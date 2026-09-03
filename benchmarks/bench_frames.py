#!/usr/bin/env python3
"""Benchmark: framed graph-state simulation, Clifford matrices vs Pauli images.

Four simulators share byte-identical graph code (adjacency as list[set],
all operations in place); they differ ONLY in how the per-vertex frame L_v
is kept in memory, so measured differences isolate the representation cost.

Baseline     "clifford": L_v stored as a 2x2 complex matrix (8 floats) --
    the representation of the accompanying eulsim module and the
    literature "graph state + local Clifford" picture. Every frame update
    is a complex matrix product (_mat2x2_mul); every semantic read
    (measurement basis transport, CZ case split, VOP decomposition)
    conjugates Paulis through the matrix (_conj_pauli) or normalises it to
    a phase-canonical key (_clifford_key).

Baseline+LUT "cliffordlut": the SAME 24-element Clifford data, but with the
    obvious rebuttal applied: since there are only 24 single-qubit
    Cliffords mod phase, any per-vertex update reachable by composing with
    one of a fixed small gate set is itself a function on a 24-element
    domain, hence just as table-driven-able as the candidate below. Every
    op that "clifford" does via _mat2x2_mul/_conj_pauli here does one dict
    lookup keyed by eulsim's own canonical key _clifford_key (the same
    key _Z_SET_KEYS/_VOP_DECOMP/_cz_block_lookup already use internally).
    This isolates what the pair encoding actually buys once "table lookup
    instead of matrix multiply" is taken off the table for both sides: a
    native small-int key (fast to hash/index, no rounding/canonicalisation
    step, 1 byte packed) versus an 8-tuple-of-rounded-floats key (dict
    hashing, and _clifford_key's phase-division+round to compute it before
    the FIRST lookup of a fresh matrix).

Baseline+ID  "cliffordid": pushes that rebuttal one step further. 24 < 32 =
    2^5, so a Clifford fits an opaque small int too, not just a tuple --
    each of the 24 Cliffords gets an arbitrary id (BFS discovery order, no
    relation to w^C/w^N) and every table is a plain LIST indexed by that id,
    exactly mirroring euler's storage shape and access pattern. What
    survives this is no longer about key size or container type (both are
    now identical to euler's): it is that euler's code is not just compact
    but *decomposable* -- axis and sign of both stored Paulis are read by
    %3/%6 arithmetic on the int itself, with NO lookup table at all, whereas
    an opaque id must always go through an explicit table (TRANSPORT_ID,
    ZAXIS_ID, ZSET_ID, ...) to answer the same question, because the id
    carries no structure to decode.

Candidate    "euler": L_v stored as the signed images of X and Z -- the
    supplementary and Eulerian elements
        (w^C, w^N) = (L X L^dag, L Z L^dag),
    encoded in ONE integer code = 6*w^C + w^N in 0..35 (24 valid codes;
    signed Pauli p = axis + 3*sign_bit, axes X=0 Y=1 Z=2). The pair is a
    bijection with C_1 mod phase (notes-EulVec-Rep-Operations, Prop. 1), so
    nothing is lost. Every frame update is a single read of a table of size
    <= 36; every semantic read is an integer compare; the Y image is
    i*w^C*w^N (table IPROD). No floats anywhere, and unlike cliffordlut's
    memoised group table, several of the rules (IPROD, NEG6, the CZ Z-axis
    test code%6 in {2,5}) are closed-form in the two Pauli-axis/sign
    sub-fields rather than opaque 24x24 lookups memorised from brute force.

All pair-space rules are DERIVED from the matrix toolkit at import time and
verified exhaustively over the 24 Cliffords (see frames._verify), so the
integer rules are machine-checked against the matrix semantics:

    right-fold V (byproducts, re-framing):     code -> TABLE_V[code]
      R_v centre   (.HS^dagH ~ sqrt(iX)):  w^N_v  -> i w^C_v w^N_v
      R_v neighbour(.S       ~ sqrt(iZ)):  w^C_u  -> i w^C_u w^N_u
      LC centre    (.HSH   ~ sqrt(-iX)):   w^N_v  -> -i w^C_v w^N_v
      LC neighbour (.S^dag ~ sqrt(-iZ)):   w^C_u  -> -i w^C_u w^N_u
      Z fold:                              w^C    -> -w^C
    left-compose gate g (physical):            code -> LGATE[g][code]
    dagger (basis transport):                  code -> DAG_PAIR[code]
    measurement transport L^dag P L:           direct read of the pair
    CZ case split (w^N on the Z axis?):        code % 6 in {2, 5}
    VOP decomposition word:                    DECOMP_PAIR[DAG_PAIR[code]]
    coupled CZ block:                          dict keyed by small ints

Operations benchmarked (the calculus of the papers/notes):
    single   physical single-qubit Clifford (left frame composition)
    reframe  re-framing move R_v: (G,L) -> (tau_v(G), L.U_v^dag)
    measure  Pauli measurement via the reduction chain X -> Y -> Z
             (mirrors eulsim.graph_ops.do_measure, delete=False)
    cz       physical CZ by the local Anders-Briegel algorithm
             (mirrors eulsim.gates.apply_cz)
    mixed    weighted mix of the above

Usage:
    bench_frames.py --selftest            rule + equivalence verification
    bench_frames.py --tables              print the derived integer tables
    bench_frames.py [--sizes ...] [--degrees ...] [--degree 6]\n                    [--n-fixed 200] [--seed 1] [--reps 3]\n                    [--drifting-warmup]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from math import log
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eulsim.frames import (           # noqa: E402
    AXES, GATE_U8, LGATE, PAIR_TO_MAT, VALID_PAIRS,
)
from eulsim.graphsim import (        # noqa: E402
    CliffordIDSim, CliffordLUTSim, CliffordSim,
)
from eulsim.graphsim.tables import LID, cz_tables   # noqa: E402
from eulsim.sim import EulerSim      # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Backends
# ═══════════════════════════════════════════════════════════════════════════
# All four live in the package now, not in this file: the Eulerian candidate
# is eulsim.sim.EulerSim (the in-place mirror of the functional core), and
# the three VOP-storage baselines are eulsim.graphsim.  They share the graph
# layer, the coupled CZ block table and the tableau fallback, so what differs
# between them is exactly the frame representation.  The pair rule tables
# themselves (eulsim.frames) are derived from the matrix toolkit at import
# and machine-checked against it over all 24 Cliffords.


def init_cz_tables() -> float:
    """Warm the coupled two-qubit block table for every backend before any
    timing starts, so the one-time build never lands inside a measured op.
    Returns the seconds it took."""
    t0 = time.perf_counter()
    cz_tables()                      # builds the core table, then re-indexes
    return time.perf_counter() - t0

# ═══════════════════════════════════════════════════════════════════════════
# Random states, abstract ops, adapters
# ═══════════════════════════════════════════════════════════════════════════

GNAMES = ("H", "S", "SDG", "X", "Y", "Z")


def random_state(rng: Random, n: int, degree: float) -> tuple[list[set], list[int]]:
    """A G(n, p) state with p chosen for the target average degree.

    Drawn by the Batagelj-Brandes geometric-skip method, so the cost is
    O(n + m) and not O(n^2): the benchmark below re-draws a state for every
    timed operation (see _bench_point), which the quadratic construction
    made unaffordable above a few hundred qubits."""
    p = min(1.0, degree / max(1, n - 1))
    adj: list[set] = [set() for _ in range(n)]
    if p >= 1.0:
        for i in range(n):
            adj[i] = set(range(n)) - {i}
    elif p > 0.0:
        lp = log(1.0 - p)
        rnd, v, w = rng.random, 1, -1
        while v < n:
            w += 1 + int(log(1.0 - rnd()) / lp)
            while w >= v and v < n:
                w -= v
                v += 1
            if v < n:
                adj[v].add(w); adj[w].add(v)
    codes = [rng.choice(VALID_PAIRS) for _ in range(n)]
    return adj, codes


def mean_degree(adj: list[set]) -> float:
    return sum(len(s) for s in adj) / len(adj)


def gen_ops(rng: Random, n: int, count: int, kind: str | None = None) -> list:
    ops = []
    for _ in range(count):
        k = kind
        if k is None:
            r = rng.random()
            k = ("g" if r < 0.40 else "r" if r < 0.55 else
                 "m" if r < 0.72 else "c")
        if k == "g":
            ops.append(("g", rng.randrange(n), rng.choice(GNAMES)))
        elif k == "r":
            ops.append(("r", rng.randrange(n)))
        elif k == "m":
            ops.append(("m", rng.randrange(n), rng.randrange(3),
                        bool(rng.randrange(2))))
        else:
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            ops.append(("c", i, j + (1 if j >= i else 0)))
    return ops


SIM_CLASSES = {"clifford": CliffordSim, "cliffordlut": CliffordLUTSim,
               "cliffordid": CliffordIDSim, "euler": EulerSim}
_GATE_ARG = {"clifford": GATE_U8, "cliffordlut": None, "cliffordid": LID, "euler": LGATE}


def bind_ops(sim, ops: list, backend: str) -> list:
    """Pre-bind (method, args) so the timed loop has identical dispatch cost
    across backends. clifford wants a raw gate matrix, cliffordlut a gate
    name string (key into LKEY), cliffordid/euler a precomputed id/pair
    table; measure wants a Pauli letter for the three Clifford-keyed
    backends, an axis int for euler."""
    euler = backend == "euler"
    tbl = _GATE_ARG[backend]
    calls = []
    for op in ops:
        k = op[0]
        if k == "g":
            g_arg = op[2] if tbl is None else tbl[op[2]]
            calls.append((sim.apply_local, (op[1], g_arg)))
        elif k == "r":
            calls.append((sim.reframe, (op[1],)))
        elif k == "m":
            calls.append((sim.measure,
                          (op[1], op[2] if euler else AXES[op[2]], op[3])))
        else:
            calls.append((sim.cz, (op[1], op[2])))
    return calls


# ═══════════════════════════════════════════════════════════════════════════
# Self-test: baseline == candidate == eulsim module, op by op
# ═══════════════════════════════════════════════════════════════════════════

class ModuleRef:
    """The functional eulsim API (copy-in/copy-out, as the server calls it),
    used as ground truth for the in-place backends."""

    def __init__(self, adj: list[set], pair_codes: list[int]):
        from eulsim.gates import apply_cz
        from eulsim.graph_ops import do_measure, reframe_move
        self._do_measure, self._reframe_move, self._apply_cz = \
            do_measure, reframe_move, apply_cz
        self.n = len(adj)
        self.adj = [set(s) for s in adj]
        self.f = list(pair_codes)

    def apply_local(self, v, gate_name):
        self.f[v] = LGATE[gate_name][self.f[v]]

    def reframe(self, v):
        self.adj, _, self.f = self._reframe_move(self.adj, self.n, v, self.f)

    def measure(self, v, basis, invert):
        self.adj, _, _, self.f = self._do_measure(
            self.adj, self.n, v, basis.lower(), self.f,
            delete=False, invert=invert)

    def cz(self, i, j):
        self.adj, self.f = self._apply_cz(self.adj, self.n, i, j, self.f)

    def pair_codes(self):
        return list(self.f)

    def adj_sets(self):
        return [set(s) for s in self.adj]


def selftest(seeds: int = 4) -> None:
    init_cz_tables()
    print("rule tables: derived and verified against the matrix toolkit (24/24)")
    for seed in range(seeds):
        rng = Random(seed)
        n = 8 if seed % 2 == 0 else 24
        adj, codes = random_state(rng, n, 3.0)
        ops = gen_ops(rng, n, 300)
        sims = {name: cls(adj, codes) for name, cls in SIM_CLASSES.items()}
        r = ModuleRef(adj, codes)
        for t, op in enumerate(ops):
            for name, sim in sims.items():
                fn, args = bind_ops(sim, [op], name)[0]
                fn(*args)
            k = op[0]
            if k == "g":
                r.apply_local(op[1], op[2])
            elif k == "r":
                r.reframe(op[1])
            elif k == "m":
                r.measure(op[1], AXES[op[2]], op[3])
            else:
                r.cz(op[1], op[2])
            ctx = f"seed={seed} step={t} op={op}"
            adjs = [sim.adj for sim in sims.values()] + [r.adj_sets()]
            assert all(a == adjs[0] for a in adjs), f"adjacency mismatch: {ctx}"
            codes_now = [sim.pair_codes() for sim in sims.values()] + [r.pair_codes()]
            assert all(c == codes_now[0] for c in codes_now), f"frame mismatch: {ctx}"
        b = sims["euler"]
        print(f"  seed {seed}: n={n}, {len(ops)} mixed ops -- "
              f"clifford == cliffordlut == euler == eulsim module "
              f"(cz paths: diag={b.n_diag} table={b.n_table} "
              f"fallback={b.n_fallback})")
    print("selftest OK")


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

# The four representations of the ladder (see the module docstring): matrix,
# matrix+LUT, opaque id, Eulerian pair.  The paper's tables compare only the
# last two -- the middle rungs are what show that the matrix backend's large
# gap is a container/key artefact and not a property of the pair -- so a run
# that only needs the paper's two columns can select them with --backends.
ALL_BACKENDS = ("clifford", "cliffordlut", "cliffordid", "euler")
BACKENDS = ALL_BACKENDS


def _timer_overhead(samples: int = 20000) -> float:
    """Seconds charged by one perf_counter() pair, subtracted from every
    timed region below.  With one operation per region (see _bench_point)
    it is ~0.11 us, which is not negligible against a cheap rule."""
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(samples):
            time.perf_counter()
            time.perf_counter()
        best = min(best, (time.perf_counter() - t0) / samples)
    return best


_T_OVERHEAD = 0.0


def _warmup_ops(rng: Random, n: int, count: int, preserve: bool,
                kind: str | None) -> list:
    """The untimed operations run before the timer starts.

    With preserve=False these are drawn like the timed ones, which is what
    earlier revisions did: a single re-framing turns a neighbourhood into a
    clique, so at high degree the state the timed operation sees is denser
    than the G(n, p) draw (256 -> 287) and the reported degree is not the
    requested one.

    With preserve=True (the default) each warm-up unit is a PAIR of
    re-framings on the same vertex.  Local complementation is an involution,
    so the pair is the sign fold R_v^2: it leaves the graph exactly as drawn
    -- the measured degree is then the nominal one -- while still walking the
    adjacency, which is what the warm-up is for (the first operation on a
    freshly allocated state costs several times a later one, and that penalty
    grows with n).  Only the frame is left changed, and the frame is random
    to begin with."""
    if not preserve:
        return gen_ops(rng, n, count, kind)
    ops = []
    for _ in range(count):
        v = rng.randrange(n)
        ops += [("r", v), ("r", v)]
    return ops


def _bench_point(n: int, degree: float, kind: str | None,
                 ops_per_chunk: int, chunks: int, seed: int, warmup: int,
                 preserve: bool = True) -> tuple[dict, float]:
    """Mean seconds/op for every backend, and the mean degree actually seen.

    A fresh G(n, p) state is drawn for every chunk (untimed) and ALL backends
    then run that same state and op stream, so the comparison between them is
    paired.

    Re-framing, measurement and CZ all rewrite the graph -- a re-framing turns
    a neighbourhood into a clique -- so a chunk of many operations leaves the
    nominal degree behind within a handful of operations and the run measures
    a drifted degree instead of the requested one.  The warm-up is
    graph-preserving for the same reason (see _warmup_ops).  The graph-rewriting
    workloads therefore use ops_per_chunk = 1, preceded by `warmup` untimed
    operations: the warm-up puts the interpreter and the freshly built
    adjacency in steady state (the very first operation on a newly allocated
    state costs several times a later one, and that penalty does grow with n),
    while a single timed operation keeps the degree at its nominal value.  The
    degree returned is measured on the state as the timed operation sees it,
    so the reported figure is the degree the timings were taken at, not a
    target.  Single-qubit Cliffords never touch the graph and keep a large
    chunk, since there is nothing to drift."""
    tot = {b: 0.0 for b in BACKENDS}
    deg_sum, n_ops = 0.0, 0
    for c in range(chunks):
        rng = Random((seed, n, degree, kind, c).__hash__())
        adj, codes = random_state(rng, n, degree)
        warm_ops = _warmup_ops(rng, n, warmup, preserve, kind)
        ops = warm_ops + gen_ops(rng, n, ops_per_chunk, kind)
        n_warm = len(warm_ops)
        n_ops += ops_per_chunk
        for backend in BACKENDS:
            sim = SIM_CLASSES[backend](adj, codes)
            calls = bind_ops(sim, ops, backend)
            gc.disable()
            for fn, args in calls[:n_warm]:
                fn(*args)
            if backend == BACKENDS[0]:
                deg_sum += mean_degree(sim.adj)   # as the timed op sees it
            t0 = time.perf_counter()
            for fn, args in calls[n_warm:]:
                fn(*args)
            tot[backend] += time.perf_counter() - t0 - _T_OVERHEAD
            gc.enable()
    return ({b: tot[b] / n_ops for b in BACKENDS}, deg_sum / chunks)


# (label, kind, ops/chunk, chunks, warm-up ops).  ops/chunk is 1 wherever the
# operation rewrites the graph; see the docstring of _bench_point.
PLANS = (
    # warm-up 1 even though a single-qubit Clifford never touches the graph:
    # without it the first-touch of the freshly built adjacency lands INSIDE
    # the timed region and is amortised over the chunk, which at n=2000,
    # d=256 reports 4.75 us/op for a rule whose steady-state cost is 0.27.
    ("single", "g", 2000, 5, 1),
    ("reframe", "r", 1, 600, 1),
    ("measure", "m", 1, 600, 1),
    ("cz", "c", 1, 600, 1),
    ("mixed", None, 1, 900, 1),
)


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def _sweep(axis: str, values: list, n: int, degree: float,
           seed: int, reps: int, scale: float, preserve: bool = True) -> None:
    """One table.  axis='n' varies the size at fixed degree, axis='d' varies
    the degree at fixed size.  Every cell is the MEAN over `reps` independent
    repetitions (each itself a mean over its chunks); +- is the sample
    standard deviation over those repetitions."""
    var = "n" if axis == "n" else "d"
    others = [b for b in BACKENDS if b != "euler"]
    hdr = (f"{'op':<9}{var:>7}{'deg':>7} "
           + " ".join(f"{b:>12}" for b in BACKENDS)
           + f" {'+-%':>6}  "
           + " ".join(f"{b + '->e':>12}" for b in others))
    print(hdr + "  (all times us/op)")
    print("-" * len(hdr))
    for label, kind, opc, chunks, warm in PLANS:
        for val in values:
            nn = val if axis == "n" else n
            dd = degree if axis == "n" else float(val)
            per_rep = {b: [] for b in BACKENDS}
            degs = []
            # a chunk costs O(n + m) to set up and the timed op costs O(d^2),
            # so the sample count is thinned at high degree to keep the run
            # time bounded; the reported +- shows the price in precision.
            ch = max(40, int(chunks * min(1.0, 8.0 / max(dd, 8.0))))
            for r in range(reps):
                res, deg = _bench_point(nn, dd, kind, opc,
                                        max(1, int(ch * scale)),
                                        seed + 1000 * r, warm, preserve)
                for b in BACKENDS:
                    per_rep[b].append(res[b] * 1e6)
                degs.append(deg)
            mean = {b: _mean_sd(per_rep[b])[0] for b in BACKENDS}
            sd_e = _mean_sd(per_rep["euler"])[1]
            e = mean["euler"]
            print(f"{label:<9}{val:>7}{sum(degs) / len(degs):>7.2f} "
                  + " ".join(f"{mean[b]:>12.2f}" for b in BACKENDS)
                  + f" {100 * sd_e / e:>5.1f}%  "
                  + " ".join(f"{mean[b] / e:>11.2f}x" for b in others))
        print()


def run_benchmarks(sizes: list[int], degrees: list[float], degree: float,
                   n_fixed: int, seed: int, reps: int, scale: float,
                   only: str | None = None, preserve: bool = True) -> None:
    global _T_OVERHEAD
    build_t = init_cz_tables()
    _T_OVERHEAD = _timer_overhead()
    print(f"coupled-CZ tables built in {build_t:.2f}s "
          f"(one-time, all backends, excluded from timings)")
    print(f"timer overhead {_T_OVERHEAD * 1e9:.0f} ns per timed region, "
          f"subtracted; mean of {reps} repetitions")
    print("warm-up: " + ("sign folds R_v^2, graph-preserving -- the measured"
                         " degree is the nominal one"
                         if preserve else
                         "same kind as the timed operation -- the measured"
                         " degree drifts above the nominal one") + "\n")
    if only != "d":
        print(f"### size sweep: target average degree {degree:g}\n")
        _sweep("n", sizes, 0, degree, seed, reps, scale, preserve)
    if only != "n":
        print(f"### degree sweep: n = {n_fixed}\n")
        _sweep("d", degrees, n_fixed, 0.0, seed, reps, scale, preserve)
    _memory_report()


def _memory_report() -> None:
    m = list(_IDENTITY_U8)
    clif_py = sys.getsizeof(m) + sum(sys.getsizeof(x) for x in m)
    key_py = sys.getsizeof(_ID_KEY) + sum(sys.getsizeof(x) for x in _ID_KEY)
    id_py = sys.getsizeof(_ID_I)
    eul_py = sys.getsizeof(ID_PAIR)
    print("per-vertex frame storage")
    print(f"  clifford matrix   : {clif_py:>4d} B as CPython list of 8 floats"
          f" | 64 B packed (8 x float64)")
    print(f"  cliffordlut key   : {key_py:>4d} B as CPython 8-tuple of floats"
          f" | 64 B packed (same 8 floats, rounded)")
    print(f"  cliffordid id     : {id_py:>4d} B as CPython int (opaque 0..23)"
          f" | 1 B packed (24 states, 5 bits) -- same as euler")
    print(f"  euler pair code   : {eul_py:>4d} B as CPython int"
          f"                | 1 B packed (24 states, 5 bits)")
    print(f"  rule tables       : {sum(1 for _ in VALID_PAIRS)} valid codes; "
          f"all euler transition tables <= 36 entries each (list, O(1) index)")
    print(f"  cliffordlut tables: same 24 states, dict-keyed by an 8-tuple "
          f"(hash + compare on lookup, vs a bare list index)")
    print(f"  cliffordid tables : same 24 states, list-indexed by an opaque"
          f" int -- storage-identical to euler, but every derived property"
          f" (axis, sign, Z-set) still needs its OWN lookup table, since the"
          f" id itself carries no arithmetic structure to decode")


# ═══════════════════════════════════════════════════════════════════════════
# Derived-table printout (paper appendix material)
# ═══════════════════════════════════════════════════════════════════════════

def _pstr(p: int | None) -> str:
    return "." if p is None else ("+-"[p // 3] + AXES[p % 3])


def print_tables() -> None:
    print("signed Pauli codes: p = axis + 3*sign  "
          "(X=0 Y=1 Z=2; +=0 -=1), pair code = 6*w^C + w^N\n")
    print("the 24 frames (pair <-> Clifford mod phase):")
    for c in VALID_PAIRS:
        wc, wn = divmod(c, 6)
        print(f"  {c:>2}  (w^C,w^N)=({_pstr(wc)},{_pstr(wn)})"
              f"   L = {_name_clifford(PAIR_TO_MAT[c])}")
    print("\nIPROD[p][q] = i P_p P_q  (the Y image: L Y L^dag = i w^C w^N):")
    print("      " + "  ".join(f"{_pstr(q):>3}" for q in range(6)))
    for p in range(6):
        print(f"  {_pstr(p):>3} " + "  ".join(f"{_pstr(IPROD[p][q]):>3}"
                                              for q in range(6)))
    print("\nright-fold rules (byproducts / re-framing), one table read each:")
    print("  R_v centre    w^N -> i w^C w^N        RV_CENTER")
    print("  R_v neighbour w^C -> i w^C w^N        RV_NEIGH")
    print("  LC centre     w^N -> -i w^C w^N       LC_CENTER")
    print("  LC neighbour  w^C -> -i w^C w^N       LC_NEIGH")
    print("  Z fold        w^C -> -w^C             ZFOLD")
    print("\nleft-compose (physical gates), code -> code over the 24 pairs:")
    for g in GNAMES:
        row = "  ".join(f"{c}>{LGATE[g][c]}" for c in VALID_PAIRS)
        print(f"  {g:<4} {row}")
    print("\nmeasurement transport (L^dag P L), direct read of the pair:")
    print("  axis(w^C)=P -> measure X with sign(w^C); axis(w^N)=P -> Z with"
          " sign(w^N); else Y with sign(i w^C w^N)")
    print("\nCZ case split: w^N on Z axis <=> code%6 in {2,5}; "
          "Z-set {I,Z,S,S^dag} <=> code%6 == 2")
    print("dagger table DAG_PAIR and decomposition words DECOMP_PAIR "
          "(shared with gates._VOP_DECOMP):")
    for c in VALID_PAIRS:
        print(f"  {c:>2} dag>{DAG_PAIR[c]:>2}  word={DECOMP_PAIR[c] or '-'}")


# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--sizes",
                    default="50,80,130,200,320,500,800,1300,2000,3200")
    ap.add_argument("--degrees", default="2,4,8,16,32,64,128,256")
    ap.add_argument("--degree", type=float, default=6.0)
    ap.add_argument("--n-fixed", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--backends",
                    default=",".join(ALL_BACKENDS),
                    help="comma-separated subset of "
                         + ",".join(ALL_BACKENDS)
                         + "; euler is always kept as the reference column."
                           " The paper's tables need only cliffordid,euler,"
                           " and dropping the other two cuts the per-chunk"
                           " setup and timing work roughly in half")
    ap.add_argument("--drifting-warmup", action="store_true",
                    help="warm up with the same operation kind that is timed,"
                         " as earlier revisions did; the neighbourhood a"
                         " re-framing cliques then pushes the measured degree"
                         " above the requested one")
    ap.add_argument("--only", choices=("n", "d"), default=None,
                    help="run only the size sweep or only the degree sweep")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every chunk count (quick smoke runs)")
    args = ap.parse_args()
    if args.tables:
        print_tables()
        return
    if args.selftest:
        selftest()
        return
    global BACKENDS
    chosen = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = [b for b in chosen if b not in ALL_BACKENDS]
    if unknown:
        ap.error(f"unknown backend(s) {','.join(unknown)}; "
                 f"choose from {','.join(ALL_BACKENDS)}")
    if "euler" not in chosen:
        chosen.append("euler")
    BACKENDS = tuple(b for b in ALL_BACKENDS if b in chosen)
    sizes = [int(s) for s in args.sizes.split(",")]
    degrees = [float(d) for d in args.degrees.split(",")]
    run_benchmarks(sizes, degrees, args.degree, args.n_fixed,
                   args.seed, args.reps, args.scale, args.only,
                   not args.drifting_warmup)


if __name__ == "__main__":
    main()
