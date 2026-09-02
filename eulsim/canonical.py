"""The canonical frame (sec:track-canonical).

The canonical frame of a stabilizer state is its restricted frame
(w^N in {+Z,+X}^n) whose Hadamard support F = {v : w^N_v = +X} is
shortlex-least; it exists, is unique, fixes the graph too (thm:gcf), and is
the trivial frame exactly on graph states (cor:canonical-graph-state).

`canonicalize` computes it with re-framings and pivots on the frame itself
(framecanon, sec:canon-algorithm).  `canonicalize_rref` computes the same
object by Gaussian elimination on the check matrix and is kept as an
independent cross-check (tests/test_canonical_frame.py).
"""
from __future__ import annotations

from .cliffords import (
    _H_U8,
    _IDENTITY_U8,
    _S_U8,
    _Z_U8,
    _dag_u8,
    _mat2x2_mul,
    _parse_mats,
)
from .framecanon import canonical_frame, u8_to_frame
from .tableau import _stab_mul, _tableau_from_state


def _emit(n: int, new_adj_sets, f: list[int], d: list[int], s: list[int]):
    """Shared tail of both routes: graph (adjacency sets), frame matrices,
    corrections, info."""
    new_adj = [set(x) for x in new_adj_sets]
    new_lu = []
    for q in range(n):                       # L_q = H^{f_q} · S^{d_q} · Z^{s_q}
        m = list(_Z_U8) if s[q] else list(_IDENTITY_U8)
        if d[q]:
            m = _mat2x2_mul(_S_U8, m)
        if f[q]:
            m = _mat2x2_mul(_H_U8, m)
        new_lu.append(m)
    trivial = not (any(f) or any(d) or any(s))
    info = {"status": "trivial" if trivial else "framed",
            "hadamards": sum(f),
            "f": [q for q in range(n) if f[q]],
            "d": [q for q in range(n) if d[q]],
            "s": [q for q in range(n) if s[q]]}
    return new_adj, new_lu, [_dag_u8(m) for m in new_lu], info


def canonicalize(adj: list[set[int]], n: int,
                 local_unitaries: list | None = None
                 ) -> tuple[list[set[int]], list[list[float]], list[list[float]], dict]:
    """Canonical frame of the framed state (G, L), by re-framing only.

    Rewrites (graph, frame) as the *unique* description of the same physical
    state |psi> = (x)_v H^{f_v} S^{d_v} Z^{s_v} |G'> whose Hadamard support
    F = supp(f) is shortlex-least for the vertex order 0 < ... < n-1
    (def:canonical-frame, thm:gcf).  The frame alphabet is the restricted set
    T = {I, Z, S, SZ, H, HZ, HS, HSZ} of prop:restricted-eulerian —
    equivalently w^N in {+Z,+X}^n — and the frame comes out trivial iff the
    state is a graph state (cor:canonical-graph-state).

    The three phases of sec:canon-algorithm run on the frame itself: restrict
    each vertex with R_v, cancel Hadamard pairs across edges inside F, then
    transport the survivors down with pivots (see framecanon).  Two framed
    states describe the same state iff their canonical frames coincide.

    Returns (new_adj, new_local_unitaries, corrections, info); applying
    corrections[v] = L'_v† per qubit collapses the residual frame to the pure
    graph state |G'> (changing the physical state);
    info = {status in {trivial, framed}, hadamards = |F|, f, d, s}."""
    if n == 0:
        return [], [], [], {"status": "trivial", "hadamards": 0,
                            "f": [], "d": [], "s": []}
    mats = _parse_mats(n, local_unitaries)
    r = canonical_frame(adj, [u8_to_frame(m) for m in mats])
    return _emit(n, r["adj"], r["f"], r["d"], r["s"])


def canonicalize_rref(adj: list[set[int]], n: int,
                      local_unitaries: list | None = None
                      ) -> tuple[list[set[int]], list[list[float]], list[list[float]], dict]:
    """The same canonical frame by Gaussian elimination on the check matrix.

    The Hadamard support is read off as the Z-block pivot set of the reduced
    row echelon form of the check matrix in the column order X_1..X_n, Z_1..Z_n,
    which is minimal (|F| = n - rank of the X-block) and, being the greedy
    pivot choice, shortlex-least.  Deterministic, O(n^3), no frame moves:
    kept as an independent check on `canonicalize`, which is the route the
    thesis prescribes."""
    if n == 0:
        return [], [], [], {"status": "trivial", "hadamards": 0,
                            "f": [], "d": [], "s": []}
    mats = _parse_mats(n, local_unitaries)
    tab = _tableau_from_state(adj, n, mats)

    def xbit(g, q): return g[1][q] in ("X", "Y")
    def zbit(g, q): return g[1][q] in ("Z", "Y")

    # 1. RREF of the check matrix (column order X_1..X_n, Z_1..Z_n): the qubits
    #    carrying Z-block pivots form the Hadamard support F.
    row = 0
    F0: list[int] = []
    for col in range(2 * n):
        q, in_x = col % n, col < n
        bit = xbit if in_x else zbit
        piv = next((r for r in range(row, n) if bit(tab[r], q)), None)
        if piv is None:
            continue
        tab[row], tab[piv] = tab[piv], tab[row]
        for r in range(n):
            if r != row and bit(tab[r], q):
                tab[r] = list(_stab_mul(tuple(tab[r]), tuple(tab[row]), n))
        if not in_x:
            F0.append(q)
        row += 1

    # 2. Conjugate by H_{F₀} (X↔Z letters; Y picks up a sign), then row-reduce
    #    the now full-rank X-block to the unique signed [I | B] form.
    fset = set(F0)
    for g in tab:
        for q in fset:
            letter = g[1][q]
            if letter == "X":
                g[1][q] = "Z"
            elif letter == "Z":
                g[1][q] = "X"
            elif letter == "Y":
                g[0] = -g[0]
    for q in range(n):
        piv = next((r for r in range(q, n) if xbit(tab[r], q)), None)
        if piv is None:                      # cannot happen: F₀ is valid
            raise ValueError("canonical frame: X-block singular")
        tab[q], tab[piv] = tab[piv], tab[q]
        for r in range(n):
            if r != q and xbit(tab[r], q):
                tab[r] = list(_stab_mul(tuple(tab[r]), tuple(tab[q]), n))

    # 3. Read off B = A + diag(d) and the signs: generator q is
    #    ±(X or Y)_q ⊗ Z_{N'(q)}, so off-diagonal letters give the graph,
    #    diagonal Y-letters give d, and negative rows give s.
    new_adj_sets = [set() for _ in range(n)]
    f = [1 if q in fset else 0 for q in range(n)]
    d = [0] * n
    s_bits = [0] * n
    for q in range(n):
        for k in range(n):
            if k != q and tab[q][1][k] != "I":
                new_adj_sets[q].add(k)
        d[q] = 1 if tab[q][1][q] == "Y" else 0
        s_bits[q] = 1 if tab[q][0] < 0 else 0
    return _emit(n, new_adj_sets, f, d, s_bits)
