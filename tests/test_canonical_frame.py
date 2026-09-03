#!/usr/bin/env python3
"""Checks canonical.canonicalize against the definition of the canonical frame.

Run:  python3 tests/test_canonical_frame.py [trials] [max qubits]

Per random framed state (G, L) it verifies that the output is
  * the same physical state, up to global phase;
  * a restricted frame: w^N_v in {+Z,+X} for every v (def:restricted-frame);
  * carrying the shortlex-least Hadamard support (def:canonical-frame): every
    subset S of V earlier in shortlex order admits no restricted frame of the
    state, which is checked directly on the check matrix — S is a valid support
    iff swapping the columns X_q, Z_q for q in S leaves the X-block invertible;
  * identical to the independent check-matrix route, canonicalize_rref.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eulsim.canonical import canonicalize, canonicalize_rref
from eulsim.frames import VALID_PAIRS, conj
from eulsim.statevector import compute_state_vector
from eulsim.tableau import _tableau_from_state


def _valid_support(tab, n, S: set) -> bool:
    """Is S the Hadamard support of some restricted frame of this state?"""
    rows = []
    for g in tab:
        bits = 0
        for q in range(n):
            letter = g[1][q]
            has = letter in ("Z", "Y") if q in S else letter in ("X", "Y")
            if has:
                bits |= 1 << q
        rows.append(bits)
    rank = 0
    for col in range(n):
        piv = next((i for i in range(rank, n) if rows[i] >> col & 1), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(n):
            if i != rank and rows[i] >> col & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank == n


def _shortlex_least_support(adj, n, frame) -> tuple:
    tab = _tableau_from_state(adj, n, frame)
    for size in range(n + 1):
        for S in itertools.combinations(range(n), size):
            if _valid_support(tab, n, set(S)):
                return S
    raise AssertionError("no valid Hadamard support")


def _same_state(a, b) -> bool:
    ph = None
    for x, y in zip(a, b):
        zx, zy = complex(x["re"], x["im"]), complex(y["re"], y["im"])
        if abs(zx) < 1e-9 and abs(zy) < 1e-9:
            continue
        if abs(zx) < 1e-9 or abs(zy) < 1e-9:
            return False
        r = zy / zx
        if ph is None:
            ph = r
        elif abs(r - ph) > 1e-6:
            return False
    return ph is None or abs(abs(ph) - 1) < 1e-6


def main(trials: int = 300, n_max: int = 6, seed: int = 20260826) -> int:
    rng = random.Random(seed)
    fails = {k: 0 for k in ("state", "restricted", "shortlex", "vs_rref")}

    for _ in range(trials):
        n = rng.randint(1, n_max)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < rng.choice([0.2, 0.5, 0.8]):
                    adj[i].add(j); adj[j].add(i)
        frame = [rng.choice(VALID_PAIRS) for _ in range(n)]

        new_adj, new_frame, _corr, info = canonicalize(adj, n, frame)

        if not _same_state(compute_state_vector(adj, n, frame),
                           compute_state_vector(new_adj, n, new_frame)):
            fails["state"] += 1
        if not all(conj(c, "Z") in ((1, "Z"), (1, "X")) for c in new_frame):
            fails["restricted"] += 1
        if tuple(info["f"]) != _shortlex_least_support(adj, n, frame):
            fails["shortlex"] += 1

        r_adj, r_frame, _, r_info = canonicalize_rref(adj, n, frame)
        if (r_adj, r_frame, r_info) != (new_adj, new_frame, info):
            fails["vs_rref"] += 1

    print(f"  trials              {trials}")
    for k, v in fails.items():
        print(f"  {k:<19} {'PASS' if v == 0 else f'FAIL x{v}'}")
    return 0 if sum(fails.values()) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
