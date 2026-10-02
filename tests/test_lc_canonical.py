#!/usr/bin/env python3
"""Checks that lc_orbit.lc_canonical is invariant under relabelling the qubits.

Run:  python3 tests/test_lc_canonical.py [trials]

  * the 7-qubit counterexample of LC_isomorph.tex §5: G and G with qubits 1, 2
    swapped used to get non-isomorphic representatives (two fewest-edge shapes
    in one orbit, picked by the labelled edge list);
  * every relabelling of every 4- and 5-qubit graph without isolated qubits,
    and random relabellings of random 6-8 qubit graphs: rep(pi(G)) ≅ rep(G);
  * the representative is in the orbit (labelled LC-equivalent to G) and has
    the fewest edges of the orbit;
  * graphs in different LC+relabelling classes (L vs R of §3) get
    non-isomorphic representatives.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eulsim.iso import isomorphic
from eulsim.lc_orbit import lc_canonical, lc_equiv_labeled


def adj_of(n, edges):
    a = [set() for _ in range(n)]
    for i, j in edges:
        a[i].add(j)
        a[j].add(i)
    return a


def edges_of(a, n):
    return [(i, j) for i in range(n) for j in a[i] if i < j]


def relabel(a, n, perm):
    return adj_of(n, [(perm[i], perm[j]) for i, j in edges_of(a, n)])


def rep(a, n):
    return lc_canonical(a, n)[0]


def main(trials: int = 30) -> int:
    rng = random.Random(20261002)
    fails = []

    # ── the §5 counterexample ────────────────────────────────────────────────
    ce_fail = []
    G = adj_of(7, [(0, 1), (0, 3), (0, 4), (0, 6), (1, 3), (1, 5), (2, 3), (3, 6)])
    Gp = relabel(G, 7, (0, 2, 1, 3, 4, 5, 6))
    rG, rGp = rep(G, 7), rep(Gp, 7)
    if not isomorphic(rG, 7, rGp, 7)[0]:
        ce_fail.append("rep(G) and rep(pi G) not isomorphic")
    if len(edges_of(rG, 7)) != 7:
        ce_fail.append(f"rep(G) has {len(edges_of(rG, 7))} edges, expected 7")
    if "2 fewest-edge shapes" not in lc_canonical(G, 7)[3]:
        ce_fail.append("message does not report the two shapes")
    fails += ce_fail
    print(f"  7-qubit counterexample   {'PASS' if not ce_fail else 'FAIL'}")

    # ── every relabelling of small graphs ────────────────────────────────────
    ex_fail = []
    for n in (4, 5):
        pairs = list(itertools.combinations(range(n), 2))
        for bits in itertools.product((0, 1), repeat=len(pairs)):
            a = adj_of(n, [e for e, b in zip(pairs, bits) if b])
            if any(not s for s in a):
                continue
            r = rep(a, n)
            for perm in itertools.permutations(range(n)):
                if not isomorphic(rep(relabel(a, n, perm), n), n, r, n)[0]:
                    ex_fail.append(f"n={n} {edges_of(a, n)} perm {perm}")
                    break
            if len(ex_fail) > 3:
                break
    fails += ex_fail
    print(f"  exhaustive n=4,5         {'PASS' if not ex_fail else 'FAIL'}")

    # ── random graphs, random relabellings ───────────────────────────────────
    rnd_fail = []
    for _ in range(trials):
        n = rng.randint(6, 8)
        p = rng.uniform(0.25, 0.6)
        a = adj_of(n, [e for e in itertools.combinations(range(n), 2) if rng.random() < p])
        r = rep(a, n)
        res, _, _ = lc_equiv_labeled(a, n, r)
        if res is not True:
            rnd_fail.append(f"n={n}: rep not in the orbit of G")
        for _ in range(4):
            perm = list(range(n))
            rng.shuffle(perm)
            if not isomorphic(rep(relabel(a, n, perm), n), n, r, n)[0]:
                rnd_fail.append(f"n={n} {edges_of(a, n)} perm {perm}")
                break
    fails += rnd_fail
    print(f"  random n=6..8            {'PASS' if not rnd_fail else 'FAIL'}")

    # ── different classes stay apart ─────────────────────────────────────────
    sep_fail = []
    L = adj_of(6, [(0, 1), (0, 2), (0, 3), (2, 4), (3, 4), (4, 5)])
    R = adj_of(6, [(0, 1), (0, 2), (0, 4), (2, 3), (2, 4), (4, 5)])
    if isomorphic(rep(L, 6), 6, rep(R, 6), 6)[0]:
        sep_fail.append("L and R (§3) got isomorphic representatives")
    fails += sep_fail
    print(f"  classes separate         {'PASS' if not sep_fail else 'FAIL'}")

    for f in fails:
        print(f"    ! {f}")
    print(f"  trials                   {trials}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
