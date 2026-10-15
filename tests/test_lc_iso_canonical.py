#!/usr/bin/env python3
"""Checks that lc_orbit.lc_iso_canonical is one fixed graph per LC+relabelling class.

Run:  python3 tests/test_lc_iso_canonical.py [trials]

  * every relabelling of every 4- and 5-qubit graph without isolated qubits,
    and random relabellings of random 6-8 qubit graphs: rep(pi(G)) == rep(G);
  * rep(tau_v(G)) == rep(G) for every vertex v;
  * the representative is in the class: isomorphic to lc_canonical's
    representative, which is LC-equivalent to G;
  * the 7-qubit counterexample of LC_isomorph.tex §5 and its relabelling get
    the same representative;
  * graphs in different LC+relabelling classes (L vs R of §3) get different
    representatives.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eulsim.graph_ops import local_complement
from eulsim.iso import isomorphic
from eulsim.lc_orbit import lc_canonical, lc_iso_canonical


def adj_of(n, edges):
    a = [set() for _ in range(n)]
    for i, j in edges:
        a[i].add(j)
        a[j].add(i)
    return a


def edges_of(a, n):
    return [(i, j) for i in range(n) for j in sorted(a[i]) if i < j]


def relabel(a, n, perm):
    return adj_of(n, [(perm[i], perm[j]) for i, j in edges_of(a, n)])


def rep(a, n):
    r, _, _, exact, _ = lc_iso_canonical(a, n)
    assert exact
    return edges_of(r, n)


def main(trials: int = 30) -> int:
    rng = random.Random(20261003)
    fails = []

    # ── the §5 counterexample ────────────────────────────────────────────────
    ce_fail = []
    G = adj_of(7, [(0, 1), (0, 3), (0, 4), (0, 6), (1, 3), (1, 5), (2, 3), (3, 6)])
    Gp = relabel(G, 7, (0, 2, 1, 3, 4, 5, 6))
    if rep(G, 7) != rep(Gp, 7):
        ce_fail.append("rep(G) != rep(pi G)")
    if len(rep(G, 7)) != 7:
        ce_fail.append(f"rep(G) has {len(rep(G, 7))} edges, expected 7")
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
                if rep(relabel(a, n, perm), n) != r:
                    ex_fail.append(f"n={n} {edges_of(a, n)} perm {perm}")
                    break
            if len(ex_fail) > 3:
                break
    fails += ex_fail
    print(f"  exhaustive n=4,5         {'PASS' if not ex_fail else 'FAIL'}")

    # ── random graphs: relabellings, LC moves, membership ────────────────────
    rnd_fail = []
    for _ in range(trials):
        n = rng.randint(6, 8)
        p = rng.uniform(0.25, 0.6)
        a = adj_of(n, [e for e in itertools.combinations(range(n), 2) if rng.random() < p])
        r = rep(a, n)
        if not isomorphic(adj_of(n, r), n, lc_canonical(a, n)[0], n)[0]:
            rnd_fail.append(f"n={n} {edges_of(a, n)}: rep not in the class")
        for v in range(n):
            if rep(local_complement(a, n, v)[0], n) != r:
                rnd_fail.append(f"n={n} {edges_of(a, n)} tau_{v}")
                break
        for _ in range(4):
            perm = list(range(n))
            rng.shuffle(perm)
            if rep(relabel(a, n, perm), n) != r:
                rnd_fail.append(f"n={n} {edges_of(a, n)} perm {perm}")
                break
    fails += rnd_fail
    print(f"  random n=6..8            {'PASS' if not rnd_fail else 'FAIL'}")

    # ── different classes stay apart ─────────────────────────────────────────
    sep_fail = []
    L = adj_of(6, [(0, 1), (0, 2), (0, 3), (2, 4), (3, 4), (4, 5)])
    R = adj_of(6, [(0, 1), (0, 2), (0, 4), (2, 3), (2, 4), (4, 5)])
    if rep(L, 6) == rep(R, 6):
        sep_fail.append("L and R (§3) got the same representative")
    fails += sep_fail
    print(f"  classes separate         {'PASS' if not sep_fail else 'FAIL'}")

    for f in fails:
        print(f"    ! {f}")
    print(f"  trials                   {trials}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
