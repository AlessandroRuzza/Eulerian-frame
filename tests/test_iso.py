#!/usr/bin/env python3
"""Checks iso.isomorphic / iso.canonical_labelling against known answers.

Run:  python3 tests/test_iso.py [trials]

  * round-trip: a random graph and a random relabelling of it are isomorphic,
    and the mapping that comes back really carries one edge set onto the other;
  * symmetric families (empty, K_n, star, complete bipartite, disjoint cliques)
    do the same — they are where a search without automorphism pruning dies;
  * invariant separation: graphs differing in edges / degrees / triangles are
    rejected without entering the search;
  * the Shrikhande graph vs the 4x4 rook's graph: both SRG(16,6,2,2), so colour
    refinement alone cannot tell them apart and the IR search has to;
  * the qubit cap and the node budget report "undecided", never a guess.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eulsim.iso import canonical_labelling, isomorphic


def adj_of(n, edges):
    a = [set() for _ in range(n)]
    for i, j in edges:
        a[i].add(j)
        a[j].add(i)
    return a


def edge_set(a, n):
    return {(min(i, j), max(i, j)) for i in range(n) for j in a[i]}


def relabel(n, edges, perm):
    return [(perm[i], perm[j]) for i, j in edges]


def mapping_is_iso(a, b, n, mapping):
    moved = {(min(mapping[i], mapping[j]), max(mapping[i], mapping[j]))
             for i, j in edge_set(a, n)}
    return moved == edge_set(b, n)


def main(trials: int = 40) -> int:
    rng = random.Random(20260101)
    fails = []

    # ── random graphs against their own relabellings ──────────────────────────
    for _ in range(trials):
        n = rng.randint(2, 18)
        p = rng.uniform(0.1, 0.6)
        edges = [(i, j) for i, j in itertools.combinations(range(n), 2)
                 if rng.random() < p]
        perm = list(range(n))
        rng.shuffle(perm)
        a, b = adj_of(n, edges), adj_of(n, relabel(n, edges, perm))
        res, msg, info = isomorphic(a, n, b, n, max_qubits=n)
        if res is not True or not mapping_is_iso(a, b, n, info["mapping"]):
            fails.append(f"random n={n}: {res} {msg}")
    print(f"  random relabellings  {'PASS' if not fails else 'FAIL'}")

    # ── symmetric families: large automorphism groups ─────────────────────────
    sym_fail = []
    for name, n, edges in [
            ("empty", 12, []),
            ("K12", 12, list(itertools.combinations(range(12), 2))),
            ("star", 12, [(0, i) for i in range(1, 12)]),
            ("K6,6", 12, [(i, 6 + j) for i in range(6) for j in range(6)]),
            ("3xK4", 12, [(b + i, b + j) for b in (0, 4, 8)
                          for i, j in itertools.combinations(range(4), 2)]),
            ("C12", 12, [(i, (i + 1) % 12) for i in range(12)])]:
        perm = list(range(n))
        rng.shuffle(perm)
        a, b = adj_of(n, edges), adj_of(n, relabel(n, edges, perm))
        res, msg, info = isomorphic(a, n, b, n, max_qubits=n)
        if res is not True or not mapping_is_iso(a, b, n, info["mapping"]):
            sym_fail.append(f"{name}: {res} {msg}")
    fails += sym_fail
    print(f"  symmetric families   {'PASS' if not sym_fail else 'FAIL'}")

    # ── separations ───────────────────────────────────────────────────────────
    sep_fail = []
    cases = [
        ("edge count", 3, [(0, 1)], [(0, 1), (1, 2)]),
        ("degrees", 4, [(0, 1), (1, 2), (2, 3)], [(0, 1), (0, 2), (0, 3)]),
        ("triangles", 6, [(i, (i + 1) % 6) for i in range(6)],
         [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]),
    ]
    for name, n, ea, eb in cases:
        res, msg, _ = isomorphic(adj_of(n, ea), n, adj_of(n, eb), n, max_qubits=n)
        if res is not False:
            sep_fail.append(f"{name}: {res} {msg}")
    if isomorphic(adj_of(3, []), 3, adj_of(4, []), 4)[0] is not False:
        sep_fail.append("different n")
    fails += sep_fail
    print(f"  invariants separate  {'PASS' if not sep_fail else 'FAIL'}")

    # ── SRG(16,6,2,2): colour refinement is blind, the search is not ──────────
    rook = [(a1 * 4 + b1, a2 * 4 + b2)
            for (a1, b1), (a2, b2) in itertools.combinations(
                [(i, j) for i in range(4) for j in range(4)], 2)
            if a1 == a2 or b1 == b2]
    diffs = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
    shrikhande = [(x1 * 4 + y1, x2 * 4 + y2)
                  for x1 in range(4) for y1 in range(4)
                  for x2 in range(4) for y2 in range(4)
                  if x1 * 4 + y1 < x2 * 4 + y2
                  and ((x2 - x1) % 4, (y2 - y1) % 4) in diffs]
    srg_fail = []
    if isomorphic(adj_of(16, rook), 16, adj_of(16, shrikhande), 16,
                  max_qubits=16)[0] is not False:
        srg_fail.append("rook ~ Shrikhande reported isomorphic")
    perm = list(range(16))
    rng.shuffle(perm)
    if isomorphic(adj_of(16, shrikhande), 16,
                  adj_of(16, relabel(16, shrikhande, perm)), 16,
                  max_qubits=16)[0] is not True:
        srg_fail.append("Shrikhande not isomorphic to itself")
    fails += srg_fail
    print(f"  SRG(16,6,2,2)        {'PASS' if not srg_fail else 'FAIL'}")

    # ── caps report undecided ────────────────────────────────────────────────
    cap_fail = []
    if isomorphic(adj_of(5, [(0, 1)]), 5, adj_of(5, [(0, 1)]), 5,
                  max_qubits=3)[0] is not None:
        cap_fail.append("qubit cap not honoured")
    if isomorphic(adj_of(12, []), 12, adj_of(12, []), 12,
                  max_qubits=12, node_budget=5)[0] is not None:
        cap_fail.append("node budget not honoured")
    _, _, _, capped = canonical_labelling(adj_of(12, []), 12, node_budget=5)
    if not capped:
        cap_fail.append("canonical_labelling did not report capping")
    fails += cap_fail
    print(f"  caps undecided       {'PASS' if not cap_fail else 'FAIL'}")

    for f in fails:
        print(f"    ! {f}")
    print(f"  trials               {trials}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
