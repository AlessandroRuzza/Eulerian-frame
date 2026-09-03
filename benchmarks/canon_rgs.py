#!/usr/bin/env python3
"""Canonicalization on chains of repeater graph states (tab:bench-canon-rgs).

Repeater i: inner qubits L_i (m of them) and R_i (m), complete bipartite
K_{m,m} between them (the merged RGS of res:rgs-merge), one outer arm per
inner qubit.  Adjacent repeaters are linked arm-to-arm: the j-th R-arm of
repeater i is joined to the j-th L-arm of repeater i+1.

Each row is REPS independent random frames on the same chain.  The canonical
frame is unique (thm:gcf) and fixes the graph, so the post-canonicalization
degree is a property of the state, not of the run: the spread across rows is
the spread over the frames drawn, and is reported as a standard deviation.

Run from anywhere:
    .env/bin/python3 benchmarks/canon_rgs.py [--reps 5]
"""
import argparse
import gc
import sys
import time
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eulsim.framecanon import canonical_frame             # noqa: E402
from eulsim.frames import ID_PAIR, VALID_PAIRS            # noqa: E402

REPS = 5               # random frames per row
BUDGET = 30.0          # seconds for one canonicalization
ARMS = (5, 10, 15, 20)
HOPS = (4, 16, 64, 256)


def rgs_chain(hops, m):
    """adjacency of a chain of `hops` merged RGSs with m arms per side."""
    adj, idx = [], {}
    def new():
        adj.append(set()); return len(adj) - 1
    def link(a, b):
        adj[a].add(b); adj[b].add(a)
    for h in range(hops):
        L = [new() for _ in range(m)]; R = [new() for _ in range(m)]
        for a in L:
            for b in R: link(a, b)                     # K_{m,m} core
        LA = [new() for _ in range(m)]; RA = [new() for _ in range(m)]
        for a, arm in zip(L, LA): link(a, arm)         # outer arms
        for a, arm in zip(R, RA): link(a, arm)
        idx[h] = (LA, RA)
    for h in range(hops - 1):                          # arm-to-arm fusion link
        for a, b in zip(idx[h][1], idx[h + 1][0]): link(a, b)
    return adj


def mean_deg(adj): return sum(len(s) for s in adj) / len(adj)
def max_deg(adj): return max(len(s) for s in adj)


def timeit(adj, frame):
    """One canonicalization, garbage collector off.  Returns (seconds, result)."""
    gc.disable()
    t0 = time.perf_counter()
    r = canonical_frame(adj, frame)
    dt = time.perf_counter() - t0
    gc.enable()
    return dt, r


def stats(xs):
    """(mean, sample standard deviation) of a list of at least one value."""
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main(reps=REPS, arms=ARMS, hops_list=HOPS, budget=BUDGET):
    print(f"RGS chains, {reps} random frames per row, "
          f"budget {budget:g}s per canonicalization")
    for m in arms:
        print(f"\n### m={m} arms per side  ({4 * m} qubits per hop)")
        print(f"{'hops':>6}{'n':>7}{'d init':>8}{'trivial [ms]':>14}"
              f"{'random [ms]':>13}{'+-':>9}{'canon deg':>11}{'+-':>7}"
              f"{'max deg':>9}{'|F|':>7}")
        for hops in hops_list:
            adj = rgs_chain(hops, m); n = len(adj)
            triv = [ID_PAIR] * n                       # (w^C,w^N)=(+X,+Z): identity
            t_triv = sum(timeit(adj, triv)[0] for _ in range(reps)) / reps
            ts, degs, maxs, fs, over = [], [], [], [], False
            for rep in range(reps):
                rng = Random(f"rgs-{m}-{hops}-{rep}")   # str seed: stable across runs
                frame = [rng.choice(VALID_PAIRS) for _ in range(n)]
                dt, r = timeit(adj, frame)
                ts.append(dt); degs.append(mean_deg(r["adj"]))
                maxs.append(max_deg(r["adj"])); fs.append(r["hadamards"])
                if dt > budget:
                    over = True
                    break
            t_m, t_sd = stats(ts); d_m, d_sd = stats(degs)
            print(f"{hops:>6}{n:>7}{mean_deg(adj):>8.2f}{t_triv * 1e3:>14.2f}"
                  f"{t_m * 1e3:>13.2f}{t_sd * 1e3:>9.2f}{d_m:>11.2f}{d_sd:>7.2f}"
                  f"{sum(maxs) / len(maxs):>9.1f}{sum(fs) / len(fs):>7.1f}",
                  flush=True)
            if over:
                print(f"  stopped: one call took {ts[-1]:.1f}s, "
                      f"over the {budget:g}s budget")
                break


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=REPS,
                    help=f"random frames per row (default {REPS})")
    ap.add_argument("--arms", type=int, nargs="+", default=list(ARMS),
                    help="values of m to sweep")
    ap.add_argument("--hops", type=int, nargs="+", default=list(HOPS),
                    help="chain lengths to sweep")
    ap.add_argument("--budget", type=float, default=BUDGET,
                    help=f"seconds per canonicalization (default {BUDGET:g})")
    a = ap.parse_args()
    main(a.reps, a.arms, a.hops, a.budget)
