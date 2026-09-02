#!/usr/bin/env python3
"""Running degree of a graph while it is being canonicalized.

Three experiments (notes/canonicalization-degree-dynamics.md):
  --traj       degree trajectory through the three phases of sec:canon-algorithm
  --attractor  end degree from several starting densities (two-sided fixed point)
  --threshold  end degree against starting degree, several n

Run from the repository root:
    .env/bin/python3 graph_states/benchmarks/canon_degree_dynamics.py --traj
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_frames as B
import eulsim.framecanon as FC


def cp(c):
    """bench int code 6*w^C + w^N -> ((sign, letter), (sign, letter))"""
    return tuple((1 if p < 3 else -1, "XYZ"[p % 3]) for p in (c // 6, c % 6))


def _run(st, n, on_reframe=None):
    """The three phases of canonical_frame, optionally traced."""
    orig = FC.FramedState.reframe
    if on_reframe is not None:
        def reframe(self, v):
            orig(self, v)
            on_reframe(self)
        FC.FramedState.reframe = reframe
    try:
        for v in range(n):
            FC._restrict_vertex(st, v)
        FC._minimize_support(st)
        FC._slide_down(st)
    finally:
        FC.FramedState.reframe = orig


def state(n, deg0, seed):
    rng = B.Random((n, deg0, seed).__hash__())
    adj, codes = B.random_state(rng, n, deg0)
    return FC.FramedState(adj, [cp(c) for c in codes])


def mean_deg(adj):
    return sum(len(s) for s in adj) / len(adj)


def traj(sizes=(100, 200, 400, 800), deg0=6.0, rows=14):
    for n in sizes:
        st = state(n, deg0, "traj")
        trace, phase = [], ["1 restrict"]
        def rec(s, phase=phase):
            trace.append((phase[0], mean_deg(s.adj), max(len(x) for x in s.adj), len(s.F)))
        orig_min, orig_slide = FC._minimize_support, FC._slide_down
        def minimize(s):
            phase[0] = "2 cancel"; orig_min(s)
        def slide(s):
            phase[0] = "3 slide"; orig_slide(s)
        FC._minimize_support, FC._slide_down = minimize, slide
        try:
            _run(st, n, rec)
        finally:
            FC._minimize_support, FC._slide_down = orig_min, orig_slide
        print(f"\n=== n={n}: {len(trace)} re-framings, "
              f"final mean degree {trace[-1][1]:.1f} ({trace[-1][1]/n:.3f} n)")
        print(f"{'move':>8}{'phase':>11}{'mean deg':>10}{'deg/n':>8}{'max':>7}{'|F|':>6}")
        step = max(1, len(trace) // rows)
        for i in range(0, len(trace), step):
            ph, md, mx, f = trace[i]
            print(f"{i+1:>8}{ph:>11}{md:>10.2f}{md/n:>8.3f}{mx:>7}{f:>6}")


def attractor(n=400, fracs=(0.005, 0.015, 0.1, 0.25, 0.5, 0.75, 0.9)):
    print(f"n={n}: end degree from several starting densities")
    print(f"{'start d/n':>11}{'end d/n':>10}{'min d/n':>10}{'max d/n':>10}")
    for fr in fracs:
        st = state(n, max(1.0, fr * n), "attr")
        seen = []
        _run(st, n, lambda s: seen.append(mean_deg(s.adj)))
        print(f"{fr:>11.3f}{seen[-1]/n:>10.3f}{min(seen)/n:>10.3f}{max(seen)/n:>10.3f}")


def threshold(sizes=(200, 400, 800), degs=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0), seeds=3):
    for n in sizes:
        print(f"\nn={n} ({seeds} seeds per starting degree)")
        print(f"{'d0':>6}{'end d':>10}{'end d/n':>10}")
        for d0 in degs:
            ends = []
            for s in range(seeds):
                st = state(n, d0, s)
                _run(st, n)
                ends.append(mean_deg(st.adj))
            m = sum(ends) / len(ends)
            print(f"{d0:>6.1f}{m:>10.1f}{m/n:>10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traj", action="store_true")
    ap.add_argument("--attractor", action="store_true")
    ap.add_argument("--threshold", action="store_true")
    a = ap.parse_args()
    if not (a.traj or a.attractor or a.threshold):
        ap.error("pick at least one of --traj, --attractor, --threshold")
    if a.traj: traj()
    if a.attractor: attractor()
    if a.threshold: threshold()
