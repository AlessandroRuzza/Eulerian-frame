"""Canonicalization cost against n, at fixed average degree 6.

One canonicalization is one operation on the whole state, so the time is
directly comparable with the per-operation times of the local rules.
Stops as soon as a single call exceeds the wall-clock budget.
"""
import sys, time, gc
sys.path.insert(0, 'graph_states'); sys.path.insert(0, 'graph_states/benchmarks')
import bench_frames as B
from eulsim.framecanon import canonical_frame

BUDGET = 60.0          # seconds for one canonicalization
SIZES = [50, 80, 130, 200, 320, 500, 800, 1300, 2000, 3200]
DEG = 6.0

def code_to_pair(c):
    """bench int code 6*w^C + w^N  ->  ((sign, letter), (sign, letter))"""
    out = []
    for p in (c // 6, c % 6):
        out.append((1 if p < 3 else -1, "XYZ"[p % 3]))
    return tuple(out)

print(f"canonicalization, average degree {DEG:g}, budget {BUDGET:g}s per call")
print(f"{'n':>6}{'deg':>7}{'reps':>6}{'us/op':>14}{'ms/op':>11}{'+-%':>7}{'us/qubit':>11}")
for n in SIZES:
    reps = 10 if n <= 500 else (5 if n <= 1300 else 3)
    ts, degs, over = [], [], False
    for r in range(reps):
        rng = B.Random((r, n, DEG, "canon").__hash__())
        adj, codes = B.random_state(rng, n, DEG)
        frame = [code_to_pair(c) for c in codes]
        degs.append(B.mean_degree(adj))
        gc.disable()
        t0 = time.perf_counter()
        canonical_frame(adj, frame)
        dt = time.perf_counter() - t0
        gc.enable()
        ts.append(dt)
        if dt > BUDGET:
            over = True
            break
    m = sum(ts) / len(ts)
    sd = (sum((t - m) ** 2 for t in ts) / (len(ts) - 1)) ** 0.5 if len(ts) > 1 else 0.0
    print(f"{n:>6}{sum(degs)/len(degs):>7.2f}{len(ts):>6}"
          f"{m*1e6:>14.1f}{m*1e3:>11.2f}{100*sd/m:>6.1f}%{m*1e6/n:>11.2f}", flush=True)
    if over:
        print(f"  stopped: one call took {ts[-1]:.1f}s, over the {BUDGET:g}s budget")
        break
