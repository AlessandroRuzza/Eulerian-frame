# Canonicalization: running degree, densification, and what it costs

Measured 2026-08-28, CPython 3.12 single-threaded, x86-64.
Scripts (run from the repository root with `.env/bin/python3`):
`graph_states/benchmarks/canon_scaling.py` (section 1),
`graph_states/benchmarks/canon_degree_dynamics.py --traj --attractor --threshold`
(sections 2 and 3), `graph_states/benchmarks/canon_rgs.py` (section 4).
Not thesis text — working notes.

## The claim under test

`subsec:cost-canonical` concludes that canonicalization costs `O(n d^2)`,
"which is `O(n^3)` on a dense graph and `O(n)` on a graph of bounded degree".

The `d` in that bound is the **running** degree, not the input degree —
`framecanon`'s own docstring says so ("`O(sum_v deg(v)^2)` on the *running*
degrees"). Canonicalization moves Hadamards with pivots, a pivot is three
re-framings, and each re-framing local-complements a neighbourhood. So the
question is whether a sparse input stays sparse. Usually it does not.

## 1. Random graphs at degree 6: the input degree is not preserved

One canonicalization per freshly drawn random state, average degree 6:

| n | time | per qubit | mean degree after |
|---|---|---|---|
| 50 | 1.89 ms | 37.9 us | 19.2 |
| 130 | 20.91 ms | 160.9 us | 60.7 |
| 320 | 295.55 ms | 923.6 us | 145.6 |
| 800 | 5.17 s | 6456.6 us | 356.7 |
| 1300 | 23.47 s | 18054.8 us | |
| 2000 | 108.76 s | 54379.0 us | |

Fitted exponent 2.99 overall, 3.32 over the top three octaves, local exponent
climbing 2.22 -> 3.56. The graph is left at degree ~0.45n, so `d = Theta(n)`
and `O(n d^2)` is `O(n^3)`. The measurement matches the bound; what fails is
reading `d` as the input degree.

## 2. The running degree has a two-sided fixed point at n/2

n = 400, varying the starting density:

| start d/n | end d/n | min seen | max seen |
|---|---|---|---|
| 0.005 | 0.023 | 0.005 | 0.048 |
| 0.015 | 0.433 | 0.015 | 0.436 |
| 0.097 | 0.500 | 0.104 | 0.503 |
| 0.252 | 0.501 | 0.280 | 0.503 |
| 0.500 | 0.498 | 0.494 | 0.504 |
| 0.750 | 0.499 | 0.433 | 0.504 |
| 0.900 | 0.498 | 0.221 | 0.672 |

Dense starts come **down**, sparse starts go **up**, `d = n/2` is stationary.

Reason: a re-framing complements the subgraph induced on `N(v)`, and
`G(n, 1/2)` is invariant under complementing any induced subgraph — each edge
is present with probability 1/2 before and after. So `p = 1/2` is the
stationary distribution of the process the algorithm drives.

The trajectory reaches the fixed point during phase 1 (restrict) and then
holds: at n=800 phase 2 performs ~240 further re-framings while the mean
degree sits at 0.41n -> 0.46n and `|F|` drains 237 -> 2.

## 3. There is a threshold, at mean degree ~1-2

End degree / n, three seeds per cell:

| start d0 | n=200 | n=400 | n=800 |
|---|---|---|---|
| 1.0 | 0.007 | 0.003 | 0.002 |
| 1.5 | 0.014 | 0.008 | 0.005 |
| 2.0 | 0.040 | 0.024 | 0.011 |
| 3.0 | 0.144 | 0.167 | 0.068 |
| 4.0 | 0.304 | 0.291 | 0.307 |
| 6.0 | 0.453 | 0.454 | 0.450 |

Below d0 ~ 1.5 the end degree is O(1) in absolute terms (1.3, 1.5, 1.6 at
n=800) — genuinely bounded, and this is the regime where the `O(n)` reading of
`subsec:cost-canonical` is correct. The crossover sits at the emergence of the
giant component in `G(n,p)` (mean degree 1): below it the graph is a scattered
forest and a re-framing at a vertex of degree <= 2 toggles at most one edge, so
nothing propagates; above it complementations cascade and the density runs to
the fixed point.

**Unexplained**: the intermediate rows (d0 = 3..5) settle at a stable fraction
strictly below 0.5, consistent across n. Plausibly the algorithm halts before
the process mixes (phase 1 is only ~n re-framings), but this was not tested.

## 4. RGS chains stay sparse — canonicalization is linear on them

Chain of merged RGSs (`res:rgs-merge`): repeater = complete bipartite `K_{m,m}`
on the inner qubits, one outer arm per inner qubit; adjacent repeaters joined
arm-to-arm. 4m qubits per hop. Random frame on the chain (a trivially framed
graph state is already canonical by `cor:canonical-graph-state`, and measures
~1 ms even at n=1536 — no re-framings happen at all).

m = 3, mean of 5 reps:

| hops | n | time | per qubit | mean deg after | max deg after | \|F\|/n |
|---|---|---|---|---|---|---|
| 4 | 48 | 0.53 ms | 10.9 us | 6.8 | 25 | 0.104 |
| 16 | 192 | 2.60 ms | 13.5 us | 10.2 | 33 | 0.078 |
| 64 | 768 | 10.75 ms | 14.0 us | 9.5 | 35 | 0.060 |
| 256 | 3072 | 53.14 ms | 17.3 us | 9.1 | 44 | 0.062 |

m = 5, mean of 5 reps:

| hops | n | time | per qubit | mean deg after | max deg after | \|F\|/n |
|---|---|---|---|---|---|---|
| 4 | 80 | 1.29 ms | 16.1 us | 6.7 | 27 | 0.075 |
| 16 | 320 | 5.27 ms | 16.5 us | 12.2 | 48 | 0.066 |
| 64 | 1280 | 23.94 ms | 18.7 us | 10.1 | 65 | 0.059 |
| 256 | 5120 | 140.33 ms | 27.4 us | 9.5 | 44 | 0.067 |

Fitted exponent **1.11** (m=3) and **1.12** (m=5) — linear, the residue being
the same memory term the local rules pay. The degree after canonicalization is
**flat in n**: ~9-10 mean and ~40 max whether the chain is 4 hops or 256.
`|F|` is ~6% of n throughout.

Why the difference from a random graph of comparable mean degree (3-4, which
in the random family already densifies to 0.3n): the chain is a sequence of
small dense blobs (`K_{m,m}`, m = 3 or 5) joined by degree-2 arm paths. A local
complementation inside a blob stays inside that blob, and the arms are too thin
to carry the cascade between blobs. **Structure, not mean degree, decides**.

## Consequences for the thesis

1. `subsec:cost-canonical`'s "`O(n)` on a graph of bounded degree" needs the
   bound to be on the running degrees, and should say so. As written it invites
   reading `d` as the input degree, which is wrong for every random graph above
   the threshold — including the degree-6 states of `tab:bench-n`.
2. The empirical section's canonicalization sweep measures `n^3`, which is not
   a contradiction of the model but an instance of it.
3. The regime the architecture actually cares about is the good one: on RGS
   chains canonicalization is linear and the degree never leaves O(1). That is
   a stronger practical statement than the bounded-degree sentence currently
   makes, and it is worth stating in terms of the resource states of
   `sec:all-photonic` rather than in terms of an abstract degree bound.

## Caveats

- One machine, CPython, single thread; absolute times are not portable.
- RGS chains tested only for m = 3, 5 and the arm-to-arm linking model above;
  a different fusion model may behave differently.
- Random-graph rows at n >= 1300 use 5 and 1 repetitions respectively — the
  n=2000 point is a single 109 s call.
