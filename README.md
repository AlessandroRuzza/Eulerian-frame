# Eulerian Frame Simulator
Open-source implementation of the Eulerian frame for MSc thesis "Quantum Error Correction for Quantum Networks"

A stabilizer state simulator built on the Eulerian frame: every qubit carries a
local Clifford stored as the *vertex basis* `(w^C_v, w^N_v) = (L_v X L_v†,
L_v Z L_v†)`, so the frame is read directly as a pair of signed Pauli letters
instead of an arbitrary index. The state is a graph plus that
frame, and every operation is a local rewrite of a closed neighbourhood.

Concretely each frame is **one integer** `6·w^C + w^N` in `0..35` (24 valid):
a signed Pauli packs into `0..5` as `axis + 3·sign_bit`. Every frame update is
then a single table read and every case split an integer test — is `w^N` on the
Z axis (`code % 6 in {2,5}`), is `L` in the Z-set `{I,Z,S,S†}` (`code % 6 == 2`),
is this vertex in the Hadamard support (`code % 6 == 0`). No matrices, no
phase canonicalisation. See `eulsim/frames.py`, whose rule tables are derived
from the 2x2 matrix toolkit at import and machine-checked against it over all
24 Cliffords.

It ships as an interactive web app: two side-by-side graph editors, live
stabilizers, state vector, canonicalization, and LC-equivalence checking.

## Running it

Needs Python 3.10+ and numpy (`pip install -r requirements.txt`).

```
source .env/bin/activate          # the venv lives in the repository root
python3 run_eulsim.py --port 8001
```

The launcher works from anywhere. `python3 -m eulsim --port 8001` is
equivalent but has to be run from root folder since that is where
the package is located.

Then open `http://127.0.0.1:8001/`. `--host` defaults to `0.0.0.0`, so the
startup banner also prints a LAN URL you can open from another machine.
`health_check.sh [port]` pings `/health` and tells you whether a server is up.

## What the app does

Two independent graph editors, A and B, sit side by side; the comparison
panels (LC-equivalence, state equality) run between them.

Per-qubit operations, all applied by clicking a vertex:

- **Local Clifford** — pick any of the 24; it lands in that vertex's frame.
- **Local complement τ_v** — the physical gate `U_v = HSH_v ⊗ S†_N(v)`. Edges
  are untouched, the state changes.
- **Re-framing R_v** — `(G, L) ↦ (τ_v(G), L·U_v†)`. Complements the
  neighbourhood *and* folds the compensating Clifford into the frame, so the
  physical state is unchanged and you are just moving between representatives.
- **Pauli measurement** in X, Y or Z, with two switches: delete the measured
  qubit (the textbook rule) or keep it, stripped and reset to `|+⟩` for reuse
  the way a photon emitter would; and force outcome 0 or 1. The reduction
  chain X → Y → Z is animated step by step.
- **CZ / CX / CY** between two clicked qubits, via the local Anders–Briegel
  algorithm.
- **Canonicalize** — rewrite to the canonical frame `⊗ H^f S^d Z^s |G'⟩` with
  shortlex-least Hadamard support. Because that form is unique and fixes the
  graph too, two states are equal iff their canonical forms match, which is
  what the state-equality panel uses.

Live panels: graph properties and adjacency matrix, stabilizer generators
`S_v = X_v ⊗ Z_N(v)`, supplementary and Eulerian vectors, dense state vector
(up to 10 qubits), and LC-orbit tools (canonical representative, orbit size,
equivalence between A and B). Presets cover the usual suspects — Bell, GHZ,
linear cluster, ring, complete graph — plus the repeater graph states and the
all-photonic and fusion constructions.

Each editor has a **↓ PNG** button that saves the shown Eulerian frame —
colour, label or ZX rendering — as a PNG cropped to the drawing.
Vertices can be relabelled, panels reordered by dragging, and the frame shown in the dual basis.

## HTTP API

Everything the frontend does goes through JSON POSTs; the server is stateless,
so each request carries the whole graph and gets the whole updated state back.

| endpoint | does |
| --- | --- |
| `POST /api/compute` | stabilizers, state vector and properties of a state |
| `POST /api/reframe` | re-framing move `R_v` (alias `/api/local_complement`) |
| `POST /api/cz` | apply CZ, CX or CY between two qubits |
| `POST /api/measure` | Pauli measurement, with `delete` and `invert` flags |
| `POST /api/canonicalize` | canonical frame, plus the residual corrections that would collapse it to a bare graph state |
| `POST /api/state_equal` | are A and B the same physical state? |
| `POST /api/lc_canonical` | LC-orbit representative and orbit size |
| `POST /api/lc_equiv` | are A and B LC-equivalent as labelled graphs? |
| `GET /health` | `{"ok": true}` |

A graph on the wire is `{"n": ..., "edges": [[i, j], ...]}`, optionally with
`"labels"` and `"local_unitaries"` — one 8-float list `[re00, im00, re01,
im01, re10, im10, re11, im11]` per qubit, defaulting to the identity.
Responses return `new_edges` in the same shape. The 8-float form is what the
page's own Clifford arithmetic speaks; `server.py` converts to and from frame
codes at the boundary and nothing below it sees a matrix.

`/api/reframe` is a JSON-API surface only: the page computes `R_v` client-side
and never calls it.

Limits: 64 qubits per request, 10 qubits for the state-vector display, and the
LC-orbit BFS stops at 20 vertices or 60000 visited states (both overridable
per request via `node_limit` and `max_bfs_states`).

## Graph representation

Adjacency sets everywhere the code computes (`adj[v] == N(v)`), edge lists
everywhere it transports. Local moves therefore cost `O(deg)` or `O(deg²)`
rather than `O(n)` or `O(n²)`, and a request/response is `O(n + m)` rather
than `O(n²)`. The only dense matrix built anywhere is the adjacency-matrix
display panel.

## Layout

```
run_eulsim.py         thin launcher (equivalent to python3 -m eulsim)
eulsim/
  frames.py           the Eulerian frame encoding: codes, rule tables, conversions
  cliffords.py        2x2 Clifford matrix toolkit — reference definition of the group
  graph_ops.py        local complementation, re-framing move, measurements
  tableau.py          stabilizer tableau: generators, reduction, UI report
  framecanon.py       canonical frame by re-framing only (R_v, pivots)
  canonical.py        canonical frame API + check-matrix cross-check
  gates.py            CZ/CX/CY: local Anders-Briegel algorithm + block table
  statevector.py      dense state-vector expansion (display)
  properties.py       graph-theoretic properties/tags
  lc_orbit.py         LC-orbit BFS: equivalence, representative, orbit size
  sim.py              in-place stateful simulator (operation streams)
  server.py           HTTP handler + JSON API endpoints
  page.py             HTML page assembly
  web/index.html      frontend (HTML/CSS/JS)
  cli.py              argument parsing + server startup
  graphsim/           VOP-storage backends (the comparison baseline)
    tables.py         the same rules over Clifford keys and opaque ids
    sim.py            CliffordSim / CliffordLUTSim / CliffordIDSim
tests/                correctness checks
benchmarks/           scaling measurements
```

`eulsim/__init__.py` re-exports the compute layer, so the package is usable as
a library without ever starting the server.

### The two representations

`eulsim` stores a frame as an Eulerian code; `eulsim.graphsim` keeps the
Anders-Briegel picture instead, where a frame is a *vertex operator* held as an
actual single-qubit Clifford — as a 2x2 matrix (`CliffordSim`), as a
phase-canonical key (`CliffordLUTSim`), or under an opaque id 0..23
(`CliffordIDSim`). The graph layer, the coupled CZ block table and the tableau
fallback are imported from the core rather than copied, so what differs between
the backends is exactly the frame representation. `bench_frames.py --selftest`
runs all four against the functional core op by op.

`cliffords.py` stays in the core because it is where the group is *defined*:
`frames.py` derives its integer rules from it and verifies them against it, the
state-vector display needs real amplitudes, and `graphsim` is built on matrices
by construction. Nothing on the compute path multiplies one.

## Tests and benchmarks

From the repository root:

```
./py_compile_test.sh                              # syntax
.env/bin/python3 tests/test_canonical_frame.py    # canonical frame
.env/bin/python3 benchmarks/bench_frames.py --selftest   # all backends agree
```

The canonical-frame test draws random framed states and checks that the output
is the same physical state up to global phase, that the frame is restricted,
that the Hadamard support is genuinely shortlex-least (verified directly on
the check matrix), and that it agrees with the independent check-matrix route
`canonicalize_rref`. Optional arguments set the number of trials and the
maximum number of qubits.

Benchmarks live in `benchmarks/` and are run the same way:

| script | measures |
| --- | --- |
| `bench_frames.py` | per-operation cost across the four frame representations |
| `canon_scaling.py` | canonicalization against `n` at fixed average degree |
| `canon_degree_dynamics.py` | how the running degree moves during canonicalization |
| `canon_rgs.py` | canonicalization of repeater-graph-state chains |
| `pivot_canonical.py` | incremental re-canonicalization after each operation |
| `plots_ch5.py` | renders the chapter-5 figures from the measured numbers |

`bench_frames.py --selftest` is the cross-check that keeps the backends honest:
it runs `clifford`, `cliffordlut`, `cliffordid`, `euler` and the functional
`eulsim` API over the same random operation stream and asserts the graph and
the frame agree after every single op.
