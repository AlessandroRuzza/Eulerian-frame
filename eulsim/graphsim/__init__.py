"""GraphSim-style backends: the same calculus, a different frame storage.

``eulsim`` proper stores each vertex frame as its Eulerian code (see
``eulsim.frames``).  This package keeps the Anders-Briegel picture instead —
a frame is a *vertex operator* (VOP), held as an actual single-qubit Clifford
— and reimplements the per-vertex rules on top of it.  Everything else is
shared: the graph layer (``eulsim.graph_ops``), the tableau
(``eulsim.tableau``), the coupled CZ block table and the defensive fallback
(``eulsim.gates``) are imported, not copied, so a measured difference between
the backends isolates the representation and nothing else.

Three storages, in increasing sympathy with the candidate:

``CliffordSim``     the VOP as a 2x2 complex matrix (8 floats).  This is the
                    literature "graph state + local Clifford" picture and the
                    representation eulsim itself used before the Eulerian
                    frame: every frame update is a complex matrix product,
                    every semantic read conjugates a Pauli through the matrix
                    or normalises it to a phase-canonical key.

``CliffordLUTSim``  the same 24 Cliffords, but the per-vertex state IS the
                    phase-canonical key and every rule is a memoised dict
                    lookup.  This applies the obvious rebuttal — "there are
                    only 24 of them, so anything you table-drive I can table-
                    drive too" — and isolates what remains: a native small-int
                    key versus an 8-tuple of rounded floats.

``CliffordIDSim``   the same 24 Cliffords under an *opaque* id 0..23 (BFS
                    discovery order, no relation to w^C/w^N), every table a
                    plain list indexed by it — matching the Eulerian backend's
                    storage shape and access pattern exactly.  What is left is
                    only that the Eulerian code is *decomposable*: axis and
                    sign of both stored Paulis come out by %3/%6 arithmetic on
                    the int itself, with no table at all, whereas an opaque id
                    must go through an explicit table to answer the same
                    question because it carries no structure to decode.

All three expose the same small interface as the Eulerian backend
(``apply_local``, ``reframe``, ``measure``, ``cz``, ``pair_codes``), so
``benchmarks/bench_frames.py`` can run them against each other and against
the live ``eulsim`` module op by op.
"""
from __future__ import annotations

from .sim import CliffordIDSim, CliffordLUTSim, CliffordSim

__all__ = ["CliffordSim", "CliffordLUTSim", "CliffordIDSim"]
