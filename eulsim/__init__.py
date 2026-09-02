"""eulsim (Eulerian Frame Simulator) - quantum graph-state simulator in Eulerian-vector form.

Concepts
--------
* Graph state |G> = prod_{(i,j) in E} CZ_ij |+>^(tensor n)
* Stabilizer generators  S_v = X_v (tensor) Z_N(v)
* Local complementation tau_v and the state-preserving re-framing move
  R_v: (G, L) -> (tau_v(G), L*U_v_dag) of the Eulerian-vector calculus
* Pauli measurements via the reduction chain X -> Y -> Z
* CZ/CX/CY by the local Anders-Briegel algorithm (quant-ph/0504117)
* LC-equivalence check via BFS over the LC orbit
* Canonical frame (tensor) H^f S^d Z^s |G'>, shortlex-least Hadamard support,
  computed with re-framings and pivots on the frame itself

Modules
-------
cliffords    2x2 Clifford matrix toolkit (8-float representation)
graph_ops    local complementation, re-framing move, Pauli measurement
tableau      stabilizer tableau: generators, reduction to graph form
framecanon   canonical frame by re-framing only (R_v, pivots)
canonical    canonical frame API + check-matrix cross-check
gates        CZ/CX/CY application (local algorithm + coupled block table)
statevector  dense state-vector expansion (display)
properties   graph-theoretic properties/tags
lc_orbit     LC-orbit BFS: equivalence, representative, orbit size
server       HTTP API handler
page         HTML page assembly (web/index.html)
cli          command-line entry point
"""
from __future__ import annotations

UI_VERSION = "1.4.0"

from .canonical import canonicalize, canonicalize_rref
from .gates import apply_controlled, apply_cz
from .graph_ops import do_measure, local_complement, reframe_move
from .lc_orbit import (
    MAX_BFS_STATES,
    NODE_LIMIT,
    lc_canonical,
    lc_equiv_labeled,
    lc_orbit_size,
)
from .properties import compute_properties
from .statevector import MAX_SV_QUBITS, compute_state_vector
from .tableau import compute_stabilizers

__all__ = [
    "UI_VERSION", "MAX_SV_QUBITS", "MAX_BFS_STATES", "NODE_LIMIT",
    "local_complement", "reframe_move", "do_measure",
    "compute_stabilizers", "canonicalize", "canonicalize_rref", "apply_cz", "apply_controlled",
    "compute_state_vector", "compute_properties",
    "lc_equiv_labeled", "lc_canonical", "lc_orbit_size",
]
