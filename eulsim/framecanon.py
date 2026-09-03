"""Canonical frame by re-framing only (sec:track-canonical, sec:canon-algorithm).

The canonical frame of a stabilizer state (def:canonical-frame) is its
*restricted* frame (def:restricted-frame, w^N_v in {+Z,+X} for every v) whose
Hadamard support F = {v : w^N_v = +X} (def:hadamard-support) is shortlex-least
for the vertex order 0 < 1 < ... < n-1.  It exists and is unique, and it fixes
the graph as well (thm:gcf); it is the trivial frame exactly on graph states
(cor:canonical-graph-state).

This module computes it the way the thesis prescribes: on the frame itself,
with re-framings R_v and the pivot P_{u,v} = R_v R_u^-1 R_v (prop:pivot), never
building a check matrix.  Everything is exact integer arithmetic on the vertex
basis (w^C_v, w^N_v) = (L_v X L_v†, L_v Z L_v†) (prop:dictionary), which is all
the algorithm ever reads — see ``frames`` for the encoding.

Primitives (both state-preserving, both O(deg)):

  R_v   re-framing: G -> tau_v(G), L_v -> L_v (HS†H), L_u -> L_u S for u in N(v)
        w^N_v -> i w^C_v w^N_v (only the centre's w^N moves), w^C_u -> i w^C_u w^N_u
  R_v^2 sign fold (prop:rv-square): the stabilizer K_v = X_v (x) Z_N(v) into the
        frame, w^N_v -> -w^N_v, w^C_u -> -w^C_u; the graph is unchanged

Phases:

  1. Restrict (prop:restricted-reachable).  Sweep v = 0..n-1 forcing w^N_v into
     {+Z,+X} with R_v, preferring +Z.  w^C_v is invariant under R_v and the
     R_v-orbit of w^N_v misses only +-w^C_v, so the reachable targets are
     forced when w^C_v is X- or Z-type and free when it is Y-type — the free
     choice is what keeps a vertex out of F when it may leave.
  2. Minimize |F|.  While an edge lives inside F, one pivot across it drops
     |F| by two (two Hadamards arriving at the same vertex cancel).  |F| is
     then the smallest any restricted frame of the state can carry.
  3. Move the Hadamards down (shortlex).  While some v in F has a neighbour
     u < v, transport its Hadamard along the edge with a pivot until no
     H-vertex has a smaller-indexed neighbour, which is the shortlex-least
     support.  Transport keeps G[F] edgeless, so the two phases run once each,
     in order (see canonical_frame).

Phases 2 and 3 use one primitive each time — R at one end, then restore the
restriction at both ends (phase 3 of sec:canon-algorithm) — which is the pivot
of prop:pivot.  Cost O(sum_v deg(v)^2) on the *running* degrees.

canonical.canonicalize wraps this for the HTTP API; canonical.canonicalize_rref
is the check-matrix route kept as a cross-check (tests/test_canonical_frame.py).
"""
from __future__ import annotations

from .frames import FDS, RV_CENTER, RV_NEIGH, XFOLD, ZFOLD, is_hadamard
from .graph_ops import lc_inplace

# Axis codes within a signed Pauli (see frames): X=0, Y=1, Z=2; +3 negates.
_X, _Y, _Z = 0, 1, 2


class FramedState:
    """(G, L) with G as adjacency sets and L as Eulerian frame codes."""

    def __init__(self, adj: list[set], frame: list[int]):
        self.adj = [set(s) for s in adj]
        self.f = list(frame)
        self.n = len(adj)
        # The Hadamard support is maintained incrementally: w^N moves only at
        # the centre of a move, so membership changes one vertex at a time.
        self.F: set = {v for v in range(self.n) if is_hadamard(self.f[v])}

    def reframe(self, v: int) -> None:
        """R_v: local complementation at v, frame updated so |psi> is unchanged."""
        f = self.f
        f[v] = RV_CENTER[f[v]]
        self._sync(v)
        for u in self.adj[v]:
            f[u] = RV_NEIGH[f[u]]
        lc_inplace(self.adj, v)

    def fold(self, v: int) -> None:
        """R_v^2 (prop:rv-square): absorb K_v into the frame, graph unchanged."""
        f = self.f
        f[v] = XFOLD[f[v]]
        self._sync(v)
        for u in self.adj[v]:
            f[u] = ZFOLD[f[u]]

    def _sync(self, v: int) -> None:
        self.F.discard(v)
        if is_hadamard(self.f[v]):
            self.F.add(v)

    def wC(self, v: int) -> int:
        return self.f[v] // 6

    def wN(self, v: int) -> int:
        return self.f[v] % 6


# ── Phases ────────────────────────────────────────────────────────────────────

def _restrict_vertex(st: FramedState, v: int) -> None:
    """Force w^N_v into {+Z,+X} with R_v, preferring +Z (prop:restricted-reachable).

    The R_v-orbit of w^N_v is {+-w^N_v, +-i w^C_v w^N_v}: every signed Pauli
    except +-w^C_v.  So w^C_v = +-X forces +Z (v stays out of F), w^C_v = +-Z
    forces +X (v is in F), and w^C_v = +-Y leaves both open — the free choice
    of phase 1.  w^C_v is invariant under R_v, so the outcome is decided before
    the first step.  A wrong sign costs one fold (prop:rv-square), not a pivot.
    """
    target = _X if st.wC(v) % 3 == _Z else _Z
    for _ in range(4):
        wn = st.wN(v)
        if wn % 3 == target:
            if wn >= 3:                       # wrong sign: one fold fixes it
                st.fold(v)
            return
        st.reframe(v)
    raise RuntimeError(f"restriction at vertex {v} did not converge")


def _drop_free(st: FramedState) -> None:
    """Take out of F every vertex free to leave it (w^C Y-type).

    Only the centre's w^N moves under R, so no vertex ever *enters* F here and
    |F| decreases monotonically; dropping v flips w^C at its neighbours, which
    can free them in turn, hence the loop."""
    while True:
        free = [v for v in sorted(st.F) if st.wC(v) % 3 == _Y]
        if not free:
            return
        for v in free:
            if is_hadamard(st.f[v]) and st.wC(v) % 3 == _Y:
                _restrict_vertex(st, v)


def _pivot(st: FramedState, u: int, v: int) -> None:
    """P_{u,v} (prop:pivot) as one R at u, then restore the restriction at both
    ends (step 3 of sec:canon-algorithm), lower end first."""
    st.reframe(u)
    _restrict_vertex(st, v)
    _restrict_vertex(st, u)


def _minimize_support(st: FramedState) -> None:
    """Phase 2: no edge inside F, i.e. |F| as small as any restricted frame of
    this state allows.  Both ends of such an edge are pinned (w^C Z-type); the
    pivot across it frees them and both Hadamards cancel."""
    for _ in range(2 * st.n + 16):
        _drop_free(st)
        edge = next(((u, v) for u in sorted(st.F) for v in sorted(st.adj[u] & st.F)),
                    None)
        if edge is None:
            return
        _pivot(st, *edge)
    raise RuntimeError("support minimization did not terminate")


def _slide_down(st: FramedState) -> None:
    """Phase 3: transport Hadamards down until no v in F has a neighbour u < v,
    which is the shortlex-least support of the fixed size reached in phase 2."""
    for _ in range(st.n * st.n + 16):
        bad = next(((v, u) for v in sorted(st.F)
                    for u in sorted(st.adj[v]) if u < v), None)
        if bad is None:
            return
        v, u = bad
        _pivot(st, u, v)
        if not (is_hadamard(st.f[u]) and st.wN(v) == _Z):
            raise RuntimeError(f"transport {v} -> {u} did not move the Hadamard")
    raise RuntimeError("Hadamard transport did not terminate")


def canonical_frame(adj: list[set], frame: list[int]) -> dict:
    """Canonical frame of the framed state (G, L), by re-framing only.

    adj: adjacency sets; frame: one Eulerian code per vertex.
    Returns {adj, frame, F, f, d, s, hadamards} with L_v = H^f_v S^d_v Z^s_v."""
    st = FramedState(adj, frame)
    n = st.n
    for v in range(n):                                   # 1. restrict
        _restrict_vertex(st, v)
    _minimize_support(st)                                # 2. cancel pairs
    _slide_down(st)                                      # 3. transport down
    # One pass each: no transport can put an edge back inside F.  Transporting
    # from v to u toggles only pairs with an end in N(v), and F avoids N(v)
    # once G[F] is edgeless; u arrives with N'[u] = N[v] (the neighbourhood
    # swap of prop:transport-locality), so it has no neighbour in F either.
    if any(st.adj[u] & st.F for u in st.F):
        raise RuntimeError("transport recreated an edge inside the Hadamard support")

    f, d, s = [0] * n, [0] * n, [0] * n
    for v in range(n):
        f[v], d[v], s[v] = FDS[st.f[v]]
    return {"adj": st.adj, "frame": st.f, "F": set(st.F),
            "f": f, "d": d, "s": s, "hadamards": len(st.F)}
