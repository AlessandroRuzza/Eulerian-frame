"""Canonical frame by re-framing only (sec:track-canonical, sec:canon-algorithm).

The canonical frame of a stabilizer state (def:canonical-frame) is its
*restricted* frame (def:restricted-frame, w^N_v in {+Z,+X} for every v) whose
Hadamard support F = {v : w^N_v = +X} (def:hadamard-support) is shortlex-least
for the vertex order 0 < 1 < ... < n-1.  It exists and is unique, and it fixes
the graph as well (thm:gcf); it is the trivial frame exactly on graph states
(cor:canonical-graph-state).

This module computes it the way the thesis prescribes: on the frame itself,
with re-framings R_v and the pivot P_{u,v} = R_v R_u^-1 R_v (prop:pivot), never
building a check matrix.  Everything is exact integer/sign arithmetic on the
vertex basis (w^C_v, w^N_v) = (L_v X L_v†, L_v Z L_v†) (prop:dictionary), which
is all the algorithm ever reads.

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

from .cliffords import (
    _H_U8,
    _IDENTITY_U8,
    _S_U8,
    _clifford_key,
    _conj_pauli,
    _mat2x2_mul,
)

# ── Exact single-qubit Clifford arithmetic on the vertex basis ────────────────
# A frame letter is its vertex basis L = (w^C, w^N); a signed Pauli is
# (sign in {+1,-1}, letter in "XYZ").

_MUL = {  # P Q = i^k R
    ("X", "X"): (0, "I"), ("X", "Y"): (1, "Z"), ("X", "Z"): (3, "Y"),
    ("Y", "X"): (3, "Z"), ("Y", "Y"): (0, "I"), ("Y", "Z"): (1, "X"),
    ("Z", "X"): (1, "Y"), ("Z", "Y"): (3, "X"), ("Z", "Z"): (0, "I"),
}

P, M = 1, -1
ID = ((P, "X"), (P, "Z"))                     # identity frame
_U_W = ((P, "X"), (P, "Y"))                   # HS†H, right-composed at the centre of R_v
_U_S = ((P, "Y"), (P, "Z"))                   # S,    right-composed at u in N(v)
_U_X = ((P, "X"), (M, "Z"))                   # X,    right-composed at the centre of R_v^2
_U_Z = ((M, "X"), (P, "Z"))                   # Z,    right-composed at u in N(v)

# (f, d, s) of L_v = H^f S^d Z^s for the eight restricted letters (tab:restricted).
_FDS = {
    ((P, "X"), (P, "Z")): (0, 0, 0),          # I
    ((M, "X"), (P, "Z")): (0, 0, 1),          # Z
    ((P, "Y"), (P, "Z")): (0, 1, 0),          # S
    ((M, "Y"), (P, "Z")): (0, 1, 1),          # SZ
    ((P, "Z"), (P, "X")): (1, 0, 0),          # H
    ((M, "Z"), (P, "X")): (1, 0, 1),          # HZ
    ((M, "Y"), (P, "X")): (1, 1, 0),          # HS
    ((P, "Y"), (P, "X")): (1, 1, 1),          # HSZ
}


def _third(c: tuple, n: tuple) -> tuple:
    """i·c·n for anticommuting signed Paulis c, n (= L Y L† when c = w^C, n = w^N)."""
    k, r = _MUL[(c[1], n[1])]
    if r == "I":
        raise ValueError("w^C and w^N must lie on distinct axes")
    return (c[0] * n[0] * (-1 if k == 1 else 1), r)


def _img(L: tuple, p: str) -> tuple:
    """L p L† for p in "XYZ", given L = (w^C, w^N)."""
    c, n = L
    return c if p == "X" else (n if p == "Z" else _third(c, n))


def _rmul(L: tuple, U: tuple) -> tuple:
    """Right composition L -> L·U, with U given as (U X U†, U Z U†)."""
    ux, uz = U
    ix, iz = _img(L, ux[1]), _img(L, uz[1])
    return ((ux[0] * ix[0], ix[1]), (uz[0] * iz[0], iz[1]))


def _u8_table() -> dict:
    """(w^C, w^N) -> 8-float matrix, the bridge to the rest of eulsim."""
    seen = {_clifford_key(_IDENTITY_U8): _IDENTITY_U8}
    frontier = [_IDENTITY_U8]
    while frontier:
        nxt = []
        for m in frontier:
            for g in (_H_U8, _S_U8):
                m2 = _mat2x2_mul(m, g)
                k = _clifford_key(m2)
                if k not in seen:
                    seen[k] = m2
                    nxt.append(m2)
        frontier = nxt
    out = {(_conj_pauli(m, "X"), _conj_pauli(m, "Z")): m for m in seen.values()}
    if len(out) != 24:
        raise RuntimeError(f"Clifford table has {len(out)} entries, expected 24")
    return out


_U8 = _u8_table()


def frame_to_u8(L: tuple) -> list:
    """Vertex basis (w^C, w^N) -> 8-float matrix."""
    return _U8[L]


def u8_to_frame(m: list) -> tuple:
    """8-float matrix -> vertex basis (w^C, w^N)."""
    return (_conj_pauli(m, "X"), _conj_pauli(m, "Z"))


# ── The framed state and its two moves ────────────────────────────────────────

class FramedState:
    """(G, L) with G as adjacency sets and L as exact vertex bases."""

    def __init__(self, adj: list[set], frame: list[tuple]):
        self.adj = [set(s) for s in adj]
        self.L = list(frame)
        self.n = len(adj)
        # The Hadamard support is maintained incrementally: w^N moves only at
        # the centre of a move, so membership changes one vertex at a time.
        self.F: set = {v for v in range(self.n) if self.L[v][1] == (P, "X")}

    def reframe(self, v: int) -> None:
        """R_v: local complementation at v, frame updated so |psi> is unchanged."""
        nb = sorted(self.adj[v])
        for i, u in enumerate(nb):
            for w in nb[i + 1:]:
                if w in self.adj[u]:
                    self.adj[u].discard(w)
                    self.adj[w].discard(u)
                else:
                    self.adj[u].add(w)
                    self.adj[w].add(u)
        self.L[v] = _rmul(self.L[v], _U_W)
        self._sync(v)
        for u in nb:
            self.L[u] = _rmul(self.L[u], _U_S)

    def fold(self, v: int) -> None:
        """R_v^2 (prop:rv-square): absorb K_v into the frame, graph unchanged."""
        self.L[v] = _rmul(self.L[v], _U_X)
        self._sync(v)
        for u in self.adj[v]:
            self.L[u] = _rmul(self.L[u], _U_Z)

    def _sync(self, v: int) -> None:
        self.F.discard(v)
        if self.L[v][1] == (P, "X"):
            self.F.add(v)

    def wC(self, v): return self.L[v][0]
    def wN(self, v): return self.L[v][1]

    def restricted(self) -> bool:
        return all(self.L[v][1] in ((P, "Z"), (P, "X")) for v in range(self.n))


# ── Phases ────────────────────────────────────────────────────────────────────

def _restrict_vertex(st: FramedState, v: int) -> None:
    """Force w^N_v into {+Z,+X} with R_v, preferring +Z (prop:restricted-reachable).

    The R_v-orbit of w^N_v is {+-w^N_v, +-i w^C_v w^N_v}: every signed Pauli
    except +-w^C_v.  So w^C_v = +-X forces +Z (v stays out of F), w^C_v = +-Z
    forces +X (v is in F), and w^C_v = +-Y leaves both open — the free choice
    of phase 1.  w^C_v is invariant under R_v, so the outcome is decided before
    the first step.  A wrong sign costs one fold (prop:rv-square), not a pivot.
    """
    target = "X" if st.wC(v)[1] == "Z" else "Z"
    for _ in range(4):
        s, p = st.wN(v)
        if p == target:
            if s == M:
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
        free = [v for v in sorted(st.F) if st.wC(v)[1] == "Y"]
        if not free:
            return
        for v in free:
            if st.wN(v) == (P, "X") and st.wC(v)[1] == "Y":
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
        if not (st.wN(u) == (P, "X") and st.wN(v) == (P, "Z")):
            raise RuntimeError(f"transport {v} -> {u} did not move the Hadamard")
    raise RuntimeError("Hadamard transport did not terminate")


def canonical_frame(adj: list[set], frame: list[tuple]) -> dict:
    """Canonical frame of the framed state (G, L), by re-framing only.

    adj: adjacency sets; frame: vertex bases (w^C, w^N) per vertex.
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

    F = set(st.F)
    f, d, s = [0] * n, [0] * n, [0] * n
    for v in range(n):
        f[v], d[v], s[v] = _FDS[st.L[v]]
    return {"adj": st.adj, "frame": st.L, "F": F,
            "f": f, "d": d, "s": s, "hadamards": len(F)}
