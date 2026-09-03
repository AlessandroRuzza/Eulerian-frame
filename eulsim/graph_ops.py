"""Graph-level operations on framed graph states (G, L).

Local complementation tau_v, the state-preserving re-framing move
R_v: (G, L) -> (tau_v(G), L*U_v_dag), and Pauli measurements via the
reduction chain X -> Y -> Z (everything terminates in Z-deletion).

The graph is an *adjacency set* list: adj[v] is the set N(v), so every
primitive costs O(deg) or O(deg^2) rather than O(n) or O(n^2).  The vertex
count n is passed alongside because vertices with no neighbours are still
vertices (adj has one entry per vertex, so n == len(adj)).

The frame is a list of Eulerian codes (see ``frames``): one small int per
vertex, so every frame update below is a single list read.
"""
from __future__ import annotations

from .frames import (
    AXES,
    DAG,
    ID_PAIR,
    PEND_HSH,
    PEND_SDG,
    RV_CENTER,
    RV_NEIGH,
    ZFOLD,
    image,
    parse_frame,
)

# ─── adjacency-set helpers ────────────────────────────────────────────────────

def copy_adj(adj: list[set[int]]) -> list[set[int]]:
    """Independent copy of an adjacency-set list."""
    return [set(s) for s in adj]


def toggle_edge(adj: list[set[int]], i: int, j: int) -> None:
    """Flip the edge {i,j} in place."""
    if j in adj[i]:
        adj[i].discard(j); adj[j].discard(i)
    else:
        adj[i].add(j); adj[j].add(i)


def set_edge(adj: list[set[int]], i: int, j: int, bit: int) -> None:
    """Force the edge {i,j} to `bit` in place."""
    if bit:
        adj[i].add(j); adj[j].add(i)
    else:
        adj[i].discard(j); adj[j].discard(i)


def edge_list(adj: list[set[int]], n: int) -> list[list[int]]:
    """Sorted [i, j] pairs with i < j (the wire format)."""
    return sorted([i, j] for i in range(n) for j in adj[i] if i < j)


def adj_from_edges(n: int, edges) -> list[set[int]]:
    """Adjacency sets from an [[i, j], ...] edge list; self-loops dropped."""
    adj: list[set[int]] = [set() for _ in range(n)]
    for e in edges:
        i, j = int(e[0]), int(e[1])
        if not (0 <= i < n and 0 <= j < n) or i == j:
            continue
        adj[i].add(j); adj[j].add(i)
    return adj


# ─── graph operations ─────────────────────────────────────────────────────────

def local_complement(adj: list[set[int]], n: int, v: int
                     ) -> tuple[list[set[int]], list[list[int]]]:
    """τ_v: invert edges in the induced subgraph on N(v).  O(deg(v)^2).
    Returns (new_adj, toggled_index_pairs)."""
    a = copy_adj(adj)
    nb = sorted(adj[v])
    toggled: list[list[int]] = []
    for i, u in enumerate(nb):
        for w in nb[i + 1:]:
            toggle_edge(a, u, w)
            toggled.append([u, w])
    return a, toggled


def lc_inplace(adj: list[set[int]], v: int) -> None:
    """τ_v in place, no allocation and no toggle list — the hot path."""
    nb = sorted(adj[v])
    for i, u in enumerate(nb):
        au = adj[u]
        for w in nb[i + 1:]:
            if w in au:
                au.discard(w); adj[w].discard(u)
            else:
                au.add(w); adj[w].add(u)


def reframe_move(adj: list[set[int]], n: int, v: int, frame: list | None
                 ) -> tuple[list[set[int]], list[list[int]], list[int]]:
    """Re-framing move R_v: (G, L) ↦ (τ_v(G), L·U_v†), the state-preserving
    representation rewrite of the Eulerian-vector calculus.
    With U_v = (HSH)_v ⊗ (S†)_{N(v)} and U_v|G⟩ = |τ_v(G)⟩ (Van den Nest),
    (⊗L)|G⟩ = (⊗ L·U_v†)·U_v|G⟩ = (⊗ L·U_v†)|τ_v(G)⟩, so per qubit
    L_v ↦ L_v·(HS†H), L_u ↦ L_u·S for u ∈ N(v).
    On the vertex bases: w^N_v ↦ i·w^C_v·w^N_v (only the centre's w^N moves)
    and w^C_u ↦ i·w^C_u·w^N_u at the neighbours (their w^N fixed).
    Returns (new_adj, toggled_index_pairs, new_frame)."""
    f = parse_frame(n, frame)
    f[v] = RV_CENTER[f[v]]
    for j in adj[v]:
        f[j] = RV_NEIGH[f[j]]
    new_adj, toggled = local_complement(adj, n, v)
    return new_adj, toggled, f


def _delete_vertex(adj: list[set[int]], n: int, v: int
                   ) -> tuple[list[set[int]], list[int]]:
    kept = [i for i in range(n) if i != v]
    idx = {old: new for new, old in enumerate(kept)}
    return [{idx[u] for u in adj[o] if u != v} for o in kept], kept


def _finish_measure(
    adj_full: list[set[int]], n: int, v: int, f: list[int],
    steps: list, delete: bool,
) -> tuple[list[set[int]], list[int], list, list[int]]:
    """Final disposition of the measured vertex.
    adj_full is the adjacency *after* any local complementations, before v
    is dealt with. If delete, v is removed (standard rule). Otherwise v is kept
    but reset for reuse: its incident edges are stripped and its local frame is
    reset to identity (a fresh |+⟩), as for an emitter qubit re-initialised
    after measurement. Either way the other qubits' graph and byproduct frames
    are unchanged."""
    if delete:
        a, kept = _delete_vertex(adj_full, n, v)
        return a, kept, steps, [f[i] for i in kept]
    a = copy_adj(adj_full)
    for u in a[v]:
        a[u].discard(v)
    a[v].clear()
    out = list(f)
    out[v] = ID_PAIR
    return a, list(range(n)), steps, out


def do_measure(
    adj: list[set[int]], n: int, v: int, basis: str,
    frame: list | None = None, delete: bool = True,
    invert: bool = False,
) -> tuple[list[set[int]], list[int], list, list[int]]:
    """Pauli measurement on vertex v via the *reduction chain*
    (notes-EulVec-Alternative-XMeasure): every case terminates in the single
    destructive primitive, Z-deletion.

    Returns (new_adj, kept_original_indices, animation_steps, new_frame);
    new_frame[i] corresponds to kept[i].

    1. *Basis transport*: a lab measurement of P on qubit v equals measuring
       Q = L_v† P L_v on the underlying |G⟩ — one read of the vertex basis of
       L_v† (a signed Pauli: sign σ ∈ {±1}, axis Q' ∈ {X,Y,Z}).
    2. *Reduction*: re-framing moves R_w rewrite the representative while the
       pending basis conjugates by the factor of U_w at v (Lemma "re-framing
       transport"):
         Q' = X → R_b at any b ∈ N(v):  (S†)X(S†)† = -Y   (pending → ∓Y),
         Q' = Y → R_v:                  (HSH)Y(HSH)† = Z  (pending → ±Z).
       Each move folds U_w† into the frames; no X/Y-specific byproduct formula
       (pivot, √(iY)) is needed. The X case lands on τ_v(τ_b(G)) — one R_b away
       from the standard pivot representative τ_b(τ_v(τ_b(G))).
    3. *Z-deletion*: delete v; the eigenvalue of the final ±Z on the underlying
       graph state is σ·(lab outcome); when -1, fold the correction byproduct
       Z on N(v).

    The reported lab outcome is +1 (result 0) by default, or -1 (result 1) when
    ``invert``. Exception: Q' = X on an isolated vertex is deterministic and
    nothing changes.

    delete=True (default) removes the measured vertex per the standard
    graph-state rules; delete=False keeps it but resets it for reuse (see
    _finish_measure).
    """
    f = parse_frame(n, frame)
    cur = copy_adj(adj)
    steps: list = []

    # 1. Basis transport: L_v† P L_v, a signed Pauli code p = axis + 3·sign_bit.
    p = image(DAG[f[v]], AXES.index({"x": "X", "y": "Y", "z": "Z"}[basis]))

    def _reframe(w: int) -> None:
        """R_w on (cur, f) + conjugation of the pending basis by the factor of
        U_w at v: HSH for v = w, S† for v ∈ N(w)."""
        nonlocal cur, p
        tab = PEND_HSH if v == w else (PEND_SDG if v in cur[w] else None)
        f[w] = RV_CENTER[f[w]]
        for u in cur[w]:
            f[u] = RV_NEIGH[f[u]]
        new_adj, tog = local_complement(cur, n, w)
        cur = new_adj
        steps.append({"op": "lc", "vertex": w, "pairs": tog})
        if tab is not None:
            p = tab[p]

    # 2. Reduction chain: X → Y → Z.
    if p % 3 == 0:                       # pending on the X axis
        if not cur[v]:                   # isolated: deterministic, nothing changes
            return _finish_measure(cur, n, v, f, steps, delete)
        b = min(cur[v], key=lambda j: (len(cur[j]), j))   # free choice: min |N(b)|
        _reframe(b)                                       # ±X → ∓Y on τ_b(G)
    if p % 3 == 1:                       # pending on the Y axis
        _reframe(v)                                       # ±Y → ±Z on τ_v(…)
    assert p % 3 == 2, p

    # 3. Z-deletion. Eigenvalue of Z on the underlying graph state = σ·lab.
    if (p // 3) ^ (1 if invert else 0):                    # correction: Z on N(v)
        for u in cur[v]:
            f[u] = ZFOLD[f[u]]
    return _finish_measure(cur, n, v, f, steps, delete)
