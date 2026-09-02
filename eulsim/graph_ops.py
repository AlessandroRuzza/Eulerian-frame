"""Graph-level operations on framed graph states (G, L).

Local complementation tau_v, the state-preserving re-framing move
R_v: (G, L) -> (tau_v(G), L*U_v_dag), and Pauli measurements via the
reduction chain X -> Y -> Z (everything terminates in Z-deletion).

The graph is an *adjacency set* list: adj[v] is the set N(v), so every
primitive costs O(deg) or O(deg^2) rather than O(n) or O(n^2).  The vertex
count n is passed alongside because vertices with no neighbours are still
vertices (adj has one entry per vertex, so n == len(adj)).
"""
from __future__ import annotations

from .cliffords import (
    _HSDGH_U8,
    _HSH_U8,
    _IDENTITY_U8,
    _S_U8,
    _SDG_U8,
    _Z_U8,
    _conj_pauli,
    _dag_u8,
    _mat2x2_mul,
    _parse_mats,
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


def adj_from_matrix(matrix, n: int) -> list[set[int]]:
    """Adjacency sets from a dense 0/1 matrix (back-compatible input path)."""
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        row = matrix[i]
        for j in range(n):
            if i != j and (int(row[j]) & 1):
                adj[i].add(j); adj[j].add(i)
    return adj


def to_matrix(adj: list[set[int]], n: int) -> list[list[int]]:
    """Dense 0/1 matrix — for display and for tests only, never for compute."""
    m = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            m[i][j] = 1
    return m


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


def _delete_vertex(adj: list[set[int]], n: int, v: int
                   ) -> tuple[list[set[int]], list[int]]:
    kept = [i for i in range(n) if i != v]
    idx = {old: new for new, old in enumerate(kept)}
    return [{idx[u] for u in adj[o] if u != v} for o in kept], kept


def _finish_measure(
    adj_full: list[set[int]], n: int, v: int, mats: list,
    steps: list, delete: bool,
) -> tuple[list[set[int]], list[int], list, list]:
    """Final disposition of the measured vertex.
    adj_full is the adjacency *after* any local complementations, before v
    is dealt with. If delete, v is removed (standard rule). Otherwise v is kept
    but reset for reuse: its incident edges are stripped and its local frame is
    reset to identity (a fresh |+⟩), as for an emitter qubit re-initialised
    after measurement. Either way the other qubits' graph and byproduct frames
    are unchanged."""
    if delete:
        a, kept = _delete_vertex(adj_full, n, v)
        return a, kept, steps, [mats[i] for i in kept]
    a = copy_adj(adj_full)
    for u in a[v]:
        a[u].discard(v)
    a[v].clear()
    out_mats = [m[:] for m in mats]
    out_mats[v] = _IDENTITY_U8[:]
    return a, list(range(n)), steps, out_mats


def do_measure(
    adj: list[set[int]], n: int, v: int, basis: str,
    local_unitaries: list | None = None, delete: bool = True,
    invert: bool = False,
) -> tuple[list[set[int]], list[int], list, list]:
    """Pauli measurement on vertex v via the *reduction chain*
    (notes-EulVec-Alternative-XMeasure): every case terminates in the single
    destructive primitive, Z-deletion.

    Returns (new_adj, kept_original_indices, animation_steps, new_local_unitaries);
    new_local_unitaries[i] corresponds to kept[i].

    1. *Basis transport*: a lab measurement of P on qubit v equals measuring
       Q = L_v† P L_v = σ·Q' on the underlying |G⟩, with σ ∈ {±1}, Q' ∈ {X,Y,Z}.
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
    mats = _parse_mats(n, local_unitaries)
    cur = copy_adj(adj)
    steps: list = []

    # 1. Basis transport: L_v† P L_v = σ·Q on |G⟩.
    P_lab = {"x": "X", "y": "Y", "z": "Z"}[basis]
    sigma, Q = _conj_pauli(_dag_u8(mats[v]), P_lab)

    def _reframe(w: int) -> None:
        """R_w on (cur, mats) + conjugation of the pending basis (σ·Q) by the
        factor of U_w at v: HSH for v = w, S† for v ∈ N(w)."""
        nonlocal cur, sigma, Q
        fac = _HSH_U8 if v == w else (_SDG_U8 if v in cur[w] else None)
        new_adj, tog, new_mats = reframe_move(cur, n, w, mats)
        mats[:] = new_mats
        cur = new_adj
        steps.append({"op": "lc", "vertex": w, "pairs": tog})
        if fac is not None:
            s2, Q2 = _conj_pauli(fac, Q)
            sigma, Q = sigma * s2, Q2

    # 2. Reduction chain: X → Y → Z.
    if Q == "X":
        if not cur[v]:                   # isolated: deterministic, nothing changes
            return _finish_measure(cur, n, v, mats, steps, delete)
        # free choice: minimise |N(b)|
        b = min(cur[v], key=lambda j: (len(cur[j]), j))
        _reframe(b)                                   # pending ±X → ∓Y on τ_b(G)
    if Q == "Y":
        _reframe(v)                                   # pending ±Y → ±Z on τ_v(…)
    assert Q == "Z"

    # 3. Z-deletion. Eigenvalue of Z on the underlying graph state = σ·lab.
    lab_outcome = -1 if invert else 1
    if sigma * lab_outcome == -1:                     # correction: Z on N(v)
        for u in cur[v]:
            mats[u] = _mat2x2_mul(mats[u], _Z_U8)
    return _finish_measure(cur, n, v, mats, steps, delete)


def reframe_move(adj: list[set[int]], n: int, v: int,
                 local_unitaries: list | None
                 ) -> tuple[list[set[int]], list[list[int]], list[list[float]]]:
    """Re-framing move R_v: (G, L) ↦ (τ_v(G), L·U_v†), the state-preserving
    representation rewrite of the Eulerian-vector calculus.
    With U_v = (HSH)_v ⊗ (S†)_{N(v)} and U_v|G⟩ = |τ_v(G)⟩ (Van den Nest),
    (⊗L)|G⟩ = (⊗ L·U_v†)·U_v|G⟩ = (⊗ L·U_v†)|τ_v(G)⟩, so per qubit
    L_v ↦ L_v·(HS†H), L_u ↦ L_u·S for u ∈ N(v).
    On the signed vectors: w^N_v ↦ i·w^C_v·w^N_v (center, w^C_v fixed) and
    w^C_u ↦ i·w^C_u·w^N_u (neighbours, w^N_u fixed).
    Returns (new_adj, toggled_index_pairs, new_local_unitaries)."""
    mats = _parse_mats(n, local_unitaries)
    mats[v] = _mat2x2_mul(mats[v], _HSDGH_U8)
    for j in adj[v]:
        mats[j] = _mat2x2_mul(mats[j], _S_U8)
    new_adj, toggled = local_complement(adj, n, v)
    return new_adj, toggled, mats
