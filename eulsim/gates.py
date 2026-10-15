"""Two-qubit gate application: CZ/CX/CY on framed graph states.

The CZ rule of the Eulerian frame (thesis, Sec. "Two-qubit Clifford Gates"):
re-frame each operand until its Eulerian letter w^N lies on the Z axis, then
toggle the edge with a sign fold per -Z.  When an operand cannot be aligned
the state has a pure-Z stabilizer on the operands and CZ acts as a Pauli.
No look-up table and no tableau fallback.

Only the operands' neighbourhoods are ever touched; the case split is decided
by integer tests on the frame codes (``frames.is_zaxis``).
"""
from __future__ import annotations

from .frames import (
    LC_CENTER,
    LC_NEIGH,
    ZFOLD,
    is_zaxis,
    left_compose,
    parse_frame,
)
from .graph_ops import copy_adj, lc_inplace, toggle_edge


def _lc_step(adj: list[set[int]], v: int, f: list[int]) -> None:
    """One local complementation τ_v with the state-preserving frame update
    L_v ↦ L_v·HSH, L_u ↦ L_u·S† for u ∈ N(v) — the re-framing R_v⁻¹ = R_v³.
    At the centre it turns w^N_v about w^C_v; at each neighbour it turns
    w^C_u about w^N_u.  Mutates adj and f in place."""
    for u in adj[v]:
        f[u] = LC_NEIGH[f[u]]
    f[v] = LC_CENTER[f[v]]
    lc_inplace(adj, v)


def _zalign(adj: list[set[int]], a: int, f: list[int], other: int,
            may_use_other: bool) -> bool:
    """Bring w^N_a onto the Z axis by at most two re-framings, never centred
    at ``other`` unless ``may_use_other``.  A re-framing at a turns w^N_a
    about w^C_a, so one suffices unless w^C_a is itself on the Z axis; then a
    re-framing at a neighbour first moves w^C_a off it.  Returns False when
    that neighbour does not exist (the degenerate case)."""
    if is_zaxis(f[a]):
        return True
    if (f[a] // 6) % 3 == 2:                     # w^C_a on the Z axis
        nb = [u for u in adj[a] if u != other]
        if not nb and may_use_other and other in adj[a]:
            nb = [other]
        if not nb:
            return False
        _lc_step(adj, min(nb, key=lambda u: (len(adj[u]), u)), f)
    _lc_step(adj, a, f)                          # w^N_a ↦ the Z axis
    return True


def _cz_degenerate(adj: list[set[int]], i: int, j: int, f: list[int],
                   stuck: int) -> None:
    """CZ when ``stuck`` cannot be aligned: N(stuck) ⊆ {other operand} and
    w^C_stuck = εZ, so the generator K_stuck read off the frame is εZ_stuck
    (isolated) or εZ_stuck·w^N_other with w^N_other = ±Z.  By
    CZ = ½(I + Z_i + Z_j − Z_iZ_j), CZ acts as a Pauli on that state:
        +Z_s → I,   −Z_s → Z on the other,   +Z_iZ_j → Z_i,   −Z_iZ_j → I."""
    other = j if stuck == i else i
    neg = f[stuck] // 18 == 1                    # sign of w^C_stuck
    if adj[stuck]:                               # N(stuck) = {other}
        neg ^= f[other] % 6 == 5                 # sign of w^N_other
        if not neg:                              # +Z_iZ_j: CZ acts as Z_i
            f[i] = left_compose(f[i], "Z")
    elif neg:                                    # −Z_stuck: Z on the other
        f[other] = left_compose(f[other], "Z")


def apply_cz(adj: list[set[int]], n: int, i: int, j: int,
             frame: list | None = None) -> tuple[list[set[int]], list[int]]:
    """Apply a physical CZ_ij gate to |ψ⟩ = (⊗L_k)|G⟩:
      1. align w^N_i on the Z axis (``_zalign``, any neighbour as pivot),
         then w^N_j (never re-framing at i, which would turn w^N_i);
      2. if both are on the Z axis, toggle the edge (i, j) and flip w^C of
         the other operand for each w^N = −Z;
      3. otherwise CZ acts as a Pauli (``_cz_degenerate``).
    Returns (new_adj, new_frame)."""
    f = parse_frame(n, frame)
    if n == 0 or i == j or not (0 <= i < n and 0 <= j < n):
        return copy_adj(adj), f
    cur = copy_adj(adj)
    if not _zalign(cur, i, f, j, may_use_other=True):
        _cz_degenerate(cur, i, j, f, stuck=i)
        return cur, f
    if not _zalign(cur, j, f, i, may_use_other=False):
        _cz_degenerate(cur, i, j, f, stuck=j)
        return cur, f
    toggle_edge(cur, i, j)
    wni, wnj = f[i] % 6, f[j] % 6
    if wni == 5:                                 # w^N_i = -Z → fold Z at j
        f[j] = ZFOLD[f[j]]
    if wnj == 5:                                 # w^N_j = -Z → fold Z at i
        f[i] = ZFOLD[f[i]]
    return cur, f


def apply_controlled(adj: list[set[int]], n: int, i: int, j: int, gate: str,
                     frame: list | None = None
                     ) -> tuple[list[set[int]], list[int]]:
    """Apply a controlled gate between qubits i, j by conjugating CZ on the
    target j: gate = W_j · CZ_ij · W_j†.
      CZ : W = I
      CX : W = H        (CX = H CZ H, since H Z H = X)
      CY : W = SH       (CY = SH·CZ·(SH)†, since SH·Z·(SH)† = S X S† = Y)
    Returns (new_adj, new_frame)."""
    gate = (gate or "cz").lower()
    # W and W† as words for frames.left_compose (leftmost applied last).
    if gate == "cx":
        W, Wdag = ("H",), ("H",)
    elif gate == "cy":
        W, Wdag = ("H", "S"), ("SDG", "H")       # SH and its dagger H·S†
    else:                                        # cz
        W = Wdag = ()
    f = parse_frame(n, frame)
    if not (0 <= i < n and 0 <= j < n and i != j):
        return copy_adj(adj), f
    if Wdag:
        f[j] = left_compose(f[j], *Wdag)         # W†_j  on |ψ⟩
    new_adj, new_f = apply_cz(adj, n, i, j, f)   # CZ_ij
    if W:
        new_f[j] = left_compose(new_f[j], *W)    # W_j   on the result
    return new_adj, new_f
