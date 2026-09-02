"""Two-qubit gate application: CZ/CX/CY on framed graph states.

The local Anders-Briegel algorithm (quant-ph/0504117) in Eulerian-vector
form: frame reduction (remove_VOP) by re-framing moves, the diagonal-axis
edge toggle, and the brute-forced coupled two-qubit block table, with a
global tableau reduction as defensive fallback.
"""
from __future__ import annotations

import numpy as np

from .cliffords import (
    _H_U8,
    _HSH_U8,
    _IDENTITY_U8,
    _S_U8,
    _SDG_U8,
    _Z_U8,
    _clifford_key,
    _conj_pauli,
    _dag_u8,
    _mat2x2_mul,
    _parse_mats,
)
from .graph_ops import copy_adj, local_complement, set_edge, toggle_edge
from .tableau import _PAULI_MUL, _reduce_tableau, _tableau_from_state

# ── Frame reduction (Anders-Briegel remove_VOP, quant-ph/0504117) ─────────────
# The frame matrices L_i play the role of the paper's vertex operators (VOPs).
# A VOP is reduced to the identity by burning, factor by factor, its shortest
# decomposition into the generators √(-iX) ∝ HSH and √(iZ) ∝ S†:
#   · a √(-iX) factor is produced by a re-framing move at the vertex,
#   · a √(iZ) factor by one at a neighbour (the "swapping partner"),
# each applied with the state-preserving frame update of _lc_step
# (Corollary 1 of the paper). Used by the local CZ algorithm below.

_I_KEY = _clifford_key(_IDENTITY_U8)
# The paper's Z = {I, Z, S, S†}: the local Cliffords commuting with CZ.
_Z_SET_KEYS = frozenset(
    _clifford_key(m) for m in (_IDENTITY_U8, _Z_U8, _S_U8, _SDG_U8))


def _build_vop_decomp() -> dict:
    """Paper's decomposition look-up table: for each of the 24 local Cliffords
    (mod phase) a shortest word over {HSH ∝ √(-iX), S† ∝ √(iZ)} whose product
    is ∝ that Clifford: tbl[key(g1·g2·…·gk)] = "XZ…" with X ↦ HSH, Z ↦ S†."""
    from collections import deque
    tbl = {_I_KEY: ""}
    q = deque([(_IDENTITY_U8, "")])
    while q:
        m, w = q.popleft()
        for mv, g in (("X", _HSH_U8), ("Z", _SDG_U8)):
            m2 = _mat2x2_mul(m, g)
            k = _clifford_key(m2)
            if k not in tbl:
                tbl[k] = w + mv
                q.append((m2, w + mv))
    return tbl


_VOP_DECOMP = _build_vop_decomp()


def _lc_step(adj: list[set[int]], n: int, v: int, mats: list) -> list[set[int]]:
    """One local complementation τ_v with the state-preserving frame update
    L_v ↦ L_v·HSH, L_u ↦ L_u·S† for u ∈ N(v) — the *directed* re-framing move
    R_v⁻¹ = R_v³ (right-multiply U_v instead of U_v†; both directions preserve
    the state since U_v² = K_v stabilises |G⟩). Kept in this direction because
    the VOP decomposition words of _VOP_DECOMP burn {HSH, S†} letters.
    Mutates mats in place; returns the new adjacency."""
    for j in adj[v]:
        mats[j] = _mat2x2_mul(mats[j], _SDG_U8)
    mats[v] = _mat2x2_mul(mats[v], _HSH_U8)
    new_adj, _ = local_complement(adj, n, v)
    return new_adj


def _frame_rank(m: list[float]) -> int:
    """0: ∝ I.  1: in Z = {I,Z,S,S†} (commutes with CZ).  2: other."""
    k = _clifford_key(m)
    if k == _I_KEY:
        return 0
    return 1 if k in _Z_SET_KEYS else 2


def _reduce_vop_at(adj: list[set[int]], n: int, a: int, mats: list,
                   avoid: int | None = None) -> list[set[int]]:
    """remove_VOP at vertex a (Anders-Briegel / notes-EulVec-Rep-Operations
    Sec. "Frame reduction"): burn the frame's shortest decomposition over
    {√(-iX) ∝ HSH, √(iZ) ∝ S†} by re-framing moves — an X-letter by a move at
    a itself, a Z-letter by one at a swapping partner b ∈ N(a), preferring
    partners ≠ avoid. Every move is state-preserving. Mutates mats in place;
    returns the new adjacency."""
    word = _VOP_DECOMP.get(_clifford_key(_dag_u8(mats[a])))
    if word is None:
        return adj
    for mv in word:
        if mv == "X":                    # burn √(-iX): re-frame at a itself
            adj = _lc_step(adj, n, a, mats)
        else:                            # burn √(iZ): re-frame at a partner
            nb = [j for j in adj[a] if j != avoid]
            if not nb:
                nb = list(adj[a])
            if not nb:
                break                    # isolated vertex: no partner available
            b = min(nb, key=lambda j: (_frame_rank(mats[j]) == 0, j))
            adj = _lc_step(adj, n, b, mats)
    return adj


# ── The coupled two-qubit CZ block (the 2·24² table of Anders-Briegel) ────────
# When an operand's frame cannot be reduced away (its only neighbour is the
# other operand), the two operand pairs are coupled and the transition function
# is genuinely table-shaped (notes-EulVec-Rep-Operations, Remark 2). The table
# is brute-forced on two-qubit stabilizer states and cached.

_CZ_INDEX: tuple | None = None
_CZ_LOOKUP_CACHE: dict = {}


def _u8_to_np(m: list[float]):
    return np.array([[complex(m[0], m[1]), complex(m[2], m[3])],
                     [complex(m[4], m[5]), complex(m[6], m[7])]])


def _all_cliffords_u8() -> dict:
    """The 24 single-qubit Cliffords (mod phase) as 8-float matrices, keyed by
    _clifford_key, from a BFS over words in {H, S}."""
    from collections import deque
    out = {_clifford_key(_IDENTITY_U8): list(_IDENTITY_U8)}
    q = deque([list(_IDENTITY_U8)])
    while q:
        m = q.popleft()
        for g in (_H_U8, _S_U8):
            m2 = _mat2x2_mul(g, m)
            k = _clifford_key(m2)
            if k not in out:
                out[k] = m2
                q.append(m2)
    return out


def _state_key_2q(psi) -> tuple:
    """Phase-canonical key for a two-qubit state vector."""
    idx = next(k for k in range(4) if abs(psi[k]) > 1e-8)
    psi = psi * (abs(psi[idx]) / psi[idx])
    return tuple(round(x, 6) + 0.0 for c in psi for x in (c.real, c.imag))


def _build_cz_index() -> tuple:
    """Enumerate all framed two-qubit representatives (ζ, L_a, L_b) ↦ state
    (L_a ⊗ L_b)·CZ^ζ|++⟩ and index them by phase-canonical state key."""
    global _CZ_INDEX
    if _CZ_INDEX is None:
        cliffs = _all_cliffords_u8()
        plus = np.full(4, 0.5, dtype=complex)
        czm = np.diag([1, 1, 1, -1]).astype(complex)
        mats_np = {k: _u8_to_np(m) for k, m in cliffs.items()}
        index: dict = {}
        for zeta in (0, 1):
            base = czm @ plus if zeta else plus
            for ka, Ma in mats_np.items():
                for kb, Mb in mats_np.items():
                    psi = np.kron(Ma, Mb) @ base
                    index.setdefault(_state_key_2q(psi), []).append((zeta, ka, kb))
        _CZ_INDEX = (cliffs, index)
    return _CZ_INDEX


def _cz_block_lookup(zeta: int, key_a: tuple, key_b: tuple,
                     need_zplus_a: bool, need_zplus_b: bool):
    """Transition of the coupled block: find (ζ', L'_a, L'_b) with
    CZ·(L_a⊗L_b)·CZ^ζ|++⟩ ∝ (L'_a⊗L'_b)·CZ^{ζ'}|++⟩.
    Among the sign-redundant representatives prefer, per the paper's
    Constraint 1, outputs keeping w^N = +Z (frame ∈ {I,Z,S,S†}) on operands
    that still carry external CZ edges (need_zplus_*), so the new frame
    commutes back through those edges. Returns (ζ', mat_a, mat_b) or None."""
    ck = (zeta, key_a, key_b, need_zplus_a, need_zplus_b)
    if ck in _CZ_LOOKUP_CACHE:
        return _CZ_LOOKUP_CACHE[ck]
    cliffs, index = _build_cz_index()
    plus = np.full(4, 0.5, dtype=complex)
    czm = np.diag([1, 1, 1, -1]).astype(complex)
    base = czm @ plus if zeta else plus
    target = czm @ (np.kron(_u8_to_np(cliffs[key_a]), _u8_to_np(cliffs[key_b])) @ base)
    cands = index.get(_state_key_2q(target), [])
    best = None
    for z2, ka2, kb2 in cands:
        cost = ((1 if need_zplus_a and ka2 not in _Z_SET_KEYS else 0)
                + (1 if need_zplus_b and kb2 not in _Z_SET_KEYS else 0))
        entry = (cost, z2, ka2, kb2)
        if best is None or entry < best:
            best = entry
    res = None
    if best is not None and best[0] == 0:
        res = (best[1], list(cliffs[best[2]]), list(cliffs[best[3]]))
    _CZ_LOOKUP_CACHE[ck] = res
    return res


def _apply_cz_tableau(adj: list[set[int]], n: int, i: int, j: int,
                      mats: list) -> tuple[list[set[int]], list[list[float]]]:
    """Fallback CZ path: conjugate the stabilizer tableau by CZ_ij
    (X_i↦X_iZ_j, Y_i↦Y_iZ_j, Z_i↦Z_i, symmetric in j), then reduce globally."""
    tab = _tableau_from_state(adj, n, mats)
    for g in tab:                                # conjugate each generator by CZ_ij
        L = g[1]
        ai = L[i] in ("X", "Y")                  # i has X-component (anticommutes with Z_i)
        aj = L[j] in ("X", "Y")
        k = 0
        if aj:                                   # → append Z on qubit i: L[i]·Z
            kk, r = _PAULI_MUL[(L[i], "Z")]; k += kk; L[i] = r
        if ai:                                   # → append Z on qubit j: Z·L[j]
            kk, r = _PAULI_MUL[("Z", L[j])]; k += kk; L[j] = r
        if k % 4 == 2:
            g[0] = -g[0]
    new_adj, new_lu, _ = _reduce_tableau(tab, n)
    return new_adj, new_lu


def apply_cz(adj: list[set[int]], n: int, i: int, j: int,
             local_unitaries: list | None = None
             ) -> tuple[list[set[int]], list[list[float]]]:
    """Apply a physical CZ_ij gate to |ψ⟩ = (⊗L_k)|G⟩ by the *local* algorithm
    of Anders-Briegel Sec. III.2 in Eulerian-vector form
    (notes-EulVec-Rep-Operations):
      1-3. reduce the operand frames by state-preserving re-framing moves,
           choosing swapping partners among non-operand neighbours — skipped
           when the frame passes through CZ anyway (both operands with w^N on
           the Z axis, handled by step 4's sign folds, or a frame in the
           Z-set {I,Z,S,S†}), keeping the rewrite minimal;
      4.   if both Eulerian entries lie on the Z axis (w^N ∈ {±Z} — the signed
           extension of the paper's Z-set {I,Z,S,S†}), CZ is an edge toggle
           plus a Z-fold on the other operand for each -Z sign;
      5.   otherwise the operands form a coupled two-qubit block: apply the
           brute-forced 2·24² transition table.
    Unlike the previous global tableau reduction, only the operands'
    neighbourhoods are touched. Returns (new_adj, new_local_unitaries)."""
    if n == 0 or i == j or not (0 <= i < n and 0 <= j < n):
        return copy_adj(adj), _parse_mats(n, local_unitaries)
    mats = _parse_mats(n, local_unitaries)
    cur = copy_adj(adj)

    def has_others(a: int, b: int) -> bool:      # non-operand neighbours of a?
        return any(k != b for k in cur[a])

    # Commutation guard: skip a reduction when the operand frame passes
    # through CZ without breaking the later case split.
    #   · If *both* operands have w^N on the Z axis, step 4 applies directly
    #     and folds the ±Z signs, so neither needs reduction.
    #   · Otherwise the other operand may end in the coupled-block table
    #     (step 5), whose factorisation pulls this operand's frame through
    #     its external CZ edges — valid only for exact commutation, i.e.
    #     frames in the Z set {I, Z, S, S†}.
    # Stable under the other operand's reduction: this operand is never a
    # move site itself (the partner search avoids it), so it only ever
    # receives the neighbour factor S† — a right-multiplied diagonal, which
    # changes neither w^N nor Z-set membership.
    def _zaxis(a: int) -> bool:
        return _conj_pauli(mats[a], "Z")[1] == "Z"
    both_zaxis = _zaxis(i) and _zaxis(j)

    def commutes(a: int) -> bool:
        return both_zaxis or _clifford_key(mats[a]) in _Z_SET_KEYS

    if has_others(i, j) and not commutes(i):
        cur = _reduce_vop_at(cur, n, i, mats, avoid=j)
    if has_others(j, i) and not commutes(j):
        cur = _reduce_vop_at(cur, n, j, mats, avoid=i)
    if has_others(i, j) and not commutes(i):
        cur = _reduce_vop_at(cur, n, i, mats, avoid=j)   # step 2 may de-reduce i

    sa, pa = _conj_pauli(mats[i], "Z")           # w^N of the operands
    sb, pb = _conj_pauli(mats[j], "Z")
    if pa == "Z" and pb == "Z":                  # 4. diagonal-axis case
        toggle_edge(cur, i, j)
        if sa < 0:                               # w^N_i = -Z → fold Z at j
            mats[j] = _mat2x2_mul(mats[j], _Z_U8)
        if sb < 0:                               # w^N_j = -Z → fold Z at i
            mats[i] = _mat2x2_mul(mats[i], _Z_U8)
        return cur, mats

    res = _cz_block_lookup(                      # 5. coupled two-qubit block
        1 if j in cur[i] else 0, _clifford_key(mats[i]), _clifford_key(mats[j]),
        need_zplus_a=has_others(i, j), need_zplus_b=has_others(j, i))
    if res is None:                              # constraint unmet — defensive
        return _apply_cz_tableau(cur, n, i, j, mats)
    zeta2, ma2, mb2 = res
    set_edge(cur, i, j, zeta2)
    mats[i], mats[j] = ma2, mb2
    return cur, mats


def apply_controlled(adj: list[set[int]], n: int, i: int, j: int, gate: str,
                     local_unitaries: list | None = None
                     ) -> tuple[list[set[int]], list[list[float]]]:
    """Apply a controlled gate between qubits i, j by conjugating CZ on the
    target j: gate = W_j · CZ_ij · W_j†.
      CZ : W = I
      CX : W = H        (CX = H CZ H, since H Z H = X)
      CY : W = SH       (CY = SH·CZ·(SH)†, since SH·Z·(SH)† = S X S† = Y)
    Returns (new_adj, new_local_unitaries)."""
    gate = (gate or "cz").lower()
    if gate == "cx":
        W, Wd = _H_U8, _H_U8
    elif gate == "cy":
        W = _mat2x2_mul(_S_U8, _H_U8); Wd = _dag_u8(W)
    else:                                            # cz
        W = Wd = None
    mats = _parse_mats(n, local_unitaries)
    if not (0 <= i < n and 0 <= j < n and i != j):
        return copy_adj(adj), mats
    if W is not None:
        mats[j] = _mat2x2_mul(Wd, mats[j])           # W†_j  on |ψ⟩
    new_adj, new_lu = apply_cz(adj, n, i, j, mats)   # CZ_ij
    if W is not None:
        new_lu[j] = _mat2x2_mul(W, new_lu[j])        # W_j   on the result
    return new_adj, new_lu

