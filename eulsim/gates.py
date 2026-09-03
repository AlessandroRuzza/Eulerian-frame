"""Two-qubit gate application: CZ/CX/CY on framed graph states.

The local Anders-Briegel algorithm (quant-ph/0504117) in Eulerian-frame form:
frame reduction (their remove_VOP) by re-framing moves, the diagonal-axis
edge toggle, and the brute-forced coupled two-qubit block table, with a
global tableau reduction as defensive fallback.

Only the operands' neighbourhoods are ever touched; the case split is decided
by integer tests on the frame codes (``frames.is_zaxis`` / ``frames.in_zset``).
"""
from __future__ import annotations

from .frames import (
    DAG,
    DECOMP,
    ID_PAIR,
    LC_CENTER,
    LC_NEIGH,
    PAIR_TO_MAT,
    VALID_PAIRS,
    ZFOLD,
    in_zset,
    is_zaxis,
    left_compose,
    parse_frame,
)
from .graph_ops import copy_adj, lc_inplace, set_edge, toggle_edge
from .tableau import _PAULI_MUL, _reduce_tableau, _tableau_from_state

# ── Frame reduction (Anders-Briegel remove_VOP, quant-ph/0504117) ─────────────
# The frame letters L_i play the role of the paper's vertex operators (VOPs).
# A VOP is reduced to the identity by burning, factor by factor, its shortest
# decomposition (frames.DECOMP) into the generators √(-iX) ∝ HSH and
# √(iZ) ∝ S†:
#   · an "X" letter is produced by a re-framing move at the vertex,
#   · a "Z" letter by one at a neighbour (the "swapping partner"),
# each applied with the state-preserving frame update of _lc_step
# (Corollary 1 of the paper).


def _lc_step(adj: list[set[int]], v: int, f: list[int]) -> None:
    """One local complementation τ_v with the state-preserving frame update
    L_v ↦ L_v·HSH, L_u ↦ L_u·S† for u ∈ N(v) — the *directed* re-framing move
    R_v⁻¹ = R_v³ (right-multiply U_v instead of U_v†; both directions preserve
    the state since U_v² = K_v stabilises |G⟩). Kept in this direction because
    the decomposition words of frames.DECOMP burn {HSH, S†} letters.
    Mutates adj and f in place."""
    for u in adj[v]:
        f[u] = LC_NEIGH[f[u]]
    f[v] = LC_CENTER[f[v]]
    lc_inplace(adj, v)


def _reduce_vop_at(adj: list[set[int]], a: int, f: list[int],
                   avoid: int | None = None) -> None:
    """remove_VOP at vertex a (Anders-Briegel / notes-EulVec-Rep-Operations
    Sec. "Frame reduction"): burn the frame's shortest decomposition over
    {√(-iX) ∝ HSH, √(iZ) ∝ S†} by re-framing moves — an "X" letter by a move
    at a itself, a "Z" letter by one at a swapping partner b ∈ N(a),
    preferring partners ≠ avoid. Every move is state-preserving."""
    word = DECOMP[DAG[f[a]]]
    if not word:
        return
    for mv in word:
        if mv == "X":                    # burn √(-iX): re-frame at a itself
            _lc_step(adj, a, f)
        else:                            # burn √(iZ): re-frame at a partner
            nb = [j for j in adj[a] if j != avoid] or list(adj[a])
            if not nb:
                break                    # isolated vertex: no partner available
            b = min(nb, key=lambda j: (f[j] == ID_PAIR, j))
            _lc_step(adj, b, f)


# ── The coupled two-qubit CZ block (the 2·24² table of Anders-Briegel) ────────
# When an operand's frame cannot be reduced away (its only neighbour is the
# other operand), the two operand frames are coupled and the transition
# function is genuinely table-shaped (notes-EulVec-Rep-Operations, Remark 2).
# The table is brute-forced once on two-qubit stabilizer states, keyed by
# frame codes, and cached for the life of the process.

_CZTAB: dict | None = None


def _state_key_2q(psi) -> tuple:
    """Phase-canonical key for a two-qubit state vector."""
    idx = next(k for k in range(4) if abs(psi[k]) > 1e-8)
    psi = psi * (abs(psi[idx]) / psi[idx])
    return tuple(round(x, 6) + 0.0 for c in psi for x in (c.real, c.imag))


def build_cz_table() -> dict:
    """Enumerate every framed two-qubit representative (ζ, L_a, L_b) ↦ state
    (L_a ⊗ L_b)·CZ^ζ|++⟩, then read off the transition

        CZ·(L_a⊗L_b)·CZ^ζ|++⟩ ∝ (L'_a⊗L'_b)·CZ^{ζ'}|++⟩.

    Among the sign-redundant representatives it keeps, per the paper's
    Constraint 1, an output with w^N = +Z (frame in the Z-set {I,Z,S,S†}) on
    each operand that still carries external CZ edges, so the new frame
    commutes back through those edges — hence the (need_a, need_b) key.
    Entries with no such representative are simply absent; apply_cz then
    falls back to the tableau route.

    Several representatives usually qualify.  The tie is broken by
    (ζ', phase-canonical key of L'_a, phase-canonical key of L'_b), which is
    an arbitrary but *fixed* rule: any winner describes the same physical
    state, and pinning it keeps the frame this simulator reports — and the
    diagonal/table path split the benchmarks measure — reproducible.

    Returns {(ζ, code_a, code_b, need_a, need_b): (ζ', code_a', code_b')}."""
    global _CZTAB
    if _CZTAB is not None:
        return _CZTAB
    import numpy as np

    def u8_to_np(m):
        return np.array([[complex(m[0], m[1]), complex(m[2], m[3])],
                         [complex(m[4], m[5]), complex(m[6], m[7])]])

    plus = np.full(4, 0.5, dtype=complex)
    czm = np.diag([1, 1, 1, -1]).astype(complex)
    mats = {c: u8_to_np(PAIR_TO_MAT[c]) for c in VALID_PAIRS}

    from .cliffords import _clifford_key
    ckey = {c: _clifford_key(PAIR_TO_MAT[c]) for c in VALID_PAIRS}

    states: dict[tuple, list] = {}
    prepared: dict = {}
    for zeta in (0, 1):
        base = czm @ plus if zeta else plus
        for ca in VALID_PAIRS:
            for cb in VALID_PAIRS:
                psi = np.kron(mats[ca], mats[cb]) @ base
                prepared[(zeta, ca, cb)] = psi
                states.setdefault(_state_key_2q(psi), []).append((zeta, ca, cb))

    tab: dict = {}
    for (zeta, ca, cb), psi in prepared.items():
        cands = states.get(_state_key_2q(czm @ psi), [])
        for na in (False, True):
            for nb in (False, True):
                best = None
                for z2, ca2, cb2 in cands:
                    if (na and not in_zset(ca2)) or (nb and not in_zset(cb2)):
                        continue
                    entry = (z2, ckey[ca2], ckey[cb2], ca2, cb2)
                    if best is None or entry < best:
                        best = entry
                if best is not None:
                    tab[(zeta, ca, cb, na, nb)] = (best[0], best[3], best[4])
    _CZTAB = tab
    return tab


def _apply_cz_tableau(adj: list[set[int]], n: int, i: int, j: int,
                      f: list[int]) -> tuple[list[set[int]], list[int]]:
    """Fallback CZ path: conjugate the stabilizer tableau by CZ_ij
    (X_i↦X_iZ_j, Y_i↦Y_iZ_j, Z_i↦Z_i, symmetric in j), then reduce globally."""
    tab = _tableau_from_state(adj, n, f)
    for g in tab:                                # conjugate each generator by CZ_ij
        L = g[1]
        ai = L[i] in ("X", "Y")                  # i has X-component
        aj = L[j] in ("X", "Y")
        k = 0
        if aj:                                   # → append Z on qubit i: L[i]·Z
            kk, r = _PAULI_MUL[(L[i], "Z")]; k += kk; L[i] = r
        if ai:                                   # → append Z on qubit j: Z·L[j]
            kk, r = _PAULI_MUL[("Z", L[j])]; k += kk; L[j] = r
        if k % 4 == 2:
            g[0] = -g[0]
    new_adj, new_f, _ = _reduce_tableau(tab, n)
    return new_adj, new_f


def apply_cz(adj: list[set[int]], n: int, i: int, j: int,
             frame: list | None = None) -> tuple[list[set[int]], list[int]]:
    """Apply a physical CZ_ij gate to |ψ⟩ = (⊗L_k)|G⟩ by the *local* algorithm
    of Anders-Briegel Sec. III.2 in Eulerian-frame form
    (notes-EulVec-Rep-Operations):
      1-3. reduce the operand frames by state-preserving re-framing moves,
           choosing swapping partners among non-operand neighbours — skipped
           when the frame passes through CZ anyway (both operands with w^N on
           the Z axis, handled by step 4's sign folds, or a frame in the
           Z-set {I,Z,S,S†}), keeping the rewrite minimal;
      4.   if both Eulerian entries lie on the Z axis (w^N ∈ {±Z} — the signed
           extension of the paper's Z-set), CZ is an edge toggle plus a Z-fold
           on the other operand for each -Z sign;
      5.   otherwise the operands form a coupled two-qubit block: apply the
           brute-forced 2·24² transition table.
    Returns (new_adj, new_frame)."""
    f = parse_frame(n, frame)
    if n == 0 or i == j or not (0 <= i < n and 0 <= j < n):
        return copy_adj(adj), f
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
    both_zaxis = is_zaxis(f[i]) and is_zaxis(f[j])

    def commutes(a: int) -> bool:
        return both_zaxis or in_zset(f[a])

    if has_others(i, j) and not commutes(i):
        _reduce_vop_at(cur, i, f, avoid=j)
    if has_others(j, i) and not commutes(j):
        _reduce_vop_at(cur, j, f, avoid=i)
    if has_others(i, j) and not commutes(i):
        _reduce_vop_at(cur, i, f, avoid=j)       # step 2 may de-reduce i

    wni, wnj = f[i] % 6, f[j] % 6                # w^N of the operands
    if wni % 3 == 2 and wnj % 3 == 2:            # 4. diagonal-axis case
        toggle_edge(cur, i, j)
        if wni == 5:                             # w^N_i = -Z → fold Z at j
            f[j] = ZFOLD[f[j]]
        if wnj == 5:                             # w^N_j = -Z → fold Z at i
            f[i] = ZFOLD[f[i]]
        return cur, f

    res = build_cz_table().get(                  # 5. coupled two-qubit block
        (1 if j in cur[i] else 0, f[i], f[j],
         has_others(i, j), has_others(j, i)))
    if res is None:                              # constraint unmet — defensive
        return _apply_cz_tableau(cur, n, i, j, f)
    zeta2, ca2, cb2 = res
    set_edge(cur, i, j, zeta2)
    f[i], f[j] = ca2, cb2
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
