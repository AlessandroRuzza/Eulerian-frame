"""Stabilizer tableau machinery.

Stabilizer generators of (tensor L_i)|G>, Pauli-string multiplication,
and Gauss-Jordan reduction of a tableau back to graph standard form
K_v = X_v (tensor) Z_N(v). Also the stabilizer report for the UI.
"""
from __future__ import annotations

from .cliffords import (
    _H_U8,
    _IDENTITY_U8,
    _SDG_U8,
    _Z_U8,
    _conj_pauli,
    _dag_u8,
    _mat2x2_mul,
    _parse_mats,
)

def compute_stabilizers(
    adj: list[set[int]], n: int, labels: list[str],
    local_unitaries: list | None = None,
) -> list[dict]:
    mats = _parse_mats(n, local_unitaries)
    result = []
    for i in range(n):
        nb = sorted(adj[i])
        overall_sign = 1
        # K_i = X_i (x) Z_N(i): only the support is conjugated, the rest is I.
        paulis: list[str] = ["I"] * n
        for k, base in [(i, "X")] + [(j, "Z") for j in nb]:
            s, p = _conj_pauli(mats[k], base)
            overall_sign *= s
            paulis[k] = p
        _PC = {"X": "#1a8078", "Y": "#b85535", "Z": "#5a1f6e", "I": "#888888"}
        def _p_html(p: str, lbl: str) -> str:
            c = _PC.get(p, "#333")
            return f'<span style="color:{c};font-weight:700">{p}<sub>{lbl}</sub></span>'
        sign_html = "&minus;" if overall_sign < 0 else ""
        non_id = [(k, paulis[k]) for k in range(n) if paulis[k] != "I"]
        if non_id:
            compact = sign_html + " &otimes; ".join(
                _p_html(p, labels[k]) for k, p in non_id)
        else:
            compact = sign_html + "I"
        pauli_str = ("-" if overall_sign < 0 else "") + "".join(paulis)
        result.append({"i": i, "label": labels[i], "compact": compact,
                       "pauli": pauli_str, "neighbors": nb, "sign": overall_sign})
    return result


# ── Stabilizer tableau reduction (used by CZ/CX/CY application) ────────────────
# Single-qubit Pauli products: P·Q = i^k · R, stored as (k mod 4, R).
_PAULI_MUL = {
    ("I","I"):(0,"I"),("I","X"):(0,"X"),("I","Y"):(0,"Y"),("I","Z"):(0,"Z"),
    ("X","I"):(0,"X"),("X","X"):(0,"I"),("X","Y"):(1,"Z"),("X","Z"):(3,"Y"),
    ("Y","I"):(0,"Y"),("Y","X"):(3,"Z"),("Y","Y"):(0,"I"),("Y","Z"):(1,"X"),
    ("Z","I"):(0,"Z"),("Z","X"):(1,"Y"),("Z","Y"):(3,"X"),("Z","Z"):(0,"I"),
}


def _stab_mul(g1: tuple, g2: tuple, n: int) -> tuple:
    """Multiply two Hermitian Pauli strings (sign, letters). Returns (sign, letters).
    Both inputs commute (stabilizer generators) so the i-phase cancels to ±1."""
    s1, l1 = g1
    s2, l2 = g2
    k = (0 if s1 > 0 else 2) + (0 if s2 > 0 else 2)
    out = []
    for a, b in zip(l1, l2):
        kk, r = _PAULI_MUL[(a, b)]
        k += kk
        out.append(r)
    k %= 4
    if k not in (0, 2):
        raise ValueError("non-commuting stabilizer multiplication")
    return (1 if k == 0 else -1, out)


def _tableau_from_state(adj: list[set[int]], n: int, mats: list) -> list[list]:
    """Stabilizer generators of |ψ⟩ = (⊗L_i)|G⟩ as a tableau.
    Each entry is [sign(±1), letters] where letters[k] ∈ {I,X,Y,Z}.
    The tableau itself is dense by nature (n rows of n letters); it is filled
    from the adjacency sets, so building it costs O(n^2) writes and O(n + m)
    conjugations."""
    tab: list[list] = []
    for v in range(n):
        sign = 1
        letters = ["I"] * n
        for k, base in [(v, "X")] + [(j, "Z") for j in sorted(adj[v])]:
            s, p = _conj_pauli(mats[k], base)
            sign *= s
            letters[k] = p
        tab.append([sign, letters])
    return tab


def _reduce_tableau(tab: list[list], n: int
                    ) -> tuple[list[set[int]], list[list[float]], list[list[float]]]:
    """Reduce a stabilizer tableau to graph standard form K'_v = X_v ⊗ Z_{N'(v)}
    (+1 signs) by Gauss-Jordan, applying local Cliffords C_i (H, S†, Z).
    Returns (new_adj as adjacency sets, new_local_unitaries = C_i†,
    corrections = C_i)."""
    C = [list(_IDENTITY_U8) for _ in range(n)]   # accumulated Clifford per qubit

    def conj_H(j: int) -> None:                  # X↔Z, Y→-Y
        for g in tab:
            l = g[1][j]
            if l == "X": g[1][j] = "Z"
            elif l == "Z": g[1][j] = "X"
            elif l == "Y": g[0] = -g[0]
        C[j] = _mat2x2_mul(_H_U8, C[j])

    def conj_Sdg(j: int) -> None:                # X→-Y, Y→X, Z→Z
        for g in tab:
            l = g[1][j]
            if l == "X": g[1][j] = "Y"; g[0] = -g[0]
            elif l == "Y": g[1][j] = "X"
        C[j] = _mat2x2_mul(_SDG_U8, C[j])

    def conj_Z(j: int) -> None:                  # flip sign where X-component present
        for g in tab:
            if g[1][j] in ("X", "Y"):
                g[0] = -g[0]
        C[j] = _mat2x2_mul(_Z_U8, C[j])

    has_x = lambda l: l in ("X", "Y")            # X-component present?

    # Gauss-Jordan: make row j pivot qubit j (X-block → identity).
    for j in range(n):
        piv = next((r for r in range(j, n) if has_x(tab[r][1][j])), None)
        if piv is None:                          # no X-comp on qubit j → Hadamard it in
            conj_H(j)
            piv = next((r for r in range(j, n) if has_x(tab[r][1][j])), None)
        if piv is None:
            continue                             # degenerate (should not happen)
        tab[j], tab[piv] = tab[piv], tab[j]
        for i in range(n):
            if i != j and has_x(tab[i][1][j]):
                tab[i] = list(_stab_mul(tuple(tab[i]), tuple(tab[j]), n))

    # Clear diagonal Y→X (residual S†) and fix signs to + (residual Z).
    for j in range(n):
        if tab[j][1][j] == "Y":
            conj_Sdg(j)
        if tab[j][0] < 0:
            conj_Z(j)

    # Read adjacency from the Z-pattern; frame is C†.
    # Symmetrise defensively (valid stabilizer states give symmetric output).
    new_adj: list[set[int]] = [set() for _ in range(n)]
    for j in range(n):
        for k in range(n):
            if k != j and tab[j][1][k] == "Z":
                new_adj[j].add(k)
                new_adj[k].add(j)

    new_lu = [_dag_u8(C[i]) for i in range(n)]
    return new_adj, new_lu, [list(C[i]) for i in range(n)]

