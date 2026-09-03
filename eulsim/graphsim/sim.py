"""The three VOP-storage backends.

Each mirrors the algorithms of ``eulsim.graph_ops`` and ``eulsim.gates``
exactly — same case splits, same free choices, same order of moves — and
differs only in how the per-vertex frame is stored and read.  The graph layer
and the defensive CZ fallback are imported from the core, so a difference in
behaviour between backends would be a bug, and a difference in cost is the
representation.

The measured vertex is kept and reset (``delete=False`` in the core's
``do_measure``): a benchmark stream needs the register size to stay fixed.
"""
from __future__ import annotations

from ..cliffords import (
    _HSDGH_U8,
    _HSH_U8,
    _IDENTITY_U8,
    _S_U8,
    _SDG_U8,
    _Z_U8,
    _clifford_key,
    _conj_pauli,
    _dag_u8,
    _mat2x2_mul,
)
from ..frames import PAIR_TO_MAT, pair_from_mat
from ..gates import _apply_cz_tableau
from ..graph_ops import lc_inplace, set_edge, toggle_edge
from .tables import (
    DAG_ID,
    DAG_KEY,
    ID_I,
    ID_KEY,
    ID_MAT,
    ID_OF_KEY,
    KEY_TO_MAT,
    LKEY,
    LC_CENTER_ID,
    LC_CENTER_KEY,
    LC_NEIGH_ID,
    LC_NEIGH_KEY,
    PEND_HSH_KEY,
    PEND_SDG_KEY,
    RV_CENTER_ID,
    RV_CENTER_KEY,
    RV_NEIGH_ID,
    RV_NEIGH_KEY,
    TRANSPORT_ID,
    TRANSPORT_KEY,
    VOP_DECOMP,
    VOPDECOMP_ID,
    ZAXIS_ID,
    ZAXIS_KEY,
    ZFOLD_ID,
    ZFOLD_KEY,
    ZSET_ID,
    ZSET_KEYS,
    cz_tables,
    frame_rank,
    mat_cz_lookup,
)


class _Base:
    """Shared bookkeeping: which CZ path each call took."""

    def __init__(self, adj: list[set]):
        self.adj = [set(s) for s in adj]
        self.n_diag = self.n_table = self.n_fallback = 0

    def _cz_common(self, i: int, j: int):
        """The reduction phase shared by all three, in terms of the subclass's
        ``_commutes`` / ``_reduce_vop_at``."""
        def has_others(a: int, b: int) -> bool:
            return any(k != b for k in self.adj[a])

        both = self._both_zaxis(i, j)
        if has_others(i, j) and not self._commutes(i, both):
            self._reduce_vop_at(i, avoid=j)
        if has_others(j, i) and not self._commutes(j, both):
            self._reduce_vop_at(j, avoid=i)
        if has_others(i, j) and not self._commutes(i, both):
            self._reduce_vop_at(i, avoid=j)     # step 2 may de-reduce i
        return has_others


# ── Storage: 2x2 Clifford matrices (the literature picture) ───────────────────

class CliffordSim(_Base):
    """Frames as 8-float matrices; every update a complex matrix product and
    every semantic read a Pauli conjugation or a phase-canonical key."""

    def __init__(self, adj: list[set], pair_codes: list[int]):
        super().__init__(adj)
        self.f = [list(PAIR_TO_MAT[c]) for c in pair_codes]

    def apply_local(self, v: int, g8: list[float]) -> None:
        self.f[v] = _mat2x2_mul(g8, self.f[v])

    def reframe(self, v: int) -> None:
        """R_v: right-multiply (HS†H)_v and S on N(v), then τ_v."""
        f = self.f
        f[v] = _mat2x2_mul(f[v], _HSDGH_U8)
        for u in self.adj[v]:
            f[u] = _mat2x2_mul(f[u], _S_U8)
        lc_inplace(self.adj, v)

    def _lc_step(self, v: int) -> None:
        f = self.f
        for u in self.adj[v]:
            f[u] = _mat2x2_mul(f[u], _SDG_U8)
        f[v] = _mat2x2_mul(f[v], _HSH_U8)
        lc_inplace(self.adj, v)

    def measure(self, v: int, basis: str, invert: bool = False) -> None:
        f, adj = self.f, self.adj
        sigma, Q = _conj_pauli(_dag_u8(f[v]), basis)     # basis transport
        if Q == "X":
            if not adj[v]:                               # deterministic
                f[v] = list(_IDENTITY_U8)
                return
            b = min(adj[v], key=lambda j: (len(adj[j]), j))
            self.reframe(b)                              # pending: conj by S†
            s2, Q = PEND_SDG_KEY[Q]
            sigma *= s2
        if Q == "Y":
            self.reframe(v)                              # pending: conj by HSH
            s2, Q = PEND_HSH_KEY[Q]
            sigma *= s2
        if sigma * (-1 if invert else 1) == -1:          # Z-deletion correction
            for u in adj[v]:
                f[u] = _mat2x2_mul(f[u], _Z_U8)
        for u in adj[v]:
            adj[u].discard(v)
        adj[v].clear()
        f[v] = list(_IDENTITY_U8)

    def _reduce_vop_at(self, a: int, avoid: int) -> None:
        word = VOP_DECOMP.get(_clifford_key(_dag_u8(self.f[a])))
        if not word:
            return
        adj, f = self.adj, self.f
        for mv in word:
            if mv == "X":
                self._lc_step(a)
            else:
                nb = [j for j in adj[a] if j != avoid] or list(adj[a])
                if not nb:
                    break
                b = min(nb, key=lambda j: (frame_rank(f[j]) == 0, j))
                self._lc_step(b)

    def _both_zaxis(self, i: int, j: int) -> bool:
        return (_conj_pauli(self.f[i], "Z")[1] == "Z"
                and _conj_pauli(self.f[j], "Z")[1] == "Z")

    def _commutes(self, a: int, both: bool) -> bool:
        return both or _clifford_key(self.f[a]) in ZSET_KEYS

    def cz(self, i: int, j: int) -> None:
        has_others = self._cz_common(i, j)
        adj, f = self.adj, self.f
        sa, pa = _conj_pauli(f[i], "Z")
        sb, pb = _conj_pauli(f[j], "Z")
        if pa == "Z" and pb == "Z":                      # diagonal-axis case
            self.n_diag += 1
            toggle_edge(adj, i, j)
            if sa < 0:
                f[j] = _mat2x2_mul(f[j], _Z_U8)
            if sb < 0:
                f[i] = _mat2x2_mul(f[i], _Z_U8)
            return
        res = mat_cz_lookup(1 if j in adj[i] else 0,     # coupled block
                            _clifford_key(f[i]), _clifford_key(f[j]),
                            has_others(i, j), has_others(j, i))
        if res is None:                                  # defensive fallback
            self.n_fallback += 1
            n = len(adj)
            new_adj, new_f = _apply_cz_tableau(
                adj, n, i, j, [pair_from_mat(m) for m in f])
            self.adj = new_adj
            self.f = [list(PAIR_TO_MAT[c]) for c in new_f]
            return
        self.n_table += 1
        zeta2, ma2, mb2 = res
        set_edge(adj, i, j, zeta2)
        f[i], f[j] = ma2, mb2

    def pair_codes(self) -> list[int]:
        return [pair_from_mat(m) for m in self.f]


# ── Storage: the phase-canonical Clifford key ─────────────────────────────────

class CliffordLUTSim(_Base):
    """The same 24 Cliffords, but the per-vertex state IS the phase-canonical
    key and every rule is a memoised dict lookup."""

    def __init__(self, adj: list[set], pair_codes: list[int]):
        super().__init__(adj)
        self.f = [_clifford_key(PAIR_TO_MAT[c]) for c in pair_codes]

    def apply_local(self, v: int, gate_name: str) -> None:
        self.f[v] = LKEY[gate_name][self.f[v]]

    def reframe(self, v: int) -> None:
        f = self.f
        f[v] = RV_CENTER_KEY[f[v]]
        for u in self.adj[v]:
            f[u] = RV_NEIGH_KEY[f[u]]
        lc_inplace(self.adj, v)

    def _lc_step(self, v: int) -> None:
        f = self.f
        for u in self.adj[v]:
            f[u] = LC_NEIGH_KEY[f[u]]
        f[v] = LC_CENTER_KEY[f[v]]
        lc_inplace(self.adj, v)

    def measure(self, v: int, basis: str, invert: bool = False) -> None:
        f, adj = self.f, self.adj
        sigma, Q = TRANSPORT_KEY[f[v]][basis]
        if Q == "X":
            if not adj[v]:
                f[v] = ID_KEY
                return
            b = min(adj[v], key=lambda j: (len(adj[j]), j))
            self.reframe(b)
            s2, Q = PEND_SDG_KEY[Q]
            sigma *= s2
        if Q == "Y":
            self.reframe(v)
            s2, Q = PEND_HSH_KEY[Q]
            sigma *= s2
        if sigma * (-1 if invert else 1) == -1:
            for u in adj[v]:
                f[u] = ZFOLD_KEY[f[u]]
        for u in adj[v]:
            adj[u].discard(v)
        adj[v].clear()
        f[v] = ID_KEY

    def _reduce_vop_at(self, a: int, avoid: int) -> None:
        word = VOP_DECOMP.get(DAG_KEY[self.f[a]])
        if not word:
            return
        adj, f = self.adj, self.f
        for mv in word:
            if mv == "X":
                self._lc_step(a)
            else:
                nb = [j for j in adj[a] if j != avoid] or list(adj[a])
                if not nb:
                    break
                b = min(nb, key=lambda j: (f[j] == ID_KEY, j))
                self._lc_step(b)

    def _both_zaxis(self, i: int, j: int) -> bool:
        return ZAXIS_KEY[self.f[i]][1] == "Z" and ZAXIS_KEY[self.f[j]][1] == "Z"

    def _commutes(self, a: int, both: bool) -> bool:
        return both or self.f[a] in ZSET_KEYS

    def cz(self, i: int, j: int) -> None:
        has_others = self._cz_common(i, j)
        adj, f = self.adj, self.f
        sa, pa = ZAXIS_KEY[f[i]]
        sb, pb = ZAXIS_KEY[f[j]]
        if pa == "Z" and pb == "Z":
            self.n_diag += 1
            toggle_edge(adj, i, j)
            if sa < 0:
                f[j] = ZFOLD_KEY[f[j]]
            if sb < 0:
                f[i] = ZFOLD_KEY[f[i]]
            return
        res = cz_tables()[0].get((1 if j in adj[i] else 0, f[i], f[j],
                                  has_others(i, j), has_others(j, i)))
        if res is None:
            self.n_fallback += 1
            n = len(adj)
            new_adj, new_f = _apply_cz_tableau(
                adj, n, i, j, [pair_from_mat(KEY_TO_MAT[k]) for k in f])
            self.adj = new_adj
            self.f = [_clifford_key(PAIR_TO_MAT[c]) for c in new_f]
            return
        self.n_table += 1
        zeta2, ka2, kb2 = res
        set_edge(adj, i, j, zeta2)
        f[i], f[j] = ka2, kb2

    def pair_codes(self) -> list[int]:
        return [pair_from_mat(KEY_TO_MAT[k]) for k in self.f]


# ── Storage: an opaque id 0..23 ───────────────────────────────────────────────

class CliffordIDSim(_Base):
    """The same 24 Cliffords under an opaque small int, every table a plain
    list — the Eulerian backend's storage shape without its structure."""

    def __init__(self, adj: list[set], pair_codes: list[int]):
        super().__init__(adj)
        self.f = [ID_OF_KEY[_clifford_key(PAIR_TO_MAT[c])] for c in pair_codes]

    def apply_local(self, v: int, table: list) -> None:
        self.f[v] = table[self.f[v]]

    def reframe(self, v: int) -> None:
        f = self.f
        f[v] = RV_CENTER_ID[f[v]]
        for u in self.adj[v]:
            f[u] = RV_NEIGH_ID[f[u]]
        lc_inplace(self.adj, v)

    def _lc_step(self, v: int) -> None:
        f = self.f
        for u in self.adj[v]:
            f[u] = LC_NEIGH_ID[f[u]]
        f[v] = LC_CENTER_ID[f[v]]
        lc_inplace(self.adj, v)

    def measure(self, v: int, basis: str, invert: bool = False) -> None:
        f, adj = self.f, self.adj
        sigma, Q = TRANSPORT_ID[f[v]][basis]
        if Q == "X":
            if not adj[v]:
                f[v] = ID_I
                return
            b = min(adj[v], key=lambda j: (len(adj[j]), j))
            self.reframe(b)
            s2, Q = PEND_SDG_KEY[Q]
            sigma *= s2
        if Q == "Y":
            self.reframe(v)
            s2, Q = PEND_HSH_KEY[Q]
            sigma *= s2
        if sigma * (-1 if invert else 1) == -1:
            for u in adj[v]:
                f[u] = ZFOLD_ID[f[u]]
        for u in adj[v]:
            adj[u].discard(v)
        adj[v].clear()
        f[v] = ID_I

    def _reduce_vop_at(self, a: int, avoid: int) -> None:
        word = VOPDECOMP_ID[DAG_ID[self.f[a]]]
        if not word:
            return
        adj, f = self.adj, self.f
        for mv in word:
            if mv == "X":
                self._lc_step(a)
            else:
                nb = [j for j in adj[a] if j != avoid] or list(adj[a])
                if not nb:
                    break
                b = min(nb, key=lambda j: (f[j] == ID_I, j))
                self._lc_step(b)

    def _both_zaxis(self, i: int, j: int) -> bool:
        return ZAXIS_ID[self.f[i]][1] == "Z" and ZAXIS_ID[self.f[j]][1] == "Z"

    def _commutes(self, a: int, both: bool) -> bool:
        return both or ZSET_ID[self.f[a]]

    def cz(self, i: int, j: int) -> None:
        has_others = self._cz_common(i, j)
        adj, f = self.adj, self.f
        sa, pa = ZAXIS_ID[f[i]]
        sb, pb = ZAXIS_ID[f[j]]
        if pa == "Z" and pb == "Z":
            self.n_diag += 1
            toggle_edge(adj, i, j)
            if sa < 0:
                f[j] = ZFOLD_ID[f[j]]
            if sb < 0:
                f[i] = ZFOLD_ID[f[i]]
            return
        res = cz_tables()[1].get((1 if j in adj[i] else 0, f[i], f[j],
                                  has_others(i, j), has_others(j, i)))
        if res is None:
            self.n_fallback += 1
            n = len(adj)
            new_adj, new_f = _apply_cz_tableau(
                adj, n, i, j, [pair_from_mat(ID_MAT[k]) for k in f])
            self.adj = new_adj
            self.f = [ID_OF_KEY[_clifford_key(PAIR_TO_MAT[c])] for c in new_f]
            return
        self.n_table += 1
        zeta2, ia2, ib2 = res
        set_edge(adj, i, j, zeta2)
        f[i], f[j] = ia2, ib2

    def pair_codes(self) -> list[int]:
        return [pair_from_mat(ID_MAT[k]) for k in self.f]
