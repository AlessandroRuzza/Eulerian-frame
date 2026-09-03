"""Rule tables for the VOP-storage backends.

The same rules ``eulsim.frames`` packs into integer arithmetic, rebuilt here
over the two other storages: the phase-canonical Clifford key, and an opaque
id 0..23.  Every table is produced by running the matrix operation once per
Clifford and memoising the result, so all three backends are correct by
construction from the same toolkit.

The coupled CZ block table is *not* rebuilt: it is re-indexed from the core's
``gates.build_cz_table`` so the physics has exactly one source.
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
from ..frames import DECOMP, GATE_U8, PAIR_TO_MAT, VALID_PAIRS, pair_from_mat

# ── Storage 1: the phase-canonical Clifford key ───────────────────────────────

KEY_TO_MAT: dict = {_clifford_key(m): m for m in PAIR_TO_MAT.values()}
ID_KEY = _clifford_key(_IDENTITY_U8)

#: The paper's Z = {I, Z, S, S†}: the local Cliffords commuting with CZ.
ZSET_KEYS = frozenset(
    _clifford_key(m) for m in (_IDENTITY_U8, _Z_U8, _S_U8, _SDG_U8))

#: VOP decomposition words, keyed by Clifford key (see ``frames.DECOMP``).
VOP_DECOMP: dict = {_clifford_key(PAIR_TO_MAT[c]): DECOMP[c] for c in VALID_PAIRS}


def _left_key_table(g8: list[float]) -> dict:
    return {k: _clifford_key(_mat2x2_mul(g8, m)) for k, m in KEY_TO_MAT.items()}


def _right_key_table(g8: list[float]) -> dict:
    return {k: _clifford_key(_mat2x2_mul(m, g8)) for k, m in KEY_TO_MAT.items()}


LKEY: dict[str, dict] = {g: _left_key_table(gm) for g, gm in GATE_U8.items()}
RV_CENTER_KEY = _right_key_table(_HSDGH_U8)
RV_NEIGH_KEY = _right_key_table(_S_U8)
LC_CENTER_KEY = _right_key_table(_HSH_U8)
LC_NEIGH_KEY = _right_key_table(_SDG_U8)
ZFOLD_KEY = _right_key_table(_Z_U8)
DAG_KEY = {k: _clifford_key(_dag_u8(m)) for k, m in KEY_TO_MAT.items()}

# Measurement basis transport L† P L and the pending-basis factors used
# mid-reduction — the analogues of frames.image / PEND_SDG / PEND_HSH.
TRANSPORT_KEY = {k: {P: _conj_pauli(_dag_u8(m), P) for P in "XYZ"}
                 for k, m in KEY_TO_MAT.items()}
PEND_SDG_KEY = {P: _conj_pauli(_SDG_U8, P) for P in "XYZ"}
PEND_HSH_KEY = {P: _conj_pauli(_HSH_U8, P) for P in "XYZ"}
# CZ diagonal-axis test: w^N axis and sign (mirrors frames.is_zaxis via %6).
ZAXIS_KEY = {k: _conj_pauli(m, "Z") for k, m in KEY_TO_MAT.items()}


def frame_rank(m: list[float]) -> int:
    """0: ∝ I.  1: in Z = {I,Z,S,S†} (commutes with CZ).  2: other."""
    k = _clifford_key(m)
    if k == ID_KEY:
        return 0
    return 1 if k in ZSET_KEYS else 2


# ── Storage 2: an opaque id 0..23 ─────────────────────────────────────────────
# BFS discovery order, deliberately unrelated to w^C/w^N.

ID_OF_KEY: dict = {k: i for i, k in enumerate(KEY_TO_MAT)}
ID_MAT: list = list(KEY_TO_MAT.values())
assert len(ID_MAT) == 24
ID_I = ID_OF_KEY[ID_KEY]


def _left_id_table(g8: list[float]) -> list:
    return [ID_OF_KEY[_clifford_key(_mat2x2_mul(g8, m))] for m in ID_MAT]


def _right_id_table(g8: list[float]) -> list:
    return [ID_OF_KEY[_clifford_key(_mat2x2_mul(m, g8))] for m in ID_MAT]


LID: dict[str, list] = {g: _left_id_table(gm) for g, gm in GATE_U8.items()}
RV_CENTER_ID = _right_id_table(_HSDGH_U8)
RV_NEIGH_ID = _right_id_table(_S_U8)
LC_CENTER_ID = _right_id_table(_HSH_U8)
LC_NEIGH_ID = _right_id_table(_SDG_U8)
ZFOLD_ID = _right_id_table(_Z_U8)
DAG_ID = [ID_OF_KEY[_clifford_key(_dag_u8(m))] for m in ID_MAT]
TRANSPORT_ID = [{P: _conj_pauli(_dag_u8(m), P) for P in "XYZ"} for m in ID_MAT]
ZAXIS_ID = [_conj_pauli(m, "Z") for m in ID_MAT]
ZSET_ID = [_clifford_key(m) in ZSET_KEYS for m in ID_MAT]
VOPDECOMP_ID = [VOP_DECOMP[k] for k in KEY_TO_MAT]


# ── The coupled CZ block, re-indexed from the core table ──────────────────────

_CZ_KEY: dict | None = None
_CZ_ID: dict | None = None


def cz_tables() -> tuple[dict, dict]:
    """The core's pair-keyed CZ block table, re-indexed by Clifford key and by
    opaque id.  Built once, on first use."""
    global _CZ_KEY, _CZ_ID
    if _CZ_KEY is None:
        from ..gates import build_cz_table
        k_of = {c: _clifford_key(PAIR_TO_MAT[c]) for c in VALID_PAIRS}
        _CZ_KEY, _CZ_ID = {}, {}
        for (z, ca, cb, na, nb), (z2, ca2, cb2) in build_cz_table().items():
            _CZ_KEY[(z, k_of[ca], k_of[cb], na, nb)] = (z2, k_of[ca2], k_of[cb2])
            _CZ_ID[(z, ID_OF_KEY[k_of[ca]], ID_OF_KEY[k_of[cb]], na, nb)] = (
                z2, ID_OF_KEY[k_of[ca2]], ID_OF_KEY[k_of[cb2]])
    return _CZ_KEY, _CZ_ID


def mat_cz_lookup(zeta: int, key_a, key_b, need_a: bool, need_b: bool):
    """Coupled-block transition for the matrix backend: same table, but the
    caller arrives with freshly computed Clifford keys.  Returns
    (ζ', mat_a, mat_b) or None."""
    res = cz_tables()[0].get((zeta, key_a, key_b, need_a, need_b))
    if res is None:
        return None
    z2, ka2, kb2 = res
    return z2, list(KEY_TO_MAT[ka2]), list(KEY_TO_MAT[kb2])


def key_to_pair(k) -> int:
    return pair_from_mat(KEY_TO_MAT[k])
