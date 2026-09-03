"""The Eulerian frame: a local Clifford stored as its vertex basis.

A frame letter L_v is kept as the pair of signed Paulis it conjugates X and Z
into — the supplementary and Eulerian elements (prop:dictionary)

    (w^C_v, w^N_v) = (L_v X L_v†, L_v Z L_v†),

which is a bijection with the 24 single-qubit Cliffords mod phase
(notes-EulVec-Rep-Operations, Prop. 1), so nothing is lost by dropping the
matrix.  A signed Pauli is packed into ``0..5`` as ``axis + 3*sign_bit``
(axes X=0, Y=1, Z=2; sign bit 0 = ``+``), and the pair into one integer

    code = 6*w^C + w^N   in 0..35,   24 of which are valid (distinct axes).

That single int is the whole per-vertex state.  Every frame update is one
list read; every semantic question the calculus asks is an integer test:

    is w^N on the Z axis (CZ case split)      code % 6 in {2, 5}
    is L in the Z-set {I, Z, S, S†}           code % 6 == 2
    is v in the Hadamard support              code % 6 == 0
    L Y L† (the third image)                  IPROD[w^C][w^N]

No floats, no phase canonicalisation, no lookup keyed by a rounded tuple.

Provenance
----------
Every table here is *derived* from the 2x2 matrix toolkit in ``cliffords`` at
import time and then checked exhaustively against it over all 24 Cliffords
(:func:`_verify`), so the integer rules are machine-verified against the
matrix semantics rather than transcribed by hand.  ``cliffords`` therefore
stays in the core as the reference definition even though the compute layer
never multiplies a matrix; ``statevector`` also needs real matrices to expand
amplitudes, and ``graphsim`` is built on them by construction.
"""
from __future__ import annotations

from .cliffords import (
    _H_U8,
    _HSDGH_U8,
    _HSH_U8,
    _IDENTITY_U8,
    _S_U8,
    _SDG_U8,
    _X_U8,
    _Y_U8,
    _Z_U8,
    _clifford_key,
    _conj_pauli,
    _dag_u8,
    _mat2x2_mul,
)

AXES = "XYZ"

#: Negate a signed Pauli (flip the sign bit).
NEG6 = [3, 4, 5, 0, 1, 2]

#: ``(w^C, w^N) = (+X, +Z)``: the identity frame.
ID_PAIR = 6 * 0 + 2


def encode(sign: int, letter: str) -> int:
    """Signed Pauli ``(±1, "XYZ")`` -> ``0..5``."""
    return AXES.index(letter) + (0 if sign > 0 else 3)


def decode(p: int) -> tuple[int, str]:
    """``0..5`` -> signed Pauli ``(±1, "XYZ")``."""
    return (1 if p < 3 else -1), AXES[p % 3]


# IPROD[p][q] = i·P_p·P_q for distinct axes — the Y image, L Y L† = i w^C w^N.
# From P_a P_b = i ε_abc P_c it follows that i P_a P_b = -ε_abc P_c.
IPROD: list[list[int | None]] = [[None] * 6 for _ in range(6)]
for _p in range(6):
    for _q in range(6):
        _a, _b = _p % 3, _q % 3
        if _a == _b:
            continue
        _c = 3 - _a - _b
        _neg = (_p // 3) ^ (_q // 3) ^ (1 if (_a + 1) % 3 == _b else 0)
        IPROD[_p][_q] = _c + 3 * _neg


def pair_from_mat(m: list[float]) -> int:
    """Frame code of a 2x2 Clifford matrix: ``(L X L†, L Z L†)``."""
    return 6 * encode(*_conj_pauli(m, "X")) + encode(*_conj_pauli(m, "Z"))


def _all_cliffords_u8() -> dict:
    """The 24 single-qubit Cliffords mod phase as 8-float matrices, keyed by
    ``_clifford_key``, from a BFS over words in {H, S}."""
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


CLIFFORDS_U8 = _all_cliffords_u8()

#: code -> the 8-float matrix of that Clifford (the bridge to ``statevector``,
#: ``graphsim`` and the JSON wire format).
PAIR_TO_MAT: dict[int, list[float]] = {
    pair_from_mat(m): m for m in CLIFFORDS_U8.values()}
VALID_PAIRS = sorted(PAIR_TO_MAT)


def mat_from_pair(c: int) -> list[float]:
    """Frame code -> 8-float matrix."""
    return PAIR_TO_MAT[c]


# ── Right composition: L -> L·U, the frame side of every representation move ──
# Indexed by code; ``None`` on the 12 invalid codes.

RV_CENTER = [None] * 36   # ·HS†H ~ √(iX)    R_v at the centre:  w^N -> i w^C w^N
RV_NEIGH = [None] * 36    # ·S    ~ √(iZ)    R_v at a neighbour: w^C -> i w^C w^N
LC_CENTER = [None] * 36   # ·HSH  ~ √(-iX)   the inverse move (VOP reduction)
LC_NEIGH = [None] * 36    # ·S†   ~ √(-iZ)
ZFOLD = [None] * 36       # ·Z               w^C -> -w^C  (Pauli byproducts)
XFOLD = [None] * 36       # ·X               w^N -> -w^N  (the R_v² sign fold)
for _wc in range(6):
    for _wn in range(6):
        if _wc % 3 == _wn % 3:
            continue
        _cd = 6 * _wc + _wn
        _y = IPROD[_wc][_wn]
        RV_CENTER[_cd] = 6 * _wc + _y
        RV_NEIGH[_cd] = 6 * _y + _wn
        LC_CENTER[_cd] = 6 * _wc + NEG6[_y]
        LC_NEIGH[_cd] = 6 * NEG6[_y] + _wn
        ZFOLD[_cd] = 6 * NEG6[_wc] + _wn
        XFOLD[_cd] = 6 * _wc + NEG6[_wn]

#: code -> code of L†  (basis transport, VOP decomposition).
DAG = [None] * 36
for _c, _m in PAIR_TO_MAT.items():
    DAG[_c] = pair_from_mat(_dag_u8(_m))


# ── Left composition: L -> g·L, a physical gate acting on the state ───────────

GATE_U8 = {"H": _H_U8, "S": _S_U8, "SDG": _SDG_U8,
           "X": _X_U8, "Y": _Y_U8, "Z": _Z_U8}

LGATE: dict[str, list] = {}
for _g, _gm in GATE_U8.items():
    _t6 = [0] * 6                       # how g moves each signed Pauli
    for _p in range(3):
        _t6[_p] = encode(*_conj_pauli(_gm, AXES[_p]))
        _t6[_p + 3] = NEG6[_t6[_p]]
    _tp = [None] * 36
    for _cd in VALID_PAIRS:
        _tp[_cd] = 6 * _t6[_cd // 6] + _t6[_cd % 6]
    LGATE[_g] = _tp


def left_compose(code: int, *gates: str) -> int:
    """``g_k ··· g_1 · L`` for named gates applied left to right."""
    for g in gates:
        code = LGATE[g][code]
    return code


# ── VOP decomposition ─────────────────────────────────────────────────────────
# For each frame letter, a shortest word over the two re-framing generators
#     "X" = HSH  ~ √(-iX)   (a move at the vertex itself)
#     "Z" = S†   ~ √(-iZ)   (a move at a neighbour, the "swapping partner")
# whose product is ∝ that letter.  Burning this word by re-framing moves is
# Anders-Briegel's remove_VOP (quant-ph/0504117), used by the local CZ
# algorithm in ``gates``.

def _build_decomp() -> list:
    from collections import deque
    tbl: list = [None] * 36
    tbl[ID_PAIR] = ""
    q = deque([ID_PAIR])
    while q:
        c = q.popleft()
        for mv, tab in (("X", LC_CENTER), ("Z", LC_NEIGH)):
            c2 = tab[c]
            if tbl[c2] is None:
                tbl[c2] = tbl[c] + mv
                q.append(c2)
    return tbl


#: code -> shortest word over {"X" = HSH, "Z" = S†} composing to that letter.
DECOMP = _build_decomp()


# ── Reading the frame ─────────────────────────────────────────────────────────

def image(code: int, axis: int) -> int:
    """``L P L†`` as a signed Pauli code, for ``axis`` in 0,1,2 = X,Y,Z."""
    wc, wn = divmod(code, 6)
    return wc if axis == 0 else (wn if axis == 2 else IPROD[wc][wn])


def conj(code: int, letter: str) -> tuple[int, str]:
    """``L P L†`` as a signed Pauli ``(±1, "XYZ")`` — the pair-space
    counterpart of ``cliffords._conj_pauli``, but a table read."""
    return decode(image(code, AXES.index(letter)))


def is_zaxis(code: int) -> bool:
    """Is ``w^N`` on the Z axis?  The CZ case split of the local algorithm."""
    return code % 6 in (2, 5)


def in_zset(code: int) -> bool:
    """Is L in the Z-set {I, Z, S, S†} — the Cliffords commuting with CZ?
    Exactly ``w^N = +Z``."""
    return code % 6 == 2


def is_hadamard(code: int) -> bool:
    """Is ``w^N = +X``, i.e. is this vertex in the Hadamard support F?"""
    return code % 6 == 0


# ── Pending measurement basis ─────────────────────────────────────────────────
# During the reduction chain X -> Y -> Z the pending basis is conjugated by
# the factor of U_w landing on the measured vertex: S† at a neighbour of the
# move site, HSH at the site itself.

PEND_SDG = [0] * 6
PEND_HSH = [0] * 6
for _p in range(3):
    PEND_SDG[_p] = encode(*_conj_pauli(_SDG_U8, AXES[_p]))
    PEND_SDG[_p + 3] = NEG6[PEND_SDG[_p]]
    PEND_HSH[_p] = encode(*_conj_pauli(_HSH_U8, AXES[_p]))
    PEND_HSH[_p + 3] = NEG6[PEND_HSH[_p]]


# ── The restricted alphabet ───────────────────────────────────────────────────
# On a restricted frame (w^N in {+Z, +X}) the letter is one of the eight of
# tab:restricted, written L_v = H^f S^d Z^s.

FDS: dict[int, tuple[int, int, int]] = {}
for _f in (0, 1):
    for _d in (0, 1):
        for _s in (0, 1):
            _m = list(_Z_U8) if _s else list(_IDENTITY_U8)
            if _d:
                _m = _mat2x2_mul(_S_U8, _m)
            if _f:
                _m = _mat2x2_mul(_H_U8, _m)
            FDS[pair_from_mat(_m)] = (_f, _d, _s)


def fds_to_pair(f: int, d: int, s: int) -> int:
    """``H^f S^d Z^s`` as a frame code."""
    return left_compose(ID_PAIR, *(("Z",) * s + ("S",) * d + ("H",) * f))


# ── Naming ────────────────────────────────────────────────────────────────────

def _build_names() -> dict[int, str]:
    """code -> short label.  Named gates win; the rest get a shortest word
    over {H, S, S†}."""
    from collections import deque
    names: dict[int, str] = {}
    for nm, m in (("I", _IDENTITY_U8), ("X", _X_U8), ("Y", _Y_U8), ("Z", _Z_U8),
                  ("H", _H_U8), ("S", _S_U8), ("S†", _SDG_U8)):
        names.setdefault(pair_from_mat(m), nm)
    words = {ID_PAIR: ""}
    q = deque([ID_PAIR])
    while q:                                    # leftmost letter applied last
        c = q.popleft()
        for nm, g in (("H", "H"), ("S", "S"), ("S†", "SDG")):
            c2 = LGATE[g][c]
            if c2 not in words:
                words[c2] = nm + words[c]
                q.append(c2)
    for c, w in words.items():
        names.setdefault(c, w or "I")
    return names


NAMES = _build_names()


def name(code: int) -> str:
    """Short label for a frame letter, for the UI's residual-correction list."""
    return NAMES.get(code, "?")


# ── Wire format ───────────────────────────────────────────────────────────────

def parse_frame(n: int, frame) -> list[int]:
    """Normalise an internal frame to exactly ``n`` codes: ``None`` (or a
    short list) means the identity frame.  Always returns a fresh list, so
    callers may mutate it freely."""
    if frame is None:
        return [ID_PAIR] * n
    return [frame[q] if q < len(frame) else ID_PAIR for q in range(n)]


def frame_from_wire(n: int, local_unitaries) -> list[int]:
    """``n`` frame codes from the JSON wire form: a list of 8-float matrices,
    or ``None`` / a short list for the identity frame."""
    if local_unitaries is None:
        return [ID_PAIR] * n
    out = []
    for q in range(n):
        raw = local_unitaries[q] if q < len(local_unitaries) else None
        out.append(ID_PAIR if raw is None or len(raw) < 8
                   else pair_from_mat([float(x) for x in raw]))
    return out


def frame_to_wire(codes: list[int]) -> list[list[float]]:
    """Frame codes -> the 8-float matrices the JSON API and the page speak."""
    return [list(PAIR_TO_MAT[c]) for c in codes]


# ── Machine check of every rule above against the matrix toolkit ──────────────

def _verify() -> None:
    assert len(VALID_PAIRS) == 24, VALID_PAIRS
    for m in CLIFFORDS_U8.values():
        c = pair_from_mat(m)
        wc, wn = divmod(c, 6)
        assert encode(*_conj_pauli(m, "Y")) == IPROD[wc][wn]
        assert pair_from_mat(_mat2x2_mul(m, _HSDGH_U8)) == RV_CENTER[c]
        assert pair_from_mat(_mat2x2_mul(m, _S_U8)) == RV_NEIGH[c]
        assert pair_from_mat(_mat2x2_mul(m, _HSH_U8)) == LC_CENTER[c]
        assert pair_from_mat(_mat2x2_mul(m, _SDG_U8)) == LC_NEIGH[c]
        assert pair_from_mat(_mat2x2_mul(m, _Z_U8)) == ZFOLD[c]
        assert pair_from_mat(_mat2x2_mul(m, _X_U8)) == XFOLD[c]
        assert pair_from_mat(_dag_u8(m)) == DAG[c]
        for g, gm in GATE_U8.items():
            assert pair_from_mat(_mat2x2_mul(gm, m)) == LGATE[g][c]
        for ax, letter in enumerate(AXES):
            assert conj(c, letter) == _conj_pauli(m, letter)
        # The Z-set {I, Z, S, S†} is exactly w^N = +Z.
        from .cliffords import _clifford_key as _k
        assert in_zset(c) == (_k(m) in _ZSET_KEYS)
    for c in VALID_PAIRS:                       # decomposition words compose back
        m = list(_IDENTITY_U8)
        for mv in DECOMP[c]:
            m = _mat2x2_mul(m, _HSH_U8 if mv == "X" else _SDG_U8)
        assert pair_from_mat(m) == c, (c, DECOMP[c])
    for (f, d, s), c in ((v, k) for k, v in FDS.items()):
        assert fds_to_pair(f, d, s) == c
    assert len(FDS) == 8


_ZSET_KEYS = frozenset(
    _clifford_key(m) for m in (_IDENTITY_U8, _Z_U8, _S_U8, _SDG_U8))

_verify()
