"""Single-qubit Clifford toolkit.

2x2 complex matrices stored as 8-float lists
[re00,im00,re01,im01,re10,im10,re11,im11]: named gate constants,
multiplication, dagger, Pauli conjugation, phase-canonical keys and
short names for the 24 single-qubit Cliffords (frame matrices / VOPs).
"""
from __future__ import annotations

_IDENTITY_U8 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
_S_U8        = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]   # S gate [[1,0],[0,i]]
_SDG_U8      = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0]   # S† gate [[1,0],[0,-i]]
_HSH_U8      = [0.5, 0.5, 0.5,-0.5, 0.5,-0.5, 0.5, 0.5]   # HSH gate
_Z_U8        = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0]   # Pauli Z
_SQRT2_F     = 0.5 ** 0.5
_H_U8        = [_SQRT2_F, 0.0, _SQRT2_F, 0.0, _SQRT2_F, 0.0, -_SQRT2_F, 0.0]  # Hadamard
_HSDGH_U8    = [0.5,-0.5, 0.5, 0.5, 0.5, 0.5, 0.5,-0.5]   # HS†H = (HSH)† ∝ √(iX)


def _mat2x2_mul(A: list[float], B: list[float]) -> list[float]:
    """A @ B for 2×2 complex matrices stored as 8-float lists."""
    a00=complex(A[0],A[1]); a01=complex(A[2],A[3])
    a10=complex(A[4],A[5]); a11=complex(A[6],A[7])
    b00=complex(B[0],B[1]); b01=complex(B[2],B[3])
    b10=complex(B[4],B[5]); b11=complex(B[6],B[7])
    r00=a00*b00+a01*b10; r01=a00*b01+a01*b11
    r10=a10*b00+a11*b10; r11=a10*b01+a11*b11
    return [round(r00.real,9),round(r00.imag,9),round(r01.real,9),round(r01.imag,9),
            round(r10.real,9),round(r10.imag,9),round(r11.real,9),round(r11.imag,9)]


def _parse_mats(n: int, local_unitaries: list | None) -> list[list[float]]:
    if local_unitaries is None:
        return [_IDENTITY_U8] * n
    out = []
    for qi in range(n):
        raw = local_unitaries[qi] if qi < len(local_unitaries) else _IDENTITY_U8
        out.append([float(x) for x in (raw if len(raw) >= 8 else _IDENTITY_U8)])
    return out


def _conj_pauli(mat8: list[float], p: str) -> tuple[int, str]:
    """Compute U·P·U† for P ∈ {X,Y,Z}. Returns (sign ∈ {±1}, Pauli ∈ {X,Y,Z})."""
    if p == "I":
        return 1, "I"
    u00 = complex(mat8[0], mat8[1]); u01 = complex(mat8[2], mat8[3])
    u10 = complex(mat8[4], mat8[5]); u11 = complex(mat8[6], mat8[7])
    # Matrix elements of P
    _P = {"X": (0, 1, 1, 0), "Y": (0, -1j, 1j, 0), "Z": (1, 0, 0, -1)}
    p00, p01, p10, p11 = _P[p]
    # UP = U @ P
    up00 = u00 * p00 + u01 * p10;  up01 = u00 * p01 + u01 * p11
    # UPU†: only need (0,0) and (0,1) to identify result
    r00 = up00 * u00.conjugate() + up01 * u01.conjugate()
    r01 = up00 * u10.conjugate() + up01 * u11.conjugate()
    EPS = 1e-6
    if abs(r00.real) > EPS:                    # result ∝ ±Z
        return (1 if r00.real > 0 else -1), "Z"
    if abs(r01.imag) > EPS:                    # result ∝ ±Y  (Y[0,1] = -i)
        return (1 if (r01 * 1j).real > 0 else -1), "Y"
    return (1 if r01.real > 0 else -1), "X"   # result ∝ ±X


def _dag_u8(m: list[float]) -> list[float]:
    """Conjugate transpose of a 2×2 matrix in 8-float form."""
    return [m[0], -m[1], m[4], -m[5], m[2], -m[3], m[6], -m[7]]


# ── Naming single-qubit Cliffords (for the canonicalisation report) ────────────
_X_U8 = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
_Y_U8 = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]


def _clifford_key(m: list[float]) -> tuple:
    """Phase-canonical key for a 2×2 unitary (Cliffords are defined up to phase)."""
    z = [complex(m[0], m[1]), complex(m[2], m[3]), complex(m[4], m[5]), complex(m[6], m[7])]
    ph = 1 + 0j
    for c in z:
        if abs(c) > 1e-6:
            ph = c / abs(c)
            break
    z = [c / ph for c in z]
    return tuple(round(c.real, 3) for c in z) + tuple(round(c.imag, 3) for c in z)


def _build_clifford_names() -> dict:
    """Map each single-qubit Clifford (phase-canonical key) to a short label.
    Named gates take priority; the rest get a shortest word in {H, S, S†}."""
    from collections import deque
    seeded = [("I", _IDENTITY_U8), ("X", _X_U8), ("Y", _Y_U8), ("Z", _Z_U8),
              ("H", _H_U8), ("S", _S_U8), ("S†", _SDG_U8)]
    names: dict = {}
    for nm, m in seeded:
        names.setdefault(_clifford_key(m), nm)
    gens = [("H", _H_U8), ("S", _S_U8), ("S†", _SDG_U8)]
    q = deque([(_IDENTITY_U8, "")])
    words = {_clifford_key(_IDENTITY_U8): ""}
    while q:
        m, w = q.popleft()
        for nm, gm in gens:
            m2 = _mat2x2_mul(gm, m)
            k = _clifford_key(m2)
            if k not in words:
                words[k] = nm + w           # operator order: leftmost applied last
                q.append((m2, nm + w))
    for k, w in words.items():
        names.setdefault(k, w or "I")
    return names


_CLIFFORD_NAMES = _build_clifford_names()


def _name_clifford(m: list[float]) -> str:
    return _CLIFFORD_NAMES.get(_clifford_key(m), "?")
