"""Dense state-vector expansion of |G,L> = (tensor_i L_i)|G> for display.

The only place the simulator needs real 2x2 matrices rather than frame codes:
amplitudes are complex numbers, so each frame letter is materialised through
``frames.PAIR_TO_MAT`` before the sweep.  This is a display path, capped at
MAX_SV_QUBITS qubits, so the conversion never shows up in the compute cost.

The graph enters only through its edge list (adjacency sets), so the phase
sweep costs O(2^n * m) rather than O(2^n * n^2)."""
from __future__ import annotations

from .cliffords import _mat2x2_mul
from .frames import PAIR_TO_MAT, parse_frame

MAX_SV_QUBITS = 10  # state-vector display cut-off

def compute_state_vector(
    adj: list[set[int]], n: int,
    frame: list[int] | None = None,
    display_basis: list[str] | None = None,
) -> list[dict] | None:
    """State vector of |G,L⟩ = (⊗_i L_i)|G⟩.
    frame: one Eulerian code per qubit (identity frame assumed when None).
    display_basis: per-qubit 'X'/'Y'/'Z' — basis in which amplitudes are expressed.
    'Z' (computational) assumed when None. Bit 0/1 of qubit q then means
    |0⟩/|1⟩ (Z), |+⟩/|-⟩ (X) or |+i⟩/|-i⟩ (Y)."""
    if n == 0 or n > MAX_SV_QUBITS:
        return None

    mats = [list(PAIR_TO_MAT[c]) for c in parse_frame(n, frame)]
    # Coefficients in basis B are ⟨b_k|ψ⟩, i.e. apply B† to ψ.
    # X: B = H (self-adjoint).  Y: B = SH → B† = H·S†.
    _S2 = 0.5 ** 0.5
    _BASIS_DAG = {
        "X": [_S2, 0.0,  _S2, 0.0, _S2, 0.0, -_S2, 0.0],
        "Y": [_S2, 0.0,  0.0, -_S2, _S2, 0.0,  0.0, _S2],
    }
    if display_basis:
        for qi in range(min(n, len(display_basis))):
            bd = _BASIS_DAG.get(str(display_basis[qi]).upper())
            if bd:
                mats[qi] = _mat2x2_mul(bd, mats[qi])

    def is_identity(m: list[float]) -> bool:
        return (abs(m[0] - 1) < 1e-9 and abs(m[1]) < 1e-9 and
                abs(m[2]) < 1e-9 and abs(m[3]) < 1e-9 and
                abs(m[4]) < 1e-9 and abs(m[5]) < 1e-9 and
                abs(m[6] - 1) < 1e-9 and abs(m[7]) < 1e-9)

    dim = 1 << n
    norm = 0.5 ** (n / 2)
    psi: list[complex] = [0j] * dim
    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    for x in range(dim):
        phase = sum(1 for i, j in edges
                    if (x >> i) & 1 and (x >> j) & 1) % 2
        psi[x] = complex((-1) ** phase * norm)

    for qi, m in enumerate(mats):
        if is_identity(m):
            continue
        # Matrix elements as complex numbers: [[a,b],[c,d]]
        a = complex(m[0], m[1])
        b = complex(m[2], m[3])
        c = complex(m[4], m[5])
        d = complex(m[6], m[7])
        stride = 1 << qi
        for outer in range(0, dim, stride << 1):
            for inner in range(outer, outer + stride):
                i0, i1 = inner, inner + stride
                a0, a1 = psi[i0], psi[i1]
                psi[i0] = a * a0 + b * a1
                psi[i1] = c * a0 + d * a1

    EPS = 1e-10
    out = []
    for x in range(dim):
        amp = psi[x]
        re = round(amp.real, 9)
        im = round(amp.imag, 9)
        prob = round(re * re + im * im, 9)
        if prob < EPS:
            re = im = prob = 0.0
        out.append({"basis": format(x, f"0{n}b"), "re": re, "im": im, "prob": prob})
    return out
