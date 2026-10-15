"""In-place stateful simulator over the Eulerian frame.

The functional core (``graph_ops``, ``gates``) copies its inputs, because the
HTTP server is stateless and every request carries a whole state.  A long
operation *stream* — a benchmark, a protocol run — wants the opposite: one
register, mutated in place, no allocation per op.  This class is that variant,
and the direct counterpart of the ``graphsim`` backends, which are in-place for
the same reason.

The algorithms are the ones documented in ``graph_ops`` and ``gates``; only
the calling convention differs.  ``benchmarks/bench_frames.py --selftest``
runs this class against the functional core op by op on random streams, so the
two cannot silently drift apart.

The measured vertex is kept and reset to |+> rather than deleted, so the
register size stays fixed along a stream.
"""
from __future__ import annotations

from .frames import (
    ID_PAIR,
    IPROD,
    LC_CENTER,
    LC_NEIGH,
    LGATE,
    PAIR_TO_MAT,
    PEND_HSH,
    PEND_SDG,
    RV_CENTER,
    RV_NEIGH,
    ZFOLD,
)
from .graph_ops import lc_inplace, toggle_edge

LZ = LGATE["Z"]


class EulerSim:
    """(G, L) with G as adjacency sets and L as one Eulerian code per vertex.
    Every frame update is one table read, every semantic test an integer
    compare."""

    def __init__(self, adj: list[set], pair_codes: list[int]):
        self.adj = [set(s) for s in adj]
        self.f = list(pair_codes)
        self.n_toggle = self.n_degenerate = 0

    # -- single-qubit physical gate (left composition) --
    def apply_local(self, v: int, table: list) -> None:
        """``table`` is one of ``frames.LGATE``'s entries."""
        self.f[v] = table[self.f[v]]

    # -- re-framing move R_v --
    def reframe(self, v: int) -> None:
        f, rvn = self.f, RV_NEIGH
        f[v] = RV_CENTER[f[v]]
        for u in self.adj[v]:
            f[u] = rvn[f[u]]
        lc_inplace(self.adj, v)

    # -- the directed move R_v^-1 used by the CZ reduction --
    def _lc_step(self, v: int) -> None:
        f, lcn = self.f, LC_NEIGH
        for u in self.adj[v]:
            f[u] = lcn[f[u]]
        f[v] = LC_CENTER[f[v]]
        lc_inplace(self.adj, v)

    # -- Pauli measurement via the reduction chain X -> Y -> Z --
    def measure(self, v: int, ax: int, invert: bool = False) -> None:
        """``ax`` is 0, 1 or 2 for X, Y, Z."""
        f, adj = self.f, self.adj
        wc, wn = divmod(f[v], 6)
        if wc % 3 == ax:                                 # transport: direct read
            sig, q = wc // 3, 0
        elif wn % 3 == ax:
            sig, q = wn // 3, 2
        else:
            wy = IPROD[wc][wn]
            sig, q = wy // 3, 1
        if q == 0:                                       # pending on the X axis
            if not adj[v]:                               # deterministic
                f[v] = ID_PAIR
                return
            b = min(adj[v], key=lambda j: (len(adj[j]), j))
            self.reframe(b)
            t = PEND_SDG[0]
            sig ^= t // 3; q = t % 3
        if q == 1:                                       # pending on the Y axis
            self.reframe(v)
            t = PEND_HSH[1]
            sig ^= t // 3; q = t % 3
        if sig ^ (1 if invert else 0):                   # Z-deletion correction
            zf = ZFOLD
            for u in adj[v]:
                f[u] = zf[f[u]]
        for u in adj[v]:
            adj[u].discard(v)
        adj[v].clear()
        f[v] = ID_PAIR

    # -- CZ: the Eulerian-frame rule (gates.apply_cz) --
    def _zalign(self, a: int, other: int, may_use_other: bool) -> bool:
        f, adj = self.f, self.adj
        if f[a] % 6 in (2, 5):                           # w^N_a on the Z axis
            return True
        if (f[a] // 6) % 3 == 2:                         # w^C_a on the Z axis
            nb = [u for u in adj[a] if u != other]
            if not nb and may_use_other and other in adj[a]:
                nb = [other]
            if not nb:
                return False
            self._lc_step(min(nb, key=lambda u: (len(adj[u]), u)))
        self._lc_step(a)
        return True

    def _cz_degenerate(self, i: int, j: int, stuck: int) -> None:
        f = self.f
        other = j if stuck == i else i
        neg = f[stuck] // 18 == 1                        # sign of w^C_stuck
        if self.adj[stuck]:                              # N(stuck) = {other}
            neg ^= f[other] % 6 == 5
            if not neg:                                  # +Z_iZ_j: CZ = Z_i
                f[i] = LZ[f[i]]
        elif neg:                                        # -Z_stuck
            f[other] = LZ[f[other]]

    def cz(self, i: int, j: int) -> None:
        if not self._zalign(i, j, True):
            self.n_degenerate += 1
            self._cz_degenerate(i, j, i)
            return
        if not self._zalign(j, i, False):
            self.n_degenerate += 1
            self._cz_degenerate(i, j, j)
            return
        self.n_toggle += 1
        f = self.f
        toggle_edge(self.adj, i, j)
        wni, wnj = f[i] % 6, f[j] % 6
        if wni == 5:                                     # w^N_i = -Z
            f[j] = ZFOLD[f[j]]
        if wnj == 5:
            f[i] = ZFOLD[f[i]]

    def pair_codes(self) -> list[int]:
        return list(self.f)

    def matrices(self) -> list[list[float]]:
        """The frame as 8-float matrices (display / wire)."""
        return [list(PAIR_TO_MAT[c]) for c in self.f]
