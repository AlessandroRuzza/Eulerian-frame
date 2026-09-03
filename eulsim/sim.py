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
    DAG,
    DECOMP,
    ID_PAIR,
    IPROD,
    LC_CENTER,
    LC_NEIGH,
    PAIR_TO_MAT,
    PEND_HSH,
    PEND_SDG,
    RV_CENTER,
    RV_NEIGH,
    ZFOLD,
)
from .gates import _apply_cz_tableau, build_cz_table
from .graph_ops import lc_inplace, set_edge, toggle_edge


class EulerSim:
    """(G, L) with G as adjacency sets and L as one Eulerian code per vertex.
    Every frame update is one table read, every semantic test an integer
    compare."""

    def __init__(self, adj: list[set], pair_codes: list[int]):
        self.adj = [set(s) for s in adj]
        self.f = list(pair_codes)
        self.n_diag = self.n_table = self.n_fallback = 0

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

    # -- CZ: the local Anders-Briegel algorithm --
    def _reduce_vop_at(self, a: int, avoid: int) -> None:
        word = DECOMP[DAG[self.f[a]]]
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
                b = min(nb, key=lambda j: (f[j] == ID_PAIR, j))
                self._lc_step(b)

    def cz(self, i: int, j: int) -> None:
        adj, f = self.adj, self.f

        def has_others(a: int, b: int) -> bool:
            return any(k != b for k in adj[a])

        both_zaxis = (f[i] % 6) % 3 == 2 and (f[j] % 6) % 3 == 2

        def commutes(a: int) -> bool:
            return both_zaxis or f[a] % 6 == 2           # Z-set: w^N = +Z

        if has_others(i, j) and not commutes(i):
            self._reduce_vop_at(i, avoid=j)
        if has_others(j, i) and not commutes(j):
            self._reduce_vop_at(j, avoid=i)
        if has_others(i, j) and not commutes(i):
            self._reduce_vop_at(i, avoid=j)

        wni, wnj = f[i] % 6, f[j] % 6
        if wni % 3 == 2 and wnj % 3 == 2:                # diagonal-axis case
            self.n_diag += 1
            toggle_edge(adj, i, j)
            if wni == 5:                                 # w^N_i = -Z
                f[j] = ZFOLD[f[j]]
            if wnj == 5:
                f[i] = ZFOLD[f[i]]
            return
        res = build_cz_table().get((1 if j in adj[i] else 0, f[i], f[j],
                                    has_others(i, j), has_others(j, i)))
        if res is None:                                  # defensive fallback
            self.n_fallback += 1
            new_adj, new_f = _apply_cz_tableau(adj, len(adj), i, j, f)
            self.adj = new_adj
            self.f = new_f
            return
        self.n_table += 1
        zeta2, pa2, pb2 = res
        set_edge(adj, i, j, zeta2)
        f[i], f[j] = pa2, pb2

    def pair_codes(self) -> list[int]:
        return list(self.f)

    def matrices(self) -> list[list[float]]:
        """The frame as 8-float matrices (display / wire)."""
        return [list(PAIR_TO_MAT[c]) for c in self.f]
