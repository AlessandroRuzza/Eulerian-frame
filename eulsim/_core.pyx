# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled counterpart of ``sim.EulerSim``: same rules, same tables, same choices.

The adjacency is owned by C++ as one sorted ``vector<int>`` per vertex, so local
complementation is a merge of sorted arrays (contiguous memory, no hashing) instead of
Python set updates. The frame stays a Python list of codes (``self.f``), so callers read
and write it exactly as with ``EulerSim``.

Every rule table is copied at import from ``frames`` (the machine-verified ones), and
every free choice (min-degree neighbour, ties by index) is made as in ``sim.py``, so the
two cores produce the same representative op by op (``benchmarks/bench_frames.py
--selftest`` and ``tests/test_core.py`` check this on random streams).
"""
from libcpp.vector cimport vector

from .frames import (ID_PAIR, IPROD, LC_CENTER, LC_NEIGH, LGATE, PEND_HSH, PEND_SDG, RV_CENTER,
                     RV_NEIGH, ZFOLD)

cdef int T_RVC[36]
cdef int T_RVN[36]
cdef int T_LCC[36]
cdef int T_LCN[36]
cdef int T_ZF[36]
cdef int T_LZ[36]
cdef int T_IPROD[36]          # IPROD[p][q] at 6*p + q
cdef int C_ID = ID_PAIR
cdef int PEND_SDG_X = PEND_SDG[0]
cdef int PEND_HSH_Y = PEND_HSH[1]


cdef void _fill(int* dst, list src):
    cdef int k
    for k in range(36):
        dst[k] = -1 if src[k] is None else src[k]


_fill(T_RVC, RV_CENTER)
_fill(T_RVN, RV_NEIGH)
_fill(T_LCC, LC_CENTER)
_fill(T_LCN, LC_NEIGH)
_fill(T_ZF, ZFOLD)
_fill(T_LZ, LGATE["Z"])
_fill(T_IPROD, [IPROD[p][q] for p in range(6) for q in range(6)])


cdef inline Py_ssize_t _find(vector[int]& a, int x) noexcept:
    """Index of the first element >= x (binary search)."""
    cdef Py_ssize_t lo = 0, hi = <Py_ssize_t>a.size(), mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline bint _has(vector[int]& a, int x) noexcept:
    cdef Py_ssize_t k = _find(a, x)
    return k < <Py_ssize_t>a.size() and a[k] == x


cdef inline void _insert(vector[int]& a, int x):
    cdef Py_ssize_t k = _find(a, x)
    if k < <Py_ssize_t>a.size() and a[k] == x:
        return
    a.insert(a.begin() + k, x)


cdef inline void _erase(vector[int]& a, int x):
    cdef Py_ssize_t k = _find(a, x)
    if k < <Py_ssize_t>a.size() and a[k] == x:
        a.erase(a.begin() + k)


cdef class CoreSim:
    """(G, L): G as sorted C++ adjacency arrays, L as a Python list of frame codes."""
    cdef vector[vector[int]] _adj
    cdef vector[int] _buf
    cdef public list f
    cdef public long n_toggle
    cdef public long n_degenerate

    def __init__(self, adj, pair_codes):
        cdef int i
        self.f = list(pair_codes)
        self._adj.resize(len(self.f))
        for i, nb in enumerate(adj):
            for u in sorted(nb):
                self._adj[i].push_back(u)
        self.n_toggle = self.n_degenerate = 0

    # ── graph access (the Python-facing API shared with EulerSim) ────────────────
    def add_vertex(self, int code) -> int:
        self._adj.push_back(vector[int]())
        self.f.append(code)
        return len(self.f) - 1

    def neighbors(self, int i) -> set:
        return {u for u in self._adj[i]}

    def degree(self, int i) -> int:
        return self._adj[i].size()

    def has_edge(self, int i, int j) -> bool:
        return _has(self._adj[i], j)

    def toggle_edge(self, int i, int j):
        self._toggle(i, j)

    def adj_sets(self) -> list:
        return [{u for u in row} for row in self._adj]

    @property
    def adj(self) -> list:
        """Copy of the adjacency as Python sets (read-only snapshot, O(n + m))."""
        return self.adj_sets()

    # ── primitives ──────────────────────────────────────────────────────────────
    cdef void _toggle(self, int i, int j):
        if _has(self._adj[i], j):
            _erase(self._adj[i], j)
            _erase(self._adj[j], i)
        else:
            _insert(self._adj[i], j)
            _insert(self._adj[j], i)

    cdef void _lc(self, int v):
        """tau_v: N(u) <- N(u) xor (N(v) minus {u}) for u in N(v), by sorted merges."""
        cdef vector[int] nb = self._adj[v]
        cdef Py_ssize_t k, a, b, na, nn = <Py_ssize_t>nb.size()
        cdef int u, x, y
        for k in range(nn):
            u = nb[k]
            self._buf.clear()
            na = <Py_ssize_t>self._adj[u].size()
            a = b = 0
            while a < na or b < nn:
                if b < nn and nb[b] == u:
                    b += 1
                    continue
                if b >= nn:
                    self._buf.push_back(self._adj[u][a]); a += 1
                elif a >= na:
                    self._buf.push_back(nb[b]); b += 1
                else:
                    x = self._adj[u][a]
                    y = nb[b]
                    if x < y:
                        self._buf.push_back(x); a += 1
                    elif y < x:
                        self._buf.push_back(y); b += 1
                    else:
                        a += 1; b += 1
            self._adj[u].swap(self._buf)

    cdef int _min_neighbor(self, int a, int skip) noexcept:
        """Neighbour of a (other than skip) with the fewest neighbours, ties by index; -1 if none."""
        cdef int best = -1, u
        cdef size_t bd = 0, d
        for u in self._adj[a]:
            if u == skip:
                continue
            d = self._adj[u].size()
            if best < 0 or d < bd:
                best, bd = u, d
        return best

    # ── single-qubit physical gate (left composition) ───────────────────────────
    def apply_local(self, int v, table):
        self.f[v] = table[self.f[v]]

    # ── re-framing move R_v ─────────────────────────────────────────────────────
    cpdef void reframe(self, int v):
        cdef list f = self.f
        cdef int u
        f[v] = T_RVC[<int>f[v]]
        for u in self._adj[v]:
            f[u] = T_RVN[<int>f[u]]
        self._lc(v)

    cdef void _lc_step(self, int v):
        cdef list f = self.f
        cdef int u
        for u in self._adj[v]:
            f[u] = T_LCN[<int>f[u]]
        f[v] = T_LCC[<int>f[v]]
        self._lc(v)

    # ── Pauli measurement via the reduction chain X -> Y -> Z ───────────────────
    cpdef void measure(self, int v, int ax, bint invert=False):
        cdef list f = self.f
        cdef int code = f[v], wc = code // 6, wn = code % 6, wy, sig, q, t, b, u
        if wc % 3 == ax:
            sig, q = wc // 3, 0
        elif wn % 3 == ax:
            sig, q = wn // 3, 2
        else:
            wy = T_IPROD[6 * wc + wn]
            sig, q = wy // 3, 1
        if q == 0:
            if self._adj[v].empty():
                f[v] = C_ID
                return
            b = self._min_neighbor(v, -1)
            self.reframe(b)
            t = PEND_SDG_X
            sig ^= t // 3; q = t % 3
        if q == 1:
            self.reframe(v)
            t = PEND_HSH_Y
            sig ^= t // 3; q = t % 3
        if sig ^ (1 if invert else 0):
            for u in self._adj[v]:
                f[u] = T_ZF[<int>f[u]]
        for u in self._adj[v]:
            _erase(self._adj[u], v)
        self._adj[v].clear()
        f[v] = C_ID

    # ── CZ: the Eulerian-frame rule ─────────────────────────────────────────────
    cdef bint _zalign(self, int a, int other, bint may_use_other):
        cdef int code = self.f[a], u
        if code % 6 == 2 or code % 6 == 5:
            return True
        if (code // 6) % 3 == 2:
            u = self._min_neighbor(a, other)
            if u < 0 and may_use_other and _has(self._adj[a], other):
                u = other
            if u < 0:
                return False
            self._lc_step(u)
        self._lc_step(a)
        return True

    cdef void _cz_degenerate(self, int i, int j, int stuck):
        cdef list f = self.f
        cdef int other = j if stuck == i else i
        cdef bint neg = (<int>f[stuck]) // 18 == 1
        if not self._adj[stuck].empty():
            neg ^= (<int>f[other]) % 6 == 5
            if not neg:
                f[i] = T_LZ[<int>f[i]]
        elif neg:
            f[other] = T_LZ[<int>f[other]]

    cpdef void cz(self, int i, int j):
        cdef list f
        cdef int wni, wnj
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
        self._toggle(i, j)
        wni, wnj = (<int>f[i]) % 6, (<int>f[j]) % 6
        if wni == 5:
            f[j] = T_ZF[<int>f[j]]
        if wnj == 5:
            f[i] = T_ZF[<int>f[i]]

    def pair_codes(self) -> list:
        return list(self.f)
