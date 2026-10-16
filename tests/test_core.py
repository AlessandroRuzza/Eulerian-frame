"""The compiled core (_core.CoreSim) makes exactly the same moves as sim.EulerSim.

Random op streams (CZ, Pauli measurements, re-framing, local gates, edge toggles, new
vertices) run on both; after every op the adjacency, the frame and the CZ counters must
be equal. Skipped when the core has not been built (python setup_core.py build_ext --inplace).
"""
import random

import pytest

from eulsim.frames import LGATE, VALID_PAIRS
from eulsim.sim import EulerSim

core = pytest.importorskip("eulsim._core")


def _random_state(rnd, n, p):
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rnd.random() < p:
                adj[i].add(j); adj[j].add(i)
    return adj, [rnd.choice(VALID_PAIRS) for _ in range(n)]


def _same(a: EulerSim, b) -> None:
    assert a.f == b.f
    assert a.adj == b.adj_sets()
    assert (a.n_toggle, a.n_degenerate) == (b.n_toggle, b.n_degenerate)


@pytest.mark.parametrize("seed", range(60))
def test_random_streams_match(seed):
    rnd = random.Random(seed)
    n = rnd.choice([2, 3, 5, 8, 16, 40])
    adj, codes = _random_state(rnd, n, rnd.choice([0.0, 0.1, 0.3, 0.7]))
    a, b = EulerSim(adj, codes), core.CoreSim(adj, codes)
    _same(a, b)
    gates = list(LGATE.values())
    for _ in range(400):
        r = rnd.random()
        n = len(a.f)
        if r < 0.35 and n >= 2:
            i, j = rnd.sample(range(n), 2)
            a.cz(i, j); b.cz(i, j)
        elif r < 0.55:
            v, ax, inv = rnd.randrange(n), rnd.randrange(3), rnd.random() < 0.5
            a.measure(v, ax, inv); b.measure(v, ax, inv)
        elif r < 0.65:
            v = rnd.randrange(n)
            a.reframe(v); b.reframe(v)
        elif r < 0.85:
            v, g = rnd.randrange(n), rnd.choice(gates)
            a.apply_local(v, g); b.apply_local(v, g)
        elif r < 0.95 and n >= 2:
            i, j = rnd.sample(range(n), 2)
            a.toggle_edge(i, j); b.toggle_edge(i, j)
            assert a.has_edge(i, j) == b.has_edge(i, j)
        else:
            c = rnd.choice(VALID_PAIRS)
            assert a.add_vertex(c) == b.add_vertex(c)
        _same(a, b)
        v = rnd.randrange(len(a.f))
        assert a.neighbors(v) == b.neighbors(v) and a.degree(v) == b.degree(v)


def test_dense_local_complementation():
    """High-degree LCs, the case the sorted-array merge is for."""
    rnd = random.Random(7)
    adj, codes = _random_state(rnd, 120, 0.5)
    a, b = EulerSim(adj, codes), core.CoreSim(adj, codes)
    for _ in range(200):
        v = rnd.randrange(120)
        a.reframe(v); b.reframe(v)
        _same(a, b)
