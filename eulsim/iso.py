"""Graph isomorphism up to renaming of the qubits, by canonical labelling.

Two graphs are isomorphic when one can be turned into the other by permuting
vertices; for us that is "the same entanglement structure under a renaming of
the qubits", a coarser relation than the labelled LC-equivalence of lc_orbit
and independent of the frame.

Algorithm: individualization-refinement (IR), the scheme every practical
isomorphism tool uses (nauty, bliss, Traces).  `_refine` runs 1-dimensional
Weisfeiler-Leman colour refinement to an equitable partition; when that leaves
a non-discrete partition the search individualizes one vertex of the first
smallest non-singleton cell, refines again, and recurses.  Every leaf of that
tree is a discrete colouring, i.e. a vertex ordering; the canonical form is the
lexicographically least adjacency bitstring over all leaves, and the ordering
that produces it is the canonical labelling.  Isomorphism is then decided by
comparing canonical forms, which also hands back an explicit vertex mapping.

Worst case is exponential (IR trees blow up on highly regular graphs), which is
why the caller passes a qubit cap and a node budget; `None` comes back when
either is hit, never a guess.

A note on Babai's quasipolynomial algorithm (2015/17) and the canonical form
that followed (2019): exp((log n)^O(1)) is asymptotic, the algorithm is a deep
group-theoretic construction (Luks' framework plus local certificates and the
split-or-Johnson routine) with no practical implementation anywhere, and its
crossover against IR is far beyond the few dozen qubits a graph state editor
ever holds.  It is not implemented here, and this module makes no claim to it;
what it shares with that line of work is the colour-refinement core.
"""
from __future__ import annotations

ISO_MAX_QUBITS = 24          # default cap on n for the isomorphism check
ISO_NODE_BUDGET = 200000     # default cap on IR search-tree nodes


def _refine(adj: list[set[int]], n: int, colors: list[int]) -> list[int]:
    """1-WL colour refinement: repeat "recolour by (colour, multiset of
    neighbour colours)" until the partition stops splitting.  Colours come back
    normalised to 0..k-1, ordered by their signature, so the result depends only
    on the graph and the input colouring — not on the vertex numbering."""
    cur = list(colors)
    while True:
        sig = [(cur[v], tuple(sorted(cur[u] for u in adj[v]))) for v in range(n)]
        order = {s: i for i, s in enumerate(sorted(set(sig)))}
        new = [order[s] for s in sig]
        if new == cur:
            return cur
        cur = new


def _target_cell(colors: list[int], n: int) -> list[int]:
    """The cell the search branches on: the smallest non-singleton colour class,
    ties going to the smallest colour.  Empty when the colouring is discrete."""
    cells: dict[int, list[int]] = {}
    for v in range(n):
        cells.setdefault(colors[v], []).append(v)
    best: list[int] = []
    for c in sorted(cells):
        cell = cells[c]
        if len(cell) > 1 and (not best or len(cell) < len(best)):
            best = cell
    return best


def _certificate(adj: list[set[int]], n: int, order: list[int]) -> tuple[int, ...]:
    """Upper-triangle adjacency bits of the graph relabelled by `order`
    (order[k] = the vertex placed at canonical position k)."""
    pos = [0] * n
    for k, v in enumerate(order):
        pos[v] = k
    rows = [[0] * n for _ in range(n)]
    for v in range(n):
        for u in adj[v]:
            rows[pos[v]][pos[u]] = 1
    return tuple(rows[i][j] for i in range(n) for j in range(i + 1, n))


def _orbits_fixing(n: int, gens: list[list[int]], prefix: list[int]) -> list[int]:
    """Union-find orbits of the vertices under the automorphisms found so far
    that fix every individualized vertex.  Those automorphisms map one branch of
    the target cell onto another, so only one vertex per orbit has to be tried —
    pruning by a subgroup of the true stabiliser, which is always sound."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in gens:
        if any(g[p] != p for p in prefix):
            continue
        for v in range(n):
            a, b = find(v), find(g[v])
            if a != b:
                parent[max(a, b)] = min(a, b)
    return [find(v) for v in range(n)]


def canonical_labelling(adj: list[set[int]], n: int, *, node_budget: int | None = None):
    """Canonical form of `adj` and the labelling that realises it.

    Returns (certificate, order, nodes, capped): `certificate` is the canonical
    adjacency bitstring, `order[k]` the vertex sitting at canonical position k,
    `nodes` the number of IR tree nodes visited and `capped` True when the node
    budget ran out — in which case the certificate is only the best found so far
    and must not be used to decide anything.

    Leaves whose certificate ties the best one hand back an automorphism (the
    permutation between the two orderings), and those generators prune the
    symmetric branches of the search — without them a graph with a large
    automorphism group, an empty graph or a K_n say, costs n! leaves."""
    budget = ISO_NODE_BUDGET if node_budget is None else node_budget
    if n == 0:
        return (), [], 0, False
    state = {"nodes": 0, "capped": False, "best": None, "order": None,
             "gens": [], "seen": set()}

    def descend(colors: list[int], prefix: list[int]):
        if state["capped"]:
            return
        state["nodes"] += 1
        if state["nodes"] > budget:
            state["capped"] = True
            return
        colors = _refine(adj, n, colors)
        cell = _target_cell(colors, n)
        if not cell:
            order = sorted(range(n), key=lambda v: colors[v])
            cert = _certificate(adj, n, order)
            if state["best"] is None or cert < state["best"]:
                state["best"], state["order"] = cert, order
            elif cert == state["best"]:
                # Both orderings relabel the graph the same way, so composing
                # them is an automorphism.
                g = [0] * n
                for k in range(n):
                    g[state["order"][k]] = order[k]
                t = tuple(g)
                # Deduplicated: the same symmetry turns up on many leaves, and
                # the orbit computation is linear in the generator count.
                if any(x != i for i, x in enumerate(t)) and t not in state["seen"]:
                    state["seen"].add(t)
                    state["gens"].append(g)
            return
        done: set[int] = set()
        reps = _orbits_fixing(n, state["gens"], prefix)
        ngens = len(state["gens"])
        for v in cell:
            if len(state["gens"]) != ngens:          # new symmetries since last time
                ngens = len(state["gens"])
                reps = _orbits_fixing(n, state["gens"], prefix)
            if reps[v] in done:
                continue                     # symmetric to a branch already taken
            done.add(reps[v])
            child = list(colors)
            child[v] = -1                    # a colour strictly below every other
            descend(child, prefix + [v])
            if state["capped"]:
                return

    descend([0] * n, [])
    return state["best"], state["order"], state["nodes"], state["capped"]


def _invariants(adj: list[set[int]], n: int):
    """Cheap isomorphism invariants: edge count, degree sequence, triangle count
    and the refined colour-class profile.  Different invariants prove the graphs
    are not isomorphic without entering the search."""
    degs = sorted(len(adj[v]) for v in range(n))
    m = sum(degs) // 2
    tri = sum(len(adj[v] & adj[u]) for v in range(n) for u in adj[v] if u > v) // 3
    colors = _refine(adj, n, [0] * n)
    profile = sorted(colors.count(c) for c in set(colors))
    return (m, tuple(degs), tri, tuple(profile))


def isomorphic(adj_a: list[set[int]], n_a: int, adj_b: list[set[int]], n_b: int, *,
               max_qubits: int | None = None, node_budget: int | None = None):
    """Decide G_A ≅ G_B up to a renaming of the qubits.

    Returns (result, msg, info) with result True / False / None (undecided:
    n above the cap, or the node budget exhausted).  `info` carries the vertex
    mapping when they are isomorphic, plus the search statistics."""
    cap = ISO_MAX_QUBITS if max_qubits is None else max_qubits
    if n_a != n_b:
        return False, f"Different qubit counts: {n_a} vs {n_b}", {}
    n = n_a
    if n == 0:
        return True, "Both empty", {"mapping": []}
    if n > cap:
        return None, f"n={n} > {cap}: above the isomorphism cap", {}

    inv_a, inv_b = _invariants(adj_a, n), _invariants(adj_b, n)
    if inv_a != inv_b:
        which = ("edge counts", "degree sequences", "triangle counts",
                 "colour-refinement profiles")
        diff = next(w for w, a, b in zip(which, inv_a, inv_b) if a != b)
        return False, f"Different {diff}", {"invariant": diff}

    cert_a, order_a, nodes_a, capped_a = canonical_labelling(adj_a, n, node_budget=node_budget)
    cert_b, order_b, nodes_b, capped_b = canonical_labelling(adj_b, n, node_budget=node_budget)
    nodes = nodes_a + nodes_b
    if capped_a or capped_b:
        return None, f"Search budget exhausted after {nodes} nodes", {"nodes": nodes}
    if cert_a != cert_b:
        return False, f"Different canonical forms ({nodes} IR nodes)", {"nodes": nodes}

    # order_x[k] is the vertex of x at canonical position k, so A's vertex
    # order_a[k] corresponds to B's vertex order_b[k].
    mapping = [0] * n
    for k in range(n):
        mapping[order_a[k]] = order_b[k]
    return True, f"Same canonical form ({nodes} IR nodes)", {
        "mapping": mapping, "nodes": nodes}
