"""LC-orbit BFS: equivalence check, canonical representative, orbit size.

Graphs are adjacency sets; the hashable orbit key is the sorted edge tuple
(O(m log m) per state instead of the O(n^2) dense row tuple).
"""
from __future__ import annotations

from .graph_ops import local_complement

# ─── LC-equivalence ───────────────────────────────────────────────────────────

MAX_BFS_STATES = 60000
NODE_LIMIT = 20


def _edge_key(adj, n):
    """Canonical edge-set key: sorted tuple of (i,j) pairs with i<j.
    Its length is the edge count, so (len(key), key) is the
    fewest-edges-then-lexicographic order used by lc_canonical."""
    return tuple(sorted((i, j) for i in range(n) for j in adj[i] if i < j))


def lc_equiv_labeled(adj1, n, adj2, *, max_bfs=None, node_limit=None):
    """BFS: can adj2 be reached from adj1 by LC sequence? (labeled, fixed vertex indices)
    Returns (result, orbit_size, msg) where result is True/False/None (too large)."""
    cap   = max_bfs    if max_bfs    is not None else MAX_BFS_STATES
    nlim  = node_limit if node_limit is not None else NODE_LIMIT
    if n == 0:
        return True, 1, "Both empty"
    if n > nlim:
        return None, 0, f"n={n} > {nlim}: too large for BFS"

    target = _edge_key(adj2, n)
    start  = _edge_key(adj1, n)
    if start == target:
        return True, 1, "Identical graphs"

    visited = {start}
    queue   = [adj1]

    while queue:
        if len(visited) >= cap:
            return None, len(visited), f"Orbit > {cap} states explored — undecided"
        cur = queue.pop(0)
        for v in range(n):
            new_adj, _ = local_complement(cur, n, v)
            h = _edge_key(new_adj, n)
            if h == target:
                return True, len(visited) + 1, f"LC-equivalent (orbit searched: {len(visited)+1})"
            if h not in visited:
                visited.add(h)
                queue.append(new_adj)

    return False, len(visited), f"Not LC-equivalent (full orbit: {len(visited)} graphs)"


def lc_canonical(adj, n, *, max_bfs=None, node_limit=None):
    """Return the LC-orbit representative with fewest edges (ties broken lexicographically
    by sorted edge list). Returns (rep_adj, orbit_size, capped, msg)."""
    cap  = max_bfs    if max_bfs    is not None else MAX_BFS_STATES
    nlim = node_limit if node_limit is not None else NODE_LIMIT
    if n == 0:
        return adj, 0, False, "Empty graph"
    if n > nlim:
        return adj, 0, False, f"n={n} > {nlim}: too large for BFS — no canonicalisation"

    start = _edge_key(adj, n)
    visited = {start}
    queue = [adj]
    capped = False
    best_adj = adj
    best_key = (len(start), start)

    while queue:
        if len(visited) >= cap:
            capped = True
            break
        cur = queue.pop(0)
        for v in range(n):
            new_adj, _ = local_complement(cur, n, v)
            h = _edge_key(new_adj, n)
            if h not in visited:
                visited.add(h)
                queue.append(new_adj)
                k = (len(h), h)
                if k < best_key:
                    best_adj, best_key = new_adj, k

    sz  = len(visited)
    msg = f"Representative found (orbit {'≥' if capped else '='} {sz} graphs)"
    return best_adj, sz, capped, msg


def lc_orbit_size(adj, n, *, max_bfs=None):
    """Return the LC-orbit size of adj (labeled). Capped at max_bfs (default MAX_BFS_STATES)."""
    cap = max_bfs if max_bfs is not None else MAX_BFS_STATES
    if n == 0 or n > 10:
        return None
    visited = {_edge_key(adj, n)}
    queue = [adj]
    while queue and len(visited) < cap:
        cur = queue.pop(0)
        for v in range(n):
            new_adj, _ = local_complement(cur, n, v)
            h = _edge_key(new_adj, n)
            if h not in visited:
                visited.add(h)
                queue.append(new_adj)
    return len(visited)
