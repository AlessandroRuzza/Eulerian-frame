"""LC-orbit BFS: equivalence check, canonical representative, orbit size.

Graphs are adjacency sets; the hashable orbit key is the sorted edge tuple
(O(m log m) per state instead of the O(n^2) dense row tuple).
"""
from __future__ import annotations

from .graph_ops import local_complement
from .iso import canonical_labelling

# ─── LC-equivalence ───────────────────────────────────────────────────────────

MAX_BFS_STATES = 60000
NODE_LIMIT = 20


def _edge_key(adj, n):
    """Canonical edge-set key: sorted tuple of (i,j) pairs with i<j.
    Its length is the edge count, the first key of lc_canonical's order."""
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


def _tie_break(mins, n):
    """Pick among the fewest-edge orbit members by isomorphism class first.

    The key is (canonical form, labelled edge key): the canonical form depends
    only on the shape, so relabelling the input graph cannot change which shape
    wins; the labelled key then picks one concrete member of that shape.
    Returns (adj, shapes, ok) — ok is False when an IR search ran out of budget
    and the choice fell back to the labelled order alone."""
    if len(mins) == 1:
        return mins[0], 1, True
    keyed = []
    for a in mins:
        cert, _, _, capped = canonical_labelling(a, n)
        if capped:
            return min(mins, key=lambda b: _edge_key(b, n)), None, False
        keyed.append((cert, _edge_key(a, n), a))
    keyed.sort(key=lambda t: t[:2])
    return keyed[0][2], len({t[0] for t in keyed}), True


def _fewest_edge_members(adj, n, cap):
    """BFS the labelled LC orbit of adj and collect its fewest-edge members.
    Returns (mins, orbit_size, capped); capped means the BFS stopped at `cap`
    states, so the orbit is partial and the true minimum may be missing."""
    start = _edge_key(adj, n)
    visited = {start}
    queue = [adj]
    capped = False
    best_m = len(start)
    mins = [adj]          # every orbit member seen with best_m edges

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
                if len(h) < best_m:
                    best_m, mins = len(h), [new_adj]
                elif len(h) == best_m:
                    mins.append(new_adj)
    return mins, len(visited), capped


def lc_canonical(adj, n, *, max_bfs=None, node_limit=None):
    """Return the LC-orbit representative: fewest edges, ties broken by the
    canonical form of the graph (iso.canonical_labelling), then by the sorted
    labelled edge list. Returns (rep_adj, orbit_size, capped, msg).

    Breaking ties by canonical form makes the choice invariant under relabelling
    the qubits: rep(pi(G)) is isomorphic to rep(G) for every permutation pi, and
    (edge count, canonical form) of the representative is a complete invariant
    for LC-equivalence up to relabelling. A plain lexicographic tie-break is not
    — from 7 qubits on an orbit can hold fewest-edge graphs of different shapes,
    and which one sorts first then depends on the labels (LC_isomorph.tex, §5).
    Not guaranteed when the BFS is capped: a partial orbit may miss the minimum."""
    cap  = max_bfs    if max_bfs    is not None else MAX_BFS_STATES
    nlim = node_limit if node_limit is not None else NODE_LIMIT
    if n == 0:
        return adj, 0, False, "Empty graph"
    if n > nlim:
        return adj, 0, False, f"n={n} > {nlim}: too large for BFS — no canonicalisation"

    mins, sz, capped = _fewest_edge_members(adj, n, cap)
    best_adj, shapes, ok = _tie_break(mins, n)
    msg = f"Representative found (orbit {'≥' if capped else '='} {sz} graphs"
    if shapes and shapes > 1:
        msg += f"; {shapes} fewest-edge shapes, tie broken by canonical form"
    if not ok:
        msg += "; isomorphism budget hit, tie broken by labels — not relabel-invariant"
    msg += ")"
    return best_adj, sz, capped, msg


def lc_iso_canonical(adj, n, *, max_bfs=None, node_limit=None):
    """Return the representative of the LC+relabelling class of adj: among the
    fewest-edge graphs of the LC orbit, the one with the least canonical form
    (iso.canonical_labelling), written in its canonical labels.
    Returns (rep_adj, orbit_size, capped, exact, msg).

    Unlike lc_canonical, which hands back a member of the labelled orbit, the
    result is one fixed labelled graph per class: rep(pi(G)) == rep(G) for every
    permutation pi and rep(tau_v(G)) == rep(G) for every v, so two graphs are
    LC-equivalent up to relabelling exactly when their representatives are
    equal. It is LC-equivalent to a relabelling of adj, in general not to adj
    itself. `exact` is False when the BFS was capped or an IR search ran out of
    budget — the result is then the input graph or a best effort, not canonical."""
    cap  = max_bfs    if max_bfs    is not None else MAX_BFS_STATES
    nlim = node_limit if node_limit is not None else NODE_LIMIT
    if n == 0:
        return adj, 0, False, True, "Empty graph"
    if n > nlim:
        return adj, 0, False, False, f"n={n} > {nlim}: too large for BFS — no canonicalisation"

    mins, sz, capped = _fewest_edge_members(adj, n, cap)
    best = None
    for a in mins:
        cert, _, _, iso_capped = canonical_labelling(a, n)
        if iso_capped:
            return adj, sz, capped, False, "Isomorphism budget hit — graph left unchanged"
        if best is None or cert < best:
            best = cert
    # Decode the certificate: upper-triangle bits in canonical labels.
    rep = [set() for _ in range(n)]
    bits = iter(best)
    for i in range(n):
        for j in range(i + 1, n):
            if next(bits):
                rep[i].add(j)
                rep[j].add(i)
    msg = f"Representative found (orbit {'≥' if capped else '='} {sz} graphs"
    if capped:
        msg += "; BFS capped, not guaranteed canonical"
    msg += ")"
    return rep, sz, capped, not capped, msg


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
