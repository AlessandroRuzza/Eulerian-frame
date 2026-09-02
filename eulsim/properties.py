"""Graph-theoretic properties and human-readable tags for the UI.

Reads the graph as adjacency sets: degrees, edge count and connectivity all
cost O(n + m) instead of O(n^2).
"""
from __future__ import annotations

def compute_properties(adj: list[set[int]], n: int, labels: list[str]) -> dict:
    if n == 0:
        return {"n": 0, "m": 0, "connected": True, "tags": ["Empty graph"], "degrees": []}
    degs = [len(adj[i]) for i in range(n)]
    m = sum(degs) // 2

    vis = [False] * n
    q = [0]; vis[0] = True
    while q:
        v = q.pop()
        for u in adj[v]:
            if not vis[u]:
                vis[u] = True; q.append(u)
    connected = all(vis)

    sdegs = sorted(degs)
    is_complete = m == n * (n - 1) // 2
    is_star = n >= 3 and sdegs == [1] * (n - 1) + [n - 1]
    is_bell = n == 2 and m == 1
    is_path = (connected and m == n - 1 and n >= 2
               and sdegs[:2] == [1, 1] and all(d <= 2 for d in degs))
    is_cycle = connected and all(d == 2 for d in degs) and n >= 3

    tags: list[str] = []
    if n == 1:        tags.append("Single qubit |+⟩")
    elif m == 0:      tags.append("Product state |+⟩⊗ⁿ — no entanglement")
    elif is_bell:     tags.append("Bell pair (2-qubit graph state)")
    elif is_complete: tags.append("Complete graph Kₙ — LC-equivalent to GHZ")
    elif is_star:     tags.append("Star graph — LC-equivalent to GHZ")
    elif is_path:     tags.append("Path / linear cluster — 1D MBQC resource")
    elif is_cycle:    tags.append("Cycle / ring cluster")
    else:             tags.append("General graph state")
    if not connected and n > 1:
        tags.append("Disconnected — separable across components")

    return {"n": n, "m": m, "connected": connected,
            "degrees": degs, "max_degree": max(degs) if degs else 0,
            "tags": tags}
