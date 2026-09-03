"""HTTP request handler: serves the demo page and the JSON compute API.

Graphs travel on the wire as edge lists ("edges": [[i, j], ...]) and live in
the compute layer as adjacency sets, so a request costs O(n + m) to parse and
a response O(n + m) to serialise.

Frames travel as "local_unitaries" — one 8-float 2x2 matrix per qubit, which
is what the page's own Clifford arithmetic speaks — and are converted to and
from Eulerian codes here, at the boundary.  Nothing below this module sees a
matrix.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

from .canonical import canonicalize
from .frames import frame_from_wire, frame_to_wire, name
from .gates import apply_controlled
from .graph_ops import adj_from_edges, do_measure, edge_list, reframe_move
from .lc_orbit import (
    MAX_BFS_STATES,
    NODE_LIMIT,
    lc_canonical,
    lc_equiv_labeled,
    lc_orbit_size,
)
from .properties import compute_properties
from .statevector import compute_state_vector
from .tableau import compute_stabilizers


# ─── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    page_html: str = ""
    server_version = "GraphStateDemo/1.1"
    protocol_version = "HTTP/1.1"

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8", status)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(self.page_html.encode(), "text/html; charset=utf-8")
        elif path == "/health":
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def _read_body(self) -> dict | None:
        cl = self.headers.get("Content-Length", "")
        try:
            n = int(cl)
        except ValueError:
            return None
        if n <= 0 or n > 500_000:
            return None
        raw = self.rfile.read(n)
        try:
            return json.loads(raw)
        except Exception:
            return None

    # Upper bound on graph size accepted by the API.  Everything the server
    # computes for a graph is polynomial (stabilizers O(n^2), canonicalisation
    # O(n^3)); the exponential parts have their own, much lower caps
    # (MAX_SV_QUBITS for the state vector, NODE_LIMIT for the LC orbit).
    MAX_QUBITS = 64

    def _parse_graph(self, p: dict) -> tuple[list[set[int]], int, list[str]]:
        """Graph as adjacency sets from the wire edge list."""
        n = int(p.get("n", 0))
        if not (0 <= n <= self.MAX_QUBITS):
            raise ValueError(f"n out of range [0,{self.MAX_QUBITS}]")
        raw_edges = p.get("edges") or []
        if len(raw_edges) > n * (n - 1) // 2:
            raise ValueError("too many edges")
        for e in raw_edges:
            if len(e) != 2:
                raise ValueError("edge not a pair")
            i, j = int(e[0]), int(e[1])
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError("edge endpoint out of range")
            if i == j:
                raise ValueError("self-loop in edge list")
        adj = adj_from_edges(n, raw_edges)
        raw_labels = p.get("labels", [])
        labels = [str(raw_labels[i]) if i < len(raw_labels) else str(i) for i in range(n)]
        return adj, n, labels

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_body()
        if body is None:
            self._json({"error": "bad request"}, 400); return
        try:
            if path == "/api/compute":
                adj, n, labels = self._parse_graph(body)
                f = frame_from_wire(n, body.get("local_unitaries"))
                display_basis = body.get("display_basis", None)
                self._json({
                    "stabilizers": compute_stabilizers(adj, n, labels, f),
                    "state_vector": compute_state_vector(adj, n, f, display_basis),
                    "properties": compute_properties(adj, n, labels),
                })
            elif path == "/api/reframe":
                # Re-framing move R_v: (G, L) ↦ (τ_v(G), L·U_v†) — same state.
                # NOTE: the bundled frontend computes R_v client-side and never
                # calls this; it is kept as a JSON-API surface. If the page is
                # ever moved onto it, delete the JS copy in web/index.html.
                adj, n, labels = self._parse_graph(body)
                v = int(body.get("vertex", -1))
                if not (0 <= v < n):
                    raise ValueError("vertex out of range")
                display_basis = body.get("display_basis", None)
                new_adj, toggled, new_f = reframe_move(
                    adj, n, v, frame_from_wire(n, body.get("local_unitaries")))
                self._json({
                    "new_edges": edge_list(new_adj, n),
                    "toggled_edges": toggled,
                    "new_local_unitaries": frame_to_wire(new_f),
                    "stabilizers": compute_stabilizers(new_adj, n, labels, new_f),
                    "state_vector": compute_state_vector(new_adj, n, new_f, display_basis),
                    "properties": compute_properties(new_adj, n, labels),
                })
            elif path == "/api/canonicalize":
                adj, n, labels = self._parse_graph(body)
                display_basis = body.get("display_basis", None)
                new_adj, new_f, corrections, canon_info = canonicalize(
                    adj, n, frame_from_wire(n, body.get("local_unitaries")))
                # Single-qubit Cliffords C_i that would collapse the residual frame
                # to the pure graph state |G'⟩ (changing the physical state).
                residual = [{"qubit": labels[i], "gate": name(corrections[i])}
                            for i in range(n) if name(corrections[i]) != "I"]
                self._json({
                    "new_edges": edge_list(new_adj, n),
                    "new_local_unitaries": frame_to_wire(new_f),
                    "residual_corrections": residual,
                    "canon_info": canon_info,
                    "stabilizers": compute_stabilizers(new_adj, n, labels, new_f),
                    "state_vector": compute_state_vector(new_adj, n, new_f, display_basis),
                    "properties": compute_properties(new_adj, n, labels),
                })
            elif path == "/api/cz":
                adj, n, labels = self._parse_graph(body)
                i = int(body.get("i", -1)); j = int(body.get("j", -1))
                gate = str(body.get("gate", "cz")).lower()
                display_basis = body.get("display_basis", None)
                if not (0 <= i < n and 0 <= j < n and i != j):
                    raise ValueError("invalid CZ qubits")
                if gate not in ("cz", "cx", "cy"):
                    raise ValueError("gate must be cz/cx/cy")
                new_adj, new_f = apply_controlled(
                    adj, n, i, j, gate, frame_from_wire(n, body.get("local_unitaries")))
                self._json({
                    "new_edges": edge_list(new_adj, n),
                    "new_local_unitaries": frame_to_wire(new_f),
                    "stabilizers": compute_stabilizers(new_adj, n, labels, new_f),
                    "state_vector": compute_state_vector(new_adj, n, new_f, display_basis),
                    "properties": compute_properties(new_adj, n, labels),
                })
            elif path == "/api/measure":
                adj, n, labels = self._parse_graph(body)
                v = int(body.get("vertex", -1))
                basis = str(body.get("basis", "z")).lower()
                if not (0 <= v < n):
                    raise ValueError("vertex out of range")
                if basis not in ("x", "y", "z"):
                    raise ValueError("basis must be x/y/z")
                delete_after = bool(body.get("delete", True))
                invert = bool(body.get("invert", False))
                new_adj, kept, steps, new_f = do_measure(
                    adj, n, v, basis, frame_from_wire(n, body.get("local_unitaries")),
                    delete=delete_after, invert=invert)
                new_n = len(kept)
                new_labels = [labels[i] for i in kept]
                display_basis = body.get("display_basis", None)
                new_db = [display_basis[i] for i in kept] if display_basis else None
                self._json({
                    "new_edges": edge_list(new_adj, new_n), "new_n": new_n,
                    "kept_indices": kept, "new_labels": new_labels,
                    "steps": steps,
                    "new_local_unitaries": frame_to_wire(new_f),
                    "stabilizers": compute_stabilizers(new_adj, new_n, new_labels, new_f),
                    "state_vector": compute_state_vector(new_adj, new_n, new_f, new_db),
                    "properties": compute_properties(new_adj, new_n, new_labels),
                })
            elif path == "/api/lc_canonical":
                g = body.get("graph", {})
                adj, n, labels = self._parse_graph(g)
                req_max_bfs    = int(body.get("max_bfs_states", MAX_BFS_STATES))
                req_node_limit = int(body.get("node_limit",    NODE_LIMIT))
                rep, sz, capped, msg = lc_canonical(
                    adj, n, max_bfs=req_max_bfs, node_limit=req_node_limit)
                edges = edge_list(rep, n)
                self._json({"n": n, "labels": labels,
                            "edges": edges, "orbit_size": sz,
                            "capped": capped, "msg": msg})
            elif path == "/api/lc_equiv":
                ga = body.get("graph_a", {})
                gb = body.get("graph_b", {})
                adj_a, n_a, labels_a = self._parse_graph(ga)
                adj_b, n_b, labels_b = self._parse_graph(gb)
                req_max_bfs   = int(body.get("max_bfs_states", MAX_BFS_STATES))
                req_node_limit = int(body.get("node_limit",    NODE_LIMIT))
                if n_a != n_b:
                    self._json({"result": False, "orbit_size_a": None,
                                "msg": f"Different qubit counts: {n_a} vs {n_b}"})
                    return
                result, orbit, msg = lc_equiv_labeled(
                    adj_a, n_a, adj_b,
                    max_bfs=req_max_bfs, node_limit=req_node_limit)
                orbit_a = lc_orbit_size(adj_a, n_a, max_bfs=req_max_bfs)
                orbit_b = lc_orbit_size(adj_b, n_b, max_bfs=req_max_bfs)
                self._json({"result": result, "msg": msg,
                            "orbit_size_a": orbit_a, "orbit_size_b": orbit_b,
                            "n": n_a})
            elif path == "/api/state_equal":
                # Equality of the two *physical* states (frames included): two
                # framed states describe the same state iff their canonical
                # frames coincide (thm:gcf; see canonical.canonicalize).
                ga = body.get("graph_a", {})
                gb = body.get("graph_b", {})
                adj_a, n_a, _ = self._parse_graph(ga)
                adj_b, n_b, _ = self._parse_graph(gb)
                if n_a != n_b:
                    self._json({"equal": False,
                                "msg": f"Different qubit counts: {n_a} vs {n_b}"})
                    return
                canon_a, _, _, info_a = canonicalize(
                    adj_a, n_a, frame_from_wire(n_a, ga.get("local_unitaries")))
                canon_b, _, _, info_b = canonicalize(
                    adj_b, n_b, frame_from_wire(n_b, gb.get("local_unitaries")))
                same_graph = canon_a == canon_b
                same_frame = all(info_a[k] == info_b[k] for k in ("f", "d", "s"))
                if same_graph and same_frame:
                    msg = f"identical canonical frames (n = {n_a})"
                elif same_graph:
                    msg = "same canonical graph, different canonical frame"
                else:
                    msg = "different canonical graphs"
                self._json({
                    "equal": same_graph and same_frame,
                    "same_graph": same_graph, "same_frame": same_frame,
                    "msg": msg, "n": n_a,
                    "canon_a": {"edges": edge_list(canon_a, n_a), **info_a},
                    "canon_b": {"edges": edge_list(canon_b, n_b), **info_b},
                })
            else:
                self._json({"error": "not found"}, 404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:
            self._json({"error": f"server error: {e}"}, 500)

