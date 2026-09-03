"""HTML page assembly: load web/index.html, substitute config placeholders."""
from __future__ import annotations

from pathlib import Path

from .lc_orbit import MAX_BFS_STATES, NODE_LIMIT

_WEB_DIR = Path(__file__).resolve().parent / "web"


def build_page(ui_version: str, share_url: str = "") -> str:
    """`share_url` is the address the Share-QR popup encodes; the page falls
    back to its own location when it is empty."""
    html = (_WEB_DIR / "index.html").read_text(encoding="utf-8")
    return (html
            .replace("__UI_VERSION__", ui_version)
            .replace("__SHARE_URL__", share_url)
            .replace("__MAX_BFS_STATES__", str(MAX_BFS_STATES))
            .replace("__MAX_BFS_STATES_JS__", str(MAX_BFS_STATES))
            .replace("__NODE_LIMIT__", str(NODE_LIMIT))
            .replace("__NODE_LIMIT_JS__", str(NODE_LIMIT)))
