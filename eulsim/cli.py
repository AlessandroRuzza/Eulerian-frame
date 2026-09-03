"""Command-line entry point for the interactive web demo server."""
from __future__ import annotations

import argparse
import os
import socket
from http.server import ThreadingHTTPServer

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    raise SystemExit("Missing: numpy.  Run: pip install numpy")

from . import UI_VERSION
from .page import build_page
from .server import Handler

# The hosted deployment: what the Share-QR code points at unless the launcher
# (or --share local) asks for this machine's own address instead.
PUBLIC_URL = "https://eulerian-frame-simulator.onrender.com"


def detect_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EulSim - Eulerian Frame Simulator: interactive web demo.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--share", choices=("public", "local"), default=None,
                   help="Address the Share-QR code encodes: the public deployment "
                        f"({PUBLIC_URL}) or this server's own address. "
                        "Overrides the launcher default.")
    return p.parse_args()


def share_url(share: str, host: str, port: int, lan_url: str) -> str:
    """Address the Share-QR popup encodes. A loopback-only server is unreachable
    from other devices, so it advertises the loopback address instead of the LAN one."""
    if share == "public":
        return PUBLIC_URL
    if host in ("127.0.0.1", "localhost", "::1"):
        return f"http://127.0.0.1:{port}/"
    return lan_url


def main(default_share: str = "public") -> int:
    args = parse_args()
    if not (1 <= args.port <= 65535):
        raise SystemExit("Port must be in [1, 65535]")

    local_ip = detect_local_ip()
    lan_url = f"http://{local_ip}:{args.port}/"
    share = args.share or default_share
    qr_url = share_url(share, args.host, args.port, lan_url)
    Handler.page_html = build_page(UI_VERSION, qr_url)

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print("\nEulSim server ready.")
    print(f"Local:  http://127.0.0.1:{args.port}/")
    print(f"LAN:    {lan_url}")
    print(f"QR:     {qr_url}  (--share {'local' if share == 'public' else 'public'} to switch)")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
