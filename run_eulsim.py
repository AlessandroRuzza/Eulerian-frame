#!/usr/bin/env python3
"""
Standalone launcher for EulSim — the Eulerian Frame Simulator.

Run:
    python3 run_eulsim.py [--port 8001]

The implementation lives in the ``eulsim`` package in this directory
(see ``eulsim/__init__.py`` for the concept overview).
Equivalent invocation:  python3 -m eulsim [--port 8001]
"""
from eulsim.cli import main

#   "public" - the hosted deployment (eulsim.cli.PUBLIC_URL), reachable by anyone;
#   "local"  - this machine's LAN address, for devices on the same network only.
# --share on the command line overrides this for a single run.
SHARE_TARGET = "public"

if __name__ == "__main__":
    raise SystemExit(main(default_share=SHARE_TARGET))
