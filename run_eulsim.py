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

if __name__ == "__main__":
    raise SystemExit(main())
