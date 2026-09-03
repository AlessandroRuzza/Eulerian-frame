#!/usr/bin/env bash
# Syntax check of the whole simulator. Run from this directory.
set -e
python3 -m compileall -q eulsim run_eulsim.py tests benchmarks && echo 'OK'
