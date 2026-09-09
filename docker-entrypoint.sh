#!/bin/sh
# EulSim container entrypoint.
#
# Env config (all optional):
#   HOST           bind address                     (default 0.0.0.0)
#   PORT           bind port; PaaS platforms such   (default 8001)
#                  as Render/Cloud Run inject this
#   EULSIM_SHARE   "local" | "public" — what the    (default local)
#                  Share-QR code encodes
#
# Any arguments passed to `docker run` are appended verbatim and therefore win
# over the env-derived flags, e.g.:  docker run ... --share public
set -eu

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
EULSIM_SHARE="${EULSIM_SHARE:-local}"

# A bare `python -m eulsim` as PID 1 has no SIGTERM handler, so the kernel drops
# the signal and `docker stop` has to wait for its timeout before SIGKILL. The
# shim installs one, turning SIGTERM into the same clean shutdown as Ctrl+C.
exec python -c '
import signal, sys
from eulsim.cli import main
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
sys.exit(main())
' --host "$HOST" --port "$PORT" --share "$EULSIM_SHARE" "$@"
