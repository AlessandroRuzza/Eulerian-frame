#!/usr/bin/env python3
"""Container healthcheck: GET /health and require {"ok": true}.

Stdlib only — the runtime image ships no curl or wget.
"""
import json
import os
import sys
import urllib.request

url = "http://127.0.0.1:%s/health" % os.environ.get("PORT", "8001")

try:
    with urllib.request.urlopen(url, timeout=4) as r:
        ok = json.load(r).get("ok") is True
except Exception as exc:  # unreachable, timeout, or non-JSON body
    print("unhealthy: %s: %s" % (url, exc), file=sys.stderr)
    sys.exit(1)

sys.exit(0 if ok else 1)
