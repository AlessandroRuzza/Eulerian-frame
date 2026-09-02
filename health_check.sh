#!/usr/bin/env bash
# Usage: ./health_check.sh [port]
PORT="${1:-8001}"
URL="http://127.0.0.1:${PORT}/health"

response=$(curl -sf --max-time 3 "$URL" 2>/dev/null)
if [ $? -ne 0 ]; then
  echo "DOWN  $URL"
  exit 1
fi

ok=$(echo "$response" | grep -o '"ok": *true')
if [ -n "$ok" ]; then
  echo "UP    $URL"
  exit 0
else
  echo "UNEXPECTED  $URL  →  $response"
  exit 2
fi
