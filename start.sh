#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  TradeBoard — start script
#  Usage: ./start.sh
# ─────────────────────────────────────────────────────────────

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="/Users/jaswanth/.venv_picker/bin"
PORT=8000

# check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "  ERROR: ANTHROPIC_API_KEY is not set."
  echo "  Run:   export ANTHROPIC_API_KEY=sk-ant-..."
  echo ""
  exit 1
fi

# kill any previous instance on the port
lsof -ti tcp:$PORT | xargs kill -9 2>/dev/null || true
sleep 0.5

echo ""
echo "  ┌───────────────────────────────────────┐"
echo "  │  TradeBoard                           │"
echo "  │  Dashboard  → http://localhost:$PORT    │"
echo "  │  Stop       → Ctrl + C               │"
echo "  └───────────────────────────────────────┘"
echo ""

# open browser after 2 s
(sleep 2 && open "http://localhost:$PORT") &

# start server
cd "$DIR"
"$VENV/uvicorn" server:app --port $PORT --reload
