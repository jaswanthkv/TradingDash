#!/bin/bash
# TradeBoard Launcher — double-click this file to start the app
# ──────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

LOGFILE="/tmp/tradeboard.log"
PORT=8000

echo "🚀 Starting TradeBoard..."

# ── find uvicorn ──────────────────────────────────────────────
UVICORN=""
for candidate in \
    "$HOME/.venv_picker/bin/uvicorn" \
    "$HOME/.local/bin/uvicorn" \
    "$HOME/.venv/bin/uvicorn" \
    "$(python3 -m site --user-base 2>/dev/null)/bin/uvicorn" \
    "/usr/local/bin/uvicorn" \
    "/opt/homebrew/bin/uvicorn"; do
    if [ -x "$candidate" ]; then
        UVICORN="$candidate"
        break
    fi
done

if [ -z "$UVICORN" ]; then
    # Try python3 -m uvicorn
    if python3 -c "import uvicorn" 2>/dev/null; then
        UVICORN="python3 -m uvicorn"
    else
        osascript -e 'display alert "TradeBoard: uvicorn not found" message "Open Terminal and run:\n\npip install uvicorn fastapi kiteconnect anthropic yfinance pandas\n\nThen try again." buttons {"OK"} default button "OK"' 2>/dev/null || true
        echo "❌ uvicorn not found. Install with: pip install uvicorn fastapi"
        read -n1 -r -p "Press any key to exit..."
        exit 1
    fi
fi

# ── load .env ─────────────────────────────────────────────────
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✅ Loaded .env"
fi

# ── kill any old server on port ───────────────────────────────
lsof -ti:$PORT | xargs kill -9 2>/dev/null && echo "♻️  Cleared port $PORT" || true
sleep 1

# ── start server ──────────────────────────────────────────────
echo "⚙️  Starting server on port $PORT..."
$UVICORN server:app --host 0.0.0.0 --port $PORT > "$LOGFILE" 2>&1 &
SERVER_PID=$!

# ── wait for server to be ready ───────────────────────────────
for i in $(seq 1 15); do
    sleep 1
    if curl -sf "http://localhost:$PORT/api/health" > /dev/null 2>&1; then
        echo "✅ Server ready!"
        break
    fi
    echo "   Waiting... ($i/15)"
done

# ── open browser ──────────────────────────────────────────────
open "http://localhost:$PORT"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TradeBoard running → http://localhost:$PORT"
echo "  Log → $LOGFILE"
echo "  Close this window to stop the server."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── keep alive until window closes ───────────────────────────
trap "kill $SERVER_PID 2>/dev/null; echo 'Server stopped.'" EXIT
wait $SERVER_PID
