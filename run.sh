#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== DeepAgent Demo ==="
echo ""

# Install deps if needed
if ! python -c "import langgraph" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install --break-system-packages --ignore-installed -r requirements.txt
fi

export PYTHONPATH="$SCRIPT_DIR"

# Kill any existing servers on our ports
pkill -f "fastapi_server" 2>/dev/null || true
pkill -f "aegra_server" 2>/dev/null || true
sleep 1

echo "Starting FastAPI server (raw SSE) on port 8000..."
uvicorn servers.fastapi_server:app --host 0.0.0.0 --port 8000 --reload \
  > /tmp/fastapi_server.log 2>&1 &
FASTAPI_PID=$!

echo "Starting Aegra server (Platform API) on port 8001..."
uvicorn servers.aegra_server:app --host 0.0.0.0 --port 8001 --reload \
  > /tmp/aegra_server.log 2>&1 &
AEGRA_PID=$!

sleep 2

echo ""
echo "=== Servers running ==="
echo ""
echo "  GUI (FastAPI mode):   http://localhost:8000"
echo "  GUI (Aegra mode):     http://localhost:8001"
echo "  Presentation:         http://localhost:8000/presentation"
echo ""
echo "  FastAPI API docs:     http://localhost:8000/docs"
echo "  Aegra API docs:       http://localhost:8001/docs"
echo ""
echo "  Logs: tail -f /tmp/fastapi_server.log"
echo "        tail -f /tmp/aegra_server.log"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $FASTAPI_PID $AEGRA_PID 2>/dev/null; exit 0" INT TERM

wait
