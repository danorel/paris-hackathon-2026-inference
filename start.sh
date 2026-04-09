#!/usr/bin/env bash
# Launch the inference server.
# Usage:  ./start.sh [port]
# The server starts on PORT (default 8000) and stays running.

set -euo pipefail

PORT="${1:-${SERVER_PORT:-8000}}"
HOST="${SERVER_HOST:-0.0.0.0}"

# Optional: point to a local model directory to avoid downloading
# export MODEL_PATH="/path/to/Qwen3.5-35B-A3B"

export SERVER_HOST="$HOST"
export SERVER_PORT="$PORT"

echo "[start.sh] Starting inference server on ${HOST}:${PORT}"

exec uvicorn server.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --loop uvloop \
    --log-level info
