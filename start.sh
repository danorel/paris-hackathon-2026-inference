#!/usr/bin/env bash
# Launch the inference server.
# Usage:  ./start.sh [port]
# The server starts on PORT (default 8000) and stays running.
#
# Key env vars:
#   MODEL_PATH         — local path or HF model ID (default: Qwen/Qwen3.5-35B-A3B)
#   ATTN_IMPLEMENTATION — "flash_attention_2" | "sdpa" | "eager" (default: sdpa)
#   MAX_BATCH_SIZE     — max requests per batch (default: 64)
#   BATCH_TIMEOUT      — seconds to wait for batch to fill (default: 0.02)

set -euo pipefail

PORT="${1:-${SERVER_PORT:-8000}}"
HOST="${SERVER_HOST:-0.0.0.0}"

export SERVER_HOST="$HOST"
export SERVER_PORT="$PORT"

# Default to sdpa; override with "flash_attention_2" if flash-attn is installed
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

mkdir -p results

echo "[start.sh] Starting inference server on ${HOST}:${PORT} (attn=${ATTN_IMPLEMENTATION})"

exec uvicorn server.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --loop uvloop \
    --log-level info
