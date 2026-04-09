#!/usr/bin/env bash
# Launch the inference server.
#
# Usage:
#   ./start.sh [port]
#
# GPU selection (GPU 0 is occupied by shared vLLM — skip it):
#   1 GPU:  CUDA_VISIBLE_DEVICES=1 ./start.sh
#   4 GPU:  CUDA_VISIBLE_DEVICES=1,2,3,4 ./start.sh
#   6 GPU:  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./start.sh   (default)
#
# When multiple GPUs are visible the engine uses device_map="auto" (pipeline parallel).
# For single GPU it loads directly to cuda:0.

set -euo pipefail

PORT="${1:-${SERVER_PORT:-8000}}"
HOST="${SERVER_HOST:-0.0.0.0}"

export SERVER_HOST="$HOST"
export SERVER_PORT="$PORT"

mkdir -p results

# Default: GPU 4 (dedicated for single-GPU experiments).
# GPU 0 occupied by shared vLLM. Override via env before calling this script.
# Multi-GPU window will be used for final benchmarking.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

echo "[start.sh] Starting inference server on ${HOST}:${PORT}"

exec uvicorn server.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --loop uvloop \
    --log-level info
