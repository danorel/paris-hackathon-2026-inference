#!/usr/bin/env bash
# Data-parallel inference: one full-model worker per GPU + round-robin proxy.
#
# Usage:
#   ./start_multi.sh [proxy-port]
#
# GPU selection (space-separated, 0-indexed CUDA device IDs):
#   WORKER_GPUS="4 5 6 7" ./start_multi.sh 8000
#
# Each worker loads the full model (~70 GB BF16) on its GPU.
# The proxy on PROXY_PORT distributes requests round-robin across all workers.
#
# Environment forwarded to every worker:
#   MODEL_PATH        — path to model weights (required)
#   MAX_BATCH_SIZE    — max active sequences per worker (default 64)
#   BATCH_TIMEOUT     — seconds to wait for a batch to fill (default 0.02)
#   USE_BATCHED_DECODE — 1 = batched cross-sequence decode (default 1)
#   USE_COMPILE       — 1 = torch.compile (default 0)

set -euo pipefail

PROXY_PORT="${1:-${SERVER_PORT:-9004}}"
PROXY_HOST="${SERVER_HOST:-0.0.0.0}"

# GPUs to use (space-separated 0-indexed CUDA IDs).
# Default: GPUs 4,5,6,7 — GPU 0 is occupied by shared vLLM on this node.
IFS=' ' read -r -a WORKER_GPUS <<< "${WORKER_GPUS:-4 5 6 7}"

BASE_WORKER_PORT=9010   # workers listen on 9010, 9011, 9012, 9013

# ---------------------------------------------------------------------------
# Kill any stale processes on the ports we're about to use
# ---------------------------------------------------------------------------
echo "[start_multi.sh] Clearing stale processes on proxy/worker ports..."
for _port in "$PROXY_PORT" 9010 9011 9012 9013; do
    _pid=$(lsof -ti tcp:"$_port" 2>/dev/null) && kill "$_pid" 2>/dev/null && echo "[start_multi.sh] killed stale PID $_pid on port $_port" || true
done
sleep 1

# Forwarded env vars (set defaults if not provided)
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
export BATCH_TIMEOUT="${BATCH_TIMEOUT:-0.02}"
export USE_BATCHED_DECODE="${USE_BATCHED_DECODE:-1}"
export USE_COMPILE="${USE_COMPILE:-0}"

WORKER_PIDS=()
PROXY_PID=""

cleanup() {
    echo ""
    echo "[start_multi.sh] Shutting down workers and proxy..."
    for pid in "${WORKER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    if [[ -n "$PROXY_PID" ]]; then
        kill "$PROXY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Launch one worker per GPU
# ---------------------------------------------------------------------------

BACKENDS=()
mkdir -p results logs

for i in "${!WORKER_GPUS[@]}"; do
    GPU="${WORKER_GPUS[$i]}"
    PORT=$((BASE_WORKER_PORT + i))
    LOG="logs/worker_gpu${GPU}.log"
    BACKENDS+=("http://localhost:${PORT}")

    echo "[start_multi.sh] GPU ${GPU} → worker port ${PORT} (log: ${LOG})"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    SERVER_HOST="127.0.0.1" \
    SERVER_PORT="${PORT}" \
        uvicorn server.main:app \
            --host 127.0.0.1 \
            --port "${PORT}" \
            --loop uvloop \
            --log-level info \
        > "${LOG}" 2>&1 &

    WORKER_PIDS+=($!)
done

echo "[start_multi.sh] ${#WORKER_GPUS[@]} workers started (PIDs: ${WORKER_PIDS[*]})"
echo "[start_multi.sh] Waiting for all workers to pass /health ..."

# ---------------------------------------------------------------------------
# Wait for every worker to be healthy before starting proxy
# ---------------------------------------------------------------------------

for backend in "${BACKENDS[@]}"; do
    until curl -sf "${backend}/health" > /dev/null 2>&1; do
        # If any worker died, exit early
        for pid in "${WORKER_PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[start_multi.sh] ERROR: worker PID ${pid} died during startup."
                echo "[start_multi.sh] Check logs in logs/ for details."
                exit 1
            fi
        done
        sleep 3
    done
    echo "[start_multi.sh] ${backend} ✓"
done

# ---------------------------------------------------------------------------
# Start the round-robin proxy (foreground so the script stays alive)
# ---------------------------------------------------------------------------

echo "[start_multi.sh] All workers ready. Starting proxy on ${PROXY_HOST}:${PROXY_PORT}"
echo "[start_multi.sh] Backends: ${BACKENDS[*]}"

PROXY_BACKENDS="$(IFS=,; echo "${BACKENDS[*]}")" \
    python -m server.proxy \
        --host "${PROXY_HOST}" \
        --port "${PROXY_PORT}" \
        --backends "${BACKENDS[@]}" \
        --no-wait &

PROXY_PID=$!
echo "[start_multi.sh] Proxy PID ${PROXY_PID}"

# Wait until proxy /health responds, then print ready banner
until curl -sf "http://localhost:${PROXY_PORT}/health" > /dev/null 2>&1; do
    sleep 1
done

echo ""
echo "================================================================"
echo " Inference cluster READY"
echo " Proxy:   http://${PROXY_HOST}:${PROXY_PORT}"
echo " Workers: ${BACKENDS[*]}"
echo "================================================================"
echo ""

# Keep script alive — exit when proxy dies
wait "$PROXY_PID"
