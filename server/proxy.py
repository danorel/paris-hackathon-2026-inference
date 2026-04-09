"""
Least-connections load-balancing proxy for data-parallel inference workers.

Each worker is a full model replica on its own GPU. The proxy routes each
request to whichever worker currently has the fewest in-flight requests.

This beats round-robin when requests have unequal latency: a worker that just
received many long requests won't get buried while others sit idle.

Usage:
    python -m server.proxy --port 8000 --backends http://localhost:8001 http://localhost:8002 ...

Or set via env:
    PROXY_BACKENDS="http://localhost:8001,http://localhost:8002" python -m server.proxy
"""

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State — populated before app starts
# ---------------------------------------------------------------------------

BACKENDS: list[str] = []
_active: list[int] = []          # in-flight request count per backend
_client: httpx.AsyncClient | None = None


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _active.extend([0] * len(BACKENDS))
    limits = httpx.Limits(max_connections=512, max_keepalive_connections=128)
    _client = httpx.AsyncClient(timeout=600.0, limits=limits)
    logger.info("Proxy ready — backends: %s", BACKENDS)
    yield
    await _client.aclose()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_backend(exclude: set[int] | None = None) -> tuple[int, str]:
    """Return (idx, url) of the least-loaded backend not in `exclude`.

    No await between read and write → safe in a single-threaded async event loop.
    Ties broken by lower index. Returns (-1, "") if all backends are excluded.
    """
    candidates = [i for i in range(len(BACKENDS)) if not exclude or i not in exclude]
    if not candidates:
        return -1, ""
    idx = min(candidates, key=lambda i: _active[i])
    _active[idx] += 1
    logger.debug("route → %s  (active: %s)", BACKENDS[idx], _active)
    return idx, BACKENDS[idx]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Check that at least one backend is alive."""
    for backend in BACKENDS:
        try:
            r = await _client.get(f"{backend}/health", timeout=5.0)
            if r.status_code == 200:
                return {"status": "ok"}
        except Exception:
            continue
    return JSONResponse({"status": "no backends available"}, status_code=503)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    body = await request.body()
    skip = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
    url_path = f"/{path}"
    if request.url.query:
        url_path = f"{url_path}?{request.url.query}"

    tried: set[int] = set()
    while len(tried) < len(BACKENDS):
        idx, backend = _pick_backend(exclude=tried)
        if idx == -1:
            break
        try:
            resp = await _client.request(
                method=request.method,
                url=f"{backend}{url_path}",
                headers=headers,
                content=body,
            )
            _active[idx] -= 1
            resp_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in {"transfer-encoding", "connection"}
            }
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            _active[idx] -= 1
            tried.add(idx)
            logger.warning("backend %s unreachable, retrying on next worker", backend)

    return JSONResponse({"error": "all backends unavailable"}, status_code=503)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _wait_for_backends(backends: list[str]) -> None:
    """Block until every backend /health returns 200."""
    import time
    import urllib.request

    for backend in backends:
        url = f"{backend}/health"
        logger.info("Waiting for %s ...", url)
        while True:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        logger.info("%s is ready", backend)
                        break
            except Exception:
                time.sleep(2)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Round-robin inference proxy")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PROXY_PORT", "8000")))
    parser.add_argument("--host", default=os.environ.get("PROXY_HOST", "0.0.0.0"))
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[b for b in os.environ.get("PROXY_BACKENDS", "").split(",") if b],
        help="Backend URLs e.g. http://localhost:8001 http://localhost:8002",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait for all backends to be healthy before starting proxy (default: true)",
    )
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    args = parser.parse_args()

    if not args.backends:
        parser.error("--backends is required (or set PROXY_BACKENDS env var)")

    BACKENDS.extend(args.backends)

    if args.wait:
        _wait_for_backends(BACKENDS)

    uvicorn.run(app, host=args.host, port=args.port, loop="uvloop", log_level="info")
