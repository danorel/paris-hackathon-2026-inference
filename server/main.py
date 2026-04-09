"""
OpenAI-compatible chat completions server.

Endpoints:
  GET  /health
  POST /v1/chat/completions
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import ATTN_IMPLEMENTATION, ENGINE_MODE, MODEL_NAME

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

if ENGINE_MODE == 2:
    from .engine_continious import InferenceEngine
else:
    # Iter 0 (eager) or Iter 1 (SDPA) — static batching
    import os
    if ENGINE_MODE == 0:
        os.environ.setdefault("ATTN_IMPLEMENTATION", "eager")
    from .engine_static import StaticBatchingEngine as InferenceEngine  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

engine_continious: InferenceEngine = None  # initialised in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_continious
    engine_continious = InferenceEngine()
    await engine_continious.start()
    logger.info("Server ready")
    yield


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MessageOut(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: MessageOut
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    result = await engine_continious.generate(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[
            Choice(
                message=MessageOut(content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )
