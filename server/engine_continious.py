"""
Inference engine: Flash Attention 2 + continuous batching.

Key ideas vs the naive HF generate() loop:
  1. SDPA (Scaled Dot Product Attention) — PyTorch built-in, uses Flash-Attention
     kernels via cuDNN on H200 without requiring flash-attn package.
  2. Continuous batching — token-by-token decode with iteration-level scheduling
     (Orca paper, 2022).  Finished sequences are removed immediately; new
     requests are prefilled and injected between decode steps.  The GPU is
     never idle waiting for the slowest sequence in a static batch.
  3. Thinking is disabled via enable_thinking=False in the chat template so
     the model never emits <think> blocks (no inference overhead, no stripping).
"""

import asyncio
import logging
import queue as thread_queue
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MAX_BATCH_SIZE, MODEL_PATH, USE_TP

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Public request / result types
# ---------------------------------------------------------------------------

@dataclass
class GenerateRequest:
    messages: list[dict]
    max_tokens: int
    temperature: float
    top_p: float
    future: asyncio.Future = field(repr=False)


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length"


# ---------------------------------------------------------------------------
# Internal: one active decode sequence
# ---------------------------------------------------------------------------

@dataclass
class _ActiveSeq:
    req: GenerateRequest
    prompt_len: int
    generated: list[int]        # tokens produced so far (excluding first sampled in prefill)
    last_token: torch.Tensor    # shape (1, 1) — input for next decode step
    past_key_values: Any        # HF KV cache


class InferenceEngine:
    def __init__(self) -> None:
        logger.info("Loading tokenizer from %s", MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        n_gpus = torch.cuda.device_count()
        if USE_TP or n_gpus > 1:
            logger.info("Loading model from %s (BF16, SDPA, device_map=auto, %d GPUs)", MODEL_PATH, n_gpus)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map="auto",
            )
        else:
            logger.info("Loading model from %s (BF16, SDPA, single GPU)", MODEL_PATH)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to("cuda")
        self.model.eval()
        # First device — where input tensors must be placed
        self._device = next(self.model.parameters()).device
        logger.info(
            "Model loaded — %d parameters",
            sum(p.numel() for p in self.model.parameters()),
        )

        # Thread-safe queue: async side puts requests, decode thread gets them
        self._request_queue: thread_queue.Queue[GenerateRequest] = thread_queue.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._thread = threading.Thread(
            target=self._continuous_decode_loop, daemon=True, name="decode"
        )

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerateResult:
        future = self._loop.create_future()
        self._request_queue.put(
            GenerateRequest(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                future=future,
            )
        )
        return await future

    # ------------------------------------------------------------------
    # Decode thread: continuous batching loop (Orca-style)
    # ------------------------------------------------------------------

    def _continuous_decode_loop(self) -> None:
        """Iteration-level scheduling.

        Each iteration:
          1. Drain the request queue → prefill new sequences, add to active set.
          2. Run ONE decode step per active sequence.
          3. Remove completed sequences and resolve their futures immediately.

        Short requests never block long ones.  The GPU never sits idle between
        static batches.
        """
        active: list[_ActiveSeq] = []

        while True:
            # 1a. If nothing is running, block until at least one request arrives
            if not active:
                req = self._request_queue.get()
                new_reqs = [req]
            else:
                new_reqs = []

            # 1b. Drain additional requests without blocking
            while len(active) + len(new_reqs) < MAX_BATCH_SIZE:
                try:
                    new_reqs.append(self._request_queue.get_nowait())
                except thread_queue.Empty:
                    break

            # Prefill each new request → get KV cache + first token
            for req in new_reqs:
                try:
                    seq = self._prefill(req)
                    active.append(seq)
                except Exception as exc:
                    logger.exception("Prefill failed")
                    self._resolve(req.future, exc)

            if not active:
                continue

            # 2. One decode step per active sequence
            still_active: list[_ActiveSeq] = []
            for seq in active:
                try:
                    done, result = self._decode_step(seq)
                except Exception as exc:
                    logger.exception("Decode step failed")
                    self._resolve(seq.req.future, exc)
                    continue

                if done:
                    self._resolve(seq.req.future, result)
                else:
                    still_active.append(seq)

            active = still_active

    # ------------------------------------------------------------------
    # Prefill: process prompt, return KV cache + first sampled token
    # ------------------------------------------------------------------

    def _prefill(self, req: GenerateRequest) -> _ActiveSeq:
        prompt = self._apply_chat_template(req.messages)
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        first_token = self._sample(out.logits[:, -1, :], req)  # (1, 1)
        return _ActiveSeq(
            req=req,
            prompt_len=input_ids.shape[1],
            generated=[first_token.item()],
            last_token=first_token,
            past_key_values=out.past_key_values,
        )

    # ------------------------------------------------------------------
    # One decode step for a single sequence
    # ------------------------------------------------------------------

    def _decode_step(self, seq: _ActiveSeq) -> tuple[bool, Optional[GenerateResult]]:
        """Returns (done, result_or_None)."""
        req = seq.req
        # position of the token we're about to generate
        pos = seq.prompt_len + len(seq.generated) - 1

        with torch.inference_mode():
            out = self.model(
                input_ids=seq.last_token,               # (1, 1)
                past_key_values=seq.past_key_values,
                use_cache=True,
                position_ids=torch.tensor([[pos]], device=self._device),
            )

        next_token = self._sample(out.logits[:, -1, :], req)
        next_id = next_token.item()

        seq.past_key_values = out.past_key_values
        seq.last_token = next_token

        eos_id = self.tokenizer.eos_token_id
        hit_eos = eos_id is not None and next_id == eos_id
        hit_max = len(seq.generated) >= req.max_tokens

        if hit_eos or hit_max:
            if not hit_eos:
                seq.generated.append(next_id)
            finish_reason = "stop" if hit_eos else "length"
            text = self.tokenizer.decode(seq.generated, skip_special_tokens=True)
            text = _THINK_RE.sub("", text).strip()
            return True, GenerateResult(
                text=text,
                prompt_tokens=seq.prompt_len,
                completion_tokens=len(seq.generated),
                finish_reason=finish_reason,
            )

        seq.generated.append(next_id)
        return False, None

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, req: GenerateRequest) -> torch.Tensor:
        """Greedy or temperature + top-p sample. Returns shape (1, 1)."""
        if req.temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / req.temperature
        if req.top_p < 1.0:
            logits = _top_p_filter(logits, req.top_p)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Apply chat template with thinking fully disabled."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # no <think> tokens generated at all
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _resolve(self, future: asyncio.Future, value: Any) -> None:
        if isinstance(value, Exception):
            self._loop.call_soon_threadsafe(future.set_exception, value)
        else:
            self._loop.call_soon_threadsafe(future.set_result, value)


# ---------------------------------------------------------------------------
# Module-level helper (not a method — avoids 'self' in hot path)
# ---------------------------------------------------------------------------

def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_logits[cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
    return logits.scatter(1, sorted_idx, sorted_logits)
