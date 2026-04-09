"""
Static-batching engine (model.generate()) — used for Iter 0 and Iter 1 baselines.

ENGINE_MODE=0  →  eager attention  (vanilla baseline)
ENGINE_MODE=1  →  SDPA attention   (Flash-Attention via cuDNN)

Requests are collected for up to BATCH_TIMEOUT seconds, then dispatched together
as a padded batch to model.generate().  Each request gets its own max_new_tokens
respected via per-sequence stopping (we take max of the batch and truncate outputs
to each request's actual budget).
"""

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ATTN_IMPLEMENTATION, BATCH_TIMEOUT, MAX_BATCH_SIZE, MODEL_PATH

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


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


class StaticBatchingEngine:
    def __init__(self) -> None:
        logger.info("Loading tokenizer from %s", MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        attn = ATTN_IMPLEMENTATION  # "sdpa" for Iter 1, "eager" for Iter 0
        logger.info(
            "Loading model from %s (BF16, attn=%s, single GPU)", MODEL_PATH, attn
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
        ).to("cuda")
        self.model.eval()

        self._pending: list[GenerateRequest] = []
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(
            "Model loaded — %d parameters",
            sum(p.numel() for p in self.model.parameters()),
        )

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        asyncio.create_task(self._batch_loop())

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerateResult:
        future = self._loop.create_future()
        async with self._lock:
            self._pending.append(
                GenerateRequest(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    future=future,
                )
            )
            self._batch_event.set()
        return await future

    async def _batch_loop(self) -> None:
        while True:
            await self._batch_event.wait()
            self._batch_event.clear()
            await asyncio.sleep(BATCH_TIMEOUT)  # collect for timeout window

            async with self._lock:
                batch = self._pending[:MAX_BATCH_SIZE]
                self._pending = self._pending[MAX_BATCH_SIZE:]
                if self._pending:
                    self._batch_event.set()

            if not batch:
                continue

            try:
                results = await self._loop.run_in_executor(
                    None, self._generate_batch, batch
                )
                for req, result in zip(batch, results):
                    req.future.set_result(result)
            except Exception as exc:
                logger.exception("Batch generation failed")
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)

    def _generate_batch(self, batch: list[GenerateRequest]) -> list[GenerateResult]:
        prompts = [self._apply_chat_template(req.messages) for req in batch]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        max_new = max(req.max_tokens for req in batch)

        gen_kwargs: dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        first = batch[0]
        if first.temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = first.temperature
            gen_kwargs["top_p"] = first.top_p

        with torch.inference_mode():
            out_ids = self.model.generate(**gen_kwargs)

        results = []
        for i, req in enumerate(batch):
            prompt_len = int(prompt_lengths[i])
            new_ids = out_ids[i, prompt_len:]
            # Truncate to per-request max_tokens
            new_ids = new_ids[: req.max_tokens]

            eos_id = self.tokenizer.eos_token_id
            finish_reason = "stop"
            if eos_id is not None and eos_id in new_ids.tolist():
                eos_pos = new_ids.tolist().index(eos_id)
                new_ids = new_ids[:eos_pos]
            else:
                finish_reason = "length"

            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            text = _THINK_RE.sub("", text).strip()
            results.append(
                GenerateResult(
                    text=text,
                    prompt_tokens=prompt_len,
                    completion_tokens=len(new_ids),
                    finish_reason=finish_reason,
                )
            )
        return results

    def _apply_chat_template(self, messages: list[dict]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
