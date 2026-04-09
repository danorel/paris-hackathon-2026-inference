"""
Inference engine: model loading, dynamic batching, autoregressive decode.
"""

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BATCH_TIMEOUT, MAX_BATCH_SIZE, MODEL_PATH

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


class InferenceEngine:
    def __init__(self) -> None:
        logger.info("Loading tokenizer from %s", MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Loading model from %s (BF16, device_map=auto)", MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        logger.info("Model loaded — %d parameters", sum(p.numel() for p in self.model.parameters()))

        self._queue: asyncio.Queue[GenerateRequest] = asyncio.Queue()
        # Single worker thread: model is not thread-safe for concurrent forward passes
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        asyncio.create_task(self._batch_loop())

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
        await self._queue.put(
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
    # Internal: batching loop
    # ------------------------------------------------------------------

    async def _batch_loop(self) -> None:
        while True:
            # Wait for the first request
            batch: list[GenerateRequest] = [await self._queue.get()]

            # Drain queue for up to BATCH_TIMEOUT to fill the batch
            deadline = self._loop.time() + BATCH_TIMEOUT
            while len(batch) < MAX_BATCH_SIZE:
                remaining = deadline - self._loop.time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            # Run generation in executor so we don't block the event loop
            await self._loop.run_in_executor(self._executor, self._generate_batch, batch)

    def _generate_batch(self, batch: list[GenerateRequest]) -> None:
        try:
            results = self._run_generate(batch)
        except Exception as exc:
            logger.exception("Generation failed")
            for req in batch:
                if not req.future.done():
                    self._loop.call_soon_threadsafe(req.future.set_exception, exc)
            return

        for req, result in zip(batch, results):
            self._loop.call_soon_threadsafe(req.future.set_result, result)

    # ------------------------------------------------------------------
    # Core generation (runs in thread)
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Format messages, disabling thinking mode."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Fallback if enable_thinking is not supported by this tokenizer version
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _strip_thinking(self, text: str) -> str:
        """Remove <think>…</think> blocks if the model emits them."""
        return _THINK_RE.sub("", text).strip()

    def _run_generate(self, batch: list[GenerateRequest]) -> list[GenerateResult]:
        prompts = [self._apply_chat_template(req.messages) for req in batch]

        # Tokenize with left-padding so all sequences are right-aligned
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        max_new_tokens = max(req.max_tokens for req in batch)
        # All requests in the batch must use compatible sampling params.
        # We use the first request's params; for a proper engine this would
        # be handled per-sequence, but transformers generate() uses a single
        # set of params for the whole batch.
        req0 = batch[0]
        do_sample = req0.temperature > 0.0

        gen_kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = req0.temperature
            gen_kwargs["top_p"] = req0.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)

        results: list[GenerateResult] = []
        for i, req in enumerate(batch):
            prompt_len = int(prompt_lengths[i])
            new_tokens = output_ids[i, prompt_len:]

            # Determine finish reason
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and len(new_tokens) > 0 and new_tokens[-1].item() == eos_id:
                finish_reason = "stop"
                # Trim the EOS token from decoded output
                decode_ids = new_tokens[:-1]
            elif len(new_tokens) >= req.max_tokens:
                finish_reason = "length"
                decode_ids = new_tokens[: req.max_tokens]
            else:
                finish_reason = "stop"
                decode_ids = new_tokens

            text = self.tokenizer.decode(decode_ids, skip_special_tokens=True)
            text = self._strip_thinking(text)

            results.append(
                GenerateResult(
                    text=text,
                    prompt_tokens=prompt_len,
                    completion_tokens=len(decode_ids),
                    finish_reason=finish_reason,
                )
            )

        return results
