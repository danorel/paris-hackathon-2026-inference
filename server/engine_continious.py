"""
Inference engine: SDPA + continuous batching + persistent batched KV cache.

Key ideas:
  1. Continuous batching (Orca) — token-by-token decode, new requests injected after
     prefill, finished sequences removed immediately.
  2. Persistent batched cache — ONE cache object shared across ALL active sequences.
     No merge/split per decode step. Cost breakdown:
       - Old: O(B × T × H × D) copies every decode step (1024 steps × overhead = slow)
       - New: O(T × H × D) copy ONCE when a sequence joins the batch
     At ISL=1024 this eliminates the dominant bottleneck at high concurrency.
  3. Cache insert on join — when a new sequence finishes prefill, its per-sequence
     cache is merged into the persistent batched cache once.
  4. Cache shrink on leave — when a sequence finishes, its row is removed once.
  5. SDPA attention, thinking disabled via enable_thinking=False.
"""

import asyncio
import copy
import logging
import queue as thread_queue
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MAX_BATCH_SIZE, MODEL_PATH, USE_COMPILE, USE_TP

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
    generated: list[int]      # tokens produced so far
    last_token: torch.Tensor  # shape (1, 1) — input for next decode step
    batch_idx: int            # row index in the persistent batched cache
    past_len: int             # number of real KV tokens in this sequence's slot


class InferenceEngine:
    def __init__(self) -> None:
        logger.info("Loading tokenizer from %s", MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        n_gpus = torch.cuda.device_count()
        if USE_TP or n_gpus > 1:
            logger.info("Loading model (BF16, SDPA, device_map=auto, %d GPUs)", n_gpus)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map="auto",
            )
        else:
            logger.info("Loading model (BF16, SDPA, single GPU)")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to("cuda")
        self.model.eval()
        self._device = next(self.model.parameters()).device

        if USE_COMPILE:
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    dynamic=True,
                    fullgraph=False,
                )
                logger.info("torch.compile enabled")
            except Exception as exc:
                logger.warning("torch.compile failed, falling back to eager: %s", exc)

        logger.info("Model loaded — %d parameters",
                    sum(p.numel() for p in self.model.parameters()))

        # Persistent batched KV cache — None when no sequences are active
        self._batched_cache: Any = None

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
        self._request_queue.put(GenerateRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            future=future,
        ))
        return await future

    # ------------------------------------------------------------------
    # Decode thread — persistent batched cache loop
    # ------------------------------------------------------------------

    def _continuous_decode_loop(self) -> None:
        active: list[_ActiveSeq] = []

        while True:
            # Block until at least one request when idle
            if not active:
                req = self._request_queue.get()
                new_reqs = [req]
                self._batched_cache = None  # reset on idle → first request
            else:
                new_reqs = []

            # Drain additional waiting requests (non-blocking)
            while len(active) + len(new_reqs) < MAX_BATCH_SIZE:
                try:
                    new_reqs.append(self._request_queue.get_nowait())
                except thread_queue.Empty:
                    break

            # Prefill new requests and insert into persistent cache
            if new_reqs:
                try:
                    new_seqs = self._prefill_and_insert(new_reqs, len(active))
                    active.extend(new_seqs)
                except Exception as exc:
                    logger.exception("Prefill failed")
                    for req in new_reqs:
                        self._resolve(req.future, exc)

            if not active:
                continue

            # Single forward pass — no merge, no split
            try:
                step_results = self._decode_step(active)
            except Exception as exc:
                logger.exception("Decode step failed")
                for seq in active:
                    self._resolve(seq.req.future, exc)
                active = []
                self._batched_cache = None
                continue

            # Separate finished from still-active
            done_indices: list[int] = []
            still_active: list[_ActiveSeq] = []
            for i, (seq, (done, result)) in enumerate(zip(active, step_results)):
                if done:
                    done_indices.append(i)
                    self._resolve(seq.req.future, result)
                else:
                    still_active.append(seq)

            # Remove finished sequences from persistent cache (one operation)
            if done_indices:
                self._remove_from_cache(done_indices, len(active))
                for new_idx, seq in enumerate(still_active):
                    seq.batch_idx = new_idx

            active = still_active

    # ------------------------------------------------------------------
    # Prefill + insert into persistent cache
    # ------------------------------------------------------------------

    def _prefill_and_insert(
        self, reqs: list[GenerateRequest], existing_count: int
    ) -> list[_ActiveSeq]:
        """Prefill all new requests and merge their caches into _batched_cache."""
        # --- Prefill (batched when multiple) ---
        if len(reqs) == 1:
            seq_caches = [self._prefill_one(reqs[0])]
        else:
            seq_caches = self._prefill_multi(reqs)

        # seq_caches: list of (seq_cache, prompt_len, first_token)

        # --- Insert each new cache into _batched_cache ---
        seqs: list[_ActiveSeq] = []
        for i, (req, (seq_cache, prompt_len, first_token)) in enumerate(
            zip(reqs, seq_caches)
        ):
            batch_idx = existing_count + i
            if self._batched_cache is None:
                # First sequence — its cache IS the batched cache (batch=1)
                self._batched_cache = seq_cache
            else:
                # Merge new sequence into existing batched cache
                self._batched_cache = _append_to_batched_cache(
                    self._batched_cache, seq_cache
                )

            seqs.append(_ActiveSeq(
                req=req,
                prompt_len=prompt_len,
                generated=[first_token.item()],
                last_token=first_token,
                batch_idx=batch_idx,
                past_len=prompt_len,
            ))

        return seqs

    def _prefill_one(self, req: GenerateRequest):
        """Single-request prefill. Returns (seq_cache, prompt_len, first_token)."""
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

        first_token = self._sample(out.logits[:, -1, :], req)
        return out.past_key_values, input_ids.shape[1], first_token

    def _prefill_multi(self, reqs: list[GenerateRequest]):
        """Batched prefill for multiple requests. Returns list of (cache, prompt_len, token)."""
        prompts = [self._apply_chat_template(req.messages) for req in reqs]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        prompt_lens = attention_mask.sum(dim=1).tolist()
        max_len = input_ids.shape[1]

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        results = []
        for i, req in enumerate(reqs):
            prompt_len = int(prompt_lens[i])
            first_token = self._sample(out.logits[i:i+1, -1, :], req)
            seq_cache = _extract_seq_cache(out.past_key_values, i, prompt_len, max_len)
            results.append((seq_cache, prompt_len, first_token))
        return results

    # ------------------------------------------------------------------
    # Decode step — uses persistent batched cache directly, no merge/split
    # ------------------------------------------------------------------

    def _decode_step(
        self, active: list[_ActiveSeq]
    ) -> list[tuple[bool, Optional[GenerateResult]]]:
        B = len(active)
        past_lens = [seq.past_len for seq in active]
        max_past = max(past_lens)

        input_ids = torch.cat([seq.last_token for seq in active], dim=0)  # (B, 1)
        position_ids = torch.tensor(
            [[seq.prompt_len + len(seq.generated) - 1] for seq in active],
            device=self._device, dtype=torch.long,
        )

        # Left-padded attention mask
        attn_mask = torch.zeros(B, max_past + 1, dtype=torch.long, device=self._device)
        for i, pl in enumerate(past_lens):
            attn_mask[i, max_past - pl:] = 1

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                past_key_values=self._batched_cache,
                position_ids=position_ids,
                use_cache=True,
            )

        # Persist the updated cache (no split)
        self._batched_cache = out.past_key_values
        for seq in active:
            seq.past_len += 1

        # Sample and check termination
        eos_id = self.tokenizer.eos_token_id
        results: list[tuple[bool, Optional[GenerateResult]]] = []
        for i, seq in enumerate(active):
            next_token = self._sample(out.logits[i:i+1, -1, :], seq.req)
            next_id = next_token.item()
            seq.last_token = next_token

            hit_eos = eos_id is not None and next_id == eos_id
            hit_max = len(seq.generated) >= seq.req.max_tokens

            if hit_eos or hit_max:
                if not hit_eos:
                    seq.generated.append(next_id)
                text = self.tokenizer.decode(seq.generated, skip_special_tokens=True)
                text = _THINK_RE.sub("", text).strip()
                results.append((True, GenerateResult(
                    text=text,
                    prompt_tokens=seq.prompt_len,
                    completion_tokens=len(seq.generated),
                    finish_reason="stop" if hit_eos else "length",
                )))
            else:
                seq.generated.append(next_id)
                results.append((False, None))

        return results

    # ------------------------------------------------------------------
    # Remove finished sequences from persistent cache
    # ------------------------------------------------------------------

    def _remove_from_cache(self, done_indices: list[int], total: int) -> None:
        """Remove rows at done_indices from _batched_cache."""
        keep = [i for i in range(total) if i not in set(done_indices)]
        if not keep:
            self._batched_cache = None
            return

        new_cache = copy.copy(self._batched_cache)
        new_cache.layers = []

        for layer in self._batched_cache.layers:
            new_layer = copy.copy(layer)
            if _is_full_attn(layer):
                new_layer.keys = layer.keys[keep]
                new_layer.values = layer.values[keep]
            else:
                if getattr(layer, "is_conv_states_initialized", False):
                    new_layer.conv_states = layer.conv_states[keep]
                if getattr(layer, "is_recurrent_states_initialized", False):
                    new_layer.recurrent_states = layer.recurrent_states[keep]
            new_cache.layers.append(new_layer)

        new_cache._seen_tokens = self._batched_cache._seen_tokens
        self._batched_cache = new_cache

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, req: GenerateRequest) -> torch.Tensor:
        if req.temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / req.temperature
        if req.top_p < 1.0:
            logits = _top_p_filter(logits, req.top_p)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

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

    def _resolve(self, future: asyncio.Future, value: Any) -> None:
        if isinstance(value, Exception):
            self._loop.call_soon_threadsafe(future.set_exception, value)
        else:
            self._loop.call_soon_threadsafe(future.set_result, value)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _is_full_attn(layer: Any) -> bool:
    k = getattr(layer, "keys", None)
    return k is not None and k.numel() > 0


def _extract_seq_cache(out_cache: Any, seq_idx: int, prompt_len: int, max_len: int) -> Any:
    """Extract a single sequence's cache from a batched prefill output."""
    pad = max_len - prompt_len
    new_cache = copy.copy(out_cache)
    new_cache.layers = []
    new_cache._seen_tokens = prompt_len

    for out_layer in out_cache.layers:
        new_layer = copy.copy(out_layer)
        if _is_full_attn(out_layer):
            new_layer.keys = out_layer.keys[seq_idx:seq_idx+1, :, pad:, :].clone()
            new_layer.values = out_layer.values[seq_idx:seq_idx+1, :, pad:, :].clone()
        else:
            if getattr(out_layer, "is_conv_states_initialized", False):
                new_layer.conv_states = out_layer.conv_states[seq_idx:seq_idx+1].clone()
            if getattr(out_layer, "is_recurrent_states_initialized", False):
                new_layer.recurrent_states = out_layer.recurrent_states[seq_idx:seq_idx+1].clone()
        new_cache.layers.append(new_layer)

    return new_cache


def _append_to_batched_cache(batched: Any, seq: Any) -> Any:
    """Append a single-sequence cache (batch=1) to the persistent batched cache.

    Called ONCE per sequence join — not on every decode step.
    Left-pads the shorter side so all sequences align at the rightmost position.
    """
    # Find max past length in each cache
    b_max = _cache_max_len(batched)
    s_max = _cache_max_len(seq)
    new_max = max(b_max, s_max)

    new_cache = copy.copy(batched)
    new_cache.layers = []
    new_cache._seen_tokens = new_max

    for layer_idx, (b_layer, s_layer) in enumerate(zip(batched.layers, seq.layers)):
        new_layer = copy.copy(b_layer)
        if _is_full_attn(b_layer):
            bk = _left_pad_kv(b_layer.keys, new_max)    # (B, H, new_max, D)
            bv = _left_pad_kv(b_layer.values, new_max)
            sk = _left_pad_kv(s_layer.keys, new_max)    # (1, H, new_max, D)
            sv = _left_pad_kv(s_layer.values, new_max)
            new_layer.keys = torch.cat([bk, sk], dim=0)
            new_layer.values = torch.cat([bv, sv], dim=0)
        else:
            if getattr(b_layer, "is_conv_states_initialized", False):
                new_layer.conv_states = torch.cat(
                    [b_layer.conv_states, s_layer.conv_states], dim=0
                )
            if getattr(b_layer, "is_recurrent_states_initialized", False):
                new_layer.recurrent_states = torch.cat(
                    [b_layer.recurrent_states, s_layer.recurrent_states], dim=0
                )
        new_cache.layers.append(new_layer)

    return new_cache


def _cache_max_len(cache: Any) -> int:
    for layer in cache.layers:
        if _is_full_attn(layer):
            return layer.keys.shape[2]
    return 0


def _left_pad_kv(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Left-pad a (B, H, T, D) tensor to target_len on dim 2."""
    T = t.shape[2]
    if T == target_len:
        return t
    pad = target_len - T
    return torch.cat(
        [t.new_zeros(t.shape[0], t.shape[1], pad, t.shape[3]), t], dim=2
    )


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_logits[cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
    return logits.scatter(1, sorted_idx, sorted_logits)
