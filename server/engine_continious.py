"""
Inference engine: SDPA + continuous batching + batched cross-sequence decode.

Key ideas:
  1. SDPA — PyTorch built-in Flash-Attention kernels via cuDNN (no flash-attn pkg needed).
  2. Continuous batching (Orca, 2022) — token-by-token decode; new requests injected
     after prefill; finished sequences removed immediately.
  3. Batched decode — ALL active sequences processed in ONE model() call per step.
     Qwen3.5-35B-A3B is a hybrid Gated-DeltaNet + MoE model with two layer types:
       - full_attention: standard KV cache  (B, H, T, D) — left-padded to max_T
       - linear_attention (DeltaNet): empty KV placeholder + conv_states + recurrent_states
         stored on the Qwen3NextDynamicCache object — concatenated along batch dim.
  4. Thinking fully disabled via enable_thinking=False in chat template.
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
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer, LinearAttentionCacheLayerMixin

from .config import MAX_BATCH_SIZE, MODEL_PATH, USE_BATCHED_DECODE, USE_COMPILE, USE_TP

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
    generated: list[int]      # tokens produced so far (excluding first sampled in prefill)
    last_token: torch.Tensor  # shape (1, 1) — input for next decode step
    past_key_values: Any      # Qwen3NextDynamicCache (one sequence, batch=1)


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
        self._device = next(self.model.parameters()).device

        if USE_COMPILE:
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    dynamic=True,
                    fullgraph=False,
                )
                logger.info("torch.compile enabled (mode=reduce-overhead, dynamic=True)")
            except Exception as exc:
                logger.warning("torch.compile failed, falling back to eager: %s", exc)

        logger.info(
            "Model loaded — %d parameters",
            sum(p.numel() for p in self.model.parameters()),
        )

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
        active: list[_ActiveSeq] = []

        while True:
            # Block until at least one request when idle
            if not active:
                req = self._request_queue.get()
                new_reqs = [req]
            else:
                new_reqs = []

            # Drain additional requests without blocking
            while len(active) + len(new_reqs) < MAX_BATCH_SIZE:
                try:
                    new_reqs.append(self._request_queue.get_nowait())
                except thread_queue.Empty:
                    break

            # Prefill new requests (batched when possible)
            if new_reqs:
                try:
                    active.extend(self._prefill_batch(new_reqs))
                except Exception as exc:
                    logger.exception("Prefill failed")
                    for req in new_reqs:
                        self._resolve(req.future, exc)

            if not active:
                continue

            # Decode step: sequential (default) or batched (USE_BATCHED_DECODE=1)
            try:
                if USE_BATCHED_DECODE:
                    step_results = self._decode_step_batched(active)
                else:
                    step_results = self._decode_step_sequential(active)
            except Exception as exc:
                logger.exception("Batched decode failed")
                for seq in active:
                    self._resolve(seq.req.future, exc)
                active = []
                continue

            still_active: list[_ActiveSeq] = []
            for seq, (done, result) in zip(active, step_results):
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

    def _prefill_batch(self, reqs: list[GenerateRequest]) -> list[_ActiveSeq]:
        """Prefill multiple requests in a single forward pass."""
        if len(reqs) == 1:
            return [self._prefill(reqs[0])]

        prompts = [self._apply_chat_template(req.messages) for req in reqs]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False,
        )
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

        seqs = []
        for i, req in enumerate(reqs):
            prompt_len = int(prompt_lens[i])
            first_token = self._sample(out.logits[i : i + 1, -1, :], req)
            seq_cache = _extract_seq_cache(
                out.past_key_values, i, prompt_len, max_len,
            )
            seqs.append(_ActiveSeq(
                req=req,
                prompt_len=prompt_len,
                generated=[first_token.item()],
                last_token=first_token,
                past_key_values=seq_cache,
            ))
        return seqs

    # ------------------------------------------------------------------
    # Batched decode: ONE forward pass for ALL active sequences
    # ------------------------------------------------------------------

    def _decode_step_batched(
        self, active: list[_ActiveSeq]
    ) -> list[tuple[bool, Optional[GenerateResult]]]:
        """Process all active sequences in a single model() call.

        Handles Qwen3.5-35B-A3B's hybrid cache:
          - full_attention layers:  key/value (B, H, T, D) → left-pad to max_T
          - linear_attention layers: empty KV placeholder (B, 0) + DeltaNet states
                                     (conv_states, recurrent_states) → concat on dim 0
        """
        B = len(active)

        # Compute past length per sequence from the first attention layer
        past_lens = _get_past_lens(active)
        max_past = max(past_lens)

        # --- Merge per-sequence caches into one batched cache ---
        merged = _merge_caches(active, past_lens, max_past)

        # --- Inputs ---
        input_ids = torch.cat([seq.last_token for seq in active], dim=0)  # (B, 1)
        position_ids = torch.tensor(
            [[seq.prompt_len + len(seq.generated) - 1] for seq in active],
            device=self._device, dtype=torch.long,
        )  # (B, 1)

        # Attention mask: (B, max_past + 1); 1 = real token, 0 = padding
        attn_mask = torch.zeros(B, max_past + 1, dtype=torch.long, device=self._device)
        for i, pl in enumerate(past_lens):
            attn_mask[i, max_past - pl:] = 1

        # --- Forward pass ---
        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                past_key_values=merged,
                position_ids=position_ids,
                use_cache=True,
            )

        # --- Split batched cache back to per-sequence ---
        _split_caches(out.past_key_values, active, past_lens, max_past)

        # --- Sample next token for each sequence and check termination ---
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
                finish_reason = "stop" if hit_eos else "length"
                text = self.tokenizer.decode(seq.generated, skip_special_tokens=True)
                text = _THINK_RE.sub("", text).strip()
                results.append((True, GenerateResult(
                    text=text,
                    prompt_tokens=seq.prompt_len,
                    completion_tokens=len(seq.generated),
                    finish_reason=finish_reason,
                )))
            else:
                seq.generated.append(next_id)
                results.append((False, None))

        return results

    # ------------------------------------------------------------------
    # Sequential decode: one model() call per sequence (proven path)
    # ------------------------------------------------------------------

    def _decode_step_sequential(
        self, active: list[_ActiveSeq]
    ) -> list[tuple[bool, Optional[GenerateResult]]]:
        """Process each sequence in its own forward pass — no merge/split."""
        eos_id = self.tokenizer.eos_token_id
        results: list[tuple[bool, Optional[GenerateResult]]] = []

        for seq in active:
            past_len = seq.prompt_len + len(seq.generated) - 1
            attn_mask = torch.ones(
                1, past_len + 1, dtype=torch.long, device=self._device,
            )

            with torch.inference_mode():
                out = self.model(
                    input_ids=seq.last_token,
                    attention_mask=attn_mask,
                    past_key_values=seq.past_key_values,
                    use_cache=True,
                )

            seq.past_key_values = out.past_key_values

            next_token = self._sample(out.logits[:, -1, :], seq.req)
            next_id = next_token.item()
            seq.last_token = next_token

            hit_eos = eos_id is not None and next_id == eos_id
            hit_max = len(seq.generated) >= seq.req.max_tokens

            if hit_eos or hit_max:
                if not hit_eos:
                    seq.generated.append(next_id)
                finish_reason = "stop" if hit_eos else "length"
                text = self.tokenizer.decode(
                    seq.generated, skip_special_tokens=True,
                )
                text = _THINK_RE.sub("", text).strip()
                results.append((True, GenerateResult(
                    text=text,
                    prompt_tokens=seq.prompt_len,
                    completion_tokens=len(seq.generated),
                    finish_reason=finish_reason,
                )))
            else:
                seq.generated.append(next_id)
                results.append((False, None))

        return results

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
# Cache helpers — module-level to keep the hot path clean
# ---------------------------------------------------------------------------

def _is_full_attn(layer: Any) -> bool:
    """True if this cache layer is a full-attention layer (has populated keys tensor)."""
    k = getattr(layer, "keys", None)
    return k is not None and k.numel() > 0


def _extract_seq_cache(
    out_cache: Any, seq_idx: int, prompt_len: int, max_len: int,
) -> Any:
    """Extract per-sequence cache from a batched prefill output.

    Uses copy.copy() to preserve the exact cache/layer subclass the model expects.
    Full-attention layers: strip left-padding (max_len - prompt_len positions).
    Linear-attention layers: slice along batch dim.
    """
    pad = max_len - prompt_len

    new_cache = copy.copy(out_cache)
    new_cache.layers = []
    new_cache._seen_tokens = prompt_len

    for out_layer in out_cache.layers:
        new_layer = copy.copy(out_layer)
        if _is_full_attn(out_layer):
            new_layer.keys = out_layer.keys[seq_idx : seq_idx + 1, :, pad:, :].clone()
            new_layer.values = out_layer.values[seq_idx : seq_idx + 1, :, pad:, :].clone()
        else:
            if getattr(out_layer, "is_conv_states_initialized", False):
                new_layer.conv_states = out_layer.conv_states[seq_idx : seq_idx + 1].clone()
            if getattr(out_layer, "is_recurrent_states_initialized", False):
                new_layer.recurrent_states = out_layer.recurrent_states[seq_idx : seq_idx + 1].clone()
        new_cache.layers.append(new_layer)

    return new_cache


def _get_past_lens(active: list[_ActiveSeq]) -> list[int]:
    """Return the KV past length for each active sequence.

    Transformers 5.5+: DynamicCache stores layers as a list of layer objects.
    Full-attention layers (DynamicLayer) have a populated .keys tensor (B, H, T, D).
    DeltaNet layers (LinearAttentionLayer) have empty .keys and use conv/recurrent states.
    """
    past_lens = []
    for seq in active:
        for layer in seq.past_key_values.layers:
            if _is_full_attn(layer):
                past_lens.append(layer.keys.shape[-2])
                break
        else:
            past_lens.append(0)
    return past_lens


def _merge_caches(
    active: list[_ActiveSeq], past_lens: list[int], max_past: int
) -> Any:
    """Merge N per-sequence DynamicCache objects into one batched cache.

    full_attention layers (DynamicLayer):  left-pad .keys/.values to max_past.
    linear_attention layers (LinearAttentionLayer): concat conv_states / recurrent_states
                                                    along batch dim.
    """
    sample_cache = active[0].past_key_values
    sample_layers = sample_cache.layers

    merged = copy.copy(sample_cache)
    merged.layers = []
    merged._seen_tokens = max_past

    for layer_idx, sample_layer in enumerate(sample_layers):
        new_layer = copy.copy(sample_layer)
        if _is_full_attn(sample_layer):
            keys, values = [], []
            for i, seq in enumerate(active):
                layer = seq.past_key_values.layers[layer_idx]
                k = layer.keys   # (1, H, T, D)
                v = layer.values
                pad = max_past - past_lens[i]
                if pad > 0:
                    k = torch.cat([k.new_zeros(1, k.shape[1], pad, k.shape[3]), k], dim=2)
                    v = torch.cat([v.new_zeros(1, v.shape[1], pad, v.shape[3]), v], dim=2)
                keys.append(k)
                values.append(v)
            new_layer.keys = torch.cat(keys, dim=0)    # (B, H, max_past, D)
            new_layer.values = torch.cat(values, dim=0)
        else:
            if getattr(sample_layer, "is_conv_states_initialized", False):
                new_layer.conv_states = torch.cat(
                    [seq.past_key_values.layers[layer_idx].conv_states for seq in active], dim=0
                )
            if getattr(sample_layer, "is_recurrent_states_initialized", False):
                new_layer.recurrent_states = torch.cat(
                    [seq.past_key_values.layers[layer_idx].recurrent_states for seq in active], dim=0
                )
        merged.layers.append(new_layer)

    return merged


def _split_caches(
    out_cache: Any,
    active: list[_ActiveSeq],
    past_lens: list[int],
    max_past: int,
) -> None:
    """Split the batched output cache back into per-sequence caches.

    full_attention layers: strip left-padding added in _merge_caches.
    The new token is already appended by DynamicCache.update() inside the model,
    so after the forward pass each sequence's cache length is past_lens[i] + 1.

    linear_attention layers: slice conv_states and recurrent_states on dim 0.
    """
    for i, seq in enumerate(active):
        new_cache = copy.copy(out_cache)
        new_cache.layers = []
        new_cache._seen_tokens = past_lens[i] + 1

        for out_layer in out_cache.layers:
            new_layer = copy.copy(out_layer)
            if _is_full_attn(out_layer):
                # (B, H, max_past+1, D) → (1, H, past_lens[i]+1, D)
                start = max_past - past_lens[i]
                new_layer.keys = out_layer.keys[i:i+1, :, start:, :].clone()
                new_layer.values = out_layer.values[i:i+1, :, start:, :].clone()
            else:
                if getattr(out_layer, "is_conv_states_initialized", False):
                    new_layer.conv_states = out_layer.conv_states[i:i+1].clone()
                if getattr(out_layer, "is_recurrent_states_initialized", False):
                    new_layer.recurrent_states = out_layer.recurrent_states[i:i+1].clone()
            new_cache.layers.append(new_layer)

        seq.past_key_values = new_cache


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_logits[cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
    return logits.scatter(1, sorted_idx, sorted_logits)
