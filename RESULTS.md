# Results Log

Each iteration records: what changed, correctness score, throughput numbers.

> **Note:** All throughput measurements use `enable_thinking=False` (suppressed at chat-template level).
> Thinking tokens are never generated — counts reflect only visible output tokens.
>
> **Benchmark config (single-GPU runs):** ISL=256, OSL=256, 8 req/level, 128 prompts, GPU 4 (H200).

---

## vLLM Baseline (reference — from README)

Source: vLLM v0.19.0, TP=8, BF16, 8×H200.

**Correctness:** 91.5% exact match / 91.0% strict-match (GSM8K-CoT, 200 problems)

**Throughput (ISL=1024, OSL=1024, 64 req/level):**

| Concurrency | tok/s | Wall Time (s) |
|---|---|---|
| 1 | 984 | 86.1 |
| 2 | 1,753 | 47.5 |
| 4 | 3,038 | 27.3 |
| 8 | 4,749 | 17.9 |
| 16 | 6,446 | 13.6 |
| 32 | 11,144 | 7.5 |
| 64 | 12,810 | 6.5 |

**Weighted score** = 1×984 + 1×1753 + 2×3038 + 2×4749 + 4×6446 + 4×11144 + 8×12810 = **248,930**

---

## Iter 0a — Multi-GPU pipeline parallel (original baseline)

**Date:** 2026-04-09 / 13:30

**Config:** `model.generate()`, `device_map="auto"` (6 GPU pipeline parallel), `MAX_BATCH_SIZE=8`, `BATCH_TIMEOUT=50ms`, eager attention.

**Correctness:** flexible-extract **89%** ✅ / strict **69%**

**Throughput** (ISL=256, OSL=256, 4 req/level, 6×H200):

| Concurrency | tok/s | vs vLLM c=1 |
|---|---|---|
| 1 | 69 | 7.0% |

**Notes:** Inter-GPU transfers dominate decode under pipeline parallel — GPUs mostly idle waiting for each other.

---

## Iter 0b — Single GPU + Eager Attention + Static Batching

**Date:** 2026-04-09 / 15:45

**Config:** `ENGINE_MODE=0`, `model.generate()`, eager attention, `CUDA_VISIBLE_DEVICES=4` (1×H200).

**Correctness:** (not re-run)

**Throughput** (ISL=256, OSL=256, 8 req/level):

| Concurrency | tok/s | vs vLLM c=1 | Notes |
|---|---|---|---|
| 1 | **45.19** | 4.6% | ✓ |
| 2 | **50.19** | 5.1% | ⚠️ token_discrepancy |

**Notes:** Static batching uses `max_new_tokens = max(batch)` for all sequences — reported token counts wrong at c=2. Wall time 62–61s.

---

## Iter 1 — Single GPU + SDPA + Static Batching

**Date:** 2026-04-09 / 15:45

**Config:** `ENGINE_MODE=1`, `model.generate()`, `attn_implementation="sdpa"`, `CUDA_VISIBLE_DEVICES=4` (1×H200).

**Correctness:** (not re-run)

**Throughput** (ISL=256, OSL=256, 8 req/level):

| Concurrency | tok/s | vs vLLM c=1 | Notes |
|---|---|---|---|
| 1 | **42.91** | 4.4% | ✓ |
| 2 | **50.19** | 5.1% | ⚠️ token_discrepancy |

**Notes:**
- SDPA slightly *slower* than eager at c=1 (42.9 vs 45.2 tok/s). Qwen3.5-35B-A3B uses DeltaNet (linear attention) for some layers — SDPA may not be beneficial for all layers.
- Same token discrepancy bug as Iter 0b at c=2 (batch uses `max_new_tokens` of slowest request).
- `flash_attn` package unavailable: CUDA 13.0 vs PyTorch 12.8 mismatch.

---

## Iter 2 — Single GPU + SDPA + Continuous Batching

**Date:** 2026-04-09 / 15:45
Environment
**Config:** `ENGINE_MODE=2`, manual token-by-token decode with `past_key_values`, `attn_implementation="sdpa"`, `CUDA_VISIBLE_DEVICES=4` (1×H200).

**Changes vs Iter 1:**
- Replaced `model.generate()` with Orca-style iteration-level scheduling
- Per-request `max_tokens`, `temperature`, `top_p` — fixes token discrepancy
- New requests injected after prefill; finished sequences removed immediately
- `enable_thinking=False` at chat-template level (no `<think>` tokens ever generated)

**Correctness:** (not re-run, same prompt path)

**Throughput** (ISL=256, OSL=256, 8 req/level):

| Concurrency | tok/s | vs vLLM c=1 | Notes |
|---|---|---|---|
| 1 | **42.47** | 4.3% | ✓ no discrepancy |
| 2 | **43.38** | 4.4% | ✓ no discrepancy |

**Notes:**
- Throughput ≈ static batching at c=1 (decode loop overhead cancels SDPA gains vs eager).
- **Key win:** zero token discrepancy at c=2 (spot checks 2/2 ✓ vs 0/2 for static).
- c=2 ≈ c=1: decode steps are still sequential per active sequence (no cross-sequence batching in one `model()` call).
- Next step: batched decode — process all active sequences in a single forward pass.

---

## Concepts

### Static Batching

Requests are collected for a fixed time window (`BATCH_TIMEOUT`), then dispatched together as one padded batch to `model.generate()`. All sequences in a batch run for `max(max_tokens)` steps — shorter sequences waste GPU time waiting for the longest one.

```
Batch of 3 requests (max_tokens = 7):

step:  1   2   3   4   5   6   7
req A: tok tok tok ✓   ---  ---  ---   ← done at step 3, idles 4 steps
req B: tok tok tok tok tok ✓   ---   ← done at step 5, idles 2 steps
req C: tok tok tok tok tok tok tok ✓  ← slowest, defines batch length

GPU utilization: low at high concurrency — padding + idle slots dominate.
```

**Used in:** Iter 0 (eager), Iter 1 (SDPA). Implementation: [engine_static.py](server/engine_static.py).

---

### Continuous Batching (Orca-style)

A background thread runs a token-by-token decode loop. Each step is a single `model()` call across **all** active sequences simultaneously. Sequences are removed immediately when they finish (EOS or `max_tokens`), and new requests are admitted after their prefill pass — no waiting for a batch window.

```
Active pool evolves each decode step:

step 1: [A, B, C]      → forward pass
step 2: [A, B, C]      → forward pass
step 3: [A, B, C]      → A hits EOS → resolve A's future immediately
step 4: [B, C, D]      → D (new request) admitted after prefill
step 5: [B, C, D]      → forward pass
step 6: [B, C, D]      → B hits EOS → resolve B's future
...

GPU utilization: high — no idle slots, no padding waste at high concurrency.
```

KV caches from different sequences have different lengths → merged with left-padding for the batched forward pass, then split back per-sequence after each step.

**Used in:** Iter 2+ (SDPA + batched decode). Implementation: [engine_continious.py](server/engine_continious.py).

---

## Iter 2b — Cache API fix for transformers 5.5

**Date:** 2026-04-09

**Problem:** `engine_continious.py` was written against the transformers <5 cache API (`cache.key_cache[]`, `cache.value_cache[]`). Transformers 5.5.1 replaced these flat lists with a `cache.layers[]` list of typed layer objects.

**Error:**
```
AttributeError: 'DynamicCache' object has no attribute 'key_cache'
```

**Root cause:** API change in transformers 5.5 — `DynamicCache` now stores:

| Old API | New API (5.5+) |
|---|---|
| `cache.key_cache[i]` | `cache.layers[i].keys` |
| `cache.value_cache[i]` | `cache.layers[i].values` |
| `k.ndim == 4` (full attn check) | `k is not None and k.numel() > 0` |
| `cache.conv_states[i]` | `cache.layers[i].conv_states` |

**Fix:** Rewrote `_get_past_lens`, `_merge_caches`, `_split_caches` in [engine_continious.py](server/engine_continious.py) to use the new layer-object API (`DynamicLayer`, `LinearAttentionLayer`).

**Status:** Fix applied — server should now handle batched decode without crashing.

---

## Iter 3 — Batched Decode (cross-sequence) + copy.copy() cache fix

**Date:** 2026-04-09

**Config:** `ENGINE_MODE=2`, `USE_BATCHED_DECODE=1`, `MAX_BATCH_SIZE=64`, `CUDA_VISIBLE_DEVICES=4` (1×H200).

**Changes vs Iter 2:**
- All active sequences processed in a single `model()` call per decode step
- KV caches merged (left-padded) into one batched cache before each step, split back after
- **Key fix:** `copy.copy()` used to construct merged/split caches — preserves the exact cache/layer subclasses the model expects (plain `DynamicCache()` caused wrong outputs and 5× slowdown)
- `_seen_tokens` set correctly on reconstructed caches
- Batched prefill: multiple new requests prefilled in one forward pass

**Correctness:** spot=2/2 ✅ at all concurrency levels

**Throughput** (ISL=256, OSL=256, 8 req/level):

| Concurrency | tok/s | vs vLLM c=1 | vs Iter 2 c=1 |
|---|---|---|---|
| 1 | **39.42** | 4.0% | ~same (merge/split overhead) |
| 2 | **65.85** | 6.7% | +66% |
| 4 | **89.10** | 9.1% | +113% |
| 8 | **117.52** | 11.9% | +180% |

**Notes:**
- c=1 slightly slower than sequential (39 vs 42 tok/s) — merge/split overhead per step
- c=2,4,8 scale well: 2 sequences share one weight load → near-linear throughput gain
- c=8 gives ~3x vs c=1 (not 8× — memory bandwidth partially saturated at larger batch)
- `USE_BATCHED_DECODE=0` (sequential, default) for correctness fallback; `=1` for throughput
- Root cause of previous batched decode failure: `DynamicCache()` is a base class, model uses subclasses internally — `copy.copy()` from actual model-returned objects fixes the type mismatch

---

## Iter 4 — torch.compile decode step

**Date:** TBD

**Changes:** `torch.compile(model, mode="reduce-overhead")` on the decode forward pass. Warm-up with dummy passes before serving.

**Correctness:** TBD

**Throughput:**

| Concurrency | tok/s | vs vLLM c=1 |
|---|---|---|
| 1 | TBD | — |
| 2 | TBD | — |
| 4 | TBD | — |

**Notes:** —

---

## Iter 5 — Multi-GPU (all-GPU window)

**Date:** TBD

**Changes:** Run best single-GPU config on all available GPUs (`CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`), `device_map="auto"`.

**Correctness:** TBD

**Throughput:**

| Concurrency | tok/s | vs vLLM |
|---|---|---|
| 1 | TBD | — |
| 4 | TBD | — |
| 16 | TBD | — |
| 64 | TBD | — |

**Notes:** —
