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

## Iter 3 — Batched Decode (cross-sequence)

**Date:** TBD

**Changes:** Process all active sequences in a single `model()` call with padded KV caches and per-sequence `position_ids`. Expected: c=2 and higher should show real throughput gains.

**Correctness:** TBD

**Throughput:**

| Concurrency | tok/s | vs vLLM c=1 |
|---|---|---|
| 1 | TBD | — |
| 2 | TBD | — |
| 4 | TBD | — |
| 8 | TBD | — |

**Notes:** —

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
