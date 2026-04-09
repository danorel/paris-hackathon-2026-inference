# Results Log

Each iteration records: what changed, correctness score, throughput numbers.

---

## vLLM Baseline (reference — from README)

Source: vLLM v0.19.0, TP=8, BF16, 8xH200.

**Correctness:** 91.5% exact match / 91.0% strict-match (GSM8K-CoT, 200 problems)

**Throughput (1024 input / 1024 output, 64 requests/level):**

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

## Iter 0 — Vanilla server (our implementation)

**Date:** 2026-04-09

**Changes:** Initial server. `transformers` `model.generate()`, `device_map="auto"`, `MAX_BATCH_SIZE=8`, `BATCH_TIMEOUT=50ms`.

**Correctness:**
- flexible-extract exact_match: **0.89** (89%) ✅ passes gate (>=87.5%)
- strict-match exact_match: **0.69** (69%) — expected, CoT outputs are verbose

**Throughput:** TBD (run full benchmark on 8xH200)

| Concurrency | tok/s | vs vLLM |
|---|---|---|
| 1 | TBD | — |
| 2 | TBD | — |
| 4 | TBD | — |
| 8 | TBD | — |
| 16 | TBD | — |
| 32 | TBD | — |
| 64 | TBD | — |

**Notes:** Correctness passes gate. Throughput bottleneck: static batching, no flash attention.

---

## Iter 1 — Flash Attention + Batch Tuning

**Date:** 2026-04-09

**Changes:**
- `attn_implementation="sdpa"` (PyTorch SDPA, faster than eager)
- `MAX_BATCH_SIZE=64` (was 8)
- `BATCH_TIMEOUT=20ms` (was 50ms)
- `enable_thinking=False` in `apply_chat_template` (suppress think tags at source)

**Correctness:**
- flexible-extract exact_match: **0.885** (88.5%) ✅ passes gate (>=87.5%)
- strict-match exact_match: **0.680** (68.0%)

**Throughput:**

| Concurrency | tok/s | vs vLLM |
|---|---|---|
| 1 | TBD | — |
| 2 | TBD | — |
| 4 | TBD | — |
| 8 | TBD | — |
| 16 | TBD | — |
| 32 | TBD | — |
| 64 | TBD | — |

**Notes:** —

---

## Iter 2 — Continuous Batching

**Date:** TBD

**Changes:**
- Replaced `model.generate()` with manual token-by-token decode loop
- Iteration-level scheduling: new requests join after prefill, finished requests leave immediately
- No padding during decode phase (1 token per active sequence per step)

**Correctness:** TBD

**Throughput:**

| Concurrency | tok/s | vs vLLM |
|---|---|---|
| 1 | TBD | — |
| 2 | TBD | — |
| 4 | TBD | — |
| 8 | TBD | — |
| 16 | TBD | — |
| 32 | TBD | — |
| 64 | TBD | — |

**Notes:** —

---

## Iter 3 — torch.compile

**Date:** TBD

**Changes:**
- `torch.compile(model, mode="reduce-overhead")` on decode step

**Correctness:** TBD

**Throughput:**

| Concurrency | tok/s | vs vLLM |
|---|---|---|
| 1 | TBD | — |
| 2 | TBD | — |
| 4 | TBD | — |
| 8 | TBD | — |
| 16 | TBD | — |
| 32 | TBD | — |
| 64 | TBD | — |

**Notes:** —

---

## Iter 4 — CUDA Graphs (if time allows)

**Date:** TBD

**Changes:** CUDA graph capture for decode step at fixed batch sizes (1, 2, 4, 8, 16, 32, 64).

**Correctness:** TBD

**Throughput:**

| Concurrency | tok/s | vs vLLM |
|---|---|---|
| 1 | TBD | — |
| 2 | TBD | — |
| 4 | TBD | — |
| 8 | TBD | — |
| 16 | TBD | — |
| 32 | TBD | — |
| 64 | TBD | — |

**Notes:** —
