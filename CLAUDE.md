# Hackathon: Inference Engine — Project Rules for Claude

## Context

5-hour hackathon. Goal: beat vLLM throughput on 8xH200 while keeping GSM8K-CoT correctness >= 87.5%.
Model: Qwen/Qwen3.5-35B-A3B (35B total, 3B active, MoE, Hybrid DeltaNet + MoE).
Hardware: 8x NVIDIA H200 (141 GB each, NVLink). Total ~1.1 TB VRAM. Model in BF16 ≈ 70 GB.

**Remaining VRAM after model load ≈ 1 TB — massive KV cache budget.**

## Priority Rule

**Speed over perfection. Ship working code in each iteration, measure, then improve.**
Only low-hanging fruits. No speculative ideas without prior art. If an optimization is uncertain, skip it.

---

## Commands

### Start server (development, single GPU for testing)
```bash
MODEL_PATH=/path/to/Qwen3.5-35B-A3B ./start.sh 8000
```

### Start server (production, 8xH200)
```bash
MODEL_PATH=/path/to/Qwen3.5-35B-A3B \
MAX_BATCH_SIZE=64 \
BATCH_TIMEOUT=0.02 \
./start.sh 8000
```

### Health / conformance check
```bash
python -m eval.check_server --base-url http://localhost:8000
```

### Correctness eval (must pass >= 87.5%)
```bash
python -m eval.correctness.run_correctness --base-url http://localhost:8000
```

### Throughput benchmark (save results)
```bash
python -m eval.throughput.run_throughput \
  --base-url http://localhost:8000 \
  --output results/iter_N_$(date +%Y%m%d_%H%M).json
```

### Quick single-concurrency smoke test
```bash
python -m eval.throughput.run_throughput \
  --base-url http://localhost:8000 \
  --concurrency 1 8 32 \
  --num-requests 8 \
  --output /tmp/smoke.json
```

---

## Iteration Plan

Each iteration: implement → start server → smoke test → full throughput → log in RESULTS.md.

### Iter 0 — Baseline (already done)
Vanilla transformers `model.generate()`, `device_map="auto"`, `MAX_BATCH_SIZE=8`.
Record current numbers in RESULTS.md before any changes.

---

### Iter 1 — Flash Attention + Batch Tuning (~30 min)
**Expected gain: 20-40% at all concurrency levels.**

Changes:
- Add `attn_implementation="flash_attention_2"` to `AutoModelForCausalLM.from_pretrained`
- Increase `MAX_BATCH_SIZE` to 64
- Decrease `BATCH_TIMEOUT` to 0.02s (20ms)
- Use `repetition_penalty` only if needed; keep sampling path clean

Risk: Flash Attention requires `flash-attn` package. Install with:
```bash
pip install flash-attn --no-build-isolation
```
If unavailable, use `attn_implementation="sdpa"` (PyTorch scaled_dot_product_attention, still faster than eager).

---

### Iter 2 — Continuous Batching (iteration-level scheduling) (~2-3 hours)
**Expected gain: 2-5x at high concurrency (c=16, 32, 64).**

This is the biggest win. Current static batching means all requests in a batch wait for the longest sequence. At concurrency=64, short requests waste GPU time waiting.

**Architecture:**
- Replace `model.generate()` with a manual decode loop
- Maintain a pool of "active sequences" (prefilled, now decoding)
- Each decode step: run one forward pass with the full active batch
- After each step: remove finished sequences, pull new requests from queue (after their prefill pass)
- Prefill: process new requests in batches to populate KV cache, then join active pool

**Key data structures:**
```python
@dataclass
class Sequence:
    req_id: int
    input_ids: torch.Tensor       # [seq_len]
    past_key_values: tuple        # KV cache for this sequence
    generated_ids: list[int]
    max_new_tokens: int
    future: asyncio.Future
```

**Steps to implement:**
1. `prefill(seq)` — single forward pass, no KV cache, returns `past_key_values` + first token
2. `decode_step(active_seqs)` — batch forward pass using cached KV, extend each sequence
3. Scheduler loop: pull from queue → prefill → add to active pool → decode until done

**Batching trick for decode:** All sequences can be processed together even with different lengths because we use KV cache (input is always 1 token per sequence). No padding needed in decode phase. Padding only in prefill phase.

**Important:** `past_key_values` from HuggingFace models is a tuple of (key, value) per layer. Stacking these for a dynamic batch requires either:
- Per-sequence KV cache (simpler, less efficient — start here)
- Batched KV cache with padding (harder, skip for now)

Start with per-sequence separate forward passes in decode if batched decode is too complex. Even sequential decode of 64 sequences with KV cache is faster than padded static batching.

---

### Iter 3 — torch.compile decode step (~1 hour)
**Expected gain: 10-25% on decode throughput.**

```python
self._decode_fn = torch.compile(
    self.model,
    mode="reduce-overhead",
    fullgraph=False,
)
```

Apply only to the decode step (single-token forward pass), not prefill.
Warm up with a few dummy passes before serving real traffic.

Risk: `torch.compile` + dynamic shapes can be tricky. Use `dynamic=True` or disable if it causes errors.

---

### Iter 4 — CUDA Graphs for decode (~30 min, if time allows)
**Expected gain: 5-15% on decode latency.**

Capture a CUDA graph for the decode step at fixed batch sizes (1, 2, 4, 8, 16, 32, 64).
Only worthwhile if Iter 3 is working.

---

## What NOT to do (time sinks)

- Custom tensor parallelism from scratch — too complex, 8+ hours
- Custom CUDA/Triton kernels for attention — Flash Attention already does this
- Speculative decoding — high risk for correctness, complex to implement correctly
- FP8/INT8 — not allowed
- Paged attention allocator from scratch — high complexity, medium gain
- Multi-process serving — complicates state management

---

## Correctness Guardrails

- **Never** remove thinking-mode stripping (`_strip_thinking`)
- **Never** change the chat template to disable thinking without verifying GSM8K score
- After any engine change, run correctness eval before recording throughput results
- Temperature=0 must be deterministic (greedy decoding, `do_sample=False`)

---

## Model Notes

- Qwen3.5-35B-A3B: Hybrid architecture — some layers are DeltaNet (linear attention), others are MoE
- The model may not support all standard attention implementations — test carefully
- `enable_thinking=False` should be set in `apply_chat_template` to suppress `<think>` tags at source (cleaner than stripping)
- Check tokenizer `apply_chat_template` signature — it has `enable_thinking` kwarg in newer Qwen tokenizers

---

## Environment

```bash
# Install server deps
uv pip install -e ".[server]"

# Flash attention (if supported)
pip install flash-attn --no-build-isolation

# Verify GPU setup
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

---

## Results Tracking

After every iteration, record results in `RESULTS.md`.
Use this format for the throughput table:

```
| Concurrency | tok/s | vs vLLM baseline |
```

Baseline vLLM numbers (from README):
| c=1: 984 | c=2: 1753 | c=4: 3038 | c=8: 4749 | c=16: 6446 | c=32: 11144 | c=64: 12810 |
