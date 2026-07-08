# Concurrency testing

## Tool

`classification_benchmark.py` — fires N concurrent document classification requests with exponential backoff
and reports latency, throughput, retry stats, and per-category accuracy.

```bash
python classification_benchmark.py --endpoint gemma-4-12b-it-q4 --users 100 --insecure
python classification_benchmark.py --endpoint gemma-4-12b-it-q4 --users 500 --insecure
python classification_benchmark.py --endpoint gemma-4-e4b-it    --users 200 --no-backoff --insecure
python classification_benchmark.py --endpoint my-ep --users 100 --output json | jq .
```

Parameters: `--endpoint` (required), `--users` (100), `--max-tokens` (150), `--max-retries` (8),
`--base-delay` (0.5s), `--no-backoff`, `--insecure`, `--profile`, `--output text|json`.

The benchmark uses 10 realistic financial services documents (one per category) cycled across users.
Accuracy is verified against expected labels. See `SAMPLE_DOCUMENTS` in the script to swap in your own.

---

## Results — gemma-4-12b-it-q4 (A10, 128K context, 0.90 gpu_mem_util)

### Short prompts (~460 tokens)

#### provisioned_concurrency=4 (1 A10 replica)

| Users | Success | Retries needed | Wall clock | p50 lat | p95 lat | Accuracy |
|---|---|---|---|---|---|---|
| 100 (no backoff) | 86% | — | 10.83s | 6.74s | 8.65s | 100% |
| 100 (backoff) | **100%** | 11% | 10.21s | 6.25s | 9.26s | 100% |
| 200 (backoff) | **100%** | 34% | 22.63s | 11.36s | 16.35s | 100% |
| 500 (backoff) | **100%** | 66% | 76.22s | 21.02s | 46.52s | 100% |

#### provisioned_concurrency=8 (2 A10 replicas)

| Users | Success | Retries needed | Wall clock | p50 lat | p95 lat | Accuracy |
|---|---|---|---|---|---|---|
| 100 (backoff) | **100%** | **0%** 🎉 | **7.76s** | 6.65s | 7.61s | 100% |
| 200 (backoff) | **100%** | 9% | **12.47s** | 7.60s | 10.41s | 100% |
| 500 (backoff) | **100%** | 48% | **25.17s** | 12.73s | 19.71s | 100% |

**Doubling concurrency from 4→8 cut wall clock 3× at 500 users (76s → 25s) and eliminated retries at 100 users.**

### Long prompts (~10,500 tokens) — concurrency=8

| Users | Success | Wall clock | p50 lat | Notes |
|---|---|---|---|---|
| 10 | 100% | 27.65s | 27.44s | All served, no retries |
| 50 | 100% | 128.80s | 122.67s | KV cache fills, requests queue |
| 100 | 36% | 182.62s | 132.40s | 64% timed out at 120s limit |

**At 10K tokens, practical concurrent capacity drops to ~10-20 users per 2×A10 replica.**
Prefill cost dominates: 50 × 10,500 tokens = 525K tokens KV cache saturates the A10.

---

## Key insights

### provisioned_concurrency ≠ strict admission limit
The endpoint serves far more than `provisioned_concurrency` requests concurrently — vLLM's continuous
batching queues and batches requests internally. `provisioned_concurrency=4` served 32 concurrent requests
with zero 429s in testing; 429s only appeared above ~64 concurrent users.

### Exponential backoff absorbs all 429s
Strategy: `delay = base_delay × 2^retry + jitter(0–1s)`, max 8 retries.
100/100, 200/200, 500/500 success rates — no failures even at extreme concurrency, at the cost of latency.

### Context length vs concurrency trade-off

```
KV cache budget (2×A10, gpu_mem_util=0.90, 12B QAT):
  Per replica: 24GB × 0.90 - 9.6GB weights ≈ 12 GB KV cache

KV per token (Gemma 4 12B — sliding window architecture):
  40 sliding layers:  capped at 1024 tokens × 8 heads × 256 dim × 2B = 160 MB fixed per request
  8 full layers:      n_tokens × 1 head × 512 dim × 2B (scales with context)

Context window | KV per request | Max concurrent per replica
  460 tokens   |   ~167 MB      |   ~70
  8K tokens    |   ~224 MB      |   ~53
  32K tokens   |   ~416 MB      |   ~29
  128K tokens  |   ~1.2 GB      |   ~10
  10,500 tok   |   ~250 MB      |   ~48  (matches observed behavior)
```

### Sliding window architecture is the key
Gemma 4's 40/48 layers use sliding attention (1,024-token window) — those layers' KV cache is capped
regardless of context length. Only 8 full-attention layers (1 KV head each) scale with context. This is
why 128K context costs only ~1.2 GB KV, vs ~157 GB for a standard transformer.

### Scaling guidance

| Workload | Recommended concurrency | Expected p95 |
|---|---|---|
| Interactive chat (≤8K ctx, ≤100 users) | 4 (1 replica) | ~10s |
| Batch classification (460 tok, ≤200 users) | 8 (2 replicas) | ~10s |
| Batch classification (460 tok, ≤500 users) | 12 (3 replicas) | ~20s |
| Long-doc processing (10K tok, ≤50 users) | 8 (2 replicas) | ~130s |
| Long-doc processing (10K tok, ≤200 users) | 32 (8 replicas) | ~130s |

Raise `provisioned_concurrency` in multiples of 4. Each +4 = one additional A10 GPU replica.
