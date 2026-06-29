# Model configurations

All models: `catalog=meli_demo  schema=default  tensor_parallel=1  workload_type=GPU_MEDIUM  hardware_accelerator=GPU_1xA10  tool_call_parser=gemma4`

## gemma-4-E2B-it

| Variable | Value | Why |
|---|---|---|
| `base_model` | `google/gemma-4-E2B-it` | |
| `model_name` | `gemma_4_e2b_it` | |
| `endpoint_name` | `gemma-4-e2b-it` | |
| `embed_weights` | `false` | weights ~9.6 GiB, borderline — false avoids artifact limit edge cases |
| `max_model_len` | `16384` | 16K context (A10 has headroom at 0.70 util) |
| `gpu_mem_util` | `0.70` | ~16.8 GB budget; ~5B bf16 weights ≈ 10 GB leaves ~6.8 GB KV |

**Benchmark (256-tok, A10)**: 74 tok/s single-stream, 608 tok/s at C=8.

## gemma-4-E4B-it

| Variable | Value | Why |
|---|---|---|
| `base_model` | `google/gemma-4-E4B-it` | |
| `model_name` | `gemma_4_e4b_it` | |
| `endpoint_name` | `gemma-4-e4b-it` | |
| `embed_weights` | `false` | weights 14.9 GiB bf16 — exceeds 10 GiB artifact limit |
| `max_model_len` | `8192` | reduced to fit KV cache at 0.80 util |
| `gpu_mem_util` | `0.80` | ~19.2 GB budget; ~14.9 GB weights leaves ~4.3 GB KV |

**Benchmark (256-tok, A10)**: 43 tok/s single-stream, 338 tok/s at C=8.

## gemma-4-12B-it-qat-w4a16-ct (QAT 4-bit)

| Variable | Value | Why |
|---|---|---|
| `base_model` | `google/gemma-4-12B-it-qat-w4a16-ct` | Official QAT compressed-tensors, NOT the GGUF variant |
| `model_name` | `gemma_4_12b_it_q4` | |
| `endpoint_name` | `gemma-4-12b-it-q4` | |
| `embed_weights` | `false` | weights 9.59 GiB — sits right at the limit; false is safer |
| `max_model_len` | `8192` | |
| `gpu_mem_util` | `0.80` | QAT 4-bit weights ≈ 9.6 GB, leaves ~9.6 GB for KV |

Requires vLLM ≥ 0.23.0 for `Gemma4UnifiedForConditionalGeneration` support (fixed in PRs #44429/#44571). `--kv-cache-dtype fp8` is valid on H100 (pass via `extra_vllm_args`) but NOT on A10 (Ampere).

**Benchmark (256-tok, A10)**: 46 tok/s single-stream, 331 tok/s at C=8 — near-identical to E4B despite 12B params.

## Adding a new model

1. Check: `curl -s "https://huggingface.co/api/models/<org>/<model>" | python3 -c "import json,sys; d=json.load(sys.stdin); print('gated:', d.get('gated'), 'params:', d.get('safetensors',{}).get('total'))"`
2. Calculate bf16 size: `params × 2 / 1e9` GiB → if > 10 GiB use `embed_weights=false`
3. Check vLLM architecture support: `vllm.config.model` registry for the `model_type` in `config.json`
4. Run with the relevant vars from this file as a template
