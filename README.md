# deploy_custom_llm_model

Deploys open-weight Gemma 4 models to **Databricks Custom LLM Serving** using vLLM,
packaged as a Databricks Asset Bundle with automated benchmarking and agent evaluation.

## Supported models

| Endpoint | Model | Size | Serving |
|---|---|---|---|
| `gemma-4-e2b-it` | `google/gemma-4-E2B-it` | ~5B eff. params, bf16 | 1×A10 |
| `gemma-4-e4b-it` | `google/gemma-4-E4B-it` | ~8B params, bf16 | 1×A10 |
| `gemma-4-12b-it-q4` | `google/gemma-4-12B-it-qat-w4a16-ct` | 12B, QAT 4-bit | 1×A10 |

All models are Apache-2.0 licensed and **not gated** — no Hugging Face token required.

## Project layout

```
databricks.yml                                   # Bundle config, variables, workspace targets
resources/deploy_gemma_serving.job.yml           # 3-task job: deploy → benchmark → agent
notebooks/
  01 - Deploy Gemma Serving Endpoint.py          # pip → download → log → register → endpoint → query
  02 - Benchmark Endpoint.py                     # Concurrency sweep; appends to endpoint_benchmarks
  03 - Agent Implementation Example.py           # LangGraph tool-calling agent with MLflow eval
benchmark.py                                     # Local CLI benchmark (shared core with notebook 02)
```

## Prerequisites

- **Databricks workspace** with Serverless GPU compute (AI Runtime) and Model Serving access
- **Unity Catalog** catalog/schema you can write to
- **Databricks CLI** installed

### Authentication

Log in using the workspace host first, then use the resulting profile in all commands:

```bash
# Login (creates or updates a profile for this workspace)
databricks auth login --host <workspace_host_url>

# Optionally assign a named profile for convenience
databricks auth login --host <workspace_host_url> --profile <profile_name>
```

Once authenticated, pass `--profile <profile_name>` to every CLI command, or set it once in `databricks.yml`:

```yaml
workspace:
  host: <workspace_host_url>
  profile: <profile_name>
```

## Deploy and run

`bundle deploy` bakes variables into the job's `base_parameters`. Always deploy before run.

### E2B
```bash
databricks bundle deploy -t dev \
  --var="catalog=meli_demo" --var="schema=default" \
  --var="base_model=google/gemma-4-E2B-it" --var="model_name=gemma_4_e2b_it" \
  --var="endpoint_name=gemma-4-e2b-it" --var="max_model_len=16384" \
  --var="gpu_mem_util=0.70" --var="embed_weights=false" --var="tool_call_parser=gemma4" \
  --var="tensor_parallel=1" --var="workload_type=GPU_MEDIUM" --var="hardware_accelerator=GPU_1xA10"

databricks bundle run -t dev deploy_gemma_serving_endpoint <same --var flags>
```

### E4B
```bash
# Same as above with: model_name=gemma_4_e4b_it endpoint_name=gemma-4-e4b-it
#                     max_model_len=8192 gpu_mem_util=0.80
```

### 12B QAT
```bash
# Same as above with: base_model=google/gemma-4-12B-it-qat-w4a16-ct
#                     model_name=gemma_4_12b_it_q4 endpoint_name=gemma-4-12b-it-q4
#                     max_model_len=8192 gpu_mem_util=0.80
```

> See `.claude/docs/runbook.md` for exact copy-paste commands for all three models.

## What the job does

1. **Install deps** — `pip<24.1` first (avoids a base-image conflict exit-code bug), then vLLM 0.23.0+cu129
2. **Download weights** — via `hf download` at container startup (all models use `embed_weights=false`)
3. **Smoke test** — disabled by default (`smoke_test=false`) due to a Databricks GPU container FIPS issue; serving endpoints are unaffected
4. **Log to MLflow** — `ChatModel` placeholder + vLLM entrypoint string as metadata
5. **Register to UC** — `mlflow.register_model(env_pack="databricks_model_serving")`
6. **Create/update endpoint** — fixed concurrency (4), `GPU_MEDIUM` (1×A10), no scale-to-zero
7. **Benchmark** — concurrency sweep; results in `<catalog>.<schema>.endpoint_benchmarks`
8. **Agent eval** — LangGraph agent with local Python executor, 5-example MLflow evaluation

## Query an endpoint

```python
from openai import OpenAI

client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=f"{DATABRICKS_HOST}/serving-endpoints")
r = client.chat.completions.create(
    model="gemma-4-e4b-it",
    messages=[{"role": "user", "content": "Hello"}],
)
print(r.choices[0].message.content)
```

Native tool calling works on all three endpoints (`--tool-call-parser gemma4`).

## Benchmark

```bash
# Local CLI
python benchmark.py --endpoint gemma-4-e4b-it --levels 1,4,8 --max-tokens 256 --insecure

# Historical results
SELECT endpoint, C, sys_tps, p50_lat_s, p50_ttft_ms
FROM meli_demo.default.endpoint_benchmarks
ORDER BY ts DESC, endpoint, C
```

**Measured throughput (A10, 256-tok):** E2B 74 tok/s → 608 tok/s at C=8 | E4B 43 tok/s → 338 tok/s | 12B QAT 46 tok/s → 331 tok/s

## Concurrency & load testing

`classification_benchmark.py` runs N concurrent document classification requests with exponential backoff:

```bash
python classification_benchmark.py --endpoint gemma-4-12b-it-q4 --users 100 --insecure
python classification_benchmark.py --endpoint gemma-4-12b-it-q4 --users 500 --insecure
python classification_benchmark.py --endpoint my-endpoint --users 200 --no-backoff --insecure
```

**Measured results on gemma-4-12b-it-q4 (128K context, A10):**

| Config | Users | Success | Wall clock | p95 latency |
|---|---|---|---|---|
| concurrency=4, ~460 tok, backoff | 100 | 100% | 10.2s | 9.3s |
| concurrency=4, ~460 tok, backoff | 500 | 100% | 76.2s | 46.5s |
| concurrency=8, ~460 tok, backoff | 100 | 100% | **7.8s** | **7.6s** |
| concurrency=8, ~460 tok, backoff | 500 | 100% | **25.2s** | **19.7s** |
| concurrency=8, ~10,500 tok, backoff | 50 | 100% | 128.8s | 128.6s |
| concurrency=8, ~10,500 tok, backoff | 100 | 36% | — | — (timeout) |

Exponential backoff (`delay = 0.5s × 2^retry + jitter`) achieves 100% success at all load levels for
short prompts. Long prompts (10K+ tokens) saturate the KV cache faster — keep concurrency ≤ ~20 users
per A10 replica for 10K token workloads.

> Full results, KV cache math, and scaling guidance: `.claude/docs/concurrency-testing.md`

## Known issues

| Issue | Fix |
|---|---|
| pip 24.1+ conflict (`databricks-serverless-gpu` requires `mlflow<3.0`) | `%pip install "pip<24.1"` before the main install — already in notebook |
| Smoke test FIPS SSL failure in GPU notebook containers | `smoke_test=false` (default) — serving containers unaffected |
| `FATAL FIPS SELFTEST FAILURE` from opencv | `env -u OPENSSL_FORCE_FIPS_MODE` in vLLM entrypoint — already in notebook |
| Artifact upload limit (~10 GiB per tar) | `embed_weights=false` — weights downloaded at container startup |
| `fastapi 0.137` breaks vLLM router | `"fastapi<0.137"` in pip install — already in notebook |

> Full details: `.claude/docs/known-issues.md`

## Extending

- **Add a model**: see `.claude/docs/extending.md`
- **Change vLLM version**: update the wheel URL in notebook 01's `%pip install` cell
- **Add eval examples**: edit `eval_dataset` list in notebook 03
- **Add a workspace target**: add entry to `targets:` in `databricks.yml`
