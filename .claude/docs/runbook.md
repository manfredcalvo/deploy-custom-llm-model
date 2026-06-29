# Runbook — deploy and run the job

## Prerequisites

```bash
databricks auth login --profile e2-demo-field-eng
```

## Run E2B (gemma-4-E2B-it)

```bash
databricks bundle deploy -t dev \
  --var="catalog=meli_demo" --var="schema=default" \
  --var="base_model=google/gemma-4-E2B-it" \
  --var="model_name=gemma_4_e2b_it" \
  --var="endpoint_name=gemma-4-e2b-it" \
  --var="tensor_parallel=1" --var="workload_type=GPU_MEDIUM" \
  --var="hardware_accelerator=GPU_1xA10" \
  --var="max_model_len=16384" --var="gpu_mem_util=0.70" \
  --var="embed_weights=false" --var="tool_call_parser=gemma4"

databricks bundle run -t dev deploy_gemma_serving_endpoint \
  --var="catalog=meli_demo" --var="schema=default" \
  --var="base_model=google/gemma-4-E2B-it" \
  --var="model_name=gemma_4_e2b_it" \
  --var="endpoint_name=gemma-4-e2b-it" \
  --var="tensor_parallel=1" --var="workload_type=GPU_MEDIUM" \
  --var="hardware_accelerator=GPU_1xA10" \
  --var="max_model_len=16384" --var="gpu_mem_util=0.70" \
  --var="embed_weights=false" --var="tool_call_parser=gemma4"
```

## Run E4B (gemma-4-E4B-it)

```bash
databricks bundle deploy -t dev \
  --var="catalog=meli_demo" --var="schema=default" \
  --var="base_model=google/gemma-4-E4B-it" \
  --var="model_name=gemma_4_e4b_it" \
  --var="endpoint_name=gemma-4-e4b-it" \
  --var="tensor_parallel=1" --var="workload_type=GPU_MEDIUM" \
  --var="hardware_accelerator=GPU_1xA10" \
  --var="max_model_len=8192" --var="gpu_mem_util=0.80" \
  --var="embed_weights=false" --var="tool_call_parser=gemma4"

databricks bundle run -t dev deploy_gemma_serving_endpoint \
  --var="catalog=meli_demo" --var="schema=default" \
  --var="base_model=google/gemma-4-E4B-it" \
  --var="model_name=gemma_4_e4b_it" \
  --var="endpoint_name=gemma-4-e4b-it" \
  --var="tensor_parallel=1" --var="workload_type=GPU_MEDIUM" \
  --var="hardware_accelerator=GPU_1xA10" \
  --var="max_model_len=8192" --var="gpu_mem_util=0.80" \
  --var="embed_weights=false" --var="tool_call_parser=gemma4"
```

## Run 12B QAT (gemma-4-12B-it-qat-w4a16-ct)

```bash
databricks bundle deploy -t dev \
  --var="catalog=meli_demo" --var="schema=default" \
  --var="base_model=google/gemma-4-12B-it-qat-w4a16-ct" \
  --var="model_name=gemma_4_12b_it_q4" \
  --var="endpoint_name=gemma-4-12b-it-q4" \
  --var="tensor_parallel=1" --var="workload_type=GPU_MEDIUM" \
  --var="hardware_accelerator=GPU_1xA10" \
  --var="max_model_len=8192" --var="gpu_mem_util=0.80" \
  --var="embed_weights=false" --var="tool_call_parser=gemma4"

databricks bundle run -t dev deploy_gemma_serving_endpoint \
  --var="catalog=meli_demo" --var="schema=default" \
  --var="base_model=google/gemma-4-12B-it-qat-w4a16-ct" \
  --var="model_name=gemma_4_12b_it_q4" \
  --var="endpoint_name=gemma-4-12b-it-q4" \
  --var="tensor_parallel=1" --var="workload_type=GPU_MEDIUM" \
  --var="hardware_accelerator=GPU_1xA10" \
  --var="max_model_len=8192" --var="gpu_mem_util=0.80" \
  --var="embed_weights=false" --var="tool_call_parser=gemma4"
```

## Notes

- **`smoke_test=false` is the default** — GPU containers have kernel FIPS mode blocking the smoke test subprocess. Do not override unless testing the fix.
- **Always `deploy` before `run`** — `bundle run --var` overrides variables at runtime but the job's `base_parameters` are baked at deploy time. Running without a fresh deploy uses the previously baked values.
- **Auth expiry** — if you get `token refresh` errors, re-run `databricks auth login --profile e2-demo-field-eng` and then redeploy.
- **Job ID**: `118697678610579` (dev target, `e2-demo-field-eng` workspace)

## Monitor a running job

```bash
# Watch run status
databricks jobs list-runs --job-id 118697678610579 -p e2-demo-field-eng --output json --limit 1 | \
  python3 -c "import json,sys; r=json.load(sys.stdin)[0]; st=r['state']; print(r['run_id'], st.get('life_cycle_state'), st.get('result_state',''))"

# Open run page in browser
# https://e2-demo-field-eng.cloud.databricks.com/#job/118697678610579
```

## Query endpoints after deployment

```bash
# E2B
databricks api post "/serving-endpoints/gemma-4-e2b-it/invocations" \
  -p e2-demo-field-eng \
  --json '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# E4B
databricks api post "/serving-endpoints/gemma-4-e4b-it/invocations" \
  -p e2-demo-field-eng \
  --json '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# 12B
databricks api post "/serving-endpoints/gemma-4-12b-it-q4/invocations" \
  -p e2-demo-field-eng \
  --json '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```
