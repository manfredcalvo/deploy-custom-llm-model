# Extending the project

## Add a new model

1. **Check model size and gating**:
   ```bash
   curl -s "https://huggingface.co/api/models/<org>/<model>" | python3 -c "
   import json,sys; d=json.load(sys.stdin)
   params=d.get('safetensors',{}).get('total',0)
   print('gated:', d.get('gated'), '| bf16 GiB:', round(params*2/2**30,1))"
   ```

2. **Determine `embed_weights`**: if bf16 size > 9.5 GiB → `embed_weights=false`

3. **Check vLLM architecture support**: look for `model_type` from `config.json` in vLLM's model registry. If not supported, need a newer vLLM release.

4. **Run**:
   ```bash
   databricks bundle deploy -t dev --var="base_model=<HF_repo>" --var="model_name=<uc_name>" \
     --var="endpoint_name=<ep_name>" --var="embed_weights=<true|false>" \
     --var="max_model_len=<N>" --var="gpu_mem_util=<0.7-0.85>" \
     --var="catalog=meli_demo" --var="schema=default" \
     --var="tensor_parallel=1" --var="workload_type=GPU_MEDIUM" \
     --var="hardware_accelerator=GPU_1xA10" --var="tool_call_parser=gemma4"
   databricks bundle run -t dev deploy_gemma_serving_endpoint <same vars>
   ```

5. **For multi-GPU models** (e.g., 27B): `workload_type=MULTIGPU_MEDIUM tensor_parallel=4 hardware_accelerator=GPU_8xH100`

---

## Add eval examples to notebook 03

Edit the `eval_dataset` list in `notebooks/03 - Agent Implementation Example.py`:
```python
eval_dataset = [
    {
        "inputs": {"input": [{"role": "user", "content": "<your question>"}]},
        "expectations": {"expected_response": "<expected answer>"},
    },
    # ... more examples
]
```
The tool-calling examples should require `execute_python` for computation. Non-tool examples (like capital of France) test the model's direct knowledge.

---

## Add a new benchmark profile

In `benchmark.py` and `notebooks/02 - Benchmark Endpoint.py`, add a new prompt constant:
```python
MY_PROMPT = "Your custom long prompt here..."
```
Then add it to the profile selector in both files. Pass `--profile my_profile` via CLI or `PROFILE=my_profile` widget in the job.

---

## Add a new target workspace

In `databricks.yml`:
```yaml
targets:
  my-workspace:
    mode: development
    workspace:
      host: https://<workspace>.cloud.databricks.com
      profile: <profile-name>
```
Then: `databricks bundle deploy -t my-workspace --var="..."` 

---

## Change vLLM version

In `notebooks/01 - Deploy Gemma Serving Endpoint.py`, update the wheel URL:
```
%pip install --upgrade https://github.com/vllm-project/vllm/releases/download/v<VERSION>/vllm-<VERSION>+cu129-cp38-abi3-manylinux_2_28_x86_64.whl ...
```
Check GitHub releases: `https://api.github.com/repos/vllm-project/vllm/releases?per_page=5`

After a version bump, verify:
- `fastapi<0.137` still needed (check if vllm pins it)
- No new architecture-level issues for target models
- Env tar size stays under 10 GiB (check with the du diagnostic cell)

---

## Run the benchmark CLI locally

```bash
python3 benchmark.py --endpoint gemma-4-e4b-it --levels 1,4,8 --max-tokens 256 --insecure
```
`--insecure` needed on corporate networks with TLS-intercepting proxies.

---

## Query benchmark history

```sql
SELECT endpoint, C, sys_tps, p50_lat_s, p50_ttft_ms, p50_tpot_ms, eff_decode_C, ts
FROM meli_demo.default.endpoint_benchmarks
WHERE ts > current_timestamp() - INTERVAL 7 DAYS
ORDER BY endpoint, C, ts DESC
```
