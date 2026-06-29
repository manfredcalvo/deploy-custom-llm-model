# Notebooks

## 01 - Deploy Gemma Serving Endpoint.py

**Purpose**: Full deployment pipeline — installs deps, downloads weights, runs smoke test (optional), logs MLflow model, registers to UC, creates/updates endpoint, runs a test query.

**Key sections**:
1. `%pip install "pip<24.1"` — downgrades pip to avoid conflict exit code (ACTIVE fix)
2. `%pip install vllm-0.23.0+cu129 ... "fastapi<0.137"` — main install
3. `%pip uninstall flashinfer-* ray` — slims env tar below 10 GiB
4. Widget definitions — all params exposed as job `base_parameters`
5. `snapshot_download` or `hf download` (via entrypoint) — weight download
6. `entrypoint(port)` / `_build_vllm_cmd(port)` — builds the vLLM server command
7. Smoke test Popen — optional, skipped by `smoke_test=false`
8. `mlflow.pyfunc.log_model` — logs placeholder ChatModel + entrypoint metadata
9. `mlflow.register_model(env_pack="databricks_model_serving")` — creates UC model version
10. `serving_endpoints.create_and_wait` / `update_config_and_wait` — endpoint CRUD
11. Query cells — SDK + OpenAI client verification

**Widgets** (= job base_parameters):
`CATALOG, SCHEMA, MODEL_NAME, BASE_MODEL, ENDPOINT_NAME, SMOKE_TEST, TENSOR_PARALLEL, WORKLOAD_TYPE, MAX_MODEL_LEN, GPU_MEM_UTIL, EMBED_WEIGHTS, TOOL_CALL_PARSER, EXTRA_VLLM_ARGS`

**Entrypoint endpoint config** (fixed concurrency, no scale-to-zero — required for entrypoint-based endpoints):
```python
ServedEntityInput(
    entity_name=UC_MODEL_NAME, entity_version=str(version),
    workload_type=WORKLOAD_TYPE,
    min_provisioned_concurrency=4, max_provisioned_concurrency=4,
    scale_to_zero_enabled=False,
)
```

---

## 02 - Benchmark Endpoint.py

**Purpose**: Fires concurrent streaming requests and measures TTFT, TPOT, sys_tps, eff_decode_C. Results displayed via `display()` and appended to `meli_demo.default.endpoint_benchmarks`.

**Imports** `benchmark.py` from the parent dir (same core logic as the CLI tool).

**Widgets**: `ENDPOINT_NAME, LEVELS, MAX_TOKENS, PROFILE (short|rag), RESULTS_TABLE`

---

## 03 - Agent Implementation Example.py

**Purpose**: LangGraph tool-calling agent using the deployed endpoint as its LLM, evaluated with MLflow GenAI scorers.

**Key details**:
- Uses `ChatDatabricks(endpoint=ENDPOINT_NAME)` — endpoint selected via widget
- Tool: local `execute_python(@tool)` — NOT the managed MCP UC tool (system.ai.python_exec was not working)
- MLflow eval dataset: 5 examples (Fibonacci, addition, prime sum, 20!, capital of France)
- Pinned libs: `mlflow==3.11.1 langgraph==1.1.8 databricks-langchain==0.19.0` — critical, mlflow 3.13.x has a trace-linking bug in `genai.evaluate` (mlflow#18284)
- `mlflow.start_run(run_name="agent_eval")` wraps eval for correct trace association
- Pre-deployment validation (`mlflow.models.predict(env_manager="uv")`) works in job runs

**Widget**: `ENDPOINT_NAME` — wired from `${var.endpoint_name}` in the job YAML
