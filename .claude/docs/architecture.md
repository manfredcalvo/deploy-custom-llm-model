# Architecture

## Component map

```
databricks.yml                  Bundle config: variables, workspace, targets (dev/prod/andreas)
resources/
  deploy_gemma_serving.job.yml  3-task Lakeflow job definition
notebooks/
  01 - Deploy ...               Main: pip install → download weights → log MLflow → register UC → create/update endpoint → query
  02 - Benchmark ...            HTTP benchmark client (imports benchmark.py), writes to endpoint_benchmarks table
  03 - Agent ...                LangGraph ResponsesAgent with local execute_python tool, MLflow eval
benchmark.py                    Local CLI version of the benchmark (same core as notebook 02)
```

## Data flow

```
HuggingFace repo
    ↓ hf download (at container startup, embed_weights=false)
    ↓ OR snapshot_download in notebook (embed_weights=true, weights <10 GiB)
Ephemeral env (GPU notebook container)
    ↓ mlflow.pyfunc.log_model  (ChatModel placeholder + entrypoint metadata)
    ↓ mlflow.register_model(env_pack="databricks_model_serving")
Unity Catalog: meli_demo.default.<model_name>  (model version N)
    ↓ serving_endpoints.create_and_wait / update_config_and_wait
Model Serving endpoint (GPU_MEDIUM = 1×A10)
    ↓ serving container boots: runs entrypoint → hf download → vLLM serve port 8080
Live endpoint (OpenAI-compatible /invocations)
    ↓ Benchmark task (HTTP, streaming)
    ↓ Agent task (LangGraph → execute_python tool)
```

## Two separate Python environments

| Context | Path | Affected by pip installs? |
|---|---|---|
| GPU notebook job container | `/databricks/python3/lib/...` (base) + ephemeral env overlay | Yes (ephemeral env) |
| Model Serving container | `env_pack` snapshot only | No — isolated from base image |

This is why kernel FIPS affects smoke tests but not serving: serving containers don't inherit base image packages.

## Entrypoint string (stored in MLflow model metadata)

For the serving container:
```
env -u OPENSSL_FORCE_FIPS_MODE VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=fork \
  python -u -m vllm.entrypoints.openai.api_server \
  --model <weights_path> --served-model-name gemma --host 0.0.0.0 --port 8080 \
  --dtype bfloat16 --max-model-len <N> --gpu-memory-utilization <N> \
  --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser gemma4
```

For `embed_weights=false`, the serving container entrypoint prepends:
```
env -u OPENSSL_FORCE_FIPS_MODE sh -c "mkdir -p /tmp/.../download && hf download <repo> --local-dir ..." &&
```
