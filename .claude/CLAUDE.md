# deploy_custom_llm_model — Claude Code Context

This project deploys open-weight Gemma 4 models to Databricks Custom LLM Serving using vLLM, packaged as a Databricks Asset Bundle with automated benchmarking and agent evaluation.

## Quick start

```bash
# Auth
databricks auth login --profile e2-demo-field-eng

# Run E4B (example)
databricks bundle deploy -t dev --var="base_model=google/gemma-4-E4B-it" --var="model_name=gemma_4_e4b_it" --var="endpoint_name=gemma-4-e4b-it" --var="max_model_len=8192" --var="gpu_mem_util=0.80" --var="embed_weights=false" --var="tool_call_parser=gemma4" --var="catalog=meli_demo" --var="schema=default" --var="tensor_parallel=1" --var="workload_type=GPU_MEDIUM" --var="hardware_accelerator=GPU_1xA10"
databricks bundle run -t dev deploy_gemma_serving_endpoint <same vars>
```

## Documentation index

Detailed docs are split into topic files — read the relevant one before making changes:

| File | When to read |
|---|---|
| [runbook.md](docs/runbook.md) | **Exact copy-paste commands** to deploy and run each model, monitor jobs, query endpoints |
| [architecture.md](docs/architecture.md) | Project structure, data flow, component roles, two-env explanation |
| [models.md](docs/models.md) | Per-model configs (E2B, E4B, 12B QAT), memory math, variables |
| [known-issues.md](docs/known-issues.md) | Active base-image bugs and their fixes (pip conflict, FIPS, artifact limits) |
| [notebooks.md](docs/notebooks.md) | What each notebook does, widgets, key code sections |
| [extending.md](docs/extending.md) | How to add a new model, eval examples, benchmark profiles, workspaces |

## Workspace

- **Profile**: `e2-demo-field-eng` | **Host**: `e2-demo-field-eng.cloud.databricks.com`
- **UC**: `meli_demo.default` | **Job ID**: `118697678610579`
- **Benchmark table**: `meli_demo.default.endpoint_benchmarks`
