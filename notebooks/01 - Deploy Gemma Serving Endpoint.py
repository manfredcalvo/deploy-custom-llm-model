# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy `google/gemma-4-E2B-it` to a Databricks Model Serving endpoint
# MAGIC
# MAGIC This notebook follows the [Custom LLM serving starter notebook](https://docs.databricks.com/aws/en/notebooks/source/machine-learning/serve-custom-llms-starter.html) strategy:
# MAGIC
# MAGIC 1. Download the model weights from Hugging Face to local disk.
# MAGIC 2. (Optional) Smoke-test the model locally with a vLLM OpenAI-compatible server on the notebook GPU.
# MAGIC 3. Log the model to MLflow with a custom vLLM entrypoint (`task: llm/v1/chat`).
# MAGIC 4. Register it to Unity Catalog with `env_pack="databricks_model_serving"`.
# MAGIC 5. Create (or update) a GPU Model Serving endpoint and query it.
# MAGIC
# MAGIC **Requirements**
# MAGIC - Run on **Serverless GPU compute** (A10 is sufficient for the ~2B-effective-parameter E2B model). As a job, use `compute.hardware_accelerator: GPU_1xA10`.
# MAGIC - The Unity Catalog destination (`CATALOG.SCHEMA.MODEL_NAME`) must be writable by you.
# MAGIC
# MAGIC [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) is Apache-2.0 licensed and not gated, so no Hugging Face token is required.

# COMMAND ----------

# MAGIC %md
# MAGIC Serving requirements. For serverless GPU jobs, dependencies must be installed in the
# MAGIC notebook (the Environments panel is not supported for serverless GPU scheduled jobs).
# MAGIC
# MAGIC The Model Serving GPU containers run an NVIDIA driver at CUDA 12.4, so vLLM's default
# MAGIC PyPI wheel (built for CUDA 13) crashes there with "NVIDIA driver too old". Install the
# MAGIC `+cu129` variant wheel and CUDA 12.9 PyTorch instead — CUDA 12.x runtimes work on the
# MAGIC 12.4 driver via minor-version compatibility, and `env_pack` snapshots this exact
# MAGIC environment into the serving container.

# COMMAND ----------

# pip 24.1+ treats conflicts between base-image packages and ephemeral-env packages as
# errors (exit 1) instead of warnings (exit 0). Downgrade pip to <24.1 first so the
# conflict with databricks-serverless-gpu's mlflow<3.0 requirement is only a warning.
# MAGIC %pip install "pip<24.1"
# MAGIC %pip install --upgrade https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl "transformers>=5.5.0" openai==2.17.0 mlflow==3.12.0 hf_transfer==0.1.9 databricks-sdk "fastapi<0.137" --extra-index-url https://download.pytorch.org/whl/cu129

# COMMAND ----------

# MAGIC %md
# MAGIC Slim the environment before it gets snapshotted by `env_pack`: the serving artifact
# MAGIC upload has a hard size limit (a ~11.4 GiB environment tar failed with "Model artifact
# MAGIC upload failed"; ~8.3 GiB worked). FlashInfer is optional for vLLM — its sampler is
# MAGIC disabled in our entrypoint and it can't JIT on the serving image anyway — and its
# MAGIC prebuilt kernels are the largest removable chunk. Ray is only needed for multi-node
# MAGIC executors. Uninstalling BEFORE the smoke test means we validate the exact slimmed
# MAGIC environment that serving will run.

# COMMAND ----------

# MAGIC %pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache ray
# MAGIC %restart_python

# COMMAND ----------

# Environment size diagnostic + trim. The env tar must stay well under ~10 GiB; only
# inert files are removed (bytecode caches, AMD GPU backend, static libs), so the
# smoke-tested environment is functionally identical to what serving runs.
import site, subprocess

sp = site.getsitepackages()[0]

def sh(cmd):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True).stdout

print("site-packages:", sp)
print("--- top 25 packages by size (MB) ---")
print(sh(f"du -sm {sp}/* 2>/dev/null | sort -rn | head -25"))

# NOTE: do NOT remove triton/backends/amd — deleting any backend dir breaks Triton's
# backend discovery entirely, and gemma4 requires the TRITON_ATTN backend.
sh(f"find {sp} -name '__pycache__' -type d -prune -exec rm -rf {{}} + 2>/dev/null")
sh(f"find {sp}/nvidia -name '*_static*' -delete 2>/dev/null; find {sp} -name '*.a' -delete 2>/dev/null")
sh("pip cache purge >/dev/null 2>&1 || true; rm -rf ~/.cache/pip")

print("--- total after trim ---")
print(sh(f"du -sh {sp}"))

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC All parameters are exposed as widgets so this notebook can run interactively or as a
# MAGIC job task with `base_parameters` (see `resources/deploy_gemma_serving.job.yml`).

# COMMAND ----------

dbutils.widgets.text("CATALOG", "main")
dbutils.widgets.text("SCHEMA", "default")
dbutils.widgets.text("MODEL_NAME", "gemma_4_e2b_it")
dbutils.widgets.text("BASE_MODEL", "google/gemma-4-E2B-it")
dbutils.widgets.text("ENDPOINT_NAME", "gemma-4-e2b-it")
dbutils.widgets.text("SMOKE_TEST", "true")  # Set "false" to skip the local vLLM test.
# Multi-GPU serving: for models that don't fit one A10 (e.g. gemma-4-12B-it, ~24 GB bf16),
# use TENSOR_PARALLEL=4 with WORKLOAD_TYPE=MULTIGPU_MEDIUM (4xA10) and run the job on
# GPU_8xH100 so the smoke test has enough GPUs/VRAM.
dbutils.widgets.text("TENSOR_PARALLEL", "1")
dbutils.widgets.text("WORKLOAD_TYPE", "GPU_MEDIUM")
# Memory knobs: larger models on a single A10 need a smaller context window and a higher
# memory utilization (e.g. E4B at ~14.9 GiB bf16: MAX_MODEL_LEN=8192, GPU_MEM_UTIL=0.80).
dbutils.widgets.text("MAX_MODEL_LEN", "16384")
dbutils.widgets.text("GPU_MEM_UTIL", "0.70")
# The serving upload has a ~10 GiB per-artifact limit. For models whose weights exceed it
# (e.g. E4B at ~14.9 GiB), set EMBED_WEIGHTS=false: the weights are NOT logged as MLflow
# artifacts and the serving container downloads them from Hugging Face at startup instead.
dbutils.widgets.text("EMBED_WEIGHTS", "true")
# Native OpenAI tool calling: vLLM needs --enable-auto-tool-choice --tool-call-parser to
# convert the model's tool-call syntax into structured tool_calls. Empty string disables.
dbutils.widgets.text("TOOL_CALL_PARSER", "gemma4")
# Extra vLLM server flags, space-separated (e.g. "--kv-cache-dtype fp8" to halve KV
# memory on the A10, or "--trust-remote-code" for repos shipping custom code).
dbutils.widgets.text("EXTRA_VLLM_ARGS", "")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
MODEL_NAME = dbutils.widgets.get("MODEL_NAME")
MODEL_REPO_ID = dbutils.widgets.get("BASE_MODEL")  # Hugging Face model to download.
ENDPOINT_NAME = dbutils.widgets.get("ENDPOINT_NAME")
SMOKE_TEST = dbutils.widgets.get("SMOKE_TEST").strip().lower() == "true"
TENSOR_PARALLEL = int(dbutils.widgets.get("TENSOR_PARALLEL"))
EMBED_WEIGHTS = dbutils.widgets.get("EMBED_WEIGHTS").strip().lower() == "true"
TOOL_CALL_PARSER = dbutils.widgets.get("TOOL_CALL_PARSER").strip()
EXTRA_VLLM_ARGS = dbutils.widgets.get("EXTRA_VLLM_ARGS").split()

# Unity Catalog destination for the registered model.
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

ARTIFACTS_PATH = "gemma"  # Local directory the weights are downloaded to.
SERVED_MODEL_NAME = "gemma"  # Name vLLM exposes the model under.

# vLLM tuning. Gemma works best with bfloat16 (supported on A10/H100; NOT on T4 —
# GPU_SMALL cannot serve these models).
DTYPE = "bfloat16"
MAX_MODEL_LEN = int(dbutils.widgets.get("MAX_MODEL_LEN"))
# Keep headroom below vLLM's startup free-memory check: the serving container has
# GPU overhead that a bare notebook GPU doesn't, so 0.85 fails there.
GPU_MEMORY_UTILIZATION = float(dbutils.widgets.get("GPU_MEM_UTIL"))

# Allowlisted ports for Serverless GPU notebooks are 3000-3999. Model Serving requires 8080.
LOCAL_PORT = 3080
SERVING_PORT = 8080

# Serving endpoint configuration. Names share a workspace, so the name must be unique.
from databricks.sdk.service.serving import ServingModelWorkloadType

WORKLOAD_TYPE = ServingModelWorkloadType(dbutils.widgets.get("WORKLOAD_TYPE"))
# Entrypoint-based custom LLM endpoints don't support autoscaling ranges (workload_size)
# or scale-to-zero; they need a fixed provisioned concurrency, in multiples of 4
# (4 = one GPU replica).
PROVISIONED_CONCURRENCY = 4
SCALE_TO_ZERO_ENABLED = False

print(f"Deploying {MODEL_REPO_ID} -> {UC_MODEL_NAME} -> endpoint '{ENDPOINT_NAME}'")

# COMMAND ----------

# Set working directory to local disk (/Workspace doesn't support large files).
import os, tempfile

workdir = tempfile.mkdtemp()
os.chdir(workdir)

# Speed up the weights download. The repo is ungated, so no HF token is needed.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Avoid FATAL FIPS SELFTEST FAILURE from opencv>=4.13's bundled OpenSSL (see entrypoint()).
os.environ.pop("OPENSSL_FORCE_FIPS_MODE", None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download the model

# COMMAND ----------

from huggingface_hub import snapshot_download

if EMBED_WEIGHTS:
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=ARTIFACTS_PATH,
    )
else:
    print("EMBED_WEIGHTS=false: the entrypoint downloads the weights at startup "
          "(both in the smoke test below and in the serving container); skipping.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the vLLM entrypoint
# MAGIC
# MAGIC Model Serving runs this command inside the endpoint container. The same command
# MAGIC (on a notebook-allowlisted port) is used for the local smoke test below.

# COMMAND ----------

# Where the entrypoint downloads weights when they are not embedded as artifacts.
WEIGHTS_DOWNLOAD_DIR = os.path.join(workdir, "model_weights")

def entrypoint(port: int) -> str:
    vllm_cmd = " ".join([
        # Databricks images set OPENSSL_FORCE_FIPS_MODE=0, which the RHEL OpenSSL bundled in
        # opencv>=4.13 (a vLLM dependency) treats as "enable FIPS", aborting the process with
        # FATAL FIPS SELFTEST FAILURE. https://github.com/opencv/opencv-python/issues/1184
        "env", "-u", "OPENSSL_FORCE_FIPS_MODE",
        "CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1",
        # Prevent OpenSSL from loading /etc/ssl/openssl.cnf which activates the FIPS provider
        # on FIPS-enabled kernels, causing SSLContext(PROTOCOL_TLS_CLIENT) to fail.
        "OPENSSL_CONF=/dev/null",
        # FlashInfer JIT-compiles sampling kernels with ninja/nvcc, which is not available
        # on the serverless image. Fall back to the PyTorch-native sampler.
        "VLLM_USE_FLASHINFER_SAMPLER=0",
        # With TP>1, spawn-based worker subprocesses hit a libstdc++ CXXABI mismatch in
        # the Model Serving container; fork avoids it (harmless for TP=1).
        "VLLM_WORKER_MULTIPROC_METHOD=fork",
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", ARTIFACTS_PATH if EMBED_WEIGHTS else WEIGHTS_DOWNLOAD_DIR,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--tensor-parallel-size", str(TENSOR_PARALLEL),
        # "--enforce-eager",  # Faster server startup at the cost of inference performance.
    ] + (
        ["--enable-auto-tool-choice", "--tool-call-parser", TOOL_CALL_PARSER]
        if TOOL_CALL_PARSER else []
    ) + EXTRA_VLLM_ARGS)
    if EMBED_WEIGHTS:
        return vllm_cmd
    download_cmd = (
        f"mkdir -p {WEIGHTS_DOWNLOAD_DIR}/.cache/huggingface/download"
        f" && hf download {MODEL_REPO_ID} --local-dir {WEIGHTS_DOWNLOAD_DIR}"
    )
    return f"bash -c '{download_cmd} && {vllm_cmd}'"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke-test the model in the notebook (optional)
# MAGIC
# MAGIC Starts vLLM on the notebook GPU, waits until it is ready (with a hard timeout so a
# MAGIC scheduled job fails fast instead of hanging), sends one chat request, then stops it.

# COMMAND ----------

import subprocess
import time
import requests

if SMOKE_TEST:
    log = open("process.log", "w")
    proc = subprocess.Popen(
        ["bash", "-lc", entrypoint(LOCAL_PORT)],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    # Wait for vLLM to come up, streaming its startup log into the cell output so
    # progress (engine init, checkpoint shard loading, CUDA graphs) is visible live.
    # Fails fast if the process crashes instead of waiting out the full deadline.
    deadline = time.time() + 45 * 60
    ready = False
    with open("process.log") as lf:
        while time.time() < deadline:
            for line in lf.readlines():
                print(line, end="", flush=True)
            try:
                if requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=5).status_code == 200:
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            if proc.poll() is not None:
                print(lf.read(), flush=True)
                raise RuntimeError(
                    f"vLLM exited with code {proc.returncode} before becoming ready (see log above)."
                )
            time.sleep(5)
        print(lf.read(), flush=True)

    if not ready:
        raise RuntimeError("vLLM did not become ready within 45 minutes (see log above).")

    resp = requests.post(
        f"http://localhost:{LOCAL_PORT}/invocations",
        json={"messages": [{"role": "user", "content": "Hello! Reply in one short sentence."}]},
    )
    resp.raise_for_status()
    print(resp.json()["choices"][0]["message"]["content"])

    # Validate native tool calling before anything gets registered.
    if TOOL_CALL_PARSER:
        tr = requests.post(
            f"http://localhost:{LOCAL_PORT}/invocations",
            json={
                "messages": [{"role": "user", "content": "What is 847 plus 2956? Use the add tool."}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers and return their sum",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }],
                "tool_choice": "auto",
                "max_tokens": 200,
            },
        )
        tr.raise_for_status()
        msg = tr.json()["choices"][0]["message"]
        print("tool_calls:", msg.get("tool_calls"))
        assert msg.get("tool_calls"), (
            "Tool-call parser is enabled but the response contains no structured tool_calls; "
            f"content was: {msg.get('content')!r}"
        )
else:
    print("SMOKE_TEST=false, skipping local vLLM test.")

# COMMAND ----------

# Stop the local server and free the GPU before logging/registering.
if SMOKE_TEST:
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"])
    time.sleep(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model with the custom entrypoint

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

# Required placeholder. Serving runs the entrypoint, not python_model.predict.
class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})

model_info = mlflow.pyfunc.log_model(
    name=SERVED_MODEL_NAME,
    python_model=LLMModel(),
    # With EMBED_WEIGHTS=false the entrypoint downloads the weights at startup, so the
    # logged model stays tiny (placeholder only) and under the artifact upload limit.
    artifacts={"model_dir": ARTIFACTS_PATH} if EMBED_WEIGHTS else None,
    metadata={
        "task": "llm/v1/chat",
        "entrypoint": entrypoint(SERVING_PORT),
    },
    extra_pip_requirements=[
        "mlflow==3.12.0",
    ],
)
model_info.model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog

# COMMAND ----------

# env_pack is required. Custom LLM Serving depends on Serverless Optimized Deployments.
# The endpoint will not work without it.
# https://docs.databricks.com/aws/en/machine-learning/model-serving/serverless-optimized-deployments
model_version = mlflow.register_model(
    model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving"
)
print(f"Registered {UC_MODEL_NAME} version {model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create (or update) the serving endpoint

# COMMAND ----------

from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

served_entity = ServedEntityInput(
    entity_name=UC_MODEL_NAME,
    entity_version=str(model_version.version),
    workload_type=WORKLOAD_TYPE,
    min_provisioned_concurrency=PROVISIONED_CONCURRENCY,
    max_provisioned_concurrency=PROVISIONED_CONCURRENCY,
    scale_to_zero_enabled=SCALE_TO_ZERO_ENABLED,
)

w = WorkspaceClient()
try:
    w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' exists, updating it to version {model_version.version}...")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[served_entity],
        timeout=timedelta(minutes=60),
    )
except NotFound:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        # Newer databricks-sdk requires `name` inside the config as well.
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),
        timeout=timedelta(minutes=60),
    )

print(f"Endpoint '{ENDPOINT_NAME}' is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the ready endpoint

# COMMAND ----------

# Query using the Databricks SDK.
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

resp = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi, what model are you?")],
)

print(resp.choices[0].message.content)

# COMMAND ----------

# Query using the OpenAI client.
from openai import OpenAI

DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
)

response = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[
        {"role": "user", "content": "Hello"},
    ],
)
print(response.choices[0].message.content)
