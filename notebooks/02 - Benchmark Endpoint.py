# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark a Custom LLM Serving endpoint
# MAGIC
# MAGIC Runs the project's concurrency / continuous-batching benchmark (imported from
# MAGIC `../benchmark.py`, synced by the bundle) against a serving endpoint and reports,
# MAGIC per concurrency level:
# MAGIC
# MAGIC - `sys_tps` — aggregate decode throughput (sum of completion tokens / wall clock)
# MAGIC - `p50_lat` / `p95_lat` — per-request end-to-end latency percentiles (s)
# MAGIC - `p50_ttft_ms` — time to first token (the interactive-UX metric)
# MAGIC - `p50_tpot_ms` — time per output token during decode
# MAGIC - `eff_decode_C` — time-weighted avg requests decoding simultaneously (tracking C
# MAGIC   proves continuous batching; ~1 means requests are serialized)
# MAGIC - `http429` — requests rejected at the admission limit
# MAGIC
# MAGIC Results are displayed as a table below and, if `RESULTS_TABLE` is set, appended to
# MAGIC that Delta table for cross-run history. Runs on serverless CPU — the benchmark is
# MAGIC a pure stdlib HTTP client; the GPU work happens on the endpoint.

# COMMAND ----------

dbutils.widgets.text("ENDPOINT_NAME", "gemma-4-e2b-it")
dbutils.widgets.text("LEVELS", "1,2,4,8")
dbutils.widgets.text("MAX_TOKENS", "256")
dbutils.widgets.text("PROFILE", "short")  # short | rag
dbutils.widgets.text("RESULTS_TABLE", "")  # catalog.schema.table; empty = don't persist

ENDPOINT_NAME = dbutils.widgets.get("ENDPOINT_NAME")
LEVELS = [int(x) for x in dbutils.widgets.get("LEVELS").split(",")]
MAX_TOKENS = int(dbutils.widgets.get("MAX_TOKENS"))
PROFILE = dbutils.widgets.get("PROFILE").strip().lower()
RESULTS_TABLE = dbutils.widgets.get("RESULTS_TABLE").strip()

print(f"Benchmarking '{ENDPOINT_NAME}' | levels={LEVELS} | {MAX_TOKENS} tok | profile={PROFILE}")

# COMMAND ----------

# Reuse the benchmark core from ../benchmark.py (synced to the workspace files root by
# `databricks bundle deploy`); a WORKSPACE-source notebook's cwd is its own folder.
import sys, os

sys.path.append(os.path.abspath(".."))
from benchmark import make_call, sweep, SHORT_PROMPT, RAG_PROMPT

# COMMAND ----------

# Authenticate with the notebook's own context — no CLI profile or proxy involved.
import json, time

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
url = f"{ctx.apiUrl().get()}/serving-endpoints/{ENDPOINT_NAME}/invocations"
tok = ctx.apiToken().get()

prompt = SHORT_PROMPT if PROFILE == "short" else RAG_PROMPT
one = make_call(url, tok, prompt, MAX_TOKENS, None)

warmup = one()
assert warmup.get("ok"), f"Warmup request failed: {warmup}"
print("warmup:", json.dumps(warmup))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the concurrency sweep

# COMMAND ----------

rows = []
for c in LEVELS:
    r = sweep(one, c)
    rows.append(r)
    print(json.dumps(r))
    time.sleep(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

from datetime import datetime, timezone

ts = datetime.now(timezone.utc)

def as_row(r):
    f = lambda k: float(r[k]) if r.get(k) is not None else None
    i = lambda k: int(r[k]) if r.get(k) is not None else None
    return {
        "endpoint": ENDPOINT_NAME, "ts": ts, "profile": PROFILE, "max_tokens": MAX_TOKENS,
        "C": i("C"), "ok": i("ok"), "http429": i("http429"),
        "wall_s": f("wall"), "sys_tps": f("sys_tps"),
        "p50_lat_s": f("p50_lat"), "p95_lat_s": f("p95_lat"),
        "p50_ttft_ms": f("p50_ttft_ms"), "p50_tpot_ms": f("p50_tpot_ms"),
        "mean_tok": i("mean_tok"), "eff_decode_C": f("eff_decode_C"),
    }

results_df = spark.createDataFrame([as_row(r) for r in rows])
display(results_df.orderBy("C"))

# COMMAND ----------

if RESULTS_TABLE:
    # Best effort: the displayed table above is the primary report; persistence failures
    # (permissions, metastore hiccups) should not fail the deployment job.
    try:
        results_df.write.mode("append").option("mergeSchema", "true").saveAsTable(RESULTS_TABLE)
        print(f"Appended {results_df.count()} rows to {RESULTS_TABLE}")
    except Exception as e:
        print(f"WARNING: could not persist results to {RESULTS_TABLE}: {e}")
else:
    print("RESULTS_TABLE not set; skipping persistence.")
