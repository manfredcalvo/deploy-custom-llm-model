# Known issues & fixes

## ACTIVE: pip 24.1+ conflict (June 2026+)

**Symptom**: `CalledProcessError: pip install returned non-zero exit status 1` despite "Successfully installed..." in output.

**Cause**: `databricks-serverless-gpu` in the base image requires `mlflow<3.0`. pip 24.1+ treats this conflict as exit code 1 (previously a warning/exit 0).

**Fix** (already in notebook 01): `%pip install "pip<24.1"` as a separate cell BEFORE the main install. pip 23.x treats the conflict as a warning and exits 0.

---

## ACTIVE: Smoke test FIPS failure (June 2026+)

**Symptom**: `ssl.SSLError: [CRYPTO] unknown error (_ssl.c:3076)` in smoke test subprocess, at `SSLContext(PROTOCOL_TLS_CLIENT)`.

**Cause**: Databricks GPU notebook containers have kernel FIPS mode enabled. `PROTOCOL_TLS_CLIENT` fails to initialize in kernel FIPS mode. This happens when `gguf_utils.py → gguf → mistral_common.audio → import requests` runs at vLLM startup — `requests 2.32.3` (base image) creates an SSL context at module load time.

**Fix**: `smoke_test=false` (now the default). The serving containers are NOT affected — they use the `env_pack` snapshot, not the base image.

**Why serving works but smoke test doesn't**: Two different Python environments. Smoke test runs in the GPU notebook container (base image `/databricks/python3/lib/...` on sys.path). Serving runs from the `env_pack` snapshot without the base image packages.

**Not fixable via env vars**: `OPENSSL_FORCE_FIPS_MODE`, `CRYPTOGRAPHY_OPENSSL_NO_LEGACY`, `OPENSSL_CONF=/dev/null`, `REQUESTS_CA_BUNDLE` — none override kernel FIPS. The fix must come from Databricks updating the container configuration.

---

## ACTIVE: sys.path priority (base image wins over ephemeral env)

**Symptom**: `%pip install urllib3<2` installs into ephemeral env but the base image's urllib3 2.x is still imported.

**Cause**: `/databricks/python3/lib/...` comes before the ephemeral env in sys.path on GPU containers. Base image packages cannot be overridden by pip installs.

**Implication**: Only packages NOT present in the base image can be reliably installed into the ephemeral env. For packages present in both, the base image wins.

---

## Artifact upload size limit (~10 GiB per artifact)

`_databricks/model_environment.tar` and `model_version.tar` each have a ~10 GiB limit. The env tar (9.75 GiB after slimming) is proven uploadable.

**Fix for large weights** (E4B 14.9 GiB, 12B 9.6 GiB borderline): use `embed_weights=false` — the serving container downloads weights at startup via `hf download` in the entrypoint. The model_version.tar becomes ~40 KB.

**Slimming the env tar**: `%pip uninstall flashinfer-python flashinfer-cubin flashinfer-jit-cache ray` removes ~2.5 GiB.

---

## opencv FIPS abort (fixed in entrypoint)

**Symptom**: `FATAL FIPS SELFTEST FAILURE` process abort (SIGABRT, exit -6).

**Cause**: `OPENSSL_FORCE_FIPS_MODE=0` set by Databricks login profile scripts. opencv ≥ 4.13 bundles RHEL OpenSSL which treats the variable's *presence* (regardless of value) as "enable FIPS". Self-test fails, process aborts.

**Fix** (in entrypoint): `env -u OPENSSL_FORCE_FIPS_MODE` removes the variable before the vLLM process starts.

---

## vLLM 0.23.0 + fastapi 0.137

`fastapi 0.137` broke vLLM's OpenAI-compatible router. Fix: `"fastapi<0.137"` in the pip install.

---

## FlashInfer JIT + fork worker method

FlashInfer tries to JIT-compile CUDA kernels with ninja/nvcc (not available in serverless). Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` in entrypoint.

TP>1 workers with spawn method hit `libstdc++ CXXABI` mismatch in serving containers. Fix: `VLLM_WORKER_MULTIPROC_METHOD=fork` in entrypoint (harmless for TP=1).
