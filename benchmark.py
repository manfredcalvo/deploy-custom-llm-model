#!/usr/bin/env python3
"""Concurrency / continuous-batching benchmark for a Databricks Custom LLM Serving endpoint.

Fires N identical chat requests concurrently against the endpoint's OpenAI-compatible
`/invocations` route (streaming) and, for each concurrency level, reports:

  - sys_tps        : aggregate decode throughput  = sum(completion_tokens) / wall_clock
  - p50/p95_lat    : per-request end-to-end latency percentiles (s)
  - p50_ttft_ms    : time to first token (s -> ms), the key interactive-UX metric
  - p50_tpot_ms    : time per output token during decode (ms)  = (total - ttft)/(tokens-1)
  - eff_decode_C   : time-weighted avg number of requests decoding simultaneously
                     (>1 and rising with C == continuous batching; ~1 == serialized)
  - http429        : requests rejected for exceeding provisioned concurrency

Interpretation:
  Continuous batching is working when, as client concurrency C rises, eff_decode_C and
  sys_tps rise while per-request latency stays roughly flat (until GPU saturation).
  Measured on gemma-4-e2b-it (A10, provisioned concurrency 4, 128-tok generations):
  short bursts up to C=32 were all served with no http429 (sys_tps ~62 -> ~1650 tok/s,
  p50 latency 2.0 -> 2.4 s). 429s may still appear under sustained heavy load; retry
  with backoff, or raise min/max_provisioned_concurrency in multiples of 4
  (each step of 4 ~= one more GPU replica).

Usage:
  python benchmark.py                                       # gemma-4-e2b-it, C=1,2,4,8
  python benchmark.py --profile rag --max-tokens 1024       # long-prompt RAG profile
  python benchmark.py --insecure                            # TLS-intercepting proxies

Auth: Databricks CLI profile from --cli-profile / DATABRICKS_CONFIG_PROFILE
(default: e2-demo-field-eng).
"""
import argparse, json, os, ssl, statistics, subprocess, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SHORT_PROMPT = ("Write a long, detailed, multi-paragraph essay about the history of computing, "
                "from the abacus to modern GPUs. Be thorough and verbose.")

# ~4K-token banking agent/RAG prompt (truncated here; pads to a stable large size).
RAG_PROMPT = ("You are a senior financial analyst assistant at a Japanese megabank. Analyze "
              "market trends, evaluate credit risk, summarize regulatory documents, and give "
              "concise recommendations grounded only in the retrieved context, always citing "
              "the relevant section.\n\n[DOC 1: BIS Capital Adequacy 2026] Basel raised the G-SIB "
              "loss-absorbing requirement by 50bp effective Jan 2027...\n[DOC 2: FSA Report 2026] "
              "Domestic G-SIBs held CET1 above 13%...\n[DOC 3: Internal Q3 Memo] Retail CET1 12.4%, "
              "adverse-case trough 9.8% by Q2 2027...\n\nUser question: Given the 50bp surcharge, "
              "does our retail division keep comfortable headroom under the FSA adverse scenario? "
              "Cite figures, judge whether capital actions are warranted before year-end, and give "
              "a step-by-step recommendation.\n\n"
              "Supplementary context: Banking professionals must monitor Basel III/IV capital "
              "adequacy, supervisory stress testing, and climate-related disclosure across credit, "
              "market, operational and liquidity risk. " * 8)


def host_token(cli_profile):
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    host = cfg[cli_profile]["host"].rstrip("/")
    tok = json.loads(subprocess.run(["databricks", "auth", "token", "-p", cli_profile],
                                     capture_output=True, text=True).stdout)["access_token"]
    return host, tok


def make_call(url, tok, prompt, max_tokens, ssl_ctx):
    def one(_=None):
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {tok}",
                                              "Content-Type": "application/json"})
        t0 = time.perf_counter(); start = time.time(); ttft = None; n = 0
        try:
            r = urllib.request.urlopen(req, timeout=600, context=ssl_ctx)
            for line in r:
                line = line.strip()
                if not line or not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    break
                try:
                    d = json.loads(line[6:])
                    ch = d.get("choices") or []
                    if ch and ch[0].get("delta", {}).get("content") and ttft is None:
                        ttft = time.perf_counter() - t0
                    if d.get("usage"):
                        n = d["usage"]["completion_tokens"]
                except Exception:
                    pass
            return {"ok": True, "start": start, "total": time.perf_counter() - t0, "ttft": ttft, "tokens": n}
        except urllib.error.HTTPError as e:
            return {"ok": False, "code": e.code, "total": time.perf_counter() - t0}
        except Exception:
            return {"ok": False, "code": "ERR", "total": time.perf_counter() - t0}
    return one


def eff_decode_concurrency(oks):
    """Time-weighted average number of requests in the decode phase simultaneously."""
    ev = []
    for r in oks:
        if r.get("ttft") is None:
            continue
        ev.append((r["start"] + r["ttft"], +1))
        ev.append((r["start"] + r["total"], -1))
    if not ev:
        return None
    ev.sort()
    cur = 0; tprev = ev[0][0]; weighted = 0.0; total = 0.0
    for t, d in ev:
        weighted += cur * (t - tprev); total += (t - tprev); cur += d; tprev = t
    return round(weighted / total, 2) if total else None


def sweep(one, c):
    t0 = time.perf_counter(); res = []
    with ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(one) for _ in range(c)]
        for f in as_completed(futs):
            res.append(f.result())
    wall = time.perf_counter() - t0
    oks = [r for r in res if r["ok"]]
    lats = sorted(r["total"] for r in oks)
    ttfts = sorted(r["ttft"] for r in oks if r.get("ttft") is not None)
    tpots = [(r["total"] - r["ttft"]) / (r["tokens"] - 1) * 1000
             for r in oks if r.get("ttft") and r["tokens"] > 1]
    n429 = sum(1 for r in res if r.get("code") == 429)
    tot = sum(r["tokens"] for r in oks)
    p = lambda a, q: round(a[min(len(a) - 1, int(q * (len(a) - 1)))], 2) if a else None
    return {"C": c, "ok": len(oks), "http429": n429,
            "wall": round(wall, 2), "sys_tps": round(tot / wall, 1) if wall else None,
            "p50_lat": p(lats, .5), "p95_lat": p(lats, .95),
            "p50_ttft_ms": round(statistics.median(ttfts) * 1000) if ttfts else None,
            "p50_tpot_ms": round(statistics.median(tpots), 1) if tpots else None,
            "mean_tok": round(tot / len(oks)) if oks else None,
            "eff_decode_C": eff_decode_concurrency(oks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="gemma-4-e2b-it")
    ap.add_argument("--profile", choices=["short", "rag"], default="short")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--cli-profile",
                    default=os.environ.get("DATABRICKS_CONFIG_PROFILE", "e2-demo-field-eng"))
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification (needed behind TLS-intercepting proxies)")
    args = ap.parse_args()

    ssl_ctx = None
    if args.insecure:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    host, tok = host_token(args.cli_profile)
    url = f"{host}/serving-endpoints/{args.endpoint}/invocations"
    prompt = SHORT_PROMPT if args.profile == "short" else RAG_PROMPT
    one = make_call(url, tok, prompt, args.max_tokens, ssl_ctx)

    print("warmup:", json.dumps(one()))
    rows = []
    for c in [int(x) for x in args.levels.split(",")]:
        r = sweep(one, c); rows.append(r); print(json.dumps(r)); time.sleep(3)
    print("\nFINAL_JSON=" + json.dumps(rows))


if __name__ == "__main__":
    main()
