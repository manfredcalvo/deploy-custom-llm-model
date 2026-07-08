#!/usr/bin/env python3
"""Document classification benchmark for Databricks Model Serving endpoints.

Fires N concurrent classification requests with exponential backoff and reports
latency, throughput, retry stats, and per-category accuracy.

Usage:
    python classification_benchmark.py --endpoint gemma-4-12b-it-q4 --users 100
    python classification_benchmark.py --endpoint gemma-4-e4b-it --users 500 --max-tokens 200
    python classification_benchmark.py --endpoint my-endpoint --users 200 --insecure --profile e2-demo-field-eng
    python classification_benchmark.py --endpoint my-endpoint --users 100 --no-backoff

Auth: Databricks CLI profile (--profile), default from DATABRICKS_CONFIG_PROFILE env var.
"""

import argparse
import json
import os
import random
import re
import ssl
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Classification system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert business document classifier for a financial services firm. \
Classify incoming documents into exactly one of these 10 categories:

1. LOAN_APPLICATION - new loan requests with borrower info, amount, purpose, collateral, repayment schedule
2. LOAN_SERVICING - existing loan accounts (has account number); payments, payoffs, modifications, forbearance
3. FRAUD_ALERT - PRIORITY: any unauthorized transactions, suspicious activity, identity theft, disputes → always this
4. COMPLIANCE_REPORT - regulatory docs: BSA, AML, KYC, FINRA, SEC, SAR filings, audit responses
5. CUSTOMER_COMPLAINT - customer dissatisfaction, wrong charges, poor service, escalation threats
6. INVESTMENT_REQUEST - buy/sell securities, portfolio changes, ticker symbols, CUSIP numbers
7. ACCOUNT_STATEMENT - periodic summaries with statement period, beginning/ending balance, transaction list
8. WIRE_TRANSFER - fund transfer instructions with sender + recipient bank details, SWIFT, IBAN, routing numbers
9. INSURANCE_CLAIM - claims with policy number, claim number, loss date, damage description, adjuster
10. LEGAL_DOCUMENT - PRIORITY: legal instruments, court orders, subpoenas, attorney letters, legal citations

Priority rules:
- FRAUD_ALERT overrides everything if fraud/unauthorized/identity-theft signals present
- LEGAL_DOCUMENT overrides all except FRAUD_ALERT if from law firm or contains legal citations

Respond ONLY with this JSON (no markdown, no code blocks):
{"category": "<NAME>", "confidence": <0.0-1.0>, "key_signals": ["signal1", "signal2"], "notes": "<max 30 words>"}"""

# ---------------------------------------------------------------------------
# Sample documents (10 types, one per category)
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    ("LOAN_APPLICATION",
     "Dear Sir/Madam, I am writing to request a mortgage loan of $450,000 to purchase a primary "
     "residence at 123 Oak Street. I have attached proof of income showing $120,000 annual salary, "
     "and I am proposing the property itself as collateral with a 30-year repayment schedule."),
    ("FRAUD_ALERT",
     "URGENT: I noticed three unauthorized transactions on my account yesterday totaling $2,340. "
     "Someone has stolen my identity and is making purchases. I did not authorize any of these "
     "charges. This is fraud and I need immediate action to dispute these transactions."),
    ("WIRE_TRANSFER",
     "Account #4521-8893: Please process a wire transfer of $75,000 USD to beneficiary Deutsche "
     "Bank Frankfurt, IBAN DE89370400440532013000, SWIFT code DEUTDEDB, for our quarterly vendor "
     "payment to Müller & Associates GmbH. Sender: Acme Corp routing 021000021."),
    ("LEGAL_DOCUMENT",
     "This letter constitutes formal notice pursuant to 15 U.S.C. § 1692g that your client, "
     "Plaintiff Anderson Holdings LLC, hereby demands full payment of the outstanding judgment "
     "entered in Case No. 2024-CV-08834. Failure to respond within 30 days will result in "
     "enforcement proceedings."),
    ("CUSTOMER_COMPLAINT",
     "I am extremely disappointed with your service. Last month you charged me $47 twice for the "
     "same transaction and despite calling three times nobody has resolved this. I am considering "
     "filing a complaint with the CFPB if this $47 overcharge is not refunded within 5 business days."),
    ("COMPLIANCE_REPORT",
     "Quarterly Compliance Report - Q3 2024. In accordance with BSA/AML requirements under "
     "31 CFR 1020.320, we are filing this Suspicious Activity Report for the period July 1 - "
     "September 30. Total SARs filed: 23. KYC reviews completed: 1,847."),
    ("INVESTMENT_REQUEST",
     "Please sell 500 shares of AAPL at market price and use the proceeds to purchase 200 units "
     "of the Vanguard S&P 500 ETF (ticker: VOO, CUSIP: 922908363). Additionally, please liquidate "
     "my position in TSLA bonds maturing 2026."),
    ("ACCOUNT_STATEMENT",
     "Statement Period: October 1 - October 31, 2024. Beginning Balance: $124,532.18. "
     "Total Deposits: $8,500.00. Total Withdrawals: $3,247.65. Service Charges: $12.00. "
     "Interest Earned: $89.43. Ending Balance: $129,861.96."),
    ("INSURANCE_CLAIM",
     "Claim #INS-2024-98234: I am submitting this insurance claim following water damage to my "
     "property on November 3rd. Policy number HO-789234. The adjuster visited and estimated "
     "$28,000 in structural damage to the basement."),
    ("LOAN_SERVICING",
     "Re: Account #LN-2024-55123. We received your request for a loan modification. Current "
     "balance $287,450, payoff valid through December 31. We can offer a forbearance plan "
     "reducing your monthly payment. Please sign the attached modification agreement."),
]


def get_documents(n: int) -> list[tuple[str, str]]:
    """Return n (expected_category, document) pairs by cycling through samples."""
    return [SAMPLE_DOCUMENTS[i % len(SAMPLE_DOCUMENTS)] for i in range(n)]


def get_token(profile: str) -> tuple[str, str]:
    """Return (host, token) from the Databricks CLI."""
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    host = cfg[profile]["host"].rstrip("/")
    result = subprocess.run(
        ["databricks", "auth", "token", "-p", profile],
        capture_output=True, text=True
    )
    tok = json.loads(result.stdout)["access_token"]
    return host, tok


def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def classify_with_backoff(
    idx: int,
    doc_pair: tuple[str, str],
    url: str,
    token: str,
    ssl_ctx: ssl.SSLContext | None,
    max_tokens: int,
    max_retries: int,
    base_delay: float,
    use_backoff: bool,
    system_prompt: str,
) -> dict:
    expected, doc = doc_pair
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify this business document:\n\n{doc}"},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req_data = json.dumps(body).encode()
    retries, total_wait = 0, 0.0
    t0 = time.perf_counter()

    while True:
        req = urllib.request.Request(
            url, data=req_data,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120, context=ssl_ctx) as r:
                d = json.loads(r.read())
            elapsed = time.perf_counter() - t0
            parsed = extract_json(d["choices"][0]["message"]["content"])
            category = parsed.get("category", "?") if parsed else "PARSE_ERR"
            confidence = parsed.get("confidence", 0.0) if parsed else 0.0
            return {
                "ok": True, "idx": idx, "elapsed": elapsed,
                "retries": retries, "total_wait": total_wait,
                "expected": expected, "category": category, "confidence": confidence,
                "correct": category == expected,
                "prompt_tokens": d.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": d.get("usage", {}).get("completion_tokens", 0),
            }
        except urllib.error.HTTPError as e:
            if e.code == 429 and use_backoff and retries < max_retries:
                delay = base_delay * (2 ** retries) + random.uniform(0, 1.0)
                total_wait += delay
                time.sleep(delay)
                retries += 1
            else:
                return {
                    "ok": False, "idx": idx, "code": e.code,
                    "elapsed": time.perf_counter() - t0,
                    "retries": retries, "total_wait": total_wait,
                    "expected": expected,
                }
        except Exception as exc:
            return {
                "ok": False, "idx": idx, "code": "ERR",
                "elapsed": time.perf_counter() - t0,
                "retries": retries, "total_wait": total_wait,
                "expected": expected, "err": str(exc)[:80],
            }

        if retries > max_retries:
            return {
                "ok": False, "idx": idx, "code": "MAX_RETRIES",
                "elapsed": time.perf_counter() - t0,
                "retries": retries, "total_wait": total_wait,
                "expected": expected,
            }


def pct(a, b):
    return a[int(len(a) * b)] if a else 0


def main():
    import urllib.request  # needed inside worker

    parser = argparse.ArgumentParser(description="Document classification load test")
    parser.add_argument("--endpoint", required=True, help="Serving endpoint name")
    parser.add_argument("--users", type=int, default=100, help="Number of concurrent users")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max output tokens per request")
    parser.add_argument("--max-retries", type=int, default=8, help="Max retry attempts per request")
    parser.add_argument("--base-delay", type=float, default=0.5, help="Base backoff delay in seconds")
    parser.add_argument("--no-backoff", action="store_true", help="Disable retry backoff (fail on 429)")
    parser.add_argument("--insecure", action="store_true", help="Skip TLS verification (for proxies)")
    parser.add_argument("--profile", default=os.environ.get("DATABRICKS_CONFIG_PROFILE", "e2-demo-field-eng"),
                        help="Databricks CLI profile name")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    # Auth
    host, token = get_token(args.profile)
    url = f"{host}/serving-endpoints/{args.endpoint}/invocations"

    # SSL context
    ssl_ctx = None
    if args.insecure:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    documents = get_documents(args.users)
    use_backoff = not args.no_backoff

    if args.output == "text":
        strategy = f"max {args.max_retries} retries, base={args.base_delay}s" if use_backoff else "no backoff"
        print(f"Endpoint:  {args.endpoint}")
        print(f"Users:     {args.users}")
        print(f"Strategy:  {'Exponential backoff (' + strategy + ')' if use_backoff else 'No retry'}")
        print(f"Profile:   {args.profile}")
        print()

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.users) as ex:
        results = list(ex.map(
            lambda p: classify_with_backoff(
                p[0], p[1], url, token, ssl_ctx,
                args.max_tokens, args.max_retries, args.base_delay, use_backoff, SYSTEM_PROMPT
            ),
            enumerate(documents)
        ))
    wall = time.perf_counter() - t_start

    oks = [r for r in results if r.get("ok")]
    fails = [r for r in results if not r.get("ok")]
    retried = [r for r in oks if r.get("retries", 0) > 0]
    lats = sorted(r["elapsed"] for r in oks)
    total_retries = sum(r.get("retries", 0) for r in results)
    correct = sum(1 for r in oks if r.get("correct"))
    n429 = sum(1 for r in fails if r.get("code") == 429)
    n_max_retries = sum(1 for r in fails if r.get("code") == "MAX_RETRIES")

    cats = {}
    for r in oks:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    if args.output == "json":
        print(json.dumps({
            "endpoint": args.endpoint, "users": args.users, "backoff": use_backoff,
            "succeeded": len(oks), "failed": len(fails),
            "failed_429": n429, "failed_max_retries": n_max_retries,
            "needed_retry": len(retried), "total_retry_attempts": total_retries,
            "accuracy": correct / max(len(oks), 1),
            "wall_s": round(wall, 2),
            "p50_lat_s": round(pct(lats, 0.5), 2),
            "p95_lat_s": round(pct(lats, 0.95), 2),
            "max_lat_s": round(lats[-1], 2) if lats else 0,
            "avg_prompt_tokens": sum(r["prompt_tokens"] for r in oks) // max(len(oks), 1),
            "avg_wait_retried_s": round(sum(r["total_wait"] for r in retried) / max(len(retried), 1), 2),
            "categories": cats,
        }, indent=2))
        return

    # Text output
    print(f"{'='*58}")
    print(f"  Total requests:       {args.users}")
    print(f"  Succeeded:            {len(oks)}/{args.users}  ({len(oks)*100//args.users}%)")
    print(f"  Failed (429, gave up):{n_max_retries + n429}")
    print(f"  Needed retry:         {len(retried)}/{len(oks)}")
    print(f"  Total retry attempts: {total_retries}")
    print(f"  Accuracy:             {correct}/{len(oks)} ({correct*100//max(len(oks),1)}%)")
    print(f"  Wall clock:           {wall:.2f}s")
    if lats:
        print(f"  p50 latency:          {pct(lats, 0.5):.2f}s")
        print(f"  p95 latency:          {pct(lats, 0.95):.2f}s")
        print(f"  Max latency:          {lats[-1]:.2f}s")
    print(f"  Avg prompt tokens:    {sum(r['prompt_tokens'] for r in oks)//max(len(oks),1)}")
    if retried:
        print(f"  Avg wait (retried):   {sum(r['total_wait'] for r in retried)/len(retried):.2f}s")
    print(f"{'='*58}")

    print(f"\nClassification results ({len(oks)} successful):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        conf_avg = sum(r["confidence"] for r in oks if r["category"] == cat) / count
        expected_count = sum(1 for r in oks if r["expected"] == cat)
        correct_count = sum(1 for r in oks if r["category"] == cat and r["correct"])
        print(f"  {cat:<25} {count:>3}x  conf:{conf_avg:.2f}  correct:{correct_count}/{expected_count}")

    if fails:
        codes: dict[str, int] = {}
        for r in fails:
            codes[str(r.get("code", "?"))] = codes.get(str(r.get("code", "?")), 0) + 1
        print(f"\nFailed breakdown: {codes}")


if __name__ == "__main__":
    import urllib.request
    main()
