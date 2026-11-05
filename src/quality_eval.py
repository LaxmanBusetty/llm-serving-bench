"""Quality evaluation: receipt field extraction scored against known gold labels.

Run this against each serving configuration so you can plot quality versus speed
(for example, does 4-bit quantization change extraction accuracy?).

Metrics:
  - json_valid_rate: output parsed as a JSON object
  - per-field accuracy: vendor, date, item_count, tax, total, payment_method
  - exact_match_rate: all six fields correct
  - bootstrap 95% confidence interval on exact_match_rate (so small differences are not over-read)

Usage:
  python src/quality_eval.py --base-url http://localhost:8000 --model Qwen/Qwen3-4B --label vllm_qwen3-4b_fp16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import build_messages, make_dataset  # noqa: E402

FIELDS = ["vendor", "date", "item_count", "tax", "total", "payment_method"]


def parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (handles <think> blocks and code fences)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "")
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def score_record(pred: dict | None, gold: dict) -> dict[str, bool]:
    if pred is None:
        return {f: False for f in FIELDS}
    out = {}
    out["vendor"] = str(pred.get("vendor", "")).strip().lower() == gold["vendor"].lower()
    out["date"] = str(pred.get("date", "")).strip() == gold["date"]
    ic = _num(pred.get("item_count"))
    out["item_count"] = ic is not None and int(ic) == gold["item_count"]
    for k in ("tax", "total"):
        v = _num(pred.get(k))
        out[k] = v is not None and abs(v - gold[k]) < 0.01
    out["payment_method"] = str(pred.get("payment_method", "")).strip().lower() == gold["payment_method"]
    return out


def bootstrap_ci(flags: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(flags)
    if n == 0:
        return (0.0, 0.0)
    means = sorted(sum(rng.choice(flags) for _ in range(n)) / n for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def summarize(scores: list[dict[str, bool]], valid: list[bool]) -> dict:
    n = len(scores)
    exact = [int(all(s.values())) for s in scores]
    lo, hi = bootstrap_ci(exact)
    return {
        "n": n,
        "json_valid_rate": round(sum(valid) / n, 4) if n else None,
        "field_accuracy": {f: round(sum(s[f] for s in scores) / n, 4) for f in FIELDS} if n else {},
        "exact_match_rate": round(sum(exact) / n, 4) if n else None,
        "exact_match_ci95": [round(lo, 4), round(hi, 4)],
    }


async def call(client, url, payload, headers, sem):
    async with sem:
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"__ERROR__ {type(e).__name__}: {e}"


async def run(args) -> dict:
    data = make_dataset(args.n, seed=args.seed)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    extra = json.loads(args.extra_body) if args.extra_body else {}
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            call(
                client, url,
                {"model": args.model, "messages": build_messages(d["text"], no_think=not args.think),
                 "max_tokens": args.max_tokens, "temperature": 0, **extra},
                headers, sem,
            )
            for d in data
        ]
        replies = await asyncio.gather(*tasks)

    scores, valid, errors, examples = [], [], 0, []
    for d, reply in zip(data, replies):
        if reply.startswith("__ERROR__"):
            errors += 1
        pred = parse_json(reply)
        valid.append(pred is not None)
        s = score_record(pred, d["gold"])
        scores.append(s)
        if not all(s.values()) and len(examples) < 5:
            examples.append({"gold": d["gold"], "reply": reply[:300]})
    summary = summarize(scores, valid)
    summary.update({"label": args.label, "model": args.model, "request_errors": errors, "failure_examples": examples})
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--extra-body", default="")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    summary = asyncio.run(run(args))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"quality_{args.label}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "failure_examples"}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
