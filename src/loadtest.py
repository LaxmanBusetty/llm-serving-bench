"""Async load test for any OpenAI-compatible /v1/chat/completions endpoint.

Works the same against vLLM (Colab GPU), Ollama (Mac), SGLang, or the FastAPI gateway
in this repo, so results are comparable across servers.

Metrics per concurrency level:
  - TTFT (time to first token): p50 / p95
  - end-to-end latency: p50 / p95
  - decode speed per request (tokens/sec after the first token): p50
  - aggregate throughput (total output tokens / wall-clock seconds)
  - request throughput (requests/sec) and error count

Usage:
  python src/loadtest.py --base-url http://localhost:8000 --model Qwen/Qwen3-4B \
      --label vllm_qwen3-4b_fp16 --levels 1,4,8,16 --requests-per-level 32
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import build_messages, make_dataset  # noqa: E402


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def one_request(client: httpx.AsyncClient, url: str, payload: dict, headers: dict, timeout: float) -> dict:
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage_tokens = None
    try:
        async with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")[:200]
                return {"ok": False, "error": f"HTTP {r.status_code}: {body}", "latency": time.perf_counter() - t0}
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens", usage_tokens)
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks += 1
    except Exception as e:  # network error, timeout, etc.
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "latency": time.perf_counter() - t0}
    latency = time.perf_counter() - t0
    tokens = usage_tokens if usage_tokens is not None else chunks
    if ttft is None or tokens == 0:
        return {"ok": False, "error": "no tokens received", "latency": latency}
    decode_time = max(latency - ttft, 1e-9)
    return {
        "ok": True,
        "latency": latency,
        "ttft": ttft,
        "tokens": tokens,
        "token_source": "usage" if usage_tokens is not None else "chunks",
        "decode_tok_s": (tokens - 1) / decode_time if tokens > 1 else None,
    }


async def run_level(args, prompts: list[list[dict]], concurrency: int, n_requests: int) -> dict:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    extra = json.loads(args.extra_body) if args.extra_body else {}
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(n_requests):
        queue.put_nowait(prompts[i % len(prompts)])
    results: list[dict] = []

    async def worker(client):
        while True:
            try:
                messages = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                **extra,
            }
            results.append(await one_request(client, url, payload, headers, args.timeout))

    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(client) for _ in range(concurrency)])
        wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    total_tokens = sum(r["tokens"] for r in ok)
    decode = [r["decode_tok_s"] for r in ok if r["decode_tok_s"] is not None]
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "ok": len(ok),
        "errors": len(errors),
        "first_error": errors[0]["error"] if errors else None,
        "wall_s": round(wall, 3),
        "output_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / wall, 2) if wall > 0 else None,
        "req_per_s": round(len(ok) / wall, 3) if wall > 0 else None,
        "ttft_p50_s": _r(percentile([r["ttft"] for r in ok], 50)),
        "ttft_p95_s": _r(percentile([r["ttft"] for r in ok], 95)),
        "latency_p50_s": _r(percentile([r["latency"] for r in ok], 50)),
        "latency_p95_s": _r(percentile([r["latency"] for r in ok], 95)),
        "decode_tok_s_p50": _r(statistics.median(decode)) if decode else None,
        "token_source": ok[0]["token_source"] if ok else None,
    }


def _r(x, nd=3):
    return None if x is None else round(x, nd)


def gpu_info() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return f"no NVIDIA GPU ({platform.machine()}, {platform.system()})"


async def main_async(args) -> dict:
    data = make_dataset(max(args.requests_per_level, 64), seed=args.seed)
    prompts = [build_messages(d["text"], no_think=not args.think) for d in data]

    if args.warmup > 0:  # warm-up so one-time costs (CUDA graphs, model load) do not skew level 1
        await run_level(args, prompts, 1, args.warmup)

    levels = []
    for c in [int(x) for x in args.levels.split(",")]:
        print(f"[{args.label}] concurrency={c} requests={args.requests_per_level} ...", flush=True)
        res = await run_level(args, prompts, c, args.requests_per_level)
        print(json.dumps(res), flush=True)
        levels.append(res)

    return {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "gpu": gpu_info(),
        "max_tokens": args.max_tokens,
        "workload": "synthetic receipt extraction, shared ~350-token system prompt, temperature 0",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "levels": levels,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="short name for this configuration, used in the output filename")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--requests-per-level", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--extra-body", default="", help='JSON merged into the request, e.g. \'{"top_p": 0.9}\'')
    ap.add_argument("--think", action="store_true", help="do not append /no_think (Qwen3 reasoning stays on)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    result = asyncio.run(main_async(args))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"loadtest_{args.label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
