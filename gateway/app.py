"""Small FastAPI gateway in front of an OpenAI-compatible model server (vLLM or Ollama).

What it adds on top of the raw model server:
  - /health that checks the upstream server, not just this process
  - per-client rate limiting (sliding window)
  - request timeout and clean error responses (502 / 504 instead of hanging)
  - structured JSON request logs (latency, tokens, status)
  - /metrics with request counts, error counts and latency percentiles
  - optional Langfuse tracing (only active if LANGFUSE_* env vars are set)

Environment variables:
  UPSTREAM_URL        default http://localhost:8000
  MODEL               model name sent upstream (required for real use)
  RATE_LIMIT_PER_MIN  default 60 per client
  TIMEOUT_S           default 60
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST  optional

Run:  uvicorn gateway.app:app --host 0.0.0.0 --port 9000
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "")
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "60"))

logger = logging.getLogger("gateway")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ---- optional Langfuse tracing (works only if the package and keys are present) ----
LANGFUSE_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
try:
    if LANGFUSE_ON:
        try:
            from langfuse import observe  # langfuse v3
        except ImportError:
            from langfuse.decorators import observe  # langfuse v2
    else:
        raise ImportError
except Exception:  # no langfuse, or keys missing: use a no-op decorator
    LANGFUSE_ON = False

    def observe(*dargs, **dkwargs):  # type: ignore[no-redef]
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return lambda fn: fn


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class Metrics:
    def __init__(self) -> None:
        self.requests = 0
        self.errors = 0
        self.rate_limited = 0
        self.latencies: deque[float] = deque(maxlen=1000)

    def snapshot(self) -> dict:
        lat = sorted(self.latencies)

        def pct(p: float):
            if not lat:
                return None
            return round(lat[min(len(lat) - 1, int(len(lat) * p / 100))], 4)

        return {
            "requests": self.requests,
            "errors": self.errors,
            "rate_limited": self.rate_limited,
            "latency_p50_s": pct(50),
            "latency_p95_s": pct(95),
            "langfuse_enabled": LANGFUSE_ON,
        }


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        q = self.hits.setdefault(key, deque())
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


app = FastAPI(title="LLM gateway")
metrics = Metrics()
limiter = RateLimiter(RATE_LIMIT_PER_MIN)


@observe()
async def call_upstream(client: httpx.AsyncClient, payload: dict) -> dict:
    r = await client.post(f"{UPSTREAM_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{UPSTREAM_URL}/v1/models", timeout=5)
        if r.status_code == 200:
            return {"status": "ok", "upstream": "ok"}
        raise HTTPException(status_code=503, detail=f"upstream returned {r.status_code}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"upstream unreachable: {type(e).__name__}")


@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


@app.post("/v1/chat")
async def chat(req: ChatRequest, request: Request):
    client_id = request.client.host if request.client else "unknown"
    if not limiter.allow(client_id):
        metrics.rate_limited += 1
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    payload = {
        "model": MODEL,
        "messages": req.messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    t0 = time.perf_counter()
    status, tokens = 200, None
    metrics.requests += 1
    try:
        async with httpx.AsyncClient() as client:
            data = await call_upstream(client, payload)
        tokens = (data.get("usage") or {}).get("completion_tokens")
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "completion_tokens": tokens, "latency_s": round(time.perf_counter() - t0, 4)}
    except httpx.TimeoutException:
        status = 504
        metrics.errors += 1
        raise HTTPException(status_code=504, detail="upstream timeout")
    except (httpx.HTTPError, KeyError, ValueError) as e:
        status = 502
        metrics.errors += 1
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}")
    finally:
        latency = time.perf_counter() - t0
        metrics.latencies.append(latency)
        logger.info(json.dumps({"event": "chat", "client": client_id, "status": status,
                                "latency_s": round(latency, 4), "completion_tokens": tokens}))
