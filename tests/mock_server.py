"""Tiny OpenAI-compatible mock server used to test the load tester and evaluator without a GPU.

It streams a fixed-length reply with small delays. When the prompt contains a receipt it also
returns a correct (or deliberately wrong, with error_rate) JSON extraction so the evaluator can be checked.
"""
from __future__ import annotations

import asyncio
import json
import random
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def make_app(delay_s: float = 0.002, error_rate: float = 0.0, seed: int = 0) -> FastAPI:
    app = FastAPI()
    rng = random.Random(seed)

    def fake_extraction(text: str) -> str:
        total = re.search(r"^TOTAL\s+\$([\d.]+)", text, flags=re.MULTILINE)
        tax = re.search(r"^TAX\s+\$([\d.]+)", text, flags=re.MULTILINE)
        lines = [l for l in text.splitlines() if re.match(r"^\d+ x ", l)]
        vendor = text.split("Receipt:\n", 1)[-1].splitlines()[0].title()
        d = {"vendor": vendor, "date": "2026-01-01", "item_count": len(lines),
             "tax": float(tax.group(1)) if tax else 0, "total": float(total.group(1)) if total else 0,
             "payment_method": "cash" if "CASH" in text else "card"}
        if rng.random() < error_rate:
            d["total"] = d["total"] + 1
        return json.dumps(d)

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "mock"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        user_text = body["messages"][-1]["content"]
        reply = fake_extraction(user_text)
        if not body.get("stream"):
            return JSONResponse({"choices": [{"message": {"content": reply}}],
                                 "usage": {"completion_tokens": len(reply.split())}})
        pieces = [reply[i : i + 6] for i in range(0, len(reply), 6)]

        async def gen():
            for p in pieces:
                await asyncio.sleep(delay_s)
                yield f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n"
            yield f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': len(pieces)}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
