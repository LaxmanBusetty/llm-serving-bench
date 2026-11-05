import asyncio
import os
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import loadtest  # noqa: E402
import quality_eval  # noqa: E402
import synth  # noqa: E402
from mock_server import make_app  # noqa: E402


class Server:
    def __init__(self, app, port):
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.port = port

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{self.port}/v1/models", timeout=0.5)
                return self
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("mock server did not start")

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(timeout=5)


# ---------- synth ----------
def test_synth_is_deterministic_and_consistent():
    a, b = synth.make_dataset(20, seed=3), synth.make_dataset(20, seed=3)
    assert a == b
    for r in a:
        g = r["gold"]
        assert abs(g["total"] - round(g["total"], 2)) < 1e-9
        assert g["payment_method"] in ("cash", "card")
        assert g["item_count"] == sum(1 for l in r["text"].splitlines() if " x " in l)
        assert f"TOTAL     ${g['total']:.2f}" in r["text"]


# ---------- quality_eval ----------
def test_parse_json_handles_think_and_fences():
    assert quality_eval.parse_json('<think>hmm {x}</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert quality_eval.parse_json("no json here") is None
    assert quality_eval.parse_json('{"a": ') is None


def test_score_record():
    gold = {"vendor": "Blue Harbor Cafe", "date": "2026-03-14", "item_count": 3, "tax": 1.2, "total": 15.5, "payment_method": "card"}
    good = {"vendor": "blue harbor cafe", "date": "2026-03-14", "item_count": 3, "tax": "$1.20", "total": 15.5, "payment_method": "Card"}
    assert all(quality_eval.score_record(good, gold).values())
    bad = dict(good, total=16.5)
    s = quality_eval.score_record(bad, gold)
    assert s["total"] is False and s["vendor"] is True
    assert not any(quality_eval.score_record(None, gold).values())


def test_bootstrap_ci_brackets_mean():
    flags = [1] * 80 + [0] * 20
    lo, hi = quality_eval.bootstrap_ci(flags)
    assert lo < 0.8 < hi and 0.65 < lo and hi < 0.92


def test_percentile():
    assert loadtest.percentile([1, 2, 3, 4], 50) == 2.5
    assert loadtest.percentile([], 50) is None


# ---------- end to end against the mock server ----------
def test_loadtest_and_eval_against_mock(tmp_path):
    with Server(make_app(delay_s=0.001), 18931):
        args = SimpleNamespace(base_url="http://127.0.0.1:18931", model="mock", label="t", levels="1,4",
                               requests_per_level=12, max_tokens=64, timeout=20.0, warmup=2, seed=1,
                               api_key="", extra_body="", think=False, out_dir=str(tmp_path))
        res = asyncio.run(loadtest.main_async(args))
        assert [lv["concurrency"] for lv in res["levels"]] == [1, 4]
        for lv in res["levels"]:
            assert lv["errors"] == 0 and lv["ok"] == 12
            assert lv["throughput_tok_s"] > 0 and lv["ttft_p50_s"] is not None
            assert lv["latency_p95_s"] >= lv["latency_p50_s"]
        assert res["levels"][1]["throughput_tok_s"] > res["levels"][0]["throughput_tok_s"]  # concurrency helps

        qargs = SimpleNamespace(base_url="http://127.0.0.1:18931", model="mock", label="t", n=40, seed=7,
                                concurrency=4, max_tokens=100, api_key="", extra_body="", think=False)
        q = asyncio.run(quality_eval.run(qargs))
        assert q["json_valid_rate"] == 1.0
        # mock cannot read dates, so date field is wrong but everything else should be right
        assert q["field_accuracy"]["total"] == 1.0 and q["field_accuracy"]["payment_method"] == 1.0


def test_loadtest_reports_errors_when_server_down():
    args = SimpleNamespace(base_url="http://127.0.0.1:1", model="m", max_tokens=8, timeout=2.0, api_key="", extra_body="")
    prompts = [synth.build_messages("x")]
    res = asyncio.run(loadtest.run_level(args, prompts, 2, 4))
    assert res["ok"] == 0 and res["errors"] == 4 and res["first_error"]


# ---------- gateway ----------
def test_rate_limiter():
    from gateway.app import RateLimiter

    rl = RateLimiter(3, window_s=10)
    assert [rl.allow("a", now=t) for t in (0, 1, 2, 3)] == [True, True, True, False]
    assert rl.allow("b", now=3) is True
    assert rl.allow("a", now=11.5) is True  # window slid


def test_gateway_end_to_end(monkeypatch):
    import importlib

    with Server(make_app(), 18932):
        monkeypatch.setenv("UPSTREAM_URL", "http://127.0.0.1:18932")
        monkeypatch.setenv("MODEL", "mock")
        monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
        import gateway.app as gw

        gw = importlib.reload(gw)
        from fastapi.testclient import TestClient

        c = TestClient(gw.app)
        assert c.get("/health").json()["upstream"] == "ok"
        msgs = synth.build_messages(synth.make_dataset(1)[0]["text"])
        codes = [c.post("/v1/chat", json={"messages": msgs}).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        m = c.get("/metrics").json()
        assert m["requests"] == 3 and m["rate_limited"] == 2 and m["errors"] == 0


def test_gateway_upstream_down_gives_502_and_unhealthy(monkeypatch):
    import importlib

    monkeypatch.setenv("UPSTREAM_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("TIMEOUT_S", "2")
    import gateway.app as gw

    gw = importlib.reload(gw)
    from fastapi.testclient import TestClient

    c = TestClient(gw.app)
    assert c.get("/health").status_code == 503
    r = c.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert c.get("/metrics").json()["errors"] == 1
