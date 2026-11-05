# LLM Serving Benchmark: vLLM vs Ollama, speed vs quality

A small, reproducible project that serves an open-source LLM (Qwen3) under different configurations, load tests each one, and measures whether the speed-ups cost any answer quality. The workload is receipt extraction (the same kind of task as my LedgerMate project), scored against known gold labels.

> **Status: fill in the results section after you run it.** Nothing below the "Results" heading is a claim until you replace the placeholders with your own numbers.

## Questions this project answers

1. How do throughput and latency change as concurrent requests go from 1 to 16?
2. What does 4-bit quantization (AWQ) save in memory and gain in speed, and does extraction accuracy change?
3. Does prefix caching reduce time to first token when every request shares a long system prompt?
4. How does a laptop (Ollama on a Mac) compare with a cloud GPU (vLLM on Colab)?
5. What does it take to wrap a model server in a service with health checks, rate limiting, timeouts and metrics?

## What is in the repo

```
src/synth.py          seeded synthetic receipts with gold labels, shared system prompt
src/loadtest.py       async streaming load test (TTFT, latency, tokens/sec, throughput, errors)
src/quality_eval.py   field-level accuracy, JSON validity, bootstrap 95% CI
src/summarize.py      markdown tables and charts from results/*.json
gateway/app.py        FastAPI gateway: /health, /v1/chat, /metrics, rate limit, timeouts, optional Langfuse
gateway/Dockerfile    container for the gateway
notebooks/            Colab notebook that runs the whole experiment on a GPU
tests/                10 pytest tests using a mock OpenAI-compatible server (no GPU needed)
```

## How to run

**1. Tests (any machine, no GPU)**
```bash
pip install -r requirements.txt
python -m pytest -q tests
```

**2. GPU experiment (Colab)**
Open `notebooks/serving_benchmark_colab.ipynb`, choose a GPU runtime, run the cells top to bottom. It starts vLLM for each configuration in the `CONFIGS` list, load tests it, scores quality, and saves everything to `results/` and Google Drive.

**3. Laptop baseline (Mac, Ollama)**
```bash
ollama pull qwen3:4b
OLLAMA_NUM_PARALLEL=8 ollama serve
python src/loadtest.py   --base-url http://localhost:11434 --model qwen3:4b --label ollama_qwen3-4b_mac
python src/quality_eval.py --base-url http://localhost:11434 --model qwen3:4b --label ollama_qwen3-4b_mac
```
The same scripts work for any OpenAI-compatible server (vLLM, Ollama, SGLang, the gateway), which is what makes the numbers comparable.

**4. Summary tables and charts**
```bash
python src/summarize.py
```

**5. Gateway (optional)**
```bash
UPSTREAM_URL=http://localhost:8000 MODEL=Qwen/Qwen3-4B uvicorn gateway.app:app --port 9000
curl localhost:9000/health
curl localhost:9000/metrics
```
Docker: `docker build -f gateway/Dockerfile -t llm-gateway .`

To trace requests with Langfuse, `pip install langfuse` and set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (and `LANGFUSE_HOST` if self-hosting). Check `/metrics` for `"langfuse_enabled": true` and confirm traces appear in the Langfuse dashboard before you describe it as done.

## Method

- **Workload:** 100 synthetic receipts (seed 7) for quality; a separate seeded set for load testing. Every request carries the same ~350-token system prompt (schema, rules, one example), then a different receipt. Temperature 0.
- **Load test:** closed loop. At concurrency N, N workers keep sending until the request budget is used. A short warm-up run is done first and discarded. Streaming is used so time to first token is measured directly. Output tokens come from the server's usage report when available.
- **Quality:** model output is parsed as JSON (think-blocks and code fences removed). Each of six fields is compared to the gold value (vendor case-insensitive, date exact ISO, numbers within one cent, item count exact, payment method exact). Exact match means all six fields right. A bootstrap 95% confidence interval is reported so small gaps are not over-read.
- **Controls:** one variable changes per configuration, and the GPU model, vLLM version and flags are saved with each result.

## Results

Fill this in from `results/summary.md`. Keep the GPU and versions next to the numbers.

- GPU: `____`   vLLM version: `____`   Date run: `____`

| Config | Cold start (s) | Peak throughput (tok/s) | TTFT p50 @ 1 (s) | p95 latency @ 16 (s) | Exact match (95% CI) |
|---|---|---|---|---|---|
| ollama qwen3-4b (Mac) | | | | | |
| vllm qwen3-4b fp16, prefix on | | | | | |
| vllm qwen3-4b fp16, prefix off | | | | | |
| vllm qwen3-4b AWQ | | | | | |
| vllm qwen3-8b AWQ | | | | | |

Charts: `results/throughput_vs_concurrency.png`, `results/p95_latency_vs_concurrency.png`, `results/quality_vs_speed.png`.

### Findings (write these after you have numbers)
- Batching: throughput went from `__` to `__` tok/s going from 1 to 16 concurrent requests, while p95 latency went from `__` to `__` s.
- Quantization: AWQ changed exact match from `__` to `__` (CIs `__` and `__`), so the difference is / is not distinguishable from noise.
- Prefix caching: TTFT p50 changed from `__` to `__` s.
- Laptop vs GPU: `__`.

### Limitations (keep these, edit to match what you actually did)
- One GPU type, one synthetic workload, 100 evaluation examples.
- Synthetic receipts are cleaner than real ones. Results show relative differences between configurations, not real-world accuracy.
- This is a benchmark, not a production deployment serving users.

## Resume bullets (use only after you have real numbers)

- Benchmarked vLLM and Ollama serving of Qwen3 across `N` configurations (fp16 vs AWQ 4-bit, prefix caching on/off); measured throughput, time to first token and p95 latency at 1 to 16 concurrent requests, reaching `X` tok/s on a `GPU`.
- Evaluated quality impact of quantization on a receipt-extraction task with gold labels (`n` examples, bootstrap CIs); AWQ changed exact-match accuracy by `X` points (`within / outside` noise) while reducing memory by `Y`%.
- Built a FastAPI gateway with upstream health checks, rate limiting, timeouts and latency metrics, covered by 10 automated tests.

## Concepts to be ready to explain

- **Continuous batching:** new requests join a running batch at every decoding step instead of waiting for the batch to finish, which is why throughput rises with concurrency.
- **KV cache and PagedAttention:** the key/value cache grows with sequence length and usually limits how many requests fit in memory. PagedAttention stores it in blocks to avoid wasted space.
- **Prefill vs decode:** prefill processes the prompt in parallel (drives TTFT); decode generates one token at a time (drives tokens/sec).
- **Quantization trade-off:** 4-bit weights use about a quarter of the memory of fp16 and can speed up decoding, with a possible accuracy cost. That is why this project measures it.
- **Prefix caching:** reuses the KV cache of a shared prompt prefix, cutting prefill work for requests that begin with the same text.
- **Latency vs throughput:** pushing concurrency up raises total tokens/sec but makes each request slower. The right setting depends on the service-level target.

## Troubleshooting

- **Out of memory when the server starts:** lower `--max-model-len`, `--max-num-seqs` or `--gpu-memory-utilization`, or pick a smaller or quantized model. Look at `results/server_<label>.log`.
- **Server will not start on a T4:** T4 has no bfloat16 and lacks some newer kernels. Keep `--dtype half`. If your vLLM version does not support your GPU, try an L4 or A100 runtime, or install an earlier vLLM release and record the version.
- **Model name not found:** check the exact repo name on Hugging Face and edit `CONFIGS`.
- **Colab disconnects:** results are copied to Drive after each configuration. Rerun and skip finished configs by removing them from the list.
- **Ollama shows no speed-up with concurrency:** set `OLLAMA_NUM_PARALLEL` before starting the server.
- **Token counts look odd:** the load test uses the server's usage report when present and falls back to counting streamed chunks (the result file records which one was used).

## Tested and not tested

The load tester, evaluator, summary tools and gateway are covered by automated tests and a full dry run against a mock server. The vLLM, Ollama and GPU parts were not run when this was written, so the first real run may need small adjustments (model names, flags for your vLLM version).
