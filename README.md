# LLM Serving Benchmark

A reproducible benchmark for self-hosted LLM inference. It serves an open-source model (Qwen3) under different configurations with vLLM and Ollama, load tests each one, and scores output quality against known answers, so speed gains can be weighed against any loss in accuracy.

The benchmark workload is structured data extraction from receipt text, scored field by field against gold labels. The tools work with any OpenAI-compatible server.

## Contents

- [What it measures](#what-it-measures)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Serving gateway](#serving-gateway)
- [Testing](#testing)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## What it measures

| Area | Metrics |
|---|---|
| Latency | Time to first token (TTFT) and end-to-end latency, p50 and p95 |
| Throughput | Aggregate output tokens per second and requests per second at 1 to N concurrent requests |
| Decode speed | Per-request tokens per second after the first token |
| Reliability | Request error count and first error message per concurrency level |
| Startup | Cold-start time from process launch to the first successful request |
| Quality | JSON validity rate, per-field accuracy, exact-match rate with a bootstrap 95% confidence interval |

Configurations that can be compared include numeric precision (fp16 vs 4-bit AWQ), prefix caching on or off, model size, and server (vLLM vs Ollama).

## Architecture

```
            synthetic receipts (seeded, gold labels)
                         |
          +--------------+---------------+
          |                              |
   src/loadtest.py               src/quality_eval.py
   (streaming, concurrent)       (JSON parse + field scoring)
          |                              |
          +--------------+---------------+
                         |
              OpenAI-compatible API
        (vLLM, Ollama, SGLang, or the gateway)
                         |
                  results/*.json
                         |
                src/summarize.py
            results/summary.md + charts
```

## Repository layout

```
src/
  synth.py            Seeded synthetic receipts with gold labels; shared system prompt
  loadtest.py         Async streaming load test
  quality_eval.py     Field-level accuracy and confidence intervals
  summarize.py        Markdown tables and charts from results/
gateway/
  app.py              FastAPI gateway in front of a model server
  Dockerfile
  requirements.txt
notebooks/
  serving_benchmark_colab.ipynb   End-to-end experiment on a Colab GPU
tests/                Automated tests using a mock OpenAI-compatible server
data/eval_set.jsonl   The default 100-record evaluation set (seed 7)
docs/background.md    Concepts behind the metrics
results/              Output directory (generated)
```

## Quick start

Requirements: Python 3.10 or later. A GPU is needed only for the vLLM experiments.

```bash
git clone https://github.com/LaxmanBusetty/llm-serving-bench.git
cd llm-serving-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

The test suite runs on any machine without a GPU or a model.

## Usage

### 1. GPU experiment (Google Colab)

Open `notebooks/serving_benchmark_colab.ipynb` in Colab with a GPU runtime. The notebook installs vLLM, then for each entry in its `CONFIGS` list it:

1. starts the vLLM server and records cold-start time,
2. runs the load test,
3. runs the quality evaluation,
4. stops the server and copies results to Google Drive.

Edit `CONFIGS` to change models and server flags. Each result file stores the GPU model, model name and server arguments.

### 2. Local baseline (Ollama)

```bash
ollama pull qwen3:4b
OLLAMA_NUM_PARALLEL=8 ollama serve

python src/loadtest.py     --base-url http://localhost:11434 --model qwen3:4b --label ollama_qwen3-4b
python src/quality_eval.py --base-url http://localhost:11434 --model qwen3:4b --label ollama_qwen3-4b
```

### 3. Against any OpenAI-compatible server

```bash
python src/loadtest.py --base-url http://HOST:PORT --model MODEL_NAME --label my_config \
    --levels 1,4,8,16 --requests-per-level 32 --max-tokens 160
```

Key options:

| Option | Default | Purpose |
|---|---|---|
| `--levels` | `1,4,8,16` | Concurrency levels to test |
| `--requests-per-level` | `32` | Requests sent at each level |
| `--max-tokens` | `160` | Output token limit |
| `--warmup` | `4` | Warm-up requests, excluded from results |
| `--extra-body` | none | JSON merged into the request body |
| `--think` | off | Keep Qwen3 reasoning mode on (off by default via `/no_think`) |

### 4. Summarize

```bash
python src/summarize.py
```

Writes `results/summary.md`, `throughput_vs_concurrency.png`, `p95_latency_vs_concurrency.png` and, when quality files exist, `quality_vs_speed.png`.

## Methodology

- **Workload.** Seeded synthetic receipts, so every run uses identical inputs. Each request carries the same system prompt of roughly 350 tokens (schema, rules and one example) followed by a different receipt. Temperature is 0.
- **Load generation.** Closed loop: at concurrency N, N workers send requests back to back until the request budget is used. A warm-up run is discarded. Responses are streamed so TTFT is measured directly. Output token counts come from the server's usage report when available, otherwise from counting streamed chunks; the result file records which was used.
- **Quality scoring.** The model reply is parsed as JSON after removing reasoning blocks and code fences. Six fields are compared with the gold values: vendor (case-insensitive), date (exact ISO), item count (exact), tax and total (within one cent), and payment method (exact). Exact match requires all six. A bootstrap 95% confidence interval (2,000 resamples) accompanies the exact-match rate.
- **Reproducibility.** Seeds are fixed, and each result file stores the hardware description, model name, label, and timestamp. The vLLM version and server flags are saved in `startup_<label>.json`.

## Results

Each run writes its figures to `results/`; see `results/summary.md` and the charts for the numbers from your hardware. Results depend on the GPU, driver, serving-engine version and model, so compare configurations only within the same environment. To share results, include the GPU model and vLLM version alongside the table.

## Serving gateway

`gateway/app.py` is a FastAPI service that sits in front of a model server and adds production basics:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Returns 200 only if the upstream model server responds |
| `POST /v1/chat` | Forwards a chat request with a timeout; returns text, token count and latency |
| `GET /metrics` | Request, error and rate-limit counts; p50 and p95 latency |

Behavior: per-client sliding-window rate limiting (HTTP 429), upstream timeout (504), upstream error (502), and structured JSON request logs.

```bash
UPSTREAM_URL=http://localhost:8000 MODEL=Qwen/Qwen3-4B uvicorn gateway.app:app --port 9000
```

| Variable | Default | Meaning |
|---|---|---|
| `UPSTREAM_URL` | `http://localhost:8000` | Model server base URL |
| `MODEL` | empty | Model name sent upstream |
| `RATE_LIMIT_PER_MIN` | `60` | Requests per client per minute |
| `TIMEOUT_S` | `60` | Upstream timeout in seconds |

Optional Langfuse tracing is enabled when `langfuse` is installed and `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. `/metrics` reports `langfuse_enabled`. Container build:

```bash
docker build -f gateway/Dockerfile -t llm-gateway .
```

## Testing

```bash
python -m pytest -q tests
```

The suite (10 tests) covers dataset determinism and label consistency, JSON extraction, field scoring, bootstrap intervals, percentile math, the load tester and evaluator against a local mock streaming server, error handling when the server is unreachable, and the gateway's health, rate limiting, error mapping and metrics. A mock OpenAI-compatible server (`tests/mock_server.py`) stands in for the model, so no GPU is required.

## Limitations

- The workload is synthetic and cleaner than real receipts. Quality figures reflect relative differences between configurations, not real-world accuracy.
- Evaluation uses 100 examples by default; use the confidence intervals before concluding that two configurations differ.
- Results are specific to the tested GPU, driver and engine versions.
- The gateway is a reference implementation for measuring and learning. It has no authentication, persistence or multi-instance coordination.
- This repository benchmarks serving behavior. It is not a production inference deployment.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Out of memory at server start | Lower `--max-model-len`, `--max-num-seqs` or `--gpu-memory-utilization`, or use a smaller or quantized model. See `results/server_<label>.log`. |
| Server fails on a T4 GPU | The T4 lacks bfloat16 and some newer kernels. Keep `--dtype half`. If your vLLM version does not support the GPU, use an L4 or A100 runtime or an earlier vLLM release. |
| Model not found | Verify the exact repository name on Hugging Face and update `CONFIGS`. |
| No speed-up with concurrency on Ollama | Set `OLLAMA_NUM_PARALLEL` before starting `ollama serve`. |
| `externally-managed-environment` on macOS | Use a virtual environment as shown in Quick start. |

## License

MIT. See [LICENSE](LICENSE).
