"""Combine results/*.json into markdown tables and charts.

Writes:
  results/summary.md
  results/throughput_vs_concurrency.png
  results/p95_latency_vs_concurrency.png
  results/quality_vs_speed.png   (only if quality_*.json files exist)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(pattern: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            out.append(json.load(f))
    return out


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    d = args.results_dir

    loads = load(os.path.join(d, "loadtest_*.json"))
    quals = {q["label"]: q for q in load(os.path.join(d, "quality_*.json"))}
    if not loads:
        raise SystemExit(f"no loadtest_*.json files in {d}")

    lines = ["# Results", ""]
    lines += ["| Config | GPU | Concurrency | Throughput (tok/s) | TTFT p50 (s) | TTFT p95 (s) | Latency p95 (s) | Decode tok/s p50 | Errors |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in loads:
        for lv in r["levels"]:
            lines.append(
                f"| {r['label']} | {r['gpu']} | {lv['concurrency']} | {fmt(lv['throughput_tok_s'])} | "
                f"{fmt(lv['ttft_p50_s'], 3)} | {fmt(lv['ttft_p95_s'], 3)} | {fmt(lv['latency_p95_s'])} | "
                f"{fmt(lv['decode_tok_s_p50'])} | {lv['errors']}/{lv['requests']} |"
            )

    if quals:
        lines += ["", "## Quality (receipt extraction, synthetic gold labels)", "",
                  "| Config | n | JSON valid | Exact match | 95% CI | vendor | date | item_count | tax | total | payment |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for label, q in quals.items():
            fa = q["field_accuracy"]
            lines.append(
                f"| {label} | {q['n']} | {fmt(q['json_valid_rate'], 3)} | {fmt(q['exact_match_rate'], 3)} | "
                f"{q['exact_match_ci95'][0]:.2f} to {q['exact_match_ci95'][1]:.2f} | "
                + " | ".join(fmt(fa.get(k), 2) for k in ["vendor", "date", "item_count", "tax", "total", "payment_method"])
                + " |"
            )

    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    for key, title, fname, ylabel in [
        ("throughput_tok_s", "Aggregate throughput vs concurrency", "throughput_vs_concurrency.png", "output tokens / second"),
        ("latency_p95_s", "p95 end-to-end latency vs concurrency", "p95_latency_vs_concurrency.png", "seconds"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for r in loads:
            xs = [lv["concurrency"] for lv in r["levels"]]
            ys = [lv[key] for lv in r["levels"]]
            ax.plot(xs, ys, marker="o", label=r["label"])
        ax.set_xlabel("concurrent requests")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(d, fname), dpi=150)
        plt.close(fig)

    if quals:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for r in loads:
            q = quals.get(r["label"])
            if not q:
                continue
            top = max(r["levels"], key=lambda lv: lv["throughput_tok_s"] or 0)
            lo, hi = q["exact_match_ci95"]
            ax.errorbar(top["throughput_tok_s"], q["exact_match_rate"],
                        yerr=[[q["exact_match_rate"] - lo], [hi - q["exact_match_rate"]]], fmt="o", capsize=3, label=r["label"])
        ax.set_xlabel("peak throughput (output tokens / second)")
        ax.set_ylabel("exact-match rate on receipts")
        ax.set_title("Quality vs speed (bars: bootstrap 95% CI)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "quality_vs_speed.png"), dpi=150)
        plt.close(fig)

    print(f"wrote {os.path.join(d, 'summary.md')} and charts")


if __name__ == "__main__":
    main()
