#!/usr/bin/env python
"""Run glassbox across one or more Ollama models and print a comparison table.

For each model this runs `batch` (grounding / faithfulness / robustness /
trustworthiness rates over the corpus) and `calibrate` (accuracy + ECE against the
labels), and reports a rough per-document latency. It's the engine behind the
"which model hallucinates least?" table.

With `--runs N` each model is evaluated N times. The main table shows the mean of
each metric; a second table shows the min-max spread across runs, which is how you
*measure* a model's non-determinism rather than just asserting it (the 7B varies
run to run even at temperature 0 -- see FINDINGS.md).

Usage:
    python benchmarks/run_benchmark.py --models qwen2.5:3b,qwen2.5:7b
    python benchmarks/run_benchmark.py --models qwen2.5:7b --runs 5
    python benchmarks/run_benchmark.py --models qwen2.5:3b --corpus benchmarks/corpus.jsonl
    python benchmarks/run_benchmark.py --models qwen2.5:3b --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean

from glassbox.backends import OllamaBackend
from glassbox.batch import load_documents, run_batch
from glassbox.calibration import evaluate, load_dataset
from glassbox.prompts import DEFAULT_POLICY, PROMPT_STYLES

# Metrics reported per run, in table order. latency is kept out of the spread
# table (it's machine noise, not model non-determinism).
RATE_METRICS = ("accuracy", "grounded", "faithful", "robust", "trustworthy")
SCORE_METRICS = ("ece",)
CSV_HEADER = (
    "model",
    "n",
    "accuracy",
    "grounded",
    "faithful",
    "robust",
    "trustworthy",
    "ece",
    "latency",
)


def benchmark(model: str, corpus: str, host: str, prompt_style: str = "plain") -> dict:
    """Run both lenses on *model* over *corpus* once and return the aggregate metrics."""
    backend = OllamaBackend(model=model, host=host)
    policy = replace(DEFAULT_POLICY, prompt_style=prompt_style)
    documents = load_documents(corpus)
    dataset = load_dataset(corpus)

    start = time.monotonic()
    batch = run_batch(documents, backend, policy)
    calibration = evaluate(dataset, backend, policy)
    elapsed = time.monotonic() - start

    return {
        "model": model,
        "n": batch.n,
        "accuracy": calibration.accuracy,
        "grounded": batch.grounded_rate,
        "faithful": batch.faithful_rate,
        "robust": batch.robust_rate,
        "trustworthy": batch.trustworthy_rate,
        "ece": calibration.ece,
        "latency": elapsed / max(batch.n, 1),  # per document, audit + one calibrate call
    }


def aggregate(runs: list[dict]) -> dict:
    """Collapse repeated runs of one model into mean / min / max per metric.

    A single run collapses to mean == min == max, so downstream formatting is the
    same whether ``--runs`` is 1 or 20.
    """
    model = runs[0]["model"]
    metrics = (*RATE_METRICS, *SCORE_METRICS, "latency")
    # N can differ run to run if the model errors on a different set of docs
    # (e.g. a prompt it can't follow); show the range so the displayed N and the
    # averaged rates aren't quietly describing different document sets.
    ns = [r["n"] for r in runs]
    n = ns[0] if len(set(ns)) == 1 else f"{min(ns)}-{max(ns)}"
    row: dict = {"model": model, "n": n, "runs": len(runs)}
    for metric in metrics:
        values = [r[metric] for r in runs]
        row[metric] = {"mean": mean(values), "min": min(values), "max": max(values)}
    return row


def to_markdown(rows: list[dict]) -> str:
    """Mean table. Reads aggregate rows (each metric is a mean/min/max dict)."""
    header = "| Model | N | Accuracy | Grounded | Faithful | Robust | Trustworthy | ECE | ~s/doc |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| `{r['model']}` | {r['n']} | {r['accuracy']['mean']:.0%} "
            f"| {r['grounded']['mean']:.0%} | {r['faithful']['mean']:.0%} "
            f"| {r['robust']['mean']:.0%} | {r['trustworthy']['mean']:.0%} "
            f"| {r['ece']['mean']:.2f} | {r['latency']['mean']:.1f} |"
        )
    return "\n".join(lines)


def _range(cell: dict, pct: bool) -> str:
    """Format a metric's min-max span; a stable metric collapses to a single value."""
    lo, hi = cell["min"], cell["max"]
    fmt = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.2f}")
    return fmt(lo) if lo == hi else f"{fmt(lo)}-{fmt(hi)}"


def to_spread_markdown(rows: list[dict]) -> str:
    """Min-max span of each metric across runs -- the non-determinism readout."""
    runs = rows[0]["runs"]
    header = "| Model | Accuracy | Grounded | Faithful | Robust | Trustworthy | ECE |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [f"### Spread across {runs} runs (min-max)", "", header, sep]
    for r in rows:
        cells = [_range(r[m], pct=True) for m in RATE_METRICS]
        cells.append(_range(r["ece"], pct=False))
        lines.append(f"| `{r['model']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict], path: str | Path) -> None:
    """Write mean benchmark rows to *path* in CSV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in rows:
            writer.writerow(
                [
                    r["model"],
                    r["n"],
                    round(r["accuracy"]["mean"], 4),
                    round(r["grounded"]["mean"], 4),
                    round(r["faithful"]["mean"], 4),
                    round(r["robust"]["mean"], 4),
                    round(r["trustworthy"]["mean"], 4),
                    round(r["ece"]["mean"], 4),
                    round(r["latency"]["mean"], 4),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark glassbox across models (Ollama).")
    parser.add_argument("--models", required=True, help="comma-separated Ollama model tags")
    parser.add_argument("--corpus", default="benchmarks/corpus.jsonl", help="JSON Lines corpus")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument(
        "--runs", type=int, default=1, help="evaluate each model this many times (default 1)"
    )
    parser.add_argument(
        "--prompt",
        choices=PROMPT_STYLES,
        default="plain",
        help="instruction style (default plain); try few-shot to test whether "
        "verbatim-quoting examples recover a model's grounding",
    )
    parser.add_argument("--csv", metavar="PATH", help="write benchmark results to CSV at PATH")
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    rows = []
    for model in (m.strip() for m in args.models.split(",") if m.strip()):
        runs = []
        for i in range(args.runs):
            suffix = f" (run {i + 1}/{args.runs})" if args.runs > 1 else ""
            print(f"# benchmarking {model} [{args.prompt}]{suffix} ...", flush=True)
            runs.append(benchmark(model, args.corpus, args.host, args.prompt))
        rows.append(aggregate(runs))

    print()
    print(to_markdown(rows))
    if args.runs > 1:
        print()
        print(to_spread_markdown(rows))

    if args.csv:
        write_csv(rows, args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
