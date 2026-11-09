"""Tests for the benchmark runner's pure aggregation/formatting (no Ollama needed).

``benchmarks/run_benchmark.py`` isn't an installed module, so load it by path.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_benchmark", _PATH)
assert _spec and _spec.loader
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def _run(model: str, **metrics: float) -> dict:
    base = {
        "model": model,
        "n": 3,
        "accuracy": 1.0,
        "grounded": 1.0,
        "faithful": 1.0,
        "robust": 1.0,
        "trustworthy": 1.0,
        "ece": 0.0,
        "latency": 10.0,
    }
    base.update(metrics)
    return base


def test_aggregate_computes_mean_min_max():
    agg = rb.aggregate(
        [
            _run("m", accuracy=0.9, grounded=1.0, ece=0.2, latency=10.0),
            _run("m", accuracy=0.8, grounded=0.8, ece=0.3, latency=12.0),
        ]
    )
    assert agg["model"] == "m"
    assert agg["runs"] == 2
    assert agg["n"] == 3
    assert agg["accuracy"]["mean"] == pytest.approx(0.85)
    assert agg["accuracy"]["min"] == 0.8 and agg["accuracy"]["max"] == 0.9
    assert agg["grounded"]["min"] == 0.8 and agg["grounded"]["max"] == 1.0
    assert agg["latency"]["mean"] == 11.0


def test_single_run_collapses_to_a_point():
    agg = rb.aggregate([_run("m", accuracy=0.9)])
    assert agg["runs"] == 1
    assert agg["accuracy"]["min"] == agg["accuracy"]["max"] == 0.9


def test_to_markdown_shows_the_mean():
    agg = rb.aggregate([_run("m", accuracy=0.9), _run("m", accuracy=0.7)])
    md = rb.to_markdown([agg])
    assert "`m`" in md
    assert "80%" in md  # mean of 0.9 and 0.7


def test_range_formatting():
    assert rb._range({"min": 0.8, "max": 0.8}, pct=True) == "80%"  # stable -> single value
    assert rb._range({"min": 0.8, "max": 0.9}, pct=True) == "80%-90%"
    assert rb._range({"min": 0.2, "max": 0.3}, pct=False) == "0.20-0.30"


def test_spread_table_reports_the_range():
    agg = rb.aggregate(
        [
            _run("big", grounded=1.0, trustworthy=0.6, ece=0.2),
            _run("big", grounded=0.6, trustworthy=0.2, ece=0.3),
        ]
    )
    spread = rb.to_spread_markdown([agg])
    assert "Spread across 2 runs" in spread
    assert "60%-100%" in spread  # grounded varied run to run
    assert "0.20-0.30" in spread  # ece


def test_write_csv_outputs_expected_header_and_rows(tmp_path: Path):
    agg = rb.aggregate(
        [
            _run("m1", accuracy=0.9, grounded=1.0, ece=0.2, latency=10.0),
            _run("m1", accuracy=0.8, grounded=0.8, ece=0.3, latency=12.0),
        ]
    )
    csv_file = tmp_path / "results.csv"
    rb.write_csv([agg], csv_file)

    assert csv_file.exists()
    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == list(rb.CSV_HEADER)
    assert rows[0] == [
        "model",
        "n",
        "accuracy",
        "grounded",
        "faithful",
        "robust",
        "trustworthy",
        "ece",
        "latency",
    ]
    assert rows[1] == ["m1", "3", "0.85", "0.9", "1.0", "1.0", "1.0", "0.25", "11.0"]


def test_main_with_csv_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    csv_file = tmp_path / "benchmarks" / "out.csv"
    dummy_run = _run("mock-model", accuracy=0.95, grounded=0.9, latency=2.5)
    monkeypatch.setattr(rb, "benchmark", lambda *args, **kwargs: dummy_run)

    exit_code = rb.main(["--models", "mock-model", "--csv", str(csv_file)])
    assert exit_code == 0
    assert csv_file.exists()

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == list(rb.CSV_HEADER)
    assert rows[1][0] == "mock-model"
    assert rows[1][2] == "0.95"
