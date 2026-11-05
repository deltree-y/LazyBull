# -*- coding: utf-8 -*-
"""P6 T0.5 性能门边界（P6-T05-02/03/04）。

全部负例通过**注入合成度量记录**触发，不制造真实慢页面：
- 体积 3,000,000 通过 / 3,000,001 失败；
- 打开 5 冷 + 5 热、每次 < 2,000ms（2,000 整失败）；
- 生成 5 次、每次 < 10,000ms（10,000 整失败）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import perf_gate  # noqa: E402


def test_p6_t05_02_size_boundary_uses_injected_metric() -> None:
    assert perf_gate.judge_size(3_000_000)["passed"] is True
    assert perf_gate.judge_size(3_000_001)["passed"] is False
    assert perf_gate.evaluate_metric({"kind": "size", "value": 3_000_001})["passed"] is False
    assert perf_gate.evaluate_metric({"kind": "size", "value": 3_000_000})["passed"] is True

    with pytest.raises(perf_gate.MeasurementError):
        perf_gate.judge_size(-1)
    with pytest.raises(perf_gate.MeasurementError):
        perf_gate.judge_size(True)
    with pytest.raises(perf_gate.MeasurementError):
        perf_gate.evaluate_metric({"kind": "unknown", "value": 1})


def _open_series(cold_ms: float = 100.0, hot_ms: float = 100.0) -> list[dict[str, float]]:
    return [{"mode": "cold", "elapsed_ms": cold_ms}] * 5 + [
        {"mode": "hot", "elapsed_ms": hot_ms}
    ] * 5


def test_p6_t05_03_open_boundary_requires_5cold_5hot() -> None:
    assert perf_gate.judge_open(1_999.999, mode="cold")["passed"] is True
    assert perf_gate.judge_open(2_000, mode="cold")["passed"] is False
    assert perf_gate.judge_open_series(_open_series())["passed"] is True
    assert perf_gate.judge_open_series(_open_series(hot_ms=1_999.0))["passed"] is True

    negative = _open_series(hot_ms=2_000.0)
    assert perf_gate.judge_open_series(negative)["passed"] is False

    with pytest.raises(perf_gate.MeasurementError):
        perf_gate.judge_open_series(_open_series()[:9])
    with pytest.raises(perf_gate.MeasurementError):
        perf_gate.judge_open(float("nan"), mode="hot")
    with pytest.raises(perf_gate.MeasurementError):
        perf_gate.judge_open(100.0, mode="warm")
    assert (
        perf_gate.evaluate_metric({"kind": "open", "value": 2_000.0, "mode": "cold"})["passed"]
        is False
    )


def test_p6_t05_04_generation_boundary_is_5_runs() -> None:
    assert perf_gate.judge_generation(9_999.999)["passed"] is True
    assert perf_gate.judge_generation(10_000)["passed"] is False
    assert perf_gate.judge_generation_series([1_000.0] * 5)["passed"] is True

    negative = perf_gate.judge_generation_series([1_000.0] * 4 + [10_000.0])
    assert negative["passed"] is False
    assert negative["counts_ok"] is True

    with pytest.raises(perf_gate.MeasurementError):
        perf_gate.judge_generation_series([1_000.0] * 4)
    with pytest.raises(perf_gate.MeasurementError):
        perf_gate.evaluate_metric({"kind": "generation", "value": float("inf")})
