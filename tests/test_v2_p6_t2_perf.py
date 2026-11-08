# -*- coding: utf-8 -*-
"""P6 T2 体积 / 打开 / 生成三门（性能 3 项，设计 §7.1 / §7.2）。

口径（**不得放宽**）：

- 体积：标准 + 压力语料各页 ``<= 3,000,000`` 字节（真实生成字节，不看压缩值）；
- 打开：5 冷 + 5 热每次 ``< 2,000`` ms —— 真实测量只由 ``scripts/v2_p6/measure_report.py``
  在验收驱动中执行；本文件验证**判定协议**（注入合成度量记录）与阈值未被放宽；
- 生成：标准语料完整链路（快照→投影→HTML→manifest→发布）逐次 ``< 10,000`` ms，
  本文件为**真实测量**（进程内 5 次，同 ``generated_at``，顺带校验字节一致）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import contracts, perf_gate  # noqa: E402
from src.lazybull.v2.reporting.generate import (  # noqa: E402
    DashboardRequest,
    generate_dashboard,
)
from tests.p6_t2_env import case_paper_root  # noqa: E402


def _request(paper: Path, quality: Path, reports_root: Path, *, attempt: str) -> DashboardRequest:
    return DashboardRequest(
        paper_root=paper,
        quality_root=quality,
        reports_root=reports_root,
        evidence_class="research",
        generated_at="20261106T020000Z",
        attempt_id=attempt,
        input_evidence_classes=("production",),
        git_record={"git_commit": "<synthetic>", "git_dirty": False},
    )


def test_p6_t2_08_size_gate_standard_and_stress(tmp_path: Path) -> None:
    """标准 + 压力语料真实字节均 <= 3MB（阈值来自冻结契约，不放宽）。"""
    assert perf_gate.MAX_REPORT_BYTES == contracts.THRESHOLDS["max_report_bytes"] == 3_000_000
    sizes: dict[str, int] = {}
    for profile in ("standard", "stress"):
        paper, quality = case_paper_root(tmp_path, profile)
        result = generate_dashboard(
            _request(paper, quality, tmp_path / f"reports_{profile}", attempt=f"size_{profile}")
        )
        payload = result["files"]["report.html"]
        sizes[profile] = len(payload)
        gate = perf_gate.judge_size(len(payload))
        assert gate["passed"], f"{profile} 超体积门：{gate}"
        assert gate["threshold"] == 3_000_000
        # 压力页必须真的包含压力规模（否则体积门是假通过）
        if profile == "stress":
            assert sizes["stress"] > sizes["standard"] * 1.5
    assert sizes["stress"] <= 3_000_000


def test_p6_t2_09_open_gate_protocol_and_thresholds_not_relaxed() -> None:
    """打开门协议：5 冷 + 5 热，逐次 <2000ms；阈值与次数不得放宽。"""
    assert perf_gate.MAX_OPEN_MS == 2_000
    assert perf_gate.COLD_REPEATS == 5 and perf_gate.HOT_REPEATS == 5
    records = [{"mode": "cold", "elapsed_ms": 120.0 + index} for index in range(5)] + [
        {"mode": "hot", "elapsed_ms": 90.0 + index} for index in range(5)
    ]
    gate = perf_gate.judge_open_series(records)
    assert gate["passed"] is True and gate["counts_ok"] is True
    assert gate["cold_count"] == 5 and gate["hot_count"] == 5

    # 边界负例：恰 2000ms 不通过（严格小于）；1999.9ms 通过
    assert perf_gate.judge_open(1_999.9, mode="cold")["passed"] is True
    assert perf_gate.judge_open(2_000.0, mode="hot")["passed"] is False
    too_slow = [dict(record) for record in records]
    too_slow[3] = {"mode": "cold", "elapsed_ms": 2_000.0}
    assert perf_gate.judge_open_series(too_slow)["passed"] is False

    # 次数不足必须报错（不得以少测冒充通过）
    import pytest

    with pytest.raises(perf_gate.MeasurementError):
        perf_gate.judge_open_series(records[:9])


def test_p6_t2_10_generation_gate_real_measurement(tmp_path: Path) -> None:
    """生成门：完整链路 5 次逐次 <10s，且同输入同 generated_at 逐字节一致。"""
    assert perf_gate.MAX_GENERATION_MS == 10_000
    assert perf_gate.GENERATION_REPEATS == 5
    paper, quality = case_paper_root(tmp_path, "standard")
    elapsed: list[float] = []
    outputs: list[dict[str, bytes]] = []
    for index in range(5):
        result = generate_dashboard(
            _request(paper, quality, tmp_path / f"gen_{index}", attempt=f"gen_{index}")
        )
        elapsed.append(result["elapsed_ms"])
        outputs.append(dict(result["files"]))
    gate = perf_gate.judge_generation_series(elapsed)
    assert gate["passed"], gate
    assert gate["repeat"] == 5
    assert all(item == outputs[0] for item in outputs), "同输入同 generated_at 必须字节一致"
    # 单次最慢也必须留出余量（10s 门下的实测值用于验收登记）
    assert max(elapsed) < perf_gate.MAX_GENERATION_MS
