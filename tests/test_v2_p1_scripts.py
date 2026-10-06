# -*- coding: utf-8 -*-
"""v2 P1 脚本组评审整改的回归测试。

覆盖（编号沿用评审意见）：
- P1-6：run_frozen_reference --serial/--parallel argparse 语义（use_parallel 推导）；
- R3-08：_iter_chunks 分块参数终止性校验（0/负数拒绝、正数分块行为不变）；
        backfill_panel argparse fail-fast（--chunk-years < 1 ⇒ SystemExit 非零）；
- R3-11：bench_panel_perf._evaluate 双口径判定（两口径各自 ≤ 门限才 PASS）；
- 复制公式消除：fix_unlock_coverage_d12._risk_flag 与源函数 compute_unlock_risk_flag 同值。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v2_p1"))

import run_frozen_reference as rfr  # noqa: E402


def _use_parallel(argv: list[str]) -> bool:
    """按 main() 的推导式计算 use_parallel（scripts/v2_p1/run_frozen_reference.py）。"""
    old_argv = sys.argv
    try:
        sys.argv = ["run_frozen_reference.py", *argv]
        args = rfr._parse_args()
    finally:
        sys.argv = old_argv
    return args.parallel and not args.serial


class TestParallelFlagSemantics:
    """P1-6：--serial 默认 False 后，--parallel 不再被恒 True 的 serial 吞掉。"""

    def test_default_is_serial(self) -> None:
        assert _use_parallel([]) is False

    def test_parallel_enables_parallel(self) -> None:
        assert _use_parallel(["--parallel"]) is True

    def test_explicit_serial_wins_over_parallel(self) -> None:
        assert _use_parallel(["--parallel", "--serial"]) is False

    def test_explicit_serial_alone(self) -> None:
        assert _use_parallel(["--serial"]) is False


class TestIterChunksTermination:
    """R3-08：chunk_years < 1 必须拒绝（0 = 不推进、负数 = 倒退，均无限循环）。"""

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_years"):
            rfr._iter_chunks("20120104", "20260702", 0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_years"):
            rfr._iter_chunks("20120104", "20260702", -1)

    def test_positive_chunking_unchanged(self) -> None:
        chunks = rfr._iter_chunks("20120104", "20260702", 5)
        assert chunks == [
            ("20120104", "20161231"),
            ("20150101", "20211231"),
            ("20200101", "20260702"),
        ]

    def test_chunk_years_one(self) -> None:
        chunks = rfr._iter_chunks("20120104", "20131231", 1)
        # 起点钳制 max(全局起点, Y−2 年)：2012 年块取全局起点，2013 年块钳到 2011-01-01
        assert chunks == [("20120104", "20121231"), ("20120104", "20131231")]


class TestBackfillChunkYearsGate:
    """R3-08：backfill_panel 对 --chunk-years < 1 fail-fast（SystemExit 非零退出）。"""

    def test_zero_exits_nonzero(self) -> None:
        import backfill_panel

        old_argv = sys.argv
        try:
            sys.argv = ["backfill_panel.py", "--chunk-years", "0"]
            with pytest.raises(SystemExit) as exc_info:
                backfill_panel._parse_args()
        finally:
            sys.argv = old_argv
        assert exc_info.value.code != 0

    def test_negative_exits_nonzero(self) -> None:
        import backfill_panel

        old_argv = sys.argv
        try:
            sys.argv = ["backfill_panel.py", "--chunk-years", "-2"]
            with pytest.raises(SystemExit) as exc_info:
                backfill_panel._parse_args()
        finally:
            sys.argv = old_argv
        assert exc_info.value.code != 0


class TestBenchDualGate:
    """R3-11：性能门双口径各自判定，均过才 PASS。"""

    def test_both_under_gate_pass(self) -> None:
        import bench_panel_perf

        verdict = bench_panel_perf._evaluate(1.2, 1.4)
        assert verdict["pass"] is True
        assert verdict["ratio_full_cols"] == 1.2
        assert verdict["ratio_sample_5_cols"] == 1.4

    def test_sample_over_gate_fails(self) -> None:
        import bench_panel_perf

        assert bench_panel_perf._evaluate(1.2, 1.6)["pass"] is False

    def test_full_over_gate_fails(self) -> None:
        import bench_panel_perf

        assert bench_panel_perf._evaluate(1.6, 1.2)["pass"] is False

    def test_boundary_equal_gate_passes(self) -> None:
        import bench_panel_perf

        assert bench_panel_perf._evaluate(1.5, 1.5)["pass"] is True


class TestUnlockRiskFlagReuse:
    """复制公式消除：_risk_flag 与 compute_unlock_risk_flag 同值同语义。"""

    def test_matches_source_function(self) -> None:
        import fix_unlock_coverage_d12 as fix

        from src.lazybull.factors.risk.announcement_factors import compute_unlock_risk_flag

        days = pd.Series([10.0, 30.0, 45.0, 90.0, 120.0, 0.0, np.nan, -5.0])
        expected = compute_unlock_risk_flag(pd.DataFrame({"days_to_unlock": days})).astype(float)
        actual = fix._risk_flag(days)
        pd.testing.assert_series_equal(actual, expected)
