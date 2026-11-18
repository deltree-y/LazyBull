"""账户层测试：6 槽 2 批、行业上限、单笔下限 V、四臂合并与信号接口。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.analyst_diffusion_ml import native_replay, spec


class _Sleeve(native_replay.PITIndustrySleeve):
    """绕过引擎构造的轻量替身（只测研究侧钩子）。"""

    def __init__(self, positions=None, mapping=None, stats=None) -> None:
        self.positions = positions or {}
        self._today_mapping = mapping or {}
        self._hook_stats = stats if stats is not None else {}


def test_min_ticket_threshold_is_absolute_for_both_capitals() -> None:
    sleeve = _Sleeve()
    date = pd.Timestamp("2024-01-02")
    assert sleeve._get_min_buy_value_threshold(date) == pytest.approx(spec.min_ticket_value())
    # 与账户规模无关（100 万与 200 万用同一 V）
    assert sleeve._get_min_buy_value_threshold(date) == pytest.approx(25588.536335721597)


def test_b_overrides_match_protocol() -> None:
    overrides = native_replay.b_overrides()
    assert overrides["rebalance_freq"] == 20
    assert overrides["holding_period"] == 20
    assert overrides["stagger_tranches"] == 2
    assert overrides["position_sizing"] == "equal"
    assert overrides["max_per_industry"] == 2
    assert overrides["max_weight_per_stock"] == pytest.approx(0.20)
    assert overrides["sell_timing"] == "open"
    assert overrides["max_retry_count"] == 0
    assert overrides["max_retry_days"] == 0
    assert overrides["enable_position_completion"] is False
    assert overrides["enable_early_rebalance_on_empty"] is False
    assert overrides["enable_trailing_early_rebalance"] is False
    assert overrides["min_buy_value_ratio"] == 0.0


def test_slippage_override_changes_cost_model() -> None:
    base = native_replay.b_overrides()
    stressed = native_replay.b_overrides(slippage=spec.STRESS["slippage_doubled"])
    assert base["cost_model"].slippage == pytest.approx(0.0005)
    assert stressed["cost_model"].slippage == pytest.approx(0.0010)
    assert base["cost_model"].min_commission == pytest.approx(5.0)


def test_industry_cap_counts_holdings_and_batch() -> None:
    pit_mapping = {f"S{i}": "电子" for i in range(5)}
    pit_mapping.update({f"T{i}": "医药生物" for i in range(3)})
    sleeve = _Sleeve(
        positions={"S0": {"shares": 100}, "T0": {"shares": 100}},
        mapping=pit_mapping,
    )
    candidates = [("S1", 1.0), ("S2", 0.9), ("T1", 0.8), ("T2", 0.7)]
    kept = sleeve._post_filter_candidates(candidates, pd.Timestamp("2024-01-02"))
    # 电子已持 1 只，只能再进 1 只；医药已持 1 只，只能再进 1 只
    assert kept == [("S1", 1.0), ("T1", 0.8)]


def test_unknown_industry_candidates_dropped() -> None:
    sleeve = _Sleeve(positions={}, mapping={"A": "电子"})
    kept = sleeve._post_filter_candidates([("A", 1.0), ("B", 0.9)], pd.Timestamp("2024-01-02"))
    assert kept == [("A", 1.0)]
    assert sleeve._hook_stats.get("dropped_unknown") == 1


def test_industry_mapping_handles_open_interval() -> None:
    pit = pd.DataFrame(
        {
            "ts_code": ["A", "B", "C"],
            "l1_code": ["270000", "440000", "270000"],
            "in_date": ["20200101", "20200101", "20230101"],
            "out_date": [None, "20220101", "20241231"],
        }
    )
    names = {"270000": "电子"}
    mapping = native_replay.industry_mapping_for_day(pit, names, "20240601")
    assert mapping["A"] == "电子"
    assert "B" not in mapping  # 已退出区间
    assert mapping["C"] == "电子"
    assert mapping.get("A") == "电子"


def test_ad_signal_top_n_is_six() -> None:
    signal = native_replay.ADSignal({"20240102": [("A", 1.0)]})
    assert signal.top_n == 6
    assert signal.generate_ranked(pd.Timestamp("2024-01-02"), None, None) == [("A", 1.0)]
    assert signal.generate(pd.Timestamp("2024-01-03"), None, None) == {}


def test_combine_arms_requires_aligned_dates_and_keeps_a100() -> None:
    index = pd.Index(["20240102", "20240103"])
    a100 = pd.Series([1_000_000.0, 1_010_000.0], index=index)
    a80 = pd.Series([800_000.0, 808_000.0], index=index)
    sleeve = pd.Series([200_000.0, 205_000.0], index=index)
    arms = native_replay.combine_arms(1_000_000, a100, a80, {"AD": sleeve}, 200_000.0)
    assert arms["A100"].tolist() == a100.tolist()
    assert arms["A80_CASH20"].tolist() == [1_000_000.0, 1_008_000.0]
    assert arms["A80_AD20"].tolist() == [1_000_000.0, 1_013_000.0]

    broken = pd.Series([200_000.0], index=["20240102"])
    with pytest.raises(native_replay.ReplayError):
        native_replay.combine_arms(1_000_000, a100, a80, {"AD": broken}, 200_000.0)


def test_combine_arms_rejects_a80_a100_mismatch() -> None:
    a100 = pd.Series([1.0, 2.0], index=["20240102", "20240103"])
    a80 = pd.Series([1.0], index=["20240102"])
    sleeve = pd.Series([1.0, 1.0], index=a100.index)
    with pytest.raises(native_replay.ReplayError):
        native_replay.combine_arms(1_000_000, a100, a80, {"AD": sleeve}, 200_000.0)


def test_spec_choices_are_protocol_derived_not_deviations() -> None:
    ids = {item["id"] for item in native_replay.SPEC_CHOICES}
    assert ids == {"no_cross_day_retry", "single_execution_window"}
    for item in native_replay.SPEC_CHOICES:
        assert item["source"] == "协议 §10.2"


def test_load_a_series_missing_raises(tmp_path) -> None:
    with pytest.raises(native_replay.ReplayError):
        native_replay.load_a_series(tmp_path, 1_000_000, "A100")
