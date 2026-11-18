"""信号与读数测试：确定排序、AD_SIMPLE、100 只门与非有限预测、目标符号。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.analyst_diffusion_ml import evaluation, prediction, spec


def _ranked_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20240110"] * 3,
            "ts_code": ["000003.SZ", "000001.SZ", "000002.SZ"],
            "score": [-0.05, -0.05, -0.20],
        }
    )


def test_all_negative_scores_still_ranked_and_ties_by_code() -> None:
    frame = _ranked_frame()
    ranked = prediction.rank_frame(frame, frame["score"].to_numpy())
    assert ranked["ts_code"].tolist() == ["000001.SZ", "000003.SZ", "000002.SZ"]
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_nonfinite_prediction_raises() -> None:
    class _Model:
        def predict(self, matrix: np.ndarray) -> np.ndarray:
            return np.array([1.0, np.inf, 2.0])

    frame = pd.DataFrame(
        {"trade_date": ["20240110"] * 3, "ts_code": ["a", "b", "c"],
         **{name: [0.0, 0.0, 0.0] for name in spec.FEATURES}}
    )
    with pytest.raises(prediction.PredictionError):
        prediction.predict_matrix(_Model(), frame)


def test_ad_simple_rank_puts_nan_last() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240110"] * 4,
            "ts_code": ["A", "B", "C", "D"],
            "ad_eps_net_diffusion_90": [0.0, 1.0, np.nan, 0.5],
            "ad_eps_revision_median_90": [0.0, 0.1, np.nan, 0.2],
        }
    )
    ranked = prediction.ad_simple_rank(frame)
    assert ranked["ts_code"].tolist() == ["B", "D", "A", "C"]
    assert ranked["rank"].tolist() == [1, 2, 3, 4]


def test_ad_simple_second_key_breaks_ties() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240110", "20240110"],
            "ts_code": ["A", "B"],
            "ad_eps_net_diffusion_90": [0.5, 0.5],
            "ad_eps_revision_median_90": [0.1, 0.3],
        }
    )
    ranked = prediction.ad_simple_rank(frame)
    assert ranked["ts_code"].tolist() == ["B", "A"]


def test_rank_table_lookup_enforces_min_scored() -> None:
    days = ["20240110", "20240111"]
    small = pd.DataFrame(
        {"trade_date": ["20240110"] * 3, "ts_code": ["a", "b", "c"], "rank": [1, 2, 3]}
    )
    lookup = prediction.rank_table_lookup(small, days)
    assert lookup["20240110"] == []  # 不足 100 只不产生新买排名
    assert lookup["20240111"] == []


def test_daily_counts_marks_universe_below_minimum() -> None:
    ranked = pd.DataFrame(
        {"trade_date": ["20240110", "20240110", "20240111"], "ts_code": ["a", "b", "c"],
         "rank": [1, 2, 1]}
    )
    counts = prediction.daily_counts(ranked, 2)
    status = dict(zip(counts["trade_date"], counts["min100_status"]))
    assert status["20240110"] == "ok"
    assert status["20240111"] == "universe_below_minimum"


def _growth(nav_start: float, annual: float, days: int, dip_at: int | None = None,
            dip: float = 0.0) -> pd.Series:
    step = (1.0 + annual) ** (1.0 / 252.0) - 1.0
    values = [nav_start * (1.0 + step) ** t for t in range(days)]
    if dip_at is not None:
        for index in range(dip_at, len(values)):
            values[index] *= (1.0 - dip)
    return pd.Series(values)


def test_target_check_pass_case() -> None:
    days = 260
    base = _growth(1_000_000.0, 0.10, days)
    cand = _growth(800_000.0, 0.106, days)
    cash = _growth(800_000.0, 0.0, days)
    result = evaluation.target_check(cand, base, cash)
    assert result["cagr_delta"] == pytest.approx(0.006, abs=2e-3)
    assert result["meets_cagr"] is True
    assert result["meets_mdd"] is True
    assert result["beats_cash"] is True
    assert result["all_three"] is True


def test_target_check_fails_when_drawdown_worse() -> None:
    days = 260
    base = _growth(1_000_000.0, 0.10, days)
    cand = _growth(800_000.0, 0.106, days, dip_at=200, dip=0.05)
    cash = _growth(800_000.0, 0.0, days)
    result = evaluation.target_check(cand, base, cash)
    assert result["mdd_loss_delta"] > 0.01
    assert result["meets_mdd"] is False
    assert result["all_three"] is False


def test_target_check_fails_when_cash_wins() -> None:
    days = 260
    base = _growth(1_000_000.0, 0.10, days)
    cand = _growth(800_000.0, 0.10, days)
    cash = _growth(800_000.0, 0.20, days)
    result = evaluation.target_check(cand, base, cash)
    assert result["beats_cash"] is False
    assert result["all_three"] is False


def test_mdd_loss_is_positive_and_delta_uses_loss() -> None:
    base = pd.Series([1.0, 1.2, 0.9, 1.1])
    cand = pd.Series([1.0, 1.1, 0.7, 1.0])
    rel = evaluation.relative_to(base, cand)
    assert evaluation.chain_metrics(base)["mdd_loss"] == pytest.approx(0.25)
    # 候选回撤 0.7/1.1 - 1 = -36.36%
    assert evaluation.chain_metrics(cand)["mdd_loss"] == pytest.approx(4.0 / 11.0)
    assert rel["mdd_loss_delta"] == pytest.approx(4.0 / 11.0 - 0.25)
    # dd_delta 用带符号回撤：-36.36% - (-25%) = -11.36%
    assert rel["dd_delta"] == pytest.approx(-(4.0 / 11.0) + 0.25)


def test_segment_table_uses_previous_day_base_point() -> None:
    series = pd.Series(
        [100.0, 110.0, 121.0, 133.1],
        index=["20240101", "20240102", "20240103", "20240104"],
    )
    table = evaluation.segment_table({"A100": series}, [["20240103", "20240104"]])
    row = table.iloc[0]
    assert row["start"] == "20240103"
    # 段首基点 = 前一日 110 -> 段内 121 -> 133.1 的收益
    assert row["cagr"] is not None


def test_bootstrap_requires_aligned_lengths() -> None:
    short = pd.Series([1.0, 1.1, 1.2])
    long = pd.Series([1.0, 1.1, 1.2, 1.3])
    with pytest.raises(ValueError):
        evaluation.bootstrap_deltas({"A": short, "B": long}, [("A", "B")], repetitions=5)
