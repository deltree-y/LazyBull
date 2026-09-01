"""Walk-forward 串联净值指标测试。"""

import numpy as np
import pandas as pd
import pytest

from scripts.compare.loading import load_chain_metrics
from src.lazybull.ml.walk_forward.chain_metrics import calculate_chain_metrics


def _build_one_year_two_split_chain() -> pd.DataFrame:
    half_year_growth = np.sqrt(1.10)
    first_split = np.geomspace(1.0, half_year_growth, 127)
    second_split = np.geomspace(half_year_growth, 1.10, 127)
    return pd.DataFrame(
        {
            "nav": np.concatenate([first_split, second_split]),
            "split_index": [0] * len(first_split) + [1] * len(second_split),
        }
    )


def test_chain_metrics_use_geometric_cagr_and_exclude_split_boundaries() -> None:
    """每个 split 重复的起始净值点不应计入有效收益区间。"""
    metrics = calculate_chain_metrics(_build_one_year_two_split_chain())

    assert metrics["trading_days"] == 252
    assert metrics["total_return"] == pytest.approx(0.10)
    assert metrics["cagr"] == pytest.approx(0.10)


def test_chain_metrics_sharpe_uses_daily_excess_returns() -> None:
    """Sharpe 应使用日收益均值和等效日无风险收益率。"""
    chain_df = pd.DataFrame(
        {
            "nav": [1.0, 1.01, 1.0, 1.0, 1.02, 1.01],
            "split_index": [0, 0, 0, 1, 1, 1],
        }
    )
    expected_returns = pd.Series([0.01, 1.0 / 1.01 - 1, 0.02, 1.01 / 1.02 - 1])
    daily_risk_free_rate = 1.03 ** (1 / 252) - 1
    expected_sharpe = (
        (expected_returns.mean() - daily_risk_free_rate) / expected_returns.std() * np.sqrt(252)
    )

    metrics = calculate_chain_metrics(chain_df)

    assert metrics["trading_days"] == 4
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)


def test_compare_loader_uses_shared_chain_metrics(tmp_path) -> None:
    """对比表加载器应采用与 WF 日志一致的几何年化口径。"""
    chain_df = _build_one_year_two_split_chain()
    chain_df.to_csv(tmp_path / "chain_nav_test_run.csv", index=False, encoding="utf-8-sig")

    metrics = load_chain_metrics(tmp_path, "test_run")

    assert metrics["chain_trading_days"] == 252
    assert metrics["chain_total_return"] == pytest.approx(0.10)
    assert metrics["chain_cagr"] == pytest.approx(0.10)
