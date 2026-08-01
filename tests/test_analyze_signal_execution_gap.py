import pandas as pd
import pytest

from scripts.ana.analyze_signal_execution_gap import (
    build_failure_reasons,
    enrich_execution_with_labels,
    summarize_execution_gap,
)


def _execution() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split_index": 6,
                "signal_date": "2022-01-03",
                "execution_date": "2022-01-04",
                "planned_stock": "000001.SZ",
                "actual_stock": "000001.SZ",
                "actual_rank": 1,
                "status": "filled",
                "reason": None,
                "signal_to_buy_return": 0.01,
            },
            {
                "split_index": 6,
                "signal_date": "2022-01-03",
                "execution_date": "2022-01-04",
                "planned_stock": "000002.SZ",
                "actual_stock": "000003.SZ",
                "actual_rank": 31,
                "status": "filled",
                "reason": None,
                "signal_to_buy_return": -0.01,
            },
            {
                "split_index": 6,
                "signal_date": "2022-01-03",
                "execution_date": "2022-01-04",
                "planned_stock": "000004.SZ",
                "actual_stock": None,
                "actual_rank": None,
                "status": "unfilled",
                "reason": "涨停",
                "signal_to_buy_return": None,
            },
        ]
    )


def _topk() -> pd.DataFrame:
    rows = []
    for rank in range(1, 32):
        rows.append(
            {
                "split_index": 6,
                "trade_date": 20220103,
                "topk": 30 if rank <= 30 else 300,
                "rank": rank,
                "ts_code": f"{rank:06d}.SZ",
                "true_return": 0.02 if rank <= 30 else -0.03,
            }
        )
    return pd.DataFrame(rows)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split_index": 6,
                "action": "sell",
                "signal_date": "2022-01-03",
                "buy_date": "2022-01-04",
                "date": "2022-01-24",
                "pnl_profit_pct": 0.01,
            },
            {
                "split_index": 6,
                "action": "sell",
                "signal_date": "2022-01-03",
                "buy_date": "2022-01-04",
                "date": "2022-01-24",
                "pnl_profit_pct": -0.02,
            },
        ]
    )


def test_enrich_execution_uses_largest_topk_snapshot() -> None:
    enriched = enrich_execution_with_labels(_execution(), _topk())

    first = enriched[enriched["actual_stock"] == "000001.SZ"].iloc[0]
    replacement = enriched[enriched["actual_stock"] == "000003.SZ"].iloc[0]
    assert first["actual_label_return"] == pytest.approx(0.02)
    assert replacement["actual_label_return"] == pytest.approx(0.02)


def test_summarize_execution_gap_separates_signal_execution_and_holding() -> None:
    summary = summarize_execution_gap(_execution(), _trades(), _topk()).iloc[0]

    assert summary["planned_slots"] == 3
    assert summary["filled_slots"] == 2
    assert summary["fill_rate"] == pytest.approx(2 / 3)
    assert summary["replacement_rate"] == pytest.approx(0.5)
    assert summary["top30_buy_coverage"] == pytest.approx(0.5)
    assert summary["signal_day_top30_hit_rate"] == pytest.approx(1.0)
    assert summary["holding_return_mean"] == pytest.approx(-0.005)


def test_build_failure_reasons_counts_by_split() -> None:
    failures = build_failure_reasons(_execution())

    assert failures.to_dict("records") == [
        {"split_index": 6, "status": "unfilled", "reason": "涨停", "count": 1}
    ]
