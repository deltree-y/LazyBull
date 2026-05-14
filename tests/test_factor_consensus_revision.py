"""一致预期修正因子双口径兼容测试。"""

import pandas as pd

from src.lazybull.factors.consensus_revision import (
    CONSENSUS_REVISION_COLS,
    CONSENSUS_REVISION_FRESHNESS_COL,
    build_consensus_revision_lookup_by_date,
)


def test_consensus_revision_supports_rec_schema():
    trade_date = "20240401"
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 6,
            "report_date": ["20231220", "20240115", "20240201", "20240220", "20240301", "20240320"],
            "rec_fore_Netprofit": [90.0, 95.0, 100.0, 110.0, 120.0, 140.0],
        }
    )

    result = build_consensus_revision_lookup_by_date(
        report_rc,
        [trade_date],
    )

    assert trade_date in result
    row = result[trade_date].loc[result[trade_date]["ts_code"] == "000001.SZ"].iloc[0]
    assert set(CONSENSUS_REVISION_COLS).issubset(result[trade_date].columns)
    assert row[CONSENSUS_REVISION_FRESHNESS_COL] >= 0
    assert pd.notna(row["cons_eps_dispersion"])
    assert pd.notna(row["cons_analyst_count_chg"])


def test_consensus_revision_supports_np_schema():
    trade_date = "20240401"
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 6,
            "report_date": ["20231220", "20240115", "20240201", "20240220", "20240301", "20240320"],
            "np": [88.0, 92.0, 96.0, 108.0, 118.0, 132.0],
        }
    )

    result = build_consensus_revision_lookup_by_date(
        report_rc,
        [trade_date],
    )

    assert trade_date in result
    row = result[trade_date].loc[result[trade_date]["ts_code"] == "000001.SZ"].iloc[0]
    assert row[CONSENSUS_REVISION_FRESHNESS_COL] >= 0
    assert pd.notna(row["cons_eps_dispersion"])
    assert pd.notna(row["cons_analyst_count_chg"])


def test_consensus_revision_analyst_count_chg_uses_density_vs_history():
    trade_date = "20240401"
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 5,
            "report_date": ["20231220", "20240115", "20240220", "20240301", "20240320"],
            "np": [90.0, 95.0, 110.0, 120.0, 140.0],
        }
    )

    result = build_consensus_revision_lookup_by_date(
        report_rc,
        [trade_date],
    )

    assert trade_date in result
    row = result[trade_date].loc[result[trade_date]["ts_code"] == "000001.SZ"].iloc[0]
    # 近 90 日共有 3 条，前序 90 日共有 2 条，因此覆盖密度变化为 (3/90 - 2/90) / (2/90) = 0.5
    assert abs(row["cons_analyst_count_chg"] - 0.5) < 1e-6
