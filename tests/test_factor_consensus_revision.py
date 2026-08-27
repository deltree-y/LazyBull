"""一致预期修正因子 v2 语义测试。"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.consensus_revision import (
    CONSENSUS_REVISION_COLS,
    CONSENSUS_REVISION_FRESHNESS_COL,
    CONSENSUS_REVISION_VERSION_COL,
    _winsorize_cross_section,
)
from src.lazybull.factors.consensus_revision import (
    build_consensus_revision_lookup_by_date as _build_consensus_revision_lookup_by_date,
)


def _with_report_identity(report_rc_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """为因子数值测试补齐真实 report_rc 身份 schema（quarter 默认 FY1）。"""
    if report_rc_df is None or len(report_rc_df) == 0:
        return report_rc_df
    result = report_rc_df.copy()
    report_dates = result["report_date"].astype("string").str.replace("-", "", regex=False)
    if "org_name" not in result.columns:
        result["org_name"] = "测试机构"
    if "author_name" not in result.columns:
        result["author_name"] = "测试分析师"
    if "report_title" not in result.columns:
        result["report_title"] = (
            "测试研报-" + result["ts_code"].astype("string") + "-" + report_dates
        )
    if "quarter" not in result.columns:
        next_year = (report_dates.str[:4].astype(int) + 1).astype(str)
        result["quarter"] = next_year + "Q4"
    return result


def build_consensus_revision_lookup_by_date(
    report_rc_df: Optional[pd.DataFrame],
    trading_dates: List[str],
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """使用完整身份测试数据调用生产修正因子入口。"""
    return _build_consensus_revision_lookup_by_date(
        _with_report_identity(report_rc_df),
        trading_dates,
        daily_data_lookup=daily_data_lookup,
    )


def _row(
    report_date: str,
    quarter: str,
    eps: float,
    rating: Optional[str] = None,
    title: str = "",
) -> dict:
    """构造一条完整身份的研报行；title 用于区分同日多份研报。"""
    row = {
        "ts_code": "000001.SZ",
        "report_date": report_date,
        "org_name": f"机构{title}",
        "author_name": f"分析师{title}",
        "report_title": f"研报{title}-{report_date}",
        "quarter": quarter,
        "eps": eps,
    }
    if rating is not None:
        row["rating"] = rating
    return row


def test_consensus_revision_outputs_v2_schema_and_sentinel():
    trade_date = "20240401"
    rows = []
    for idx, (report_date, fy_eps) in enumerate(
        [
            ("20240201", 10.0),
            ("20240201", 12.0),
            ("20240301", 11.0),
            ("20240301", 13.0),
            ("20240320", 12.0),
            ("20240320", 14.0),
        ]
    ):
        rows.append(_row(report_date, "2025Q4", fy_eps, title=str(idx)))

    result = build_consensus_revision_lookup_by_date(pd.DataFrame(rows), [trade_date])

    assert trade_date in result
    cols = set(result[trade_date].columns)
    for col in CONSENSUS_REVISION_COLS:
        assert col in cols
    assert CONSENSUS_REVISION_FRESHNESS_COL in cols
    assert "cons_revision_target_upside" not in cols
    assert (result[trade_date][CONSENSUS_REVISION_VERSION_COL] == 2).all()


def test_consensus_revision_eps_prefers_fy1_and_does_not_mix_fy0():
    """同一报告日多预测期行：EPS 指标只取 FY1，不混入量级不同的 FY0。"""
    rows = []
    idx = 0
    for report_date, fy1_eps in (("20240201", 10.0), ("20240301", 11.0), ("20240320", 12.0)):
        # FY1：eps 量级 10
        rows.append(_row(report_date, "2025Q4", fy1_eps, title=str(idx)))
        idx += 1
        rows.append(_row(report_date, "2025Q4", fy1_eps + 2.0, title=str(idx)))
        idx += 1
        # FY0：eps 量级 100（若混入，分歧度会完全偏离）
        rows.append(_row(report_date, "2024Q4", 100.0, title=str(idx)))
        idx += 1
        rows.append(_row(report_date, "2024Q4", 120.0, title=str(idx)))
        idx += 1

    result = build_consensus_revision_lookup_by_date(pd.DataFrame(rows), ["20240401"])

    row = result["20240401"].iloc[0]
    # 仅 FY1：日分歧度依次为 1.414/11、1.414/12、1.414/13，均值约 0.118
    # 若混入 FY0（量级 100），日分歧度将接近 1.0
    assert pd.notna(row["cons_eps_dispersion"])
    assert 0.10 < row["cons_eps_dispersion"] < 0.13


def test_consensus_revision_accel_uses_calendar_time():
    """修正速度按报告日真实日历时间拟合，而非研报行序号或 yyyymmdd 整数。"""
    dates = ["20240102", "20240103", "20240228", "20240315", "20240330"]
    eps_values = [10.0, 10.1, 10.5, 10.8, 11.0]
    rows = [_row(d, "2025Q4", e, title=str(i)) for i, (d, e) in enumerate(zip(dates, eps_values))]

    result = build_consensus_revision_lookup_by_date(pd.DataFrame(rows), ["20240401"])

    row = result["20240401"].iloc[0]
    days = (pd.to_datetime(dates, format="%Y%m%d") - pd.Timestamp("1970-01-01")).days.to_numpy(
        dtype=float
    )
    medians = np.array(eps_values, dtype=float)
    expected = float(np.polyfit(days, medians, 1)[0] / np.mean(medians))
    # 若误用 yyyymmdd 整数回归，跨月/跨年日期距离会严重失真（约仅为真实日率的 37%）
    wrong_days = np.array([int(d) for d in dates], dtype=float)
    wrong_expected = float(np.polyfit(wrong_days, medians, 1)[0] / np.mean(medians))
    assert abs(expected - wrong_expected) / max(abs(expected), 1e-12) > 0.5
    assert pd.notna(row["cons_eps_revision_accel"])
    assert abs(row["cons_eps_revision_accel"] - expected) < 1e-12


def test_consensus_revision_rating_upgrade_ratio_reads_rating():
    """评级上调占比真实读取 rating 列，不再借用目标价变化。"""
    earlier = [("20240105", "买入"), ("20240120", "增持"), ("20240205", "增持")]
    recent30 = [("20240310", "买入"), ("20240320", "中性")]
    rows = [
        _row(d, "2025Q4", 1.0, rating=r, title=str(i))
        for i, (d, r) in enumerate(earlier + recent30)
    ]

    result = build_consensus_revision_lookup_by_date(pd.DataFrame(rows), ["20240401"])

    row = result["20240401"].iloc[0]
    # 基线 (5+4+4)/3 ≈ 4.33；近 30 日 [5, 3] 中仅 5 高于基线 → 1/2
    assert abs(row["cons_rating_upgrade_ratio"] - 0.5) < 1e-12


def test_consensus_revision_counts_unique_reports_not_forecast_rows():
    """覆盖变化按唯一研报计数，同研报多预测期不放大，且按 30/90 日折算。"""
    earlier_dates = ["20240105", "20240115", "20240125", "20240201", "20240210"]
    recent_dates = ["20240305", "20240310", "20240315", "20240320"]
    rows = []
    idx = 0
    for report_date in earlier_dates + recent_dates:
        title = str(idx)
        idx += 1
        for quarter in ("2024Q4", "2025Q4"):
            rows.append(_row(report_date, quarter, 1.0, title=title))

    result = build_consensus_revision_lookup_by_date(pd.DataFrame(rows), ["20240401"])

    row = result["20240401"].iloc[0]
    # 基线 = 5 份研报 / 3 = 1.667，近 30 日 4 份 → (4-1.667)/1.667 = 1.4
    assert abs(row["cons_analyst_count_chg"] - 1.4) < 1e-9


def test_consensus_revision_absolute_fiscal_year_no_cross_year_mix():
    """绝对财年隔离：锚定 2025 年时，2024 年报告的 FY1(2025) 与 2025 年报告的
    FY1(2026) 不得混入同一序列。"""
    rows = []
    idx = 0
    # 2024 年报告的 FY1=2025（绝对财年 2025）：量级 100，若混入会完全改变分歧度
    rows.append(_row("20241220", "2025Q4", 100.0, title=str(idx)))
    idx += 1
    rows.append(_row("20241220", "2025Q4", 120.0, title=str(idx)))
    idx += 1
    # 2025 年报告的 FY1=2026（绝对财年 2026）：量级 10，三个报告日各两份研报
    for report_date, base in (("20250115", 10.0), ("20250210", 11.0), ("20250310", 12.0)):
        rows.append(_row(report_date, "2026Q4", base, title=str(idx)))
        idx += 1
        rows.append(_row(report_date, "2026Q4", base + 2.0, title=str(idx)))
        idx += 1

    result = build_consensus_revision_lookup_by_date(pd.DataFrame(rows), ["20250320"])

    row = result["20250320"].iloc[0]
    # 仅绝对财年 2026：1.414/11、1.414/12、1.414/13 → 均值约 0.118
    # 若混入绝对财年 2025（量级 100），日分歧度将接近 1.0
    assert pd.notna(row["cons_eps_dispersion"])
    assert 0.10 < row["cons_eps_dispersion"] < 0.13


def test_consensus_revision_shuffled_input_does_not_leak_future_reports():
    """乱序输入的研报数据必须先排序再窗口定位，未来研报不得计入历史窗口（PIT）。"""
    ordered_rows = [
        _row("20240115", "2025Q4", 1.0, title="a"),
        _row("20240201", "2025Q4", 1.0, title="b"),
        _row("20240301", "2025Q4", 1.0, title="c"),
        _row("20240320", "2025Q4", 1.0, title="d"),
    ]
    shuffled_rows = [ordered_rows[3], ordered_rows[0], ordered_rows[2], ordered_rows[1]]

    ordered = build_consensus_revision_lookup_by_date(pd.DataFrame(ordered_rows), ["20240310"])
    shuffled = build_consensus_revision_lookup_by_date(pd.DataFrame(shuffled_rows), ["20240310"])

    # 20240310 当天只有 20240115/20240201/20240301 三份研报可见，20240320 为未来
    row_ordered = ordered["20240310"].iloc[0]
    row_shuffled = shuffled["20240310"].iloc[0]
    assert (
        abs(row_ordered["cons_analyst_count_chg"] - row_shuffled["cons_analyst_count_chg"]) < 1e-12
    )
    assert (
        row_ordered["cons_revision_freshness_days"] == row_shuffled["cons_revision_freshness_days"]
    )
    # 乱序输入下窗口定位必须与有序输入完全一致（都只看到 3 份历史研报）
    assert (
        abs(row_shuffled["cons_analyst_count_chg"] - row_ordered["cons_analyst_count_chg"]) < 1e-12
    )


def test_consensus_revision_state_kept_for_365_days_from_latest_report():
    """修正状态锚定最新研报保留 365 日，不在第 90 日硬消失。"""
    report_rc = pd.DataFrame(
        [
            _row("20240102", "2025Q4", 90.0, title="a"),
            _row("20240115", "2025Q4", 100.0, title="b"),
            _row("20240201", "2025Q4", 110.0, title="c"),
        ]
    )
    kept_date = "20240515"
    expired_date = "20250202"

    result = build_consensus_revision_lookup_by_date(report_rc, [kept_date, expired_date])

    assert kept_date in result
    assert result[kept_date].iloc[0][CONSENSUS_REVISION_FRESHNESS_COL] == 104
    assert expired_date not in result


def test_consensus_revision_eps_with_single_valid_value_does_not_warn():
    report_rc = pd.DataFrame(
        [
            _row("20240201", "2025Q4", 100.0, title="a"),
            _row("20240215", "2025Q4", float("nan"), title="b"),
            _row("20240301", "2025Q4", float("nan"), title="c"),
            _row("20240310", "2025Q4", float("nan"), title="d"),
            _row("20240320", "2025Q4", float("nan"), title="e"),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = build_consensus_revision_lookup_by_date(report_rc, ["20240401"])

    assert "20240401" in result
    row = result["20240401"].iloc[0]
    assert pd.isna(row["cons_eps_dispersion"])
    assert pd.isna(row["cons_eps_dispersion_chg"])
    assert pd.isna(row["cons_eps_revision_accel"])


def test_consensus_revision_rejects_incomplete_identity_schema():
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "report_date": ["20240320"],
            "eps": [1.0],
        }
    )

    with pytest.raises(ValueError, match="report_rc 身份 schema 不完整"):
        _build_consensus_revision_lookup_by_date(report_rc, ["20240401"])


def test_consensus_revision_missing_eps_column_raises():
    """v2 不回退净利润口径：缺 eps 列必须明确失败，不能伪装成无数据零因子。"""
    report_rc = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "report_date": "20240320",
                "org_name": "机构",
                "author_name": "分析师",
                "report_title": "研报",
                "quarter": "2024Q4",
                "np": 1.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="缺少 eps 列"):
        _build_consensus_revision_lookup_by_date(report_rc, ["20240401"])


def test_consensus_revision_prefers_fiscal_year_with_more_report_days():
    """覆盖日数多者优先：FY1 仅 3 日、FY0 有 5 日时选 FY0，accel 不因少量 FY1 压空。"""
    rows = []
    idx = 0
    # FY1（2025）：3 个报告日，各两份研报（同日双份可算 dispersion，但日数少）
    for report_date, base in (("20240201", 10.0), ("20240215", 10.2), ("20240301", 10.4)):
        rows.append(_row(report_date, "2025Q4", base, title=str(idx)))
        idx += 1
        rows.append(_row(report_date, "2025Q4", base + 2.0, title=str(idx)))
        idx += 1
    # FY0（2024）：5 个报告日，各一份研报（同日单份无 dispersion，但 accel 日数充足）
    for idx2, (report_date, eps) in enumerate(
        [
            ("20240110", 8.0),
            ("20240125", 8.2),
            ("20240210", 8.5),
            ("20240225", 8.8),
            ("20240315", 9.0),
        ]
    ):
        rows.append(_row(report_date, "2024Q4", eps, title=f"fy0_{idx2}"))

    result = build_consensus_revision_lookup_by_date(pd.DataFrame(rows), ["20240401"])

    row = result["20240401"].iloc[0]
    # 日数 5 的 FY0 胜出 → accel 由 FY0 序列计算，非 NaN
    assert pd.notna(row["cons_eps_revision_accel"])


def test_consensus_revision_winsorizes_cross_section():
    """截面 winsorize 裁剪极端值，NaN 保持不动。"""
    values = [0.1 + i * 0.01 for i in range(24)] + [10.0]
    df = pd.DataFrame({"cons_eps_dispersion": values})

    out = _winsorize_cross_section(df, ["cons_eps_dispersion"])

    assert out["cons_eps_dispersion"].isna().sum() == 0
    # 极端值 10.0 被裁剪到 99% 分位附近，不再等于原值
    assert out["cons_eps_dispersion"].max() < 10.0
    assert out["cons_eps_dispersion"].max() > 0.34


def test_live_consensus_revision_cols_includes_size_neutralized_derivatives():
    """存活列识别需覆盖基础 zscore 列与市值中性化 _sz 派生列。"""
    from src.lazybull.ml.walk_forward.summary import _live_consensus_revision_cols

    result = {
        "feature_columns": [
            "zscore_cons_analyst_count_chg",
            "zscore_cons_analyst_count_chg_sz",
            "zscore_cons_eps_dispersion",
            "zscore_pe_ttm",
            "cons_revision_freshness_days",
        ]
    }

    live = _live_consensus_revision_cols(result)

    assert live == (
        "zscore_cons_analyst_count_chg,zscore_cons_analyst_count_chg_sz,"
        "zscore_cons_eps_dispersion"
    )
