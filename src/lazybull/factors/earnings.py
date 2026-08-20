"""业绩预告因子模块

将事件驱动的业绩预告（forecast）数据
按 ann_date point-in-time 对齐到日频。

数据来源：
- Tushare forecast_vip API（5000 积分）：业绩预告

关键防前视：只用 ann_date（公告日），不用 end_date（报告期末）。
公告后前向填充，直到被同一报告期的新公告或下一报告期覆盖。
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.date_utils import normalize_series_to_yyyymmdd
from .announcement_utils import build_latest_announcement_lookup_by_date


# 业绩预告类型 → 评分
FORECAST_TYPE_SCORE = {
    "预增": 1.0,
    "略增": 0.5,
    "续盈": 0.3,
    "扭亏": 0.8,
    "不确定": 0.0,
    "略减": -0.5,
    "预减": -1.0,
    "首亏": -1.0,
    "续亏": -0.8,
}

EARNINGS_COLS = [
    "forecast_type_score",
    "forecast_chg_mid",
]

EARNINGS_FRESHNESS_COL = "forecast_freshness_days"


def build_earnings_lookup_by_date(
    forecast_df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """将业绩预告数据按 ann_date point-in-time 对齐到日频

    Args:
        forecast_df: 业绩预告 DataFrame，需包含
            ts_code, ann_date, end_date, type, p_change_min, p_change_max
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序）

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, forecast_type_score, ...)}
    """
    # ── 处理业绩预告 ────────────────────────────────────────────
    if forecast_df is not None and len(forecast_df) > 0:
        fc = forecast_df.copy()
        fc["ann_date"] = normalize_series_to_yyyymmdd(fc["ann_date"])
        fc = fc.dropna(subset=["ann_date"])

        # 类型评分：未知/缺失类型保留 NaN（与"不确定"评分 0.0 区分，
        # 由 XGBoost/LightGBM 原生 NaN 处理学习"无预告类型"的缺失方向）
        if "type" in fc.columns:
            fc["forecast_type_score"] = fc["type"].map(FORECAST_TYPE_SCORE)
        else:
            fc["forecast_type_score"] = np.nan

        # 变动幅度中值
        for col in ["p_change_min", "p_change_max"]:
            if col in fc.columns:
                fc[col] = pd.to_numeric(fc[col], errors="coerce")
        if "p_change_min" in fc.columns and "p_change_max" in fc.columns:
            fc["forecast_chg_mid"] = (fc["p_change_min"] + fc["p_change_max"]) / 2
        else:
            fc["forecast_chg_mid"] = np.nan

        # 仅去除完全重复记录，保留同一报告期多次公告版本，
        # 由 PIT 查询按交易日选择（end_col 模式下同报告期取最新修正版）
        fc = fc.sort_values(["ts_code", "end_date", "ann_date"])
        fc = fc.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")
        fc = fc.sort_values(["ts_code", "ann_date"])

        factor_df = fc[["ts_code", "ann_date", "end_date"] + EARNINGS_COLS].copy()
        return build_latest_announcement_lookup_by_date(
            factor_df,
            trading_dates,
            value_cols=EARNINGS_COLS,
            freshness_col=EARNINGS_FRESHNESS_COL,
            end_col="end_date",
            log_name="业绩预告",
        )

    return {}
