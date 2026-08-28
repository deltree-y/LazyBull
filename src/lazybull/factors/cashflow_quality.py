"""现金流量表质量因子模块（v3：版本化 PIT + 事件驱动 TTM）

将季度频率的现金流量表数据（cashflow）前向填充到日频，构建每日现金流质量查询表。

v3 核心修正（2026-08 审计）：
- 版本化 PIT：数据可用时间取 f_ann_date（实际公告日，缺失回退 ann_date），
  修订记录不再回填到原始公告日（消除训练数据前视污染）；
- 去重键 (ts_code, end_date, f_ann_date)：同报告期的多次修订按版本保留，
  由 PIT 查询按交易日选择当日可见的最新版本；
- 事件驱动 TTM：当前期及两个历史依赖期的修订都会在可用日重算当前 TTM，
    避免依赖期晚到修订后最新报告期仍冻结旧值；
- 确定性去重：同一版本键冲突时按 TuShare `update_flag=1` 最新语义选择，
    不再由接口或 parquet 行顺序决定供应商 FCF；
- 自由现金流直接采用 TuShare `free_cashflow` 字段（供应商口径），
  不再使用本地 OCF-|capex| 代理混同口径；
- 数值稳定性：分母经济尺度下限 + 比值有界裁剪，替代仅 1e-6 元的弱过滤。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..data.financial_statement_versions import deduplicate_prefer_latest_update_flag
from .announcement_utils import build_latest_announcement_lookup_by_date

# 现金流因子输出列（前向填充后的日频列名）
CASHFLOW_COLS = [
    "ocf",  # TTM 经营活动现金流净额
    "ocf_to_revenue",  # TTM OCF / TTM 销售商品提供劳务收到的现金（现金含量）
    "ocf_to_profit",  # TTM OCF / TTM 净利润（利润质量；Q1/Q3 覆盖极低）
    "fcf",  # TTM 自由现金流（TuShare free_cashflow 供应商口径）
    "fcf_yield",  # FCF / 总市值（现金回报率，handler 层计算）
    "capex_to_ocf",  # TTM 资本支出 / TTM OCF
]

CASHFLOW_FRESHNESS_COL = "cashflow_freshness_days"

# 现金流质量因子 schema 哨兵：v3 = 依赖修订事件驱动 TTM + 确定性版本去重。
# handler 对当日全截面恒写当前版本号（含无数据股票），训练入口校验
# 哨兵列缺失、NaN 或版本不符必须失败；语义重做时递增哨兵值。
CASHFLOW_QUALITY_SCHEMA_VERSION = 3
CASHFLOW_QUALITY_VERSION_COL = "cashflow_quality_schema_v2"

# ── 数值稳定性参数 ───────────────────────────────────────────
# 分母最小绝对值（元）。低于该经济尺度（1000 万元）视为不可信，比值置 NaN。
_MIN_ABS_DENOMINATOR = 1e7
# 比值有界裁剪（保持符号，裁剪极端幅度；负 OCF/负利润的方向仍保留）
_CLIP_OCF_TO_REVENUE = (-10.0, 10.0)
_CLIP_OCF_TO_PROFIT = (-50.0, 50.0)
_CLIP_CAPEX_TO_OCF = (-50.0, 50.0)
# fcf_yield 在 handler 层计算：总市值分母下限（元，1 亿元）与裁剪界
_MIN_ABS_TOTAL_MV_YUAN = 1e8
_CLIP_FCF_YIELD = (-1.0, 1.0)

# TTM 计算所需的累计期数值列
_CUM_NUMERIC_COLS = [
    "n_cashflow_act",  # 经营活动现金流净额（累计）
    "c_pay_acq_const_fiolta",  # 购建固定资产等资本支出（累计）
    "c_fr_sale_sg",  # 销售商品、提供劳务收到的现金（累计）
    "net_profit",  # 净利润（累计）
    "free_cashflow",  # 企业自由现金流（累计，供应商口径）
]


def _normalize_date(value) -> Optional[str]:
    """标准化日期为 YYYYMMDD 字符串（无法解析返回 None）。"""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.strftime("%Y%m%d")
    text = str(value).strip().replace("-", "")
    if not text or text == "nan":
        return None
    return text[:8] if len(text) >= 8 else None


def _build_ttm_table(cashflow_raw: pd.DataFrame) -> pd.DataFrame:
    """把累计期现金流量表原始行转换为版本化 TTM 表。

    输入需包含 ts_code/ann_date/end_date（f_ann_date 可选），
    输出含 ts_code、end_date、avail_date（数据可用时间）与 TTM 因子列。

    TTM 推导（q_y 为 y 年第 q 季度）：
        TTM(q_y) = cum(q_y) - cum(q_{y-1}) + cum(Q4_{y-1})
    每个目标报告期在自身、去年同季度或去年 Q4 任一版本可用时生成事件快照，
    各累计值均按事件日之前最近可见版本解析。依赖期晚到修订因此会重算当前
    报告期 TTM，但不会跨修订可用时间向历史回填。
    Q4 行两项相消退化为 cum(Q4_y)（年报即全年累计，与公式自洽）。
    """
    work = cashflow_raw.copy()
    for col in ("ann_date", "end_date", "f_ann_date"):
        if col in work.columns:
            work[col] = work[col].map(_normalize_date)
    work = work.dropna(subset=["ts_code", "ann_date", "end_date"])

    # 可用时间：f_ann_date（实际公告日）缺失回退 ann_date
    avail_col = "f_ann_date" if "f_ann_date" in work.columns else "ann_date"
    work["avail_date"] = work[avail_col].fillna(work["ann_date"])
    work = work.dropna(subset=["avail_date"])

    cum_cols_present = [c for c in _CUM_NUMERIC_COLS if c in work.columns]
    for col in cum_cols_present:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # 同一实际公告日缺少更细的修订时刻，按 TuShare 官方 update_flag=1 最新
    # 语义确定性去重，禁止由接口或 parquet 行顺序决定供应商 FCF 等字段。
    work = deduplicate_prefer_latest_update_flag(
        work,
        ["ts_code", "end_date", "avail_date"],
        deterministic_ties=True,
    )

    if len(work) == 0:
        return work

    # 目标报告期：去年同季度与去年 Q4
    work["_year"] = work["end_date"].str[:4].astype(int)
    work["_mmdd"] = work["end_date"].str[4:8]
    work["_q"] = work["end_date"].str[4:6]
    work["_q_last_year"] = (work["_year"] - 1).astype(str) + work["_mmdd"]
    work["_q4_last_year"] = (work["_year"] - 1).astype(str) + "1231"
    work["_avail_num"] = work["avail_date"].astype(int)

    targets = work[["ts_code", "end_date", "_q", "_q_last_year", "_q4_last_year"]].drop_duplicates()
    targets = targets.rename(columns={"end_date": "_target_end_date"})
    source_events = work[["ts_code", "end_date", "_avail_num"]].rename(
        columns={"end_date": "_source_end_date", "_avail_num": "_event_num"}
    )

    def _events_for_period(period_col: str, target_rows: pd.DataFrame) -> pd.DataFrame:
        events = target_rows.merge(
            source_events,
            left_on=["ts_code", period_col],
            right_on=["ts_code", "_source_end_date"],
            how="inner",
        )
        return events[
            [
                "ts_code",
                "_target_end_date",
                "_q",
                "_q_last_year",
                "_q4_last_year",
                "_event_num",
            ]
        ]

    # Q4 直接使用全年累计，不受依赖期变化影响；非 Q4 才订阅两个依赖期事件。
    own_events = _events_for_period("_target_end_date", targets)
    non_annual_targets = targets[targets["_q"] != "12"]
    event_frames = [own_events]
    if len(non_annual_targets) > 0:
        event_frames.extend(
            [
                _events_for_period("_q_last_year", non_annual_targets),
                _events_for_period("_q4_last_year", non_annual_targets),
            ]
        )
    events = pd.concat(event_frames, ignore_index=True)
    events = events.drop_duplicates(
        subset=["ts_code", "_target_end_date", "_event_num"]
    ).sort_values(["ts_code", "_target_end_date", "_event_num"], kind="mergesort")
    events = events.reset_index(drop=True)
    events["_row_id"] = np.arange(len(events))

    # 解析事件日可见的指定报告期版本。显式左连接后按可用日倒序取最后一条，
    # 规避 pandas merge_asof 对跨组全局排序的限制。
    def _resolve_asof(period_col: str, prefix: str) -> pd.DataFrame:
        right = work[["ts_code", "end_date", "_avail_num"] + cum_cols_present].copy()
        right = right.rename(
            columns={
                "end_date": "_source_end_date",
                "_avail_num": f"_{prefix}_avail_num",
                **{col: f"{prefix}_{col}" for col in cum_cols_present},
            }
        )
        merged = events[["_row_id", "ts_code", period_col, "_event_num"]].merge(
            right,
            left_on=["ts_code", period_col],
            right_on=["ts_code", "_source_end_date"],
            how="left",
        )
        avail_col = f"_{prefix}_avail_num"
        valid = merged[merged[avail_col] <= merged["_event_num"]]
        valid = valid.sort_values(["_row_id", avail_col], kind="mergesort")
        best = valid.drop_duplicates(subset="_row_id", keep="last")
        return best.set_index("_row_id").reindex(events["_row_id"])

    current = _resolve_asof("_target_end_date", "current")
    prev_q = _resolve_asof("_q_last_year", "prev_q")
    prev_q4 = _resolve_asof("_q4_last_year", "prev_q4")
    event_work = events.set_index("_row_id")
    event_work = event_work[current["_current_avail_num"].notna()].copy()

    # TTM = cum(今年) - cum(去年同季度) + cum(去年Q4)；Q4 行退化为当年累计
    for col in cum_cols_present:
        current_value = current.loc[event_work.index, f"current_{col}"]
        ttm = (
            current_value
            - prev_q.loc[event_work.index, f"prev_q_{col}"]
            + prev_q4.loc[event_work.index, f"prev_q4_{col}"]
        )
        event_work[col] = ttm.where(event_work["_q"] != "12", current_value)

    event_work["end_date"] = event_work["_target_end_date"]
    event_work["avail_date"] = event_work["_event_num"].astype(str)

    # 因子列映射（TTM 口径）
    event_work["ocf"] = (
        event_work["n_cashflow_act"] if "n_cashflow_act" in event_work.columns else np.nan
    )
    event_work["fcf"] = (
        event_work["free_cashflow"] if "free_cashflow" in event_work.columns else np.nan
    )

    if "c_fr_sale_sg" in event_work.columns:
        event_work["ocf_to_revenue"] = np.where(
            event_work["c_fr_sale_sg"].abs() >= _MIN_ABS_DENOMINATOR,
            np.clip(
                event_work["ocf"] / event_work["c_fr_sale_sg"],
                *_CLIP_OCF_TO_REVENUE,
            ),
            np.nan,
        )
    else:
        event_work["ocf_to_revenue"] = np.nan

    if "net_profit" in event_work.columns:
        event_work["ocf_to_profit"] = np.where(
            event_work["net_profit"].abs() >= _MIN_ABS_DENOMINATOR,
            np.clip(event_work["ocf"] / event_work["net_profit"], *_CLIP_OCF_TO_PROFIT),
            np.nan,
        )
    else:
        event_work["ocf_to_profit"] = np.nan

    if "c_pay_acq_const_fiolta" in event_work.columns:
        event_work["capex_to_ocf"] = np.where(
            event_work["ocf"].abs() >= _MIN_ABS_DENOMINATOR,
            np.clip(
                event_work["c_pay_acq_const_fiolta"].abs() / event_work["ocf"],
                *_CLIP_CAPEX_TO_OCF,
            ),
            np.nan,
        )
    else:
        event_work["capex_to_ocf"] = np.nan

    event_work = event_work.drop(
        columns=[
            c
            for c in [
                "_target_end_date",
                "_q",
                "_q_last_year",
                "_q4_last_year",
                "_event_num",
            ]
            if c in event_work.columns
        ]
    )
    return event_work.reset_index(drop=True)


def build_cashflow_quality_lookup_by_date(
    cashflow_raw: pd.DataFrame,
    trading_dates: List[str],
    daily_basic_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """将季度现金流量表前向填充到日频，构建每日查询表。

    Args:
        cashflow_raw: 现金流量表原始 DataFrame，需包含
                      ts_code, ann_date, end_date（f_ann_date 可选），
                      以及 n_cashflow_act/c_pay_acq_const_fiolta/c_fr_sale_sg/
                      net_profit/free_cashflow 中的可用列。
        trading_dates: 交易日列表（YYYYMMDD 格式字符串，已排序）
        daily_basic_lookup: 保留参数（历史接口兼容；fcf_yield 由 handler 层计算）

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, ocf, ocf_to_revenue, ...)}
    """
    if cashflow_raw is None or len(cashflow_raw) == 0:
        logger.warning("现金流量表数据为空，跳过现金流因子构建")
        return {}

    factor_df = _build_ttm_table(cashflow_raw)
    if len(factor_df) == 0:
        logger.warning("现金流量表数据清洗/版本化后为空，跳过现金流因子构建")
        return {}

    available_cols = [c for c in CASHFLOW_COLS if c in factor_df.columns and c != "fcf_yield"]

    logger.info(
        f"现金流质量查询表构建(schema_v{CASHFLOW_QUALITY_SCHEMA_VERSION}): "
        f"{factor_df['ts_code'].nunique()} 只股票, {len(trading_dates)} 个交易日"
    )

    result_dict = build_latest_announcement_lookup_by_date(
        factor_df[["ts_code", "avail_date", "end_date"] + available_cols],
        trading_dates,
        value_cols=available_cols,
        end_col="end_date",
        ann_col="avail_date",
        freshness_col=CASHFLOW_FRESHNESS_COL,
        log_name="现金流质量",
    )

    logger.info(f"现金流质量日频查询表构建完成: {len(result_dict)} 个交易日")
    return result_dict


def cashflow_quality_live_columns(feature_columns: List[str]) -> List[str]:
    """从训练门禁后的 feature_columns 中提取实际入模的现金流质量列（含 _sz 变体）。"""
    base = {
        "zscore_ocf_to_revenue",
        "zscore_ocf_to_profit",
        "zscore_fcf_yield",
        "zscore_capex_to_ocf",
        "cashflow_freshness_days",
    }
    sz = {f"{c}_sz" for c in base if c != "cashflow_freshness_days"}
    return sorted(c for c in feature_columns if c in base or c in sz)
