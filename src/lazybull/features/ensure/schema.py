# -*- coding: utf-8 -*-
"""ensure 子包：features 缓存完整性校验。"""

from typing import Collection, Dict, List, Optional, Tuple

from loguru import logger

from ...data import Storage

# 已缓存 features 必须包含的基础因子列（缺失则触发重建）。
# 可选因子组由调用方按本次构建开关显式传入，避免关闭组永久使缓存失效。
_BASE_REQUIRED_FACTOR_COLS = [
    "rzye_chg_5",
    "rzye_chg_20",
    "rqye_rzye_ratio",  # 融资融券
    "short_balance_change_5",  # 融资融券（风控列，v0.95.2 补检旧缓存）
    "zscore_bp",
    "zscore_dv_ttm",
    "zscore_amount_ma20",  # 截面 z-score
    "neu_ret_5",  # 行业中性化收益
    "alpha_industry_5",  # 行业 alpha
    "ind_momentum_rank",  # 行业动量
    "mkt_atr_pct",  # 市场级 ATR 当前值
    "mkt_atr_pct_ma250",  # 市场级 ATR 250 日均值
    "roe_waa",  # 基本面因子
    "q_gr_yoy",  # 基本面单季增速
    "cf_sales",  # 基本面现金流/营收
    "cf_nm",  # 基本面现金流/净利润
    "grossprofit_margin",  # 基本面扩展因子（利润率）
    "int_to_talcap",  # 无形资产/总资本比
    "inv_turn",  # 存货周转率
    "fundamental_freshness_days",  # 基本面 freshness
    "holder_num_chg",  # 股东人数因子
    "holder_freshness_days",  # 股东人数 freshness
    "forecast_type_score",  # 业绩预告因子
    "forecast_freshness_days",  # 业绩预告 freshness
    "winner_rate",  # 筹码胜率因子
    "weight_avg_bias",  # 筹码成本偏离度（v0.95.4 修复后正式产出，旧 4 列缓存缺此列自动重建）
    "fund_hold_ratio",  # 基金持仓因子
    "fund_portfolio_freshness_days",  # 基金持仓 freshness
    "express_revenue_yoy",  # 业绩快报因子
    "express_profit_yoy",  # 业绩快报因子（v0.95.6 补检）
    "express_roe",  # 业绩快报因子（v0.95.6 补检）
    "express_surprise",  # 业绩快报因子（v0.95.6 补检）
    "express_freshness_days",  # 业绩快报 freshness
    "lhb_cont_on_list",  # 龙虎榜因子（v0.95.0 新增列，旧缓存缺列需重建）
    "lhb_cont_up_days_5",  # 龙虎榜连续异动近 5 日累计（v0.95.0 新增列）
    "lhb_cont_up_days_20",  # 龙虎榜连续异动近 20 日累计（v0.95.0 新增列）
    "consensus_freshness_days",  # 一致预期 freshness
]


OPTIONAL_FACTOR_GROUP_CASHFLOW_QUALITY = "cashflow_quality"
OPTIONAL_FACTOR_GROUP_CONSENSUS_REVISION = "consensus_revision"
OPTIONAL_FACTOR_GROUP_DIVIDEND_POLICY = "dividend_policy"

_OPTIONAL_FACTOR_REQUIRED_COLS: Dict[str, List[str]] = {
    OPTIONAL_FACTOR_GROUP_CASHFLOW_QUALITY: [
        "cashflow_freshness_days",
        "cashflow_quality_schema_v2",
    ],
    OPTIONAL_FACTOR_GROUP_CONSENSUS_REVISION: [
        "cons_revision_freshness_days",
        "cons_eps_revision_accel",
        "cons_eps_dispersion",
        "cons_eps_dispersion_chg",
        "cons_target_upside_chg",
        "cons_analyst_count_chg",
        "cons_rating_upgrade_ratio",
        "cons_revision_schema_v2",
    ],
    OPTIONAL_FACTOR_GROUP_DIVIDEND_POLICY: [
        "dividend_continuity_5y",
        "dividend_stability_5y",
        "dividend_growth_3y",
        "dividend_growth_5y",
        "dividend_payout_ratio",
        "dividend_yield_hist_12m",
        "dividend_days_to_ex_date",
        "dividend_recent_imp_ann_10d",
        "zscore_dividend_continuity_5y",
        "zscore_dividend_stability_5y",
        "zscore_dividend_growth_3y",
        "zscore_dividend_growth_5y",
        "zscore_dividend_payout_ratio",
        "zscore_dividend_yield_hist_12m",
        "dividend_freshness_days",
        "dividend_hist_missing",
        "dividend_schema_v1",
    ],
}

# 兼容既有导入：表示基础列与全部可选组的并集，不再直接用于配置感知校验。
_REQUIRED_FACTOR_COLS = list(_BASE_REQUIRED_FACTOR_COLS)
for _group_columns in _OPTIONAL_FACTOR_REQUIRED_COLS.values():
    _REQUIRED_FACTOR_COLS.extend(_group_columns)


def _optional_factor_sentinel_specs() -> Dict[str, Tuple[str, int]]:
    """返回可选因子组的稳定哨兵列及当前语义版本。"""
    from ...factors.cashflow_quality import (
        CASHFLOW_QUALITY_SCHEMA_VERSION,
        CASHFLOW_QUALITY_VERSION_COL,
    )
    from ...factors.consensus_revision import (
        CONSENSUS_REVISION_SCHEMA_VERSION,
        CONSENSUS_REVISION_VERSION_COL,
    )
    from ...factors.dividend import (
        DIVIDEND_POLICY_SCHEMA_VERSION,
        DIVIDEND_POLICY_VERSION_COL,
    )

    return {
        OPTIONAL_FACTOR_GROUP_CASHFLOW_QUALITY: (
            CASHFLOW_QUALITY_VERSION_COL,
            CASHFLOW_QUALITY_SCHEMA_VERSION,
        ),
        OPTIONAL_FACTOR_GROUP_CONSENSUS_REVISION: (
            CONSENSUS_REVISION_VERSION_COL,
            CONSENSUS_REVISION_SCHEMA_VERSION,
        ),
        OPTIONAL_FACTOR_GROUP_DIVIDEND_POLICY: (
            DIVIDEND_POLICY_VERSION_COL,
            DIVIDEND_POLICY_SCHEMA_VERSION,
        ),
    }


def _resolve_required_optional_groups(
    required_optional_groups: Optional[Collection[str]],
) -> List[str]:
    """规范化调用方要求的可选因子组；None 表示全部组。"""
    groups = (
        set(_OPTIONAL_FACTOR_REQUIRED_COLS)
        if required_optional_groups is None
        else set(required_optional_groups)
    )
    unknown = groups.difference(_OPTIONAL_FACTOR_REQUIRED_COLS)
    if unknown:
        raise ValueError(f"未知可选因子 schema 组: {sorted(unknown)}")
    return [group for group in _OPTIONAL_FACTOR_REQUIRED_COLS if group in groups]


def _check_features_schema(
    storage: Storage,
    trade_date: str,
    subdir: str = "cs_train",
    required_optional_groups: Optional[Collection[str]] = None,
) -> bool:
    """快速检查已缓存 features 是否满足本次构建配置。

    基础列始终必需；可选组由 ``required_optional_groups`` 指定，None 表示
    全部三组（纸面推理默认契约）。已启用组同时校验稳定哨兵版本值。
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    optional_groups = _resolve_required_optional_groups(required_optional_groups)
    required_columns = list(_BASE_REQUIRED_FACTOR_COLS)
    for group in optional_groups:
        required_columns.extend(_OPTIONAL_FACTOR_REQUIRED_COLS[group])

    target_path = storage.features_path / subdir
    file_path = target_path / f"{trade_date}.parquet"
    if not file_path.exists():
        return False

    try:
        schema = pq.read_schema(str(file_path))
        col_names = set(schema.names)
        missing = [column for column in required_columns if column not in col_names]
        if missing:
            logger.debug(f"features 缓存缺失列: {missing}")
            return False
        sentinel_specs = _optional_factor_sentinel_specs()
        for group in optional_groups:
            version_col, expected_version = sentinel_specs[group]
            version_values = pq.read_table(str(file_path), columns=[version_col]).column(0)
            all_current = pc.all(pc.equal(version_values, expected_version)).as_py()
            if not version_values or version_values.null_count > 0 or all_current is not True:
                logger.debug(f"features 缓存 {group} 哨兵版本不符: 期望 {expected_version}")
                return False
        return True
    except Exception:
        return False
