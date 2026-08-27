# -*- coding: utf-8 -*-
"""ensure 子包：features 缓存完整性校验。"""

from typing import List

from loguru import logger

from ...data import Storage

# 已缓存 features 必须包含的因子列（缺失则触发重建）
# 每个因子组至少一个代表性列，确保旧缓存或因子组缺失时自动淘汰
_REQUIRED_FACTOR_COLS = [
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
    "cashflow_freshness_days",  # 现金流质量 freshness
    "cons_revision_freshness_days",  # 一致预期修正 freshness
    "cons_eps_revision_accel",  # 一致预期修正六值列（v2 重做后补检）
    "cons_eps_dispersion",
    "cons_eps_dispersion_chg",
    "cons_target_upside_chg",
    "cons_analyst_count_chg",
    "cons_rating_upgrade_ratio",
    "cons_revision_schema_v2",  # v2 重做哨兵列，缺失即强制重建旧语义缓存
]


def _check_features_schema(storage: Storage, trade_date: str, subdir: str = "cs_train") -> bool:
    """快速检查已缓存 features 是否包含必要的因子列

    仅读取 Parquet schema（不加载数据），开销极低。
    若文件损坏或缺失必要列则返回 False，触发重建。
    """
    import pyarrow.parquet as pq

    target_path = storage.features_path / subdir
    file_path = target_path / f"{trade_date}.parquet"
    if not file_path.exists():
        return False

    try:
        schema = pq.read_schema(str(file_path))
        col_names = set(schema.names)
        missing = [c for c in _REQUIRED_FACTOR_COLS if c not in col_names]
        if missing:
            logger.debug(f"features 缓存缺失列: {missing}")
            return False
        return True
    except Exception:
        return False
