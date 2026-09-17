# -*- coding: utf-8 -*-
"""因子体检常量：家族映射与候选判定阈值。"""

from dataclasses import dataclass
from typing import Dict, List

DEFAULT_LABEL_COLUMN = "neu_y_ret_20"
DEFAULT_MARKET_VOL_COLUMN = "mkt_vol_20"

# 训练侧家族开关常量（ml.train_core.constants）-> 报告用家族名
FAMILY_CONSTANT_NAMES: Dict[str, str] = {
    "FUNDAMENTAL_FEATURE_COLUMNS": "fundamental",
    "ALT_FEATURE_COLUMNS": "alt",
    "MARGIN_FEATURE_COLUMNS": "margin",
    "CYQ_FEATURE_COLUMNS": "cyq",
    "FUND_FEATURE_COLUMNS": "fund",
    "EXPRESS_FEATURE_COLUMNS": "express",
    "NORTH_FEATURE_COLUMNS": "north",
    "LHB_FEATURE_COLUMNS": "lhb",
    "CONSENSUS_FEATURE_COLUMNS": "consensus",
    "ENHANCED_FEATURE_COLUMNS": "enhanced",
    "CASHFLOW_QUALITY_FEATURE_COLUMNS": "cashflow",
    "CONSENSUS_REVISION_FEATURE_COLUMNS": "cons_rev",
    "DIVIDEND_POLICY_FEATURE_COLUMNS": "dividend",
    "MISSING_MARKER_FEATURE_COLUMNS": "missing_marker",
    "HOLDERTRADE_FEATURE_COLUMNS": "holdertrade",
}

# 基础特征清单（prepare.py 内联列表）按主题分组，仅用于报告归类展示
BASE_GROUPS: Dict[str, List[str]] = {
    "动量趋势": [
        "neu_ret_1",
        "neu_ret_20",
        "neu_ret_5",
        "alpha_industry_20",
        "alpha_industry_5",
        "ind_ret_avg",
        "ind_momentum_rank",
        "zscore_ma_deviation_20",
        "zscore_acceleration",
        "zscore_macd_hist",
        "bb_pct",
    ],
    "流动性资金": [
        "zscore_turnover_rate",
        "vol_ratio_20",
        "vol_burst_20",
        "zscore_amount_ma20",
        "zscore_net_mf_amount",
        "zscore_elg_net_amount_sum_20",
        "lg_net_amount_sum_5",
    ],
    "波动形态": [
        "zscore_volatility_20",
        "zscore_volatility_5",
        "amplitude",
        "zscore_bb_width",
        "upper_shadow",
        "lower_shadow",
        "spec_score",
        "rsi_14",
        "kdj_j",
    ],
    "估值质量": [
        "zscore_size",
        "zscore_bp",
        "zscore_dv_ttm",
        "zscore_pe_ttm",
        "is_loss",
        "list_days",
    ],
    "市场环境": [
        "mkt_adv_dec_ratio",
        "mkt_ret_avg_20",
        "mkt_turnover_std",
        "mkt_vol_20",
    ],
}


@dataclass(frozen=True)
class HealthThresholds:
    """候选清单判定阈值（全部可经 CLI 覆盖）。"""

    low_coverage: float = 0.6  # 全期覆盖低于该值 -> 低覆盖候选
    low_year_coverage: float = 0.4  # 任一年份覆盖低于该值 -> 低覆盖候选
    weak_abs_t: float = 1.5  # |t| 低于该值视为弱信息
    weak_gain_share: float = 0.005  # gain 份额低于该值视为未被模型使用
    unused_split_use: float = 0.5  # 分裂使用率低于该值视为几乎未用
    unused_gain_share: float = 0.002  # gain 份额低于该值视为几乎未用
    flip_min_abs_year_ic: float = 0.01  # 年 |IC| 低于该值不计入符号统计
    flip_years_opposite: int = 3  # 异号年份数达到该值 -> 翻号候选
    flip_year_count: int = 2  # 相邻年份翻转次数达到该值 -> 翻号候选
    cluster_abs_corr: float = 0.85  # |rho| 达到该值合并为同一簇
    min_pairs: int = 200  # 单日截面计算 IC 所需最少配对样本
    corr_min_periods: int = 100  # 相关矩阵最少配对样本
