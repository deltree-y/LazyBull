#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练核心逻辑模块

从 train_ml_model.py 抽取的可复用训练函数，供训练脚本和 walk-forward 脚本共用。

功能：
- 加载特征数据
- 准备训练数据（切分训练集/验证集）
- 标签变换（截面 z-score、分类标签生成）
- 训练 XGBoost 模型
- 验证集逐日评估
"""

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError:
    logger.error("需要安装 xgboost: pip install xgboost")
    raise

from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml.eval_utils import (
    compute_diagnostic_statistics,
    evaluate_predictions_by_date,
    print_diagnostic_report,
    summarize_daily_metrics,
)

# 基本面因子特征列（行业 z-score 后的列名）
FUNDAMENTAL_FEATURE_COLUMNS = [
    "zscore_roe_waa",  # 加权平均ROE
    "zscore_or_yoy",  # 营业收入同比增速
    "zscore_netprofit_yoy",  # 净利润同比增速
    "zscore_debt_to_assets",  # 资产负债率
    "zscore_q_gr_yoy",  # 单季度营收同比增速
    "fundamental_freshness_days",  # 最近一次基本面公告距当日天数
]

# 融资融券因子特征列
MARGIN_FEATURE_COLUMNS = [
    "rzye_chg_5",  # 融资余额5日变动率
    "rzye_chg_20",  # 融资余额20日变动率
    "rqye_rzye_ratio",  # 融券/融资余额比
    "margin_net_buy_ratio",  # 融资净买入/成交额
]

# 另类数据因子特征列（不含融资融券）
ALT_FEATURE_COLUMNS = [
    # 股东人数 (2)
    "holder_num_chg",  # 股东人数环比变动率
    "holder_num_chg_2q",  # 股东人数两期变动率
    "holder_freshness_days",  # 最近一次股东人数公告距当日天数
    # 业绩预告 (2)
    "forecast_type_score",  # 业绩预告类型评分
    "forecast_chg_mid",  # 业绩预告变动幅度中值
    "forecast_freshness_days",  # 最近一次业绩预告公告距当日天数
]

# 筹码胜率因子特征列（5000 积分）
CYQ_FEATURE_COLUMNS = [
    "winner_rate",  # 胜率
    "weight_avg_bias",  # 加权平均成本偏离度
    "cost_concentration",  # 筹码集中度
    "winner_rate_chg_5",  # 5日胜率变化
    "winner_rate_chg_20",  # 20日胜率变化
]

# 基金持仓因子特征列（5000 积分）
FUND_FEATURE_COLUMNS = [
    "fund_hold_ratio",  # 基金持股占流通股比例
    "fund_hold_ratio_chg",  # 基金持股比例季度变化
    "fund_count",  # 持仓基金数量
    "fund_count_chg",  # 持仓基金数量季度变化
    "fund_portfolio_freshness_days",  # 最近一次基金持仓公告距当日天数
]

# 业绩快报因子特征列（5000 积分）
EXPRESS_FEATURE_COLUMNS = [
    "express_revenue_yoy",  # 营业收入同比增速
    "express_profit_yoy",  # 净利润同比增速
    "express_roe",  # 快报ROE
    "express_surprise",  # 业绩惊喜
    "express_freshness_days",  # 最近一次业绩快报公告距当日天数
]

# 北向资金因子特征列（市场级, 广播到全部 ts_code）
NORTH_FEATURE_COLUMNS = [
    "north_flow",  # 当日北向净流入（亿元）
    "north_flow_ma5",  # 5日移动均值
    "north_flow_ma20",  # 20日移动均值
    "north_flow_z20",  # 20日滚动 z-score
    "north_flow_sum5",  # 5日累计净流入
    "north_flow_sign_streak",  # 连续同方向天数
]

# 龙虎榜因子特征列（稀疏, 未上榜填 0）
LHB_FEATURE_COLUMNS = [
    "lhb_on_list",  # 当日是否上榜
    "lhb_net_amount",  # 龙虎榜净买入额
    "lhb_net_rate",  # 净买入占流通市值比
    "lhb_amount_rate",  # 龙虎榜成交占比
    "lhb_up_days_20",  # 近 20 日累计上榜次数
    "lhb_net_sum_5",  # 近 5 日净买入累计
    "lhb_net_sum_20",  # 近 20 日净买入累计
    "lhb_reason_count",  # 当日上榜理由数
]

# 一致预期因子特征列（分析师研报聚合）
CONSENSUS_FEATURE_COLUMNS = [
    "cons_analyst_count_30d",  # 近 30 日覆盖的研报数
    "cons_eps_mean_fy1",  # 近 90 日 FY1 EPS 预测均值
    "cons_eps_revision_30d",  # 近 30 日 EPS 预测修正率
    "cons_target_price_mid",  # 近 90 日目标价中值
    "cons_rating_score",  # 近 90 日平均评级得分
]

# 增强因子特征列（从已有日线/moneyflow数据计算，无需额外积分）
ENHANCED_FEATURE_COLUMNS = [
    "zscore_opening_strength",       # 开盘强度（隔夜情绪代理）
    "zscore_intraday_vol_structure",  # 日内波动结构（多空力量对比）
    "zscore_order_imbalance",        # 特大单订单失衡
    "order_imbalance_mean_5",        # 5日订单失衡均值
    "order_imbalance_mean_20",       # 20日订单失衡均值
]


# ── 自定义早停 eval metric ──────────────────────────────────────
def neg_rank_ic(y_true, y_pred):
    """Spearman Rank IC（XGBoost sklearn 早停用）。
    返回负值以适配 XGBoost minimize 约定；函数名自动作为 metric name。"""
    corr, _ = spearmanr(y_true, y_pred)
    return float(-corr if not np.isnan(corr) else 0)


def _rank_ic_eval_lgb(y_true, y_pred):
    """Spearman Rank IC（LightGBM 早停用，higher_is_better=True）"""
    corr, _ = spearmanr(y_true, y_pred)
    return "rank_ic", float(corr if not np.isnan(corr) else 0), True


def load_features_data(
    storage: Storage, loader: DataLoader, start_date: str, end_date: str
) -> tuple:
    """加载指定日期区间的特征数据

    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD

    Returns:
        (df, trade_days_count) 元组：合并后的特征 DataFrame 和交易日数量
    """
    logger.info(f"加载特征数据: {start_date} 至 {end_date}")

    # 获取交易日列表
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()

    trade_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    logger.info(f"共 {len(trade_dates)} 个交易日")

    # 加载每日特征数据
    all_features = []
    missing_dates = []
    for trade_date in trade_dates:
        features = storage.load_cs_train_day(trade_date)
        if features is not None and len(features) > 0:
            all_features.append(features)
        else:
            logger.debug(f"日期 {trade_date} 没有特征数据")
            missing_dates.append(trade_date)

    if missing_dates:
        logger.info(
            f"共 {len(missing_dates)} 个交易日无特征数据（跳过）: {missing_dates[0]} ~ {missing_dates[-1]}"
        )

    if not all_features:
        raise ValueError(f"指定日期区间内没有特征数据")

    # 合并所有数据
    df = pd.concat(all_features, ignore_index=True)
    logger.info(
        f"成功加载 {len(df)} 条样本（{len(all_features)}/{len(trade_dates)} 个交易日有数据）"
    )

    return df, len(trade_dates)


def split_train_val_by_date(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    date_col: str = "trade_date",
    delta: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """按 trade_date 粒度切分训练集和验证集

    确保同一交易日的所有样本不会被拆分到不同集合，彻底避免截面统计量跨集合污染。
    以唯一交易日列表为单位，最后 ceil(n_dates * val_ratio) 个日期作为验证集。

    Args:
        df: 输入 DataFrame（需包含 date_col 列）
        val_ratio: 验证集比例，默认 0.2
        date_col: 日期列名，默认 trade_date
        delta: 训练集末尾与验证集开头之间的间隔交易日数，用于防止标签前向泄露；
               应设置为标签 horizon（如 y_ret_20 对应 delta=20）。

    Returns:
        (df_train, df_val, stats) 元组：
            - df_train: 训练集 DataFrame
            - df_val: 验证集 DataFrame
            - stats: 包含日期统计信息的字典（train_n_dates/val_n_dates/train_start_date/
                     train_end_date/val_start_date/val_end_date）
    """
    all_dates = sorted(df[date_col].unique())
    n_dates = len(all_dates)

    if n_dates == 0:
        empty_stats = {
            "train_n_dates": 0,
            "val_n_dates": 0,
            "train_start_date": "N/A",
            "train_end_date": "N/A",
            "val_start_date": "N/A",
            "val_end_date": "N/A",
        }
        return df.iloc[:0].copy(), df.iloc[:0].copy(), empty_stats

    n_val_dates = max(1, math.ceil(n_dates * val_ratio))
    n_train_dates = n_dates - n_val_dates

    if n_train_dates <= 0:
        n_train_dates = 0
        n_val_dates = n_dates

    # 训练集末尾扣除 delta 天间隔，防止标签泄漏到验证集
    if n_train_dates > delta:
        train_dates_set = set(all_dates[: n_train_dates - delta])
    else:
        # 训练日期数不足以扣除 delta 间隔，训练集将无隔离带
        logger.warning(
            f"训练/验证集分割: 训练日期数({n_train_dates}) <= delta({delta})，"
            f"无法从训练集末尾扣除隔离带，训练集保持完整"
        )
        train_dates_set = set(all_dates[:n_train_dates])

    if n_train_dates + delta < n_dates:
        val_dates_set = set(all_dates[n_train_dates + delta :])
    else:
        # 数据不足以保留完整 delta 间隔，存在标签泄漏风险
        # 为避免训练结果不可靠，返回空验证集而非使用有泄漏的数据
        logger.error(
            f"训练/验证集分割: 数据不足以保留 {delta} 天间隔 "
            f"(总日期={n_dates}, 训练={n_train_dates}, delta={delta})，"
            f"放弃验证集以避免标签泄漏"
        )
        val_dates_set = set()

    df_train = df[df[date_col].isin(train_dates_set)].copy()
    df_val = df[df[date_col].isin(val_dates_set)].copy()

    actual_train_n_dates = len(train_dates_set)
    sorted_train_dates = sorted(train_dates_set)
    sorted_val_dates = sorted(val_dates_set)
    stats = {
        "train_n_dates": actual_train_n_dates,  # 实际参与训练的日期数（已扣除末尾 delta 天间隔）
        "val_n_dates": len(val_dates_set),
        "train_start_date": str(sorted_train_dates[0]) if sorted_train_dates else "N/A",
        "train_end_date": str(sorted_train_dates[-1]) if sorted_train_dates else "N/A",
        "val_start_date": str(sorted_val_dates[0]) if sorted_val_dates else "N/A",
        "val_end_date": str(sorted_val_dates[-1]) if sorted_val_dates else "N/A",
    }

    logger.info(f"按 trade_date 粒度切分（共 {n_dates} 个交易日，delta={delta} 天间隔）:")
    logger.info(
        f"  训练集: {stats['train_start_date']} 至 {stats['train_end_date']}"
        f"（{actual_train_n_dates} 个交易日，{len(df_train)} 条样本）"
    )
    logger.info(
        f"  验证集: {stats['val_start_date']} 至 {stats['val_end_date']}"
        f"（{stats['val_n_dates']} 个交易日，{len(df_val)} 条样本）"
    )

    return df_train, df_val, stats


def filter_stable_features(
    df_train: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    date_col: str = "trade_date",
    n_splits: int = 3,
    min_abs_ic: float = 0.02,
) -> Tuple[List[str], Dict]:
    """筛选在所有时间段截面IC方向一致且显著的特征

    将训练集按时间等分成 n_splits 段，每段计算各特征与标签的截面 Spearman IC 均值。
    仅保留所有段IC方向一致且平均 |IC| 超过阈值的特征。

    Args:
        df_train: 训练集 DataFrame（已完成标签变换）
        feature_columns: 候选特征列名列表
        label_column: 标签列名
        date_col: 日期列名
        n_splits: 将训练集按时间等分的段数，默认 3
        min_abs_ic: 平均 |IC| 的最低阈值，默认 0.02

    Returns:
        (stable_features, filter_info) 元组：
            - stable_features: 通过筛选的特征列名列表
            - filter_info: 筛选统计信息字典
    """
    dates = sorted(df_train[date_col].unique())
    n_dates = len(dates)
    if n_dates < n_splits * 10:
        logger.warning(
            f"特征稳定性筛选: 训练集仅 {n_dates} 个交易日，不足以分成 {n_splits} 段，跳过筛选"
        )
        return feature_columns, {"skipped": True, "reason": "交易日不足"}

    split_size = n_dates // n_splits

    # 每个时段计算各特征的截面IC均值
    ic_matrix: Dict[str, List[float]] = {col: [] for col in feature_columns}

    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < n_splits - 1 else n_dates
        split_dates = set(dates[start:end])
        split_df = df_train[df_train[date_col].isin(split_dates)]

        # 向量化计算：逐日 rank 后用 corr 算出各特征的截面IC
        # 某些交易日某些特征值全相同（std=0），corr 返回 NaN 并触发 numpy 告警，
        # 这是预期行为，NaN 已被跳过，抑制告警避免刷屏
        daily_ics: Dict[str, List[float]] = {col: [] for col in feature_columns}
        with np.errstate(divide="ignore", invalid="ignore"):
            for _, group in split_df.groupby(date_col):
                if len(group) < 30:
                    continue
                ranked = group[feature_columns + [label_column]].rank()
                label_rank = ranked[label_column]
                for col in feature_columns:
                    corr = ranked[col].corr(label_rank)
                    if not np.isnan(corr):
                        daily_ics[col].append(corr)

        for col in feature_columns:
            ics = daily_ics[col]
            ic_matrix[col].append(np.mean(ics) if ics else 0.0)

    # 筛选条件：所有段IC符号一致 且 平均|IC| >= min_abs_ic
    stable_features = []
    removed_details = []
    for col in feature_columns:
        ics = ic_matrix[col]
        nonzero_signs = [np.sign(ic) for ic in ics if abs(ic) > 1e-6]

        if len(nonzero_signs) == 0:
            removed_details.append((col, ics, "IC全为零"))
            continue

        all_same_sign = all(s == nonzero_signs[0] for s in nonzero_signs)
        avg_abs_ic = np.mean([abs(ic) for ic in ics])

        if all_same_sign and avg_abs_ic >= min_abs_ic:
            stable_features.append(col)
        else:
            reason = "IC方向不一致" if not all_same_sign else f"|IC|={avg_abs_ic:.4f}<{min_abs_ic}"
            removed_details.append((col, ics, reason))

    filter_info = {
        "skipped": False,
        "total_features": len(feature_columns),
        "stable_count": len(stable_features),
        "removed_count": len(removed_details),
        "removed_details": removed_details,
    }

    logger.info(
        f"特征稳定性筛选: {len(feature_columns)} → {len(stable_features)} 个特征"
        f"（移除 {len(removed_details)} 个不稳定特征）"
    )
    if removed_details:
        for col, ics, reason in removed_details:
            ic_str = ", ".join(f"{ic:+.4f}" for ic in ics)
            logger.debug(f"  移除 {col}: [{ic_str}] — {reason}")

    return stable_features, filter_info


def prepare_training_data(
    df: pd.DataFrame,
    label_column: str = "neu_y_ret_20",
    val_ratio: float = 0.2,
    label_transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    enable_fundamental_features: bool = False,
    enable_alt_features: bool = False,
    enable_margin_features: bool = False,
    enable_cyq_features: bool = False,
    enable_fund_features: bool = False,
    enable_express_features: bool = False,
    enable_enhanced_features: bool = False,
    enable_north_features: bool = False,
    enable_lhb_features: bool = False,
    enable_consensus_features: bool = False,
    feature_stability_filter: bool = False,
) -> tuple:
    """准备训练数据，并按 trade_date 粒度切分训练集和验证集

    Args:
        df: 特征 DataFrame
        label_column: 标签列名
        val_ratio: 验证集比例，默认 0.2（最后 20% 的交易日作为验证集）
        label_transform_fn: 可选的标签变换函数，接受 DataFrame 并返回变换后的 DataFrame。
            若提供，将在按日切分后分别对训练集与验证集独立调用，避免跨集合统计量污染。
            典型用法：cs_zscore 变换（见 transform_labels_cs_zscore）。

    Returns:
        (X_train, y_train, X_val, y_val, feature_columns, df_train_split, df_val_split, data_stats,
         df_val_split_original) 元组
        - df_val_split: 标签已变换（供模型训练/early stopping 使用）
        - df_val_split_original: 标签变换前的原始快照（供逐日收益评估使用，保持真实收益单位）
        data_stats 包含：samples_after_filter, val_start_date, val_end_date
    """
    logger.info("准备训练数据...")

    # 确认标签列存在
    if label_column not in df.columns:
        raise ValueError(f"标签列 {label_column} 不存在")

    # 定义需要排除的列（非特征列）
    # 标识列
    id_columns = ["ts_code", "trade_date", "name"]
    # 标签列
    label_columns = [col for col in df.columns if col.startswith("y_")]
    # 过滤标记列（使用统一的列名，与clean层一致）
    # 收盘买入策略：涨停无法买入需过滤；跌停可以买入（有人卖出），保留参与训练
    filter_columns = ["is_st", "is_suspended", "is_limit_up"]
    # 其他非特征列
    other_exclude_columns = [
        "tradable",
        "list_date",
        "list_days",
        "is_limit_up",
        "is_limit_down",
        "industry",
    ]
    # 临时过滤掉的列
    temp_test_exclude_columns = (
        ["total_mv", "circ_mv", "log_circ_mv"]
        + ["kdj_k", "kdj_d"]
        + ["bb_upper", "bb_lower"]
        + ["macd_dif", "macd_dea"]
        + ["ps_ttm", "ep_ttm"]
        + [
            "amount_ma10",
            "amount_ma20",
            "volume_ratio",
            "log_circ_mv",
            "net_mf_amount_mean_5",
            "net_mf_amount_mean_20",
            "vol_burst_10",
            "vol_burst_20",
            "kdj_d",
            "macd_dea",
            "bb_upper",
            "bb_lower",
        ]
    )

    exclude_columns = (
        id_columns
        + label_columns
        + filter_columns
        + other_exclude_columns
        + temp_test_exclude_columns
    )

    # 获取特征列
    # feature_columns = [col for col in df.columns if col not in exclude_columns]
    feature_columns = [
        # 1. 中性化动量与趋势 (9个) - 剔除行业/市值后的纯选股动量
        "neu_ret_1",  # 超短期个股中性化反转（A股隔日反转效应）
        "neu_ret_20",  # 中期个股中性化超额
        # "neu_ret_10",              # 中期个股中性化超额
        "neu_ret_5",  # 短期个股中性化超额
        "alpha_industry_20",  # 行业动量（保留此特征以保留行业轮动视角）
        # "alpha_industry_10",       # 行业动量（保留此特征以保留行业轮动视角）
        "alpha_industry_5",  # 行业短期爆发力
        "ind_ret_avg",  # 所属行业平均收益（行业绝对动量）
        "ind_momentum_rank",  # 行业动量百分位排名（0~1，1=最强行业）
        "zscore_ma_deviation_20",  # 20日均线乖离率
        "zscore_acceleration",  # 动量加速度
        "zscore_macd_hist",  # MACD能量柱（动能切换）
        "bb_pct",  # 布林带位置
        # 2. 流动性与资金博弈 (7个) - 识别虚假繁荣与主力意图
        # volume_ratio（实时量比）与 vol_ratio_20 高度重叠，已移除
        "zscore_turnover_rate",  # 换手率级别
        "vol_ratio_20",  # 20日量比
        "vol_burst_20",  # 20日爆量系数
        "zscore_amount_ma20",  # 20日成交额基准
        "zscore_net_mf_amount",  # 当日净流入资金
        "zscore_elg_net_amount_sum_20",  # 20日特大单累积（主力深度）
        "lg_net_amount_sum_5",  # 5日大单累积
        # 3. 波动风险与形态特征 (8个) - 解决”早夭”与压制回撤
        # body_length 可由 amplitude - upper_shadow - lower_shadow 推导，已移除
        "zscore_volatility_20",  # 20日波动率
        # "zscore_volatility_10",    # 10日波动率
        "zscore_volatility_5",  # 5日波动率
        "amplitude",  # 当日振幅
        "zscore_bb_width",  # 布林带宽度（波动挤压/释放）
        "upper_shadow",  # 上影线（压力位）
        "lower_shadow",  # 下影线（支撑位）
        "spec_score",  # 投机分（高波动小市值复合得分）
        "rsi_14",  # 强弱指标（超买超卖）
        "kdj_j",  # 随机指标J值（灵敏度高）
        # 4. 估值、质量与安全边际 (6个) - 风格锚点，提供底层防御
        # pb 与 zscore_bp 线性冗余（BP=1/PB），已移除 pb
        "zscore_size",  # 市值因子（核心锚点）
        "zscore_bp",  # 账面市值比（价值挖掘）
        "zscore_dv_ttm",  # 股息率
        "zscore_pe_ttm",  # PE分位
        "is_loss",  # 是否亏损（质量过滤）
        "list_days",  # 上市天数
        # 5. 市场环境特征 (7个) - 环境感知 + 择时信号
        # is_limit_up 过滤后恒为 0，已移除；is_limit_down 保留（跌停股参与训练，不作为特征）
        "mkt_adv_dec_ratio",  # 市场涨跌比
        "mkt_ret_avg_20",  # 市场平均收益（20日）
        "mkt_turnover_std",  # 市场成交额波动
        "mkt_vol_20",  # 市场总体成交量
    ]

    # 基本面因子（可选）
    if enable_fundamental_features:
        available_funda = [col for col in FUNDAMENTAL_FEATURE_COLUMNS if col in df.columns]
        if available_funda:
            feature_columns.extend(available_funda)
            logger.info(f"启用基本面因子: {available_funda}")
        else:
            logger.warning("enable_fundamental_features=True，但数据中未找到基本面列，跳过")

    # 另类数据因子（可选）
    if enable_alt_features:
        available_alt = [col for col in ALT_FEATURE_COLUMNS if col in df.columns]
        if available_alt:
            feature_columns.extend(available_alt)
            logger.info(f"启用另类数据因子: {available_alt}")
        else:
            logger.warning("enable_alt_features=True，但数据中未找到另类数据列，跳过")

    # 融资融券因子（可选）
    if enable_margin_features:
        available_margin = [col for col in MARGIN_FEATURE_COLUMNS if col in df.columns]
        if available_margin:
            feature_columns.extend(available_margin)
            logger.info(f"启用融资融券因子: {available_margin}")
        else:
            logger.warning("enable_margin_features=True，但数据中未找到融资融券列，跳过")

    # 筹码胜率因子（可选，5000 积分）
    if enable_cyq_features:
        available_cyq = [col for col in CYQ_FEATURE_COLUMNS if col in df.columns]
        if available_cyq:
            feature_columns.extend(available_cyq)
            logger.info(f"启用筹码胜率因子: {available_cyq}")
        else:
            logger.warning("enable_cyq_features=True，但数据中未找到筹码胜率列，跳过")

    # 基金持仓因子（可选，5000 积分）
    if enable_fund_features:
        available_fund = [col for col in FUND_FEATURE_COLUMNS if col in df.columns]
        if available_fund:
            feature_columns.extend(available_fund)
            logger.info(f"启用基金持仓因子: {available_fund}")
        else:
            logger.warning("enable_fund_features=True，但数据中未找到基金持仓列，跳过")

    # 业绩快报因子（可选，5000 积分）
    if enable_express_features:
        available_express = [col for col in EXPRESS_FEATURE_COLUMNS if col in df.columns]
        if available_express:
            feature_columns.extend(available_express)
            logger.info(f"启用业绩快报因子: {available_express}")
        else:
            logger.warning("enable_express_features=True，但数据中未找到业绩快报列，跳过")

    # 增强因子（可选，从已有数据计算）
    if enable_enhanced_features:
        available_enhanced = [col for col in ENHANCED_FEATURE_COLUMNS if col in df.columns]
        if available_enhanced:
            feature_columns.extend(available_enhanced)
            logger.info(f"启用增强因子: {available_enhanced}")
        else:
            logger.warning("enable_enhanced_features=True，但数据中未找到增强因子列，跳过")

    # 北向资金因子（可选, 市场级, 广播到全部 ts_code）
    if enable_north_features:
        available_north = [col for col in NORTH_FEATURE_COLUMNS if col in df.columns]
        if available_north:
            feature_columns.extend(available_north)
            logger.info(f"启用北向资金因子: {available_north}")
        else:
            logger.warning("enable_north_features=True，但数据中未找到北向资金列，跳过")

    # 龙虎榜因子（可选, 稀疏, 未上榜填 0）
    if enable_lhb_features:
        available_lhb = [col for col in LHB_FEATURE_COLUMNS if col in df.columns]
        if available_lhb:
            feature_columns.extend(available_lhb)
            logger.info(f"启用龙虎榜因子: {available_lhb}")
        else:
            logger.warning("enable_lhb_features=True，但数据中未找到龙虎榜列，跳过")

    # 一致预期因子（可选, 分析师研报聚合）
    if enable_consensus_features:
        available_cons = [col for col in CONSENSUS_FEATURE_COLUMNS if col in df.columns]
        if available_cons:
            feature_columns.extend(available_cons)
            logger.info(f"启用一致预期因子: {available_cons}")
        else:
            logger.warning("enable_consensus_features=True，但数据中未找到一致预期列，跳过")

    logger.info(f"特征列数量: {len(feature_columns)}")
    logger.debug(f"特征列: {feature_columns[:10]}...")  # 只显示前10个

    # 过滤可训练样本（移除含有过滤标记的样本）
    mask = pd.Series([True] * len(df), index=df.index)
    for col in filter_columns:
        if col in df.columns:
            mask = mask & (~df[col].astype(bool))

    df_train = df[mask].copy()
    logger.info(f"过滤后样本数: {len(df_train)} / {len(df)}")
    samples_after_filter = len(df_train)

    # 移除标签为 NaN 的样本
    df_train = df_train.dropna(subset=[label_column])
    logger.info(f"移除标签 NaN 后样本数: {len(df_train)}")

    if len(df_train) == 0:
        raise ValueError("没有可用的训练样本")

    # 从标签列名自动推断 delta（例如 neu_y_ret_20 -> horizon=20，y_ret_5 -> horizon=5）
    # delta 是训练集末尾与验证集开头之间的交易日间隔，需 >= 标签实际跨越的交易日数以防止标签泄露
    # 当前标签语义为 T+1 收盘买入 / T+1+N 开盘卖出，实际跨越 N+1 个交易日，故 +1
    try:
        inferred_horizon = int(label_column.rstrip("d").split("_")[-1])
    except (ValueError, IndexError):
        inferred_horizon = 20
    label_delta = max(inferred_horizon + 1, 5)  # 最少 5 个交易日间隔

    # 按 trade_date 粒度切分训练集和验证集（确保同日样本不被拆分到两侧）
    df_train_split, df_val_split, split_stats = split_train_val_by_date(
        df_train, val_ratio=val_ratio, delta=label_delta
    )

    # 在标签变换前保存 val 原始 df 快照，用于逐日评估（保持真实收益单位）
    df_val_split_original = df_val_split.copy()

    # 如果提供了标签变换函数，切分后各自独立变换（避免跨集合统计量污染）
    if label_transform_fn is not None:
        logger.info("切分后分别对训练集与验证集独立进行标签变换...")
        df_train_split = label_transform_fn(df_train_split)
        if len(df_val_split) > 0:
            df_val_split = label_transform_fn(df_val_split)

    # 特征稳定性筛选：移除跨时期IC方向不一致的特征
    feature_filter_info = None
    if feature_stability_filter:
        feature_columns, feature_filter_info = filter_stable_features(
            df_train=df_train_split,
            feature_columns=feature_columns,
            label_column=label_column,
        )

    # 获取验证集的时间范围
    val_start_date = split_stats["val_start_date"]
    val_end_date = split_stats["val_end_date"]

    # 准备训练集 X 和 y
    X_train = df_train_split[feature_columns].copy()
    y_train = df_train_split[label_column].copy()

    # 准备验证集 X 和 y
    X_val = df_val_split[feature_columns].copy()
    y_val = df_val_split[label_column].copy()

    # NaN 处理：XGBoost / LightGBM 原生支持 NaN（自动学习缺失值的最优分裂方向），
    # 不再 fillna(0)，保留 NaN 让模型区分"无数据"与"值为0"。

    logger.info(f"训练数据准备完成: X_train shape={X_train.shape}, X_val shape={X_val.shape}")

    # 数据统计
    data_stats = {
        "samples_after_filter": samples_after_filter,
        "val_start_date": str(val_start_date),
        "val_end_date": str(val_end_date),
        "feature_filter_info": feature_filter_info,
    }

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        feature_columns,
        df_train_split,
        df_val_split,
        data_stats,
        df_val_split_original,
    )


def transform_labels_cs_zscore(
    df: pd.DataFrame, label_column: str, winsorize_p: float = 0.01
) -> pd.DataFrame:
    """对标签进行截面 winsorize + zscore 变换

    仅在训练阶段生效，对每个 trade_date 的原始回归标签进行：
    1. 截面 winsorize（截断极端值）
    2. 截面 zscore（标准化：均值=0，标准差=1）

    Args:
        df: 训练数据 DataFrame
        label_column: 标签列名
        winsorize_p: winsorize 参数，默认 0.01（截断上下1%极端值）

    Returns:
        变换后的 DataFrame（标签列已替换为标准化后的值）
    """
    # 使用别名以区别于 normalization.cross_sectional_zscore（后者处理多列 DataFrame）
    from src.lazybull.common.feature_utils import cross_sectional_zscore as _single_col_zscore

    cross_sectional_zscore = _single_col_zscore

    logger.info(f"对标签 {label_column} 进行截面 z-score 标准化...")
    logger.info(f"  winsorize 参数: {winsorize_p}")

    df_transformed = df.copy()
    nan_count_ori = df_transformed[label_column].isna().sum()
    logger.info(f"原始标签 NaN 数量: {nan_count_ori}")

    # 按 trade_date 分组进行截面标准化
    df_transformed[label_column] = cross_sectional_zscore(
        df_transformed,
        value_col=label_column,
        group_col="trade_date",
        winsorize_limits=(winsorize_p, winsorize_p),
        ddof=0,
    )

    # 统计标准化后的效果
    mean = df_transformed[label_column].mean()
    std = df_transformed[label_column].std()
    logger.info(f"标准化后: 均值={mean:.6f}, 标准差={std:.6f}")

    # 检查是否有 NaN（可能由于某天标准差为0）
    nan_count = df_transformed[label_column].isna().sum()
    if nan_count > 0:
        logger.warning(f"标准化后产生 {nan_count} 个 NaN（可能某天标准差为0），将被移除")
        df_transformed = df_transformed.dropna(subset=[label_column])

    # --- 新增：硬截断，防止标准化后依然存在离群值干扰 MSE ---
    # 哪怕 winsorize 过了，如果有极端分布，z-score 后依然可能出现 > 5 的值
    df_transformed[label_column] = df_transformed[label_column].clip(-5.0, 5.0)

    return df_transformed


def generate_classification_labels(
    df: pd.DataFrame,
    label_column: str,
    pos_quantile: Optional[float] = None,
    pos_topk: Optional[int] = None,
) -> pd.DataFrame:
    """生成分类标签（TopN 正类）

    按每个交易日截面，将原始标签按分位阈值或数量阈值转为 0/1 标签。

    Args:
        df: 训练数据 DataFrame
        label_column: 原始标签列名
        pos_quantile: 百分比阈值（例如 0.2 表示 Top20% 为正类）
        pos_topk: 数量阈值（例如 300 表示每个交易日收益最高的 300 只为正类）

    Returns:
        添加了二分类标签的 DataFrame（新增列 {label_column}_binary）

    Note:
        pos_quantile 和 pos_topk 二选一，pos_topk 优先级更高
        使用 rank(method='first') 确保 topk 数量严格等于 k（打散并列）
    """
    logger.info(f"生成分类标签（基于 {label_column}）...")

    if pos_quantile is None and pos_topk is None:
        raise ValueError("必须指定 pos_quantile 或 pos_topk 之一")

    if pos_topk is not None and pos_quantile is not None:
        logger.warning("同时指定了 pos_topk 和 pos_quantile，使用 pos_topk（优先级更高）")

    df_labeled = df.copy()
    binary_label_col = f"{label_column}_binary"

    # 初始化标签列为 NaN
    df_labeled[binary_label_col] = np.nan

    # 按 trade_date 分组，对每组的标签进行排名
    df_labeled["_rank"] = df_labeled.groupby("trade_date")[label_column].rank(
        method="first", ascending=False, na_option="keep"
    )

    if pos_topk is not None:
        # 数量模式：Top K（排名 <= K 为正类）
        df_labeled[binary_label_col] = (df_labeled["_rank"] <= pos_topk).astype(float)
        df_labeled.loc[df_labeled["_rank"].isna(), binary_label_col] = np.nan
    else:
        # 百分比模式：Top X%
        valid_counts = df_labeled.groupby("trade_date")["_rank"].transform("count")
        threshold_ranks = (valid_counts * pos_quantile).clip(lower=1).astype(int)
        df_labeled[binary_label_col] = (df_labeled["_rank"] <= threshold_ranks).astype(float)
        df_labeled.loc[df_labeled["_rank"].isna(), binary_label_col] = np.nan

    # 删除临时排名列
    df_labeled = df_labeled.drop(columns=["_rank"])

    # 统计正类比例
    total_valid = df_labeled[binary_label_col].notna().sum()
    pos_count = df_labeled[binary_label_col].sum()
    pos_ratio = pos_count / total_valid if total_valid > 0 else 0

    logger.info(f"分类标签生成完成:")
    logger.info(
        f"  模式: {'pos_topk=' + str(pos_topk) if pos_topk else 'pos_quantile=' + str(pos_quantile)}"
    )
    logger.info(f"  正类样本数: {pos_count:.0f} / {total_valid:.0f} ({pos_ratio:.2%})")

    if pos_topk is not None:
        pos_counts_per_day = df_labeled.groupby("trade_date")[binary_label_col].sum()
        logger.debug(
            f"  各交易日正类数量统计: min={pos_counts_per_day.min():.0f}, max={pos_counts_per_day.max():.0f}, mean={pos_counts_per_day.mean():.1f}"
        )

    return df_labeled


def build_time_decay_weights(
    df_train: pd.DataFrame, half_life_years: float = 1.0, date_col: str = "trade_date"
) -> np.ndarray:
    """按时间衰减构造训练样本权重

    越近的样本权重越高，使模型更重视近期市场模式。
    使用指数衰减：weight = 0.5 ^ (distance_years / half_life_years)

    - half_life_years=1.0 → 1年前的样本权重为0.5，2年前为0.25
    - half_life_years=2.0 → 2年前的样本权重为0.5，4年前为0.25（衰减更慢）

    Args:
        df_train: 训练集 DataFrame，需包含 trade_date 列
        half_life_years: 半衰期（年），即权重衰减到 0.5 所需的年数
        date_col: 日期列名

    Returns:
        与 df_train 行数相同的 numpy 数组，包含每个样本的权重（最新样本=1.0）
    """
    if date_col not in df_train.columns:
        logger.warning(f"日期列 {date_col} 不存在，返回全为1的权重")
        return np.ones(len(df_train), dtype=float)

    dates = pd.to_datetime(df_train[date_col].astype(str))
    max_date = dates.max()

    # 距最新日期的交易日数（用实际行数近似，避免需要交易日历）
    # 按唯一日期排序，给每个日期赋予序号距离
    unique_dates = sorted(dates.unique())
    date_to_rank = {d: i for i, d in enumerate(unique_dates)}
    total_days = len(unique_dates)

    ranks = dates.map(date_to_rank).values.astype(float)
    # distance: 距最新日期的交易日距离（最新=0，最旧=total_days-1）
    distance = (total_days - 1) - ranks
    distance_years = distance / 252.0

    weights = np.power(0.5, distance_years / half_life_years)

    logger.info(
        f"时间衰减权重构造完成: half_life={half_life_years}y, "
        f"训练跨度={total_days}交易日(≈{total_days/252:.1f}y), "
        f"最旧样本权重={weights.min():.4f}, 最新样本权重={weights.max():.4f}"
    )
    return weights


def build_rank_sample_weights(
    df_train: pd.DataFrame,
    label_column: str,
    topk: int = 30,
    top_weight: float = 5.0,
    date_col: str = "trade_date",
) -> np.ndarray:
    """按日截面排名构造训练样本权重

    对训练集按每个交易日截面排序，将每日 Top K 和 Bottom K 样本的权重设为
    top_weight，其余样本权重为 1.0。用于强化模型对极端头部/尾部样本的预测精度。

    处理规则：
    - 若某日样本数 <= 2*topk，则该日全部样本均设为 top_weight（防止权重分配异常）。
    - 排名依据：标签列在当日截面内的值（升序排名最小/最大分别对应 Bottom/Top）。

    Args:
        df_train: 训练集 DataFrame，需包含 trade_date 列和标签列
        label_column: 排名所用标签列名（如 neu_y_ret_20）
        topk: 每日 Top/Bottom 取前 K 个样本，默认 30
        top_weight: Top/Bottom K 样本的权重，默认 5.0
        date_col: 日期列名，默认 trade_date

    Returns:
        与 df_train 行数相同的 numpy 数组，包含每个样本的权重
    """
    weights = np.ones(len(df_train), dtype=float)

    if label_column not in df_train.columns:
        logger.warning(f"标签列 {label_column} 不存在，返回全为1的权重")
        return weights

    if date_col not in df_train.columns:
        logger.warning(f"日期列 {date_col} 不存在，返回全为1的权重")
        return weights

    # 按日截面处理
    for date, grp_idx in df_train.groupby(date_col).groups.items():
        grp = df_train.loc[grp_idx, label_column].dropna()
        n = len(grp)
        if n == 0:
            continue

        if n <= 2 * topk:
            # 样本数不足时，整组都赋予 top_weight（退化处理）
            positions = df_train.index.get_indexer_for(grp_idx)
            valid_positions = positions[positions >= 0]
            weights[valid_positions] = top_weight
            continue

        # 排序取 Top K（最大值）和 Bottom K（最小值）
        sorted_vals = grp.sort_values()
        bottom_k_idx = sorted_vals.iloc[:topk].index
        top_k_idx = sorted_vals.iloc[-topk:].index

        # 将 Top/Bottom K 的位置映射到 weights 数组位置（使用 get_indexer_for 确保正确映射）
        top_positions = df_train.index.get_indexer_for(top_k_idx)
        bottom_positions = df_train.index.get_indexer_for(bottom_k_idx)
        weights[top_positions[top_positions >= 0]] = top_weight
        weights[bottom_positions[bottom_positions >= 0]] = top_weight

    top_bottom_count = int((weights > 1.0).sum())
    logger.info(
        f"样本权重构造完成: Top/Bottom {topk} 增强，"
        f"加权样本数={top_bottom_count}，权重={top_weight}，"
        f"普通样本数={len(weights) - top_bottom_count}"
    )
    return weights


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str = "regression",
    skip_label_winsorize: bool = False,
    scale_pos_weight: Optional[float] = None,
    sample_weight: Optional[np.ndarray] = None,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    min_child_weight: int = 100,
    reg_alpha: float = 0.05,
    reg_lambda: float = 1.0,
    gamma: float = 0.1,
    objective_type: str = "mse",
    df_train_for_group: Optional[pd.DataFrame] = None,
    df_val_for_group: Optional[pd.DataFrame] = None,
    early_stopping_rounds: Optional[int] = 200,
    early_stopping_metric: str = "auto",
) -> tuple:
    """训练 XGBoost 模型（支持回归、分类和排序学习）

    Args:
        task: 任务类型，"regression" 或 "classification"
        skip_label_winsorize: 是否跳过标签 winsorize（当 label_transform=cs_zscore 时为 True）
        scale_pos_weight: 正类权重（分类任务），None 表示自动计算为 neg/pos
        sample_weight: 样本权重数组（可选），用于 Top/Bottom K 强化训练精度，
                       由 build_rank_sample_weights() 生成；None 表示不使用样本权重
        X_train: 训练特征数据
        y_train: 训练标签数据
        X_val: 验证特征数据
        y_val: 验证标签数据
        n_estimators: 树的数量
        max_depth: 树的最大深度
        learning_rate: 学习率
        subsample: 样本采样比例
        colsample_bytree: 特征采样比例
        random_state: 随机种子
        min_child_weight: 叶节点最少样本权重和，防止过拟合，默认 100（金融数据建议 100-500）
        reg_alpha: L1 正则化系数，默认 0.05
        reg_lambda: L2 正则化系数，默认 1.0
        gamma: 节点分裂最小损失下降，默认 0.1
        objective_type: 目标函数类型，"mse"（回归，默认）或 "lambdarank"（排序学习，
                        直接优化股票排序而非预测收益绝对值，与 RankIC 评估指标对齐）
        df_train_for_group: 训练集 DataFrame（仅 lambdarank 需要，用于提取 trade_date 分组信息）
        df_val_for_group: 验证集 DataFrame（仅 lambdarank 需要，用于提取 trade_date 分组信息）

    Returns:
        (model, train_params, train_metrics, val_metrics) 元组
    """
    logger.info(f"开始训练 XGBoost 模型（任务类型: {task}）...")

    # 对回归标签进行 winsorize 处理（分类标签不需要，cs_zscore 标签也不需要）
    if task == "regression" and not skip_label_winsorize:
        from scipy.stats import mstats

        y_train_processed = pd.Series(
            mstats.winsorize(y_train, limits=[0.01, 0.01]), index=y_train.index
        )
        logger.info("对回归标签进行 winsorize 处理（截断上下1%极端值），用于稳定训练")
    else:
        y_train_processed = y_train
        if task == "classification":
            logger.info("分类任务，跳过标签 winsorize 处理")
        elif skip_label_winsorize:
            logger.info("标签已在 cs_zscore 步骤中 winsorize，训练阶段跳过 winsorize")

    # 计算 scale_pos_weight（分类任务）
    computed_scale_pos_weight = None
    if task == "classification":
        pos_count = (y_train_processed == 1).sum()
        neg_count = (y_train_processed == 0).sum()

        if scale_pos_weight is None:
            if pos_count > 0:
                computed_scale_pos_weight = neg_count / pos_count
                logger.info(
                    f"自动计算 scale_pos_weight: {computed_scale_pos_weight:.4f} (负类={neg_count}, 正类={pos_count})"
                )
            else:
                logger.warning("训练集中无正类样本，无法计算 scale_pos_weight")
                computed_scale_pos_weight = 1.0
        else:
            computed_scale_pos_weight = scale_pos_weight
            logger.info(
                f"使用用户指定 scale_pos_weight: {computed_scale_pos_weight:.4f} (负类={neg_count}, 正类={pos_count})"
            )

    if sample_weight is not None:
        logger.info(f"使用样本权重（rank-weight），加权样本数={int((sample_weight > 1.0).sum())}")
    else:
        logger.info("未使用样本权重（rank-weight 未启用）")

    # 判断是否使用 LambdaRank（排序学习）
    use_lambdarank = task == "regression" and objective_type == "lambdarank"

    if use_lambdarank:
        if df_train_for_group is None:
            raise ValueError(
                "lambdarank 目标需要 df_train_for_group 参数（用于按 trade_date 分组）"
            )
        logger.info("使用 LambdaRank 排序学习目标（直接优化股票排序，与 RankIC 评估对齐）")

    # 准备训练参数
    if use_lambdarank:
        train_params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": -1,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
        }
    else:
        # 确定 eval_metric：rank_ic 用自定义函数（尺度无关，跨 split 更稳定）
        if early_stopping_metric == "rank_ic" and task == "regression":
            xgb_eval_metric = neg_rank_ic
        else:
            xgb_eval_metric = "mae" if task == "regression" else "auc"

        train_params = {
            "objective": "reg:squarederror" if task == "regression" else "binary:logistic",
            "eval_metric": xgb_eval_metric,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": -1,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
        }

    # 早停设置：early_stopping_rounds=None 或 0 表示禁用早停，使用固定 n_estimators
    if early_stopping_rounds:
        train_params["early_stopping_rounds"] = early_stopping_rounds

    # 分类任务添加 scale_pos_weight
    if task == "classification" and computed_scale_pos_weight is not None:
        train_params["scale_pos_weight"] = computed_scale_pos_weight

    # 日志中显示可读的 eval_metric 名称（callable 替换为字符串）
    log_params = {k: (v.__name__ if callable(v) else v) for k, v in train_params.items()}
    logger.info(f"训练参数: {log_params}")
    if early_stopping_rounds:
        es_metric_display = (
            early_stopping_metric
            if early_stopping_metric != "auto"
            else log_params.get("eval_metric", "mae")
        )
        logger.info(f"使用早停机制（rounds={early_stopping_rounds}, metric={es_metric_display}）")
    else:
        logger.info(f"未使用早停机制，固定训练 {n_estimators} 棵树")

    # LambdaRank 需要构造 qid（query group ID），每个 trade_date 为一个 query group
    # 同时需要将连续收益率标签转换为非负整数等级（XGBoost rank 要求）
    qid_train = None
    qid_val = None
    if use_lambdarank:
        # 按 trade_date 排序并构造 qid（同一天的股票属于同一组，组内进行排序优化）
        train_dates = df_train_for_group["trade_date"].values
        val_dates = (
            df_val_for_group["trade_date"].values
            if df_val_for_group is not None and len(df_val_for_group) > 0
            else np.array([])
        )

        # qid: 将日期映射为整数 group id
        unique_train_dates = sorted(set(train_dates))
        date_to_qid = {d: i for i, d in enumerate(unique_train_dates)}
        qid_train = np.array([date_to_qid[d] for d in train_dates])

        if len(val_dates) > 0:
            offset = len(unique_train_dates)
            unique_val_dates = sorted(set(val_dates))
            val_date_to_qid = {d: i + offset for i, d in enumerate(unique_val_dates)}
            qid_val = np.array([val_date_to_qid[d] for d in val_dates])

        logger.info(
            f"LambdaRank 分组: 训练集 {len(unique_train_dates)} 个交易日, "
            f"验证集 {len(unique_val_dates) if len(val_dates) > 0 else 0} 个交易日"
        )

        # 将连续收益率转换为按日截面排名等级 (0~31)
        # XGBoost rank:pairwise + NDCG 指数增益要求标签 <= 31
        # ~3000 只股票 / 32 级 ≈ 每级 ~94 只，粒度足够保留排序信息
        max_grade = 31

        def _returns_to_grades(y: pd.Series, dates: np.ndarray) -> pd.Series:
            """按每日截面将连续收益率转为 0~max_grade 的整数等级"""
            grades = pd.Series(0, index=y.index, dtype=int)
            for date in sorted(set(dates)):
                mask = dates == date
                daily_y = y[mask]
                n = len(daily_y)
                if n <= 1:
                    grades[mask] = 0
                else:
                    pct_rank = daily_y.rank(method="average") / n  # (0, 1]
                    grades[mask] = (pct_rank * max_grade).clip(0, max_grade).astype(int)
            return grades

        y_train_processed = _returns_to_grades(y_train_processed, train_dates)
        logger.info(f"LambdaRank 标签转换完成: 连续收益 → 排名等级 (0~{max_grade})")
        logger.info(
            f"  等级范围: {y_train_processed.min()} ~ {y_train_processed.max()}, "
            f"均值: {y_train_processed.mean():.1f}"
        )

        # 验证集标签也需要转换
        if len(val_dates) > 0:
            y_val = _returns_to_grades(y_val, val_dates)

    # 创建并训练模型
    if use_lambdarank:
        model = xgb.XGBRanker(**train_params)
    elif task == "regression":
        model = xgb.XGBRegressor(**train_params)
    else:
        model = xgb.XGBClassifier(**train_params)

    # 如果有验证集，使用早停机制
    if len(X_val) > 0:
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }
        if use_lambdarank:
            fit_kwargs["qid"] = qid_train
            if qid_val is not None:
                fit_kwargs["eval_qid"] = [qid_val]
        else:
            fit_kwargs["sample_weight"] = sample_weight

        model.fit(X_train, y_train_processed, **fit_kwargs)
        if early_stopping_rounds:
            logger.info(f"模型训练完成（最佳迭代: {model.best_iteration}）")
        else:
            logger.info(f"模型训练完成（固定 {n_estimators} 棵树）")
    else:
        fit_kwargs = {"verbose": False}
        if use_lambdarank:
            fit_kwargs["qid"] = qid_train
        else:
            fit_kwargs["sample_weight"] = sample_weight

        model.fit(X_train, y_train_processed, **fit_kwargs)
        logger.info("模型训练完成（无验证集，未使用早停）")

    importance = model.feature_importances_
    feature_names = X_train.columns
    feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    logger.info(f"Model Features sorted:")
    logger.warning(f"\n{feat_imp.head(len(feat_imp)).to_string(float_format='%.3f')}")

    # 计算训练集性能指标
    # 使用 y_train_processed（winsorize 后）与预测值比较，保持与训练目标一致
    if task == "regression":
        y_train_pred = model.predict(X_train)
        y_train_eval = pd.Series(y_train_processed, index=y_train.index)
        train_mse = mean_squared_error(y_train_eval, y_train_pred)
        train_rmse = train_mse**0.5
        train_r2 = r2_score(y_train_eval, y_train_pred)
        train_ic = y_train_eval.corr(pd.Series(y_train_pred, index=y_train.index))

        train_metrics = {
            "mse": float(train_mse),
            "rmse": float(train_rmse),
            "r2": float(train_r2),
            "ic": float(train_ic),
        }

        logger.info(
            f"训练集性能: MSE={train_mse:.6f}, RMSE={train_rmse:.6f}, R2={train_r2:.4f}, IC={train_ic:.4f}"
        )
    else:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred_binary = model.predict(X_train)

        train_acc = accuracy_score(y_train, y_train_pred_binary)
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        train_precision = precision_score(y_train, y_train_pred_binary)
        train_recall = recall_score(y_train, y_train_pred_binary)

        train_metrics = {
            "accuracy": float(train_acc),
            "auc": float(train_auc),
            "precision": float(train_precision),
            "recall": float(train_recall),
        }

        logger.info(
            f"训练集性能: ACC={train_acc:.4f}, AUC={train_auc:.4f}, Precision={train_precision:.4f}, Recall={train_recall:.4f}"
        )

    # 计算验证集性能指标
    # 注意：当使用 label_transform=cs_zscore 时，y_val 是截面 z-score 标准化后的标签（均值≈0，标准差≈1），
    #       val_mse/val_ic 等指标均在 z-score 空间计算，与 train_metrics（使用 y_train_processed，
    #       同样是处理后的标签）可比；但与真实收益单位的 val 逐日评估结果不可直接比较。
    if len(X_val) > 0:
        if task == "regression":
            y_val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_rmse = val_mse**0.5
            val_r2 = r2_score(y_val, y_val_pred)
            val_ic = y_val.corr(pd.Series(y_val_pred, index=y_val.index))
            val_rank_ic, _ = spearmanr(y_val, y_val_pred)

            val_metrics = {
                "mse": float(val_mse),
                "rmse": float(val_rmse),
                "r2": float(val_r2),
                "ic": float(val_ic),
                "rank_ic": float(val_rank_ic),
            }

            logger.info("=" * 60)
            logger.info("验证集评估结果（回归任务）")
            logger.info("=" * 60)
            logger.info(f"验证集样本数: {len(X_val)}")
            logger.info(f"MSE（均方误差）: {val_mse:.6f}")
            logger.info(f"RMSE（均方根误差）: {val_rmse:.6f}")
            logger.info(f"R2（决定系数）: {val_r2:.4f}")
            logger.info(
                f"IC（信息系数）: {val_ic:.4f}  <- 重要指标（cs_zscore 模式下为 z-score 空间）"
            )
            logger.info(f"RankIC（排序IC）: {val_rank_ic:.4f}  <- 选股策略关键指标")
            logger.info("=" * 60)
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

            y_val_pred_proba = model.predict_proba(X_val)[:, 1]
            y_val_pred_binary = model.predict(X_val)

            val_acc = accuracy_score(y_val, y_val_pred_binary)
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            val_precision = precision_score(y_val, y_val_pred_binary)
            val_recall = recall_score(y_val, y_val_pred_binary)

            val_metrics = {
                "accuracy": float(val_acc),
                "auc": float(val_auc),
                "precision": float(val_precision),
                "recall": float(val_recall),
            }

            logger.info("=" * 60)
            logger.info("验证集评估结果（分类任务）")
            logger.info("=" * 60)
            logger.info(f"验证集样本数: {len(X_val)}")
            logger.info(f"Accuracy（准确率）: {val_acc:.4f}")
            logger.info(f"AUC（ROC曲线下面积）: {val_auc:.4f}  <- 重要指标")
            logger.info(f"Precision（精确率）: {val_precision:.4f}")
            logger.info(f"Recall（召回率）: {val_recall:.4f}")
            logger.info("=" * 60)
    else:
        val_metrics = {}
        logger.warning("验证集为空，无法评估")

    # 添加 best_iteration 到 train_params
    if len(X_val) > 0 and hasattr(model, "best_iteration"):
        train_params["best_iteration"] = int(model.best_iteration)
    train_params["early_stopping_metric"] = early_stopping_metric
    # 确保 eval_metric 可序列化（callable 替换为函数名）
    if callable(train_params.get("eval_metric")):
        train_params["eval_metric"] = train_params["eval_metric"].__name__

    return model, train_params, train_metrics, val_metrics


def evaluate_validation_daily(
    model,
    df_val: pd.DataFrame,
    feature_columns: List[str],
    original_return_col: str,
    task: str,
    topk_values: Optional[List[int]] = None,
) -> Dict:
    """对验证集进行逐日评估（贴近交易场景）

    Args:
        model: 训练好的模型
        df_val: 验证集 DataFrame（包含 trade_date, ts_code, 特征列, 原始收益列）
        feature_columns: 特征列名列表
        original_return_col: 原始真实收益列名（如 y_ret_20）
        task: 任务类型
        topk_values: TopK 评估的 K 值列表

    Returns:
        逐日评估结果字典
    """
    if len(df_val) == 0:
        logger.warning("验证集为空，跳过逐日评估")
        return {}

    if original_return_col not in df_val.columns:
        logger.warning(f"验证集缺少原始收益列 {original_return_col}，跳过逐日评估")
        return {}

    if topk_values is None:
        topk_values = [30, 100, 300]

    logger.info("=" * 60)
    logger.info("验证集逐日评估（贴近交易场景）")
    logger.info("=" * 60)

    # 准备预测数据
    df_eval = df_val.copy()
    X_val_features = df_val[feature_columns].fillna(0)

    # 预测
    if task == "classification":
        y_pred_proba = model.predict_proba(X_val_features)[:, 1]
        df_eval["pred_score"] = y_pred_proba
    else:
        y_pred = model.predict(X_val_features)
        df_eval["pred_score"] = y_pred

    # 逐日评估
    daily_results = evaluate_predictions_by_date(
        df=df_eval,
        date_col="trade_date",
        prediction_col="pred_score",
        return_col=original_return_col,
        topk_values=topk_values,
    )

    # 汇总统计
    summary = summarize_daily_metrics(daily_results)

    # 输出结果
    logger.info(f"评估天数: {len(daily_results)}")
    logger.info(f"逐日 RankIC 均值: {summary.get('RankIC_均值', np.nan):.4f}")
    logger.info(f"逐日 RankIC 标准差: {summary.get('RankIC_标准差', np.nan):.4f}")
    logger.info(f"逐日 RankIC IR: {summary.get('RankIC_IR', np.nan):.4f}")

    for k in topk_values:
        mean_key = f"Top{k}平均收益_均值"
        std_key = f"Top{k}平均收益_标准差"
        if mean_key in summary:
            logger.info(
                f"Top{k} 平均收益（跨日）: 均值={summary[mean_key]:.4f}, 标准差={summary[std_key]:.4f}"
            )

    logger.info("=" * 60)

    # 计算并打印诊断统计
    diagnostics = compute_diagnostic_statistics(
        df=df_eval,
        date_col="trade_date",
        prediction_col="pred_score",
        return_col=original_return_col,
        topk_values=topk_values,
    )

    print_diagnostic_report(diagnostics)

    # 返回汇总结果（包含诊断统计）
    result = {
        "daily_rankic_mean": summary.get("RankIC_均值", np.nan),
        "daily_rankic_std": summary.get("RankIC_标准差", np.nan),
        "daily_rankic_ir": summary.get("RankIC_IR", np.nan),
        **{f"top{k}_return_mean": summary.get(f"Top{k}平均收益_均值", np.nan) for k in topk_values},
        **{
            f"top{k}_return_std": summary.get(f"Top{k}平均收益_标准差", np.nan) for k in topk_values
        },
    }

    # 添加诊断统计
    result.update({f"diagnostic_{k}": v for k, v in diagnostics.items()})

    return result


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str = "regression",
    skip_label_winsorize: bool = False,
    scale_pos_weight: Optional[float] = None,
    sample_weight: Optional[np.ndarray] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    num_leaves: Optional[int] = None,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    min_child_weight: int = 20,
    reg_alpha: float = 0.05,
    reg_lambda: float = 1.0,
    gamma: float = 0.1,
    early_stopping_rounds: Optional[int] = 200,
    early_stopping_metric: str = "auto",
) -> tuple:
    """训练 LightGBM 模型（支持回归和分类）

    参数签名与 train_xgboost_model() 基本一致，便于在训练脚本中通过 --algorithm 切换。
    gamma 参数映射为 LightGBM 的 min_split_gain。
    num_leaves 为 LightGBM 独有参数，不指定时默认 31。

    Returns:
        (model, train_params, train_metrics, val_metrics) 元组
    """
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError:
        logger.error("需要安装 lightgbm: pip install lightgbm")
        raise

    logger.info(f"开始训练 LightGBM 模型（任务类型: {task}）...")

    # 对回归标签进行 winsorize 处理
    if task == "regression" and not skip_label_winsorize:
        from scipy.stats import mstats

        y_train_processed = pd.Series(
            mstats.winsorize(y_train, limits=[0.01, 0.01]), index=y_train.index
        )
        logger.info("对回归标签进行 winsorize 处理（截断上下1%极端值），用于稳定训练")
    else:
        y_train_processed = y_train
        if task == "classification":
            logger.info("分类任务，跳过标签 winsorize 处理")
        elif skip_label_winsorize:
            logger.info("标签已在 cs_zscore 步骤中 winsorize，训练阶段跳过 winsorize")

    # 计算 scale_pos_weight（分类任务）
    computed_scale_pos_weight = None
    if task == "classification":
        pos_count = (y_train_processed == 1).sum()
        neg_count = (y_train_processed == 0).sum()

        if scale_pos_weight is None:
            if pos_count > 0:
                computed_scale_pos_weight = neg_count / pos_count
                logger.info(
                    f"自动计算 scale_pos_weight: {computed_scale_pos_weight:.4f} (负类={neg_count}, 正类={pos_count})"
                )
            else:
                logger.warning("训练集中无正类样本，无法计算 scale_pos_weight")
                computed_scale_pos_weight = 1.0
        else:
            computed_scale_pos_weight = scale_pos_weight
            logger.info(
                f"使用用户指定 scale_pos_weight: {computed_scale_pos_weight:.4f} (负类={neg_count}, 正类={pos_count})"
            )

    if sample_weight is not None:
        logger.info(f"使用样本权重（rank-weight），加权样本数={int((sample_weight > 1.0).sum())}")
    else:
        logger.info("未使用样本权重（rank-weight 未启用）")

    # num_leaves：LightGBM 的核心复杂度控制器
    # 如果未显式指定，使用默认值 31（LightGBM 官方默认值，适度复杂度）
    if num_leaves is None:
        num_leaves = 31

    # 准备训练参数（映射 XGBoost 参数到 LightGBM）
    train_params = {
        "objective": "regression" if task == "regression" else "binary",
        "metric": "mae" if task == "regression" else "auc",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "n_jobs": 10,
        "min_split_gain": gamma,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "min_child_weight": min_child_weight,
        "verbosity": -1,
    }

    if early_stopping_rounds:
        train_params["early_stopping_rounds"] = early_stopping_rounds

    logger.info(f"训练参数: {train_params}")

    # 创建模型
    if task == "regression":
        model = lgb.LGBMRegressor(**train_params)
    else:
        model = lgb.LGBMClassifier(**train_params)

    # 训练（LightGBM 使用 callbacks 进行早停）
    if early_stopping_rounds and len(X_val) > 0:
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=0),  # 静默
        ]
        es_metric_display = (
            early_stopping_metric
            if early_stopping_metric != "auto"
            else train_params.get("metric", "mae")
        )
        logger.info(f"使用早停机制（rounds={early_stopping_rounds}, metric={es_metric_display}）")

        fit_kwargs = {
            "sample_weight": sample_weight,
            "eval_set": [(X_val, y_val)],
            "callbacks": callbacks,
        }
        # 自定义早停 eval metric（rank_ic 替代 mae，尺度无关更稳定）
        if early_stopping_metric == "rank_ic" and task == "regression":
            fit_kwargs["eval_metric"] = _rank_ic_eval_lgb

        model.fit(X_train, y_train_processed, **fit_kwargs)
        logger.info(f"模型训练完成（最佳迭代: {model.best_iteration_}）")
    elif len(X_val) > 0:
        callbacks = [lgb.log_evaluation(period=0)]
        logger.info(f"未使用早停机制，固定训练 {n_estimators} 棵树")
        model.fit(
            X_train,
            y_train_processed,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        logger.info(f"模型训练完成（固定 {n_estimators} 棵树）")
    else:
        logger.info(f"未使用早停机制，固定训练 {n_estimators} 棵树")
        model.fit(
            X_train,
            y_train_processed,
            sample_weight=sample_weight,
        )
        logger.info("模型训练完成（无验证集）")

    importance = model.feature_importances_
    feature_names = X_train.columns
    feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    logger.info(f"Model Top 20 Features:")
    logger.warning(f"\n{feat_imp.head(20)}")

    # 计算训练集性能指标
    if task == "regression":
        y_train_pred = model.predict(X_train)
        y_train_eval = pd.Series(y_train_processed, index=y_train.index)
        train_mse = mean_squared_error(y_train_eval, y_train_pred)
        train_rmse = train_mse**0.5
        train_r2 = r2_score(y_train_eval, y_train_pred)
        train_ic = y_train_eval.corr(pd.Series(y_train_pred, index=y_train.index))

        train_metrics = {
            "mse": float(train_mse),
            "rmse": float(train_rmse),
            "r2": float(train_r2),
            "ic": float(train_ic),
        }

        logger.info(
            f"训练集性能: MSE={train_mse:.6f}, RMSE={train_rmse:.6f}, R2={train_r2:.4f}, IC={train_ic:.4f}"
        )
    else:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred_binary = model.predict(X_train)

        train_acc = accuracy_score(y_train, y_train_pred_binary)
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        train_precision = precision_score(y_train, y_train_pred_binary)
        train_recall = recall_score(y_train, y_train_pred_binary)

        train_metrics = {
            "accuracy": float(train_acc),
            "auc": float(train_auc),
            "precision": float(train_precision),
            "recall": float(train_recall),
        }

        logger.info(
            f"训练集性能: ACC={train_acc:.4f}, AUC={train_auc:.4f}, Precision={train_precision:.4f}, Recall={train_recall:.4f}"
        )

    # 计算验证集性能指标
    if len(X_val) > 0:
        if task == "regression":
            y_val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_rmse = val_mse**0.5
            val_r2 = r2_score(y_val, y_val_pred)
            val_ic = y_val.corr(pd.Series(y_val_pred, index=y_val.index))
            val_rank_ic, _ = spearmanr(y_val, y_val_pred)

            val_metrics = {
                "mse": float(val_mse),
                "rmse": float(val_rmse),
                "r2": float(val_r2),
                "ic": float(val_ic),
                "rank_ic": float(val_rank_ic),
            }

            logger.info("=" * 60)
            logger.info("验证集评估结果（回归任务）")
            logger.info("=" * 60)
            logger.info(f"验证集样本数: {len(X_val)}")
            logger.info(f"MSE（均方误差）: {val_mse:.6f}")
            logger.info(f"RMSE（均方根误差）: {val_rmse:.6f}")
            logger.info(f"R2（决定系数）: {val_r2:.4f}")
            logger.info(f"IC（信息系数）: {val_ic:.4f}")
            logger.info(f"RankIC（排序IC）: {val_rank_ic:.4f}  <- 选股策略关键指标")
            logger.info("=" * 60)
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

            y_val_pred_proba = model.predict_proba(X_val)[:, 1]
            y_val_pred_binary = model.predict(X_val)

            val_acc = accuracy_score(y_val, y_val_pred_binary)
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            val_precision = precision_score(y_val, y_val_pred_binary)
            val_recall = recall_score(y_val, y_val_pred_binary)

            val_metrics = {
                "accuracy": float(val_acc),
                "auc": float(val_auc),
                "precision": float(val_precision),
                "recall": float(val_recall),
            }

            logger.info("=" * 60)
            logger.info("验证集评估结果（分类任务）")
            logger.info("=" * 60)
            logger.info(f"验证集样本数: {len(X_val)}")
            logger.info(f"Accuracy（准确率）: {val_acc:.4f}")
            logger.info(f"AUC（ROC曲线下面积）: {val_auc:.4f}  <- 重要指标")
            logger.info(f"Precision（精确率）: {val_precision:.4f}")
            logger.info(f"Recall（召回率）: {val_recall:.4f}")
            logger.info("=" * 60)
    else:
        val_metrics = {}
        logger.warning("验证集为空，无法评估")

    # 添加 best_iteration 到 train_params
    if len(X_val) > 0 and hasattr(model, "best_iteration_"):
        train_params["best_iteration"] = int(model.best_iteration_)
    train_params["early_stopping_metric"] = early_stopping_metric

    return model, train_params, train_metrics, val_metrics
