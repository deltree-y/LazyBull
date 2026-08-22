# -*- coding: utf-8 -*-
"""prepare：train_core 拆分模块。"""

from loguru import logger
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
import gc
import numpy as np
import pandas as pd

from .constants import (
    ALT_FEATURE_COLUMNS,
    CASHFLOW_QUALITY_FEATURE_COLUMNS,
    CONSENSUS_FEATURE_COLUMNS,
    CONSENSUS_REVISION_FEATURE_COLUMNS,
    CYQ_FEATURE_COLUMNS,
    ENHANCED_FEATURE_COLUMNS,
    EVENT_FRESHNESS_TO_VALUE_COLUMNS,
    EXPRESS_FEATURE_COLUMNS,
    FRESHNESS_STRATEGY_DROP_ALL,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY,
    FUNDAMENTAL_FEATURE_COLUMNS,
    FUND_FEATURE_COLUMNS,
    LHB_FEATURE_COLUMNS,
    MARGIN_FEATURE_COLUMNS,
    MISSING_MARKER_FEATURE_COLUMNS,
    NORTH_FEATURE_COLUMNS,
    STATE_FRESHNESS_COLUMNS,
)
from .split import (
    split_train_val_by_date,
    split_val_for_selection_protocol_by_date,
)
from .features import (
    _load_factor_exclude_list,
    filter_stable_features,
)
from .freshness import apply_event_freshness_decay


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
    enable_cashflow_quality_features: bool = False,
    enable_consensus_revision_features: bool = False,
    feature_stability_filter: bool = False,
    factor_prune: bool = False,
    factor_exclude_file: Optional[str] = None,
    max_feature_missing_ratio: float = 0.6,
    feature_columns_override: Optional[List[str]] = None,
    freshness_strategy: str = FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
    event_freshness_half_life_days: float = 45.0,
) -> tuple:
    """准备训练数据，并按 trade_date 粒度切分训练集和验证集

    Args:
        df: 特征 DataFrame
        label_column: 标签列名
        val_ratio: 验证集比例，默认 0.2（最后 20% 的交易日作为验证集）
        label_transform_fn: 可选的标签变换函数，接受 DataFrame 并返回变换后的 DataFrame。
            若提供，将在按日切分后分别对训练集与验证集独立调用，避免跨集合统计量污染。
            典型用法：cs_zscore 变换（见 transform_labels_cs_zscore）。
        factor_prune: 是否启用因子精简（从 data/models/factor_exclude_list.json
            加载排除列表并过滤特征列）。默认 False。
        factor_exclude_file: 因子精简使用的显式清单路径；未提供时使用默认生产清单。
        max_feature_missing_ratio: 训练入口特征缺失率上限，超过该阈值的特征将被移除。
            默认 0.6。
        feature_columns_override: 若提供，则在训练入口特征质量门禁之后强制将特征列
            对齐到该列表（用于多窗口集成统一子模型特征 schema），数据中缺失的列补 NaN。
            默认 None（不启用）。
        freshness_strategy: freshness 处理策略。
            - state_keep_event_decay（默认）：状态型 freshness 保留，事件型 freshness 仅用于衰减对应特征值
            - state_keep_event_no_decay：状态型 freshness 保留，事件型 freshness 删除且不衰减对应特征值
            - drop_all：删除全部 freshness 特征，不做衰减
        event_freshness_half_life_days: 事件型特征衰减半衰期（天），默认 45。

    Returns:
                (X_train, y_train, X_val, y_val, feature_columns, df_train_split, df_val_split, data_stats,
                 df_val_split_original) 元组
                - df_val_split: early stopping 子集（仅供模型训练/early stopping 使用）
                - df_val_split_original: calibration 子集的原始快照；若 calibration 为空则回退到
                    early stopping 子集。供逐日收益评估、候选比较与验证评估使用。
                data_stats 包含：samples_after_filter, val_start_date, val_end_date 及
                    val_es_* / val_calib_* / val_embargo_* 统计
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

    # 估值缺失标记因子（可选：旧 schema 特征分区无此列时自动跳过，兼容旧数据直接训练）
    # 新构建的特征分区（含 dv_ttm_missing/pe_ttm_missing）加入训练，模型可显式利用缺失状态
    available_missing_markers = [
        col for col in MISSING_MARKER_FEATURE_COLUMNS if col in df.columns
    ]
    if available_missing_markers:
        feature_columns.extend(available_missing_markers)
        logger.info(f"启用估值缺失标记因子: {available_missing_markers}")
    else:
        logger.warning(
            "数据中未找到估值缺失标记列 "
            f"{MISSING_MARKER_FEATURE_COLUMNS}（旧 schema 特征分区），"
            "缺失标记特征将被跳过；如需启用请重建特征"
        )

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

    # 现金流质量因子（可选，需 cashflow 接口，2000 积分）
    if enable_cashflow_quality_features:
        available_cfq = [col for col in CASHFLOW_QUALITY_FEATURE_COLUMNS if col in df.columns]
        if available_cfq:
            feature_columns.extend(available_cfq)
            logger.info(f"启用现金流质量因子: {available_cfq}")
        else:
            logger.warning(
                "enable_cashflow_quality_features=True，但数据中未找到现金流质量列，跳过"
            )

    # 一致预期修正因子（可选，基于已有 report_rc 构建时序修正信号）
    if enable_consensus_revision_features:
        available_cr = [
            col for col in CONSENSUS_REVISION_FEATURE_COLUMNS if col in df.columns
        ]
        if available_cr:
            feature_columns.extend(available_cr)
            logger.info(f"启用一致预期修正因子: {available_cr}")
        else:
            logger.warning(
                "enable_consensus_revision_features=True，但数据中未找到一致预期修正列，跳过"
            )

    # ── 市值中性化特征：仅纳入核心特征列表中稳定因子对应的 zscore_*_sz 列 ──
    # 避免稀疏因子（如一致预期）的 _sz 列在不同日期间存在/缺失导致 schema 不一致
    sz_cols = [
        c for c in df.columns
        if c.startswith("zscore_") and c.endswith("_sz")
        and c[:-3] in feature_columns  # 仅当基础 zscore_* 列已在核心特征列表中
    ]
    if sz_cols:
        feature_columns.extend(sz_cols)
        logger.info(f"自动发现市值中性化特征: {len(sz_cols)} 个 (zscore_*_sz)")

    # ── 因子精简（可选）──
    if factor_prune:
        exclude_list = _load_factor_exclude_list(
            exclude_file=Path(factor_exclude_file) if factor_exclude_file else None
        )
        if exclude_list:
            # 联动剔除：
            # 1) 排除 zscore_x 时同步排除 zscore_x_sz
            # 2) 排除 zscore_x_sz 时同步排除 zscore_x
            expanded_excludes = set(exclude_list)
            for col in list(exclude_list):
                if col.startswith("zscore_") and not col.endswith("_sz"):
                    expanded_excludes.add(f"{col}_sz")
                if col.startswith("zscore_") and col.endswith("_sz"):
                    expanded_excludes.add(col[:-3])

            before = len(feature_columns)
            feature_columns = [c for c in feature_columns if c not in expanded_excludes]
            removed = before - len(feature_columns)
            logger.info(f"因子精简: 排除 {removed} 个因子, 剩余 {len(feature_columns)} 个")

    freshness_cols = [c for c in feature_columns if "freshness" in c]
    removed_freshness_features: List[str] = []
    event_freshness_cols_used: List[str] = []
    state_freshness_cols_kept: List[str] = []
    decay_helper_cols: List[str] = []

    if freshness_strategy not in {
        FRESHNESS_STRATEGY_DROP_ALL,
        FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
        FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY,
    }:
        raise ValueError(
            "freshness_strategy 非法，"
            f"可选: {FRESHNESS_STRATEGY_DROP_ALL} | "
            f"{FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY} | "
            f"{FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY}"
        )

    if freshness_strategy == FRESHNESS_STRATEGY_DROP_ALL:
        removed_freshness_features = sorted(freshness_cols)
        if removed_freshness_features:
            feature_columns = [c for c in feature_columns if c not in removed_freshness_features]
            logger.info(f"训练入口移除 freshness 特征: {len(removed_freshness_features)} 个")
    else:
        state_freshness_cols_kept = sorted(
            [c for c in freshness_cols if c in STATE_FRESHNESS_COLUMNS]
        )
        event_freshness_cols_used = sorted(
            [c for c in freshness_cols if c in EVENT_FRESHNESS_TO_VALUE_COLUMNS]
        )
        removed_freshness_features = sorted(event_freshness_cols_used)
        if removed_freshness_features:
            feature_columns = [c for c in feature_columns if c not in removed_freshness_features]
            logger.info(
                f"freshness 策略({freshness_strategy}): "
                f"保留状态型 {len(state_freshness_cols_kept)} 列，"
                f"移除事件型 freshness {len(removed_freshness_features)} 列"
            )
        if freshness_strategy == FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY:
            decay_helper_cols = list(event_freshness_cols_used)

    # 去重，保持顺序稳定
    feature_columns = list(dict.fromkeys(feature_columns))

    if not feature_columns:
        raise ValueError("特征列为空（在因子精简/freshness 策略后）")

    logger.info(f"特征列数量: {len(feature_columns)}")
    logger.debug(f"特征列: {feature_columns[:10]}...")  # 只显示前10个

    # ── 内存优化：只保留训练必需的列，避免 copy 时 OOM ──
    needed_cols = set(
        feature_columns + [label_column, "trade_date", "ts_code"] + filter_columns + decay_helper_cols
    )
    needed_cols &= set(df.columns)  # 仅保留实际存在的列
    kept = list(needed_cols)
    n_before = len(df.columns)
    df = df[kept]
    logger.debug(f"内存优化: {n_before} → {len(df.columns)} 列（仅保留训练必需列）")

    # 过滤可训练样本（移除含有过滤标记的样本）
    mask = pd.Series([True] * len(df), index=df.index)
    for col in filter_columns:
        if col in df.columns:
            mask = mask & (~df[col].astype(bool))

    # df[mask]（布尔索引）已复制独立数据，无需再 deep copy；后续
    # df_train[col] = np.nan 等写操作作用于独立副本，不会写穿 df。
    # 大训练集上 .copy() 会触发 BlockManager 合并导致额外全量内存分配（OOM）。
    df_train = df[mask]
    logger.info(f"过滤后样本数: {len(df_train)} / {len(df)}")
    samples_after_filter = len(df_train)

    # 移除标签为 NaN 的样本
    df_train = df_train.dropna(subset=[label_column])
    logger.info(f"移除标签 NaN 后样本数: {len(df_train)}")

    if len(df_train) == 0:
        raise ValueError("没有可用的训练样本")

    decay_applied_stats: Dict[str, int] = {}
    if (
        freshness_strategy == FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY
        and event_freshness_cols_used
    ):
        df_train, decay_applied_stats = apply_event_freshness_decay(
            df_train,
            event_freshness_cols=event_freshness_cols_used,
            half_life_days=float(event_freshness_half_life_days),
        )
        if decay_applied_stats:
            logger.info(
                "事件型 freshness 衰减已应用: "
                + ", ".join(
                    [f"{k}->{v}" for k, v in sorted(decay_applied_stats.items())]
                )
            )

    # 训练入口特征质量门禁：删除高缺失、全空、常数列
    available_feature_cols = [c for c in feature_columns if c in df_train.columns]
    removed_high_missing: List[str] = []
    removed_all_nan: List[str] = []
    removed_constant: List[str] = []
    if available_feature_cols:
        missing_ratio = df_train[available_feature_cols].isna().mean()
        removed_high_missing = sorted(
            missing_ratio[missing_ratio > float(max_feature_missing_ratio)].index.tolist()
        )
        removed_all_nan = sorted(missing_ratio[missing_ratio >= 1.0].index.tolist())

        candidate_cols = [c for c in available_feature_cols if c not in removed_high_missing]
        if candidate_cols:
            nunique = df_train[candidate_cols].nunique(dropna=True)
            removed_constant = sorted(nunique[nunique <= 1].index.tolist())

        base_removed = set(removed_high_missing) | set(removed_all_nan) | set(removed_constant)
        linked_removed: Set[str] = set(base_removed)
        for col in list(base_removed):
            if col.startswith("zscore_") and col.endswith("_sz"):
                linked_removed.add(col[:-3])
            if col.startswith("zscore_") and not col.endswith("_sz"):
                linked_removed.add(f"{col}_sz")

        if linked_removed:
            before = len(feature_columns)
            feature_columns = [c for c in feature_columns if c not in linked_removed]
            removed_count = before - len(feature_columns)
            logger.info(
                "训练入口特征清洗: "
                f"移除 {removed_count} 列（高缺失>{max_feature_missing_ratio}: {len(removed_high_missing)}，"
                f"全空: {len(removed_all_nan)}，常数: {len(removed_constant)}）"
            )
            # 打印各类移除列的详细名称，便于定位数据链路问题（全空是高缺失的子集，计数独立）
            detail_lines: List[str] = []
            if removed_high_missing:
                detail_lines.append(
                    f"高缺失>{max_feature_missing_ratio}（{len(removed_high_missing)}列）: "
                    f"{', '.join(removed_high_missing)}"
                )
            if removed_all_nan:
                detail_lines.append(
                    f"全空（{len(removed_all_nan)}列）: {', '.join(removed_all_nan)}"
                )
            if removed_constant:
                detail_lines.append(
                    f"常数（{len(removed_constant)}列）: {', '.join(removed_constant)}"
                )
            linked_extra = sorted(linked_removed - base_removed)
            if linked_extra:
                detail_lines.append(
                    f"联动移除（{len(linked_extra)}列）: {', '.join(linked_extra)}"
                )
            if detail_lines:
                logger.info("训练入口特征清洗明细:\n" + "\n".join(detail_lines))

    if not feature_columns:
        raise ValueError("特征列为空（在缺失率/常数列过滤后）")

    # 多窗口集成特征列统一：以基础窗口特征列为准，保证集成子模型特征 schema 一致。
    # 不同窗口稀疏因子缺失率不同，高缺失门禁可能产生不同特征列（如 express_revenue_yoy），
    # 若不统一会导致集成预测时 XGBoost feature_names mismatch。
    if feature_columns_override is not None:
        override_missing_in_data = [
            c for c in feature_columns_override if c not in df_train.columns
        ]
        for col in override_missing_in_data:
            df_train[col] = np.nan
        if len(feature_columns) != len(feature_columns_override):
            logger.info(
                "训练入口特征列统一(override): "
                f"{len(feature_columns)} -> {len(feature_columns_override)} 列（以基础窗口为准）"
            )
        if override_missing_in_data:
            logger.warning(
                "训练入口特征列统一(override): "
                f"当前窗口数据缺失列补 NaN {len(override_missing_in_data)} 个: "
                f"{override_missing_in_data}"
            )
        feature_columns = list(feature_columns_override)
        if not feature_columns:
            raise ValueError("特征列为空（在 override 统一后）")

    # 从标签列名自动推断 delta（例如 neu_y_ret_20 -> horizon=20，y_ret_5 -> horizon=5）
    # delta 是训练集末尾与验证集开头之间的交易日间隔，需 >= 标签实际跨越的交易日数以防止标签泄露
    # 当前标签语义为 T+1 收盘买入 / T+1+N 开盘卖出，实际跨越 N+1 个交易日，故 +1
    try:
        inferred_horizon = int(label_column.rstrip("d").split("_")[-1])
    except (ValueError, IndexError):
        inferred_horizon = 20
    label_delta = max(inferred_horizon + 1, 5)  # 最少 5 个交易日间隔

    # 按 trade_date 粒度切分训练集和验证集（确保同日样本不被拆分到两侧）
    df_train_split, df_val_split_raw, split_stats = split_train_val_by_date(
        df_train, val_ratio=val_ratio, delta=label_delta
    )

    # 从验证集尾部自动隔离与测试期可能重叠的标签窗口样本，
    # 仅隔离后的子集参与 early stopping / best_iteration 选择。
    (
        df_val_split,
        df_val_split_calib,
        df_val_split_embargo,
        val_protocol_stats,
    ) = split_val_for_selection_protocol_by_date(
        df_val_split_raw,
        embargo_days=label_delta,
    )
    logger.info(
        "验证集协议拆分（按标签自动推导）: "
        f"raw={val_protocol_stats['val_raw_start_date']}~{val_protocol_stats['val_raw_end_date']} "
        f"({val_protocol_stats['val_raw_n_dates']}日/{val_protocol_stats['val_raw_samples']}样本), "
        f"es={val_protocol_stats['val_es_start_date']}~{val_protocol_stats['val_es_end_date']} "
        f"({val_protocol_stats['val_es_n_dates']}日/{val_protocol_stats['val_es_samples']}样本), "
        f"calib={val_protocol_stats['val_calib_start_date']}~{val_protocol_stats['val_calib_end_date']} "
        f"({val_protocol_stats['val_calib_n_dates']}日/{val_protocol_stats['val_calib_samples']}样本), "
        f"embargo={val_protocol_stats['val_embargo_start_date']}~{val_protocol_stats['val_embargo_end_date']} "
        f"({val_protocol_stats['val_embargo_n_dates']}日/{val_protocol_stats['val_embargo_samples']}样本), "
        f"embargo_days={val_protocol_stats['val_embargo_days_requested']}"
    )

    # 在标签变换前保存 calibration 原始 df 引用；若 calibration 为空则回退到 es 子集。
    # label_transform_fn 内部会创建新 df，原始 df 不会被修改，无需深拷贝。
    df_val_split_original = (
        df_val_split_calib if len(df_val_split_calib) > 0 else df_val_split
    )

    # 释放已不再需要的原始/中间 DataFrame，回收 ~4-5 GiB 内存
    # 提前保存后续 data_stats 需要的值
    _val_raw_samples = len(df_val_split_raw)
    del df, df_train, df_val_split_raw
    gc.collect()

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

    # 获取用于 early stopping 的验证集时间范围（标签变换后可能进一步收缩）
    val_es_dates = sorted(df_val_split["trade_date"].unique()) if len(df_val_split) > 0 else []
    val_start_date = str(val_es_dates[0]) if val_es_dates else "N/A"
    val_end_date = str(val_es_dates[-1]) if val_es_dates else "N/A"

    # 准备训练集 X 和 y（float32 内存减半，XGBoost/LightGBM 原生支持）
    X_train = df_train_split[feature_columns].astype(np.float32)
    y_train = df_train_split[label_column].astype(np.float32)

    # 准备验证集 X 和 y
    X_val = df_val_split[feature_columns].astype(np.float32)
    y_val = df_val_split[label_column].astype(np.float32)

    # 显式释放中间 DataFrame 引用，帮助 GC 回收内存
    gc.collect()

    # NaN 处理：XGBoost / LightGBM 原生支持 NaN（自动学习缺失值的最优分裂方向），
    # 不再 fillna(0)，保留 NaN 让模型区分"无数据"与"值为0"。

    logger.info(f"训练数据准备完成: X_train shape={X_train.shape}, X_val shape={X_val.shape}")

    # 数据统计
    data_stats = {
        "samples_after_filter": samples_after_filter,
        "train_start_date": split_stats["train_start_date"],
        "train_end_date": split_stats["train_end_date"],
        "val_start_date": str(val_start_date),
        "val_end_date": str(val_end_date),
        "val_raw_start_date": split_stats["val_start_date"],
        "val_raw_end_date": split_stats["val_end_date"],
        "val_raw_n_dates": split_stats["val_n_dates"],
        "val_raw_samples": _val_raw_samples,
        "val_es_start_date": str(val_start_date),
        "val_es_end_date": str(val_end_date),
        "val_es_n_dates": len(val_es_dates),
        "val_es_samples": len(df_val_split),
        "val_calib_start_date": val_protocol_stats["val_calib_start_date"],
        "val_calib_end_date": val_protocol_stats["val_calib_end_date"],
        "val_calib_n_dates": val_protocol_stats["val_calib_n_dates"],
        "val_calib_samples": len(df_val_split_original),
        "val_embargo_days": label_delta,
        "val_embargo_days_applied": val_protocol_stats["val_embargo_days_applied"],
        "val_embargo_n_dates": val_protocol_stats["val_embargo_n_dates"],
        "val_embargo_samples": len(df_val_split_embargo),
        "val_embargo_start_date": val_protocol_stats["val_embargo_start_date"],
        "val_embargo_end_date": val_protocol_stats["val_embargo_end_date"],
        "val_calibration_ratio": val_protocol_stats["val_calibration_ratio"],
        "feature_filter_info": feature_filter_info,
        "max_feature_missing_ratio": float(max_feature_missing_ratio),
        "removed_high_missing_features": removed_high_missing,
        "removed_all_nan_features": removed_all_nan,
        "removed_constant_features": removed_constant,
        "freshness_strategy": freshness_strategy,
        "event_freshness_half_life_days": float(event_freshness_half_life_days),
        "removed_freshness_features": removed_freshness_features,
        "kept_state_freshness_features": state_freshness_cols_kept,
        "event_freshness_columns_used": event_freshness_cols_used,
        "event_decay_applied_stats": decay_applied_stats,
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
