# -*- coding: utf-8 -*-
"""prepare_families：v2/train/core 训练模块（P3-T2 搬运改写，§16.1-B / §16.2-bis）。

自 ``prepare.py`` 提取的特征族扩展助手群（复杂度契约 §4.4 文件二分）——语句顺序 /
数值操作顺序 / 日志调用顺序逐函数保持，不做泛化重构；与旧模块
``ml/train_core/prepare.py`` 对应段双源并存至 v2 切换。
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.lazybull.v2.common.model_schema import (
    CASHFLOW_QUALITY_SCHEMA_VERSION,
    CASHFLOW_QUALITY_VERSION_COL,
    CONSENSUS_REVISION_SCHEMA_VERSION,
    CONSENSUS_REVISION_VERSION_COL,
    DIVIDEND_POLICY_SCHEMA_VERSION,
    DIVIDEND_POLICY_VERSION_COL,
)

from .constants import (
    ALT_FEATURE_COLUMNS,
    CASHFLOW_QUALITY_FEATURE_COLUMNS,
    CONSENSUS_FEATURE_COLUMNS,
    CONSENSUS_REVISION_FEATURE_COLUMNS,
    CYQ_FEATURE_COLUMNS,
    DIVIDEND_POLICY_FEATURE_COLUMNS,
    ENHANCED_FEATURE_COLUMNS,
    EVENT_FRESHNESS_TO_VALUE_COLUMNS,
    EXPRESS_FEATURE_COLUMNS,
    FRESHNESS_STRATEGY_DROP_ALL,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY,
    FUND_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
    LHB_FEATURE_COLUMNS,
    MARGIN_FEATURE_COLUMNS,
    MISSING_MARKER_FEATURE_COLUMNS,
    NORTH_FEATURE_COLUMNS,
    STATE_FRESHNESS_COLUMNS,
)
from .features import _load_factor_exclude_list
from .runtime_derive import RuntimeDerivers


def _base_feature_columns() -> List[str]:
    """核心特征字面量列表（旧 :213-263 原样）。"""
    return [
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


def _extend_missing_markers(df: pd.DataFrame, feature_columns: List[str]) -> None:
    """估值缺失标记（旧 :265-276 原样）：按列存在性自动加入/跳过。"""
    # 估值缺失标记因子（可选：旧 schema 特征分区无此列时自动跳过，兼容旧数据直接训练）
    # 新构建的特征分区（含 dv_ttm_missing/pe_ttm_missing）加入训练，模型可显式利用缺失状态
    available_missing_markers = [col for col in MISSING_MARKER_FEATURE_COLUMNS if col in df.columns]
    if available_missing_markers:
        feature_columns.extend(available_missing_markers)
        logger.info(f"启用估值缺失标记因子: {available_missing_markers}")
    else:
        logger.warning(
            "数据中未找到估值缺失标记列 "
            f"{MISSING_MARKER_FEATURE_COLUMNS}（旧 schema 特征分区），"
            "缺失标记特征将被跳过；如需启用请重建特征"
        )


def _extend_availability_markers(
    df: pd.DataFrame,
    feature_columns: List[str],
    enable: bool,
    derivers: Optional[RuntimeDerivers],
) -> None:
    """可用性标记运行时派生（旧 :278-286；DI 化，§16.4）。"""
    # 运行时可用性标记（结构性缺失显式化；由 factors/availability.py 统一派生，
    # 不写回 cs_train/cs_infer 任何产物；推理侧由模型 feature_columns 驱动对齐）
    if not enable:
        return
    if derivers is None or derivers.availability is None:
        raise ValueError(
            "enable_availability_markers=True 需要注入可用性标记派生函数"
            "（runtime_derivers.availability；v2/train 不 import factors，由调用方"
            "组装 factors.availability.derive_availability_markers 注入）"
        )
    added_markers = derivers.availability(df)
    if added_markers:
        feature_columns.extend(added_markers)
        logger.info(f"启用可用性标记因子: {added_markers}")
    else:
        logger.warning("可用性标记未派生（来源列缺失或已存在），已跳过")


def _extend_declared_family_columns_a(
    df: pd.DataFrame,
    feature_columns: List[str],
    *,
    enable_fundamental_features: bool,
    enable_alt_features: bool,
    enable_margin_features: bool,
    enable_cyq_features: bool,
) -> None:
    """声明式族 toggle（旧 :288-322 原样）：fundamental / alt / margin / cyq。"""
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


def _extend_declared_family_columns_b(
    df: pd.DataFrame,
    feature_columns: List[str],
    *,
    enable_fund_features: bool,
    enable_express_features: bool,
    enable_enhanced_features: bool,
    enable_north_features: bool,
    enable_lhb_features: bool,
) -> None:
    """声明式族 toggle（旧 :324-367 原样）：fund / express / enhanced / north / lhb。"""
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


def _extend_consensus_columns(df: pd.DataFrame, feature_columns: List[str], enable: bool) -> None:
    """一致预期因子（旧 :369-378 原样）：fail-fast 缺列。"""
    # 一致预期因子（可选, 分析师研报聚合）
    if enable:
        missing_cons = [col for col in CONSENSUS_FEATURE_COLUMNS if col not in df.columns]
        if missing_cons:
            raise ValueError(
                "enable_consensus_features=True，但一致预期特征 schema 不完整，"
                f"缺少列: {missing_cons}。请使用 --enable-consensus 重新构建特征"
            )
        feature_columns.extend(CONSENSUS_FEATURE_COLUMNS)
        logger.info(f"启用一致预期因子: {CONSENSUS_FEATURE_COLUMNS}")


def _extend_cashflow_quality_columns(
    df: pd.DataFrame, feature_columns: List[str], enable: bool
) -> None:
    """现金流质量因子（旧 :380-411 原样；哨兵版本取自 v2.common.model_schema）。"""
    # 现金流质量因子（可选，需 cashflow 接口，2000 积分）
    if enable:
        # fail-fast：开关开启时 schema 必须完整，缺列直接失败（与一致预期开关同级）
        missing_cfq = [col for col in CASHFLOW_QUALITY_FEATURE_COLUMNS if col not in df.columns]
        if missing_cfq:
            raise ValueError(
                "enable_cashflow_quality_features=True，但现金流质量特征 schema 不完整，"
                f"缺少列: {missing_cfq}。请使用 --enable-cashflow-quality-features 重新构建特征"
            )
        # 哨兵校验：拦截旧语义缓存（ann_date 口径/累计期口径）被误用于训练
        sentinel = (
            df[CASHFLOW_QUALITY_VERSION_COL] if CASHFLOW_QUALITY_VERSION_COL in df.columns else None
        )
        sentinel_valid = (
            sentinel is not None
            and not sentinel.isna().any()
            and bool((sentinel == CASHFLOW_QUALITY_SCHEMA_VERSION).all())
        )
        if not sentinel_valid:
            raise ValueError(
                "enable_cashflow_quality_features=True，但现金流质量哨兵列 "
                f"{CASHFLOW_QUALITY_VERSION_COL} 缺失、含 NaN 或版本不等于 "
                f"{CASHFLOW_QUALITY_SCHEMA_VERSION}；当前特征分区可能为旧语义"
                "（ann_date 口径/累计期口径），请重建特征"
            )
        feature_columns.extend(CASHFLOW_QUALITY_FEATURE_COLUMNS)
        logger.info(f"启用现金流质量因子: {CASHFLOW_QUALITY_FEATURE_COLUMNS}")


def _dedup_cashflow_alias(feature_columns: List[str]) -> tuple:
    """zscore_cf_nm 构建别名去重（旧 :413-420 原样）。"""
    removed_duplicate_features: List[str] = []
    if "zscore_cf_nm" in feature_columns and "zscore_ocf_to_profit" in feature_columns:
        feature_columns.remove("zscore_cf_nm")
        removed_duplicate_features.append("zscore_cf_nm")
        logger.info(
            "训练入口因子去重: zscore_cf_nm 是 zscore_ocf_to_profit 的构建别名，"
            "两者同时启用时保留后者"
        )
    return feature_columns, removed_duplicate_features


def _extend_consensus_revision_columns(
    df: pd.DataFrame, feature_columns: List[str], enable: bool
) -> None:
    """一致预期修正因子（旧 :422-457 原样；哨兵版本取自 v2.common.model_schema）。"""
    # 一致预期修正因子（可选，基于已有 report_rc 构建时序修正信号）
    if enable:
        missing_revision = [
            col for col in CONSENSUS_REVISION_FEATURE_COLUMNS if col not in df.columns
        ]
        if missing_revision:
            raise ValueError(
                "enable_consensus_revision_features=True，但一致预期修正特征 schema 不完整，"
                f"缺少列: {missing_revision}。请使用 --enable-consensus-revision-features "
                "重新构建特征"
            )
        # v2 哨兵校验：防止 v1 旧语义缓存（列同名但经济语义错误）被误用于训练
        if CONSENSUS_REVISION_VERSION_COL not in df.columns:
            raise ValueError(
                "enable_consensus_revision_features=True，但特征分区缺少 v2 哨兵列 "
                f"{CONSENSUS_REVISION_VERSION_COL}，疑似 v1 旧语义缓存。"
                "请使用 --enable-consensus-revision-features 重新构建特征"
            )
        sentinel_series = df[CONSENSUS_REVISION_VERSION_COL]
        if (
            sentinel_series.isna().any()
            or (sentinel_series.fillna(-1) != CONSENSUS_REVISION_SCHEMA_VERSION).any()
        ):
            raise ValueError(
                "enable_consensus_revision_features=True，但特征分区的哨兵列 "
                f"{CONSENSUS_REVISION_VERSION_COL} 存在缺失或版本不等于 "
                f"{CONSENSUS_REVISION_SCHEMA_VERSION}（疑似混入 v1 旧语义分区或未构建），"
                "请重建训练窗口内全部特征"
            )
        feature_columns.extend(CONSENSUS_REVISION_FEATURE_COLUMNS)
        logger.info(f"启用一致预期修正因子: {CONSENSUS_REVISION_FEATURE_COLUMNS}")


def _extend_dividend_policy_columns(
    df: pd.DataFrame, feature_columns: List[str], enable: bool
) -> None:
    """分红政策质量因子（旧 :459-492 原样；哨兵版本取自 v2.common.model_schema）。"""
    # 分红政策质量因子（可选，需 dividend 接口，2000 积分）
    if enable:
        # fail-fast：开关开启时 schema 必须完整，缺列直接失败（与现金流/修正开关同级）
        missing_div = [col for col in DIVIDEND_POLICY_FEATURE_COLUMNS if col not in df.columns]
        if missing_div:
            raise ValueError(
                "enable_dividend_policy_features=True，但分红政策特征 schema 不完整，"
                f"缺少列: {missing_div}。请使用 --enable-dividend-policy-features 重新构建特征"
            )
        # 哨兵校验：拦截旧语义缓存被误用于训练
        if DIVIDEND_POLICY_VERSION_COL not in df.columns:
            raise ValueError(
                "enable_dividend_policy_features=True，但特征分区缺少哨兵列 "
                f"{DIVIDEND_POLICY_VERSION_COL}，疑似旧语义缓存。"
                "请使用 --enable-dividend-policy-features 重新构建特征"
            )
        sentinel_series = df[DIVIDEND_POLICY_VERSION_COL]
        if (
            sentinel_series.isna().any()
            or (sentinel_series.fillna(-1) != DIVIDEND_POLICY_SCHEMA_VERSION).any()
        ):
            raise ValueError(
                "enable_dividend_policy_features=True，但特征分区的哨兵列 "
                f"{DIVIDEND_POLICY_VERSION_COL} 存在缺失或版本不等于 "
                f"{DIVIDEND_POLICY_SCHEMA_VERSION}（疑似混入旧语义分区或未构建），"
                "请重建训练窗口内全部特征"
            )
        feature_columns.extend(DIVIDEND_POLICY_FEATURE_COLUMNS)
        logger.info(f"启用分红政策因子: {DIVIDEND_POLICY_FEATURE_COLUMNS}")


def _require_deriver(derivers: Optional[RuntimeDerivers], family: str, enable_flag_name: str):
    """取族 deriver；启用而未注入 ⇒ 报错（§16.4；与「启用未传派生表必须报错」同级）。"""
    deriver = getattr(derivers, family, None) if derivers is not None else None
    if deriver is None:
        raise ValueError(
            f"{enable_flag_name}=True 需要注入 {family} 派生能力"
            f"（runtime_derivers.{family}；v2/train 不 import factors，由调用方"
            "组装 factors 侧 derive/列集/哨兵常量注入）"
        )
    return deriver


def _extend_holdertrade_columns(
    df: pd.DataFrame,
    feature_columns: List[str],
    enable: bool,
    holdertrade_lookup: Optional[dict],
    holdertrade_feature_set: str,
    derivers: Optional[RuntimeDerivers],
) -> None:
    """股东增减持因子运行时派生（旧 :494-539 原样；DI 化，§16.4）。"""
    if enable:
        # 本族列**运行时派生**（不写回 cs_train）：与可用性标记同一先例，保证
        # cs_train 不被重建污染；查询表由调用方（CLI / walk_forward）构建并传入。
        deriver = _require_deriver(derivers, "holdertrade", "enable_holdertrade_features")
        if holdertrade_lookup is None:
            raise ValueError(
                "enable_holdertrade_features=True 需要传入 holdertrade_lookup"
                "（股东增减持因子为运行时派生：本族列不写入 cs_train）。"
                "请通过 train_ml_model.py / walk_forward.py 的"
                " --enable-holdertrade-features 启用，或先运行"
                " python scripts/download_raw.py --download stk_holdertrade"
            )
        ht_columns = deriver.feature_columns_fn(holdertrade_feature_set)
        derived_ht = deriver.derive_fn(
            df,
            holdertrade_lookup,
            wanted=ht_columns,
            log_prefix="[训练入口] ",
        )
        valid_cols = {c for c in df.columns}
        missing_ht = [col for col in ht_columns if col not in valid_cols]
        if missing_ht:
            raise ValueError(
                "enable_holdertrade_features=True，但股东增减持特征 schema 不完整，"
                f"缺少列: {missing_ht}（已尝试运行时派生 {derived_ht}）。"
                "请检查 raw/stk_holdertrade 数据与查询表日期覆盖"
            )
        # 哨兵校验：拦截旧语义特征分区（若分区自带本族列则不会被派生覆盖）被误用
        ht_sentinel = df[deriver.version_col]
        if ht_sentinel.isna().any() or (ht_sentinel.fillna(-1) != deriver.schema_version).any():
            raise ValueError(
                "enable_holdertrade_features=True，但特征分区的哨兵列 "
                f"{deriver.version_col} 存在缺失或版本不等于 "
                f"{deriver.schema_version}（疑似混入旧语义分区），"
                "请确认特征分区不含本族列（本族列一律运行时派生）"
            )
        feature_columns.extend(ht_columns)
        logger.info(
            f"启用股东增减持因子（运行时派生，列集={holdertrade_feature_set}）: {ht_columns}"
        )


def _extend_repurchase_columns(
    df: pd.DataFrame,
    feature_columns: List[str],
    enable: bool,
    repurchase_lookup: Optional[dict],
    repurchase_feature_set: str,
    derivers: Optional[RuntimeDerivers],
) -> None:
    """股票回购因子运行时派生（旧 :541-583 原样；DI 化，§16.4）。"""
    if enable:
        # 本族列同为**运行时派生**（不写回 cs_train）：查询表由调用方（CLI / walk_forward）构建。
        deriver = _require_deriver(derivers, "repurchase", "enable_repurchase_features")
        if repurchase_lookup is None:
            raise ValueError(
                "enable_repurchase_features=True 需要传入 repurchase_lookup"
                "（股票回购因子为运行时派生：本族列不写入 cs_train）。"
                "请通过 train_ml_model.py / walk_forward.py 的"
                " --enable-repurchase-features 启用，或先运行"
                " python scripts/download_raw.py --download repurchase"
            )
        rp_columns = deriver.feature_columns_fn(repurchase_feature_set)
        derived_rp = deriver.derive_fn(
            df,
            repurchase_lookup,
            wanted=rp_columns,
            log_prefix="[训练入口] ",
        )
        valid_cols = {c for c in df.columns}
        missing_rp = [col for col in rp_columns if col not in valid_cols]
        if missing_rp:
            raise ValueError(
                "enable_repurchase_features=True，但股票回购特征 schema 不完整，"
                f"缺少列: {missing_rp}（已尝试运行时派生 {derived_rp}）。"
                "请检查 raw/repurchase 数据与查询表日期覆盖"
            )
        # 哨兵校验：拦截旧语义特征分区（若分区自带本族列则不会被派生覆盖）被误用
        rp_sentinel = df[deriver.version_col]
        if rp_sentinel.isna().any() or (rp_sentinel.fillna(-1) != deriver.schema_version).any():
            raise ValueError(
                "enable_repurchase_features=True，但特征分区的哨兵列 "
                f"{deriver.version_col} 存在缺失或版本不等于 "
                f"{deriver.schema_version}（疑似混入旧语义分区），"
                "请确认特征分区不含本族列（本族列一律运行时派生）"
            )
        feature_columns.extend(rp_columns)
        logger.info(f"启用股票回购因子（运行时派生，列集={repurchase_feature_set}）: {rp_columns}")


def _extend_top10fh_columns(
    df: pd.DataFrame,
    feature_columns: List[str],
    enable: bool,
    top10fh_panel: Optional[pd.DataFrame],
    top10fh_feature_set: str,
    derivers: Optional[RuntimeDerivers],
) -> None:
    """十大流通股东因子运行时派生（旧 :585-631 原样；DI 化，§16.4）。"""
    if enable:
        # 本族列同为**运行时派生**（不写回 cs_train）：报告期面板由调用方（CLI / walk_forward）构建。
        deriver = _require_deriver(derivers, "top10fh", "enable_top10fh_features")
        if top10fh_panel is None:
            raise ValueError(
                "enable_top10fh_features=True 需要传入 top10fh_panel"
                "（十大流通股东因子为运行时派生：本族列不写入 cs_train）。"
                "请通过 train_ml_model.py / walk_forward.py 的"
                " --enable-top10fh-features 启用，或先运行"
                " python scripts/download_raw.py --download top10_floatholders"
            )
        tfh_columns = deriver.feature_columns_fn(top10fh_feature_set)
        derived_tfh = deriver.derive_fn(
            df,
            top10fh_panel,
            wanted=tfh_columns,
            log_prefix="[训练入口] ",
        )
        valid_cols = {c for c in df.columns}
        missing_tfh = [col for col in tfh_columns if col not in valid_cols]
        if missing_tfh:
            raise ValueError(
                "enable_top10fh_features=True，但十大流通股东特征 schema 不完整，"
                f"缺少列: {missing_tfh}（已尝试运行时派生 {derived_tfh}）。"
                "请检查 raw/top10_floatholders 数据与报告期面板"
            )
        # 哨兵校验：拦截旧语义特征分区（若分区自带本族列则不会被派生覆盖）被误用
        tfh_sentinel = df[deriver.version_col]
        if tfh_sentinel.isna().any() or (tfh_sentinel.fillna(-1) != deriver.schema_version).any():
            raise ValueError(
                "enable_top10fh_features=True，但特征分区的哨兵列 "
                f"{deriver.version_col} 存在缺失或版本不等于 "
                f"{deriver.schema_version}（疑似混入旧语义分区），"
                "请确认特征分区不含本族列（本族列一律运行时派生）"
            )
        feature_columns.extend(tfh_columns)
        logger.info(
            f"启用十大流通股东因子（运行时派生，列集={top10fh_feature_set}）: {tfh_columns}"
        )


def _extend_top_inst_columns(
    df: pd.DataFrame,
    feature_columns: List[str],
    enable: bool,
    top_inst_lookup: Optional[dict],
    derivers: Optional[RuntimeDerivers],
) -> None:
    """龙虎榜机构席位因子运行时派生（旧 :633-676 原样；DI 化，§16.4；已终结族代码保留）。"""
    if enable:
        # 本族列同为**运行时派生**（不写回 cs_train）：查询表由调用方（CLI / walk_forward）构建。
        deriver = _require_deriver(derivers, "top_inst", "enable_top_inst_features")
        if top_inst_lookup is None:
            raise ValueError(
                "enable_top_inst_features=True 需要传入 top_inst_lookup"
                "（龙虎榜机构席位因子为运行时派生：本族列不写入 cs_train）。"
                "请通过 walk_forward.py 的 --enable-top-inst-features 启用，"
                "且 raw/top_inst 年分区已下载"
            )
        ti_columns = deriver.feature_columns_fn()
        derived_ti = deriver.derive_fn(
            df,
            top_inst_lookup,
            wanted=ti_columns,
            log_prefix="[训练入口] ",
        )
        valid_cols = {c for c in df.columns}
        missing_ti = [col for col in ti_columns if col not in valid_cols]
        if missing_ti:
            raise ValueError(
                "enable_top_inst_features=True，但龙虎榜机构席位特征 schema 不完整，"
                f"缺少列: {missing_ti}（已尝试运行时派生 {derived_ti}）。"
                "请检查 raw/top_inst 数据与查询表日期覆盖"
            )
        # 哨兵校验：拦截旧语义特征分区（若分区自带本族列则不会被派生覆盖）被误用
        ti_sentinel = df[deriver.version_col]
        if ti_sentinel.isna().any() or (ti_sentinel.fillna(-1) != deriver.schema_version).any():
            raise ValueError(
                "enable_top_inst_features=True，但特征分区的哨兵列 "
                f"{deriver.version_col} 存在缺失或版本不等于 "
                f"{deriver.schema_version}（疑似混入旧语义分区），"
                "请确认特征分区不含本族列（本族列一律运行时派生）"
            )
        feature_columns.extend(ti_columns)
        logger.info(f"启用龙虎榜机构席位因子（运行时派生）: {ti_columns}")


def _append_sz_columns(df: pd.DataFrame, feature_columns: List[str]) -> None:
    """市值中性化特征自动发现（旧 :678-689 原样）。"""
    # ── 市值中性化特征：仅纳入核心特征列表中稳定因子对应的 zscore_*_sz 列 ──
    # 避免稀疏因子（如一致预期）的 _sz 列在不同日期间存在/缺失导致 schema 不一致
    sz_cols = [
        c
        for c in df.columns
        if c.startswith("zscore_")
        and c.endswith("_sz")
        and c[:-3] in feature_columns  # 仅当基础 zscore_* 列已在核心特征列表中
    ]
    if sz_cols:
        feature_columns.extend(sz_cols)
        logger.info(f"自动发现市值中性化特征: {len(sz_cols)} 个 (zscore_*_sz)")


def _apply_factor_prune(
    feature_columns: List[str], factor_prune: bool, factor_exclude_file: Optional[str]
) -> List[str]:
    """因子精简（旧 :691-710 原样）：排除清单 + zscore 成对联动剔除。"""
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
    return feature_columns


def _resolve_freshness_columns(feature_columns: List[str], freshness_strategy: str) -> tuple:
    """freshness 策略处理（旧 :711-750 原样）。"""
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

    return (
        feature_columns,
        removed_freshness_features,
        event_freshness_cols_used,
        state_freshness_cols_kept,
        decay_helper_cols,
    )
