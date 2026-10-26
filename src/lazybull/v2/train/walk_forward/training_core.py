#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward 训练核心函数（v2 侧）。

P3-T3 搬运改写自 src.lazybull.ml.walk_forward.training_core（2026-10-22），改动点（§17.1-B）：

1. **数据入口上移**：``_train_model_on_window`` 不再自行 ``load_features_data``，
   改接收调用方经适配层 + ``TrainingFrameCache`` 加载的 ``TrainingFrame``
   （同折 3 种子共享一帧，T2 加载成本门）；四族 lookup 数据表由
   ``frame.runtime_lookups`` 携带（cs_train 验证源模式），panel 模式四族全关
   （物化直读，适配层校验防双写）；
2. **参数面 args namespace → ``TrainFoldParams`` frozen dataclass**（字段默认值 =
   旧 cli.py 默认值逐项一致；映射对照测试锁定）；
3. **lgb 分支移除**（Q6-A）：``algorithm != "xgboost"`` ⇒ ValueError（拒绝路径
   一并清理，``train_lightgbm_model`` 不再 import）；
4. **显式日期缝透传**：``explicit_train_end`` / ``explicit_val_start`` 透传 prepare
   （F10 D1：比例→日期换算归编排器，``resolve_explicit_split_dates`` 为换算 helper，
   复算 ``split_train_val_by_date`` 同数学的日历日域表达——现役语义前提 = 过滤后
   日域与外层窗日历日域一致，偏差由 §7.6 逐折对账门（R2-06）兜住）；
5. ``_train_model_on_window``（旧 178 行在册）按 §17.8 最小提取式拆分；
6. import 改指 ``v2.train.core``（T2 落点）；旧 ``data.DataLoader/Storage`` 仅
   保留类型注解引用（验收专用过渡例外，cs_train_verify 路径，去除节点 P4 评审）。
"""

import copy
import gc
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

from src.lazybull.v2.train.core import (
    RuntimeDerivers,
    add_blended_return_label,
    build_rank_sample_weights,
    build_time_decay_weights,
    evaluate_validation_daily,
    generate_classification_labels,
    prepare_training_data,
    train_xgboost_model,
    transform_labels_cs_zscore,
)
from src.lazybull.v2.train.data_adapter import TrainingFrame

from .training_reporting import _safe_float

SEED_ENSEMBLE_KEEP_TOP_RATIO = 0.30
MIN_MODELS = 3
SEED_ENSEMBLE_KEEP_MIN_MODELS = MIN_MODELS


@dataclass(frozen=True)
class TrainFoldParams:
    """单折训练参数面（P3-T3 配置层；字段默认值 = 旧 cli.py 默认值逐项一致）。

    由薄入口/宿主从 ``ModelConfig``（label_spec / feature_recipe / split_spec /
    hyperparams）组装；``ensemble_seeds`` 承载 ``Sequence[int]``（字符串解析归
    薄入口，§13.3.1；逗号字符串输入兼容保留旧解析路径）。四族数据查询表不经
    本对象（由 ``TrainingFrame.runtime_lookups`` 携带）；派生函数集合经
    ``runtime_derivers`` 独立注入（DI，v2/train 不 import factors）。
    """

    # 标签/任务
    label_column: str = "y_ret_5"
    neutral_label_blend_weight: float = 0.0
    task: str = "regression"
    label_transform: str = "raw"
    winsorize_p: float = 0.01
    pos_quantile: Optional[float] = None
    pos_topk: Optional[int] = None
    scale_pos_weight: Optional[float] = None
    # 切分
    val_ratio: float = 0.1
    # 特征族开关（13 族 + 可用性标记）
    enable_fundamental_features: bool = False
    enable_alt_features: bool = False
    enable_margin_features: bool = False
    enable_cyq_features: bool = False
    enable_fund_features: bool = False
    enable_express_features: bool = False
    enable_enhanced_features: bool = False
    enable_north_features: bool = False
    enable_lhb_features: bool = False
    enable_consensus_features: bool = False
    enable_cashflow_quality_features: bool = False
    enable_consensus_revision_features: bool = False
    enable_dividend_policy_features: bool = False
    enable_availability_markers: bool = False
    enable_holdertrade_features: bool = False
    holdertrade_feature_set: str = "full"
    enable_repurchase_features: bool = False
    repurchase_feature_set: str = "full"
    enable_top10fh_features: bool = False
    top10fh_feature_set: str = "full"
    enable_top_inst_features: bool = False
    # 筛选/裁剪
    feature_stability_filter: bool = False
    factor_prune: bool = False
    factor_exclude_file: Optional[str] = None
    max_feature_missing_ratio: float = 0.6
    # freshness
    freshness_strategy: str = "state_keep_event_decay"
    event_freshness_half_life_days: float = 45.0
    # 样本权重
    rank_weight_enabled: bool = True
    rank_weight_topk: int = 30
    rank_weight: float = 5.0
    rank_weight_topk_weight_mode: str = "linear_decay"
    time_decay_half_life: float = 0
    # 训练超参
    algorithm: str = "xgboost"  # Q6-A：仅 xgboost；其余 ⇒ ValueError
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 100
    reg_alpha: float = 0.05
    reg_lambda: float = 1.0
    gamma: float = 0.1
    random_state: int = 42
    early_stopping_rounds: int = 200
    early_stopping_metric: str = "rank_ic"
    min_best_iteration: int = 0
    objective: str = "mse"
    # 集成
    ensemble_seeds: Optional[Sequence[int]] = None  # 空/None = 单模型(random_state)
    ensemble_offsets: int = 0  # 多偏移窗口（月）；0 = 单窗口
    ensemble_seed_keep_top_ratio: float = 0.3
    ensemble_seed_keep_min_models: int = 3
    # 域标识（解析职责在调用方/薄入口；train 不 import universe）
    stock_domain: str = "main"


def _build_feature_flag_train_params(params: TrainFoldParams) -> Dict[str, Any]:
    """构造需随模型注册保存的可选特征开关（含列集等字符串型签名维度）。"""
    return {
        "stock_domain": str(params.stock_domain or "main"),
        "enable_consensus_features": bool(params.enable_consensus_features),
        "enable_cashflow_quality_features": bool(params.enable_cashflow_quality_features),
        "enable_consensus_revision_features": bool(params.enable_consensus_revision_features),
        "enable_dividend_policy_features": bool(params.enable_dividend_policy_features),
        "enable_holdertrade_features": bool(params.enable_holdertrade_features),
        "holdertrade_feature_set": str(params.holdertrade_feature_set),
        "enable_repurchase_features": bool(params.enable_repurchase_features),
        "repurchase_feature_set": str(params.repurchase_feature_set),
        "enable_top10fh_features": bool(params.enable_top10fh_features),
        "top10fh_feature_set": str(params.top10fh_feature_set),
        "enable_top_inst_features": bool(params.enable_top_inst_features),
        "enable_availability_markers": bool(params.enable_availability_markers),
    }


def _build_main_board_codes(stock_basic: pd.DataFrame, markets: tuple = ("主板",)) -> set:
    """从 stock_basic 构建域股票代码集合（默认主板 = 生产现状）。

    Args:
        stock_basic: 股票基础信息
        markets: 市场白名单（来自 ``universe/domains.py`` 域定义；C 臂域扩展）。
            默认 ``("主板",)`` 与历史行为逐位一致。
    """
    if stock_basic is None or len(stock_basic) == 0:
        raise ValueError("stock_basic 为空，无法构建主板股票池")
    if "ts_code" not in stock_basic.columns or "market" not in stock_basic.columns:
        raise ValueError("stock_basic 缺少 ts_code/market 列，无法做主板过滤")

    board_df = stock_basic[stock_basic["market"].isin(markets)]
    board_codes = set(board_df["ts_code"].astype(str).tolist())
    if not board_codes:
        raise ValueError(f"stock_basic 中 market∈{list(markets)} 的股票为空，无法构建域股票池")
    return board_codes


def _filter_to_main_board(df: pd.DataFrame, main_board_codes: set, stage: str) -> pd.DataFrame:
    """按域股票池过滤样本，确保训练/评估与交易口径一致（池由 ``_build_main_board_codes`` 构建）。"""
    if df is None or len(df) == 0:
        return df
    if "ts_code" not in df.columns:
        raise ValueError(f"{stage} 数据缺少 ts_code 列，无法做主板过滤")

    before = len(df)
    # 布尔掩码索引 df[mask] 本身已按行 take 复制出独立数据，无需再 deep copy。
    # 大训练集（如 293 列 × 400 万行）上 .copy() 会触发 pandas BlockManager
    # _consolidate_inplace 合并全部列块，额外申请一份全量连续内存（≈8.8 GiB），
    # 极易导致 OOM。
    filtered = df[df["ts_code"].astype(str).isin(main_board_codes)]
    after = len(filtered)
    logger.info(f"{stage} 主板过滤: {before} -> {after}（移除 {before - after}）")
    if after == 0:
        raise ValueError(f"{stage} 主板过滤后样本为空，请检查数据与股票池配置")
    return filtered


def _align_to_trade_date(date_str: str, trade_dates: List[str], forward: bool = True) -> str:
    """将日期对齐到最近的交易日。"""
    if forward:
        for td in trade_dates:
            if td >= date_str:
                return td
        return trade_dates[-1]
    for td in reversed(trade_dates):
        if td <= date_str:
            return td
    return trade_dates[0]


def compute_offset_windows(
    train_start: str,
    train_end: str,
    offset_months: int,
    trade_cal: pd.DataFrame,
) -> List[Tuple[str, str]]:
    """计算多偏移训练窗口。"""
    all_dates = trade_cal[trade_cal["is_open"] == 1]["cal_date"].sort_values().tolist()
    windows = [(train_start, train_end)]

    for sign in [-1, 1]:
        start_dt = datetime.strptime(train_start, "%Y%m%d") + relativedelta(
            months=sign * offset_months
        )
        end_dt = datetime.strptime(train_end, "%Y%m%d") + relativedelta(months=sign * offset_months)
        start_aligned = _align_to_trade_date(start_dt.strftime("%Y%m%d"), all_dates, forward=True)
        end_aligned = _align_to_trade_date(end_dt.strftime("%Y%m%d"), all_dates, forward=False)
        windows.append((start_aligned, end_aligned))

    return windows


def resolve_explicit_split_dates(
    outer_train_start: str,
    outer_val_end: str,
    val_ratio: float,
    trade_dates: Sequence[str],
    label_delta: int,
) -> Tuple[str, str, str, str]:
    """比例→显式四日期换算（F10 D1：换算归编排器，train 侧不再二次比例切分）。

    复算 ``split_train_val_by_date`` 同数学的日历日域表达：
    ``n_val = max(1, ceil(n × val_ratio))``、``n_train = n - n_val``、
    训练尾扣 ``label_delta`` 隔离带 ⇒ ``train_end = D[n_train-delta-1]``、
    ``val_start = D[n_train+delta]``（D = 外层窗交易日升序序列，长度 n）。

    现役语义前提 = 过滤后日域与外层窗日历日域一致（T0.5b 首折实证成立）；
    偏差场景由 §7.6 逐折对账门（R2-06）检出，禁止静默通过。
    """
    days = [d for d in trade_dates if outer_train_start <= d <= outer_val_end]
    n = len(days)
    if n == 0:
        raise ValueError(
            f"显式切分换算失败: 外层窗 [{outer_train_start}, {outer_val_end}] 内无交易日"
        )
    n_val = max(1, math.ceil(n * val_ratio))
    n_train = n - n_val
    if n_train <= 0:
        raise ValueError(f"显式切分换算失败: val_ratio={val_ratio} 下训练日期数为零（n={n}）")
    if n_train + label_delta >= n:
        raise ValueError(
            f"显式切分换算失败: 数据不足以保留 {label_delta} 天隔离带"
            f"（n={n}, n_train={n_train}）"
        )
    train_end_idx = max(n_train - label_delta - 1, 0)
    val_start_idx = n_train + label_delta
    return (days[0], days[train_end_idx], days[val_start_idx], days[-1])


def _prepare_labels(
    df_train: pd.DataFrame, params: TrainFoldParams
) -> Tuple[pd.DataFrame, str, Optional[Callable[[pd.DataFrame], pd.DataFrame]]]:
    """标签混合/分类二值化与标签变换闭包（P3-T3 提取自 _train_model_on_window，逐字）。

    返回 (df_train, actual_label_column, label_transform_fn)——**帧必须回传**：
    ``generate_classification_labels`` 返回新帧（T3 评审 codex-R04 修复：拆分曾丢帧，
    属回归；旧实现接住了新帧）。``add_blended_return_label`` 为原地写（blend>0）。
    """
    actual_label_column = add_blended_return_label(
        df_train,
        params.label_column,
        params.neutral_label_blend_weight,
    )
    if params.task == "classification":
        if params.pos_quantile is None and params.pos_topk is None:
            raise ValueError("分类任务必须指定 --pos-quantile 或 --pos-topk")
        df_train = generate_classification_labels(
            df_train,
            label_column=params.label_column,
            pos_quantile=params.pos_quantile,
            pos_topk=params.pos_topk,
        )
        binary_label_col = f"{actual_label_column}_binary"
        actual_label_column = binary_label_col

    label_transform_fn = None
    if params.task == "regression" and params.label_transform == "cs_zscore":
        label_transform_fn = lambda d: transform_labels_cs_zscore(
            d, label_column=actual_label_column, winsorize_p=params.winsorize_p
        )
    return df_train, actual_label_column, label_transform_fn


def _build_sample_weights(
    df_train_split: pd.DataFrame,
    actual_label_column: str,
    params: TrainFoldParams,
) -> Optional[pd.Series]:
    """rank 权重 × 可选时间衰减权重（P3-T3 提取自 _train_model_on_window，逐字）。"""
    rank_sample_weight = None
    if params.rank_weight_enabled:
        rank_sample_weight = build_rank_sample_weights(
            df_train=df_train_split,
            label_column=actual_label_column,
            topk=params.rank_weight_topk,
            top_weight=params.rank_weight,
            topk_weight_mode=params.rank_weight_topk_weight_mode,
        )
    if params.time_decay_half_life > 0:
        td_weights = build_time_decay_weights(
            df_train=df_train_split,
            half_life_years=params.time_decay_half_life,
        )
        if rank_sample_weight is not None:
            rank_sample_weight = rank_sample_weight * td_weights
        else:
            rank_sample_weight = td_weights
    return rank_sample_weight


def _train_model_on_window(
    frame: TrainingFrame,
    params: TrainFoldParams,
    main_board_codes: set,
    runtime_derivers: Optional[RuntimeDerivers] = None,
    random_state_override: Optional[int] = None,
    feature_columns_override: Optional[List[str]] = None,
    explicit_train_end: Optional[str] = None,
    explicit_val_start: Optional[str] = None,
) -> Dict:
    """在指定训练窗口上训练单个模型（v2：训练帧由调用方经适配层 + 帧缓存供给）。

    Args:
        frame: 外层训练窗 ``TrainingFrame``（df 平铺镜像 cs_train 形态；
            四族数据查询表由 ``frame.runtime_lookups`` 携带，cs_train 验证源模式）；
        feature_columns_override: 若提供，训练入口特征质量门禁后强制将特征列
            对齐到该列表（多窗口集成统一子模型特征 schema）；
        explicit_train_end / explicit_val_start: 显式日期缝（F10 D1；None =
            旧比例路径，``val_ratio`` 生效）；两者必须同给（prepare 内校验）。
    """
    df_train = _filter_to_main_board(frame.df, main_board_codes, "训练窗口")
    train_days_count = frame.trade_days
    total_train_samples = len(df_train)

    df_train, actual_label_column, label_transform_fn = _prepare_labels(df_train, params)
    runtime_lookups = frame.runtime_lookups or {}

    (
        X_train,
        y_train,
        X_val,
        y_val,
        feature_columns,
        df_train_split,
        df_val_split,
        data_stats,
        df_val_split_original,
    ) = prepare_training_data(
        df_train,
        actual_label_column,
        val_ratio=params.val_ratio,
        label_transform_fn=label_transform_fn,
        enable_fundamental_features=params.enable_fundamental_features,
        enable_alt_features=params.enable_alt_features,
        enable_margin_features=params.enable_margin_features,
        enable_cyq_features=params.enable_cyq_features,
        enable_fund_features=params.enable_fund_features,
        enable_express_features=params.enable_express_features,
        enable_enhanced_features=params.enable_enhanced_features,
        enable_north_features=params.enable_north_features,
        enable_lhb_features=params.enable_lhb_features,
        enable_consensus_features=params.enable_consensus_features,
        enable_cashflow_quality_features=params.enable_cashflow_quality_features,
        enable_consensus_revision_features=params.enable_consensus_revision_features,
        enable_dividend_policy_features=params.enable_dividend_policy_features,
        enable_holdertrade_features=params.enable_holdertrade_features,
        holdertrade_lookup=runtime_lookups.get("holdertrade_lookup"),
        holdertrade_feature_set=params.holdertrade_feature_set,
        enable_repurchase_features=params.enable_repurchase_features,
        repurchase_lookup=runtime_lookups.get("repurchase_lookup"),
        repurchase_feature_set=params.repurchase_feature_set,
        enable_top10fh_features=params.enable_top10fh_features,
        top10fh_panel=runtime_lookups.get("top10fh_panel"),
        top10fh_feature_set=params.top10fh_feature_set,
        enable_top_inst_features=params.enable_top_inst_features,
        top_inst_lookup=runtime_lookups.get("top_inst_lookup"),
        enable_availability_markers=params.enable_availability_markers,
        feature_stability_filter=params.feature_stability_filter,
        factor_prune=params.factor_prune,
        factor_exclude_file=params.factor_exclude_file,
        max_feature_missing_ratio=params.max_feature_missing_ratio,
        feature_columns_override=feature_columns_override,
        freshness_strategy=params.freshness_strategy,
        event_freshness_half_life_days=params.event_freshness_half_life_days,
        explicit_train_end=explicit_train_end,
        explicit_val_start=explicit_val_start,
        runtime_derivers=runtime_derivers,
    )

    del df_train
    gc.collect()

    rank_sample_weight = _build_sample_weights(df_train_split, actual_label_column, params)

    skip_label_winsorize = params.task == "regression" and params.label_transform == "cs_zscore"
    if params.algorithm != "xgboost":
        raise ValueError(
            f"v2 训练引擎仅支持 xgboost（Q6-A 裁决 LightGBM 不迁移、登记退役备用臂）: "
            f"{params.algorithm!r}"
        )

    extra_kwargs: Dict[str, Any] = {}
    objective_type = params.objective
    extra_kwargs["objective_type"] = objective_type
    if objective_type == "lambdarank":
        extra_kwargs["df_train_for_group"] = df_train_split
    # 验证集 df 无条件传入：lambdarank 分组与 rank_ic_daily 早停指标共用
    # （df_val_split 为 ES 段，行序与 X_val 一致；行数不一致时训练函数会明确报错）
    extra_kwargs["df_val_for_group"] = df_val_split
    extra_kwargs["min_best_iteration"] = params.min_best_iteration

    es_rounds = params.early_stopping_rounds if params.early_stopping_rounds else None

    model, train_params, train_metrics, val_metrics = train_xgboost_model(
        X_train,
        y_train,
        X_val,
        y_val,
        task=params.task,
        skip_label_winsorize=skip_label_winsorize,
        scale_pos_weight=params.scale_pos_weight,
        sample_weight=rank_sample_weight,
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        random_state=(
            params.random_state if random_state_override is None else random_state_override
        ),
        min_child_weight=params.min_child_weight,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        gamma=params.gamma,
        early_stopping_rounds=es_rounds,
        early_stopping_metric=params.early_stopping_metric,
        **extra_kwargs,
    )

    return {
        "model": model,
        "feature_columns": feature_columns,
        "train_params": train_params,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "df_train_split": df_train_split,
        "df_val_split": df_val_split,
        "df_val_split_original": df_val_split_original,
        "data_stats": data_stats,
        "train_days_count": train_days_count,
        "total_train_samples": total_train_samples,
        "X_train_len": len(X_train),
        "X_val_len": len(X_val),
        "label_column": actual_label_column,
    }


def _resolve_ensemble_seeds(params: TrainFoldParams) -> List[int]:
    """解析 ensemble_seeds，返回去重保序的种子列表。

    承载 ``Sequence[int]``（薄入口已完成字符串解析，§13.3.1）；逗号字符串
    输入兼容保留旧解析路径。
    """
    raw = params.ensemble_seeds
    if not raw:
        return [params.random_state]
    if isinstance(raw, str):
        seeds: List[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            seed = int(token)
            if seed not in seeds:
                seeds.append(seed)
        return seeds if seeds else [params.random_state]
    seeds = []
    for seed in raw:
        seed = int(seed)
        if seed not in seeds:
            seeds.append(seed)
    return seeds if seeds else [params.random_state]


def _select_ensemble_validation_result(train_results: List[Dict]) -> Optional[Dict]:
    """选择所有保留子模型共同未见的 calibration 面板。"""
    dated_results = []
    for train_result in train_results:
        data_stats = train_result.get("data_stats", {})
        train_end = str(data_stats.get("train_end_date", "")).replace("-", "")[:8]
        val_es_end = str(data_stats.get("val_es_end_date", "")).replace("-", "")[:8]
        validation_frame = train_result.get("df_val_split_original")
        if not train_end.isdigit() or validation_frame is None or len(validation_frame) == 0:
            continue
        seen_end = max(train_end, val_es_end) if val_es_end.isdigit() else train_end
        validation_dates = (
            validation_frame["trade_date"].astype(str).str.replace("-", "", regex=False)
        )
        val_start = validation_dates.min()[:8]
        if val_start.isdigit():
            dated_results.append((seen_end, val_start, train_result))

    if not dated_results:
        return train_results[0] if train_results else None

    max_seen_end = max(item[0] for item in dated_results)
    safe_results = [item for item in dated_results if item[1] > max_seen_end]
    if safe_results:
        _, val_start, selected_result = min(safe_results, key=lambda item: item[1])
        logger.info(
            f"集成验证统一口径: 所有子模型训练/早停截止<={max_seen_end}, "
            f"共同未见 calibration 起始={val_start}"
        )
        return selected_result

    fallback_result = copy.copy(max(dated_results, key=lambda item: item[0])[2])
    fallback_frame = fallback_result["df_val_split_original"]
    fallback_result["df_val_split_original"] = fallback_frame.iloc[:0].copy()
    logger.warning(
        f"集成验证已禁用: 无法找到晚于所有子模型训练/早停截止日 {max_seen_end} "
        "的 calibration 面板"
    )
    return fallback_result


def _score_sub_models(
    sub_model_records: List[Dict],
    params: TrainFoldParams,
    topk_values: Optional[List[int]],
    keep_top_ratio: float,
    keep_min_models: int,
    prefix: str,
) -> Tuple[List, Dict, List[Dict], Dict[str, Any]]:
    """多种子子模型筛选段（P3-T3 提取自 _build_ensemble_sub_models，逐字）。

    返回 (sub_models, base_result, retained_train_results, ensemble_meta_partial)。
    """
    eval_topk_values = topk_values if topk_values else [30]
    scored_records: List[Dict] = []
    for record in sub_model_records:
        train_result = record["train_result"]
        seed_metrics = _evaluate_train_result_val_daily(
            train_result,
            train_result["label_column"],
            params.task,
            eval_topk_values,
            emit_logs=False,
        )
        score = _seed_model_sort_score(seed_metrics)
        scored_records.append(
            {
                "train_result": train_result,
                "seed_metrics": seed_metrics,
                "score": score,
            }
        )

    scored_records.sort(key=lambda item: item["score"], reverse=True)
    raw_keep_count = int(np.ceil(len(scored_records) * keep_top_ratio))
    keep_count = min(len(scored_records), max(keep_min_models, raw_keep_count))
    kept_records = scored_records[:keep_count]
    sub_models = [item["train_result"]["model"] for item in kept_records]
    base_result = kept_records[0]["train_result"]

    logger.warning(
        f"{prefix}多种子筛选: total={len(scored_records)}, keep={keep_count}, "
        f"ratio={keep_top_ratio:.0%}, min_keep={keep_min_models}"
    )
    for rank_idx, item in enumerate(kept_records, start=1):
        train_result = item["train_result"]
        metrics = item["seed_metrics"]
        seed_used = train_result["train_params"].get("random_state")
        top30_med = _safe_float(metrics.get("diagnostic_Top30_逐日均值_50分位"))
        val_ir = _safe_float(metrics.get("daily_rankic_ir"))
        logger.warning(
            f"  保留子模型#{rank_idx}: seed={seed_used}, "
            f"top30_median={'nan' if top30_med is None else f'{top30_med:.6f}'}, "
            f"val_ir={'nan' if val_ir is None else f'{val_ir:.4f}'}"
        )

    sub_model_best_iters: List[Tuple] = []
    for item in kept_records:
        tr = item["train_result"]
        seed_val = tr["train_params"].get("random_state")
        best_iter_val = tr["train_params"].get("best_iteration")
        sub_model_best_iters.append((seed_val, best_iter_val))
    retained_train_results = [item["train_result"] for item in kept_records]
    return (
        sub_models,
        base_result,
        retained_train_results,
        {"sub_model_best_iterations": sub_model_best_iters},
    )


def _build_ensemble_sub_models(
    windows: List[tuple],
    frame_provider: Callable[[str, str], TrainingFrame],
    params: TrainFoldParams,
    main_board_codes: set,
    seeds: List[int],
    runtime_derivers: Optional[RuntimeDerivers] = None,
    is_deploy: bool = False,
    topk_values: Optional[List[int]] = None,
    explicit_dates: Optional[Tuple[str, str]] = None,
) -> tuple:
    """对（窗口 × 种子）笛卡尔训练子模型。

    v2 改动：数据经 ``frame_provider``（适配层 + 帧缓存；同折同窗口多种子共享
    一帧）供给；``explicit_dates`` = （显式 train_end, 显式 val_start）逐窗口
    共用（多偏移窗口的显式缝按各窗口换算由调用方逐窗提供时透传——
    本期 offset=0 单窗口，显式缝单组透传）。
    """
    sub_models: List = []
    sub_model_records: List[Dict] = []
    base_result: Optional[Dict] = None
    ensemble_meta: Dict[str, Any] = {}
    keep_top_ratio = float(params.ensemble_seed_keep_top_ratio)
    keep_top_ratio = min(1.0, max(0.01, keep_top_ratio))
    keep_min_models = int(params.ensemble_seed_keep_min_models)
    keep_min_models = max(1, keep_min_models)
    total = len(windows) * len(seeds)
    idx = 0
    prefix = "部署" if is_deploy else ""
    explicit_train_end = explicit_dates[0] if explicit_dates else None
    explicit_val_start = explicit_dates[1] if explicit_dates else None
    for win_idx, (win_start, win_end) in enumerate(windows):
        win_label = ["基础", "前移", "后移"][win_idx] if win_idx < 3 else f"偏移{win_idx}"
        frame = frame_provider(win_start, win_end)
        for seed in seeds:
            idx += 1
            seed_note = f" seed={seed}" if len(seeds) > 1 else ""
            logger.info(
                f"{'='*60}\n"
                f"  {prefix}子模型 {idx}/{total}（{win_label}{seed_note}）: "
                f"{win_start} ~ {win_end}\n"
                f"{'='*60}"
            )
            selected_tr = _train_model_on_window(
                frame,
                params,
                main_board_codes,
                runtime_derivers=runtime_derivers,
                random_state_override=seed,
                # 以首个（基础窗口）子模型的特征列为准，强制后续子模型对齐，
                # 避免不同窗口高缺失门禁产生不一致的特征 schema 导致集成预测失败
                feature_columns_override=(
                    base_result["feature_columns"] if base_result is not None else None
                ),
                explicit_train_end=explicit_train_end,
                explicit_val_start=explicit_val_start,
            )

            sub_models.append(selected_tr["model"])
            sub_model_records.append({"train_result": selected_tr})
            if base_result is None:
                base_result = selected_tr
            elif set(selected_tr["feature_columns"]) != set(base_result["feature_columns"]):
                logger.warning(
                    f"  子模型 {idx} 特征列数量({len(selected_tr['feature_columns'])})"
                    f"与基础模型({len(base_result['feature_columns'])})不一致"
                )

    if len(seeds) > 1 and len(sub_model_records) > 0:
        sub_models, base_result, retained_train_results, meta_part = _score_sub_models(
            sub_model_records,
            params,
            topk_values,
            keep_top_ratio,
            keep_min_models,
            prefix,
        )
        ensemble_meta.update(meta_part)
    else:
        sub_model_best_iters: List[Tuple] = []
        for record in sub_model_records:
            tr = record["train_result"]
            seed_val = tr["train_params"].get("random_state")
            best_iter_val = tr["train_params"].get("best_iteration")
            sub_model_best_iters.append((seed_val, best_iter_val))
        ensemble_meta["sub_model_best_iterations"] = sub_model_best_iters
        retained_train_results = [record["train_result"] for record in sub_model_records]

    if len(retained_train_results) < len(sub_model_records):
        validation_result = copy.copy(base_result)
        validation_frame = validation_result["df_val_split_original"]
        validation_result["df_val_split_original"] = validation_frame.iloc[:0].copy()
        logger.warning("集成验证已禁用: calibration 已参与子模型筛选，不能复用为独立验证集")
    else:
        validation_result = _select_ensemble_validation_result(retained_train_results)
    ensemble_meta["_ensemble_validation_result"] = validation_result
    return sub_models, base_result, ensemble_meta


def _evaluate_train_result_val_daily(
    train_result: Dict,
    original_return_col: str,
    task: str,
    topk_values: List[int],
    emit_logs: bool = True,
) -> Dict:
    df_val = train_result.get("df_val_split_original")
    if df_val is None or len(df_val) == 0:
        return {}
    return evaluate_validation_daily(
        model=train_result["model"],
        df_val=df_val,
        feature_columns=train_result["feature_columns"],
        original_return_col=original_return_col,
        task=task,
        topk_values=topk_values,
        emit_logs=emit_logs,
    )


def _seed_model_sort_score(metrics: Dict) -> Tuple[float, float]:
    """多种子子模型排序评分：Top30 逐日收益中位数优先，其次逐日 RankIC IR。"""
    top30_median = _safe_float(metrics.get("diagnostic_Top30_逐日均值_50分位"))
    rankic_ir = _safe_float(metrics.get("daily_rankic_ir"))
    return (
        -np.inf if top30_median is None else top30_median,
        -np.inf if rankic_ir is None else rankic_ir,
    )
