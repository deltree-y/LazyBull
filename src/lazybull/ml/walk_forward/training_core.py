#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward 训练核心函数。"""

import copy
import gc
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml.train_core import (
    add_blended_return_label,
    build_rank_sample_weights,
    build_time_decay_weights,
    evaluate_validation_daily,
    generate_classification_labels,
    load_features_data,
    prepare_training_data,
    train_lightgbm_model,
    train_xgboost_model,
    transform_labels_cs_zscore,
)

from .training_reporting import _safe_float

SEED_ENSEMBLE_KEEP_TOP_RATIO = 0.30
MIN_MODELS = 3
SEED_ENSEMBLE_KEEP_MIN_MODELS = MIN_MODELS


def _build_feature_flag_train_params(args: Any) -> Dict[str, bool]:
    """构造需随模型注册保存的可选特征开关。"""
    return {
        "enable_consensus_features": bool(getattr(args, "enable_consensus_features", False)),
        "enable_cashflow_quality_features": bool(
            getattr(args, "enable_cashflow_quality_features", False)
        ),
        "enable_consensus_revision_features": bool(
            getattr(args, "enable_consensus_revision_features", False)
        ),
        "enable_dividend_policy_features": bool(
            getattr(args, "enable_dividend_policy_features", False)
        ),
    }


def _build_main_board_codes(stock_basic: pd.DataFrame) -> set:
    """从 stock_basic 构建主板股票代码集合。"""
    if stock_basic is None or len(stock_basic) == 0:
        raise ValueError("stock_basic 为空，无法构建主板股票池")
    if "ts_code" not in stock_basic.columns or "market" not in stock_basic.columns:
        raise ValueError("stock_basic 缺少 ts_code/market 列，无法做主板过滤")

    board_df = stock_basic[stock_basic["market"] == "主板"]
    board_codes = set(board_df["ts_code"].astype(str).tolist())
    if not board_codes:
        raise ValueError("stock_basic 中 market=主板 的股票为空，无法做主板过滤")
    return board_codes


def _filter_to_main_board(df: pd.DataFrame, main_board_codes: set, stage: str) -> pd.DataFrame:
    """按主板股票池过滤样本，确保训练/评估与交易口径一致。"""
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


def _train_model_on_window(
    train_start: str,
    train_end: str,
    storage: Storage,
    loader: DataLoader,
    args,
    main_board_codes: set,
    random_state_override: Optional[int] = None,
    feature_columns_override: Optional[List[str]] = None,
) -> Dict:
    """在指定训练窗口上训练单个模型。

    Args:
        feature_columns_override: 若提供，训练入口特征质量门禁后强制将特征列
            对齐到该列表（多窗口集成统一子模型特征 schema）。
    """
    df_train, train_days_count = load_features_data(storage, loader, train_start, train_end)
    df_train = _filter_to_main_board(df_train, main_board_codes, "训练窗口")
    total_train_samples = len(df_train)

    actual_label_column = add_blended_return_label(
        df_train,
        args.label_column,
        getattr(args, "neutral_label_blend_weight", 0.0),
    )
    if args.task == "classification":
        if args.pos_quantile is None and args.pos_topk is None:
            raise ValueError("分类任务必须指定 --pos-quantile 或 --pos-topk")
        df_train = generate_classification_labels(
            df_train,
            label_column=args.label_column,
            pos_quantile=args.pos_quantile,
            pos_topk=args.pos_topk,
        )
        binary_label_col = f"{actual_label_column}_binary"
        actual_label_column = binary_label_col

    label_transform_fn = None
    if args.task == "regression" and args.label_transform == "cs_zscore":
        label_transform_fn = lambda d: transform_labels_cs_zscore(
            d, label_column=actual_label_column, winsorize_p=args.winsorize_p
        )
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
        val_ratio=args.val_ratio,
        label_transform_fn=label_transform_fn,
        enable_fundamental_features=args.enable_fundamental_features,
        enable_alt_features=args.enable_alt_features,
        enable_margin_features=args.enable_margin_features,
        enable_cyq_features=args.enable_cyq_features,
        enable_fund_features=args.enable_fund_features,
        enable_express_features=args.enable_express_features,
        enable_enhanced_features=getattr(args, "enable_enhanced_features", False),
        enable_north_features=getattr(args, "enable_north_features", False),
        enable_lhb_features=getattr(args, "enable_lhb_features", False),
        enable_consensus_features=getattr(args, "enable_consensus_features", False),
        enable_cashflow_quality_features=getattr(args, "enable_cashflow_quality_features", False),
        enable_consensus_revision_features=getattr(
            args, "enable_consensus_revision_features", False
        ),
        enable_dividend_policy_features=getattr(args, "enable_dividend_policy_features", False),
        feature_stability_filter=args.feature_stability_filter,
        factor_prune=args.factor_prune,
        factor_exclude_file=getattr(args, "factor_exclude_file", None),
        max_feature_missing_ratio=getattr(args, "max_feature_missing_ratio", 0.6),
        feature_columns_override=feature_columns_override,
        freshness_strategy=getattr(args, "freshness_strategy", "state_keep_event_decay"),
        event_freshness_half_life_days=getattr(args, "event_freshness_half_life_days", 45.0),
    )

    del df_train
    gc.collect()

    rank_sample_weight = None
    if args.rank_weight_enabled:
        rank_sample_weight = build_rank_sample_weights(
            df_train=df_train_split,
            label_column=actual_label_column,
            topk=args.rank_weight_topk,
            top_weight=args.rank_weight,
            topk_weight_mode=getattr(args, "rank_weight_topk_weight_mode", "linear_decay"),
        )
    if args.time_decay_half_life > 0:
        td_weights = build_time_decay_weights(
            df_train=df_train_split,
            half_life_years=args.time_decay_half_life,
        )
        if rank_sample_weight is not None:
            rank_sample_weight = rank_sample_weight * td_weights
        else:
            rank_sample_weight = td_weights

    skip_label_winsorize = args.task == "regression" and args.label_transform == "cs_zscore"
    algorithm = getattr(args, "algorithm", "xgboost")
    train_fn = train_lightgbm_model if algorithm == "lightgbm" else train_xgboost_model

    extra_kwargs: Dict[str, Any] = {}
    if algorithm == "lightgbm":
        num_leaves_val = getattr(args, "num_leaves", None)
        if num_leaves_val is not None:
            extra_kwargs["num_leaves"] = num_leaves_val
    if algorithm == "xgboost":
        objective_type = getattr(args, "objective", "mse")
        extra_kwargs["objective_type"] = objective_type
        if objective_type == "lambdarank":
            extra_kwargs["df_train_for_group"] = df_train_split
        # 验证集 df 无条件传入：lambdarank 分组与 rank_ic_daily 早停指标共用
        # （df_val_split 为 ES 段，行序与 X_val 一致；行数不一致时训练函数会明确报错）
        extra_kwargs["df_val_for_group"] = df_val_split
        extra_kwargs["min_best_iteration"] = getattr(args, "min_best_iteration", 0)

    es_rounds = args.early_stopping_rounds if args.early_stopping_rounds else None

    model, train_params, train_metrics, val_metrics = train_fn(
        X_train,
        y_train,
        X_val,
        y_val,
        task=args.task,
        skip_label_winsorize=skip_label_winsorize,
        scale_pos_weight=args.scale_pos_weight,
        sample_weight=rank_sample_weight,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=(
            args.random_state if random_state_override is None else random_state_override
        ),
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        early_stopping_rounds=es_rounds,
        early_stopping_metric=args.early_stopping_metric,
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


def _resolve_ensemble_seeds(args) -> List[int]:
    """解析 --ensemble-seeds，返回去重保序的种子列表。"""
    raw = getattr(args, "ensemble_seeds", None)
    if not raw:
        return [args.random_state]
    seeds: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        seed = int(token)
        if seed not in seeds:
            seeds.append(seed)
    return seeds if seeds else [args.random_state]


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


def _build_ensemble_sub_models(
    windows: List[tuple],
    storage: Storage,
    loader: DataLoader,
    args,
    main_board_codes: set,
    seeds: List[int],
    is_deploy: bool = False,
    topk_values: Optional[List[int]] = None,
) -> tuple:
    """对（窗口 × 种子）笛卡尔训练子模型。"""
    sub_models: List = []
    sub_model_records: List[Dict] = []
    base_result: Optional[Dict] = None
    ensemble_meta: Dict[str, Any] = {}
    keep_top_ratio = float(
        getattr(args, "ensemble_seed_keep_top_ratio", SEED_ENSEMBLE_KEEP_TOP_RATIO)
    )
    keep_top_ratio = min(1.0, max(0.01, keep_top_ratio))
    keep_min_models = int(
        getattr(args, "ensemble_seed_keep_min_models", SEED_ENSEMBLE_KEEP_MIN_MODELS)
    )
    keep_min_models = max(1, keep_min_models)
    total = len(windows) * len(seeds)
    idx = 0
    prefix = "部署" if is_deploy else ""
    for win_idx, (win_start, win_end) in enumerate(windows):
        win_label = ["基础", "前移", "后移"][win_idx] if win_idx < 3 else f"偏移{win_idx}"
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
                win_start,
                win_end,
                storage,
                loader,
                args,
                main_board_codes,
                random_state_override=seed,
                # 以首个（基础窗口）子模型的特征列为准，强制后续子模型对齐，
                # 避免不同窗口高缺失门禁产生不一致的特征 schema 导致集成预测失败
                feature_columns_override=(
                    base_result["feature_columns"] if base_result is not None else None
                ),
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
        eval_topk_values = topk_values if topk_values else [30]
        scored_records: List[Dict] = []
        for record in sub_model_records:
            train_result = record["train_result"]
            seed_metrics = _evaluate_train_result_val_daily(
                train_result,
                train_result["label_column"],
                args.task,
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
        ensemble_meta["sub_model_best_iterations"] = sub_model_best_iters
        retained_train_results = [item["train_result"] for item in kept_records]
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
