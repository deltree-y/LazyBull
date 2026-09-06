#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward split 训练执行模块。"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml import ModelRegistry
from src.lazybull.ml.ensemble import EnsembleModel
from src.lazybull.ml.run_logger import (
    create_training_run_record_from_training_session,
    write_training_run_to_csv,
)
from src.lazybull.ml.train_core import (
    DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
    add_blended_return_label,
    apply_serving_event_decay,
    attach_cashflow_quality_train_params,
    attach_cons_revision_schema_version,
    attach_dividend_train_params,
    evaluate_validation_daily,
    load_features_data,
)

from .reporting import build_daily_topk_detail_df
from .training_core import (
    MIN_MODELS,
    SEED_ENSEMBLE_KEEP_TOP_RATIO,
    _build_ensemble_sub_models,
    _build_feature_flag_train_params,
    _evaluate_train_result_val_daily,
    _filter_to_main_board,
    _resolve_ensemble_seeds,
    _seed_model_sort_score,
    _select_ensemble_validation_result,
    _train_model_on_window,
    compute_offset_windows,
)
from .training_reporting import (
    _fmt_metric,
    _fmt_pct,
    _print_oos_focus_panel,
    _print_pre_backtest_model_summary,
    _safe_float,
)
from .utils import WalkForwardSplit


def _build_split_training_candidate(
    split: WalkForwardSplit,
    storage: Storage,
    loader: DataLoader,
    args,
    main_board_codes: set,
    topk_values: List[int],
    trade_cal: Optional[pd.DataFrame] = None,
    candidate_name: str = "base",
) -> Dict:
    ensemble_offsets = getattr(args, "ensemble_offsets", 0)
    ensemble_seeds = _resolve_ensemble_seeds(args)
    use_ensemble = (ensemble_offsets > 0 and trade_cal is not None) or len(ensemble_seeds) > 1

    if use_ensemble:
        if ensemble_offsets > 0 and trade_cal is not None:
            windows = compute_offset_windows(
                split.train_start, split.train_end, ensemble_offsets, trade_cal
            )
        else:
            windows = [(split.train_start, split.train_end)]
        logger.info(
            f"{candidate_name} 集成训练: {len(windows)}个窗口 × {len(ensemble_seeds)}个种子 "
            f"= {len(windows) * len(ensemble_seeds)}个子模型"
            f"（偏移±{ensemble_offsets}个月, seeds={ensemble_seeds}）"
        )

        sub_models, base_result, ensemble_meta = _build_ensemble_sub_models(
            windows,
            storage,
            loader,
            args,
            main_board_codes,
            ensemble_seeds,
            topk_values=topk_values,
        )

        model = EnsembleModel(sub_models)
        validation_result = ensemble_meta.pop("_ensemble_validation_result", None) or base_result
        feature_columns = base_result["feature_columns"]
        train_params = base_result["train_params"]
        train_metrics = base_result["train_metrics"]
        val_metrics = base_result["val_metrics"]
        df_val_split_original = validation_result["df_val_split_original"]
        validation_feature_columns = validation_result["feature_columns"]
        validation_label_column = validation_result["label_column"]
        data_stats = base_result["data_stats"]
        train_days_count = base_result["train_days_count"]
        total_train_samples = base_result["total_train_samples"]
        X_train_len = base_result["X_train_len"]
        X_val_len = base_result["X_val_len"]

        logger.info(f"{candidate_name} 集成模型创建完成: {model}")
    else:
        tr = _train_model_on_window(
            split.train_start,
            split.train_end,
            storage,
            loader,
            args,
            main_board_codes,
            random_state_override=ensemble_seeds[0],
        )
        model = tr["model"]
        feature_columns = tr["feature_columns"]
        train_params = tr["train_params"]
        train_metrics = tr["train_metrics"]
        val_metrics = tr["val_metrics"]
        df_val_split_original = tr["df_val_split_original"]
        validation_feature_columns = feature_columns
        validation_label_column = tr["label_column"]
        data_stats = tr["data_stats"]
        train_days_count = tr["train_days_count"]
        total_train_samples = tr["total_train_samples"]
        X_train_len = tr["X_train_len"]
        X_val_len = tr["X_val_len"]
        ensemble_meta = {
            "sub_model_best_iterations": [(ensemble_seeds[0], train_params.get("best_iteration"))]
        }

    val_daily_metrics = {}
    if len(df_val_split_original) > 0:
        val_daily_metrics = evaluate_validation_daily(
            model=model,
            df_val=df_val_split_original,
            feature_columns=validation_feature_columns,
            original_return_col=validation_label_column,
            task=args.task,
            topk_values=topk_values,
        )

    return {
        "candidate_name": candidate_name,
        "model": model,
        "feature_columns": feature_columns,
        "train_params": train_params,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_daily_metrics": val_daily_metrics,
        "df_val_split_original": df_val_split_original,
        "data_stats": data_stats,
        "train_days_count": train_days_count,
        "total_train_samples": total_train_samples,
        "X_train_len": X_train_len,
        "X_val_len": X_val_len,
        "label_column": validation_label_column,
        "ensemble_meta": ensemble_meta,
    }


def execute_split_training(
    split: WalkForwardSplit,
    wf_run_id: str,
    storage: Storage,
    loader: DataLoader,
    registry: ModelRegistry,
    args,
    main_board_codes: set,
    topk_values: List[int],
    trade_cal: Optional[pd.DataFrame] = None,
) -> Dict:
    """执行单个 split 的训练。"""
    logger.info("=" * 80)
    logger.info(f"开始训练 Split {split.split_index}")
    logger.info(f"  训练区间: {split.train_start} 至 {split.train_end}")
    logger.info(f"  测试区间: {split.test_start} 至 {split.test_end}")
    logger.info("=" * 80)

    selected_candidate = _build_split_training_candidate(
        split,
        storage,
        loader,
        args,
        main_board_codes,
        topk_values,
        trade_cal,
        candidate_name="base",
    )
    ensemble_meta = selected_candidate.get("ensemble_meta", {})

    model = selected_candidate["model"]
    feature_columns = selected_candidate["feature_columns"]
    train_params = selected_candidate["train_params"]
    train_metrics = selected_candidate["train_metrics"]
    val_metrics = selected_candidate["val_metrics"]
    val_daily_metrics = selected_candidate["val_daily_metrics"]
    data_stats = selected_candidate["data_stats"]
    train_days_count = selected_candidate["train_days_count"]
    total_train_samples = selected_candidate["total_train_samples"]
    X_train_len = selected_candidate["X_train_len"]
    X_val_len = selected_candidate["X_val_len"]

    df_test, test_days_count = load_features_data(storage, loader, split.test_start, split.test_end)
    df_test = _filter_to_main_board(df_test, main_board_codes, "测试窗口")
    total_test_samples = len(df_test)

    logger.info("=" * 60)
    logger.info("样本外测试集评估（OOS Evaluation）")
    logger.info("=" * 60)

    df_test_eval = df_test.copy()
    evaluation_label_column = add_blended_return_label(
        df_test_eval,
        args.label_column,
        getattr(args, "neutral_label_blend_weight", 0.0),
    )

    filter_columns = ["is_st", "is_suspended", "is_limit_up"]
    mask = pd.Series(True, index=df_test_eval.index)
    for col in filter_columns:
        if col in df_test_eval.columns:
            mask = mask & (~df_test_eval[col].astype(bool))
    df_test_eval = df_test_eval[mask].copy()

    if evaluation_label_column in df_test_eval.columns:
        df_test_eval = df_test_eval.dropna(subset=[evaluation_label_column])

    logger.info(f"测试集样本数（过滤后）: {len(df_test_eval)}")

    # OOS 评估侧复现训练时的事件型 freshness 衰减（train/serve 一致）
    df_test_eval = apply_serving_event_decay(
        df_test_eval,
        freshness_strategy=getattr(
            args, "freshness_strategy", FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY
        ),
        event_freshness_half_life_days=getattr(
            args, "event_freshness_half_life_days", DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS
        ),
    )

    X_test_features = df_test_eval[feature_columns]

    if args.task == "classification":
        y_test_pred_proba = model.predict_proba(X_test_features)[:, 1]
        df_test_eval["pred_score"] = y_test_pred_proba
    else:
        y_test_pred = model.predict(X_test_features)
        df_test_eval["pred_score"] = y_test_pred

    test_score_column = "pred_score"

    topk_detail_df = build_daily_topk_detail_df(
        df_eval=df_test_eval,
        original_return_col=evaluation_label_column,
        topk_values=(20, 30),
        score_column=test_score_column,
    )

    test_daily_metrics = evaluate_validation_daily(
        model=model,
        df_val=df_test_eval,
        feature_columns=feature_columns,
        original_return_col=evaluation_label_column,
        task=args.task,
        topk_values=topk_values,
        emit_logs=False,
        prediction_col=test_score_column,
    )

    _print_oos_focus_panel(split.split_index, test_daily_metrics)

    if getattr(args, "oos_detail_metrics", False):
        _print_pre_backtest_model_summary(
            split_index=split.split_index,
            ensemble_meta=ensemble_meta,
            val_daily_metrics=val_daily_metrics,
            test_daily_metrics=test_daily_metrics,
        )
    else:
        logger.info(
            f"Split {split.split_index} OOS简报: "
            f"RankIC={_fmt_metric(_safe_float(test_daily_metrics.get('daily_rankic_mean')), '.4f')} | "
            f"Top20_hit={_fmt_pct(_safe_float(test_daily_metrics.get('diagnostic_Top20_命中率_日均收益为正')))} | "
            f"Top20_median={_fmt_metric(_safe_float(test_daily_metrics.get('diagnostic_Top20_逐日均值_50分位')), '.6f')} | "
            f"Top30_hit={_fmt_pct(_safe_float(test_daily_metrics.get('diagnostic_Top30_命中率_日均收益为正')))} | "
            f"Top30_median={_fmt_metric(_safe_float(test_daily_metrics.get('diagnostic_Top30_逐日均值_50分位')), '.6f')}"
        )

    performance_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "validation_daily": val_daily_metrics,
        "test": {},
        "test_daily": test_daily_metrics,
    }

    algorithm = getattr(args, "algorithm", "xgboost")
    full_train_params = train_params.copy()
    attach_cons_revision_schema_version(
        full_train_params,
        getattr(args, "enable_consensus_revision_features", False),
    )
    attach_cashflow_quality_train_params(
        full_train_params,
        getattr(args, "enable_cashflow_quality_features", False),
        feature_columns=feature_columns,
    )
    attach_dividend_train_params(
        full_train_params,
        getattr(args, "enable_dividend_policy_features", False),
    )
    full_train_params.update(
        {
            "algorithm": algorithm,
            "task": args.task,
            "label_transform": args.label_transform if args.task == "regression" else None,
            "winsorize_p": args.winsorize_p if args.label_transform == "cs_zscore" else None,
            "pos_quantile": args.pos_quantile if args.task == "classification" else None,
            "pos_topk": args.pos_topk if args.task == "classification" else None,
            "scale_pos_weight_manual": args.scale_pos_weight is not None,
            **_build_feature_flag_train_params(args),
            "freshness_strategy": getattr(
                args, "freshness_strategy", FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY
            ),
            "event_freshness_half_life_days": getattr(
                args, "event_freshness_half_life_days", DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS
            ),
        }
    )
    if isinstance(model, EnsembleModel):
        full_train_params["ensemble_offsets"] = getattr(args, "ensemble_offsets", 0)
        full_train_params["ensemble_seeds"] = _resolve_ensemble_seeds(args)
        full_train_params["ensemble_seed_keep_top_ratio"] = getattr(
            args, "ensemble_seed_keep_top_ratio", SEED_ENSEMBLE_KEEP_TOP_RATIO
        )
        full_train_params["ensemble_seed_keep_min_models"] = getattr(
            args, "ensemble_seed_keep_min_models", MIN_MODELS
        )
        full_train_params["ensemble_n_models"] = model.n_models

    version = registry.register_model(
        model=model,
        model_type=f"{algorithm}_{args.task}_wf",
        train_start_date=split.train_start,
        train_end_date=split.train_end,
        feature_columns=feature_columns,
        label_column=args.label_column,
        n_samples=X_train_len + X_val_len,
        train_params=full_train_params,
        performance_metrics=performance_metrics,
    )

    logger.info(f"模型已注册: v{version}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ffi = data_stats.get("feature_filter_info")
    feature_filter_summary = {}
    if ffi and not ffi.get("skipped", True):
        feature_filter_summary = {
            "feature_total": ffi["total_features"],
            "feature_stable": ffi["stable_count"],
            "feature_removed": ffi["removed_count"],
        }

    complete_data_stats = {
        "trade_days_count": train_days_count,
        "total_samples": total_train_samples,
        "samples_after_filter": data_stats["samples_after_filter"],
        "train_samples": X_train_len,
        "val_samples": X_val_len,
        "val_start_date": data_stats["val_start_date"],
        "val_end_date": data_stats["val_end_date"],
        "val_ratio": args.val_ratio,
        "val_raw_start_date": data_stats.get("val_raw_start_date", data_stats["val_start_date"]),
        "val_raw_end_date": data_stats.get("val_raw_end_date", data_stats["val_end_date"]),
        "val_raw_n_dates": data_stats.get("val_raw_n_dates", 0),
        "val_raw_samples": data_stats.get("val_raw_samples", 0),
        "val_es_start_date": data_stats.get("val_es_start_date", data_stats["val_start_date"]),
        "val_es_end_date": data_stats.get("val_es_end_date", data_stats["val_end_date"]),
        "val_es_n_dates": data_stats.get("val_es_n_dates", 0),
        "val_es_samples": data_stats.get("val_es_samples", X_val_len),
        "val_calib_start_date": data_stats.get("val_calib_start_date", "N/A"),
        "val_calib_end_date": data_stats.get("val_calib_end_date", "N/A"),
        "val_calib_n_dates": data_stats.get("val_calib_n_dates", 0),
        "val_calib_samples": data_stats.get("val_calib_samples", 0),
        "val_embargo_days": data_stats.get("val_embargo_days", 0),
        "val_embargo_days_applied": data_stats.get("val_embargo_days_applied", 0),
        "val_embargo_n_dates": data_stats.get("val_embargo_n_dates", 0),
        "val_embargo_samples": data_stats.get("val_embargo_samples", 0),
        "val_embargo_start_date": data_stats.get("val_embargo_start_date", "N/A"),
        "val_embargo_end_date": data_stats.get("val_embargo_end_date", "N/A"),
        **feature_filter_summary,
    }

    run_record = create_training_run_record_from_training_session(
        timestamp=timestamp,
        start_date=split.train_start,
        end_date=split.train_end,
        label_column=args.label_column,
        task=args.task,
        model_version=version,
        train_params=full_train_params,
        data_stats=complete_data_stats,
        performance_metrics=performance_metrics,
        wf_run_id=wf_run_id,
        split_index=split.split_index,
        test_start_date=split.test_start,
        test_end_date=split.test_end,
    )

    csv_path = (
        args.run_log_csv if args.run_log_csv else f"{args.data_root}/models/ml_train_runs.csv"
    )
    write_training_run_to_csv(run_record, csv_path)

    logger.info(f"训练运行日志已记录到: {csv_path}")

    return {
        "split_index": split.split_index,
        "train_start": split.train_start,
        "train_end": split.train_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "model_version": version,
        "feature_columns": feature_columns,
        "train_samples": X_train_len,
        "val_samples": X_val_len,
        "val_es_samples": data_stats.get("val_es_samples", X_val_len),
        "val_embargo_samples": data_stats.get("val_embargo_samples", 0),
        "val_embargo_days": data_stats.get("val_embargo_days", 0),
        "test_samples": len(df_test_eval),
        "best_iteration": train_params.get("best_iteration"),
        "best_iteration_floor_triggered": bool(
            train_params.get("best_iteration_floor_triggered", False)
        ),
        "val_rankic_ir": val_daily_metrics.get("daily_rankic_ir"),
        "test_daily_metrics": test_daily_metrics,
        "_topk_detail_df": topk_detail_df,
    }
