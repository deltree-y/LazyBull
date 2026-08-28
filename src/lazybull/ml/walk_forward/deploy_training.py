#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward deploy 训练执行模块。"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
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
    attach_cons_revision_schema_version,
    attach_cashflow_quality_train_params,
    evaluate_validation_daily,
)

from .training_core import (
    MIN_MODELS,
    SEED_ENSEMBLE_KEEP_TOP_RATIO,
    _build_ensemble_sub_models,
    _build_feature_flag_train_params,
    _resolve_ensemble_seeds,
    _train_model_on_window,
    compute_offset_windows,
)
from .utils import resolve_deploy_train_window


def execute_deploy_training(
    deploy_train_end: str,
    wf_run_id: str,
    storage: Storage,
    loader: DataLoader,
    registry: ModelRegistry,
    args,
    main_board_codes: set,
    topk_values: List[int],
    trade_cal: pd.DataFrame,
) -> Optional[Dict]:
    """在 walk-forward 评估完成后，用最新数据训练部署模型。"""
    train_start_dt = datetime.strptime(deploy_train_end, "%Y%m%d") - relativedelta(
        years=args.train_window_years
    )
    train_start_str = train_start_dt.strftime("%Y%m%d")

    train_start, train_end = resolve_deploy_train_window(
        trade_cal=trade_cal,
        deploy_train_end=deploy_train_end,
        train_window_years=args.train_window_years,
    )

    if train_start is None:
        logger.error(f"无法找到有效的部署模型 train_start（目标: {train_start_str}）")
        return None

    if train_end is None:
        logger.error(f"无法找到有效的部署模型 train_end（目标: {deploy_train_end}）")
        return None

    logger.info("=" * 80)
    logger.info("部署模型训练（Deploy Training）")
    logger.info(f"  训练区间: {train_start} 至 {train_end}")
    logger.info("  （无测试区间，用于部署）")
    logger.info("=" * 80)

    ensemble_offsets = getattr(args, "ensemble_offsets", 0)
    ensemble_seeds = _resolve_ensemble_seeds(args)
    use_ensemble = ensemble_offsets > 0 or len(ensemble_seeds) > 1

    if use_ensemble:
        if ensemble_offsets > 0:
            windows = compute_offset_windows(train_start, train_end, ensemble_offsets, trade_cal)
        else:
            windows = [(train_start, train_end)]
        logger.info(
            f"部署模型集成训练: {len(windows)}个窗口 × {len(ensemble_seeds)}个种子 "
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
            is_deploy=True,
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

        logger.info(f"部署集成模型创建完成: {model}")
    else:
        tr = _train_model_on_window(
            train_start,
            train_end,
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

    performance_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "validation_daily": val_daily_metrics,
        "test": {},
        "test_daily": {},
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
    full_train_params.update(
        {
            "algorithm": algorithm,
            "task": args.task,
            "label_transform": args.label_transform if args.task == "regression" else None,
            "winsorize_p": args.winsorize_p if args.label_transform == "cs_zscore" else None,
            "pos_quantile": args.pos_quantile if args.task == "classification" else None,
            "pos_topk": args.pos_topk if args.task == "classification" else None,
            "scale_pos_weight_manual": args.scale_pos_weight is not None,
            "is_deploy": True,
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
        full_train_params["ensemble_offsets"] = ensemble_offsets
        full_train_params["ensemble_seeds"] = ensemble_seeds
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
        train_start_date=train_start,
        train_end_date=train_end,
        feature_columns=feature_columns,
        label_column=args.label_column,
        n_samples=X_train_len + X_val_len,
        train_params=full_train_params,
        performance_metrics=performance_metrics,
    )

    logger.info(f"部署模型已注册: v{version}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ffi_deploy = data_stats.get("feature_filter_info")
    ff_summary_deploy = {}
    if ffi_deploy and not ffi_deploy.get("skipped", True):
        ff_summary_deploy = {
            "feature_total": ffi_deploy["total_features"],
            "feature_stable": ffi_deploy["stable_count"],
            "feature_removed": ffi_deploy["removed_count"],
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
        **ff_summary_deploy,
    }

    run_record = create_training_run_record_from_training_session(
        timestamp=timestamp,
        start_date=train_start,
        end_date=train_end,
        label_column=args.label_column,
        task=args.task,
        model_version=version,
        train_params=full_train_params,
        data_stats=complete_data_stats,
        performance_metrics=performance_metrics,
        wf_run_id=wf_run_id,
        split_index="deploy",
        test_start_date=None,
        test_end_date=None,
    )

    csv_path = (
        args.run_log_csv if args.run_log_csv else f"{args.data_root}/models/ml_train_runs.csv"
    )
    write_training_run_to_csv(run_record, csv_path)

    logger.info(f"部署模型训练运行日志已记录到: {csv_path}")

    return {
        "split_index": "deploy",
        "train_start": train_start,
        "train_end": train_end,
        "test_start": None,
        "test_end": None,
        "model_version": version,
        "train_samples": X_train_len,
        "val_samples": X_val_len,
        "val_es_samples": data_stats.get("val_es_samples", X_val_len),
        "val_embargo_samples": data_stats.get("val_embargo_samples", 0),
        "val_embargo_days": data_stats.get("val_embargo_days", 0),
        "test_samples": 0,
        "best_iteration": train_params.get("best_iteration"),
        "val_rankic_ir": val_daily_metrics.get("daily_rankic_ir"),
        "test_daily_metrics": {},
    }
