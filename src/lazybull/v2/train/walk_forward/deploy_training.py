#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward deploy 训练执行模块(v2 侧)。

P3-T3 搬运改写自 src.lazybull.ml.walk_forward.deploy_training(2026-10-22),改动点:

1. 帧供给改 ``frame_provider``(适配层 + 帧缓存;deploy 走**比例路径**——无显式
   日期缝,``resolve_deploy_train_window`` 窗口 + ``val_ratio`` 语义不变);
2. 参数面 args → ``TrainFoldParams``;注册表改指 ``v2.common.model_registry``;
   ``EnsembleModel`` 改指 ``v2.train.ensemble``;台账 ``csv_path`` 显式必传;
3. 拆分(旧 ``execute_deploy_training`` 277 行在册):窗口解析 / 训练段 /
   注册+健康度简报+台账段;
4. Q10-B:P3 回放显式关闭 deploy(编排器 ``deploy_train=False`` 默认 + 验收入口
   登记理由);本模块代码保持可用——生产部署训练留 P4 验证。
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.train.core import (
    RuntimeDerivers,
    attach_cashflow_quality_train_params,
    attach_cons_revision_schema_version,
    attach_dividend_train_params,
    evaluate_validation_daily,
)
from src.lazybull.v2.train.data_adapter import TrainingFrame
from src.lazybull.v2.train.ensemble import EnsembleModel
from src.lazybull.v2.train.run_logger import (
    create_training_run_record_from_training_session,
    write_training_run_to_csv,
)

from .training_core import (
    TrainFoldParams,
    _build_ensemble_sub_models,
    _build_feature_flag_train_params,
    _resolve_ensemble_seeds,
    _train_model_on_window,
)
from .utils import resolve_deploy_train_window


def _resolve_deploy_window(
    deploy_train_end: str,
    train_window_years: int,
    trade_cal: pd.DataFrame,
) -> tuple:
    """部署窗口解析与日志(P3-T3 提取自 execute_deploy_training,逐字)。

    ``train_window_years`` 由编排器从 ``ModelConfig.train_window_months`` 换算注入
    (§13.3.1:train_window_years ×12 → 协议 train_window_months)。
    """
    train_start_dt = datetime.strptime(deploy_train_end, "%Y%m%d") - relativedelta(
        years=train_window_years
    )
    train_start_str = train_start_dt.strftime("%Y%m%d")

    train_start, train_end = resolve_deploy_train_window(
        trade_cal=trade_cal,
        deploy_train_end=deploy_train_end,
        train_window_years=train_window_years,
    )

    if train_start is None:
        logger.error(f"无法找到有效的部署模型 train_start(目标: {train_start_str})")
        return None, None

    if train_end is None:
        logger.error(f"无法找到有效的部署模型 train_end(目标: {deploy_train_end})")
        return None, None

    logger.info("=" * 80)
    logger.info("部署模型训练(Deploy Training)")
    logger.info(f"  训练区间: {train_start} 至 {train_end}")
    logger.info("  (无测试区间,用于部署)")
    logger.info("=" * 80)
    return train_start, train_end


def _train_deploy_model(
    train_start: str,
    train_end: str,
    *,
    frame_provider: Callable[[str, str], TrainingFrame],
    params: TrainFoldParams,
    main_board_codes: set,
    topk_values: List[int],
    trade_cal: pd.DataFrame,
    runtime_derivers: Optional[RuntimeDerivers],
) -> tuple:
    """部署模型训练段(集成/单模型;P3-T3 提取自 execute_deploy_training,逐字)。

    返回 (model, feature_columns, train_params, train_metrics, val_metrics,
    df_val_split_original, validation_feature_columns, validation_label_column,
    data_stats, train_days_count, total_train_samples, X_train_len, X_val_len,
    ensemble_meta, ensemble_seeds, ensemble_offsets)。
    """
    ensemble_offsets = params.ensemble_offsets
    ensemble_seeds = _resolve_ensemble_seeds(params)
    use_ensemble = ensemble_offsets > 0 or len(ensemble_seeds) > 1

    if use_ensemble:
        if ensemble_offsets > 0:
            # deploy 比例路径(无显式缝),多偏移窗口组合保持可用
            from .training_core import compute_offset_windows

            windows = compute_offset_windows(train_start, train_end, ensemble_offsets, trade_cal)
        else:
            windows = [(train_start, train_end)]
        logger.info(
            f"部署模型集成训练: {len(windows)}个窗口 × {len(ensemble_seeds)}个种子 "
            f"= {len(windows) * len(ensemble_seeds)}个子模型"
            f"(偏移±{ensemble_offsets}个月, seeds={ensemble_seeds})"
        )

        sub_models, base_result, ensemble_meta = _build_ensemble_sub_models(
            windows,
            frame_provider,
            params,
            main_board_codes,
            ensemble_seeds,
            runtime_derivers=runtime_derivers,
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
            frame_provider(train_start, train_end),
            params,
            main_board_codes,
            runtime_derivers=runtime_derivers,
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
        ensemble_meta = {}

    return (
        model,
        feature_columns,
        train_params,
        train_metrics,
        val_metrics,
        df_val_split_original,
        validation_feature_columns,
        validation_label_column,
        data_stats,
        train_days_count,
        total_train_samples,
        X_train_len,
        X_val_len,
        ensemble_meta,
        ensemble_seeds,
        ensemble_offsets,
    )


def _assemble_deploy_full_params(
    train_params: Dict[str, Any],
    params: TrainFoldParams,
    feature_columns: List[str],
    model: Any,
    ensemble_seeds: List[int],
    ensemble_offsets: int,
) -> Dict[str, Any]:
    """注册参数组装段(P3-T3 提取,语句逐字保持)。"""
    full_train_params = train_params.copy()
    attach_cons_revision_schema_version(
        full_train_params,
        params.enable_consensus_revision_features,
    )
    attach_cashflow_quality_train_params(
        full_train_params,
        params.enable_cashflow_quality_features,
        feature_columns=feature_columns,
    )
    attach_dividend_train_params(
        full_train_params,
        params.enable_dividend_policy_features,
    )
    full_train_params.update(
        {
            "algorithm": params.algorithm,
            "task": params.task,
            "label_transform": params.label_transform if params.task == "regression" else None,
            "winsorize_p": params.winsorize_p if params.label_transform == "cs_zscore" else None,
            "pos_quantile": params.pos_quantile if params.task == "classification" else None,
            "pos_topk": params.pos_topk if params.task == "classification" else None,
            "scale_pos_weight_manual": params.scale_pos_weight is not None,
            "is_deploy": True,
            **_build_feature_flag_train_params(params),
            "freshness_strategy": params.freshness_strategy,
            "event_freshness_half_life_days": params.event_freshness_half_life_days,
        }
    )
    if isinstance(model, EnsembleModel):
        full_train_params["ensemble_offsets"] = ensemble_offsets
        full_train_params["ensemble_seeds"] = ensemble_seeds
        full_train_params["ensemble_seed_keep_top_ratio"] = params.ensemble_seed_keep_top_ratio
        full_train_params["ensemble_seed_keep_min_models"] = params.ensemble_seed_keep_min_models
        full_train_params["ensemble_n_models"] = model.n_models
    return full_train_params


def _build_deploy_data_stats(
    data_stats: Dict[str, Any],
    params: TrainFoldParams,
    train_days_count: int,
    total_train_samples: int,
    X_train_len: int,
    X_val_len: int,
) -> Dict[str, Any]:
    """deploy 台账 data_stats 组装段(P3-T3 提取,逐字)。"""
    ffi_deploy = data_stats.get("feature_filter_info")
    ff_summary_deploy = {}
    if ffi_deploy and not ffi_deploy.get("skipped", True):
        ff_summary_deploy = {
            "feature_total": ffi_deploy["total_features"],
            "feature_stable": ffi_deploy["stable_count"],
            "feature_removed": ffi_deploy["removed_count"],
        }

    return {
        "trade_days_count": train_days_count,
        "total_samples": total_train_samples,
        "samples_after_filter": data_stats["samples_after_filter"],
        "train_samples": X_train_len,
        "val_samples": X_val_len,
        "val_start_date": data_stats["val_start_date"],
        "val_end_date": data_stats["val_end_date"],
        "val_ratio": params.val_ratio,
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


def _register_and_log_deploy(
    deploy_result: tuple,
    wf_run_id: str,
    *,
    train_start: str,
    train_end: str,
    registry: ModelRegistry,
    params: TrainFoldParams,
    topk_values: List[int],
    run_log_csv: str,
) -> Dict:
    """注册 + 健康度简报 + 台账段(P3-T3 提取自 execute_deploy_training,逐字)。"""
    (
        model,
        feature_columns,
        train_params,
        train_metrics,
        val_metrics,
        df_val_split_original,
        validation_feature_columns,
        validation_label_column,
        data_stats,
        train_days_count,
        total_train_samples,
        X_train_len,
        X_val_len,
        _ensemble_meta,
        ensemble_seeds,
        ensemble_offsets,
    ) = deploy_result

    val_daily_metrics = {}
    if len(df_val_split_original) > 0:
        val_daily_metrics = evaluate_validation_daily(
            model=model,
            df_val=df_val_split_original,
            feature_columns=validation_feature_columns,
            original_return_col=validation_label_column,
            task=params.task,
            topk_values=topk_values,
        )

    performance_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "validation_daily": val_daily_metrics,
        "test": {},
        "test_daily": {},
    }

    full_train_params = _assemble_deploy_full_params(
        train_params, params, feature_columns, model, ensemble_seeds, ensemble_offsets
    )

    version = registry.register_model(
        model=model,
        model_type=f"{params.algorithm}_{params.task}_wf",
        train_start_date=train_start,
        train_end_date=train_end,
        feature_columns=feature_columns,
        label_column=params.label_column,
        n_samples=X_train_len + X_val_len,
        train_params=full_train_params,
        performance_metrics=performance_metrics,
    )

    logger.info(f"部署模型已注册: v{version}")

    # 部署轮健康度简报:部署模型无 OOS 可评,验证段(早停窗口)质量是唯一健康度信号。
    # 部署窗口尾部紧贴当前行情,best_iteration 偏低与 val_daily_rankic_ir 偏弱通常
    # 反映验证段与训练主体的制度断裂(信号早熟),需结合后续纸面表现判断。
    floor_flag_deploy = bool(train_params.get("best_iteration_floor_triggered", False))
    logger.info(
        "部署轮健康度简报: "
        f"best_iteration={train_params.get('best_iteration')}"
        f"{'(低于下限,请检查)' if floor_flag_deploy else ''}, "
        f"val_es={data_stats.get('val_es_start_date')}~{data_stats.get('val_es_end_date')}"
        f"({data_stats.get('val_es_n_dates')}日), "
        f"val_daily_rankic_ir={val_daily_metrics.get('daily_rankic_ir')}, "
        f"val_daily_rankic_mean={val_daily_metrics.get('daily_rankic_mean')}, "
        f"val_top30_return_mean={val_daily_metrics.get('top30_return_mean')}"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    complete_data_stats = _build_deploy_data_stats(
        data_stats, params, train_days_count, total_train_samples, X_train_len, X_val_len
    )

    run_record = create_training_run_record_from_training_session(
        timestamp=timestamp,
        start_date=train_start,
        end_date=train_end,
        label_column=params.label_column,
        task=params.task,
        model_version=version,
        train_params=full_train_params,
        data_stats=complete_data_stats,
        performance_metrics=performance_metrics,
        wf_run_id=wf_run_id,
        split_index="deploy",
        test_start_date=None,
        test_end_date=None,
    )

    write_training_run_to_csv(run_record, run_log_csv)

    logger.info(f"部署模型训练运行日志已记录到: {run_log_csv}")

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
        "best_iteration_floor_triggered": bool(
            train_params.get("best_iteration_floor_triggered", False)
        ),
        "val_rankic_ir": val_daily_metrics.get("daily_rankic_ir"),
        "test_daily_metrics": {},
    }


def execute_deploy_training(
    deploy_train_end: str,
    wf_run_id: str,
    *,
    frame_provider: Callable[[str, str], TrainingFrame],
    registry: ModelRegistry,
    params: TrainFoldParams,
    train_window_years: int,
    main_board_codes: set,
    topk_values: List[int],
    trade_cal: pd.DataFrame,
    run_log_csv: str,
    runtime_derivers: Optional[RuntimeDerivers] = None,
) -> Optional[Dict]:
    """在 walk-forward 评估完成后,用最新数据训练部署模型(输出根隔离:registry
    与 run_log_csv 均由编排器注入隔离根路径)。

    Q10-B:P3 回放不调用本函数(编排器 deploy_train=False 默认);生产部署训练
    留 P4 验证。
    """
    train_start, train_end = _resolve_deploy_window(deploy_train_end, train_window_years, trade_cal)
    if train_start is None or train_end is None:
        return None

    deploy_result = _train_deploy_model(
        train_start,
        train_end,
        frame_provider=frame_provider,
        params=params,
        main_board_codes=main_board_codes,
        topk_values=topk_values,
        trade_cal=trade_cal,
        runtime_derivers=runtime_derivers,
    )
    return _register_and_log_deploy(
        deploy_result,
        wf_run_id,
        train_start=train_start,
        train_end=train_end,
        registry=registry,
        params=params,
        topk_values=topk_values,
        run_log_csv=run_log_csv,
    )
