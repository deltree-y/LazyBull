#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward split 训练执行模块（v2 侧）。

P3-T3 搬运改写自 src.lazybull.ml.walk_forward.split_training(2026-10-22),改动点(§17.1-B):

1. **评估读取上移**:``evaluate_test_window`` 不再自行 ``load_features_data``,改接收
   调用方经适配层供给的测试窗 ``TrainingFrame``(panel 默认 / cs_train_verify 验收);
2. **注册段移入 ``V2Trainer.train_fold``**(协议职责:训练+注册);本模块保留
   评估 + 台账 + 报表编排;
3. **四族/availability 评估侧 DI 化**(v2/train 不 import factors):派生函数经
   ``RuntimeDerivers`` 注入、族出列清单经 ``family_output_cols`` 注入、可用性标记
   经 ``availability_ensurer`` 注入——panel 模式四族物化直读(帧内已有列,derive
   段跳过;``frame.runtime_lookups`` 非空 ⇒ 适配层已报错防双写),cs_train_verify
   模式同源 derive(语句/日志逐字);
4. **downside_penalty 段删除**(已终结家族,§13.3.1 ⑤;启用拒绝由编排器
   ``reject_retired_switches`` 守卫,本层无参数承载);
5. 台账 ``csv_path`` 显式必传(``v2/train/run_logger.py``,隔离台账义务);
6. ``evaluate_test_window``(旧 183 行/mccabe 16 在册)与 ``execute_split_training``
   (旧 224 行在册)按 §17.8 最小提取式拆分;
7. 注册表改指 ``v2.common.model_registry``;``EnsembleModel`` 改指 ``v2.train.ensemble``。
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.train.core import (
    RuntimeDerivers,
    add_blended_return_label,
    apply_serving_event_decay,
    evaluate_validation_daily,
)
from src.lazybull.v2.train.data_adapter import TrainingFrame
from src.lazybull.v2.train.run_logger import (
    create_training_run_record_from_training_session,
    write_training_run_to_csv,
)

from .reporting import build_daily_topk_detail_df
from .training_core import (
    TrainFoldParams,
    _filter_to_main_board,
    resolve_explicit_split_dates,
)
from .training_reporting import (
    _fmt_metric,
    _fmt_pct,
    _print_oos_focus_panel,
    _print_pre_backtest_model_summary,
    _safe_float,
)
from .utils import WalkForwardSplit

if TYPE_CHECKING:
    # 类型注解专用;运行时不 import(trainer → training_core 触发包 __init__ 的环,断环)
    from src.lazybull.v2.train.trainer import V2Trainer

#: 评估侧四族段配置(族名 → (runtime_lookups 键, RuntimeDerivers 属性, 日志标签))
_EVAL_DERIVE_FAMILIES = (
    ("holdertrade", "holdertrade_lookup", "holdertrade", "股东增减持"),
    ("repurchase", "repurchase_lookup", "repurchase", "股票回购"),
    ("top10fh", "top10fh_panel", "top10fh", "十大流通股东"),
    ("top_inst", "top_inst_lookup", "top_inst", "龙虎榜机构席位"),
)


def _load_eval_frame(frame: TrainingFrame, main_board_codes: set) -> pd.DataFrame:
    """测试窗帧取出 + 主板过滤(P3-T3 提取自 evaluate_test_window;读取已上移)。"""
    df_test = _filter_to_main_board(frame.df, main_board_codes, "测试窗口")

    logger.info("=" * 60)
    logger.info("样本外测试集评估(OOS Evaluation)")
    logger.info("=" * 60)
    return df_test


def _apply_eval_filters(df_test: pd.DataFrame, params: TrainFoldParams) -> tuple:
    """评估域过滤链:混合标签 → ST/停牌/涨停剔除 → 标签 dropna(逐字保持)。"""
    df_test_eval = df_test.copy()
    evaluation_label_column = add_blended_return_label(
        df_test_eval,
        params.label_column,
        params.neutral_label_blend_weight,
    )

    filter_columns = ["is_st", "is_suspended", "is_limit_up"]
    mask = pd.Series(True, index=df_test_eval.index)
    for col in filter_columns:
        if col in df_test_eval.columns:
            mask = mask & (~df_test_eval[col].astype(bool))
    df_test_eval = df_test_eval[mask].copy()

    if evaluation_label_column in df_test_eval.columns:
        df_test_eval = df_test_eval.dropna(subset=[evaluation_label_column])

    logger.info(f"测试集样本数(过滤后): {len(df_test_eval)}")
    return df_test_eval, evaluation_label_column


def _apply_eval_freshness(df_test_eval: pd.DataFrame, params: TrainFoldParams) -> pd.DataFrame:
    """OOS 评估侧复现训练时的事件型 freshness 衰减(train/serve 一致,逐字)。"""
    return apply_serving_event_decay(
        df_test_eval,
        freshness_strategy=params.freshness_strategy,
        event_freshness_half_life_days=params.event_freshness_half_life_days,
    )


def _derive_eval_families(
    df_test_eval: pd.DataFrame,
    feature_columns: List[str],
    frame: TrainingFrame,
    runtime_derivers: Optional[RuntimeDerivers],
    family_output_cols: Optional[Dict[str, List[str]]],
    availability_ensurer: Optional[Callable[[pd.DataFrame, List[str]], List[str]]],
) -> None:
    """运行时派生族就地拼接(DI 化;panel 模式物化直读跳过,cs_train_verify 同源 derive)。

    - panel 模式:``frame.runtime_lookups`` 为 None(适配层强制,传 lookups 已报错),
      四族段自然跳过——物化列已在帧内,**禁止 derive 双写/覆盖**(§3.3-③);
    - cs_train_verify 模式:lookup 由 ``frame.runtime_lookups`` 携带;族启用
      (lookup 非空且模型列含本族出列)而缺 deriver/出列清单 ⇒ 报错(DI 契约);
    - ``availability_ensurer`` 为 None ⇒ 跳过(等价于旧 ``ensure_availability_markers``
      无相关 wanted 列时的无操作路径;模型消费缺失列由缺列硬报错兜底)。
    """
    if availability_ensurer is not None:
        # OOS 评估侧与 MLSignal 一致:按模型特征列派生运行时可用性标记
        derived_markers = availability_ensurer(df_test_eval, feature_columns)
        if derived_markers:
            logger.info(f"OOS 测试集派生可用性标记: {derived_markers}")

    runtime_lookups = frame.runtime_lookups or {}
    for family, lookup_key, deriver_attr, label in _EVAL_DERIVE_FAMILIES:
        lookup = runtime_lookups.get(lookup_key)
        output_cols = (family_output_cols or {}).get(family) or []
        if lookup is None or not any(col in feature_columns for col in output_cols):
            continue
        deriver = getattr(runtime_derivers, deriver_attr, None) if runtime_derivers else None
        if deriver is None:
            raise ValueError(
                f"OOS 评估侧 {label} 派生缺注入(runtime_derivers.{deriver_attr};"
                "v2/train 不 import factors,由调用方从 factors 组装注入)"
            )
        derived = deriver.derive_fn(
            df_test_eval,
            lookup,
            wanted=feature_columns,
            log_prefix="[OOS 评估] ",
        )
        if derived:
            logger.info(f"OOS 测试集派生{label}列: {derived}")


def _predict_eval_scores(
    df_test_eval: pd.DataFrame,
    model: Any,
    feature_columns: List[str],
    params: TrainFoldParams,
) -> str:
    """缺列硬报错 + 预测分数列(downside_penalty 段已删除——退役家族,逐字保持其余)。"""
    missing_test_columns = [col for col in feature_columns if col not in df_test_eval.columns]
    if missing_test_columns:
        raise ValueError(
            "OOS 测试集缺少模型特征列(禁止静默降级): "
            f"{missing_test_columns[:10]}{'...' if len(missing_test_columns) > 10 else ''}"
            f"(共 {len(missing_test_columns)} 列)"
        )

    X_test_features = df_test_eval[feature_columns]

    if params.task == "classification":
        y_test_pred_proba = model.predict_proba(X_test_features)[:, 1]
        df_test_eval["pred_score"] = y_test_pred_proba
    else:
        y_test_pred = model.predict(X_test_features)
        df_test_eval["pred_score"] = y_test_pred

    return "pred_score"


def evaluate_test_window(
    split: WalkForwardSplit,
    model: Any,
    feature_columns: List[str],
    *,
    frame: TrainingFrame,
    params: TrainFoldParams,
    main_board_codes: set,
    topk_values: List[int],
    runtime_derivers: Optional[RuntimeDerivers] = None,
    family_output_cols: Optional[Dict[str, List[str]]] = None,
    availability_ensurer: Optional[Callable[[pd.DataFrame, List[str]], List[str]]] = None,
) -> Dict:
    """在测试窗口上执行 OOS 评估(逐日 Top-K 明细 + 测试指标)。

    训练路径(``execute_split_training``)与 skip-training 评估路径
    (``execute_skip_training_evaluation``)共用本函数,保证两边过滤/派生口径一致。

    v2 改动:测试窗帧由调用方经适配层供给(``frame``);四族/可用性 DI 化;
    downside_penalty 段删除(退役家族——启用拒绝归编排器 reject_retired_switches)。
    """
    df_test = _load_eval_frame(frame, main_board_codes)
    df_test_eval, evaluation_label_column = _apply_eval_filters(df_test, params)
    df_test_eval = _apply_eval_freshness(df_test_eval, params)
    _derive_eval_families(
        df_test_eval,
        feature_columns,
        frame,
        runtime_derivers,
        family_output_cols,
        availability_ensurer,
    )
    test_score_column = _predict_eval_scores(df_test_eval, model, feature_columns, params)

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
        task=params.task,
        topk_values=topk_values,
        emit_logs=False,
        prediction_col=test_score_column,
    )

    return {
        "topk_detail_df": topk_detail_df,
        "test_daily_metrics": test_daily_metrics,
        "test_samples": int(len(df_test_eval)),
    }


def execute_skip_training_evaluation(
    split: WalkForwardSplit,
    wf_run_id: str,
    *,
    registry: ModelRegistry,
    eval_frame_provider: Callable[[str, str], TrainingFrame],
    params: TrainFoldParams,
    main_board_codes: set,
    topk_values: List[int],
    model_version: int,
    runtime_derivers: Optional[RuntimeDerivers] = None,
    family_output_cols: Optional[Dict[str, List[str]]] = None,
    availability_ensurer: Optional[Callable[[pd.DataFrame, List[str]], List[str]]] = None,
) -> Dict:
    """skip-training 模式下的 OOS 评估(不训练、不注册新版本、不写训练台账)。

    用途:惩罚/政策类实验复用旧折模型时,仍需标准逐日 Top-K 明细(信号层尺子)
    与测试指标;配 ``skip_training_eval`` 开启,默认跳过以保持 NAV-only 复跑的最小开销。
    """
    logger.info("=" * 80)
    logger.info(f"[跳过训练] 评估 Split {split.split_index}(模型 v{model_version})")
    logger.info(f"  测试区间: {split.test_start} 至 {split.test_end}")
    logger.info("=" * 80)

    model, metadata = registry.load_model(version=model_version, strict_version_check=True)
    feature_columns = list(metadata.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError(f"[skip-training-eval] v{model_version} 缺少 feature_columns,禁止静默降级")

    test_eval = evaluate_test_window(
        split=split,
        model=model,
        feature_columns=feature_columns,
        frame=eval_frame_provider(split.test_start, split.test_end),
        params=params,
        main_board_codes=main_board_codes,
        topk_values=topk_values,
        runtime_derivers=runtime_derivers,
        family_output_cols=family_output_cols,
        availability_ensurer=availability_ensurer,
    )
    _print_oos_focus_panel(split.split_index, test_eval["test_daily_metrics"])

    return {
        "split_index": split.split_index,
        "train_start": split.train_start,
        "train_end": split.train_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "model_version": model_version,
        "feature_columns": feature_columns,
        "test_samples": test_eval["test_samples"],
        "test_daily_metrics": test_eval["test_daily_metrics"],
        "_topk_detail_df": test_eval["topk_detail_df"],
        "skipped_training": True,
    }


def _print_split_reports(
    split: WalkForwardSplit,
    ensemble_meta: Dict[str, Any],
    val_daily_metrics: Dict[str, Any],
    test_daily_metrics: Dict[str, Any],
    oos_detail_metrics: bool,
) -> None:
    """OOS 焦点面板 + 可选回测前简报(P3-T3 提取自 execute_split_training,逐字)。"""
    _print_oos_focus_panel(split.split_index, test_daily_metrics)

    if oos_detail_metrics:
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


def _write_split_run_log(
    split: WalkForwardSplit,
    wf_run_id: str,
    *,
    version: int,
    params: TrainFoldParams,
    full_train_params: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    data_stats: Dict[str, Any],
    train_days_count: int,
    total_train_samples: int,
    X_train_len: int,
    X_val_len: int,
    run_log_csv: str,
) -> None:
    """训练台账段(P3-T3 提取自 execute_split_training;csv_path 显式必传,逐字)。"""
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
        **feature_filter_summary,
    }

    run_record = create_training_run_record_from_training_session(
        timestamp=timestamp,
        start_date=split.train_start,
        end_date=split.train_end,
        label_column=params.label_column,
        task=params.task,
        model_version=version,
        train_params=full_train_params,
        data_stats=complete_data_stats,
        performance_metrics=performance_metrics,
        wf_run_id=wf_run_id,
        split_index=split.split_index,
        test_start_date=split.test_start,
        test_end_date=split.test_end,
    )

    write_training_run_to_csv(run_record, run_log_csv)

    logger.info(f"训练运行日志已记录到: {run_log_csv}")


def execute_split_training(
    split: WalkForwardSplit,
    wf_run_id: str,
    *,
    trainer: "V2Trainer",
    config: Any,
    store: Any,
    params: TrainFoldParams,
    main_board_codes: set,
    topk_values: List[int],
    run_log_csv: str,
    eval_frame_provider: Callable[[str, str], TrainingFrame],
    trade_cal: Optional[pd.DataFrame] = None,
    oos_detail_metrics: bool = False,
    runtime_derivers: Optional[RuntimeDerivers] = None,
    family_output_cols: Optional[Dict[str, List[str]]] = None,
    availability_ensurer: Optional[Callable[[pd.DataFrame, List[str]], List[str]]] = None,
) -> Dict:
    """执行单个 split 的训练(v2:训练+注册归 ``V2Trainer.train_fold``,本函数编排评估/台账)。

    显式日期缝换算(F10 D1,编排器职责):外层窗 ``[split.train_start, split.train_end]``
    × ``params.val_ratio`` ⇒ 四日期;``label_delta = max(horizon+1, 5)`` 与 prepare 语义一致。
    """
    logger.info("=" * 80)
    logger.info(f"开始训练 Split {split.split_index}")
    logger.info(f"  训练区间: {split.train_start} 至 {split.train_end}")
    logger.info(f"  测试区间: {split.test_start} 至 {split.test_end}")
    logger.info("=" * 80)

    trade_dates = _split_trade_dates(trade_cal, split)
    label_delta = _label_delta(params.label_column)
    train_start, train_end, val_start, val_end = resolve_explicit_split_dates(
        split.train_start, split.train_end, params.val_ratio, trade_dates, label_delta
    )
    version = int(trainer.train_fold(config, train_start, train_end, val_start, val_end, store))
    selected = trainer.last_fold_result
    if selected is None or selected.model_version != version:
        raise RuntimeError(
            f"train_fold 后 last_fold_result 缺失或版本不符(split {split.split_index})"
        )

    ensemble_meta = selected.ensemble_meta
    model = selected.model
    feature_columns = selected.feature_columns
    train_params = selected.train_params
    val_daily_metrics = selected.val_daily_metrics
    data_stats = selected.data_stats
    train_days_count = selected.train_days_count
    total_train_samples = selected.total_train_samples
    X_train_len = selected.X_train_len
    X_val_len = selected.X_val_len

    test_eval = evaluate_test_window(
        split=split,
        model=model,
        feature_columns=feature_columns,
        frame=eval_frame_provider(split.test_start, split.test_end),
        params=params,
        main_board_codes=main_board_codes,
        topk_values=topk_values,
        runtime_derivers=runtime_derivers,
        family_output_cols=family_output_cols,
        availability_ensurer=availability_ensurer,
    )
    topk_detail_df = test_eval["topk_detail_df"]
    test_daily_metrics = test_eval["test_daily_metrics"]

    _print_split_reports(
        split, ensemble_meta, val_daily_metrics, test_daily_metrics, oos_detail_metrics
    )

    performance_metrics = {
        "train": selected.train_metrics,
        "validation": selected.val_metrics,
        "validation_daily": val_daily_metrics,
        "test": {},
        "test_daily": test_daily_metrics,
    }

    _write_split_run_log(
        split,
        wf_run_id,
        version=version,
        params=params,
        full_train_params=selected.full_train_params,
        performance_metrics=performance_metrics,
        data_stats=data_stats,
        train_days_count=train_days_count,
        total_train_samples=total_train_samples,
        X_train_len=X_train_len,
        X_val_len=X_val_len,
        run_log_csv=run_log_csv,
    )

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
        "test_samples": test_eval["test_samples"],
        "best_iteration": train_params.get("best_iteration"),
        "best_iteration_floor_triggered": bool(
            train_params.get("best_iteration_floor_triggered", False)
        ),
        "val_rankic_ir": val_daily_metrics.get("daily_rankic_ir"),
        "test_daily_metrics": test_daily_metrics,
        "_topk_detail_df": topk_detail_df,
    }


def _split_trade_dates(trade_cal: Optional[pd.DataFrame], split: WalkForwardSplit) -> List[str]:
    """外层窗交易日序列(显式缝换算输入);trade_cal 缺失 ⇒ 报错(编排器必须供给)。"""
    if trade_cal is None or len(trade_cal) == 0:
        raise ValueError(
            f"显式日期缝换算需要交易日历(split {split.split_index});" "编排器必须注入 trade_cal"
        )
    dates = trade_cal[trade_cal["is_open"] == 1]["cal_date"].astype(str).tolist()
    return sorted(dates)


def _label_delta(label_column: str) -> int:
    """``label_delta = max(horizon+1, 5)``(与 prepare N+1 防泄漏语义一致)。"""
    digits = "".join(ch for ch in label_column if ch.isdigit())
    horizon = int(digits) if digits else 0
    return max(horizon + 1, 5)
