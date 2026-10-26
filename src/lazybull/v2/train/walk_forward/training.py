#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward 训练模块统一导出门面（v2 侧）。

P3-T3 搬运原样自 src.lazybull.ml.walk_forward.training(2026-10-22),包内改指。
"""

from .deploy_training import execute_deploy_training

# 注:旧 _build_split_training_candidate 职责已移入 V2Trainer.train_fold(训练+注册,
# §17.1-B);本门面导出折流程函数 execute_split_training / execute_skip_training_evaluation。
from .split_training import execute_skip_training_evaluation, execute_split_training
from .training_core import (
    MIN_MODELS,
    SEED_ENSEMBLE_KEEP_MIN_MODELS,
    SEED_ENSEMBLE_KEEP_TOP_RATIO,
    _align_to_trade_date,
    _build_ensemble_sub_models,
    _build_main_board_codes,
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
    _metric_value,
    _print_oos_focus_panel,
    _print_pre_backtest_model_summary,
    _safe_float,
    _topk_key_metrics,
)

__all__ = [
    "MIN_MODELS",
    "SEED_ENSEMBLE_KEEP_MIN_MODELS",
    "SEED_ENSEMBLE_KEEP_TOP_RATIO",
    "_align_to_trade_date",
    "_build_ensemble_sub_models",
    "_build_main_board_codes",
    "_evaluate_train_result_val_daily",
    "_filter_to_main_board",
    "_resolve_ensemble_seeds",
    "_seed_model_sort_score",
    "_select_ensemble_validation_result",
    "_train_model_on_window",
    "compute_offset_windows",
    "execute_deploy_training",
    "execute_skip_training_evaluation",
    "execute_split_training",
    "_fmt_metric",
    "_fmt_pct",
    "_metric_value",
    "_print_oos_focus_panel",
    "_print_pre_backtest_model_summary",
    "_safe_float",
    "_topk_key_metrics",
]
