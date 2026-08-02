#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward 训练兼容门面。"""

from .deploy_training import execute_deploy_training
from .split_training import (
    _build_split_training_candidate,
    execute_split_training,
)
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
    "SEED_ENSEMBLE_KEEP_TOP_RATIO",
    "MIN_MODELS",
    "SEED_ENSEMBLE_KEEP_MIN_MODELS",
    "_build_main_board_codes",
    "_filter_to_main_board",
    "_align_to_trade_date",
    "compute_offset_windows",
    "_train_model_on_window",
    "_resolve_ensemble_seeds",
    "_select_ensemble_validation_result",
    "_build_ensemble_sub_models",
    "_evaluate_train_result_val_daily",
    "_seed_model_sort_score",
    "_safe_float",
    "_fmt_metric",
    "_fmt_pct",
    "_metric_value",
    "_topk_key_metrics",
    "_print_oos_focus_panel",
    "_print_pre_backtest_model_summary",
    "_build_split_training_candidate",
    "execute_split_training",
    "execute_deploy_training",
]
