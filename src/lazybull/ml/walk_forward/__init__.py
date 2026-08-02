# -*- coding: utf-8 -*-
"""Walk-forward 模块子包门面（re-export 公共 API）。"""

from .backtest import run_oos_backtest
from .cli import (
    _normalize_selected_split_indices,
    build_walk_forward_parser,
    parse_walk_forward_args,
)
from .reporting import (
    build_daily_topk_detail_df,
    chain_nav_splits,
    write_walk_forward_topk_details,
    write_walk_forward_trade_details,
)
from .runner import (
    _filter_splits_by_selected_indices,
    run_walk_forward,
)
from .summary import write_walk_forward_summary
from .training import (
    MIN_MODELS,
    SEED_ENSEMBLE_KEEP_MIN_MODELS,
    SEED_ENSEMBLE_KEEP_TOP_RATIO,
    _align_to_trade_date,
    _build_ensemble_sub_models,
    _build_main_board_codes,
    _build_split_training_candidate,
    _evaluate_train_result_val_daily,
    _filter_to_main_board,
    _fmt_metric,
    _fmt_pct,
    _metric_value,
    _print_oos_focus_panel,
    _print_pre_backtest_model_summary,
    _resolve_ensemble_seeds,
    _safe_float,
    _seed_model_sort_score,
    _select_ensemble_validation_result,
    _topk_key_metrics,
    _train_model_on_window,
    compute_offset_windows,
    execute_deploy_training,
    execute_split_training,
)
from .utils import (
    WalkForwardSplit,
    generate_walk_forward_splits,
    generate_walk_forward_splits_by_count,
    print_splits_summary,
    resolve_deploy_train_window,
)

__all__ = [
    "MIN_MODELS",
    "SEED_ENSEMBLE_KEEP_MIN_MODELS",
    "SEED_ENSEMBLE_KEEP_TOP_RATIO",
    "WalkForwardSplit",
    "_align_to_trade_date",
    "_build_ensemble_sub_models",
    "_build_main_board_codes",
    "_build_split_training_candidate",
    "_evaluate_train_result_val_daily",
    "_filter_splits_by_selected_indices",
    "_filter_to_main_board",
    "_fmt_metric",
    "_fmt_pct",
    "_metric_value",
    "_normalize_selected_split_indices",
    "_print_oos_focus_panel",
    "_print_pre_backtest_model_summary",
    "_resolve_ensemble_seeds",
    "_safe_float",
    "_seed_model_sort_score",
    "_select_ensemble_validation_result",
    "_topk_key_metrics",
    "_train_model_on_window",
    "build_daily_topk_detail_df",
    "build_walk_forward_parser",
    "chain_nav_splits",
    "compute_offset_windows",
    "execute_deploy_training",
    "execute_split_training",
    "generate_walk_forward_splits",
    "generate_walk_forward_splits_by_count",
    "parse_walk_forward_args",
    "print_splits_summary",
    "resolve_deploy_train_window",
    "run_oos_backtest",
    "run_walk_forward",
    "write_walk_forward_summary",
    "write_walk_forward_topk_details",
    "write_walk_forward_trade_details",
]
