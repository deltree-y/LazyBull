# -*- coding: utf-8 -*-
"""v2/train/walk_forward 滚动训练编排子包门面(P3-T3)。

镜像旧 ``ml/walk_forward/__init__.py`` re-export 面,按新落点改指:
- runner → ``orchestrator``(V2WalkForwardOrchestrator / OrchestratorHostConfig /
  reject_retired_switches);
- backtest → 薄组装(BacktestHostParams / build_oos_backtest_kwargs /
  merge_backtest_metrics;OOS 回测本体 = v2/hosts/backtest/replay,宿主注入);
- cli → 不迁(薄入口化归 scripts 侧;退役开关守卫 = reject_retired_switches);
- data_state 成员不入 __all__(旧门面同口径,函数经 .data_state 模块直接用)。
"""

from .backtest import (
    BacktestHostParams,
    build_oos_backtest_kwargs,
    merge_backtest_metrics,
)
from .reporting import (
    build_daily_topk_detail_df,
    chain_nav_splits,
    write_walk_forward_holdings_snapshot,
    write_walk_forward_topk_details,
    write_walk_forward_trade_details,
)
from .summary import write_walk_forward_summary
from .training import (
    MIN_MODELS,
    SEED_ENSEMBLE_KEEP_MIN_MODELS,
    SEED_ENSEMBLE_KEEP_TOP_RATIO,
    _align_to_trade_date,
    _build_ensemble_sub_models,
    _build_main_board_codes,
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
    execute_skip_training_evaluation,
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
    "BacktestHostParams",
    "MIN_MODELS",
    "OrchestratorHostConfig",
    "SEED_ENSEMBLE_KEEP_MIN_MODELS",
    "SEED_ENSEMBLE_KEEP_TOP_RATIO",
    "V2WalkForwardOrchestrator",
    "WalkForwardSplit",
    "_align_to_trade_date",
    "_build_ensemble_sub_models",
    "_build_main_board_codes",
    "_evaluate_train_result_val_daily",
    "_filter_splits_by_selected_indices",
    "_filter_to_main_board",
    "_fmt_metric",
    "_fmt_pct",
    "_metric_value",
    "_print_oos_focus_panel",
    "_print_pre_backtest_model_summary",
    "_resolve_ensemble_seeds",
    "_safe_float",
    "_seed_model_sort_score",
    "_select_ensemble_validation_result",
    "_topk_key_metrics",
    "_train_model_on_window",
    "build_daily_topk_detail_df",
    "build_oos_backtest_kwargs",
    "chain_nav_splits",
    "compute_offset_windows",
    "execute_deploy_training",
    "execute_skip_training_evaluation",
    "execute_split_training",
    "generate_walk_forward_splits",
    "generate_walk_forward_splits_by_count",
    "merge_backtest_metrics",
    "print_splits_summary",
    "reject_retired_switches",
    "resolve_deploy_train_window",
    "write_walk_forward_holdings_snapshot",
    "write_walk_forward_summary",
    "write_walk_forward_topk_details",
    "write_walk_forward_trade_details",
]


# orchestrator 组懒加载(PEP 562):trainer.py → 本包 training_core 的导入会触发
# 本 __init__,顶层直导 orchestrator(→ trainer)构成循环;懒加载断环。
_LAZY_ORCHESTRATOR = {
    "OrchestratorHostConfig",
    "V2WalkForwardOrchestrator",
    "_filter_splits_by_selected_indices",
    "reject_retired_switches",
}


def __getattr__(name: str):
    if name in _LAZY_ORCHESTRATOR:
        from . import orchestrator

        return getattr(orchestrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
