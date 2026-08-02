#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward 薄入口脚本。"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lazybull.ml.walk_forward.backtest import run_oos_backtest
from src.lazybull.ml.walk_forward.cli import (
    _normalize_selected_split_indices,
    parse_walk_forward_args,
)
from src.lazybull.ml.walk_forward.reporting import (
    build_daily_topk_detail_df,
    chain_nav_splits,
    write_walk_forward_topk_details,
    write_walk_forward_trade_details,
)
from src.lazybull.ml.walk_forward.runner import (
    _filter_splits_by_selected_indices,
    run_walk_forward,
)
from src.lazybull.ml.walk_forward.summary import write_walk_forward_summary
from src.lazybull.ml.walk_forward.training import (
    SEED_ENSEMBLE_KEEP_MIN_MODELS,
    SEED_ENSEMBLE_KEEP_TOP_RATIO,
    _align_to_trade_date,
    _build_ensemble_sub_models,
    _build_main_board_codes,
    _build_split_training_candidate,
    _evaluate_train_result_val_daily,
    _filter_to_main_board,
    _resolve_ensemble_seeds,
    _seed_model_sort_score,
    _select_ensemble_validation_result,
    _train_model_on_window,
    compute_offset_windows,
    execute_deploy_training,
    execute_split_training,
)


def main():
    """主函数。"""
    args = parse_walk_forward_args()
    run_walk_forward(args)


if __name__ == "__main__":
    main()
