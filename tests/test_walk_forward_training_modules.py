#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""walk_forward_training 门面模块一致性测试。"""

import ast
from pathlib import Path

from src.lazybull.ml import walk_forward_training as facade
from src.lazybull.ml import walk_forward_training_core as core
from src.lazybull.ml import walk_forward_training_reporting as reporting
from src.lazybull.ml.walk_forward_deploy_training import execute_deploy_training
from src.lazybull.ml.walk_forward_split_training import (
    _build_split_training_candidate,
    execute_split_training,
)


def test_facade_symbol_identity_matches_owner_modules():
    assert facade.SEED_ENSEMBLE_KEEP_TOP_RATIO == core.SEED_ENSEMBLE_KEEP_TOP_RATIO
    assert facade.SEED_ENSEMBLE_KEEP_MIN_MODELS == core.SEED_ENSEMBLE_KEEP_MIN_MODELS
    assert facade.MIN_MODELS == core.MIN_MODELS

    assert facade._build_main_board_codes is core._build_main_board_codes
    assert facade._filter_to_main_board is core._filter_to_main_board
    assert facade._align_to_trade_date is core._align_to_trade_date
    assert facade.compute_offset_windows is core.compute_offset_windows
    assert facade._train_model_on_window is core._train_model_on_window
    assert facade._resolve_ensemble_seeds is core._resolve_ensemble_seeds
    assert facade._select_ensemble_validation_result is core._select_ensemble_validation_result
    assert facade._build_ensemble_sub_models is core._build_ensemble_sub_models
    assert facade._evaluate_train_result_val_daily is core._evaluate_train_result_val_daily
    assert facade._seed_model_sort_score is core._seed_model_sort_score

    assert facade._safe_float is reporting._safe_float
    assert facade._fmt_metric is reporting._fmt_metric
    assert facade._fmt_pct is reporting._fmt_pct
    assert facade._metric_value is reporting._metric_value
    assert facade._topk_key_metrics is reporting._topk_key_metrics
    assert facade._print_oos_focus_panel is reporting._print_oos_focus_panel
    assert facade._print_pre_backtest_model_summary is reporting._print_pre_backtest_model_summary

    assert facade._build_split_training_candidate is _build_split_training_candidate
    assert facade.execute_split_training is execute_split_training
    assert facade.execute_deploy_training is execute_deploy_training


def test_facade_file_has_no_functiondef_nodes():
    source = Path(facade.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_defs = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert function_defs == []
