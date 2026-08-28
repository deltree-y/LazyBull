#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""walk_forward_training 门面模块一致性测试。"""

import ast
from pathlib import Path

from src.lazybull.ml.walk_forward import training as facade
from src.lazybull.ml.walk_forward import training_core as core
from src.lazybull.ml.walk_forward import training_reporting as reporting
from src.lazybull.ml.walk_forward.deploy_training import execute_deploy_training
from src.lazybull.ml.walk_forward.split_training import (
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


# ── 一致预期修正 v2 链路缺口回归 ─────────────────────────────


def test_skip_training_metadata_reads_feature_columns_from_features_file(tmp_path):
    """skip-training 的存活列必须从独立 features 文件读取，而不是 metadata 内字段。"""
    import json
    import types

    from src.lazybull.ml.walk_forward.runner import _load_skip_training_metadata

    features_file = tmp_path / "v123_features.json"
    features_file.write_text(
        json.dumps(["zscore_cons_analyst_count_chg", "zscore_pe_ttm"]),
        encoding="utf-8",
    )

    class _StubRegistry:
        def __init__(self, models_dir, metadata):
            self.models_dir = models_dir
            self._metadata = metadata

        def _load_metadata(self, version):
            return self._metadata

    metadata = {
        "version": 123,
        "version_str": "v123",
        "features_file": features_file.name,
        "train_params": {"enable_consensus_revision_features": False},
    }
    registry = _StubRegistry(tmp_path, metadata)
    args = types.SimpleNamespace(enable_consensus_revision_features=False)

    result = _load_skip_training_metadata(registry, 123, args)

    assert result is not None
    assert result["feature_columns"] == ["zscore_cons_analyst_count_chg", "zscore_pe_ttm"]


def test_legacy_revision_model_warns_when_schema_version_missing():
    """含修正列但未记录 v2 schema 版本的旧模型加载时必须告警。"""
    from loguru import logger as loguru_logger

    from src.lazybull.ml.model_registry import _warn_legacy_consensus_revision_model

    messages = []
    sink_id = loguru_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        _warn_legacy_consensus_revision_model(
            feature_columns=["zscore_cons_analyst_count_chg"],
            train_params={"enable_consensus_revision_features": True},
            version_str="v22715",
        )
    finally:
        loguru_logger.remove(sink_id)

    assert any("zscore_cons_analyst_count_chg" in m and "train/serve" in m for m in messages)

    # 已记录 v2 版本的新模型不告警
    messages.clear()
    sink_id = loguru_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        _warn_legacy_consensus_revision_model(
            feature_columns=["zscore_cons_analyst_count_chg"],
            train_params={
                "enable_consensus_revision_features": True,
                "cons_revision_schema_version": 2,
            },
            version_str="v30000",
        )
    finally:
        loguru_logger.remove(sink_id)

    assert messages == []


def test_legacy_cashflow_model_warns_when_schema_missing_or_mismatched():
    """普通模型加载前必须识别同名异义的旧现金流质量模型。"""
    from loguru import logger as loguru_logger

    from src.lazybull.factors.cashflow_quality import CASHFLOW_QUALITY_SCHEMA_VERSION
    from src.lazybull.ml.model_registry import _warn_legacy_cashflow_quality_model

    messages = []
    sink_id = loguru_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        _warn_legacy_cashflow_quality_model(
            feature_columns=["zscore_fcf_yield", "cashflow_freshness_days"],
            train_params={"cashflow_quality_schema_version": 2},
            version_str="v23348",
        )
    finally:
        loguru_logger.remove(sink_id)

    assert any("zscore_fcf_yield" in message and "train/serve" in message for message in messages)

    messages.clear()
    sink_id = loguru_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        _warn_legacy_cashflow_quality_model(
            feature_columns=["zscore_fcf_yield", "cashflow_freshness_days"],
            train_params={"cashflow_quality_schema_version": CASHFLOW_QUALITY_SCHEMA_VERSION},
            version_str="v30000",
        )
    finally:
        loguru_logger.remove(sink_id)

    assert messages == []


def test_read_cons_revision_schema_version_tolerates_malformed_value():
    """异常 schema 值（如 \"v2\"）必须安全返回 -1，不能抛异常中断 split 循环。"""
    from src.lazybull.ml.train_core.constants import read_cons_revision_schema_version

    assert read_cons_revision_schema_version({}) == -1
    assert read_cons_revision_schema_version({"cons_revision_schema_version": "v2"}) == -1
    assert read_cons_revision_schema_version({"cons_revision_schema_version": None}) == -1
    assert read_cons_revision_schema_version({"cons_revision_schema_version": 2}) == 2
    assert read_cons_revision_schema_version({"cons_revision_schema_version": "2"}) == 2
