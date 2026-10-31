# -*- coding: utf-8 -*-
"""P4-T3：E5 过渡验收工具与 E9 config 依赖点的机器锁定。"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from scripts.v2_p2a.compare_replay_vs_b0 import CompareReport, compare_fingerprint_gate
from src.lazybull.v2.train.data_adapter import SOURCE_CS_TRAIN_VERIFY, SOURCE_PANEL, AdapterConfig

ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "src" / "lazybull" / "v2"
F18_FUNCTIONS = {
    "get_data_root",
    "get_stock_selection_models_root",
    "get_shenwan_level",
    "get_reports_root",
    "get_models_root",
    "get_cost_settings",
}


def test_cs_train_verify_is_guarded_transitional_acceptance_tool():
    config = AdapterConfig(start_date="20260101", end_date="20260102")
    assert config.source_mode == SOURCE_PANEL
    assert SOURCE_CS_TRAIN_VERIFY == "cs_train_verify"
    for relative in (
        "train/__init__.py",
        "train/data_adapter.py",
        "hosts/backtest/replay.py",
    ):
        text = (V2_ROOT / relative).read_text(encoding="utf-8")
        assert "保活至旧链路物理删除" in text
        assert "永久验收工具" not in text


def test_e9_redundant_module_import_removed_but_function_fallback_explicitly_retained():
    path = V2_ROOT / "train" / "core" / "features.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    config_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "src.lazybull.common.config"
    ]
    assert len(config_imports) == 1
    assert config_imports[0].col_offset > 0


def test_e9_all_current_f18_import_points_are_accounted_for():
    actual = []
    for path in V2_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "src.lazybull.common.config":
                functions = tuple(alias.name for alias in node.names if alias.name in F18_FUNCTIONS)
                if functions:
                    actual.append((path.relative_to(V2_ROOT).as_posix(), functions))
    expected = [
        ("common/cost.py", ("get_cost_settings",)),
        ("common/model_registry.py", ("get_models_root",)),
        ("core/decision/industry_constraint.py", ("get_shenwan_level",)),
        ("core/signal/factory.py", ("get_stock_selection_models_root",)),
        ("core/signal/ml_signal.py", ("get_stock_selection_models_root",)),
        ("hosts/backtest/replay.py", ("get_data_root",)),
        ("hosts/backtest/reporter.py", ("get_reports_root",)),
        ("hosts/backtest/runtime.py", ("get_stock_selection_models_root",)),
        ("hosts/paper/config_adapter.py", ("get_stock_selection_models_root",)),
        ("hosts/paper/factory.py", ("get_data_root", "get_stock_selection_models_root")),
        ("hosts/paper/reporting.py", ("get_stock_selection_models_root",)),
        ("store/label_refresh.py", ("get_shenwan_level",)),
        ("train/core/features.py", ("get_data_root",)),
        ("train/walk_forward/summary.py", ("get_data_root",)),
    ]
    assert sorted(actual) == sorted(expected)
    assert sum(len(functions) for _, functions in actual) == 15


def test_e9_paper_root_import_is_separately_locked():
    actual = []
    for path in V2_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "src.lazybull.common.config":
                if any(alias.name == "get_paper_root" for alias in node.names):
                    actual.append(path.relative_to(V2_ROOT).as_posix())
    assert actual == ["hosts/paper/state_store.py"]


def test_gate2b_mode_and_sleeve_are_checked_per_fold():
    expected = {
        "merger_mode": "single_sleeve_passthrough",
        "merger_sleeve_id": "value",
    }
    frozen = pd.DataFrame({"split_index": [13]})
    bad = pd.DataFrame(
        {
            "split_index": [13],
            "merger_mode": ["legacy_bypass"],
            "merger_sleeve_id": ["wrong_sleeve"],
        }
    )
    report = CompareReport()
    compare_fingerprint_gate(bad, frozen, report, migration_only_values=expected)
    assert any("merger_mode 错值折=[13]" in diff for diff in report.gate3_diffs)
    assert any("merger_sleeve_id 错值折=[13]" in diff for diff in report.gate3_diffs)

    good = bad.assign(**expected)
    control = CompareReport()
    compare_fingerprint_gate(good, frozen, control, migration_only_values=expected)
    assert control.gate3_diffs == []
    source = (ROOT / "scripts/v2_p4/t3_gate2b_evidence.py").read_text(encoding="utf-8")
    assert "migration_only_values" in source
    assert '"merger_mode": "single_sleeve_passthrough"' in source
    assert '"merger_sleeve_id": "value"' in source
