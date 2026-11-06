# -*- coding: utf-8 -*-
"""P6 T1 配置/起始日/模型/年化投影（P6-T1-02/03/08）。

覆盖：实跑模型无来源固定降级且不构造 Registry/Storage（写拦截）；``account_start_date``
展平前冲突检测与 YAML 块顺序无关；年化只调唯一实现，初资/总值非正与起始日晚于 NAV 日 invalid。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.hosts.paper.performance import calculate_annualized_return  # noqa: E402
from src.lazybull.v2.reporting import config_view  # noqa: E402
from src.lazybull.v2.reporting.cockpit import (  # noqa: E402
    build_dashboard_projection,
    capture_paper_inputs,
)
from src.lazybull.v2.reporting.readonly_audit import ReadonlyGuard  # noqa: E402
from src.lazybull.v2.reporting.snapshot import capture_stable_snapshot  # noqa: E402
from tests.p6_t1_env import (  # noqa: E402
    INITIAL_CAPITAL,
    TOTAL_VALUE,
    TRADE_DATE,
    build_paper_root,
    build_projection,
)

BASE_CONFIG = {
    "engine": "v2",
    "initial_capital": INITIAL_CAPITAL,
    "model_version": 24001,
    "model_version_b": None,
    "ensemble_weight_a": 1.0,
    "account_start_date": "20260101",
}


def _imported_names(module: object) -> set[str]:
    """AST 枚举模块的 import 名称（用于断言不构造 Registry/Storage/日历）。"""
    import ast

    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


def _project(tmp_path: Path, **kwargs: object) -> dict:
    built = build_paper_root(tmp_path / "paper", **kwargs)  # type: ignore[arg-type]
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging"
    )
    return projection


def test_p6_t1_02_actual_model_not_persisted_and_no_registry(tmp_path: Path) -> None:
    """新 v2 无实跑模型来源 ⇒ 固定 no_persisted_actual_model；且不构造 Registry/Storage。"""
    built = build_paper_root(
        tmp_path / "paper",
        config={**BASE_CONFIG, "model_version": None},
    )
    info: dict = {}

    def reader(snapshot: object) -> None:
        info.clear()
        info.update(capture_paper_inputs(snapshot))  # type: ignore[arg-type]

    snapshot, _audit = capture_stable_snapshot(
        roots={"paper": built["root"]},
        staging_root=tmp_path / "staging",
        attempt_ids=["p6_t1_attempt_a"],
        reader=reader,
        max_retries=0,
    )
    with ReadonlyGuard(allowed_patterns=(), worktree_root=built["root"]) as guard:
        projection = build_dashboard_projection(
            snapshot=snapshot, generated_at="20261105T000000Z", capture_info=info
        )
    assert guard.violations == []
    model = projection["config"]["model"]
    assert model["actual_model"]["report_status"] == "not_available"
    assert model["actual_model"]["degradation_reason_code"] == "no_persisted_actual_model"
    assert model["follows_latest"] is True
    assert model["report_status"] == "degraded"
    assert model["degradation_reason_code"] == "no_readonly_resolution"
    assert model["model_version"] is None

    legacy = _project(
        tmp_path / "legacy", config={**BASE_CONFIG, "model_version": 24001}
    )
    legacy_model = legacy["config"]["model"]
    assert legacy_model["model_version"] == 24001
    assert legacy_model["actual_model"]["degradation_reason_code"] == (
        "no_persisted_actual_model"
    )

    forbidden_imports = {
        "ModelRegistry",
        "PanelDataStore",
        "Storage",
        "load_trade_cal",
        "get_trade_cal",
    }
    from src.lazybull.v2.reporting import authority, cockpit, quality, status

    for module in (config_view, cockpit, quality, authority, status):
        assert not (_imported_names(module) & forbidden_imports), module.__name__
    assert "model_registry" not in " ".join(
        name for name in _imported_names(config_view) if isinstance(name, str)
    )


def test_p6_t1_03a_start_date_same_value_any_yaml_order(tmp_path: Path) -> None:
    """extra 与顶层起始日相同 ⇒ 可用；两种 YAML 块顺序结果一致。"""
    forward = {"extra": {"account_start_date": "20260101"}, **BASE_CONFIG}
    backward = {**BASE_CONFIG, "extra": {"account_start_date": "20260101"}}
    first = _project(tmp_path / "forward", config=forward)
    second = _project(tmp_path / "backward", config=backward)
    for projection in (first, second):
        row = projection["config"]
        assert row["report_status"] == "available"
        assert row["account_start_date"] == "20260101"
        assert row["annualization"]["report_status"] == "available"
    assert (
        first["config"]["account_start_date"] == second["config"]["account_start_date"]
    )
    assert (
        first["config"]["annualization"]["annualized_return"]
        == second["config"]["annualization"]["annualized_return"]
    )


def test_p6_t1_03b_start_date_conflict_is_invalid(tmp_path: Path) -> None:
    """extra 与顶层起始日不同 ⇒ invalid/account_start_date_conflict（展平前拒绝）。"""
    conflicted = {**BASE_CONFIG, "account_start_date": "20251231",
                  "extra": {"account_start_date": "20260101"}}
    projection = _project(tmp_path, config=conflicted)
    row = projection["config"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "account_start_date_conflict"
    assert "model" not in row and "annualization" not in row
    assert {item["field_path"] for item in projection["degradations"]} == {"config"}


def test_p6_t1_annualization_uses_unique_implementation(tmp_path: Path) -> None:
    """年化值必须等于唯一实现 ``calculate_annualized_return`` 的返回（百分数）。"""
    projection = _project(tmp_path, config=BASE_CONFIG)
    row = projection["config"]["annualization"]
    expected = calculate_annualized_return(
        INITIAL_CAPITAL, TOTAL_VALUE, TRADE_DATE, "20260101"
    )
    assert row["report_status"] == "available"
    assert row["annualized_return"] == pytest.approx(expected)
    assert row["annualized_return"] == pytest.approx(expected, rel=1e-12)
    assert row["initial_capital"] == INITIAL_CAPITAL
    assert row["current_value"] == pytest.approx(TOTAL_VALUE)
    assert row["nav_date"] == TRADE_DATE
    assert row["verification_level"] == "authority_verified"


@pytest.mark.parametrize(
    ("config_patch", "reason"),
    [
        ({"initial_capital": 0.0}, "annualization_input_invalid"),
        ({"initial_capital": -1.0}, "annualization_input_invalid"),
        ({"account_start_date": "20270101"}, "annualization_start_after_nav"),
    ],
)
def test_p6_t1_annualization_invalid_inputs(
    tmp_path: Path, config_patch: dict, reason: str
) -> None:
    """初资非正与起始日晚于 NAV 日 ⇒ invalid（不得静默返回 0）。"""
    projection = _project(tmp_path, config={**BASE_CONFIG, **config_patch})
    row = projection["config"]["annualization"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == reason


def test_p6_t1_config_missing_is_not_available(tmp_path: Path) -> None:
    """配置缺失 ⇒ 配置区 not_available/config_missing（不猜默认值）。"""
    built = build_paper_root(tmp_path / "paper")
    (built["root"] / "config.yaml").unlink()
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging"
    )
    row = projection["config"]
    assert row["report_status"] == "not_available"
    assert row["degradation_reason_code"] == "config_missing"
