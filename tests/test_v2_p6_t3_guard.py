# -*- coding: utf-8 -*-
"""P6 T3 生成链拦截 bootstrap 与自动裁决（P6-T3-14：禁止 bootstrap_delta/table）。

覆盖两层防线（B-02）：

1. 静态 AST：字面引用 / ``from ... import ... as bd`` 别名 / ``getattr``/``__import__`` 常量参数
   / ``__dict__[...]`` 下标常量；报告包自身必须零命中；
2. 运行期哨兵：生成链执行期内被禁函数一旦被调用即 ``BootstrapForbiddenError``（可恢复），
   并以端到端 ``generate_experiment`` 证明整链（含 CSV/渲染/manifest）不触发。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare import fold_subset  # noqa: E402
from src.lazybull.v2.reporting import (  # noqa: E402
    experiment_guard,
    experiment_projection,
    generate,
)
from tests.p6_t3_env import standard_projection, write_runs_batch  # noqa: E402

REPORTING_ROOT = ROOT / "src/lazybull/v2/reporting"


def test_p6_t3_24_reporting_package_has_no_bootstrap_reference():
    """静态：报告包零命中（含别名/动态取用形态）。"""
    report = experiment_guard.assert_generation_chain(REPORTING_ROOT)
    assert report["hits"] == []
    assert report["scanned"] >= 20


@pytest.mark.parametrize(
    "source",
    [
        "from scripts.compare.fold_subset import bootstrap_delta as bd\n",
        "from scripts.compare.fold_subset import bootstrap_table\n",
        "import bootstrap_delta\n",
        "value = getattr(module, 'bootstrap_delta')\n",
        "value = module.__dict__['bootstrap_table']\n",
        "value = importlib.import_module('bootstrap_delta')\n",
    ],
)
def test_p6_t3_25_ast_guard_catches_alias_and_dynamic_forms(source: str):
    """别名/动态取用不得绕过静态守卫。"""
    hits = experiment_guard.scan_source(source, label="fixture")
    assert hits, source


def test_p6_t3_25b_ast_guard_folds_simple_string_concat():
    """``getattr(m, "bootstrap_" + "delta")`` 的常量拼接同样命中（B-R2-01）。"""
    hits = experiment_guard.scan_source(
        "value = getattr(m, 'bootstrap_' + 'delta')\n", label="fixture"
    )
    assert hits


def test_p6_t3_25c_generation_guard_scans_production_entrypoint():
    """扫描域含生产入口脚本（``scripts/p6_report.py``），载荷登记 scope（B-R2-01）。"""
    report = experiment_guard.assert_generation_chain(
        REPORTING_ROOT, extra_paths=(ROOT / "scripts/p6_report.py",)
    )
    assert report["hits"] == []
    assert report["scope"]["extra_paths"]
    assert report["scanned"] >= len(list(REPORTING_ROOT.glob("*.py")))


def test_p6_t3_26_ast_guard_rejects_temp_module(tmp_path):
    """含别名导入的临时模块使生成链检查 fail-closed。"""
    (tmp_path / "sneaky.py").write_text(
        "from scripts.compare.fold_subset import bootstrap_delta as bd\n", encoding="utf-8"
    )
    with pytest.raises(experiment_guard.BootstrapForbiddenError, match="拦截"):
        experiment_guard.assert_generation_chain(tmp_path)


def test_p6_t3_27_runtime_sentinel_blocks_and_restores():
    """运行期哨兵：调用即抛，退出后恢复原函数。"""
    original = fold_subset.bootstrap_delta
    with experiment_guard.forbid_bootstrap_calls() as evidence:
        assert "bootstrap_delta" in evidence["guarded"]
        with pytest.raises(experiment_guard.BootstrapForbiddenError):
            fold_subset.bootstrap_delta(None, None, [0])
    assert fold_subset.bootstrap_delta is original


def test_p6_t3_28_end_to_end_generation_never_calls_bootstrap(tmp_path, monkeypatch):
    """整链（快照→投影→五 CSV→渲染→manifest）不触发 bootstrap；投影无裁决字段。"""

    def _boom(*_args, **_kwargs):
        raise AssertionError("生成链禁止调用 bootstrap_delta/bootstrap_table")

    monkeypatch.setattr(fold_subset, "bootstrap_delta", _boom)
    monkeypatch.setattr(fold_subset, "bootstrap_table", _boom)
    base = write_runs_batch(tmp_path / "runs_base", batch_id="batch_base")
    arm = write_runs_batch(tmp_path / "runs_arm", batch_id="batch_arm", nav_dip=0.04)
    request = generate.ExperimentRequest(
        reports_root=tmp_path / "reports",
        evidence_class="research",
        generated_at="20261106T000000Z",
        attempt_id="gen_t3_guard",
        input_evidence_classes=("research",),
        arms=(f"base=runs:{base}", f"arm=runs:{arm}"),
        baseline="base",
        publish_run=False,
    )
    result = generate.generate_experiment(request)
    assert result["bootstrap_guard"]["sentinel"]["guarded"]
    projection = result["projection"]
    assert projection["schema"] == experiment_projection.EXPERIMENT_SCHEMA
    text = repr(projection)
    for key in ("verdict", "decision", "adopt", "gate_pass"):
        assert f"'{key}'" not in text, f"投影不得包含裁决字段：{key}"
    _arms, built = standard_projection(tmp_path / "second")
    assert built["projection"]["schema"] == experiment_projection.EXPERIMENT_SCHEMA
