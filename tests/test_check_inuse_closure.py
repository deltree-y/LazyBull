# -*- coding: utf-8 -*-
"""在用全域闭包检查脚本测试（scripts/check/check_inuse_closure.py，F10R-4）。

两类口径：
- 合成模块树（tmp_path）：验证闭包展开 / 相对导入层级 / 大小写精确匹配 / BOM 容错 /
  动态加载并集等机制本身，不依赖真实仓库；
- 真实仓 ratchet：守恒性质恒成立；在用总数锚定基线值（P2a 迁移改变结构时须显式更新
  基线——与 complexity_baseline.json 同款棘轮语义）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_CHECK = Path(__file__).resolve().parent.parent / "scripts" / "check"
sys.path.insert(0, str(SCRIPTS_CHECK))

import check_inuse_closure as cic  # noqa: E402

# 真实仓 ratchet 基线（2026-10-06 对账定稿；P2a 迁移后显式更新：
# T1（2026-10-07）新增 v2/common 复制件 13 模块（rules×5 + 支撑×8），
# 均不在两条主链路闭包内 ⇒ 不在用 49→62、全仓 249→262；静态闭包与在用数不变；
# T2（2026-10-07）新增 v2/common/protocols 定义件 3 模块（signal/core/hosts，
# 纯 Protocol 定义、无实现侧消费方）⇒ 不在用 62→65、全仓 262→265；其余不变；
# T4（2026-10-07）新增 v2/core 部件 15 模块（decision 2 + execution 3 +
# accounting 4 + signal 1 + 包 __init__×5），均不在两条主链路闭包内
# ⇒ 不在用 65→80、全仓 265→280；静态闭包与在用数不变；
# T3（2026-10-07）新增 v2/core/decision 复制件 4 模块（buy_plan/sizing/
# industry_constraint/weight_processor），均不在两条主链路闭包内
# ⇒ 不在用 80→84、全仓 280→284；静态闭包与在用数不变；
# T5（2026-10-07）新增 v2/core/execution 交付件 8 模块（run_loop/
# buy_execution/sell_execution/signal_execution/pending_execution/
# reporting/engine/ml_signal_feed），均不在两条主链路闭包内
# ⇒ 不在用 84→92、全仓 284→292；静态闭包与在用数不变；
# T6（2026-10-08）新增 v2/core/signal 交付件 3 模块（ml_signal/
# ensemble_signal/factory），均不在两条主链路闭包内
# ⇒ 不在用 92→95、全仓 292→295；静态闭包与在用数不变；
# T7（2026-10-08）新增 v2/hosts/backtest 交付件 6 模块（包 __init__×2 +
# runtime/replay/reporter/runs_writer），均不在两条主链路闭包内
# ⇒ 不在用 95→101、全仓 295→301；静态闭包与在用数不变）
INUSE_BASELINE = 200
NOTIN_BASELINE = 101
STATIC_BASELINE = 196


@pytest.fixture()
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """构造合成仓库：src/lazybull 两级包 + 入口脚本。"""
    lb = tmp_path / "src" / "lazybull"
    (lb / "alpha").mkdir(parents=True)
    (lb / "alpha" / "__init__.py").write_text("", encoding="utf-8")
    (lb / "alpha" / "core.py").write_text(
        "from src.lazybull.alpha.helper import util\n", encoding="utf-8"
    )
    (lb / "alpha" / "helper.py").write_text(
        "from . import sibling\n", encoding="utf-8"  # __init__ 内相对导入已空；普通模块砍 1 层
    )
    # helper 的相对导入目标是 alpha 包 init（普通模块内 from . import x → 包本身）
    (lb / "beta").mkdir()
    (lb / "beta" / "__init__.py").write_text("from . import leaf\n", encoding="utf-8")
    (lb / "beta" / "leaf.py").write_text(
        "def f():\n    from src.lazybull.alpha import core\n", encoding="utf-8"
    )  # 函数体内惰性 import（闭包必须覆盖）
    (lb / "storage.py").write_text("", encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "entry_main.py").write_text(
        "from src.lazybull.alpha.core import util\nfrom src.lazybull.beta import leaf\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cic, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_static_closure_spans_lazy_and_function_level_imports(fake_repo: Path):
    """闭包覆盖跨包 import 与函数体内惰性 import；scripts 入口计入。"""
    entries = [fake_repo / "scripts" / "entry_main.py"]
    closure = cic.compute_static_closure(entries)
    lb = {m for m in closure if m.startswith("src.lazybull")}
    assert "src.lazybull.alpha.core" in lb
    assert "src.lazybull.alpha.helper" in lb
    assert "src.lazybull.beta.leaf" in lb          # 经 beta/__init__ 相对导入
    assert "scripts.entry_main" in closure


def test_case_sensitive_module_resolution(fake_repo: Path):
    """大写模块名不得匹配小写文件（Windows 不敏感文件系统的防护）。"""
    assert cic._module_to_path("src.lazybull.Storage") is None  # 仓库只有 storage.py
    assert cic._module_to_path("src.lazybull.storage") is not None


def test_dynamic_union_and_conservation(fake_repo: Path, monkeypatch: pytest.MonkeyPatch):
    """动态加载清单并入在用全域；守恒 = 在用 + 不在用 = 全仓。"""
    monkeypatch.setattr(
        cic, "DYNAMIC_RISK_MODULES", ["src.lazybull.alpha.hidden_dyn"]
    )
    # 额外落一个只有动态加载可达的模块
    (fake_repo / "src" / "lazybull" / "alpha" / "hidden_dyn.py").write_text(
        "", encoding="utf-8"
    )
    report = cic.build_inuse_report(["scripts/entry_main.py"])
    assert report["dynamic_extra"] == ["src.lazybull.alpha.hidden_dyn"]
    assert report["conservation_ok"] is True
    assert report["inuse_count"] + report["notin_count"] == report["all_modules_count"]


def test_bom_init_is_parseable(fake_repo: Path):
    """带 UTF-8 BOM 的 __init__.py 必须可解析（真实仓三处踩坑回归）。"""
    bom_init = fake_repo / "src" / "lazybull" / "beta" / "__init__.py"
    bom_init.write_text("﻿from . import leaf\n", encoding="utf-8")
    closure = cic.compute_static_closure([fake_repo / "scripts" / "entry_main.py"])
    assert "src.lazybull.beta.leaf" in {m for m in closure if m.startswith("src.lazybull")}


def test_real_repo_ratchet_conservation():
    """真实仓：守恒性质 + 总数棘轮（P2a 迁移后显式更新基线）。"""
    report = cic.build_inuse_report()
    assert report["conservation_ok"] is True, f"守恒破坏: {report['missing_files']}"
    assert report["static_closure_count"] == STATIC_BASELINE
    assert report["inuse_count"] == INUSE_BASELINE
    assert report["notin_count"] == NOTIN_BASELINE
    # 动态加载缺口持续存在（机制必要性锚点）
    assert "src.lazybull.factors.risk.downside_factors" in report["dynamic_extra"]


def test_real_repo_matches_committed_baseline_fixture():
    """实时计算与已提交底稿 fixture 逐集合一致（底稿 = 可复核证据，进版本控制）。"""
    import json

    fixture = (
        Path(__file__).resolve().parent / "fixtures" / "inuse_closure_baseline_20261006.json"
    )
    baseline = json.loads(fixture.read_text(encoding="utf-8"))
    report = cic.build_inuse_report()
    assert set(report["inuse_modules"]) == set(baseline["inuse_modules"])
    assert set(report["notin_modules"]) == set(baseline["notin_modules"])
