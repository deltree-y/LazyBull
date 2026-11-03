# -*- coding: utf-8 -*-
"""P5b T1 写域与身份共用件测试（tests/ 全量豁免复杂度）。

覆盖：生产根硬禁（项目/配置根 + 声明 data_root 下的 paper/runs/models）、外部根需显式放开、
全流程递归快照证明生产目录零写入、不触达 LedgerEntry、身份摘要口径与代码态双档口径。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t1_env as envmod  # noqa: E402

from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    TestbedScopeError,
    assert_write_scope,
    identity_summary,
    initialize_d0_fresh,
    resolve_campaign_root,
    testbed_root,
)
from src.lazybull.v2.hosts.testbed import d0 as d0_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import identity as identity_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import paths as paths_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import registration as registration_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import storage as storage_module  # noqa: E402

_TESTBED_MODULES = (
    paths_module,
    identity_module,
    storage_module,
    registration_module,
    d0_module,
)


def _recursive_entries(path: Path) -> set:
    if not path.is_dir():
        return set()
    return {entry.relative_to(path).as_posix() for entry in path.rglob("*")}


@pytest.mark.parametrize("subdir", ["paper", "runs", "models"])
def test_project_production_roots_are_hard_forbidden(subdir):
    target = paths_module.project_root() / "data" / subdir / "p5b_probe"
    for allow_external in (False, True):
        with pytest.raises(TestbedScopeError, match="禁止写入生产根"):
            assert_write_scope(target, allow_external_root=allow_external)


@pytest.mark.parametrize("subdir", ["paper", "runs", "models"])
def test_declared_data_root_production_roots_are_hard_forbidden(tmp_path, subdir):
    """隔离数据根下的 paper/runs/models 同样禁写（R1-T1-04 反例）。"""
    data_root = tmp_path / "data"
    target = data_root / subdir / "p5b_probe"
    with pytest.raises(TestbedScopeError, match="禁止写入生产根"):
        assert_write_scope(
            target, data_root=str(data_root), allow_external_root=True
        )


def test_campaign_root_inside_production_paper_is_rejected(tmp_path):
    """无 allow_external_root 且目标落在生产面时，注册入口必须直接拒绝。"""
    env = envmod.build_env(tmp_path)
    production = paths_module.project_root() / "data" / "paper"
    with pytest.raises(TestbedScopeError, match="禁止写入生产根"):
        envmod.register_base_campaign_with_base(env, production, "t1_in_paper")


def test_external_root_requires_explicit_opt_in(tmp_path):
    external = tmp_path / "somewhere" / "campaign"
    with pytest.raises(TestbedScopeError, match="越界"):
        assert_write_scope(external)
    assert assert_write_scope(external, allow_external_root=True) == external.resolve()


def test_unc_and_posix_remote_paper_root_are_not_local_forbidden(monkeypatch):
    """配置 paper 根为远端（UNC / 双斜杠）时不得并入本地禁写面（R2R-07）。"""
    from src.lazybull.v2.hosts.testbed import paths as module

    for remote in ("//host/share/paper", "\\\\host\\share\\paper", "relative/paper"):
        monkeypatch.setattr(
            "src.lazybull.common.config.get_paper_root", lambda remote=remote: remote
        )
        assert module._configured_paper_root() is None


def test_declared_data_root_is_accepted_as_testbed_base(tmp_path):
    data_root = tmp_path / "iso_data"
    target = data_root / "paper_shadow" / "p5b_testbed" / "probe_campaign"
    assert assert_write_scope(target, data_root=str(data_root)) == target.resolve()


def test_default_testbed_root_layout():
    root = testbed_root()
    assert root.parts[-2:] == ("paper_shadow", "p5b_testbed")
    campaign = resolve_campaign_root("p5b_layout_probe", allow_external_root=True)
    assert campaign.name == "p5b_layout_probe"
    assert campaign.parent == root


def test_full_t1_flow_writes_only_campaign_root(tmp_path):
    project_data = paths_module.project_root() / "data"
    watched_names = ("paper", "runs", "models")
    watched = {name: _recursive_entries(project_data / name) for name in watched_names}
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    d0_record = initialize_d0_fresh(prepared.root, allow_external_root=True)
    from src.lazybull.v2.hosts.testbed import DuplicateRegistrationError

    with pytest.raises(DuplicateRegistrationError):
        envmod.register_base_arm(env, prepared.root, "late_arm")
    with pytest.raises(DuplicateRegistrationError):
        initialize_d0_fresh(prepared.root, allow_external_root=True)
    for name, before in watched.items():
        assert _recursive_entries(project_data / name) == before, name
    written = {path.name for path in (prepared.root / "registered").iterdir()}
    assert {"campaign.json", "arms", "d0_fresh.json"} <= written
    assert set(d0_record["arms"]) == {"aa_1", "aa_2"}
    assert prepared.root.is_relative_to(env.base_root)


def test_no_ledger_write_path_in_testbed_modules():
    """源码级守卫：试车场不引用 LedgerEntry 写入入口，也不装配 DataStore。"""
    for module in _TESTBED_MODULES:
        source = inspect.getsource(module)
        assert "append_ledger_entry" not in source, module.__name__
        assert "data_store import" not in source, module.__name__
        assert "LedgerEntry(" not in source, module.__name__


def test_identity_summary_projects_all_five_segments(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    for arm in prepared.arms.values():
        identity = arm["identity"]
        summary = arm["identity_summary"]
        assert summary["input"] == identity["input"]["input_sha256"]
        assert summary["config"] == identity["config"]["fingerprint"]
        assert summary["model"] == identity["model"]["content_fingerprint"]
        assert summary["code"] == identity["code"]["execution_digest"]
        assert summary["window_state"] == identity["window_state"]["window_state_id"]
        assert set(summary) == {
            "input",
            "config",
            "model",
            "code",
            "window_state",
            "environment",
        }
        # 代码态双档：门禁用执行闭包，全仓摘要是不同口径的登记值
        assert identity["code"]["execution_digest"] != identity["code"]["workspace_source_digest"]
        assert identity["code"]["execution_file_count"] > 0
        assert identity_summary(identity) == summary


def test_aa_clone_arms_share_all_identity_hashes(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    left, right = prepared.arms["aa_1"], prepared.arms["aa_2"]
    assert left["identity_summary"] == right["identity_summary"]
    assert left["root"] != right["root"]


def test_mechanical_perturbation_changes_config_identity_only(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    perturbed = dict(env.config)
    perturbed["top_n"] = int(env.config["top_n"]) + 1
    record = envmod.register_base_arm(
        env, prepared.root, "perturbed_topn", role="mechanical_perturbation", config=perturbed
    )
    base_summary = prepared.arms["aa_1"]["identity_summary"]
    new_summary = record["identity_summary"]
    assert new_summary["config"] != base_summary["config"]
    for key in ("input", "model", "code", "window_state", "environment"):
        assert new_summary[key] == base_summary[key]
