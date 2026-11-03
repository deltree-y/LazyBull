# -*- coding: utf-8 -*-
"""P5b T1 模拟臂注册契约测试（tests/ 全量豁免复杂度）。

覆盖：campaign/arm 注册成功、重复注册拒绝（含独占创建）、未注册臂拒绝、真实候选角色
拒绝、跨数据态拼臂拒绝、窗口漂移拒绝 / 水位变化不拒绝、记录封存复算、半初始化现场拒绝、
配置来源复用 P4、模型训练截止早于 D0。
"""

from __future__ import annotations

import json
import shutil
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
    ARM_ROLES,
    DuplicateRegistrationError,
    TestbedError,
    UnregisteredArmError,
    list_arms,
    load_arm,
    load_campaign,
    register_arm,
    register_campaign,
)
from src.lazybull.v2.hosts.testbed import registration as registration_module  # noqa: E402


def test_register_campaign_records_identity_and_calibration_flags(tmp_path):
    env = envmod.build_env(tmp_path)
    record = envmod.register_base_campaign(env, "t1_campaign")
    assert record["schema"] == "p5b_testbed_campaign/v1"
    assert record["scope"] == "historical_testbed"
    assert record["evidence_class"] == "calibration_only"
    assert record["virtual_clock"] is True
    assert record["not_forward_evidence"] is True
    assert record["not_sla_evidence"] is True
    assert record["not_real_candidate"] is True
    # 行域是"声明"，不得以回执形态呈现（R1-T1-02）
    assert record["row_domain"] == "structural"
    assert record["row_domain_verified"] is False
    assert record["row_domain_receipt"] is None
    assert record["label_cutoff"]["status"] == "not_applicable_t1"
    # 身份五段在 campaign 级只有 input/code/window-state，其余显式 None（R1-T1-09）
    assert record["identity"]["config"] is None
    assert record["identity"]["model"] is None
    summary = record["identity_summary"]
    assert summary["config"] is None and summary["model"] is None
    assert len(summary["input"]) == 64
    assert len(summary["code"]) == 64
    assert len(summary["window_state"]) == 16
    assert record["record_sha256"]
    reloaded = load_campaign(Path(record["campaign_root"]))
    assert reloaded["record_sha256"] == record["record_sha256"]


def test_campaign_clock_records_contiguity_and_calendar_identity(tmp_path):
    env = envmod.build_env(tmp_path)
    record = envmod.register_base_campaign(env)
    clock = record["clock"]
    assert clock["d0_date"] == env.days[0]
    assert clock["first_return_date"] == env.days[1]
    assert clock["n_return_days"] == len(env.days) - 1
    assert env.days[0] not in clock["return_days"]
    contiguity = clock["contiguity"]
    assert contiguity["verified"] is True
    assert contiguity["basis"] == "trading_calendar"
    assert contiguity["calendar"]["day_count"] == len(env.calendar)
    assert len(contiguity["calendar"]["calendar_sha256"]) == 64


def test_multi_day_window_requires_calendar(tmp_path):
    env = envmod.build_env(tmp_path)
    with pytest.raises(TestbedError, match="必须提供交易日历"):
        register_campaign(
            campaign_id="t1_campaign",
            d0_date=env.days[0],
            virtual_days=env.days,
            data_root=str(env.data_root),
            base_root=str(env.base_root),
            repo_root=str(env.repo_root),
            allow_external_root=True,
        )


def test_duplicate_campaign_registration_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    envmod.register_base_campaign(env, "t1_campaign")
    with pytest.raises(DuplicateRegistrationError):
        envmod.register_base_campaign(env, "t1_campaign")


def test_campaign_root_existing_without_record_is_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    leftover = env.base_root / "t1_campaign"
    leftover.mkdir(parents=True)
    with pytest.raises(DuplicateRegistrationError):
        envmod.register_base_campaign(env, "t1_campaign")


def test_campaign_id_and_window_validated(tmp_path):
    env = envmod.build_env(tmp_path)
    with pytest.raises(TestbedError):
        envmod.register_base_campaign(env, "Bad-Campaign")
    with pytest.raises(TestbedError):
        register_campaign(
            campaign_id="t1_campaign",
            d0_date=env.days[1],
            virtual_days=env.days,
            calendar=env.calendar,
            data_root=str(env.data_root),
            base_root=str(env.base_root),
            repo_root=str(env.repo_root),
            allow_external_root=True,
        )


def test_register_arm_records_five_identity_blocks(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    record = envmod.register_base_arm(env, root, "aa_1")
    assert record["schema"] == "p5b_testbed_arm/v1"
    assert record["arm_role"] == "aa_clone"
    assert record["root"] == "arms/aa_1"
    assert record["evidence_class"] == "calibration_only"
    assert record["not_forward_evidence"] is True
    assert record["not_sla_evidence"] is True
    assert record["record_sha256"]
    identity = record["identity"]
    assert set(identity) == {"input", "config", "model", "code", "window_state"}
    assert identity["model"]["version"] == 1
    assert set(identity["model"]["artifacts"]) == {"model.joblib", "metadata.json", "features.json"}
    # 模型产物载荷与 P4 门 1 同形（bytes/sha256），可跨阶段比对（R2-T1-15）
    assert set(identity["model"]["artifacts"]["model.joblib"]) == {"bytes", "sha256"}
    assert identity["model"]["train_end_date"] == envmod.TRAIN_END_DATE
    assert identity["model"]["trained_before_d0"] is True
    assert identity["config"]["execution"]["engine"] == "v2"
    assert identity["model"]["content_fingerprint"] == record["identity_summary"]["model"]
    assert (
        identity["window_state"]["window_state_id"]
        == campaign["identity"]["window_state"]["window_state_id"]
    )
    frozen = load_arm(root, "aa_1")
    assert frozen["record_sha256"] == record["record_sha256"]
    assert list_arms(root).keys() == {"aa_1"}


def test_duplicate_arm_registration_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    envmod.register_base_arm(env, root, "aa_1")
    with pytest.raises(DuplicateRegistrationError):
        envmod.register_base_arm(env, root, "aa_1")


def test_half_initialized_arm_dir_rejected_explicitly(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    half = root / "registered" / "arms" / "half_arm"
    half.mkdir(parents=True)
    (half / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TestbedError, match="半初始化现场"):
        envmod.register_base_arm(env, root, "half_arm")


def test_unregistered_arm_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    with pytest.raises(UnregisteredArmError):
        load_arm(root, "ghost")
    with pytest.raises(UnregisteredArmError):
        load_campaign(tmp_path / "testbed" / "not_registered")


@pytest.mark.parametrize("role", ["real_candidate", "forward_shadow", "production", "shadow"])
def test_real_candidate_roles_rejected(tmp_path, role):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    with pytest.raises(TestbedError, match="禁止注册真实候选"):
        envmod.register_base_arm(env, root, "bad_arm", role=role)
    assert list_arms(root) == {}


def test_unknown_role_rejected_but_allowed_roles_accepted(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    with pytest.raises(TestbedError, match="未知模拟臂角色"):
        envmod.register_base_arm(env, root, "bad_arm", role="anonymous_arm")
    registered = {
        role: envmod.register_base_arm(env, root, role, role=role)["arm_role"]
        for role in ARM_ROLES
    }
    assert registered == {role: role for role in ARM_ROLES}


def test_legacy_engine_arm_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    config = dict(env.config)
    config["engine"] = "legacy"
    with pytest.raises(TestbedError, match="只复用 v2 纸面宿主"):
        envmod.register_base_arm(env, root, "aa_legacy", config=config)


def test_arm_data_root_must_match_campaign(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    other = tmp_path / "other_data"
    envmod.write_partitions(other)
    with pytest.raises(TestbedError, match="data_root 与 campaign 不一致"):
        register_arm(
            campaign_root=root,
            arm_id="aa_1",
            arm_role="aa_clone",
            config_flat=env.config,
            models_dir=str(env.models_dir),
            data_root=str(other),
            repo_root=str(env.repo_root),
            allow_external_root=True,
        )


def test_window_content_drift_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    (env.data_root / "clean" / "daily" / "2026-07-20.parquet").write_bytes(b"drifted")
    with pytest.raises(TestbedError, match="窗口数据态漂移"):
        envmod.register_base_arm(env, root, "aa_1")


def test_watermark_change_outside_window_is_not_drift(tmp_path):
    """窗口外新增分区只改水位，不得把 campaign 冻死（R2-T1-12 反例）。"""
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    envmod.write_partitions(env.data_root, ("20260723",))
    record = envmod.register_base_arm(env, root, "aa_late")
    assert record["identity"]["window_state"]["global_watermark"]["clean/daily"] == "20260723"
    assert (
        record["identity"]["window_state"]["window_state_id"]
        == campaign["identity"]["window_state"]["window_state_id"]
    )


def test_optional_window_dataset_participates_in_identity(tmp_path):
    env = envmod.build_env(tmp_path)
    envmod.write_optional_features(env.data_root, ("20260717",))
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    paths = {item["path"] for item in campaign["identity"]["window_state"]["window_files"]}
    assert any("features/cs_infer" in path for path in paths)
    (env.data_root / "features" / "cs_infer" / "20260717.parquet").write_bytes(b"changed")
    with pytest.raises(TestbedError, match="窗口数据态漂移"):
        envmod.register_base_arm(env, root, "aa_1")


def test_tampered_campaign_record_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    path = root / "registered" / "campaign.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["clock"]["days"] = [payload["clock"]["days"][0], payload["clock"]["days"][2]]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TestbedError, match="封存哈希不一致"):
        load_campaign(root)


def test_moved_campaign_root_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    moved = env.base_root / "t1_moved"
    shutil.move(str(root), str(moved))
    with pytest.raises(TestbedError, match="注册记录与目录不一致"):
        load_campaign(moved)


def test_config_source_root_reuses_p4_parsing(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    source_root = tmp_path / "frozen_paper_root"
    source_root.mkdir()
    (source_root / "config.yaml").write_text(
        "\n".join(f"{key}: {value}" for key, value in env.config.items()) + "\n",
        encoding="utf-8",
    )
    from_source = register_arm(
        campaign_root=root,
        arm_id="from_source",
        arm_role="aa_clone",
        config_source=source_root,
        models_dir=str(env.models_dir),
        repo_root=str(env.repo_root),
        allow_external_root=True,
    )
    from_flat = envmod.register_base_arm(env, root, "from_flat")
    assert from_source["identity_summary"]["config"] == from_flat["identity_summary"]["config"]
    with pytest.raises(TestbedError, match="必须是含 config.yaml 的纸面根"):
        register_arm(
            campaign_root=root,
            arm_id="bad_source",
            arm_role="aa_clone",
            config_source=source_root / "config.yaml",
            models_dir=str(env.models_dir),
            repo_root=str(env.repo_root),
            allow_external_root=True,
        )


def test_model_version_must_be_explicit(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    config = dict(env.config)
    config.pop("model_version")
    with pytest.raises(TestbedError, match="必须显式冻结 model_version"):
        envmod.register_base_arm(env, root, "aa_1", config=config)


def test_model_trained_after_d0_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    envmod.write_models(env.data_root, train_end_date="20260801")
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    with pytest.raises(TestbedError, match="必须严格早于 D0"):
        envmod.register_base_arm(env, root, "aa_1")


def test_model_date_format_cannot_bypass_training_cutoff(tmp_path):
    """分隔符形态的晚于 D0 的日期不得绕过训练截止门禁（复审 R1R-T1-01/R2R-01）。"""
    env = envmod.build_env(tmp_path)
    envmod.write_models(env.data_root, train_end_date="2026/08/01")
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    with pytest.raises(TestbedError, match="必须严格早于 D0"):
        envmod.register_base_arm(env, root, "aa_1")


def test_model_metadata_requires_type_and_dates(tmp_path):
    env = envmod.build_env(tmp_path)
    metadata_path = env.data_root / "models" / "stock_selection" / "v1_metadata.json"
    metadata_path.write_text(
        json.dumps({"train_start_date": "20200101", "train_end_date": "20260710"}),
        encoding="utf-8",
    )
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    with pytest.raises(TestbedError, match="缺 model_type"):
        envmod.register_base_arm(env, root, "aa_1")


def test_race_branch_rejects_concurrent_arm_creation(tmp_path, monkeypatch):
    """预检查被绕过时，独占创建仍是重复注册的唯一裁决点（复审要求①）。"""
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    (root / "registered" / "arms" / "aa_1").mkdir(parents=True)
    monkeypatch.setattr(
        registration_module, "_assert_arm_dir_free", lambda *args, **kwargs: None
    )
    with pytest.raises(DuplicateRegistrationError, match="并发写入被独占创建拒绝"):
        envmod.register_base_arm(env, root, "aa_1")


def test_retired_config_key_does_not_change_execution_fingerprint(tmp_path):
    """退役键被 load_paper_config 剔除，但不得改变有效执行指纹（R2-T1-21 固化）。"""
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    baseline = envmod.register_base_arm(env, root, "aa_1")
    with_retired = dict(env.config)
    with_retired["take_profit_threshold"] = 0.12
    variant = envmod.register_base_arm(env, root, "aa_retired", config=with_retired)
    assert (
        variant["identity_summary"]["config"]
        == baseline["identity_summary"]["config"]
    )
    assert (
        variant["identity"]["config"]["config_sha256"]
        != baseline["identity"]["config"]["config_sha256"]
    )


def test_arm_repo_root_must_match_campaign(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    other_repo = tmp_path / "other_repo"
    other_repo.mkdir()
    with pytest.raises(TestbedError, match="repo_root 与 campaign 不一致"):
        register_arm(
            campaign_root=root,
            arm_id="aa_1",
            arm_role="aa_clone",
            config_flat=env.config,
            models_dir=str(env.models_dir),
            repo_root=str(other_repo),
            allow_external_root=True,
        )


def test_campaign_record_persists_repo_root(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    assert campaign["repo_root"] == str(Path(env.repo_root).resolve()).replace("\\", "/")


def test_config_flat_and_source_are_exclusive(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = envmod.register_base_campaign(env)
    root = Path(campaign["campaign_root"])
    with pytest.raises(TestbedError, match="必须二选一"):
        register_arm(
            campaign_root=root,
            arm_id="aa_1",
            arm_role="aa_clone",
            models_dir=str(env.models_dir),
            repo_root=str(env.repo_root),
            allow_external_root=True,
        )
