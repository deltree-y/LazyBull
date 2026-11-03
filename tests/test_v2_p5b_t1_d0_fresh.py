# -*- coding: utf-8 -*-
"""P5b T1 D0 fresh 契约测试（tests/ 全量豁免复杂度）。

覆盖：各臂同一历史 D0 空现金起点（盘上观测）、不复制生产状态、D0 只估值 / D1 才有第一个
相邻收益、重复初始化拒绝、半初始化拒绝、中段失败回滚、复核只读、D0 记录篡改检出、
窗口数据态漂移拒绝初始化。
"""

from __future__ import annotations

import json
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
    D0FreshError,
    DuplicateRegistrationError,
    TestbedError,
    build_virtual_clock,
    empty_account_digest,
    initialize_d0_fresh,
    register_campaign,
    verify_d0_fresh,
)
from src.lazybull.v2.hosts.testbed import d0 as d0_module  # noqa: E402
from src.lazybull.v2.hosts.testbed.storage import seal_record  # noqa: E402


def _tree(root: Path) -> set:
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def test_all_arms_start_from_identical_observed_empty_cash(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    record = initialize_d0_fresh(prepared.root, allow_external_root=True)
    assert record["schema"] == "p5b_testbed_d0_fresh/v1"
    assert record["source"] == "fresh_cash"
    assert record["production_state_copied"] is False
    assert record["evidence_class"] == "calibration_only"
    assert record["record_sha256"]
    arms = record["arms"]
    assert set(arms) == {"aa_1", "aa_2"}
    digests = {arm_id: payload["empty_state_sha256"] for arm_id, payload in arms.items()}
    assert len(set(digests.values())) == 1
    # 观测摘要必须与期望空账户摘要一致（形态对齐，非自证）
    assert digests["aa_1"] == empty_account_digest(float(env.config["initial_capital"]))
    for arm_id, payload in arms.items():
        observed = payload["empty_state"]
        assert observed["positions"] == {}
        assert observed["pending_buys"] == [] and observed["pending_sells"] == []
        assert observed["instructions"] == []
        assert observed["nav_rows"] == 0 and observed["trade_rows"] == 0
        assert observed["coverage_history"] is False
        assert observed["state_files"] == []
        assert observed["store_consistency"] == "fresh"
        assert observed["cash_basis"] == "config_initial_capital"
        assert payload["initial_capital"] == float(env.config["initial_capital"])
        arm_root = prepared.root / payload["root"]
        assert (arm_root / "config.yaml").is_file()
        empty_dirs = (
            "state",
            "trades",
            "nav",
            "runs",
            "pending_buys",
            "pending_sells",
            "instructions",
        )
        for name in empty_dirs:
            assert list((arm_root / name).iterdir()) == []


def test_d0_is_valuation_only_and_first_return_is_d1(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    record = initialize_d0_fresh(prepared.root, allow_external_root=True)
    assert record["d0_role"] == "valuation_only"
    assert record["d0_date"] == env.days[0]
    assert record["first_return_date"] == env.days[1]
    clock = record["clock"]
    assert clock["d0_date"] == env.days[0]
    assert env.days[0] not in clock["return_days"]
    assert clock["return_days"] == list(env.days[1:])
    assert clock["adjacent_return_pairs"][0] == [env.days[0], env.days[1]]


def test_single_day_window_has_no_return_sample(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = register_campaign(
        campaign_id="t1_single_day",
        d0_date=env.days[0],
        data_root=str(env.data_root),
        base_root=str(env.base_root),
        repo_root=str(env.repo_root),
        allow_external_root=True,
    )
    assert campaign["clock"]["first_return_date"] is None
    assert campaign["clock"]["n_return_days"] == 0
    assert campaign["clock"]["contiguity"]["basis"] == "vacuous_single_day"
    clock = build_virtual_clock(d0_date=campaign["clock"]["d0_date"])
    assert clock.adjacent_return_pairs() == ()


def test_duplicate_d0_initialization_rejected(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    with pytest.raises(DuplicateRegistrationError):
        initialize_d0_fresh(prepared.root, allow_external_root=True)


def test_d0_requires_all_registered_arms(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    with pytest.raises(D0FreshError, match="禁止半初始化"):
        initialize_d0_fresh(prepared.root, arm_ids=["aa_1"], allow_external_root=True)
    assert not (prepared.root / "arms" / "aa_1").exists()
    assert not (prepared.root / "registered" / "d0_fresh.json").exists()


def test_d0_rejects_preexisting_arm_root_without_partial_state(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    (prepared.root / "arms" / "aa_2").mkdir(parents=True)
    with pytest.raises(D0FreshError, match="全新臂根"):
        initialize_d0_fresh(prepared.root, allow_external_root=True)
    assert not (prepared.root / "arms" / "aa_1").exists()
    assert not (prepared.root / "registered" / "d0_fresh.json").exists()


def test_midway_failure_rolls_back_all_arm_roots(tmp_path, monkeypatch):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    original = d0_module._materialize_config
    calls = {"count": 0}

    def fail_on_second(arm_root, frozen):
        calls["count"] += 1
        if calls["count"] == 2:
            raise TestbedError("注入：第二臂配置落盘失败")
        return original(arm_root, frozen)

    monkeypatch.setattr(d0_module, "_materialize_config", fail_on_second)
    with pytest.raises(TestbedError, match="注入"):
        initialize_d0_fresh(prepared.root, allow_external_root=True)
    assert not (prepared.root / "arms").exists()
    assert not (prepared.root / "registered" / "d0_fresh.json").exists()


def test_missing_frozen_config_blocks_d0_with_context(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    (prepared.root / "registered" / "arms" / "aa_1" / "config.json").unlink()
    with pytest.raises(TestbedError, match="冻结配置缺失"):
        initialize_d0_fresh(prepared.root, allow_external_root=True)
    assert not (prepared.root / "arms" / "aa_1").exists()


def test_arm_registration_closed_after_d0(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    with pytest.raises(DuplicateRegistrationError, match="禁止事后新增臂"):
        envmod.register_base_arm(env, prepared.root, "late_arm")


def test_data_drift_after_campaign_blocks_d0(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    (env.data_root / "clean" / "daily" / "2026-07-21.parquet").write_bytes(b"drift")
    with pytest.raises(D0FreshError, match="窗口数据态漂移"):
        initialize_d0_fresh(prepared.root, allow_external_root=True)
    assert not (prepared.root / "arms" / "aa_1").exists()


def test_verify_d0_fresh_is_readonly_and_reports_arms(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    record = initialize_d0_fresh(prepared.root, allow_external_root=True)
    before = _tree(prepared.root)
    verified = verify_d0_fresh(prepared.root)
    assert _tree(prepared.root) == before
    assert verified["campaign_id"] == "t1_campaign"
    assert verified["d0_date"] == env.days[0]
    assert verified["first_return_date"] == env.days[1]
    assert verified["arms"] == {"aa_1": "fresh", "aa_2": "fresh"}
    assert verified["record_sha256"] == record["record_sha256"]
    assert verified["initial_capital"] == float(env.config["initial_capital"])


def test_verify_d0_fresh_detects_config_tampering(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    config_path = prepared.root / "arms" / "aa_1" / "config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("top_n: 5", "top_n: 9"),
        encoding="utf-8",
    )
    with pytest.raises(D0FreshError, match="配置指纹"):
        verify_d0_fresh(prepared.root)


def test_verify_d0_fresh_detects_record_tampering(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    path = prepared.root / "registered" / "d0_fresh.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["initial_capital"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TestbedError, match="封存哈希不一致"):
        verify_d0_fresh(prepared.root)


def test_verify_d0_fresh_rejects_root_traversal(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    path = prepared.root / "registered" / "d0_fresh.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["arms"]["aa_1"]["root"] = "../../data/paper"
    path.write_text(json.dumps(seal_record(payload)), encoding="utf-8")
    with pytest.raises(D0FreshError, match="root 非法"):
        verify_d0_fresh(prepared.root)


def test_verify_d0_fresh_detects_arm_set_mismatch(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    path = prepared.root / "registered" / "d0_fresh.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["arms"].pop("aa_2")
    path.write_text(json.dumps(seal_record(payload)), encoding="utf-8")
    with pytest.raises(D0FreshError, match="臂集合与已注册臂不一致"):
        verify_d0_fresh(prepared.root)


def test_verify_d0_fresh_detects_model_content_drift(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    (env.models_dir / "v1_model.joblib").write_bytes(b"tampered")
    with pytest.raises(D0FreshError, match="模型产物内容已漂移"):
        verify_d0_fresh(prepared.root)


def test_verify_d0_fresh_detects_deleted_arm_root(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    import shutil

    shutil.rmtree(prepared.root / "arms" / "aa_1")
    with pytest.raises(D0FreshError, match="臂根不存在"):
        verify_d0_fresh(prepared.root)


def test_verify_d0_fresh_detects_resealed_capital_mismatch(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    path = prepared.root / "registered" / "d0_fresh.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["initial_capital"] = 1.0
    path.write_text(json.dumps(seal_record(payload)), encoding="utf-8")
    with pytest.raises(D0FreshError, match="顶层 initial_capital"):
        verify_d0_fresh(prepared.root)


def test_verify_d0_fresh_detects_arm_registration_identity_swap(tmp_path):
    """重封存后替换臂注册身份（config/model）仍必须被 D0 复核拦住（N1 反例）。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    path = prepared.root / "registered" / "arms" / "aa_1" / "registration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["identity"]["config"]["fingerprint"] = "0" * 16
    path.write_text(json.dumps(seal_record(payload)), encoding="utf-8")
    with pytest.raises(D0FreshError, match="注册配置身份与 D0 记录不一致"):
        verify_d0_fresh(prepared.root)


def test_verify_d0_fresh_requires_record(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    with pytest.raises(TestbedError, match="D0 fresh 记录缺失"):
        verify_d0_fresh(prepared.root)


def test_virtual_clock_rejects_skipped_day(tmp_path):
    env = envmod.build_env(tmp_path)
    with pytest.raises(TestbedError, match="禁止跳日"):
        register_campaign(
            campaign_id="t1_campaign",
            d0_date="20260717",
            virtual_days=["20260717", "20260721"],
            calendar=env.calendar,
            data_root=str(env.data_root),
            base_root=str(env.base_root),
            repo_root=str(env.repo_root),
            allow_external_root=True,
        )
    contiguous = register_campaign(
        campaign_id="t1_contiguous",
        d0_date="20260717",
        virtual_days=["20260717", "20260720", "20260721"],
        calendar=env.calendar,
        data_root=str(env.data_root),
        base_root=str(env.base_root),
        repo_root=str(env.repo_root),
        allow_external_root=True,
    )
    assert contiguous["clock"]["days"] == list(env.days)


def test_virtual_clock_rejects_backwards_duplicate_and_invalid_dates(tmp_path):
    with pytest.raises(TestbedError, match="严格升序"):
        build_virtual_clock(d0_date="20260720", virtual_days=["20260720", "20260717"])
    with pytest.raises(TestbedError, match="重复日期"):
        build_virtual_clock(d0_date="20260717", virtual_days=["20260717", "20260717"])
    with pytest.raises(TestbedError, match="不是合法日期"):
        build_virtual_clock(d0_date="20261332")


def test_missing_partition_fails_closed(tmp_path):
    env = envmod.build_env(tmp_path)
    (env.data_root / "clean" / "daily" / "2026-07-21.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="缺必需清洗分区"):
        envmod.register_base_campaign(env)


def test_d0_record_carries_declaration_flags_only(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_two_arms(env)
    initialize_d0_fresh(prepared.root, allow_external_root=True)
    record = json.loads(
        (prepared.root / "registered" / "d0_fresh.json").read_text(encoding="utf-8")
    )
    assert record["not_forward_evidence"] is True
    assert record["not_sla_evidence"] is True
    assert record["not_real_candidate"] is True
    assert record["scope"] == "historical_testbed"
    assert record["row_domain"] == "structural"
    assert record["row_domain_verified"] is False
    assert record["row_domain_receipt"] is None
    assert set(record["evidence_notes"]) == {"empty_state", "all_arms_identical"}
    assert record["campaign_identity_summary"]["window_state"]
