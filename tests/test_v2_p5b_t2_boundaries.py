# -*- coding: utf-8 -*-
"""P5b T2 边界与守卫测试（tests/ 全量豁免复杂度）。

覆盖：行域回执不可伪造（manifest 被改成非 structural 即拒）；写域硬禁（生产根/campaign 外）；
campaign 级并发运行锁；单日资格（run_kind 大小写、驱动日期、估值行唯一、导航有限）；
不联网/不补齐（源码级守卫）；不写 LedgerEntry（源码级守卫）；D0 只作估值点。
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as envmod  # noqa: E402

from src.lazybull.v2.hosts.paper.replay import ReplayRowDomainError  # noqa: E402
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore  # noqa: E402
from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    ArmSpec,
    CampaignBusyError,
    CampaignInvalidError,
    CampaignLayout,
    CampaignRunLock,
    DrivenQualificationError,
    FamilyTxnError,
    TestbedError,
    VirtualDayNotReady,
    drive_virtual_day,
    layout_for,
    read_authoritative_valuation,
    run_virtual_campaign,
)
from src.lazybull.v2.hosts.testbed import arm_runtime as arm_runtime_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import family_txn as family_txn_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import journal as journal_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import layout as layout_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import runner as runner_module  # noqa: E402

_T2_MODULES = (
    layout_module,
    journal_module,
    arm_runtime_module,
    family_txn_module,
    runner_module,
)


def _spec(prepared, arm_id: str = "aa_1") -> ArmSpec:
    from src.lazybull.v2.hosts.testbed import load_arm_specs

    return load_arm_specs(prepared.root)[arm_id]


def test_forged_non_structural_partition_is_rejected(tmp_path):
    """把某日 panel 分区的 manifest 行域改成 label_filtered ⇒ 回执必须拒绝。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    day = env.days[0]
    manifest_path = env.data_root / "features" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["partitions"][f"panel/{day}/core.parquet"]["row_domain"] = "label_filtered"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    spec = _spec(prepared)
    host = envmod.make_host_factory(env)(spec, prepared.root / "arms" / "aa_1")
    with pytest.raises(ReplayRowDomainError, match="实际行域"):
        drive_virtual_day(host, spec, day)


def test_pre_structural_date_is_rejected_by_row_domain_receipt(tmp_path):
    """更早历史不得靠改 D0 绕过：label_filtered 段日期在回执处直接拒绝。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    spec = _spec(prepared)
    host = envmod.make_host_factory(env)(spec, prepared.root / "arms" / "aa_1")
    with pytest.raises(ReplayRowDomainError, match="声明行域"):
        drive_virtual_day(host, spec, "20260702")


def test_campaign_run_lock_blocks_concurrent_runner(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    layout = layout_for(campaign)
    with CampaignRunLock(layout):
        with pytest.raises(CampaignBusyError, match="另一运行器|已被另一运行器持有"):
            with CampaignRunLock(layout):
                pass
    with CampaignRunLock(layout):
        pass


def test_run_scope_rejects_production_roots(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    production = ROOT / "data" / "paper"
    with pytest.raises(TestbedError, match="禁止写入生产根"):
        run_virtual_campaign(
            production / "t2_probe",
            days=[env.days[0]],
            host_factory=envmod.make_host_factory(env),
            allow_external_root=True,
        )
    # campaign 根不得落在生产面（即便存在也不得被当作试车场）
    assert not (production / "t2_probe").exists()
    assert prepared.root.is_relative_to(env.base_root)


def test_external_root_requires_explicit_opt_in(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    with pytest.raises(TestbedError, match="越界"):
        run_virtual_campaign(
            prepared.root, days=[env.days[0]], host_factory=envmod.make_host_factory(env)
        )


def test_valuation_must_be_exactly_one_authoritative_row(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    spec = _spec(prepared)
    day = env.days[0]
    envmod.run_campaign(env, prepared, days=[day])
    store = PaperStateStore(root_path=str(prepared.root / "arms" / "aa_1"), create_layout=False)
    with pytest.raises(DrivenQualificationError, match="必 1 条|必须恰 1 条"):
        read_authoritative_valuation(store, "20991231", spec.initial_capital)
    valuation = read_authoritative_valuation(store, day, spec.initial_capital)
    assert valuation["source"] == "engine_state.portfolio_values"
    assert valuation["row_count"] == 1
    assert valuation["nav"] == pytest.approx(1.0)


def test_not_ready_day_raises_dedicated_error(tmp_path):
    broken = envmod.DAYS[2]
    env = envmod.build_env(tmp_path, row_subset={broken: 2})
    prepared = envmod.prepare_campaign(env)
    spec = _spec(prepared)
    host = envmod.make_host_factory(env)(spec, prepared.root / "arms" / "aa_1")
    with pytest.raises(VirtualDayNotReady) as excinfo:
        drive_virtual_day(host, spec, broken)
    assert excinfo.value.arm_id == "aa_1"
    assert excinfo.value.gaps
    # 未就绪不得留下权威代（同日重试从父代重算）
    store = PaperStateStore(root_path=str(prepared.root / "arms" / "aa_1"), create_layout=False)
    assert store.get_authority_last_date() is None


def test_single_day_specs_cannot_bypass_d0(monkeypatch):
    """驱动入口必须是 run_kind='t1'（小写）且日期一致——不允许 t0 冒充成功。"""
    captured = {}

    class _FakeResult:
        requested_date = "20260706"
        corrected_date = "20260706"
        run_kind = "t0"
        driven = True
        t0_status = "success"
        readiness = None

    class _FakeHost:
        last_day_result = _FakeResult()
        store = None
        panel_store = None

        def run_t1(self, date, fills=None):
            captured["date"] = date
            return None

    monkeypatch.setattr(
        arm_runtime_module,
        "assert_single_day_row_domain",
        lambda store, date, expected_domain: {
            "declared_row_domain": expected_domain,
            "actual_row_domains": [expected_domain],
            "files": [],
            "query_start": date,
            "query_end": date,
            "passed": True,
        },
    )
    spec = ArmSpec(
        arm_id="probe",
        arm_role="aa_clone",
        root=Path("."),
        config_flat={},
        config_fingerprint="x",
        model_version=1,
        models_dir=".",
        model_content_fingerprint="y",
        train_end_date="20260101",
        data_root=".",
        initial_capital=1.0,
    )
    with pytest.raises(DrivenQualificationError, match="t1"):
        drive_virtual_day(_FakeHost(), spec, "20260706")
    assert captured["date"] == "20260706"


def test_t2_modules_have_no_network_or_backfill_entrypoints():
    """源码级守卫：试车场 T2 模块不 import 联网/补齐入口，也不构造 LedgerEntry。"""
    forbidden_imports = (
        "tushare",
        "akshare",
        "efinance",
        "requests",
        "urllib",
        "features.ensure",
        "data.ensure",
        "append_ledger_entry",
        "LedgerEntry(",
    )
    for module in _T2_MODULES:
        source = inspect.getsource(module)
        for token in forbidden_imports:
            assert token not in source, f"{module.__name__} 命中禁用依赖: {token}"


def test_layout_paths_stay_inside_campaign_root(tmp_path):
    campaign = tmp_path / "campaign"
    layout = CampaignLayout(campaign)
    sample = [
        layout.arm_root("aa_1"),
        layout.staged_arm_root("20260706", "att-1", "aa_1"),
        layout.backup_arm_root("20260706", "att-1", "aa_1"),
        layout.failed_arm_root("20260706", "att-1", "aa_1"),
        layout.shard_path("aa_1", "20260706"),
        layout.manifest_path("aa_1"),
        layout.journal_path("20260706"),
        layout.index_path,
    ]
    for path in sample:
        assert path.resolve().is_relative_to(layout.root)
    assert layout.root == campaign.resolve()


def test_paper_state_roots_are_per_arm_and_isolated(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    envmod.run_campaign(env, prepared, days=[env.days[0]])
    roots = {arm: prepared.root / "arms" / arm for arm in ("aa_1", "aa_2")}
    assert roots["aa_1"] != roots["aa_2"]
    stores = {
        arm: PaperStateStore(root_path=str(root), create_layout=False)
        for arm, root in roots.items()
    }
    for store in stores.values():
        assert store.get_authority_last_date() == env.days[0]
    # 两臂独立：改一臂的投影文件不影响另一臂
    account = roots["aa_1"] / "state" / "account.json"
    payload = json.loads(account.read_text(encoding="utf-8"))
    payload["cash"] = -1.0
    account.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert stores["aa_2"].load_account()["cash"] != -1.0


def test_engine_state_and_nav_are_consistent_but_nav_parquet_is_sparse(tmp_path):
    """权威估值条数 = 处理日数；nav.parquet 只含有成交日（稀疏），不得当权威。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    envmod.run_campaign(env, prepared, days=env.days[:4])
    root = prepared.root / "arms" / "aa_1"
    store = PaperStateStore(root_path=str(root), create_layout=False)
    state = store.load_engine_state()
    assert len(state.portfolio_values) == 4
    nav_frame = store.load_all_nav()
    sparse_days = set() if nav_frame is None else set(nav_frame["trade_date"].astype(str))
    fills_days = {
        day for day in env.days[:4] if envmod.shard(prepared, "aa_1", day)["evidence"]["fills"]
    }
    assert sparse_days == fills_days
    assert len(sparse_days) < 4


def test_campaign_lock_is_released_after_run(tmp_path):
    """运行结束后锁必须**已释放**（OS 锁文件保留，但可被再次获取）。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    envmod.run_campaign(env, prepared, days=[env.days[0]])
    layout = layout_for(prepared.root)
    with runner_module.CampaignRunLock(layout, timeout_seconds=5.0):
        pass  # 能再次取得锁即证明已释放


def test_missing_clean_partition_is_detected_before_run(tmp_path):
    """窗口内缺必需清洗分区 ⇒ 数据态/漂移门拒绝（不静默跳日）。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    target = (
        env.data_root
        / "clean"
        / "daily"
        / f"{env.days[5][:4]}-{env.days[5][4:6]}-{env.days[5][6:]}.parquet"
    )
    target.unlink()
    with pytest.raises((CampaignInvalidError, FileNotFoundError), match="缺必需清洗分区|漂移"):
        envmod.run_campaign(env, prepared, days=[env.days[0]])


def test_pending_queue_projection_matches_engine_state(tmp_path):
    """pending 队列投影随代提升迁移：文件内容与引擎快照的 pending_order_manager 一致。"""
    blocked_day = envmod.DAYS[6]
    env = envmod.build_env(tmp_path, limit_down={blocked_day: [envmod.CODES[0]]})
    prepared = envmod.prepare_campaign(env)
    envmod.run_campaign(env, prepared, days=env.days[:7])
    queue = envmod.pending(prepared, "aa_1", "sells")
    store = PaperStateStore(root_path=str(prepared.root / "arms" / "aa_1"), create_layout=False)
    from src.lazybull.v2.core.accounting.engine_state import loads_blob

    fields = loads_blob(store.load_engine_state_blob())["fields"]
    manager = fields.get("pending_order_manager") or {}
    engine_sells = set()
    for order in (manager.get("pending_orders") or {}).values():
        stock = getattr(order, "stock", None) or (
            order.get("stock") if isinstance(order, dict) else None
        )
        action = getattr(order, "action", None) or (
            order.get("action") if isinstance(order, dict) else None
        )
        if str(action) == "sell":
            engine_sells.add(str(stock))
    assert {row["ts_code"] for row in queue} == engine_sells
    assert engine_sells == {envmod.CODES[0]}


def test_family_txn_requires_genesis_index(tmp_path):
    from src.lazybull.v2.hosts.testbed import load_arm_specs, open_family_transaction

    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    layout = layout_for(prepared.root)
    specs = load_arm_specs(prepared.root)
    with pytest.raises(FamilyTxnError, match="缺已发布 index"):
        open_family_transaction(layout, "t2_campaign", specs, env.days[0])


def test_second_runner_concurrency_guard(tmp_path, monkeypatch):
    """运行中再次进入 ⇒ CampaignBusyError（不在同一 campaign 上并发写）。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    lock = CampaignRunLock(layout_for(prepared.root))
    with lock:
        with pytest.raises(CampaignBusyError):
            envmod.run_campaign(env, prepared, days=[env.days[0]])


def test_index_generation_is_monotonic(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    envmod.run_campaign(env, prepared, days=env.days[:3])
    first = envmod.index(prepared)["published_generation"]
    envmod.run_campaign(env, prepared, days=env.days[3:5])
    second = envmod.index(prepared)["published_generation"]
    assert second == first + 2
    days = [entry["generation"] for entry in envmod.index(prepared)["days"]]
    assert days == sorted(days) == [1, 2, 3, 4, 5]


def test_non_finite_or_non_json_value_is_rejected_before_write():
    """派生投影可重建的前提 = 严格 JSON（NaN/Inf 与非原生类型不得被静默字符串化）。"""
    from src.lazybull.v2.hosts.testbed.journal import assert_json_safe

    assert_json_safe({"ok": 1.5, "text": "中文"}, what="probe")
    with pytest.raises(FamilyTxnError, match="非 JSON 安全值"):
        assert_json_safe({"nav": float("nan")}, what="probe")
    with pytest.raises(FamilyTxnError, match="非 JSON 安全值"):
        assert_json_safe({"nav": float("inf")}, what="probe")
    with pytest.raises(FamilyTxnError, match="非 JSON 安全值"):
        assert_json_safe({"obj": object()}, what="probe")


def test_testbed_root_is_never_created_under_project_data(tmp_path):
    """全部 T2 测试只写 tmp_path；项目 data/paper_shadow/p5b_testbed 不得被创建。"""
    from src.lazybull.v2.hosts.testbed import testbed_root

    target = testbed_root()
    before = target.exists()
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    envmod.run_campaign(env, prepared, days=[env.days[0]], write_report=True)
    assert target.exists() == before
    assert prepared.root.is_relative_to(tmp_path)


def test_report_declares_limitations_and_no_sla_claims(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    report = envmod.run_campaign(env, prepared, days=[env.days[0]], write_report=True)
    assert report["not_sla_evidence"] is True
    assert report["not_real_candidate"] is True
    assert report["evidence_class"] == "calibration_only"
    text = "\n".join(report["limitations"])
    for token in ("NOT_FORWARD_EVIDENCE", "T3", "T4/T5", "A/A", "标签截止", "次日指令帧"):
        assert token in text, f"报告限制声明缺 {token}"
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["schema"] == runner_module.REPORT_SCHEMA
