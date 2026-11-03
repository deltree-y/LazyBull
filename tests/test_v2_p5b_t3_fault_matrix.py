# -*- coding: utf-8 -*-
"""P5b T3 正式故障注入矩阵测试（tests/ 全量豁免复杂度）。

覆盖设计 v4 §4.3「必须注入的故障点」全部九项 + 0 roots 行 + INVALIDATED 失效项 + 生产零写入：
逐项断言终态、恢复后置条件（**由落盘证据推导**，不是执行侧回填）、代增量与五条不变量，
并导出判定表与 containment 表；另含五条不变量的**负例**（证明每项都会失败，非恒真）。
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

import p5b_t3_env as envmod  # noqa: E402

from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    CONTAINMENT,
    INVARIANTS,
    MATRIX,
    POSTCONDITIONS,
    build_verdict_table,
    check_invariants,
    evaluate_case,
    layout_for,
    matrix_fingerprint,
    snapshot_campaign,
    table_markdown,
)
from src.lazybull.v2.hosts.testbed import family_txn as family_txn_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import fault_matrix as fault_matrix_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import journal as journal_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import ledger_audit as ledger_audit_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import ledger_common as ledger_common_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import ledger_repair as ledger_repair_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import shard_chain as shard_chain_module  # noqa: E402

_T3_MODULES = (
    shard_chain_module,
    ledger_common_module,
    ledger_audit_module,
    ledger_repair_module,
    fault_matrix_module,
)
#: 设计 §4.3 必须注入的故障点（与矩阵行一一对应）
REQUIRED_FAULT_POINTS = {
    "after_stage": "0 个 root 提升即中断（状态表第 1 行）",
    "after_promote": "第 1/第 2 个 root 提升后崩溃、部分 roots 提升",
    "after_promote_all": "全部 roots 提升但分片未写",
    "after_shard": "只写部分臂分片",
    "after_shards": "分片已写、manifest 未发布",
    "after_manifests": "manifest 已发布、index 未写",
    "after_abort_restore": "回滚/恢复过程中再次崩溃",
    "virtual_date_regression": "虚拟日期回退",
    "data_fingerprint_drift": "数据指纹漂移",
    "production_zero_write": "生产根/数据资产零写入",
}


@pytest.fixture(scope="module")
def matrix_run(tmp_path_factory):
    """跑完整矩阵一次（每行独立 campaign），失败信息按行聚合而不中断其余行。"""
    records = {}
    failures = {}
    for case in MATRIX:
        tmp = tmp_path_factory.mktemp(case.case_id.lower().replace("-", "_"))
        try:
            env = envmod.build_env(tmp)
            prepared = envmod.prepare(env)
            result = envmod.run_matrix_case(env, prepared, case)
            records[case.case_id] = result["record"]
        except Exception as exc:  # noqa: BLE001 - 矩阵需要逐行隔离失败
            failures[case.case_id] = f"{type(exc).__name__}: {exc}"
    return {"records": records, "failures": failures}


@pytest.mark.parametrize("case", MATRIX, ids=[case.case_id for case in MATRIX])
def test_matrix_case_terminal_postconditions_and_five_invariants(matrix_run, case):
    assert case.case_id not in matrix_run["failures"], matrix_run["failures"].get(case.case_id)
    record = matrix_run["records"][case.case_id]
    assert record["terminal"] == case.expected_terminal
    assert record["generation_delta"] == case.expected_generation_delta
    assert record["isolation"] == case.isolation
    assert record["error_type_matched"] is True
    # 恢复动作列必须由证据推导（观测集合 == 期望集合，而不是回填字符串）
    assert record["observed_postconditions"] == sorted(case.expected_postconditions)
    assert record["missing_postconditions"] == []
    assert record["unexpected_postconditions"] == []
    assert record["recovery"] == case.expected_recovery
    assert record["missing_expected_gaps"] == []
    for name in INVARIANTS:
        assert record["invariants"][name] is True, f"{case.case_id} 不变量 {name} 失败: " + str(
            record["details"][name]
        )
    assert record["passed"] is True


def test_matrix_covers_every_required_fault_point_and_state_row():
    covered = {case.fault_point for case in MATRIX}
    assert REQUIRED_FAULT_POINTS.keys() <= covered
    for point in REQUIRED_FAULT_POINTS:
        assert any(case.fault_point == point for case in MATRIX), point
    # §4.3 状态表：0 / 部分 / 全部 roots × 未发布 均有确定动作
    assert any(case.case_id == "T3-FM-00" for case in MATRIX)  # 0 roots
    second = next(case for case in MATRIX if case.case_id == "T3-FM-02")
    assert second.arm_id == "aa_2" and second.min_promoted == 2  # 部分（三臂中 2 已提升）
    assert any(case.case_id == "T3-FM-03" for case in MATRIX)  # 全部 roots
    # 已发布侧：COMMITTED 行不重跑，只补齐
    published = [case for case in MATRIX if case.expected_terminal == "COMMITTED"]
    assert {case.case_id for case in published} == {"T3-FM-06", "T3-FM-07", "T3-FM-12"}


def test_verdict_table_all_passed_and_has_containment(matrix_run):
    assert matrix_run["failures"] == {}
    table = build_verdict_table(list(matrix_run["records"].values()), campaign_id="t3_campaign")
    assert table["schema"] == "p5b_testbed_fault_matrix/v1"
    assert table["evidence_class"] == "calibration_only"
    assert table["not_sla_evidence"] is True
    assert table["case_count"] == len(MATRIX)
    assert table["passed_count"] == len(MATRIX)
    assert table["all_passed"] is True
    assert table["invariants"] == list(INVARIANTS)
    assert table["postconditions"] == list(POSTCONDITIONS)
    assert table["matrix_fingerprint"] == matrix_fingerprint()
    assert table["matrix_case_ids"] == [case.case_id for case in MATRIX]
    for row in table["cases"]:
        assert set(row["invariants"]) == set(INVARIANTS)
        assert set(row["postconditions"]) == set(POSTCONDITIONS)
        assert row["terminal"] in {"ABORTED", "COMMITTED", "INVALIDATED", "REJECTED"}
    levels = {entry["level"] for entry in table["containment"]}
    assert levels == {"attempt", "arm", "campaign", "data_state"}
    for entry in table["containment"]:
        assert entry["no_sla_claim"] is True
        assert entry["deterministic_action"]
        assert entry["escalation"] or entry["level"] == "data_state"
    text = "\n".join(table["limitations"])
    assert "NOT_FORWARD_EVIDENCE" in text and "SLA" in text


def test_verdict_table_markdown_columns(matrix_run):
    markdown = table_markdown(list(matrix_run["records"].values()))
    header = markdown.splitlines()[0]
    assert "注入点" in header and "终态（期望/实际）" in header and "判定" in header
    assert "恢复后置条件（实测）" in header
    for name in INVARIANTS:
        assert name in header
    assert markdown.count("\n") == len(MATRIX) + 1
    assert "失败" not in markdown


def test_containment_has_no_sla_or_recovery_time_claims():
    for level in CONTAINMENT:
        payload = json.dumps(level.to_dict(), ensure_ascii=False)
        for banned in ("SLA", "时限", "可用性承诺", "小时内", "RTO", "RPO"):
            assert banned not in payload.replace("no_sla_claim", "")
        assert level.terminal_state
    # arm 级升级阈值与代码一致：任一臂不可还原即 campaign 失效
    arm_level = next(level for level in CONTAINMENT if level.level == "arm")
    assert "任一臂" in arm_level.escalation


def test_production_roots_receive_zero_writes(tmp_path):
    """递归快照：data/paper 与 data/runs 零写入；data/models 元数据零变化。"""
    before = envmod.production_snapshot()
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env)
    envmod.run(env, prepared, env.days[:2])
    with pytest.raises(Exception):
        envmod.run(
            env,
            prepared,
            [env.days[2]],
            fault_hook=envmod.fault_hook("after_promote", arm_id="aa_1"),
        )
    envmod.recover(prepared)
    after = envmod.production_snapshot()
    assert envmod.production_unchanged(before, after)
    assert prepared.root.is_relative_to(tmp_path)


# ---------------------------------------------------------------------------
# 五条不变量的负例（证明每项都会失败；防止「恒真断言」）
# ---------------------------------------------------------------------------


def _case(case_id: str = "T3-FM-01"):
    return next(case for case in MATRIX if case.case_id == case_id)


def _live_snapshot(tmp_path: Path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    return env, prepared, layout, snapshot_campaign(layout, envmod.CAMPAIGN_ID)


def test_invariant_no_double_booking_detects_duplicate_generation(tmp_path):
    _env, _prepared, layout, after = _live_snapshot(tmp_path)
    broken = json.loads(json.dumps(after))
    broken["index_days"][1]["generation"] = broken["index_days"][0]["generation"]
    broken["published_generation"] = 1
    checks = check_invariants(layout, envmod.CAMPAIGN_ID, _case(), after, broken)
    assert checks["no_double_booking"][0] is False
    assert any("代序" in problem for problem in checks["no_double_booking"][1]["problems"])


def test_invariant_no_double_booking_detects_rewritten_history(tmp_path):
    _env, _prepared, layout, after = _live_snapshot(tmp_path)
    broken = json.loads(json.dumps(after))
    arm = sorted(broken["shards"])[0]
    date = sorted(broken["shards"][arm])[0]
    broken["shards"][arm][date] = "sha-rewritten"
    checks = check_invariants(layout, envmod.CAMPAIGN_ID, _case(), after, broken)
    assert checks["no_double_booking"][0] is False


def test_invariant_unpublished_detects_stray_shard(tmp_path):
    _env, _prepared, layout, after = _live_snapshot(tmp_path)
    broken = json.loads(json.dumps(after))
    broken["shards"]["aa_1"]["20260731"] = "sha-orphan"
    checks = check_invariants(layout, envmod.CAMPAIGN_ID, _case(), after, broken)
    assert checks["unpublished_not_in_generation"][0] is False
    assert checks["unpublished_not_in_generation"][1]["strays"] == 1


def test_invariant_determinism_detects_tampered_shard(tmp_path):
    _env, _prepared, layout, after = _live_snapshot(tmp_path)
    broken = json.loads(json.dumps(after))
    arm = sorted(broken["shards"])[0]
    date = sorted(broken["shards"][arm])[0]
    broken["shards"][arm][date] = "sha-forged"
    checks = check_invariants(layout, envmod.CAMPAIGN_ID, _case(), after, broken)
    assert checks["deterministic_recompute"][0] is False


def test_invariant_read_side_detects_generation_ahead(tmp_path):
    _env, _prepared, layout, after = _live_snapshot(tmp_path)
    broken = json.loads(json.dumps(after))
    broken["published_generation"] = 99
    checks = check_invariants(layout, envmod.CAMPAIGN_ID, _case(), after, broken)
    assert checks["read_side_published_only"][0] is False


def test_invariant_backup_detects_tampered_parent(tmp_path):
    _env, prepared, layout, after = _live_snapshot(tmp_path)
    attempt = next(item for item in after["attempts"] if item["status"] == "COMMITTED")
    backup = layout.backup_arm_root(
        str(attempt["virtual_date"]), str(attempt["attempt_id"]), "aa_1"
    )
    target = sorted(backup.rglob("*"))[0]
    while target.is_dir():
        target = sorted(target.rglob("*"))[0]
    target.write_text("tampered", encoding="utf-8")
    checks = check_invariants(layout, envmod.CAMPAIGN_ID, _case(), after, after)
    assert checks["parent_backup_verifiable"][0] is False


def test_invariants_are_not_vacuous_on_empty_snapshot(tmp_path):
    """空集合不得恒真：无已发布日/无 attempt/无备份 ⇒ 每项都判失败。"""
    _env, _prepared, layout, _after = _live_snapshot(tmp_path)
    empty = {
        "index_days": [],
        "published_generation": 0,
        "attempts": [],
        "shards": {},
        "arms": [],
        "invalidated": [],
        "gaps_by_class": {},
        "three_way_consistent": False,
    }
    checks = check_invariants(layout, envmod.CAMPAIGN_ID, _case(), empty, empty)
    for name in INVARIANTS:
        assert checks[name][0] is False, f"{name} 在空快照上恒真（不允许）"


def test_wrong_error_type_fails_the_case(tmp_path):
    """异常白名单必须真的生效：错误类名 ⇒ `error_type_matched=False` ⇒ 整行判失败。"""
    env, prepared, layout, before = _live_snapshot(tmp_path)
    case = next(item for item in MATRIX if item.case_id == "T3-FM-09")
    record = evaluate_case(
        layout,
        envmod.CAMPAIGN_ID,
        case,
        before,
        target_date=env.days[2],
        error_type="UnexpectedBoom",
        detected_gaps=(),
        injected_attempt_id="",
        measurements={},
    )
    assert record["error_type_matched"] is False
    assert record["passed"] is False


def test_missing_postcondition_fails_the_case(tmp_path):
    """未真正发生故障的现场不能「通过」故障行（缺后置条件 ⇒ 判失败，不是包含关系）。"""
    env, prepared, layout, before = _live_snapshot(tmp_path)
    case = next(item for item in MATRIX if item.case_id == "T3-FM-01")
    record = evaluate_case(
        layout,
        envmod.CAMPAIGN_ID,
        case,
        before,
        target_date=env.days[1],
        error_type="",
        detected_gaps=(),
        injected_attempt_id="",
        measurements={},
    )
    assert record["missing_postconditions"]
    assert record["observed_postconditions"] != record["expected_postconditions"]
    assert record["passed"] is False


def test_t3_modules_have_no_ledger_or_network_entrypoints():
    """源码级守卫：T3 模块不写 LedgerEntry、不联网、不补齐、不写持仓。"""
    forbidden = (
        "tushare",
        "akshare",
        "efinance",
        "requests",
        "urllib",
        "features.ensure",
        "data.ensure",
        "append_ledger_entry",
        "LedgerEntry(",
        "pending_buys",
        "pending_sells",
    )
    for module in _T3_MODULES:
        source = inspect.getsource(module)
        for token in forbidden:
            assert token not in source, f"{module.__name__} 命中禁用依赖: {token}"


def test_fault_points_tuple_documents_t3_injection_points():
    points = family_txn_module.FAULT_POINTS
    for required in ("after_promote_all", "after_shard", "after_manifests"):
        assert required in points
    for legacy in (
        "after_backup",
        "after_stage",
        "after_drive",
        "before_promote",
        "mid_promote",
        "after_promote",
        "after_shards",
        "after_publish",
        "abort_begin",
        "after_abort_restore",
        "abort_done",
    ):
        assert legacy in points
    # 恢复期幂等补齐不挂故障钩子（避免 after_manifests 跨日误伤）
    source = inspect.getsource(family_txn_module.top_up_if_needed)
    assert "top_up_published(layout, campaign_id, attempt)" in source
    assert "fault_hook" not in source
    # 暂存阶段必须先把全部臂的父代备份做完（否则 0 roots 行会因后续臂备份缺失而升级失效）
    order = inspect.getsource(family_txn_module.FamilyTransaction.open)
    assert order.index("self._make_backup(arm_id)") < order.index("self._make_staged_root(arm_id)")


def test_journal_and_shard_schema_unchanged_by_t3(tmp_path):
    """T3 不得改变已发布分片的字节/schema：分片 schema 仍是 T2 值。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env)
    envmod.run(env, prepared, [env.days[0]])
    shard = json.loads(envmod.shard_path(prepared, "aa_1", env.days[0]).read_text("utf-8"))
    assert shard["schema"] == journal_module.SHARD_SCHEMA == "p5b_testbed_ledger_shard/v1"
    assert set(shard) == {
        "schema",
        "campaign_id",
        "scope",
        "evidence_class",
        "virtual_clock",
        "not_forward_evidence",
        "not_sla_evidence",
        "not_real_candidate",
        "arm_id",
        "arm_role",
        "virtual_date",
        "generation",
        "gen_id",
        "family_txn_id",
        "parent_generation",
        "evidence",
        "root_identity",
        "record_sha256",
    }
