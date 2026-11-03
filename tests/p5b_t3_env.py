# -*- coding: utf-8 -*-
"""P5b T3 测试环境（tests/ 全量豁免复杂度；不是测试用例模块）。

在 T2 合成环境（``p5b_t2_env``）之上只做**增量**：

- 三臂 campaign（矩阵需要「第 2 个 root 提升后崩溃」与「部分 roots 未提升」）；
- 故障注入钩子（按注入点 + 臂 + 已提升数过滤）；
- 矩阵单例驱动器（注入 → 启动恢复 → 缺口修复 → 同日重算 → 判定表记录）；
- 只读取证助手（现场树快照、生产根快照、分片篡改、链/审计产物读取）；
- 长窗口（≥40 连续虚拟日）规模环境。

驱动器只上报**原始测量**（实际抛出的异常类名、生产根快照是否变化、被注入 attempt id），
期望值一律留在 ``fault_matrix.MATRIX``，由 ``evaluate_case`` 独立比对。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as base  # noqa: E402

from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    FaultCase,
    InjectedFault,
    LedgerInvalidatedError,
    audit_ledger,
    evaluate_case,
    first_attempt_id,
    layout_for,
    published_days,
    query_shards,
    read_index,
    recover_campaign,
    repair_ledger,
    snapshot_campaign,
)
from src.lazybull.v2.hosts.testbed.ledger_common import (  # noqa: E402
    ledger_paths,
    load_invalidations,
)
from src.lazybull.v2.hosts.testbed.storage import seal_record  # noqa: E402

#: 三臂（矩阵需要 2 提升 + 1 未提升的部分 roots 场景）
THREE_ARMS: Tuple[str, ...] = ("aa_1", "aa_2", "aa_3")
#: 默认 campaign 标识
CAMPAIGN_ID = "t3_campaign"
#: 规模窗口虚拟日数（≥40，跨多代）
SCALE_DAYS = 40
#: 生产根（零写入断言对象）
PRODUCTION_DIRS: Tuple[str, ...] = ("paper", "runs", "models")


def build_env(
    tmp_path: Path,
    *,
    days: Sequence[str] = base.DAYS,
    calendar: Sequence[str] = base.CALENDAR,
    **kwargs: Any,
) -> SimpleNamespace:
    """构造隔离环境（复用 T2：真实 parquet 的 clean/daily + features/panel + 模型三件套）。"""
    return base.build_env(tmp_path, days=days, calendar=calendar, **kwargs)


def prepare(
    env: SimpleNamespace,
    *,
    arms: Sequence[str] = THREE_ARMS,
    campaign_id: str = CAMPAIGN_ID,
) -> SimpleNamespace:
    """登记 campaign + 臂 + D0 fresh（全部 ``aa_clone`` 角色）。"""
    return base.prepare_campaign(env, campaign_id=campaign_id, arms=arms)


def scale_days(count: int = SCALE_DAYS) -> Tuple[str, ...]:
    """长窗口虚拟日（连续交易日；最后 2 日留作窗口外已知日）。"""
    return tuple(pd.bdate_range("2026-07-06", periods=int(count) + 2).strftime("%Y%m%d"))


def run(
    env: SimpleNamespace,
    prepared: SimpleNamespace,
    days: Sequence[str],
    *,
    fault_hook: Any = None,
) -> Dict[str, Any]:
    """以合成宿主工厂运行指定虚拟日（T3 统一入口）。"""
    return base.run_campaign(env, prepared, days=list(days), fault_hook=fault_hook)


def fault_hook(point: str, *, arm_id: Optional[str] = None, min_promoted: Optional[int] = None):
    """命中注入点（可选臂/已提升数过滤）即模拟进程崩溃（不回滚、不清理）。"""

    def hook(name: str, payload: Mapping[str, Any]) -> None:
        if name != point:
            return
        if arm_id is not None and payload.get("arm_id") != arm_id:
            return
        if min_promoted is not None and int(payload.get("promoted_count", 0)) < min_promoted:
            return
        raise InjectedFault(f"injected fault at {name}（payload={dict(payload)}）")

    return hook


def case_hook(case: FaultCase):
    """矩阵行 → 故障钩子（无注入点的案例返回 None）。"""
    return fault_hook(
        case.fault_point,
        arm_id=case.arm_id or None,
        min_promoted=case.min_promoted,
    )


def recover(prepared: SimpleNamespace, *, fault_hook: Any = None) -> Dict[str, Any]:
    """启动恢复（T2 语义：已发布只补齐，未发布终止回滚）。"""
    return recover_campaign(layout_for(prepared.root), CAMPAIGN_ID, fault_hook=fault_hook)


def audit(prepared: SimpleNamespace) -> Dict[str, Any]:
    """只读三方一致性审计。"""
    return audit_ledger(layout_for(prepared.root), CAMPAIGN_ID)


def gap_kinds(prepared: SimpleNamespace) -> List[str]:
    """当前审计中出现的缺口分类（去重排序）。"""
    report = audit(prepared)
    return sorted(kind for kind, count in report["gaps_by_class"].items() if count)


def shard_path(prepared: SimpleNamespace, arm_id: str, day: str) -> Path:
    return layout_for(prepared.root).shard_path(arm_id, day)


def tamper_shard(prepared: SimpleNamespace, arm_id: str, day: str, *, reseal: bool = True) -> None:
    """篡改已发布分片（``reseal=True`` ⇒ 封存仍合法但内容身份不符）。"""
    path = shard_path(prepared, arm_id, day)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["valuation"]["nav"] = 0.123456
    body = seal_record(payload) if reseal else payload
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def rerun_same_day(env: SimpleNamespace, prepared: SimpleNamespace, day: str) -> Dict[str, Any]:
    """同日以新 attempt 重算（由已恢复父代驱动）。"""
    return run(env, prepared, [day])


def rewrite_partition_content(data_root: Path, day: str) -> None:
    """改写某日 ``clean/daily`` 分区内容（构造窗口数据指纹漂移）。"""
    path = data_root / "clean" / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
    frame = pd.read_parquet(path)
    frame["close"] = frame["close"].astype(float) + 1.0
    frame["close_adj"] = frame["close"]
    frame.to_parquet(path, index=False)


def tree_state(root: Path) -> Dict[str, Any]:
    """目录递归快照（相对路径 → {size, sha256}；用于「零写入」断言）。"""
    base_path = Path(root)
    if not base_path.exists():
        return {}
    state: Dict[str, Any] = {}
    for path in sorted(base_path.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state[path.relative_to(base_path).as_posix()] = {
            "size": path.stat().st_size,
            "sha256": digest,
        }
    return state


def metadata_state(root: Path) -> Dict[str, Any]:
    """目录元数据快照（相对路径 → (size, mtime_ns)）；大目录免哈希。"""
    base_path = Path(root)
    if not base_path.exists():
        return {}
    state: Dict[str, Any] = {}
    for path in sorted(base_path.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        state[path.relative_to(base_path).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return state


def production_snapshot() -> Dict[str, Any]:
    """生产根快照：``data/paper``、``data/runs`` 递归哈希 + ``data/models`` 元数据。"""
    return {
        "paper": tree_state(ROOT / "data" / "paper"),
        "runs": tree_state(ROOT / "data" / "runs"),
        "models": metadata_state(ROOT / "data" / "models"),
        "testbed_root_exists": (ROOT / "data" / "paper_shadow" / "p5b_testbed").exists(),
    }


def production_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    """生产根零写入判定（由快照比对得出，不依赖矩阵期望）。"""
    return dict(before) == dict(after)


def read_chain_artifact(prepared: SimpleNamespace) -> Dict[str, Any]:
    path = ledger_paths(layout_for(prepared.root)).chain_path
    return json.loads(path.read_text(encoding="utf-8"))


def read_audit_artifact(prepared: SimpleNamespace) -> Dict[str, Any]:
    path = ledger_paths(layout_for(prepared.root)).audit_path
    return json.loads(path.read_text(encoding="utf-8"))


def invalidations(prepared: SimpleNamespace) -> Dict[str, Any]:
    return load_invalidations(layout_for(prepared.root))


def index(prepared: SimpleNamespace) -> Dict[str, Any]:
    """读侧 index（已发布代唯一视图）。"""
    return json.loads((prepared.root / "_index.json").read_text(encoding="utf-8"))


def journal(prepared: SimpleNamespace, day: str) -> Dict[str, Any]:
    """读当日族级事务 journal。"""
    return json.loads((prepared.root / "journal" / f"{day}.json").read_text(encoding="utf-8"))


def _error_name(exc: BaseException) -> str:
    """实际异常类名（原始测量值，供矩阵白名单比对）。"""
    return type(exc).__name__


def _inject_and_recover(
    env: SimpleNamespace, prepared: SimpleNamespace, case: FaultCase
) -> Dict[str, Any]:
    """注入 → 启动恢复 → （必要时）缺口修复 → 同日重算。"""
    target = env.days[1]
    try:
        run(env, prepared, [target], fault_hook=case_hook(case))
    except InjectedFault:
        pass  # 模拟进程崩溃：不做清理，交给 recover_campaign
    detected = gap_kinds(prepared)
    mid = snapshot_campaign(layout_for(prepared.root), CAMPAIGN_ID)
    recover(prepared)
    if case.requires_repair:
        repair_ledger(layout_for(prepared.root), CAMPAIGN_ID, apply=True)
    rerun_same_day(env, prepared, target)
    return {
        "target_date": target,
        "error_type": "",
        "measurements": {},
        "detected_gaps": detected,
        "injected_attempt_id": first_attempt_id(mid, target),
        "mid_published_generation": int(mid.get("published_generation") or 0),
    }


def _crash_during_abort(
    env: SimpleNamespace, prepared: SimpleNamespace, case: FaultCase
) -> Dict[str, Any]:
    """回滚/恢复过程中再次崩溃：续同一事务（不新建 attempt），随后同日重算。"""
    target = env.days[1]
    try:
        run(
            env,
            prepared,
            [target],
            fault_hook=fault_hook("after_promote", arm_id="aa_1"),
        )
    except InjectedFault:
        pass
    detected = gap_kinds(prepared)
    mid = snapshot_campaign(layout_for(prepared.root), CAMPAIGN_ID)
    attempt_id = first_attempt_id(mid, target)
    try:
        recover(prepared, fault_hook=fault_hook("after_abort_restore"))
    except InjectedFault:
        pass
    resumed = snapshot_campaign(layout_for(prepared.root), CAMPAIGN_ID)
    resumed_attempt = next(
        (
            item
            for item in resumed.get("attempts") or []
            if str(item.get("attempt_id")) == attempt_id
        ),
        None,
    )
    assert resumed_attempt is not None, "恢复中崩溃后 attempt 不得消失"
    assert str(resumed_attempt.get("status")) in {"ABORTING", "ABORTED"}
    recover(prepared)
    rerun_same_day(env, prepared, target)
    return {
        "target_date": target,
        "error_type": "",
        "measurements": {},
        "detected_gaps": detected,
        "injected_attempt_id": attempt_id,
        "mid_published_generation": int(mid.get("published_generation") or 0),
        "resumed_status": str(resumed_attempt.get("status")),
    }


def _reject_by_regression(
    env: SimpleNamespace, prepared: SimpleNamespace, case: FaultCase
) -> Dict[str, Any]:
    """虚拟日期回退（请求日序倒置）⇒ 拒绝推进。"""
    error_type = ""
    try:
        run(env, prepared, [env.days[2], env.days[1]])
    except Exception as exc:  # noqa: BLE001 - 记录原始异常类名，供白名单比对
        error_type = _error_name(exc)
    return {
        "target_date": env.days[1],
        "error_type": error_type,
        "measurements": {},
        "detected_gaps": [],
        "injected_attempt_id": "",
    }


def _reject_by_drift(
    env: SimpleNamespace, prepared: SimpleNamespace, case: FaultCase
) -> Dict[str, Any]:
    """窗口数据指纹漂移 ⇒ 拒绝推进（在任何写盘之前）。"""
    rewrite_partition_content(env.data_root, env.days[1])
    error_type = ""
    try:
        run(env, prepared, [env.days[1]])
    except Exception as exc:  # noqa: BLE001
        error_type = _error_name(exc)
    return {
        "target_date": env.days[1],
        "error_type": error_type,
        "measurements": {},
        "detected_gaps": [],
        "injected_attempt_id": "",
    }


def _invalidate_shard(
    env: SimpleNamespace, prepared: SimpleNamespace, case: FaultCase
) -> Dict[str, Any]:
    """已发布分片内容身份不符 ⇒ 登记 INVALIDATED 并拒绝推进。"""
    run(env, prepared, [env.days[1]])
    tamper_shard(prepared, case.arm_id or "aa_1", env.days[0])
    # 篡改属于外部损坏：快照取自「篡改之后」，避免把损坏读成框架重复入账
    before = snapshot_campaign(layout_for(prepared.root), CAMPAIGN_ID)
    detected = gap_kinds(prepared)
    repair_ledger(layout_for(prepared.root), CAMPAIGN_ID, apply=True)
    error_type = ""
    try:
        run(env, prepared, [env.days[2]])
    except LedgerInvalidatedError as exc:
        error_type = _error_name(exc)
        assert "INVALIDATED" in str(exc) or "拒绝推进" in str(exc)
    return {
        "target_date": env.days[2],
        "error_type": error_type,
        "measurements": {},
        "detected_gaps": detected,
        "injected_attempt_id": "",
        "before": before,
    }


def _zero_write_case(
    env: SimpleNamespace, prepared: SimpleNamespace, case: FaultCase
) -> Dict[str, Any]:
    """完整提交一日，并在前后比对生产根快照（原始测量，不是期望）。"""
    before_prod = production_snapshot()
    run(env, prepared, [env.days[1]])
    after_prod = production_snapshot()
    return {
        "target_date": env.days[1],
        "error_type": "",
        "measurements": {
            "production_roots_unchanged": production_unchanged(before_prod, after_prod),
            "production_snapshot_keys": sorted(before_prod),
        },
        "detected_gaps": [],
        "injected_attempt_id": "",
    }


_CASE_DRIVERS = {
    "virtual_date_regression": _reject_by_regression,
    "data_fingerprint_drift": _reject_by_drift,
    "published_shard_mismatch": _invalidate_shard,
    "production_zero_write": _zero_write_case,
    "after_abort_restore": _crash_during_abort,
}


def run_matrix_case(
    env: SimpleNamespace, prepared: SimpleNamespace, case: FaultCase
) -> Dict[str, Any]:
    """执行一条矩阵行并返回判定表记录（含中间态断言材料）。

    基线：先发布 ``days[0]``（保证「历史已发布账本」存在，可验证不重复入账）；
    再在 ``days[1]``（拒绝类为被拒推进日）注入故障并按矩阵行的恢复序列收敛。
    """
    run(env, prepared, [env.days[0]])
    before = snapshot_campaign(layout_for(prepared.root), CAMPAIGN_ID)
    driver = _CASE_DRIVERS.get(case.fault_point)
    outcome = (
        driver(env, prepared, case)
        if driver is not None
        else _inject_and_recover(env, prepared, case)
    )
    record = evaluate_case(
        layout_for(prepared.root),
        CAMPAIGN_ID,
        case,
        outcome.get("before") or before,
        target_date=str(outcome.get("target_date") or env.days[1]),
        error_type=str(outcome.get("error_type") or ""),
        detected_gaps=outcome.get("detected_gaps") or (),
        injected_attempt_id=str(outcome.get("injected_attempt_id") or ""),
        measurements=outcome.get("measurements") or {},
    )
    return {"record": record, "outcome": outcome, "before": before}


def assert_unpublished_before_recovery(
    prepared: SimpleNamespace, *, target_day: str, expected_generation: int
) -> None:
    """中间态：崩溃后该日不得出现在读侧（读侧只见 published_generation）。"""
    layout = layout_for(prepared.root)
    index = read_index(layout)
    days = published_days(index)
    assert target_day not in days, f"崩溃后 {target_day} 不得进入读侧"
    assert int(index["published_generation"]) == int(expected_generation)
    for arm_id in THREE_ARMS:
        result = query_shards(layout, arm_id)
        assert target_day not in {day["virtual_date"] for day in result["days"]}


def shard_digests(prepared: SimpleNamespace, arm_id: str) -> Dict[str, str]:
    """已发布分片的 ``date → day_digest``（不变量 ③ 的逐值比对材料）。"""
    result: Dict[str, str] = {}
    for path in sorted((prepared.root / "ledger" / arm_id).glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[path.stem] = str((payload.get("evidence") or {}).get("day_digest"))
    return result


def attempt_evidence(prepared: SimpleNamespace, day: str) -> Dict[str, Any]:
    """当日全部 attempt 的逐臂驱动证据（determinism 比对材料）。"""
    path = prepared.root / "journal" / f"{day}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(attempt["attempt_id"]): {
            str(arm_id): dict(payload.get("evidence") or {})
            for arm_id, payload in (attempt.get("arms") or {}).items()
        }
        for attempt in doc.get("attempts") or []
    }
