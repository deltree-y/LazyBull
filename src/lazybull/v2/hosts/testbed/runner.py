# -*- coding: utf-8 -*-
"""P5b 试车场 T2 虚拟时钟多臂运行器（历史窗口、逐日推进、族级事务提交）。

契约来源：设计 v4 §4.1（逐日推进 / ``schedule_mode=replay`` / 不联网 / 不补齐）、
§4.1.1（虚拟日成功资格）、§4.3（族级事务与恢复）、§4.4（10 个连续历史交易日）、
§7.2（报告字段）。

执行形态
========
1. 启动即恢复（``recover_campaign``）：已发布 attempt 只补齐派生投影；未发布 attempt 终止回滚；
2. 已发布代由 ``_index.json`` 唯一决定；index 缺失时以 D0 fresh 建立 genesis（generation 0）；
3. 逐虚拟日：已发布日只做幂等补齐并推进；未发布日开新 attempt（备份父代 → 暂存根驱动 →
   逐臂资格校验 → 提升 → 分片 → 唯一发布点）；
4. 任一臂 ``not_ready`` ⇒ 整族停留同一虚拟日（attempt 终止、不写 index、不推进）；
5. 报告落 ``reports/``，字段覆盖 §7.2 的 T2 部分（窗口 / 行域回执 / 模型截止 / 五段身份 /
   每臂 driven-日期-完整代-权威估值条数 / family_txn_id / parent-published generation /
   roots 内容身份 / 唯一发布点 / 故障注入与恢复 / 机器墙钟）。

边界：不写 ``data/paper`` / ``data/runs`` / ``data/models``，不写 ``LedgerEntry``，不联网，
不做真实数据补齐，不注册真实候选；全部产物 ``evidence_class=calibration_only``。
"""

from __future__ import annotations

import json
import os
import socket
import uuid
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.v2.hosts.testbed.arm_runtime import (
    ArmSpec,
    HostFactory,
    VirtualDayAlreadyRun,
    VirtualDayNotReady,
    drive_virtual_day,
    load_arm_specs,
    make_default_host_factory,
)
from src.lazybull.v2.hosts.testbed.clock import VirtualClock, calendar_identity
from src.lazybull.v2.hosts.testbed.d0 import load_d0_record
from src.lazybull.v2.hosts.testbed.family_txn import (
    FaultHook,
    InjectedFault,
    cache_manifests,
    open_family_transaction,
    recover_campaign,
    top_up_if_needed,
)
from src.lazybull.v2.hosts.testbed.identity import (
    collect_code_identity,
    collect_data_state_identity,
)
from src.lazybull.v2.hosts.testbed.journal import (
    create_genesis_index,
    load_day_journal,
    published_days,
    read_index,
)
from src.lazybull.v2.hosts.testbed.layout import (
    CampaignInvalidError,
    CampaignLayout,
    FamilyTxnError,
    gen_id,
    layout_for,
    optional_identity,
    root_identity,
)
from src.lazybull.v2.hosts.testbed.paths import (
    EVIDENCE_CLASS,
    SCOPE,
    assert_write_scope,
)
from src.lazybull.v2.hosts.testbed.registration import load_campaign
from src.lazybull.v2.hosts.testbed.storage import seal_record, write_json_atomic
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "CampaignBusyError",
    "CampaignRunLock",
    "LONG_TASK_THRESHOLD_SECONDS",
    "REPORT_SCHEMA",
    "RUN_LOCK_FILENAME",
    "assert_window_drift_free",
    "campaign_fingerprint",
    "run_virtual_campaign",
]

#: 运行报告 schema
REPORT_SCHEMA = "p5b_testbed_t2_run_report/v1"
#: campaign 级运行锁（并发运行器的唯一裁决点）
RUN_LOCK_FILENAME = ".run_attempt"
#: 运行锁 stale 判定（秒；仅在锁文件不可读时作兜底信息，不用于抢占活跃持有者）
GUARD_STALE_SECONDS = 30.0
RUN_LOCK_TIMEOUT_SECONDS = 3600.0
#: 单批机器时长授权阈值（秒；设计 §4.4「预计 >20 分钟须单独授权」）
LONG_TASK_THRESHOLD_SECONDS = 1200.0
#: 成功资格要求小写 ``t1``（§4.1.1 A1）
_REQUIRED_RUN_KIND = "t1"


class CampaignBusyError(FamilyTxnError):
    """已有运行器持有 campaign 锁（并发运行禁止）。"""

    __test__ = False


class CampaignRunLock:
    """campaign 级独占运行锁（``O_CREAT|O_EXCL``）。

    stale 判定 = **同主机且持有者 pid 已不存在**（``psutil.pid_exists``），不做纯 mtime
    超时抢占——整批历史回放可能运行超过任一固定阈值，按时间抢占会与活跃运行器并发。
    锁文件不可读/信息缺失 ⇒ 一律视为被持有（fail-closed），由人工核对后清理。
    ``timeout_seconds`` 参数**不参与抢占判定**（保留给 T3 心跳阈值），不得当作保护阈值使用。
    """

    def __init__(
        self, layout: CampaignLayout, *, timeout_seconds: float = RUN_LOCK_TIMEOUT_SECONDS
    ) -> None:
        self.layout = layout
        self.timeout_seconds = float(timeout_seconds)
        self.path = layout.journal_root / RUN_LOCK_FILENAME
        self.acquired = False

    def __enter__(self) -> "CampaignRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.token = uuid.uuid4().hex
        holder = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "token": self.token,
            "acquired_at": datetime.now().isoformat(timespec="seconds"),
        }
        # 所有对锁文件的读-改-写都在**串行化守卫**内完成：守卫用 O_EXCL 创建，
        # 保证「判定死持有者 → 摘除 → 创建新锁」这一段临界区互斥（R2-P5B-02：
        # 单纯 read→rename/unlink 都会按路径操作，能把别人的新锁摘走）。
        deadline = time.monotonic() + float(self.timeout_seconds)
        while True:
            if self._acquire_guard(deadline):
                try:
                    existing = self._read_holder() if self.path.exists() else {}
                    if existing and not self._holder_is_dead(existing):
                        raise CampaignBusyError(
                            f"campaign 已被另一运行器持有（holder={existing}）: {self.path}；"
                            "并发运行会破坏族级事务的父代比较交换，必须先释放或人工清理"
                        )
                    if existing:
                        logger.warning(
                            f"检测到已死持有者的 campaign 运行锁（holder={existing}），在守卫内摘除"
                        )
                        self.path.unlink(missing_ok=True)
                    descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        json.dump(holder, handle, ensure_ascii=False)
                finally:
                    self._release_guard()
                self.acquired = True
                return self
            if time.monotonic() >= deadline:
                raise CampaignBusyError(f"campaign 运行锁抢占失败（守卫超时）: {self.path}")
            time.sleep(0.01)

    def _guard_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.guard")

    def _acquire_guard(self, deadline: float) -> bool:
        """取得串行化守卫（O_EXCL）；陈旧守卫（持有者已死或超时）可按 mtime 清理。"""
        guard = self._guard_path()
        try:
            descriptor = os.open(str(guard), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - guard.stat().st_mtime
            except OSError:
                return False
            if age > GUARD_STALE_SECONDS:
                logger.warning(f"检测到陈旧锁守卫（{age:.0f}s），按超时清理: {guard}")
                guard.unlink(missing_ok=True)
            return False
        payload = {"pid": os.getpid(), "token": self.token}
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return True

    def _release_guard(self) -> None:
        guard = self._guard_path()
        try:
            payload = json.loads(guard.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if str((payload or {}).get("token") or "") == self.token:
            guard.unlink(missing_ok=True)

    def _read_holder(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"campaign 运行锁读取失败（按被持有处理）: {self.path}（{exc}）")
            return {}
        return payload if isinstance(payload, dict) else {}

    def _holder_is_dead(self, holder: Mapping[str, Any]) -> bool:
        """仅当同主机且 pid 确认不存在时判死；其余一律视为仍在运行。"""
        if str(holder.get("host") or "") != socket.gethostname():
            return False
        pid = holder.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil 为项目依赖
            logger.warning("psutil 不可用，无法判定锁持有者存活；按被持有处理")
            return False
        return not psutil.pid_exists(pid)

    def __exit__(self, *exc_info: Any) -> None:
        if not self.acquired:
            return
        # 释放同样在守卫内完成，并校验令牌：只删自己持有的那把锁
        deadline = time.monotonic() + float(self.timeout_seconds)
        guard_ok = self._acquire_guard(deadline)
        try:
            if str(self._read_holder().get("token") or "") == self.token:
                self.path.unlink(missing_ok=True)
            else:
                logger.warning(f"campaign 运行锁已不再属于本进程，跳过删除: {self.path}")
        finally:
            if guard_ok:
                self._release_guard()
        self.acquired = False


@contextmanager
def _campaign_lock(layout: CampaignLayout) -> Iterator[CampaignRunLock]:
    with CampaignRunLock(layout) as lock:
        yield lock


def assert_window_drift_free(campaign: Mapping[str, Any]) -> Dict[str, Any]:
    """运行前漂移门：窗口内容数据态 + 执行闭包代码态必须与 campaign 登记一致。

    口径与 T1 ``d0._assert_drift_free`` 同源（同一 ``collect_data_state_identity`` /
    ``collect_code_identity`` 与同两个门禁字段）；此处不直接复用该私有函数的原因 = T1 冻结件
    不可改，且其异常语义（``D0FreshError``/「禁止初始化 D0」）不适用于运行期。差异仅限
    异常类型与消息，**判定字段完全一致**。
    """
    identity = campaign.get("identity") or {}
    clock = campaign["clock"]
    data_root = str((identity.get("input") or {})["data_root"])
    window_state = collect_data_state_identity(
        data_root=data_root, virtual_days=[str(day) for day in clock["days"]]
    )
    recorded_window = (identity.get("window_state") or {}).get("window_state_id")
    if window_state["window_state_id"] != recorded_window:
        raise CampaignInvalidError(
            "窗口数据态漂移：campaign 登记后窗口内历史分区内容已变化，禁止推进虚拟时钟: "
            f"{window_state['window_state_id']} vs {recorded_window}"
        )
    code_identity = collect_code_identity(str(campaign.get("repo_root") or "."))
    recorded_code = (identity.get("code") or {}).get("execution_digest")
    if code_identity["execution_digest"] != recorded_code:
        raise CampaignInvalidError(
            "执行闭包代码态漂移：campaign 登记后试车场/复用件源码已变化，禁止推进虚拟时钟: "
            f"{code_identity['execution_digest']} vs {recorded_code}"
        )
    return {
        "window_state_id": window_state["window_state_id"],
        "execution_digest": code_identity["execution_digest"],
        "global_watermark": window_state.get("global_watermark"),
    }


def _campaign_clock(campaign: Mapping[str, Any]) -> VirtualClock:
    """虚拟时钟唯一来源 = campaign 注册记录（连续性已在注册期按交易日历校验）。"""
    payload = campaign["clock"]
    return VirtualClock(
        d0_date=str(payload["d0_date"]),
        days=tuple(str(day) for day in payload["days"]),
    )


def _ensure_genesis(
    layout: CampaignLayout,
    campaign: Mapping[str, Any],
    d0_record: Mapping[str, Any],
    specs: Mapping[str, ArmSpec],
) -> Dict[str, Any]:
    """建立或核验 D0 fresh 起点 index（generation 0 = 已发布代的地基）。"""
    index = read_index(layout)
    if index is None:
        identities = {arm_id: root_identity(layout.arm_root(arm_id)) for arm_id in sorted(specs)}
        index = create_genesis_index(
            layout,
            str(campaign["campaign_id"]),
            identities,
            str(d0_record["record_sha256"]),
        )
        logger.info(
            f"试车场 {campaign['campaign_id']} 建立 genesis index（generation 0，"
            f"D0={d0_record.get('d0_date')}）"
        )
        return index
    for arm_id, entry in sorted((index.get("arms") or {}).items()):
        identity = optional_identity(layout.arm_root(arm_id))
        if identity is None or identity["content_sha"] != entry.get("content_sha"):
            raise CampaignInvalidError(
                f"臂 {arm_id!r} 的现场根与已发布代不一致（先执行恢复或人工核对）: "
                f"现场={None if identity is None else identity['content_sha']} "
                f"已发布={entry.get('content_sha')}"
            )
    return index


def _resolve_days(
    clock_days: Sequence[str], published: Mapping[str, Any], requested: Optional[Sequence[str]]
) -> List[str]:
    """解析本次要处理的虚拟日：禁止跳过未运行日、禁止乱序、禁止重复。"""
    pending = [str(day) for day in clock_days if str(day) not in published]
    if requested is None:
        return pending
    days = [str(day) for day in requested]
    unknown = [day for day in days if day not in set(str(item) for item in clock_days)]
    if unknown:
        raise FamilyTxnError(f"请求虚拟日不在 campaign 窗口内（禁止越窗）: {unknown}")
    if days != sorted(days) or len(set(days)) != len(days):
        raise FamilyTxnError(f"请求虚拟日必须升序且无重复（禁止时钟回退）: {days}")
    unpublished = [day for day in days if day not in published]
    if unpublished != pending[: len(unpublished)]:
        raise FamilyTxnError(
            "请求虚拟日必须从最早未发布日连续推进（禁止跳日）: "
            f"requested_unpublished={unpublished} pending={pending}"
        )
    return days


def _attempt_for_day(layout: CampaignLayout, campaign_id: str, virtual_date: str) -> Dict[str, Any]:
    """取该虚拟日已发布 attempt（幂等补齐的原料；缺失 ⇒ campaign 失效）。"""
    doc = load_day_journal(layout, virtual_date)
    for attempt in doc.get("attempts") or []:
        if attempt.get("status") == "COMMITTED":
            return dict(attempt)
    raise CampaignInvalidError(
        f"index 声明虚拟日 {virtual_date} 已发布，但 journal 无 COMMITTED attempt（campaign 失效）: "
        f"{campaign_id}"
    )


def _run_day(
    layout: CampaignLayout,
    campaign: Mapping[str, Any],
    specs: Mapping[str, ArmSpec],
    virtual_date: str,
    host_factory: HostFactory,
    fault_hook: Optional[FaultHook],
) -> Dict[str, Any]:
    """单虚拟日的族级事务：开 attempt → 逐臂驱动 → 提升 → 分片 → 唯一发布点。"""
    campaign_id = str(campaign["campaign_id"])
    txn = open_family_transaction(layout, campaign_id, specs, virtual_date, fault_hook=fault_hook)
    started = time.perf_counter()
    try:
        for arm_id in sorted(specs):
            host = host_factory(specs[arm_id], txn.staged_arm_root(arm_id))
            evidence = drive_virtual_day(host, specs[arm_id], virtual_date)
            txn.record_driven(arm_id, evidence)
        txn.finish_driving()
        txn.promote()
        txn.write_artifacts()
        txn.publish()
    except InjectedFault:
        # 故障注入 = 模拟进程崩溃：不做任何清理，交给 recover_campaign 收拾现场
        raise
    except VirtualDayAlreadyRun as exc:
        txn.abort("already_run_unpublished")
        raise CampaignInvalidError(
            f"未发布 attempt 的 already_run 不得直接接纳（§4.1.1 B2）：{exc}；"
            "attempt 已终止回滚，父代已含该日但未发布 ⇒ 现场不一致，需人工核对"
        ) from exc
    except VirtualDayNotReady:
        txn.abort("not_ready")
        raise
    except Exception:
        # 驱动/提升失败：父代未被部分前滚（提升阶段之前）或已整体回滚（提升阶段之后）
        if txn.attempt.get("status") != "ABORTED":
            txn.abort("drive_error")
        raise
    attempt = txn.attempt
    return {
        "virtual_date": virtual_date,
        "mode": "committed",
        "family_txn_id": attempt["family_txn_id"],
        "attempt_id": attempt["attempt_id"],
        "parent_generation": int(attempt["parent_generation"]),
        "published_generation": int(attempt["published_generation"]),
        "parent_roots": {
            arm_id: dict(payload["parent"]) for arm_id, payload in attempt["arms"].items()
        },
        "new_roots": {
            arm_id: dict(payload["staged"] or {}) for arm_id, payload in attempt["arms"].items()
        },
        "promoted_count": int(attempt["promoted_count"]),
        "unique_publish_point": {
            "field": "journal.attempt.published_txn_id",
            "value": attempt["published_txn_id"],
        },
        "arms": {
            arm_id: dict(payload["evidence"] or {}) for arm_id, payload in attempt["arms"].items()
        },
        "seconds": time.perf_counter() - started,
    }


def _aa_consistency(
    specs: Mapping[str, ArmSpec], day_records: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """A/A 克隆臂逐值一致（日期/成交/现金/持仓/NAV，经确定性日摘要比对）。"""
    aa_arms = sorted(arm_id for arm_id, spec in specs.items() if spec.arm_role == "aa_clone")
    result: Dict[str, Any] = {
        "arms": aa_arms,
        "comparable": len(aa_arms) >= 2,
        "compared_days": [],
        "identical": True,
        "mismatches": [],
    }
    if not result["comparable"]:
        result["note"] = "aa_clone 臂不足 2 个，A/A 逐值一致不可判定"
        return result
    for record in day_records:
        if record.get("mode") != "committed":
            continue
        date = str(record["virtual_date"])
        digests = {
            arm_id: str((record.get("arms") or {}).get(arm_id, {}).get("day_digest"))
            for arm_id in aa_arms
        }
        result["compared_days"].append(date)
        if len(set(digests.values())) != 1:
            result["identical"] = False
            result["mismatches"].append({"virtual_date": date, "digests": digests})
    return result


def _row_domain_evidence(day_records: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], bool]:
    """逐日逐臂行域回执汇总（仅"日期已物化"不算 PIT 通过）。

    非真空判定：任一已提交日缺臂回执或任一臂回执未通过 ⇒ ``verified=False``。
    """
    per_day: Dict[str, Any] = {}
    verified = True
    for record in day_records:
        if record.get("mode") != "committed":
            continue
        date = str(record["virtual_date"])
        arm_receipts: Dict[str, Any] = {}
        for arm_id, evidence in sorted((record.get("arms") or {}).items()):
            receipt = evidence.get("row_domain_receipt") or {}
            passed = receipt.get("passed") is True and set(
                receipt.get("actual_row_domains") or []
            ) == {"structural"}
            verified = verified and passed
            arm_receipts[arm_id] = {
                "actual_row_domains": receipt.get("actual_row_domains"),
                "files": receipt.get("files"),
                "passed": passed,
            }
        if not arm_receipts:
            verified = False
        per_day[date] = arm_receipts
    return per_day, verified


def _arm_summaries(
    specs: Mapping[str, ArmSpec], day_records: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """每臂：driven 日数 / 日期 / 完整代数 / 权威估值条数 / 末次 NAV / 根内容身份。"""
    summaries: Dict[str, Any] = {}
    for arm_id, spec in sorted(specs.items()):
        dates: List[str] = []
        values = 0
        complete = 0
        last_nav: Optional[float] = None
        digests: List[str] = []
        for record in day_records:
            evidence = (record.get("arms") or {}).get(arm_id) or {}
            if not evidence:
                continue
            dates.append(str(record["virtual_date"]))
            values += int((evidence.get("valuation") or {}).get("row_count", 0))
            complete += 1 if evidence.get("complete_generation") else 0
            last_nav = float((evidence.get("valuation") or {}).get("nav", last_nav or 0.0))
            digests.append(str(evidence.get("day_digest")))
        summaries[arm_id] = {
            "arm_role": spec.arm_role,
            "driven_days": len(dates),
            "driven_dates": dates,
            "complete_generations": complete,
            "authoritative_values": values,
            "last_nav": last_nav,
            "day_digests": digests,
            "identity": {
                "config_fingerprint": spec.config_fingerprint,
                "model_version": spec.model_version,
                "model_content_fingerprint": spec.model_content_fingerprint,
                "train_end_date": spec.train_end_date,
                "initial_capital": spec.initial_capital,
            },
        }
    return summaries


def _build_report(
    layout: CampaignLayout,
    campaign: Mapping[str, Any],
    d0_record: Mapping[str, Any],
    specs: Mapping[str, ArmSpec],
    clock: VirtualClock,
    index: Mapping[str, Any],
    drift: Mapping[str, Any],
    day_records: Sequence[Mapping[str, Any]],
    recovery: Mapping[str, Any],
    paused: Optional[Mapping[str, Any]],
    machine_time: Mapping[str, Any],
) -> Dict[str, Any]:
    """组装 §7.2 的 T2 覆盖字段（不含 T3/T4/T5 的账本与裁判结论）。"""
    per_day_row_domain, row_domain_verified = _row_domain_evidence(day_records)
    committed_days = [record for record in day_records if record.get("mode") == "committed"]
    return seal_record(
        {
            "schema": REPORT_SCHEMA,
            **_report_scope(campaign, layout),
            "window": _report_window(clock),
            "row_domain": {
                "declared": str(campaign.get("row_domain")),
                "verified": bool(row_domain_verified and committed_days),
                "per_day": per_day_row_domain,
            },
            "label_cutoff": dict(campaign.get("label_cutoff") or {}),
            "model_availability": _report_model_availability(specs, clock),
            "identity": _report_identity(campaign, d0_record, drift, clock),
            "arms": _arm_summaries(specs, day_records),
            "aa_consistency": _aa_consistency(specs, day_records),
            "transactions": [
                {key: value for key, value in record.items() if key not in {"arms", "seconds"}}
                for record in day_records
            ],
            "published": _report_published(index),
            "recovery": dict(recovery),
            "fault_injections": [
                {"point": entry.get("point"), "virtual_date": entry.get("virtual_date")}
                for record in day_records
                for entry in (record.get("faults") or [])
            ],
            "paused": None if paused is None else dict(paused),
            "machine_time": dict(machine_time),
            "limitations": [
                "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE：历史虚拟时钟试车场，不是前向影子",
                "T2 只覆盖多臂运行与族级事务；分片长期账本/正式故障注入矩阵/containment 归 T3",
                "收益与回撤裁判（含 A/A 误报、MDE、临界值）归 T4/T5，本报告不产生裁判结论",
                "A/A 逐值一致不能替代行域回执、PIT 与模型截止证明",
                "窗口末日仍按宿主 T+1 语义生成次日指令帧（落在 campaign 根内、不参与分片证据、"
                "不代表窗口外已运行）",
                "标签截止沿用 campaign 声明（T2 机械层不消费标签，标签截止属统计层登记，"
                "本报告未做核验）",
            ],
        }
    )


def _report_scope(campaign: Mapping[str, Any], layout: CampaignLayout) -> Dict[str, Any]:
    """报告范围声明（calibration_only 边界；不含任何前向/SLA 主张）。"""
    return {
        "scope": SCOPE,
        "evidence_class": EVIDENCE_CLASS,
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "not_real_candidate": True,
        "campaign_id": campaign["campaign_id"],
        "campaign_root": layout.root.as_posix(),
        "schedule_mode": "replay",
    }


def _report_window(clock: VirtualClock) -> Dict[str, Any]:
    """虚拟窗口字段（D0 只作估值点；D1 起才有相邻收益配对）。"""
    return {
        "d0_date": clock.d0_date,
        "d0_role": "valuation_only",
        "first_return_date": clock.first_return_date,
        "virtual_days": list(clock.days),
        "day_count": len(clock.days),
        "history_start": clock.days[0],
        "history_end": clock.days[-1],
        "window_identity": calendar_identity(list(clock.days)),
        "adjacent_return_pairs": [list(pair) for pair in clock.adjacent_return_pairs()],
    }


def _report_model_availability(specs: Mapping[str, ArmSpec], clock: VirtualClock) -> Dict[str, Any]:
    """模型可得性（训练截止必须严格早于 D0）。"""
    return {
        arm_id: {
            "model_version": spec.model_version,
            "train_end_date": spec.train_end_date,
            "trained_before_d0": spec.train_end_date < clock.d0_date,
            "model_content_fingerprint": spec.model_content_fingerprint,
        }
        for arm_id, spec in sorted(specs.items())
    }


def _report_identity(
    campaign: Mapping[str, Any],
    d0_record: Mapping[str, Any],
    drift: Mapping[str, Any],
    clock: VirtualClock,
) -> Dict[str, Any]:
    """五段身份 + D0 fresh 记录链接（数据态/代码态为门禁值）。"""
    return {
        "window_state_id": drift["window_state_id"],
        "execution_digest": drift["execution_digest"],
        "global_watermark": drift.get("global_watermark"),
        "campaign_identity_summary": dict(campaign.get("identity_summary") or {}),
        "d0_fresh_record_sha256": d0_record["record_sha256"],
        "d0_date": d0_record["d0_date"],
        "initial_capital": float(d0_record["initial_capital"]),
        "calendar_day_count": len(clock.days),
    }


def _report_published(index: Mapping[str, Any]) -> Dict[str, Any]:
    """已发布头（唯一发布点投影）。"""
    return {
        "published_generation": int(index["published_generation"]),
        "published_txn_id": index["published_txn_id"],
        "published_virtual_date": index.get("published_virtual_date"),
        "head_gen_id": gen_id(int(index["published_generation"])),
        "arms": dict(index.get("arms") or {}),
    }


def _write_report(layout: CampaignLayout, report: Mapping[str, Any]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = layout.reports_root / f"t2_run_{stamp}.json"
    write_json_atomic(path, dict(report))
    return path


def run_virtual_campaign(
    campaign_root: str | Path,
    *,
    days: Optional[Sequence[str]] = None,
    host_factory: Optional[HostFactory] = None,
    fault_hook: Optional[FaultHook] = None,
    write_report: bool = True,
    allow_external_root: bool = False,
    host_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """在已物化的 structural 历史窗口上逐日推进多臂虚拟时钟并提交族级事务。

    Args:
        campaign_root: 已登记 campaign 根（T1 注册 + D0 fresh 已就绪）。
        days: 本次处理的虚拟日（缺省 = 全部未发布日）；必须从最早未发布日连续推进。
        host_factory: 宿主工厂（缺省 = ``create_paper_host`` + ``schedule_mode="replay"``）。
        fault_hook: 故障注入钩子（测试专用；None = 不插桩）。
        write_report: 是否把报告落 ``reports/``。
        allow_external_root: 仅测试/临时脚本放开外部根（不解除生产根禁令）。
        host_overrides: 仅供测试/影子注入桩信号与数据面。

    Returns:
        运行报告（含 window/row_domain/identity/arms/aa_consistency/transactions/paused/
        recovery/machine_time）。``paused`` 非空表示整族停留该虚拟日未推进。

    Raises:
        CampaignBusyError: 另一运行器持有 campaign 锁。
        CampaignInvalidError: 现场与已发布代不一致 / 父代不可验证 / 数据态漂移。
        TestbedScopeError: 产物根越界（生产根硬禁）。
    """
    started_at = datetime.now()
    started = time.perf_counter()
    root = Path(campaign_root)
    # 写域守卫先行：生产根硬禁（不依赖 campaign 注册记录是否可读）
    assert_write_scope(root, allow_external_root=allow_external_root)
    campaign = load_campaign(root)
    data_root = str((campaign.get("identity") or {})["input"]["data_root"])
    assert_write_scope(root, data_root=data_root, allow_external_root=allow_external_root)
    clock = _campaign_clock(campaign)
    drift = assert_window_drift_free(campaign)
    d0_record = load_d0_record(root)
    specs = load_arm_specs(root)
    layout = layout_for(root)
    # T3 推进守卫（只读）：INVALIDATED 终态 / 不可重建缺口 / 未发布残片 ⇒ 拒绝推进
    from src.lazybull.v2.hosts.testbed.ledger_repair import assert_ledger_advanceable

    assert_ledger_advanceable(layout, str(campaign["campaign_id"]), clock_days=list(clock.days))
    factory = host_factory or make_default_host_factory(
        data_root=data_root,
        replay_calendar=[pd.Timestamp(day) for day in clock.days],
        host_overrides=host_overrides,
    )
    with _campaign_lock(layout):
        # 不可读的读侧投影先隔离留证（否则恢复/推进阶段会以未分类异常中断）
        from src.lazybull.v2.hosts.testbed.ledger_repair import (
            quarantine_unreadable_projections,
        )

        quarantined = quarantine_unreadable_projections(layout)
        recovery = recover_campaign(layout, str(campaign["campaign_id"]), fault_hook=fault_hook)
        if quarantined:
            recovery = {**recovery, "quarantined_projections": quarantined}
        index = _ensure_genesis(layout, campaign, d0_record, specs)
        run_days = _resolve_days(clock.days, published_days(index), days)
        day_records, paused = _drive_days(layout, campaign, specs, run_days, factory, fault_hook)
        index = read_index(layout) or index
        # T3 收尾（仍在锁内）：刷新审计/链产物 + 回收已终态 attempt 的暂存残留
        ledger_tail = _ledger_tail(layout, str(campaign["campaign_id"]), clock)
        machine_time = _machine_time(day_records, started_at=started_at, started=started)
        machine_time = {**machine_time, **ledger_tail}
        report = _build_report(
            layout,
            campaign,
            d0_record,
            specs,
            clock,
            index,
            drift,
            day_records,
            recovery,
            paused,
            machine_time,
        )
        report_path = _write_report(layout, report) if write_report else None
    if report_path is not None:
        logger.info(
            f"试车场 {campaign['campaign_id']} 运行完成："
            f"已发布代 {report['published']['published_generation']}，报告 {report_path}"
        )
    return {**report, "report_path": None if report_path is None else report_path.as_posix()}


def _ledger_tail(layout: CampaignLayout, campaign_id: str, clock: VirtualClock) -> Dict[str, Any]:
    """运行收尾的账本旁路维护（**只写** ``audit/`` 与清理已终态 staging，不改分片字节）。

    - 刷新 ``audit/shard_chain.json`` 与 ``audit/ledger_audit.json``：健康 campaign 的审计状态
      不再被「链产物尚未写」稀释成 GAPS；
    - 回收已终态 attempt 的暂存残留（T2 遗留项 1：整臂复制导致磁盘线性增长）。
    """
    from src.lazybull.v2.hosts.testbed.ledger_audit import (
        audit_ledger,
        write_ledger_artifacts,
    )
    from src.lazybull.v2.hosts.testbed.ledger_repair import prune_staging

    clock_days = list(clock.days)
    pre = audit_ledger(layout, campaign_id, clock_days=clock_days)
    artifacts = write_ledger_artifacts(layout, campaign_id, pre)
    # 写入链/审计产物后再审计一次：报告的是**稳定态**（含链产物新鲜度）
    settled = audit_ledger(layout, campaign_id, clock_days=clock_days)
    pruning = prune_staging(layout, apply=True, take_lock=False)
    return {
        "ledger_audit": {
            "status": settled["status"],
            "three_way_status": settled.get("three_way_status"),
            "artifact_status": settled.get("artifact_status"),
            "shard_chain_digest": (settled.get("chain") or {}).get("chain_digest"),
            "gaps_by_class": {
                kind: count for kind, count in (settled.get("gaps_by_class") or {}).items() if count
            },
            "gaps_before_artifact_write": [gap["kind"] for gap in pre.get("gaps") or []],
            **artifacts,
        },
        "staging_pruned": {
            "files_before": pruning["files_before"],
            "bytes_before": pruning["bytes_before"],
            "bytes_after": pruning["bytes_after"],
            "attempts_removed": len(pruning["removed"]),
        },
    }


def _drive_days(
    layout: CampaignLayout,
    campaign: Mapping[str, Any],
    specs: Mapping[str, ArmSpec],
    run_days: Sequence[str],
    factory: HostFactory,
    fault_hook: Optional[FaultHook],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """逐日推进：已发布日只做幂等补齐；未发布日走族级事务；``not_ready`` 整族停留。

    已发布日的补齐走 :func:`top_up_if_needed`（投影完整即零动作、不读 journal），
    因此「重跑已发布日」是线性成本，不再逐日触发全量 manifest/index 重建。
    """
    campaign_id = str(campaign["campaign_id"])
    day_records: List[Dict[str, Any]] = []
    paused: Optional[Dict[str, Any]] = None
    manifests: Dict[str, Any] = {}
    for virtual_date in run_days:
        index = read_index(layout)
        if index is not None and virtual_date in published_days(index):
            attempt = _attempt_for_day(layout, campaign_id, virtual_date)
            cache_manifests(layout, attempt, manifests)
            topped = top_up_if_needed(
                layout, campaign_id, attempt, index=index, manifests=manifests
            )
            day_records.append(
                {
                    "virtual_date": virtual_date,
                    "mode": "already_published",
                    "family_txn_id": attempt["family_txn_id"],
                    "published_generation": int(attempt["published_generation"]),
                    "top_up": topped or _noop_top_up(attempt),
                }
            )
            continue
        try:
            day_records.append(_run_day(layout, campaign, specs, virtual_date, factory, fault_hook))
            manifests.clear()  # 新发布日已重写清单/索引 ⇒ 缓存失效
        except VirtualDayNotReady as exc:
            paused = {
                "virtual_date": virtual_date,
                "reason": "not_ready",
                "detail": str(exc),
                "gaps": exc.gaps,
                "index_written": False,
                "advanced": False,
            }
            day_records.append(
                {"virtual_date": virtual_date, "mode": "paused_not_ready", "detail": str(exc)}
            )
            logger.warning(f"试车场整族停留虚拟日 {virtual_date}（未推进、未写 index）: {exc}")
            break
    return day_records, paused


def _noop_top_up(attempt: Mapping[str, Any]) -> Dict[str, Any]:
    """已发布日投影已完整时的零动作摘要（与 top_up 同形状，便于报告对齐）。"""
    return {
        "virtual_date": attempt["virtual_date"],
        "family_txn_id": attempt["family_txn_id"],
        "action": "noop",
        "status": "COMMITTED",
        "shards_restored": [],
        "manifests_written": [],
        "index_written": False,
    }


def _machine_time(
    day_records: Sequence[Mapping[str, Any]], *, started_at: datetime, started: float
) -> Dict[str, Any]:
    """机器墙钟（逐日 + 总计；单批 >20 分钟须先报预算，此处只登记实测）。"""
    committed = [record for record in day_records if record.get("mode") == "committed"]
    seconds = [float(record.get("seconds", 0.0)) for record in committed]
    total_seconds = time.perf_counter() - started
    return {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": total_seconds,
        "per_day_seconds": {
            str(record["virtual_date"]): float(record.get("seconds", 0.0)) for record in committed
        },
        "generation_txn_avg_seconds": (sum(seconds) / max(1, len(seconds))),
        # 设计 §4.4/§9：单批预计 >20 分钟须先报预算并获授权；此标记为实测告警，不是授权
        "long_task_threshold_seconds": LONG_TASK_THRESHOLD_SECONDS,
        "long_task_threshold_exceeded": bool(total_seconds > LONG_TASK_THRESHOLD_SECONDS),
    }


def campaign_fingerprint(report: Mapping[str, Any]) -> str:
    """报告内容指纹（供复现登记；排除墙钟字段后计算）。"""
    payload = {
        key: value
        for key, value in dict(report).items()
        if key not in {"record_sha256", "machine_time", "report_path"}
    }
    return canonical_json_sha256(payload)
