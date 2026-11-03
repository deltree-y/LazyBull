# -*- coding: utf-8 -*-
"""P5b T6 历史端到端演练编排：冻结 → 注册/D0 → 运行（宿主层注入）→ 墙钟与授权。

契约来源：设计 v4 §6.1~§6.3（全链路演练）、§7.1（产物根）、§9（>20 分钟先报预算并获授权）。

编排顺序即纪律
==============
1. ``freeze_rehearsal``：先冻结 T6 manifest（窗口/五臂/δ 倍数/λ/ζ/H6 规则/上游链接），
   之后不得按读数调整；
2. ``register_rehearsal``：登记 campaign + 五臂 + D0 fresh（同一 D0、同一 structural 输入根、
   固定 ``schedule_mode=replay``）；
3. ``run_rehearsal``：``run_virtual_campaign`` 逐日推进，注入在宿主层（见 :mod:`hosts`）；
   单日实测外推整批 >20 分钟且无授权记录 ⇒ ``LongTaskAuthorizationRequired``（退出码 4）；
4. 读数、报告与校验见 :mod:`assembly`。

边界：不写 ``data/paper`` / ``data/runs`` / ``data/models``，不写正式 runs 或 LedgerEntry，
不注册真实候选，不产生前向/SLA 证据。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from src.lazybull.v2.hosts.testbed.d0 import initialize_d0_fresh
from src.lazybull.v2.hosts.testbed.layout import CampaignLayout, layout_for
from src.lazybull.v2.hosts.testbed.paths import (
    assert_write_scope,
    resolve_campaign_root,
    testbed_root,
)
from src.lazybull.v2.hosts.testbed.registration import (
    load_campaign,
    register_arm,
    register_campaign,
)
from src.lazybull.v2.hosts.testbed.rehearsal import artifacts
from src.lazybull.v2.hosts.testbed.rehearsal.hosts import make_injected_host_factory
from src.lazybull.v2.hosts.testbed.rehearsal.injection import (
    InjectionError,
    arm_effect_plan,
    window_mask_from_frozen,
)
from src.lazybull.v2.hosts.testbed.rehearsal.manifest import (
    MANIFEST_FILENAME,
    ManifestError,
    build_manifest_document,
    calibrator_identity,
    file_sha256,
    freeze_manifest,
    manifest_sha256,
    repo_manifest_path,
)
from src.lazybull.v2.hosts.testbed.rehearsal.sections import ARM_ROLES
from src.lazybull.v2.hosts.testbed.rehearsal.sources import (
    SourcesError,
    anchor_parameters,
    frozen_windows_from_t5,
    production_flat_config,
    production_model_train_end,
    require_repo_manifest,
    resolve_structural_window,
    resolve_trading_calendar,
    resolve_upstream,
)
from src.lazybull.v2.hosts.testbed.runner import (
    LONG_TASK_THRESHOLD_SECONDS,
    run_virtual_campaign,
)

__all__ = [
    "LongTaskAuthorizationRequired",
    "RehearsalError",
    "append_machine_time",
    "campaign_root_for",
    "freeze_rehearsal",
    "load_authorization",
    "project_batch_seconds",
    "register_rehearsal",
    "rehearsal_layout",
    "run_rehearsal",
    "verify_rehearsal_window",
]

#: 五臂固定顺序（来自 manifest 分节；运行中不得改名）
_ARM_ORDER = tuple(sorted(ARM_ROLES))


class RehearsalError(SourcesError):
    """演练编排非法（窗口不一致、campaign 缺失、产物越界）。"""

    __test__ = False


class LongTaskAuthorizationRequired(RehearsalError):
    """单批预计超过 20 分钟且缺少用户授权记录（设计 §9）。"""

    __test__ = False

    def __init__(self, projection: Mapping[str, Any]) -> None:
        self.projection = dict(projection)
        super().__init__(
            "单批预计耗时超过 20 分钟阈值，需用户授权后继续: "
            f"projected={projection.get('projected_seconds')}s basis={projection.get('basis')}"
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def campaign_root_for(campaign_id: str) -> Path:
    """试车场根下的 campaign 根（写域守卫在注册时再校验一次）。"""
    return resolve_campaign_root(campaign_id, base_root=str(testbed_root()))


def freeze_rehearsal(
    *,
    repo_root: str | Path,
    campaign_id: str,
    run_id: str,
    data_root: Optional[str] = None,
    apply: bool = False,
    suffix: str = "",
    lineage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """冻结 T6 manifest（仓库件；``apply=False`` 只算不写）。

    ``suffix``/``lineage`` 仅供重冻（必须换新 sequence 并登记前身哈希与原因）。
    """
    root = Path(repo_root)
    resolved_data = str((Path(data_root) if data_root else root / "data").resolve())
    campaign_root = campaign_root_for(campaign_id)
    window = resolve_structural_window(
        resolved_data, model_train_end=production_model_train_end(root)
    )
    upstream = resolve_upstream(root)
    frozen = frozen_windows_from_t5(root)
    started = time.perf_counter()
    parameters = anchor_parameters(root, frozen["frozen_list"])
    doc = build_manifest_document(
        run_id=str(run_id),
        created_at=datetime.now().strftime("%Y-%m-%d"),
        campaign_id=str(campaign_id),
        campaign_root=campaign_root.as_posix(),
        base_root=testbed_root().as_posix(),
        data_root=resolved_data,
        days=window["days"],
        watermark=window["watermark"],
        selection_rule=window["selection_rule"],
        frozen_windows=frozen["frozen_list"],
        frozen_list_sha256=frozen["sha256"],
        parameters={
            "dd_improve": parameters["dd_improve"],
            "dd_worsen": parameters["dd_worsen"],
            "dd_improve_target_pp": parameters["targets_pp"]["dd_improve"],
            "dd_worsen_target_pp": parameters["targets_pp"]["dd_worsen"],
        },
        upstream=upstream,
        lineage=lineage,
        solver={
            "targets_pp": dict(parameters["targets_pp"]),
            "anchor_thresholds_pp": dict(parameters["anchor_thresholds_pp"]),
            "target_derivation": str(parameters["target_derivation"]),
            "solver_semantics": (
                "T5 求解器语义：101 点扫描 + ≤80 次细化 + 0.05pp 容差 + 多解取最小干预"
            ),
            "solve_source": "frozen_B0_anchor_window（盲态；不读演练读数）",
        },
    )
    outcome: Dict[str, Any] = {
        "dry_run": not apply,
        "campaign_root": campaign_root.as_posix(),
        "window_days": int(window["day_count"]),
        "return_days": int(window["return_days"]),
        "window_first": window["d0_date"],
        "window_last": window["watermark"],
        "parameters": {
            key: parameters[key]
            for key in (
                "dd_improve",
                "dd_worsen",
                "dd_improve_status",
                "dd_worsen_status",
                "dd_improve_pure_pp",
                "dd_worsen_pure_pp",
            )
        },
        "calibrator_sha256": calibrator_identity()["sha256"],
        "seconds": time.perf_counter() - started,
    }
    if not apply:
        outcome["target"] = repo_manifest_path(root, doc["created_at"], suffix).as_posix()
        return outcome
    target = repo_manifest_path(root, doc["created_at"], suffix)
    outcome.update(freeze_manifest(doc, target))
    outcome["doc"] = doc
    outcome["lineage_sequence"] = int(doc["lineage"]["sequence"])
    return outcome


def register_rehearsal(
    *,
    repo_root: str | Path,
    campaign_id: str,
    allow_external_root: bool = False,
) -> Dict[str, Any]:
    """登记 campaign + 五臂 + D0 fresh（同一输入根 / 同一 D0 / 固定 replay）。"""
    root = Path(repo_root)
    campaign_root = campaign_root_for(campaign_id)
    if campaign_root.exists():
        raise RehearsalError(f"campaign 根已存在（禁止复用或覆盖）: {campaign_root}")
    manifest_path = require_repo_manifest(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    window = manifest["window"]
    data_root = manifest["campaign"]["data_root"]
    calendar = resolve_trading_calendar(data_root)
    started = time.perf_counter()
    register_campaign(
        campaign_id=campaign_id,
        d0_date=str(window["d0_date"]),
        virtual_days=list(window["virtual_days"]),
        calendar=calendar,
        data_root=data_root,
        allow_external_root=allow_external_root,
    )
    flat = production_flat_config(root)
    arms: Dict[str, str] = {}
    for arm_id in _ARM_ORDER:
        record = register_arm(
            campaign_root=campaign_root,
            arm_id=arm_id,
            arm_role=str(ARM_ROLES[arm_id]),
            config_flat=flat,
            data_root=data_root,
            note=str(manifest["arms"][arm_id]["note"]),
            allow_external_root=allow_external_root,
        )
        arms[arm_id] = str(record["identity"]["model"]["content_fingerprint"])
    d0 = initialize_d0_fresh(campaign_root, allow_external_root=allow_external_root)
    copy = campaign_root / "reports" / MANIFEST_FILENAME
    copy.parent.mkdir(parents=True, exist_ok=True)
    if not copy.is_file():
        copy.write_bytes(manifest_path.read_bytes())
    if copy.read_bytes() != manifest_path.read_bytes():
        raise ManifestError("campaign manifest 副本与仓库冻结件不一致（拒绝注册）")
    before_path = artifacts.artifact_path(campaign_root, artifacts.ARTIFACT_PRODUCTION_BEFORE)
    if not before_path.is_file():
        from src.lazybull.v2.hosts.testbed.rehearsal.sources import (
            collect_production_fingerprints,
        )

        artifacts.write_json_once(
            before_path,
            {
                "stage": "register",
                "captured_at": _now(),
                "snapshot": collect_production_fingerprints(root),
                "window": "register → report（跨阶段）",
                "note": "生产根零写入的 before 快照；报告阶段与 after 比对",
            },
            what="生产根 before 快照",
        )
    return {
        "campaign_root": campaign_root.as_posix(),
        "arms": arms,
        "d0_record_sha256": str(d0["record_sha256"]),
        "calendar_days": len(calendar),
        "production_before": before_path.as_posix(),
        "seconds": time.perf_counter() - started,
    }


def project_batch_seconds(per_day_seconds: float, day_count: int) -> Dict[str, Any]:
    """按单日实测外推整批墙钟（授权门唯一依据；不缩量规避）。"""
    per_day = float(per_day_seconds)
    days = max(1, int(day_count))
    projected = per_day * days
    return {
        "per_day_seconds": per_day,
        "virtual_days": days,
        "projected_seconds": projected,
        "threshold_seconds": LONG_TASK_THRESHOLD_SECONDS,
        "exceeds_threshold": bool(projected > LONG_TASK_THRESHOLD_SECONDS),
        "basis": (
            f"单日实测 {per_day:.3f}s（五臂族级）× {days} 日；" "注册/D0/报告阶段单列，不计入该外推"
        ),
    }


def load_authorization(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """读取用户长任务授权记录（缺省 None ⇒ 未授权）。"""
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        raise RehearsalError(f"授权记录文件不存在: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    for key in ("authorized_at", "authorized_by", "quote", "basis", "scope"):
        if key not in payload:
            raise RehearsalError(f"授权记录缺字段 {key!r}: {target}")
    scope = payload.get("scope") or {}
    for key in ("campaign_id", "max_virtual_days"):
        if key not in scope:
            raise RehearsalError(f"授权记录 scope 缺字段 {key!r}: {target}")
    if not str(scope.get("campaign_id") or "").strip():
        raise RehearsalError(f"授权记录 scope.campaign_id 不得为空: {target}")
    if int(scope.get("max_virtual_days") or 0) < 1:
        raise RehearsalError(f"授权记录 scope.max_virtual_days 必须 >= 1: {target}")
    return {**payload, "path": target.as_posix(), "sha256": file_sha256(target)}


def append_machine_time(path: Path, payload: Mapping[str, Any]) -> Path:
    """追加墙钟登记（非确定性产物；不入任何身份哈希）。"""
    target = Path(path)
    existing: Dict[str, Any] = {}
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
    history = list(existing.get("history") or [])
    history.append(dict(payload))
    merged = {**existing, **dict(payload), "history": history}
    batches = [
        {"stage": entry.get("stage"), "days": len(entry.get("days") or [])} for entry in history
    ]
    merged["virtual_days_total"] = sum(item["days"] for item in batches)
    merged["batches"] = batches
    merged["top_level_semantics"] = (
        "顶层键（days/per_day_seconds/batch_seconds）是**最后一次**批次的值；"
        "整 run 口径看 virtual_days_total/batches 与 history"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def _mask_table(document: Mapping[str, Any], return_days: int) -> Dict[str, Any]:
    """DD 臂的窗口掩码（窗越界时不给掩码，由该臂 ``not_constructible`` 承担）。"""
    if return_days <= 0:
        return {}
    frozen = list(document["injection_window"]["frozen_list"])
    try:
        mask, _ = window_mask_from_frozen(frozen, return_days)
    except InjectionError:  # 窗越界属预期分支 ⇒ 该臂 not_constructible，无掩码
        return {}
    return {
        arm: mask
        for arm, payload in dict(document["arms"]).items()
        if str((payload.get("injection") or {}).get("kind")) == "crash_window_transform"
    }


def _default_factory(document: Mapping[str, Any]):
    from src.lazybull.v2.hosts.testbed.arm_runtime import make_default_host_factory

    days = list(document["window"]["virtual_days"])
    return make_default_host_factory(
        data_root=str(document["campaign"]["data_root"]),
        replay_calendar=[pd.Timestamp(day) for day in days],
    )


def run_rehearsal(
    *,
    repo_root: str | Path,
    campaign_id: str,
    days: Optional[Sequence[str]] = None,
    authorization: Optional[Mapping[str, Any]] = None,
    allow_external_root: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """在已登记 campaign 上逐日推进（宿主层注入；超阈值需授权）。"""
    campaign_root = campaign_root_for(campaign_id)
    assert_write_scope(campaign_root, allow_external_root=allow_external_root)
    manifest_path = campaign_root / "reports" / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RehearsalError(f"campaign 缺冻结 manifest（先执行 freeze --apply）: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return_days = int(manifest["window"]["return_days"])
    effects = arm_effect_plan(manifest, return_days)
    factory = make_injected_host_factory(
        _default_factory(manifest),
        effects=effects,
        masks=_mask_table(manifest, return_days),
    )
    if dry_run:
        return {
            "dry_run": True,
            "campaign_root": campaign_root.as_posix(),
            "effects": {arm: effect.to_dict() for arm, effect in sorted(effects.items())},
        }
    # §9 授权门：**跑前**按已登记的单日基线 × 本次待跑日数判定（不缩量、不延后判定）
    pending_days = days if days is not None else manifest["window"]["virtual_days"]
    pending = [str(day) for day in pending_days]
    if authorization is not None:
        _assert_authorization_scope(authorization, campaign_id, len(pending))
    baseline = _per_day_baseline(campaign_root)
    if baseline is None:
        if len(pending) > 1:
            if authorization is None:
                raise LongTaskAuthorizationRequired(
                    {
                        "per_day_seconds": None,
                        "virtual_days": len(pending),
                        "projected_seconds": None,
                        "threshold_seconds": LONG_TASK_THRESHOLD_SECONDS,
                        "exceeds_threshold": True,
                        "basis": (
                            "无已登记的单日基线 ⇒ 先跑 --days 1 探针并按实测外推，"
                            "或提供覆盖本次批次的用户授权记录"
                        ),
                    }
                )
            baseline = {
                "per_day_seconds": None,
                "basis": "无历史基线 + 用户授权覆盖本次批次（不入外推）",
            }
            projection = {
                "per_day_seconds": None,
                "virtual_days": len(pending),
                "projected_seconds": None,
                "threshold_seconds": LONG_TASK_THRESHOLD_SECONDS,
                "exceeds_threshold": False,
                "basis": baseline["basis"],
            }
        else:
            projection = {
                "per_day_seconds": None,
                "virtual_days": 1,
                "projected_seconds": None,
                "threshold_seconds": LONG_TASK_THRESHOLD_SECONDS,
                "exceeds_threshold": False,
                "basis": "单日探针（无历史基线，本次即基线）",
            }
    else:
        projection = project_batch_seconds(baseline["per_day_seconds"], len(pending))
        if projection["exceeds_threshold"] and authorization is None:
            raise LongTaskAuthorizationRequired(projection)
    started = time.perf_counter()
    report = run_virtual_campaign(
        campaign_root,
        days=list(days) if days is not None else None,
        host_factory=factory,
        write_report=False,
        allow_external_root=allow_external_root,
    )
    seconds = time.perf_counter() - started
    per_day = dict(report["machine_time"]["per_day_seconds"])
    append_machine_time(
        artifacts.artifact_path(campaign_root, artifacts.ARTIFACT_MACHINE_TIME_RUN),
        {
            "stage": "run",
            "measured_at": _now(),
            "days": list(per_day),
            "per_day_seconds": per_day,
            "batch_seconds": seconds,
            "projection": {
                **projection,
                "basis": (baseline["basis"] if baseline else "单日探针（无历史基线，本次即基线）"),
            },
            "authorization_sha256": None if authorization is None else authorization.get("sha256"),
        },
    )
    _assert_injection_effective(campaign_root, manifest, effects, per_day)
    return {
        "campaign_root": campaign_root.as_posix(),
        "published_generation": int(report["published"]["published_generation"]),
        "per_day_seconds": per_day,
        "batch_seconds": seconds,
        "projection": projection,
        "effects": {arm: effect.to_dict() for arm, effect in sorted(effects.items())},
        "long_task_authorization": None if authorization is None else dict(authorization),
        "aa_consistency_field": dict(report["aa_consistency"]),
    }


def _per_day_baseline(campaign_root: Path) -> Optional[Dict[str, Any]]:
    """跑前单日基线：取本 campaign 已登记的逐日墙钟均值（无历史 ⇒ ``None``，需先跑探针）。"""
    path = artifacts.artifact_path(Path(campaign_root), artifacts.ARTIFACT_MACHINE_TIME_RUN)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history") or [payload]
    values: List[float] = []
    for entry in history:
        values.extend(float(value) for value in (entry.get("per_day_seconds") or {}).values())
    if not values:
        return None
    mean = float(sum(values) / len(values))
    return {
        "per_day_seconds": mean,
        "basis": f"跑前基线 = 已登记 {len(values)} 个虚拟日的逐日均值 {mean:.3f}s（五臂族级）",
    }


def _assert_authorization_scope(
    authorization: Mapping[str, Any], campaign_id: str, pending_days: int
) -> None:
    """授权必须覆盖本次 campaign 与本次批次日数（一份授权不得跨 campaign/跨批量复用）。"""
    scope = dict(authorization.get("scope") or {})
    granted_campaign = str(scope.get("campaign_id") or "").strip()
    if not granted_campaign or granted_campaign != campaign_id:
        raise RehearsalError(
            f"授权记录的 campaign_id={granted_campaign!r} 与本次 {campaign_id!r} 不符（拒绝运行）"
        )
    max_days = int(scope.get("max_virtual_days") or 0)
    if pending_days > max_days:
        raise LongTaskAuthorizationRequired(
            {
                "per_day_seconds": None,
                "virtual_days": pending_days,
                "projected_seconds": None,
                "threshold_seconds": LONG_TASK_THRESHOLD_SECONDS,
                "exceeds_threshold": True,
                "basis": (
                    f"授权仅覆盖 {max_days} 个虚拟日，本次待跑 {pending_days} 日 ⇒ 需重新授权"
                ),
            }
        )


def _assert_injection_effective(
    campaign_root: Path,
    manifest: Mapping[str, Any],
    effects: Mapping[str, Any],
    per_day: Mapping[str, Any],
) -> None:
    """可构造臂的注入必须真的改变 NAV（禁止静默退化成克隆）。"""
    from src.lazybull.v2.hosts.testbed.layout import layout_for
    from src.lazybull.v2.hosts.testbed.rehearsal.aa import load_snapshots

    from src.lazybull.v2.hosts.testbed.journal import read_index

    layout = layout_for(Path(campaign_root))
    if read_index(layout) is None:  # 尚无已发布 index ⇒ 无账本可判，交由报告阶段报错
        return
    base = {snap.virtual_date: snap.nav for snap in load_snapshots(layout, "base")}
    frozen = list((manifest.get("injection_window") or {}).get("frozen_list") or [])
    first_window = min((int(item["window_start_ordinal"]) for item in frozen), default=0)
    for arm_id, effect in effects.items():
        if not effect.active or effect.kind != "crash_window_transform":
            continue
        if not per_day:
            continue
        arm = {snap.virtual_date: snap.nav for snap in load_snapshots(layout, arm_id)}
        shared = sorted(set(base) & set(arm))
        # 只在本批**已覆盖注入窗序号**时才判“静默克隆”（否则小批尚未触及窗，克隆属预期）
        # shared[0] 为 D0（无收益）；shared[i] 的收益序号 = i（1 起）
        covered = [day for ordinal, day in enumerate(shared) if ordinal >= first_window]
        if covered and all(arm[day] == base[day] for day in covered):
            raise RehearsalError(
                f"臂 {arm_id!r} 声明 constructible 但注入窗内 NAV 与 base 完全相同（静默克隆）"
            )


def rehearsal_layout(repo_root: str | Path, campaign_id: str) -> CampaignLayout:
    """campaign 布局（只读）。"""
    return layout_for(campaign_root_for(campaign_id))


def verify_rehearsal_window(repo_root: str | Path, campaign_id: str) -> Dict[str, Any]:
    """校验 campaign 窗口与冻结窗口逐日一致（防止窗口被改写或删减）。"""
    root = campaign_root_for(campaign_id)
    campaign = load_campaign(root)
    manifest_path = root / "reports" / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = [str(day) for day in manifest["window"]["virtual_days"]]
    actual = [str(day) for day in campaign["clock"]["days"]]
    if recorded != actual:
        raise RehearsalError(
            "campaign 虚拟窗口与冻结窗口不一致（禁止事后删减或改写）: "
            f"{len(actual)} vs {len(recorded)}"
        )
    repo_path = require_repo_manifest(Path(repo_root))
    if manifest_sha256(manifest_path) != manifest_sha256(repo_path):
        raise ManifestError("campaign manifest 与仓库冻结件不一致（拒绝运行）")
    return {
        "virtual_days": len(actual),
        "window_state_id": (campaign["identity"]["window_state"]["window_state_id"]),
    }
