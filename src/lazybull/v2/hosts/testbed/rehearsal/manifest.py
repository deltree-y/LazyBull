# -*- coding: utf-8 -*-
"""P5b T6 演练 manifest：冻结、字段/门槛/谱系校验、上游链接与三环互链。

契约来源：设计 v4 §6.2（唯一构造）、§6.3（验收）、§5.5「H6 数值」行、§7.1/§7.2（产物与报告）、
§8-H1/H6/H7、§9（预算与墙钟授权）。

冻结纪律
========
1. 冻结发生在**任何演练读数之前**：λ/ζ 只在冻结的 B0 锚窗上求（盲态），δ 倍数与目标 pp
   一律取设计 §5.5 的冻结值；
2. 上游链接必须逐字节核验：T4 收益 manifest 与临界值、T5 回撤 manifest / 临界值 / 功效产物，
   任一哈希不符即**拒绝运行**（防止用被改写或替换的仪器读阈值）；
3. 三环互链：``manifest`` ↔ ``calibrator``（身份覆盖的模块 sha） ↔ ``判定表产物``，
   缺任一环或哈希不符即拒绝；
4. 独占创建：已存在即拒绝（重跑必须换新 campaign，禁止覆盖既有证据）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import ANCHOR_HORIZON
from src.lazybull.v2.hosts.testbed.rehearsal.sections import (
    ARM_ROLES,
    LIMITATION_MARKERS,
    WINDOW_MAPPING_MODE,
    anchor_solver_section,
    arms_section,
    campaign_section,
    checklists_section,
    evidence_section,
    h6_section,
    injection_section,
    limitations_section,
    lineage_section,
    upstream_section,
    window_section,
)
from src.lazybull.v2.hosts.testbed.storage import (
    read_json,
    seal_record,
    verify_seal,
)
from src.lazybull.v2.store.row_domain import STRUCTURAL, STRUCTURAL_FROM, row_domain_for_date

__all__ = [
    "CALIBRATOR_MODULES",
    "IMPLEMENTATION_ID",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "ManifestError",
    "build_manifest_document",
    "calibrator_identity",
    "file_sha256",
    "freeze_manifest",
    "load_manifest",
    "manifest_sha256",
    "repo_manifest_path",
    "validate_manifest",
    "verify_chain",
    "verify_upstream",
]

#: manifest schema
MANIFEST_SCHEMA = "p5b_testbed_rehearsal_manifest/v1"
#: campaign ``reports/`` 内的 manifest 副本名
MANIFEST_FILENAME = "rehearsal_manifest.json"
#: 演练器实现编号（写进校准器身份与产物）
IMPLEMENTATION_ID = "p5b_t6_rehearsal/v1"
#: 校准器身份覆盖的模块（``__init__`` 仅导出面，按排除项登记）
CALIBRATOR_MODULES: Tuple[str, ...] = (
    "aa.py",
    "artifacts.py",
    "assembly.py",
    "hosts.py",
    "injection.py",
    "manifest.py",
    "readings.py",
    "report.py",
    "runner.py",
    "sections.py",
    "sources.py",
    "verdict.py",
)
#: 校准器身份排除项（显式登记理由，校验时逐条存在）
CALIBRATOR_EXCLUSIONS: Mapping[str, str] = {
    "__init__.py": "仅导出面，无判据或注入逻辑",
    "testbed/runner.py": "上游共享层（T2 虚拟时钟运行器），由 T2 冻结链负责",
    "testbed/registration.py": "上游共享层（T1 注册），由 T1 冻结链负责",
    "testbed/d0.py": "上游共享层（D0 fresh），由 T1 冻结链负责",
    "testbed/ledger_repair.py": "上游只读查询 ``query_shards`` 的宿主，由 T3 冻结链负责",
    "referee/ledger_source.py": "上游共享层（配对账本读取），由 T4 校准器身份负责",
    "referee/statistics.py": "上游共享层（§5.1 look 映射与注册统计量），由 T4 校准器身份负责",
    "referee/drawdown/injection.py": "T5 注入器与求解器（只读复用），由 T5 校准器身份负责",
    "referee/drawdown/mdd.py": "T5 回撤读数（只读复用），由 T5 校准器身份负责",
}
#: 校验必填字段（点路径）
_REQUIRED_PATHS: Tuple[str, ...] = (
    "schema",
    "run_id",
    "created_at",
    "status",
    "scope",
    "evidence",
    "campaign",
    "window",
    "arms",
    "injection_window",
    "upstream",
    "h6_numeric",
    "checklists",
    "counts",
    "files",
    "limitations",
    "lineage",
)
#: 上游链接的必登记项（六项：含被引用的 T4 holdout —— R2-P5B-06 整改后不再“只记录当前哈希”）
_UPSTREAM_KEYS: Tuple[str, ...] = (
    "t4_manifest",
    "t4_thresholds",
    "t4_holdout",
    "t5_manifest",
    "t5_thresholds",
    "t5_power",
)


class ManifestError(TestbedError):
    """manifest 缺失/字段不全/口径不符/哈希链断裂（一律 fail-closed）。"""

    __test__ = False


def file_sha256(path: Path) -> str:
    """文件字节 sha256（哈希链唯一口径）。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_sha256(path: Path) -> str:
    """manifest 文件字节 sha256（语义化别名）。"""
    return file_sha256(path)


def repo_manifest_path(repo_root: Path, created_at: str, suffix: str = "") -> Path:
    """仓库冻结件路径（``docs/plans/v2/`` 为本地档案惯例目录）。

    ``suffix`` 仅供重冻使用（例如 ``_r2``），首次冻结必须为空 ⇒ 文件名为
    ``p5b_t6_rehearsal_manifest_<日期>.json``。
    """
    name = f"p5b_t6_rehearsal_manifest_{created_at}{suffix}.json"
    return Path(repo_root) / "docs" / "plans" / "v2" / name


def calibrator_identity(
    module_dir: Optional[Path] = None,
    *,
    implementation_id: str = IMPLEMENTATION_ID,
    modules: Sequence[str] = CALIBRATOR_MODULES,
) -> Dict[str, Any]:
    """演练器代码身份 = ``{implementation_id, modules, sha256}``（逐模块字节哈希）。"""
    base = Path(module_dir) if module_dir is not None else Path(__file__).resolve().parent
    digests: Dict[str, str] = {}
    for name in modules:
        path = base / name
        if not path.is_file():
            raise ManifestError(f"演练器模块缺失（无法计算校准器身份）: {path}")
        digests[name] = file_sha256(path)
    payload = {"implementation_id": str(implementation_id), "modules": digests}
    combined = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"implementation_id": str(implementation_id), "modules": digests, "sha256": combined}


def _get(doc: Mapping[str, Any], path: str) -> Any:
    node: Any = doc
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _missing_fields(doc: Mapping[str, Any]) -> List[str]:
    missing: List[str] = []
    for path in _REQUIRED_PATHS:
        try:
            _get(doc, path)
        except KeyError:
            missing.append(path)
    return missing


def _check_evidence(doc: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    expected = {
        "evidence_class": "calibration_only",
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "no_ledger_entry": True,
        "row_domain": STRUCTURAL,
        "schedule_mode": "replay",
    }
    for key, value in expected.items():
        if doc["evidence"].get(key) != value:
            errors.append(f"evidence.{key} 必须是 {value!r}，实际 {doc['evidence'].get(key)!r}")
    if str(doc.get("scope")) != "historical_testbed":
        errors.append(f"scope 必须是 historical_testbed，实际 {doc.get('scope')!r}")
    if str(doc.get("status")) != "frozen_before_evidence":
        errors.append(f"status 必须是 frozen_before_evidence，实际 {doc.get('status')!r}")
    return errors


def _check_campaign(campaign: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(campaign.get("schedule_mode")) != "replay":
        errors.append("campaign.schedule_mode 必须是 replay（禁止切到 live）")
    if str(campaign.get("row_domain")) != STRUCTURAL:
        errors.append("campaign.row_domain 必须是 structural")
    root = str(campaign.get("campaign_root") or "").replace("\\", "/").rstrip("/")
    base = str(campaign.get("base_root") or "").replace("\\", "/").rstrip("/")
    campaign_id = str(campaign.get("campaign_id") or "")
    if root.rsplit("/", 1)[-1] != campaign_id:
        errors.append(
            f"campaign.campaign_root 必须以 campaign_id 结尾: {root!r} vs {campaign_id!r}"
        )
    if root.rsplit("/", 1)[0] != base:
        errors.append(f"campaign.campaign_root 必须是 base_root 的直接子目录: {root!r} vs {base!r}")
    if "p5b_t4" in campaign_id or "p5b_t5" in campaign_id:
        errors.append(f"campaign_id 不得占用 T4/T5 campaign 名: {campaign_id!r}")
    if campaign.get("distinct_from_t4_t5_campaigns") is not True:
        errors.append("campaign.distinct_from_t4_t5_campaigns 必须为 true（禁止占用 T4/T5 目录）")
    return errors


def _check_window(window: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    days = [str(day) for day in window.get("virtual_days") or []]
    if int(window.get("h1_minimum_days", -1)) != 10:
        errors.append("window.h1_minimum_days 必须是 10（H1 下限）")
    if len(days) < 10:
        errors.append(f"window.virtual_days 至少 10 日（H1 下限），实际 {len(days)}")
    if days != sorted(days) or len(set(days)) != len(days):
        errors.append("window.virtual_days 必须严格升序且无重复（禁止时钟回退/跳日）")
    if days and str(window.get("d0_date")) != days[0]:
        errors.append("window.d0_date 必须等于首日（D0 只作估值点）")
    if int(window.get("day_count", -1)) != len(days):
        errors.append("window.day_count 与 virtual_days 长度不符")
    if int(window.get("return_days", -1)) != max(0, len(days) - 1):
        errors.append("window.return_days 必须等于 day_count-1")
    bad = [day for day in days if row_domain_for_date(day) != STRUCTURAL]
    if bad:
        errors.append(f"window 含行域非 structural 的虚拟日（边界 {STRUCTURAL_FROM}）: {bad[:3]}")
    if window.get("deletion_forbidden") is not True:
        errors.append("window.deletion_forbidden 必须为 true（不得删减日期）")
    return errors


def _check_arms(arms: Mapping[str, Any], frozen_list: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    if sorted(arms) != sorted(ARM_ROLES):
        errors.append(f"arms 必须恰为五臂 {sorted(ARM_ROLES)}，实际 {sorted(arms)}")
    for arm_id, role in sorted(ARM_ROLES.items()):
        payload = arms.get(arm_id) or {}
        if str(payload.get("role")) != role:
            errors.append(f"arms.{arm_id}.role 必须是 {role}，实际 {payload.get('role')!r}")
    base = dict((arms.get("base") or {}).get("injection") or {})
    clone = dict((arms.get("mech_aa") or {}).get("injection") or {})
    if base != clone or base.get("kind") != "none":
        errors.append("base 与 mech_aa 必须同为无注入臂（机械 A/A 只验逐值一致）")
    errors.extend(
        _check_income_arm(dict((arms.get("income_minus_5d") or {}).get("injection") or {}))
    )
    errors.extend(
        _check_window_arm(dict((arms.get("dd_improve_12pp") or {}).get("injection") or {}))
    )
    errors.extend(
        _check_window_arm(dict((arms.get("dd_worsen_12pp") or {}).get("injection") or {}))
    )
    if not frozen_list:
        errors.append("injection_window.frozen_list 不能为空（回撤臂无窗即不可构造）")
    return errors


def _check_income_arm(spec: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(spec.get("kind")) != "income_log_shift":
        errors.append("income_minus_5d.injection.kind 必须是 income_log_shift")
    if float(spec.get("delta_multiple", -1.0)) != 5.0:
        errors.append(f"收益臂位移倍数必须是冻结值 5.0，实际 {spec.get('delta_multiple')!r}")
    return errors


def _check_window_arm(spec: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(spec.get("kind")) != "crash_window_transform":
        errors.append("回撤臂 injection.kind 必须是 crash_window_transform")
    if int(spec.get("anchor", -1)) != ANCHOR_HORIZON:
        errors.append(f"回撤臂锚必须是 {ANCHOR_HORIZON}，实际 {spec.get('anchor')!r}")
    direction = str(spec.get("direction"))
    target = float(spec.get("target_pp", 0.0))
    # 目标由预登记推导（DD-I: max(12, c^I_anchor+2)；DD-W: min(-12, c^W_anchor-2)）：
    # 只校验方向与量级区间，具体数值由 anchor_solver.targets_pp 与本字段互证。
    if direction == "dd_improve" and not (0.0 < target <= 40.0):
        errors.append(f"{direction} 目标必须为正且 ≤40pp，实际 {target}")
    if direction == "dd_worsen" and not (-40.0 <= target < 0.0):
        errors.append(f"{direction} 目标必须为负且 ≥-40pp，实际 {target}")
    parameter = spec.get("parameter")
    if parameter is not None:
        value = float(parameter)
        low, high = (0.0, 1.0) if direction == "dd_improve" else (0.0, 2.0)
        if not (low <= value <= high):
            errors.append(f"{direction} 参数必须落在 [{low},{high}]，实际 {value}")
    return errors


def _check_injection_window(section: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(section.get("mapping_mode")) != WINDOW_MAPPING_MODE:
        errors.append(f"injection_window.mapping_mode 必须是 {WINDOW_MAPPING_MODE}")
    for key in ("reselect_forbidden", "truncation_forbidden"):
        if section.get(key) is not True:
            errors.append(f"injection_window.{key} 必须为 true")
    for entry in section.get("frozen_list") or []:
        start = int(entry["window_start_ordinal"])
        stop = int(entry["window_stop_ordinal"])
        if stop <= start:
            errors.append(f"冻结窗序号非法: {entry}")
    return errors


def _check_upstream(section: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in _UPSTREAM_KEYS:
        payload = section.get(key) or {}
        path = str(payload.get("path") or "")
        digest = str(payload.get("sha256") or "")
        if not path:
            errors.append(f"upstream.{key}.path 缺失")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            errors.append(f"upstream.{key}.sha256 必须是 64 位小写 hex，实际 {digest!r}")
    if section.get("verification") is not True:
        errors.append("upstream.verification 必须为 true（冻结时已核验）")
    return errors


def _check_h6(section: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    items = list(section.get("items") or [])
    names = [str(item.get("name")) for item in items]
    if names != ["income_minus_5d", "dd_improve_12pp", "dd_worsen_12pp"]:
        errors.append(f"h6_numeric.items 必须是三条（收益/DD-I/DD-W），实际 {names}")
    if str(section.get("frequency_source") or "").find("T4/T5") < 0:
        errors.append("h6_numeric.frequency_source 必须显式指向 T4/T5 holdout（禁止自产频率）")
    for key in ("directional_only", "mutual_offset_forbidden"):
        if section.get(key) is not True:
            errors.append(f"h6_numeric.{key} 必须为 true")
    if str(section.get("status")) != "frozen_before_evidence":
        errors.append("h6_numeric.status 必须是 frozen_before_evidence")
    return errors


def _check_checklists(section: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in ("h1", "h6", "h7"):
        if not list(section.get(key) or []):
            errors.append(f"checklists.{key} 不能为空")
    if str(section.get("gate_owner")) != "user_at_merge_acceptance":
        errors.append("checklists.gate_owner 必须是 user_at_merge_acceptance")
    return errors


def _check_limitations(items: Sequence[Any]) -> List[str]:
    errors: List[str] = []
    texts = [str(item) for item in items]
    if len(texts) < 4:
        errors.append(f"limitations 至少 4 条（§5.6 全文口径），实际 {len(texts)}")
    joined = "\n".join(texts)
    for key, marker in sorted(LIMITATION_MARKERS.items()):
        if marker not in joined:
            errors.append(f"limitations 缺少 §5.6 要点 {key!r}（需含 {marker!r}）")
    return errors


def _check_files(section: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    calibrator = section.get("calibrator") or {}
    covered = list(calibrator.get("covered_modules") or [])
    modules = dict(calibrator.get("modules") or {})
    if sorted(covered) != sorted(CALIBRATOR_MODULES):
        errors.append("files.calibrator.covered_modules 与冻结模块清单不符")
    if sorted(modules) != sorted(CALIBRATOR_MODULES):
        errors.append("files.calibrator.modules 与冻结模块清单不符（禁止漏覆盖）")
    if len(str(calibrator.get("sha256") or "")) != 64:
        errors.append("files.calibrator.sha256 必须是 64 位 hex")
    exclusions = dict(calibrator.get("exclusions") or {})
    for key in CALIBRATOR_EXCLUSIONS:
        if key not in exclusions:
            errors.append(f"files.calibrator.exclusions 缺少必登记排除项 {key!r}")
    return errors


def _check_lineage(section: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    sequence = int(section.get("sequence", 0) or 0)
    if sequence < 1:
        errors.append(f"lineage.sequence 必须 >= 1，实际 {section.get('sequence')!r}")
    if sequence == 1:
        for key in ("superseded_manifest_sha256", "superseded_calibrator_sha256"):
            if section.get(key) is not None:
                errors.append(f"lineage.{key} 必须为 null（首次冻结无前身）")
        return errors
    for key in ("superseded_manifest_sha256", "superseded_calibrator_sha256"):
        value = str(section.get(key) or "")
        if len(value) != 64:
            errors.append(f"重冻必须登记 64 位 {key}，实际 {value!r}")
    if not str(section.get("supersede_reason") or "").strip():
        errors.append("重冻必须登记 supersede_reason（说明为何重冻）")
    return errors


def _check_counts(counts: Mapping[str, Any], window: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if int(counts.get("arms", -1)) != len(ARM_ROLES):
        errors.append(f"counts.arms 必须是 {len(ARM_ROLES)}")
    if int(counts.get("virtual_days", -1)) != int(window.get("day_count", -2)):
        errors.append("counts.virtual_days 必须等于 window.day_count")
    if int(counts.get("runs", -1)) != 1:
        errors.append("counts.runs 必须是 1（单次演练；重跑必须换新 campaign）")
    return errors


def validate_manifest(doc: Mapping[str, Any]) -> List[str]:
    """逐节校验 manifest；返回错误清单（空 = 合法）。"""
    missing = _missing_fields(doc)
    if missing:
        return [f"缺必填字段: {path}" for path in missing]
    if str(doc.get("schema")) != MANIFEST_SCHEMA:
        return [f"schema 非法: {doc.get('schema')!r}（期望 {MANIFEST_SCHEMA}）"]
    errors: List[str] = []
    errors.extend(_check_evidence(doc))
    errors.extend(_check_campaign(doc["campaign"]))
    errors.extend(_check_window(doc["window"]))
    errors.extend(_check_arms(doc["arms"], list(doc["injection_window"].get("frozen_list") or [])))
    errors.extend(_check_injection_window(doc["injection_window"]))
    errors.extend(_check_upstream(doc["upstream"]))
    errors.extend(_check_h6(doc["h6_numeric"]))
    errors.extend(_check_checklists(doc["checklists"]))
    errors.extend(_check_files(doc["files"]))
    errors.extend(_check_limitations(doc["limitations"]))
    errors.extend(_check_lineage(doc["lineage"]))
    errors.extend(_check_counts(doc["counts"], doc["window"]))
    return errors


def verify_upstream(doc: Mapping[str, Any], *, repo_root: Path) -> Dict[str, Dict[str, str]]:
    """核验上游链接五项的物理哈希（不符即拒绝运行）。

    Returns:
        ``{key: {path, sha256}}``（已核验）。

    Raises:
        ManifestError: 文件缺失或哈希不符。
    """
    root = Path(repo_root)
    resolved: Dict[str, Dict[str, str]] = {}
    for key in _UPSTREAM_KEYS:
        payload = dict(doc["upstream"][key])
        path = root / str(payload["path"])
        if not path.is_file():
            raise ManifestError(f"上游链接文件不存在（禁止在仪器缺失时运行）: {path}")
        actual = file_sha256(path)
        if actual != str(payload["sha256"]):
            raise ManifestError(
                f"上游链接哈希不符（{key}）：登记={payload['sha256']} 实际={actual}（拒绝运行）"
            )
        resolved[key] = {"path": str(payload["path"]), "sha256": actual}
    frozen = list(doc["injection_window"]["frozen_list"])
    expected = str(doc["injection_window"]["frozen_list_sha256"])
    actual_frozen = _sha256_json(frozen)
    if actual_frozen != expected:
        raise ManifestError(
            f"冻结崩盘窗清单哈希不符：登记={expected} 实际={actual_frozen}（禁止重选窗）"
        )
    return resolved


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_manifest(doc: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    """冻结 manifest 到指定路径（独占创建；dry-run 由调用方控制）。

    Raises:
        ManifestError: 校验不通过或目标已存在。
    """
    errors = validate_manifest(doc)
    if errors:
        raise ManifestError("manifest 不可冻结: " + "; ".join(errors))
    target = Path(path)
    sealed = seal_record(dict(doc))
    text = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ManifestError(f"manifest 已存在（禁止覆盖既有冻结件）: {target}") from exc
    return {
        "path": target.as_posix(),
        "sha256": manifest_sha256(target),
        "record_sha256": str(sealed["record_sha256"]),
    }


def load_manifest(
    path: Path,
    *,
    repo_manifest: Optional[Path] = None,
    verify_calibrator: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """读取并校验 manifest（缺省同时核验校准器身份与仓库副本逐字节一致）。

    Raises:
        ManifestError: 封存/schema/字段/校准器身份/仓库副本不一致。
    """
    target = Path(path)
    if not target.is_file():
        raise ManifestError(f"manifest 不存在（禁止在未冻结状态下运行）: {target}")
    doc = read_json(target, what="T6 演练 manifest")
    verify_seal(doc, what=f"T6 演练 manifest（{target}）")
    errors = validate_manifest(doc)
    if errors:
        raise ManifestError("manifest 校验失败: " + "; ".join(errors))
    if verify_calibrator:
        identity = calibrator_identity()
        recorded = dict(doc["files"]["calibrator"])
        if str(recorded.get("sha256")) != str(identity["sha256"]):
            raise ManifestError(
                "演练器代码态与冻结身份不一致（禁止在改动后沿用旧 manifest）: "
                f"{identity['sha256']} vs {recorded.get('sha256')}"
            )
    if repo_manifest is not None:
        repo_path = Path(repo_manifest)
        if not repo_path.is_file():
            raise ManifestError(f"仓库冻结件不存在（禁止在未冻结状态下运行）: {repo_path}")
        if repo_path.read_bytes() != target.read_bytes():
            raise ManifestError(
                "仓库冻结件与 campaign 副本不是逐字节一致（拒绝运行）: " f"{repo_path} vs {target}"
            )
    return doc, manifest_sha256(target)


def verify_chain(
    *,
    manifest_doc: Mapping[str, Any],
    manifest_file_sha256: str,
    calibrator: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
) -> Dict[str, str]:
    """三环互链校验：manifest ↔ 校准器 ↔ 判定表/报告产物。

    Raises:
        ManifestError: 任一环哈希不符或缺失。
    """
    recorded_calibrator = str(manifest_doc["files"]["calibrator"]["sha256"])
    if recorded_calibrator != str(calibrator["sha256"]):
        raise ManifestError(
            f"三环断裂：manifest 登记的校准器 {recorded_calibrator} ≠ 实际 {calibrator['sha256']}"
        )
    for name, digest in sorted(artifact_hashes.items()):
        if len(str(digest)) != 64:
            raise ManifestError(f"三环断裂：产物 {name} 哈希非法: {digest!r}")
    return {
        "manifest_sha256": str(manifest_file_sha256),
        "calibrator_sha256": str(calibrator["sha256"]),
        **{f"{name}_sha256": str(digest) for name, digest in sorted(artifact_hashes.items())},
    }


def build_manifest_document(
    *,
    run_id: str,
    created_at: str,
    campaign_id: str,
    campaign_root: str,
    base_root: str,
    data_root: str,
    days: Sequence[str],
    watermark: str,
    selection_rule: str,
    frozen_windows: Sequence[Mapping[str, Any]],
    frozen_list_sha256: str,
    parameters: Mapping[str, Optional[float]],
    upstream: Mapping[str, Any],
    calibrator: Optional[Mapping[str, Any]] = None,
    lineage: Optional[Mapping[str, Any]] = None,
    solver: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """构造完整 manifest 文档（不含封存字段；由 :func:`freeze_manifest` 封存）。

    Args:
        run_id: 演练身份。
        created_at: 冻结日期（``YYYY-MM-DD``）。
        campaign_id / campaign_root / base_root / data_root: 演练现场声明。
        days: 冻结的虚拟窗口（含 D0；≥10 日连续 structural 交易日）。
        watermark: 数据水位（登记用）。
        selection_rule: 窗口选取规则（必须与结果无关，写进证据）。
        frozen_windows: T5 冻结崩盘窗清单（原文引用）。
        frozen_list_sha256: 上述清单的规范化哈希。
        parameters: ``{dd_improve: λ|None, dd_worsen: ζ|None}``（盲态预登记）。
        upstream: :func:`sections.upstream_section` 的入参五元组。
        calibrator: 演练器身份（缺省 = 复算）。
    """
    identity = dict(calibrator) if calibrator is not None else calibrator_identity()
    window = window_section(days, selection_rule=selection_rule, watermark=watermark)
    arms = arms_section(parameters)
    return {
        "schema": MANIFEST_SCHEMA,
        "run_id": str(run_id),
        "created_at": str(created_at),
        "status": "frozen_before_evidence",
        "scope": "historical_testbed",
        "evidence": evidence_section(),
        "campaign": campaign_section(
            campaign_id=campaign_id,
            campaign_root=campaign_root,
            data_root=data_root,
            base_root=base_root,
        ),
        "window": window,
        "arms": arms["arms"],
        "arm_renaming_forbidden": True,
        "injection_window": injection_section(
            frozen_windows,
            frozen_list_sha256=frozen_list_sha256,
            source_sha256=str(upstream["t5_manifest"]["sha256"]),
        ),
        "upstream": upstream_section(**dict(upstream)),
        "h6_numeric": h6_section(
            dd_improve_target=float(parameters.get("dd_improve_target_pp", 12.0)),
            dd_worsen_target=float(parameters.get("dd_worsen_target_pp", -12.0)),
        ),
        "checklists": checklists_section(),
        "counts": {
            "virtual_days": int(window["day_count"]),
            "return_days": int(window["return_days"]),
            "arms": int(arms["count"]),
            "runs": 1,
        },
        "files": {
            "calibrator": {
                "implementation_id": identity["implementation_id"],
                "sha256": identity["sha256"],
                "modules": dict(identity["modules"]),
                "covered_modules": list(CALIBRATOR_MODULES),
                "exclusions": dict(CALIBRATOR_EXCLUSIONS),
            }
        },
        "limitations": limitations_section(),
        "lineage": dict(lineage) if lineage is not None else lineage_section(),
        "anchor_solver": (
            dict(solver)
            if solver is not None
            else anchor_solver_section(
                {
                    "dd_improve": parameters.get("dd_improve"),
                    "dd_worsen": parameters.get("dd_worsen"),
                }
            )
        ),
    }
