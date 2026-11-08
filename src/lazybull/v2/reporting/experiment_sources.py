# -*- coding: utf-8 -*-
"""实验报告臂来源适配（P6 T3：规范化 runs 与旧 WF **分路**读取）。

硬口径（设计 §4.2 / §5.3）：

- 两类来源分别适配，**不互相冒充**：``runs`` 走 ``load_runs_batch`` 契约读入桥（loader 负责
  §9 全部硬校验与 ``config_fingerprint`` 重算），``legacy_wf`` 走 ``scripts/compare`` 白名单
  子集（``fold_subset.load_run`` 提供 chain/窗口，``walk_forward_summary_*.csv`` 提供逐折指标）；
- **坏文件不得跳过**：多候选文件、缺必需文件、契约违例一律 visible ``invalid``，不按“最新”
  静默择一（旧 compare loader 的 ``_pick_latest`` 行为在此升级为显式状态）；
- 只读**快照副本**：所有解析都在 ``StableSnapshot`` 的副本上进行，原根只被读取一次字节；
- 本模块不含任何指标公式：链式指标/差值/分折由 ``experiment_metrics`` 调用既有唯一实现。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.compare.fold_subset import RunArtifacts, load_run
from src.lazybull.v2.evidence.runs_loader import RunsBatch, load_runs_batch
from src.lazybull.v2.reporting import status
from src.lazybull.v2.reporting.snapshot import StableSnapshot
from src.lazybull.v2.reporting.sources import verify_chain_rows

__all__ = [
    "ARM_ID_PATTERN",
    "ArmData",
    "ArmSourceError",
    "ArmSpec",
    "FoldData",
    "LEGACY_CHAIN_GLOB",
    "LEGACY_STATE_GLOB",
    "LEGACY_SUMMARY_GLOB",
    "RUNS_FIXED_FILES",
    "RUNS_FOLD_FILES",
    "SOURCE_KINDS",
    "build_arm_specs",
    "capture_arm_files",
    "load_arm",
    "parse_arm_spec",
    "validate_chain_rows",
]

#: 臂标识形态（CSV 行键；限 ASCII 以免下游列名/路径歧义）
ARM_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{0,31}$")
#: 两类来源（闭集）
SOURCE_KINDS: tuple[str, ...] = ("runs", "legacy_wf")
#: 规范化 runs 批次的必需文件（契约 §9：缺失即报错）
RUNS_FIXED_FILES: tuple[str, ...] = ("batch_meta.json", "summary.parquet", "chain_nav.parquet")
#: 逐折目录内的文件名（trades 必需，其余按契约局部降级）
RUNS_FOLD_FILES: tuple[str, ...] = (
    "trades.parquet",
    "daily.parquet",
    "attribution.parquet",
    "topk_detail.parquet",
    "_meta.json",
)
#: 旧 WF 白名单子集（只认这三类固定前缀）
LEGACY_CHAIN_GLOB = "chain_nav_*.csv"
LEGACY_SUMMARY_GLOB = "walk_forward_summary_*.csv"
LEGACY_STATE_GLOB = "data_state_*.json"


class ArmSourceError(ValueError):
    """臂规格/来源不可解析（命令参数问题，归设置错误而非报告状态）。"""


@dataclass(frozen=True)
class ArmSpec:
    """一个显式臂：``arm_id`` + 来源类型 + 根目录 + manifest 根标识。"""

    arm_id: str
    source_kind: str
    path: Path
    root_id: str


@dataclass
class FoldData:
    """单折的可选来源文件（缺文件为 ``None``，由调用方登记降级）。"""

    split_index: int
    trades: pd.DataFrame | None = None
    daily: pd.DataFrame | None = None
    attribution: pd.DataFrame | None = None
    topk_detail: pd.DataFrame | None = None
    missing: list[str] = field(default_factory=list)


@dataclass
class ArmData:
    """一个臂的只读数据面（链/逐折/归因/成本），不含任何派生指标。"""

    spec: ArmSpec
    status: dict[str, Any]
    label: str
    batch_id: str
    host_mode: str | None
    implementation: str
    data_state_id: str | None
    config_fingerprint: str | None
    config: dict[str, Any]
    chain: pd.DataFrame
    summary: pd.DataFrame
    windows: pd.DataFrame
    folds: dict[int, FoldData]
    artifacts: RunArtifacts | None
    source_refs: dict[str, str]
    degradations: list[dict[str, Any]]
    watermarks: dict[str, Any]
    #: 真实相对路径（CSV ``source_ref`` 用；缺省空 dict 时退回语义别名）
    source_paths: dict[str, str] = field(default_factory=dict)
    rows_total: int = 0
    rows_used: int = 0
    split_count: int = 0

    @property
    def usable(self) -> bool:
        """链行全部有效且臂本身不是 invalid（degraded 仍可出指标，但显式可见）。"""
        return self.status.get("report_status") != "invalid" and self.rows_used > 0


def parse_arm_spec(text: str, *, index: int) -> ArmSpec:
    """解析 ``arm_id=kind:path``（kind ∈ {runs, legacy_wf}）。"""
    raw = str(text or "").strip()
    if "=" not in raw:
        raise ArmSourceError(f"臂规格须为 arm_id=kind:path：{text!r}")
    arm_id, target = raw.split("=", 1)
    arm_id = arm_id.strip()
    if not ARM_ID_PATTERN.match(arm_id):
        raise ArmSourceError(f"臂标识形态非法（要求 {ARM_ID_PATTERN.pattern}）：{arm_id!r}")
    if ":" not in target:
        raise ArmSourceError(f"臂来源须为 kind:path（kind ∈ {list(SOURCE_KINDS)}）：{text!r}")
    kind, path_text = target.split(":", 1)
    kind = kind.strip()
    if kind not in SOURCE_KINDS:
        raise ArmSourceError(f"臂来源类型不受支持：{kind!r}（合法 {list(SOURCE_KINDS)}）")
    path = Path(path_text.strip())
    if not str(path):
        raise ArmSourceError(f"臂来源路径为空：{text!r}")
    return ArmSpec(arm_id=arm_id, source_kind=kind, path=path, root_id=f"arm_{index}")


def build_arm_specs(
    raw_specs: Sequence[str], baseline: str
) -> tuple[list[ArmSpec], str]:
    """装配臂清单并确认基线（基线必须出现在清单内且唯一）。"""
    specs = [parse_arm_spec(item, index=index) for index, item in enumerate(raw_specs)]
    if not specs:
        raise ArmSourceError("未提供任何臂（至少需要基线与一个对照臂）")
    ids = [spec.arm_id for spec in specs]
    duplicated = sorted({item for item in ids if ids.count(item) > 1})
    if duplicated:
        raise ArmSourceError(f"臂标识重复：{duplicated}")
    if str(baseline) not in ids:
        raise ArmSourceError(f"基线臂 {baseline!r} 不在臂清单 {ids} 内（必须显式确认基线）")
    return specs, str(baseline)


def _require_root(spec: ArmSpec) -> Path:
    """来源根必须是**本地目录**（受信判定由快照统一负责）。"""
    path = Path(spec.path)
    if not path.is_dir():
        raise ArmSourceError(f"臂 {spec.arm_id} 的来源根不存在或不是目录：{path}")
    return path


def capture_arm_files(snapshot: StableSnapshot, spec: ArmSpec) -> dict[str, Any]:
    """把该臂的**全部**相关文件捕获进快照副本，并返回候选清单（不改原根）。

    返回 ``{"fixed": [...], "folds": {name: [...]}, "legacy": {...}}``：runs 侧登记实际
    捕获的固定文件与逐折文件；legacy 侧登记三类候选文件名（多候选由 ``load_arm`` 判 invalid）。
    """
    plan: dict[str, Any] = {"fixed": [], "folds": {}, "legacy": {}}
    if spec.source_kind == "runs":
        for name in RUNS_FIXED_FILES:
            if snapshot.capture(spec.root_id, name):
                plan["fixed"].append(name)
        for fold_name in snapshot.list_dirs(spec.root_id, "folds"):
            captured: list[str] = []
            for file_name in RUNS_FOLD_FILES:
                if snapshot.capture(spec.root_id, f"folds/{fold_name}/{file_name}"):
                    captured.append(file_name)
            plan["folds"][fold_name] = captured
        return plan

    names = snapshot.list_dir(spec.root_id, "")
    for key, pattern in (
        ("chain", LEGACY_CHAIN_GLOB),
        ("summary", LEGACY_SUMMARY_GLOB),
        ("state", LEGACY_STATE_GLOB),
    ):
        prefix, suffix = pattern.split("*")
        selected = [name for name in names if name.startswith(prefix) and name.endswith(suffix)]
        for name in selected:
            snapshot.capture(spec.root_id, name)
        plan["legacy"][key] = selected
    return plan


def _status(word: str, reason: str | None, *, source_ref: str | None, detail: str) -> dict[str, Any]:
    """状态行（复用闭集构造器，禁止自由文本冒充状态词）。"""
    return status.make_status(word, reason_code=reason, source_ref=source_ref, detail=detail)


def _degradation(field_path: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """降级登记项（manifest ``degradations[]`` 的唯一构造口径）。"""
    return status.degradation_entry(row, field_path=field_path)


def validate_chain_rows(chain: pd.DataFrame) -> dict[str, Any]:
    """链式净值全行门（**唯一实现** = ``sources.verify_chain_rows``）。

    旧 WF 的 ``chain_nav_*.csv`` 用折内序号作 ``date``，故允许序数日期；真实日期仍走
    ``normalize_date``。返回状态行（含 ``rows_total/rows_used/split_count/trading_days``
    与唯一实现算出的 ``metrics``），不在此另写第二套判定。
    """
    columns = [name for name in ("date", "nav", "split_index") if name in chain.columns]
    return verify_chain_rows(
        chain[columns] if columns else chain,
        allow_ordinal_date=True,
    )


def _legacy_single(plan_files: Mapping[str, Any], key: str, arm_id: str) -> tuple[str | None, dict[str, Any] | None]:
    """旧 WF 单候选选择：0 个 ⇒ 缺必需来源，>1 个 ⇒ 拒绝择一（visible invalid）。"""
    names = list(plan_files.get(key) or [])
    if len(names) == 1:
        return names[0], None
    if not names:
        return None, _status(
            "invalid",
            "required_source_missing",
            source_ref=f"arm:{arm_id}",
            detail=f"缺少必需来源文件 {key}（候选 0 个）",
        )
    return None, _status(
        "invalid",
        "multiple_source_files",
        source_ref=f"arm:{arm_id}",
        detail=f"{key} 候选 {len(names)} 个 {names}——拒绝按最新静默择一",
    )


def _runs_folds(arm_id: str, batch: RunsBatch) -> tuple[dict[int, FoldData], list[dict[str, Any]]]:
    """逐折数据面与降级登记（``topk_detail`` 缺失不算臂级降级）。"""
    folds: dict[int, FoldData] = {}
    degradations: list[dict[str, Any]] = []
    for split_index, fold in sorted(batch.folds.items()):
        data = FoldData(
            split_index=split_index,
            trades=fold.trades,
            daily=fold.daily,
            attribution=fold.attribution,
            topk_detail=fold.topk_detail,
        )
        # topk_detail 属“契约可选且报告侧固定降级”的来源：缺失不构成臂级降级；
        # daily/attribution 缺失会使对应展示与 CSV 局部不可用，按契约登记降级。
        for name in ("daily", "attribution"):
            if getattr(fold, name) is None:
                data.missing.append(name)
                row = _status(
                    "degraded",
                    "fold_file_missing",
                    source_ref=f"arm:{arm_id}/folds/split{split_index}",
                    detail=f"该折缺 {name}.parquet——按契约局部降级，不补造",
                )
                degradations.append(_degradation(f"arms.{arm_id}.folds[{split_index}].{name}", row))
        if fold.topk_detail is None:
            data.missing.append("topk_detail")
        folds[split_index] = data
    return folds, degradations


def _fold_directories(plan_files: Mapping[str, Any]) -> dict[int, str]:
    """捕获计划中的**实际**逐折目录名（兼容生产补零 ``split00`` 与夹具 ``split0``）。"""
    mapping: dict[int, str] = {}
    for name in (plan_files.get("folds") or {}):
        match = re.fullmatch(r"split(\d+)", str(name))
        if match:
            mapping[int(match.group(1))] = str(name)
    return mapping


def _load_runs(
    snapshot: StableSnapshot, spec: ArmSpec, plan_files: Mapping[str, Any]
) -> ArmData:
    """规范化 runs：``load_runs_batch`` 契约读入（loader 负责指纹/链/折集合校验）。"""
    empty = _empty_frame()
    try:
        batch: RunsBatch = load_runs_batch(snapshot.root_copy(spec.root_id))
    except FileNotFoundError as exc:
        return _invalid_arm(spec, "required_source_missing", f"runs 必需文件缺失：{exc}")
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        return _invalid_arm(
            spec, "source_contract_violation", f"runs 契约校验失败：{type(exc).__name__}: {exc}"
        )
    dirs = _fold_directories(plan_files)
    frame = batch.chain_nav.copy()
    frame["split_index"] = pd.to_numeric(frame["split_index"], errors="raise").astype(int)
    frame = frame.reset_index(drop=True)
    frame["source_row_index"] = frame.groupby("split_index").cumcount()
    folds, degradations = _runs_folds(spec.arm_id, batch)
    artifacts = batch.to_run_artifacts()
    artifacts = RunArtifacts(
        label=spec.arm_id,
        directory=artifacts.directory,
        chain=artifacts.chain,
        windows=artifacts.windows,
        data_state_id=artifacts.data_state_id,
        chain_path=None,
    )
    data_state = dict(batch.batch_meta.get("data_state") or {})
    code_state = dict(batch.batch_meta.get("code_state") or {})
    if degradations:
        arm_status = _status(
            "degraded",
            "fold_file_missing",
            source_ref=f"arm:{spec.arm_id}",
            detail=f"runs 契约读入通过，但 {len(degradations)} 个可选项下载降级（归因/逐日/TopK）",
        )
    else:
        arm_status = _status(
            "available", None, source_ref=f"arm:{spec.arm_id}", detail="runs 契约读入通过"
        )
    return ArmData(
        spec=spec,
        status=arm_status,
        label=spec.arm_id,
        batch_id=str(batch.batch_id),
        host_mode=str(batch.batch_meta.get("host_mode") or ""),
        implementation="runs",
        data_state_id=str(data_state.get("data_state_id") or "") or None,
        config_fingerprint=str(batch.batch_meta.get("config_fingerprint") or "") or None,
        config=dict(batch.batch_meta.get("config") or {}),
        chain=frame if not frame.empty else empty,
        summary=batch.summary.copy(),
        windows=artifacts.windows.copy(),
        folds=folds,
        artifacts=artifacts,
        source_refs={},
        degradations=degradations,
        watermarks=_watermarks(data_state, code_state),
        source_paths={
            "summary": "summary.parquet",
            "chain_nav": "chain_nav.parquet",
            **{
                f"attribution:{int(split)}": (
                    f"folds/{dirs.get(int(split), f'split{int(split)}')}/attribution.parquet"
                )
                for split in folds
            },
        },
    )


def _load_legacy(
    snapshot: StableSnapshot, spec: ArmSpec, plan_files: Mapping[str, Any]
) -> ArmData:
    """旧 WF：白名单子集读取；多候选/缺文件一律 visible invalid（不跳过坏文件）。"""
    chain_name, chain_row = _legacy_single(plan_files, "chain", spec.arm_id)
    summary_name, summary_row = _legacy_single(plan_files, "summary", spec.arm_id)
    for row in (chain_row, summary_row):
        if row is not None:
            return _invalid_arm(spec, str(row["degradation_reason_code"]), str(row["detail"]))
    copy_root = snapshot.root_copy(spec.root_id)
    # 数据态候选先校验（在调用旧 loader 之前）：旧 loader 会读末件且对根类型不做保护，
    # 必须先于任何读取把它收敛为 visible invalid（R3-03）。
    state_names = list(plan_files.get("state") or [])
    if len(state_names) > 1:
        return _invalid_arm(
            spec,
            "multiple_source_files",
            f"数据态文件多候选（{sorted(state_names)}）：拒绝按排序首件择一",
        )
    state: dict[str, Any] = {}
    if state_names:
        try:
            parsed = json.loads((copy_root / str(state_names[0])).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            # 损坏与缺失分开处理：坏语法/坏编码都是 visible invalid，不得退回 {} 后继续配对
            # （R3-01/R3-03；UnicodeDecodeError 不属于 OSError/JSONDecodeError，需显式捕获）
            return _invalid_arm(
                spec,
                "invalid_json",
                f"数据态文件损坏：{state_names[0]}（{type(exc).__name__}）",
            )
        if not isinstance(parsed, dict):
            return _invalid_arm(
                spec,
                "invalid_json",
                f"数据态文件根类型不是对象：{state_names[0]}（{type(parsed).__name__}）",
            )
        state = parsed
    try:
        artifacts = load_run(copy_root, label=spec.arm_id)
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        return _invalid_arm(
            spec, "source_contract_violation", f"旧 WF 读取失败：{type(exc).__name__}: {exc}"
        )
    chain = artifacts.chain.copy()
    chain["split_index"] = pd.to_numeric(chain["split_index"], errors="raise").astype(int)
    chain = chain.reset_index(drop=True)
    chain["source_row_index"] = chain.groupby("split_index").cumcount()
    summary = pd.read_csv(copy_root / str(summary_name), encoding="utf-8-sig")
    summary["split_index"] = pd.to_numeric(summary["split_index"], errors="raise").astype(int)
    params = _legacy_config(summary)
    row = _status(
        "degraded",
        "no_config_fingerprint",
        source_ref=f"arm:{spec.arm_id}/{summary_name}",
        detail="旧 WF 无合法配置指纹来源（不另造 canonical 指纹）；臂指标仍可用",
    )
    degradations = [_degradation(f"arms.{spec.arm_id}.config_fingerprint", row)]
    folds = {
        int(split): FoldData(split_index=int(split))
        for split in sorted(set(chain["split_index"].tolist()))
    }
    for fold in folds.values():
        fold.missing.extend(["trades", "daily", "attribution", "topk_detail"])
    state_id = str(state.get("data_state_id") or "").strip() or None
    artifacts = RunArtifacts(
        label=spec.arm_id,
        directory=artifacts.directory,
        chain=artifacts.chain,
        windows=artifacts.windows,
        # 全部读侧消费同一份已校验的数据态文件（不让 loader 的首/末件选择覆盖，R3-03）
        data_state_id=state_id,
        chain_path=artifacts.chain_path,
    )
    return ArmData(
        spec=spec,
        status=row,
        label=spec.arm_id,
        batch_id=str(state.get("batch_run_id") or spec.path.name),
        host_mode="backtest",
        implementation="legacy_wf",
        data_state_id=artifacts.data_state_id,
        config_fingerprint=None,
        config=params,
        chain=chain,
        summary=summary,
        windows=artifacts.windows.copy(),
        folds=folds,
        artifacts=artifacts,
        source_refs={},
        degradations=degradations,
        watermarks=_watermarks(state, state),
        source_paths={
            "summary": str(summary_name),
            "chain_nav": Path(str(artifacts.chain_path or chain_name)).name,
        },
    )


def _legacy_config(summary: pd.DataFrame) -> dict[str, Any]:
    """旧 WF 参数：取 summary 首行的参数字段（动态/统计输出由稳定性层统一剔除）。"""
    if summary.empty:
        return {}
    first = summary.iloc[0]
    config: dict[str, Any] = {}
    for column in summary.columns:
        if column == "split_index":
            continue
        value = first[column]
        config[str(column)] = None if pd.isna(value) else value
    return config


def _watermarks(data_state: Mapping[str, Any], code_state: Mapping[str, Any]) -> dict[str, Any]:
    """数据态/代码态水位（只取原值，缺失写 ``None``，不按 mtime 推）。"""
    partitions = dict(data_state.get("raw_latest_partitions") or {})
    coverage = data_state.get("dividend_coverage")
    if isinstance(coverage, Mapping):
        coverage_value: Any = dict(coverage)
    else:
        coverage_value = coverage
    return {
        "data_state_id": data_state.get("data_state_id"),
        "git_commit": code_state.get("git_commit") or data_state.get("git_commit"),
        "git_dirty": code_state.get("git_dirty", data_state.get("git_dirty")),
        "data_daily_latest": partitions.get("daily"),
        "data_cs_train_latest": data_state.get("features_cs_train_latest"),
        "data_dividend_coverage": coverage_value,
    }


def _empty_frame() -> pd.DataFrame:
    """空链帧（列集固定，便于下游统一判空）。"""
    return pd.DataFrame({"split_index": [], "date": [], "nav": [], "source_row_index": []})


def _invalid_arm(spec: ArmSpec, reason: str, detail: str) -> ArmData:
    """构造不可用臂（链空、无 artifacts；CSV 仍保留该臂行并标 invalid）。"""
    row = _status("invalid", reason, source_ref=f"arm:{spec.arm_id}", detail=detail)
    return ArmData(
        spec=spec,
        status=row,
        label=spec.arm_id,
        batch_id=spec.path.name,
        host_mode=None,
        implementation=spec.source_kind,
        data_state_id=None,
        config_fingerprint=None,
        config={},
        chain=_empty_frame(),
        summary=pd.DataFrame(),
        windows=pd.DataFrame(),
        folds={},
        artifacts=None,
        source_refs={},
        degradations=[_degradation(f"arms.{spec.arm_id}", row)],
        watermarks=_watermarks({}, {}),
    )


def load_arm(
    snapshot: StableSnapshot, spec: ArmSpec, plan: Mapping[str, Any] | None = None
) -> ArmData:
    """读入一个臂（runs/legacy 分路），并完成链行逐行校验与折集合登记。"""
    plan_files = dict(plan or {})
    if spec.source_kind == "runs":
        if not plan_files:
            plan_files = capture_arm_files(snapshot, spec)
        arm = _load_runs(snapshot, spec, plan_files)
    elif spec.source_kind == "legacy_wf":
        if not plan_files:
            plan_files = capture_arm_files(snapshot, spec)
        arm = _load_legacy(snapshot, spec, dict(plan_files.get("legacy") or {}))
    else:  # pragma: no cover - 构造期已拒绝
        raise ArmSourceError(f"臂来源类型不受支持：{spec.source_kind!r}")
    arm.source_refs = {
        key: receipt.sha256
        for key, receipt in sorted(snapshot.receipts.items())
        if receipt.root_id == spec.root_id
    }
    if arm.status.get("report_status") == "invalid":
        # 读入阶段已判无效（缺必需文件/多候选/契约违例）：保留原原因，不覆盖为“链为空”
        return arm
    chain_row = validate_chain_rows(arm.chain)
    arm.rows_total = int(chain_row.get("rows_total", len(arm.chain)) or 0)
    arm.rows_used = int(chain_row.get("rows_used", 0) or 0) if status.is_usable(chain_row) else 0
    arm.split_count = int(chain_row.get("split_count") or arm.chain["split_index"].nunique())
    if status.is_usable(chain_row):
        arm.status = _merge_status(arm.status, chain_row)
    else:
        arm.status = chain_row
        arm.degradations.append(_degradation(f"arms.{spec.arm_id}.chain", chain_row))
    return arm


#: 状态严重度（臂级合并取较严重者；与视图层同一序，不新增第七种状态）
_SEVERITY: dict[str, int] = {
    "available": 0,
    "empty_valid": 0,
    "stale": 1,
    "degraded": 2,
    "not_available": 3,
    "invalid": 4,
}


def _merge_status(base: Mapping[str, Any], chain_row: Mapping[str, Any]) -> dict[str, Any]:
    """臂级状态 = 自身状态与链行状态的**较严重者**（不把 invalid 读成可用）。"""
    base_word = str(base.get("report_status") or "not_available")
    chain_word = str(chain_row.get("report_status") or "not_available")
    if _SEVERITY.get(chain_word, 3) > _SEVERITY.get(base_word, 3):
        return dict(chain_row)
    return dict(base)
