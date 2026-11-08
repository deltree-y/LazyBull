# -*- coding: utf-8 -*-
"""趋势库来源适配（P6 T4）：三类证据域**分路**发现、身份规则与输入白名单。

硬口径（设计 §2.2 / §4.3 / §5.6 / §5.7）：

- 三类证据域输入**互斥**：``production`` 只收 P4 纸面根；``research`` 只收规范化 runs 批次
  与旧 WF 运行目录；``calibration_only`` 只收经封存校验的 P5b bundle（``t5_seq7``）；
  收到越类输入即抛 ``TrendSourceError``（CLI 非零退出），不生成“有效混合报告”；
- **身份**：run 目录有 ``manifest.json``/``report_manifest.json``（bundle 为
  ``drawdown_manifest.json``）时 ``source_identity_sha256`` = manifest **字节 SHA**；
  无 manifest 时取“实际消费的白名单源文件（chain/summary/trades/可得 attribution）的
  root-relative path + 文件 SHA”按路径排序后的 canonical JSON 哈希，
  ``source_manifest_sha256=null`` 并在 projection 标 ``manifest_absent``；
- **增量**：上一快照的行按 ``run_key`` 原样继承；只有**新 run** 才读指标文件
  （全历史重扫只在单独授权的批任务里出现）；
- **禁止 mtime / 空串占位**：身份只由路径与字节哈希构成；
- 只读**快照副本**：发现与解析只对副本进行；本模块不构造 Storage/Registry/日历。

指标读取/观测装配在 :mod:`trend_observe`（本模块只负责发现、身份与快照计划）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.lazybull import PROJECT_ROOT
from src.lazybull.v2.reporting import manifest, status
from src.lazybull.v2.reporting.snapshot import StableSnapshot
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "EVIDENCE_SOURCE_KINDS",
    "MANIFEST_NAMES",
    "PAPER_ROOT_ID",
    "PREVIOUS_ROOT_ID",
    "SOURCE_KINDS",
    "RunObservation",
    "RunSeed",
    "SourceSpec",
    "TrendSourceError",
    "build_observations",
    "build_sources",
    "capture_new_run_files",
    "load_previous_snapshot",
    "plan_and_capture",
    "previous_run_keys",
]

#: research 面的两类来源（闭集）
SOURCE_KINDS: tuple[str, ...] = ("runs", "legacy_wf")
#: 证据域 → 允许的输入类型（输入白名单，跨类即失败）
EVIDENCE_SOURCE_KINDS: dict[str, tuple[str, ...]] = {
    "production": ("p4_paper",),
    "research": SOURCE_KINDS,
    "calibration_only": ("p5b_bundle",),
}
#: run 目录内的 manifest 候选名（有则身份 = 该文件字节 SHA）
MANIFEST_NAMES: tuple[str, ...] = ("manifest.json", "report_manifest.json")
#: 趋势 run 的固定四件产出（manifest 之外；继承前必须逐件核字节，R3-06）
PREVIOUS_OUTPUTS: tuple[str, ...] = (
    "projection.json",
    "report.html",
    "run_index.csv",
    "trend.csv",
)
#: P5b 回撤族 manifest（calibration 面身份来源）
BUNDLE_MANIFEST = "drawdown_manifest.json"
#: P5b 回撤族报告（log-MDD/pp 指标来源）
BUNDLE_REPORT = "drawdown_report.json"
#: 固定根标识
PAPER_ROOT_ID = "paper"
PREVIOUS_ROOT_ID = "prev"
#: P4 run 记录文件名（``{run_kind}_{YYYYMMDD}.json``；coverage/rebalance 不匹配）
RUN_RECORD_PATTERN = re.compile(r"^[a-z][a-z0-9]*_(\d{8})\.json$")
#: P4 宿主与 run 记录同目录写出的诊断旁路（readiness/gap），不是交易提交记录（R3-02）
PAPER_DIAGNOSTIC_PREFIXES: tuple[str, ...] = ("readiness_", "gap_")
#: 身份白名单（runs 形态；daily 只服务金额指标、不参与身份）
RUNS_IDENTITY_FILES: tuple[str, ...] = ("chain_nav.parquet", "summary.parquet")
RUNS_FOLD_IDENTITY: tuple[str, ...] = ("trades.parquet", "attribution.parquet")


class TrendSourceError(ValueError):
    """来源规格/白名单/身份不可用（命令参数或输入形态问题，归设置错误/策略失败）。"""


@dataclass(frozen=True)
class SourceSpec:
    """一个显式来源根（kind 与路径由命令行给出，root_id 由路径令牌确定性生成）。"""

    kind: str
    path: Path
    root_id: str


@dataclass
class RunSeed:
    """发现阶段的一个 run（身份已定，指标尚未读取）。"""

    root_id: str
    relative_run_path: str
    kind: str
    run_dir_name: str
    identity_ref: str
    identity_sha: str
    manifest_ref: str | None = None
    manifest_sha: str | None = None
    identity_files: list[str] = field(default_factory=list)
    metric_files: list[str] = field(default_factory=list)
    run_key: str = ""

    @property
    def key_ref(self) -> str:
        """run 原始路径标识（``root_id:relative``；根自身写 ``.``）。"""
        return f"{self.root_id}:{self.relative_run_path or '.'}"


@dataclass
class RunObservation:
    """一个 run 的索引行、五条指标行与降级登记。"""

    index_row: dict[str, Any]
    metric_rows: list[dict[str, Any]]
    degradations: list[dict[str, Any]]


def _root_token(path: Path) -> str:
    """来源根稳定标识：**仓库相对路径**（POSIX）的 SHA-256 前 8 位。

    锚点固定为 ``PROJECT_ROOT``（不是 ``Path.cwd()``）：否则同一输入在不同工作目录
    会得到不同 root_id → 不同 run_key/report_id，违反“同输入字节一致”（T4-R2-N1）。
    若把 ``--source`` 出现顺序并入身份，顺序变化就会改变 root_id → ``run_key``，
    从而在同一趋势库里把同一 run 记成两份（T4-R1-06）；因此本函数刻意不读取顺序。仓库外根退化为绝对路径
    哈希（仅本机稳定，属显式登记的非可移植边界）。
    """
    resolved = Path(path).expanduser().resolve()
    try:
        key = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        key = resolved.as_posix()
    return sha256(key.encode("utf-8")).hexdigest()[:8]


def _spec(kind: str, path: Path, index: int) -> SourceSpec:
    """来源根（``root_id`` 由 kind + 路径令牌确定性生成：同根同 run_key、与顺序无关）。"""
    prefix = {"runs": "src", "legacy_wf": "legacy", "p5b_bundle": "bundle"}.get(kind, "src")
    del index  # 顺序不再参与身份；保留参数以兼容既有调用点
    return SourceSpec(kind=kind, path=Path(path), root_id=f"{prefix}_{_root_token(path)}")


def build_sources(
    evidence_class: str,
    *,
    raw_sources: Sequence[str],
    paper_root: Path | str | None,
    bundles: Sequence[str] | str | None,
) -> list[SourceSpec]:
    """按证据域装配来源清单并执行**输入白名单**（越类即失败）。"""
    if evidence_class not in EVIDENCE_SOURCE_KINDS:
        raise TrendSourceError(f"未知证据类：{evidence_class!r}")
    allowed = EVIDENCE_SOURCE_KINDS[evidence_class]
    bundle_items = [bundles] if isinstance(bundles, (str, Path)) else list(bundles or [])
    if evidence_class == "production":
        if raw_sources or bundle_items:
            raise TrendSourceError(
                "production 面只接受 P4 纸面根（收到 runs/legacy/bundle 输入）：禁止跨类混合"
            )
        if paper_root is None:
            raise TrendSourceError("production 面必须显式给出纸面根")
        return [SourceSpec("p4_paper", Path(paper_root), PAPER_ROOT_ID)]
    if evidence_class == "research":
        if paper_root is not None or bundle_items:
            raise TrendSourceError(
                "research 面只接受 runs/legacy_wf 来源（收到纸面根或 P5b bundle）：禁止跨类混合"
            )
        specs = [_split_source(item, index) for index, item in enumerate(raw_sources)]
        if not specs:
            raise TrendSourceError("research 面至少需要一个 --source runs:<dir>|legacy_wf:<dir>")
        return specs
    if paper_root is not None or raw_sources:
        raise TrendSourceError(
            "calibration_only 面只接受 P5b sealed bundle（收到纸面根或 runs/legacy 输入）"
        )
    specs = [_spec("p5b_bundle", Path(item), index) for index, item in enumerate(bundle_items)]
    if not specs:
        raise TrendSourceError("calibration_only 面必须显式给出至少一个 P5b bundle 目录")
    if "p5b_bundle" not in allowed:  # pragma: no cover - 构造期不可达
        raise TrendSourceError(f"证据类 {evidence_class} 不接受 bundle 输入")
    return specs


def _split_source(text: str, index: int) -> SourceSpec:
    """解析 ``kind:path``（kind ∈ runs/legacy_wf）。"""
    raw = str(text or "").strip()
    if ":" not in raw:
        raise TrendSourceError(f"来源须为 kind:path（kind ∈ {list(SOURCE_KINDS)}）：{text!r}")
    kind, path_text = raw.split(":", 1)
    kind = kind.strip()
    if kind not in SOURCE_KINDS:
        raise TrendSourceError(f"来源类型不受支持：{kind!r}（合法 {list(SOURCE_KINDS)}）")
    path = Path(path_text.strip())
    if not path_text.strip():
        raise TrendSourceError(f"来源路径为空：{text!r}")
    return _spec(kind, path, index)


def _run_key(evidence_class: str, seed: RunSeed) -> str:
    """``run_key=canonical_json_sha256({evidence_class,root_id,relative_run_path,identity})``。"""
    return canonical_json_sha256(
        {
            "evidence_class": evidence_class,
            "root_id": seed.root_id,
            "relative_run_path": seed.relative_run_path,
            "source_identity_sha256": seed.identity_sha,
        }
    )


def _rel(prefix: str, name: str) -> str:
    """拼相对路径（POSIX 分隔符；根自身为空前缀）。"""
    return f"{prefix}/{name}" if prefix else name


def _receipt_sha(snapshot: StableSnapshot, root_id: str, relative: str) -> str | None:
    """副本回执的字节 SHA（未捕获即 ``None``）。"""
    receipt = snapshot.receipts.get(f"{root_id}:{relative}")
    return None if receipt is None else receipt.sha256


def _identity_from_files(
    snapshot: StableSnapshot,
    root_id: str,
    relatives: Sequence[str],
    *,
    manifest_name: str | None,
    run_rel: str,
) -> tuple[str, str, str | None, str | None]:
    """返回 ``(identity_sha, identity_ref, manifest_ref, manifest_sha)``。

    有 manifest 时身份 = manifest 字节 SHA（``identity_ref`` 指向该文件）；
    否则 = 实际消费文件的 ``[{path, sha256}]`` 排序后 canonical JSON 哈希，
    ``identity_ref`` 指向 **run 目录**（复合摘要与单文件 SHA 不同粒度，
    不得写成其中某一文件，避免“按 ref 取 SHA 对不上”）。
    """
    if manifest_name is not None:
        rel = manifest_name
        receipt = snapshot.receipts.get(f"{root_id}:{rel}")
        if receipt is None:
            raise TrendSourceError(f"manifest 未进入快照：{root_id}:{rel}")
        ref = f"{root_id}:{rel}"
        return receipt.sha256, ref, ref, receipt.sha256
    rows: list[dict[str, str]] = []
    for rel in sorted(set(relatives)):
        receipt = snapshot.receipts.get(f"{root_id}:{rel}")
        if receipt is None:
            continue
        rows.append({"path": rel, "sha256": receipt.sha256})
    if not rows:
        raise TrendSourceError(f"{root_id}:{'.'} 无任何白名单源文件，身份不可计算")
    digest = canonical_json_sha256(rows)
    return digest, f"{root_id}:{run_rel or '.'}", None, None


def _capture_first(
    snapshot: StableSnapshot, root_id: str, relatives: Sequence[str]
) -> list[str]:
    """按序捕获存在的文件，返回实际捕获的相对路径（缺失静默跳过，由调用方判状态）。"""
    captured: list[str] = []
    for rel in relatives:
        if snapshot.exists(root_id, rel):
            # 共享文件（如 P4 的 nav/trades）只复制一次：副本已在则复用回执
            captured.append(rel)
            continue
        if snapshot.capture(root_id, rel):
            captured.append(rel)
    return captured


def _manifest_of(snapshot: StableSnapshot, root_id: str, prefix: str) -> str | None:
    """在 run 目录内找 manifest（候选名闭集；多候选即失败，不按最新择一）。

    先列**原目录**（``list_dir`` 已在二次身份核验内），再捕获该 manifest 字节。
    """
    names = set(snapshot.list_dir(root_id, prefix))
    found = [_rel(prefix, name) for name in MANIFEST_NAMES if name in names]
    if len(found) > 1:
        raise TrendSourceError(f"run 目录存在多个 manifest 候选：{found}（拒绝择一）")
    if not found:
        return None
    if not snapshot.capture(root_id, found[0]):
        return None
    return found[0]


def _plan_runs(snapshot: StableSnapshot, spec: SourceSpec) -> list[RunSeed]:
    """规范化 runs：根自身是批次或子目录逐层（一层）扫描。"""
    files = snapshot.list_dir(spec.root_id, "")
    if "summary.parquet" in files:
        prefixes = [""]
    else:
        prefixes = snapshot.list_dirs(spec.root_id, "")
    if not prefixes:
        return []
    seeds: list[RunSeed] = []
    for prefix in prefixes:
        manifest_rel = _manifest_of(snapshot, spec.root_id, prefix)
        identity: list[str] = []
        if manifest_rel is None:
            # 无 manifest 才需要“实际消费文件 path+sha”作为身份（有 manifest 时身份 =
            # manifest 字节 SHA，逐折/逐批文件只为**新 run**的指标解析捕获）。
            identity = _capture_first(
                snapshot, spec.root_id, [_rel(prefix, name) for name in RUNS_IDENTITY_FILES]
            )
            fold_dirs = snapshot.list_dirs(spec.root_id, _rel(prefix, "folds"))
            for fold in fold_dirs:
                identity.extend(
                    _capture_first(
                        snapshot,
                        spec.root_id,
                        [_rel(prefix, f"folds/{fold}/{name}") for name in RUNS_FOLD_IDENTITY],
                    )
                )
            _capture_first(snapshot, spec.root_id, [_rel(prefix, "batch_meta.json")])
        digest, ref, mref, msha = _identity_from_files(
            snapshot,
            spec.root_id,
            identity,
            manifest_name=manifest_rel,
            run_rel=prefix,
        )
        seeds.append(
            RunSeed(
                root_id=spec.root_id,
                relative_run_path=prefix,
                kind="runs",
                run_dir_name=Path(prefix).name if prefix else spec.path.name,
                identity_ref=ref,
                identity_sha=digest,
                manifest_ref=mref,
                manifest_sha=msha,
                identity_files=list(identity),
            )
        )
    return seeds


def _legacy_files(snapshot: StableSnapshot, root_id: str, prefix: str) -> dict[str, list[str]]:
    """旧 WF 白名单文件（chain/summary/state 三类固定前缀）。"""
    names = snapshot.list_dir(root_id, prefix)
    selected: dict[str, list[str]] = {"chain": [], "summary": [], "state": []}
    for name in names:
        if name.startswith("chain_nav_") and name.endswith(".csv"):
            selected["chain"].append(name)
        elif name.startswith("walk_forward_summary_") and name.endswith(".csv"):
            selected["summary"].append(name)
        elif name.startswith("data_state_") and name.endswith(".json"):
            selected["state"].append(name)
    return selected


def _plan_legacy(snapshot: StableSnapshot, spec: SourceSpec) -> list[RunSeed]:
    """旧 WF：根自身是运行目录，或一层子目录各为一个运行。"""
    root_files = _legacy_files(snapshot, spec.root_id, "")
    if root_files["chain"] or root_files["summary"]:
        prefixes = [""]
    else:
        prefixes = snapshot.list_dirs(spec.root_id, "")
    if not prefixes:
        return []
    seeds: list[RunSeed] = []
    for prefix in prefixes:
        manifest_rel = _manifest_of(snapshot, spec.root_id, prefix)
        files = _legacy_files(snapshot, spec.root_id, prefix)
        identity: list[str] = []
        for name in files["chain"] + files["summary"]:
            identity.extend(_capture_first(snapshot, spec.root_id, [_rel(prefix, name)]))
        for name in files["state"]:
            _capture_first(snapshot, spec.root_id, [_rel(prefix, name)])
        digest, ref, mref, msha = _identity_from_files(
            snapshot,
            spec.root_id,
            identity,
            manifest_name=manifest_rel,
            run_rel=prefix,
        )
        seeds.append(
            RunSeed(
                root_id=spec.root_id,
                relative_run_path=prefix,
                kind="legacy_wf",
                run_dir_name=Path(prefix).name if prefix else spec.path.name,
                identity_ref=ref,
                identity_sha=digest,
                manifest_ref=mref,
                manifest_sha=msha,
                identity_files=list(identity),
            )
        )
    return seeds


def _plan_paper(snapshot: StableSnapshot, spec: SourceSpec) -> list[RunSeed]:
    """P4 纸面根：每个 ``runs/{run_kind}_{YYYYMMDD}.json`` 是一个 run。"""
    names = snapshot.list_dir(spec.root_id, "runs")
    # 只收“原生交易提交记录”闭集：readiness/gap 是宿主诊断旁路，不产生趋势指标（R3-02）
    records = sorted(
        name
        for name in names
        if RUN_RECORD_PATTERN.match(name) and not name.startswith(PAPER_DIAGNOSTIC_PREFIXES)
    )
    seeds: list[RunSeed] = []
    for name in records:
        rel = _rel("runs", name)
        _capture_first(snapshot, spec.root_id, [rel])
        receipt = snapshot.receipts[f"{spec.root_id}:{rel}"]
        seeds.append(
            RunSeed(
                root_id=spec.root_id,
                relative_run_path=rel,
                kind="p4_paper",
                run_dir_name=name,
                identity_ref=f"{spec.root_id}:{rel}",
                identity_sha=receipt.sha256,
                identity_files=[rel],
            )
        )
    return seeds


def _plan_bundles(snapshot: StableSnapshot, specs: Sequence[SourceSpec]) -> list[RunSeed]:
    """P5b：每个 bundle 目录是一个 run（身份 = 回撤族 manifest 字节 SHA）。"""
    seeds: list[RunSeed] = []
    for spec in specs:
        names = snapshot.list_dir(spec.root_id, "")
        captured = _capture_first(snapshot, spec.root_id, names)
        if BUNDLE_MANIFEST not in captured:
            raise TrendSourceError(f"bundle 缺回撤族 manifest（{BUNDLE_MANIFEST}）：{spec.path}")
        digest, ref, mref, msha = _identity_from_files(
            snapshot,
            spec.root_id,
            captured,
            manifest_name=BUNDLE_MANIFEST,
            run_rel="",
        )
        seeds.append(
            RunSeed(
                root_id=spec.root_id,
                relative_run_path="",
                kind="p5b_bundle",
                run_dir_name=str(spec.path),
                identity_ref=ref,
                identity_sha=digest,
                manifest_ref=mref,
                manifest_sha=msha,
                identity_files=list(captured),
                metric_files=[BUNDLE_REPORT] if BUNDLE_REPORT in captured else [],
            )
        )
    return seeds


def plan_and_capture(
    snapshot: StableSnapshot, evidence_class: str, specs: Sequence[SourceSpec]
) -> dict[str, Any]:
    """发现 + 捕获身份文件（含 calibration bundle 全量文件）。"""
    if evidence_class == "production":
        # 权威文件与 run 数量无关：零 run 时也要有输入回执证明“合法空/权威缺失”（R3-01-R3）
        _capture_first(snapshot, specs[0].root_id, ["state/engine_state.bin"])
        seeds = _plan_paper(snapshot, specs[0])
    elif evidence_class == "calibration_only":
        seeds = _plan_bundles(snapshot, specs)
    else:
        seeds = []
        for spec in specs:
            seeds.extend(
                _plan_runs(snapshot, spec) if spec.kind == "runs" else _plan_legacy(snapshot, spec)
            )
    owner: dict[str, str] = {}
    for spec in specs:
        resolved = str(Path(spec.path).expanduser().resolve())
        mapped = owner.setdefault(spec.root_id, resolved)
        if mapped != resolved:
            # 8 hex 截断碰撞：不同来源根映射到同一 root_id 会静默错语义，直接拒绝
            raise TrendSourceError(
                f"root_id 冲突（不同来源根映射到同一标识）：{spec.root_id} → {mapped} / {resolved}"
            )
    for seed in seeds:
        seed.run_key = _run_key(evidence_class, seed)
    duplicated = sorted({seed.run_key for seed in seeds if _count(seeds, seed.run_key) > 1})
    if duplicated:
        raise TrendSourceError(f"同一来源内 run_key 重复（身份冲突）：{duplicated[:5]}")
    # root_id 由来源根的仓库相对路径派生（与顺序无关）：同一物理 run 出现在两个来源根
    # 会被算成两条身份 ⇒ 索引/趋势双份、run_total 虚增；此处按“同 kind+路径+身份”拒绝。
    # 物理位置 = 来源根 + 该 run 在根内的相对路径（集合根与直接根的重叠在解析后同址）。
    # 判重**只看物理位置与来源类型**，不掺内容身份：无 manifest 的旧 WF 身份含
    # root-relative 路径，父/子根下摘要不同，若把身份并入签名仍会漏判（R3-05-R1）。
    roots_by_id = {spec.root_id: Path(spec.path).expanduser().resolve() for spec in specs}
    owners: dict[tuple[str, str], str] = {}
    for seed in seeds:
        physical = str((roots_by_id[seed.root_id] / seed.relative_run_path).resolve())
        signature = (seed.kind, physical)
        if signature in owners and owners[signature] != seed.root_id:
            raise TrendSourceError(
                "同一物理 run 出现在多个来源根（解析后同址，静默重复即拒绝）："
                f"{signature[0]}:{physical} @ {owners[signature]}/{seed.root_id}"
            )
        owners[signature] = seed.root_id
    return {"evidence_class": evidence_class, "specs": list(specs), "seeds": seeds}


def _count(seeds: Sequence[RunSeed], key: str) -> int:
    return sum(1 for seed in seeds if seed.run_key == key)


def previous_run_keys(previous_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """上一快照已登记的 run_key 集合（增量判定的唯一依据）。"""
    return {str(row.get("run_key") or "") for row in previous_rows}


def capture_new_run_files(
    snapshot: StableSnapshot, plan: Mapping[str, Any], inherited: set[str]
) -> list[RunSeed]:
    """只为**新 run** 捕获指标文件（继承行不重读，保证“上一快照 + 新 run”增量）。"""
    fresh: list[RunSeed] = []
    for seed in plan.get("seeds") or []:
        if seed.run_key in inherited:
            continue
        _capture_seed_metrics(snapshot, seed)
        fresh.append(seed)
    return fresh


def _capture_seed_metrics(snapshot: StableSnapshot, seed: RunSeed) -> None:
    """按来源类型捕获该 run 的指标文件（缺失跳过，由读取阶段登记状态行）。"""
    if seed.kind == "runs":
        prefix = seed.relative_run_path
        names = ["batch_meta.json", "summary.parquet", "chain_nav.parquet"]
        _capture_first(snapshot, seed.root_id, [_rel(prefix, name) for name in names])
        for fold in snapshot.list_dirs(seed.root_id, _rel(prefix, "folds")):
            _capture_first(
                snapshot,
                seed.root_id,
                [
                    _rel(prefix, f"folds/{fold}/{name}")
                    for name in (
                        "trades.parquet",
                        "attribution.parquet",
                        "daily.parquet",
                        "topk_detail.parquet",
                    )
                ],
            )
    elif seed.kind == "legacy_wf":
        prefix = seed.relative_run_path
        files = _legacy_files(snapshot, seed.root_id, prefix)
        _capture_first(
            snapshot,
            seed.root_id,
            [_rel(prefix, name) for name in files["chain"] + files["summary"] + files["state"]],
        )
    elif seed.kind == "p4_paper":
        # 权威代（paper_generation/portfolio_values）是 production 数值的前置门（R3-01）
        _capture_first(
            snapshot,
            seed.root_id,
            [
                "state/engine_state.bin",
                "config.yaml",
                "nav/nav.parquet",
                "trades/trades.parquet",
            ],
        )
    elif seed.kind == "p5b_bundle":
        _capture_first(snapshot, seed.root_id, seed.identity_files)
    # 指标文件只取该 run 自己的（集合根下不能把其他 run 的文件算进来；R3-03）。
    # 直接批次根（relative_run_path == ""）时该根下的文件就是本 run 指标文件，
    # 不能因空相对路径退化成空集合摘要（R3-03-R1）。
    if seed.kind in ("runs", "legacy_wf"):
        prefix = f"{seed.root_id}:{seed.relative_run_path}"
        needle = prefix + "/" if seed.relative_run_path else prefix
        seed.metric_files = sorted(
            key.split(":", 1)[1] for key in snapshot.receipts if key.startswith(needle)
        )
    else:
        seed.metric_files = sorted(
            key.split(":", 1)[1]
            for key in snapshot.receipts
            if key.startswith(f"{seed.root_id}:")
        )


def _previous_manifest_doc(
    snapshot: StableSnapshot, *, evidence_class: str
) -> Mapping[str, Any]:
    """上一快照 manifest：形态/封存/类型/证据类四道门（不满足即拒绝继承）。"""
    raw, reason = snapshot.read_json(PREVIOUS_ROOT_ID, "report_manifest.json")
    if reason is not None:
        raise TrendSourceError(f"上一快照 manifest 损坏：{reason}")
    if not isinstance(raw, Mapping):
        raise TrendSourceError("上一快照缺 report_manifest.json（增量生成必须显式给出）")
    row = manifest.verify_manifest(raw)
    if not status.is_usable(row):
        raise TrendSourceError(
            f"上一快照 manifest 校验失败：{row.get('degradation_reason_code')} / {row.get('detail')}"
        )
    if str(raw.get("report_kind")) != "trend":
        raise TrendSourceError(f"上一快照不是趋势 run：report_kind={raw.get('report_kind')!r}")
    previous_class = str(raw.get("evidence_class"))
    if previous_class != evidence_class:
        raise TrendSourceError(
            f"跨证据类引用：上一快照 {previous_class!r} ≠ 本次 {evidence_class!r}"
        )
    return raw


def _previous_tables(
    snapshot: StableSnapshot, manifest_doc: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """上一快照两份 CSV：先对 manifest 登记核字节，再解析（拒绝被改写的历史）。"""
    from src.lazybull.v2.reporting import trend_csv  # 局部导入避免循环

    declared = dict(manifest_doc.get("outputs") or {})
    index_text = _read_text(snapshot, "run_index.csv")
    trend_text = _read_text(snapshot, "trend.csv")
    for name, text in (("run_index.csv", index_text), ("trend.csv", trend_text)):
        if declared.get(name) != _sha256_text(text):
            raise TrendSourceError(
                f"上一快照 {name} 字节与 manifest 登记不一致（拒绝继承被改写的历史）"
            )
    return trend_csv.parse_run_index(index_text), trend_csv.parse_trend(trend_text)


def _previous_outputs(
    snapshot: StableSnapshot, manifest_doc: Mapping[str, Any]
) -> None:
    """上一快照固定四件产出：键集与字节都必须与 manifest 登记一致（R3-06）。"""
    declared = dict(manifest_doc.get("outputs") or {})
    if set(declared) != set(PREVIOUS_OUTPUTS):
        raise TrendSourceError(
            f"上一快照 outputs 键集不符：{sorted(declared)} ≠ {sorted(PREVIOUS_OUTPUTS)}"
        )
    for name in PREVIOUS_OUTPUTS:
        receipt = snapshot.receipts.get(f"{PREVIOUS_ROOT_ID}:{name}")
        if receipt is None:
            raise TrendSourceError(f"上一快照缺 {name}（五件不完整，拒绝继承）")
        if declared.get(name) != receipt.sha256:
            raise TrendSourceError(
                f"上一快照 {name} 字节与 manifest 登记不一致（拒绝继承被改写/缺件的历史）"
            )


def _previous_projection(
    snapshot: StableSnapshot, manifest_doc: Mapping[str, Any]
) -> Mapping[str, Any]:
    """上一快照 projection：字节与登记一致 + 根为对象（供 root_map/继承来源）。"""
    declared = dict(manifest_doc.get("outputs") or {})
    text = _read_text(snapshot, "projection.json")
    if declared.get("projection.json") != _sha256_text(text):
        raise TrendSourceError("上一快照 projection.json 字节与 manifest 登记不一致")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrendSourceError(f"上一快照 projection.json 不是合法 JSON：{exc}") from exc
    if not isinstance(parsed, Mapping):
        raise TrendSourceError("上一快照 projection.json 根不是对象")
    return parsed


def load_previous_snapshot(
    snapshot: StableSnapshot, *, evidence_class: str
) -> dict[str, Any]:
    """读取并**先验 seal**校验上一不可变趋势快照（同 ID 历史不覆盖）。"""
    _capture_first(
        snapshot,
        PREVIOUS_ROOT_ID,
        ["report_manifest.json", *PREVIOUS_OUTPUTS],
    )
    raw = _previous_manifest_doc(snapshot, evidence_class=evidence_class)
    _previous_outputs(snapshot, raw)
    index_rows, trend_rows = _previous_tables(snapshot, raw)
    foreign = sorted({str(item.get("evidence_class")) for item in index_rows} - {evidence_class})
    if foreign:
        raise TrendSourceError(f"上一快照 index 含跨证据类行：{foreign}")
    previous_projection = _previous_projection(snapshot, raw)
    manifest_receipt = snapshot.receipts.get(f"{PREVIOUS_ROOT_ID}:report_manifest.json")
    return {
        "report_id": str(raw.get("report_id")),
        "manifest": dict(raw),
        "manifest_sha256": manifest_receipt.sha256 if manifest_receipt else None,
        "input_manifest_sha256": raw.get("input_manifest_sha256"),
        "root_map": {
            str(key): str(value)
            for key, value in dict(previous_projection.get("root_map") or {}).items()
        },
        "index_rows": index_rows,
        "trend_rows": trend_rows,
    }


def _read_text(snapshot: StableSnapshot, relative: str) -> str:
    """读副本文本（缺失即失败；解码失败也失败，不猜编码）。"""
    payload = snapshot.read_bytes(PREVIOUS_ROOT_ID, relative)
    if payload is None:
        raise TrendSourceError(f"上一快照缺 {relative}")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrendSourceError(f"上一快照 {relative} 不是 UTF-8：{exc}") from exc


def _sha256_text(text: str) -> str:
    from hashlib import sha256

    return sha256(text.encode("utf-8")).hexdigest()


def build_observations(
    snapshot: StableSnapshot,
    plan: Mapping[str, Any],
    fresh: Sequence[RunSeed],
) -> list[RunObservation]:
    """为**新 run** 计算身份行与五类指标行（实现见 ``trend_observe``）。"""
    from src.lazybull.v2.reporting import trend_observe

    return trend_observe.build_observations(snapshot, plan, fresh)
