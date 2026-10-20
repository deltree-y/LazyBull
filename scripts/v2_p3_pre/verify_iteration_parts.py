# -*- coding: utf-8 -*-
"""P3 pre T4 验收辅助：输入身份、冻结段证据、影子 fixture 与故障注入（只读生产）。

身份清单覆盖管线实际读取窗口（D-7 自然月 / ti 24 交易日前置）的 clean 三数据集、
静态单文件、raw/top_inst 年分区、附录 B 冻结指纹与旧链资产锚点；冻结段后验以
全新只读 manifest 视图复算，不依赖缓存；故障注入在 scratch 内以全新实例重放
WAL 并核对盘上终态，不触生产。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts" / "v2_p1") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "v2_p1"))
if str(_REPO_ROOT / "scripts" / "v2_p3_pre") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "v2_p3_pre"))

from migrate_store_manifest import _write_json_atomic  # noqa: E402
from verify_immutable_partitions import _compare_partition_content  # noqa: E402

from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import (  # noqa: E402
    ARCHIVE_HOT_BOUNDARY,
    LABEL_TABLES,
    PANEL_GROUPS,
)
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.label_refresh import (  # noqa: E402
    LabelRefreshRecoveryError,
    _current_status,
    refresh_labels_for_day,
    validate_label_day_readiness,
)
from src.lazybull.v2.store.manifest import Manifest, sha256_16_of_file  # noqa: E402
from src.lazybull.v2.store.panel_builder import bootstrap_manifest  # noqa: E402
from src.lazybull.v2.store.row_domain import (  # noqa: E402
    LABEL_FILTERED_THROUGH,
    STRUCTURAL_FROM,
)
from src.lazybull.v2.store.store_identity import (  # noqa: E402
    collect_file_identities,
    file_identity,
    producer_identity,
    verify_file_identities,
    workspace_identity,
)

_ALL_GROUPS = tuple(PANEL_GROUPS)
_LABEL_NAMES = tuple(LABEL_TABLES)
_ARCHIVE_BOUNDARY_MONTH = f"{ARCHIVE_HOT_BOUNDARY[:4]}-{ARCHIVE_HOT_BOUNDARY[4:6]}"

#: P1 数据态预检（docs/data/v2_p1_build_freeze.md 附录 B，sha256 前 16 位，2026-10-02 冻结）
APPENDIX_B: tuple[tuple[str, str, str], ...] = (
    ("raw/daily/2026-07-31.parquet", "daily（2026-07-31）", "2783617a92ab2ce9"),
    ("raw/adj_factor/2026-07-31.parquet", "adj_factor（2026-07-31）", "43c16bd328e4e29f"),
    ("raw/daily_basic/2026-07-31.parquet", "daily_basic（2026-07-31）", "48ad81b4621f63ba"),
    ("raw/moneyflow/2026-07-31.parquet", "moneyflow（2026-07-31）", "9071cc45a5946ebf"),
    ("raw/stk_limit/2026-07-31.parquet", "stk_limit（2026-07-31）", "f79334470d421dc8"),
    ("raw/suspend/2026-07-31.parquet", "suspend（2026-07-31）", "a4db2d44e1e1e9f2"),
    ("raw/stock_st/2026-07-31.parquet", "stock_st（2026-07-31）", "31124dcea2facd25"),
    ("raw/margin_detail/2026-07-31.parquet", "margin_detail（2026-07-31）", "2bfdbeb9ee41de34"),
    ("features/cs_train/20260702.parquet", "cs_train（20260702）", "54678be3decbd5d5"),
    ("clean/daily/2026-07-31.parquet", "clean/daily（2026-07-31）", "d6a6f7857393896f"),
)

#: 冷区抽验样本（信号日, 所属冷区月）：月文件指纹 + 文件内 trade_date 过滤
COLD_SPOT_SAMPLES: tuple[tuple[str, str], ...] = (
    ("20120104", "2012-01"),
    ("20170103", "2017-01"),
    ("20220104", "2022-01"),
    ("20250630", "2025-06"),
)
HOT_SPOT_DAY = LABEL_FILTERED_THROUGH
COLD_SPOT_MONTH = "2025-05"


class AcceptanceError(RuntimeError):
    """验收门失败；报告落盘（路径已验证）后以非零退出码结束。"""


# ---------- 附录 B 数据态预检 ----------


def appendix_b_report(
    data_root: Path, table: Sequence[tuple[str, str, str]] = APPENDIX_B
) -> dict[str, Any]:
    """逐项重算 P1 附录 B 指纹；任一不匹配先归因输入修订，不判实现漂移也不放行。"""
    items: list[dict[str, Any]] = []
    for relative, name, expected in table:
        path = data_root / relative
        actual = sha256_16_of_file(path) if path.is_file() else None
        items.append(
            {
                "dataset": name,
                "path": relative,
                "expected": expected,
                "actual": actual,
                "match": actual == expected,
            }
        )
    matched = sum(1 for item in items if item["match"])
    return {
        "items": items,
        "matched": matched,
        "total": len(items),
        "all_match": matched == len(items),
    }


# ---------- 输入身份清单（M1/R2-T4-03 整改：覆盖管线实际读取窗口） ----------


def min_input_date(
    calendar: Sequence[str],
    shadow_days: Sequence[str],
    dates: Sequence[str],
    capture_start: Any,
) -> str:
    """管线实际读取窗口左边界。

    取三者最小值：影子首个 T+1；日更捕获起点（D-7 自然月）再减 7 个自然月的管线预热
    （panel_builder._daily_capture_start → features/pipeline.py:384 的二次 warmup）；
    ti 24 交易日前置。
    """
    candidates = [str(calendar[calendar.index(shadow_days[0]) + 1])]
    for date in dates:
        capture = str(capture_start(date))
        warmup = (pd.to_datetime(capture, format="%Y%m%d") - pd.DateOffset(months=7)).strftime(
            "%Y%m%d"
        )
        candidates.append(warmup)
    shadow_idx = calendar.index(shadow_days[0])
    candidates.append(str(calendar[max(0, shadow_idx - 24)]))
    return min(candidates)


def _dataset_window_paths(
    directory: Path, dataset: str, min_date: str, watermark: str
) -> list[Path]:
    if not directory.is_dir():
        raise AcceptanceError(f"输入身份必需数据集目录缺失: {directory}（{dataset}）")
    paths: list[Path] = []
    for path in sorted(directory.glob("*.parquet")):
        date = path.stem.replace("-", "")
        if len(date) == 8 and date.isdigit() and min_date <= date <= watermark:
            paths.append(path)
    return paths


def _required_single_files(loader: Any) -> list[Path]:
    storage = loader.storage
    singles = [
        Path(storage.clean_path) / "stock_basic.parquet",
        Path(storage.clean_path) / "trade_cal.parquet",
        Path(storage.raw_path) / "stock_basic.parquet",
        Path(storage.raw_path) / "trade_cal.parquet",
        Path(storage.raw_path) / "shenwan_industry.parquet",
    ]
    missing = [str(path) for path in singles if not path.is_file()]
    if missing:
        raise AcceptanceError(f"输入身份必需文件缺失: {missing}")
    return singles


def pipeline_input_identities(
    loader: Any,
    calendar: Sequence[str],
    shadow_days: Sequence[str],
    dates: Sequence[str],
    watermark: str,
    capture_start: Any,
) -> tuple[list[dict[str, Any]], str]:
    """预采集基线：clean 窗口三数据集、静态单文件、top_inst 全年分区、附录 B 与旧链锚点。

    该清单是运行前基线；管线实际读取（含已启用因子的 raw 输入与二次预热窗口）由
    `recording_parquet_reads` 在读取层记录并于首次读取时哈希，两者合并后运行前后复核。
    """
    storage = loader.storage
    root = Path(storage.root_path)
    left = min_input_date(calendar, shadow_days, dates, capture_start)
    paths: list[Path] = []
    for dataset in ("daily", "daily_basic", "moneyflow"):
        paths.extend(
            _dataset_window_paths(Path(storage.clean_path) / dataset, dataset, left, watermark)
        )
    paths.extend(_required_single_files(loader))
    top_inst_dir = Path(storage.raw_path) / "top_inst"
    if top_inst_dir.is_dir():  # load_top_inst 默认读取全年份，全量纳入
        paths.extend(sorted(top_inst_dir.glob("*.parquet")))
    paths.extend(root / relative for relative, _, _ in APPENDIX_B)
    cs_infer_dir = root / "features" / "cs_infer"
    paths.extend(sorted(cs_infer_dir.glob("*.parquet")))
    cs_train_names = sorted((root / "features" / "cs_train").glob("*.parquet"))
    if cs_train_names:
        paths.append(cs_train_names[-1])
    # 附录 B/cs 锚点与窗口分区天然重叠（如 clean/daily 尾部）；去重保序，禁止歧义证据
    paths = list(dict.fromkeys(paths))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AcceptanceError(f"输入身份清单含缺失文件: {missing[:5]}")
    return collect_file_identities(paths, base_dir=root), left


def directory_listing_evidence(data_root: Path) -> dict[str, list[str] | None]:
    """生产文件集合快照：raw/clean 全数据集、cs 锚点与 store 资产目录名（新增/删除可检出）。

    运行前冻结全部会被消费的 raw/clean 文件集合；store 资产目录名同步冻结。
    扁平键便于逐目录 diff；报告只落 digest 与比对结论，不落全量名清单。
    """
    listing: dict[str, list[str] | None] = {}
    for top in ("raw", "clean"):
        base = data_root / top
        if not base.is_dir():
            listing[top] = None
            continue
        singles = sorted(path.name for path in base.glob("*.parquet"))
        if singles:
            listing[f"{top}/<files>"] = singles
        for child in sorted(base.iterdir()):
            if child.is_dir():
                listing[f"{top}/{child.name}"] = sorted(
                    path.name for path in child.glob("*.parquet")
                )
    features = data_root / "features"
    for anchor in ("cs_train", "cs_infer"):
        directory = features / anchor
        listing[f"features/{anchor}"] = (
            sorted(path.name for path in directory.glob("*.parquet"))
            if directory.is_dir()
            else None
        )
    for store_dir in ("panel", "panel_archive"):
        directory = features / store_dir
        listing[f"features/{store_dir}"] = (
            sorted(path.name for path in directory.iterdir() if path.is_dir())
            if directory.is_dir()
            else None
        )
    labels = data_root / "labels"
    listing["labels"] = (
        sorted(
            f"{table.name}/{path.name}"
            for table in sorted(labels.iterdir())
            if table.is_dir()
            for path in table.glob("*.parquet")
        )
        if labels.is_dir()
        else None
    )
    return listing


def diff_directory_listing(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, list[str]]:
    """逐目录列出新增/删除的文件名；空 dict 表示完全一致。"""
    diff: dict[str, list[str]] = {}
    for key in sorted(set(before) | set(after)):
        old, new = set(before.get(key) or []), set(after.get(key) or [])
        added, removed = sorted(new - old), sorted(old - new)
        if added or removed:
            diff[key] = [*[f"+{name}" for name in added], *[f"-{name}" for name in removed]]
    return diff


@contextmanager
def recording_parquet_reads(
    data_root: Path, preloaded_identities: Sequence[Mapping[str, Any]] = ()
) -> Any:
    """读取层身份：生产根内 parquet 首次读取前即采集身份，读取后立即核验，运行结束再复核。

    预采集基线按绝对路径键预填（运行前身份）；首次出现的文件读取前采集、读取后不一致
    即当次拒绝，禁止"读旧内容、登记新哈希"。scratch（生产根之外）的读取不记录。
    """
    root = data_root.resolve()
    seen: dict[str, dict[str, Any]] = {}
    for record in preloaded_identities:
        path = Path(str(record["path"]))
        resolved = path if path.is_absolute() else (root / path).resolve()
        seen[str(resolved)] = dict(record)
    original = pd.read_parquet

    def _recording(path: Any, *args: Any, **kwargs: Any) -> Any:
        resolved: Path | None = None
        if isinstance(path, (str, Path)):
            try:
                candidate = Path(path).resolve()
            except (OSError, ValueError):
                candidate = None
            if (
                candidate is not None
                and candidate.suffix == ".parquet"
                and candidate.is_relative_to(root)
                and str(candidate) not in seen
            ):
                resolved = candidate
        if resolved is None:
            return original(path, *args, **kwargs)
        before = file_identity(resolved, base_dir=root)
        frame = original(path, *args, **kwargs)
        if file_identity(resolved, base_dir=root) != before:
            raise AcceptanceError(f"读取期间输入漂移（读前/读后身份不一致）: {resolved}")
        seen[str(resolved)] = before
        return frame

    pd.read_parquet = _recording
    try:
        yield seen
    finally:
        pd.read_parquet = original


def legacy_asset_evidence(data_root: Path) -> dict[str, Any]:
    """旧 cs_train/cs_infer 零修改的机械证据：分区名清单（指纹由输入身份清单覆盖）。"""
    cs_train_dir = data_root / "features" / "cs_train"
    cs_infer_dir = data_root / "features" / "cs_infer"
    train_names = sorted(path.name for path in cs_train_dir.glob("*.parquet"))
    infer_names = sorted(path.name for path in cs_infer_dir.glob("*.parquet"))
    return {
        "cs_train_count": len(train_names),
        "cs_train_latest": train_names[-1] if train_names else None,
        "cs_infer_partitions": infer_names,
        "cs_infer_beyond_seam": [
            name for name in infer_names if name.removesuffix(".parquet") > LABEL_FILTERED_THROUGH
        ],
    }


def assert_run_identities(
    input_identities: Sequence[Mapping[str, Any]],
    input_root: Path,
    producer: Mapping[str, Any],
    workspace: Mapping[str, Any],
) -> None:
    """运行前后输入/生产者/工作区身份复核；任何漂移 fail-closed。"""
    mismatches = verify_file_identities(input_identities, base_dir=input_root)
    if mismatches:
        raise AcceptanceError(f"验收运行前后输入身份变化: {mismatches[:5]}")
    if producer_identity(_REPO_ROOT)["producer_digest"] != producer["producer_digest"]:
        raise AcceptanceError("验收运行前后生产者代码或环境身份变化")
    current = workspace_identity(_REPO_ROOT)
    for key in ("git_commit", "source_digest"):
        if current.get(key) != workspace.get(key):
            raise AcceptanceError(f"验收运行前后工作区身份变化: {key}")


# ---------- 影子 fixture（复制生产 → 显式 forming + 清空水位） ----------


def copy_shadow_day_to_scratch(prod_store: Any, scratch_store: Any, date: str) -> None:
    """复制生产当日 panel 八族与三表标签；标签显式置 forming（原 sealed 副本的影子初态）。"""
    for group in _ALL_GROUPS:
        frame = pd.read_parquet(prod_store.panel_dir / date / f"{group}.parquet")
        scratch_store.append_features(TradeDate.from_str(date), group, frame)
    for name in _LABEL_NAMES:
        frame = pd.read_parquet(prod_store.labels_dir / name / f"{date}.parquet")
        scratch_store.append_labels(name, frame.assign(maturity_status="forming"))


def assert_forming_fixture(scratch_store: Any, days: Sequence[str]) -> None:
    """影子初态：三表全 forming 且封存水位已清空。"""
    for name in _LABEL_NAMES:
        sealed_through = scratch_store.manifest.label_sealed_through(name)
        if sealed_through is not None:
            raise AcceptanceError(f"影子封存水位未清空: {name}: {sealed_through}")
    for date in days:
        for name in _LABEL_NAMES:
            path = scratch_store.labels_dir / name / f"{date}.parquet"
            frame = pd.read_parquet(path, columns=["maturity_status"])
            if set(frame["maturity_status"].astype(str)) != {"forming"}:
                raise AcceptanceError(f"影子初态不是 forming: {name}/{date}")


def assert_fixture_panel_fidelity(prod_store: Any, scratch_store: Any, days: Sequence[str]) -> None:
    """复用 P1 只读比较辅助，逐日证明影子 core 键域与中性化输入忠实于生产。"""
    problems: list[str] = []
    for date in days:
        problems.extend(
            _compare_partition_content(
                prod_store.panel_dir / date / "core.parquet",
                scratch_store.panel_dir / date / "core.parquet",
                "core",
            )
        )
    if problems:
        raise AcceptanceError(f"影子 panel core 与生产不一致: {problems[:3]}")


def production_source_identities(store: Any, days: Sequence[str]) -> list[dict[str, Any]]:
    """影子复制来源的生产 panel/labels 文件身份（只读证据，运行前后必须不变）。"""
    paths: list[Path] = []
    for date in days:
        for group in _ALL_GROUPS:
            paths.append(store.panel_dir / date / f"{group}.parquet")
        for name in _LABEL_NAMES:
            paths.append(store.labels_dir / name / f"{date}.parquet")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AcceptanceError(f"影子来源文件缺失: {missing[:5]}")
    return collect_file_identities(paths, base_dir=store.root)


# ---------- 冻结段证据（R2-T4-04 整改：全新只读 manifest 视图复算） ----------


def frozen_registered_mapping(manifest: Manifest) -> dict[str, str]:
    """全量 manifest 中 ≤20260702 的 panel/labels 登记 sha256 映射（按分区种类解析）。"""
    snapshot = manifest.snapshot()
    mapping: dict[str, str] = {}
    for rel, meta in snapshot["partitions"].items():
        parts = rel.split("/")
        if parts[0] == "panel" and parts[1] <= LABEL_FILTERED_THROUGH:
            mapping[rel] = str(meta["sha256_16"])
        elif parts[0] == "panel_archive" and parts[1] <= _ARCHIVE_BOUNDARY_MONTH:
            mapping[rel] = str(meta["sha256_16"])
    for name, entry in snapshot["labels"].items():
        for filename, meta in entry.get("partitions", {}).items():
            if filename.removesuffix(".parquet") <= LABEL_FILTERED_THROUGH:
                mapping[f"labels/{name}/{filename}"] = str(meta["sha256_16"])
    return mapping


def check_file_against_registration(
    registered: str | None, path: Path, target: str
) -> dict[str, Any]:
    actual = sha256_16_of_file(path) if path.is_file() else None
    return {
        "target": target,
        "registered": registered,
        "actual": actual,
        "match": registered is not None and actual == registered,
    }


def physical_spot_checks(store: Any, manifest: Manifest) -> list[dict[str, Any]]:
    """热区按日文件、冷区按月文件指纹 + 文件内 trade_date 过滤抽验。"""
    checks: list[dict[str, Any]] = []
    for group in _ALL_GROUPS:
        rel = f"panel/{HOT_SPOT_DAY}/{group}.parquet"
        checks.append(
            check_file_against_registration(
                manifest.partition_fingerprint(rel),
                store.panel_dir / HOT_SPOT_DAY / f"{group}.parquet",
                rel,
            )
        )
    for name in _LABEL_NAMES:
        checks.append(
            check_file_against_registration(
                manifest.label_partition_fingerprint(name, HOT_SPOT_DAY),
                store.labels_dir / name / f"{HOT_SPOT_DAY}.parquet",
                f"labels/{name}/{HOT_SPOT_DAY}.parquet",
            )
        )
    cold_months = {COLD_SPOT_MONTH} | {month for _, month in COLD_SPOT_SAMPLES}
    for month in sorted(cold_months):
        for group in _ALL_GROUPS:
            rel = f"panel_archive/{month}/{group}.parquet"
            checks.append(
                check_file_against_registration(
                    manifest.partition_fingerprint(rel),
                    store.archive_dir / month / f"{group}.parquet",
                    rel,
                )
            )
    for date, month in COLD_SPOT_SAMPLES:
        path = store.archive_dir / month / "core.parquet"
        rows = 0
        if path.is_file():
            frame = pd.read_parquet(path, columns=["trade_date"])
            rows = int((frame["trade_date"].astype(str) == date).sum())
        checks.append(
            {"target": f"panel_archive/{month}/core#{date}", "rows": rows, "match": rows > 0}
        )
    return checks


def capture_frozen_before(store: Any, data_root: Path) -> dict[str, Any]:
    """验收前冻结段基线：登记映射、抽验、repairs、manifest/.bak 文件身份与旧链资产证据。"""
    manifest_path = store.features_dir / "manifest.json"
    fresh = Manifest(manifest_path)
    failed = [check for check in physical_spot_checks(store, fresh) if not check["match"]]
    if failed:
        raise AcceptanceError(f"验收前冻结段物理抽验失败: {[c['target'] for c in failed][:5]}")
    legacy = legacy_asset_evidence(data_root)
    if legacy["cs_infer_beyond_seam"]:
        raise AcceptanceError(
            f"生产 cs_infer 存在跨缝分区（应不存在）: {legacy['cs_infer_beyond_seam']}"
        )
    manifest_files = [manifest_path]
    bak_path = store.manifest.backup_path
    bak_existed = bak_path.is_file()
    if bak_existed:
        manifest_files.append(bak_path)
    return {
        "registered_mapping": frozen_registered_mapping(fresh),
        "repairs": fresh.snapshot().get("repairs", []),
        "content_fingerprint": fresh.content_fingerprint(),
        "manifest_files": collect_file_identities(manifest_files),
        "bak_existed": bak_existed,
        "legacy_assets": legacy,
    }


def frozen_segment_after_check(
    store: Any, data_root: Path, before: Mapping[str, Any]
) -> dict[str, Any]:
    """验收后复算冻结段：manifest 文件身份、登记映射、抽验、repairs、指纹与旧链资产不变。"""
    mismatches = verify_file_identities(before["manifest_files"])
    if mismatches:
        raise AcceptanceError(f"生产 manifest 文件在验收前后变化: {mismatches}")
    if store.manifest.backup_path.is_file() != before["bak_existed"]:
        raise AcceptanceError("生产 manifest.bak 存在性在验收前后变化")
    if store.wal_path.exists() and store.wal_path.stat().st_size > 0:
        raise AcceptanceError("验收后生产存在未处理 WAL；保留现场，禁止借普通 store 重放压实")
    fresh = Manifest(store.features_dir / "manifest.json")
    after_mapping = frozen_registered_mapping(fresh)
    if after_mapping != before["registered_mapping"]:
        raise AcceptanceError("冻结段登记指纹映射在验收前后不一致")
    checks = physical_spot_checks(store, fresh)
    failed = [check for check in checks if not check["match"]]
    if failed:
        raise AcceptanceError(f"冻结段物理抽验失败: {[c['target'] for c in failed][:5]}")
    if fresh.snapshot().get("repairs", []) != before["repairs"]:
        raise AcceptanceError("repairs 在验收前后不一致")
    if fresh.content_fingerprint() != before["content_fingerprint"]:
        raise AcceptanceError("生产 manifest 内容指纹在验收前后变化")
    if legacy_asset_evidence(data_root) != before["legacy_assets"]:
        raise AcceptanceError("旧 cs_train/cs_infer 资产证据在验收前后变化")
    return {
        "registered_partitions": len(after_mapping),
        "mapping_unchanged": True,
        "spot_checks": checks,
        "spot_check_failures": len(failed),
        "repairs_count": len(before["repairs"]),
        "repairs_unchanged": True,
        "manifest_fingerprint_unchanged": True,
        "manifest_file_identity_unchanged": True,
        "legacy_assets_unchanged": True,
    }


# ---------- 故障注入（R2-T4-06 整改：全新实例重放 WAL 并核对终态） ----------


def _audit_counts(store: Any) -> tuple[int, int]:
    return (
        len(store.manifest.label_refresh_intents()),
        len(store.manifest.label_refreshes()),
    )


def _wal_is_live(store: Any) -> bool:
    return store.wal_path.exists() and store.wal_path.stat().st_size > 0


@contextmanager
def _death_defer(store: Any) -> Any:
    """进程死亡等价的 defer：保持写锁/计数语义，出口不做 sync/save/压实收尾。"""
    store.write_lock().acquire()
    store._manifest_defer += 1
    try:
        yield
    finally:
        store._manifest_defer -= 1
        store.write_lock().release(during_exception=True)


def _expect_crash(call: Any, marker: str) -> None:
    try:
        call()
    except RuntimeError as exc:
        if marker not in str(exc):
            raise
    else:
        raise AcceptanceError(f"故障注入未按预期抛错: {marker}")


@contextmanager
def _manifest_save_crash() -> Any:
    original = Manifest.save

    def _crash(self: Any) -> None:
        raise RuntimeError("T4 故障注入：manifest.save 崩溃")

    Manifest.save = _crash
    try:
        yield
    finally:
        Manifest.save = original


def _assert_terminal_state_fresh(root: Path, date: str) -> None:
    """恢复终态可由磁盘独立重建：全新实例核对指纹、登记、水位、行域、审计与 WAL。"""
    fresh = PanelDataStore(root)
    for name in _LABEL_NAMES:
        path = fresh.labels_dir / name / f"{date}.parquet"
        registered = fresh.manifest.label_partition_fingerprint(name, date)
        if registered is None or sha256_16_of_file(path) != registered:
            raise AcceptanceError(f"恢复终态登记与文件不一致: {name}/{date}")
        if fresh.manifest.label_sealed_through(name) != date:
            raise AcceptanceError(f"恢复终态封存水位异常: {name}/{date}")
        meta = fresh.manifest.label_partition_meta(name, date)
        if meta is None or meta.get("row_domain") != "label_filtered":
            raise AcceptanceError(f"恢复终态行域异常: {name}/{date}")
    if _audit_counts(fresh) != (1, 1):
        raise AcceptanceError(f"恢复终态审计计数异常: {_audit_counts(fresh)}")
    if _wal_is_live(fresh):
        raise AcceptanceError("恢复终态仍存在未压实 WAL")


def _assert_rerun_noop(store: Any, date: str, refresh_kwargs: Mapping[str, Any]) -> None:
    if validate_label_day_readiness(store, date)["ready"] is not True:
        raise AcceptanceError("恢复后 readiness 未通过")
    if refresh_labels_for_day(store, date, **refresh_kwargs).status != "no_op":
        raise AcceptanceError("恢复后继续重跑不是 no_op")
    if _audit_counts(store) != (1, 1):
        raise AcceptanceError("恢复后审计被追加")


def _new_fault_store(prod_store: Any, root: Path, date: str) -> Any:
    store = PanelDataStore(root)
    bootstrap_manifest(store)
    copy_shadow_day_to_scratch(prod_store, store, date)
    return store


def intent_crash_scenario(
    prod_store: Any, scratch_root: Path, date: str, refresh_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """刷新意图前崩溃：零副作用现场，普通入口重跑完成刷新（§7.3 注入点 1）。"""
    root = scratch_root / "shadow_fault_intent"
    store = _new_fault_store(prod_store, root, date)
    wal = store._wal
    original = wal.append_label_refresh_intent

    def _crash(record: Mapping[str, Any]) -> bool:
        raise RuntimeError("T4 故障注入：刷新意图前崩溃")

    wal.append_label_refresh_intent = _crash
    try:
        _expect_crash(
            lambda: refresh_labels_for_day(store, date, **refresh_kwargs), "刷新意图前崩溃"
        )
    finally:
        wal.append_label_refresh_intent = original
    if _wal_is_live(store):
        raise AcceptanceError("意图前崩溃不应产生活动 WAL")
    reopened = PanelDataStore(root)
    if _audit_counts(reopened) != (0, 0) or _current_status(reopened, date) != "forming":
        raise AcceptanceError("意图前崩溃后现场不干净")
    refreshed = refresh_labels_for_day(reopened, date, **refresh_kwargs)
    if refreshed.status != "refreshed" or _audit_counts(reopened) != (1, 1):
        raise AcceptanceError("意图前崩溃后普通重跑未完成刷新")
    _assert_terminal_state_fresh(root, date)
    return {
        "scenario": "intent_crash_clean_rerun",
        "injected_day": date,
        "wal_live_at_crash": False,
        "rerun_status": "refreshed",
        "audit_counts": [1, 1],
    }


def partial_tables_crash_scenario(
    prod_store: Any, scratch_root: Path, date: str, refresh_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """分区 WAL 已写后崩溃：首表 sealed、后两表 forming 的活动 WAL 现场恢复（注入点 2）。"""
    root = scratch_root / "shadow_fault_partial"
    store = _new_fault_store(prod_store, root, date)
    original_append = store.append_labels
    calls = {"count": 0}

    def _crash_second(name: str, table: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("T4 故障注入：分区 WAL 已写后崩溃")
        return original_append(name, table)

    store.append_labels = _crash_second
    store.defer_manifest_save = lambda: _death_defer(store)  # 进程死亡等价：阻止异常收尾
    _expect_crash(
        lambda: refresh_labels_for_day(store, date, **refresh_kwargs), "分区 WAL 已写后崩溃"
    )
    if not _wal_is_live(store):
        raise AcceptanceError("部分表崩溃后活动 WAL 应保留（进程死亡等价）")
    reopened = PanelDataStore(root)
    if _audit_counts(reopened) != (1, 0):
        raise AcceptanceError(f"部分表崩溃重放后审计计数异常: {_audit_counts(reopened)}")
    if reopened.manifest.label_sealed_through("y_ret_5") != date:
        raise AcceptanceError("首表封存水位未按 WAL 重放推进")
    if _current_status(reopened, date) != "mixed":
        raise AcceptanceError("部分表崩溃后应为首表 sealed、其余 forming 的部分态")
    recovered = refresh_labels_for_day(reopened, date, **refresh_kwargs)
    if recovered.status != "recovered" or _audit_counts(reopened) != (1, 1):
        raise AcceptanceError("部分表崩溃恢复未按同一事件键恰补一次")
    _assert_rerun_noop(reopened, date, refresh_kwargs)
    _assert_terminal_state_fresh(root, date)
    return {
        "scenario": "partial_tables_live_wal",
        "injected_day": date,
        "wal_live_at_crash": True,
        "recovered_exactly_once": True,
        "audit_counts": [1, 1],
    }


def completion_crash_scenario(
    prod_store: Any, scratch_root: Path, date: str, refresh_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """第三表后完成审计前崩溃：活动 WAL 留存，全新实例重放后恰补一次（注入点 3）。"""
    root = scratch_root / "shadow_fault_completion"
    store = _new_fault_store(prod_store, root, date)
    wal = store._wal
    original = wal.append_label_refresh_completion

    def _crash(record: Mapping[str, Any]) -> bool:
        raise RuntimeError("T4 故障注入：完成审计前崩溃")

    wal.append_label_refresh_completion = _crash
    store.defer_manifest_save = lambda: _death_defer(store)  # 进程死亡等价：阻止异常收尾
    try:
        _expect_crash(
            lambda: refresh_labels_for_day(store, date, **refresh_kwargs), "完成审计前崩溃"
        )
    finally:
        wal.append_label_refresh_completion = original
    if not _wal_is_live(store):
        raise AcceptanceError("完成审计前崩溃后活动 WAL 应保留（进程死亡等价）")
    reopened = PanelDataStore(root)
    if _audit_counts(reopened) != (1, 0):
        raise AcceptanceError(f"崩溃后全新实例重放审计计数异常: {_audit_counts(reopened)}")
    if _current_status(reopened, date) != "sealed":
        raise AcceptanceError("崩溃后三表盘上终态不是 sealed")
    recovered = refresh_labels_for_day(reopened, date, **refresh_kwargs)
    if recovered.status != "recovered" or _audit_counts(reopened) != (1, 1):
        raise AcceptanceError("崩溃恢复未按同一事件键恰补一次")
    _assert_rerun_noop(reopened, date, refresh_kwargs)
    _assert_terminal_state_fresh(root, date)
    return {
        "scenario": "completion_crash_live_wal",
        "injected_day": date,
        "wal_live_at_crash": True,
        "recovered_exactly_once": True,
        "audit_counts": [1, 1],
    }


def manifest_save_crash_scenario(
    prod_store: Any, scratch_root: Path, date: str, refresh_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """manifest.save 前崩溃：活动 WAL 全事件重放恢复；禁用重放则门必失败（注入点 4）。"""
    root = scratch_root / "shadow_fault_save"
    store = _new_fault_store(prod_store, root, date)
    with _manifest_save_crash():
        _expect_crash(
            lambda: refresh_labels_for_day(store, date, **refresh_kwargs), "manifest.save 崩溃"
        )
    if not _wal_is_live(store):
        raise AcceptanceError("manifest.save 崩溃后活动 WAL 应保留")
    disk_only = Manifest(root / "features" / "manifest.json")
    if disk_only.label_refreshes():
        raise AcceptanceError("盘上 manifest 不应含完成审计（重放依赖证据）")
    reopened = PanelDataStore(root)
    if _audit_counts(reopened) != (1, 1):
        raise AcceptanceError(
            f"活动 WAL 重放后审计计数异常: {_audit_counts(reopened)}（禁用重放时此处必失败）"
        )
    if _wal_is_live(reopened):
        raise AcceptanceError("重放完成后 WAL 应已压实")
    if refresh_labels_for_day(reopened, date, **refresh_kwargs).status != "no_op":
        raise AcceptanceError("重放恢复后继续重跑不是 no_op")
    _assert_terminal_state_fresh(root, date)
    return {
        "scenario": "manifest_save_crash_wal_replay",
        "injected_day": date,
        "wal_live_at_crash": True,
        "replay_required_evidence": {
            "disk_manifest_completions": 0,
            "replayed_audit_counts": [1, 1],
        },
        "audit_counts": [1, 1],
    }


def _inject_replace_wal_crash(store: Any, date: str, refresh_kwargs: Mapping[str, Any]) -> None:
    wal = store._wal
    original = wal.register_label_partition

    def _crash(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("T4 故障注入：replace 后分区 WAL 前崩溃")

    wal.register_label_partition = _crash
    try:
        _expect_crash(
            lambda: refresh_labels_for_day(store, date, **refresh_kwargs), "分区 WAL 前崩溃"
        )
    finally:
        wal.register_label_partition = original


def _normal_entry_event_key(store: Any, date: str, refresh_kwargs: Mapping[str, Any]) -> str:
    """replace→WAL 崩溃窗：普通入口必须 fail-closed 并给出受控恢复事件键。"""
    try:
        refresh_labels_for_day(store, date, **refresh_kwargs)
    except LabelRefreshRecoveryError as exc:
        if "--recover-event" not in str(exc):
            raise
        return str(exc).split("--recover-event ", 1)[1].split()[0]
    raise AcceptanceError("replace→分区 WAL 崩溃窗普通入口未 fail-closed")


def replace_wal_window_scenario(
    prod_store: Any, scratch_root: Path, date: str, refresh_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """replace→分区 WAL 崩溃窗：普通入口拒绝并保留现场，受控 roll-forward 后终态一致。"""
    root = scratch_root / "shadow_fault_replace"
    store = _new_fault_store(prod_store, root, date)
    _inject_replace_wal_crash(store, date, refresh_kwargs)
    reopened = PanelDataStore(root)
    event_key = _normal_entry_event_key(reopened, date, refresh_kwargs)
    planned = refresh_labels_for_day(
        reopened, date, recover_event_key=event_key, dry_run=True, **refresh_kwargs
    )
    if planned.status != "recovery_planned":
        raise AcceptanceError(f"受控恢复 dry-run 状态异常: {planned.status}")
    recovered = refresh_labels_for_day(
        reopened, date, recover_event_key=event_key, **refresh_kwargs
    )
    if recovered.status != "recovered" or _audit_counts(reopened) != (1, 1):
        raise AcceptanceError("受控 roll-forward 未按同一事件键恢复")
    _assert_rerun_noop(reopened, date, refresh_kwargs)
    _assert_terminal_state_fresh(root, date)
    return {
        "scenario": "replace_wal_window_controlled_rollforward",
        "injected_day": date,
        "normal_entry_rejected": True,
        "recovered_exactly_once": True,
        "audit_counts": [1, 1],
    }


def rollback_scenario(
    prod_store: Any, scratch_root: Path, date: str, refresh_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """同代回滚反例：故障活动 WAL 隔离 + 同代文件恢复后，全新 store 不重放故障 WAL。"""
    root = scratch_root / "shadow_fault_rollback"
    store = _new_fault_store(prod_store, root, date)
    snapshot_dir = scratch_root / "rollback_snapshot"
    relatives = [Path("features") / "manifest.json", Path("features") / "manifest.json.bak"]
    relatives += [Path("labels") / name / f"{date}.parquet" for name in _LABEL_NAMES]
    saved: dict[str, str] = {}
    for rel in relatives:
        source = root / rel
        if source.exists():
            target = snapshot_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            saved[str(rel)] = sha256_16_of_file(source)
    manifest_fingerprint = Manifest(root / "features" / "manifest.json").content_fingerprint()
    with _manifest_save_crash():
        _expect_crash(
            lambda: refresh_labels_for_day(store, date, **refresh_kwargs), "manifest.save 崩溃"
        )
    if not _wal_is_live(store):
        raise AcceptanceError("回滚反例要求故障后活动 WAL 非空")
    quarantine = scratch_root / "rollback_quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    shutil.move(str(store.wal_path), quarantine / "manifest_wal.jsonl")
    for rel_str, fingerprint in saved.items():
        shutil.copy2(snapshot_dir / rel_str, root / rel_str)
        if sha256_16_of_file(root / rel_str) != fingerprint:
            raise AcceptanceError(f"回滚文件指纹不一致: {rel_str}")
    fresh = PanelDataStore(root)
    if _audit_counts(fresh) != (0, 0):
        raise AcceptanceError("回滚后故障 WAL 仍被重放")
    if _current_status(fresh, date) != "forming":
        raise AcceptanceError("回滚后标签未回到 forming 初态")
    for name in _LABEL_NAMES:
        if fresh.manifest.label_sealed_through(name) is not None:
            raise AcceptanceError(f"回滚后封存水位未清空: {name}")
    if Manifest(root / "features" / "manifest.json").content_fingerprint() != manifest_fingerprint:
        raise AcceptanceError("回滚后 manifest 内容指纹与同代快照不一致")
    refreshed = refresh_labels_for_day(fresh, date, **refresh_kwargs)
    if refreshed.status != "refreshed" or _audit_counts(fresh) != (1, 1):
        raise AcceptanceError("回滚后干净重跑未完成刷新")
    _assert_terminal_state_fresh(root, date)
    return {
        "scenario": "same_generation_rollback",
        "injected_day": date,
        "wal_quarantined": True,
        "fault_wal_replayed": False,
        "rerun_status": "refreshed",
        "audit_counts": [1, 1],
    }


def fault_injection_check(
    prod_store: Any, scratch_root: Path, date: str, refresh_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """scratch 内六场景故障门：§7.3 四注入点 + 受控恢复 + 同代回滚，全新实例核验终态。"""
    scenarios = [
        intent_crash_scenario(prod_store, scratch_root, date, refresh_kwargs),
        partial_tables_crash_scenario(prod_store, scratch_root, date, refresh_kwargs),
        completion_crash_scenario(prod_store, scratch_root, date, refresh_kwargs),
        manifest_save_crash_scenario(prod_store, scratch_root, date, refresh_kwargs),
        replace_wal_window_scenario(prod_store, scratch_root, date, refresh_kwargs),
        rollback_scenario(prod_store, scratch_root, date, refresh_kwargs),
    ]
    return {
        "scenarios": scenarios,
        "coverage_note": "§7.3 注入点覆盖：意图前、分区 WAL 已写后、第三表后完成审计前、"
        "manifest.save 前、replace→WAL 崩溃窗受控 roll-forward、故障 WAL 非空同代回滚；"
        "活动 WAL 场景均以进程死亡等价注入保留现场，全新实例重放并核对终态；"
        "manifest_save 场景含重放依赖反变异证据（盘上无完成审计、重放后方完整）",
    }


# ---------- 跨缝影响计数与补充验收报告（R2-T4-CLOSE-01 整改） ----------


def _sum_cross_seam_impacts(impacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """聚合多个 `cross_seam_impact_for_day` 结果。"""
    total: dict[str, Any] = {}
    for label in ("y_ret_5", "y_ret_10"):
        by_industry: dict[str, int] = {}
        rows = 0
        for impact in impacts:
            rows += int(impact[label]["affected_rows"])
            for industry, count in impact[label]["by_industry"].items():
                by_industry[industry] = by_industry.get(industry, 0) + int(count)
        total[label] = {"affected_rows": rows, "by_industry": by_industry}
    return total


def cross_seam_impact_for_day(store: Any, date: str) -> dict[str, Any]:
    """当日跨缝影响计数：y20 空 & y5/y10 非空的行数与分行业计数（panel core 行业 × 三表标签）。

    结构段（forming 且 E20 未到水位）的 y20 大面积为空但 y5/y10 可非空，本函数是
    §3.1/§7.2 第 6 条对**真实 structural 域**的机械计数入口；label_filtered 影子域的
    计数由刷新结果自带，两者不得混用。
    """
    core = pd.read_parquet(
        store.panel_dir / date / "core.parquet",
        columns=["trade_date", "ts_code", "sw_industry"],
    )
    merged = core.assign(
        trade_date=core["trade_date"].astype(str), ts_code=core["ts_code"].astype(str)
    )
    for name, short in (("y_ret_5", "y5"), ("y_ret_10", "y10"), ("y_ret_20", "y20")):
        table = pd.read_parquet(
            store.labels_dir / name / f"{date}.parquet",
            columns=["trade_date", "ts_code", "label_value"],
        )
        table["trade_date"] = table["trade_date"].astype(str)
        table["ts_code"] = table["ts_code"].astype(str)
        merged = merged.merge(
            table.rename(columns={"label_value": short}),
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
    result: dict[str, Any] = {}
    for short, label in (("y5", "y_ret_5"), ("y10", "y_ret_10")):
        mask = merged["y20"].isna() & merged[short].notna()
        result[label] = {
            "affected_rows": int(mask.sum()),
            "by_industry": {
                str(key): int(value)
                for key, value in merged.loc[mask]
                .groupby("sw_industry", dropna=False)
                .size()
                .items()
            },
        }
    return result


def _dir_watermark(data_root: Path) -> str:
    stems = [
        path.stem.replace("-", "") for path in (data_root / "clean" / "daily").glob("*.parquet")
    ]
    if not stems:
        raise AcceptanceError(f"clean/daily 分区目录为空: {data_root / 'clean' / 'daily'}")
    return max(stems)


def _production_append_scope(prod_store: Any, manifest_view: Manifest) -> dict[str, Any]:
    """生产追加段（structural 全部已登记 labels 日）的逐日与合计跨缝计数。

    manifest_view 为纯只读视图（`Manifest(path)`，不触发 WAL 持久化重放）；
    prod_store 仅用于路径解析，不访问其 manifest 属性。
    """
    snapshot = manifest_view.snapshot()
    append_dates = sorted(
        name.removesuffix(".parquet")
        for name in snapshot["labels"].get("y_ret_20", {}).get("partitions", {})
        if STRUCTURAL_FROM <= name.removesuffix(".parquet")
    )
    per_day = [
        {"trade_date": date, **cross_seam_impact_for_day(prod_store, date)} for date in append_dates
    ]
    return {
        "available": bool(append_dates),
        "row_domain": "structural",
        "days": len(append_dates),
        "range": f"{append_dates[0]}~{append_dates[-1]}" if append_dates else None,
        "per_day": per_day,
        "total": _sum_cross_seam_impacts(per_day),
        "scope_note": "D 后生产当前状态；forming 段 y20 未到水位大面积为空，y5/y10 受影响为"
        "暂态计数（随 E20 到达刷新后收敛），不构成投资损益或永久母域差异",
    }


def _scratch_daily_scope(original: Mapping[str, Any]) -> dict[str, Any]:
    """原 scratch 日更域重算；scratch 已清理时显式标 unavailable。"""
    scratch_daily = Path(str(original["scratch_root"])) / "daily"
    if not scratch_daily.is_dir():
        return {"available": False, "note": "原 scratch 日更 store 已清理，无法重算"}
    scratch_store = PanelDataStore(scratch_daily)
    daily_dates = [entry["trade_date"] for entry in original["real_daily"]["entries"]]
    per_day = [
        {"trade_date": date, **cross_seam_impact_for_day(scratch_store, date)}
        for date in daily_dates
    ]
    return {
        "available": True,
        "row_domain": "structural",
        "per_day": per_day,
        "total": _sum_cross_seam_impacts(per_day),
    }


def validate_output_path(
    data_root: Path, out_path: Path, *, scratch_root: Path | None = None
) -> Path:
    """输出路径隔离（主验收与补充报告共享）：规范 .json、最终/原子临时目标不得存在、
    生产数据根内仅允许 reports/ 子树、不得落在 scratch 内；任何读取之前调用。"""
    resolved = out_path.resolve()
    if resolved.suffix != ".json":
        raise AcceptanceError(f"输出路径须为规范 .json 文件: {resolved}")
    if resolved.exists():
        raise AcceptanceError(f"输出已存在，拒绝覆盖: {resolved}")
    tmp = resolved.with_suffix(".json.tmp")  # 与 _write_json_atomic 的实际临时目标一致
    if tmp.exists():
        raise AcceptanceError(f"输出原子临时目标已存在，拒绝覆盖既有现场: {tmp}")
    root = data_root.resolve()
    if resolved.is_relative_to(root) and not resolved.is_relative_to(root / "reports"):
        raise AcceptanceError(f"输出落在生产数据根内但不在 reports/ 下: {resolved}")
    if scratch_root is not None and resolved.is_relative_to(scratch_root.resolve()):
        raise AcceptanceError(f"输出不得落在 scratch 根内: {resolved}")
    return resolved


def supplement_cross_seam(
    original_report_path: Path | str,
    out_path: Path | str,
    *,
    data_root: Path | str | None = None,
) -> dict[str, Any]:
    """跨缝补充验收报告（R2-T4-CLOSE-01 整改）：影子域/scratch 日更域/生产追加段三范围分列。

    只读既有报告、原 scratch 日更 store 与当前生产标签/panel core，不重建、不改写任何资产。
    """
    original_path = Path(original_report_path)
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes.decode("utf-8"))
    if original.get("status") != "passed":
        raise AcceptanceError(f"原验收报告非 passed，拒绝补充: {original.get('status')}")
    root = Path(data_root) if data_root is not None else Path(original["data_root"])
    out = validate_output_path(
        root, Path(out_path), scratch_root=Path(str(original["scratch_root"]))
    )
    wal_path = root / "features" / "manifest_wal.jsonl"
    if wal_path.exists() and wal_path.stat().st_size > 0:
        raise AcceptanceError(f"只读补充拒绝活动 WAL，禁止借补充自动恢复生产登记: {wal_path}")
    manifest_path = root / "features" / "manifest.json"
    bak_path = root / "features" / "manifest.json.bak"
    manifest_files = [manifest_path] + ([bak_path] if bak_path.is_file() else [])
    bak_existed = bak_path.is_file()
    identity_before = collect_file_identities(manifest_files)
    manifest_view = Manifest(manifest_path)  # 纯只读视图，不触发 WAL 持久化重放
    shadow_impacts = [day["cross_seam_impact"] for day in original["historical_shadow"]["per_day"]]
    historical_shadow = {
        **_sum_cross_seam_impacts(shadow_impacts),
        "days": original["precheck"]["shadow_count"],
        "range": original["precheck"]["shadow_range"],
        "row_domain": "label_filtered",
        "scope_note": "影子域 y20 已成熟，受影响计数趋零是该域必然结果，不代表 structural 域",
    }
    # 全部读取先行（生产枚举、scratch 日更、目录水位），终核覆盖完整读取过程
    prod_store = PanelDataStore(root)
    production_append = _production_append_scope(prod_store, manifest_view)
    scratch_scope = _scratch_daily_scope(original)
    clean_watermark = _dir_watermark(root)
    if verify_file_identities(identity_before) or bak_path.is_file() != bak_existed:
        raise AcceptanceError("补充报告运行期间生产 manifest/.bak 状态变化；保留现场并拒绝")
    if wal_path.exists() and wal_path.stat().st_size > 0:
        raise AcceptanceError("补充报告运行期间出现活动 WAL；保留现场并拒绝")
    report = {
        "supplement_of": str(original_path),
        "original_run_id": original["run_id"],
        "original_report_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_identity": {
            "data_root": str(root.resolve()),
            "manifest_content_fingerprint": manifest_view.content_fingerprint(),
            "label_watermark": original["identity"]["loaded_watermark"],
            "clean_dir_watermark": clean_watermark,
            "manifest_file_identity_unchanged": True,
        },
        "scopes": {
            "historical_shadow": historical_shadow,
            "real_daily_scratch": scratch_scope,
            "production_append": production_append,
        },
        "restriction": "B0 14 折不跨缝；y5/y10 跨缝母域差异未裁决前不得跨缝使用",
        "status": "passed",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out, report)
    print(f"SUPPLEMENT_REPORT {out} status=passed")
    prod_total = production_append["total"]
    print(
        f"CROSS_SEAM production_append days={production_append['days']} "
        f"y5_affected={prod_total['y_ret_5']['affected_rows']} "
        f"y10_affected={prod_total['y_ret_10']['affected_rows']}"
    )
    return report
