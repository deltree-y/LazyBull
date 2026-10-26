"""P3 T0.5b 运行基线（消费集合 + 生产守卫快照）获取与核验（S11 拆文件）。

职责：消费集合枚举（按层真实命名归一）、基线快照首次创建（append-only）、
续跑基线加载与即时复核（R2-T05B-04：先核验后复用，禁止先覆盖再核验；
**已有缓存结果但原始基线缺失 ⇒ 拒绝重新采集后复用——该检查独立于
``--skip-existing`` 且先于任何快照写入**，复核三余项）。

P3 T5 评审处置（R2-T5A-02）：``enumerate_consumption_files`` 增
``include_panel_labels`` 可选扩展——臂③（v2_panel）真实消费还包括
``features/panel`` 热区日目录、``features/panel_archive`` 冷区月目录、
``labels/`` 三表分区与 ``features/manifest.json(.bak)`` / ``manifest_wal.jsonl``；
panel/labels 分区入字节门（≤ 冻结段末过滤、零覆盖即拒绝），**manifest 族不入
字节门**，走相关性 diff 语义（复用 ``t1_reconcile_adapter._manifest_relevance_diff``
口径，另落 ``manifest_baseline.json`` 基线）。默认 ``False`` ⇒ 与 spike 现状逐位一致。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

from scripts.v2_p3.t05b_isolated_root import (
    collect_consumption_snapshot,
    snapshot_production_models_guard,
    verify_consumption_snapshot_closed,
    verify_production_models_guard,
)

# 消费集合枚举上限（冻结段口径，§13.3.3：窗口 ⊂ 冻结段 ≤20260702）；
# 超出该日的日更追加分区不属于本 spike 消费集合（不触发拒绝，M5 口径）
_FROZEN_SEGMENT_END = "20260702"

# 按日分区的消费目录（cs_train=训练/评估；clean/daily=OOS 回测行情；raw/suspend=卖出停牌日历）
_DATED_CONSUMPTION_DIRS = ("features/cs_train", "clean/daily", "raw/suspend")
#: 闭市日判定 raw 输入（R2-T5AE-02）：仅 T5 臂评分启用（从 score_days 的 raw 三列全空
#: 决定 excluded_days，直接影响覆盖门）；历史 spike 默认不纳入以保持既有行为。
_MONEYFLOW_HSGT_DIR = "raw/moneyflow_hsgt"
# 单文件消费（域股票池与交易日历）
_SINGLE_CONSUMPTION_GLOBS = ("clean/stock_basic*", "clean/trade_cal*", "raw/trade_cal*")
_SINGLE_FILES_LABEL = "（单文件）"

# --- R2-T5A-02 扩展枚举（include_panel_labels=True，臂③真实消费） ---
_PANEL_HOT_DIR = "features/panel"
_PANEL_ARCHIVE_DIR = "features/panel_archive"
_LABELS_DIR = "labels"
#: manifest 族文件名（审计身份保留，**不入字节门**，走相关性 diff 语义）
MANIFEST_FAMILY_NAMES = ("manifest.json", "manifest.json.bak", "manifest_wal.jsonl")
_MANIFEST_FAMILY_LABEL = "（manifest 族，审计不入字节门）"
#: 消费快照分列计数的路径前缀 → 目录标签（含扩展目录；默认枚举无此类记录 ⇒ 现状不变）
_COUNT_PREFIX_DIRS = _DATED_CONSUMPTION_DIRS + (_PANEL_ARCHIVE_DIR, _PANEL_HOT_DIR, _LABELS_DIR)


def _normalize_partition_day(stem: str) -> str:
    """分区文件名日期归一：``YYYYMMDD`` / ``YYYY-MM-DD`` 两种命名统一为 ``YYYYMMDD``（N1）。"""
    return stem.replace("-", "")


def enumerate_consumption_files(
    production_data_root: Path,
    include_panel_labels: bool = False,
    include_moneyflow_hsgt: bool = False,
) -> Dict[str, List[Path]]:
    """枚举 spike 实际消费文件，**按目录分列**返回（§13.3.4 消费集合；冻结段保守超集）。

    - ``features/cs_train`` / ``clean/daily`` / ``raw/suspend`` 的 ``*.parquet``：
      文件名日期归一（``YYYY-MM-DD`` → ``YYYYMMDD``，N1）后取 ≤ 冻结段末；
      任一目录冻结段内零覆盖即拒绝（零覆盖须可复核理由，R2-T05B-01 复核口径）；
    - ``clean/stock_basic*``、``clean/trade_cal*``、``raw/trade_cal*``（域池与日历）。

    ``include_panel_labels=True``（R2-T5A-02，臂③ v2_panel 真实消费）追加：

    - ``features/panel`` 热区日目录 + ``features/panel_archive`` 冷区月目录
      （按族冷热同源解析，镜像 ``t1_reconcile_adapter._panel_partition_files`` 口径，
      均 ≤ 冻结段末；冷热合计零覆盖即拒绝，防静默漏枚举）；
    - ``labels/`` 三表分区（``*.parquet`` 文件名日期 ≤ 冻结段末；目录缺失
      ⇒ FileNotFoundError，冻结段内零覆盖 ⇒ ValueError，均显式失败）；
    - ``features/manifest.json(.bak)`` / ``manifest_wal.jsonl``（存在才收录，
      列于 ``（manifest 族，审计不入字节门）``——调用方不得将其并入字节门快照，
      判定走 ``verify_manifest_baseline`` 的相关性 diff 语义）。

    ``include_moneyflow_hsgt=True``（R2-T5AE-02，T5 臂评分）追加
    ``raw/moneyflow_hsgt/*.parquet``（闭市日窄口径判定输入；目录/冻结段零覆盖显式失败）。

    默认 ``False`` ⇒ 与 spike 现状逐位一致。
    """
    production_data_root = Path(production_data_root)
    by_dir: Dict[str, List[Path]] = {}
    for sub in _DATED_CONSUMPTION_DIRS:
        by_dir[sub] = _dated_files_upto_frozen(
            production_data_root / sub,
            missing_msg="消费集合目录缺失",
            empty_msg="消费集合目录冻结段内零覆盖，须登记可复核理由",
        )
    singles: List[Path] = []
    for pattern in _SINGLE_CONSUMPTION_GLOBS:
        singles.extend(sorted(production_data_root.glob(pattern)))
    singles = [path for path in singles if path.is_file()]
    if not singles:
        raise ValueError(f"消费集合单文件（stock_basic/trade_cal）缺失: {production_data_root}")
    by_dir[_SINGLE_FILES_LABEL] = singles
    if include_moneyflow_hsgt:
        by_dir[_MONEYFLOW_HSGT_DIR] = _dated_files_upto_frozen(
            production_data_root / _MONEYFLOW_HSGT_DIR,
            missing_msg="闭市日判定输入目录缺失",
            empty_msg="闭市日判定输入冻结段内零覆盖",
        )
    if include_panel_labels:
        _extend_consumption_with_panel_labels(production_data_root, by_dir)
    return by_dir


def _dated_files_upto_frozen(directory: Path, *, missing_msg: str, empty_msg: str) -> List[Path]:
    """收集目录内 ``*.parquet`` 中日期 ≤ 冻结段末的文件；缺失/零覆盖均显式失败。"""
    if not directory.is_dir():
        raise FileNotFoundError(f"{missing_msg}: {directory}")
    files = []
    for path in sorted(directory.glob("*.parquet")):
        day = _normalize_partition_day(path.stem)
        if len(day) == 8 and day.isdigit() and day <= _FROZEN_SEGMENT_END:
            files.append(path)
    if not files:
        raise ValueError(f"{empty_msg}: {directory}")
    return files


def _is_yyyymm(value: str) -> bool:
    parts = value.split("-")
    return (
        len(parts) == 2
        and len(parts[0]) == 4
        and len(parts[1]) == 2
        and parts[0].isdigit()
        and parts[1].isdigit()
    )


def _panel_partitions_through_frozen(root: Path) -> Tuple[List[Path], List[Path]]:
    """(热区日分区, 冷区月分区) 清单，≤ 冻结段末，**按族冷热同源**（R2-T5A-02）。

    镜像 ``t1_reconcile_adapter._panel_partition_files``（其镜像 store
    ``_load_group_frame`` / ``_months_in_range`` 语义）的冷热裁决：一族当月已封存
    ⇒ 热区该月日分区让位冷区。唯一差异 = 直接扫描现存目录代替月份区间迭代
    （枚举语义 = 现存分区中 ≤ 冻结段末的全集，等价于 start ≤ 最早分区的窗口）。
    """
    from src.lazybull.v2.store.column_groups import PANEL_GROUPS

    frozen_month = f"{_FROZEN_SEGMENT_END[:4]}-{_FROZEN_SEGMENT_END[4:6]}"
    archive_root = root / _PANEL_ARCHIVE_DIR
    hot_root = root / _PANEL_HOT_DIR
    archive_months: List[str] = []
    if archive_root.is_dir():
        archive_months = sorted(
            entry.name
            for entry in archive_root.iterdir()
            if entry.is_dir() and _is_yyyymm(entry.name) and entry.name <= frozen_month
        )
    hot_days: List[str] = []
    if hot_root.is_dir():
        hot_days = sorted(
            entry.name
            for entry in hot_root.iterdir()
            if entry.is_dir()
            and len(entry.name) == 8
            and entry.name.isdigit()
            and entry.name <= _FROZEN_SEGMENT_END
        )
    hot_files: List[Path] = []
    archive_files: List[Path] = []
    for group in PANEL_GROUPS:
        archived = [
            month
            for month in archive_months
            if (archive_root / month / f"{group}.parquet").is_file()
        ]
        for month in archived:
            archive_files.append(archive_root / month / f"{group}.parquet")
        archived_set = set(archived)
        for day in hot_days:
            if f"{day[:4]}-{day[4:6]}" in archived_set:
                continue  # 该族当月已封存，冷区为准（与 _load_group_frame 同口径）
            candidate = hot_root / day / f"{group}.parquet"
            if candidate.is_file():
                hot_files.append(candidate)
    return hot_files, archive_files


def _label_partitions_through_frozen(root: Path) -> List[Path]:
    """labels 三表分区（``<表>/<YYYYMMDD>.parquet``，≤ 冻结段末）；目录缺失显式失败。"""
    labels_root = root / _LABELS_DIR
    if not labels_root.is_dir():
        raise FileNotFoundError(
            f"消费集合 labels 目录缺失（臂③真实消费，启用扩展后禁止静默跳过）: {labels_root}"
        )
    files: List[Path] = []
    for table_dir in sorted(labels_root.iterdir()):
        if not table_dir.is_dir():
            continue
        for path in sorted(table_dir.glob("*.parquet")):
            if len(path.stem) == 8 and path.stem.isdigit() and path.stem <= _FROZEN_SEGMENT_END:
                files.append(path)
    return files


def _extend_consumption_with_panel_labels(root: Path, by_dir: Dict[str, List[Path]]) -> None:
    """扩展枚举落地（R2-T5A-02）：panel 热/冷区 + labels 三表 + manifest 族（审计）。"""
    hot_files, archive_files = _panel_partitions_through_frozen(root)
    if not (hot_files or archive_files):
        raise ValueError(
            f"消费集合 panel 热/冷区冻结段内零覆盖，须登记可复核理由: {root / 'features'}"
        )
    by_dir[_PANEL_HOT_DIR] = hot_files
    by_dir[_PANEL_ARCHIVE_DIR] = archive_files
    label_files = _label_partitions_through_frozen(root)
    if not label_files:
        raise ValueError(f"消费集合 labels 冻结段内零覆盖，须登记可复核理由: {root / _LABELS_DIR}")
    by_dir[_LABELS_DIR] = label_files
    manifest_files: List[Path] = []
    for name in MANIFEST_FAMILY_NAMES:
        candidate = root / "features" / name
        if candidate.is_file():
            manifest_files.append(candidate)
    by_dir[_MANIFEST_FAMILY_LABEL] = manifest_files


def flatten_consumption_files(by_dir: Dict[str, List[Path]]) -> List[Path]:
    """按目录分列的消费集合压平为文件清单。"""
    return [path for files in by_dir.values() for path in files]


def _sha256_file_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _consumption_counts_from_snapshot(snapshot_path: Path) -> Dict[str, int]:
    """从落盘清单推导目录分列计数（路径前缀匹配；含 R2-T5A-02 扩展目录）。"""
    records = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    counts: Dict[str, int] = {}
    for record in records:
        display = str(record["path"]).replace("\\", "/")
        for sub in _COUNT_PREFIX_DIRS:
            if f"data/{sub}/" in display:
                counts[sub] = counts.get(sub, 0) + 1
                break
        else:
            counts[_SINGLE_FILES_LABEL] = counts.get(_SINGLE_FILES_LABEL, 0) + 1
    return counts


def _scope_days_for_manifest_diff(
    by_dir: Dict[str, List[Path]], manifest_snapshot: Dict[str, object]
) -> List[str]:
    """manifest 相关性 diff 的日域口径（冻结段内实际消费日 + 基线登记日兜底）。

    来源：枚举的 panel 热区日目录名 / labels 分区文件名 / panel_archive 冷区月
    （以伪首日 ``YYYYMM01`` 承载月份入 month_set——``_manifest_relevance_diff``
    仅支持日序列入参）；叠加基线 manifest 窗内已登记的 panel/labels 分区日
    （登记先于文件或孤儿登记的窗内改写亦可检出）。窗口外（> 冻结段末）日更
    追加不在日域 ⇒ 放行（T1 reviewer2-N2 口径沿用）。
    """
    days = set()
    for path in by_dir.get(_PANEL_HOT_DIR, ()):
        days.add(path.parent.name)
    for path in by_dir.get(_LABELS_DIR, ()):
        days.add(path.stem)
    frozen_month = f"{_FROZEN_SEGMENT_END[:4]}-{_FROZEN_SEGMENT_END[4:6]}"
    for path in by_dir.get(_PANEL_ARCHIVE_DIR, ()):
        month = path.parent.name  # YYYY-MM
        days.add(f"{month[:4]}{month[5:7]}01")
    partitions = manifest_snapshot.get("partitions", {})
    for rel in partitions:
        parts = str(rel).split("/")
        if len(parts) != 3:
            continue
        if parts[0] == "panel" and parts[1] <= _FROZEN_SEGMENT_END:
            days.add(parts[1])
        elif parts[0] == "panel_archive" and parts[1] <= frozen_month:
            days.add(parts[1].replace("-", "") + "01")
    labels = manifest_snapshot.get("labels", {})
    for label_entry in labels.values():
        for filename in (label_entry or {}).get("partitions", {}):  # type: ignore[union-attr]
            stem = str(filename).removesuffix(".parquet")
            if len(stem) == 8 and stem.isdigit() and stem <= _FROZEN_SEGMENT_END:
                days.add(stem)
    return sorted(days)


def _write_manifest_baseline(
    out_path: Path,
    production_data_root: Path,
    base_dir: Path,
    by_dir: Dict[str, List[Path]],
) -> None:
    """manifest 族基线落盘（``manifest_baseline.json``；R2-T5A-02，不入字节门）。

    内容 = manifest 深拷贝快照（T1 同口径：经 store 读取，WAL 吸收后）+ WAL
    待处理态 + 相关性 diff 日域 + manifest 族文件审计身份（仅展示，不参与判定）。
    """
    from scripts.v2_p3.t1_reconcile_adapter import _manifest_snapshot, _wal_pending
    from src.lazybull.v2.store.store_identity import collect_file_identities

    snapshot = dict(_manifest_snapshot(production_data_root))
    wal_pending = _wal_pending(production_data_root)
    manifest_files = list(by_dir.get(_MANIFEST_FAMILY_LABEL, ()))
    audit = collect_file_identities(manifest_files, base_dir=base_dir) if manifest_files else []
    doc = {
        "spec": (
            "manifest 族基线（R2-T5A-02）：不入字节门；复核 = "
            "t1_reconcile_adapter._manifest_relevance_diff 相关性 diff"
            "（列定义/窗内分区登记/sealed_through/行域段绑定；窗口外正常追加放行）"
            "+ WAL 待处理态前后一致；audit_identities 仅审计展示"
        ),
        "frozen_segment_end": _FROZEN_SEGMENT_END,
        "manifest_snapshot": snapshot,
        "wal_pending": wal_pending,
        "scope_days": _scope_days_for_manifest_diff(by_dir, snapshot),
        "audit_identities": audit,
    }
    Path(out_path).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"manifest 族基线落盘: {out_path}（日域 {len(doc['scope_days'])} 天）")


def verify_manifest_baseline(baseline_path: Path, production_data_root: Path) -> List[str]:
    """按落盘基线重算 manifest 相关性 diff + WAL 待处理态一致；空列表 = 零漂移。

    复用 ``t1_reconcile_adapter`` 口径（R2-T5A-02）：列定义 / 窗内 panel 与标签
    分区登记 / sealed_through / 行域段 / manifest_version；窗口外正常追加放行；
    复核解析失败（运行期损坏）同样判漂移。
    """
    from scripts.v2_p3.t1_reconcile_adapter import (
        _manifest_relevance_diff,
        _manifest_snapshot,
        _wal_pending,
    )

    doc = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    days = [str(day) for day in doc["scope_days"]]
    before: Dict[str, object] = doc["manifest_snapshot"]  # type: ignore[assignment]
    try:
        after = _manifest_snapshot(production_data_root)
        issues = list(_manifest_relevance_diff(before, after, days))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        issues = [f"manifest 复核解析失败（运行期损坏）: {exc}"]
    wal_after = _wal_pending(production_data_root)
    if wal_after != bool(doc["wal_pending"]):
        issues.append(f"WAL 待处理态漂移: 基线={doc['wal_pending']} 当前={wal_after}")
    return issues


def _acquire_baselines(
    run_dir: Path,
    production_data_root: Path,
    skip_existing: bool,
    cache_probe: Optional[Callable[[Path], bool]] = None,
    include_panel_labels: bool = False,
    include_moneyflow_hsgt: bool = False,
) -> Dict[str, object]:
    """运行基线获取：首次新建（append-only）或续跑核验（R2-T05B-04 三轮复核余项）。

    分支口径：
    - **已有任一 run 缓存指纹 ⇒ 两份原始基线必须齐全（任一或同时缺 ⇒ 拒绝）——
      该检查与 ``--skip-existing`` 无关且先于任何快照写入**（复核三：堵住
      「不带参数启动失败但已重建基线 → 再带参数续跑洗白基线」路径）；
    - 基线齐全且指定续跑 ⇒ 即时复核守卫与消费集合（漂移即拒绝）后沿用；
    - 无缓存且基线已存在：未指定续跑 ⇒ 拒绝覆盖（append-only）；
    - 无缓存且无基线：首次运行，采集并落盘两份快照。

    ``cache_probe``（P3 T5）：自定义「已有运行缓存」判定（T5 臂布局 =
    folds/splitNN + runs/，与 spike 的 run{1,2}/ 不同）；None ⇒ spike 现状判定，
    行为逐位不变。

    ``include_panel_labels``（R2-T5A-02，臂③）：消费集合枚举扩展到 panel 热/冷区
    与 labels 三表（入字节门），manifest 族另落 ``manifest_baseline.json``（相关性
    diff 基线，不入字节门）；启用后基线齐备口径 = 三份（缺 manifest 基线同判缺失）。

    ``include_moneyflow_hsgt``（R2-T5AE-02，T5 臂）：把闭市日判定 raw 输入
    ``raw/moneyflow_hsgt`` 纳入字节门；默认 ``False`` ⇒ spike 现状逐位一致。
    """
    guard_snapshot_path = run_dir / "before_guard.json"
    consumption_snapshot_path = run_dir / "consumption_snapshot.json"
    manifest_baseline_path = run_dir / "manifest_baseline.json"
    # 显示路径基目录 = 生产数据根的父级（生产=REPO_ROOT；测试可注入临时数据根）
    base_dir = production_data_root.resolve().parent
    guard_exists = guard_snapshot_path.exists()
    consumption_exists = consumption_snapshot_path.exists()
    manifest_exists = manifest_baseline_path.exists()
    if cache_probe is None:
        has_run_cache = any((run_dir / f"run{idx}" / "fingerprint.json").exists() for idx in (1, 2))
    else:
        has_run_cache = bool(cache_probe(run_dir))

    baselines_complete = (
        guard_exists and consumption_exists and (manifest_exists or not include_panel_labels)
    )
    if has_run_cache and not baselines_complete:
        required = "before_guard/consumption_snapshot"
        if include_panel_labels:
            required += "/manifest_baseline"
        raise RuntimeError(
            f"已有缓存结果但原始基线缺失（{required} 任一或同时缺失），"
            "禁止重新采集后复用（R2-T05B-04 复核三：检查独立于 --skip-existing，"
            "先于任何快照写入）；请换新 --run-name 重新取证"
        )
    if skip_existing and baselines_complete:
        guard_before = json.loads(guard_snapshot_path.read_text(encoding="utf-8"))
        guard_drift = verify_production_models_guard(guard_before)
        # 续跑基线核验（R2-T5AE-R1-02）：闭合核验 = 逐文件身份 + 重枚举闭合
        # （基线采集后到续跑开始前，消费目录内新增文件同样判漂移）
        rescanned_by_dir = enumerate_consumption_files(
            production_data_root,
            include_panel_labels=include_panel_labels,
            include_moneyflow_hsgt=include_moneyflow_hsgt,
        )
        rescan_files = [
            path
            for label, files in rescanned_by_dir.items()
            if label != _MANIFEST_FAMILY_LABEL
            for path in files
        ]
        consumption_drift = verify_consumption_snapshot_closed(
            consumption_snapshot_path, base_dir=base_dir, rescan_paths=rescan_files
        )
        manifest_drift: List[str] = []
        if include_panel_labels:
            manifest_drift = verify_manifest_baseline(manifest_baseline_path, production_data_root)
        if guard_drift or consumption_drift or manifest_drift:
            raise RuntimeError(
                f"续跑基线漂移，禁止复用（R2-T05B-04）: guard={guard_drift[:3]} "
                f"consumption={consumption_drift[:3]} manifest={manifest_drift[:3]}"
            )
        counts = _consumption_counts_from_snapshot(consumption_snapshot_path)
        resumed = True
        logger.info(f"续跑基线核验通过（未改写快照）: {consumption_snapshot_path}")
    else:
        if guard_exists or consumption_exists or (include_panel_labels and manifest_exists):
            raise RuntimeError("基线快照已存在且未指定 --skip-existing，禁止覆盖（append-only）")
        logger.info("生产模型目录守卫快照（运行前）...")
        guard_before = snapshot_production_models_guard(production_data_root)
        guard_snapshot_path.write_text(
            json.dumps(guard_before, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("消费集合物理身份快照（运行前，§13.3.4 唯一硬门）...")
        by_dir = enumerate_consumption_files(
            production_data_root,
            include_panel_labels=include_panel_labels,
            include_moneyflow_hsgt=include_moneyflow_hsgt,
        )
        # manifest 族不入字节门（相关性 diff 语义），其余枚举全部入字节门快照
        consumption_files = [
            path
            for label, files in by_dir.items()
            if label != _MANIFEST_FAMILY_LABEL
            for path in files
        ]
        collect_consumption_snapshot(
            consumption_files, base_dir=base_dir, out_path=consumption_snapshot_path
        )
        if include_panel_labels:
            _write_manifest_baseline(manifest_baseline_path, production_data_root, base_dir, by_dir)
        counts = {name: len(files) for name, files in by_dir.items()}
        resumed = False
    return {
        "guard_before": guard_before,
        "guard_snapshot_path": guard_snapshot_path,
        "consumption_snapshot_path": consumption_snapshot_path,
        "manifest_baseline_path": manifest_baseline_path if include_panel_labels else None,
        "production_data_root": production_data_root,
        "consumption_counts": counts,
        "consumption_files_count": sum(counts.values()),
        "input_snapshot_digest": _sha256_file_bytes(consumption_snapshot_path),
        "resumed": resumed,
        "base_dir": base_dir,
    }
