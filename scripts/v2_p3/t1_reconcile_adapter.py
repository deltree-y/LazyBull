"""P3 T1 冻结段小窗对账工具（P3 规划 §4.2 三类结论 + cs_train 验证源逐值一致门）。

对账范围 = 列集 + 键集 + 全部特征值 + 六标签值：

- **panel 腿**：适配层 panel 模式输出 vs 现役 ``load_features_data``（cs_train 路径）
  同窗比对，结论三类——**容差内一致项**（atol=1e-6，P1 归因门口径，非逐位精确）/
  **已批准差异项**（D-13 裁决的 PIT 修复类等，按 F11 词表逐列登记）/ **拒绝项**
  （不在已批准清单 ⇒ 停止，§7.4-7）。比对/分类复用 P1 归因门同一实现
  （``replay_compare`` 数值口径 + ``panel_reconcile._compare_day/_merge_and_judge/
  _build_verdict``，词表唯一来源）；
- **cs_train 验证源腿**：适配层 cs_train_verify 模式输出 vs 现役 ``load_features_data``
  必须**逐值完全一致**（``check_exact=True``，含六标签与键集、行列序、dtype，G-code 数据前提）；
- **输入稳定硬门（R2-P3-T1-01/N1，§5-1/§7.3 唯一硬门）**：日历身份 → 选日 → 日历复核 →
  水位 → 枚举+采哈希 → 水位/日历再复核，绑定为**一致输入上下文**（采集期间变化 ⇒ 丢弃
  干净重读一次，再变 ⇒ 拒绝）；消费后复核——非 manifest 族文件字节门 + 水位前后比对 +
  **manifest 相关性 diff**（列定义/窗内分区登记/sealed_through/行域段；窗口外正常追加
  放行，reviewer2-N2）+ WAL 待处理态一致 + 覆盖日域一致性（预期日=readiness=两腿）。
  任一漂移/不一致 ⇒ verdict FAIL，事后身份无权追认先前读取。

窗口选取理由随 ``--rationale`` 落盘（不得以「无差异月份」规避已知差异列族）；报告落
``data/reports/v2_p3_t1_adapter_reconcile_<UTC 时间戳>.json``（可用 --out 覆盖）。

用法（只读消费生产数据 + 写报告；小窗预计 <10min，超授权口径前须用户授权，§7.4-9）：

    python scripts/v2_p3/t1_reconcile_adapter.py --start-date 20181126 --end-date 20181221 \
        --rationale "首折 OOS 起点窗，命中 D-12/D-04③/D-05 已知差异列族"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lazybull.ml.train_core.split import load_features_data  # noqa: E402
from src.lazybull.v2.store.column_groups import (  # noqa: E402
    LABEL_VALUE_COLUMNS,
    PANEL_GROUPS,
)
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

# 比对数值口径（atol=1e-6、NaN==NaN 一致）与 F11 词表判定的唯一来源（禁止复制）
from src.lazybull.v2.store.panel_reconcile import _compare_day  # noqa: E402
from src.lazybull.v2.store.panel_reconcile import (  # noqa: E402
    PANEL_FEATURE_COLUMNS,
    _build_verdict,
    _merge_and_judge,
)
from src.lazybull.v2.store.row_domain import LABEL_FILTERED_THROUGH  # noqa: E402
from src.lazybull.v2.store.store_identity import (  # noqa: E402
    collect_file_identities,
    file_identity,
    verify_file_identities,
    workspace_identity,
)
from src.lazybull.v2.train.data_adapter import (  # noqa: E402
    SOURCE_CS_TRAIN_VERIFY,
    AdapterConfig,
    load_training_frame,
    resolve_input_watermark,
    resolve_trading_dates,
)

_KEY_COLUMNS = ("trade_date", "ts_code")
_REPORT_DEFAULT_DIR = Path("data") / "reports"
#: panel 腿数值口径（P1 归因门复用）：容差内一致，非逐位精确相等（R2-P3-T1-S1）
_COMPARE_LEGEND = (
    "panel 腿数值口径 atol=1e-6（P1 归因门复用）：identical_columns = 容差内一致，"
    "非逐位精确相等；cs_train 验证源腿 check_exact=True 逐位一致"
)


# ---------------------------------------------------------------------------
# panel 腿：适配层 panel 模式 vs 现役 cs_train 读取（逐日比对 + 归因门判定）
# ---------------------------------------------------------------------------


def _load_cs_train_days(root: Path, days: Sequence[str]) -> dict[str, pd.DataFrame]:
    """逐日读 cs_train 分区（保留逐日列集，供列集/键集/值三层比对）。"""
    from src.lazybull.data.storage import Storage

    storage = Storage(root_path=str(root))
    frames: dict[str, pd.DataFrame] = {}
    for day in days:
        frame = storage.load_cs_train_day(day)
        if frame is not None and len(frame) > 0:
            frames[day] = frame
    return frames


def _compare_panel_leg(
    adapter_df: pd.DataFrame, reference_days: dict[str, pd.DataFrame], days: Sequence[str]
) -> dict[str, Any]:
    """逐日 ``_compare_day`` + ``_merge_and_judge``（归因门），输出三类结论。

    两侧重复键先行 fail-fast（reviewer1-P3-5）：``_compare_day`` 的 set 键域比较
    掩盖同日重复行，重复键会使值比较错位而不报错。
    """
    duplicate_key_days = []
    duplicated_days: set[str] = set()
    for day in days:
        ref = reference_days.get(day)
        ref_dup = ref is not None and ref.duplicated(list(_KEY_COLUMNS)).any()
        adapter_day = adapter_df[adapter_df["trade_date"] == day]
        adapter_dup = adapter_day.duplicated(list(_KEY_COLUMNS)).any()
        if ref_dup:
            duplicate_key_days.append({"date": day, "side": "cs_train"})
        if adapter_dup:
            duplicate_key_days.append({"date": day, "side": "adapter"})
        if ref_dup or adapter_dup:
            duplicated_days.add(day)
    day_results = []
    for day in days:
        if day in duplicated_days:
            continue  # 重键日已登记并判 FAIL；_compare_day 的 set 键域比较对重键会错位崩溃
        ref = reference_days.get(day)
        if ref is None:
            day_results.append({"date": day, "status": "missing_ref"})
            continue
        adapter_day = adapter_df[adapter_df["trade_date"] == day]
        day_results.append(_compare_day(day, ref, adapter_day))
    merged = _merge_and_judge(day_results, mode="attrib")
    columns = merged["columns"]
    diff_cols = {c: a for c, a in columns.items() if a["over_rows"] > 0}
    gate_verdict = _build_verdict(merged, diff_cols, "attrib")
    identical = [col for col in [*_panel_leg_expected_columns()] if col not in columns]
    approved = {
        col: _column_report(agg)
        for col, agg in sorted(columns.items())
        if not agg["boundary_violation"]
    }
    rejected = {
        col: _column_report(agg)
        for col, agg in sorted(columns.items())
        if agg["boundary_violation"]
    }
    return {
        "verdict_pass": bool(gate_verdict["pass"]) and not duplicate_key_days,
        "verdict_detail": gate_verdict,
        # 重键日被跳过比较（已判 FAIL），计数须扣除（reviewer1-R4，防「比较了 N 日」虚报）
        "trade_days_compared": len(days) - len(merged["missing_days"]) - len(duplicated_days),
        "missing_ref_days": merged["missing_days"],
        "key_diff_days": merged["row_diff_days"],
        "duplicate_key_days": duplicate_key_days,
        "cols_only_cs_train_days": merged["cols_only_ref_days"],
        "cols_only_adapter": merged["cols_only_panel"],
        "identical_columns": identical,
        "approved_differences": approved,
        "rejected_differences": rejected,
    }


def _panel_leg_expected_columns() -> list[str]:
    return [*PANEL_FEATURE_COLUMNS, *LABEL_VALUE_COLUMNS]


def _column_report(agg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attribution": agg["attribution"],
        "over_rows": agg["over_rows"],
        "over_share": agg["over_share"],
        "max_abs_diff": agg["max_abs_diff"],
        "diff_day_summary": agg["diff_day_summary"],
        "samples": agg["samples"],
    }


# ---------------------------------------------------------------------------
# cs_train 验证源腿：适配层 cs_train_verify vs 现役 load_features_data 逐值一致
# ---------------------------------------------------------------------------


def verify_cs_train_source_parity(
    root: Path, start_date: str, end_date: str, label_column: str
) -> dict[str, Any]:
    """G-code 数据前提：两侧输出逐值完全一致（含六标签与键集、行列序、dtype）。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    storage = Storage(root_path=str(root))
    loader = DataLoader(storage)
    config = AdapterConfig(
        start_date=start_date,
        end_date=end_date,
        label_column=label_column,
        source_mode=SOURCE_CS_TRAIN_VERIFY,
    )
    adapted = load_training_frame(
        config, storage=storage, loader=loader, acceptance_purpose="g_code_parity"
    )
    reference, reference_days = load_features_data(storage, loader, start_date, end_date)
    detail = ""
    try:
        pd.testing.assert_frame_equal(adapted.df, reference, check_exact=True)
    except AssertionError as exc:
        detail = str(exc)[:4000]
    passed = not detail
    return {
        "pass": passed,
        "detail": detail,
        "rows": len(reference),
        "trade_days": reference_days,
        "dates": sorted(str(day) for day in reference["trade_date"].unique()),
        "columns": len(reference.columns),
        "label_columns_present": [c for c in LABEL_VALUE_COLUMNS if c in reference.columns],
    }


# ---------------------------------------------------------------------------
# 消费集合身份（只读证据：实际读取文件的内容哈希清单）
# ---------------------------------------------------------------------------


def enumerate_window_input_files(
    root: Path, days: Sequence[str], start_date: str, end_date: str
) -> list[Path]:
    """枚举实际消费文件（消费前固定集合，R2-P3-T1-01）。

    - cs_train 日分区：按**已校验日域**（验证源腿为日历驱动逐日读取）；
    - panel 分区 / 三表标签分区：按**读取区间 [start,end]**（store 区间语义——
      panel 按族冷热同源解析（镜像 ``_load_group_frame``）、标签按区间 glob，
      覆盖「为过滤而读取的额外日文件」的身份边界，reviewer2-N1 余项）；
    - manifest 系列（读取校验 / 列族解析 / 成熟度判定的依据）+ 交易日历（影响窗口日枚举）
      + clean/daily 最新分区（``W_in`` 来源，影响成熟度判定）。
    """
    files: list[Path] = []
    for day in days:
        candidate = root / "features" / "cs_train" / f"{day}.parquet"
        if candidate.is_file():
            files.append(candidate)
    files.extend(_panel_partition_files(root, start_date, end_date))
    labels_root = root / "labels"
    label_dirs = sorted(labels_root.iterdir()) if labels_root.is_dir() else []
    for name_dir in label_dirs:
        if not name_dir.is_dir():
            continue
        for file in sorted(name_dir.glob("*.parquet")):
            if len(file.stem) == 8 and file.stem.isdigit() and start_date <= file.stem <= end_date:
                files.append(file)
    features_dir = root / "features"
    for name in ("manifest.json", "manifest.json.bak", "manifest_wal.jsonl"):
        candidate = features_dir / name
        if candidate.is_file():
            files.append(candidate)
    calendar = _calendar_path(root)
    files.append(calendar)
    daily_max = _daily_max_partition(root)
    if daily_max is not None:
        files.append(daily_max)
    return files


def _daily_max_partition(root: Path) -> Path | None:
    """clean/daily 最新分区文件（W_in 的物理来源；无分区返回 None）。"""
    daily_dir = root / "clean" / "daily"
    if not daily_dir.is_dir():
        return None
    candidates = []
    for path in daily_dir.glob("*.parquet"):
        digits = path.stem.replace("-", "")
        if len(digits) == 8 and digits.isdigit():
            candidates.append((digits, path))
    return max(candidates)[1] if candidates else None


def _panel_partition_files(root: Path, start_date: str, end_date: str) -> list[Path]:
    """按族解析 panel 消费文件（与 store 实际读取同源，R2-P3-T1-01 清单缺口修复）。"""
    # 月份区间与自然月对齐（store._months_in_range 同口径；reviewer1-R3-③ 顺手对齐项）
    months = PanelDataStore._months_in_range(start_date, end_date)
    files: list[Path] = []
    for group in PANEL_GROUPS:
        archived = {
            month
            for month in months
            if (root / "features" / "panel_archive" / month / f"{group}.parquet").is_file()
        }
        for month in sorted(archived):
            files.append(root / "features" / "panel_archive" / month / f"{group}.parquet")
        panel_dir = root / "features" / "panel"
        if not panel_dir.is_dir():
            continue
        for day_dir in sorted(panel_dir.iterdir()):
            if not day_dir.is_dir() or not _is_yyyymmdd(day_dir.name):
                continue
            if not (start_date <= day_dir.name <= end_date):
                continue
            if f"{day_dir.name[:4]}-{day_dir.name[4:6]}" in archived:
                continue  # 该族当月已封存，冷区为准（与 _load_group_frame 同口径）
            hot_file = day_dir / f"{group}.parquet"
            if hot_file.is_file():
                files.append(hot_file)
    return files


def _is_yyyymmdd(value: str) -> bool:
    return len(value) == 8 and value.isdigit()


# ---------------------------------------------------------------------------
# 编排与报告
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _InputContext:
    """绑定一致的输入上下文（reviewer2-N1）：选日 + 水位 + 消费集合身份 + manifest 快照 + WAL 态。

    ``identities`` = 非 manifest 族消费文件（字节门）；``manifest_audit`` = manifest 族身份
    （仅审计展示，判定走 ``_manifest_relevance_diff`` 语义口径，N2：窗口外正常追加放行）。
    """

    days: tuple[str, ...]
    watermark: str
    identities: tuple[dict[str, Any], ...]
    manifest_audit: tuple[dict[str, Any], ...]
    manifest_snapshot: Mapping[str, Any]
    wal_pending: bool


_MANIFEST_FAMILY = frozenset({"manifest.json", "manifest.json.bak", "manifest_wal.jsonl"})


def _calendar_path(root: Path) -> Path:
    calendar = root / "clean" / "trade_cal.parquet"
    return calendar if calendar.is_file() else root / "raw" / "trade_cal.parquet"


def _wal_pending(root: Path) -> bool:
    wal = root / "features" / "manifest_wal.jsonl"
    return wal.is_file() and wal.stat().st_size > 0


def _manifest_snapshot(root: Path) -> Mapping[str, Any]:
    """经全新 store 实例读取 manifest 深拷贝快照（避免内存缓存掩盖运行期磁盘变更）。"""
    return PanelDataStore(root).manifest.snapshot()


def _acquire_input_context(root: Path, start_date: str, end_date: str) -> _InputContext:
    """首次读取全程受核验覆盖（N1）：日历/水位在采集期间变化 ⇒ 丢弃干净重读一次；再变 ⇒ 拒绝。

    顺序写死：日历身份 → 选日 → 日历复核 → 水位 → 枚举+采哈希 → 水位/日历再复核，
    三项一致才固定为基线（不保留旧清单再换基线）。
    """
    for attempt in (1, 2):
        calendar = _calendar_path(root)
        cal_before = file_identity(calendar)
        days = [d for d in resolve_trading_dates(root) if start_date <= d <= end_date]
        if not days:
            raise ValueError(f"窗口 [{start_date}, {end_date}] 内无交易日")
        cal_mid = file_identity(calendar)
        watermark_before = resolve_input_watermark(root)
        files = enumerate_window_input_files(root, days, start_date, end_date)
        identities = collect_file_identities(files, base_dir=root)
        watermark_after = resolve_input_watermark(root)
        cal_after = file_identity(calendar)
        if cal_before == cal_mid == cal_after and watermark_before == watermark_after:
            audit, locked = _split_manifest_family(identities)
            return _InputContext(
                days=tuple(days),
                watermark=watermark_before,
                identities=tuple(locked),
                manifest_audit=tuple(audit),
                manifest_snapshot=_manifest_snapshot(root),
                wal_pending=_wal_pending(root),
            )
        logger.warning(f"输入上下文第 {attempt} 次采集期间发生变化（日历/水位），丢弃并干净重读")
    raise RuntimeError(
        "输入上下文连续两次采集不一致（交易日历或输入水位在读取期间变化），拒绝继续"
        "（N1：首次读取必须受核验覆盖，不保留旧清单再换基线）"
    )


def _split_manifest_family(
    identities: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    locked: list[dict[str, Any]] = []
    for record in identities:
        name = str(record["path"]).rsplit("/", 1)[-1]
        (audit if name in _MANIFEST_FAMILY else locked).append(dict(record))
    return audit, locked


def _manifest_relevance_diff(
    before: Mapping[str, Any], after: Mapping[str, Any], days: Sequence[str]
) -> list[str]:
    """查询相关 manifest 变更清单（reviewer2-N2）：窗口外正常追加不判漂移。

    判定绑定到本次实际依赖的内容——列定义（columns）/ 查询窗内 panel 与标签分区登记 /
    sealed_through / 行域段 / manifest_version；窗口外新增分区（正常日更追加）放行；
    repairs / 刷新审计 / updated_at 不参与判定（分区内容与登记变更已被前述项覆盖）。
    """
    issues: list[str] = []
    if before.get("columns") != after.get("columns"):
        issues.append("manifest.columns 列定义变更")
    day_set = set(days)
    month_set = {f"{day[:4]}-{day[4:6]}" for day in days}
    issues.extend(_partition_registration_diff(before, after, day_set, month_set))
    issues.extend(_label_registration_diff(before, after, day_set))
    for key in ("row_domain_segments", "manifest_version"):
        if before.get(key) != after.get(key):
            issues.append(f"manifest.{key} 变更")
    return issues


def _partition_registration_diff(
    before: Mapping[str, Any], after: Mapping[str, Any], day_set: set[str], month_set: set[str]
) -> list[str]:
    before_parts = before.get("partitions", {})
    after_parts = after.get("partitions", {})
    issues = []
    for rel in sorted(set(before_parts) | set(after_parts)):
        parts = str(rel).split("/")
        in_scope = (len(parts) == 3 and parts[0] == "panel" and parts[1] in day_set) or (
            len(parts) == 3 and parts[0] == "panel_archive" and parts[1] in month_set
        )
        if in_scope and before_parts.get(rel) != after_parts.get(rel):
            issues.append(f"panel 分区登记变更: {rel}")
    return issues


def _label_registration_diff(
    before: Mapping[str, Any], after: Mapping[str, Any], day_set: set[str]
) -> list[str]:
    issues = []
    names = set(before.get("labels", {})) | set(after.get("labels", {}))
    for name in sorted(names):
        before_lab = before.get("labels", {}).get(name, {})
        after_lab = after.get("labels", {}).get(name, {})
        if before_lab.get("sealed_through") != after_lab.get("sealed_through"):
            issues.append(f"labels.{name}.sealed_through 变更")
        before_pp = before_lab.get("partitions", {})
        after_pp = after_lab.get("partitions", {})
        for filename in sorted(set(before_pp) | set(after_pp)):
            if filename.removesuffix(".parquet") not in day_set:
                continue
            if before_pp.get(filename) != after_pp.get(filename):
                issues.append(f"标签分区登记变更: {name}/{filename}")
    return issues


def _verify_input_stability(
    root: Path, context: _InputContext, days: Sequence[str]
) -> dict[str, Any]:
    """消费后复核（R2-P3-T1-01/N1/N2）：字节门 + 水位 + manifest 相关性 diff + WAL 态。"""
    drift_files = sorted(verify_file_identities(context.identities, base_dir=root))
    watermark_after = resolve_input_watermark(root)
    try:
        manifest_issues = _manifest_relevance_diff(
            context.manifest_snapshot, _manifest_snapshot(root), days
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        manifest_issues = [f"manifest 复核解析失败（运行期损坏）: {exc}"]
    wal_after = _wal_pending(root)
    watermark_stable = watermark_after == context.watermark
    passed = (
        not drift_files
        and not manifest_issues
        and wal_after == context.wal_pending
        and watermark_stable
    )
    return {
        "checked": True,
        "pass": passed,
        "drift_files": drift_files,
        "manifest_relevance_issues": manifest_issues,
        "wal_pending_before": context.wal_pending,
        "wal_pending_after": wal_after,
        "watermark_before": context.watermark,
        "watermark_after": watermark_after,
        "watermark_stable": watermark_stable,
        "locked_file_count": len(context.identities),
        "manifest_audit_count": len(context.manifest_audit),
        "manifest_audit_note": "审计 = 运行前快照（reviewer1-N-4）；通过判定走 manifest 相关性 diff",
    }


def _coverage_consistency(
    days: Sequence[str],
    adapter_frame: Any,
    panel_leg: Mapping[str, Any],
    parity_leg: Mapping[str, Any],
) -> dict[str, Any]:
    """预期 / readiness / 实际适配帧 / 验证源帧 / 比较日域按**日期集合**一致性核对（N1 余项）。

    计数一致不等于日域一致（同数不同日必须拒绝）；panel 侧由 ``missing_ref`` / 重键
    自身的 FAIL 语义兜底，此处核对三侧日期集合与面板比较日域账。
    """
    expected = {str(day) for day in days}
    readiness_dates = {str(entry["trade_date"]) for entry in adapter_frame.day_readiness}
    adapter_dates = {str(day) for day in adapter_frame.df["trade_date"].unique()}
    parity_dates = {str(day) for day in parity_leg.get("dates", [])}
    compared_dates = (
        expected
        - set(panel_leg["missing_ref_days"])
        - {str(entry["date"]) for entry in panel_leg["duplicate_key_days"]}
    )
    consistent = (
        readiness_dates == expected
        and adapter_dates == expected
        and parity_dates == expected
        and compared_dates == expected
    )
    return {
        "consistent": consistent,
        "expected_days": len(expected),
        "readiness_days": len(readiness_dates),
        "adapter_days": len(adapter_dates),
        "parity_trade_days": len(parity_dates) if parity_dates else None,
        "panel_compared_days": len(compared_dates),
        "missing_ref_days": len(panel_leg["missing_ref_days"]),
        "duplicate_key_days": len(panel_leg["duplicate_key_days"]),
        "excluded_days": list(adapter_frame.excluded_days),
        "date_mismatches": {
            "readiness_minus_expected": sorted(readiness_dates - expected),
            "expected_minus_readiness": sorted(expected - readiness_dates),
            "adapter_minus_expected": sorted(adapter_dates - expected),
            "expected_minus_adapter": sorted(expected - adapter_dates),
            "parity_minus_expected": sorted(parity_dates - expected),
            "expected_minus_parity": sorted(expected - parity_dates),
        },
    }


def run_window_reconcile(
    data_root: Path | str,
    start_date: str,
    end_date: str,
    *,
    label_column: str = "neu_y_ret_20",
    rationale: str = "",
    _post_panel_leg_hook: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """对账主入口：panel 腿三类结论 + cs_train 验证源腿逐值一致 + 输入稳定硬门。

    顺序写死（R2-P3-T1-01/N1）：一致输入上下文（日历+选日+水位+身份绑定）→ 两腿消费 →
    消费后复核（字节门 + 水位 + manifest 相关性 diff + WAL 态）→ 覆盖一致性核对；
    任一漂移/不一致 ⇒ verdict FAIL。``_post_panel_leg_hook`` 为测试注入点，生产不用。
    """
    root = Path(data_root)
    started = datetime.now(timezone.utc)
    context = _acquire_input_context(root, start_date, end_date)
    days = list(context.days)
    store = PanelDataStore(root)
    adapter_frame = load_training_frame(
        AdapterConfig(
            start_date=start_date,
            end_date=end_date,
            feature_columns=tuple(PANEL_FEATURE_COLUMNS),
            label_column=label_column,
        ),
        store=store,
    )
    reference_days = _load_cs_train_days(root, days)
    panel_leg = _compare_panel_leg(adapter_frame.df, reference_days, days)
    if _post_panel_leg_hook is not None:
        _post_panel_leg_hook(root)
    try:
        parity_leg = verify_cs_train_source_parity(root, start_date, end_date, label_column)
    except (RuntimeError, ValueError) as exc:
        # 验证源腿自身无法完成（如分区覆盖硬失败）⇒ 记 FAIL 而非崩溃，保留 panel 腿证据
        parity_leg = {"pass": False, "detail": f"验证源腿执行失败: {exc}"[:4000]}
    input_stability = _verify_input_stability(root, context, days)
    coverage = _coverage_consistency(days, adapter_frame, panel_leg, parity_leg)
    verdict_pass = (
        bool(panel_leg["verdict_pass"])
        and bool(parity_leg["pass"])
        and input_stability["pass"]
        and coverage["consistent"]
    )
    report = {
        "tool": "t1_reconcile_adapter",
        "generated_at": started.isoformat(),
        "data_root": str(root),
        "window": {
            "start": start_date,
            "end": end_date,
            "trade_days": len(days),
            "frozen_through": LABEL_FILTERED_THROUGH,
            "rationale": rationale,
        },
        "label_column": label_column,
        "comparison_legend": _COMPARE_LEGEND,
        "panel_vs_cs_train": panel_leg,
        "cs_train_verify_parity": parity_leg,
        "adapter_evidence": {
            "watermarks": _watermarks_report(adapter_frame),
            "manifest_fingerprint": adapter_frame.manifest_fingerprint,
            "day_readiness": list(adapter_frame.day_readiness),
            "adapter_dates": sorted(str(day) for day in adapter_frame.df["trade_date"].unique()),
            "excluded_days": list(adapter_frame.excluded_days),
        },
        "input_stability": input_stability,
        "coverage_consistency": coverage,
        "input_identities": list(context.identities) + list(context.manifest_audit),
        "workspace_identity": workspace_identity(_REPO_ROOT),
        "verdict": {
            "pass": verdict_pass,
            "rule": (
                "§4.2：列集+键集+全部特征值+六标签值；拒绝项（未入已批准清单的差异 / "
                "键集不一致或重复 / 结构越界 / cs_train 验证源腿非逐位一致 / "
                "消费前后输入身份漂移（查询相关口径）/ 覆盖日域不一致）任一出现 ⇒ 停止"
                "（§7.4-7 + §5-1/§7.3 唯一硬门；窗口外正常追加不判漂移）"
            ),
        },
    }
    logger.info(
        f"对账完成: verdict={'PASS' if verdict_pass else 'FAIL'}；"
        f"容差内一致 {len(panel_leg['identical_columns'])} 列 / "
        f"已批准差异 {len(panel_leg['approved_differences'])} 列 / "
        f"拒绝 {len(panel_leg['rejected_differences'])} 列；"
        f"cs_train 验证源腿 {'逐位一致' if parity_leg['pass'] else '不一致'}；"
        f"输入稳定 {'通过' if input_stability['pass'] else '漂移'}；"
        f"覆盖一致 {'是' if coverage['consistent'] else '否'}"
    )
    return report


def _watermarks_report(adapter_frame: Any) -> dict[str, Any]:
    marks = adapter_frame.watermarks
    return {
        "input_watermark": marks.input_watermark,
        "label_sealed_through": dict(marks.label_sealed_through),
        "label_latest_partition": dict(marks.label_latest_partition),
        "panel_latest_partition": marks.panel_latest_partition,
        "cs_train_latest_partition": marks.cs_train_latest_partition,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="窗口起点 YYYYMMDD（≤20260702）")
    parser.add_argument("--end-date", required=True, help="窗口终点 YYYYMMDD（≤20260702）")
    parser.add_argument("--data-root", default="data", help="生产数据根（只读消费）")
    parser.add_argument("--label-column", default="neu_y_ret_20")
    parser.add_argument(
        "--rationale",
        required=True,
        help="窗口选取理由（必填，落盘报告 window.rationale；不得以无差异月份规避已知差异列族）",
    )
    parser.add_argument("--out", default=None, help="报告输出路径（默认 data/reports/ 下按时间戳）")
    args = parser.parse_args(argv)

    report = run_window_reconcile(
        Path(args.data_root),
        args.start_date,
        args.end_date,
        label_column=args.label_column,
        rationale=args.rationale,
    )
    out_path = Path(args.out) if args.out else None
    if out_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = _REPORT_DEFAULT_DIR / f"v2_p3_t1_adapter_reconcile_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"对账报告落盘: {out_path}")
    print(f"verdict: {'PASS' if report['verdict']['pass'] else 'FAIL'} -> {out_path}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
