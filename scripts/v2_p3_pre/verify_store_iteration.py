# -*- coding: utf-8 -*-
"""P3 pre T4：真实 scratch 验收（只读生产输入，只写 scratch 与正式报告）。

执行顺序：报告路径隔离校验（已存在/命中生产资产即拒绝，不落任何报告）→ 授权耗时预检 →
机械预检（逐样本 T/B/E20、目录水位、预期状态、sealed 候选数，先打印后计时/写入）→
P1 附录 B 十项指纹预检 → 真实日更 forming、共享 readiness 与 Q6 键集锚 → 历史影子
forming→sealed 刷新（恰 N 日转换、raw+neu 逐值对账、总耗时硬门、build_features_data=0、
二次幂等、两场景故障注入）→ ti 块头对照 → 冻结段指纹（全新只读 manifest 视图）与
输入/工作树/环境身份核对。样本或附录 B 预检任一不符，在计时与写入前报错；
下层异常同样落正式 failed 报告；历史影子证据不等于真实日更段已 sealed。

用法：
    python -X utf8 scripts/v2_p3_pre/verify_store_iteration.py \
        --dates 20260703,20260731 --historical-shadow-count 20 \
        --scratch-root temp/p3_pre_store_acceptance_<RUN_ID> \
        --out data/reports/v2_p3_pre_store_acceptance_<RUN_ID>.json
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v2_p3_pre"))
sys.path.insert(0, str(ROOT / "scripts" / "v2_p1"))

from migrate_store_manifest import _reject_pending_wal, _write_json_atomic  # noqa: E402
from update_daily_store import (  # noqa: E402
    _BUILD_SECONDS_PER_DAY,
    _REFRESH_SECONDS_PER_DAY,
    _clean_candidate_paths,
    _read_clean_partition,
    _refresh_kwargs,
)
from verify_iteration_parts import (  # noqa: E402
    AcceptanceError,
    _sum_cross_seam_impacts,
    appendix_b_report,
    assert_fixture_panel_fidelity,
    assert_forming_fixture,
    assert_run_identities,
    capture_frozen_before,
    copy_shadow_day_to_scratch,
    cross_seam_impact_for_day,
    diff_directory_listing,
    directory_listing_evidence,
    fault_injection_check,
    frozen_segment_after_check,
    pipeline_input_identities,
    production_source_identities,
    recording_parquet_reads,
    supplement_cross_seam,
    validate_output_path,
)

from src.lazybull.data.loader import DataLoader  # noqa: E402
from src.lazybull.data.storage import Storage  # noqa: E402
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import (  # noqa: E402
    LABEL_TABLES,
    PANEL_GROUPS,
    columns_of_group,
)
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.label_refresh import (  # noqa: E402
    plan_label_refreshes,
    refresh_labels_for_day,
    validate_label_day_readiness,
)
from src.lazybull.v2.store.labels_builder import maturity_status_for  # noqa: E402
from src.lazybull.v2.store.panel_builder import V2PanelBuilder, bootstrap_manifest  # noqa: E402
from src.lazybull.v2.store.row_domain import (  # noqa: E402
    LABEL_FILTERED_THROUGH,
    STRUCTURAL_FROM,
    row_domain_for_date,
)
from src.lazybull.v2.store.store_identity import (  # noqa: E402
    canonical_json_sha256,
    collect_file_identities,
    producer_identity,
    workspace_identity,
)

_ALL_GROUPS = tuple(PANEL_GROUPS)
_LABEL_NAMES = tuple(LABEL_TABLES)
_KEY_COLUMNS = ["trade_date", "ts_code"]
#: ti_* 6 列经物化归入 announcement 族（columns_of_group = 10 基础列 + 34 物化列）
_TI_COLUMNS = tuple(
    column for column in columns_of_group("announcement") if column.startswith("ti_")
)
_REPORT_PREFIX = "v2_p3_pre_store_acceptance_"

#: 授权耗时静态估算（S6 处置；实测以报告为准；哈希含读取层记录，预算放宽；
#: 构建/参照按 T4 实测 452~462s/日 + 余量取 500s，2026-10-19 用户裁决口径）
_REFERENCE_SECONDS_PER_DAY = 500
_TI_BLOCK_HEAD_SECONDS = 240
_IDENTITY_HASHING_SECONDS = 420


class ReportPathRejected(RuntimeError):
    """报告路径已存在或命中生产资产/scratch；不落任何报告直接拒绝。"""


@dataclass
class _Context:
    """run 级只读上下文；生产 store 仅读，不写。"""

    store: Any
    loader: Any
    builder: Any
    calendar: list[str]
    watermark: str
    producer: Mapping[str, Any]
    workspace: Mapping[str, Any]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", help="真实日更验收日期，逗号分隔 YYYYMMDD")
    parser.add_argument("--historical-shadow-count", type=int, default=20, help="历史影子日数")
    parser.add_argument("--scratch-root", help="scratch 根目录（须不存在或为空）")
    parser.add_argument("--out", required=True, help="正式报告输出路径（已存在则拒绝）")
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="v2 store 生产数据根")
    parser.add_argument(
        "--shadow-budget-seconds",
        type=float,
        default=60.0,
        help="历史影子刷新总耗时硬门（默认 60 秒）",
    )
    parser.add_argument(
        "--authorized-minutes", type=float, default=60.0, help="本次预授权机器分钟数（默认 60）"
    )
    parser.add_argument(
        "--supplement-cross-seam",
        help="跨缝补充报告模式：既有 passed 验收报告路径（与 --dates 互斥）",
    )
    args = parser.parse_args(argv)
    if args.supplement_cross_seam:
        if args.dates or args.scratch_root:
            parser.error("--supplement-cross-seam 与 --dates/--scratch-root 互斥")
        return args
    if not args.dates:
        parser.error("--dates 不能为空（或使用 --supplement-cross-seam）")
    if not args.scratch_root:
        parser.error("--scratch-root 不能为空（或使用 --supplement-cross-seam）")
    dates = [item.strip() for item in str(args.dates).split(",") if item.strip()]
    for date in dates:
        if len(date) != 8 or not date.isdigit():
            parser.error(f"--dates 含非法日期: {date!r}")
        if date < STRUCTURAL_FROM:
            parser.error(f"真实日更验收日期不得早于 {STRUCTURAL_FROM}: {date}")
    if len(set(dates)) != len(dates):
        parser.error("--dates 含重复日期")
    if args.historical_shadow_count < 1:
        parser.error("--historical-shadow-count 必须为正数")
    if args.shadow_budget_seconds <= 0:
        parser.error("--shadow-budget-seconds 必须为正数")
    if args.authorized_minutes <= 0:
        parser.error("--authorized-minutes 必须为正数")
    args.dates = sorted(dates)
    return args


# ---------- 机械预检（计时/写入前） ----------


def _endpoints_of(calendar: Sequence[str], trade_date: str) -> tuple[str | None, str | None]:
    """信号日 T 的买入端点 B=idx(T)+1 与最大端点 E20=idx(T)+21；越界返回 None。"""
    dates = list(calendar)
    index = dates.index(str(trade_date))
    buy = dates[index + 1] if index + 1 < len(dates) else None
    endpoint = dates[index + 21] if index + 21 < len(dates) else None
    return buy, endpoint


def _sealed_candidates(calendar: Sequence[str], watermark: str) -> list[str]:
    """结构段内 E20 已达实际水位的日期；当前数据态必须为空。"""
    return [
        date
        for date in calendar
        if STRUCTURAL_FROM <= date <= watermark
        and maturity_status_for(date, calendar, data_end=watermark) == "sealed"
    ]


def _select_shadow_days(calendar: Sequence[str], watermark: str, count: int) -> list[str]:
    """最近 count 个 E20 ≤ 实际水位的冻结段信号日（升序）。"""
    selected: list[str] = []
    for date in reversed([item for item in calendar if item <= LABEL_FILTERED_THROUGH]):
        _, endpoint = _endpoints_of(calendar, date)
        if endpoint is not None and endpoint <= watermark:
            selected.append(date)
        if len(selected) == count:
            break
    return sorted(selected)


def _assert_daily_entries(entries: Sequence[Mapping[str, Any]]) -> None:
    for entry in entries:
        if entry["row_domain"] != "structural":
            raise AcceptanceError(f"{entry['trade_date']} 行域应为 structural: {entry}")
        if entry["expected_status"] != "forming":
            raise AcceptanceError(
                f"{entry['trade_date']} 状态与预检不符：预期 forming，实得 "
                f"{entry['expected_status']}（E20={entry['endpoint_date']}，目录水位 "
                f"{entry['input_watermark']}）；禁止移动行域边界或改写样本"
            )


def _assert_production_segment_absent(store: Any, dates: Sequence[str]) -> None:
    snapshot = store.manifest.snapshot()
    for date in dates:
        present = [rel for rel in snapshot["partitions"] if rel.startswith(f"panel/{date}/")]
        present += [
            f"labels/{name}/{date}.parquet"
            for name in _LABEL_NAMES
            if f"{date}.parquet" in snapshot["labels"].get(name, {}).get("partitions", {})
        ]
        if present:
            raise AcceptanceError(f"生产已存在 {date} 日更段资产，禁止重复初建: {present[:3]}")


def _assert_shadow_sample_integrity(store: Any, days: Sequence[str]) -> None:
    """影子样本必须为生产已登记、行域 label_filtered 且盘上 sealed 的完整日。"""
    snapshot = store.manifest.snapshot()
    for date in days:
        if row_domain_for_date(date) != "label_filtered":
            raise AcceptanceError(f"影子日 {date} 不在冻结段")
        core = snapshot["partitions"].get(f"panel/{date}/core.parquet")
        if core is None or core.get("row_domain") != "label_filtered":
            raise AcceptanceError(f"影子日 {date} 缺 panel core 登记或行域异常")
        for name in _LABEL_NAMES:
            meta = snapshot["labels"].get(name, {}).get("partitions", {}).get(f"{date}.parquet")
            if meta is None or meta.get("row_domain") != "label_filtered":
                raise AcceptanceError(f"影子日 {date} 缺标签登记或行域异常: {name}")
            sealed_through = snapshot["labels"][name].get("sealed_through") or ""
            if sealed_through < date:
                raise AcceptanceError(f"影子日 {date} 不在生产封存水位内: {name}")
            path = store.labels_dir / name / f"{date}.parquet"
            frame = pd.read_parquet(path, columns=["maturity_status"])
            statuses = set(frame["maturity_status"].astype(str))
            if statuses != {"sealed"}:
                raise AcceptanceError(f"影子日 {date} 盘上状态不是 sealed: {name}: {statuses}")


def _mechanical_precheck(ctx: _Context, dates: Sequence[str], shadow_count: int) -> dict[str, Any]:
    """只读机械预检：任何不符在计时与写入前抛出。"""
    entries: list[dict[str, Any]] = []
    for date in dates:
        if date not in ctx.calendar:
            raise AcceptanceError(f"{date} 不在实际交易日历中")
        if date > ctx.watermark:
            raise AcceptanceError(f"样本日期 {date} 晚于实际目录水位 {ctx.watermark}，无法构建")
        buy, endpoint = _endpoints_of(ctx.calendar, date)
        entries.append(
            {
                "trade_date": date,
                "buy_date": buy,
                "endpoint_date": endpoint,
                "input_watermark": ctx.watermark,
                "expected_status": maturity_status_for(date, ctx.calendar, data_end=ctx.watermark),
                "row_domain": row_domain_for_date(date),
            }
        )
    _assert_daily_entries(entries)
    candidates = _sealed_candidates(ctx.calendar, ctx.watermark)
    if candidates:
        raise AcceptanceError(f"当前真实日更 sealed 候选必须为 0，实得: {candidates}")
    _assert_production_segment_absent(ctx.store, dates)
    shadow_days = _select_shadow_days(ctx.calendar, ctx.watermark, shadow_count)
    if len(shadow_days) != shadow_count:
        raise AcceptanceError(
            f"历史影子不足 {shadow_count} 日（实得 {len(shadow_days)}），禁止以零候选 no-op 充数"
        )
    _assert_shadow_sample_integrity(ctx.store, shadow_days)
    return {
        "daily_entries": entries,
        "sealed_candidates": candidates,
        "shadow_days": shadow_days,
        "shadow_count": len(shadow_days),
        "shadow_range": f"{shadow_days[0]}~{shadow_days[-1]}",
        "dir_watermark": ctx.watermark,
    }


def _print_precheck(precheck: Mapping[str, Any]) -> None:
    """机械打印样本锚点；先于任何计时/写入。"""
    for entry in precheck["daily_entries"]:
        print(
            f"PRECHECK date={entry['trade_date']} B={entry['buy_date']} "
            f"E20={entry['endpoint_date']} dir_watermark={entry['input_watermark']} "
            f"expected={entry['expected_status']} domain={entry['row_domain']}"
        )
    print(f"PRECHECK sealed_candidates={len(precheck['sealed_candidates'])}")
    print(
        f"PRECHECK shadow_count={precheck['shadow_count']} "
        f"shadow_range={precheck['shadow_range']} dir_watermark={precheck['dir_watermark']}"
    )


# ---------- clean 输入加载（frames 供影子刷新；身份覆盖刷新真实读取） ----------


def _load_shadow_clean_frames(
    ctx: _Context, shadow_days: Sequence[str]
) -> tuple[dict[str, pd.DataFrame], str, list[dict[str, Any]]]:
    """加载影子首个 T+1 至尾部的 clean/daily 帧；尾部读取失败不得回退更早水位。"""
    first_input = ctx.calendar[ctx.calendar.index(shadow_days[0]) + 1]
    daily_dir = ctx.loader.storage.clean_path / "daily"
    candidates = _clean_candidate_paths(daily_dir, first_input)
    input_root = Path(ctx.loader.storage.root_path)
    identities = [collect_file_identities([path], base_dir=input_root)[0] for _, path in candidates]
    frames: dict[str, pd.DataFrame] = {}
    for partition_date, path in candidates:
        frames[partition_date] = _read_clean_partition(path, partition_date)
    watermark = max(frames)
    if watermark != ctx.watermark:
        raise AcceptanceError(f"实际加载水位 {watermark} 与目录水位 {ctx.watermark} 不一致")
    calendar_path = Path(ctx.loader.storage.clean_path) / "trade_cal.parquet"
    if not calendar_path.is_file():
        raise AcceptanceError(f"交易日历输入缺失: {calendar_path}")
    identities = sorted(
        [*identities, *collect_file_identities([calendar_path], base_dir=input_root)],
        key=lambda record: record["path"],
    )
    return frames, watermark, identities


# ---------- 历史影子 scratch ----------


@contextmanager
def _counting_build_calls():
    """给 panel_builder.build_features_data 挂计数器；专用刷新路径必须保持 0 次。"""
    import src.lazybull.v2.store.panel_builder as panel_builder_module

    calls = {"count": 0}
    original = panel_builder_module.build_features_data

    def _counting(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        return original(*args, **kwargs)

    panel_builder_module.build_features_data = _counting
    try:
        yield calls
    finally:
        panel_builder_module.build_features_data = original


def _refresh_shadow_days(
    scratch_store: Any, days: Sequence[str], refresh_kwargs: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], float]:
    """升序直调同一日级原语；返回逐日结果与实际总耗时。"""
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    for date in days:
        result = refresh_labels_for_day(scratch_store, date, **refresh_kwargs)
        results.append(result.to_dict())
    return results, time.monotonic() - started


def _assert_shadow_transitions(results: Sequence[Mapping[str, Any]], expected: int) -> None:
    if len(results) != expected:
        raise AcceptanceError(f"影子转换日数 {len(results)} ≠ 预期 {expected}")
    for result in results:
        if result["status"] != "refreshed" or result["actual_status"] != "sealed":
            raise AcceptanceError(f"影子日 {result['trade_date']} 未实际转换: {result['status']}")
        if result["row_domain"] != "label_filtered":
            raise AcceptanceError(f"影子日 {result['trade_date']} 行域异常: {result['row_domain']}")


def _reconcile_shadow_day(prod_store: Any, scratch_store: Any, date: str) -> list[str]:
    """刷新后三表与 P1 原 sealed 逐值对账（先列集断言，含 schema 与成熟度）。"""
    mismatches: list[str] = []
    for name in _LABEL_NAMES:
        expected = pd.read_parquet(prod_store.labels_dir / name / f"{date}.parquet")
        actual = pd.read_parquet(scratch_store.labels_dir / name / f"{date}.parquet")
        if set(actual.columns) != set(expected.columns):
            mismatches.append(
                f"{name}/{date}: 列集不一致 prod={sorted(expected.columns)} "
                f"scratch={sorted(actual.columns)}"
            )
            continue
        for frame in (expected, actual):
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame["ts_code"] = frame["ts_code"].astype(str)
        expected = expected.sort_values(_KEY_COLUMNS).reset_index(drop=True)
        actual = actual[expected.columns].sort_values(_KEY_COLUMNS).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        except AssertionError as exc:
            mismatches.append(f"{name}/{date}: {str(exc)[:400]}")
    return mismatches


def _label_file_bytes(scratch_store: Any, days: Sequence[str]) -> dict[str, bytes]:
    return {
        f"{name}/{date}": (scratch_store.labels_dir / name / f"{date}.parquet").read_bytes()
        for date in days
        for name in _LABEL_NAMES
    }


def _noop_rerun_check(
    scratch_store: Any, days: Sequence[str], refresh_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """二次重跑必须全部 no_op，标签字节、manifest 指纹与审计数量均不变。"""
    before_bytes = _label_file_bytes(scratch_store, days)
    before_fingerprint = scratch_store.manifest.content_fingerprint()
    before_audits = (
        len(scratch_store.manifest.label_refresh_intents()),
        len(scratch_store.manifest.label_refreshes()),
    )
    statuses = [
        refresh_labels_for_day(scratch_store, date, **refresh_kwargs).status for date in days
    ]
    if any(status != "no_op" for status in statuses):
        raise AcceptanceError(f"二次重跑存在非 no_op: {sorted(set(statuses))}")
    after_audits = (
        len(scratch_store.manifest.label_refresh_intents()),
        len(scratch_store.manifest.label_refreshes()),
    )
    if after_audits != before_audits:
        raise AcceptanceError(f"二次重跑新增审计: {before_audits} → {after_audits}")
    if scratch_store.manifest.content_fingerprint() != before_fingerprint:
        raise AcceptanceError("二次重跑 manifest 内容指纹变化")
    if _label_file_bytes(scratch_store, days) != before_bytes:
        raise AcceptanceError("二次重跑标签字节变化")
    return {"rerun_days": len(days), "all_no_op": True, "audit_counts": list(after_audits)}


def _per_day_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    """逐日证据（S1 处置：含水位/摘要/输出指纹/行数/键集/孤儿意图/跨缝计数/耗时）。"""
    keys = (
        "trade_date",
        "buy_date",
        "endpoint_date",
        "input_watermark",
        "row_domain",
        "status",
        "actual_status",
        "event_key",
        "input_digest",
        "rows",
        "keyset_valid",
        "output_fingerprints",
        "orphan_intents",
        "cross_seam_impact",
        "elapsed_seconds",
    )
    return {key: result[key] for key in keys}


def _aggregate_cross_seam(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """跨缝 y5/y10 母域受影响行数合计与分行业计数（§7.2 第 6 条机械证据）。"""
    total: dict[str, Any] = {}
    for shorter in ("y_ret_5", "y_ret_10"):
        by_industry: dict[str, int] = {}
        rows = 0
        for result in results:
            impact = result["cross_seam_impact"][shorter]
            rows += int(impact["affected_rows"])
            for industry, count in impact["by_industry"].items():
                by_industry[industry] = by_industry.get(industry, 0) + int(count)
        total[shorter] = {"affected_rows": rows, "by_industry": by_industry}
    return total


def _historical_shadow_phase(
    ctx: _Context,
    scratch_root: Path,
    shadow_days: Sequence[str],
    refresh_kwargs: Mapping[str, Any],
    budget_seconds: float,
    fragment: dict[str, Any],
) -> None:
    """历史影子主阶段（渐进填充 fragment；失败时已完成部分仍进正式报告）。"""
    fragment.update(
        {
            "days": list(shadow_days),
            "budget_seconds": budget_seconds,
            "evidence_note": "真实冻结数据的历史影子证据，不等于真实日更段已 sealed",
        }
    )
    shadow_store = PanelDataStore(scratch_root / "shadow")
    bootstrap_manifest(shadow_store)
    for date in shadow_days:
        copy_shadow_day_to_scratch(ctx.store, shadow_store, date)
    assert_forming_fixture(shadow_store, shadow_days)
    assert_fixture_panel_fidelity(ctx.store, shadow_store, shadow_days)
    fragment["fixture"] = {
        "copied_days": len(shadow_days),
        "forming": True,
        "watermark_cleared": True,
        "panel_core_fidelity": True,
    }
    with _counting_build_calls() as build_calls:
        results, elapsed = _refresh_shadow_days(shadow_store, shadow_days, refresh_kwargs)
    fragment["per_day"] = [_per_day_entry(result) for result in results]
    fragment["total_elapsed_seconds"] = round(elapsed, 3)
    fragment["build_features_data_calls"] = build_calls["count"]
    _assert_shadow_transitions(results, len(shadow_days))
    fragment["transitions"] = len(results)
    if build_calls["count"] != 0:
        raise AcceptanceError(f"专用刷新触发整日重建 {build_calls['count']} 次")
    if elapsed > budget_seconds:
        raise AcceptanceError(
            f"历史影子 {len(shadow_days)} 日刷新总耗时 {elapsed:.1f}s 超预算 "
            f"{budget_seconds:.1f}s（分解见 per_day）；不得以整日重建代替"
        )
    fragment["within_budget"] = True
    mismatches: list[str] = []
    for date in shadow_days:
        mismatches.extend(_reconcile_shadow_day(ctx.store, shadow_store, date))
    fragment["reconcile"] = {
        "compared_days": len(shadow_days),
        "compared_tables": len(shadow_days) * len(_LABEL_NAMES),
        "mismatches": mismatches,
    }
    if mismatches:
        raise AcceptanceError(f"raw+neu 逐值对账不等价 {len(mismatches)} 项: {mismatches[:3]}")
    if plan_label_refreshes(shadow_store, list(shadow_days), ctx.calendar, ctx.watermark):
        raise AcceptanceError("生产 planner 不应选择历史影子日")
    fragment["planner_excludes_shadow"] = True
    fragment["idempotency"] = _noop_rerun_check(shadow_store, shadow_days, refresh_kwargs)
    fragment["fault_injection"] = fault_injection_check(
        ctx.store, scratch_root, shadow_days[0], refresh_kwargs
    )
    fragment["cross_seam_impact_total"] = _aggregate_cross_seam(results)


# ---------- 真实日更 forming、共享 readiness 与 Q6 键集 ----------


def _assert_daily_day_state(
    daily_store: Any,
    date: str,
    reference_frame: pd.DataFrame,
    calendar: Sequence[str],
    watermark: str,
) -> dict[str, Any]:
    """结构段初建：八族/行域/status 快查 + 共享 consumer readiness + Q6 键集精确相等。"""
    snapshot = daily_store.manifest.snapshot()
    for group in _ALL_GROUPS:
        meta = snapshot["partitions"].get(f"panel/{date}/{group}.parquet")
        if meta is None or meta.get("row_domain") != "structural":
            raise AcceptanceError(f"{date} panel 族缺失或行域异常: {group}")
    for name in _LABEL_NAMES:
        meta = snapshot["labels"].get(name, {}).get("partitions", {}).get(f"{date}.parquet")
        if meta is None or meta.get("row_domain") != "structural":
            raise AcceptanceError(f"{date} 标签缺失或行域异常: {name}")
        frame = pd.read_parquet(
            daily_store.labels_dir / name / f"{date}.parquet", columns=["maturity_status"]
        )
        if set(frame["maturity_status"].astype(str)) != {"forming"}:
            raise AcceptanceError(f"{date} 初建标签不是 forming: {name}")
    readiness = validate_label_day_readiness(
        daily_store, date, consumer=True, trading_dates=calendar, input_watermark=watermark
    )
    core = pd.read_parquet(daily_store.panel_dir / date / "core.parquet", columns=_KEY_COLUMNS)
    core_keys = pd.MultiIndex.from_frame(core.astype(str))
    reference = reference_frame.copy()
    reference["trade_date"] = reference["trade_date"].astype(str)
    reference["ts_code"] = reference["ts_code"].astype(str)
    reference_keys = pd.MultiIndex.from_frame(reference[_KEY_COLUMNS])
    if reference_keys.has_duplicates or core_keys.has_duplicates:
        raise AcceptanceError(f"{date} Q6 键集存在重复键")
    if reference_keys.difference(core_keys).size or core_keys.difference(reference_keys).size:
        raise AcceptanceError(
            f"{date} Q6 键集不对齐: 参照 {len(reference_keys)} vs core {len(core_keys)}"
        )
    return {
        "core_rows": len(core_keys),
        "reference_rows": len(reference_keys),
        "readiness_status": readiness["status"],
        "readiness_rows": readiness["rows"],
    }


def _real_daily_phase(ctx: _Context, scratch_root: Path, dates: Sequence[str]) -> dict[str, Any]:
    """逐日 build_daily 落 scratch；旧构建器结构域参照仅作键集锚（Q6，不作数值证据）。"""
    daily_store = PanelDataStore(scratch_root / "daily")
    bootstrap_manifest(daily_store)
    entries: list[dict[str, Any]] = []
    for date in dates:
        started = time.monotonic()
        ctx.builder.build_daily(TradeDate.from_str(date), _ALL_GROUPS, daily_store)
        build_seconds = time.monotonic() - started
        reference_started = time.monotonic()
        captured = ctx.builder._run_capture(
            ctx.builder._daily_capture_start(date), date, keep_dates={date}, structural=True
        )
        reference_seconds = time.monotonic() - reference_started
        if len(captured) != 1 or captured[0][0] != date:
            raise AcceptanceError(f"{date} 旧构建器结构域参照捕获异常: {len(captured)}")
        state = _assert_daily_day_state(
            daily_store, date, captured[0][1], ctx.calendar, ctx.watermark
        )
        impact = cross_seam_impact_for_day(daily_store, date)
        entries.append(
            {
                "trade_date": date,
                "build_seconds": round(build_seconds, 3),
                "reference_seconds": round(reference_seconds, 3),
                "label_status": "forming",
                "row_domain": "structural",
                "shared_readiness": True,
                "q6_keyset_equal": True,
                "cross_seam_impact": impact,
                **state,
            }
        )
        print(
            f"REAL_DAILY date={date} status=forming domain=structural readiness=True "
            f"q6_keyset_equal=True rows={state['core_rows']} "
            f"build={build_seconds:.1f}s reference={reference_seconds:.1f}s"
        )
    return {
        "entries": entries,
        "cross_seam_impact_total": _sum_cross_seam_impacts(
            [entry["cross_seam_impact"] for entry in entries]
        ),
        "q6_note": "scratch 旧构建器结构域参照仅作键集精确相等断言；生产 cs_infer 在 "
        ">=20260703 不存在，本参照不作数值等价证据",
        "duplicate_capture_note": "build_daily 内部捕获与 Q6 参照捕获为两次独立管线运行，"
        "耗时分别记录（reviewer1 S5 处置，不改变 T3 构建语义）",
    }


# ---------- ti 块头对照 ----------

#: 分层判据（§19.9 用户裁决）：`np.cumsum` 前缀差分列在全历史与切片上下文的浮点尾差
#: 物理不可逐位一致（实测 ≤1.4e-17 / 相对 ≤1.2e-13），按容差比对并记录尾差量级；
#: 窗口完整性由整数列（ti_inst_days_20）与其余当日值列的精确一致承担。
_TI_TOLERANCE_COLUMNS = ("ti_inst_net_sum_20",)
_TI_SUM_RTOL = 1e-9
_TI_SUM_ATOL = 1e-12


def _compare_ti_columns(
    prod_path: Path, scratch_path: Path, day: str
) -> tuple[list[str], dict[str, float]]:
    """块头日 ti_* 分层判据：精确列逐位一致；容差列超容差才记差异，并返回尾差量级。"""
    if not scratch_path.is_file():
        return [f"{day}: scratch announcement 分区缺失"], {}
    columns = [*_KEY_COLUMNS, *_TI_COLUMNS]
    prod = pd.read_parquet(prod_path, columns=columns)
    scratch = pd.read_parquet(scratch_path, columns=columns)
    for frame in (prod, scratch):
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
    prod = prod.sort_values(_KEY_COLUMNS).reset_index(drop=True)
    scratch = scratch.sort_values(_KEY_COLUMNS).reset_index(drop=True)
    exact_columns = [c for c in _TI_COLUMNS if c not in _TI_TOLERANCE_COLUMNS]
    try:
        pd.testing.assert_frame_equal(
            scratch[[*_KEY_COLUMNS, *exact_columns]],
            prod[[*_KEY_COLUMNS, *exact_columns]],
            check_exact=True,
        )
    except AssertionError as exc:
        return [f"{day}: 精确列不一致: {str(exc)[:400]}"], {}
    problems: list[str] = []
    stats: dict[str, float] = {}
    for column in _TI_TOLERANCE_COLUMNS:
        expected = prod[column].to_numpy(dtype="float64")
        actual = scratch[column].to_numpy(dtype="float64")
        expected_finite = np.isfinite(expected)
        actual_finite = np.isfinite(actual)
        stats[f"{column}_nonfinite_rows"] = int((~actual_finite).sum())
        if not (expected_finite == actual_finite).all():
            problems.append(f"{day}: {column} 有限性/NaN 模式不一致（NaN 位置不同）")
            continue
        mask = expected_finite  # 两侧模式已一致
        if not mask.any():
            problems.append(f"{day}: {column} 全部为 NaN，结构异常")
            continue
        delta = np.abs(actual[mask] - expected[mask])
        limit = _TI_SUM_ATOL + _TI_SUM_RTOL * np.abs(expected[mask])  # np.allclose 同语义
        stats[f"{column}_max_abs"] = float(delta.max()) if len(delta) else 0.0
        stats[f"{column}_max_rel"] = (
            float((delta / np.maximum(np.abs(expected[mask]), _TI_SUM_ATOL)).max())
            if len(delta)
            else 0.0
        )
        over = int((delta > limit).sum())
        if over:
            problems.append(
                f"{day}: {column} 超容差 {over} 行（max_abs={stats[f'{column}_max_abs']:.2e}，"
                f"容差 atol={_TI_SUM_ATOL} + rtol={_TI_SUM_RTOL}×|expected|）"
            )
    return problems, stats


def _ti_block_head_phase(
    ctx: _Context, scratch_root: Path, window_start: str, window_end: str
) -> dict[str, Any]:
    """短 backfill 块头 19 个交易日的 ti_* 分层判据对照（§19.9 用户裁决）。"""
    if not _TI_COLUMNS:
        raise AcceptanceError("announcement 族 ti_* 列集为空")
    ti_store = PanelDataStore(scratch_root / "ti")
    bootstrap_manifest(ti_store)
    started = time.monotonic()
    ctx.builder.backfill(
        TradeDate.from_str(window_start),
        TradeDate.from_str(window_end),
        ("announcement",),
        ti_store,
    )
    elapsed = time.monotonic() - started
    range_days = [d for d in ctx.calendar if window_start <= d <= window_end]
    head_days = range_days[:19]
    if len(head_days) != 19:
        raise AcceptanceError(f"ti 块头窗口不足 19 个交易日: {window_start}~{window_end}")
    mismatches: list[str] = []
    tail_stats: dict[str, float] = {}
    for day in head_days:
        problems, stats = _compare_ti_columns(
            ctx.store.panel_dir / day / "announcement.parquet",
            ti_store.panel_dir / day / "announcement.parquet",
            day,
        )
        mismatches.extend(problems)
        for key, value in stats.items():
            tail_stats[key] = max(tail_stats.get(key, 0.0), value)
    if mismatches:
        raise AcceptanceError(f"ti 块头对照差异 {len(mismatches)} 项: {mismatches[:3]}")
    return {
        "window": f"{window_start}~{window_end}",
        "head_days": head_days,
        "ti_columns": list(_TI_COLUMNS),
        "layered_criterion": {
            "exact_columns": [c for c in _TI_COLUMNS if c not in _TI_TOLERANCE_COLUMNS],
            "tolerance_columns": list(_TI_TOLERANCE_COLUMNS),
            "atol": _TI_SUM_ATOL,
            "rtol": _TI_SUM_RTOL,
            "adjudication": "§19.9 用户裁决（2026-10-19）：cumsum 前缀差分浮点尾差分层判据",
        },
        "tail_stats": tail_stats,
        "compared_days": len(head_days),
        "mismatches": mismatches,
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------- 报告路径隔离与授权耗时预检 ----------


def _validate_report_path(data_root: Path, scratch_root: Path, out_path: Path) -> Path:
    """报告路径隔离（委托 parts 共享守卫）：已存在/命中生产资产/scratch 拒绝，不落报告。"""
    try:
        return validate_output_path(data_root, out_path, scratch_root=scratch_root)
    except AcceptanceError as exc:
        raise ReportPathRejected(str(exc)) from exc


def _validate_roots(data_root: Path, scratch_root: Path) -> None:
    if not (data_root / "features" / "manifest.json").is_file():
        raise AcceptanceError(f"生产 manifest 缺失: {data_root}")
    if scratch_root.resolve().is_relative_to(data_root.resolve()):
        raise AcceptanceError("scratch 根不得落在生产数据根内")
    if scratch_root.exists() and any(scratch_root.iterdir()):
        raise AcceptanceError(f"scratch 根非空，拒绝覆盖既有现场: {scratch_root}")


def _estimate_acceptance_seconds(dates_count: int, shadow_count: int) -> int:
    """授权耗时静态估算；实测以报告为准，预计超授权在启动前拒绝。"""
    build = dates_count * (_BUILD_SECONDS_PER_DAY + _REFERENCE_SECONDS_PER_DAY)
    refresh = shadow_count * _REFRESH_SECONDS_PER_DAY
    return build + refresh + _TI_BLOCK_HEAD_SECONDS + _IDENTITY_HASHING_SECONDS


def _run_id_of(out_path: Path) -> str:
    stem = out_path.name.removesuffix(".json")
    return stem.removeprefix(_REPORT_PREFIX) if stem.startswith(_REPORT_PREFIX) else stem


# ---------- 编排 ----------


def _open_context(
    data_root: Path,
    *,
    store: Any = None,
    loader: Any = None,
    builder: Any = None,
    producer: Mapping[str, Any] | None = None,
    workspace: Mapping[str, Any] | None = None,
) -> _Context:
    store = store or PanelDataStore(data_root)
    _reject_pending_wal(store.wal_path)
    loader = loader or DataLoader(Storage(root_path=str(data_root)))
    builder = builder or V2PanelBuilder(loader)
    calendar = list(builder._full_calendar())
    watermark = builder._data_horizon()
    return _Context(
        store=store,
        loader=loader,
        builder=builder,
        calendar=calendar,
        watermark=watermark,
        producer=dict(producer or producer_identity(ROOT)),
        workspace=dict(workspace or workspace_identity(ROOT)),
    )


def _run_phases(
    ctx: _Context,
    report: dict[str, Any],
    scratch_root: Path,
    budget_seconds: float,
) -> None:
    run_id = report["run_id"]
    shadow_days: list[str] = list(report["precheck"]["shadow_days"])
    clean_frames, loaded_watermark, refresh_identities = _load_shadow_clean_frames(ctx, shadow_days)
    pipeline_identities, left = pipeline_input_identities(
        ctx.loader,
        ctx.calendar,
        shadow_days,
        report["dates"],
        ctx.watermark,
        ctx.builder._daily_capture_start,
    )
    source_identities = production_source_identities(ctx.store, shadow_days)
    merged = {
        record["path"]: record
        for record in [*refresh_identities, *pipeline_identities, *source_identities]
    }
    baseline_identities = sorted(merged.values(), key=lambda record: record["path"])
    input_root = Path(ctx.loader.storage.root_path)
    dir_listing_before = directory_listing_evidence(input_root)
    frozen_before = capture_frozen_before(ctx.store, input_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    with recording_parquet_reads(input_root, baseline_identities) as read_layer:
        report["real_daily"] = _real_daily_phase(ctx, scratch_root, report["dates"])
        refresh_kwargs = _refresh_kwargs(
            ctx.calendar,
            clean_frames,
            loaded_watermark,
            ctx.builder,
            ctx.producer,
            refresh_identities,
            input_root,
            run_id,
        )
        shadow_fragment: dict[str, Any] = {}
        report["historical_shadow"] = shadow_fragment
        _historical_shadow_phase(
            ctx, scratch_root, shadow_days, refresh_kwargs, budget_seconds, shadow_fragment
        )
        report["ti_block_head"] = _ti_block_head_phase(
            ctx, scratch_root, shadow_days[0], LABEL_FILTERED_THROUGH
        )
    all_identities = sorted(read_layer.values(), key=lambda record: record["path"])
    report["identity"] = {
        "input_files": all_identities,
        "input_file_count": len(all_identities),
        "baseline_file_count": len(baseline_identities),
        "read_layer_file_count": len(all_identities) - len(baseline_identities),
        "dir_watermark": ctx.watermark,
        "loaded_watermark": loaded_watermark,
        "min_input_date": left,
        "producer": dict(ctx.producer),
        "workspace": dict(ctx.workspace),
        "identity_scope": "读取层实际读取记录（pd.read_parquet 为全仓唯一内容读取入口；"
        "首次读取前即采集身份、读取后立即核验、运行结束再复核）+ 预采集基线（附录 B 十件、"
        "clean 三数据集 D-14 月窗口、静态单文件、top_inst 全年分区、cs 锚点、影子来源）"
        "+ 生产 raw/clean 全数据集与 store 资产目录的文件集合前后冻结比对",
    }
    report["frozen_segment"] = frozen_segment_after_check(ctx.store, input_root, frozen_before)
    listing_after = directory_listing_evidence(input_root)
    listing_diff = diff_directory_listing(dir_listing_before, listing_after)
    if listing_diff:
        raise AcceptanceError(f"生产文件集合在验收前后新增/删除: {listing_diff}")
    report["identity"]["dir_listing_unchanged"] = True
    report["identity"]["dir_listing_digest"] = canonical_json_sha256(listing_after)
    assert_run_identities(all_identities, input_root, ctx.producer, ctx.workspace)
    report["identity"]["input_identity_unchanged"] = True
    report["cross_seam"] = {
        "historical_shadow": shadow_fragment.get("cross_seam_impact_total", {}),
        "real_daily_scratch": report["real_daily"].get("cross_seam_impact_total", {}),
        "scope_note": "historical_shadow 为 label_filtered 影子域（y20 已成熟，受影响计数趋零）；"
        "real_daily_scratch 为本次真实 structural 日更域（y20 未到水位大面积为空，"
        "y5/y10 受影响计数为实际值）；生产追加段的 D 后计数以跨缝补充报告为准；"
        "跨缝使用 neu_y_ret_5/10 前仍须另行裁决母域差异",
        "b0_declaration": "当前 B0 14 折窗口全部 ≤20260702（不跨缝）；跨缝训练须另行预登记，"
        "本阶段不跑跨缝训练/收益实验",
    }


def verify_store_iteration(
    data_root: Path | str,
    scratch_root: Path | str,
    out_path: Path | str,
    *,
    dates: Sequence[str],
    shadow_count: int = 20,
    budget_seconds: float = 60.0,
    authorized_minutes: float = 60.0,
    store: Any = None,
    loader: Any = None,
    builder: Any = None,
    producer: Mapping[str, Any] | None = None,
    workspace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """T4 真实 scratch 验收主流程；门失败先落正式报告再以非零退出。

    报告路径已存在或命中生产资产/scratch 时不落任何报告直接拒绝（rejected）；
    下层具体异常（OSError/ValueError/RuntimeError/KeyError/TypeError）同样落 failed
    报告并记录异常类别与失败阶段。
    """
    root, scratch, out = Path(data_root), Path(scratch_root), Path(out_path)
    base: dict[str, Any] = {
        "run_id": _run_id_of(out),
        "tool": "verify_store_iteration",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(root.resolve()),
        "scratch_root": str(scratch.resolve()),
        "report_path": str(out.resolve()),
        "dates": sorted(str(date) for date in dates),
        "historical_shadow_count": shadow_count,
        "evidence_boundary": "只读生产输入，只写 scratch 与正式报告；历史影子 ≠ 真实日更段 sealed",
    }
    try:
        resolved_out = _validate_report_path(root, scratch, out)
    except ReportPathRejected as exc:
        print(f"REPORT_REJECTED {exc}")
        return {**base, "status": "rejected", "failure_reason": str(exc)}
    report: dict[str, Any] = {**base, "status": "running"}
    precheck_passed = False
    stage = "validate_roots"
    try:
        _validate_roots(root, scratch)
        stage = "estimate"
        estimated = _estimate_acceptance_seconds(len(report["dates"]), shadow_count)
        report["estimated_seconds"] = estimated
        report["authorized_minutes"] = authorized_minutes
        if estimated > authorized_minutes * 60:
            raise AcceptanceError(
                f"预计耗时 {estimated / 60:.1f} 分钟超过本次授权 {authorized_minutes:.1f} 分钟；"
                "启动前拒绝，未写文件"
            )
        stage = "open_context"
        ctx = _open_context(
            root,
            store=store,
            loader=loader,
            builder=builder,
            producer=producer,
            workspace=workspace,
        )
        stage = "mechanical_precheck"
        report["precheck"] = _mechanical_precheck(ctx, report["dates"], shadow_count)
        _print_precheck(report["precheck"])
        stage = "appendix_b"
        report["appendix_b"] = appendix_b_report(root)
        print(f"APPENDIX_B {report['appendix_b']['matched']}/{report['appendix_b']['total']} match")
        if not report["appendix_b"]["all_match"]:
            report["status"] = "precheck_failed"
            report["failure_reason"] = (
                "P1 附录 B 指纹预检不匹配：先报告并归因输入修订，重新证明同数据态；"
                "既不得判为实现漂移，也不得放行"
            )
            _finalize_report(report, resolved_out)
            return report
        precheck_passed = True
        stage = "run_phases"
        _run_phases(ctx, report, scratch, budget_seconds)
        report["status"] = "passed"
    except AcceptanceError as exc:
        report["status"] = "failed" if precheck_passed else "precheck_failed"
        report["failure_reason"] = str(exc)
        report["failed_stage"] = stage
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        report["status"] = "failed" if precheck_passed else "precheck_failed"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        report["exception_type"] = type(exc).__name__
        report["failed_stage"] = stage
    _finalize_report(report, resolved_out)
    return report


def _finalize_report(report: Mapping[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out_path, dict(report))
    status = report.get("status")
    print(f"REPORT {out_path} status={status}")
    if status == "passed":
        print("STORE_ACCEPTANCE=PASS")
    else:
        print(f"STORE_ACCEPTANCE={str(status).upper()} reason={report.get('failure_reason')}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.supplement_cross_seam:
        try:
            report = supplement_cross_seam(
                args.supplement_cross_seam, args.out, data_root=args.data_root
            )
        except AcceptanceError as exc:
            print(f"SUPPLEMENT_REJECTED {exc}")
            return 1
        return 0 if report["status"] == "passed" else 1
    report = verify_store_iteration(
        args.data_root,
        args.scratch_root,
        args.out,
        dates=args.dates,
        shadow_count=args.historical_shadow_count,
        budget_seconds=args.shadow_budget_seconds,
        authorized_minutes=args.authorized_minutes,
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
