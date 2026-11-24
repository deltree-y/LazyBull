# -*- coding: utf-8 -*-
"""run-a 结构域重建与逐日打分（协议 §5.1、§5.2-5；审点二 A 批）。

结构重建（恢复路径）
====================
按**行业变更日**切分输出窗（窗内 117 个变更日 → 118 段）：段内「已在段首
上市的股票」行业映射恒定 ⇒ 段首日 PIT 快照对**全段每个保留日**精确正确
（行业依赖列 = alpha_industry_5/20、ind_ret_avg、ind_momentum_rank，共 4 列，
逐日组统计来自帧内 sw 列）。段中新上市股票行业列为 NaN，但其 ``list_days``
远小于 365 门槛、不进合格域，不影响打分。

预热口径：``build_features_data`` 自带「捕获起点再回推 7 个自然月」的加载
预热（pipeline.py 生产常量）——段首保留日的加载史 ≈ 7 个月 ≈ 145 交易日，
覆盖最大股票级滚动窗（60 行，volatility_factors）与全部 PIT 回看；与
checkpoint-1 已评审通过的单日验证（90 自然日捕获窗 ≈ 同量级加载史）同口径。
捕获起点 = 段首日 ⇒ 计算日 = 段交易日（预热只发生在加载层，不重复计算帧）。

逐日核验：对「PIT 命中 ∧ list_days ≥ 365 ∧ 帧内有行」的股票，帧内
sw_l1/l2/l3 必须等于当日 PIT 区间命中值；任何不符 ⇒ 该日单独重捕
（捕获起点 = 当日）并覆写缓存件。

打分
====
资格 = 审点一 :func:`probes.apply_selection` 同一实现（母表 → 时点可知
逐层排除 → 成交额/市值门槛）；模型输入走 ``MLSignal.generate_ranked``
（选择/质量门/事件衰减与生产同源，无 latest 回退）；并列分数以股票代码
升序决胜（协议 §5.2-5）；逐日宽度台账落盘。
"""

from __future__ import annotations

import json
import time
from bisect import bisect_left
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from scripts.v2_p7.cashflow_sleeve_split import probes, spec, structbuild
from scripts.v2_p7.common import sha256_file, write_json

#: 缓存子目录（campaign/cache/<spec16>/structural_full/<段>）
CACHE_LEAF = "structural_full"

#: 训练历史窗缓存子目录（checkpoint-2 第 2 项：PIT 叠加需全历史结构帧）
HISTORY_CACHE_LEAF = "structural_history"

#: 行业依赖的帧列（逐日核验对象）
INDUSTRY_FRAME_COLUMNS = ("sw_l1_code", "sw_l2_code", "sw_l3_code")


class ScoringError(RuntimeError):
    """结构重建/打分失败（fail-fast，不静默缩域）。"""


def cache_root(camp: Path, history: bool = False) -> Path:
    """研究缓存根（协议 §2.2：campaign/cache/<input_digest>/）。"""
    leaf = HISTORY_CACHE_LEAF if history else CACHE_LEAF
    return Path(camp) / "cache" / spec.spec_sha256()[:16] / leaf


def _required_columns_for_day(
    required: set[str], frame_file: Path, cs_dir: Optional[Path], pq: Any
) -> set[str]:
    """该日应核对的列集 = 路由列 ∩ 生产同日分区列集（生产缺分区则保守取全集）。"""
    if cs_dir is None:
        return required
    prod_file = cs_dir / f"{frame_file.stem}.parquet"
    if prod_file.exists():
        return required & set(pq.read_schema(prod_file).names)
    return required


def assert_route_columns_coverage(
    camp: Path,
    routes: Mapping[int, Mapping[str, Any]],
    emit: Callable[[str], None],
    prod_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """段缓存路由列覆盖断言（fail-closed，2026-11-22 整列消失事故防复发）。

    口径（2026-11-23 修订）：段帧列集必须覆盖 **路由列 ∩ 生产同日分区列集**。
    生产列集随数据源起点演化（2018 前无筹码列等），早期段不能要求全路由列
    在场；逐日对照生产同日分区——段帧缺「生产也缺」的列属既定演化，段帧缺
    「生产存在」的路由列才是整列消失缺陷（share_float/north 窄窗命不中型），
    fail-closed。生产分区缺失的日按 required 全集保守核对。
    """
    import pyarrow.parquet as pq

    required: set[str] = set()
    for arm_cols in routes.values():
        for arm in ("F", "R", "B"):
            required |= set(arm_cols.get(arm) or ())
    if not required:
        raise ScoringError("路由列空表，无法执行覆盖断言")
    # 只核对当前 spec 代际目录：cache/ 下可能有 _bak_* 备份目录（旧代际产物，
    # 2026-11-22 教训——误扫备份目录会把历史缺列段误判为本代缺陷）
    digest_dir = Path(camp) / "cache" / spec.spec_sha256()[:16]
    if not digest_dir.exists():
        raise ScoringError(f"结构缓存不存在：{digest_dir}")
    cs_dir = (Path(prod_root) / "data" / "features" / "cs_train") if prod_root else None
    files_checked = 0
    missing: Dict[str, List[str]] = {}
    for leaf in (CACHE_LEAF, HISTORY_CACHE_LEAF):
        root = digest_dir / leaf
        if not root.exists():
            continue
        for seg in sorted(p for p in root.iterdir() if p.is_dir()):
            for frame_file in sorted(seg.glob("*.parquet")):
                cols = set(pq.read_schema(frame_file).names)
                need = _required_columns_for_day(required, frame_file, cs_dir, pq)
                miss = sorted(need - cols)
                if miss:
                    missing[f"{leaf}/{seg.name}/{frame_file.name}"] = miss
                files_checked += 1
    if missing:
        sample = list(missing.items())[:5]
        raise ScoringError(
            f"段缓存路由列覆盖断言失败：{len(missing)} 个文件缺列（前 5：{sample}）；"
            "疑似公告类整列消失缺陷复发或缓存代际混杂，禁止继续训练/打分"
        )
    emit(f"      路由列覆盖断言：{len(required)} 列 × {files_checked} 文件全通过")
    return {"required_columns": len(required), "files_checked": files_checked}


def account_calendar(root: Path) -> List[str]:
    """账户窗交易日（协议 §4：2019-04-01 ~ 2026-10-29，SSE 开市日）。"""
    from scripts.v2_p7.cashflow_sleeve_split.training import load_trade_calendar

    full = load_trade_calendar(root)
    days = [day for day in full if str(spec.WINDOW["start"]) <= day <= str(spec.WINDOW["end"])]
    if len(days) != int(spec.WINDOW["return_intervals"]):
        raise ScoringError(f"账户窗交易日数不符：{len(days)} != {spec.WINDOW['return_intervals']}")
    return days


def _window_days(root: Path, history_start: Optional[str]) -> List[str]:
    """重建日历：账户窗；``history_start`` ⇒ 训练历史窗（起点提前至该日）。"""
    if not history_start:
        return account_calendar(root)
    from scripts.v2_p7.cashflow_sleeve_split.training import load_trade_calendar

    full = load_trade_calendar(root)
    days = [day for day in full if str(history_start) <= day <= str(spec.WINDOW["end"])]
    if not days:
        raise ScoringError(f"历史窗日历为空：{history_start}~{spec.WINDOW['end']}")
    return days


def _late_coverage_dates(root: Path, pit: Any) -> set:
    """晚覆盖首现日集合：老股票（list_date 早于首区间 ≥365 天）窗内首现。"""
    basic_path = Path(root) / "data/clean/stock_basic.parquet"
    if not basic_path.exists():
        raise ScoringError(f"stock_basic 缺件：{basic_path}")
    basic = pd.read_parquet(basic_path, columns=["ts_code", "list_date"])
    list_date = basic.set_index(basic["ts_code"].astype(str))["list_date"].astype(str)
    dates: set = set()
    first_intervals = pit.drop_duplicates("ts_code", keep="first")
    for _, row in first_intervals.iterrows():
        in_date = str(row["in_date"])
        listed = str(list_date.get(str(row["ts_code"]), ""))
        if len(listed) != 8 or not listed.isdigit():
            continue
        gap_days = (pd.Timestamp(in_date) - pd.Timestamp(listed)).days
        if gap_days >= int(spec.PROBE_SELECTION["min_list_days"]):
            dates.add(in_date)
    return dates


def industry_change_dates(
    root: Path, window_days: Sequence[str], *, snap_from: Optional[str] = None
) -> List[str]:
    """窗内行业映射边界日（升序；分块边界唯一来源）。

    两类边界：
    1. **变更日**：同一 ``ts_code`` 相邻 PIT 区间的 l1/l2/l3 任一不同 ⇒ 后一区间
       ``in_date``。新上市（首区间）不算变更；
    2. **晚覆盖首现日**：老股票（``stock_basic.list_date`` 早于首区间 in_date
       ≥ 365 天）的 PIT 覆盖在窗内才出现（代码复用实例：001914.SZ 首区间
       20191223、list_date 19940928）——段首快照在 in_date 前不含它，必须断段。

    ``snap_from`` 仅用于保持已有缓存目录的边界。实际捕获按当前交易窗吸附。
    """
    pit_path = Path(root) / spec.INDUSTRY_PIT_INJECTION["pit_relative"]
    if not pit_path.exists():
        raise ScoringError(f"行业 PIT 缺件：{pit_path}")
    pit = pd.read_parquet(pit_path).sort_values(["ts_code", "in_date"])
    changed: set = set()
    codes = pit["ts_code"].astype(str).values
    in_dates = pit["in_date"].astype(str).values
    l1 = pit["l1_code"].astype(str).values
    l2 = pit["l2_code"].astype(str).values
    l3 = pit["ind_code"].astype(str).values
    for index in range(1, len(pit)):
        if codes[index] != codes[index - 1]:
            continue
        if (l1[index], l2[index], l3[index]) != (l1[index - 1], l2[index - 1], l3[index - 1]):
            changed.add(str(in_dates[index]))
    changed |= _late_coverage_dates(root, pit)
    window_set = set(window_days)
    snapped: set = set()
    for day in changed:
        if day in window_set:
            snapped.add(day)
            continue
        if not window_days or not (window_days[0] <= day <= window_days[-1]):
            continue
        if snap_from is None or day >= snap_from:
            snapped.add(window_days[bisect_left(window_days, day)])
    return sorted(snapped)


def plan_segments(window_days: Sequence[str], change_dates: Sequence[str]) -> List[Dict[str, Any]]:
    """按变更日切段：每段内「已在市股票」行业映射恒定。"""
    cuts = set(change_dates)
    segments: List[Dict[str, Any]] = []
    current: List[str] = []
    for day in window_days:
        if current and day in cuts:
            segments.append(_segment(current))
            current = []
        current.append(day)
    if current:
        segments.append(_segment(current))
    return segments


def _segment(days: List[str]) -> Dict[str, Any]:
    return {"first": days[0], "last": days[-1], "days": list(days), "n": len(days)}


def _verify_day_industry(frame: Any, day: str, root: Path) -> Dict[str, Any]:
    """当日帧行业映射 vs PIT 区间命中（仅核「PIT 命中 ∧ list_days≥365」的股票）。

    向量化合并比对：帧列 sw_l1/l2/l3 必须逐只等于当日 PIT 快照（注入口径）。
    """
    snapshot = structbuild.industry_pit_snapshot(str(day), root)
    expected = snapshot["frame"].copy()
    expected["ts_code"] = expected["ts_code"].astype(str)
    merged = frame.merge(
        expected[["ts_code", *INDUSTRY_FRAME_COLUMNS]],
        on="ts_code",
        how="inner",
        suffixes=("", "_pit"),
    )
    list_days = pd.to_numeric(merged.get("list_days"), errors="coerce")
    merged = merged[list_days.fillna(-1) >= int(spec.PROBE_SELECTION["min_list_days"])]
    mismatched: List[Dict[str, str]] = []
    mismatch_total = 0
    for column in INDUSTRY_FRAME_COLUMNS:
        left = merged[column].fillna("").astype(str)
        right = merged[f"{column}_pit"].fillna("").astype(str)
        bad_mask = (left != right).to_numpy()
        mismatch_total += int(bad_mask.sum())
        bad = merged.loc[bad_mask]
        for _, row in bad.head(max(8 - len(mismatched), 0)).iterrows():
            mismatched.append(
                {
                    "ts_code": str(row["ts_code"]),
                    "column": column,
                    "expected": str(row[f"{column}_pit"]),
                    "actual": str(row[column]),
                }
            )
    return {
        "day": str(day),
        "checked_n": int(len(merged)),
        "mismatch_n": mismatch_total,
        "mismatches": mismatched,
        "pit_codes": int(len(expected)),
    }


def _segment_manifest_path(seg_dir: Path) -> Path:
    return seg_dir / "MANIFEST.json"


def _write_segment_manifest(
    seg_dir: Path, seg: Mapping[str, Any], recaptured: Sequence[str]
) -> None:
    """分段内容清单（逐件 SHA256；续跑内容核验的唯一依据）。"""
    payload = {
        "schema": "p7_cashflow_split_segment_manifest/v1",
        "first": str(seg["first"]),
        "last": str(seg["last"]),
        "days": [str(day) for day in seg["days"]],
        "recaptured_days": [str(day) for day in recaptured],
        "files": {f"{day}.parquet": sha256_file(seg_dir / f"{day}.parquet") for day in seg["days"]},
        "spec_sha256": spec.spec_sha256(),
    }
    manifest_path = _segment_manifest_path(seg_dir)
    staging_path = manifest_path.with_suffix(".tmp")
    write_json(staging_path, payload)
    staging_path.replace(manifest_path)


def _segment_cache_status(seg_dir: Path, seg: Mapping[str, Any]) -> str:
    """以完成清单、日期集合与逐件 SHA 核验分段缓存。"""
    manifest_path = _segment_manifest_path(seg_dir)
    if not manifest_path.exists():
        return "missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return "mismatch"
    if not isinstance(manifest, dict):
        return "mismatch"
    days = [str(day) for day in seg["days"]]
    plan_identity = {
        "schema": "p7_cashflow_split_segment_manifest/v1",
        "first": str(seg["first"]),
        "last": str(seg["last"]),
        "days": days,
        "spec_sha256": spec.spec_sha256(),
    }
    if any(manifest.get(key) != value for key, value in plan_identity.items()):
        return "mismatch"
    files = manifest.get("files")
    expected = {f"{day}.parquet" for day in days}
    if not isinstance(files, dict) or set(files) != expected:
        return "mismatch"
    if {path.name for path in seg_dir.glob("*.parquet") if path.is_file()} != expected:
        return "mismatch"
    if any(sha256_file(seg_dir / name) != digest for name, digest in files.items()):
        return "mismatch"
    return "verified"


def _capture_segment_frames(base: Path, seg_dir: Path, sub: Mapping[str, Any]) -> Dict[str, Any]:
    """按子段段首的 PIT 快照构建该子段全部交易日。"""
    snapshot = structbuild.industry_pit_snapshot(sub["first"], base)
    result = structbuild.capture(
        base,
        sub["first"],
        sub["last"],
        keep_dates=sub["days"],
        label_mode="normal",
        industry_frame=snapshot["frame"],
        industry_source="industry_pit_injected",
        write_dir=seg_dir,
        sample_rss=False,
    )
    if sorted(result["dates"]) != sub["days"]:
        raise ScoringError(f"段 {sub['first']} 捕获日期不符：{len(result['dates'])} != {sub['n']}")
    return result


def _capture_segment_task(args: Tuple[str, str, Mapping[str, Any], bool]) -> Dict[str, Any]:
    """进程池任务：分子段捕获、逐日核验、重捕后发布完成清单。"""
    root_str, camp_str, seg, history = args
    base = Path(root_str)
    camp = Path(camp_str)
    seg_dir = cache_root(camp, history=history) / f"{seg['first']}_{seg['last']}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    # 历史窗的元旦等非交易日也会改变行业；沿用原缓存目录，每次捕获用实际生效日切分。
    cuts = industry_change_dates(base, seg["days"])
    subs = plan_segments(seg["days"], cuts)
    started = time.perf_counter()
    capture_seconds = 0.0
    write_bytes = 0
    columns_n = 0
    for sub in subs:
        result = _capture_segment_frames(base, seg_dir, sub)
        capture_seconds += result["elapsed_seconds"]
        write_bytes += int((result.get("write") or {}).get("bytes") or 0)
        columns_n = max(columns_n, result["columns_n"])
        del result
    recaptured: List[str] = []
    for day in seg["days"]:
        frame = pd.read_parquet(
            seg_dir / f"{day}.parquet", columns=["ts_code", "list_days", *INDUSTRY_FRAME_COLUMNS]
        )
        check = _verify_day_industry(frame, day, base)
        if check["mismatch_n"] > 0:
            day_snapshot = structbuild.industry_pit_snapshot(day, base)
            fix = structbuild.capture(
                base,
                day,
                day,
                keep_dates=[day],
                label_mode="normal",
                industry_frame=day_snapshot["frame"],
                industry_source="industry_pit_injected",
                write_dir=None,
                sample_rss=False,
            )
            fixed_frame = fix["frames"][0][1]
            fixed_frame.to_parquet(seg_dir / f"{day}.parquet")
            recheck = _verify_day_industry(fixed_frame, day, base)
            if recheck["mismatch_n"] > 0:
                raise ScoringError(f"{day} 重捕后行业核验仍不符：{recheck['mismatches'][:3]}")
            recaptured.append(day)
    _write_segment_manifest(seg_dir, seg, recaptured)
    return {
        "segment": seg["first"],
        "last": seg["last"],
        "days": seg["n"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "capture_seconds": round(capture_seconds, 3),
        "capture_windows": [[sub["first"], sub["last"]] for sub in subs],
        "verified_days": seg["n"],
        "recaptured_days": recaptured,
        "columns_n": columns_n,
        "write_bytes": write_bytes,
    }


def rebuild_segments(
    root: Path,
    camp: Path,
    state: Dict[str, Any],
    emit: Callable[[str], None],
    *,
    resume: bool,
    persist: Optional[Callable[[], None]] = None,
    workers: int = 1,
    history_start: Optional[str] = None,
) -> Dict[str, Any]:
    """全窗结构重建（分段捕获 + 逐日行业核验 + 必要时单日重捕）。

    断点续跑：分段 MANIFEST 逐件 SHA 核验通过 ⇒ 复用（内容寻址，与 run 无关，
    协议 §2.2 分片复用）。``workers`` > 1 时按进程池并行捕获（各段独立写各自
    目录；生产输入只读；任务函数自包含可 pickle）。``history_start`` ⇒ 训练
    历史窗重建（独立缓存叶 structural_history；checkpoint-2 第 2 项）。
    """
    base = Path(root)
    window = _window_days(base, history_start)
    changes = industry_change_dates(base, window, snap_from=str(spec.WINDOW["start"]))
    segments = plan_segments(window, changes)
    emit(
        f"    分块：{len(segments)} 段（行业变更日 {len(changes)} 个；workers={workers}；"
        f"窗 {window[0]}~{window[-1]}）"
    )
    out_root = cache_root(camp, history=history_start is not None)
    out_root.mkdir(parents=True, exist_ok=True)

    totals = {"segments": len(segments), "days": 0, "recaptured_days": 0, "verified_days": 0}
    records: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for index, seg in enumerate(segments, 1):
        seg_dir = out_root / f"{seg['first']}_{seg['last']}"
        cached_status = _segment_cache_status(seg_dir, seg)
        if cached_status == "verified":
            # 内容寻址缓存：清单核验通过即复用（与 run 无关，协议 §2.2 分片复用）
            emit(
                f"    段 {index}/{len(segments)} {seg['first']}~{seg['last']} 已缓存（SHA 核验跳过）"
            )
            records.append(
                {
                    "segment": seg["first"],
                    "last": seg["last"],
                    "days": seg["n"],
                    "status": "cached_sha_verified",
                }
            )
            totals["days"] += seg["n"]
            continue
        if cached_status == "mismatch":
            raise ScoringError(
                f"段 {seg['first']} 缓存内容与清单不符（fail-closed；删除该段目录后重跑）"
            )
        pending.append({"index": index, "seg": seg})

    if pending and workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        emit(f"    并行捕获 {len(pending)} 段（{workers} workers）…")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _capture_segment_task,
                    (str(base), str(camp), item["seg"], history_start is not None),
                ): item
                for item in pending
            }
            done_n = 0
            for future in as_completed(futures):
                item = futures[future]
                record = future.result()
                done_n += 1
                seg = item["seg"]
                records.append(record)
                state.setdefault("segments", {})[seg["first"]] = {
                    "last": seg["last"],
                    "days": seg["n"],
                    "recaptured_days": record["recaptured_days"],
                    "columns_n": record["columns_n"],
                }
                if persist is not None:
                    persist()
                totals["days"] += seg["n"]
                totals["recaptured_days"] += len(record["recaptured_days"])
                totals["verified_days"] += record["verified_days"]
                emit(
                    f"    段 {done_n}/{len(pending)} 完成 {seg['first']}~{seg['last']}"
                    f"（{record['elapsed_seconds']:.0f}s，{seg['n']} 日，重捕 "
                    f"{len(record['recaptured_days'])} 日）"
                )
    else:
        for item in pending:
            seg = item["seg"]
            emit(
                f"    段 {item['index']}/{len(segments)} "
                f"{seg['first']}~{seg['last']}（{seg['n']} 日）捕获中…"
            )
            record = _capture_segment_task(
                (str(base), str(camp), seg, history_start is not None)
            )
            records.append(record)
            state.setdefault("segments", {})[seg["first"]] = {
                "last": seg["last"],
                "days": seg["n"],
                "recaptured_days": record["recaptured_days"],
                "columns_n": record["columns_n"],
            }
            if persist is not None:
                persist()
            totals["days"] += seg["n"]
            totals["recaptured_days"] += len(record["recaptured_days"])
            totals["verified_days"] += record["verified_days"]
            emit(
                f"      完成 {record['elapsed_seconds']:.0f}s；核验 {record['verified_days']} 日，"
                f"重捕 {len(record['recaptured_days'])} 日"
            )
    return {
        "segments_n": len(segments),
        "change_dates_n": len(changes),
        "workers": workers,
        "cache_root": str(out_root),
        "totals": totals,
        "records": records,
    }


def load_day_frame(camp: Path, day: str) -> Any:
    """读当日结构缓存帧（缺件即失败，禁止旧域回退）。"""
    base = cache_root(camp)
    matches = sorted(base.glob(f"*/{day}.parquet"))
    if not matches:
        raise ScoringError(f"结构缓存缺当日分区：{day}（{base}）")
    if len(matches) > 1:
        raise ScoringError(f"结构缓存当日分区重复：{day} → {matches}")
    return pd.read_parquet(matches[0])


def score_day(
    root: Path,
    frame: Any,
    day: str,
    signal: Any,
    basic_frame: Any,
    industry_hit: Any,
    mother: Any,
    routes: Mapping[int, Mapping[str, Any]],
    route_version: int,
    arm_columns: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    """单日打分：资格（审点一同源）→ MLSignal.generate_ranked → 决胜排序。

    返回逐臂 ``(codes, scores)`` 全排名 + 宽度台账行（``width ledger``）。
    """
    mother_frame = mother[["ts_code", "list_days", "is_st", "is_suspended", "tradable"]].copy()
    mother_frame["trade_date"] = str(day)
    selection = probes.apply_selection(
        mother_frame, frame, basic_frame, industry_hit, str(day), "structural_recovered"
    )
    if selection["status"] != "ok":
        raise ScoringError(f"{day} 资格选择失败：{selection['status']}")
    eligible = selection["eligible"]
    universe = eligible["ts_code"].astype(str).tolist()
    arm_outputs: Dict[str, List[Tuple[str, float]]] = {}
    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm, columns in arm_columns.items():
        ranked = signal[arm].generate_ranked(
            pd.Timestamp(day), universe, {"features": frame.copy()}
        )
        ordered = sorted(
            ((str(code), float(score)) for code, score in ranked),
            key=lambda item: (-item[1], item[0]),
        )
        arm_outputs[arm] = ordered
        finite = [(code, score) for code, score in ordered if pd.notna(score)]
        code_set = {code for code, _ in finite}
        # 有效宽度 = 有限分 ∧ 至少一个臂列有真实来源（有限值）
        if len(finite):
            arm_frame = frame[frame["ts_code"].astype(str).isin(code_set)]
            usable = [c for c in columns if c in arm_frame.columns]
            source_mask = probes._finite_mask(arm_frame, usable) if usable else None
            effective_n = int(source_mask.sum()) if source_mask is not None else 0
        else:
            effective_n = 0
        empty_reason = ""
        if not finite:
            selected, _, _ = signal[arm]._apply_selection_filters(
                frame.loc[frame["ts_code"].astype(str).isin(universe)].copy()
            )
            empty_reason = (
                "universe_empty"
                if not universe
                else "selection_empty" if selected.empty else "kernel_refused"
            )
        per_arm[arm] = {
            "ranked_rows": len(finite),
            "effective_n": effective_n,
            "empty_reason": empty_reason,
            "route_version": int(route_version),
        }
    return {
        "day": str(day),
        "universe": universe,
        "arm_outputs": arm_outputs,
        "eligible_n": int(len(eligible)),
        "mother_n": int(len(mother_frame)),
        "universe_n": int(selection.get("universe_n") or 0),
        "structural_gap_n": int(selection.get("structural_gap_n") or 0),
        "per_arm": per_arm,
        "selection_steps": selection["steps"],
    }


def score_window(
    root: Path,
    camp: Path,
    ranked_dir: Path,
    routes: Mapping[int, Mapping[str, Any]],
    version_map: Mapping[str, Mapping[str, int]],
    state: Dict[str, Any],
    emit: Callable[[str], None],
    *,
    resume: bool,
    persist: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """全域逐日打分（F/R 两臂；日期绑定模型；逐日 checkpoint + 宽度台账）。

    ``version_map``：``{"F": {frozen_version: registered_version}, "R": {...}}``。
    逐日排名落 ``ranked_dir/<arm>/<day>.parquet``（列 trade_date/rank/ts_code/score），
    全部完成后由 :func:`assemble_ranked` 拼装整窗文件。
    """
    from scripts.v2_p3.t5_arm_executors import _signal_selection_mapping
    from scripts.v2_p3.t05b_fingerprint import _configure_signal_selection
    from src.lazybull.v2.core.signal.ml_signal import MLSignal

    base = Path(root)
    window = account_calendar(base)
    ranked_dir.mkdir(parents=True, exist_ok=True)
    for arm in version_map:
        (ranked_dir / arm).mkdir(parents=True, exist_ok=True)

    signals: Dict[str, Any] = {}
    for arm, mapping in version_map.items():
        models_dir = Path(camp) / ("models/f" if arm == "F" else "models/r")
        signal = MLSignal(
            top_n=int(spec.BIG_LEG["top_n"]),
            model_version=None,
            models_dir=str(models_dir),
            verbose=False,
        )
        _configure_signal_selection(signal, _signal_selection_mapping())
        signals[arm] = signal

    basic = probes.read_stock_basic(base)
    if basic["status"] != "ok":
        raise ScoringError(basic["reason"])
    pit = probes.read_industry_pit(base)
    if pit["status"] != "ok":
        raise ScoringError(pit["reason"])
    # 缓存 stock_basic 只读一次；母表逐日读（当日全部行）
    cached_mother: Dict[str, Any] = {}

    ledger: List[Dict[str, Any]] = list(state.get("ledger", []))
    ledger_days = {str(row["day"]) for row in ledger}
    started = time.perf_counter()
    for index, day in enumerate(window, 1):
        if resume and day in ledger_days:
            continue
        summary = _score_one_day(
            base,
            camp,
            ranked_dir,
            day,
            signals,
            version_map,
            routes,
            basic["frame"],
            pit["frame"],
            cached_mother,
        )
        ledger.append(summary)
        state["ledger"] = ledger
        if index % 100 == 0 or index == len(window):
            if persist is not None:
                persist()
            done_before = sum(1 for done in ledger_days if done < day)
            rate = (time.perf_counter() - started) / max(index - done_before, 1)
            arm_f = summary["per_arm"]["F"]["ranked_rows"]
            arm_r = summary["per_arm"]["R"]["ranked_rows"]
            emit(
                f"    打分 {index}/{len(window)}（{summary['day']}）"
                f"F={arm_f} R={arm_r} eligible={summary['eligible_n']} ~{rate:.1f}s/日"
            )
    ledger = sorted(ledger, key=lambda row: str(row["day"]))
    widths = _width_summary(ledger)
    return {
        "days": len(window),
        "ledger_rows": len(ledger),
        "widths": widths,
        "ledger": ledger,
    }


def _score_one_day(
    base: Path,
    camp: Path,
    ranked_dir: Path,
    day: str,
    signals: Mapping[str, Any],
    version_map: Mapping[str, Mapping[str, int]],
    routes: Mapping[int, Mapping[str, Any]],
    basic_frame: Any,
    pit_frame: Any,
    cached_mother: Dict[str, Any],
) -> Dict[str, Any]:
    """单日打分 + 逐日排名 checkpoint（供 score_window 主循环调用）。"""
    from scripts.v2_p7.cashflow_sleeve_split.training import resolve_day_route_version

    route_version = resolve_day_route_version(base, day)
    if route_version is None:
        raise ScoringError(f"{day} 不在任何折/接缝/部署域内")
    for arm in version_map:
        registered = version_map[arm].get(str(route_version))
        if registered is None:
            raise ScoringError(f"{day} 臂 {arm} 缺注册版本映射（v{route_version}）")
        signal = signals[arm]
        if getattr(signal, "model_version", None) != int(registered):
            signal.update_model_version(int(registered))
    route = routes.get(int(route_version))
    if route is None:
        raise ScoringError(f"{day} 缺 v{route_version} 路由")
    frame = load_day_frame(camp, day)
    mother = cached_mother.get(day)
    if mother is None:
        mother = probes.read_mother(base, day)
        if mother["status"] != "ok":
            raise ScoringError(f"{day} 母表缺件：{mother.get('reason')}")
        cached_mother[day] = mother
    industry_hit = probes.industry_for_day(pit_frame, day)
    result = score_day(
        base,
        frame,
        day,
        signals,
        basic_frame,
        industry_hit,
        mother["frame"],
        routes,
        route_version,
        {"F": route["F"], "R": route["R"]},
    )
    _write_day_ranked(ranked_dir, day, result["arm_outputs"])
    return {
        "day": day,
        "route_version": int(route_version),
        "mother_n": result["mother_n"],
        "universe_n": result["universe_n"],
        "eligible_n": result["eligible_n"],
        "structural_gap_n": result["structural_gap_n"],
        "per_arm": result["per_arm"],
    }


def _write_day_ranked(
    ranked_dir: Path, day: str, arm_outputs: Mapping[str, Sequence[Tuple[str, float]]]
) -> None:
    """逐臂写当日排名 checkpoint（名次 1..N）。"""
    for arm, rows in arm_outputs.items():
        out = pd.DataFrame(
            {
                "trade_date": [day] * len(rows),
                "rank": list(range(1, len(rows) + 1)),
                "ts_code": [code for code, _ in rows],
                "score": [score for _, score in rows],
            }
        )
        out.to_parquet(ranked_dir / arm / f"{day}.parquet", index=False)


def _width_summary(ledger: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """逐臂宽度摘要（最小值、<500 日清单、空排名日）。"""
    summary: Dict[str, Any] = {}
    for arm in ("F", "R"):
        effective = [int(row["per_arm"][arm]["effective_n"]) for row in ledger]
        ranked = [int(row["per_arm"][arm]["ranked_rows"]) for row in ledger]
        empty = [str(row["day"]) for row in ledger if int(row["per_arm"][arm]["ranked_rows"]) == 0]
        narrow = [
            {"day": str(row["day"]), "effective_n": int(row["per_arm"][arm]["effective_n"])}
            for row in ledger
            if int(row["per_arm"][arm]["effective_n"]) < 500
        ]
        summary[arm] = {
            "min_ranked_rows": min(ranked) if ranked else None,
            "min_effective_n": min(effective) if effective else None,
            "empty_days": empty,
            "narrow_days_lt500": narrow,
            "narrow_n": len(narrow),
        }
    return summary


def assemble_ranked(ranked_dir: Path, window: Sequence[str]) -> Dict[str, Path]:
    """拼装整窗 ranked parquet（列 trade_date/rank/ts_code/score，名次 1..N 连续）。"""
    outputs: Dict[str, Path] = {}
    for arm in ("F", "R"):
        frames = []
        for day in window:
            path = ranked_dir / arm / f"{day}.parquet"
            if not path.exists():
                raise ScoringError(f"逐日排名缺件：{arm}/{day}")
            frames.append(pd.read_parquet(path))
        combined = pd.concat(frames, ignore_index=True)
        outputs[arm] = ranked_dir / f"ranked_{arm}.parquet"
        combined.to_parquet(outputs[arm], index=False)
    return outputs


def load_ranked_dict(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    """读整窗 ranked parquet → ``{day: [(code, score)]}``（名次/递减校验）。"""
    frame = pd.read_parquet(path, columns=["trade_date", "rank", "ts_code", "score"])
    ranked: Dict[str, List[Tuple[str, float]]] = {}
    for day, group in frame.groupby("trade_date", sort=False):
        day = str(day)
        if group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise ScoringError(f"ranked 名次非法：{day}")
        ranked[day] = list(zip(group.ts_code.astype(str), group.score.astype(float)))
    return ranked


__all__ = [
    "CACHE_LEAF",
    "ScoringError",
    "account_calendar",
    "assemble_ranked",
    "cache_root",
    "industry_change_dates",
    "load_day_frame",
    "load_ranked_dict",
    "plan_segments",
    "rebuild_segments",
    "score_day",
    "score_window",
]
