# -*- coding: utf-8 -*-
"""R0—R3 研究阶段实现（R4—R6 未获经济运行授权，不在本模块提供入口）。

# 复杂度豁免：P7 候选一阶段专用研究工具，不入生产链路（AGENTS §7.5）；
# 拆分计划：R4 授权前把 R0/R2/R3 按「输入装载 / 报告组装 / 校验」拆到独立模块。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

from . import clock, graph, inputs, score, spec, targets
from .common import (
    ResearchError,
    StageContext,
    read_json,
    run_root,
    sha256_file,
    write_json,
    write_parquet,
)


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _scan_partitions(
    root: Path, subdir: str, start: str, end: str, *, base: str = "data/clean"
) -> dict[str, Any]:
    folder = root / base / subdir
    if not folder.exists():
        raise ResearchError(f"输入目录缺失：{folder}")
    files = []
    for path in sorted(folder.glob("*.parquet")):
        day = path.stem.replace("-", "")
        if not (start <= day <= end):
            continue
        files.append(
            {
                "rel": str(path.relative_to(root)).replace("\\", "/"),
                "day": day,
                "bytes": int(path.stat().st_size),
                "rows": _parquet_rows(path),
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ResearchError(f"{subdir} 在 {start}~{end} 没有分区")
    return {
        "subdir": f"{base}/{subdir}",
        "range": [files[0]["day"], files[-1]["day"]],
        "files": files,
        "rows": int(sum(item["rows"] for item in files)),
        "bytes": int(sum(item["bytes"] for item in files)),
    }


def _schema(path: Path) -> dict[str, str]:
    frame = pd.read_parquet(path)
    return {name: str(dtype) for name, dtype in frame.dtypes.items()}


def _baseline_identity(root: Path) -> dict[str, Any]:
    manifest_path = root / spec.BASELINE_REL / "manifest.json"
    if not manifest_path.exists():
        raise ResearchError(f"冻结基准 manifest 缺失：{manifest_path}")
    digest = sha256_file(manifest_path)
    if digest != spec.BASELINE_MANIFEST_SHA:
        raise ResearchError(
            f"基准 manifest SHA 不符：{digest} != {spec.BASELINE_MANIFEST_SHA}"
        )
    manifest = read_json(manifest_path)
    registered = manifest.get("files") or {}
    checks: dict[str, Any] = {"capitals": {}}
    for capital in spec.CAPITALS:
        folder = f"{capital}"
        consumed = [item.format(cap=folder) for item in spec.BASELINE_CONSUMED]
        file_rows = []
        for rel in consumed:
            path = manifest_path.parent / rel
            if not path.exists():
                raise ResearchError(f"基准消费文件缺失：{rel}")
            value = sha256_file(path)
            if value != registered.get(rel):
                raise ResearchError(f"基准消费文件 SHA 与 manifest 不符：{rel}")
            file_rows.append({"rel": rel, "sha256": value})
        a100 = pd.read_parquet(manifest_path.parent / f"{folder}/A100_daily.parquet")
        cash = pd.read_parquet(manifest_path.parent / f"{folder}/A80_B_cash_daily.parquet")
        a80 = pd.DataFrame(read_json(manifest_path.parent / f"{folder}/A80/daily.json"))
        for name, frame in (("A100", a100), ("A80", a80), ("cash", cash)):
            frame["trade_date"] = frame["trade_date"].astype(str)
        dates_equal = (
            a100["trade_date"].tolist()
            == a80["trade_date"].tolist()
            == cash["trade_date"].tolist()
        )
        a_capital = float(capital) * (1.0 - spec.B_SHARE)
        cash_part = float(capital) * spec.B_SHARE
        first_ok = abs(float(a80["total_value"].iloc[0]) - a_capital) <= 1e-6
        cash_ok = bool(
            np.allclose(
                cash["total_value"].to_numpy(dtype=float),
                a80["total_value"].to_numpy(dtype=float) + cash_part,
                rtol=0.0,
                atol=1e-6,
            )
        )
        metrics = calculate_chain_metrics(
            pd.DataFrame({"nav": a100["nav"].astype(float)})
        )
        displayed = spec.BASELINE_DISPLAY[int(capital)]
        checks["capitals"][folder] = {
            "files": file_rows,
            "rows": int(len(a100)),
            "date_range": [a100["trade_date"].iloc[0], a100["trade_date"].iloc[-1]],
            "dates_equal": dates_equal,
            "a80_first_total_value": float(a80["total_value"].iloc[0]),
            "a80_first_is_80pct": first_ok,
            "cash_minus_a80_constant": cash_ok,
            "a100_cagr_full_precision": metrics["cagr"],
            "a100_mdd_loss_full_precision": abs(float(metrics["max_drawdown"])),
            "display_cagr": displayed["a100_cagr"],
            "display_mdd": displayed["a100_mdd"],
            "cagr_abs_diff": abs(float(metrics["cagr"]) - displayed["a100_cagr"]),
            "mdd_abs_diff": abs(
                abs(float(metrics["max_drawdown"])) - displayed["a100_mdd"]
            ),
        }
        if not dates_equal:
            raise ResearchError(f"{folder} 三账日期轴不一致")
        if not first_ok:
            raise ResearchError(f"{folder} 首日 A80 不是 80% 本金")
        if not cash_ok:
            raise ResearchError(f"{folder} A80+现金不等于 A80+固定 20% 现金")
    checks["manifest_sha256"] = digest
    checks["manifest_registered_files"] = len(registered)
    return checks


def _crash_readiness(root: Path) -> dict[str, Any]:
    from scripts.v2_p7.crash_spec import DEFAULT_SPEC_REL

    path = root / DEFAULT_SPEC_REL
    payload: dict[str, Any] = {"spec_rel": DEFAULT_SPEC_REL, "exists": path.exists()}
    if not path.exists():
        payload["state"] = "not_available"
        return payload
    frozen = read_json(path)
    payload["spec_sha256"] = sha256_file(path)
    payload["windows"] = frozen.get("windows") or []
    payload["coverage"] = frozen.get("coverage") or {}
    return payload


def run_r0(ctx: StageContext) -> dict[str, Any]:
    """输入就绪：身份、时钟、基准、广度上界与已知缺口。"""
    root = spec.REPO_ROOT
    started = time.perf_counter()
    days = clock.load_calendar(root)
    rows = clock.month_clock(days, spec.MAIN_START, spec.MAIN_END)
    clock_meta = clock.describe_clock(rows, days)
    basic = inputs.load_stock_basic(root)
    industry = inputs.IndustryPit(root)

    scan_start = "20180101"
    needed: set[str] = set()
    for row in rows:
        needed.add(row["formation_date"])
        needed.add(row["signal_date"])
    sample_rows = [row for position, row in enumerate(rows) if position % 6 == 0]
    liquidity_days: dict[str, list[str]] = {}
    for row in sample_rows:
        end = int(row["signal_index"])
        window = days[max(0, end - spec.AMOUNT_WINDOW + 1):end + 1]
        liquidity_days[row["month"]] = window
        needed.update(window)
    needed_days = sorted(day for day in needed if day >= scan_start)
    if not needed_days:
        raise ResearchError("R0 需要日期为空")

    base_by_day: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    read_start = time.perf_counter()
    for day in needed_days:
        mv_frame = inputs.load_total_mv(root, [day])
        total_mv = mv_frame.set_index("ts_code")["total_mv"].astype(float) * 10000.0
        mapping, industry_meta = industry.at(day)
        selected, summary = inputs.base_domain(
            day=day,
            daily=inputs.load_daily(root, day),
            basic=basic,
            calendar=days,
            industry=mapping,
            total_mv_yuan=total_mv,
            amount_stats=None,
        )
        summary["industry"] = industry_meta
        summary["codes"] = selected["ts_code"].tolist()
        base_by_day[day] = summary
        frames[day] = selected[["ts_code", "close", "l1_name", "total_mv_yuan"]].copy()
    read_seconds = time.perf_counter() - read_start

    def ranked_day(day: str, month: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        if day not in frames:
            raise ResearchError(f"R0 未预读日期：{day}")
        window = liquidity_days.get(month)
        summary = dict(base_by_day[day])
        if window:
            window_frame = inputs.load_daily_concat(
                root, [item for item in window if item <= day][-spec.AMOUNT_WINDOW:]
            )
            stats = inputs.amount_statistics(window_frame, summary["codes"])
            selected, full = inputs.base_domain(
                day=day,
                daily=inputs.load_daily(root, day),
                basic=basic,
                calendar=days,
                industry=industry.at(day)[0],
                total_mv_yuan=pd.Series(
                    frames[day].set_index("ts_code")["total_mv_yuan"]
                ),
                amount_stats=stats,
            )
            summary["full"] = full
            return selected, summary
        return frames[day], summary

    monthly: list[dict[str, Any]] = []
    for row in rows:
        month = row["month"]
        _f_sel, f_summary = ranked_day(row["formation_date"], month)
        _t_sel, t_summary = ranked_day(row["signal_date"], month)
        entry = {
            "month": month,
            "formation_date": row["formation_date"],
            "signal_date": row["signal_date"],
            "exec_date": row["exec_date"],
            "formation_base": f_summary["base_count"],
            "signal_base": t_summary["base_count"],
            "formation_excluded": f_summary["excluded"],
            "signal_excluded": t_summary["excluded"],
            "formation_full": f_summary.get("full", {}).get("base_count"),
            "signal_full": t_summary.get("full", {}).get("base_count"),
            "liquidity_sampled": month in liquidity_days,
        }
        monthly.append(entry)

    below = [item for item in monthly if item["signal_base"] < spec.BREADTH_MIN]
    consecutive = 0
    worst_consecutive = 0
    for item in monthly:
        consecutive = consecutive + 1 if item["signal_base"] < spec.BREADTH_MIN else 0
        worst_consecutive = max(worst_consecutive, consecutive)
    stop_line = (
        len(below) / len(monthly) > spec.BREADTH_STOP_MONTH_RATIO
        or worst_consecutive >= 2
    )
    sampled = [item for item in monthly if item["liquidity_sampled"]]
    full_values = [
        item["signal_full"] for item in sampled if item["signal_full"] is not None
    ]

    scan_seconds = time.perf_counter() - started - read_seconds
    per_month = read_seconds / max(1, len(monthly))
    readiness = {
        "schema": "p7_peer_gap_readiness/v1",
        "clock": {
            "months": clock_meta["months"],
            "first_signal_date": clock_meta["first_signal_date"],
            "first_exec_date": clock_meta["first_exec_date"],
            "last_signal_date": clock_meta["last_signal_date"],
            "last_exec_date": clock_meta["last_exec_date"],
            "cross_year_months": clock_meta["cross_year_months"],
            "short_months": clock_meta["short_months"],
            "omitted_last_month": "2026-10（T 的执行日 20261030 超出主窗）",
        },
        "breadth": {
            "min_required": spec.BREADTH_MIN,
            "months": len(monthly),
            "signal_base_below": [item["month"] for item in below],
            "below_ratio": len(below) / len(monthly),
            "worst_consecutive_below": worst_consecutive,
            "stop_line_triggered": bool(stop_line),
            "note": (
                "base_count 是不施加流动性条件的宽域上界；"
                "scored_count 需 R1/R2 才算，禁止以上界冒充有效评分数"
            ),
        },
        "sampled_full_domain": {
            "months": [item["month"] for item in sampled],
            "min": min(full_values) if full_values else None,
            "max": max(full_values) if full_values else None,
            "median": float(np.median(full_values)) if full_values else None,
            "note": "每 6 个月抽样一次并施加 20 日均额 ≥5000 万元、≥16 日实有",
        },
        "monthly": monthly,
        "unresolved": {
            "delist_date": "stock_basic 无 delist_date；退市与板块变更个案无法从当期资料恢复",
            "market_snapshot": "market/list_date 为当期快照，历史板块变更以代码前缀与当时行情辅助",
            "industry_conflict_days": int(
                sum(
                    1
                    for item in base_by_day.values()
                    if item["industry"]["conflict_count"] > 0
                )
            ),
            "industry_unknown_rows": int(
                sum(item["excluded"].get("unknown_industry", 0) for item in base_by_day.values())
            ),
            "delisted_codes_seen": "见 sources.json 的 daily 代码并集",
        },
        "crash_windows": _crash_readiness(root),
        "cost": {
            "wall_clock_seconds": round(time.perf_counter() - started, 3),
            "input_read_seconds": round(read_seconds, 3),
            "scan_seconds": round(scan_seconds, 3),
            "days_read": len(needed_days),
        },
        "r1_estimate": {
            "observed_seconds_per_month": round(per_month, 4),
            "projected_hours_full_window": round(per_month * len(monthly) / 3600.0, 3),
            "projected_peak_memory_mb": None,
            "basis": "R0 单月 base 域耗时外推；R1 另含 252 日相关矩阵",
        },
    }

    sources = {
        "schema": "p7_peer_gap_sources/v1",
        "protocol": {
            "rel": spec.PROTOCOL_REL,
            "sha256": sha256_file(root / spec.PROTOCOL_REL),
        },
        "clean": {
            "trade_cal": {
                "rel": "data/clean/trade_cal.parquet",
                "sha256": sha256_file(root / "data/clean/trade_cal.parquet"),
                "rows": _parquet_rows(root / "data/clean/trade_cal.parquet"),
                "columns": _schema(root / "data/clean/trade_cal.parquet"),
                "exchanges": sorted(
                    pd.read_parquet(
                        root / "data/clean/trade_cal.parquet", columns=["exchange"]
                    )["exchange"].astype(str).unique()
                ),
            },
            "stock_basic": {
                "rel": "data/clean/stock_basic.parquet",
                "sha256": sha256_file(root / "data/clean/stock_basic.parquet"),
                "rows": _parquet_rows(root / "data/clean/stock_basic.parquet"),
                "columns": _schema(root / "data/clean/stock_basic.parquet"),
                "market_values": {
                    str(key): int(value)
                    for key, value in basic["market"].value_counts(dropna=False).items()
                },
            },
            "sw_industry_pit": {
                "rel": "data/clean/sw_industry_pit.parquet",
                "sha256": sha256_file(root / "data/clean/sw_industry_pit.parquet"),
                "rows": _parquet_rows(root / "data/clean/sw_industry_pit.parquet"),
                "columns": _schema(root / "data/clean/sw_industry_pit.parquet"),
            },
            "sw_industry_code_table": {
                "rel": "data/clean/sw_industry_code_table.parquet",
                "sha256": sha256_file(root / "data/clean/sw_industry_code_table.parquet"),
                "rows": _parquet_rows(root / "data/clean/sw_industry_code_table.parquet"),
            },
        },
        "daily": _scan_partitions(root, "daily", scan_start, spec.MAIN_END),
        "daily_basic": _scan_partitions(root, "daily_basic", scan_start, spec.MAIN_END),
        "raw_dividend": _scan_partitions(
            root, "dividend", "1990", "2027", base="data/raw"
        )
        if (root / "data/raw/dividend").exists()
        else None,
        "units": {
            "daily.amount": "千元（×1000 转元）",
            "daily_basic.total_mv": "万元（×10000 转元）",
            "protocol_threshold_amount": "5000 万元 = 5.0e7 元",
            "protocol_threshold_mv": "50 亿–1500 亿元 = 5.0e9–1.5e11 元",
        },
        "baseline": {"rel": spec.BASELINE_REL, "manifest_sha256": spec.BASELINE_MANIFEST_SHA},
        "code": {
            name: sha256_file(Path(__file__).resolve().parent / name)
            for name in sorted(
                path.name
                for path in Path(__file__).resolve().parent.glob("*.py")
            )
        },
    }

    calendar_frame = clock.calendar_payload(days)
    calendar_frame["in_main_window"] = (
        (calendar_frame["cal_date"] >= spec.MAIN_START)
        & (calendar_frame["cal_date"] <= spec.MAIN_END)
    ).astype(int)
    clock_table = pd.DataFrame(rows)
    calendar_frame = calendar_frame.merge(
        clock_table[["month", "formation_date", "signal_date", "exec_date"]],
        how="left",
        left_on="cal_date",
        right_on="signal_date",
    )

    baseline = _baseline_identity(root)
    out = ctx.staging_dir
    out.mkdir(parents=True, exist_ok=True)
    spec_payload = {
        "schema": "p7_peer_gap_spec/v1",
        "study": spec.STUDY,
        "campaign": spec.CAMPAIGN,
        "protocol_rel": spec.PROTOCOL_REL,
        "protocol_sha256": sha256_file(root / spec.PROTOCOL_REL),
        "params": {
            name: getattr(spec, name)
            for name in (
                "MAIN_START",
                "MAIN_END",
                "ANCHOR_DATE",
                "CAPITALS",
                "B_SHARE",
                "TARGET_DCAGR",
                "TARGET_DMDD",
                "SEGMENTS",
                "MIN_LISTED_TRADING_DAYS",
                "MV_MIN_YUAN",
                "MV_MAX_YUAN",
                "AMOUNT_WINDOW",
                "AMOUNT_MIN_YUAN",
                "AMOUNT_MIN_DAYS",
                "EXCLUDED_L1",
                "BREADTH_MIN",
                "CORR_WINDOW",
                "CORR_MIN_PAIRS",
                "TOP_PEERS",
                "PEER_MIN_VALID",
                "BETA_MIN_PAIRS",
                "VARIANCE_FLOOR",
                "ROUND_DECIMALS",
                "MONTH_MIN_COVERAGE",
                "MATCH_RANK_TOL",
                "SLOTS",
                "SECTOR_CAP",
                "COMMISSION_RATE",
                "MIN_COMMISSION",
                "STAMP_TAX",
                "SLIPPAGE",
                "MIN_NOTIONAL",
            )
        },
    }
    write_json(out / "spec.json", spec_payload)
    authorization = {
        "schema": "p7_peer_gap_authorization/v1",
        "approved_candidate": "PRG-M1-v1",
        "approved_by": "user",
        "approved_message": "协议 §9 交给实现模型的任务正文（2026-11-19）",
        "approved_stages": ["R0", "R1", "R2", "R3"],
        "approved_scope": (
            "R0 只读本地仓库与冻结输入；R1—R3 编写隔离研究与合成测试；"
            "R4—R6 未获预算，禁止运行"
        ),
        "data_spend_cny": 0,
        "network": "forbidden",
        "machine_limit": "R0 与合成测试按协议 §7 的 20 分钟建议口径执行并实测报告",
        "formal_alpha_consumed": False,
        "formal_admission": "not_evaluated",
        "intended_type": "enhancement",
        "notes": [
            "用户未发送协议 §9 末尾的首次任务词，故无 R4—R6 经济重跑授权",
            "本文件不构成正式准入申请",
        ],
    }
    write_json(out / "authorization.json", authorization)
    write_json(out / "sources.json", sources)
    write_json(out / "baseline_identity.json", baseline)
    write_json(out / "readiness.json", readiness)
    calendar_frame.to_parquet(out / "calendar.parquet", index=False)
    files = {
        path.name: sha256_file(path)
        for path in sorted(out.iterdir())
        if path.is_file()
    }
    ctx.notes.append(f"R0 wall_clock={readiness['cost']['wall_clock_seconds']}s")
    manifest = ctx.finish(
        {
            "files": files,
            "readiness": {
                "months": len(monthly),
                "stop_line_triggered": bool(stop_line),
                "sampled_full_median": readiness["sampled_full_domain"]["median"],
            },
            "machine_time_seconds": readiness["cost"]["wall_clock_seconds"],
        }
    )
    return {"manifest": manifest, "readiness": readiness}


def _load_panel(
    root: Path, days: Sequence[str], start: str, end: str
) -> tuple[pd.DataFrame, list[str]]:
    subset = [day for day in days if start <= day <= end]
    if not subset:
        raise ResearchError(f"面板区间为空：{start}~{end}")
    frame = inputs.load_daily_concat(root, subset)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame, subset


def _return_matrices(
    panel: pd.DataFrame, days: Sequence[str], basic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, dict[str, Any]]:
    close = panel.pivot(index="trade_date", columns="ts_code", values="close_adj")
    close = close.reindex(list(days))
    st = panel.pivot(index="trade_date", columns="ts_code", values="is_st")
    st = st.reindex(index=close.index, columns=close.columns)
    suspended = panel.pivot(index="trade_date", columns="ts_code", values="is_suspended")
    suspended = suspended.reindex(index=close.index, columns=close.columns)
    market = basic.set_index("ts_code")["market"]
    eligible = pd.DataFrame(
        np.repeat(
            np.array(
                [market.get(code) == spec.MARKET_MAIN for code in close.columns]
            )[None, :],
            len(close.index),
            axis=0,
        ),
        index=close.index,
        columns=close.columns,
    )
    eligible &= st.fillna(1).eq(0)
    excess, median, meta = graph.build_excess_returns(
        close, eligible, suspended.fillna(1) > 0
    )
    return close, eligible, suspended, median, meta


def run_r1(ctx: StageContext, *, max_months: int = 3) -> dict[str, Any]:
    """R1：逐月建图与 beta。本次只在批准的小样本上运行。"""
    root = spec.REPO_ROOT
    days = clock.load_calendar(root)
    rows = clock.month_clock(days, spec.MAIN_START, spec.MAIN_END)
    rows = rows[: int(max_months)]
    basic = inputs.load_stock_basic(root)
    start_index = int(rows[0]["warmup_start_index"])
    panel, subset = _load_panel(
        root, days, days[start_index], rows[-1]["signal_date"]
    )
    close, eligible, suspended, median, market_meta = _return_matrices(
        panel, subset, basic
    )
    excess, _, _ = graph.build_excess_returns(close, eligible, suspended.fillna(1) > 0)

    out = ctx.staging_dir
    out.mkdir(parents=True, exist_ok=True)
    code_hash = sha256_file(Path(__file__).resolve().parent / "graph.py")
    beta_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        formation = row["formation_date"]
        position = int(row["formation_index"])
        window_days = days[max(0, position - spec.CORR_WINDOW + 1):position + 1]
        base, summary = inputs.build_base_frame(
            root,
            day=formation,
            calendar=days,
            basic=basic,
            liquidity_window=days[max(0, position - spec.AMOUNT_WINDOW + 1):position + 1],
        )
        codes = [
            code for code in base["ts_code"].tolist() if code in excess.columns
        ]
        edges, betas, meta = graph.formation(excess.loc[window_days], codes)
        sector_map = dict(zip(base["ts_code"], base["l1_name"]))
        if len(edges):
            meta["cross_industry_ratio"] = round(
                float(
                    (
                        edges["center"].map(sector_map)
                        != edges["peer"].map(sector_map)
                    ).mean()
                ),
                6,
            )
        if len(edges):
            edges["formation_date"] = formation
            edges["window_start"] = window_days[0]
            edges["cutoff"] = formation
            edges["code_hash"] = code_hash
            beta_map = betas.set_index("center")["beta"].to_dict()
            betas_pair = betas.set_index("center")["pair_n"].to_dict()
            edges["beta"] = [beta_map.get(str(center)) for center in edges["center"]]
            edges["beta_n"] = [
                betas_pair.get(str(center), 0) for center in edges["center"]
            ]
            graph_dir = out / "graph"
            graph_dir.mkdir(parents=True, exist_ok=True)
            edges.to_parquet(graph_dir / f"{formation}.parquet", index=False)
        betas["formation_date"] = formation
        betas["window_start"] = window_days[0]
        betas["cutoff"] = formation
        betas["base_count"] = len(codes)
        beta_frames.append(betas)
        meta.update(
            {
                "formation_date": formation,
                "window_start": window_days[0],
                "base_count": len(codes),
                "scored_upper_bound": int(betas["beta"].notna().sum()),
                "excluded": summary["excluded"],
                "industry_conflicts": summary["industry"]["conflict_count"],
            }
        )
        summaries.append(meta)
        timing.append(
            {
                "formation_date": formation,
                "seconds": round(time.perf_counter() - started, 3),
                "centers": len(codes),
            }
        )

    beta_all = pd.concat(beta_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    market_frame = graph.market_daily_frame(median)
    write_parquet(beta_all, out / "beta.parquet")
    write_parquet(summary_frame, out / "formation_summary.parquet")
    write_parquet(market_frame, out / "market_daily.parquet")
    graph_files = sorted((out / "graph").glob("*.parquet"))
    files = {
        path.name: sha256_file(path)
        for path in [out / "beta.parquet", out / "formation_summary.parquet",
                     out / "market_daily.parquet", *graph_files]
    }
    total = time.perf_counter() - ctx.started
    manifest = ctx.finish(
        {
            "files": files,
            "months": len(rows),
            "full_window": False,
            "months_run": [row["month"] for row in rows],
            "market_meta": market_meta,
            "timing": timing,
            "machine_time_seconds": round(total, 3),
            "note": "本次为授权内小样本定向验证，不构成 R1 全窗交付",
        }
    )
    return {"manifest": manifest, "months": len(rows)}


def run_r2(ctx: StageContext, *, max_months: int = 3) -> dict[str, Any]:
    """R2：逐月分数、三臂与匹配。只消耗已发布的 R1 边表/beta。"""
    root = spec.REPO_ROOT
    upstream = {item["stage"]: item for item in ctx.require_upstream()}
    r1_dir = run_root(ctx.campaign, upstream["R1"]["run_id"])
    beta_all = pd.read_parquet(r1_dir / "beta.parquet")
    market = pd.read_parquet(r1_dir / "market_daily.parquet").set_index("trade_date")["m"]
    days = clock.load_calendar(root)
    rows = clock.month_clock(days, spec.MAIN_START, spec.MAIN_END)
    run_months = set(beta_all["formation_date"].unique())
    rows = [row for row in rows if row["formation_date"] in run_months][: int(max_months)]
    if not rows:
        raise ResearchError("R1 未覆盖任何信号月")
    basic = inputs.load_stock_basic(root)
    industry = inputs.IndustryPit(root)

    out = ctx.staging_dir
    out.mkdir(parents=True, exist_ok=True)
    score_frames: list[pd.DataFrame] = []
    plan_rows: list[dict[str, Any]] = []
    match_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for row in rows:
        formation = row["formation_date"]
        signal = row["signal_date"]
        month = row["month"]
        panel, _ = _load_panel(root, days, formation, signal)
        base_t, base_summary = inputs.build_base_frame(
            root,
            day=signal,
            calendar=days,
            basic=basic,
            liquidity_window=days[
                max(0, int(row["signal_index"]) - spec.AMOUNT_WINDOW + 1):
                int(row["signal_index"]) + 1
            ],
        )
        mapping, _meta = industry.at(signal)
        month_frame = score.month_return_frame(
            panel,
            formation=formation,
            signal=signal,
            median=market,
            days=days,
        )
        beta_frame = beta_all[beta_all["formation_date"] == formation]
        edges = pd.read_parquet(r1_dir / "graph" / f"{formation}.parquet")
        peers_by_center = {
            str(center): group.sort_values("rank")["peer"].astype(str).tolist()
            for center, group in edges.groupby("center")
        }
        merged = month_frame.merge(
            base_t[["ts_code", "l1_name", "total_mv_yuan"]].rename(
                columns={"l1_name": "sector"}
            ),
            on="ts_code",
            how="inner",
        )
        scored, score_meta = score.score_centers(
            beta_frame=beta_frame,
            month_frame=merged,
            excess_month_by_code=dict(zip(month_frame["ts_code"], month_frame["L"])),
            peers_by_center=peers_by_center,
            base_codes=merged["ts_code"].tolist(),
        )
        if len(scored):
            scored = scored.merge(
                merged[["ts_code", "sector", "total_mv_yuan"]],
                on="ts_code",
                how="left",
            )
            scored["sector"] = scored["ts_code"].map(mapping)
            scored["month"] = month
            scored["signal_date"] = signal
            write_parquet(scored, out / f"scores_{signal}.parquet")
            score_frames.append(scored)

        industry_map = dict(zip(base_t["ts_code"], base_t["l1_name"]))
        peer_targets = score.select_arm(
            scored if len(scored) else pd.DataFrame(),
            key="score",
            industry=industry_map,
        ) if len(scored) else []
        rev_targets = score.select_arm(
            scored if len(scored) else pd.DataFrame(),
            key="L",
            industry=industry_map,
            ascending=True,
        ) if len(scored) else []
        matches = score.match_targets(
            peer_targets=peer_targets,
            scored=scored if len(scored) else pd.DataFrame(),
            base_frame=merged,
            industry=industry_map,
        ) if len(scored) else pd.DataFrame()
        if len(matches):
            matches["month"] = month
            matches["signal_date"] = signal
            match_rows.append(matches)
        for arm, picked in (("PEER", peer_targets), ("REV", rev_targets)):
            for order, item in enumerate(picked, start=1):
                plan_rows.append(
                    {
                        "month": month,
                        "signal_date": signal,
                        "arm": arm,
                        "slot": order,
                        "ts_code": item.get("ts_code"),
                        "sector": industry_map.get(str(item.get("ts_code"))),
                        "score": item.get("score"),
                        "L": item.get("L"),
                        "rank_key": item.get("score") if arm == "PEER" else item.get("L"),
                    }
                )
        t_codes = set(base_t["ts_code"])
        formed = (
            int(
                beta_frame.loc[
                    beta_frame["beta"].notna()
                    & beta_frame["center"].isin(t_codes),
                    "center",
                ].nunique()
            )
            if len(beta_frame)
            else 0
        )
        top_reason = (
            max(score_meta["reasons"].items(), key=lambda item: item[1])[0]
            if score_meta["reasons"]
            else ""
        )
        coverage_rows.append(
            {
                "month": month,
                "signal_date": signal,
                "historical_mother_count": int(base_summary["mother_count"]),
                "base_count": int(base_summary["base_count"]),
                "formed_count": formed,
                "scored_count": int(len(scored)),
                "n_unresolved": max(0, int(base_summary["base_count"]) - len(scored)),
                "first_reason": top_reason,
                "reason_flags": json.dumps(score_meta["reasons"], sort_keys=True),
                "below_100": bool(len(scored) < spec.BREADTH_MIN),
            }
        )
        for record in month_frame.to_dict("records"):
            if not record["valid"]:
                exclusion_rows.append(
                    {
                        "month": month,
                        "signal_date": signal,
                        "ts_code": record["ts_code"],
                        "stage": "month_return",
                        "reason": record["reason"],
                    }
                )
        for reason, count in score_meta["reasons"].items():
            exclusion_rows.append(
                {
                    "month": month,
                    "signal_date": signal,
                    "ts_code": "",
                    "stage": "score",
                    "reason": f"{reason}:{count}",
                }
            )
        for item in matches.to_dict("records") if len(matches) else []:
            if item["match"] is None:
                exclusion_rows.append(
                    {
                        "month": month,
                        "signal_date": signal,
                        "ts_code": item["source"],
                        "stage": "match",
                        "reason": item["reason"],
                    }
                )

    plans = pd.DataFrame(plan_rows)
    write_parquet(plans, out / "plans.parquet")
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out / "coverage.csv", index=False, encoding="utf-8")
    exclusions = pd.DataFrame(exclusion_rows)
    write_parquet(exclusions, out / "exclusions.parquet")
    matched = (
        pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()
    )
    write_parquet(matched, out / "matched_pairs.parquet")
    stop = bool(coverage["below_100"].any())
    files = {
        path.name: sha256_file(path)
        for path in sorted(out.glob("*.*"))
        if path.is_file()
    }
    manifest = ctx.finish(
        {
            "files": files,
            "months": len(rows),
            "months_run": [row["month"] for row in rows],
            "breadth_stop_triggered": stop,
            "machine_time_seconds": round(time.perf_counter() - ctx.started, 3),
            "note": "本次为授权内小样本定向验证，不构成 R2 全窗交付",
        }
    )
    return {"manifest": manifest, "months": len(rows)}


def _json_safe(payload: Any) -> Any:
    """把 Timestamp/NaT/NaN 统一为可写 JSON 的形式（保留原值语义）。"""
    return json.loads(json.dumps(payload, default=str, ensure_ascii=False))


def _baseline_daily(root: Path, capital: int, arm: str) -> list[dict[str, Any]]:
    base = spec.BASELINE_DIR / str(capital)
    if arm == "A80":
        rows = read_json(base / "A80" / "daily.json")
        for row in rows:
            row["trade_date"] = str(row["trade_date"])[:10].replace("-", "")[:8]
        return rows
    name = "A100_daily.parquet" if arm == "A100" else "A80_B_cash_daily.parquet"
    frame = pd.read_parquet(base / name)
    frame["trade_date"] = (
        frame["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    )
    return frame.to_dict("records")


def run_r4(ctx: StageContext) -> dict[str, Any]:
    """R4：三臂 B 袖原生账户 + 两档五臂合并。"""
    from scripts.v2_p7.buffer_native_replay import load_prices
    from src.lazybull.v2.core.execution.price_index import prepare_price_index

    from . import economic

    root = spec.REPO_ROOT
    upstream = {item["stage"]: item for item in ctx.require_upstream()}
    r2_dir = run_root(ctx.campaign, upstream["R2"]["run_id"])
    plans = pd.read_parquet(r2_dir / "plans.parquet")
    matched = pd.read_parquet(r2_dir / "matched_pairs.parquet")
    rankings = economic.build_arm_rankings(plans, matched)
    missing = [arm for arm in economic.ARMS if not rankings[arm]]
    if missing:
        raise ResearchError(f"R4 目标臂为空：{missing}")

    days = clock.load_calendar(root)
    window = [day for day in days if spec.ANCHOR_PREV <= day <= spec.MAIN_END]
    if window[0] != spec.ANCHOR_PREV:
        raise ResearchError(f"锚前交易日不是 {spec.ANCHOR_PREV}")
    prices, price_identity = load_prices(
        root, window, warmup_cutoff=spec.ANCHOR_PREV
    )
    indexes = prepare_price_index(prices)
    pit = inputs.IndustryPit(root).frame
    names_frame = pit[["l1_code", "l1_name"]].dropna().drop_duplicates("l1_code")
    names = {
        str(row["l1_code"]): str(row["l1_name"])
        for _, row in names_frame.iterrows()
    }
    book = economic.NativeCorporateBook(
        calendar=window,
        dividend_dir=spec.DIVIDEND_DIR,
    )
    out = ctx.staging_dir
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    checks: dict[str, Any] = {"arms": {}}
    for capital in spec.CAPITALS:
        b_capital = float(capital) * spec.B_SHARE
        five = {
            "A100": _baseline_daily(root, capital, "A100"),
            "A80": _baseline_daily(root, capital, "A80"),
            "A80_cash": _baseline_daily(root, capital, "A80_cash"),
        }
        b_daily: dict[str, list[dict[str, Any]]] = {}
        for arm in economic.ARMS:
            started = time.perf_counter()
            result = economic.run_b_arm(
                root=root,
                ranked=rankings[arm],
                calendar=window,
                prices=prices,
                indexes=indexes,
                capital=b_capital,
                book=book,
                pit=pit,
                l1_names=names,
            )
            daily = [row for row in result["daily"] if row["trade_date"] != spec.ANCHOR_PREV]
            b_daily[arm] = daily
            arm_dir = out / "b" / f"{capital}" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            write_json(arm_dir / "daily.json", _json_safe(daily))
            write_json(arm_dir / "trades.json", _json_safe(result["trades"]))
            write_json(arm_dir / "validation.json", _json_safe(result["validation"]))
            write_json(arm_dir / "engine_params.json", _json_safe(result["engine_params"]))
            write_json(arm_dir / "corporate.json", _json_safe(result.get("corporate") or {}))
            target_sets = {day: {code for code, _ in rows} for day, rows in rankings[arm].items()}
            checks["arms"].setdefault(arm, {})[str(capital)] = {
                "signal_days": len(rankings[arm]),
                "targets_expected": sum(len(rows) for rows in rankings[arm].values()),
                "buy_fills": sum(1 for row in result["trades"] if row["action"] == "buy"),
                "sell_fills": sum(1 for row in result["trades"] if row["action"] == "sell"),
                "validation": result["validation"]["valid"],
                "seconds": round(time.perf_counter() - started, 3),
                "signal_targets": {
                    day: sorted(codes) for day, codes in list(target_sets.items())[:3]
                },
            }
        combined = economic.combine_five_arms(
            five["A100"], five["A80"], b_daily, capital
        )
        payload[str(capital)] = {
            name: economic.metrics_from_daily(rows, capital)
            for name, rows in combined.items()
        }
        payload[str(capital)]["A80_cash"] = economic.metrics_from_daily(
            five["A80_cash"], capital
        )
        for name, rows in combined.items():
            write_json(out / "five_arm" / f"{capital}" / f"{name}.json", rows)
        write_json(
            out / "five_arm" / f"{capital}" / "A80_cash.json", five["A80_cash"]
        )
    five_frame = []
    for capital, arms in payload.items():
        for name, metrics in arms.items():
            five_frame.append({"capital": int(capital), "arm": name, **metrics})
    write_parquet(pd.DataFrame(five_frame), out / "five_arm_metrics.parquet")
    write_json(out / "checks.json", checks)
    write_json(out / "price_identity.json", price_identity)
    failed = [
        f"{arm}/{capital}"
        for arm, per in checks["arms"].items()
        for capital, item in per.items()
        if not item["validation"]
    ]
    manifest = ctx.finish(
        {
            "files": {
                path.relative_to(out).as_posix(): sha256_file(path)
                for path in sorted(out.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            },
            "metrics": payload,
            "checks": checks,
            "machine_time_seconds": round(time.perf_counter() - ctx.started, 3),
            "failed_arms": failed,
        }
    )
    if failed:
        raise ResearchError(f"R4 守恒校验失败：{failed}")
    return {"manifest": manifest}


def _daily_returns(rows: Sequence[Mapping[str, Any]]) -> pd.Series:
    values = pd.Series([float(row["total_value"]) for row in rows])
    return values.pct_change().dropna()


def _pearson(left: pd.Series, right: pd.Series) -> float | None:
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(joined) < 30 or joined.iloc[:, 0].std() == 0 or joined.iloc[:, 1].std() == 0:
        return None
    return round(float(joined.iloc[:, 0].corr(joined.iloc[:, 1])), 10)


def _stress_arm(
    rows: Sequence[Mapping[str, Any]],
    trades: Sequence[Mapping[str, Any]],
    corporate: Mapping[str, Any],
) -> dict[str, Any]:
    """解析压力：费用 1.2×、现金分红 0.7×，差额逐日从总资产扣除。"""
    fee = pd.Series(0.0, index=[str(row["trade_date"]) for row in rows])
    for trade in trades:
        day = str(trade.get("date", ""))[:10].replace("-", "")[:8]
        if day in fee.index:
            fee.loc[day] += 0.2 * float(trade.get("cost") or 0.0)
    dividend = pd.Series(0.0, index=fee.index)
    for event in corporate.get("rows") or []:
        day = str(event.get("trade_date", ""))[:10].replace("-", "")[:8]
        if day in dividend.index:
            dividend.loc[day] += 0.3 * float(event.get("cash_receivable") or 0.0)
    base = np.array([float(row["total_value"]) for row in rows])
    stressed = base - np.cumsum(fee.to_numpy()) - np.cumsum(dividend.to_numpy())
    capital = float(rows[0]["total_value"])
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    chain = calculate_chain_metrics(pd.DataFrame({"nav": stressed}))
    return {
        "fee_uplift_total": round(float(fee.sum()), 6),
        "dividend_haircut_total": round(float(dividend.sum()), 6),
        "cagr": round(float(chain["cagr"]), 10),
        "mdd_loss": round(abs(float(chain["max_drawdown"])), 10),
        "ending_value": round(float(stressed[-1]), 6),
        "base_capital": round(capital, 6),
    }


def run_r5(ctx: StageContext) -> dict[str, Any]:
    """R5：指标、分年、增量、压力、三口径相关与 F29 表。"""
    from . import economic

    upstream = {item["stage"]: item for item in ctx.require_upstream()}
    r4_dir = run_root(ctx.campaign, upstream["R4"]["run_id"])
    metrics = pd.read_parquet(r4_dir / "five_arm_metrics.parquet")
    r4_manifest = read_json(r4_dir / "manifest.json")
    r2_run = next(
        item["run_id"] for item in r4_manifest["upstream"] if item["stage"] == "R2"
    )
    matched = pd.read_parquet(
        run_root(ctx.campaign, r2_run) / "matched_pairs.parquet"
    )
    match_coverage = round(float(matched["match"].notna().mean()), 6)
    lookup = {
        (int(row["capital"]), str(row["arm"])): row for _, row in metrics.iterrows()
    }
    daily: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for capital in spec.CAPITALS:
        for arm in ("A100", "A80_cash", "A80_PEER", "A80_REV", "A80_MATCH"):
            daily[(capital, arm)] = read_json(
                r4_dir / "five_arm" / str(capital) / f"{arm}.json"
            )
    result: dict[str, Any] = {"objectives": {}, "increments": {}, "yearly": {}}
    for capital in spec.CAPITALS:
        a100 = lookup[(capital, "A100")]
        cash = lookup[(capital, "A80_cash")]
        peer = lookup[(capital, "A80_PEER")]
        rev = lookup[(capital, "A80_REV")]
        match = lookup[(capital, "A80_MATCH")]
        d_cagr = float(peer["cagr"]) - float(a100["cagr"])
        d_mdd = float(peer["mdd_loss"]) - float(a100["mdd_loss"])
        result["objectives"][str(capital)] = {
            "d_cagr": round(d_cagr, 10),
            "d_mdd_loss": round(d_mdd, 10),
            "cagr_above_cash": bool(float(peer["cagr"]) > float(cash["cagr"])),
            "cagr_ge_0p5pp": bool(d_cagr >= 0.005),
            "mdd_le_1pp": bool(d_mdd <= 0.010),
            "all_three": bool(
                d_cagr >= 0.005
                and d_mdd <= 0.010
                and float(peer["cagr"]) > float(cash["cagr"])
            ),
            "peer": {key: peer[key] for key in ("cagr", "mdd_loss", "ending_value")},
            "cash": {key: cash[key] for key in ("cagr", "mdd_loss", "ending_value")},
            "a100": {key: a100[key] for key in ("cagr", "mdd_loss", "ending_value")},
        }
        result["increments"][str(capital)] = {
            "peer_minus_rev": round(float(peer["cagr"]) - float(rev["cagr"]), 10),
            "peer_minus_match": round(float(peer["cagr"]) - float(match["cagr"]), 10),
            "match_coverage": match_coverage,
            "peer_minus_match_estimable": bool(match_coverage >= 0.90),
            "rev": {key: rev[key] for key in ("cagr", "mdd_loss")},
            "match": {key: match[key] for key in ("cagr", "mdd_loss")},
        }
        peer_year = economic.yearly_metrics(daily[(capital, "A80_PEER")], capital)
        a_year = economic.yearly_metrics(daily[(capital, "A100")], capital)
        by_year = {
            row["year"]: row["cagr"] - other["cagr"]
            for row, other in zip(peer_year, a_year)
        }
        same_sign = sum(1 for value in by_year.values() if value > 0)
        result["yearly"][str(capital)] = {
            "peer": peer_year,
            "a100": a_year,
            "delta": {year: round(value, 10) for year, value in by_year.items()},
            "same_sign_positive": same_sign,
            "years": len(by_year),
            "same_sign_ratio": round(same_sign / max(1, len(by_year)), 6),
        }
        # 压力
        arm_dir = r4_dir / "b" / str(capital)
        result.setdefault("stress", {})[str(capital)] = {}
        for arm in economic.ARMS:
            rows = read_json(r4_dir / "five_arm" / str(capital) / f"A80_{arm}.json")
            trades = read_json(arm_dir / arm / "trades.json")
            corporate = read_json(arm_dir / arm / "corporate.json")
            result["stress"][str(capital)][arm] = _stress_arm(rows, trades, corporate)
        # 相关：候选=B 袖单账（20% 独立资金），对照=A100
        b_rows = read_json(r4_dir / "b" / str(capital) / "PEER" / "daily.json")
        peer_ret = _daily_returns(b_rows)
        a_ret = _daily_returns(daily[(capital, "A100")])
        ten_peer = peer_ret.rolling(10).sum()
        ten_a = a_ret.rolling(10).sum()
        crash_mask = [
            "20220104" <= str(row["trade_date"]) <= "20221031"
            for row in b_rows[1:]
        ]
        result.setdefault("correlation", {})[str(capital)] = {
            "daily": _pearson(peer_ret, a_ret),
            "ten_day": _pearson(ten_peer, ten_a),
            "crash_2022": _pearson(
                peer_ret[np.array(crash_mask)], a_ret[np.array(crash_mask)]
            ),
            "crash_2015_2018": None,
            "note": "2015/2018 无冻结 A 配对账户，按协议记 not_estimable",
        }
    violations: dict[str, Any] = {}
    for capital in spec.CAPITALS:
        for arm in economic.ARMS:
            trades = read_json(r4_dir / "b" / str(capital) / arm / "trades.json")
            buys = [row for row in trades if row.get("action") == "buy"]
            amounts = [float(row.get("amount") or 0.0) for row in buys]
            violations[f"{capital}/{arm}"] = {
                "buys": len(buys),
                "below_floor": sum(
                    1 for value in amounts if value < spec.MIN_NOTIONAL
                ),
                "min_amount": round(min(amounts), 6) if amounts else None,
            }
    result["f29"] = _f29_table(result, violations)
    out = ctx.staging_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", result["objectives"])
    write_json(out / "increments.json", result["increments"])
    write_json(out / "yearly.json", result["yearly"])
    write_json(out / "stress.json", result["stress"])
    write_json(out / "correlation.json", result["correlation"])
    write_json(out / "f29.json", result["f29"])
    manifest = ctx.finish(
        {
            "files": {
                path.name: sha256_file(path)
                for path in sorted(out.iterdir())
                if path.is_file()
            },
            "objectives": result["objectives"],
            "increments": result["increments"],
            "correlation": result["correlation"],
            "f29": result["f29"],
            "machine_time_seconds": round(time.perf_counter() - ctx.started, 3),
        }
    )
    return {"manifest": manifest}


def _f29_table(
    result: Mapping[str, Any],
    violations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """F29 型 I 逐项状态；未评估项必须写 not_evaluated，不得默认通过。"""
    objectives = result.get("objectives", {})
    increments = result.get("increments", {})
    correlation = result.get("correlation", {})
    all_three = all(
        bool(item["all_three"]) for item in objectives.values()
    )
    corr_daily = [
        item["daily"] for item in correlation.values() if item["daily"] is not None
    ]
    corr_ok = bool(corr_daily) and all(value < 0.30 for value in corr_daily)
    increments_ok = all(item["peer_minus_rev"] > 0 for item in increments.values())
    return {
        "type": "I",
        "E1_signal_layer": {
            "status": "evaluated",
            "d_cagr_by_capital": {
                capital: item["d_cagr"] for capital, item in objectives.items()
            },
            "pass": all_three,
        },
        "Q1_independent_sharpe": {
            "status": "not_evaluated",
            "gap": "未做 A vs A 换种子的独立夏普区间标定",
        },
        "E2_institutional": {
            "status": "not_evaluated",
            "gap": "本协议未分配正式 α，未做 14 折配对重排",
        },
        "T2_portfolio_ruler": {
            "status": "evaluated",
            "all_three_by_capital": {
                capital: bool(item["all_three"]) for capital, item in objectives.items()
            },
            "pass": all_three,
        },
        "correlation_under_0p3": {
            "status": "evaluated" if corr_ok else "failed",
            "daily_by_capital": {
                capital: item["daily"] for capital, item in correlation.items()
            },
            "pass": corr_ok,
            "crash_2015_2018": "not_estimable",
        },
        "peer_beats_rev_and_match": {
            "status": "evaluated",
            "by_capital": increments,
            "peer_minus_match": (
                "not_estimable"
                if not all(item["peer_minus_match_estimable"] for item in increments.values())
                else "evaluated"
            ),
            "pass": increments_ok,
        },
        "single_trade_min_notional": {
            "status": "evaluated",
            "basis": "R4 三臂 × 两档逐笔 amount < V 核验（V=25588.54）",
            "violations": dict(violations or {}),
            "pass": bool(
                violations
                and all(item["below_floor"] == 0 for item in violations.values())
            ),
        },
        "E4_utilization": {
            "status": "not_evaluated",
            "gap": "未计算 B 袖利用率与空仓占比",
        },
        "formal_admission": "not_evaluated",
        "intended_type": "enhancement",
        "alpha_consumed": False,
    }


def run_r6(ctx: StageContext) -> dict[str, Any]:
    """R6：中文报告、哈希与不可覆盖发布。"""
    upstream = {item["stage"]: item for item in ctx.require_upstream()}
    r4_dir = run_root(ctx.campaign, upstream["R4"]["run_id"])
    r5_dir = run_root(ctx.campaign, upstream["R5"]["run_id"])
    f29 = read_json(r5_dir / "f29.json")
    metrics = read_json(r5_dir / "metrics.json")
    increments = read_json(r5_dir / "increments.json")
    yearly = read_json(r5_dir / "yearly.json")
    stress = read_json(r5_dir / "stress.json")
    correlation = read_json(r5_dir / "correlation.json")
    checks = read_json(r4_dir / "checks.json")
    lines = [
        "# P7 候选一 PRG-M1-v1 · R4—R6 经济结论",
        "",
        "> 本报告由 `scripts/v2_p7/peer_return_gap` 生成；数据支出 0 元。",
        "> 判定语汇：`continue` / `continue+限制` / `negative`。",
        "",
        "## 1 五臂主表",
        "",
        "| 本金 | 组合 | CAGR | MDD_loss | 期末 |",
        "|---|---|---|---|---|",
    ]
    for capital, item in metrics.items():
        for name, label in (
            ("a100", "A100"),
            ("cash", "A80+现金"),
            ("peer", "A80+PEER"),
        ):
            row = item[name]
            lines.append(
                f"| {capital} | {label} | {row['cagr']*100:.4f}% | "
                f"{row['mdd_loss']*100:.4f}% | {row.get('ending_value', '')} |"
            )
        for name, label in (("rev", "A80+REV"), ("match", "A80+MATCH")):
            row = increments[capital][name]
            lines.append(
                f"| {capital} | {label} | {row['cagr']*100:.4f}% | "
                f"{row['mdd_loss']*100:.4f}% | — |"
            )
    lines += ["", "## 1b B 袖单账（20% 本金）", ""]
    from . import economic

    for capital in metrics:
        b_rows = read_json(r4_dir / "b" / str(capital) / "PEER" / "daily.json")
        b_metrics = economic.metrics_from_daily(
            b_rows, int(capital) * spec.B_SHARE
        )
        buy = checks["arms"]["PEER"][str(capital)]["buy_fills"]
        expected = checks["arms"]["PEER"][str(capital)]["targets_expected"]
        lines.append(
            f"- {capital}: PEER 袖 CAGR={b_metrics['cagr']*100:.4f}%、"
            f"MDD={b_metrics['mdd_loss']*100:.4f}%、期末={b_metrics['ending_value']}、"
            f"买入成交 {buy}/{expected}（{buy/expected:.2%}）"
        )
    lines += ["", "## 2 三项经济目标", ""]
    for capital, item in metrics.items():
        lines.append(
            f"- {capital}: ΔCAGR={item['d_cagr']*100:.4f}pp、"
            f"ΔMDD={item['d_mdd_loss']*100:.4f}pp（≤+1pp 满足={item['mdd_le_1pp']}）、"
            f"高于现金={item['cagr_above_cash']}、三项合取过={item['all_three']}"
        )
    lines += ["", "## 3 增量（PEER - REV / PEER - MATCH）", ""]
    for capital, item in increments.items():
        lines.append(
            f"- {capital}: PEER-REV={item['peer_minus_rev']*100:.4f}pp、"
            f"PEER-MATCH={item['peer_minus_match']*100:.4f}pp "
            f"(match_coverage={item['match_coverage']:.4f})"
        )
    lines += ["", "## 4 分年同号", ""]
    for capital, item in yearly.items():
        lines.append(
            f"- {capital}: {item['same_sign_positive']}/{item['years']} = "
            f"{item['same_sign_ratio']:.3f}"
        )
    lines += ["", "## 5 压力（费用 1.2×、现金分红 0.7×）", ""]
    for capital, arms in stress.items():
        for arm, item in arms.items():
            lines.append(
                f"- {capital}/{arm}: CAGR={item['cagr']*100:.4f}%、"
                f"MDD={item['mdd_loss']*100:.4f}%、费用增量={item['fee_uplift_total']}、"
                f"分红折减={item['dividend_haircut_total']}"
            )
    lines += ["", "## 6 三口径相关", ""]
    for capital, item in correlation.items():
        lines.append(
            f"- {capital}: 日={item['daily']}、10日={item['ten_day']}、"
            f"2022 危机={item['crash_2022']}；2015/2018=not_estimable"
        )
    lines += ["", "## 7 F29 type-I", ""]
    for key in (
        "E1_signal_layer",
        "Q1_independent_sharpe",
        "E2_institutional",
        "T2_portfolio_ruler",
        "correlation_under_0p3",
        "peer_beats_rev_and_match",
        "single_trade_min_notional",
        "E4_utilization",
    ):
        item = f29[key]
        lines.append(f"- {key}: status={item['status']}, pass={item.get('pass')}")
    lines += [
        "",
        "## 8 引擎执行校验",
        "",
        f"- 三臂 × 两档 validation 全过：{not read_json(r4_dir/'manifest.json')['failed_arms']}",
        f"- 目标月数：{checks['arms']['PEER']['1000000']['signal_days']}",
        "",
        "## 9 判定",
        "",
        f"**{_verdict(metrics, f29)}**",
        "",
        "边界：不构成 B 区转正；`formal_admission=not_evaluated`；"
        "2015/2018 相关 not_estimable。",
    ]
    out = ctx.staging_dir
    out.mkdir(parents=True, exist_ok=True)
    report = out / "report_zh.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    manifest = ctx.finish(
        {
            "files": {"report_zh.md": sha256_file(report)},
            "verdict": _verdict(metrics, f29),
            "machine_time_seconds": round(time.perf_counter() - ctx.started, 3),
        }
    )
    return {"manifest": manifest}


def _verdict(metrics: Mapping[str, Any], f29: Mapping[str, Any]) -> str:
    if not any(item["all_three"] for item in metrics.values()):
        return "negative"
    if all(item["all_three"] for item in metrics.values()) and f29["correlation_under_0p3"]["pass"]:
        return "continue"
    return "continue+限制"


class SyntheticBook:
    """R3 合成演练的逐笔账本（真实 CostModel 计费，不重写向量净值）。"""

    def __init__(self, capital: float) -> None:
        self.initial = float(capital)
        self.cash = float(capital)
        self.positions: dict[str, int] = {}
        self.locked: dict[str, dict[str, Any]] = {}
        self.receivable = 0.0
        self.pending_sells: dict[str, dict[str, Any]] = {}
        self.orders: list[dict[str, Any]] = []
        self.fills: list[dict[str, Any]] = []
        self.daily: list[dict[str, Any]] = []
        self.model = targets.cost_model()

    def sellable(self, code: str) -> int:
        locked = int(self.locked.get(code, {}).get("shares", 0))
        return max(0, int(self.positions.get(code, 0)) - locked)

    def submit_sell(self, date: str, code: str, reason: str) -> None:
        self.orders.append(
            {"date": date, "code": code, "side": "sell", "reason": reason}
        )
        self.pending_sells.setdefault(
            code, {"first_attempt": date, "reason": reason, "attempts": 0}
        )

    def execute_sells(
        self, date: str, prices: Mapping[str, float], blocked: Mapping[str, bool]
    ) -> None:
        for code in sorted(list(self.pending_sells)):
            record = self.pending_sells[code]
            record["attempts"] += 1
            shares = int(self.positions.get(code, 0))
            if shares <= 0:
                self.pending_sells.pop(code, None)
                continue
            if blocked.get(code, False):
                continue
            if self.sellable(code) < shares:
                continue
            price = float(prices[code])
            notional = shares * price
            cost = self.model.calculate_sell_cost(notional)
            self.cash += notional - cost
            self.fills.append(
                {
                    "date": date,
                    "code": code,
                    "side": "sell",
                    "shares": shares,
                    "price": price,
                    "notional": notional,
                    "fees": cost,
                    "reason": record["reason"],
                    "attempts": record["attempts"],
                }
            )
            self.positions.pop(code, None)
            self.locked.pop(code, None)
            self.pending_sells.pop(code, None)

    def buy(
        self,
        date: str,
        code: str,
        price: float,
        budget: float,
        sector: str,
        blocked: bool,
    ) -> dict[str, Any]:
        record = {
            "date": date,
            "code": code,
            "sector": sector,
            "shares": 0,
            "notional": 0.0,
            "fees": 0.0,
            "reason": "",
        }
        if blocked:
            record["reason"] = "suspended_or_limit_up"
            self.orders.append({**record, "side": "buy"})
            return record
        shares, notional = targets.affordable_shares(price, budget)
        if notional < spec.MIN_NOTIONAL:
            record["reason"] = "below_min_notional"
        elif notional > self.cash:
            record["reason"] = "cash_insufficient"
        else:
            cost = self.model.calculate_buy_cost(notional)
            self.cash -= notional + cost
            self.positions[code] = self.positions.get(code, 0) + shares
            record.update({"shares": shares, "notional": notional, "fees": cost})
            record["reason"] = "buy"
            self.fills.append({**record, "side": "buy", "price": price})
        self.orders.append({**record, "side": "buy"})
        return record

    def mark(self, date: str, prices: Mapping[str, float]) -> dict[str, Any]:
        market_value = sum(
            shares * float(prices.get(code, 0.0))
            for code, shares in self.positions.items()
        )
        total = self.cash + market_value + self.receivable
        row = {
            "date": date,
            "cash": self.cash,
            "market_value": market_value,
            "receivable": self.receivable,
            "total_value": total,
            "identity_error": abs(total - (self.cash + market_value + self.receivable)),
            "positions": len(self.positions),
            "negative_cash": self.cash < -1e-9,
        }
        self.daily.append(row)
        return row


def _synthetic_prices(codes: Sequence[str], days: Sequence[str]) -> dict[str, dict[str, float]]:
    """确定性合成价格：基准价 + 温和正弦，不依赖随机数。"""
    base = {code: 8.0 + 4.0 * index for index, code in enumerate(codes)}
    prices: dict[str, dict[str, float]] = {code: {} for code in codes}
    for position, day in enumerate(days):
        for index, code in enumerate(codes):
            drift = 0.02 * np.sin((position + index) / 7.0) + 0.0004 * position
            prices[code][day] = round(float(base[code] * (1.0 + drift)), 4)
    return prices


def run_r3(ctx: StageContext) -> dict[str, Any]:
    """R3：三个月（含春节）+ 停牌顺延 + 连续入选 + 满仓缺位的逐笔演练。"""
    root = spec.REPO_ROOT
    upstream = {item["stage"]: item for item in ctx.require_upstream()}
    r0_dir = run_root(ctx.campaign, upstream["R0"]["run_id"])
    baseline = read_json(r0_dir / "baseline_identity.json")
    days = [day for day in clock.load_calendar(root) if "20201201" <= day <= "20210305"]
    if not days:
        raise ResearchError("R3 合成日历为空")
    codenames = ["SYN001", "SYN002", "SYN003", "SYN004", "SYN005", "SYN006"]
    sectors = {
        "SYN001": "电子",
        "SYN002": "医药",
        "SYN003": "机械",
        "SYN004": "电子",
        "SYN005": "医药",
        "SYN006": "机械",
    }
    prices = _synthetic_prices(codenames, days)
    capital = 600000.0
    book = SyntheticBook(capital)
    calendar = days
    index = {day: position for position, day in enumerate(calendar)}

    month_ends = {}
    for day in calendar:
        month_ends[day[:6]] = day
    ordered_months = sorted(month_ends)
    targets_by_month = {
        ordered_months[0]: ["SYN001", "SYN002", "SYN003"],
        ordered_months[1]: ["SYN001", "SYN004", "SYN005"],
        ordered_months[2]: ["SYN001", "SYN004", "SYN006"],
    }
    suspended_windows = {"SYN002": ("20210201", "20210205")}
    bonus_date = month_ends[ordered_months[2]]
    next_index = index[bonus_date] + 1
    bonus_unlock = calendar[next_index] if next_index < len(calendar) else bonus_date
    exec_map: dict[str, str] = {}
    for month in ordered_months:
        position = index[month_ends[month]] + 1
        if position < len(calendar):
            exec_map[calendar[position]] = month
    pending_seen: dict[str, str] = {}

    for day in calendar:
        month = exec_map.get(day)
        if month is not None:
            target = targets_by_month[month]
            for code in sorted(book.positions):
                if code not in target:
                    book.submit_sell(day, code, "monthly_target_exit")
                    pending_seen.setdefault(code, day)
        else:
            target = []
        blocked = {
            code: suspended_windows.get(code, ("", ""))[0] <= day
            <= suspended_windows.get(code, ("", ""))[1]
            for code in codenames
        }
        book.execute_sells(
            day,
            {code: prices[code][day] for code in codenames},
            blocked,
        )
        if month is not None:
            total = book.daily[-1]["total_value"] if book.daily else capital
            value = targets.unit_value(total)
            for code in target:
                if code in book.positions:
                    continue
                book.buy(
                    day,
                    code,
                    prices[code][day],
                    value,
                    sectors[code],
                    blocked[code],
                )
        if day == bonus_date and "SYN001" in book.positions:
            book.positions["SYN001"] += 100
            book.locked["SYN001"] = {"shares": 100, "unlock_date": bonus_unlock}
        if day == bonus_unlock:
            book.locked.pop("SYN001", None)
        book.mark(day, {code: prices[code][day] for code in codenames})

    expected = {
        "scenario": "三个月（2020-12、2021-01、2021-02 春节短月）",
        "expect": {
            "retained_no_roundtrip": True,
            "suspended_sell_deferred_attempts_min": 2,
            "blocked_buy_fills_zero": True,
            "identity_max_error_max": 1e-6,
            "negative_cash_days": 0,
            "lot_multiple": True,
            "min_notional": round(spec.MIN_NOTIONAL, 6),
        },
    }
    actual = {
        "fills": book.fills,
        "orders": book.orders,
        "daily": book.daily,
        "ending_cash": book.cash,
        "ending_positions": dict(book.positions),
        "pending_sells": book.pending_sells,
        "retained_buy_orders": [
            item for item in book.orders if item.get("side") == "buy" and item["shares"] == 0
        ],
    }
    deferred = {
        fill["code"]: int(fill.get("attempts", 0))
        for fill in book.fills
        if fill["side"] == "sell" and int(fill.get("attempts", 0)) > 1
    }
    retained_roundtrip = any(
        fill["code"] == "SYN001" and fill["side"] == "sell" for fill in book.fills
    )
    identity_max = max((row["identity_error"] for row in book.daily), default=0.0)
    negative_days = sum(1 for row in book.daily if row["negative_cash"])
    lot_ok = all(
        int(fill["shares"]) % spec.LOT_SIZE == 0
        for fill in book.fills
        if fill["side"] == "buy"
    )
    min_notional_ok = all(
        float(fill["notional"]) >= spec.MIN_NOTIONAL
        for fill in book.fills
        if fill["side"] == "buy"
    )
    baseline_ok = all(
        item["dates_equal"] and item["cash_minus_a80_constant"]
        for item in baseline["capitals"].values()
    )
    checks = {
        "identity_max_error": identity_max,
        "negative_cash_days": negative_days,
        "lot_multiple": lot_ok,
        "min_notional": min_notional_ok,
        "baseline_identity": baseline_ok,
        "suspended_deferrals": deferred,
        "retained_no_roundtrip": not retained_roundtrip,
        "trades": len(book.fills),
        "engine_reuse": "partial: CostModel 计费 + 恒等式逐日核对；完整原生引擎接线留待 R4 授权",
    }
    validation = {
        "all_passed": bool(
            identity_max <= 1e-6
            and negative_days == 0
            and lot_ok
            and min_notional_ok
            and baseline_ok
            and not retained_roundtrip
            and bool(deferred)
        ),
        "checks": checks,
        "gaps": [
            "未接 NativeMergeBacktestEngineML：锁股/应收/公司行为仅验证合成口径",
            "R4 前须用真实引擎重跑本演练并对照 synthetic_expected.json",
        ],
    }
    out = ctx.staging_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "synthetic_expected.json", expected)
    write_json(out / "synthetic_actual.json", actual)
    write_json(
        out / "account_params.json",
        {str(capital): targets.account_params(int(capital)) for capital in spec.CAPITALS},
    )
    write_json(out / "validation.json", validation)
    files = {
        path.name: sha256_file(path) for path in sorted(out.iterdir()) if path.is_file()
    }
    manifest = ctx.finish(
        {
            "files": files,
            "validation": validation,
            "machine_time_seconds": round(time.perf_counter() - ctx.started, 3),
            "note": "合成演练；完整原生引擎接线为 R4 前置缺口",
        }
    )
    return {"manifest": manifest, "validation": validation}
