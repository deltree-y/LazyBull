"""R1 数据集合成：基础域 × 分析师覆盖 × 15 列特征 × 行业中性 20 日标签。

口径（协议 §4~§7）：
- 母表行键来自 clean/daily 的 `(trade_date, ts_code)`，不由 cs_train/旧 panel/现存名单决定；
- 分析师覆盖不足（`N_active<2`）的行不进评分缓存，但逐日计数进台账；
- 前 7 列来自 `factors/consensus_revision.py`（沿用其既有 1%/99% 截面 winsorize），
  后 8 列由本包 `diffusion.py` 构建；
- 标签行业中性化只在**基础合格域**上取中位数，保存原始值与中性值两列。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import (
    diffusion,
    labels,
    panel,
    report_events,
    spec,
    universe,
)

FEATURE_SCHEMA = "ad_ml_15cols/v1"


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def load_inputs(root: Path, start: str, end: str) -> dict[str, Any]:
    """装载价格/市值/基础表/行业 PIT（只读）。"""
    calendar = panel.load_calendar(root)
    prices = panel.load_daily_panel(root, start, end, calendar)
    basic = panel.load_basic_panel(root, start, end, calendar)
    frame_basic = universe.stock_basic_frame(root)
    pit = universe.industry_pit_frame(root)
    table = universe.industry_code_table_frame(root)
    return {
        "calendar": calendar,
        "prices": prices,
        "basic": basic,
        "stock_basic": frame_basic,
        "industry_pit": pit,
        "l1_names": universe.industry_l1_names(table, pit),
        "dates": [str(d) for d in prices["close_adj"].index],
        "codes": [str(c) for c in prices["close_adj"].columns],
    }


def consensus_frame(
    raw: pd.DataFrame,
    trading_dates: Sequence[str],
    emit: Callable[[str], None],
    asof_dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """前 7 列（现有语义）：按日查询表 -> 长表。

    传 `asof_dates` 时，第 i 天的特征取 `asof_dates[i]` 的查询结果（延迟压力用），
    使**全部 15 列**都只使用 `report_date <= asof` 的信息，而不是只延后后 8 列。
    """
    from src.lazybull.factors.consensus_revision import (
        CONSENSUS_REVISION_COLS,
        CONSENSUS_REVISION_FRESHNESS_COL,
        build_consensus_revision_lookup_by_date,
    )

    started = time.perf_counter()
    days = [str(day) for day in trading_dates]
    asof = [str(day) for day in (asof_dates or days)]
    if len(asof) != len(days):
        raise ValueError(f"asof 轴长度不一致：{len(asof)} != {len(days)}")
    lookup = build_consensus_revision_lookup_by_date(raw, sorted(set(asof)))
    emit(f"consensus lookup: {len(lookup)} days in {time.perf_counter() - started:.1f}s")
    columns = list(CONSENSUS_REVISION_COLS) + [CONSENSUS_REVISION_FRESHNESS_COL]
    frames = []
    for day, asof_day in zip(days, asof):
        frame = lookup.get(asof_day)
        if frame is None or frame.empty:
            continue
        part = frame[["ts_code"] + columns].copy()
        part["trade_date"] = str(day)
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code"] + columns)
    out = pd.concat(frames, ignore_index=True)
    for name in columns:
        out[name] = pd.to_numeric(out[name], errors="coerce").astype(float)
    return out


def build_dataset(
    root: Path,
    *,
    feature_start: str,
    end: str,
    out_dir: Path,
    source_manifest_sha: str,
    lag_trading_days: int = 0,
    log: Callable[[str], None] | None = None,
    write: bool = True,
    keep_frame: bool = True,
) -> dict[str, Any]:
    """构建并（可选）落盘分年缓存，返回摘要与台账。"""
    emit = _emit(log)
    started = time.perf_counter()
    feature_start = panel.norm_day(feature_start)
    end = panel.norm_day(end)
    calendar = panel.load_calendar(root)
    first = next(i for i, day in enumerate(calendar) if day >= feature_start)
    warmup = int(spec.WINDOW["dataset_load_warmup_trading_days"])
    load_start = calendar[max(0, first - warmup)]
    emit(f"装载窗口 {load_start}..{end}")

    inputs = load_inputs(root, load_start, end)
    dates = inputs["dates"]
    codes = inputs["codes"]
    emit(f"面板：{len(dates)} 日 × {len(codes)} 只，耗时 {time.perf_counter()-started:.1f}s")

    result = universe.build_universe(
        inputs["prices"],
        inputs["basic"],
        inputs["stock_basic"],
        inputs["industry_pit"],
        min_list_days=int(spec.UNIVERSE["min_list_days"]),
        min_mv_yuan=float(spec.UNIVERSE["min_total_mv_yuan"]),
        max_mv_yuan=float(spec.UNIVERSE["max_total_mv_yuan"]),
        min_amount20_yuan=float(spec.UNIVERSE["min_amount20_yuan"]),
        amount_min_periods=int(spec.UNIVERSE["amount_min_periods"]),
        listing_calendar=calendar,
        l1_names=inputs["l1_names"],
    )
    emit(f"基础域：{time.perf_counter()-started:.1f}s，条件计数 {result.condition_counts}")

    raw_reports = report_events.load_report_rc(root)
    events = report_events.build_events(
        raw_reports, trading_days=calendar, lag_trading_days=lag_trading_days, log=emit
    )
    emit(f"事件层：{time.perf_counter()-started:.1f}s")

    coverage = report_events.analyst_coverage_matrix(
        events["activity"], events["primary"], dates, codes
    )
    n_active = coverage["n_active"]
    report_age = coverage["report_age"]
    min_active = int(spec.UNIVERSE["min_active_analysts_90"])
    window_mask = np.array([feature_start <= day <= end for day in dates])
    base_eligible = result.eligible & window_mask.reshape(-1, 1)
    scored_mask = base_eligible & (n_active >= min_active)
    rate = scored_mask.sum() / max(1, window_mask.sum())
    emit(f"评分域：{int(scored_mask.sum())} 行（日均 {rate:.1f}）")

    diffusion_frame = diffusion.compute_features(
        events, dates, codes, scored_mask, n_active, report_age, log=emit
    )
    emit(f"扩散特征：{time.perf_counter()-started:.1f}s")

    asof_dates = None
    if lag_trading_days > 0:
        # 延迟压力必须覆盖**全部 15 列**：把 7 列共识特征的 as-of 轴整体前移
        position = {str(day): index for index, day in enumerate(calendar)}
        asof_dates = [
            calendar[max(0, position[str(day)] - int(lag_trading_days))] for day in dates
        ]
    consensus = consensus_frame(raw_reports, list(dates), emit, asof_dates=asof_dates)
    emit(f"现有 7 列：{len(consensus)} 行（lag={lag_trading_days}）")

    raw_label, book = labels.build_labels(inputs["prices"], dates)
    bucket_matrix, bucket_names = universe.industry_bucket_matrix(
        inputs["industry_pit"], dates, codes, inputs["l1_names"]
    )
    neutral, median_matrix = labels.industry_neutralize(
        raw_label,
        book["label_status"],
        base_eligible,
        bucket_matrix,
        min_members=int(spec.LABEL["min_industry_members"]),
    )
    label_long = labels.extract_label_frame(
        raw_label,
        book["label_status"],
        neutral,
        median_matrix,
        scored_mask,
        dates,
        codes,
        dates,
    )
    emit(f"标签：{time.perf_counter()-started:.1f}s")

    frame = diffusion_frame.merge(
        consensus, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
    )
    frame = frame.merge(
        label_long, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
    )
    rows, cols = np.nonzero(scored_mask)
    bucket_values = bucket_matrix[rows, cols]
    bucket_labels = np.array(["UNKNOWN"] + bucket_names, dtype=object)
    frame["industry_bucket"] = bucket_labels[bucket_values]
    frame["reason_code"] = result.reason[rows, cols].astype(np.int32)
    frame["reason_flags"] = result.flags[rows, cols].astype(np.int32)
    frame["n_active_analysts_90"] = n_active[rows, cols].astype(np.int32)
    frame["feature_schema"] = FEATURE_SCHEMA
    frame["source_manifest_sha"] = source_manifest_sha
    frame["asof_date"] = frame["trade_date"]

    domain = universe.daily_domain(
        result, n_active, min_active, int(spec.UNIVERSE["min_scored_per_day"])
    )
    domain = domain[domain["trade_date"] >= feature_start]
    ledger = universe.reason_ledger(result)
    maturity = labels.maturity_frame(book["label_status"], dates)
    maturity = maturity[maturity["trade_date"] >= feature_start]

    summary: dict[str, Any] = {
        "feature_start": feature_start,
        "end": end,
        "rows": int(len(frame)),
        "days": int(frame["trade_date"].nunique()),
        "codes": int(frame["ts_code"].nunique()),
        "base_eligible_mean": float(domain["base_eligible_count"].mean()),
        "analyst_eligible_mean": float(domain["analyst_eligible_count"].mean()),
        "analyst_eligible_min": int(domain["analyst_eligible_count"].min()),
        "days_below_100": int((domain["min100_status"] != "ok").sum()),
        "label_status_counts": frame["label_status"].value_counts().to_dict(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "feature_schema": FEATURE_SCHEMA,
        "lag_trading_days": int(lag_trading_days),
    }
    written: dict[str, Any] = {}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        for year, part in frame.groupby(frame["trade_date"].str[:4]):
            path = out_dir / f"panel_{year}.parquet"
            part.reset_index(drop=True).to_parquet(path, index=False)
            written[path.name] = int(len(part))
        domain.to_parquet(out_dir / "daily_domain.parquet", index=False)
        ledger.to_parquet(out_dir / "reason_ledger.parquet", index=False)
        maturity.to_parquet(out_dir / "label_maturity.parquet", index=False)
        diffusion.feature_coverage(frame).to_parquet(
            out_dir / "feature_coverage.parquet", index=False
        )
        (out_dir / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "summary": summary,
        "written": written,
        "domain": domain,
        "ledger": ledger,
        "maturity": maturity,
        "events": events,
        "frame": frame if keep_frame else None,
    }


def coverage_gate(domain: pd.DataFrame, months_share: float = 0.01) -> dict[str, Any]:
    """协议 §4.2 覆盖门：不足 100 的计划评分日占比 >1% 或连续 5 日不足即 not_ready。"""
    below = (domain["min100_status"] != "ok").astype(bool).to_list()
    share = float(np.mean(below)) if below else 0.0
    longest = 0
    current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    status = "ready"
    if share > months_share or longest >= 5:
        status = "not_ready"
    return {
        "days": int(len(domain)),
        "below_days": int(sum(below)),
        "below_share": share,
        "longest_below_streak": int(longest),
        "threshold_share": float(months_share),
        "status": status,
    }
