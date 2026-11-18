"""R1 数据集合成：母截面资格 + 19 列特征 + 60 日标签 -> 分年缓存。

口径（协议 §4~§6）：
- 母表行键来自 clean/daily 的 `(trade_date, ts_code)`，不由 cs_train/旧 panel/现存名单决定；
- 资格不合格的行**不进入**训练/评分缓存，但原因逐日留台账；
- 特征 NaN 保留行；标签不可观测按七态登记，不静默丢弃。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.defensive_ml import features, labels, panel, spec, universe

FEATURE_SCHEMA = "def_ml_19cols/v1"


def _log(factory: Callable[[str], None] | None) -> Callable[[str], None]:
    return factory or (lambda message: None)


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


def market_benchmark(
    prices: dict[str, pd.DataFrame], stock_basic: pd.DataFrame
) -> pd.DataFrame:
    """全 clean 母截面主板股票的等权有效相邻日收益（标签过滤前）。"""
    close = prices["close_adj"]
    returns = panel.daily_returns(close)
    main_codes = set(stock_basic.loc[stock_basic["market"] == "主板", "ts_code"].astype(str))
    mask = pd.DataFrame(
        np.repeat(
            np.array([c in main_codes for c in close.columns]).reshape(1, -1),
            len(close.index),
            axis=0,
        ),
        index=close.index,
        columns=close.columns,
    )
    return panel.market_series(returns, mask)


def cashflow_lookup(root: Path, calendar: Sequence[str], start: str, end: str,
                    log: Callable[[str], None]) -> dict[str, pd.DataFrame]:
    """现金流 schema v3 逐日查询表（PIT 可用日语义由构建器保证）。"""
    from src.lazybull.factors.cashflow_quality import build_cashflow_quality_lookup_by_date

    files = sorted((Path(root) / "data/raw/cashflow").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("现金流分区为空")
    heavy = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    heavy = heavy[heavy["end_date"].astype(str) >= "20100101"]
    low, high = panel.norm_day(start), panel.norm_day(end)
    days = [day for day in calendar if low <= day <= high]
    started = time.perf_counter()
    lookup = build_cashflow_quality_lookup_by_date(heavy, days)
    log(f"cashflow lookup: {len(lookup)} days in {time.perf_counter() - started:.1f}s")
    return lookup


def build_dataset(
    root: Path,
    *,
    feature_start: str,
    end: str,
    out_dir: Path,
    source_manifest_sha: str,
    log: Callable[[str], None] | None = None,
    with_mdd: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """构建并（可选）落盘分年缓存，返回摘要与台账。"""
    emit = _log(log)
    started = time.perf_counter()
    feature_start = panel.norm_day(feature_start)
    end = panel.norm_day(end)
    load_start = panel.load_calendar(root)
    first = next(i for i, day in enumerate(load_start) if day >= feature_start)
    warmup_start = load_start[max(0, first - 300)]
    emit(f"load window {warmup_start}..{end}")
    inputs = load_inputs(root, warmup_start, end)
    calendar = inputs["calendar"]
    dates = inputs["dates"]
    emit(
        f"loaded {len(dates)} days x {len(inputs['codes'])} codes "
        f"in {time.perf_counter() - started:.1f}s"
    )

    bench = market_benchmark(inputs["prices"], inputs["stock_basic"])
    returns = panel.daily_returns(inputs["prices"]["close_adj"])
    result = universe.build_universe(
        inputs["prices"],
        inputs["basic"],
        returns,
        inputs["stock_basic"],
        inputs["industry_pit"],
        min_list_days=int(spec.UNIVERSE["min_list_days"]),
        min_mv_yuan=float(spec.UNIVERSE["min_total_mv_yuan"]),
        max_mv_yuan=float(spec.UNIVERSE["max_total_mv_yuan"]),
        min_amount20_yuan=float(spec.UNIVERSE["min_amount20_yuan"]),
        min_valid_returns=int(spec.UNIVERSE["min_valid_returns_in_window"]),
        listing_calendar=calendar,
        l1_names=inputs["l1_names"],
    )
    emit(f"universe: {time.perf_counter()-started:.1f}s")

    window_mask = np.array([feature_start <= day <= end for day in dates])
    eligible = result.eligible & window_mask.reshape(-1, 1)

    columns = features.compute_columns(
        inputs["prices"], inputs["basic"], returns,
        bench["mkt_ret"].reindex(returns.index), with_mdd=with_mdd,
    )
    emit(f"features(16): {time.perf_counter()-started:.1f}s")

    frame = features.extract_feature_frame(columns, eligible, dates, result.codes)
    del columns

    lookup = cashflow_lookup(root, calendar, dates[0], end, emit)
    cashflow_long = features.cashflow_frame(lookup, dates)
    frame = features.attach_cashflow(frame, cashflow_long, dates)
    emit(f"cashflow attached: {len(frame)} rows, {len(cashflow_long)} source rows")

    # 标签轴 = 已装载窗口（其末日即冻结数据水位），右删失据此判定
    _, label_book = labels.build_labels(inputs["prices"], dates)
    label_long = labels.extract_label_frame(
        label_book, eligible, dates, result.codes, dates
    )
    emit(f"labels: {time.perf_counter()-started:.1f}s")

    bucket_matrix, bucket_names = universe.industry_bucket_matrix(
        inputs["industry_pit"], dates, result.codes, inputs["l1_names"]
    )
    rows, cols = np.nonzero(eligible)
    bucket_values = bucket_matrix[rows, cols]
    bucket_labels = np.array(["UNKNOWN"] + bucket_names, dtype=object)

    merged = frame.merge(
        label_long, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
    )
    merged["industry_bucket"] = bucket_labels[bucket_values]
    merged["reason_code"] = result.reason[rows, cols]
    merged["reason_flags"] = result.flags[rows, cols].astype(np.int32)
    merged["n_valid_returns_120"] = result.n_valid_returns_120[rows, cols].astype(np.int32)
    merged["feature_schema"] = FEATURE_SCHEMA
    merged["source_manifest_sha"] = source_manifest_sha
    merged["asof_date"] = merged["trade_date"]

    domain = universe.daily_domain(result)
    domain = domain[domain["trade_date"] >= feature_start]
    ledger = universe.reason_ledger(result)
    maturity = labels.maturity_frame(label_book["label_status"], dates)
    maturity = maturity[maturity["trade_date"] >= feature_start]

    summary: dict[str, Any] = {
        "feature_start": feature_start,
        "end": end,
        "rows": int(len(merged)),
        "days": int(merged["trade_date"].nunique()),
        "codes": int(merged["ts_code"].nunique()),
        "eligible_mean": float(domain["eligible_count"].mean()),
        "eligible_min": int(domain["eligible_count"].min()),
        "days_below_100": int((domain["min100_status"] != "ok").sum()),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "feature_schema": FEATURE_SCHEMA,
    }
    written: dict[str, Any] = {}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        for year, part in merged.groupby(merged["trade_date"].str[:4]):
            path = out_dir / f"panel_{year}.parquet"
            part.reset_index(drop=True).to_parquet(path, index=False)
            written[path.name] = int(len(part))
        domain.to_parquet(out_dir / "daily_domain.parquet", index=False)
        ledger.to_parquet(out_dir / "reason_ledger.parquet", index=False)
        maturity.to_parquet(out_dir / "label_maturity.parquet", index=False)
        coverage = feature_coverage(merged)
        coverage.to_parquet(out_dir / "feature_coverage.parquet", index=False)
        (out_dir / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "summary": summary,
        "written": written,
        "domain": domain,
        "ledger": ledger,
        "maturity": maturity,
        "frame": merged,
    }


def feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """逐列 × 逐年覆盖率（NaN 不算排除，但必须可量化）。"""
    work = frame.copy()
    work["year"] = work["trade_date"].str[:4]
    rows = []
    for year, part in work.groupby("year"):
        row: dict[str, Any] = {"year": year, "rows": int(len(part))}
        for name in spec.FEATURES:
            row[name] = float(part[name].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows)
