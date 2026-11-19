"""R2 数据集合成：基础合格域 × 分钟特征 × 6 日线控制列 × 行业中性 10 日标签。

口径（协议 §5~§7）：
- 行键来自 clean/daily 的 `(trade_date, ts_code)`，**不由 cs_train、旧 panel 或现存名单决定**；
- 基础合格域每一行都保留 24 列输入与状态：`minute_informed=False` 的行保留为不可评分，
  不从覆盖分母消失，也不退化为纯日线分数；
- 训练/对照行域严格一致（都用 `base_eligible & minute_informed & label valid`）；
- 标签行业中性化只在**基础合格域**上取中位数，保存原始值与中性值两列。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import domain, features, labels, panel, spec, splits, store

FEATURE_SCHEMA = "lr5m_ml_24cols/v1"


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def load_inputs(root: Path, start: str, end: str) -> dict[str, Any]:
    """装载价格/市值/基础表/行业 PIT（只读，不写生产目录）。"""
    calendar = panel.load_calendar(root)
    prices = panel.load_daily_panel(root, start, end, calendar)
    basic = panel.load_basic_panel(root, start, end, calendar)
    frame_basic = domain.stock_basic_frame(root)
    pit = domain.industry_pit_frame(root)
    table = domain.industry_code_table_frame(root)
    return {
        "calendar": calendar,
        "prices": prices,
        "basic": basic,
        "stock_basic": frame_basic,
        "industry_pit": pit,
        "l1_names": domain.industry_l1_names(table, pit),
        "dates": [str(d) for d in prices["close_adj"].index],
        "codes": [str(c) for c in prices["close_adj"].columns],
    }


def _minute_rows_for_day(
    store_obj: store.MinuteStore,
    day: str,
    eligible_codes: Sequence[str],
    history: dict[str, deque],
) -> dict[str, dict[str, Any]]:
    """单日逐股计算分钟列并推进历史（T 之前的 20 个交易日，不含 T）。"""
    frames = store_obj.day_frames(day, eligible_codes)
    out: dict[str, dict[str, Any]] = {}
    for code in eligible_codes:
        frame = frames.get(str(code))
        if frame is None:
            row = {name: float("nan") for name in spec.MINUTE_FEATURES}
            row.update(
                {
                    "session_complete": False,
                    "missing_slots": "all",
                    "minute_finite_columns": 0,
                    "minute_informed": False,
                    "minute_status": "no_minute_source",
                    "A_core": float("nan"),
                    "q_late": float("nan"),
                }
            )
        else:
            row = features.minute_feature_row(frame, list(history.get(str(code), [])))
        history.setdefault(str(code), deque(maxlen=features.HISTORY_WINDOW)).append(
            {"q_late": row["q_late"], "A_core": row["A_core"]}
        )
        out[str(code)] = row
    return out


def build_dataset(
    root: Path,
    *,
    minute_dir: Path,
    feature_start: str,
    end: str,
    out_dir: Path,
    source_manifest_sha: str,
    log: Callable[[str], None] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """构建并（可选）落盘分年面板，返回摘要与台账。"""
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

    result = domain.build_domain(
        inputs["prices"],
        inputs["basic"],
        inputs["stock_basic"],
        inputs["industry_pit"],
        listing_calendar=calendar,
        l1_names=inputs["l1_names"],
    )
    emit(f"基础域：{time.perf_counter()-started:.1f}s，条件计数 {result.condition_counts}")

    raw_label, book = labels.build_labels(inputs["prices"], dates)
    bucket_matrix, bucket_names = domain.industry_bucket_matrix(
        inputs["industry_pit"], dates, codes, inputs["l1_names"]
    )
    neutral, median_matrix = labels.industry_neutralize(
        raw_label,
        book["label_status"],
        result.eligible,
        bucket_matrix,
        min_members=int(spec.LABEL["min_industry_members"]),
    )

    controls = features.daily_controls(
        inputs["prices"]["close_adj"],
        inputs["prices"]["amount"].astype(float) * float(spec.UNITS["daily_amount_to_yuan"]),
        inputs["basic"]["turnover_rate"].reindex(index=dates, columns=codes),
        inputs["basic"]["total_mv"].reindex(index=dates, columns=codes)
        * float(spec.UNITS["total_mv_to_yuan"]),
    )
    controls["trade_date"] = controls["trade_date"].astype(str)
    controls = controls.set_index(["trade_date", "ts_code"])
    emit(f"日线控制列：{len(controls)} 行，耗时 {time.perf_counter()-started:.1f}s")

    window_mask = np.array(
        [[feature_start <= day <= end for _ in codes] for day in dates], dtype=bool
    )
    label_long = labels.extract_label_frame(
        raw_label,
        book["label_status"],
        neutral,
        median_matrix,
        result.eligible & window_mask,
        dates,
        codes,
        dates,
    )
    label_long = label_long.set_index(["trade_date", "ts_code"])

    store_obj = store.MinuteStore(minute_dir)
    history: dict[str, deque] = {}
    day_index = {str(day): i for i, day in enumerate(dates)}
    minute_complete = np.zeros((len(dates), len(codes)), dtype=bool)
    minute_informed = np.zeros((len(dates), len(codes)), dtype=bool)
    code_index = {code: j for j, code in enumerate(codes)}

    yearly_rows: dict[str, list[dict[str, Any]]] = {}
    for day in dates:
        i = day_index[day]
        eligible_codes = [code for code in codes if result.eligible[i, code_index[code]]]
        if not eligible_codes:
            continue
        minute_rows = _minute_rows_for_day(store_obj, day, eligible_codes, history)
        for code in eligible_codes:
            minute_complete[i, code_index[code]] = bool(minute_rows[code]["session_complete"])
            minute_informed[i, code_index[code]] = bool(minute_rows[code]["minute_informed"])
        if day < feature_start or day > end:
            continue
        records: list[dict[str, Any]] = []
        rows: list[int] = []
        cols: list[int] = []
        control_block = controls.loc[day] if day in controls.index.get_level_values(0) else None
        for code, minute_row in minute_rows.items():
            payload: dict[str, Any] = {"trade_date": day, "ts_code": code}
            for name in spec.DAILY_FEATURES:
                value = float("nan")
                if control_block is not None and code in control_block.index:
                    candidate = control_block.at[code, name]
                    if pd.notna(candidate):
                        value = float(candidate)
                payload[name] = value
            for name in spec.MINUTE_FEATURES:
                payload[name] = float(minute_row[name])
            payload["session_complete"] = bool(minute_row["session_complete"])
            payload["missing_slots"] = str(minute_row["missing_slots"])
            payload["minute_finite_columns"] = int(minute_row["minute_finite_columns"])
            payload["minute_informed"] = bool(minute_row["minute_informed"])
            payload["minute_status"] = str(minute_row["minute_status"])
            records.append(payload)
            rows.append(day_index[day])
            cols.append(code_index[code])
        if not records:
            continue
        block = pd.DataFrame(records)
        block = block.merge(
            label_long.reset_index(),
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        rows_array = np.asarray(rows, dtype=int)
        cols_array = np.asarray(cols, dtype=int)
        block["industry_bucket"] = [
            "UNKNOWN" if bucket_matrix[r, c] == 0 else bucket_names[bucket_matrix[r, c] - 1]
            for r, c in zip(rows_array, cols_array)
        ]
        block["reason_code"] = result.reason[rows_array, cols_array].astype(np.int32)
        block["reason_flags"] = result.flags[rows_array, cols_array].astype(np.int32)
        block["feature_schema"] = FEATURE_SCHEMA
        block["source_manifest_sha"] = source_manifest_sha
        block["asof_date"] = block["trade_date"]
        yearly_rows.setdefault(day[:4], []).append(block)

    frame = (
        pd.concat([part for blocks in yearly_rows.values() for part in blocks], ignore_index=True)
        if yearly_rows
        else pd.DataFrame(columns=["trade_date", "ts_code"])
    )
    emit(f"面板行：{len(frame)}，耗时 {time.perf_counter()-started:.1f}s")

    domain_frame = domain.daily_domain(
        result,
        minute_complete,
        minute_informed,
        int(spec.UNIVERSE["min_scored_per_day"]),
        mother_mask=np.isfinite(inputs["prices"]["close_adj"].to_numpy(dtype=float)),
    )
    domain_frame = domain_frame[domain_frame["trade_date"] >= feature_start]
    ledger = domain.reason_ledger(result)
    maturity = labels.maturity_frame(book["label_status"], dates)
    maturity = maturity[maturity["trade_date"] >= feature_start]
    gate = domain.coverage_gate(domain_frame)

    summary: dict[str, Any] = {
        "feature_start": feature_start,
        "end": end,
        "rows": int(len(frame)),
        "days": int(frame["trade_date"].nunique()) if len(frame) else 0,
        "codes": int(frame["ts_code"].nunique()) if len(frame) else 0,
        "base_eligible_mean": float(domain_frame["base_eligible_count"].mean()),
        "minute_informed_mean": float(domain_frame["minute_informed_count"].mean()),
        "minute_informed_min": int(domain_frame["minute_informed_count"].min()),
        "days_below_100": int((domain_frame["min100_status"] != "ok").sum()),
        "coverage_gate": gate,
        "label_status_counts": (
            frame["label_status"].value_counts().to_dict() if "label_status" in frame else {}
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "feature_schema": FEATURE_SCHEMA,
        "source_manifest_sha": source_manifest_sha,
    }
    written: dict[str, Any] = {}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        for year, part in frame.groupby(frame["trade_date"].str[:4]):
            path = out_dir / f"panel_{year}.parquet"
            part.reset_index(drop=True).to_parquet(path, index=False)
            written[path.name] = int(len(part))
        domain_frame.to_parquet(out_dir / "daily_domain.parquet", index=False)
        ledger.to_parquet(out_dir / "reason_ledger.parquet", index=False)
        maturity.to_parquet(out_dir / "label_maturity.parquet", index=False)
        features.feature_coverage(frame).to_parquet(
            out_dir / "feature_coverage.parquet", index=False
        )
        split_table = splits.fold_table(splits.build_folds(calendar, end))
        split_table.to_parquet(out_dir / "folds.parquet", index=False)
        (out_dir / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "summary": summary,
        "written": written,
        "domain": domain_frame,
        "ledger": ledger,
        "maturity": maturity,
        "frame": frame,
        "minute_store_years": store_obj.years_seen,
    }
