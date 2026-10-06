# -*- coding: utf-8 -*-
"""v2 P1 单元 4：panel 性能闸门（冻结 §7：全历史跨列族加载 ≤ 现状 1.5 倍）。

双口径（各自判 ≤1.5×，均过才 PASS——R3-11 整改）：
- 全列口径：panel 377 列 vs 基线全列；
- 抽 5 列口径：pe_ttm/pb/net_mf_amount/roe_waa/volatility_20（基线同窗口同列投影）。
各 --runs 次取中位。

基线 = `ml/train_core/split.py::load_features_data`（旧链路逐日 parquet 加载 + concat）。
该函数不支持列裁剪，抽 5 列基线为本脚本内的同语义薄实现（同一交易日历、逐日读
cs_train 分区做列投影、缺失日跳过、concat；不改 split.py 数值行为）。
v2 = `PanelDataStore.load_features`（热/冷透明，跨族 outer join）。

内存注意（本机 ~33GB）：全历史全列双侧结果帧均为 ~15.8M 行 × ~380 列（数十 GB 量级），
超内存时按 --start/--end 收窄窗口（双侧同口径，相对比较保持）并在产物注明实际口径。
产物：data/reports/v2_p1_panel_perf_<date>.json + 控制台计时表。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.data.loader import DataLoader  # noqa: E402
from src.lazybull.data.storage import Storage  # noqa: E402
from src.lazybull.ml.train_core.split import load_features_data  # noqa: E402
from src.lazybull.v2.common.types import FeatureQuery, TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import PANEL_GROUPS  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

#: 抽 5 列口径（冻结 §7 指定）
SAMPLE_5_COLS = ["pe_ttm", "pb", "net_mf_amount", "roe_waa", "volatility_20"]

#: panel 全列 = 377 − 2 键列（键列以 index 承载）
_ALL_FEATURE_COLS = [
    c for cols in PANEL_GROUPS.values() for c in cols if c not in ("trade_date", "ts_code")
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 P1 panel 性能闸门")
    parser.add_argument("--start", default="20120104", help="起点 YYYYMMDD（默认全历史）")
    parser.add_argument("--end", default="20260702", help="终点 YYYYMMDD")
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="v2 store 数据根")
    parser.add_argument("--runs", type=int, default=3, help="每口径重复次数（取中位）")
    parser.add_argument("--out", default=None, help="产物 JSON 路径")
    return parser.parse_args()


#: 性能门倍率（冻结 §7）
GATE_RATIO = 1.5


def _baseline_load_sample(
    storage: Storage, loader: DataLoader, start: str, end: str, columns: list[str]
) -> pd.DataFrame:
    """基线抽列口径薄实现（R3-11）：与 `load_features_data` 同语义
    （同一交易日历、逐日读 cs_train 分区、缺失日跳过、concat），仅加列投影。

    split.py 不支持列裁剪，本函数不改动其数值行为，仅服务于基线抽列计时。
    """
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()
    trade_dates = trade_cal[
        (trade_cal["cal_date"] >= start)
        & (trade_cal["cal_date"] <= end)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    cs_dir = storage.features_path / "cs_train"
    frames = []
    for trade_date in trade_dates:
        path = cs_dir / f"{trade_date}.parquet"
        if not path.exists():
            continue
        available = set(pq.ParquetFile(path).schema_arrow.names)
        read_cols = [c for c in ("trade_date", "ts_code", *columns) if c in available]
        df = pd.read_parquet(path, columns=read_cols)
        for col in columns:  # 早期分区缺列补 NaN，保持列集稳定
            if col not in df.columns:
                df[col] = float("nan")
        if len(df) > 0:
            frames.append(df)
    if not frames:
        raise ValueError(f"指定日期区间内没有特征数据: {start}~{end}")
    return pd.concat(frames, ignore_index=True)


def _evaluate(ratio_full: float, ratio_sample: float) -> dict:
    """双口径判定（R3-11 整改）：全列与抽 5 列各自 ≤ 门限，均过才 PASS。"""
    return {
        "pass": ratio_full <= GATE_RATIO and ratio_sample <= GATE_RATIO,
        "ratio_full_cols": ratio_full,
        "ratio_sample_5_cols": ratio_sample,
        "rule": f"v2 ≤ 基线 ×{GATE_RATIO}（全列与抽 5 列双口径各自判定，均过才 PASS）",
    }


def _time_median(fn, runs: int) -> tuple[float, int, int]:
    """计时中位，返回 (中位秒, 行数, 列数)。每次跑完 gc 释放。"""
    times = []
    shape = (0, 0)
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        df = fn()
        times.append(time.perf_counter() - t0)
        shape = df.shape
        del df
    return sorted(times)[len(times) // 2], shape[0], shape[1]


def main() -> int:
    args = _parse_args()
    storage = Storage()
    loader = DataLoader(storage)
    store = PanelDataStore(args.data_root)
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": {"start": args.start, "end": args.end},
        "runs": args.runs,
        "gate": f"v2 ≤ 基线 ×{GATE_RATIO}（双口径各自判定，均过才 PASS）",
        "note": None,
    }

    def _v2_load(cols: list[str]):
        frame = store.load_features(
            FeatureQuery(
                columns=cols,
                start_date=TradeDate.from_str(args.start),
                end_date=TradeDate.from_str(args.end),
            )
        )
        return frame.df

    def _baseline_load():
        df, _ = load_features_data(storage, loader, args.start, args.end)
        return df

    def _baseline_load_5():
        return _baseline_load_sample(storage, loader, args.start, args.end, SAMPLE_5_COLS)

    # 全列口径
    base_full_t, base_rows, base_cols = _time_median(_baseline_load, args.runs)
    v2_full_t, v2_rows, v2_cols = _time_median(lambda: _v2_load(_ALL_FEATURE_COLS), args.runs)
    # 抽 5 列口径（基线同窗口同列投影，与 v2 同口径计时）
    base_5_t, base5_rows, base5_cols = _time_median(_baseline_load_5, args.runs)
    v2_5_t, v2_5_rows, v2_5_cols = _time_median(lambda: _v2_load(SAMPLE_5_COLS), args.runs)

    ratio_full = v2_full_t / base_full_t if base_full_t > 0 else float("inf")
    ratio_5 = v2_5_t / base_5_t if base_5_t > 0 else float("inf")
    report["results"] = {
        "full_cols": {
            "columns": f"全列（panel {v2_cols} 列 vs 基线 {base_cols} 列）",
            "baseline": {"median_s": base_full_t, "rows": base_rows, "cols": base_cols},
            "v2": {"median_s": v2_full_t, "rows": v2_rows, "cols": v2_cols},
            "ratio": ratio_full,
            "gate": GATE_RATIO,
            "pass": ratio_full <= GATE_RATIO,
        },
        "sample_5_cols": {
            "columns": SAMPLE_5_COLS,
            "baseline": {"median_s": base_5_t, "rows": base5_rows, "cols": base5_cols},
            "v2": {"median_s": v2_5_t, "rows": v2_5_rows, "cols": v2_5_cols},
            "ratio": ratio_5,
            "gate": GATE_RATIO,
            "pass": ratio_5 <= GATE_RATIO,
        },
    }
    verdict = _evaluate(ratio_full, ratio_5)
    report["verdict"] = verdict
    if args.start != "20120104" or args.end != "20260702":
        report["note"] = f"窗口收窄口径（内存约束，双侧同窗口）: {args.start}~{args.end}"

    out_path = (
        Path(args.out)
        if args.out
        else (ROOT / "data" / "reports" / f"v2_p1_panel_perf_{datetime.now():%Y%m%d}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"基线全列: {base_full_t:.1f}s（{base_rows} 行 × {base_cols} 列）")
    print(f"v2 全列: {v2_full_t:.1f}s（{v2_rows} 行 × {v2_cols} 列），比值 {ratio_full:.2f}")
    print(f"基线抽5列: {base_5_t:.1f}s（{base5_rows} 行 × {base5_cols} 列）")
    print(f"v2 抽5列: {v2_5_t:.1f}s（{v2_5_rows} 行 × {v2_5_cols} 列），比值 {ratio_5:.2f}")
    print(f"判定: {'PASS' if verdict['pass'] else 'FAIL'}（双口径各 ≤ ×{GATE_RATIO}）；报告: {out_path}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
