# -*- coding: utf-8 -*-
"""v2 P1 单元 4：panel 性能闸门（冻结 §7：全历史跨列族加载 ≤ 现状 1.5 倍）。

双口径：全列（panel 377 列 vs 基线全列）+ 抽 5 列（pe_ttm/pb/net_mf_amount/roe_waa/
volatility_20）。各 --runs 次取中位。

基线 = `ml/train_core/split.py::load_features_data`（旧链路逐日 parquet 加载 + concat）。
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
    parser.add_argument("--data-root", default="data", help="v2 store 数据根")
    parser.add_argument("--runs", type=int, default=3, help="每口径重复次数（取中位）")
    parser.add_argument("--out", default=None, help="产物 JSON 路径")
    return parser.parse_args()


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
        "gate": "v2 ≤ 基线 ×1.5（双口径）",
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

    # 全列口径
    base_full_t, base_rows, base_cols = _time_median(_baseline_load, args.runs)
    v2_full_t, v2_rows, v2_cols = _time_median(lambda: _v2_load(_ALL_FEATURE_COLS), args.runs)
    # 抽 5 列口径
    v2_5_t, _, _ = _time_median(lambda: _v2_load(SAMPLE_5_COLS), args.runs)
    report["results"] = {
        "baseline_full_cols": {"median_s": base_full_t, "rows": base_rows, "cols": base_cols},
        "v2_full_377": {"median_s": v2_full_t, "rows": v2_rows, "cols": v2_cols},
        "v2_sample_5": {"median_s": v2_5_t},
    }
    ratio_full = v2_full_t / base_full_t if base_full_t > 0 else float("inf")
    verdict = ratio_full <= 1.5
    report["verdict"] = {
        "pass": verdict,
        "ratio_full_cols": ratio_full,
        "rule": "v2 全列 ≤ 基线 ×1.5（抽 5 列口径仅登记，不设门）",
    }
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
    print(f"v2 抽5列: {v2_5_t:.1f}s")
    print(f"判定: {'PASS' if verdict else 'FAIL'}（门 ×1.5）；报告: {out_path}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
