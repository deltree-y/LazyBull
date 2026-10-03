# -*- coding: utf-8 -*-
"""v2 P1 单元 3：重放对账薄入口（冻结参照分区 vs cs_train）。

用法：
    python scripts/v2_p1/compare_replay.py --replay-root temp/p1_frozen_reference_b \
        --start 20260401 --end 20260630 [--jobs -1] [--out PATH]

产物：--out 指定 JSON（默认 data/reports/v2_p1_frozen_ref_compare_<date>.json）+ 控制台汇总。
业务逻辑在 `src/lazybull/v2/store/replay_compare.py`（scripts 薄入口契约）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.v2.store.replay_compare import compare_partitions  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 P1 冻结参照重放对账")
    parser.add_argument("--replay-root", required=True, help="重放 scratch 根")
    parser.add_argument(
        "--cs-train-root",
        default=str(ROOT / "data" / "features" / "cs_train"),
        help="cs_train 分区根（生产只读）",
    )
    parser.add_argument("--start", required=True, help="比对起点 YYYYMMDD")
    parser.add_argument("--end", required=True, help="比对终点 YYYYMMDD")
    parser.add_argument("--jobs", type=int, default=-1, help="joblib worker 数（1 = 串行）")
    parser.add_argument("--out", default=None, help="报告 JSON 路径")
    return parser.parse_args()


def _dates_in_range(replay_root: Path, cs_root: Path, start: str, end: str) -> list[str]:
    """比对日期 = 两侧分区名在区间内的并集（缺侧在对账内按 missing 登记）。"""
    dates: set[str] = set()
    for root, sub in ((replay_root, "features/cs_train"), (cs_root, "")):
        base = root / sub if sub else root
        if not base.exists():
            continue
        for file in base.glob("*.parquet"):
            if len(file.stem) == 8 and file.stem.isdigit() and start <= file.stem <= end:
                dates.add(file.stem)
    return sorted(dates)


def main() -> int:
    args = _parse_args()
    replay_root = Path(args.replay_root)
    cs_root = Path(args.cs_train_root)
    dates = _dates_in_range(replay_root, cs_root, args.start, args.end)
    if not dates:
        print(f"区间内无分区: {args.start}~{args.end}")
        return 1
    report = compare_partitions(replay_root, cs_root, dates, n_jobs=args.jobs)

    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "data" / "reports" / f"v2_p1_frozen_ref_compare_{datetime.now():%Y%m%d}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    days = report["days"]
    summary = report["summary"]
    print(f"对账窗口: {report['window']['start']}~{report['window']['end']}")
    print(
        f"日级: 请求 {days['requested']}，比对 {days['compared']}，缺分区 {len(days['missing'])}，"
        f"行集差日 {len(days['row_set_diff_days'])}"
    )
    print(
        f"列级: 比对列 {summary['total_columns_compared']}，差异列 {summary['columns_with_diff']}，"
        f"over_gate {len(summary['over_gate_columns'])}"
    )
    print(f"归因分布: {summary['by_attribution']}")
    for item in summary["worst"][:5]:
        print(
            f"  {item['column']}: share={item['over_share']:.2e}, max|Δ|={item['max_abs_diff']:.3e}"
        )
    print(f"报告: {out_path}")
    return 1 if summary["over_gate_columns"] else 0


if __name__ == "__main__":
    sys.exit(main())
