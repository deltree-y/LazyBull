# -*- coding: utf-8 -*-
"""v2 P1 单元 4：panel 对账门薄入口（严格门 + 归因门）。

用法：
    python scripts/v2_p1/reconcile_panel.py --mode strict \
        [--panel-root data] [--ref-root data/frozen_reference/v2_p1] \
        [--start 20120104] [--end 20260702] [--jobs -1] [--out PATH]
    python scripts/v2_p1/reconcile_panel.py --mode attrib [--ref-root data/features/cs_train] ...

mode：strict = panel vs 冻结参照（出口 = 残差 100% 归因登记类（L1/D-04/D-11/D-12 豁免）
+ 结构项合法性判定（行集零差异）；未归因差异即实现漂移嫌疑 ⇒ 出口 1）；
attrib = panel vs cs_train（差异须 100% 落入五类预期清单且行集零差异 ⇒ 出口 1 并输出越界清单）。
产物：--out 指定 JSON（默认 data/reports/v2_p1_panel_reconcile_<date>.json /
v2_p1_panel_vs_cstrain_<date>.json）+ 控制台汇总。
业务逻辑：`src/lazybull/v2/store/panel_reconcile.py`（scripts 薄入口契约）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.v2.store.panel_reconcile import reconcile_panel  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 P1 panel 对账门（严格门/归因门）")
    parser.add_argument("--mode", choices=["strict", "attrib"], required=True)
    parser.add_argument("--panel-root", default=str(ROOT / "data"), help="v2 store 数据根")
    parser.add_argument("--ref-root", default=None, help="参照分区根（默认随 mode）")
    parser.add_argument("--start", required=True, help="比对起点 YYYYMMDD")
    parser.add_argument("--end", required=True, help="比对终点 YYYYMMDD")
    parser.add_argument("--jobs", type=int, default=-1, help="joblib worker 数（1 = 串行）")
    parser.add_argument("--out", default=None, help="报告 JSON 路径")
    return parser.parse_args()


def _dates_in_range(cs_train_root: Path, start: str, end: str) -> list[str]:
    """比对日 = cs_train 分区名在区间内的清单（参照日历 = cs_train 覆盖日）。"""
    root = cs_train_root
    if (root / "features" / "cs_train").exists():
        root = root / "features" / "cs_train"
    if not root.exists():
        raise ValueError(f"参照分区目录不存在: {root}")
    return sorted(
        f.stem
        for f in root.glob("*.parquet")
        if len(f.stem) == 8 and f.stem.isdigit() and start <= f.stem <= end
    )


def main() -> int:
    args = _parse_args()
    ref_root = (
        Path(args.ref_root)
        if args.ref_root
        else (
            ROOT / "data" / "frozen_reference" / "v2_p1"
            if args.mode == "strict"
            else ROOT / "data" / "features" / "cs_train"
        )
    )
    dates = _dates_in_range(ROOT / "data" / "features" / "cs_train", args.start, args.end)
    if not dates:
        print(f"区间内无 cs_train 分区: {args.start}~{args.end}")
        return 1
    report = reconcile_panel(args.panel_root, ref_root, dates, mode=args.mode, n_jobs=args.jobs)

    default_name = "v2_p1_panel_reconcile" if args.mode == "strict" else "v2_p1_panel_vs_cstrain"
    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "data" / "reports" / f"{default_name}_{datetime.now():%Y%m%d}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    days = report["days"]
    summary = report["summary"]
    verdict = report["verdict"]
    print(f"对账（{args.mode}）: 窗口 {report['window']['start']}~{report['window']['end']}")
    print(
        f"日级: 请求 {days['requested']}，比对 {days['compared']}，"
        f"缺参照分区 {len(days['missing_ref_days'])}，行集差日 {len(days['row_set_diff_days'])}"
    )
    print(
        f"列级: 差异列 {summary['columns_with_diff']}，over_gate {len(summary['over_gate_columns'])}"
    )
    print(f"归因分布: {summary['by_attribution']}")
    print(f"判定: {'PASS' if verdict['pass'] else 'FAIL'}（{verdict['rule']}）")
    if not verdict["pass"]:
        print(
            f"  违规: {verdict.get('offenders') or list(verdict.get('boundary_violations', {}))[:10]}"
        )
    print(f"报告: {out_path}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
