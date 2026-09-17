#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""日频信号级配对对比（薄入口）。

把 WF 单臂对比从「14 个折净值读数」（噪声带 5~8pp）提升到「逐日配对读数」
（14 折 × 约 1700 个交易日），用于**筛选**：这个改动是否真的动了信号。

使用示例：
    # 1) 标定噪声带：同配置仅换种子
    python scripts/compare_wf_signal_metrics.py ^
        --baseline data/walk_forward/batches/factor_ab_20260916_071157_base ^
        --arm data/walk_forward/batches/factor_ab_20260916_123214_noiseB0 ^
        --arm-label B0 --baseline-label base

    # 2) 候选臂台账（可一次传多个 --arm）
    python scripts/compare_wf_signal_metrics.py --baseline <base> \
        --arm <A> --arm <markers> --arm-label A_hard --arm-label markers

口径边界：本口径**不含**交易成本、调仓节奏、Kelly 仓位与路径效应，只作筛选；
最终裁决仍看链式净值判据（ΔMaxDD 为主判据）。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger  # noqa: E402

from scripts.compare.signal_metrics import (  # noqa: E402
    DEFAULT_BLOCK_DAYS,
    DEFAULT_TOPK_LIST,
    compare_signal_metrics,
)

BOUNDARY_NOTE = (
    "口径边界：日频信号级代理不含交易成本/调仓节奏/Kelly 仓位/路径效应，"
    "仅用于筛选；最终裁决以链式净值判据（ΔMaxDD 为主判据）为准。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="日频信号级配对对比（只读 WF 产物）")
    parser.add_argument("--baseline", required=True, help="基线批次目录或 raw 目录")
    parser.add_argument("--arm", action="append", required=True, help="实验臂目录（可重复）")
    parser.add_argument("--arm-label", action="append", default=[], help="与 --arm 顺序对应的标签")
    parser.add_argument("--baseline-label", default="", help="基线标签，默认取目录名")
    parser.add_argument(
        "--topk",
        type=int,
        default=list(DEFAULT_TOPK_LIST),
        nargs="+",
        help=f"Top-K 档位，默认 {list(DEFAULT_TOPK_LIST)}",
    )
    parser.add_argument(
        "--block-days",
        type=int,
        default=list(DEFAULT_BLOCK_DAYS),
        nargs="+",
        help=f"分块自举块长（交易日），默认 {list(DEFAULT_BLOCK_DAYS)}",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="重采样次数，默认 1000")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="重采样种子，默认 42")
    parser.add_argument("--ci", type=float, default=0.95, help="区间置信水平，默认 0.95")
    parser.add_argument(
        "--allow-state-mismatch",
        action="store_true",
        help="允许数据态不一致（仅限「仅 git 标记不同、数据水位一致」的显式例外）",
    )
    parser.add_argument(
        "--allow-day-mismatch",
        action="store_true",
        help="允许两臂交易日集合不完全一致（默认报错，配对设计失效时不得静默继续）",
    )
    parser.add_argument(
        "--out", default="", help="产物目录（默认 data/reports/wf_signal_compare/<时间戳>）"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_signal_metrics(
        baseline=Path(args.baseline),
        arms=[Path(arm) for arm in args.arm],
        arm_labels=args.arm_label,
        topk_list=args.topk,
        block_days_list=args.block_days,
        n_resamples=args.bootstrap,
        seed=args.bootstrap_seed,
        ci=args.ci,
        allow_state_mismatch=args.allow_state_mismatch,
        allow_day_mismatch=args.allow_day_mismatch,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("data/reports/wf_signal_compare") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "日频信号对比.csv"
    folds_path = out_dir / "逐折信号对比.csv"
    result["指标表"].to_csv(metrics_path, index=False, encoding="utf-8-sig")
    result["逐折表"].to_csv(folds_path, index=False, encoding="utf-8-sig")
    note_path = out_dir / "口径说明.md"
    note_path.write_text(
        "\n".join(
            [
                "# 日频信号级配对对比",
                "",
                f"- 基线：{result['基线']}（{result['基线目录']}）",
                f"- 实验臂：{', '.join(result['臂'])}",
                f"- 块长（交易日）：{list(args.block_days)}；重采样次数：{args.bootstrap}",
                f"- 区间水平：{args.ci}",
                "",
                BOUNDARY_NOTE,
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info(f"产物目录: {out_dir}")
    for name, path in (
        ("日频信号对比", metrics_path),
        ("逐折信号对比", folds_path),
        ("口径说明", note_path),
    ):
        logger.info(f"  {name} -> {path}")
    logger.info(BOUNDARY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
