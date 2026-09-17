#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""折子集链式对比入口（薄入口）。

用途：消融实验只跑部分折（``walk_forward.py --selected-split-indices``）时，
把基线与之放在**同一折子集**上重算链式指标对比；只读既有产物，不重训、不写回。

使用示例：
    # 单臂对比最近 6 折（0-based 折号 8..13）
    python scripts/compare_wf_fold_subset.py \
        --baseline data/walk_forward/batches/wf_batch_A \
        --arm data/walk_forward/batches/wf_batch_B --splits 8-13

    # 多臂 + 指定产物目录
    python scripts/compare_wf_fold_subset.py --baseline <dir> --arm <dir1> --arm <dir2> \
        --splits 8-13 --out data/reports/wf_fold_subset/runA

本文件为薄入口：CLI 解析 + 编排 scripts/compare/fold_subset.py。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from scripts.compare.fold_subset import (
    bootstrap_table,
    compare_runs,
    load_run,
    parse_split_spec,
    validate_alignment,
)
from src.lazybull.common.config import get_data_root, get_logs_dir
from src.lazybull.common.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description="折子集链式对比（只读既有 walk-forward 产物）")
    parser.add_argument(
        "--baseline", required=True, help="基线目录（batch 目录 / raw 目录 / chain_nav 文件）"
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="对照臂目录，可重复传入；标签默认取目录名",
    )
    parser.add_argument("--arm-label", action="append", default=[], help="与 --arm 顺序对应的标签")
    parser.add_argument("--baseline-label", default="", help="基线标签，默认取目录名")
    parser.add_argument("--splits", default="", help="折子集规格（如 8-13）；默认全部折")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="折级自举次数（0 = 跳过；判定口径见 docs/factor_pruning_ab_protocol.md）",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="自举随机种子")
    parser.add_argument(
        "--allow-state-mismatch",
        action="store_true",
        help="数据态 ID 不一致时仅告警（仅限“仅 git 标记不同、数据水位一致”的显式例外）",
    )
    parser.add_argument(
        "--out", default="", help="产物目录，默认 data/reports/wf_fold_subset/<时间戳>"
    )
    return parser


def main() -> None:
    """编排折子集对比流程。"""
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(
        log_level="INFO",
        log_file=str(Path(get_logs_dir()) / f"wf_fold_subset_{timestamp}.log"),
    )

    baseline = load_run(Path(args.baseline), label=args.baseline_label or None)
    arms = []
    for index, path in enumerate(args.arm):
        label = args.arm_label[index] if index < len(args.arm_label) else None
        arms.append(load_run(Path(path), label=label))
    if not arms:
        raise SystemExit("至少需要一个 --arm 对照臂")

    validate_alignment([baseline] + arms, allow_state_mismatch=args.allow_state_mismatch)
    splits = parse_split_spec(args.splits) if args.splits else None

    result = compare_runs(
        baseline, arms, splits, allow_state_mismatch=args.allow_state_mismatch
    )
    summary = result["summary"]
    folds = result["folds"]

    out_dir = (
        Path(args.out)
        if args.out
        else Path(get_data_root()) / "reports" / "wf_fold_subset" / timestamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "折子集对比.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out_dir / "逐折对比.csv", index=False, encoding="utf-8-sig")

    target_splits = sorted(folds["折序号"].unique().tolist())
    windows = baseline.windows[baseline.windows["split_index"].isin(target_splits)]
    window_text = ", ".join(
        f"折{int(r.split_index)} {r.test_start}~{r.test_end}" for r in windows.itertuples()
    )
    arm_text = ", ".join(f"`{run.label}`（{run.directory}）" for run in arms)
    lines = [
        "# 折子集链式对比说明",
        "",
        f"- 生成时间：{timestamp}",
        f"- 基线：`{baseline.label}`（{baseline.directory}）",
        f"- 对照臂：{arm_text}",
        f"- 折子集：{target_splits}",
        f"- 数据态 ID：{baseline.data_state_id or '缺失'}",
        f"- 折窗口：{window_text}",
        "",
        "> 口径：子集指标 = 过滤目标折后净值归一化到起点，复用 `ml/walk_forward/chain_metrics.py`；",
        "> 逐折指标按折内起止净值计算。跨折边界收益不计入交易日数，与全周期口径一致。",
        "> 数据态不可复用时应重新冻结数据态复跑，禁止跨数据态比较。",
        f"> 判据自举：折级有放回重采样 {args.bootstrap} 次（种子 {args.bootstrap_seed}），"
        "判定规则见 `docs/factor_pruning_ab_protocol.md`（ΔCAGR/Δ夏普 自举区间下限 > 0、ΔMaxDD ≥ 0、逐折同向 ≥ 70%）。",
    ]
    (out_dir / "折子集对比说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(f"折子集对比完成，产物目录: {out_dir}")
    with pd.option_context(
        "display.width", 220, "display.max_columns", 40, "display.float_format", "{:.4f}".format
    ):
        logger.info("汇总表:\n" + summary.to_string(index=False))
        logger.info("逐折收益:\n" + result["fold_returns"].round(4).to_string())

    if args.bootstrap > 0:
        criteria = bootstrap_table(
            baseline,
            arms,
            target_splits,
            n_boot=args.bootstrap,
            seed=args.bootstrap_seed,
            allow_state_mismatch=args.allow_state_mismatch,
        )
        criteria.to_csv(out_dir / "判据结论.csv", index=False, encoding="utf-8-sig")
        logger.info("预登记判据（折级自举）:\n" + criteria.to_string(index=False))


if __name__ == "__main__":
    main()
