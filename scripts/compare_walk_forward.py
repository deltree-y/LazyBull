# -*- coding: utf-8 -*-
"""
Walk-forward 实验对比脚本

功能：
- 无参时自动扫描 data/walk_forward/raw/ 与 data/walk_forward/batches/*/raw/ 两类来源
- 按 wf_run_id 分组，跨 split 聚合各项指标
- 生成对比表格（行=实验，列=聚合指标+训练参数）
- 输出到 Excel 文件

使用示例：
    python scripts/compare_walk_forward.py
    python scripts/compare_walk_forward.py --data-root ./data
    python scripts/compare_walk_forward.py --raw-dir ./data/walk_forward/raw --output ./data/walk_forward/wf_comparison.csv

本文件为薄入口：CLI 参数解析 + 从 scripts/compare/ 子包 re-export 公共 API。
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.lazybull.common.config import get_data_root
from src.lazybull.common.logger import setup_logger

from scripts.compare.constants import (
    BATCH_EXPERIMENT_CORE_COLS,
    BATCH_EXPERIMENT_EXCLUDED_PARAM_COLS,
    BATCH_EXPERIMENT_PARAM_CANDIDATES,
    CANDIDATE_MIN_CHAIN_CAGR_WORST,
    CANDIDATE_MIN_CHAIN_MAX_DRAWDOWN,
    CANDIDATE_MIN_EFFECTIVE_PAIR_CONTEXTS,
    CANDIDATE_MIN_MODEL_ALPHA,
    CANDIDATE_SCORE_CONFIG,
    COL_NAMES,
    MODEL_ALPHA_SCORE_CONFIG,
    MODEL_PARAM_KEYS,
    PAIR_CONTEXT_KEYS,
    PARAM_COLS,
    SCORE_CONFIG,
    SEED_STABILITY_EXCLUDED_MODEL_KEYS,
    SUMMARY_CSV_DTYPE,
    TRADE_PARAM_KEYS,
    TRADE_ROBUST_SCORE_CONFIG,
    TRADE_YIELD_SCORE_CONFIG,
)
from scripts.compare.loading import (
    build_auto_compare_jobs,
    load_all_summaries,
    load_all_summaries_from_raw_dirs,
    load_chain_metrics,
)
from scripts.compare.aggregate import (
    aggregate_run,
    build_comparison_table,
    build_period_stability_table,
    sort_by_latest_run_time,
)
from scripts.compare.scoring import (
    build_live_candidate_score_table,
    build_model_alpha_score_table,
    build_model_seed_stability_table,
    build_trade_param_score_table,
    compute_composite_score,
    compute_selection_score,
)
from scripts.compare.metrics_desc import build_metric_descriptions
from scripts.compare.detail_display import (
    build_split_detail_table,
    compact_experiment_sheet_for_display,
    reorder_comparison_columns,
    sort_by_run_time,
)
from scripts.compare.excel import (
    format_excel_output,
    print_comparison_table,
    write_empty_report,
)
from scripts.compare.report import (
    generate_comparison_report,
    run_auto_compare_jobs,
)

def main():
    parser = argparse.ArgumentParser(description="Walk-forward 实验对比分析")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录；未指定时使用 configs/base.yaml 中的 data.root",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="walk_forward 汇总CSV目录，默认 {data_root}/walk_forward/raw",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="对比Excel输出路径，默认 {data_root}/walk_forward/wf_comparison.xlsx",
    )
    args = parser.parse_args()

    setup_logger()

    effective_data_root = Path(args.data_root or get_data_root())

    logger.info("=" * 70)
    logger.info("Walk-forward 实验对比分析")
    logger.info("=" * 70)

    if args.raw_dir is None and args.output is None:
        output_paths = run_auto_compare_jobs(effective_data_root)
        if len(output_paths) == 0:
            logger.error("raw 与 batches 均没有可用数据，退出")
            return
        logger.info(f"自动扫描完成，共生成 {len(output_paths)} 个对比文件")
    else:
        raw_dir = (
            Path(args.raw_dir) if args.raw_dir else effective_data_root / "walk_forward" / "raw"
        )
        output_path = (
            Path(args.output)
            if args.output
            else effective_data_root / "walk_forward" / "wf_comparison.xlsx"
        )
        if not generate_comparison_report(
            [raw_dir], output_path, "single", data_root=effective_data_root
        ):
            logger.error("没有可用数据，退出")
            return

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
