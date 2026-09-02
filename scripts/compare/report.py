# -*- coding: utf-8 -*-
"""对比报告生成与无参自动扫描。"""

from typing import Optional

from loguru import logger
import pandas as pd

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.compare.constants import (
    COL_NAMES,
    SCORE_CONFIG,
)
from scripts.compare.data_state import warn_cross_data_state
from scripts.compare.loading import (
    build_auto_compare_jobs,
    load_all_summaries_from_raw_dirs,
)
from scripts.compare.aggregate import (
    build_comparison_table,
    build_period_stability_table,
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
    _score_sheet_or_placeholder,
    format_excel_output,
    write_empty_report,
)


def generate_comparison_report(
    raw_dirs: list[Path],
    output_path: Path,
    source_label: str,
    data_root: Optional[Path] = None,
    write_empty_output: bool = False,
) -> bool:
    """加载指定来源的汇总CSV并写出对比 Excel。"""
    logger.info("-" * 70)
    logger.info(f"来源标签:   {source_label}")
    if len(raw_dirs) == 1:
        logger.info(f"汇总CSV目录: {raw_dirs[0]}")
    else:
        logger.info(f"汇总CSV目录: 共 {len(raw_dirs)} 个 raw 目录（来源: {source_label}）")
    logger.info(f"输出路径:     {output_path}")

    all_df = load_all_summaries_from_raw_dirs(raw_dirs, data_root=data_root)
    if all_df.empty:
        logger.warning(f"[{source_label}] 没有可用数据，跳过")
        if write_empty_output:
            write_empty_report(output_path, source_label)
            logger.info(f"[{source_label}] 已生成空白占位文件: {output_path}")
            return True
        return False

    comp_df = build_comparison_table(all_df)
    if comp_df.empty:
        logger.warning(f"[{source_label}] 构建对比表失败，跳过")
        return False

    # 数据态一致性检查：跨数据态的运行混在一张表时必须显式告警
    if warn_cross_data_state(comp_df):
        logger.warning(f"[{source_label}] 本次对比表包含多个数据态，相关结论需冻结数据态复跑确认")

    comp_df.insert(1, "综合得分", compute_composite_score(comp_df))
    comp_df.insert(2, "选股综合得分", compute_selection_score(comp_df))
    comp_df = reorder_comparison_columns(comp_df)
    logger.info(
        f"综合得分计算完成（参与评分指标数: {sum(1 for k, _, _ in SCORE_CONFIG if COL_NAMES.get(k) in comp_df.columns)}）"
    )
    logger.info("选股综合得分计算完成（指标: RankIC均值30%/ICIR30%/Top30超额40%）")

    desc_df = build_metric_descriptions()
    split_df = build_split_detail_table(all_df)
    logger.info(f"逐Split明细表: {len(split_df)} 行")
    period_stability_df = build_period_stability_table(comp_df)
    logger.info(f"跨时间段稳定性表: {len(period_stability_df)} 行")
    model_alpha_df = build_model_alpha_score_table(comp_df)
    logger.info(f"模型Alpha评分表: {len(model_alpha_df)} 行")
    model_seed_stability_df = build_model_seed_stability_table(comp_df, model_alpha_df)
    logger.info(f"模型Seed稳定性表: {len(model_seed_stability_df)} 行")
    trade_score_df = build_trade_param_score_table(comp_df)
    logger.info(f"交易参数收益评分表: {len(trade_score_df)} 行")
    candidate_df = build_live_candidate_score_table(comp_df, model_alpha_df, trade_score_df)
    logger.info(f"实盘候选评分表: {len(candidate_df)} 行")
    comp_df = sort_by_run_time(comp_df)
    display_comp_df = compact_experiment_sheet_for_display(comp_df, source_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _score_sheet_or_placeholder(
            candidate_df,
            "实盘候选评分",
            "缺少模型Alpha评分或有效交易参数配对评分，无法计算实盘候选分。",
        ).to_excel(writer, sheet_name="实盘候选评分", index=False)
        _score_sheet_or_placeholder(
            model_alpha_df,
            "模型Alpha评分",
            "缺少可聚合的模型参数或选股指标，无法计算模型Alpha分。",
        ).to_excel(writer, sheet_name="模型Alpha评分", index=False)
        _score_sheet_or_placeholder(
            model_seed_stability_df,
            "模型Seed稳定性",
            "缺少 seed 字段或模型Alpha评分，无法按忽略 seed 的超参分组统计稳定性。",
        ).to_excel(writer, sheet_name="模型Seed稳定性", index=False)
        _score_sheet_or_placeholder(
            trade_score_df,
            "交易参数收益评分",
            "没有相同模型参数 + 相同时间段下的多交易参数候选，无法计算配对交易参数评分。",
        ).to_excel(writer, sheet_name="交易参数收益评分", index=False)
        display_comp_df.to_excel(writer, sheet_name="实验对比", index=False)
        if not period_stability_df.empty:
            period_stability_df.to_excel(writer, sheet_name="跨时间段稳定性", index=False)
        desc_df.to_excel(writer, sheet_name="指标说明", index=False)
        if not split_df.empty:
            split_df.to_excel(writer, sheet_name="逐Split明细", index=False)
        format_excel_output(writer.book, desc_df)
    logger.info(f"[{source_label}] 对比表已保存: {output_path}（{len(comp_df)} 个实验）")
    return True


def run_auto_compare_jobs(data_root: Path) -> list[Path]:
    """无参模式：自动扫描 raw 与 batches 两类来源。"""
    output_paths: list[Path] = []
    jobs = build_auto_compare_jobs(data_root)
    logger.info("无参模式：自动扫描 raw 与 batches 目录")
    for job in jobs:
        if generate_comparison_report(
            job["raw_dirs"],
            job["output_path"],
            job["label"],
            data_root=data_root,
            write_empty_output=True,
        ):
            output_paths.append(job["output_path"])
    return output_paths
