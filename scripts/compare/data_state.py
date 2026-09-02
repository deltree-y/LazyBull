# -*- coding: utf-8 -*-
"""数据态一致性检查：按数据态 ID 分组，跨数据态对比时输出显式告警。

背景：walk-forward 实验的结论只在同一数据态（git 版本 + 数据水位）内成立；
数据修复、增量下载或代码提交都会让跨日期的复跑结果不可比。
本模块在对比报告生成时检查运行的数据态分布，避免把数据态漂移误读为配置差异。
"""

from typing import Dict, List

import pandas as pd
from loguru import logger

from scripts.compare.constants import COL_NAMES

# comp_df 已重命名为中文列
_STATE_COL = COL_NAMES["data_state_id"]
_RUN_COL = COL_NAMES["wf_run_id"]
_GIT_COL = COL_NAMES["git_commit"]
_DAILY_COL = COL_NAMES["data_daily_latest"]
_UNKNOWN_STATE = "未知(历史运行)"


def _display_state(value) -> str:
    """缺失数据态（历史运行无血缘记录）统一显示为未知态。"""
    if pd.isna(value) or str(value).strip().lower() in ("", "none", "nan"):
        return _UNKNOWN_STATE
    return str(value).strip()


def summarize_data_states(comp_df: pd.DataFrame) -> Dict[str, List[dict]]:
    """按数据态 ID 分组运行明细：{数据态ID: [运行信息字典]}。"""
    groups: Dict[str, List[dict]] = {}
    for _, row in comp_df.iterrows():
        state_id = _display_state(row.get(_STATE_COL))
        groups.setdefault(state_id, []).append(
            {
                "run_id": str(row.get(_RUN_COL, "?")),
                "git_commit": row.get(_GIT_COL),
                "daily_latest": row.get(_DAILY_COL),
            }
        )
    return groups


def warn_cross_data_state(comp_df: pd.DataFrame) -> bool:
    """检查对比表内运行的数据态分布；存在多个数据态时输出告警。

    Returns:
        True 表示存在跨数据态对比（结果需谨慎解读）。
    """
    if comp_df.empty or _STATE_COL not in comp_df.columns:
        return False

    groups = summarize_data_states(comp_df)
    if len(groups) <= 1:
        only_state = next(iter(groups), None)
        logger.info(f"数据态一致性检查通过: 全部运行同属数据态 {only_state or _UNKNOWN_STATE}")
        return False

    logger.warning("=" * 70)
    logger.warning(f"检测到 {len(groups)} 个不同数据态，跨数据态的指标差异不具可比性！")
    for state_id, runs in sorted(groups.items()):
        git_commits = sorted(
            {str(r.get("git_commit")) for r in runs if pd.notna(r.get("git_commit"))}
        )
        daily_watermarks = sorted(
            {str(r.get("daily_latest")) for r in runs if pd.notna(r.get("daily_latest"))}
        )
        run_preview = ", ".join(r["run_id"] for r in runs[:5])
        suffix = " ..." if len(runs) > 5 else ""
        logger.warning(
            f"  数据态 {state_id}: {len(runs)} 个运行 | git={git_commits or '未知'} | "
            f"raw/daily水位={daily_watermarks or '未知'}"
        )
        logger.warning(f"    运行: {run_preview}{suffix}")
    logger.warning("建议: 配置差异只在同一数据态内比较；跨数据态结论需冻结当前数据态复跑对照确认。")
    logger.warning("=" * 70)
    return True
