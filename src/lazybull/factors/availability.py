# -*- coding: utf-8 -*-
"""运行时可用性标记：把「结构性缺失」显式化（不写回任何特征产物）。

背景（2026-09-16 因子诊断）：
    一致预期（研报覆盖 44~63%）、两融（**标的资格**：2013 年 22% → 2026 年 71%）、
    基金持仓、业绩快报等因子的缺失是**结构性的**（资格名单 / 覆盖 / 披露口径），
    不是随机噪声；且缺口只是规模代理（有值/缺失标签中位差≈0，规模分位差 +0.75~0.90）。

做法：**运行时派生**。训练矩阵组装（`ml/train_core/prepare.py`）与推理特征准备
（`signals/ml_signal.py`）共用本模块，按现有列派生 `has_*` 标记，**不修改 cs_train /
cs_infer 任何产物**（无需重建 3000+ 分区）。

标记语义：``has_x = 1`` 当且仅当该股票当日**至少一个**来源列有值（其余为 0）。

train/serve 一致：推理侧只补**模型 ``feature_columns`` 中出现过的**标记（由模型驱动），
因此不需要额外配置项，也不会出现「训练有、推理无 → 静默 NaN」的偏差。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

# 标记 -> 来源列（任一非空即置 1）
AVAILABILITY_MARKER_SOURCES: Dict[str, Tuple[str, ...]] = {
    # 卖方一致预期：无分析师覆盖 / 无目标价披露的股票为结构性缺失
    "has_cons_coverage": (
        "cons_analyst_count_30d",
        "cons_eps_yield_fy1",
        "cons_eps_yield_fy2",
    ),
    # 两融：非标的证券无余额（资格名单历史扩张，且 rqye 口径 2024-07 后退化）
    "has_margin_balance": (
        "rzye_chg_5",
        "rzye_chg_20",
        "rqye_rzye_ratio",
    ),
    # 基金持仓：无基金持仓披露的股票为结构性缺失
    "has_fund_holding": (
        "fund_hold_ratio",
        "fund_count",
    ),
    # 业绩快报：仅发布快报的股票有值
    "has_express_data": (
        "express_profit_yoy",
        "express_roe",
        "express_surprise",
    ),
}


def availability_marker_names() -> List[str]:
    """全部可用性标记名（顺序稳定）。"""
    return list(AVAILABILITY_MARKER_SOURCES.keys())


def derive_availability_markers(
    frame: pd.DataFrame,
    wanted: Optional[Iterable[str]] = None,
    log_prefix: str = "",
) -> List[str]:
    """按需派生可用性标记（**就地**修改 frame），返回本次新增的标记列名。

    Args:
        frame: 训练矩阵 / 推理特征 DataFrame（需含来源列）。
        wanted: 期望的标记名集合；``None`` 表示全量（训练侧）。推理侧传入模型的
            ``feature_columns`` 即可天然对齐。
        log_prefix: 日志前缀（如 ``[推理]``），便于定位来源。

    Returns:
        实际新增的标记列名（来源列缺失或已存在的标记不计入）。
    """
    candidates = availability_marker_names()
    if wanted is not None:
        wanted_set = {str(name) for name in wanted}
        candidates = [name for name in candidates if name in wanted_set]
    new_columns: Dict[str, pd.Series] = {}
    for name in candidates:
        if name in frame.columns:
            continue
        sources = [col for col in AVAILABILITY_MARKER_SOURCES[name] if col in frame.columns]
        if not sources:
            logger.warning(
                f"{log_prefix}可用性标记 {name} 跳过：来源列 "
                f"{AVAILABILITY_MARKER_SOURCES[name]} 均不在数据中"
                "（若模型训练时含该标记，本次计算会因缺列被补成 NaN，形成 train/serve 偏差；"
                "请确认推理侧特征分区包含对应家族列）"
            )
            continue
        new_columns[name] = frame[sources].notna().any(axis=1).astype("int8")
    if not new_columns:
        return []
    # 一次性多列赋值（逐列赋值会让大 DataFrame 产生分片告警与额外内存开销）
    marker_df = pd.DataFrame(new_columns, index=frame.index)
    frame[list(marker_df.columns)] = marker_df
    return list(marker_df.columns)


def ensure_availability_markers(
    frame: pd.DataFrame, wanted: Optional[Sequence[str]] = None
) -> List[str]:
    """推理侧入口：只为模型需要的标记补齐（等价于 :func:`derive_availability_markers`）。"""
    return derive_availability_markers(frame, wanted=wanted, log_prefix="[推理]")
