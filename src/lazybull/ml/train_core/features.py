# -*- coding: utf-8 -*-
"""features：train_core 拆分模块。"""

from loguru import logger
from pathlib import Path
from src.lazybull.common.config import get_data_root
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import json
import math
import numpy as np
import pandas as pd

from .constants import (
    FACTOR_EXCLUDE_LIST_FILE,
)

_factor_exclude_cache: Dict[Path, Set[str]] = {}


def _load_factor_exclude_list(
    models_dir: Optional[Path] = None,
    exclude_file: Optional[Path] = None,
) -> Set[str]:
    """加载因子排除列表（带缓存）

    从 data/models/factor_exclude_list.json 读取排除的因子名称。
    该文件由 scripts/ana/generate_factor_exclude_list.py 生成。

    严格模式：factor_prune 启用意味着本次训练必须执行因子精简，
    清单缺失或非法时直接抛异常终止训练，禁止静默降级为"全部因子保留"
    （静默跳过会产生与无精简完全相同的实验结果，浪费整轮 walk-forward）。

    Args:
        models_dir: 模型目录，为 None 时自动从 config 推断
        exclude_file: 显式排除清单路径；未提供时读取模型目录下的默认清单

    Returns:
        排除的因子名称集合

    Raises:
        FileNotFoundError: 排除清单文件不存在
        ValueError: 排除清单内容非法（JSON 解析失败或缺少 exclude_factors）
    """
    if exclude_file is None:
        if models_dir is None:
            from src.lazybull.common.config import get_data_root

            data_root = Path(get_data_root())
            models_dir = data_root / "models"
        exclude_file = models_dir / FACTOR_EXCLUDE_LIST_FILE
    exclude_file = Path(exclude_file).resolve()

    if exclude_file in _factor_exclude_cache:
        return _factor_exclude_cache[exclude_file]

    if not exclude_file.exists():
        raise FileNotFoundError(
            f"因子精简已启用但排除清单不存在: {exclude_file}。"
            f"请先运行 scripts/ana/generate_factor_exclude_list.py 生成生产默认清单，"
            f"或通过 --factor-exclude-file 显式指定已存在的清单文件（注意使用相对项目根的路径）。"
        )

    try:
        with open(exclude_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"因子排除清单加载失败: {exclude_file}，{e}") from e

    if not isinstance(data, dict) or not isinstance(data.get("exclude_factors"), list):
        raise ValueError(
            f"因子排除清单格式非法: {exclude_file}，"
            f"缺少 exclude_factors 列表字段。"
            f"该文件应由 scripts/ana/generate_factor_exclude_list.py 生成。"
        )

    exclude_list = data.get("exclude_factors", [])
    _factor_exclude_cache[exclude_file] = set(exclude_list)
    logger.info(
        f"已加载因子排除列表: {exclude_file}，"
        f"{len(_factor_exclude_cache[exclude_file])} 个因子 "
        f"(min_icir={data.get('min_icir')}, "
        f"min_coverage={data.get('min_coverage')})"
    )
    return _factor_exclude_cache[exclude_file]


def filter_stable_features(
    df_train: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    date_col: str = "trade_date",
    n_splits: int = 3,
    min_abs_ic: float = 0.02,
) -> Tuple[List[str], Dict]:
    """筛选在所有时间段截面IC方向一致且显著的特征

    将训练集按时间等分成 n_splits 段，每段计算各特征与标签的截面 Spearman IC 均值。
    仅保留所有段IC方向一致且平均 |IC| 超过阈值的特征。

    Args:
        df_train: 训练集 DataFrame（已完成标签变换）
        feature_columns: 候选特征列名列表
        label_column: 标签列名
        date_col: 日期列名
        n_splits: 将训练集按时间等分的段数，默认 3
        min_abs_ic: 平均 |IC| 的最低阈值，默认 0.02

    Returns:
        (stable_features, filter_info) 元组：
            - stable_features: 通过筛选的特征列名列表
            - filter_info: 筛选统计信息字典
    """
    dates = sorted(df_train[date_col].unique())
    n_dates = len(dates)
    if n_dates < n_splits * 10:
        logger.warning(
            f"特征稳定性筛选: 训练集仅 {n_dates} 个交易日，不足以分成 {n_splits} 段，跳过筛选"
        )
        return feature_columns, {"skipped": True, "reason": "交易日不足"}

    split_size = n_dates // n_splits

    # 每个时段计算各特征的截面IC均值
    ic_matrix: Dict[str, List[float]] = {col: [] for col in feature_columns}

    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < n_splits - 1 else n_dates
        split_dates = set(dates[start:end])
        split_df = df_train[df_train[date_col].isin(split_dates)]

        # 向量化计算：逐日 rank 后用 corr 算出各特征的截面IC
        # 先做最小样本与零方差校验，避免触发 numpy 的无效自由度告警
        daily_ics: Dict[str, List[float]] = {col: [] for col in feature_columns}
        for _, group in split_df.groupby(date_col):
            if len(group) < 30:
                continue
            ranked = group[feature_columns + [label_column]].rank()
            label_rank = ranked[label_column]
            for col in feature_columns:
                pair = pd.concat([ranked[col], label_rank], axis=1).dropna()
                if len(pair) < 2:
                    continue
                if pair.iloc[:, 0].std(ddof=1) == 0 or pair.iloc[:, 1].std(ddof=1) == 0:
                    continue
                corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                if not np.isnan(corr):
                    daily_ics[col].append(corr)

        for col in feature_columns:
            ics = daily_ics[col]
            ic_matrix[col].append(np.mean(ics) if ics else 0.0)

    # 筛选条件：所有段IC符号一致 且 平均|IC| >= min_abs_ic
    stable_features = []
    removed_details = []
    for col in feature_columns:
        ics = ic_matrix[col]
        nonzero_signs = [np.sign(ic) for ic in ics if abs(ic) > 1e-6]

        if len(nonzero_signs) == 0:
            removed_details.append((col, ics, "IC全为零"))
            continue

        all_same_sign = all(s == nonzero_signs[0] for s in nonzero_signs)
        avg_abs_ic = np.mean([abs(ic) for ic in ics])

        if all_same_sign and avg_abs_ic >= min_abs_ic:
            stable_features.append(col)
        else:
            reason = "IC方向不一致" if not all_same_sign else f"|IC|={avg_abs_ic:.4f}<{min_abs_ic}"
            removed_details.append((col, ics, reason))

    filter_info = {
        "skipped": False,
        "total_features": len(feature_columns),
        "stable_count": len(stable_features),
        "removed_count": len(removed_details),
        "removed_details": removed_details,
    }

    logger.info(
        f"特征稳定性筛选: {len(feature_columns)} → {len(stable_features)} 个特征"
        f"（移除 {len(removed_details)} 个不稳定特征）"
    )
    if removed_details:
        for col, ics, reason in removed_details:
            ic_str = ", ".join(f"{ic:+.4f}" for ic in ics)
            logger.debug(f"  移除 {col}: [{ic_str}] — {reason}")

    return stable_features, filter_info

def _format_feature_importance_compact(
    feat_imp: pd.Series,
    n_cols: int = 4,
    float_format: str = "%.3f",
) -> str:
    """将特征重要性 Series 格式化为紧凑的多列字符串，减少显示行数。"""
    n = len(feat_imp)
    if n == 0:
        return "(空)"
    per_col = math.ceil(n / n_cols)
    max_name_len = max(len(str(name)) for name in feat_imp.index)
    lines: List[str] = []
    for row_idx in range(per_col):
        parts: List[str] = []
        for col_idx in range(n_cols):
            item_idx = col_idx * per_col + row_idx
            if item_idx < n:
                name = str(feat_imp.index[item_idx])
                score = float_format % feat_imp.iloc[item_idx]
                parts.append(f"{name:<{max_name_len}}  {score}")
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)
