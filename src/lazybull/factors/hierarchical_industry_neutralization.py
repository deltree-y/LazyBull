"""分层行业回退中性化模块

实现 L3→L2→L1→全市场 的四层回退统计，适用于：
- hierarchical_zscore: 指标的行业内 Z-Score（前缀 zscore_）
- hierarchical_demean: 收益率/标签的行业内去均值（前缀 neu_）

回退规则（每列单独判断）：
  当日 L3 行业内 tradable==1 样本数 < min_group_size → 回退到对应 L2 行业统计
  L2 仍不足 → 回退到 L1 行业统计
  L1 仍不足 → 回退到全市场（tradable==1）统计

统计过程严格无前瞻：仅使用当日截面数据。
"""

from typing import List

import numpy as np
import pandas as pd
from loguru import logger


def _build_level_stats(
    col_values: pd.Series,
    group_keys: pd.Series,
    tradable_mask: pd.Series,
    min_group_size: int,
) -> dict:
    """预计算每个分组的可用统计量（均值、标准差）

    仅对 tradable==1 的样本计算统计量；组内可交易样本数 < min_group_size 时不纳入字典
    （调用方会跳过该组，向上层回退）。

    Args:
        col_values: 待统计的特征列
        group_keys: 行业 code 列（L1/L2/L3 均可）
        tradable_mask: 布尔 Series，True=可交易
        min_group_size: 最小组内可交易样本数

    Returns:
        dict: {group_key: (mean, std)}，标准差为 0 或 NaN 的组不纳入
    """
    if False:   #原始实现
        stats = {}
        for key, grp_idx in group_keys.groupby(group_keys, dropna=True).groups.items():
            tradable_vals = col_values.loc[
                grp_idx.intersection(col_values.index[tradable_mask])
            ].dropna()
            if len(tradable_vals) < max(min_group_size, 2):
                continue
            m = float(tradable_vals.mean())
            s = float(tradable_vals.std())  # ddof=1，与 normalization.py 保持一致
            if np.isnan(s) or s == 0:
                continue
            stats[key] = (m, s)
        return stats
    else:   #Gemini 优化实现：一次性全局过滤 + 向量化聚合，极大提升效率
        # 1. 预先全局过滤数据：同时满足“可交易”和“非空”
        # 相比在循环里做 dropna 和 intersection，这里是一次性布尔索引，速度极快
        valid_mask = tradable_mask.fillna(False).astype(bool) & col_values.notna()
        
        valid_vals = col_values[valid_mask]
        valid_keys = group_keys[valid_mask]
        
        # 防御性检查：如果没有有效数据，直接返回
        if valid_vals.empty:
            return {}

        # 2. 向量化聚合：一次性算出所有组的 mean, std, count
        # Pandas 底层会用 C/Cython 层面实现加速
        agg_df = valid_vals.groupby(valid_keys).agg(
            mean='mean',
            std='std',
            count='count'
        )
        
        # 3. 批量过滤不符合条件的组
        min_count = max(min_group_size, 2)
        valid_agg = agg_df[
            (agg_df['count'] >= min_count) & 
            (agg_df['std'].notna()) & 
            (agg_df['std'] > 0)
        ]
        
        # 4. 组装结果字典
        # 使用 itertuples 遍历 DataFrame 是最高效的做法之一
        return {
            row.Index: (row.mean, row.std)
            for row in valid_agg.itertuples()
        }

def _build_level_means(
    col_values: pd.Series,
    group_keys: pd.Series,
    tradable_mask: pd.Series,
    min_group_size: int,
) -> dict:
    """预计算每个分组的可用均值（用于 demean）

    Args:
        col_values: 待统计的特征列
        group_keys: 行业 code 列
        tradable_mask: 布尔 Series
        min_group_size: 最小组内可交易样本数

    Returns:
        dict: {group_key: mean}
    """
    means = {}
    for key, grp_idx in group_keys.groupby(group_keys, dropna=True).groups.items():
        tradable_vals = col_values.loc[
            grp_idx.intersection(col_values.index[tradable_mask])
        ].dropna()
        if len(tradable_vals) < min_group_size:
            continue
        m = float(tradable_vals.mean())
        if np.isnan(m):
            continue
        means[key] = m
    return means


def hierarchical_zscore(
    df: pd.DataFrame,
    columns: List[str],
    l3_col: str,
    l2_col: str,
    l1_col: str,
    tradable_col: str = "tradable",
    min_group_size: int = 5,
    prefix: str = "zscore_",
) -> pd.DataFrame:
    """分层行业内 Z-Score 标准化（L3→L2→L1→全市场回退）

    对每列独立计算：仅使用 tradable==1 的样本确定各层统计量；
    最终对全部行（含不可交易）施加 Z-Score，结果存入 ``{prefix}{col}`` 列。

    Args:
        df: 输入截面 DataFrame（单日）
        columns: 待标准化列名列表
        l3_col: L3 行业 code 列名（如 'sw_industry_code'）
        l2_col: L2 行业 code 列名（如 'sw_l2_code'）
        l1_col: L1 行业 code 列名（如 'sw_l1_code'）
        tradable_col: 可交易标记列（1=可交易）
        min_group_size: 最小组内可交易样本数，默认 5
        prefix: 输出列前缀，默认 'zscore_'

    Returns:
        添加了 zscore 列的 DataFrame（原列不变）
    """
    result = df.copy()

    if tradable_col in result.columns:
        tradable_mask = (result[tradable_col] == 1)
    else:
        logger.warning(f"hierarchical_zscore: 未找到 {tradable_col} 列，使用全部样本")
        tradable_mask = pd.Series(True, index=result.index)

    for col in columns:
        if col not in result.columns:
            logger.warning(f"hierarchical_zscore: 列 {col} 不存在，跳过")
            continue

        output_col = f"{prefix}{col}"

        # 全市场统计（最终兜底）
        global_tradable = result.loc[tradable_mask, col].dropna()
        if len(global_tradable) < 2:
            logger.debug(f"hierarchical_zscore: {col} 全市场可交易样本不足，跳过")
            result[output_col] = np.nan
            continue

        global_mean = float(global_tradable.mean())
        global_std = float(global_tradable.std())
        if np.isnan(global_std) or global_std == 0:
            logger.warning(f"hierarchical_zscore: {col} 全市场标准差为 0 或 NaN，跳过")
            result[output_col] = np.nan
            continue

        # 各层统计量预计算
        l3_stats = (
            _build_level_stats(result[col], result[l3_col], tradable_mask, min_group_size)
            if l3_col in result.columns
            else {}
        )
        l2_stats = (
            _build_level_stats(result[col], result[l2_col], tradable_mask, min_group_size)
            if l2_col in result.columns
            else {}
        )
        l1_stats = (
            _build_level_stats(result[col], result[l1_col], tradable_mask, min_group_size)
            if l1_col in result.columns
            else {}
        )

        if False:   #原始实现：循环匹配每行，效率较低
            # 向量化匹配：优先级 L3 > L2 > L1 > 全市场
            l3_mean = result[l3_col].map({k: v[0] for k, v in l3_stats.items()}) if l3_col in result.columns else pd.Series(np.nan, index=result.index)
            l3_std  = result[l3_col].map({k: v[1] for k, v in l3_stats.items()}) if l3_col in result.columns else pd.Series(np.nan, index=result.index)

            l2_mean = result[l2_col].map({k: v[0] for k, v in l2_stats.items()}) if l2_col in result.columns else pd.Series(np.nan, index=result.index)
            l2_std  = result[l2_col].map({k: v[1] for k, v in l2_stats.items()}) if l2_col in result.columns else pd.Series(np.nan, index=result.index)

            l1_mean = result[l1_col].map({k: v[0] for k, v in l1_stats.items()}) if l1_col in result.columns else pd.Series(np.nan, index=result.index)
            l1_std  = result[l1_col].map({k: v[1] for k, v in l1_stats.items()}) if l1_col in result.columns else pd.Series(np.nan, index=result.index)

            # 分层回退：L3 有效 → 用 L3；否则 L2；否则 L1；否则全市场
            use_mean = l3_mean.fillna(l2_mean).fillna(l1_mean).fillna(global_mean)
            use_std  = l3_std.fillna(l2_std).fillna(l1_std).fillna(global_std)

            result[output_col] = (result[col] - use_mean) / use_std
        else:   # Gemini 优化实现：全局预分配 Numpy 数组 + 分层覆盖更新，极大提升效率
            # 1. 提前分配好 Numpy 数组，用兜底的 global 值进行初始化
            n_rows = len(result)
            use_mean = np.full(n_rows, global_mean, dtype=float)
            use_std  = np.full(n_rows, global_std, dtype=float)

            # 2. 定义按优先级升序排列的层级结构（L1 -> L2 -> L3）
            # 升序的好处：高优先级（如 L3）的有效值会在最后直接覆盖低优先级的值
            levels = [
                (l1_col, l1_stats),
                (l2_col, l2_stats),
                (l3_col, l3_stats)
            ]

            for lvl_col, lvl_stats in levels:
                # 只有当列存在且字典不为空时才处理，彻底避免生成全 NaN 的 Series
                if lvl_col in result.columns and lvl_stats:
                    # 拆解字典（当前层级只遍历这一遍）
                    mean_dict = {k: v[0] for k, v in lvl_stats.items()}
                    std_dict  = {k: v[1] for k, v in lvl_stats.items()}
                    
                    # 使用 Pandas 高效的 map 后直接取 .values 转换为 Numpy 数组
                    col_data = result[lvl_col]
                    curr_mean = col_data.map(mean_dict).values
                    curr_std  = col_data.map(std_dict).values
                    
                    # 寻找映射成功的有效位置（非 NaN 的位置）
                    valid_mask = ~np.isnan(curr_std)
                    
                    # Numpy 布尔索引原地覆盖更新，零拷贝开销！
                    use_mean[valid_mask] = curr_mean[valid_mask]
                    use_std[valid_mask]  = curr_std[valid_mask]

            # 3. 最终运算直接使用 Numpy 数组，避开 Pandas 的索引对齐，最后再赋值回 DataFrame
            result[output_col] = (result[col].values - use_mean) / use_std

        # 统计回退情况
        if l3_col in result.columns:
            l3_hit = result[l3_col].map(l3_stats).notna().sum()
            l2_hit = (~result[l3_col].map(l3_stats).notna()) & result.get(l2_col, pd.Series()).map(l2_stats).notna() if l2_col in result.columns else pd.Series(False, index=result.index)
            logger.debug(
                f"hierarchical_zscore [{col}]: L3 命中 {l3_hit} 行，"
                f"L2 回退 {int(l2_hit.sum()) if hasattr(l2_hit, 'sum') else 0} 行"
            )

    return result


def hierarchical_demean(
    df: pd.DataFrame,
    columns: List[str],
    l3_col: str,
    l2_col: str,
    l1_col: str,
    tradable_col: str = "tradable",
    min_group_size: int = 5,
    prefix: str = "neu_",
) -> pd.DataFrame:
    """分层行业内去均值中性化（L3→L2→L1→全市场回退）

    对每列独立计算：仅使用 tradable==1 的样本确定各层均值；
    最终对全部行（含不可交易）执行 x - mean，结果存入 ``{prefix}{col}`` 列。
    适用于收益率/标签列（neu_ 前缀）。

    Args:
        df: 输入截面 DataFrame（单日）
        columns: 待去均值列名列表
        l3_col: L3 行业 code 列名
        l2_col: L2 行业 code 列名
        l1_col: L1 行业 code 列名
        tradable_col: 可交易标记列（1=可交易）
        min_group_size: 最小组内可交易样本数，默认 5
        prefix: 输出列前缀，默认 'neu_'

    Returns:
        添加了去均值列的 DataFrame（原列不变）
    """
    result = df.copy()

    if tradable_col in result.columns:
        tradable_mask = (result[tradable_col] == 1)
    else:
        logger.warning(f"hierarchical_demean: 未找到 {tradable_col} 列，使用全部样本")
        tradable_mask = pd.Series(True, index=result.index)

    for col in columns:
        if col not in result.columns:
            logger.warning(f"hierarchical_demean: 列 {col} 不存在，跳过")
            continue

        output_col = f"{prefix}{col}"

        # 全市场均值（最终兜底）
        global_tradable = result.loc[tradable_mask, col].dropna()
        if len(global_tradable) == 0:
            logger.warning(f"hierarchical_demean: {col} 全市场可交易样本为空，跳过")
            result[output_col] = np.nan
            continue

        global_mean = float(global_tradable.mean())
        if np.isnan(global_mean):
            result[output_col] = np.nan
            continue

        # 各层均值预计算
        l3_means = (
            _build_level_means(result[col], result[l3_col], tradable_mask, min_group_size)
            if l3_col in result.columns
            else {}
        )
        l2_means = (
            _build_level_means(result[col], result[l2_col], tradable_mask, min_group_size)
            if l2_col in result.columns
            else {}
        )
        l1_means = (
            _build_level_means(result[col], result[l1_col], tradable_mask, min_group_size)
            if l1_col in result.columns
            else {}
        )

        # 向量化匹配：L3 > L2 > L1 > 全市场
        l3_mean_s = result[l3_col].map(l3_means) if l3_col in result.columns else pd.Series(np.nan, index=result.index)
        l2_mean_s = result[l2_col].map(l2_means) if l2_col in result.columns else pd.Series(np.nan, index=result.index)
        l1_mean_s = result[l1_col].map(l1_means) if l1_col in result.columns else pd.Series(np.nan, index=result.index)

        use_mean = l3_mean_s.fillna(l2_mean_s).fillna(l1_mean_s).fillna(global_mean)
        result[output_col] = result[col] - use_mean

    return result
