"""ML 模型评估工具模块

提供逐日评估函数，用于训练和回测场景：
- 逐日 RankIC 计算（Spearman）
- 逐日 TopK 收益评估
- 统一的日度评估接口
"""

from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_daily_rankic(
    predictions: pd.Series,
    true_returns: pd.Series
) -> float:
    """计算单日 RankIC（Spearman 秩相关）
    
    Args:
        predictions: 预测分数（Series，index 为股票代码）
        true_returns: 真实收益（Series，index 为股票代码）
        
    Returns:
        RankIC 值（Spearman 相关系数）
    """
    # 对齐两个 Series
    common_idx = predictions.index.intersection(true_returns.index)
    if len(common_idx) == 0:
        return np.nan
    
    pred_aligned = predictions.loc[common_idx]
    ret_aligned = true_returns.loc[common_idx]
    
    # 移除 NaN
    valid_mask = pred_aligned.notna() & ret_aligned.notna()
    if valid_mask.sum() < 2:  # 至少需要 2 个样本
        return np.nan
    
    pred_valid = pred_aligned[valid_mask]
    ret_valid = ret_aligned[valid_mask]
    
    # 计算 Spearman 相关
    rank_ic, _ = spearmanr(pred_valid, ret_valid)
    return rank_ic


def compute_daily_topk_returns(
    predictions: pd.Series,
    true_returns: pd.Series,
    k_values: List[int]
) -> Dict[str, float]:
    """计算单日 TopK 平均收益
    
    Args:
        predictions: 预测分数（Series，index 为股票代码）
        true_returns: 真实收益（Series，index 为股票代码）
        k_values: K 值列表（例如 [30, 100, 300]）
        
    Returns:
        字典，key 为 "Top{k}平均收益"，value 为均值
    """
    # 对齐两个 Series
    common_idx = predictions.index.intersection(true_returns.index)
    if len(common_idx) == 0:
        return {f"Top{k}平均收益": np.nan for k in k_values}
    
    pred_aligned = predictions.loc[common_idx]
    ret_aligned = true_returns.loc[common_idx]
    
    # 移除 NaN
    valid_mask = pred_aligned.notna() & ret_aligned.notna()
    pred_valid = pred_aligned[valid_mask]
    ret_valid = ret_aligned[valid_mask]
    
    if len(pred_valid) == 0:
        return {f"Top{k}平均收益": np.nan for k in k_values}
    
    # 按预测分数降序排序
    sorted_idx = pred_valid.sort_values(ascending=False).index
    
    # 计算每个 K 的 TopK 平均收益
    topk_returns = {}
    for k in k_values:
        actual_k = min(k, len(sorted_idx))
        if actual_k > 0:
            topk_idx = sorted_idx[:actual_k]
            topk_mean_return = ret_valid.loc[topk_idx].mean()
            topk_returns[f"Top{k}平均收益"] = topk_mean_return
        else:
            topk_returns[f"Top{k}平均收益"] = np.nan
    
    return topk_returns


def evaluate_predictions_by_date(
    df: pd.DataFrame,
    date_col: str,
    prediction_col: str,
    return_col: str,
    topk_values: Optional[List[int]] = None
) -> pd.DataFrame:
    """对多日预测进行逐日评估
    
    Args:
        df: DataFrame，包含日期、预测分数、真实收益列
        date_col: 日期列名
        prediction_col: 预测分数列名
        return_col: 真实收益列名
        topk_values: TopK 评估的 K 值列表，None 表示不计算 TopK
        
    Returns:
        评估结果 DataFrame，每行为一个交易日的指标
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'trade_date': ['20230101', '20230101', '20230102', '20230102'],
        ...     'ts_code': ['A', 'B', 'A', 'B'],
        ...     'pred_score': [0.8, 0.6, 0.7, 0.9],
        ...     'y_ret_5': [0.05, 0.02, 0.03, 0.06]
        ... })
        >>> results = evaluate_predictions_by_date(
        ...     df, 'trade_date', 'pred_score', 'y_ret_5', topk_values=[1]
        ... )
    """
    if topk_values is None:
        topk_values = []
    
    daily_metrics = []
    
    for trade_date in df[date_col].unique():
        day_df = df[df[date_col] == trade_date]
        
        predictions = day_df.set_index('ts_code')[prediction_col]
        true_returns = day_df.set_index('ts_code')[return_col]
        
        # 计算 RankIC
        rank_ic = compute_daily_rankic(predictions, true_returns)
        
        # 计算 TopK 收益
        topk_returns = compute_daily_topk_returns(predictions, true_returns, topk_values)
        
        # 合并指标
        metrics = {
            date_col: trade_date,
            'RankIC': rank_ic,
            '样本数': len(predictions)
        }
        metrics.update(topk_returns)
        
        daily_metrics.append(metrics)
    
    return pd.DataFrame(daily_metrics)


def summarize_daily_metrics(daily_metrics: pd.DataFrame) -> Dict[str, float]:
    """汇总逐日指标，计算均值、标准差、IR
    
    Args:
        daily_metrics: 逐日指标 DataFrame（来自 evaluate_predictions_by_date）
        
    Returns:
        汇总统计字典
    """
    summary = {}
    
    # RankIC 统计
    if 'RankIC' in daily_metrics.columns:
        rankic_series = daily_metrics['RankIC'].dropna()
        if len(rankic_series) > 0:
            summary['RankIC_均值'] = rankic_series.mean()
            summary['RankIC_标准差'] = rankic_series.std()
            if rankic_series.std() > 0:
                summary['RankIC_IR'] = rankic_series.mean() / rankic_series.std()
            else:
                summary['RankIC_IR'] = np.nan
        else:
            summary['RankIC_均值'] = np.nan
            summary['RankIC_标准差'] = np.nan
            summary['RankIC_IR'] = np.nan
    
    # TopK 收益统计
    topk_cols = [col for col in daily_metrics.columns if col.startswith('Top') and col.endswith('平均收益')]
    for col in topk_cols:
        topk_series = daily_metrics[col].dropna()
        if len(topk_series) > 0:
            summary[f"{col}_均值"] = topk_series.mean()
            summary[f"{col}_标准差"] = topk_series.std()
        else:
            summary[f"{col}_均值"] = np.nan
            summary[f"{col}_标准差"] = np.nan
    
    return summary
