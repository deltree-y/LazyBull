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


def compute_diagnostic_statistics(
    df: pd.DataFrame,
    date_col: str,
    prediction_col: str,
    return_col: str,
    topk_values: List[int]
) -> Dict[str, any]:
    """计算逐日评估诊断统计（用于排查 TopK/RankIC 不一致风险）
    
    诊断项包括：
    - 验证集全市场收益的逐日均值与标准差（跨日汇总）
    - TopK 均值相对全市场均值的提升（TopK - UniverseMean）
    - 每日可选样本数的分布（min/median/max）
    - TopK 收益的分位数（25%/50%/75%）
    
    Args:
        df: DataFrame，包含日期、预测分数、真实收益列
        date_col: 日期列名
        prediction_col: 预测分数列名
        return_col: 真实收益列名（例如 y_ret_20）
        topk_values: TopK 评估的 K 值列表
        
    Returns:
        诊断统计字典
    """
    diagnostics = {}
    
    # 1. 全市场收益的逐日统计
    daily_universe_stats = []
    for trade_date in df[date_col].unique():
        day_df = df[df[date_col] == trade_date]
        valid_returns = day_df[return_col].dropna()
        
        if len(valid_returns) > 0:
            daily_universe_stats.append({
                'date': trade_date,
                'mean': valid_returns.mean(),
                'std': valid_returns.std(),
                'count': len(valid_returns)
            })
    
    universe_stats_df = pd.DataFrame(daily_universe_stats)
    
    if len(universe_stats_df) > 0:
        diagnostics['全市场收益_逐日均值的均值'] = universe_stats_df['mean'].mean()
        diagnostics['全市场收益_逐日均值的标准差'] = universe_stats_df['mean'].std()
        diagnostics['全市场收益_逐日标准差的均值'] = universe_stats_df['std'].mean()
        
        # 样本数分布
        diagnostics['每日样本数_最小'] = int(universe_stats_df['count'].min())
        diagnostics['每日样本数_中位数'] = int(universe_stats_df['count'].median())
        diagnostics['每日样本数_最大'] = int(universe_stats_df['count'].max())
    
    # 2. 计算 TopK 收益及其相对提升
    for k in topk_values:
        topk_daily_returns = []
        
        for trade_date in df[date_col].unique():
            day_df = df[df[date_col] == trade_date]
            
            # 过滤有效样本
            valid_mask = day_df[prediction_col].notna() & day_df[return_col].notna()
            valid_df = day_df[valid_mask]
            
            if len(valid_df) == 0:
                continue
            
            # 按预测分数降序排序，选择 TopK
            sorted_df = valid_df.sort_values(prediction_col, ascending=False)
            actual_k = min(k, len(sorted_df))
            topk_df = sorted_df.head(actual_k)
            
            # 计算 TopK 平均收益
            topk_mean = topk_df[return_col].mean()
            
            # 计算全市场平均收益（用于计算提升）
            universe_mean = valid_df[return_col].mean()
            
            topk_daily_returns.append({
                'date': trade_date,
                'topk_mean': topk_mean,
                'universe_mean': universe_mean,
                'lift': topk_mean - universe_mean  # 提升 = TopK - 全市场
            })
        
        topk_returns_df = pd.DataFrame(topk_daily_returns)
        
        if len(topk_returns_df) > 0:
            # TopK 均值统计
            diagnostics[f'Top{k}_逐日均值的均值'] = topk_returns_df['topk_mean'].mean()
            diagnostics[f'Top{k}_逐日均值的标准差'] = topk_returns_df['topk_mean'].std()
            
            # 提升统计
            diagnostics[f'Top{k}_相对全市场提升_均值'] = topk_returns_df['lift'].mean()
            diagnostics[f'Top{k}_相对全市场提升_标准差'] = topk_returns_df['lift'].std()
            
            # 分位数统计（用于判断是否被极端日驱动）
            diagnostics[f'Top{k}_逐日均值_25分位'] = topk_returns_df['topk_mean'].quantile(0.25)
            diagnostics[f'Top{k}_逐日均值_50分位'] = topk_returns_df['topk_mean'].quantile(0.50)
            diagnostics[f'Top{k}_逐日均值_75分位'] = topk_returns_df['topk_mean'].quantile(0.75)
    
    return diagnostics


def print_diagnostic_report(diagnostics: Dict[str, any]) -> None:
    """打印诊断报告（格式化输出）
    
    Args:
        diagnostics: 诊断统计字典（来自 compute_diagnostic_statistics）
    """
    from loguru import logger
    
    logger.info("=" * 80)
    logger.info("逐日评估诊断报告（排查 TopK/RankIC 不一致风险）")
    logger.info("=" * 80)
    
    # 1. 全市场收益统计
    logger.info("\n【1. 全市场收益统计】")
    if '全市场收益_逐日均值的均值' in diagnostics:
        logger.info(f"  逐日均值的均值: {diagnostics['全市场收益_逐日均值的均值']:.6f}")
        logger.info(f"  逐日均值的标准差: {diagnostics['全市场收益_逐日均值的标准差']:.6f}")
        logger.info(f"  逐日标准差的均值: {diagnostics['全市场收益_逐日标准差的均值']:.6f}")
    
    # 2. 样本数分布
    logger.info("\n【2. 每日样本数分布】")
    if '每日样本数_最小' in diagnostics:
        logger.info(f"  最小: {diagnostics['每日样本数_最小']}")
        logger.info(f"  中位数: {diagnostics['每日样本数_中位数']}")
        logger.info(f"  最大: {diagnostics['每日样本数_最大']}")
    
    # 3. TopK 收益与提升
    logger.info("\n【3. TopK 收益统计与相对提升】")
    topk_keys = [k for k in diagnostics.keys() if k.startswith('Top') and '_逐日均值的均值' in k]
    for key in sorted(topk_keys):
        k_str = key.split('_')[0]  # 提取 "Top30", "Top100" 等
        logger.info(f"\n  {k_str}:")
        logger.info(f"    逐日均值的均值: {diagnostics[key]:.6f}")
        logger.info(f"    逐日均值的标准差: {diagnostics[key.replace('逐日均值的均值', '逐日均值的标准差')]:.6f}")
        logger.info(f"    相对全市场提升（均值）: {diagnostics[key.replace('逐日均值的均值', '相对全市场提升_均值')]:.6f}")
        logger.info(f"    相对全市场提升（标准差）: {diagnostics[key.replace('逐日均值的均值', '相对全市场提升_标准差')]:.6f}")
        logger.info(f"    分位数 (25%/50%/75%): {diagnostics[key.replace('逐日均值的均值', '逐日均值_25分位')]:.6f} / {diagnostics[key.replace('逐日均值的均值', '逐日均值_50分位')]:.6f} / {diagnostics[key.replace('逐日均值的均值', '逐日均值_75分位')]:.6f}")
    
    logger.info("=" * 80)

