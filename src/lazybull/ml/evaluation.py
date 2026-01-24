"""机器学习模型评估模块

提供IC/RankIC等量化因子评估指标的计算工具。
支持按日IC序列计算、统计指标、分位数分析等。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr


def calculate_daily_ic(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    date_col: str = 'trade_date',
    method: str = 'pearson'
) -> pd.Series:
    """计算按日IC序列
    
    对每个交易日，计算预测值与真实标签的相关系数。
    
    Args:
        df: 数据DataFrame，需包含 date_col, pred_col, label_col
        pred_col: 预测值列名
        label_col: 真实标签列名
        date_col: 日期列名，默认 'trade_date'
        method: 相关系数方法，'pearson' 或 'spearman'，默认 'pearson'
        
    Returns:
        pd.Series: 按日IC序列，index为日期，value为IC值
    """
    daily_ics = {}
    
    for date, group in df.groupby(date_col):
        # 获取有效样本（同时有预测值和标签）
        valid_mask = group[pred_col].notna() & group[label_col].notna()
        
        if valid_mask.sum() < 2:
            # 有效样本数不足，无法计算相关系数
            daily_ics[date] = np.nan
            continue
        
        pred_values = group.loc[valid_mask, pred_col]
        label_values = group.loc[valid_mask, label_col]
        
        # 计算相关系数
        if method == 'pearson':
            ic = pred_values.corr(label_values)
        elif method == 'spearman':
            ic, _ = spearmanr(pred_values, label_values)
        else:
            raise ValueError(f"不支持的相关系数方法: {method}")
        
        daily_ics[date] = ic
    
    return pd.Series(daily_ics)


def calculate_ic_statistics(
    daily_ic_series: pd.Series,
    include_quantiles: bool = True
) -> Dict[str, float]:
    """计算IC序列的统计指标
    
    Args:
        daily_ic_series: 按日IC序列
        include_quantiles: 是否包含分位数统计，默认True
        
    Returns:
        统计指标字典，包含：
        - ic_mean: IC均值
        - ic_std: IC标准差
        - ic_ir: IC信息比率 (IC均值 / IC标准差)
        - ic_positive_rate: IC>0的比例（胜率）
        - ic_min: 最小IC
        - ic_max: 最大IC
        - ic_p10, ic_p50, ic_p90: 分位数（可选）
    """
    # 过滤NaN值
    valid_ics = daily_ic_series.dropna()
    
    if len(valid_ics) == 0:
        logger.warning("IC序列为空，无法计算统计指标")
        return {
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'ic_ir': np.nan,
            'ic_positive_rate': np.nan
        }
    
    stats = {
        'ic_mean': float(valid_ics.mean()),
        'ic_std': float(valid_ics.std()),
        'ic_positive_rate': float((valid_ics > 0).sum() / len(valid_ics)),
        'ic_min': float(valid_ics.min()),
        'ic_max': float(valid_ics.max()),
    }
    
    # 计算IR（信息比率）
    if stats['ic_std'] > 1e-8:
        stats['ic_ir'] = stats['ic_mean'] / stats['ic_std']
    else:
        stats['ic_ir'] = np.nan
    
    # 分位数统计
    if include_quantiles:
        stats['ic_p10'] = float(valid_ics.quantile(0.10))
        stats['ic_p50'] = float(valid_ics.quantile(0.50))
        stats['ic_p90'] = float(valid_ics.quantile(0.90))
    
    return stats


def evaluate_model_ic(
    y_true: pd.Series,
    y_pred: pd.Series,
    trade_dates: Optional[pd.Series] = None,
    return_daily_series: bool = False
) -> Dict[str, any]:
    """评估模型的IC指标（完整版）
    
    计算IC、RankIC及其统计指标。
    如果提供trade_dates，则计算按日IC序列及详细统计。
    
    Args:
        y_true: 真实标签，pd.Series
        y_pred: 预测值，pd.Series或numpy.ndarray
        trade_dates: 交易日期序列，可选。如提供，将计算按日IC
        return_daily_series: 是否返回按日IC序列，默认False
        
    Returns:
        评估指标字典，包含：
        - ic: 总体IC（Pearson）
        - rank_ic: 总体RankIC（Spearman）
        - 如果提供trade_dates，还包含：
          - ic_mean, ic_std, ic_ir: IC序列统计
          - rank_ic_mean, rank_ic_std, rank_ic_ir: RankIC序列统计
          - ic_p10, ic_p50, ic_p90: IC分位数
          - rank_ic_p10, rank_ic_p50, rank_ic_p90: RankIC分位数
        - 如果return_daily_series=True，还包含：
          - ic_series: 按日IC序列
          - rank_ic_series: 按日RankIC序列
    """
    # 确保y_pred是pd.Series
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_true.index)
    
    # 过滤有效样本
    valid_mask = y_true.notna() & y_pred.notna()
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) < 2:
        logger.warning("有效样本数不足，无法计算IC")
        return {
            'ic': np.nan,
            'rank_ic': np.nan
        }
    
    # 计算总体IC和RankIC
    ic = float(y_true_valid.corr(y_pred_valid))
    rank_ic, _ = spearmanr(y_true_valid, y_pred_valid)
    rank_ic = float(rank_ic)
    
    results = {
        'ic': ic,
        'rank_ic': rank_ic
    }
    
    # 如果提供了日期，计算按日IC序列
    if trade_dates is not None:
        # 确保trade_dates与y_true对齐
        if len(trade_dates) != len(y_true):
            logger.warning("trade_dates长度与y_true不匹配，跳过按日IC计算")
        else:
            # 构建DataFrame
            df = pd.DataFrame({
                'trade_date': trade_dates,
                'y_true': y_true,
                'y_pred': y_pred
            })
            
            # 计算按日IC序列（Pearson）
            ic_series = calculate_daily_ic(
                df,
                pred_col='y_pred',
                label_col='y_true',
                date_col='trade_date',
                method='pearson'
            )
            
            # 计算按日RankIC序列（Spearman）
            rank_ic_series = calculate_daily_ic(
                df,
                pred_col='y_pred',
                label_col='y_true',
                date_col='trade_date',
                method='spearman'
            )
            
            # 计算IC统计指标
            ic_stats = calculate_ic_statistics(ic_series, include_quantiles=True)
            rank_ic_stats = calculate_ic_statistics(rank_ic_series, include_quantiles=True)
            
            # 添加前缀并合并
            for key, value in ic_stats.items():
                results[key] = value
            
            for key, value in rank_ic_stats.items():
                results[f'rank_{key}'] = value
            
            # 如果需要返回序列
            if return_daily_series:
                results['ic_series'] = ic_series
                results['rank_ic_series'] = rank_ic_series
    
    return results


def print_ic_evaluation_report(
    metrics: Dict[str, any],
    title: str = "IC评估报告"
):
    """打印格式化的IC评估报告
    
    Args:
        metrics: 评估指标字典（由 evaluate_model_ic 返回）
        title: 报告标题
    """
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)
    
    # 总体IC
    if 'ic' in metrics:
        logger.info(f"总体 IC（信息系数）: {metrics['ic']:.4f}")
    
    if 'rank_ic' in metrics:
        logger.info(f"总体 RankIC（排序IC）: {metrics['rank_ic']:.4f}")
    
    # 按日IC统计
    if 'ic_mean' in metrics:
        logger.info("")
        logger.info("-" * 70)
        logger.info("按日IC序列统计")
        logger.info("-" * 70)
        logger.info(f"  IC均值: {metrics['ic_mean']:.4f}")
        logger.info(f"  IC标准差: {metrics['ic_std']:.4f}")
        logger.info(f"  IC信息比率(IR): {metrics['ic_ir']:.4f}")
        logger.info(f"  IC胜率(>0): {metrics['ic_positive_rate']:.2%}")
        logger.info(f"  IC范围: [{metrics['ic_min']:.4f}, {metrics['ic_max']:.4f}]")
        
        if 'ic_p10' in metrics:
            logger.info(f"  IC分位数: P10={metrics['ic_p10']:.4f}, "
                       f"P50={metrics['ic_p50']:.4f}, P90={metrics['ic_p90']:.4f}")
    
    # 按日RankIC统计
    if 'rank_ic_mean' in metrics:
        logger.info("")
        logger.info("-" * 70)
        logger.info("按日RankIC序列统计")
        logger.info("-" * 70)
        logger.info(f"  RankIC均值: {metrics['rank_ic_mean']:.4f}")
        logger.info(f"  RankIC标准差: {metrics['rank_ic_std']:.4f}")
        logger.info(f"  RankIC信息比率(IR): {metrics['rank_ic_ir']:.4f}")
        logger.info(f"  RankIC胜率(>0): {metrics['rank_ic_positive_rate']:.2%}")
        logger.info(f"  RankIC范围: [{metrics['rank_ic_min']:.4f}, {metrics['rank_ic_max']:.4f}]")
        
        if 'rank_ic_p10' in metrics:
            logger.info(f"  RankIC分位数: P10={metrics['rank_ic_p10']:.4f}, "
                       f"P50={metrics['rank_ic_p50']:.4f}, P90={metrics['rank_ic_p90']:.4f}")
    
    logger.info("=" * 70)
    logger.info("提示：")
    logger.info("  - IC > 0.03 通常认为有一定预测能力")
    logger.info("  - RankIC > 0.05 说明排序能力较好（对选股策略更重要）")
    logger.info("  - IR（信息比率）> 1.0 说明稳定性较好")
    logger.info("=" * 70)
