"""权重后处理模块

提供权重限制和归一化功能，用于组合构建中的权重约束管理。
"""

from typing import Dict
import numpy as np
from loguru import logger


def cap_and_normalize_weights(
    weights: Dict[str, float],
    max_weight_per_stock: float,
    verbose: bool = False
) -> Dict[str, float]:
    """对权重进行限权并重新归一化
    
    处理逻辑：
    1. 过滤掉权重 <= 0 的股票
    2. 对每个股票的权重应用上限约束
    3. 将超出上限的权重按原比例分配给未触顶股票
    
    边界情况处理：
    - 空权重字典：返回空字典
    - 全为0或负数：返回空字典
    - NaN值：将NaN视为0并过滤掉
    - 权重和为0：返回空字典
    
    Args:
        weights: 原始权重字典 {股票代码: 权重}
        max_weight_per_stock: 单个股票的最大权重（0-1之间）
        verbose: 是否输出详细日志
        
    Returns:
        处理后的权重字典，每个权重均不超过上限；约束可行时权重和为 1，
        不可行时权重和小于 1，剩余部分保留为现金
        
    Raises:
        ValueError: 当 max_weight_per_stock 不在有效范围内
    """
    if max_weight_per_stock <= 0 or max_weight_per_stock > 1:
        raise ValueError(f"max_weight_per_stock 必须在 (0, 1] 范围内，当前值: {max_weight_per_stock}")
    
    if not weights:
        return {}
    
    # 过滤有效权重：将 NaN 视为 0，过滤掉 <= 0 的权重
    valid_weights = {}
    for stock, weight in weights.items():
        # 处理 NaN
        if np.isnan(weight):
            if verbose:
                logger.warning(f"股票 {stock} 权重为 NaN，已过滤")
            continue
        
        # 过滤 <= 0 的权重
        if weight <= 0:
            if verbose:
                logger.debug(f"股票 {stock} 权重 <= 0 ({weight:.6f})，已过滤")
            continue
        
        valid_weights[stock] = weight
    
    if not valid_weights:
        if verbose:
            logger.warning("所有权重均无效（<= 0 或 NaN），返回空字典")
        return {}
    
    remaining_stocks = list(valid_weights)
    normalized_weights: Dict[str, float] = {}
    remaining_weight = 1.0

    while remaining_stocks and remaining_weight > 0:
        source_total = sum(valid_weights[stock] for stock in remaining_stocks)
        if source_total <= 0:
            break

        proposed_weights = {
            stock: remaining_weight * valid_weights[stock] / source_total
            for stock in remaining_stocks
        }
        capped_stocks = [
            stock
            for stock, weight in proposed_weights.items()
            if weight > max_weight_per_stock + 1e-12
        ]
        if not capped_stocks:
            normalized_weights.update(proposed_weights)
            remaining_weight = 0.0
            break

        for stock in capped_stocks:
            normalized_weights[stock] = max_weight_per_stock
            remaining_stocks.remove(stock)
            remaining_weight -= max_weight_per_stock

    if remaining_stocks:
        for stock in remaining_stocks:
            normalized_weights[stock] = min(
                max_weight_per_stock,
                normalized_weights.get(stock, 0.0),
            )

    normalized_weights = {
        stock: normalized_weights[stock]
        for stock in valid_weights
        if stock in normalized_weights
    }

    unallocated_weight = max(0.0, 1.0 - sum(normalized_weights.values()))
    
    if verbose:
        logger.info(
            f"  权重后处理完成: 原始 {len(weights)} 只 → 有效 {len(valid_weights)} 只 → "
            f"限权完成（单股上限 {max_weight_per_stock:.2%}, "
            f"现金保留 {unallocated_weight:.2%}）"
        )
        # 显示前3只股票的最终权重
        sample_stocks = list(normalized_weights.keys())[:3]
        weights_str = ", ".join([f"{stock}: {normalized_weights[stock]:.4f}" for stock in sample_stocks])
        logger.info(f"  样本权重抽样: {weights_str}") 
        
    return normalized_weights


def resolve_tranche_weight_cap(
    max_weight_per_stock: float,
    tranche_capital_fraction: float,
) -> float:
    """将全组合单股上限换算为批内归一化权重上限。"""
    if max_weight_per_stock <= 0 or max_weight_per_stock > 1:
        raise ValueError(
            f"max_weight_per_stock 必须在 (0, 1] 范围内，当前值: {max_weight_per_stock}"
        )
    if tranche_capital_fraction <= 0 or tranche_capital_fraction > 1:
        raise ValueError(
            "tranche_capital_fraction 必须在 (0, 1] 范围内，"
            f"当前值: {tranche_capital_fraction}"
        )
    return min(max_weight_per_stock / tranche_capital_fraction, 1.0)
