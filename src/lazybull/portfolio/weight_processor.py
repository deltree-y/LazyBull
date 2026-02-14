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
    3. 重新归一化使权重和为 1.0
    
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
        处理后的权重字典，权重和为 1.0，每个权重 <= max_weight_per_stock
        
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
    
    # 迭代应用权重上限和归一化（确保最终所有权重都不超过上限）
    # 迭代原因：限权后归一化可能导致其他权重超过上限，需要多次迭代直到收敛
    current_weights = valid_weights.copy()
    total_capped_count = 0
    max_iterations = 100  # 防止无限循环
    
    for iteration in range(max_iterations):
        # 应用权重上限
        capped_weights = {}
        capped_count = 0
        for stock, weight in current_weights.items():
            if weight > max_weight_per_stock:
                capped_weights[stock] = max_weight_per_stock
                capped_count += 1
            else:
                capped_weights[stock] = weight
        
        total_capped_count += capped_count
        
        # 重新归一化
        total_weight = sum(capped_weights.values())
        
        if total_weight == 0:
            if verbose:
                logger.warning("限权后权重和为 0，返回空字典")
            return {}
        
        normalized_weights = {
            stock: weight / total_weight
            for stock, weight in capped_weights.items()
        }
        
        # 检查是否收敛（所有权重都不超过上限）
        if all(w <= max_weight_per_stock + 1e-10 for w in normalized_weights.values()):
            break
        
        # 继续下一次迭代
        current_weights = normalized_weights
    else:
        # 达到最大迭代次数仍未收敛，记录警告但返回当前结果
        if verbose:
            logger.warning(f"权重限制迭代未在 {max_iterations} 次内收敛，返回当前结果")
    
    if verbose:
        logger.info(
            f"  权重后处理完成: 原始 {len(weights)} 只 → 有效 {len(valid_weights)} 只 → "
            f"归一化完成（单股上限 {max_weight_per_stock:.2%}）"
        )
        # 显示前3只股票的最终权重
        sample_stocks = list(normalized_weights.keys())[:3]
        weights_str = ", ".join([f"{stock}: {normalized_weights[stock]:.4f}" for stock in sample_stocks])
        logger.info(f"  样本权重抽样: {weights_str}") 
        
    return normalized_weights
