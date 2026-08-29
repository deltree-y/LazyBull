"""行业约束模块

提供基于行业的持仓数量约束功能，用于组合构建中的行业分散化管理。
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger

from ..common.config import get_shenwan_level, normalize_shenwan_level


def load_industry_mapping(
    shenwan_industry: pd.DataFrame,
    verbose: bool = False,
    shenwan_level: Optional[str] = None,
) -> Dict[str, str]:
    """从申万行业分类数据加载行业映射

    Args:
        shenwan_industry: 申万行业分类 DataFrame，必须包含 ts_code 列
            以及 sw_industry/sw_l1/sw_l2/sw_l3/sw_name 中至少一列
        verbose: 是否输出详细日志
        shenwan_level: 主行业口径层级，支持 l1/l2/l3；未传时从项目配置读取

    Returns:
        行业映射字典 {股票代码: 行业名称}，行业缺失的股票映射到 "未知行业"

    Raises:
        ValueError: 当 shenwan_industry 缺少必需列时
    """
    if shenwan_industry is None or shenwan_industry.empty:
        if verbose:
            logger.warning("shenwan_industry 为空，返回空行业映射")
        return {}

    if 'ts_code' not in shenwan_industry.columns:
        raise ValueError("shenwan_industry 必须包含 ts_code 列")

    if not {'sw_industry', 'sw_l1', 'sw_l2', 'sw_name', 'sw_l3'}.intersection(shenwan_industry.columns):
        raise ValueError("shenwan_industry 必须包含 sw_industry、sw_l1、sw_l2、sw_name 或 sw_l3 列")

    resolved_level = (
        normalize_shenwan_level(shenwan_level)
        if shenwan_level is not None
        else get_shenwan_level()
    )
    preferred_columns = {
        'l1': ['sw_l1', 'sw_industry', 'sw_name', 'sw_l2', 'sw_l3'],
        'l2': ['sw_l2', 'sw_industry', 'sw_name', 'sw_l3', 'sw_l1'],
        'l3': ['sw_l3', 'sw_industry', 'sw_l2', 'sw_name', 'sw_l1'],
    }
    selected_column = next(
        (col for col in preferred_columns[resolved_level] if col in shenwan_industry.columns),
        None,
    )

    if selected_column is None:
        raise ValueError("shenwan_industry 缺少可用的行业名称列")

    if verbose:
        logger.info(f"行业映射使用申万{resolved_level.upper()}主口径列: {selected_column}")

    # 构建映射：将 NaN/None 映射为 "未知行业"
    industry_mapping = {}
    unknown_count = 0

    for _, row in shenwan_industry.iterrows():
        ts_code = row['ts_code']
        industry = row.get(selected_column)
        
        # 处理缺失值
        if pd.isna(industry) or industry == '' or industry is None:
            industry_mapping[ts_code] = "未知行业"
            unknown_count += 1
        else:
            industry_mapping[ts_code] = str(industry)
    
    if verbose:
        unique_industries = set(industry_mapping.values())
        logger.info(
            f"行业映射加载完成: 共 {len(industry_mapping)} 只股票, "
            f"{len(unique_industries)} 个行业, 未知行业 {unknown_count} 只"
        )
    
    return industry_mapping


def apply_industry_constraint(
    ranked_candidates: List[Tuple[str, float]],
    industry_mapping: Dict[str, str],
    max_per_industry: int,
    target_n: int,
    verbose: bool = False,
    initial_industry_counts: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, float]]:
    """应用行业持仓数量约束，从排序候选中选择股票
    
    选股逻辑：
    1. 按候选列表顺序（已按分数降序排列）从高到低遍历
    2. 检查该股票所属行业当前持仓数量
    3. 如果未达到行业上限，则选入；否则跳过
    4. 继续遍历直到选满 target_n 只或候选耗尽
    
    行业缺失处理：
    - 行业缺失（未在 industry_mapping 中）的股票归为 "未知行业"
    - "未知行业" 同样受 max_per_industry 限制
    
    Args:
        ranked_candidates: 排序后的候选列表 [(股票代码, 分数), ...]，按分数降序排列
        industry_mapping: 行业映射字典 {股票代码: 行业名称}
        max_per_industry: 单个行业最大持仓数量，必须 > 0
        target_n: 目标选股数量
        verbose: 是否输出详细日志
        initial_industry_counts: 已有持仓及预留买单的行业占用数量
        
    Returns:
        满足行业约束的股票列表 [(股票代码, 分数), ...]
        
    Raises:
        ValueError: 当 max_per_industry <= 0 时
    """
    if max_per_industry <= 0:
        raise ValueError(f"max_per_industry 必须 > 0，当前值: {max_per_industry}")
    
    if not ranked_candidates:
        return []
    
    if target_n <= 0:
        return []
    
    selected = []
    industry_counts = dict(initial_industry_counts or {})  # {行业: 已占用及已选数量}
    skipped_by_industry = {}  # {行业: 跳过数量}（用于日志）
    
    for stock, score in ranked_candidates:
        # 查找行业，缺失则归为 "未知行业"
        industry = industry_mapping.get(stock, "未知行业")
        
        # 检查行业约束
        current_count = industry_counts.get(industry, 0)
        
        if current_count < max_per_industry:
            # 未达到上限，选入
            selected.append((stock, score))
            industry_counts[industry] = current_count + 1
            
            # 达到目标数量，停止
            if len(selected) >= target_n:
                break
        else:
            # 超过行业上限，跳过
            skipped_by_industry[industry] = skipped_by_industry.get(industry, 0) + 1
    
    if verbose:
        logger.info(
            f"  行业约束选股: 候选 {len(ranked_candidates)} 只 → 选中 {len(selected)}/{target_n} 只 "
            f"(行业上限 {max_per_industry})"
        )
        
        # 显示各行业持仓分布
        if industry_counts:
            industry_dist = sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)
            top_industries = industry_dist[:5]  # 显示前5个行业
            dist_str = ', '.join([f"{ind}({cnt})" for ind, cnt in top_industries])
            logger.info(f"    行业分布（前5）: {dist_str}")
        
        # 显示被跳过的行业统计
        if skipped_by_industry:
            skipped_industries = sorted(skipped_by_industry.items(), key=lambda x: x[1], reverse=True)
            top_skipped = skipped_industries[:3]  # 显示前3个被跳过最多的行业
            skipped_str = ', '.join([f"{ind}({cnt})" for ind, cnt in top_skipped])
            logger.info(f"    因行业限制跳过（前3）: {skipped_str}")
    
    return selected
