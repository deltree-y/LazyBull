"""行业相关因子模块

提供基于行业的 alpha、偏离等特征
"""

import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger


def generate_industry_encoding(industry_series: pd.Series) -> Dict[str, int]:
    """生成行业编码映射
    
    Args:
        industry_series: 行业名称 Series
        
    Returns:
        行业名称到整数编码的字典
    """
    unique_industries = sorted(industry_series.dropna().unique())
    encoding = {ind: idx for idx, ind in enumerate(unique_industries, start=1)}
    return encoding


def add_industry_features(
    features_df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    ret_col: str = 'ret_1'
) -> pd.DataFrame:
    """添加行业相关特征
    
    计算：
    - industry_id: 行业整数编码
    - alpha_industry: 个股收益 - 行业平均收益
    
    Args:
        features_df: 特征DataFrame，需包含 ts_code, trade_date, ret_col
        stock_basic: 股票基础信息，需包含 ts_code, industry
        ret_col: 收益率列名，默认 'ret_1'
        
    Returns:
        DataFrame，添加了行业相关特征
    """
    logger.info("添加行业相关特征...")
    result = features_df.copy()
    
    # 检查 stock_basic 是否包含 industry 字段
    if 'industry' not in stock_basic.columns:
        raise ValueError(
            "stock_basic 数据中缺少 'industry' 字段。\n"
            "请确保已使用最新的数据拉取脚本更新 stock_basic 数据，"
            "并确保 TuShare 接口返回了 industry 字段。"
        )
    
    # 合并行业信息
    result = result.merge(
        stock_basic[['ts_code', 'industry']],
        on='ts_code',
        how='left'
    )
    
    # 检查合并后是否有 industry 缺失
    missing_industry = result['industry'].isna().sum()
    missing_industry_ts_codes = result[result['industry'].isna()]['ts_code'].unique()
    if missing_industry > 0:
        logger.warning(f"有 {missing_industry} 个样本缺少行业信息，涉及股票代码: {missing_industry_ts_codes}")
    
    # 生成 industry_id 编码（保证相同 industry 映射一致）
    industry_encoding = generate_industry_encoding(result['industry'])
    result['industry_id'] = result['industry'].map(industry_encoding)
    
    # 对于缺失行业的样本，industry_id 设为 0
    result['industry_id'] = result['industry_id'].fillna(0).astype(int)
    
    # 计算行业平均收益（按 trade_date + industry 分组）
    if ret_col in result.columns:
        industry_avg = result.groupby(['trade_date', 'industry'])[ret_col].transform('mean')
        
        # 计算 alpha_industry = 个股收益 - 行业平均收益
        result['alpha_industry'] = result[ret_col] - industry_avg
    else:
        logger.warning(f"特征中缺少 {ret_col} 列，无法计算 alpha_industry")
        result['alpha_industry'] = np.nan
    
    return result


def calculate_industry_alpha_windows(
    df: pd.DataFrame,
    ret_windows: list,
    industry_col: str = 'industry'
) -> pd.DataFrame:
    """计算多个窗口的行业 alpha

    Args:
        df: DataFrame，需包含 ts_code, trade_date, industry, ret_N (N in ret_windows)
        ret_windows: 收益率窗口列表，例如 [5, 10, 20]
        industry_col: 行业列名，默认 'industry'

    Returns:
        DataFrame，包含 ts_code, trade_date, alpha_industry_{window}
    """
    logger.info("计算多个窗口的行业 alpha...")
    result = df[['ts_code', 'trade_date']].copy()

    for window in ret_windows:
        ret_col = f'ret_{window}'
        alpha_col = f'alpha_industry_{window}'

        if ret_col in df.columns and industry_col in df.columns:
            # 计算行业平均收益
            industry_avg = df.groupby(['trade_date', industry_col])[ret_col].transform('mean')

            # 计算 alpha
            result[alpha_col] = df[ret_col] - industry_avg
        else:
            logger.warning(f"缺少 {ret_col} 或 {industry_col}，{alpha_col} 设为空")
            result[alpha_col] = np.nan

    return result


def calculate_industry_momentum_features(
    df: pd.DataFrame,
    industry_col: str = 'sw_industry',
    ret_col: str = 'ret_20',
) -> pd.DataFrame:
    """计算行业动量特征（下沉到个股）

    对每个交易日，先计算各行业的平均收益（行业动量），再对行业动量做
    百分位排名，最后将行业级特征广播回每只股票。

    输出特征：
    - ind_ret_avg: 所属行业的平均 ret_col（行业绝对动量）
    - ind_momentum_rank: 行业动量在所有行业中的百分位排名（0~1，1=最强）

    Args:
        df: 当日截面 DataFrame，需包含 ts_code, trade_date, industry_col, ret_col
        industry_col: 行业列名，默认 'sw_industry'
        ret_col: 用于计算行业动量的收益列，默认 'ret_20'

    Returns:
        DataFrame，包含 ts_code, trade_date, ind_ret_avg, ind_momentum_rank
    """
    result = df[['ts_code', 'trade_date']].copy()
    result['ind_ret_avg'] = np.nan
    result['ind_momentum_rank'] = np.nan

    if industry_col not in df.columns or ret_col not in df.columns:
        logger.debug(f"缺少 {industry_col} 或 {ret_col}，跳过行业动量特征")
        return result

    for date, grp in df.groupby('trade_date'):
        valid = grp.dropna(subset=[industry_col, ret_col])
        if len(valid) == 0:
            continue

        # 行业平均收益
        ind_avg = valid.groupby(industry_col)[ret_col].mean()

        # 行业百分位排名（0~1，1=最强）
        ind_rank = ind_avg.rank(pct=True)

        # 映射回个股
        idx = grp.index
        result.loc[idx, 'ind_ret_avg'] = grp[industry_col].map(ind_avg).values
        result.loc[idx, 'ind_momentum_rank'] = grp[industry_col].map(ind_rank).values

    n_valid = result['ind_momentum_rank'].notna().sum()
    logger.debug(f"行业动量特征: {n_valid}/{len(result)} 条有效")
    return result
