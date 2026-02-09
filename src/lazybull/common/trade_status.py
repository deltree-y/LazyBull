"""交易状态检查工具模块

提供检查股票涨跌停、停牌状态的工具函数，用于选股和交易阶段的过滤。
"""

from typing import Optional, Dict, Any

import pandas as pd
from loguru import logger


def is_suspended_by_suspend_df(
    ts_code: str,
    trade_date: str,
    suspend_df: pd.DataFrame
) -> bool:
    """基于 suspend 数据判断股票是否停牌
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期（YYYYMMDD格式字符串）
        suspend_df: 停牌数据，需包含 ts_code, trade_date, suspend_type 列
                   suspend_type: 'S' 表示停牌，'R' 表示复牌/恢复交易
        
    Returns:
        True 表示停牌，False 表示未停牌或复牌
        
    注意:
        - 若当日记录 suspend_type == 'S' => 停牌 True
        - 若当日记录 suspend_type == 'R' => 复牌/可交易 False
        - 若当日无记录 => 视为未停牌 False
    """
    try:
        if suspend_df is None or suspend_df.empty:
            return False
        
        mask = (suspend_df['ts_code'] == ts_code) & (suspend_df['trade_date'] == trade_date)
        if mask.sum() == 0:
            # 当日无停牌记录，视为未停牌
            return False
        
        row = suspend_df[mask].iloc[0]
        suspend_type = row.get('suspend_type', '')
        
        if suspend_type == 'S':
            # 停牌
            return True
        elif suspend_type == 'R':
            # 复牌/恢复交易
            return False
        else:
            # 其他类型，视为未停牌
            return False
    except Exception as e:
        logger.warning(f"检查停牌状态时出错 {ts_code} {trade_date}: {e}")
        return False


def is_suspended(
    ts_code: str,
    trade_date: str,
    quote_data: pd.DataFrame,
    suspend_df: Optional[pd.DataFrame] = None
) -> bool:
    """检查股票是否停牌
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期（YYYYMMDD格式字符串）
        quote_data: 行情数据，需包含 is_suspended 列
        suspend_df: （可选）停牌数据，若提供则优先使用
        
    Returns:
        True 表示停牌，False 表示未停牌
        
    注意:
        - 若提供 suspend_df，则以其为准（suspend_type='S'为停牌，'R'为复牌，无记录为未停牌）
        - 若未提供 suspend_df，则从 quote_data 的 is_suspended 列判断
        - 最后降级：quote_data 缺行时假定停牌（保守策略）
    """
    # 优先使用 suspend_df
    if suspend_df is not None and not suspend_df.empty:
        return is_suspended_by_suspend_df(ts_code, trade_date, suspend_df)
    
    # 降级到 quote_data
    try:
        mask = (quote_data['ts_code'] == ts_code) & (quote_data['trade_date'] == trade_date)
        if mask.sum() == 0:
            logger.debug(f"未找到 {ts_code} 在 {trade_date} 的行情数据，假定停牌")
            return True
        
        row = quote_data[mask].iloc[0]
        if 'is_suspended' in row:
            return bool(row['is_suspended'] == 1)
        
        # 备用方案：检查成交量
        if 'vol' in row:
            return bool(row['vol'] <= 0 or pd.isna(row['vol']))
        
        return False
    except Exception as e:
        logger.warning(f"检查停牌状态时出错 {ts_code} {trade_date}: {e}")
        return False


def is_limit_up(
    ts_code: str,
    trade_date: str,
    quote_data: pd.DataFrame
) -> bool:
    """检查股票是否涨停
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期（YYYYMMDD格式字符串）
        quote_data: 行情数据，需包含 is_limit_up 列
        
    Returns:
        True 表示涨停，False 表示未涨停
    """
    try:
        mask = (quote_data['ts_code'] == ts_code) & (quote_data['trade_date'] == trade_date)
        if mask.sum() == 0:
            logger.warning(f"未找到 {ts_code} 在 {trade_date} 的行情数据，假定未涨停")
            return False
        
        row = quote_data[mask].iloc[0]
        if 'is_limit_up' in row:
            return bool(row['is_limit_up'] == 1)
        
        return False
    except Exception as e:
        logger.warning(f"检查涨停状态时出错 {ts_code} {trade_date}: {e}")
        return False


def is_limit_down(
    ts_code: str,
    trade_date: str,
    quote_data: pd.DataFrame
) -> bool:
    """检查股票是否跌停
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期（YYYYMMDD格式字符串）
        quote_data: 行情数据，需包含 is_limit_down 列
        
    Returns:
        True 表示跌停，False 表示未跌停
    """
    try:
        mask = (quote_data['ts_code'] == ts_code) & (quote_data['trade_date'] == trade_date)
        if mask.sum() == 0:
            logger.debug(f"未找到 {ts_code} 在 {trade_date} 的行情数据，假定未跌停")
            return False
        
        row = quote_data[mask].iloc[0]
        if 'is_limit_down' in row:
            return bool(row['is_limit_down'] == 1)
        
        return False
    except Exception as e:
        logger.warning(f"检查跌停状态时出错 {ts_code} {trade_date}: {e}")
        return False


def is_tradeable(
    ts_code: str,
    trade_date: str,
    quote_data: pd.DataFrame,
    action: str = 'buy',
    suspend_df: Optional[pd.DataFrame] = None
) -> tuple[bool, Optional[str]]:
    """检查股票是否可交易
    
    综合检查停牌、涨跌停状态，判断股票是否可交易。
    
    交易规则：
    - 停牌：买卖均不可交易
    - 涨停：买入不可交易（难以成交），卖出可交易
    - 跌停：卖出不可交易（难以成交），买入可交易
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期（YYYYMMDD格式字符串）
        quote_data: 行情数据
        action: 操作类型，'buy' 或 'sell'
        suspend_df: （可选）停牌数据，若提供则优先使用
        
    Returns:
        (可交易标志, 不可交易原因)
        - True, None: 可交易
        - False, "停牌": 停牌不可交易
        - False, "涨停": 涨停买入不可交易
        - False, "跌停": 跌停卖出不可交易
    """
    if quote_data.empty:
        logger.warning(f"行情数据为空，假定股票可交易 {ts_code} {trade_date}")
        return True, None

    # 检查停牌
    if is_suspended(ts_code, trade_date, quote_data, suspend_df):
        return False, "停牌"
    
    # 检查涨跌停
    if action == 'buy':
        # 买入时涨停难以成交
        if is_limit_up(ts_code, trade_date, quote_data):
            return False, "涨停"
    elif action == 'sell':
        # 卖出时跌停难以成交
        if is_limit_down(ts_code, trade_date, quote_data):
            return False, "跌停"
    
    return True, None


def get_trade_status_info(
    ts_code: str,
    trade_date: str,
    quote_data: pd.DataFrame,
    suspend_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """获取股票的完整交易状态信息
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期（YYYYMMDD格式字符串）
        quote_data: 行情数据
        suspend_df: （可选）停牌数据，若提供则优先使用
        
    Returns:
        包含交易状态的字典：
        {
            'is_suspended': bool,
            'is_limit_up': bool,
            'is_limit_down': bool,
            'can_buy': bool,
            'can_sell': bool,
            'close': float or None,
            'pct_chg': float or None
        }
    """
    if quote_data.empty:
        logger.warning(f"行情数据为空，无法获取交易状态信息 {ts_code} {trade_date}")
        return {
            'is_suspended': False,
            'is_limit_up': False,
            'is_limit_down': False,
            'can_buy': True,
            'can_sell': True,
            'close': None,
            'pct_chg': None
        }
    suspended = is_suspended(ts_code, trade_date, quote_data, suspend_df)
    limit_up = is_limit_up(ts_code, trade_date, quote_data)
    limit_down = is_limit_down(ts_code, trade_date, quote_data)
    
    # 获取价格信息
    close_price = None
    pct_chg = None
    try:
        mask = (quote_data['ts_code'] == ts_code) & (quote_data['trade_date'] == trade_date)
        if mask.sum() > 0:
            row = quote_data[mask].iloc[0]
            close_price = row.get('close', None)
            pct_chg = row.get('pct_chg', None)
    except Exception as e:
        logger.warning(f"获取价格信息时出错 {ts_code} {trade_date}: {e}")
    
    # 判断能否买卖
    can_buy = not suspended and not limit_up
    can_sell = not suspended and not limit_down
    
    return {
        'is_suspended': suspended,
        'is_limit_up': limit_up,
        'is_limit_down': limit_down,
        'can_buy': can_buy,
        'can_sell': can_sell,
        'close': close_price,
        'pct_chg': pct_chg
    }
