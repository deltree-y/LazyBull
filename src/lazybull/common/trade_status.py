"""交易状态检查工具模块

提供检查股票涨跌停、停牌状态的工具函数，用于选股和交易阶段的过滤。
"""

from typing import Any, Dict, Mapping, Optional

import pandas as pd
from loguru import logger


def evaluate_trade_status(
    status: Mapping[str, Any],
    action: str,
    *,
    require_tradable: bool = False,
) -> tuple[bool, Optional[str]]:
    """根据单条状态记录判断买卖可执行性。"""
    if status.get("is_suspended", 0) == 1:
        return False, "停牌"
    if action == "buy" and status.get("is_limit_up", 0) == 1:
        return False, "涨停"
    if action == "sell" and status.get("is_limit_down", 0) == 1:
        return False, "跌停"
    if require_tradable and status.get("tradable", 1) == 0:
        return False, "不可交易（ST/上市不足等）"
    return True, None


def is_suspended(
    ts_code: str,
    trade_date: str,
    quote_data: pd.DataFrame
) -> bool:
    """检查股票是否停牌
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期（YYYYMMDD格式字符串）
        quote_data: 行情数据，需包含 is_suspended 列
        
    Returns:
        True 表示停牌，False 表示未停牌
    """
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
    action: str = 'buy'
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

    status = {
        "is_suspended": is_suspended(ts_code, trade_date, quote_data),
        "is_limit_up": is_limit_up(ts_code, trade_date, quote_data),
        "is_limit_down": is_limit_down(ts_code, trade_date, quote_data),
    }
    return evaluate_trade_status(status, action)


def get_trade_status_info(
    ts_code: str,
    trade_date: str,
    quote_data: pd.DataFrame
) -> Dict[str, Any]:
    """获取股票的完整交易状态信息
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期（YYYYMMDD格式字符串）
        quote_data: 行情数据
        
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
    suspended = is_suspended(ts_code, trade_date, quote_data)
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
