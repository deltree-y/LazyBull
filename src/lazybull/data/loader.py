"""数据加载模块"""

from typing import Optional

import pandas as pd
from loguru import logger

from .storage import Storage


class DataLoader:
    """数据加载器
    
    提供标准化的数据加载接口
    """
    
    def __init__(self, storage: Optional[Storage] = None):
        """初始化数据加载器
        
        Args:
            storage: 存储实例，如不提供则创建默认实例
        """
        self.storage = storage or Storage()
    
    def load_trade_cal(self) -> Optional[pd.DataFrame]:
        """加载交易日历
        
        Returns:
            交易日历DataFrame
        """
        df = self.storage.load_raw("trade_cal")
        if df is not None:
            # 转换日期格式
            if 'cal_date' in df.columns:
                df['cal_date'] = pd.to_datetime(df['cal_date'], format='%Y%m%d')
            if 'pretrade_date' in df.columns:
                df['pretrade_date'] = pd.to_datetime(df['pretrade_date'], format='%Y%m%d', errors='coerce')
        return df
    
    def load_stock_basic(self) -> Optional[pd.DataFrame]:
        """加载股票基本信息
        
        Returns:
            股票基本信息DataFrame
        """
        df = self.storage.load_raw("stock_basic")
        if df is not None:
            # 转换日期格式
            if 'list_date' in df.columns:
                df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
        return df
    
    def load_daily(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """加载日线行情数据
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            
        Returns:
            日线行情DataFrame
        """
        df = self.storage.load_raw("daily")
        if df is None:
            return None
        
        # 转换日期格式
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        # 日期过滤
        if start_date:
            df = df[df['trade_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['trade_date'] <= pd.to_datetime(end_date)]
        
        return df
    
    def load_daily_basic(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """加载每日指标数据
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            
        Returns:
            每日指标DataFrame
        """
        df = self.storage.load_raw("daily_basic")
        if df is None:
            return None
        
        # 转换日期格式
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        # 日期过滤
        if start_date:
            df = df[df['trade_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['trade_date'] <= pd.to_datetime(end_date)]
        
        return df
    
    def get_trading_dates(self, start_date: str, end_date: str) -> list:
        """获取指定范围内的交易日列表
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            
        Returns:
            交易日期列表
        """
        df = self.load_trade_cal()
        if df is None:
            logger.warning("交易日历未加载，返回空列表")
            return []
        
        # 筛选交易日
        mask = (
            (df['cal_date'] >= pd.to_datetime(start_date)) &
            (df['cal_date'] <= pd.to_datetime(end_date)) &
            (df['is_open'] == 1)
        )
        
        trading_dates = df[mask]['cal_date'].tolist()
        return sorted(trading_dates)
