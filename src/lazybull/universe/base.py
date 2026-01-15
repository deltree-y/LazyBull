"""股票池基类"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
from loguru import logger


class Universe(ABC):
    """股票池基类
    
    定义股票池的标准接口
    """
    
    def __init__(self, name: str = "base"):
        """初始化股票池
        
        Args:
            name: 股票池名称
        """
        self.name = name
    
    @abstractmethod
    def get_stocks(self, date: pd.Timestamp) -> List[str]:
        """获取指定日期的股票列表
        
        Args:
            date: 查询日期
            
        Returns:
            股票代码列表
        """
        pass
    
    def filter_st(self, stocks: pd.DataFrame) -> pd.DataFrame:
        """过滤ST股票
        
        Args:
            stocks: 股票DataFrame，需包含name字段
            
        Returns:
            过滤后的DataFrame
        """
        if 'name' in stocks.columns:
            mask = ~stocks['name'].str.contains('ST|退', na=False)
            return stocks[mask]
        return stocks
    
    def filter_market_cap(self, stocks: pd.DataFrame, min_cap: float) -> pd.DataFrame:
        """按市值过滤
        
        Args:
            stocks: 股票DataFrame，需包含total_mv字段（单位：万元）
            min_cap: 最小市值（单位：亿元）
            
        Returns:
            过滤后的DataFrame
        """
        if 'total_mv' in stocks.columns:
            min_cap_wan = min_cap * 10000  # 转换为万元
            return stocks[stocks['total_mv'] >= min_cap_wan]
        return stocks
    
    def filter_list_days(self, stocks: pd.DataFrame, date: pd.Timestamp, min_days: int) -> pd.DataFrame:
        """按上市天数过滤
        
        Args:
            stocks: 股票DataFrame，需包含list_date字段
            date: 当前日期
            min_days: 最少上市天数
            
        Returns:
            过滤后的DataFrame
        """
        if 'list_date' in stocks.columns:
            stocks = stocks.copy()
            stocks['list_date'] = pd.to_datetime(stocks['list_date'], errors='coerce')
            stocks['days_listed'] = (date - stocks['list_date']).dt.days
            return stocks[stocks['days_listed'] >= min_days]
        return stocks


class BasicUniverse(Universe):
    """基础股票池
    
    简单的股票池实现，根据基本条件筛选
    """
    
    def __init__(
        self,
        stock_basic: pd.DataFrame,
        exclude_st: bool = True,
        min_market_cap: Optional[float] = None,
        min_list_days: Optional[int] = None,
        markets: Optional[List[str]] = None
    ):
        """初始化基础股票池
        
        Args:
            stock_basic: 股票基本信息DataFrame
            exclude_st: 是否排除ST股票
            min_market_cap: 最小市值（亿元）- 暂未实现，需要额外的市值数据
            min_list_days: 最少上市天数
            markets: 市场列表，如 ["主板", "创业板"]
        """
        super().__init__("basic")
        self.stock_basic = stock_basic.copy()
        self.exclude_st = exclude_st
        self.min_market_cap = min_market_cap
        self.min_list_days = min_list_days
        self.markets = markets
    
    def get_stocks(self, date: pd.Timestamp) -> List[str]:
        """获取指定日期的股票列表
        
        Args:
            date: 查询日期
            
        Returns:
            股票代码列表
        """
        stocks = self.stock_basic.copy()
        
        # 市场过滤
        if self.markets:
            stocks = stocks[stocks['market'].isin(self.markets)]
        
        # ST过滤
        if self.exclude_st:
            stocks = self.filter_st(stocks)
        
        # 上市天数过滤
        if self.min_list_days:
            stocks = self.filter_list_days(stocks, date, self.min_list_days)
        
        # 市值过滤（需要daily_basic数据，当前未实现）
        # TODO: 实现市值过滤需要在调用时传入daily_basic数据
        # if self.min_market_cap and daily_basic is not None:
        #     stocks = self.filter_market_cap(stocks, self.min_market_cap)
        
        stock_list = stocks['ts_code'].tolist()
        logger.debug(f"股票池 {self.name} 在 {date.date()} 选出 {len(stock_list)} 只股票")
        
        return stock_list
