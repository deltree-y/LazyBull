"""股票池基类"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
from loguru import logger

from ..common.trade_status import is_tradeable


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
    def get_stocks(
        self, 
        date: pd.Timestamp,
        quote_data: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """获取指定日期的股票列表
        
        Args:
            date: 查询日期
            quote_data: 行情数据（可选），用于过滤停牌、涨跌停股票
            
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
        markets: Optional[List[str]] = None,
        filter_suspended: bool = True,
        filter_limit_stocks: bool = True
    ):
        """初始化基础股票池
        
        Args:
            stock_basic: 股票基本信息DataFrame
            exclude_st: 是否排除ST股票
            min_market_cap: 最小市值（亿元）- 暂未实现，需要额外的市值数据
            min_list_days: 最少上市天数
            markets: 市场列表，如 ["主板", "创业板"]
            filter_suspended: 是否过滤停牌股票，默认True
            filter_limit_stocks: 是否过滤涨跌停股票，默认True
        """
        super().__init__("basic")
        self.stock_basic = stock_basic.copy()
        self.exclude_st = exclude_st
        self.min_market_cap = min_market_cap
        self.min_list_days = min_list_days
        self.markets = markets
        self.filter_suspended = filter_suspended
        self.filter_limit_stocks = filter_limit_stocks
    
    def get_stocks(
        self, 
        date: pd.Timestamp,
        quote_data: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """获取指定日期的股票列表
        
        Args:
            date: 查询日期
            quote_data: 行情数据（可选），用于过滤停牌、涨跌停股票
            
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
        
        # 如果提供了行情数据，进一步过滤停牌和涨跌停股票
        if quote_data is not None and not quote_data.empty and (self.filter_suspended or self.filter_limit_stocks):
            stock_list = self._filter_untradeable_stocks(
                stock_list, date, quote_data
            )
        
        logger.debug(f"股票池 {self.name} 在 {date.date()} 选出 {len(stock_list)} 只股票")
        
        return stock_list
    
    def _filter_untradeable_stocks(
        self,
        stock_list: List[str],
        date: pd.Timestamp,
        quote_data: pd.DataFrame
    ) -> List[str]:
        """过滤不可交易的股票（停牌、涨停）
        
        注意：此方法在Universe级别调用，使用T日数据。
        但根据新的交易逻辑，涨跌停应该在信号生成后基于T+1日数据过滤。
        因此这里只过滤停牌股票（如果配置了filter_suspended），
        不过滤涨跌停股票（即使配置了filter_limit_stocks）。
        
        Args:
            stock_list: 股票代码列表
            date: 查询日期
            quote_data: 行情数据
            
        Returns:
            过滤后的股票代码列表
        """
        trade_date_str = date.strftime('%Y%m%d')
        filtered_stocks = []
        filtered_count = {'停牌': 0}
        
        for stock in stock_list:
            # 仅检查停牌状态（涨跌停过滤已移至信号生成阶段基于T+1数据）
            if quote_data.empty:
                # 行情数据为空，无法判断交易状态，保留股票
                logger.warning(
                    f"Universe过滤时行情数据为空，保留股票 {stock} 在 {date.date()}"
                )
                filtered_stocks.append(stock)
                continue
            tradeable, reason = is_tradeable(
                stock, trade_date_str, quote_data, action='buy'
            )
            
            if tradeable:
                filtered_stocks.append(stock)
            else:
                # 只过滤停牌股票
                if reason == "停牌" and self.filter_suspended:
                    filtered_count['停牌'] = filtered_count.get('停牌', 0) + 1
                else:
                    # 涨跌停不在此过滤，保留在股票池中
                    filtered_stocks.append(stock)
        
        # 输出过滤日志
        if sum(filtered_count.values()) > 0:
            logger.info(
                f"Universe过滤 {date.date()}: 原始 {len(stock_list)} 只，"
                f"过滤停牌 {filtered_count['停牌']} 只，"
                f"最终 {len(filtered_stocks)} 只"
            )
        
        return filtered_stocks
