"""TuShare数据接口客户端"""

import os
import time
from typing import Any, Dict, Optional

import pandas as pd
import tushare as ts
from loguru import logger


class TushareClient:
    """TuShare Pro API客户端
    
    封装TuShare接口调用，提供限频和重试机制
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit: int = 200
    ):
        """初始化TuShare客户端
        
        Args:
            token: TuShare token，如不提供则从环境变量TS_TOKEN读取
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            rate_limit: 每分钟请求限制
        """
        # 获取token
        self.token = token or os.getenv("TS_TOKEN")
        if not self.token:
            raise ValueError(
                "未找到TuShare token！\n"
                "请设置环境变量 TS_TOKEN 或创建 .env 文件。\n"
                "获取token: https://tushare.pro/register"
            )
        
        # 设置token
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        
        # 配置参数
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit = rate_limit
        
        # 参数验证
        if rate_limit <= 0:
            raise ValueError(f"rate_limit 必须大于0，当前值: {rate_limit}")
        
        # 限频控制
        self._last_request_time = 0.0
        self._request_interval = 60.0 / rate_limit  # 每次请求最小间隔
        
        logger.info(f"TuShare客户端初始化成功，限频: {rate_limit}次/分钟")
    
    def _rate_limit_wait(self) -> None:
        """执行限频等待"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            wait_time = self._request_interval - elapsed
            time.sleep(wait_time)
        self._last_request_time = time.time()
    
    def query(
        self,
        api_name: str,
        fields: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """调用TuShare API
        
        Args:
            api_name: API名称，如 'trade_cal', 'stock_basic'
            fields: 返回字段，逗号分隔
            **kwargs: API参数
            
        Returns:
            查询结果DataFrame
        """
        for attempt in range(self.max_retries):
            try:
                # 限频等待
                self._rate_limit_wait()
                
                # 调用API
                logger.debug(f"调用API: {api_name}, 参数: {kwargs}")
                df = self.pro.query(api_name, fields=fields, **kwargs)
                
                logger.debug(f"API {api_name} 返回 {len(df)} 条记录")
                return df
                
            except Exception as e:
                logger.warning(
                    f"API调用失败 ({attempt + 1}/{self.max_retries}): {api_name}, "
                    f"错误: {str(e)}"
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"API调用最终失败: {api_name}")
                    raise
        
        return pd.DataFrame()
    
    def get_trade_cal(
        self,
        start_date: str,
        end_date: str,
        exchange: str = "SSE"
    ) -> pd.DataFrame:
        """获取交易日历
        
        Args:
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            exchange: 交易所，SSE上交所/SZSE深交所
            
        Returns:
            交易日历DataFrame
        """
        return self.query(
            "trade_cal",
            fields="exchange,cal_date,is_open,pretrade_date",
            exchange=exchange,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_stock_basic(
        self,
        list_status: str = "L",
        fields: Optional[str] = None
    ) -> pd.DataFrame:
        """获取股票列表
        
        Args:
            list_status: 上市状态，L上市/D退市/P暂停上市
            fields: 返回字段
            
        Returns:
            股票列表DataFrame
        """
        if fields is None:
            fields = "ts_code,symbol,name,area,industry,market,list_date"
        
        return self.query("stock_basic", fields=fields, list_status=list_status)
    
    def get_daily(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取日线行情
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            日线行情DataFrame
        """
        return self.query(
            "daily",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_daily_basic(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取每日指标（PE、PB等）
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            每日指标DataFrame
        """
        return self.query(
            "daily_basic",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_adj_factor(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取复权因子
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            复权因子DataFrame，包含 ts_code, trade_date, adj_factor 等字段
        """
        return self.query(
            "adj_factor",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_suspend_d(
        self,
        ts_code: Optional[str] = None,
        suspend_date: Optional[str] = None,
        resume_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取停复牌信息
        
        Args:
            ts_code: 股票代码
            suspend_date: 停牌日期
            resume_date: 复牌日期
            
        Returns:
            停复牌信息DataFrame
        """
        return self.query(
            "suspend_d",
            ts_code=ts_code,
            suspend_date=suspend_date,
            resume_date=resume_date
        )
    
    def get_stk_limit(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取每日涨跌停价格
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            涨跌停价格DataFrame，包含 up_limit, down_limit 等字段
        """
        return self.query(
            "stk_limit",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_namechange(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取股票名称变更历史
        
        用于判断ST状态等
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            名称变更历史DataFrame
        """
        return self.query(
            "namechange",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
