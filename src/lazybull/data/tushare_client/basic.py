# -*- coding: utf-8 -*-
"""TushareClient 基础信息 mixin：交易日历/股票列表/名称变更/行业分类/成分股。"""

from typing import Dict, List, Optional

import pandas as pd
import tushare as ts


class ClientBasicMixin:
    """TushareClient 基础信息 mixin。"""

    def get_trade_cal(
        self,
        start_date: str = None,
        end_date: str = None,
        exchange: str = "SSE"
    ) -> pd.DataFrame:
        """获取交易日历
        
        Args:
            start_date: 开始日期，格式YYYYMMDD（不指定则获取全部数据）
            end_date: 结束日期，格式YYYYMMDD（不指定则获取全部数据）
            exchange: 交易所，SSE上交所/SZSE深交所
            
        Returns:
            交易日历DataFrame
        """
        # 构建查询参数
        kwargs = {"exchange": exchange}
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
            
        return self.query(
            "trade_cal",
            fields="exchange,cal_date,is_open,pretrade_date",
            **kwargs
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

    def get_index_classify(
        self,
        level: str = "L1",
        src: str = "SW2021",
        **kwargs
    ) -> pd.DataFrame:
        """获取申万行业分类
        
        Args:
            level: 行业级别，L1=一级/L2=二级/L3=三级
            src: 申万分类版本，SW2021=申万2021版/SW2014=申万2014版
            **kwargs: 其他参数
            
        Returns:
            申万行业分类DataFrame，包含以下字段：
            - index_code: 指数代码
            - industry_name: 行业名称
            - level: 行业级别
            - industry_code: 行业代码
            - parent_code: 父级代码
            - src: 分类来源
        """
        return self.query(
            "index_classify",
            level=level,
            src=src,
            **kwargs
        )

    def get_index_member(
        self,
        l1_code: str = None,
        l2_code: str = None,
        l3_code: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """获取指数成分股
        
        Args:
            l1_code: 一级行业代码
            l2_code: 二级行业代码
            l3_code: 三级行业代码
            **kwargs: 其他参数
            
        Returns:
            指数成分股DataFrame，包含以下字段：
            - l1_code: 一级行业代码
            - l1_name: 一级行业名称
            - l2_code: 二级行业代码
            - l2_name: 二级行业名称
            - l3_code: 三级行业代码
            - l3_name: 三级行业名称
            - ts_code: 成分股代码
            - ts_name: 成分股名称
            - in_date: 加入日期
            - out_date: 退出日期
            - is_new: 是否最新成分股，1=是，0=否
        """
        return self.query(
            "index_member_all",
            l1_code=l1_code,
            l2_code=l2_code,
            l3_code=l3_code,
            **kwargs
        )

    def get_realtime_quote(self, ts_codes: str) -> pd.DataFrame:
        """获取实时行情

        Args:
            ts_codes: 股票代码，多个以逗号分隔，如 '000001.SZ,000002.SZ'

        Returns:
            实时行情DataFrame，包含 ts_code, name, price, pre_close, open,
            high, low, volume, amount, time 等字段
        """
        #return self.query("realtime_quote", ts_code=ts_codes)
        return ts.realtime_quote(ts_code=ts_codes)