# -*- coding: utf-8 -*-
"""TushareClient 日线行情 mixin：日线/每日指标/复权/停复牌/涨跌停/ST/资金流。"""

from typing import Optional

import pandas as pd

# TuShare 单次查询行数上限（全市场单日已逼近该值，超出会静默截断）
_TUSHARE_PAGE_LIMIT = 6000


class ClientDailyMixin:
    """TushareClient 日线行情 mixin。"""

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
        if ts_code is None:
            # 全市场查询（单日约 5400+ 条，逼近单次 6000 上限）自动分页，避免静默截断
            return self._query_with_pagination(
                "daily",
                page_limit=_TUSHARE_PAGE_LIMIT,
                ts_code=ts_code,
                trade_date=trade_date,
                start_date=start_date,
                end_date=end_date,
            )
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
        if ts_code is None:
            # 全市场查询（单日约 5400+ 条，逼近单次 6000 上限）自动分页，避免静默截断
            return self._query_with_pagination(
                "daily_basic",
                page_limit=_TUSHARE_PAGE_LIMIT,
                ts_code=ts_code,
                trade_date=trade_date,
                start_date=start_date,
                end_date=end_date,
            )
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
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        suspend_type: Optional[str] = None
    ) -> pd.DataFrame:
        """获取停复牌信息
        
        注意：此API已更新参数，旧版本使用suspend_date/resume_date的代码需要迁移
        
        Args:
            ts_code: 股票代码，支持多个股票
            trade_date: 交易日期，格式YYYYMMDD
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            suspend_type: 停复牌类型，S=停牌，R=复牌
            
        Returns:
            停复牌信息DataFrame，包含以下字段：
            - ts_code: 股票代码
            - trade_date: 停复牌日期
            - suspend_timing: 盘中停复牌时段（如有）
            - suspend_type: S=停牌，R=复牌
            
        Examples:
            >>> # 获取某日所有停牌股票
            >>> client.get_suspend_d(trade_date='20230315', suspend_type='S')
            >>> # 获取某个时间段某只股票的停复牌记录
            >>> client.get_suspend_d(ts_code='000001.SZ', start_date='20230101', end_date='20230331')
        """
        return self.query(
            "suspend_d",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            suspend_type=suspend_type
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
        if ts_code is None:
            # 全市场查询（含指数约 7400 条，超单次 6000 上限）自动分页，避免静默截断
            return self._query_with_pagination(
                "stk_limit",
                page_limit=_TUSHARE_PAGE_LIMIT,
                ts_code=ts_code,
                trade_date=trade_date,
                start_date=start_date,
                end_date=end_date,
            )
        return self.query(
            "stk_limit",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

    def get_stock_st(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取 ST 状态数据（stock_st）。

        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            ST 状态 DataFrame，通常包含 ts_code、trade_date、is_st 等字段
        """
        return self.query(
            "stock_st",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
        )

    def get_moneyflow(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取个股资金流向
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            资金流向DataFrame，包含以下字段：
            - ts_code: 股票代码
            - trade_date: 交易日期
            - buy_sm_vol: 小单买入量（手）
            - buy_sm_amount: 小单买入金额（万元）
            - sell_sm_vol: 小单卖出量（手）
            - sell_sm_amount: 小单卖出金额（万元）
            - buy_md_vol: 中单买入量（手）
            - buy_md_amount: 中单买入金额（万元）
            - sell_md_vol: 中单卖出量（手）
            - sell_md_amount: 中单卖出金额（万元）
            - buy_lg_vol: 大单买入量（手）
            - buy_lg_amount: 大单买入金额（万元）
            - sell_lg_vol: 大单卖出量（手）
            - sell_lg_amount: 大单卖出金额（万元）
            - buy_elg_vol: 特大单买入量（手）
            - buy_elg_amount: 特大单买入金额（万元）
            - sell_elg_vol: 特大单卖出量（手）
            - sell_elg_amount: 特大单卖出金额（万元）
            - net_mf_vol: 净流入量（手）
            - net_mf_amount: 净流入额（万元）
        """
        if ts_code is None:
            # 全市场查询（单日约 5400+ 条，逼近单次 6000 上限）自动分页，避免静默截断
            return self._query_with_pagination(
                "moneyflow",
                page_limit=_TUSHARE_PAGE_LIMIT,
                ts_code=ts_code,
                trade_date=trade_date,
                start_date=start_date,
                end_date=end_date,
            )
        return self.query(
            "moneyflow",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )