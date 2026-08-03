# -*- coding: utf-8 -*-
"""TushareClient 另类数据 mixin：筹码/基金持仓/股东人数/北向/龙虎榜/一致预期。"""

from typing import Optional

import pandas as pd


class ClientAltMixin:
    """TushareClient 另类数据 mixin。"""

    def get_cyq_perf(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取筹码胜率数据（cyq_perf，5000 积分）

        支持两种查询方式：
        1. 按 trade_date 获取全市场当日数据（推荐，单次获取所有股票）
        2. 按 ts_code + start_date/end_date 获取单只股票历史数据

        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期，格式 YYYYMMDD（可选，与 ts_code 二选一）
            start_date: 开始日期，格式 YYYYMMDD（配合 ts_code 使用）
            end_date: 结束日期，格式 YYYYMMDD（配合 ts_code 使用）

        Returns:
            筹码胜率 DataFrame
        """
        kwargs = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if trade_date is not None:
            kwargs["trade_date"] = trade_date
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("cyq_perf", **kwargs)

    def get_fund_portfolio(
        self,
        ts_code: Optional[str] = None,
        period: Optional[str] = None,
        ann_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取公募基金持仓数据（fund_portfolio，5000 积分）

        Args:
            ts_code: 基金代码（按单只基金查询）
            period: 报告期，格式 YYYYMMDD（如 20231231）
            ann_date: 公告日期，格式 YYYYMMDD

        Returns:
            基金持仓 DataFrame
        """
        kwargs = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if period is not None:
            kwargs["period"] = period
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        return self.query("fund_portfolio", **kwargs)

    def get_stk_holdernumber(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取股东人数数据

        支持多种查询方式：
        1. 按 ann_date 获取当日公告的全市场数据
        2. 按 start_date/end_date 获取一段时间内全市场数据（单次限3000条）
        3. 按 ts_code 获取单只股票历史数据

        Args:
            ts_code: 股票代码（可选）
            ann_date: 公告日期，格式 YYYYMMDD（可选）
            start_date: 开始日期，格式 YYYYMMDD（可选）
            end_date: 结束日期，格式 YYYYMMDD（可选）

        Returns:
            股东人数 DataFrame
        """
        kwargs: dict = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("stk_holdernumber", **kwargs)

    def get_moneyflow_hsgt(
        self,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取沪深股通资金流向（moneyflow_hsgt，2000 积分）

        市场级日度数据，返回沪股通/深股通当日整体买卖与净流入，
        用作北向资金宏观因子（广播到全部 ts_code）。

        Args:
            trade_date: 交易日期，格式 YYYYMMDD（可选）
            start_date: 开始日期，格式 YYYYMMDD（可选）
            end_date: 结束日期，格式 YYYYMMDD（可选）

        Returns:
            DataFrame，主要字段：
            - trade_date: 交易日期
            - ggt_ss: 港股通（上海）
            - ggt_sz: 港股通（深圳）
            - hgt: 沪股通（亿元）
            - sgt: 深股通（亿元）
            - north_money: 北向资金净流入（亿元）
            - south_money: 南向资金净流入（亿元）
        """
        kwargs: dict = {}
        if trade_date is not None:
            kwargs["trade_date"] = trade_date
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("moneyflow_hsgt", **kwargs)

    def get_top_list(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取龙虎榜个股明细（top_list，2000 积分）

        Args:
            trade_date: 交易日期，格式 YYYYMMDD（可选）
            ts_code: 股票代码（可选）
            start_date: 开始日期，格式 YYYYMMDD（可选）
            end_date: 结束日期，格式 YYYYMMDD（可选）

        Returns:
            DataFrame，主要字段：
            - trade_date, ts_code, name, close
            - pct_change: 涨跌幅
            - turnover_rate: 换手率
            - amount: 总成交额
            - l_sell/l_buy: 龙虎榜卖/买入额
            - l_amount: 龙虎榜成交额
            - net_amount: 龙虎榜净买入额
            - net_rate: 龙虎榜净买入额占比
            - amount_rate: 龙虎榜成交额占比
            - float_values: 当日流通市值
            - reason: 上榜理由
        """
        kwargs: dict = {}
        if trade_date is not None:
            kwargs["trade_date"] = trade_date
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        # 官方未明示限频, 局部放宽到 1000 次/分钟 (60ms/次), 加速历史批量下载
        # 实测 60ms 间隔连续请求 30 次无限流 (瓶颈是服务端 ~3.8s/请求),
        # 若触发限流 client.query 会自动解析"频率超限(X次/分钟)"并降频
        return self.query("top_list", rate_limit_override=1000, **kwargs)

    def get_report_rc(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取卖方研报一致预期（report_rc，2000 积分）

        Args:
            ts_code: 股票代码（可选）
            report_date: 研报日期，格式 YYYYMMDD（可选）
            start_date: 报告日期起（可选）
            end_date: 报告日期止（可选）

        Returns:
            DataFrame，主要字段：
            - ts_code, name
            - report_date: 研报日期
            - report_title, report_type
            - classify, org_name, author_name
            - quarter: 预测季度
            - op_rt: 预测营收增长率
            - op_pr: 预测营收
            - tp: 预测净利润
            - np: 预测净利润
            - eps: 每股收益预测
            - pe/rd/roe/ev_ebitda: 估值/收益指标
            - rating: 评级
            - max_price, min_price: 预测价格区间
        """
        kwargs: dict = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if report_date is not None:
            kwargs["report_date"] = report_date
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("report_rc", **kwargs)
