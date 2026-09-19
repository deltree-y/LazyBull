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

    def get_top10_floatholders(
        self,
        ts_code: Optional[str] = None,
        period: Optional[str] = None,
        ann_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取十大流通股东数据（top10_floatholders）。

        **单页 6000 行且超限不报错**（Phase 0 实测：`limit=10000` 仍回 6000）⇒ 批量拉取必须
        用 `_query_with_pagination(page_limit=6000)` 按 `period` 翻页读满；
        **不要使用 `start_date/end_date`**（审计 §3：语义不透明，疑似按报告期过滤）；
        PIT 锚点 = `ann_date`（缺失 0%）。本方法只用于单笔/单股查询。

        Args:
            ts_code: 股票代码
            period: 报告期，格式 YYYYMMDD（如 20231231）
            ann_date: 公告日期，格式 YYYYMMDD（返回行会跨多个报告期，慎用）

        Returns:
            十大流通股东 DataFrame
        """
        kwargs = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if period is not None:
            kwargs["period"] = period
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        return self.query("top10_floatholders", **kwargs)

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
        # top_list 官方限频 500 次/分钟, 由 _API_RATE_LIMITS_DEFAULT["top_list"]
        # 控制客户端侧限频 (当前 400, 低于官方限频避免被限流)。
        # 不要传 rate_limit_override: 它会绕过接口级/全局令牌桶, 曾导致按 1000
        # 次/分钟并发触发官方限流。若仍触发限流, client.query 会自动解析
        # "频率超限(X次/分钟)"并自适应降频。
        return self.query("top_list", **kwargs)

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

    def get_stk_holdertrade(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取股东增减持数据（stk_holdertrade）。

        **注意分页**：接口单次请求最多 3000 行，超出时**不报错**、默认返回最新 3000 行；
        批量下载必须走 `client._query_with_pagination(page_limit=3000, ...)` 或
        `data/holdertrade_raw.py::download_holdertrade()`，本方法只用于单笔/单股查询。

        Args:
            ts_code: 股票代码（可选；单独指定可拉该股全历史）
            start_date: 公告日期起（YYYYMMDD，可选）
            end_date: 公告日期止（YYYYMMDD，可选）

        Returns:
            DataFrame，字段：ts_code, ann_date, holder_name, holder_type, in_de,
            change_vol, change_ratio, after_share, after_ratio, avg_price, total_share
        """
        kwargs: dict = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("stk_holdertrade", **kwargs)

    def get_repurchase(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取股票回购数据（repurchase）。

        **注意分页**：接口单次请求最多 2000 行，超出时**不报错**、默认返回最新 2000 行
        （实测 2022 全年真实 6034 行，不翻页丢 67%）；批量下载必须走
        `client._query_with_pagination(page_limit=2000, ...)` 或
        `data/repurchase_raw.py::download_repurchase()`，本方法只用于单笔/单股查询。

        Args:
            ts_code: 股票代码（可选；单独指定可拉该股全历史）
            start_date: 公告日期起（YYYYMMDD，可选）
            end_date: 公告日期止（YYYYMMDD，可选）

        Returns:
            DataFrame，字段：ts_code, ann_date, end_date, proc, exp_date,
            vol, amount, high_limit, low_limit
        """
        kwargs: dict = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("repurchase", **kwargs)
