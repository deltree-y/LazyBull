# -*- coding: utf-8 -*-
"""TushareClient 基本面 mixin：财务指标/现金流量表/业绩预告/业绩快报。"""

from typing import Optional

import pandas as pd

from .core import FINA_INDICATOR_DEFAULT_FIELDS

INCOME_DEFAULT_FIELDS = (
    "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type," "n_income_attr_p,update_flag"
)


class ClientFundamentalMixin:
    """TushareClient 基本面 mixin。"""

    def get_fina_indicator(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取财务指标数据（fina_indicator）

        注意：此 API 只支持按单个股票查询，每次最多返回 100 条记录。
        需要 2000 积分权限。

        Args:
            ts_code: 股票代码（必须，单只股票，如 '000001.SZ'）
            start_date: 报告期开始日期，格式 YYYYMMDD
            end_date: 报告期结束日期，格式 YYYYMMDD
            fields: 返回字段，逗号分隔

        Returns:
            财务指标 DataFrame
        """
        if fields is None:
            fields = FINA_INDICATOR_DEFAULT_FIELDS

        return self.query(
            "fina_indicator",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def get_fina_indicator_by_date(
        self,
        ann_date: Optional[str] = None,
        period: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """按公告日或报告期获取全市场财务指标（fina_indicator_vip，5000 积分）

        Args:
            ann_date: 公告日期，格式 YYYYMMDD（与 period 二选一）
            period: 报告期，格式 YYYYMMDD，如 20231231（与 ann_date 二选一）
            fields: 返回字段，逗号分隔

        Returns:
            全市场财务指标 DataFrame
        """
        if fields is None:
            fields = FINA_INDICATOR_DEFAULT_FIELDS
        kwargs: dict = {"fields": fields}
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        if period is not None:
            kwargs["period"] = period
        return self.query("fina_indicator_vip", **kwargs)

    def get_cashflow(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取现金流量表数据（cashflow，2000 积分）

        用于构建经营现金流质量、自由现金流等因子。

        Args:
            ts_code: 股票代码（可选，不传则按报告期查全市场）
            start_date: 报告期开始日期，格式 YYYYMMDD
            end_date: 报告期结束日期，格式 YYYYMMDD
            fields: 返回字段，逗号分隔

        Returns:
            现金流量表 DataFrame
        """
        if fields is None:
            fields = (
                "ts_code,ann_date,f_ann_date,end_date,"
                "net_profit,"
                "c_fr_sale_sg,"
                "n_cashflow_act,"
                "c_pay_acq_const_fiolta,"
                "st_cash_out_act,"
                "n_cashflow_inv_act,"
                "free_cashflow"
            )
        kwargs: dict = {"fields": fields}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("cashflow", **kwargs)

    def get_cashflow_by_period(
        self,
        period: str,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """按报告期获取全市场现金流量表（cashflow_vip，5000 积分）

        Args:
            period: 报告期，格式 YYYYMMDD，如 20231231
            fields: 返回字段，逗号分隔

        Returns:
            全市场现金流量表 DataFrame
        """
        if fields is None:
            fields = (
                "ts_code,ann_date,f_ann_date,end_date,"
                "net_profit,"
                "c_fr_sale_sg,"
                "n_cashflow_act,"
                "c_pay_acq_const_fiolta,"
                "st_cash_out_act,"
                "n_cashflow_inv_act,"
                "free_cashflow"
            )
        return self.query("cashflow_vip", fields=fields, period=period)

    def get_income_by_period(
        self,
        period: str,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """按报告期获取全市场利润表（income_vip，5000 积分）。"""
        return self.query(
            "income_vip",
            fields=fields or INCOME_DEFAULT_FIELDS,
            period=period,
        )

    def get_forecast_by_date(
        self,
        ann_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """按公告日或报告期获取全市场业绩预告（forecast_vip，5000 积分）

        Args:
            ann_date: 公告日期，格式 YYYYMMDD（与 period 二选一）
            period: 报告期，格式 YYYYMMDD，如 20231231（与 ann_date 二选一）

        Returns:
            全市场业绩预告 DataFrame
        """
        kwargs: dict = {}
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        if period is not None:
            kwargs["period"] = period
        return self.query("forecast_vip", **kwargs)

    def get_express_vip(
        self,
        ann_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """按公告日/报告期获取全市场业绩快报（express_vip，5000 积分）

        Args:
            ann_date: 公告日期，格式 YYYYMMDD
            period: 报告期，格式 YYYYMMDD（如 20231231）

        Returns:
            业绩快报 DataFrame
        """
        kwargs = {}
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        if period is not None:
            kwargs["period"] = period
        return self.query("express_vip", **kwargs)
