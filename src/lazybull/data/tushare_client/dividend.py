# -*- coding: utf-8 -*-
"""TushareClient 分红送股 mixin：TuShare `dividend` 接口（2000+ 积分）。"""

from typing import Optional

import pandas as pd

# 显式请求字段：TuShare `dividend` 接口默认字段不包含 base_share（基准股本，
# 万股），而不请求该字段服务端不会返回；分红政策因子 dividend_payout_ratio
# 的现金分红总额 = cash_div_tax（每股税前）× base_share，缺失会导致支付率
# 静默全 NaN。因此所有 dividend 查询必须显式携带本字段清单。
DIVIDEND_FIELDS = (
    "ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,"
    "cash_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate,"
    "imp_ann_date,base_date,base_share"
)


class ClientDividendMixin:
    """TushareClient 分红送股 mixin。"""

    def get_dividend(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        record_date: Optional[str] = None,
        ex_date: Optional[str] = None,
        imp_ann_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取分红送股数据（接口仅支持单值查询参数，不支持日期区间）。

        输出字段（TuShare 官方口径）：
          - cash_div: 每股分红（税后）
          - cash_div_tax: 每股分红（税前）
          - stk_div: 每股送转（股数）
          - div_proc: 实施进度（预案/决案/实施）
          - ann_date: 公告日（预案/决案）
          - imp_ann_date: 实施公告日
          - ex_date: 除权除息日
          - record_date: 股权登记日
          - end_date: 分红年度
          - base_date: 基准日
          - base_share: 基准股本（万股，必须显式请求，默认字段不返回）

        Args:
            ts_code: TS 代码（如 600848.SH）
            ann_date: 公告日 YYYYMMDD
            record_date: 股权登记日 YYYYMMDD
            ex_date: 除权除息日 YYYYMMDD
            imp_ann_date: 实施公告日 YYYYMMDD

        Returns:
            分红送股 DataFrame（可能为空）
        """
        return self.query(
            "dividend",
            ts_code=ts_code,
            ann_date=ann_date,
            record_date=record_date,
            ex_date=ex_date,
            imp_ann_date=imp_ann_date,
            fields=DIVIDEND_FIELDS,
        )
