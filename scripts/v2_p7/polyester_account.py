"""涤纶两股固定试点的研究账户入口；仅调整槽数，执行沿原研究账。"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook
from scripts.v2_p7.operating_monthly_account import _MonthlyAccount


def run_polyester_account(
    *,
    calendar: Sequence[str],
    quotes: pd.DataFrame,
    signals: pd.DataFrame,
    capital: float,
    corporate_book: NativeCorporateBook,
    arm_name: str,
    codes: Sequence[str],
    dividend_haircut: float = 0.0,
) -> dict[str, Any]:
    """执行两个固定槽；未知信号、可知时钟与两股名单由冻结研究规格处理。

    输入列和公司行为口径沿经营月报账户。T日收盘计划，每个入选槽占本账户
    净资产一半，未入选槽留现金；T+1开盘卖出、收盘买入，不借用A资金。
    不改变已有公开经营月报入口的四槽规则。
    """
    return _MonthlyAccount(
        calendar,
        quotes,
        signals,
        capital,
        corporate_book,
        arm_name,
        codes,
        dividend_haircut,
        expected_slots=2,
    ).run()
