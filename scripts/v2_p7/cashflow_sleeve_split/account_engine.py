# -*- coding: utf-8 -*-
"""CF-SPLIT 原生账户的实际持仓上限。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from scripts.v2_p7.operating_monthly_baseline import OperatingBaselineEngine


class CashflowAccountEngine(OperatingBaselineEngine):
    """所有建仓入口共用同一真实槽位检查，买卖时钟与候补排期沿用原生账户。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.position_limit_rejections: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _buy_stock_direct(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: pd.Timestamp | None = None,
        requested_shares: int | None = None,
    ) -> None:
        """在延迟订单和补齐共享的成交点复核新股可用槽位。"""
        capacity = int(self.signal.top_n)
        if stock not in self.positions and len(self.positions) >= capacity:
            self.position_limit_rejections.append(
                {
                    "trade_date": date.strftime("%Y%m%d"),
                    "signal_date": signal_date.strftime("%Y%m%d") if signal_date else None,
                    "ts_code": stock,
                    "position_count": len(self.positions),
                    "position_limit": capacity,
                    "reason": "position_capacity_full",
                }
            )
            return
        super()._buy_stock_direct(date, stock, target_value, signal_date, requested_shares)
