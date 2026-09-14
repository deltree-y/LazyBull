"""回测引擎的持仓快照旁路（只读记录，不参与任何决策）。

动机（terminal_loss 政策层 P2-1）：主策略 OOS 回测目前只导出成交、执行归因、
TopK 名单与净值链，**没有逐日持仓快照**，导致"当时持有什么、持有到第几天、
到期执行日是哪天"只能靠成交流水反推（计划 §8.2 明确禁止靠不完整交易日志猜测
持仓上下文）。本 mixin 在每日估值后追加一行只读记录。

硬约束：

- **不得改变任何决策**：只在 ``record_holdings_snapshot`` 为 True 时追加列表，
  不读写任何参与买卖判断的状态；启用/禁用对成交、净值、排期逐位一致。
- 估值口径与 ``_calculate_portfolio_value`` 完全一致（成交价优先、缺失回退
  缓存价、再回退买入价），避免与净值曲线出现系统性偏差。
- 到期执行日按**标准持有期**推算（``买入日历位置 + holding_period``）；超出
  回测窗口时为 None（记录空值而不是猜测）。盈利延续等动态延期不在本表口径内，
  以引擎真实调度为准的那一版由 P2-3（shadow 接入）提供。
"""

from typing import Dict, List, Optional

import pandas as pd


class BacktestHoldingsSnapshotMixin:
    """逐日持仓快照（只读旁路）。"""

    def get_holdings_snapshot(self) -> pd.DataFrame:
        """获取持仓快照 DataFrame（无记录时返回空帧，列齐但不含行）。"""
        from ..common.sidecar_schema import SNAPSHOT_KEYS

        if not self.holdings_snapshots:
            return pd.DataFrame(columns=SNAPSHOT_KEYS)
        return pd.DataFrame(self.holdings_snapshots)

    def _record_holdings_snapshot(
        self,
        date: pd.Timestamp,
        portfolio_value: float,
        trading_dates: List[pd.Timestamp],
        date_to_idx: Dict[pd.Timestamp, int],
    ) -> None:
        """追加当日持仓快照（仅记录，不做任何判断）。

        Args:
            date: 当前交易日
            portfolio_value: 当日组合总值（与净值曲线同一口径）
            trading_dates: 回测交易日序列（升序）
            date_to_idx: {交易日: 序列下标}
        """
        if not getattr(self, "record_holdings_snapshot", False):
            return
        if not self.positions:
            return

        idx = date_to_idx.get(date)
        if idx is None:
            # 交易日映射缺失属链路异常（快照依赖日历位置推算到期日）
            raise ValueError(f"持仓快照需要 {date} 在交易日序列中的位置，当前映射缺失")

        for stock, info in self.positions.items():
            # 只读：估值口径与 _calculate_portfolio_value 一致，但不写回任何状态
            trade_price = self._get_trade_price(date, stock)
            if trade_price is None:
                trade_price = info.get("last_known_price")
                if trade_price is None:
                    trade_price = info.get("buy_trade_price", 0.0)

            shares = info["shares"]
            market_value = shares * trade_price
            buy_date = info.get("buy_date")
            buy_idx = date_to_idx.get(buy_date) if buy_date is not None else None

            holding_days: Optional[int] = None
            planned_exit_date: Optional[pd.Timestamp] = None
            remaining_intervals: Optional[int] = None
            if buy_idx is not None:
                holding_days = idx - buy_idx
                exit_idx = buy_idx + self.holding_period
                if exit_idx < len(trading_dates):
                    planned_exit_date = trading_dates[exit_idx]
                    # 与标签口径一致：E = T+1+h ⇒ h = E 位置 − T 位置 − 1
                    remaining_intervals = exit_idx - idx - 1

            self.holdings_snapshots.append(
                {
                    "date": date,
                    "ts_code": stock,
                    "shares": int(shares),
                    "market_value": float(market_value),
                    "weight": float(market_value / portfolio_value) if portfolio_value else None,
                    "portfolio_value": float(portfolio_value),
                    "buy_date": buy_date,
                    "signal_date": info.get("signal_date") or buy_date,
                    "holding_days": holding_days,
                    "planned_exit_date": planned_exit_date,
                    "remaining_intervals": remaining_intervals,
                }
            )
