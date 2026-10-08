# -*- coding: utf-8 -*-
"""宿主层协议（契约 docs/contracts/protocols.md §4 的代码化，R4-M6/H6 修正）。

BacktestHost（回测宿主，薄适配器：只提供时钟 / 数据 / 成交通道）与
PaperHost（纸面宿主，真实时间驱动；run_t0 / run_t1 分别暴露
Executor.execute_daily 的两个阶段，同一执行语义的宿主适配）。
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from src.lazybull.v2.common.protocols.core import (
    DecisionMaker,
    Executor,
    Ledger,
    PortfolioMerger,
)
from src.lazybull.v2.common.protocols.signal import Signal
from src.lazybull.v2.common.protocols.store import DataStore
from src.lazybull.v2.common.types import Fill, Order, TradeDate

__all__ = ["BacktestHost", "PaperHost"]


@runtime_checkable
class BacktestHost(Protocol):
    """回测宿主（薄适配器，只提供时钟 / 数据 / 成交通道）。"""

    def run(
        self,
        start_date: TradeDate,
        end_date: TradeDate,
        sleeves: Mapping[str, Signal],
        merger: PortfolioMerger,  # R4-M6：合并器进编排链（单袖时代 = 直通实现）
        decision_maker: DecisionMaker,
        executor: Executor,
        ledger: Ledger,
        store: DataStore,
    ) -> None:
        """运行回测（驱动每日循环：信号 → 合并 → 决策 → 执行 → 入账 → 快照）。

        副作用：写账本快照 + 成交记录
        产出：runs/<批ID>/ 目录
        """
        ...


@runtime_checkable
class PaperHost(Protocol):
    """纸面宿主（真实时间驱动）。"""

    def run_t0(self, date: TradeDate) -> Sequence[Order]:
        """T0 判定（收盘后生成次日指令）。

        产出：钉钉推送 + 指令文件
        """
        ...

    def run_t1(self, date: TradeDate, fills: Sequence[Fill]) -> None:
        """T+1 执行（手工回填实际成交）。

        副作用：更新账本
        """
        ...
