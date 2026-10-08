# -*- coding: utf-8 -*-
"""v2 协议层（typing.Protocol 定义集中地，契约 = docs/contracts/protocols.md）。

依赖方向：本包只允许依赖 v2.common.types（值对象）与标准库 / pandas，
禁止依赖 store / sleeves / core / hosts / evidence 任何实现侧模块。
（包内互引仅限协议定义之间：signal → store.DataStore、hosts → core/signal，
均为类型标注引用，不引入实现侧。）
"""

from src.lazybull.v2.common.protocols.core import (
    DecisionMaker,
    Executor,
    FundScheduler,
    Ledger,
    PortfolioMerger,
)
from src.lazybull.v2.common.protocols.hosts import BacktestHost, PaperHost
from src.lazybull.v2.common.protocols.signal import (
    Signal,
    Trainer,
    WalkForwardOrchestrator,
)
from src.lazybull.v2.common.protocols.store import DataStore, FeatureBuilder

__all__ = [
    "BacktestHost",
    "DataStore",
    "DecisionMaker",
    "Executor",
    "FeatureBuilder",
    "FundScheduler",
    "Ledger",
    "PaperHost",
    "PortfolioMerger",
    "Signal",
    "Trainer",
    "WalkForwardOrchestrator",
]
