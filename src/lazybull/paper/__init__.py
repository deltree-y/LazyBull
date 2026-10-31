"""纸面交易模块"""

from .account import PaperAccount
from .broker import PaperBroker
from .models import AccountState, Fill, NAVRecord, Order, PendingBuy, PendingSell, Position, TargetWeight, TradeInstruction
from .runner import PaperTradingRunner
from .runtime import PaperTradeExecutionResult, PaperTradeRuntimeContext, create_paper_trade_runtime, execute_trade_workflow
from .storage import PaperStorage

# reporting 六符号经本包顶层 re-export 的消费面不变（F15 钉钉五符号 + 快照）；
# P4-T2 起 reporting 本体迁至 v2/hosts/paper/reporting.py（旧路径为壳），
# 壳内 import 旧 paper.* 部件会回本包——若此处保持急切 import 则成环，
# 故改 PEP 562 惰性解析（首次属性访问时经壳取回，行为逐位不变）。
_REPORTING_EXPORTS = frozenset({
    "PaperPositionSnapshot",
    "format_model_info",
    "format_next_day_instructions",
    "format_positions_mobile",
    "format_trade_result",
    "load_position_snapshot",
})


def __getattr__(name: str):
    if name in _REPORTING_EXPORTS:
        from . import reporting

        return getattr(reporting, name)
    if name in ("reporting", "performance"):
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'PaperAccount',
    'PaperBroker',
    'PaperStorage',
    'PaperTradingRunner',
    'AccountState',
    'Position',
    'Order',
    'Fill',
    'TargetWeight',
    'NAVRecord',
    'PendingBuy',
    'PendingSell',
    'TradeInstruction',
    'PaperTradeRuntimeContext',
    'PaperTradeExecutionResult',
    'PaperPositionSnapshot',
    'create_paper_trade_runtime',
    'execute_trade_workflow',
    'format_model_info',
    'format_next_day_instructions',
    'load_position_snapshot',
    'format_positions_mobile',
    'format_trade_result',
]
