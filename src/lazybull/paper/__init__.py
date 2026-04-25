"""纸面交易模块"""

from .account import PaperAccount
from .broker import PaperBroker
from .models import AccountState, Fill, NAVRecord, Order, PendingBuy, PendingSell, Position, TargetWeight, TradeInstruction
from .reporting import PaperPositionSnapshot, format_model_info, format_positions_mobile, format_trade_result, load_position_snapshot
from .runner import PaperTradingRunner
from .runtime import PaperTradeExecutionResult, PaperTradeRuntimeContext, create_paper_trade_runtime, execute_trade_workflow
from .storage import PaperStorage

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
    'load_position_snapshot',
    'format_positions_mobile',
    'format_trade_result',
]
