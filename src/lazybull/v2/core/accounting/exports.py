"""成交记录与执行归因导出（P2a-T4，engine.py 函数级拆分件）。

来源：旧 ``backtest/engine.py`` 的 ``get_trades`` /
``get_execution_attribution``（T0 规划 §3.3）。方法转函数、显式入参
替代 self；导出语义逐字保留。
"""

from typing import Dict, List

import pandas as pd


def get_trades(trades: List[Dict]) -> pd.DataFrame:
    """获取交易记录

    Args:
        trades: 交易记录列表

    Returns:
        交易记录DataFrame
    """
    return pd.DataFrame(trades)


def get_execution_attribution(execution_attribution_records: List[Dict]) -> pd.DataFrame:
    """获取信号槽位到实际买入的归因记录。

    Args:
        execution_attribution_records: 信号槽位到实际成交的旁路记录列表

    Returns:
        归因记录DataFrame
    """
    return pd.DataFrame(execution_attribution_records)
