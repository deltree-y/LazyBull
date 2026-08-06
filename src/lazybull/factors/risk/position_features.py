"""持仓上下文特征

H 类因子（5 个）：风控模型独有，选股模型不涉及。

这些因子在**回测/纸面交易运行时**动态计算（非 builder.py 离线构建），
因为需要持仓状态信息（入场价、持有天数等）。

因子清单：
  days_held_ratio, unrealized_pnl, drawdown_from_entry_peak,
  pnl_relative_to_mkt, holding_rank_in_portfolio
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def build_holding_context_features(
    position: dict,
    date: str,
    current_price: float,
    market_return_over_period: float = 0.0,
    holding_period: int = 20,
    all_holdings_pnl: Optional[List[float]] = None,
) -> Dict[str, float]:
    """为单个持仓构建上下文特征。

    Args:
        position: 持仓字典，必须含 entry_price, entry_date, peak_price（可选）
        date: 当前日期 YYYYMMDD
        current_price: 当前价格
        market_return_over_period: 持仓期间市场收益率
        holding_period: 计划持有天数
        all_holdings_pnl: 所有持仓的浮动盈亏列表（用于计算组合内排名）

    Returns:
        {特征名: 值} 字典
    """
    entry_price = position.get('entry_price', current_price)
    entry_date = position.get('entry_date', date)
    peak_price = position.get('peak_price', max(entry_price, current_price))

    # 持有天数
    try:
        days_held = (pd.Timestamp(date) - pd.Timestamp(entry_date)).days
    except Exception:
        days_held = 0
    days_held = max(days_held, 0)

    # 浮盈/浮亏
    unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

    # 从持仓最高点回撤
    drawdown_peak = (peak_price - current_price) / peak_price if peak_price > 0 else 0.0

    # 超额收益
    pnl_vs_mkt = unrealized_pnl - market_return_over_period

    # 组合内排名
    if all_holdings_pnl and len(all_holdings_pnl) > 1:
        rank = sum(1 for p in all_holdings_pnl if p < unrealized_pnl) / len(all_holdings_pnl)
    else:
        rank = 0.5

    return {
        'days_held_ratio': min(days_held / holding_period, 1.0) if holding_period > 0 else 1.0,
        'unrealized_pnl': unrealized_pnl,
        'drawdown_from_entry_peak': drawdown_peak,
        'pnl_relative_to_mkt': pnl_vs_mkt,
        'holding_rank_in_portfolio': rank,
    }
