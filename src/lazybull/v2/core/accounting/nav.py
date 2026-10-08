"""组合估值与净值曲线（P2a-T4，engine.py 函数级拆分件）。

来源：旧 ``backtest/engine.py`` 的 ``_calculate_portfolio_value`` /
``_position_market_value`` / ``_generate_nav_curve``（T0 规划 §3.3）。
方法转函数、显式入参替代 self；数值操作顺序逐字保留。

含估值回退链 2 处（D4，旧 engine.py:743 / :777，原样搬运）：当日成交价
→ 仓位缓存最后已知价 → 买入价兜底；回退链收敛为单一 helper 是切换后
改进项（F11 勘误协议 §11 登记），迁移期逐点等价锁定。
"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.lazybull.v2.core.execution.price_index import PriceIndexes, get_trade_price


def calculate_portfolio_value(
    positions: Dict[str, Dict],
    current_capital: float,
    price_indexes: PriceIndexes,
    date: pd.Timestamp,
) -> float:
    """计算组合市值（基于成交价格）

    Args:
        positions: 当前持仓 {股票代码: 持仓信息}（会在估值时更新
            ``last_known_price`` 缓存，与旧方法副作用一致）
        current_capital: 当前可用资金
        price_indexes: 四套价格序列
        date: 计算日期

    Returns:
        组合总市值
    """
    market_value = 0.0

    for stock, info in positions.items():
        shares = info["shares"]
        trade_price = get_trade_price(price_indexes, date, stock)
        if trade_price is None:
            # 估值回退（D4 第 1 处，旧 engine.py:743 原样搬运）：
            # 股票当日无价格（可能已退市/停牌），使用仓位中缓存的最后已知价格
            # 避免市值突降为 0 导致净值曲线出现虚假跳水
            trade_price = info.get("last_known_price")
            if trade_price is None:
                # 兜底：使用买入价
                trade_price = info.get("buy_trade_price", 0.0)
                if trade_price > 0:
                    logger.warning(
                        f"股票 {stock} 在 {date.date()} 无价格数据，"
                        f"用买入价 {trade_price:.2f} 估值（可能已退市）"
                    )
        else:
            # 更新最后已知价格缓存
            info["last_known_price"] = trade_price

        market_value += shares * trade_price

    return current_capital + market_value


def position_market_value(
    positions: Dict[str, Dict],
    price_indexes: PriceIndexes,
    date: pd.Timestamp,
    stock: str,
) -> Optional[float]:
    """单只持仓的市值（价格口径与 ``calculate_portfolio_value`` 完全一致：
    当日成交价 → 最后已知价 → 买入价兜底），非持仓或市值不可用返回 None。

    估值回退（D4 第 2 处，旧 engine.py:777 原样搬运）。
    """
    info = positions.get(stock)
    if not info or info.get("shares", 0) <= 0:
        return None
    trade_price = get_trade_price(price_indexes, date, stock)
    if trade_price is None:
        trade_price = info.get("last_known_price")
        if trade_price is None:
            trade_price = info.get("buy_trade_price", 0.0)
    if trade_price is None or trade_price <= 0:
        return None
    return float(info["shares"]) * float(trade_price)


def generate_nav_curve(
    portfolio_values: List[Dict],
    initial_capital: float,
) -> pd.DataFrame:
    """生成净值曲线

    Args:
        portfolio_values: 组合价值历史（每日一条记录，含 portfolio_value 键）
        initial_capital: 初始资金

    Returns:
        净值曲线DataFrame
    """
    df = pd.DataFrame(portfolio_values)
    df["nav"] = df["portfolio_value"] / initial_capital
    df["return"] = df["nav"] - 1.0
    return df
