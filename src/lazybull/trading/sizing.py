"""仓位规模共享逻辑

回测与纸面交易共用的 Kelly 权重计算、方差估计与最小买入市值阈值。
此前两侧各自维护近乎相同的实现，历史上多次出现口径漂移，
现统一收敛到本模块。
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np


def estimate_variance_from_prices(prices: np.ndarray) -> Optional[float]:
    """由价格序列估计日对数收益率方差。

    统一口径：过滤非有限值与非正价格后，要求至少 10 个价格点、
    5 个对数收益率，否则返回 None（由调用侧降级处理）。

    Args:
        prices: 按时间升序排列的价格数组

    Returns:
        对数收益率方差；数据不足时返回 None
    """
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) < 10:
        return None

    log_returns = np.diff(np.log(prices))
    if len(log_returns) < 5:
        return None

    return float(np.var(log_returns))


def compute_kelly_weights(
    signals: Dict[str, float],
    variance_fn: Callable[[str], Optional[float]],
    half: bool = False,
    max_leverage: float = 1.0,
) -> Tuple[Dict[str, float], int]:
    """计算 Kelly / 半 Kelly 仓位权重（回测与纸面共用）。

    f* = score_rank / σ²，分数排名为主项，波动率负相关（低波动 → 更高权重）：
    1. score_rank = 分数百分位排名（0~1），量级稳定
    2. f* = score_rank / σ²（分数高 + 波动低 → 权重高）
    3. 半 Kelly：归一化后与等权 50/50 混合，更保守
    4. 归一化总和为 1.0，再按 max_leverage 做单股上限（迭代重归一化）

    Args:
        signals: {股票代码: 原始分数}
        variance_fn: 单只股票收益率方差估计回调，无法估计时返回 None
        half: 是否使用半 Kelly
        max_leverage: 单股权重上限（>=1.0 表示不限制）

    Returns:
        (权重字典, 波动率估计失败回退到截面中位数的股票数)
    """
    n = len(signals)
    if n == 0:
        return {}, 0

    # 分数百分位排名（0~1），仅对正分数股票计算
    positive_stocks = {stock: score for stock, score in signals.items() if score > 0}
    if not positive_stocks:
        weight = 1.0 / n
        return {stock: weight for stock in signals}, 0

    sorted_stocks = sorted(positive_stocks.items(), key=lambda item: item[1])
    m = len(sorted_stocks)
    score_ranks = {stock: (idx + 1) / m for idx, (stock, _) in enumerate(sorted_stocks)}

    # 每只股票的 1/σ²（无数据时回退截面中位数）
    vol_adjusts: Dict[str, float] = {}
    fallback_stocks = []
    for stock in positive_stocks:
        vol_sq = variance_fn(stock)
        if vol_sq is not None and vol_sq > 0:
            vol_adjusts[stock] = 1.0 / float(vol_sq)
        else:
            fallback_stocks.append(stock)

    median_vol_adj = float(np.median(list(vol_adjusts.values()))) if vol_adjusts else 1.0
    for stock in fallback_stocks:
        vol_adjusts[stock] = median_vol_adj

    # f* = score_rank / σ²
    raw_kelly = {
        stock: score_ranks[stock] * vol_adjusts[stock] for stock in positive_stocks
    }

    # 非正分数股票分配中位 kelly 值
    median_kelly = float(np.median(list(raw_kelly.values()))) if raw_kelly else 1.0 / n
    for stock in signals:
        if stock not in raw_kelly:
            raw_kelly[stock] = median_kelly

    total = sum(raw_kelly.values())
    if total <= 0:
        weight = 1.0 / n
        return {stock: weight for stock in signals}, len(fallback_stocks)
    result = {stock: weight / total for stock, weight in raw_kelly.items()}

    # half_kelly: 50% kelly + 50% 等权
    if half:
        eq_weight = 1.0 / n
        result = {stock: 0.5 * weight + 0.5 * eq_weight for stock, weight in result.items()}
        total = sum(result.values())
        if total > 0:
            result = {stock: weight / total for stock, weight in result.items()}

    # 单股权重上限（迭代重归一化）
    if max_leverage < 1.0:
        for _ in range(10):
            capped = {stock: min(weight, max_leverage) for stock, weight in result.items()}
            cap_total = sum(capped.values())
            if cap_total <= 0:
                break
            result = {stock: weight / cap_total for stock, weight in capped.items()}
            if all(weight <= max_leverage + 1e-9 for weight in result.values()):
                break

    return result, len(fallback_stocks)


def compute_min_buy_value_threshold(
    total_assets: float,
    target_count: int,
    ratio: float,
) -> float:
    """计算最小买入后持仓市值阈值（回测与纸面共用口径）。

    阈值 = 总资产 / 目标持仓数 * 比例；任一输入非法时返回 0（关闭）。
    """
    ratio = float(ratio or 0.0)
    target_count = int(target_count or 0)
    total_assets = float(total_assets or 0.0)
    if ratio <= 0 or target_count <= 0 or total_assets <= 0:
        return 0.0
    avg_position_value = total_assets / float(target_count)
    return avg_position_value * ratio
