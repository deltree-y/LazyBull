"""回测侧信号权重包装（P2a-T4，engine.py 函数级拆分件；D5：只迁回测侧语义）。

来源：旧 ``backtest/engine.py`` 的 ``_calculate_volatility`` /
``_apply_risk_budget`` / ``_normalize_signals`` / ``_kelly_weights`` /
``_estimate_stock_variance``（T0 规划 §3.3）。方法转函数、显式入参替代
self；公开语义、数值操作顺序、默认值逐字保留。

D5 边界：本模块为**回测侧**语义（全分数求和 / 静默等权回退 / 方差窗口取
``kelly_vol_window`` 属性）；纸面侧（``paper/runner/signals.py`` /
``paper/runner/pricing.py``）语义分歧属实，**迁移期禁止合并**，统一 =
P4 前置预登记裁决项（F11 登记）。

依赖处置（§3.5 登记表）：
- ``compute_kelly_weights`` / ``estimate_variance_from_prices`` 委托
  ``core/decision/sizing``（T3 已归位，改指 v2 复制件；原 T4 期的旧
  ``trading.sizing`` 只读过渡依赖随之去除）；
- ``to_trade_date_str`` 改指 v2 ``common.date_utils``（T1 已迁）。
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.v2.common.date_utils import to_trade_date_str
from src.lazybull.v2.core.decision.sizing import (
    compute_kelly_weights,
    estimate_variance_from_prices,
)

# 常量：每年交易日数量（用于年化波动率计算）
TRADING_DAYS_PER_YEAR = 252


def calculate_volatility(
    pnl_price_index: pd.Series,
    stock: str,
    end_date: pd.Timestamp,
    vol_window: int,
    vol_epsilon: float,
) -> float:
    """计算个股历史波动率（基于绩效价格，避免未来函数）

    使用 end_date 之前的 vol_window 个交易日的收益率计算波动率

    Args:
        pnl_price_index: 收盘绩效价格索引（MultiIndex: (trade_date, ts_code)）
        stock: 股票代码
        end_date: 结束日期（不包含，只使用该日期之前的数据）
        vol_window: 波动率计算窗口（交易日）
        vol_epsilon: 波动率缩放的最小波动率，防止除零

    Returns:
        年化波动率
    """
    try:
        # 获取该股票的所有绩效价格（按日期排序）
        stock_prices = pnl_price_index.xs(stock, level="ts_code").sort_index()

        # 筛选 end_date 之前的数据
        stock_prices = stock_prices[stock_prices.index < end_date]

        if len(stock_prices) < 2:
            return vol_epsilon

        # 取最近 vol_window 个交易日
        recent_prices = stock_prices.iloc[-vol_window:]

        if len(recent_prices) < 2:
            return vol_epsilon

        # 计算日收益率（显式 fill_method=None：停牌缺价不得被前向填充冒充零收益）
        returns = recent_prices.pct_change(fill_method=None).dropna()

        if len(returns) < 2:
            return vol_epsilon

        # 计算波动率（年化，假设每年252个交易日）
        vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 确保波动率不低于 epsilon
        return max(vol, vol_epsilon)

    except (KeyError, ValueError, IndexError) as e:
        logger.warning(f"计算 {stock} 波动率时出错: {e}，使用默认值 {vol_epsilon}")
        return vol_epsilon


def apply_risk_budget(
    signals: Dict[str, float],
    date: pd.Timestamp,
    pnl_price_index: pd.Series,
    vol_window: int,
    vol_epsilon: float,
) -> Dict[str, float]:
    """应用风险预算（波动率缩放）

    调整权重: adj_weight ∝ raw_weight / volatility
    然后归一化使权重和为1

    Args:
        signals: 原始信号 {stock: weight}
        date: 当前日期（买入日期）
        pnl_price_index: 收盘绩效价格索引（MultiIndex: (trade_date, ts_code)）
        vol_window: 波动率计算窗口（交易日）
        vol_epsilon: 波动率缩放的最小波动率，防止除零

    Returns:
        调整后的信号 {stock: adj_weight}
    """
    if not signals:
        return signals

    # 计算每只股票的波动率（使用 date 之前的数据）
    volatilities = {}
    for stock in signals:
        vol = calculate_volatility(pnl_price_index, stock, date, vol_window, vol_epsilon)
        volatilities[stock] = vol

    # 计算调整后的权重: raw_weight / volatility
    adj_weights = {}
    for stock, weight in signals.items():
        adj_weights[stock] = weight / volatilities[stock]

    # 归一化
    total_adj_weight = sum(adj_weights.values())
    if total_adj_weight > 0:
        for stock in adj_weights:
            adj_weights[stock] /= total_adj_weight
    else:
        # 如果总权重为0，均分
        n = len(adj_weights)
        for stock in adj_weights:
            adj_weights[stock] = 1.0 / n if n > 0 else 0.0

    return adj_weights


def normalize_signals(
    signals: Dict[str, float],
    date: pd.Timestamp,
    position_sizing: str,
    normalize_log_count: int,
    verbose: bool,
    kelly_max_leverage: float,
    price_data_cache: Optional[pd.DataFrame],
    kelly_vol_window: int,
) -> Tuple[Dict[str, float], int]:
    """将分数字典归一化为权重字典

    支持 4 种模式（由 position_sizing 控制）:
    - equal: 等权
    - score: 按预测分数线性加权
    - kelly: Kelly 最优仓位（基于分数 × 波动率估计）
    - half_kelly: 半 Kelly（Kelly 仓位的 50%，更保守）

    Kelly 公式: f* = μ / σ², 其中:
    - μ: 预期超额收益（用 ML 分数代理）
    - σ²: 收益率方差（从近期价格数据估计）
    最终 clip 到 [0, kelly_max_leverage], 再归一化总和为 1.0。

    Args:
        signals: 分数字典 {股票代码: 分数}
        date: 当前日期
        position_sizing: 仓位管理模式 equal|score|kelly|half_kelly
        normalize_log_count: 权重诊断日志计数（原 ``self._normalize_log_count``，
            只打印前5次；本函数不原地修改，以返回值回写）
        verbose: 是否输出详细日志
        kelly_max_leverage: 单只股票 Kelly 仓位上限（占总资产）
        price_data_cache: 价格数据缓存（Kelly 方差估计用）
        kelly_vol_window: Kelly 波动率估计窗口（交易日）

    Returns:
        (权重字典, 更新后的权重诊断日志计数)
    """
    if not signals:
        return {}, normalize_log_count

    sizing = position_sizing

    if sizing == "equal":
        weight = 1.0 / len(signals)
        if normalize_log_count < 5:
            scores = list(signals.values())
            s_max, s_min = max(scores), min(scores)
            logger.info(
                f"  [权重诊断 {normalize_log_count + 1}/5] equal, "
                f"n={len(signals)}, 每只权重={weight:.4f}, "
                f"分数范围=[{s_min:.4f}, {s_max:.4f}], 分数差距={s_max - s_min:.4f}"
            )
            normalize_log_count += 1
        return {stock: weight for stock in signals.keys()}, normalize_log_count

    elif sizing == "score":
        total_score = sum(signals.values())
        if total_score > 0:
            result = {stock: score / total_score for stock, score in signals.items()}
            if normalize_log_count < 5:
                weights = sorted(result.values(), reverse=True)
                w_max, w_min = weights[0], weights[-1]
                eq_weight = 1.0 / len(result)
                concentration = w_max / eq_weight
                sample_stocks = sorted(result.items(), key=lambda x: x[1], reverse=True)[:3]
                weights_str = ", ".join(
                    [f"{stock}: {weight:.4f}" for stock, weight in sample_stocks]
                )
                logger.info(
                    f"  [权重诊断 {normalize_log_count + 1}/5] score, "
                    f"n={len(result)}, 等权基准={eq_weight:.4f}, "
                    f"最高={w_max:.4f}({concentration:.1f}x), 最低={w_min:.4f}, "
                    f"示例（前3）: {weights_str}"
                )
                normalize_log_count += 1
            return result, normalize_log_count
        else:
            weight = 1.0 / len(signals)
            if verbose:
                logger.warning(f"所有分数 <= 0，回退到等权分配，每只股票权重 {weight:.4f}")
            return {stock: weight for stock in signals.keys()}, normalize_log_count

    elif sizing in ("kelly", "half_kelly"):
        return (
            kelly_weights(
                signals,
                date,
                kelly_max_leverage=kelly_max_leverage,
                price_data_cache=price_data_cache,
                kelly_vol_window=kelly_vol_window,
                verbose=verbose,
                half=(sizing == "half_kelly"),
            ),
            normalize_log_count,
        )

    else:
        weight = 1.0 / len(signals)
        return {stock: weight for stock in signals.keys()}, normalize_log_count


def kelly_weights(
    signals: Dict[str, float],
    date: pd.Timestamp,
    kelly_max_leverage: float,
    price_data_cache: Optional[pd.DataFrame],
    kelly_vol_window: int,
    verbose: bool,
    half: bool = False,
) -> Dict[str, float]:
    """计算 Kelly / 半 Kelly 仓位权重（委托 core/decision/sizing 共享实现）。

    Args:
        signals: 分数字典 {股票代码: 分数}
        date: 当前日期（方差估计取该日期及之前的数据）
        kelly_max_leverage: 单只股票 Kelly 仓位上限（占总资产）
        price_data_cache: 价格数据缓存（方差估计用）
        kelly_vol_window: Kelly 波动率估计窗口（交易日）
        verbose: 是否输出详细日志
        half: True 时计算半 Kelly
    """
    result, fallback_count = compute_kelly_weights(
        signals,
        variance_fn=lambda stock: estimate_stock_variance(
            price_data_cache, stock, date, kelly_vol_window
        ),
        half=half,
        max_leverage=kelly_max_leverage,
    )

    if verbose and result:
        mode_name = "half_kelly" if half else "kelly"
        sample = list(result.items())[:3]
        weights_str = ", ".join([f"{s}: {w:.4f}" for s, w in sample])
        logger.info(
            f"  权重方法: {mode_name}, 示例权重（前3只）: {weights_str}, "
            f"fallback={fallback_count}只"
        )
    return result


def estimate_stock_variance(
    price_data_cache: Optional[pd.DataFrame],
    stock: str,
    date: pd.Timestamp,
    kelly_vol_window: int,
) -> Optional[float]:
    """估计股票近期收益率方差,供 Kelly 仓位计算使用

    从 price_data_cache 中取近 kelly_vol_window 日收盘价,计算日收益率方差。
    数据不足时返回 None。
    """
    if price_data_cache is None:
        return None

    date_str = to_trade_date_str(date)
    stock_data = price_data_cache[
        (price_data_cache["ts_code"] == stock) & (price_data_cache["trade_date"] <= date_str)
    ]

    if len(stock_data) < 20:
        return None

    # 取最近 kelly_vol_window 条
    stock_data = stock_data.sort_values("trade_date").tail(kelly_vol_window)

    # 优先使用后复权价格，方差口径统一由 core/decision/sizing 计算
    price_col = "close_adj" if "close_adj" in stock_data.columns else "close"
    return estimate_variance_from_prices(stock_data[price_col].values.astype(float))
