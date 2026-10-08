"""统一止损检查逻辑

将 paper_trade.py `_check_stop_loss()` 和 BacktestEngine `_check_stop_loss()`
中重复的止损检查核心流程提取为纯函数，消除两套独立实现。

v2 迁移复制件（P2a-T1，行为冻结）：与旧模块 ``risk/stop_loss_checker.py``
双源并存至 v2 切换；差异 = import 改指（StopLossMonitor 改指 v2 复制件）+
§4.4「迁移即重构」提取式拆分（check_positions_stop_loss 圈复杂度 20 > 硬上限 12，
拆出 _check_suspension / _get_valid_price / _extract_buy_price，判定顺序与日志不变）。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from src.lazybull.v2.common.rules.stop_loss import StopLossMonitor


@dataclass
class StopLossAction:
    """止损检查结果"""

    ts_code: str
    reason: str
    triggered: bool
    trigger_type: Optional[str] = None
    is_limit_down: bool = False
    can_execute: bool = True


def _check_suspension(
    suspend_calendar, ts_code: str, trade_date: str, verbose: bool
) -> Optional[bool]:
    """停牌检查。返回 True=停牌跳过；False=未停牌；None=数据异常（中止整轮检查）。"""
    if suspend_calendar is None:
        return False
    try:
        if suspend_calendar.is_suspended(ts_code, trade_date):
            if verbose:
                logger.info(f"股票 {ts_code} 停牌，跳过止损检查")
            return True
    except FileNotFoundError as e:
        logger.error(f"停牌数据文件缺失，无法进行止损检查：{e}")
        return None
    except Exception as e:
        logger.error(f"加载停牌数据失败，无法进行止损检查：{e}")
        return None
    return False


def _get_valid_price(prices: Dict[str, float], ts_code: str, verbose: bool) -> Optional[float]:
    """价格有效性检查。返回有效收盘价；无行情或价格无效返回 None（跳过该持仓）。"""
    if ts_code not in prices:
        if verbose:
            logger.warning(f"股票 {ts_code} 无行情数据，跳过止损检查")
        return None
    current_price = prices[ts_code]
    if current_price is None or current_price <= 0:
        if verbose:
            logger.warning(f"股票 {ts_code} 价格无效（{current_price}），跳过止损检查")
        return None
    return current_price


def _extract_buy_price(pos_info) -> float:
    """提取买入价（兼容 Position 对象和 dict）。"""
    if hasattr(pos_info, "buy_price"):
        return pos_info.buy_price
    if isinstance(pos_info, dict):
        return pos_info.get("buy_trade_price", pos_info.get("buy_price", 0))
    return 0


def check_positions_stop_loss(
    positions: Dict[str, dict],
    stop_loss_monitor: StopLossMonitor,
    prices: Dict[str, float],
    limit_down_info: Dict[str, bool],
    suspend_calendar,
    trade_date: str,
    *,
    verbose: bool = True,
) -> List[StopLossAction]:
    """检查所有持仓的止损触发条件。

    这是一个纯粹的检查函数，不做任何副作用操作（不修改队列、不执行卖出）。
    调用方（paper_trade / BacktestEngine）根据返回结果各自处理后续动作。

    Args:
        positions: 持仓字典。
            - paper_trade 传入: {ts_code: Position对象}，Position.buy_price 为买入价
            - BacktestEngine 传入: {ts_code: dict}，dict['buy_trade_price'] 为买入价
        stop_loss_monitor: StopLossMonitor 实例
        prices: {ts_code: close_price} 当日收盘价
        limit_down_info: {ts_code: bool} 是否跌停
        suspend_calendar: SuspendCalendar 实例（可为 None）
        trade_date: 当前交易日字符串 YYYYMMDD
        verbose: 是否输出详细日志

    Returns:
        触发止损的动作列表
    """
    actions: List[StopLossAction] = []

    if not positions:
        if verbose:
            logger.info("当前无持仓，跳过止损检查")
        return actions

    for ts_code, pos_info in list(positions.items()):
        # ── 停牌检查 ──
        suspended = _check_suspension(suspend_calendar, ts_code, trade_date, verbose)
        if suspended is None:
            return actions
        if suspended:
            continue

        # ── 价格有效性检查 ──
        current_price = _get_valid_price(prices, ts_code, verbose)
        if current_price is None:
            continue

        is_limit_down = limit_down_info.get(ts_code, False)

        # ── 提取买入价（兼容 Position 对象和 dict） ──
        buy_price = _extract_buy_price(pos_info)

        if buy_price is None or buy_price <= 0:
            if verbose:
                logger.warning(f"股票 {ts_code} 买入价无效（{buy_price}），跳过止损检查")
            continue

        # ── 调用 StopLossMonitor 核心判断 ──
        triggered, trigger_type, reason = stop_loss_monitor.check_stop_loss(
            ts_code,
            buy_price,
            current_price,
            is_limit_down,
        )

        if triggered:
            actions.append(
                StopLossAction(
                    ts_code=ts_code,
                    reason=reason,
                    triggered=True,
                    trigger_type=trigger_type.value if trigger_type else "unknown",
                    is_limit_down=is_limit_down,
                    can_execute=not is_limit_down,
                )
            )

    if verbose:
        logger.info(f"止损检查完成：触发 {len(actions)} 个止损信号")
    return actions
