"""卖出规则共享逻辑

回测（T0 排队 / T+1 执行）与纸面交易（调仓卖出指令）共用的
持有期到期判定与调仓卖出候选筛选。

两侧的既有口径差异通过参数保留：
- 回测调仓卖出阈值 = max(0, holding_period - 1)
- 纸面调仓卖出阈值 = max(1, rebalance_freq - 1)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


def is_holding_period_expired(holding_days: int, holding_period: int) -> bool:
    """持有期是否到期（holding_days >= holding_period）。"""
    return holding_days >= holding_period


def min_holding_days_for_rebalance_sell(rebalance_freq: int, floor: int = 0) -> int:
    """调仓卖出的最低持有天数阈值。

    仅卖出已接近或超过持有期的持仓（rebalance_freq - 1），
    floor 用于保留两侧既有下限差异（回测 0 / 纸面 1）。
    """
    return max(floor, int(rebalance_freq) - 1)


@dataclass
class RebalanceSellDecision:
    """调仓卖出筛选结果。"""

    sells: List[str] = field(default_factory=list)  # 应卖出的股票代码（保持输入顺序）
    skipped_protected: int = 0  # 因保护跳过
    skipped_target: int = 0  # 因在新目标中跳过
    skipped_queued: int = 0  # 因已在其他卖出队列跳过
    skipped_too_young: int = 0  # 因未满持有期跳过


def select_rebalance_sell_candidates(
    holding_days_map: Dict[str, Optional[int]],
    min_holding_days: int,
    target_codes: Optional[Set[str]] = None,
    protected_codes: Optional[Set[str]] = None,
    queued_codes: Optional[Set[str]] = None,
) -> RebalanceSellDecision:
    """筛选调仓日应卖出的持仓（回测与纸面共用语义）。

    跳过规则（优先级从高到低，与两侧既有实现一致）：
    1. 已在其他卖出队列（避免重复卖出）
    2. 受保护股票
    3. 在新信号目标中（保留/加仓，不卖出）
    4. 持有天数未达阈值（提前调仓时保护年轻持仓）；
       holding_days 为 None 表示无法计算持有天数，不做年轻过滤

    Args:
        holding_days_map: {股票代码: 持有交易日数或 None}，迭代顺序即输出顺序
        min_holding_days: 最低持有天数阈值
        target_codes: 新信号目标股票集合
        protected_codes: 受保护股票集合
        queued_codes: 已在其他卖出队列中的股票集合

    Returns:
        RebalanceSellDecision（卖出列表 + 各类跳过计数）
    """
    targets = target_codes or set()
    protected = protected_codes or set()
    queued = queued_codes or set()

    decision = RebalanceSellDecision()
    for ts_code, holding_days in holding_days_map.items():
        if ts_code in queued:
            decision.skipped_queued += 1
            continue
        if ts_code in protected:
            decision.skipped_protected += 1
            continue
        if ts_code in targets:
            decision.skipped_target += 1
            continue
        if holding_days is not None and holding_days < min_holding_days:
            decision.skipped_too_young += 1
            continue
        decision.sells.append(ts_code)

    return decision
