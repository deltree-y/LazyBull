"""分批调仓（staggered rebalance）共享纯函数模块

回测引擎（backtest.engine）与纸面交易（paper.runner）共用的分批调仓核心逻辑，
消除两侧实现漂移：

- 排期：将 rebalance_freq 周期均匀切分为 K 个批次，错开 rebalance_freq/K 天
- 槽位拆分：将总 top_n 拆成 K 份（余数优先分配给较早批次）
- 预算比例：每批预算 = 组合总资产 × (本批槽位数 / top_n)

设计约定：本模块内函数均为纯函数（不依赖引擎/账户状态），
日期类型在回测侧为 pd.Timestamp、纸面交易侧为 str（YYYYMMDD），
通过 generic 参数兼容。
"""

from typing import Dict, List, TypeVar

DateT = TypeVar("DateT")


def get_tranche_target_count(
    tranche_idx: int, total_target: int, stagger_tranches: int
) -> int:
    """获取当前批次应占用的目标持仓槽位数。

    将 total_target 拆成 K 份，余数优先分配给较早批次。
    例如 top_n=30, K=4 → 8/8/7/7；top_n=20, K=4 → 5/5/5/5。

    Args:
        tranche_idx: 批次索引（0-based）
        total_target: 组合最终总持仓数（top_n）
        stagger_tranches: 批次数（1=不分批）

    Returns:
        本批次应占用的槽位数
    """
    if stagger_tranches <= 1:
        return total_target

    base_count, remainder = divmod(total_target, stagger_tranches)
    return base_count + (1 if tranche_idx < remainder else 0)


def get_tranche_capital_fraction(
    tranche_idx: int, total_target: int, stagger_tranches: int
) -> float:
    """获取当前批次占组合总资产的预算比例。

    按本批槽位数占总 top_n 的比例分配，所有批次合计为 100%。

    Args:
        tranche_idx: 批次索引（0-based）
        total_target: 组合最终总持仓数（top_n）
        stagger_tranches: 批次数（1=不分批）

    Returns:
        本批预算比例（0.0 ~ 1.0）
    """
    if stagger_tranches <= 1 or total_target <= 0:
        return 1.0 if stagger_tranches <= 1 else 0.0

    return get_tranche_target_count(tranche_idx, total_target, stagger_tranches) / total_target


def compute_tranche_schedule(
    trading_dates: List[DateT], rebalance_freq: int, stagger_tranches: int
) -> Dict[DateT, int]:
    """计算全区间分批调仓排期表（回测用）。

    不分批时：每隔 rebalance_freq 天调仓一次，tranche 均为 0。
    分批时：按周期比例均匀分布 K 个 tranche。
    例如 20 日分 3 批时偏移为 0/7/13，循环间隔为 7/6/7。

    Args:
        trading_dates: 交易日列表（pd.Timestamp 或 str）
        rebalance_freq: 每个批次自身的完整持有周期（交易日）
        stagger_tranches: 批次数（1=不分批）

    Returns:
        字典 {日期: tranche_idx}
    """
    n = rebalance_freq
    if n <= 0:
        raise ValueError(f"调仓频率必须为正整数，当前值: {n}")

    if stagger_tranches <= 1:
        return {trading_dates[i]: 0 for i in range(0, len(trading_dates), n)}

    k = stagger_tranches
    schedule: Dict[DateT, int] = {}
    for t in range(k):
        start = (2 * t * n + k) // (2 * k)
        for i in range(start, len(trading_dates), n):
            schedule[trading_dates[i]] = t
    return schedule


def build_tranche_schedule_from_anchor(
    anchor_date: str,
    trade_dates: List[str],
    rebalance_freq: int,
    stagger_tranches: int,
) -> Dict[str, int]:
    """基于批次0锚定日推算分批调仓排期表（纸面交易用）。

    纸面交易没有"回测区间起点"的概念，而是以最近一次批次0调仓日为锚，
    向未来推算各批次的排期。

    例如 anchor=20260101, freq=20, K=3 时：
    - 批次0: 20260101, +20, +40, ...
    - 批次1: +7, +27, +47, ...
    - 批次2: +13, +33, +53, ...

    Args:
        anchor_date: 批次0锚定日（YYYYMMDD），即最近一次批次0调仓日
        trade_dates: 开市交易日列表（YYYYMMDD，升序）
        rebalance_freq: 每个批次自身的完整持有周期（交易日）
        stagger_tranches: 批次数（1=不分批）

    Returns:
        字典 {日期: tranche_idx}，仅包含 anchor_date 之后的调仓日
    """
    if stagger_tranches <= 1:
        # 不分批：从锚定日开始每隔 rebalance_freq 天
        schedule: Dict[str, int] = {}
        try:
            anchor_idx = trade_dates.index(anchor_date)
        except ValueError:
            return schedule
        for i in range(anchor_idx, len(trade_dates), rebalance_freq):
            schedule[trade_dates[i]] = 0
        return schedule

    # 找到锚定日在交易日列表中的索引
    try:
        anchor_idx = trade_dates.index(anchor_date)
    except ValueError:
        return {}

    # 从锚定日开始构建完整的排期（利用共享的 compute_tranche_schedule）
    remaining_dates = trade_dates[anchor_idx:]
    return compute_tranche_schedule(remaining_dates, rebalance_freq, stagger_tranches)
