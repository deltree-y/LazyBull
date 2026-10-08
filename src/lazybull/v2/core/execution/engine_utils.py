"""回测引擎组装辅助纯函数（P2a-T4；§3.3 映射到组装体的 4 个方法）。

按 T0 规划 §3.3，``_get_suspend_calendar`` / ``_get_rebalance_dates`` /
``_snapshot_early_rebalance_state`` / ``_restore_early_rebalance_state``
的目标落位是组装体 ``core/execution/engine.py``；组装体随 T5 建立
（V2R-05：mixin 基类齐备前不存在可组装的中间态），故本 task 以独立
纯函数形式先行交付，供 T5 组装接线。方法转函数、显式入参替代 self，
语义与操作顺序逐字保留。
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.lazybull.data.storage import Storage  # 只读过渡依赖（§3.5；去除节点 P2b/P3）
from src.lazybull.v2.common.rules.stagger import compute_tranche_schedule
from src.lazybull.v2.common.suspend_calendar import SuspendCalendar, get_suspend_calendar


def ensure_suspend_calendar(
    suspend_calendar: Optional[SuspendCalendar],
    data_storage: Optional[Storage],
) -> Tuple[SuspendCalendar, Optional[Storage]]:
    """获取停牌日历实例（延迟创建，共用 common 构建函数）

    纯函数形态：原方法把实例缓存在 ``self._suspend_calendar`` 并回写
    ``self.data_storage``；现由调用方持有返回值并回写（组装体负责缓存）。

    Args:
        suspend_calendar: 已缓存的停牌日历实例，None 表示尚未创建
        data_storage: 数据存储实例，None 时由 common 构建函数创建默认实例

    Returns:
        (calendar, data_storage)：日历实例与实际使用的存储实例
    """
    if suspend_calendar is None:
        suspend_calendar, data_storage = get_suspend_calendar(data_storage)

    return suspend_calendar, data_storage


def get_rebalance_dates(
    trading_dates: List[pd.Timestamp],
    rebalance_freq: int,
    stagger_tranches: int,
) -> Dict[pd.Timestamp, int]:
    """获取调仓日期及对应的 tranche 索引（委托 common.rules.stagger 共享实现）

    Args:
        trading_dates: 交易日列表
        rebalance_freq: 调仓频率（交易日数）
        stagger_tranches: 分批调仓批次数（1=不分批）

    Returns:
        字典 {日期: tranche_idx}。stagger_tranches=1 时所有日期的 tranche 均为 0。
    """
    return compute_tranche_schedule(trading_dates, rebalance_freq, stagger_tranches)


def snapshot_early_rebalance_state(
    last_ranked_candidates: list,
    last_signal_date: Optional[pd.Timestamp],
    last_rebalance_nav: Optional[float],
) -> Dict:
    """快照提前调仓可能污染的状态，用于失败时回滚。

    原方法以 ``getattr(self, "_last_rebalance_nav", None)`` 防御性读取；
    纯函数形态下该防御由调用方（组装体）在取值时保留。
    """
    snapshot = {
        "last_ranked_candidates": list(last_ranked_candidates),
        "last_signal_date": last_signal_date,
        "last_rebalance_nav": last_rebalance_nav,
    }
    return snapshot


def restore_early_rebalance_state(snapshot: Dict) -> Dict:
    """回滚提前调仓过程中修改的状态（纯函数形态：返回应回写的字段，组装体负责赋值）。

    语义与旧方法一致：``last_rebalance_nav`` 仅在快照值非 None 时回写
    （返回 dict 不含该键 = 调用方不回写），其余两个字段无条件回写。
    """
    restored = {
        "last_ranked_candidates": snapshot["last_ranked_candidates"],
        "last_signal_date": snapshot["last_signal_date"],
    }
    if snapshot["last_rebalance_nav"] is not None:
        restored["last_rebalance_nav"] = snapshot["last_rebalance_nav"]
    return restored
