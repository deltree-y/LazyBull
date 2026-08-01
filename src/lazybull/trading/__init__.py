"""共享交易决策核心

回测引擎（backtest.engine）与纸面交易（paper.runner/broker）共用的
纯函数决策逻辑，消除两侧实现漂移：

- sizing: Kelly/半Kelly 权重、最小买入市值阈值、收益率方差估计
- sell_rules: 持有期到期判定、调仓卖出候选筛选
- buy_plan: 槽位-候选匹配骨架（补位/调仓买入的顺位语义）
- stagger: 分批调仓排期、槽位拆分、预算比例

设计约定：本包内函数均为纯函数（不依赖引擎/账户状态），
数据获取与执行副作用通过回调注入，日志由调用侧负责。
"""

from .buy_plan import (
    REASON_ALREADY_BOUGHT,
    REASON_EXECUTION_FAILED,
    SlotMatchResult,
    fill_slots_from_candidates,
)
from .sell_rules import (
    RebalanceSellDecision,
    is_holding_period_expired,
    min_holding_days_for_rebalance_sell,
    select_rebalance_sell_candidates,
)
from .sizing import (
    compute_kelly_weights,
    compute_min_buy_value_threshold,
    estimate_variance_from_prices,
)
from .stagger import (
    build_tranche_schedule_from_anchor,
    compute_tranche_schedule,
    get_tranche_capital_fraction,
    get_tranche_target_count,
)

__all__ = [
    "REASON_ALREADY_BOUGHT",
    "REASON_EXECUTION_FAILED",
    "SlotMatchResult",
    "fill_slots_from_candidates",
    "RebalanceSellDecision",
    "is_holding_period_expired",
    "min_holding_days_for_rebalance_sell",
    "select_rebalance_sell_candidates",
    "compute_kelly_weights",
    "compute_min_buy_value_threshold",
    "estimate_variance_from_prices",
    "build_tranche_schedule_from_anchor",
    "compute_tranche_schedule",
    "get_tranche_capital_fraction",
    "get_tranche_target_count",
]
