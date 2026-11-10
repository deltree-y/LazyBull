"""物理成交观察点（P6.5 T3 受控注入；默认生产路径零行为变化）。

契约依据：设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md``
§4.4-3/§4.6（执行侧只加受控注入点或订单构造适配，默认生产路径行为不变）与
T3 任务 §3.1（结算批次的物理成交/现金事实必须由执行链产出，不得手工注入）。

职责与边界：
- 本模块只提供**买入侧**入账调用封装（``_buy_stock_direct`` / ``_add_to_position``
  两个调用点；``sell_execution._sell_stock_direct`` 仍直调 ``apply_fills``，本阶段
  不接入）：无观察者时逐字等价于 ``engine.ledger.apply_fills(engine, entries)``
  直调，不新增任何数值操作、不改变语句顺序，因此生产默认路径（回测/纸面/回放）
  行为逐位不变；
- 合成全链（T3）由宿主显式注入引擎属性 ``physical_fill_observer``：账本落账
  成功后逐笔回调 ``observer(entry, cash_before, cash_after)``，供
  ``hosts/order_chain.py`` 构造 ``PhysicalFillFact`` 与物理现金前后值；
- 观察者**只读**入账结果（``BookingEntry`` 与现金前后值），不得回写账户；
  回调发生在落账之后，回调异常不构成回滚——调用方必须在执行前完成全部可拒绝
  校验（零副作用要求由宿主承担，见 ``order_chain`` 的预演门）；
- 受控注入点只支持**单笔**入账批次：多笔批次无法从账户总额反推逐笔现金前后值，
  以免观察者拿到不可证的中间值（fail-closed）。
"""

from __future__ import annotations

from typing import Any, Sequence

from src.lazybull.v2.core.accounting.ledger import BookingEntry


def apply_booking_entries(engine: Any, entries: Sequence[BookingEntry]) -> None:
    """物理入账唯一入口封装（无观察者时与 ``ledger.apply_fills`` 直调等价）。

    Args:
        engine: 引擎（账户 = ``current_capital`` / ``positions`` / ``trades`` 的载体）
        entries: 入账指令序列（执行器已完成全部数值计算）

    Raises:
        ValueError: 注入观察者时收到多笔批次（逐笔现金前后值不可证）。
        LedgerError: 透传 ``Ledger.apply_fills`` 的重复/乱序/形状拒绝。
    """
    observer = getattr(engine, "physical_fill_observer", None)
    if observer is None:
        engine.ledger.apply_fills(engine, entries)
        return
    entries = list(entries)
    if len(entries) != 1:
        raise ValueError(
            f"阶段=P6.5 受控注入点：注入 physical_fill_observer 时只支持单笔入账批次，"
            f"实际 {len(entries)} 笔；多笔批次的逐笔现金前后值不可由账户总额反推"
        )
    cash_before = float(engine.current_capital)
    engine.ledger.apply_fills(engine, entries)
    observer(entries[0], cash_before, float(engine.current_capital))
