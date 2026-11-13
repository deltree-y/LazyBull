# -*- coding: utf-8 -*-
"""P7 T5：原始需求、配额、日内净额、借用会计与死局（VV-H / VV-S 共用机械实现）。

模块
====
- ``params``：冻结参数（配额比率 ρ、T4 完成根引用、旧面身份）；
- ``quota``：原始目标 G → 需求 D → 常规额度 R → 授权 A（含费用预留）；
- ``borrow``：FIFO 借款 lot（部分归还、逐 lot 状态）；
- ``recall``：T+1 召回队列、``deadlocked`` 永久登记、``late_recovery``；
- ``credit``：出借方分摊 / 借入方决胜 / 本金对消（``Σcredit = 0``）；
- ``state``：``p7_t5_state/v1`` 版本化信封（不与 ``p7_t4_state/v2`` 混用）；
- ``host``：逐日生命周期宿主（复用 T4 的预净额/费用/双口径/价源）；
- ``scenarios``：黄金例与负例（``synthetic`` / ``constructed_on_history``）；
- ``runner`` / ``run_t5``：证据落盘与薄入口。

硬契约
======
1. 借款只在「原始需求超过常规额度且费用后现金不足」时发生，``L`` 只按本日转账一次入账；
2. 原始卖出不得直接入市场（仍走预净额 → 最终净额）；
3. 逐袖 ``cash_i + credit_i >= 0`` 逐日硬门、``Σ credit_i = 0``、本金转移零 P&L；
4. ``credit=off & quota=off`` 与 T4 完成根逐位一致（T4 兼容门）；
5. 零 α 消耗：``alpha_consumed=false``，借期与 cap-binding 只作机械量。
"""

T5_MODULES: tuple[str, ...] = (
    "params",
    "quota",
    "borrow",
    "recall",
    "credit",
    "state",
    "host",
    "scenarios",
    "runner",
    "restart",
    "run_t5",
)
