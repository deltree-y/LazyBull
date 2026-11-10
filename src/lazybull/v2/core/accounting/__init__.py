"""v2 核算层：组合估值、净值曲线、成交导出、引擎状态初始化、引擎状态快照（engine_state）、
物理账本（ledger），以及 P6.5 合成对账件：虚拟账本（virtual_ledger）、物理 lot 伴随视图
（physical_lot_view）、三恒等式校验器（reconciliation）与 T2 内部结算批次
（settlement_batch：Q13 费用瀑布 / Q18 股数分摊 / 结算后 I1/I2 回验）。"""
