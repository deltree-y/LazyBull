# -*- coding: utf-8 -*-
"""P7 T4：P7 专用生命周期、价源、双口径与状态。

分包职责（全部为 **P7 专用新增件**，不就地修改任何受审实现）
==========================================================
- :mod:`params`   冻结参数、T2.5 输入身份、受审旧面 SHA 清单；
- :mod:`fees`     Q3 纯函数（S5 实际费用分摊、独立费用聚合、有符号节省）；
- :mod:`price`    成交价（只认真实 T+1 开盘）与三级估值解析（逐股记日期/来源）；
- :mod:`state`    P7 专用状态信封（lot/cash/credit/残差/借款/召回/死局/幂等键/generation）；
- :mod:`netting`  净额化生命周期（S1/S2/S3/S4/S5/S6/S7，零副作用预净额）；
- :mod:`ledger`   物理/虚拟账本落账与 I1/I2/I3 双口径；
- :mod:`scenarios` 自然实例提取 + 历史状态上的构造正负例；
- :mod:`host`     ``HistoricalVirtualSleeveHost``（跨折连续账户逐日驱动）；
- :mod:`runner`   编排、证据落盘与自验。

边界：本包不产生任何 pass/fail 经济判定（``alpha_consumed=false``），
不改 ``src/lazybull/**``，也不接入生产装配。
"""
