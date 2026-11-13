# -*- coding: utf-8 -*-
"""P7 T2.5 工具包：A 袖逐日信号再生、相位目标与原始意图预检。

模块分工
========
- :mod:`params`：冻结输入身份与派生日历（锚批次/输入清单/T0.5 precheck 逐值校验）；
- :mod:`signal_layer`：逐折打分、锚一致性、全排名与估计价提供者、逐位重放复核；
- :mod:`target_layer`：内核同源目标选择、跨折连续持仓账、折界与 runoff 登记；
- :mod:`intent_precheck`：S1~S5 分字段计数、额度缺口/召回/重叠/相关性（估计口径）；
- :mod:`runner`：编排与证据落盘；:mod:`pilot_fold`：单折回归夹具（试点保留）。
"""
