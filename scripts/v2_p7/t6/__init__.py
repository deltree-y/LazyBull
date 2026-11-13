# -*- coding: utf-8 -*-
"""P7 T6：两臂运行、绩效与三连指标经济裁定（``alpha_consumed=true``）。

模块划分
========
- ``calibration``：§6.2 预登记判据（**运行前冻结**，含阈值与噪声带定义 + 稳定摘要）；
- ``params``：T6 冻结输入 = T5 冻结参数 + T5 完成根身份（逐件 SHA）+ T5 模块 SHA；
- ``arms``：两臂（VV-H / VV-S）全窗运行与确定性重跑（只读复用 T4/T5 宿主）；
- ``daily``：逐日/逐袖序列装配（nav / equity / contribution / drawdown / turnover / fees / credit）；
- ``metrics``：绩效、回撤、稳定性、链式指标、费用与换手（全部走既有唯一入口）；
- ``noise``：配对块 bootstrap 噪声带与分年符号检验；
- ``verdict``：C1–C5 逐条判定与 ``verdict``；
- ``golden``：§4.6 黄金例三条（口径锚）；
- ``gates``：G1–G7 机械门；
- ``artifacts``：证据九件套装配与落盘；
- ``runner``：编排；``run_t6``：薄入口。

边界
====
``verdict`` 只对本窗口、本机械与费用口径成立；不构成阶段授权、B 区转正、
生产多袖能力、前向/日历验证或实盘可用结论。
"""

from __future__ import annotations

SCHEMA = "p7_t6/v1"
TASK = "P7 T6 — 两臂运行、绩效与三连指标经济裁定"

__all__ = ["SCHEMA", "TASK"]
