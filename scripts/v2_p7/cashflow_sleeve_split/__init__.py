# -*- coding: utf-8 -*-
"""CF-SPLIT-20-v1 隔离研究包（A 袖现金流因子拆分）。

本包是**阶段专用研究工具**，不入生产链路（CLAUDE §7.5 / §7.8 临时开发文件口径）：

- 只写 ``data/reports/p7_cashflow_split_20261120/**`` 与 ``temp/**``；
- 不读生产全局 registry，不写 ``data/models/**``、``data/features/**``、``data/paper/**``；
- 入口 = :mod:`scripts.v2_p7.cashflow_sleeve_split.run`（薄 CLI，禁止无参数默认全跑）。

模块划分（协议 §2.1）：

===================  ==========================================================
``spec``             冻结规格与机器预登记（唯一数值口径）
``artifacts``        campaign / run 目录布局、写域守卫、封存与发布标记
``identity``         git / 代码脏改动身份、输入清单、旧 A 锚哈希核验
``feature_routes``   逐折 F / R / B 列路由与逐列校验
``fold_clock``       折表、训练/早停/校准/隔离分段、接缝与部署时钟
``readiness``        数据与行域清点、阻断项
``probes``           七日只读小探针（不产生任何收益读数）
``budget``           探针实测成本与主批次预算
===================  ==========================================================
"""

from __future__ import annotations

__all__ = ["__version__"]

#: 包内实现版本（只用于证据登记，不参与数值身份）
__version__ = "1.0.0"
