# -*- coding: utf-8 -*-
"""P6 报告系统实现包（T0.5 起，T1/T2 扩面）。

本包只承载本地、离线、确定性、只读报告投影与渲染的实现。当前包含：

- ``contracts``：T0.5 冻结契约（compare 映射 / canonical 字段 / 精确写白名单 / validation 根）；
- ``deps``：Jinja2 兼容上下限与缺失 fail-closed；
- ``perf_gate``：体积 / 打开 / 生成阈值判定与注入合成度量记录；
- ``validation``：validation 根、新目录守卫与隔离写盘；
- T1：``status`` 状态词/原因码、``snapshot`` 受信根与稳定私有快照、
  ``authority`` 权威内容核对、``cockpit`` 驾驶舱 projection、``instructions`` 指令/pending/
  调仓投影、``config_view`` 配置模型年化投影、``quality`` 质量聚合、``sources`` 链式净值
  有限性门与 runs/旧 WF 白名单读口、``bundle`` P5b sealed bundle 校验、``manifest`` 自封存与
  复算、``readonly_audit`` 精确写审计器；
- T2：``text_safety``/``security``/``html_check`` 安全面、``svg`` 静态图元、
  ``view_primitives``/``dashboard_sections``/``dashboard_view`` 视图层、``render`` 渲染入口、
  ``evidence`` 证据域隔离与固定横幅、``publish`` 原子发布与有界子树审计、
  ``generate`` 生成编排、``templates/`` 自包含模板。

全包**禁止 import Playwright**（仅 ``scripts/v2_p6/measure_report.py`` 等验收侧脚本在函数内
导入）；渲染与生成不得写 ``logs/``/``temp/``。
"""

from __future__ import annotations

__all__: list[str] = []
