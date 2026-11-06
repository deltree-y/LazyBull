# -*- coding: utf-8 -*-
"""P6 报告系统实现包（T0.5 起，T1 扩面）。

本包只承载本地、离线、确定性、只读报告投影的实现。当前包含：
- ``contracts``：T0.5 冻结契约（compare 映射 / canonical 字段 / 精确写白名单 / validation 根）；
- ``deps``：Jinja2 兼容上下限与缺失 fail-closed；
- ``perf_gate``：体积 / 打开 / 生成阈值判定与注入合成度量记录；
- ``validation``：validation 根、新目录守卫与隔离写盘；
- T1：``status`` 状态词/原因码、``snapshot`` 受信根与稳定私有快照、
  ``authority`` 权威内容核对、``cockpit`` 驾驶舱 projection、``instructions`` 指令/pending/
  调仓投影、``config_view`` 配置模型年化投影、``quality`` 质量聚合、``sources`` 链式净值
  有限性门与 runs/旧 WF 白名单读口、``bundle`` P5b sealed bundle 校验、``manifest`` 自封存与
  复算、``readonly_audit`` 精确写审计器。

渲染（HTML/CSV）归 T2~T4；本包**禁止 import Playwright**（仅验收侧脚本可导入）。
"""

from __future__ import annotations

__all__: list[str] = []
