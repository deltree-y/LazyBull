# -*- coding: utf-8 -*-
"""P6 报告系统实现包（T0.5 起）。

本包只承载本地、离线、确定性、只读报告投影的实现。当前版本（T0.5）仅包含：
- ``contracts``：T0.5 冻结契约（compare 映射 / canonical 字段 / 精确写白名单 / validation 根）；
- ``deps``：Jinja2 兼容上下限与缺失 fail-closed；
- ``perf_gate``：体积 / 打开 / 生成阈值判定与注入合成度量记录；
- ``validation``：validation 根、新目录守卫与隔离写盘。

渲染、投影、manifest 自封存等实现归 T1~T4；本包**禁止 import Playwright**（仅验收侧脚本可导入）。
"""

from __future__ import annotations

__all__: list[str] = []
