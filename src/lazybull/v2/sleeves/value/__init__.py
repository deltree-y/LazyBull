# -*- coding: utf-8 -*-
"""袖子 A（value）：现范式迁移的价值截面袖子（规格 + 声明；信号实现维持
``v2/core/signal/`` 单处，P3 规划 §1.1-4）。

依赖方向：本包只允许依赖 ``v2/sleeves/spec.py``（规格值对象）与
common.types / common.protocols / store（只读）；**禁止 import 兄弟袖子包**
（互不 import，「独立存活」的结构性保证）。
"""

from src.lazybull.v2.sleeves.value.spec import VALUE_SLEEVE_SPEC

__all__ = ["VALUE_SLEEVE_SPEC"]
