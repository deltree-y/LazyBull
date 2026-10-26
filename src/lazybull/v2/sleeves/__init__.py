# -*- coding: utf-8 -*-
"""v2 袖子包（策略模块）：每袖子 = 规格（列集 / 标签 / 模型配置 / 排序与后处理）
+ 信号逻辑；**不承载因子计算**（因子唯一计算归属地 = ``factors/``，架构 §4.3）。

依赖方向（契约 protocols.md §7）：
- 允许：sleeves → common.types / common.protocols / store（只读接口）；
- 禁止：sleeves -/-> core / train / hosts / evidence / factors / universe；
  袖子互不 import（组合只允许发生在 core 层，「独立存活」的结构性保证）；
- 注册表 = ``_registry.py``（封闭，未知袖子 fail-fast；注册文件唯一）。

本期边界（P3 规划 §1.1-4）：袖子 A（value）规格化落地；信号实现维持
``v2/core/signal/`` 单处，规格以 ``signal_impl`` 字符串引用信号类，
由调用方（薄入口 / 宿主）解析构造。
"""

from src.lazybull.v2.sleeves._registry import get_sleeve_spec, list_sleeve_ids
from src.lazybull.v2.sleeves.spec import SleeveSpec

__all__ = ["SleeveSpec", "get_sleeve_spec", "list_sleeve_ids"]
