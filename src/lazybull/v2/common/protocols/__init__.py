# -*- coding: utf-8 -*-
"""v2 协议层（typing.Protocol 定义集中地，契约 = docs/contracts/protocols.md）。

依赖方向：本包只允许依赖 v2.common.types（值对象）与标准库 / pandas，
禁止依赖 store / sleeves / core / hosts / evidence 任何实现侧模块。
"""

from src.lazybull.v2.common.protocols.store import DataStore, FeatureBuilder

__all__ = ["DataStore", "FeatureBuilder"]
