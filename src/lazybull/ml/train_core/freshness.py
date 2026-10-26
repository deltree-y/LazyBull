# -*- coding: utf-8 -*-
"""事件型 freshness 衰减（P3-T2 起唯一来源上移至 `v2/common/freshness.py`）。

本模块仅 re-export 保持旧导入路径（同一函数对象，生产行为逐位不变）；
`state_keep_event_decay` 训练/推理同一实现的唯一落点见 v2.common.freshness
（事件衰减契约；P3 实施规划 §16.1-A，Q4-D）。
"""

from src.lazybull.v2.common.freshness import (
    apply_event_freshness_decay,
    apply_serving_event_decay,
)

__all__ = [
    "apply_event_freshness_decay",
    "apply_serving_event_decay",
]
