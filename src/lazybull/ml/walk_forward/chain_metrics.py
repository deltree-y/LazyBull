# -*- coding: utf-8 -*-
"""Walk-forward 串联净值指标计算（P3-T2 起唯一来源上移至 `v2/common/chain_metrics.py`）。

本模块仅 re-export 保持旧导入路径（同一函数对象，生产行为逐位不变）；
链式指标唯一实现见 v2.common.chain_metrics（P3 实施规划 §16.1-A，M9）。
"""

from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

__all__ = ["calculate_chain_metrics"]
