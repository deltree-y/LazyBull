"""纸面展示与格式化兼容壳（P4-T2 起本体迁至 ``v2/hosts/paper/reporting.py``）。

本壳仅 re-export 保持旧导入路径（同一函数/常量对象，行为逐位不变）；
唯一实现见 ``src.lazybull.v2.hosts.paper.reporting``（D6 形态，
docs/plans/v2/p4_implementation_plan_20261026.md）。本壳保留至旧链路物理删除
（壳删除前提 = 钉钉/树莓派等消费方迁移完成，F15）。
"""

from src.lazybull.v2.hosts.paper.reporting import *  # noqa: F401,F403
from src.lazybull.v2.hosts.paper.reporting import (
    MODEL_INFO_MAX_CHARS,
    PaperPositionSnapshot,
    format_model_info,
    format_next_day_instructions,
    format_positions_mobile,
    format_trade_result,
    load_position_snapshot,
)

__all__ = [
    "MODEL_INFO_MAX_CHARS",
    "PaperPositionSnapshot",
    "format_model_info",
    "format_next_day_instructions",
    "format_positions_mobile",
    "format_trade_result",
    "load_position_snapshot",
]
