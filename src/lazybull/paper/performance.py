"""纸面账户绩效口径兼容壳（P4-T2 起本体迁至 ``v2/hosts/paper/performance.py``）。

本壳仅 re-export 保持旧导入路径（同一函数对象，行为逐位不变）；
年化/起始日/净值序列唯一实现见 ``src.lazybull.v2.hosts.paper.performance``
（D3 过渡例外，观察期冻结禁改，docs/plans/v2/p4_implementation_plan_20261026.md）。
本壳保留至旧链路物理删除（壳删除前提 = 树莓派等消费方迁移完成，F15）。
"""

from src.lazybull.v2.hosts.paper.performance import *  # noqa: F401,F403
from src.lazybull.v2.hosts.paper.performance import (
    build_account_value_series,
    calculate_annualized_return,
    load_account_start_date,
)

__all__ = [
    "build_account_value_series",
    "calculate_annualized_return",
    "load_account_start_date",
]
