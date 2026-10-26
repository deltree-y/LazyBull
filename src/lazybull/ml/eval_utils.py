"""ML 模型评估工具模块（P3-T2 起唯一来源上移至 `v2/common/eval_utils.py`）。

本模块仅 re-export 保持旧导入路径（同一函数对象，生产行为逐位不变）；
判据数学唯一实现见 v2.common.eval_utils（P3 实施规划 §16.1-A，Q4-D）。
"""

from src.lazybull.v2.common.eval_utils import (
    compute_daily_rankic,
    compute_daily_topk_returns,
    compute_diagnostic_statistics,
    evaluate_predictions_by_date,
    print_diagnostic_report,
    summarize_daily_metrics,
)

__all__ = [
    "compute_daily_rankic",
    "compute_daily_topk_returns",
    "compute_diagnostic_statistics",
    "evaluate_predictions_by_date",
    "print_diagnostic_report",
    "summarize_daily_metrics",
]
