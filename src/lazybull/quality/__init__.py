"""数据质量扫描与静态报告。"""

from .metrics import collect_partition_metrics

__all__ = ["collect_partition_metrics"]
