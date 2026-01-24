"""机器学习模块"""

from .evaluation import (
    calculate_daily_ic,
    calculate_ic_statistics,
    evaluate_model_ic,
    print_ic_evaluation_report,
)
from .model_registry import ModelRegistry
from .preprocess import (
    cross_sectional_winsorize,
    cross_sectional_zscore,
    process_labels_cross_sectional,
    validate_cross_sectional_standardization,
)

__all__ = [
    "ModelRegistry",
    # 预处理
    "cross_sectional_winsorize",
    "cross_sectional_zscore",
    "process_labels_cross_sectional",
    "validate_cross_sectional_standardization",
    # 评估
    "calculate_daily_ic",
    "calculate_ic_statistics",
    "evaluate_model_ic",
    "print_ic_evaluation_report",
]
