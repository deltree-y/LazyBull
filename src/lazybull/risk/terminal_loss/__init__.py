"""期末异常亏损风险模型（terminal_vol_scaled_loss）

按 docs/plans/terminal_loss_risk_model_plan.md 实施的分阶段模块：

- labels: 标签、端点与成熟度校验（第一阶段）
- dataset: 多期限瘦表、时间分割、样本关联（第一阶段）
- train: 二分类训练、早停、校准与元数据（第一阶段）
- model: 加载、输入契约、批量预测（第一/二阶段）

因子本体归 factors/risk（含 sigma_daily_20 严格日历口径面板）；
本包只保留标签、训练与决策逻辑，不复制因子实现。
"""

from .labels import (
    LABEL_STATUS_ENDPOINT_MISSING,
    LABEL_STATUS_IMMATURE,
    LABEL_STATUS_SIGMA_UNAVAILABLE,
    LABEL_STATUS_VALID,
    TerminalLossLabelConfig,
    build_terminal_loss_labels,
    summarize_label_coverage,
)
from .dataset import (
    BASE_FEATURES,
    DERIVED_FEATURES,
    META_COLUMNS,
    TERMINAL_LOSS_FEATURES,
    DatasetConfig,
    StageSpec,
    add_pct_features,
    attach_horizon_features,
    build_training_matrix,
    load_clean_daily_panels,
    load_cs_train_days,
    load_trade_calendar,
    split_stages_with_label_isolation,
    subsample_dates,
    subsample_h_per_group,
    validate_feature_manifest,
)
from .model import TerminalLossModel, TerminalLossModelConfig
from .train import (
    SigmoidCalibrator,
    TerminalLossTrainConfig,
    TerminalLossTrainResult,
    evaluate_probability_quality,
    train_terminal_loss_model,
)

__all__ = [
    "LABEL_STATUS_ENDPOINT_MISSING",
    "LABEL_STATUS_IMMATURE",
    "LABEL_STATUS_SIGMA_UNAVAILABLE",
    "LABEL_STATUS_VALID",
    "TerminalLossLabelConfig",
    "build_terminal_loss_labels",
    "summarize_label_coverage",
    "BASE_FEATURES",
    "DERIVED_FEATURES",
    "META_COLUMNS",
    "TERMINAL_LOSS_FEATURES",
    "DatasetConfig",
    "StageSpec",
    "add_pct_features",
    "attach_horizon_features",
    "build_training_matrix",
    "load_clean_daily_panels",
    "load_cs_train_days",
    "load_trade_calendar",
    "split_stages_with_label_isolation",
    "subsample_dates",
    "subsample_h_per_group",
    "validate_feature_manifest",
    "TerminalLossModel",
    "TerminalLossModelConfig",
    "SigmoidCalibrator",
    "TerminalLossTrainConfig",
    "TerminalLossTrainResult",
    "evaluate_probability_quality",
    "train_terminal_loss_model",
]
