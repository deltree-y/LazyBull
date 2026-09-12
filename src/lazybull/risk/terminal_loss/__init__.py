"""期末异常亏损风险模型（terminal_vol_scaled_loss）

按 docs/plans/terminal_loss_risk_model_plan.md 实施的分阶段模块：

- labels: 标签、端点与成熟度校验（第一阶段）
- dataset: 多期限瘦表、时间分割、样本关联（第一阶段）
- mother_section: pct_* 完整同日母截面重建（第一阶段，方案 4.4）
- train: 二分类训练、早停、校准与元数据（第一阶段）
- model: 加载、输入契约、批量预测（第一/二阶段）
- coverage_audit: 停牌/端点缺失占比与代理条件事件率审计（方案 8.1）
- block_stats: 分块重采样区间与成对比较（方案第 6 节规则 4/5）
- artifacts: 产物落盘（注册制版本化 + 固定名别名）

因子本体归 factors/risk（含 sigma_daily_20 严格日历口径面板）；
本包只保留标签、训练、审计与统计逻辑，不复制因子实现。
"""

from .artifacts import (
    ES_PREDICTION_COLUMNS,
    TERMINAL_LOSS_LABEL_COLUMN,
    TERMINAL_LOSS_MODEL_TYPE,
    build_performance_metrics,
    save_flat_artifacts,
    save_terminal_loss_artifacts,
    save_versioned_artifacts,
)
from .block_stats import (
    BLOCK_DAYS_SENSITIVITY,
    DAY_NORM_SCORE_COL,
    ES_BLOCK_DAYS,
    ES_BLOCK_DAYS_SENSITIVITY,
    PRIMARY_BLOCK_DAYS,
    BootstrapConfig,
    add_day_percentile_score,
    block_paired_delta,
    fold_level_gate,
    moving_block_metric_ci,
    moving_block_metric_sensitivity,
)
from .coverage_audit import (
    AUDIT_PROXY_COLUMNS,
    DELAYED_SENSITIVITY_SHARE_THRESHOLD,
    ProxyProfileAccumulator,
    coverage_audit_required,
    delayed_endpoint_sensitivity,
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
    empty_training_matrix,
    load_clean_daily_panels,
    load_cs_train_days,
    load_trade_calendar,
    split_stages_with_label_isolation,
    subsample_dates,
    subsample_h_per_group,
    validate_feature_manifest,
)
from .labels import (
    COVERAGE_COLUMNS,
    LABEL_STATUS_ENDPOINT_MISSING,
    LABEL_STATUS_IMMATURE,
    LABEL_STATUS_SIGMA_UNAVAILABLE,
    LABEL_STATUS_VALID,
    LabelCoverageAccumulator,
    TerminalLossLabelConfig,
    build_terminal_loss_labels,
    summarize_label_coverage,
)
from .model import TerminalLossModel, TerminalLossModelConfig
from .mother_section import (
    MOTHER_HARD_ATOL,
    MOTHER_HISTORY_MONTHS,
    MOTHER_OUTLIER_SHARE_LIMIT,
    MOTHER_SECTION_FACTORS,
    MotherSectionCache,
    assert_mother_section_validation_ok,
    build_mother_section,
    load_clean_daily_long,
    merge_mother_section_validation,
    validate_mother_section_against_cs_train,
)
from .train import (
    SigmoidCalibrator,
    TerminalLossTrainConfig,
    TerminalLossTrainResult,
    evaluate_probability_quality,
    train_terminal_loss_model,
)

__all__ = [
    "ES_PREDICTION_COLUMNS",
    "TERMINAL_LOSS_LABEL_COLUMN",
    "TERMINAL_LOSS_MODEL_TYPE",
    "build_performance_metrics",
    "save_flat_artifacts",
    "save_terminal_loss_artifacts",
    "save_versioned_artifacts",
    "BootstrapConfig",
    "BLOCK_DAYS_SENSITIVITY",
    "DAY_NORM_SCORE_COL",
    "ES_BLOCK_DAYS",
    "ES_BLOCK_DAYS_SENSITIVITY",
    "PRIMARY_BLOCK_DAYS",
    "add_day_percentile_score",
    "block_paired_delta",
    "fold_level_gate",
    "moving_block_metric_ci",
    "moving_block_metric_sensitivity",
    "AUDIT_PROXY_COLUMNS",
    "DELAYED_SENSITIVITY_SHARE_THRESHOLD",
    "ProxyProfileAccumulator",
    "coverage_audit_required",
    "delayed_endpoint_sensitivity",
    "COVERAGE_COLUMNS",
    "LABEL_STATUS_ENDPOINT_MISSING",
    "LABEL_STATUS_IMMATURE",
    "LABEL_STATUS_SIGMA_UNAVAILABLE",
    "LABEL_STATUS_VALID",
    "LabelCoverageAccumulator",
    "TerminalLossLabelConfig",
    "build_terminal_loss_labels",
    "summarize_label_coverage",
    "MOTHER_HARD_ATOL",
    "MOTHER_HISTORY_MONTHS",
    "MOTHER_OUTLIER_SHARE_LIMIT",
    "MOTHER_SECTION_FACTORS",
    "MotherSectionCache",
    "assert_mother_section_validation_ok",
    "build_mother_section",
    "load_clean_daily_long",
    "merge_mother_section_validation",
    "validate_mother_section_against_cs_train",
    "BASE_FEATURES",
    "DERIVED_FEATURES",
    "META_COLUMNS",
    "TERMINAL_LOSS_FEATURES",
    "DatasetConfig",
    "StageSpec",
    "add_pct_features",
    "attach_horizon_features",
    "build_training_matrix",
    "empty_training_matrix",
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
