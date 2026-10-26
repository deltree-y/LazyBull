# -*- coding: utf-8 -*-
"""v2/train/core 训练核心子包门面（P3-T2，镜像旧 ``ml/train_core/__init__.py`` re-export）。

与旧门面差异：无 lgb（Q6-A 不迁移）；无 ``load_features_data``
（数据入口 = ``v2/train/data_adapter.py``）；freshness 两函数唯一来源 =
``v2.common.freshness``（经本门面 re-export 保持消费形态）；新增
``split_train_val_by_explicit_dates``（F10 显式日期缝）与 ``RuntimeDerivers``（§16.4 DI）。
"""

from src.lazybull.v2.common.freshness import (
    apply_event_freshness_decay,
    apply_serving_event_decay,
)

from .constants import (
    ALT_FEATURE_COLUMNS,
    CASHFLOW_QUALITY_FEATURE_COLUMNS,
    CONSENSUS_FEATURE_COLUMNS,
    CONSENSUS_REVISION_FEATURE_COLUMNS,
    CYQ_FEATURE_COLUMNS,
    DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS,
    DIVIDEND_POLICY_FEATURE_COLUMNS,
    ENHANCED_FEATURE_COLUMNS,
    EVENT_FRESHNESS_TO_VALUE_COLUMNS,
    EXPRESS_FEATURE_COLUMNS,
    FACTOR_EXCLUDE_LIST_FILE,
    FRESHNESS_STRATEGY_DROP_ALL,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY,
    FUND_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
    LHB_FEATURE_COLUMNS,
    MARGIN_FEATURE_COLUMNS,
    NORTH_FEATURE_COLUMNS,
    STATE_FRESHNESS_COLUMNS,
    attach_cashflow_quality_train_params,
    attach_cons_revision_schema_version,
    attach_dividend_train_params,
    read_cashflow_quality_schema_version,
    read_cons_revision_schema_version,
    read_dividend_policy_schema_version,
)
from .eval import (
    _rank_ic_eval_lgb,
    evaluate_validation_daily,
    neg_rank_ic,
)
from .features import (
    _factor_exclude_cache,
    _format_feature_importance_compact,
    _load_factor_exclude_list,
    filter_stable_features,
)
from .labels import (
    add_blended_return_label,
    generate_classification_labels,
    transform_labels_cs_zscore,
)
from .prepare import prepare_training_data
from .runtime_derive import RuntimeDerivers, RuntimeFamilyDeriver
from .split import (
    split_train_val_by_date,
    split_train_val_by_explicit_dates,
    split_val_for_early_stopping_by_date,
    split_val_for_selection_protocol_by_date,
)
from .weights import (
    build_rank_sample_weights,
    build_time_decay_weights,
)
from .xgb import train_xgboost_model

__all__ = [
    "ALT_FEATURE_COLUMNS",
    "CASHFLOW_QUALITY_FEATURE_COLUMNS",
    "CONSENSUS_FEATURE_COLUMNS",
    "CONSENSUS_REVISION_FEATURE_COLUMNS",
    "CYQ_FEATURE_COLUMNS",
    "DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS",
    "DIVIDEND_POLICY_FEATURE_COLUMNS",
    "ENHANCED_FEATURE_COLUMNS",
    "EVENT_FRESHNESS_TO_VALUE_COLUMNS",
    "EXPRESS_FEATURE_COLUMNS",
    "FACTOR_EXCLUDE_LIST_FILE",
    "FRESHNESS_STRATEGY_DROP_ALL",
    "FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY",
    "FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY",
    "FUND_FEATURE_COLUMNS",
    "FUNDAMENTAL_FEATURE_COLUMNS",
    "LHB_FEATURE_COLUMNS",
    "MARGIN_FEATURE_COLUMNS",
    "NORTH_FEATURE_COLUMNS",
    "STATE_FRESHNESS_COLUMNS",
    "RuntimeDerivers",
    "RuntimeFamilyDeriver",
    "_factor_exclude_cache",
    "_format_feature_importance_compact",
    "_load_factor_exclude_list",
    "_rank_ic_eval_lgb",
    "add_blended_return_label",
    "apply_event_freshness_decay",
    "apply_serving_event_decay",
    "attach_cashflow_quality_train_params",
    "attach_cons_revision_schema_version",
    "attach_dividend_train_params",
    "build_rank_sample_weights",
    "build_time_decay_weights",
    "evaluate_validation_daily",
    "filter_stable_features",
    "generate_classification_labels",
    "neg_rank_ic",
    "prepare_training_data",
    "read_cashflow_quality_schema_version",
    "read_cons_revision_schema_version",
    "read_dividend_policy_schema_version",
    "split_train_val_by_date",
    "split_train_val_by_explicit_dates",
    "split_val_for_early_stopping_by_date",
    "split_val_for_selection_protocol_by_date",
    "train_xgboost_model",
    "transform_labels_cs_zscore",
]
