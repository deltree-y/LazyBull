# -*- coding: utf-8 -*-
"""训练核心逻辑（train_core 子包门面，re-export 全部符号）。"""

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
from .freshness import (
    apply_event_freshness_decay,
    apply_serving_event_decay,
)
from .labels import (
    add_blended_return_label,
    generate_classification_labels,
    transform_labels_cs_zscore,
)
from .lgb import train_lightgbm_model
from .prepare import prepare_training_data
from .split import (
    load_features_data,
    split_train_val_by_date,
    split_val_for_early_stopping_by_date,
    split_val_for_selection_protocol_by_date,
)
from .weights import (
    build_rank_sample_weights,
    build_time_decay_weights,
)
from .xgb import train_xgboost_model
