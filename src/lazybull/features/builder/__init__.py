# -*- coding: utf-8 -*-
"""特征构建编排器（builder 子包门面）。

FeatureBuilder 由 cache / orchestration / helpers / factors 四个 mixin 组合而成；
静态函数拆分至 static_core / static_extra，供串行与并行路径共用。
"""

from .cache import FeatureCacheMixin
from .factors import FeatureFactorsMixin
from .helpers import FeatureHelpersMixin
from .orchestration import FeatureOrchestrationMixin
from .static_core import (
    _backfill_fundamental_proxy_features_static,
    _calculate_base_features,
    _calculate_window_features_static,
    _get_lookback_dates_static,
)
from .static_extra import (
    _add_advanced_factors_static,
    _add_filter_flags_static,
    _add_limit_flags_static,
    _add_moneyflow_features_static,
    _add_new_individual_features_static,
    _add_value_dividend_features_static,
    _apply_filters_static,
    _attach_risk_factors_static,
)


class FeatureBuilder(
    FeatureCacheMixin, FeatureOrchestrationMixin, FeatureHelpersMixin, FeatureFactorsMixin
):
    """特征构建编排器（缓存管理 + 单日构建编排 mixin 组合）。"""


__all__ = [
    "FeatureBuilder",
    "FeatureCacheMixin",
    "FeatureOrchestrationMixin",
    "FeatureHelpersMixin",
    "FeatureFactorsMixin",
    "_add_advanced_factors_static",
    "_add_filter_flags_static",
    "_add_limit_flags_static",
    "_add_moneyflow_features_static",
    "_add_new_individual_features_static",
    "_add_value_dividend_features_static",
    "_apply_filters_static",
    "_attach_risk_factors_static",
    "_backfill_fundamental_proxy_features_static",
    "_calculate_base_features",
    "_calculate_window_features_static",
    "_get_lookback_dates_static",
]
