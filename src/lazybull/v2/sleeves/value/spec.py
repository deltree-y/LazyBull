# -*- coding: utf-8 -*-
"""袖子 A（value）规格实例——价值截面模型，现范式迁移（架构 §4.3）。

取值 = 裁决后的生产默认配置（逐字段来源见 P3 规划 §18.2 表）：
``scripts/batch/batch_walk_forward.ps1`` 现行默认 + ``universe/domains.py``
域定义 + ``v2/core/signal/ml_signal.py`` 生产过滤默认。MVP 只做等价复刻，
特征瘦身等改动属迁移后的预登记实验，不在规格中顺手调整。
"""

from src.lazybull.v2.sleeves.spec import SleeveSpec
from src.lazybull.v2.sleeves.value.feature_columns import VALUE_FEATURE_COLUMNS

__all__ = ["VALUE_SLEEVE_SPEC"]

#: 袖子 A 标签定义（生产：neu_y_ret_20 回归 + 截面 zscore，纯行业中性标签 blend=0）
_VALUE_LABEL_SPEC = {
    "version": 1,
    "task": "regression",
    "label_transform": "cs_zscore",
    "neutral_label_blend_weight": 0.0,
    "winsorize_p": 0.01,
    "pos_quantile": None,
    "pos_topk": None,
    "scale_pos_weight": None,
}

#: 袖子 A 特征配方（生产 13 族开关：fundamental/alt/margin/cyq/fund/express/north/
#: lhb/consensus/cashflow_quality/enhanced 开，consensus_revision/dividend_policy 关；
#: 运行时派生五族生产全关——§16.4 batch_walk_forward.ps1 实测）
_VALUE_FEATURE_RECIPE = {
    "version": 1,
    "enable_fundamental_features": True,
    "enable_alt_features": True,
    "enable_margin_features": True,
    "enable_cyq_features": True,
    "enable_fund_features": True,
    "enable_express_features": True,
    "enable_enhanced_features": True,
    "enable_north_features": True,
    "enable_lhb_features": True,
    "enable_consensus_features": True,
    "enable_consensus_revision_features": False,
    "enable_cashflow_quality_features": True,
    "enable_dividend_policy_features": False,
    "enable_availability_markers": False,
    "enable_holdertrade_features": False,
    "holdertrade_feature_set": "full",
    "enable_repurchase_features": False,
    "repurchase_feature_set": "full",
    "enable_top10fh_features": False,
    "top10fh_feature_set": "full",
    "enable_top_inst_features": False,
    "feature_stability_filter": False,
    "factor_prune": False,
    "factor_exclude_file": None,
    "max_feature_missing_ratio": 0.6,
}

#: 袖子 A 切分规格（生产 val_ratio=0.2；比例→日期换算职责在编排器，§13.3.2 D1）
_VALUE_SPLIT_SPEC = {"version": 1, "val_ratio": 0.2}

#: 袖子 A 模型超参（生产口径：批脚本超参段 + 样本权重 + freshness + 集成配套；
#: 算法名由 hyperparams.algorithm 承载，C03）
_VALUE_HYPERPARAMS = {
    "algorithm": "xgboost",
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_child_weight": 200,
    "colsample_bytree": 0.3,
    "subsample": 0.8,
    "reg_alpha": 0.03,
    "reg_lambda": 7.0,
    "gamma": 0.5,
    "objective": "mse",
    "early_stopping_rounds": 50,
    "early_stopping_metric": "rank_ic_daily",
    "min_best_iteration": 30,
    "random_state": 42,
    "rank_weight_enabled": True,
    "rank_weight_topk": 120,
    "rank_weight": 100.0,
    "rank_weight_topk_weight_mode": "linear_decay",
    "time_decay_half_life": 0.0,
    "freshness_strategy": "state_keep_event_decay",
    "event_freshness_half_life_days": 120.0,
    "ensemble_offsets": 0,
    "ensemble_seed_keep_top_ratio": 1.0,
    "ensemble_seed_keep_min_models": 3,
}

#: 袖子 A 规格（注册唯一入口 = ``v2/sleeves/_registry.py``）
VALUE_SLEEVE_SPEC = SleeveSpec(
    sleeve_id="value",
    spec_version=1,
    model_family="stock_selection",
    label_name="neu_y_ret_20",
    label_spec=_VALUE_LABEL_SPEC,
    feature_recipe=_VALUE_FEATURE_RECIPE,
    split_spec=_VALUE_SPLIT_SPEC,
    hyperparams=_VALUE_HYPERPARAMS,
    ensemble_seeds=(42, 61, 82),
    train_window_months=72,
    domain="main",
    feature_columns=VALUE_FEATURE_COLUMNS,
    top_n=20,
    holding_period_days=20,
    min_amount_ma20=50000.0,
    exclude_financial=True,
    signal_impl="src.lazybull.v2.core.signal.ml_signal.MLSignal",
)
