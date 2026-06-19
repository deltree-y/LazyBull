import warnings

import numpy as np
import pandas as pd

from src.lazybull.ml.train_core import filter_stable_features


def test_filter_stable_features_no_runtimewarning_on_low_dof_pairs():
    """有效配对样本不足时应直接跳过，不触发 numpy 自由度告警。"""
    n_dates = 30
    samples_per_day = 30
    n = n_dates * samples_per_day
    dates = [f"2024{(i // 30) + 1:02d}{(i % 30) + 1:02d}" for i in range(n_dates) for _ in range(samples_per_day)]

    # feature_sparse 每个交易日只有 1 个非空值：旧逻辑在 corr 内部会触发 RuntimeWarning
    sparse = np.full(n, np.nan)
    for day in range(n_dates):
        sparse[day * samples_per_day] = float(day)

    df = pd.DataFrame(
        {
            "trade_date": dates,
            "feature_sparse": sparse,
            "feature_dense": np.arange(n, dtype=float),
            "neu_y_ret_20": np.arange(n, dtype=float),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        stable_features, filter_info = filter_stable_features(
            df_train=df,
            feature_columns=["feature_sparse", "feature_dense"],
            label_column="neu_y_ret_20",
            n_splits=3,
            min_abs_ic=0.0,
        )

    assert filter_info["skipped"] is False
    assert "feature_dense" in stable_features
