# -*- coding: utf-8 -*-
"""train_core 逐日截面 Spearman 早停指标与 best_iteration 下限监控契约测试。

背景：整段 Spearman 早停指标会被单一事件期样本（极端行情日样本）主导，
导致训练窗口尾部验证段（val_es）上早停异常提前。rank_ic_daily 将早停指标
切换为逐日截面 Spearman 均值，与 evaluate_validation_daily 的 daily_rankic
评估口径对齐；min_best_iteration 仅告警与标记，不改变模型行为。
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.lazybull.ml import train_core as tc
from src.lazybull.ml.train_core.eval import daily_spearman_mean
from src.lazybull.ml.train_core.eval import make_neg_rank_ic_daily


class DummyRegressor:
    """最小 XGBRegressor 替身，仅记录构造参数与 fit kwargs。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.feature_importances_ = np.array([0.5, 0.5], dtype=float)
        self.best_iteration = 2
        self.fit_kwargs = None

    def fit(self, X, y, **fit_kwargs):
        self.fit_kwargs = fit_kwargs
        return self

    def predict(self, X):
        return np.linspace(0.0, 1.0, len(X)) if len(X) else np.array([])


def _make_panel(n_days: int, n_stocks: int, seed: int = 7):
    """构造多日面板数据，返回 (dates, y, p) 三个等长数组。"""
    rng = np.random.default_rng(seed)
    dates_list = []
    y_list = []
    p_list = []
    for i in range(n_days):
        date = f"202401{str(i + 1).zfill(2)}"
        y = rng.normal(size=n_stocks)
        p = y * 0.8 + rng.normal(size=n_stocks) * 0.4
        dates_list.extend([date] * n_stocks)
        y_list.extend(y.tolist())
        p_list.extend(p.tolist())
    return np.array(dates_list), np.array(y_list), np.array(p_list)


def test_daily_spearman_mean_matches_scipy():
    """逐日 Spearman 均值必须与 scipy 逐日 spearmanr 后取均值一致。"""
    dates, y, p = _make_panel(n_days=6, n_stocks=40)
    expected = np.mean(
        [
            spearmanr(y[dates == d], p[dates == d])[0]
            for d in np.unique(dates)
        ]
    )
    assert daily_spearman_mean(dates, y, p) == pytest.approx(expected, abs=1e-12)


def test_daily_spearman_mean_perfect_and_inverse():
    """逐日完美单调为 +1，逐日完全反向为 -1。"""
    dates, y, _ = _make_panel(n_days=4, n_stocks=30)
    p_perfect = y * 1.0
    p_inverse = -y * 1.0
    assert daily_spearman_mean(dates, y, p_perfect) == pytest.approx(1.0)
    assert daily_spearman_mean(dates, y, p_inverse) == pytest.approx(-1.0)


def test_daily_spearman_mean_skips_invalid_days():
    """单样本日、组内常数日、NaN/inf 行均被剔除，不产生错误值。"""
    dates = np.array(["20240101"] * 5 + ["20240102"] * 1 + ["20240103"] * 5)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [9.0] + [1.0, 2.0, 3.0, 4.0, 5.0])
    p = np.array([1.0, 2.0, 3.0, 4.0, 6.0] + [7.0] + [5.0, 4.0, 3.0, 2.0, 1.0])
    # 20240102 只有 1 个样本：剔除；20240103 完全反向 -1；20240101 近似 0.9
    val = daily_spearman_mean(dates, y, p)
    expected = np.mean([spearmanr(y[:5], p[:5])[0], -1.0])
    assert val == pytest.approx(expected, abs=1e-12)


def test_daily_spearman_mean_nan_inf_dropped():
    """NaN/inf 出现在标签或预测中时按行剔除。"""
    dates = np.array(["20240101"] * 6)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    p = np.array([1.0, 2.0, 0.0, 4.0, 5.0, 6.0])
    # 第 6 行 y=NaN 被剔除后，剩余样本 p 中 0 与 3 错位，IC 降低但仍有效
    val = daily_spearman_mean(dates, y, p)
    expected = spearmanr(y[:5], p[:5])[0]
    assert val == pytest.approx(expected, abs=1e-12)


def test_neg_rank_ic_daily_closure():
    """闭包 metric：名称固定、符号取负、结果确实按日分组（乱序日期改变结果）。"""
    dates, y, p = _make_panel(n_days=6, n_stocks=40)
    metric = make_neg_rank_ic_daily(dates)
    assert metric.__name__ == "neg_rank_ic_daily"

    val = metric(y, p)
    assert val == pytest.approx(-daily_spearman_mean(dates, y, p))

    # 将 dates 打乱为与 y/p 不再对齐的顺序，逐日 IC 必须改变
    shuffled = dates.copy()
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(dates))
    shuffled = shuffled[idx]
    val_shuffled = make_neg_rank_ic_daily(shuffled)(y, p)
    assert val_shuffled != pytest.approx(val, abs=1e-9)


def test_metric_instance_picklable():
    """metric 必须可 pickle：XGBoost sklearn wrapper 会把 eval_metric 存入模型对象，
    模型注册 joblib.dump 时按模块路径查找；闭包函数无法满足，必须用模块级类实例。"""
    import io

    import joblib
    import xgboost as xgb

    dates, y, p = _make_panel(n_days=4, n_stocks=20)
    metric = make_neg_rank_ic_daily(dates)

    # metric 实例独立 pickle 往返后行为一致
    buf = io.BytesIO()
    joblib.dump(metric, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    assert restored.__name__ == "neg_rank_ic_daily"
    assert restored(y, p) == pytest.approx(metric(y, p))

    # 复现原故障链路：XGBRegressor 构造即持有 metric，dump 不再报 PicklingError
    model = xgb.XGBRegressor(eval_metric=metric, n_estimators=2, max_depth=2)
    buf2 = io.BytesIO()
    joblib.dump(model, buf2)


def _make_train_data():
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [4.0, 3.0, 2.0, 1.0]})
    y_train = pd.Series([0.01, 0.02, -0.01, 0.0])
    X_val = pd.DataFrame({"f1": [2.0, 3.0], "f2": [3.0, 2.0]})
    y_val = pd.Series([0.01, -0.01])
    df_val = pd.DataFrame({"trade_date": ["20240101", "20240102"]})
    return X_train, y_train, X_val, y_val, df_val


def test_xgb_rank_ic_daily_metric_wiring(monkeypatch):
    """rank_ic_daily：闭包 metric 注入构造参数，序列化后名称正确。"""
    monkeypatch.setattr("xgboost.XGBRegressor", DummyRegressor)

    X_train, y_train, X_val, y_val, df_val = _make_train_data()

    _, train_params, _, _ = tc.train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="regression",
        early_stopping_rounds=50,
        early_stopping_metric="rank_ic_daily",
        df_val_for_group=df_val,
        n_estimators=10,
        max_depth=2,
        min_child_weight=1,
    )

    assert train_params["eval_metric"] == "neg_rank_ic_daily"
    # 闭包 metric 名固定，保证 train_params 序列化稳定
    assert train_params["early_stopping_metric"] == "rank_ic_daily"
    assert train_params["best_iteration"] == 2
    assert train_params["best_iteration_floor_triggered"] is False


def test_xgb_rank_ic_daily_requires_val_df(monkeypatch):
    """rank_ic_daily 缺少行序一致的验证 df 时必须明确报错，禁止静默降级。"""
    monkeypatch.setattr("xgboost.XGBRegressor", DummyRegressor)

    X_train, y_train, X_val, y_val, df_val = _make_train_data()

    with pytest.raises(ValueError, match="rank_ic_daily"):
        tc.train_xgboost_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task="regression",
            early_stopping_rounds=50,
            early_stopping_metric="rank_ic_daily",
            df_val_for_group=None,
            n_estimators=10,
            max_depth=2,
            min_child_weight=1,
        )


def test_min_best_iteration_flag_triggered(monkeypatch):
    """best_iteration 低于下限时标记触发，且不改变模型/早停行为。"""
    monkeypatch.setattr("xgboost.XGBRegressor", DummyRegressor)

    X_train, y_train, X_val, y_val, df_val = _make_train_data()

    _, train_params, _, _ = tc.train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="regression",
        early_stopping_rounds=50,
        early_stopping_metric="rank_ic_daily",
        df_val_for_group=df_val,
        min_best_iteration=10,
        n_estimators=10,
        max_depth=2,
        min_child_weight=1,
    )

    assert train_params["best_iteration"] == 2
    assert train_params["best_iteration_floor_triggered"] is True


def test_min_best_iteration_flag_disabled(monkeypatch):
    """min_best_iteration=0（默认）时不触发下限标记。"""
    monkeypatch.setattr("xgboost.XGBRegressor", DummyRegressor)

    X_train, y_train, X_val, y_val, df_val = _make_train_data()

    _, train_params, _, _ = tc.train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="regression",
        early_stopping_rounds=50,
        early_stopping_metric="rank_ic_daily",
        df_val_for_group=df_val,
        min_best_iteration=0,
        n_estimators=10,
        max_depth=2,
        min_child_weight=1,
    )

    assert train_params["best_iteration_floor_triggered"] is False
