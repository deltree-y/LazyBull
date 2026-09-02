# -*- coding: utf-8 -*-
"""train_core.xgb 早停参数构造契约测试。

xgboost >= 2.1 在构造函数携带 early_stopping_rounds 且 fit 未提供 eval_set 时，
会自动从训练集切出 20% 作早停验证，导致实际训练样本悄然减少。
本测试锁定"无验证集 = 构造参数中不携带早停"的语义（xgboost 3.x 升级配套）。
"""

import numpy as np
import pandas as pd

from src.lazybull.ml import train_core as tc


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
        # 返回非常数序列，避免 spearmanr 对常数输入报 RuntimeWarning
        return np.linspace(0.0, 1.0, len(X)) if len(X) else np.array([])


def _make_data(with_val: bool):
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [4.0, 3.0, 2.0, 1.0]})
    y_train = pd.Series([0.01, 0.02, -0.01, 0.00])
    if with_val:
        X_val = pd.DataFrame({"f1": [2.0, 3.0], "f2": [3.0, 2.0]})
        y_val = pd.Series([0.01, -0.01])
    else:
        X_val = pd.DataFrame({"f1": [], "f2": []})
        y_val = pd.Series([], dtype=float)
    return X_train, y_train, X_val, y_val


def test_early_stopping_dropped_without_validation_set(monkeypatch):
    """无验证集时必须移除 early_stopping_rounds，避免 xgboost 自动切分训练数据。"""
    monkeypatch.setattr("xgboost.XGBRegressor", DummyRegressor)

    X_train, y_train, X_val, y_val = _make_data(with_val=False)

    _, train_params, _, _ = tc.train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="regression",
        early_stopping_rounds=50,
        n_estimators=10,
        max_depth=2,
        min_child_weight=1,
    )

    assert "early_stopping_rounds" not in train_params


def test_early_stopping_kept_with_validation_set(monkeypatch):
    """有验证集时早停参数正常保留并传入构造函数与 fit(eval_set)。"""
    monkeypatch.setattr("xgboost.XGBRegressor", DummyRegressor)

    X_train, y_train, X_val, y_val = _make_data(with_val=True)

    model, train_params, _, _ = tc.train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="regression",
        early_stopping_rounds=50,
        n_estimators=10,
        max_depth=2,
        min_child_weight=1,
    )

    assert train_params["early_stopping_rounds"] == 50
    assert model.kwargs["early_stopping_rounds"] == 50
    assert "eval_set" in model.fit_kwargs
    assert train_params["best_iteration"] == 2
