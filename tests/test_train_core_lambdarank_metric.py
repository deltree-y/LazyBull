import numpy as np
import pandas as pd

from src.lazybull.ml import train_core as tc


class DummyRanker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.feature_importances_ = np.array([0.6, 0.4], dtype=float)
        self.best_iteration = 3
        self.fit_kwargs = None

    def fit(self, X, y, **fit_kwargs):
        self.fit_kwargs = fit_kwargs
        return self

    def predict(self, X):
        n = len(X)
        if n == 0:
            return np.array([], dtype=float)
        return np.linspace(0.0, 1.0, n)


def test_lambdarank_keeps_rank_ic_as_early_stopping_metric(monkeypatch):
    monkeypatch.setattr(tc.xgb, "XGBRanker", DummyRanker)

    X_train = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "f2": [6, 5, 4, 3, 2, 1]})
    y_train = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.00])
    X_val = pd.DataFrame({"f1": [2, 3, 4, 5], "f2": [5, 4, 3, 2]})
    y_val = pd.Series([0.02, -0.01, 0.01, 0.00])

    df_train_for_group = pd.DataFrame({
        "trade_date": ["20240102", "20240102", "20240103", "20240103", "20240104", "20240104"]
    })
    df_val_for_group = pd.DataFrame({"trade_date": ["20240105", "20240105", "20240108", "20240108"]})

    model, train_params, train_metrics, val_metrics = tc.train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="regression",
        objective_type="lambdarank",
        df_train_for_group=df_train_for_group,
        df_val_for_group=df_val_for_group,
        early_stopping_rounds=10,
        early_stopping_metric="rank_ic",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
    )

    assert model.kwargs["objective"] == "rank:pairwise"
    assert callable(model.kwargs["eval_metric"])
    assert model.kwargs["eval_metric"].__name__ == "neg_rank_ic"
    assert int(model.kwargs["n_jobs"]) >= 1
    assert "qid" in model.fit_kwargs
    assert "eval_qid" in model.fit_kwargs

    assert train_params["eval_metric"] == "neg_rank_ic"
    assert train_params["early_stopping_metric"] == "rank_ic"
    assert "mse" in train_metrics
    assert "rank_ic" in val_metrics
