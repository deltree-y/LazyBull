"""rank:pairwise + auc 自实现池化 AUC 早停（v0.112.1）回归测试（合成数据）。

背景：XGBoost 内置 ranking AUC 对每个 query group 做 O(n_g²) 成对展开
（``auc.cu`` 的 ``RankingAUC`` 断言 ``Σ_g (n_g+2)(n_g-1)/2 < INT32_MAX``）。
terminal_loss 的 Val 段是“每个交易日一个 group × 每日数千行”，实测 6 个月
Val 的成对数 29.7 亿 > 21.47 亿上限 → GPU 后端每折 XGBoostError。
本文件锁定三类事实：
1. 大规模 query group（旧实现必然爆 O(n²) 的规模）在新实现下训练正常；
2. 逐树 margin 增量 AUC 与全量截断预测完全一致（排序等价）；
3. auc 早停不得再把评估集交回 XGBoost 内置评估路径（防回归）。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from xgboost import DMatrix, XGBRanker

import src.lazybull.risk.terminal_loss.train as train_mod
from src.lazybull.ml.train_core.eval import ValPooledAUCStopping
from src.lazybull.risk.terminal_loss.train import (
    EVAL_METRIC_AUC,
    OBJECTIVE_RANK_PAIRWISE,
    TerminalLossTrainConfig,
    train_terminal_loss_model,
)

FEATURES = ["f1", "f2", "remaining_intervals"]


def _synthetic_matrix(n: int, seed: int, days: int = 20, signal: float = 2.0) -> pd.DataFrame:
    """弱信号合成矩阵：p = sigmoid(signal * f1)，标签按 p 伯努利抽样。"""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    h = rng.integers(1, 21, n)
    p = 1 / (1 + np.exp(-signal * f1))
    y = (rng.random(n) < p).astype(int)
    dates = pd.bdate_range("2024-01-02", periods=days).strftime("%Y%m%d")
    return pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "remaining_intervals": h,
            "h": h,
            "trade_date": rng.choice(dates, n),
            "loss_label": y,
            "sample_weight": np.full(n, 1 / 20),
        }
    )


def _big_group_matrix(rows_per_day: int, days: int, seed: int) -> pd.DataFrame:
    """构造“每日一个 query group、组规模极大”的矩阵。

    旧内置 ranking AUC 的成对数 Σ(n_g+2)(n_g-1)/2 会随组规模平方增长：
    65000 行/天 × 2 天 = 42.2 亿 > INT32_MAX（实测 GPU 复现崩溃）。
    """
    rng = np.random.default_rng(seed)
    n = rows_per_day * days
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    h = rng.integers(1, 21, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-0.8 * f1))).astype(int)
    dates = np.repeat([f"2024{bday:02d}01" for bday in range(1, days + 1)], rows_per_day)
    return pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "remaining_intervals": h,
            "h": h,
            "trade_date": dates,
            "loss_label": y,
            "sample_weight": np.full(n, 1 / 20),
        }
    )


class TestPooledAUCStopping:
    def test_validates_inputs(self):
        """patience/单类标签/行数与标签不一致必须明确失败。"""
        dval = DMatrix(np.zeros((4, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="patience 必须 >= 1"):
            ValPooledAUCStopping(dval, np.array([0, 1, 0, 1]), rounds=0)
        with pytest.raises(ValueError, match="只有一个类别"):
            ValPooledAUCStopping(dval, np.array([1, 1, 1, 1]), rounds=3)
        with pytest.raises(ValueError, match="不一致"):
            ValPooledAUCStopping(dval, np.array([0, 1]), rounds=3)

    def test_incremental_margin_matches_full_prediction(self):
        """逐树 margin 增量 AUC 必须与全量截断预测完全一致（排序等价）。"""
        tr = _synthetic_matrix(3000, seed=11, days=15, signal=0.8)
        va = _synthetic_matrix(1200, seed=12, days=6, signal=0.8).sort_values(
            "trade_date", kind="stable"
        )
        tr = tr.sort_values("trade_date", kind="stable")
        dval = DMatrix(va[FEATURES], label=va["loss_label"].astype(float))
        stopper = ValPooledAUCStopping(dval, va["loss_label"].astype(int), rounds=3)
        ranker = XGBRanker(
            objective="rank:pairwise",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            device="cpu",
            callbacks=[stopper],
        )
        ranker.fit(
            tr[FEATURES],
            tr["loss_label"].astype(int),
            qid=np.unique(tr["trade_date"], return_inverse=True)[1].astype(int),
            verbose=False,
        )
        # 早停生效：最多只多跑 patience 棵（+1 为触发轮本身）
        rounds = ranker.get_booster().num_boosted_rounds()
        assert rounds <= stopper.best_iteration + 1 + 3
        assert rounds < 200
        full = ranker.predict(va[FEATURES], iteration_range=(0, stopper.best_iteration + 1))
        assert stopper.best_score == pytest.approx(
            float(roc_auc_score(va["loss_label"], full)), abs=1e-12
        )


class TestRankPairwiseAucPath:
    @staticmethod
    def _cfg(**kwargs) -> TerminalLossTrainConfig:
        base = dict(
            n_estimators=60,
            early_stopping_rounds=5,
            device="cpu",
            objective=OBJECTIVE_RANK_PAIRWISE,
            eval_metric=EVAL_METRIC_AUC,
        )
        base.update(kwargs)
        return TerminalLossTrainConfig(**base)

    def test_auc_path_bypasses_builtin_ranking_auc(self, monkeypatch):
        """auc 早停必须由自实现回调承担：不得把评估集交回内置评估路径。"""
        captured: dict = {}

        class _SpyRanker(XGBRanker):
            def __init__(self, **kwargs):
                captured["init"] = dict(kwargs)
                super().__init__(**kwargs)

            def fit(self, *args, **kwargs):
                captured["fit"] = dict(kwargs)
                return super().fit(*args, **kwargs)

        monkeypatch.setattr(train_mod, "XGBRanker", _SpyRanker)
        train_terminal_loss_model(
            _synthetic_matrix(600, seed=31),
            _synthetic_matrix(300, seed=32),
            FEATURES,
            self._cfg(n_estimators=20, early_stopping_rounds=3),
        )
        init_kwargs = captured["init"]
        fit_kwargs = captured["fit"]
        # 内置 ranking auc 会对每个 query group 做 O(n_g²) 成对展开并在大规模组上
        # 直接崩溃（auc.cu 的 int32 断言）——这里必须完全没有评估集入口
        assert fit_kwargs.get("eval_set") is None
        assert fit_kwargs.get("eval_qid") is None
        assert init_kwargs.get("eval_metric") != "auc"
        stoppers = [
            cb
            for cb in (init_kwargs.get("callbacks") or [])
            if isinstance(cb, ValPooledAUCStopping)
        ]
        assert len(stoppers) == 1
        # 回调已被清理：不得把 Val DMatrix / 逐轮 margin 序列化进模型
        assert captured["fit"].get("callbacks") is None

    def test_model_trimmed_to_stop_iteration(self):
        """模型本体必须裁剪到停点（XGBoost 官方 bst[0:best+1]），默认预测即停点。"""
        va = _synthetic_matrix(400, seed=42).sort_values("trade_date", kind="stable")
        result = train_terminal_loss_model(
            _synthetic_matrix(800, seed=41),
            va,
            FEATURES,
            self._cfg(),
        )
        assert result.classifier.get_booster().num_boosted_rounds() == result.best_iteration + 1
        default = result.classifier.predict(va[FEATURES])
        explicit = result.classifier.predict(
            va[FEATURES], iteration_range=(0, result.best_iteration + 1)
        )
        np.testing.assert_allclose(default, explicit)

    def test_large_query_groups_train_with_pooled_auc_callback(self):
        """旧实现必崩的规模（42.2 亿成对数 > INT32_MAX）必须训练成功。

        实测复现（GPU，2026-09-13）：2 天 × 65000 行的内置 ranking auc 评估
        报 ``auc.cu:520 Check failed: n_threads < std::numeric_limits<int32_t>
        ::max() (4224935000 vs. 2147483647)``；自实现回调路径无组规模上限。
        """
        tr = _big_group_matrix(rows_per_day=65000, days=2, seed=51)
        va = _big_group_matrix(rows_per_day=65000, days=2, seed=52)
        result = train_terminal_loss_model(
            tr,
            va,
            FEATURES,
            self._cfg(n_estimators=3, early_stopping_rounds=1, max_depth=2),
        )
        assert 0 <= result.best_iteration < 3
        assert result.calibrator is not None
        assert result.classifier.get_booster().num_boosted_rounds() == result.best_iteration + 1
