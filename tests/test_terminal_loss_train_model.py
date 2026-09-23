"""期末异常亏损训练与模型封装测试（合成数据）"""

import warnings

import numpy as np
import pandas as pd
import pytest
import joblib

from src.lazybull.risk.terminal_loss.model import (
    MODEL_ARTIFACT_VERSION,
    RANK_TIE_BREAK_EPS,
    TerminalLossModel,
    TerminalLossModelConfig,
    _restore_ranking_resolution,
)
from src.lazybull.risk.terminal_loss.train import (
    EVAL_METRIC_AUC,
    EVAL_METRIC_NDCG,
    EVAL_METRIC_RANK_IC_DAILY,
    OBJECTIVE_RANK_PAIRWISE,
    SigmoidCalibrator,
    TerminalLossTrainConfig,
    evaluate_probability_quality,
    train_terminal_loss_model,
)

FEATURES = ["f1", "f2", "remaining_intervals"]


def _synthetic_matrix(n: int, seed: int, signal: float = 2.0, with_dates: int = 20):
    """弱信号合成矩阵：p = sigmoid(signal * f1)，标签按 p 伯努利抽样。"""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    h = rng.integers(1, 21, n)
    p = 1 / (1 + np.exp(-signal * f1))
    y = (rng.random(n) < p).astype(int)
    dates = pd.bdate_range("2024-01-02", periods=with_dates).strftime("%Y%m%d")
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


class TestTrain:
    def test_train_runs_with_early_stopping_and_weights(self):
        train_df = _synthetic_matrix(800, seed=1)
        es_df = _synthetic_matrix(300, seed=2)
        result = train_terminal_loss_model(
            train_df,
            es_df,
            FEATURES,
            TerminalLossTrainConfig(n_estimators=120, early_stopping_rounds=10, device="cpu"),
        )
        assert 0 < result.best_iteration <= 120
        meta = result.to_metadata()
        assert meta["train_config"]["regularization_scale_policy"].startswith("A_")
        assert meta["n_train"] == 800 and meta["n_val"] == 300
        # 早停段事件率与 h 分布登记（评估段 ES 的行数/事件率由调用方登记）
        assert 0.0 < meta["val_event_rate"] < 1.0
        assert meta["val_h_distribution"]
        assert 0.0 < meta["train_event_rate"] < 1.0

    def test_rank_ic_daily_early_stopping_metric(self, tmp_path):
        """早停指标可切逐日截面 RankIC（与门禁 lift 同向），口径入元数据。"""
        train_df = _synthetic_matrix(800, seed=11)
        es_df = _synthetic_matrix(300, seed=12)
        result = train_terminal_loss_model(
            train_df,
            es_df,
            FEATURES,
            TerminalLossTrainConfig(
                n_estimators=60,
                early_stopping_rounds=10,
                device="cpu",
                eval_metric="rank_ic_daily",
            ),
        )
        assert 0 < result.best_iteration <= 60
        assert result.to_metadata()["train_config"]["eval_metric"] == "rank_ic_daily"
        # 指标对象必须可 pickle（模型注册 joblib.dump 要求，闭包会触发 PicklingError）
        joblib.dump(result.classifier, tmp_path / "model.joblib")

    def test_unknown_eval_metric_raises(self):
        """未知早停指标必须明确失败，不静默回退到 logloss。"""
        train_df = _synthetic_matrix(200, seed=13)
        es_df = _synthetic_matrix(100, seed=14)
        with pytest.raises(ValueError, match="不支持的早停指标"):
            train_terminal_loss_model(
                train_df,
                es_df,
                FEATURES,
                TerminalLossTrainConfig(device="cpu", eval_metric="aucpr"),
            )

    def test_rank_ic_daily_requires_trade_date(self):
        """rank_ic_daily 需要 Val 矩阵带 trade_date 分组列，缺列明确报错。"""
        train_df = _synthetic_matrix(200, seed=15).drop(columns=["trade_date"])
        es_df = _synthetic_matrix(100, seed=16).drop(columns=["trade_date"])
        with pytest.raises(ValueError, match="需要 Val 矩阵包含 trade_date"):
            train_terminal_loss_model(
                train_df,
                es_df,
                FEATURES,
                TerminalLossTrainConfig(device="cpu", eval_metric="rank_ic_daily"),
            )

    def test_early_stopping_follows_val_segment(self):
        """早停只由 Val 段（内部验证段）决定：Val 信号强弱改变 best_iteration。

        门禁评估段 ES 不参与早停（方案 5.2）：本函数签名里根本没有 ES 矩阵，
        ES 段仅在调用方（训练脚本）用于概率质量报告与门禁。
        """
        train_df = _synthetic_matrix(1500, seed=21, signal=2.0)
        strong_val = _synthetic_matrix(600, seed=22, signal=3.0)
        weak_val = _synthetic_matrix(600, seed=23, signal=0.0)
        cfg = dict(n_estimators=200, early_stopping_rounds=10, device="cpu")
        it_strong = train_terminal_loss_model(
            train_df, strong_val, FEATURES, TerminalLossTrainConfig(**cfg)
        ).best_iteration
        it_weak = train_terminal_loss_model(
            train_df, weak_val, FEATURES, TerminalLossTrainConfig(**cfg)
        ).best_iteration
        # 噪声 Val 立即触底（早停），有信号 Val 持续改善 → 树数明显更多
        assert it_strong > it_weak

    def test_stage_dates_include_val(self):
        """三段（train/val/es）日期必须落入元数据供审计。"""
        train_df = _synthetic_matrix(300, seed=31)
        val_df = _synthetic_matrix(150, seed=32)
        result = train_terminal_loss_model(
            train_df,
            val_df,
            FEATURES,
            TerminalLossTrainConfig(device="cpu", n_estimators=20),
            stage_dates={
                "train": ("20230101", "20231231"),
                "val": ("20240101", "20240630"),
                "es": ("20240701", "20241231"),
            },
        )
        meta = result.to_metadata()
        assert meta["stage_dates"]["val"] == ["20240101", "20240630"]
        assert meta["stage_dates"]["es"] == ["20240701", "20241231"]
        assert "n_es" not in meta  # 评估段行数由调用方登记，避免两处口径分叉

    def test_missing_feature_column_raises(self):
        train_df = _synthetic_matrix(100, seed=3)
        es_df = _synthetic_matrix(50, seed=4)
        with pytest.raises(ValueError, match="缺少特征列"):
            train_terminal_loss_model(train_df, es_df, FEATURES + ["ghost_col"])

    def test_empty_stage_rejected(self):
        train_df = _synthetic_matrix(100, seed=5)
        with pytest.raises(ValueError, match="为空"):
            train_terminal_loss_model(train_df, train_df.iloc[:0], FEATURES)


class TestEvaluateQuality:
    def test_perfect_prediction(self):
        rng = np.random.default_rng(6)
        y = (rng.random(500) < 0.3).astype(int)
        p = np.where(y == 1, 0.99, 0.01)
        h = rng.integers(1, 21, 500)
        report = evaluate_probability_quality(y, p, h)
        assert report["pr_auc"] == pytest.approx(1.0)
        assert report["logloss"] < 0.1
        assert report["brier"] < 0.02
        assert {r["h"] for r in report["by_h"]} <= set(range(1, 21))

    def test_by_h_and_sigma_calibration_table(self):
        rng = np.random.default_rng(7)
        y = (rng.random(400) < 0.4).astype(int)
        p = np.clip(y * 0.5 + rng.normal(0.25, 0.1, 400), 0.01, 0.99)
        h = np.repeat([5, 10], 200)
        sigma = np.abs(rng.normal(0.02, 0.01, 400))
        report = evaluate_probability_quality(y, p, h, sigma_values=sigma, sigma_bins=4)
        calib = report["calibration_by_h_sigma"]
        assert set(calib["h"]) == {5, 10}
        # 每个 h 至少覆盖 q1..q4 四个分位组
        for _, grp in calib.groupby("h"):
            assert set(grp["sigma_bin"]) >= {"q1", "q2", "q3", "q4"}
        assert calib["n"].sum() == 400

    def test_nan_sigma_handled(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.8, 0.3, 0.7])
        h = np.array([1, 1, 2, 2])
        sigma = np.array([np.nan, 0.02, np.nan, 0.03])
        report = evaluate_probability_quality(y, p, h, sigma_values=sigma)
        assert "calibration_by_h_sigma" in report


class TestSigmoidCalibrator:
    def test_monotone_mapping(self):
        rng = np.random.default_rng(8)
        scores = rng.normal(0, 1, 1000)
        y = (rng.random(1000) < 1 / (1 + np.exp(-2 * scores))).astype(int)
        calib = SigmoidCalibrator().fit(scores, y)
        mapped = calib.transform(np.array([-2.0, 0.0, 2.0]))
        assert mapped[0] < mapped[1] < mapped[2]
        assert ((mapped > 0) & (mapped < 1)).all()

    def test_single_class_rejected(self):
        with pytest.raises(ValueError, match="单一类别"):
            SigmoidCalibrator().fit(np.array([1.0, 2.0]), np.array([1, 1]))

    def test_unfitted_transform_rejected(self):
        with pytest.raises(RuntimeError, match="未拟合"):
            SigmoidCalibrator().transform(np.array([1.0]))


class TestTerminalLossModel:
    def _model(self) -> TerminalLossModel:
        train_df = _synthetic_matrix(400, seed=9)
        es_df = _synthetic_matrix(200, seed=10)
        result = train_terminal_loss_model(
            train_df,
            es_df,
            FEATURES,
            TerminalLossTrainConfig(n_estimators=40, early_stopping_rounds=5, device="cpu"),
        )
        config = TerminalLossModelConfig(
            task_id=result.label_config.task_id,
            feature_names=result.feature_names,
            train_config=result.train_config,
            label_config=result.label_config,
            metadata=result.to_metadata(),
        )
        return TerminalLossModel(config, result.classifier)

    def test_predict_proba_contract(self):
        model = self._model()
        df = pd.DataFrame({"f1": [0.1, -0.2], "f2": [0.0, 0.3], "remaining_intervals": [5, 19]})
        p = model.predict_proba(df)
        assert len(p) == 2
        assert ((p > 0) & (p < 1)).all()

    def test_missing_column_raises(self):
        model = self._model()
        df = pd.DataFrame({"f1": [0.1], "remaining_intervals": [5]})
        with pytest.raises(ValueError, match="缺少特征列"):
            model.predict_proba(df)

    def test_save_load_roundtrip(self, tmp_path):
        model = self._model()
        path = str(tmp_path / "terminal_loss_model.joblib")
        model.save(path)
        loaded = TerminalLossModel.load(path)
        assert loaded.feature_names == model.feature_names
        df = pd.DataFrame({"f1": [0.3], "f2": [0.1], "remaining_intervals": [10]})
        np.testing.assert_allclose(loaded.predict_proba(df), model.predict_proba(df))
        assert (tmp_path / "terminal_loss_model.json").exists()

    def test_artifact_version_mismatch_rejected(self, tmp_path):
        model = self._model()
        path = str(tmp_path / "bad_version.joblib")
        joblib.dump(
            {
                "artifact_version": MODEL_ARTIFACT_VERSION + 1,
                "config": {"task_id": "x", "feature_names": FEATURES},
                "classifier": model._clf,
            },
            path,
        )
        with pytest.raises(ValueError, match="版本不符"):
            TerminalLossModel.load(path)

    def test_load_suppresses_xgboost_pickle_warning(self, monkeypatch):
        """load 抑制 XGBoost 跨版本反序列化告警（旧模型 unlock 时的日志噪音）。"""
        payload = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "config": {"task_id": "x", "feature_names": FEATURES},
            "classifier": object(),
        }
        msg = (
            "[10:22:34] WARNING: If you are loading a serialized model "
            "(like pickle in Python, RDS in R) or\n"
            "configuration generated by an older version of XGBoost"
        )

        def fake_load(path):
            warnings.warn(msg, UserWarning)
            return payload

        monkeypatch.setattr(joblib, "load", fake_load)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = TerminalLossModel.load("dummy.joblib")
        assert model.feature_names == FEATURES


class TestRankingResolutionRestore:
    """isotonic 档位内恢复严格排序（v0.112.2；小改动、不改变概率语义）。"""

    def test_monotone_in_score_within_same_level(self):
        base = np.full(5, 0.2)
        scores = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        out = _restore_ranking_resolution(base, scores)
        assert np.all(np.diff(out) > 0)
        assert np.all((out >= 0.0) & (out <= 1.0))
        assert np.allclose(out, 0.2, atol=1e-8)

    def test_level_order_preserved_and_bounds_stay_in_range(self):
        base = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        scores = np.array([1.0, -1.0, 0.0, -1.0, 1.0])
        out = _restore_ranking_resolution(base, scores)
        assert out[0] > out[1]  # 同档内按原始分数
        assert out[2] > out[0] and out[3] > out[2] and out[4] > out[3]  # 档位顺序
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_identical_scores_stay_tied(self):
        base = np.array([0.3, 0.3])
        scores = np.array([1.5, 1.5])
        out = _restore_ranking_resolution(base, scores)
        assert out[0] == out[1]

    def test_perturbation_is_negligible(self):
        base = np.array([0.0, 0.5, 1.0])
        scores = np.array([-3.0, 0.0, 3.0])
        out = _restore_ranking_resolution(base, scores)
        assert np.max(np.abs(out - base)) <= 3 * RANK_TIE_BREAK_EPS


class TestRankPairwiseObjective:
    """排序目标（v0.111.0）：rank:pairwise + qid=trade_date + Val isotonic 校准。"""

    @staticmethod
    def _result(seed: int = 31, eval_metric: str = EVAL_METRIC_AUC):
        return train_terminal_loss_model(
            _synthetic_matrix(600, seed=seed),
            _synthetic_matrix(300, seed=seed + 1),
            FEATURES,
            TerminalLossTrainConfig(
                n_estimators=60,
                early_stopping_rounds=5,
                device="cpu",
                objective=OBJECTIVE_RANK_PAIRWISE,
                eval_metric=eval_metric,
            ),
        )

    @staticmethod
    def _model(result) -> TerminalLossModel:
        config = TerminalLossModelConfig(
            task_id=result.label_config.task_id,
            feature_names=result.feature_names,
            train_config=result.train_config,
            label_config=result.label_config,
            metadata=result.to_metadata(),
        )
        return TerminalLossModel(config, result.classifier, result.calibrator)

    def test_trains_ranker_with_val_isotonic_calibration(self):
        result = self._result()
        assert type(result.classifier).__name__ == "XGBRanker"
        assert result.calibrator is not None
        meta = result.to_metadata()
        # 目标与校准口径必须入元数据（无校准的排序分数不是概率，政策层不可用）
        assert meta["objective"] == OBJECTIVE_RANK_PAIRWISE
        assert meta["calibration"] == "isotonic_val"
        # 默认停点口径是池化 AUC（自实现回调 ValPooledAUCStopping，与门禁第三
        # 判据 auc_lift 同向；不用 XGBoost 内置 ranking auc 的 O(n²) 成对展开）
        assert meta["train_config"]["eval_metric"] == EVAL_METRIC_AUC
        assert 0 < result.best_iteration <= 60

    def test_ndcg_metric_still_supported_for_rank(self):
        """ndcg 保留供对照：显式指定时可用，但记录在签名里（em=ndcg）。"""
        result = self._result(eval_metric=EVAL_METRIC_NDCG)
        assert type(result.classifier).__name__ == "XGBRanker"
        assert result.to_metadata()["train_config"]["eval_metric"] == EVAL_METRIC_NDCG

    def test_rank_ic_daily_rejected_for_rank_objective(self):
        """逐日 RankIC 是逐行 callable，在 XGBRanker 的 qid eval_set 下不兼容。"""
        with pytest.raises(ValueError, match="不允许 eval_metric"):
            train_terminal_loss_model(
                _synthetic_matrix(200, seed=61),
                _synthetic_matrix(100, seed=62),
                FEATURES,
                TerminalLossTrainConfig(
                    device="cpu",
                    objective=OBJECTIVE_RANK_PAIRWISE,
                    eval_metric=EVAL_METRIC_RANK_IC_DAILY,
                ),
            )

    def test_calibrated_probability_is_monotone_in_score(self):
        result = self._result()
        model = self._model(result)
        df = pd.DataFrame(
            {
                "f1": [0.1, -0.2, 0.5, 0.0],
                "f2": [0.0, 0.3, -0.1, 0.2],
                "remaining_intervals": [5, 19, 3, 10],
            }
        )
        prob = model.predict_proba(df)
        raw = result.classifier.predict(df[FEATURES])
        assert ((prob >= 0) & (prob <= 1)).all()
        order = np.argsort(raw)
        assert np.all(np.diff(prob[order]) >= -1e-12)

    def test_calibration_keeps_full_ranking_resolution(self):
        """校准档位不得压平排序：输出必须恢复全分辨率的严格序（v0.112.2）。

        背景（实测）：sklearn 1.9 的 IsotonicRegression 在大样本上把拟合压缩
        成很少的档位（合成 20 万样本 → transform 只有 10 个不同输出；生产
        rank 折 100 万行 Val → 67–127 档），直接当 p_loss 输出会让每日只剩
        几十个不同分数（实测 26–289 vs binary 5800–9325），门禁三个只看排序
        的判据被系统性压低。
        """
        result = self._result()
        model = self._model(result)
        n = 400
        frame = pd.DataFrame(
            {
                "f1": np.linspace(-3.0, 3.0, n),
                "f2": np.linspace(2.0, -2.0, n),
                "remaining_intervals": np.arange(n) % 20 + 1,
            }
        )
        prob = model.predict_proba(frame)
        raw = result.classifier.predict(frame[FEATURES])
        assert ((prob >= 0) & (prob <= 1)).all()
        # 全分辨率：不同分数 ⇒ 不同输出（只有同分数才允许并列）
        assert len(np.unique(prob)) == len(np.unique(raw))
        order = np.argsort(raw, kind="stable")
        assert np.all(np.diff(prob[order]) >= 0)

    def test_save_load_roundtrip_keeps_calibrator(self, tmp_path):
        model = self._model(self._result())
        path = str(tmp_path / "rank_model.joblib")
        model.save(path)
        loaded = TerminalLossModel.load(path)
        df = pd.DataFrame({"f1": [0.3], "f2": [0.1], "remaining_intervals": [10]})
        np.testing.assert_allclose(loaded.predict_proba(df), model.predict_proba(df))
        # 排序模型 + 校准器必须可 pickle（模型注册 joblib.dump 的硬要求）
        joblib.dump(model, tmp_path / "registry.joblib")

    def test_rank_model_without_calibrator_rejected(self):
        result = self._result()
        config = TerminalLossModelConfig(
            task_id=result.label_config.task_id,
            feature_names=result.feature_names,
            train_config=result.train_config,
            label_config=result.label_config,
        )
        model = TerminalLossModel(config, result.classifier)  # 故意不传校准器
        df = pd.DataFrame({"f1": [0.3], "f2": [0.1], "remaining_intervals": [10]})
        with pytest.raises(ValueError, match="必须携带 Val 段 isotonic 校准器"):
            model.predict_proba(df)

    def test_objective_and_metric_combinations_validated(self):
        train_df = _synthetic_matrix(200, seed=41)
        val_df = _synthetic_matrix(100, seed=42)
        # 排序目标不允许概率口径早停（logloss 是二分类默认，必须显式换口径）
        with pytest.raises(ValueError, match="不允许 eval_metric"):
            train_terminal_loss_model(
                train_df,
                val_df,
                FEATURES,
                TerminalLossTrainConfig(
                    device="cpu",
                    objective=OBJECTIVE_RANK_PAIRWISE,
                    eval_metric="logloss",
                ),
            )
        # 二分类目标不允许列表口径 ndcg
        with pytest.raises(ValueError, match="不允许 eval_metric"):
            train_terminal_loss_model(
                train_df,
                val_df,
                FEATURES,
                TerminalLossTrainConfig(device="cpu", eval_metric=EVAL_METRIC_NDCG),
            )
        with pytest.raises(ValueError, match="不支持的训练目标"):
            train_terminal_loss_model(
                train_df,
                val_df,
                FEATURES,
                TerminalLossTrainConfig(device="cpu", objective="lambdarank"),
            )

    def test_rank_pairwise_requires_trade_date(self):
        train_df = _synthetic_matrix(200, seed=43).drop(columns=["trade_date"])
        val_df = _synthetic_matrix(100, seed=44)
        with pytest.raises(ValueError, match="需要train矩阵包含 trade_date"):
            train_terminal_loss_model(
                train_df,
                val_df,
                FEATURES,
                TerminalLossTrainConfig(
                    device="cpu",
                    objective=OBJECTIVE_RANK_PAIRWISE,
                    eval_metric=EVAL_METRIC_AUC,
                ),
            )
