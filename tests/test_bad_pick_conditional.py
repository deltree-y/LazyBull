#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""条件式 Bad-Pick 模型测试

覆盖：
- BadPickConfig 序列化/反序列化
- detect_market_regime 正确性
- apply_conditional_penalty 阈值门控逻辑
- 边界：空特征、全缺失、单 regime
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.risk.bad_pick import (
    BAD_PICK_CLASSIFIER_FEATURES,
    MARKET_STATE_FEATURES,
    BadPickConfig,
    RegimeBadPickConfig,
    apply_conditional_penalty,
    detect_market_regime,
    prepare_classifier_features,
)


class TestBadPickConfig:
    """BadPickConfig 序列化/反序列化测试"""

    def test_roundtrip_enabled(self):
        """完整配置的序列化往返"""
        config = BadPickConfig(
            enabled=True,
            bad_pick_model_version=5,
            classifier_features=["zscore_volatility_20", "rsi_14"],
            regime_bear_pct=-0.02,
            regime_vol_pct=0.025,
            regime_dd_pct=-0.05,
            regime_configs={
                "normal": RegimeBadPickConfig(threshold=0.5, penalty_lambda=0.1),
                "stressed": RegimeBadPickConfig(threshold=0.3, penalty_lambda=0.2),
            },
            calibration_samples=1000,
            calibration_bad_samples=300,
            calibration_bad_rate=0.3,
            calibration_auc=0.72,
            regime_sample_counts={"normal": 700, "stressed": 300},
            baseline_topk_median=0.001,
            selected_topk_median=0.002,
        )
        d = config.to_dict()
        assert d["version"] == 2
        assert d["enabled"] is True
        assert d["bad_pick_model_version"] == 5

        restored = BadPickConfig.from_dict(d)
        assert restored.enabled is True
        assert restored.bad_pick_model_version == 5
        assert restored.calibration_auc == 0.72
        assert restored.regime_configs["normal"].threshold == 0.5
        assert restored.regime_configs["stressed"].penalty_lambda == 0.2

    def test_roundtrip_disabled(self):
        """禁用配置的序列化往返"""
        config = BadPickConfig(enabled=False)
        d = config.to_dict()
        restored = BadPickConfig.from_dict(d)
        assert restored.enabled is False
        assert restored.bad_pick_model_version == 0

    def test_from_empty_dict(self):
        """空字典反序列化不应报错"""
        config = BadPickConfig.from_dict({})
        assert config.enabled is False


class TestDetectMarketRegime:
    """市场状态检测测试"""

    def test_normal_regime(self):
        """正常市场 → normal"""
        config = BadPickConfig(
            regime_bear_pct=-0.03,
            regime_vol_pct=0.03,
            regime_dd_pct=-0.08,
        )
        mkt = {
            "mkt_ret_avg_20": 0.001,  # 微正
            "mkt_vol_20": 0.015,  # 低波动
            "mkt_drawdown_20": -0.02,  # 轻微回撤
        }
        assert detect_market_regime(mkt, config) == "normal"

    def test_bear_regime(self):
        """趋势走弱 → stressed"""
        config = BadPickConfig(
            regime_bear_pct=-0.02,
            regime_vol_pct=0.03,
            regime_dd_pct=-0.08,
        )
        mkt = {
            "mkt_ret_avg_20": -0.03,  # 低于 bear 阈值
            "mkt_vol_20": 0.015,
            "mkt_drawdown_20": -0.02,
        }
        assert detect_market_regime(mkt, config) == "stressed"

    def test_high_vol_regime(self):
        """高波动 → stressed"""
        config = BadPickConfig(
            regime_bear_pct=-0.03,
            regime_vol_pct=0.02,
            regime_dd_pct=-0.08,
        )
        mkt = {
            "mkt_ret_avg_20": 0.001,
            "mkt_vol_20": 0.025,  # 高于 vol 阈值
            "mkt_drawdown_20": -0.02,
        }
        assert detect_market_regime(mkt, config) == "stressed"

    def test_drawdown_regime(self):
        """深度回撤 → stressed"""
        config = BadPickConfig(
            regime_bear_pct=-0.03,
            regime_vol_pct=0.03,
            regime_dd_pct=-0.05,
        )
        mkt = {
            "mkt_ret_avg_20": 0.001,
            "mkt_vol_20": 0.015,
            "mkt_drawdown_20": -0.10,  # 深于 dd 阈值
        }
        assert detect_market_regime(mkt, config) == "stressed"

    def test_missing_features(self):
        """缺失市场特征 → 不应报错，默认 normal"""
        config = BadPickConfig()
        mkt = {}
        assert detect_market_regime(mkt, config) == "normal"


class TestPrepareClassifierFeatures:
    """分类器特征准备测试"""

    def test_basic_extraction(self):
        """基本特征提取"""
        df = pd.DataFrame(
            {
                "zscore_volatility_20": [0.1, 0.2, 0.3],
                "rsi_14": [50, 60, 70],
                "other_col": [1, 2, 3],
            }
        )
        result = prepare_classifier_features(df, ["zscore_volatility_20", "rsi_14"])
        assert list(result.columns) == ["zscore_volatility_20", "rsi_14"]
        assert len(result) == 3

    def test_missing_columns(self):
        """缺失特征列 → 不应包含在结果中"""
        df = pd.DataFrame({"zscore_volatility_20": [0.1, 0.2]})
        result = prepare_classifier_features(
            df, ["zscore_volatility_20", "rsi_14", "kdj_j"]
        )
        assert "zscore_volatility_20" in result.columns
        assert "rsi_14" not in result.columns

    def test_nan_fill(self):
        """NaN 值应被填充为中位数"""
        df = pd.DataFrame({"zscore_volatility_20": [0.1, np.nan, 0.3]})
        result = prepare_classifier_features(df, ["zscore_volatility_20"])
        assert not result["zscore_volatility_20"].isna().any()
        assert result.loc[1, "zscore_volatility_20"] == pytest.approx(0.2)

    def test_all_nan_column(self):
        """全 NaN 列应填 0"""
        df = pd.DataFrame({"zscore_volatility_20": [np.nan, np.nan]})
        result = prepare_classifier_features(df, ["zscore_volatility_20"])
        assert (result["zscore_volatility_20"] == 0.0).all()

    def test_empty_df(self):
        """空 DataFrame → 返回空 DataFrame"""
        df = pd.DataFrame()
        result = prepare_classifier_features(df, ["zscore_volatility_20"])
        assert len(result) == 0


class TestApplyConditionalPenalty:
    """条件式惩罚应用测试"""

    def _make_features_df(self, n=10):
        """构造测试用特征 DataFrame"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "ts_code": [f"00000{i}.SZ" for i in range(n)],
                "ml_score": np.linspace(0.1, 1.0, n),
                "zscore_volatility_20": np.random.randn(n) * 0.5 + 0.5,
                "rsi_14": np.random.uniform(30, 80, n),
                "mkt_ret_avg_20": [0.001] * n,
                "mkt_vol_20": [0.015] * n,
                "mkt_drawdown_20": [-0.02] * n,
                "mkt_adv_dec_ratio": [1.2] * n,
                "mkt_turnover_std": [0.01] * n,
            }
        )
        return df

    def _make_mock_classifier(self, p_bad_values):
        """构造模拟分类器，返回指定的 P(bad_pick) 值"""

        class MockClassifier:
            def predict_proba(self, X):
                n = len(X)
                p = np.array(p_bad_values[:n])
                return np.column_stack([1 - p, p])

        return MockClassifier()

    def test_disabled_config(self):
        """禁用配置 → 不惩罚"""
        df = self._make_features_df()
        config = BadPickConfig(enabled=False)
        clf = self._make_mock_classifier([0.9] * 10)
        result, col = apply_conditional_penalty(df, config, clf)
        assert col == "ml_score"
        assert (result["final_score"] == result["ml_score"]).all()

    def test_threshold_gating_no_penalty(self):
        """P(bad) 低于阈值 → 不惩罚"""
        df = self._make_features_df()
        config = BadPickConfig(
            enabled=True,
            bad_pick_model_version=1,
            classifier_features=["zscore_volatility_20", "rsi_14"],
            regime_bear_pct=-0.03,
            regime_vol_pct=0.03,
            regime_dd_pct=-0.08,
            regime_configs={
                "normal": RegimeBadPickConfig(threshold=0.8, penalty_lambda=0.1),
            },
        )
        # 所有 P(bad) = 0.3，低于 threshold=0.8
        clf = self._make_mock_classifier([0.3] * 10)
        result, col = apply_conditional_penalty(df, config, clf)
        assert col == "final_score"
        # 不应被惩罚 → final_score == ml_score
        assert (result["final_score"] == result["ml_score"]).all()

    def test_threshold_gating_with_penalty(self):
        """P(bad) 高于阈值 → 部分被惩罚"""
        df = self._make_features_df(10)
        config = BadPickConfig(
            enabled=True,
            bad_pick_model_version=1,
            classifier_features=["zscore_volatility_20", "rsi_14"],
            regime_bear_pct=-0.03,
            regime_vol_pct=0.03,
            regime_dd_pct=-0.08,
            regime_configs={
                "normal": RegimeBadPickConfig(threshold=0.5, penalty_lambda=0.2),
            },
        )
        # p_bad = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.3, 0.2]
        p_bad_vals = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.3, 0.2]
        clf = self._make_mock_classifier(p_bad_vals)
        result, col = apply_conditional_penalty(df, config, clf)
        assert col == "final_score"

        # 前3个 p_bad <= 0.5 → 不惩罚
        assert result.loc[0, "final_score"] == pytest.approx(result.loc[0, "ml_score"])
        assert result.loc[1, "final_score"] == pytest.approx(result.loc[1, "ml_score"])
        assert result.loc[2, "final_score"] == pytest.approx(result.loc[2, "ml_score"])

        # p_bad=0.6 → penalty = 0.2 * (0.6 - 0.5) = 0.02
        expected = result.loc[3, "ml_score"] - 0.2 * (0.6 - 0.5)
        assert result.loc[3, "final_score"] == pytest.approx(expected)

        # p_bad=0.9 → penalty = 0.2 * (0.9 - 0.5) = 0.08
        expected = result.loc[6, "ml_score"] - 0.2 * (0.9 - 0.5)
        assert result.loc[6, "final_score"] == pytest.approx(expected)

    def test_no_classifier(self):
        """分类器为 None → 跳过惩罚"""
        df = self._make_features_df()
        config = BadPickConfig(enabled=True, bad_pick_model_version=1)
        result, col = apply_conditional_penalty(df, config, None)
        assert col == "ml_score"

    def test_empty_features(self):
        """空特征 DataFrame → 不报错"""
        df = pd.DataFrame({"ml_score": []})
        config = BadPickConfig(enabled=True, bad_pick_model_version=1)
        clf = self._make_mock_classifier([])
        result, col = apply_conditional_penalty(df, config, clf)
        assert col == "ml_score"

    def test_risk_score_column_added(self):
        """risk_score 列应反映 P(bad_pick)"""
        df = self._make_features_df(5)
        config = BadPickConfig(
            enabled=True,
            bad_pick_model_version=1,
            classifier_features=["zscore_volatility_20", "rsi_14"],
            regime_bear_pct=-0.03,
            regime_vol_pct=0.03,
            regime_dd_pct=-0.08,
            regime_configs={
                "normal": RegimeBadPickConfig(threshold=0.5, penalty_lambda=0.1),
            },
        )
        clf = self._make_mock_classifier([0.1, 0.4, 0.6, 0.8, 0.3])
        result, _ = apply_conditional_penalty(df, config, clf)
        assert "risk_score" in result.columns
        assert result.loc[0, "risk_score"] == pytest.approx(0.1)
        assert result.loc[3, "risk_score"] == pytest.approx(0.8)


class TestBadPickClassifierFeatures:
    """特征列表完整性测试"""

    def test_all_features_unique(self):
        """所有候选特征不应重复"""
        assert len(BAD_PICK_CLASSIFIER_FEATURES) == len(set(BAD_PICK_CLASSIFIER_FEATURES))

    def test_market_state_features_present(self):
        """MARKET_STATE_FEATURES 应包含核心市场特征"""
        required = ["mkt_ret_avg_20", "mkt_vol_20", "mkt_drawdown_20"]
        for feat in required:
            assert feat in MARKET_STATE_FEATURES, f"{feat} missing from MARKET_STATE_FEATURES"
