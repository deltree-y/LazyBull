# -*- coding: utf-8 -*-
"""P1优化测试: 因子增强(2.2)、多特征子集集成(2.1)、模型质量监控(3.3)"""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.lazybull.ml.ensemble import EnsembleModel, SubsetEnsembleModel
from src.lazybull.ml.train_core import (
    ENHANCED_FEATURE_COLUMNS,
    SUBSET_CAPITAL_FLOW_FEATURES,
    SUBSET_FUNDAMENTAL_FEATURES,
    SUBSET_MOMENTUM_FEATURES,
    get_subset_ensemble_configs,
)


# ── 2.2 因子增强 ──────────────────────────────────────────────

class TestEnhancedFeatures:
    """测试增强因子常量定义"""

    def test_enhanced_feature_columns_count(self):
        """增强因子应包含5个列"""
        assert len(ENHANCED_FEATURE_COLUMNS) == 5

    def test_enhanced_feature_columns_content(self):
        """增强因子应包含开盘强度、日内波动结构和委托不平衡"""
        assert "zscore_opening_strength" in ENHANCED_FEATURE_COLUMNS
        assert "zscore_intraday_vol_structure" in ENHANCED_FEATURE_COLUMNS
        assert "zscore_order_imbalance" in ENHANCED_FEATURE_COLUMNS
        assert "order_imbalance_mean_5" in ENHANCED_FEATURE_COLUMNS
        assert "order_imbalance_mean_20" in ENHANCED_FEATURE_COLUMNS


# ── 2.1 多特征子集集成 ────────────────────────────────────────

class TestSubsetEnsembleModel:
    """测试 SubsetEnsembleModel"""

    @pytest.fixture
    def dummy_models_and_features(self):
        """创建3个简单的XGBoost模型，各自使用不同特征子集"""
        np.random.seed(42)
        n = 200

        # 所有可用特征
        all_cols = ["f1", "f2", "f3", "f4", "f5", "f6"]
        X_full = pd.DataFrame(
            np.random.randn(n, len(all_cols)), columns=all_cols
        )
        y = np.random.randn(n)

        # 子集定义
        subsets = [["f1", "f2"], ["f3", "f4"], ["f5", "f6"]]
        names = ["momentum", "fundamental", "capital"]

        models = []
        for cols in subsets:
            model = xgb.XGBRegressor(
                max_depth=2, n_estimators=5, verbosity=0
            )
            model.fit(X_full[cols], y)
            models.append(model)

        return models, subsets, names, X_full

    def test_predict_shape(self, dummy_models_and_features):
        """预测结果长度应与输入一致"""
        models, subsets, names, X_full = dummy_models_and_features
        ensemble = SubsetEnsembleModel(
            sub_models=models,
            sub_feature_columns=subsets,
            sub_names=names,
        )
        pred = ensemble.predict(X_full)
        assert len(pred) == len(X_full)

    def test_predict_weighted(self, dummy_models_and_features):
        """加权预测应与手动加权一致"""
        models, subsets, names, X_full = dummy_models_and_features
        weights = [0.5, 0.3, 0.2]
        ensemble = SubsetEnsembleModel(
            sub_models=models,
            sub_feature_columns=subsets,
            sub_names=names,
            weights=weights,
        )
        pred = ensemble.predict(X_full)

        # 手动计算加权平均
        manual = np.zeros(len(X_full))
        for m, cols, w in zip(models, subsets, weights):
            manual += m.predict(X_full[cols]) * w
        manual /= sum(weights)

        np.testing.assert_allclose(pred, manual, atol=1e-6)

    def test_all_feature_columns(self, dummy_models_and_features):
        """all_feature_columns 应为所有子模型特征的并集"""
        models, subsets, names, X_full = dummy_models_and_features
        ensemble = SubsetEnsembleModel(
            sub_models=models,
            sub_feature_columns=subsets,
            sub_names=names,
        )
        expected = sorted(set(sum(subsets, [])))
        assert sorted(ensemble.all_feature_columns) == expected

    def test_n_models(self, dummy_models_and_features):
        """n_models 应返回子模型数量"""
        models, subsets, names, X_full = dummy_models_and_features
        ensemble = SubsetEnsembleModel(
            sub_models=models,
            sub_feature_columns=subsets,
            sub_names=names,
        )
        assert ensemble.n_models == 3

    def test_equal_weights_default(self, dummy_models_and_features):
        """默认权重应为等权"""
        models, subsets, names, X_full = dummy_models_and_features
        ensemble = SubsetEnsembleModel(
            sub_models=models,
            sub_feature_columns=subsets,
            sub_names=names,
        )
        # 等权预测
        pred_eq = ensemble.predict(X_full)

        # 显式等权
        ensemble_explicit = SubsetEnsembleModel(
            sub_models=models,
            sub_feature_columns=subsets,
            sub_names=names,
            weights=[1.0, 1.0, 1.0],
        )
        pred_explicit = ensemble_explicit.predict(X_full)
        np.testing.assert_allclose(pred_eq, pred_explicit, atol=1e-6)


class TestSubsetConfigs:
    """测试 get_subset_ensemble_configs 函数"""

    @pytest.fixture
    def sample_df(self):
        """创建包含各类特征列的DataFrame"""
        n = 100
        cols = {}
        # 动量特征
        for c in SUBSET_MOMENTUM_FEATURES:
            cols[c] = np.random.randn(n)
        # 基本面特征
        for c in SUBSET_FUNDAMENTAL_FEATURES:
            cols[c] = np.random.randn(n)
        # 资金流特征
        for c in SUBSET_CAPITAL_FLOW_FEATURES:
            cols[c] = np.random.randn(n)
        # 标签
        cols["trade_date"] = pd.date_range("2023-01-01", periods=n)
        cols["ts_code"] = "000001.SZ"
        return pd.DataFrame(cols)

    def test_returns_three_configs(self, sample_df):
        """应返回3个子集配置"""
        configs = get_subset_ensemble_configs(sample_df)
        assert len(configs) == 3

    def test_config_structure(self, sample_df):
        """每个配置应包含 name 和 features"""
        configs = get_subset_ensemble_configs(sample_df)
        for cfg in configs:
            assert "name" in cfg
            assert "features" in cfg
            assert len(cfg["features"]) > 0

    def test_config_names(self, sample_df):
        """配置名应为 momentum, fundamental, capital_flow"""
        configs = get_subset_ensemble_configs(sample_df)
        names = [c["name"] for c in configs]
        assert "momentum" in names
        assert "fundamental" in names
        assert "capital_flow" in names

    def test_features_exist_in_df(self, sample_df):
        """所有返回的特征应存在于输入DataFrame中"""
        configs = get_subset_ensemble_configs(sample_df)
        for cfg in configs:
            for f in cfg["features"]:
                assert f in sample_df.columns, f"特征 {f} 不在DataFrame中"


# ── 3.3 模型质量监控逻辑 ──────────────────────────────────────

class TestModelQualityDegradation:
    """测试模型降级逻辑（纯逻辑验证，不依赖walk_forward.py实际执行）"""

    def _simulate_quality_check(
        self, results_ir_list, threshold=0.03
    ):
        """模拟walk_forward主循环中的质量监控逻辑

        Args:
            results_ir_list: 每个split的val_rankic_ir值列表
            threshold: 降级阈值

        Returns:
            最终使用的model_version列表, 降级次数
        """
        prev_good_version = None
        degradation_count = 0
        used_versions = []

        for i, ir in enumerate(results_ir_list):
            current_version = i + 1  # 假设版本号从1开始
            result_version = current_version

            if ir is not None and ir < threshold:
                if prev_good_version is not None:
                    degradation_count += 1
                    result_version = prev_good_version
                else:
                    prev_good_version = current_version
            else:
                prev_good_version = current_version

            used_versions.append(result_version)

        return used_versions, degradation_count

    def test_no_degradation_when_all_good(self):
        """所有split质量合格时不应触发降级"""
        irs = [0.05, 0.06, 0.04, 0.07]
        versions, count = self._simulate_quality_check(irs)
        assert versions == [1, 2, 3, 4]
        assert count == 0

    def test_degradation_triggered(self):
        """低质量split应回退到上一合格版本"""
        irs = [0.05, 0.01, 0.06, 0.02]
        versions, count = self._simulate_quality_check(irs)
        # split 0: 合格 → v1, split 1: 低 → 回退v1
        # split 2: 合格 → v3, split 3: 低 → 回退v3
        assert versions == [1, 1, 3, 3]
        assert count == 2

    def test_first_split_low_quality_no_fallback(self):
        """第一个split低质量时无可用历史模型，继续使用当前"""
        irs = [0.01, 0.05, 0.06]
        versions, count = self._simulate_quality_check(irs)
        # split 0: 低但无历史 → v1 (作为基准)
        # split 1: 合格 → v2, split 2: 合格 → v3
        assert versions == [1, 2, 3]
        assert count == 0

    def test_consecutive_low_quality(self):
        """连续低质量split应持续回退到最后一个合格版本"""
        irs = [0.05, 0.01, 0.005, 0.02, 0.07]
        versions, count = self._simulate_quality_check(irs)
        # split 0: 合格v1, split 1/2/3: 低→回退v1, split 4: 合格v5
        assert versions == [1, 1, 1, 1, 5]
        assert count == 3

    def test_none_ir_treated_as_good(self):
        """val_rankic_ir为None时不触发降级（数据不足等情况）"""
        irs = [0.05, None, 0.01]
        versions, count = self._simulate_quality_check(irs)
        # split 0: 合格v1, split 1: None→合格v2, split 2: 低→回退v2
        assert versions == [1, 2, 2]
        assert count == 1
