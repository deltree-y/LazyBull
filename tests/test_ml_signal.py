"""ML 信号测试"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.lazybull.ml import ModelRegistry
from src.lazybull.signals import MLSignal


class MockMLModel:
    """模拟 ML 模型（用于测试）"""
    
    def predict(self, X):
        """返回模拟预测值"""
        # 返回简单的预测值（基于第一个特征）
        if len(X.columns) > 0:
            return X.iloc[:, 0].values * 0.1
        return np.random.randn(len(X))


@pytest.fixture
def temp_models_dir():
    """创建临时模型目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def trained_model(temp_models_dir):
    """创建一个训练好的模型"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    model = MockMLModel()
    version = registry.register_model(
        model=model,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["f1", "f2", "f3"],
        label_column="y_ret_5",
        n_samples=1000,
        train_params={"n_estimators": 100}
    )
    
    return temp_models_dir, version


def test_ml_signal_init():
    """测试 ML 信号初始化"""
    signal = MLSignal(top_n=50, model_version=1)
    
    assert signal.top_n == 50
    assert signal.model_version == 1
    assert signal.weight_method == "equal"
    assert signal.model is None  # 延迟加载


def test_ml_signal_load_model(trained_model):
    """测试模型加载"""
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=30,
        model_version=version,
        models_dir=models_dir
    )
    
    # 模型应该是延迟加载的
    assert signal.model is None
    
    # 触发加载
    signal._load_model()
    
    assert signal.model is not None
    assert signal.metadata is not None
    assert signal.feature_columns == ["f1", "f2", "f3"]


def test_ml_signal_generate_equal_weight(trained_model):
    """测试生成等权信号"""
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=5,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal"
    )
    
    # 准备测试数据
    date = pd.Timestamp("2023-06-15")
    universe = [
        "000001.SZ", "000002.SZ", "000003.SZ",
        "600000.SH", "600001.SH", "600002.SH",
        "600003.SH", "600004.SH"
    ]
    
    # 创建特征数据（8只股票）
    features_df = pd.DataFrame({
        "ts_code": universe,
        "f1": [10, 8, 12, 6, 15, 5, 9, 7],  # 预测值会基于此列
        "f2": np.random.randn(8),
        "f3": np.random.randn(8)
    })
    
    data = {"features": features_df}
    
    # 生成信号
    signals = signal.generate(date, universe, data)
    
    # 验证结果
    assert len(signals) == 5  # Top 5
    assert abs(sum(signals.values()) - 1.0) < 1e-6  # 权重和为1
    
    # 等权，每只股票权重应该是 1/5 = 0.2
    for weight in signals.values():
        assert abs(weight - 0.2) < 1e-6


def test_ml_signal_generate_score_weight(trained_model):
    """测试生成按分数加权的信号"""
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="score"
    )
    
    # 准备测试数据
    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]
    
    features_df = pd.DataFrame({
        "ts_code": universe,
        "f1": [10, 5, 8, 3],  # 预测值基于此列：预测值为 [1.0, 0.5, 0.8, 0.3]
        "f2": [1, 2, 3, 4],
        "f3": [5, 6, 7, 8]
    })
    
    data = {"features": features_df}
    
    # 生成信号
    signals = signal.generate(date, universe, data)
    
    # 验证结果
    assert len(signals) == 3  # Top 3
    assert abs(sum(signals.values()) - 1.0) < 1e-6  # 权重和为1
    
    # 分数高的股票权重应该更大
    # f1=[10, 5, 8] -> 预测值=[1.0, 0.5, 0.8]
    # 应该选择 000001.SZ, 000003.SZ, 000002.SZ
    assert "000001.SZ" in signals
    assert "000003.SZ" in signals
    assert "000002.SZ" in signals
    
    # 验证权重不相等（非等权）
    weights = list(signals.values())
    assert len(set(weights)) > 1, "score权重应该产生不同的权重值，而不是等权"
    
    # 验证权重大小关系：000001.SZ > 000003.SZ > 000002.SZ
    assert signals["000001.SZ"] > signals["000003.SZ"], \
        "分数更高的股票权重应该更大"
    assert signals["000003.SZ"] > signals["000002.SZ"], \
        "分数更高的股票权重应该更大"


def test_ml_signal_generate_no_features(trained_model):
    """测试没有特征数据的情况"""
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=5,
        model_version=version,
        models_dir=models_dir
    )
    
    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ"]
    
    # 没有特征数据
    data = {}
    
    signals = signal.generate(date, universe, data)
    
    assert len(signals) == 0


def test_ml_signal_generate_empty_features(trained_model):
    """测试空特征数据的情况"""
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=5,
        model_version=version,
        models_dir=models_dir
    )
    
    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ"]
    
    # 空的特征数据
    features_df = pd.DataFrame()
    data = {"features": features_df}
    
    signals = signal.generate(date, universe, data)
    
    assert len(signals) == 0


def test_ml_signal_generate_with_universe_filter(trained_model):
    """测试股票池过滤"""
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir
    )
    
    date = pd.Timestamp("2023-06-15")
    
    # 股票池只包含3只股票
    universe = ["000001.SZ", "000002.SZ", "000003.SZ"]
    
    # 特征数据包含5只股票
    features_df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH", "600001.SH"],
        "f1": [10, 8, 12, 15, 20],
        "f2": [1, 2, 3, 4, 5],
        "f3": [5, 6, 7, 8, 9]
    })
    
    data = {"features": features_df}
    
    signals = signal.generate(date, universe, data)
    
    # 应该只选择股票池内的股票
    assert len(signals) == 3
    for stock in signals.keys():
        assert stock in universe


def test_ml_signal_generate_with_features_method(trained_model):
    """测试使用便捷方法生成信号"""
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir
    )
    
    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ"]
    
    features_df = pd.DataFrame({
        "ts_code": universe,
        "f1": [10, 8, 12],
        "f2": [1, 2, 3],
        "f3": [5, 6, 7]
    })
    
    # 使用便捷方法
    signals = signal.generate_with_features(date, universe, features_df)
    
    assert len(signals) == 3
    assert abs(sum(signals.values()) - 1.0) < 1e-6


def test_ml_signal_get_model_info(trained_model):
    """测试获取模型信息"""
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=30,
        model_version=version,
        models_dir=models_dir
    )
    
    info = signal.get_model_info()
    
    assert info["version"] == version
    assert info["model_type"] == "xgboost"
    assert info["feature_count"] == 3
    assert info["n_samples"] == 1000


def test_ml_signal_confidence_gate_scales_weights(trained_model):
    """测试信号置信度门控会缩放权重并保留现金。"""
    models_dir, version = trained_model

    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal",
        signal_confidence_gate_enabled=True,
        signal_confidence_gate_thresholds=[0.0],
        signal_confidence_gate_exposure_levels=[0.4],
    )

    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]
    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [12, 9, 4, 1],
            "f2": [1, 2, 3, 4],
            "f3": [5, 6, 7, 8],
        }
    )

    signals = signal.generate(date, universe, {"features": features_df})

    assert len(signals) == 3
    assert abs(sum(signals.values()) - 0.4) < 1e-6
    for weight in signals.values():
        assert abs(weight - (0.4 / 3)) < 1e-6

    gate_state = signal.get_last_confidence_gate_state()
    assert gate_state.enabled is True
    assert gate_state.exposure == 0.4


def test_ml_signal_confidence_gate_blocks_negative_regression_scores(trained_model):
    """测试回归分数整体不为正时，门控直接持币。"""
    models_dir, version = trained_model

    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal",
        signal_confidence_gate_enabled=True,
        signal_confidence_gate_thresholds=[0.8, 1.2],
        signal_confidence_gate_exposure_levels=[0.5, 1.0],
    )

    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]
    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [-1, -3, -5, -7],
            "f2": [1, 2, 3, 4],
            "f3": [5, 6, 7, 8],
        }
    )

    signals = signal.generate(date, universe, {"features": features_df})

    assert signals == {}
    gate_state = signal.get_last_confidence_gate_state()
    assert gate_state.enabled is True
    assert gate_state.exposure == 0.0
    assert "无正向alpha" in gate_state.reason


def test_ml_signal_top_n_larger_than_universe(trained_model):
    """测试 Top N 大于股票池大小的情况"""
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=10,  # 要求 Top 10
        model_version=version,
        models_dir=models_dir
    )
    
    date = pd.Timestamp("2023-06-15")
    
    # 但股票池只有5只股票
    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH", "600001.SH"]
    
    features_df = pd.DataFrame({
        "ts_code": universe,
        "f1": [10, 8, 12, 6, 15],
        "f2": [1, 2, 3, 4, 5],
        "f3": [5, 6, 7, 8, 9]
    })
    
    data = {"features": features_df}
    
    signals = signal.generate(date, universe, data)
    
    # 应该返回所有5只股票
    assert len(signals) == 5
    assert abs(sum(signals.values()) - 1.0) < 1e-6
