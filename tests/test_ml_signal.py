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
        train_params={"n_estimators": 100},
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

    signal = MLSignal(top_n=30, model_version=version, models_dir=models_dir)

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

    signal = MLSignal(top_n=5, model_version=version, models_dir=models_dir, weight_method="equal")

    # 准备测试数据
    date = pd.Timestamp("2023-06-15")
    universe = [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "600000.SH",
        "600001.SH",
        "600002.SH",
        "600003.SH",
        "600004.SH",
    ]

    # 创建特征数据（8只股票）
    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [10, 8, 12, 6, 15, 5, 9, 7],  # 预测值会基于此列
            "f2": np.random.randn(8),
            "f3": np.random.randn(8),
        }
    )

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

    signal = MLSignal(top_n=3, model_version=version, models_dir=models_dir, weight_method="score")

    # 准备测试数据
    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]

    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [10, 5, 8, 3],  # 预测值基于此列：预测值为 [1.0, 0.5, 0.8, 0.3]
            "f2": [1, 2, 3, 4],
            "f3": [5, 6, 7, 8],
        }
    )

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
    assert signals["000001.SZ"] > signals["000003.SZ"], "分数更高的股票权重应该更大"
    assert signals["000003.SZ"] > signals["000002.SZ"], "分数更高的股票权重应该更大"


def test_ml_signal_generate_no_features(trained_model):
    """测试没有特征数据的情况"""
    models_dir, version = trained_model

    signal = MLSignal(top_n=5, model_version=version, models_dir=models_dir)

    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ"]

    # 没有特征数据
    data = {}

    signals = signal.generate(date, universe, data)

    assert len(signals) == 0


def test_ml_signal_generate_empty_features(trained_model):
    """测试空特征数据的情况"""
    models_dir, version = trained_model

    signal = MLSignal(top_n=5, model_version=version, models_dir=models_dir)

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

    signal = MLSignal(top_n=3, model_version=version, models_dir=models_dir)

    date = pd.Timestamp("2023-06-15")

    # 股票池只包含3只股票
    universe = ["000001.SZ", "000002.SZ", "000003.SZ"]

    # 特征数据包含5只股票
    features_df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH", "600001.SH"],
            "f1": [10, 8, 12, 15, 20],
            "f2": [1, 2, 3, 4, 5],
            "f3": [5, 6, 7, 8, 9],
        }
    )

    data = {"features": features_df}

    signals = signal.generate(date, universe, data)

    # 应该只选择股票池内的股票
    assert len(signals) == 3
    for stock in signals.keys():
        assert stock in universe


def test_ml_signal_generate_with_features_method(trained_model):
    """测试使用便捷方法生成信号"""
    models_dir, version = trained_model

    signal = MLSignal(top_n=3, model_version=version, models_dir=models_dir)

    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ"]

    features_df = pd.DataFrame(
        {"ts_code": universe, "f1": [10, 8, 12], "f2": [1, 2, 3], "f3": [5, 6, 7]}
    )

    # 使用便捷方法
    signals = signal.generate_with_features(date, universe, features_df)

    assert len(signals) == 3
    assert abs(sum(signals.values()) - 1.0) < 1e-6


def test_ml_signal_get_model_info(trained_model):
    """测试获取模型信息"""
    models_dir, version = trained_model

    signal = MLSignal(top_n=30, model_version=version, models_dir=models_dir)

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


def test_ml_signal_composite_gate_cost_blocks(trained_model):
    """测试 composite 门控：预测收益低于成本阈值时持币。"""
    models_dir, version = trained_model

    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal",
        signal_gate_mode="composite",
        signal_gate_cost_multiplier=2.0,
        signal_gate_round_trip_cost=0.003,  # 成本0.3%，阈值0.6%
        signal_confidence_gate_top_k=2,
    )

    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]
    # f1 值很小，乘以0.1后 top_mean 远低于 0.006
    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [0.03, 0.02, 0.01, 0.005],
            "f2": [1, 2, 3, 4],
            "f3": [5, 6, 7, 8],
        }
    )

    signals = signal.generate(date, universe, {"features": features_df})

    assert signals == {}
    gate_state = signal.get_last_confidence_gate_state()
    assert gate_state.enabled is True
    assert gate_state.cost_gate_passed is False
    assert "成本门控" in gate_state.reason


def test_ml_signal_composite_gate_passes_with_high_scores(trained_model):
    """测试 composite 门控：高预测分数时满仓通过（预热期内默认放行）。"""
    models_dir, version = trained_model

    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal",
        signal_gate_mode="composite",
        signal_gate_cost_multiplier=2.0,
        signal_gate_round_trip_cost=0.003,
        signal_gate_percentile_warmup=20,
        signal_confidence_gate_top_k=2,
    )

    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]
    # f1 值较大，乘以0.1后 top_mean > 0.006
    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [10, 8, 5, 2],
            "f2": [1, 2, 3, 4],
            "f3": [5, 6, 7, 8],
        }
    )

    signals = signal.generate(date, universe, {"features": features_df})

    # 预热期内成本门控通过后，exposure=1.0（默认放行）
    assert len(signals) == 3
    assert abs(sum(signals.values()) - 1.0) < 1e-6
    gate_state = signal.get_last_confidence_gate_state()
    assert gate_state.enabled is True
    assert gate_state.cost_gate_passed is True
    assert gate_state.exposure == 1.0


def test_ml_signal_composite_gate_accumulates_history(trained_model):
    """测试 composite 门控：历史缓冲区正确累积。"""
    models_dir, version = trained_model

    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal",
        signal_gate_mode="composite",
        signal_gate_cost_multiplier=2.0,
        signal_gate_round_trip_cost=0.003,
        signal_gate_percentile_warmup=5,  # 较低预热期便于测试
        signal_confidence_gate_top_k=2,
    )

    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]
    base_features = {
        "ts_code": universe,
        "f2": [1, 2, 3, 4],
        "f3": [5, 6, 7, 8],
    }

    # 模拟多次调仓，让历史累积
    for i in range(8):
        date = pd.Timestamp(f"2023-06-{15 + i}")
        f1_values = [10 + i, 8 + i, 5 + i, 2 + i]
        features_df = pd.DataFrame({**base_features, "f1": f1_values})
        # 直接调用门控评估
        signal._load_model()  # 确保模型已加载
        ranked = [(stock, score) for stock, score in zip(universe, [v * 0.1 for v in f1_values])]
        signal._calculate_confidence_gate_state(ranked, date=date)

    # 验证历史缓冲区已累积
    assert len(signal._separation_history) == 8
    assert len(signal._composite_score_history) == 8


def test_ml_signal_disabled_gate_mode(trained_model):
    """测试 disabled 模式：门控完全不介入。"""
    models_dir, version = trained_model

    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal",
        signal_gate_mode="disabled",
    )

    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]
    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [-1, -3, -5, -7],  # 全负分
            "f2": [1, 2, 3, 4],
            "f3": [5, 6, 7, 8],
        }
    )

    signals = signal.generate(date, universe, {"features": features_df})

    # disabled 模式下不门控，即使分数全负也应生成信号
    assert len(signals) == 3
    gate_state = signal.get_last_confidence_gate_state()
    assert gate_state.enabled is False


def test_ml_signal_update_model_version_preserves_history(trained_model):
    """测试 update_model_version：切换模型时保留门控历史缓冲区。"""
    models_dir, version = trained_model

    # 注册第二个版本的模型
    registry = ModelRegistry(models_dir=models_dir)
    new_version = registry.register_model(
        model=MockMLModel(),
        model_type="xgboost",
        train_start_date="20240101",
        train_end_date="20241231",
        feature_columns=["f1", "f2", "f3"],
        label_column="y_ret_5",
        n_samples=1000,
        train_params={"n_estimators": 100},
    )

    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal",
        signal_gate_mode="composite",
        signal_gate_cost_multiplier=2.0,
        signal_gate_round_trip_cost=0.003,
        signal_gate_percentile_warmup=20,
    )

    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]
    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [10.0, 8.0, 5.0, 2.0],
            "f2": [1, 2, 3, 4],
            "f3": [5, 6, 7, 8],
        }
    )

    # 积累一些历史
    signal._load_model()
    for i in range(5):
        ranked = [("000001.SZ", 1.0 + i * 0.1), ("000002.SZ", 0.8 + i * 0.1),
                  ("000003.SZ", 0.5), ("600000.SH", 0.2)]
        signal._calculate_confidence_gate_state(ranked)

    assert len(signal._separation_history) == 5
    assert len(signal._composite_score_history) == 5

    # 切换到新版本
    signal.update_model_version(new_version)

    # 历史缓冲区应保留
    assert len(signal._separation_history) == 5
    assert len(signal._composite_score_history) == 5
    # 模型缓存应被清空（等待延迟加载）
    assert signal.model is None
    assert signal.model_version == new_version

    # 切换后仍可正常生成信号
    signals = signal.generate(pd.Timestamp("2024-06-15"), universe, {"features": features_df})
    assert len(signals) > 0


def test_ml_signal_update_model_version_none_raises_on_generate(trained_model):
    """测试初始 model_version=None 在未调用 update_model_version 时 generate 会抛异常。"""
    models_dir, _ = trained_model

    signal = MLSignal(
        top_n=3,
        model_version=None,
        models_dir=models_dir,
        signal_gate_mode="composite",
    )

    universe = ["000001.SZ", "000002.SZ"]
    features_df = pd.DataFrame({"ts_code": universe, "f1": [1.0, 2.0], "f2": [1, 2], "f3": [1, 2]})

    with pytest.raises(RuntimeError, match="model_version 为 None"):
        signal.generate(pd.Timestamp("2024-01-01"), universe, {"features": features_df})


def test_ml_signal_composite_gate_zscore_different_multipliers(temp_models_dir):
    """测试 composite 门控：cs_zscore 标签时不同 cost_multiplier 产生不同的 abs_quality_score。"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    version = registry.register_model(
        model=MockMLModel(),
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["f1", "f2", "f3"],
        label_column="y_ret_5",
        n_samples=1000,
        train_params={"n_estimators": 100, "label_transform": "cs_zscore"},
    )

    ranked = [
        ("000001.SZ", 0.25),
        ("000002.SZ", 0.20),   # top_k=2: top_mean=0.225
        ("000003.SZ", 0.05),
        ("600000.SH", -0.10),  # score_std ≈ 0.137
    ]

    results = []
    for mult in [1.0, 1.5, 2.0]:
        signal = MLSignal(
            top_n=3,
            model_version=version,
            models_dir=temp_models_dir,
            signal_gate_mode="composite",
            signal_gate_cost_multiplier=mult,
            signal_confidence_gate_top_k=2,
        )
        signal._load_model()
        state = signal._calculate_confidence_gate_state(ranked)
        results.append(state.abs_quality_score)

    # 不同 multiplier 应产生不同（且递减）的 abs_quality_score
    assert results[0] > results[1] > results[2]
    # multiplier=2.0 时 midpoint ≈ 2×0.137=0.274，top_mean=0.225 < midpoint → quality < 1.0
    assert results[2] < 1.0


def test_ml_signal_composite_gate_zscore_cost_blocks_weak_signal(temp_models_dir):
    """测试 composite 门控：cs_zscore 标签时，cost_multiplier=1.0 可拦截弱于 1σ 的信号。"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    version = registry.register_model(
        model=MockMLModel(),
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["f1", "f2", "f3"],
        label_column="y_ret_5",
        n_samples=1000,
        train_params={"n_estimators": 100, "label_transform": "cs_zscore"},
    )

    # cost_multiplier=1.0: 要求 top_mean > 1 × score_std
    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=temp_models_dir,
        signal_gate_mode="composite",
        signal_gate_cost_multiplier=1.0,  # top_mean 必须 > score_std
        signal_gate_round_trip_cost=0.003,
        signal_confidence_gate_top_k=2,
    )

    signal._load_model()
    # 弱信号：top_mean=0.025，std≈0.1 → threshold = 1.0×0.1=0.10，0.025 < 0.10 → 拦截
    ranked = [
        ("000001.SZ", 0.05),
        ("000002.SZ", 0.00),   # top_mean=0.025
        ("000003.SZ", -0.05),
        ("600000.SH", -0.10),
    ]
    state = signal._calculate_confidence_gate_state(ranked)

    assert state.cost_gate_passed is False
    assert state.exposure == 0.0
    assert "z分数模式" in state.reason


def test_ml_signal_top_n_larger_than_universe(trained_model):
    """测试 Top N 大于股票池大小的情况"""
    models_dir, version = trained_model

    signal = MLSignal(top_n=10, model_version=version, models_dir=models_dir)  # 要求 Top 10

    date = pd.Timestamp("2023-06-15")

    # 但股票池只有5只股票
    universe = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH", "600001.SH"]

    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [10, 8, 12, 6, 15],
            "f2": [1, 2, 3, 4, 5],
            "f3": [5, 6, 7, 8, 9],
        }
    )

    data = {"features": features_df}

    signals = signal.generate(date, universe, data)

    # 应该返回所有5只股票
    assert len(signals) == 5
    assert abs(sum(signals.values()) - 1.0) < 1e-6
