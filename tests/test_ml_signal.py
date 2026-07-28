"""ML 信号测试"""

import io
import tempfile

import numpy as np
import pandas as pd
import pytest
from loguru import logger

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


def test_ml_signal_generate_scores(trained_model):
    """测试 generate() 返回原始 ml_score（非归一化），供引擎层统一做权重分配"""
    models_dir, version = trained_model

    signal = MLSignal(top_n=5, model_version=version, models_dir=models_dir)

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
            "f1": [10, 8, 12, 6, 15, 5, 9, 7],
            "f2": np.random.randn(8),
            "f3": np.random.randn(8),
        }
    )

    data = {"features": features_df}

    # 生成信号
    signals = signal.generate(date, universe, data)

    # 验证结果：返回 Top 5，分数为正数（非归一化权重）
    assert len(signals) == 5
    for score in signals.values():
        assert score > 0, "generate() 应返回正数原始分数"
    # 总和不等于 1（原始分数，非归一化）
    assert sum(signals.values()) > 1.0


def test_ml_signal_generate_score_weight(trained_model):
    """测试生成按分数加权的信号"""
    models_dir, version = trained_model

    signal = MLSignal(top_n=3, model_version=version, models_dir=models_dir)

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

    # 验证结果：返回 Top 3 原始分数
    assert len(signals) == 3

    # 分数高的股票值应该更大
    # f1=[10, 5, 8] -> 预测值=[1.0, 0.5, 0.8]
    # 应该选择 000001.SZ, 000003.SZ, 000002.SZ
    assert "000001.SZ" in signals
    assert "000003.SZ" in signals
    assert "000002.SZ" in signals

    # 验证分数不相等（原始 ml_score 因输入不同而各异）
    scores = list(signals.values())
    assert len(set(scores)) > 1, "原始分数应该各不相同"

    # 验证分数大小关系：000001.SZ > 000003.SZ > 000002.SZ
    assert signals["000001.SZ"] > signals["000003.SZ"], "分数更高的股票值应该更大"
    assert signals["000003.SZ"] > signals["000002.SZ"], "分数更高的股票值应该更大"


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


def test_generate_ranked_no_longer_logs_prediction_summary(trained_model):
    """generate_ranked 不再输出独立的过滤/预测入口日志。"""

    models_dir, version = trained_model
    signal = MLSignal(top_n=3, model_version=version, models_dir=models_dir, verbose=False)

    date = pd.Timestamp("2023-06-15")
    universe = ["000001.SZ", "000002.SZ", "000003.SZ"]
    features_df = pd.DataFrame(
        {
            "ts_code": universe,
            "f1": [10, 9, 8],
            "f2": [1, 2, 3],
            "f3": [4, 5, 6],
            "amount_ma20": [60000.0, 40000.0, 70000.0],
            "total_mv": [1000000.0, 1000000.0, 1000000.0],
            "sw_l1_code": ["801010", "801010", "801010"],
        }
    )

    stream = io.StringIO()
    sink_id = logger.add(stream, format="{message}")
    try:
        ranked = signal.generate_ranked(date, universe, {"features": features_df})
    finally:
        logger.remove(sink_id)

    output = stream.getvalue()
    assert len(ranked) == 2
    assert "选股/预测(ranked):" not in output
    assert "选股过滤合计" not in output
    assert "开始模型预测(ranked)" not in output


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
    assert all(v > 0 for v in signals.values()), "generate() 应返回正数原始分数"


def test_ml_signal_get_model_info(trained_model):
    """测试获取模型信息"""
    models_dir, version = trained_model

    signal = MLSignal(top_n=30, model_version=version, models_dir=models_dir)

    info = signal.get_model_info()

    assert info["version"] == version
    assert info["model_type"] == "xgboost"
    assert info["feature_count"] == 3
    assert info["n_samples"] == 1000


def test_ml_signal_update_model_version_preserves_lazy_load(trained_model):
    """测试 update_model_version：切换模型后清空缓存并可正常生成信号。"""
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

    signal._load_model()
    assert signal.model is not None

    # 切换到新版本
    signal.update_model_version(new_version)

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
    )

    universe = ["000001.SZ", "000002.SZ"]
    features_df = pd.DataFrame({"ts_code": universe, "f1": [1.0, 2.0], "f2": [1, 2], "f3": [1, 2]})

    with pytest.raises(RuntimeError, match="model_version 为 None"):
        signal.generate(pd.Timestamp("2024-01-01"), universe, {"features": features_df})


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
    assert all(v > 0 for v in signals.values()), "generate() 应返回正数原始分数"
