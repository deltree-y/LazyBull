"""纸面交易展示模块测试（format_model_info 模型信息回退路径）。"""

from pathlib import Path

import numpy as np
import pytest

from src.lazybull.ml import ModelRegistry
from src.lazybull.paper import format_model_info


class _MockModel:
    """最小可序列化的模拟模型（仅用于注册占位）。"""

    def predict(self, X):
        return np.zeros(len(X))


class _FakePaperStorage:
    """替换 PaperStorage，避免测试依赖真实纸面交易配置。"""

    def __init__(self, config):
        self._config = config

    def load_config(self):
        return self._config


@pytest.fixture
def paper_config(monkeypatch):
    """注入最小纸面交易配置（未指定 model_version，取最新模型）。"""
    config = {"model_version": None, "top_n": 20, "rebalance_freq": 5}
    monkeypatch.setattr(
        "src.lazybull.paper.reporting.PaperStorage",
        lambda: _FakePaperStorage(config),
    )
    return config


def _register_model(models_dir: Path, version_tag: int) -> None:
    """向临时目录注册一个模拟模型。"""
    registry = ModelRegistry(models_dir=str(models_dir))
    registry.register_model(
        model=_MockModel(),
        model_type="xgboost",
        train_start_date="20240101",
        train_end_date="20241231",
        feature_columns=[f"feature_{i}" for i in range(3)],
        label_column="y_ret_5",
        n_samples=1000 + version_tag,
        train_params={"n_estimators": 100},
        performance_metrics={"validation": {"rank_ic": 0.05}},
    )


def test_format_model_info_falls_back_to_metadata_sidecar(tmp_path, paper_config):
    """model_registry.json 缺失时应回退读取 v{N}_metadata.json 旁路文件。"""
    _register_model(tmp_path, 1)
    (tmp_path / "model_registry.json").unlink()

    text = format_model_info(models_dir=str(tmp_path))

    assert "当前模型: v1 (最新)" in text
    assert "训练区间: 20240101 ~ 20241231" in text
    assert "训练样本: 1001" in text


def test_format_model_info_prefers_registry_when_present(tmp_path, paper_config):
    """注册表存在时优先走原路径，行为不因回退逻辑改变。"""
    _register_model(tmp_path, 1)

    text = format_model_info(models_dir=str(tmp_path))

    assert "当前模型: v1 (最新)" in text


def test_format_model_info_no_models_at_all(tmp_path, paper_config):
    """既无注册表也无旁路文件时提示训练模型。"""
    text = format_model_info(models_dir=str(tmp_path))

    assert "没有已注册的模型" in text


def test_format_model_info_specified_version_from_sidecar(tmp_path, paper_config):
    """指定版本时也应能从旁路文件命中对应模型。"""
    _register_model(tmp_path, 1)
    _register_model(tmp_path, 2)
    (tmp_path / "model_registry.json").unlink()
    paper_config["model_version"] = 1

    text = format_model_info(models_dir=str(tmp_path))

    assert "当前模型: v1" in text
    assert "(最新)" not in text
