"""纸面交易展示模块测试（format_model_info 模型信息回退与限幅路径）。"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.lazybull.ml import ModelRegistry
from src.lazybull.paper import format_model_info
from src.lazybull.paper.reporting import MODEL_INFO_MAX_CHARS


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


def _write_raw_metadata(models_dir: Path, version: int, extra: dict) -> None:
    """直接写入旁路元数据文件（format_model_info 只读元数据，无需模型文件）。"""
    metadata = {
        "version": version,
        "version_str": f"v{version}",
        "model_type": "xgboost",
        "model_file": f"v{version}_model.joblib",
        "features_file": f"v{version}_features.json",
        "train_start_date": "20240101",
        "train_end_date": "20241231",
        "feature_count": 3,
        "label_column": "y_ret_5",
        "n_samples": 1000,
        "train_params": {},
        "performance_metrics": {},
        "created_at": "2026-01-01 00:00:00",
    }
    metadata.update(extra)
    with open(models_dir / f"v{version}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


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


def test_format_model_info_truncates_oversized_output(tmp_path, paper_config):
    """超长 metadata（如内嵌大清单的 train_params）输出不得超出钉钉消息上限。"""
    oversized_params = {f"param_{i}": "x" * 200 for i in range(60)}
    _write_raw_metadata(tmp_path, 1, {"train_params": oversized_params})

    text = format_model_info(models_dir=str(tmp_path))

    assert len(text) <= MODEL_INFO_MAX_CHARS
    assert "当前模型: v1" in text
    # 非白名单参数不逐项展示，仅给出省略计数
    assert "param_0" not in text
    assert "(其余 60 个参数省略)" in text


def test_format_model_info_shows_only_key_params(tmp_path, paper_config):
    """只展示白名单关键超参，其余参数省略计数。"""
    params = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 3000,
        "internal_diagnostic_list": "600000.SH," * 100,
        "private_flag": True,
    }
    _write_raw_metadata(tmp_path, 1, {"train_params": params})

    text = format_model_info(models_dir=str(tmp_path))

    assert "max_depth: 5" in text
    assert "learning_rate: 0.05" in text
    assert "n_estimators: 3000" in text
    assert "internal_diagnostic_list" not in text
    assert "private_flag" not in text
    assert "(其余 2 个参数省略)" in text


def test_format_model_info_truncates_single_huge_param_value(tmp_path, paper_config):
    """白名单内单个超大参数值应被截断展示，不影响其余参数。"""
    params = {
        "label_transform": "x" * 300,
        "max_depth": 5,
    }
    _write_raw_metadata(tmp_path, 1, {"train_params": params})

    text = format_model_info(models_dir=str(tmp_path))

    assert "max_depth: 5" in text
    huge_line = next(line for line in text.splitlines() if "label_transform" in line)
    assert len(huge_line) < 120
    assert huge_line.endswith("...")
    assert len(text) <= MODEL_INFO_MAX_CHARS


def test_format_model_info_falls_back_to_sidecar_when_registry_lacks_version(
    tmp_path, paper_config
):
    """注册表存在但不含指定版本时，应回退旁路元数据而非报未找到。"""
    _register_model(tmp_path, 1)
    # 模拟陈旧整包注册表：注册表条目与旁路文件不同步（删除旁路后重写注册表结构）
    registry_file = tmp_path / "model_registry.json"
    stale_registry = {
        "models": [
            {
                "version": 100,
                "version_str": "v100",
                "model_type": "xgboost",
                "train_start_date": "20200101",
                "train_end_date": "20201231",
            },
            {
                "version": 101,
                "version_str": "v101",
                "model_type": "xgboost",
                "train_start_date": "20210101",
                "train_end_date": "20211231",
            },
        ],
        "next_version": 102,
    }
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(stale_registry, f, ensure_ascii=False, indent=2)
    paper_config["model_version"] = 1

    text = format_model_info(models_dir=str(tmp_path))

    assert "当前模型: v1" in text
    assert "未找到" not in text


def test_format_model_info_truncates_huge_available_version_list(tmp_path, paper_config):
    """指定版本缺失且可用版本数百个时，提示只展示最近 10 个。"""
    stale_registry = {
        "models": [
            {"version": v, "version_str": f"v{v}", "model_type": "xgboost"}
            for v in range(200, 500)
        ],
        "next_version": 500,
    }
    with open(tmp_path / "model_registry.json", "w", encoding="utf-8") as f:
        json.dump(stale_registry, f, ensure_ascii=False, indent=2)
    paper_config["model_version"] = 1

    text = format_model_info(models_dir=str(tmp_path))

    assert "未找到版本 1 的模型" in text
    assert "可用版本共 300 个" in text
    assert "最近 10 个: [490," in text
    # 数百个版本号不得撑爆单条消息
    assert len(text) <= MODEL_INFO_MAX_CHARS
