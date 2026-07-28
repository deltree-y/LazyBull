"""测试配置管理模块"""

import pytest

import src.lazybull.common.config as config_module

from src.lazybull.common.config import (
    Config,
    get_cost_settings,
    get_data_path,
    get_data_root,
    get_models_root,
    get_paper_root,
    get_reports_root,
    get_tushare_settings,
    normalize_shenwan_level,
)


def test_config_init():
    """测试配置初始化"""
    config = Config()
    assert config is not None


def test_config_set_get():
    """测试配置设置和获取"""
    config = Config()
    
    # 设置简单值
    config.set("test.key", "value")
    assert config.get("test.key") == "value"
    
    # 设置嵌套值
    config.set("test.nested.key", 123)
    assert config.get("test.nested.key") == 123
    
    # 获取不存在的键
    assert config.get("not.exist", "default") == "default"


def test_config_nested_keys():
    """测试嵌套键访问"""
    config = Config()
    
    config.set("level1.level2.level3", "deep")
    assert config.get("level1.level2.level3") == "deep"
    
    # 获取中间层级
    level2 = config.get("level1.level2")
    assert isinstance(level2, dict)
    assert level2["level3"] == "deep"


def test_config_get_env(monkeypatch):
    """测试环境变量获取"""
    config = Config()
    
    # 设置环境变量
    monkeypatch.setenv("TEST_VAR", "test_value")
    
    assert config.get_env("TEST_VAR") == "test_value"
    assert config.get_env("NOT_EXIST", "default") == "default"


def test_normalize_shenwan_level():
    """测试申万行业层级标准化"""
    assert normalize_shenwan_level(None) == "l2"
    assert normalize_shenwan_level("L1") == "l1"
    assert normalize_shenwan_level(" l3 ") == "l3"

    with pytest.raises(ValueError, match="仅支持"):
        normalize_shenwan_level("foo")


def test_project_default_helpers(monkeypatch, tmp_path):
    """测试项目级默认配置 helper。"""
    config = Config()
    data_root = tmp_path / "custom_data"
    config.set("data.root", str(data_root))
    config.set("data.raw", "./data/raw")
    config.set("data.clean", str(tmp_path / "clean_area"))
    config.set("tushare.max_retries", 7)
    config.set("tushare.retry_delay", 2.5)
    config.set("tushare.rate_limit", 123)
    config.set("costs.commission_rate", 0.0004)
    config.set("costs.min_commission", 6.0)
    config.set("costs.stamp_tax", 0.0006)
    config.set("costs.slippage", 0.0007)
    monkeypatch.setattr(config_module, "_global_config", config)

    assert get_data_root() == str(data_root)
    assert get_data_path("raw") == str(data_root / "raw")
    assert get_data_path("clean") == str(tmp_path / "clean_area")
    assert get_models_root() == str(data_root / "models")
    assert get_reports_root() == str(data_root / "reports")
    assert get_paper_root() == str(data_root / "paper")
    tushare_settings = get_tushare_settings()
    assert tushare_settings["max_retries"] == 7
    assert tushare_settings["retry_delay"] == 2.5
    assert tushare_settings["rate_limit"] == 123
    cost_settings = get_cost_settings()
    assert cost_settings["commission_rate"] == 0.0004
    assert cost_settings["min_commission"] == 6.0
    assert cost_settings["stamp_tax"] == 0.0006
    assert cost_settings["slippage"] == 0.0007
