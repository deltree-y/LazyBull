"""测试配置管理模块"""

import pytest

from src.lazybull.common.config import Config


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
