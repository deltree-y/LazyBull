"""测试交易成本模型"""

import pytest

import src.lazybull.common.config as config_module

from src.lazybull.common.config import Config
from src.lazybull.common.cost import CostModel, get_default_cost_model


def test_cost_model_init():
    """测试成本模型初始化"""
    model = CostModel(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax=0.001,
        slippage=0.001
    )
    
    assert model.commission_rate == 0.0003
    assert model.min_commission == 5.0
    assert model.stamp_tax == 0.001
    assert model.slippage == 0.001


def test_calculate_commission():
    """测试佣金计算"""
    model = CostModel(commission_rate=0.0003, min_commission=5.0)
    
    # 测试最小佣金
    assert model.calculate_commission(1000) == 5.0  # 1000 * 0.0003 = 0.3 < 5
    
    # 测试正常佣金 (使用近似比较)
    assert abs(model.calculate_commission(100000) - 30.0) < 0.01  # 100000 * 0.0003 ≈ 30


def test_calculate_stamp_tax():
    """测试印花税计算"""
    model = CostModel(stamp_tax=0.001)
    
    assert model.calculate_stamp_tax(100000) == 100.0  # 100000 * 0.001


def test_calculate_slippage():
    """测试滑点计算"""
    model = CostModel(slippage=0.001)
    
    assert model.calculate_slippage(100000) == 100.0  # 100000 * 0.001


def test_calculate_buy_cost():
    """测试买入成本计算"""
    model = CostModel(
        commission_rate=0.0003,
        min_commission=5.0,
        slippage=0.001
    )
    
    # 买入10万元
    buy_cost = model.calculate_buy_cost(100000)
    expected = 30.0 + 100.0  # 佣金30 + 滑点100
    assert buy_cost == expected


def test_calculate_sell_cost():
    """测试卖出成本计算"""
    model = CostModel(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax=0.001,
        slippage=0.001
    )
    
    # 卖出10万元
    sell_cost = model.calculate_sell_cost(100000)
    expected = 30.0 + 100.0 + 100.0  # 佣金30 + 印花税100 + 滑点100
    assert sell_cost == expected


def test_calculate_total_cost():
    """测试买卖双向总成本"""
    model = CostModel(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax=0.001,
        slippage=0.001
    )
    
    # 买入卖出各10万元
    total_cost = model.calculate_total_cost(100000, 100000)
    buy_cost = 30.0 + 100.0
    sell_cost = 30.0 + 100.0 + 100.0
    assert total_cost == buy_cost + sell_cost


def test_get_default_cost_model():
    """测试获取默认成本模型"""
    model = get_default_cost_model()

    assert isinstance(model, CostModel)
    assert model.commission_rate == 0.0001954
    assert model.min_commission == 5.0
    assert model.stamp_tax == 0.0005
    assert model.slippage == 0.0005


def test_cost_model_uses_project_config_defaults(monkeypatch):
    """测试 CostModel 默认值来自项目配置。"""
    config = Config()
    config.set("costs.commission_rate", 0.00042)
    config.set("costs.min_commission", 7.0)
    config.set("costs.stamp_tax", 0.00066)
    config.set("costs.slippage", 0.00088)
    monkeypatch.setattr(config_module, "_global_config", config)

    model = CostModel()

    assert model.commission_rate == 0.00042
    assert model.min_commission == 7.0
    assert model.stamp_tax == 0.00066
    assert model.slippage == 0.00088


def test_get_default_cost_model_uses_project_config(monkeypatch):
    """测试 get_default_cost_model 与项目配置保持一致。"""
    config = Config()
    config.set("costs.commission_rate", 0.00031)
    config.set("costs.min_commission", 8.0)
    config.set("costs.stamp_tax", 0.00051)
    config.set("costs.slippage", 0.00091)
    monkeypatch.setattr(config_module, "_global_config", config)

    model = get_default_cost_model()

    assert model.commission_rate == 0.00031
    assert model.min_commission == 8.0
    assert model.stamp_tax == 0.00051
    assert model.slippage == 0.00091
