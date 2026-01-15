"""测试交易成本模型"""

import pytest

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
    
    # 测试正常佣金
    assert model.calculate_commission(100000) == 30.0  # 100000 * 0.0003 = 30


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
    assert model.commission_rate == 0.0003
    assert model.min_commission == 5.0
    assert model.stamp_tax == 0.001
    assert model.slippage == 0.001
