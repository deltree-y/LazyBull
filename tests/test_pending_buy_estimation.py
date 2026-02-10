"""测试补位买入股数估算方法"""

import pytest
from unittest.mock import Mock, patch

from src.lazybull.paper.runner import PaperTradingRunner
from src.lazybull.paper.account import PaperAccount
from src.lazybull.paper.broker import PaperBroker
from src.lazybull.common.cost import CostModel


@pytest.fixture
def runner():
    """创建测试用的 PaperTradingRunner 实例"""
    # 创建基础依赖
    account = PaperAccount(initial_capital=1000000.0)
    cost_model = CostModel(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax=0.001,
        slippage=0.001
    )
    broker = PaperBroker(account=account, cost_model=cost_model)
    
    # 创建 runner（最小化依赖）
    runner = PaperTradingRunner.__new__(PaperTradingRunner)
    runner.account = account
    runner.broker = broker
    
    return runner


def test_estimate_pending_buy_shares_normal(runner):
    """测试正常情况下的股数估算"""
    # 设置账户现金
    runner.account.state.cash = 500000.0
    
    # 参数
    ts_code = "000001.SZ"
    price = 10.0
    target_weight = 0.1  # 10%
    total_pending_count = 5
    retention_ratio = 0.3  # 保留30%
    
    # 估算股数
    shares = runner._estimate_pending_buy_shares(
        ts_code=ts_code,
        price=price,
        target_weight=target_weight,
        total_pending_count=total_pending_count,
        pendding_capital_retention_ratio=retention_ratio
    )
    
    # 验证计算逻辑：
    # total_cash = 500000 * (1 - 0.3) = 350000
    # available_cash = 350000 / 5 = 70000
    # target_value = 350000 * 0.1 = 35000
    # estimated_cost ≈ 35000 * (0.0003 + 0.001) = 45.5 (但min是5，所以实际是45.5)
    # target_value + cost = 35000 + 45.5 = 35045.5 < 70000，所以不调整
    # buy_shares = floor(35000 / 10 / 100) * 100 = 3500
    
    assert shares == 3500


def test_estimate_pending_buy_shares_cash_limit(runner):
    """测试现金受限情况"""
    # 设置账户现金
    runner.account.state.cash = 100000.0
    
    # 参数（权重较高，会超出单个补位的可用现金）
    ts_code = "000001.SZ"
    price = 10.0
    target_weight = 0.5  # 50%
    total_pending_count = 2
    retention_ratio = 0.3
    
    # 估算股数
    shares = runner._estimate_pending_buy_shares(
        ts_code=ts_code,
        price=price,
        target_weight=target_weight,
        total_pending_count=total_pending_count,
        pendding_capital_retention_ratio=retention_ratio
    )
    
    # 验证计算逻辑：
    # total_cash = 100000 * (1 - 0.3) = 70000
    # available_cash = 70000 / 2 = 35000
    # target_value = 70000 * 0.5 = 35000
    # estimated_cost ≈ 35000 * 0.0013 = 45.5
    # target_value + cost = 35045.5 > 35000，需要调整
    # target_value = 35000 - 45.5 = 34954.5
    # buy_shares = floor(34954.5 / 10 / 100) * 100 = 3400
    
    assert shares == 3400


def test_estimate_pending_buy_shares_less_than_one_lot(runner):
    """测试不足一手的情况"""
    # 设置账户现金
    runner.account.state.cash = 5000.0
    
    # 参数
    ts_code = "600000.SH"
    price = 100.0  # 高价股
    target_weight = 0.1
    total_pending_count = 10
    retention_ratio = 0.3
    
    # 估算股数
    shares = runner._estimate_pending_buy_shares(
        ts_code=ts_code,
        price=price,
        target_weight=target_weight,
        total_pending_count=total_pending_count,
        pendding_capital_retention_ratio=retention_ratio
    )
    
    # 验证计算逻辑：
    # total_cash = 5000 * (1 - 0.3) = 3500
    # available_cash = 3500 / 10 = 350
    # target_value = 3500 * 0.1 = 350
    # estimated_cost ≈ 350 * 0.0013 = 0.455 (但min是5)
    # target_value + cost = 350 + 5 = 355 > 350，需要调整
    # target_value = 350 - 5 = 345
    # buy_shares = floor(345 / 100 / 100) * 100 = 0
    
    assert shares == 0


def test_estimate_pending_buy_shares_zero_price(runner):
    """测试价格为0的异常情况"""
    runner.account.state.cash = 100000.0
    
    shares = runner._estimate_pending_buy_shares(
        ts_code="000001.SZ",
        price=0.0,
        target_weight=0.1,
        total_pending_count=5,
        pendding_capital_retention_ratio=0.3
    )
    
    assert shares == 0


def test_estimate_pending_buy_shares_zero_pending_count(runner):
    """测试补位数量为0的异常情况"""
    runner.account.state.cash = 100000.0
    
    shares = runner._estimate_pending_buy_shares(
        ts_code="000001.SZ",
        price=10.0,
        target_weight=0.1,
        total_pending_count=0,
        pendding_capital_retention_ratio=0.3
    )
    
    assert shares == 0


def test_estimate_pending_buy_shares_multiple_targets(runner):
    """测试多个补位目标的情况（确保现金均分）"""
    # 设置账户现金
    runner.account.state.cash = 1000000.0
    
    # 参数
    price = 20.0
    target_weight = 0.05  # 每个5%
    total_pending_count = 10  # 10个补位目标
    retention_ratio = 0.3
    
    # 估算股数
    shares = runner._estimate_pending_buy_shares(
        ts_code="000001.SZ",
        price=price,
        target_weight=target_weight,
        total_pending_count=total_pending_count,
        pendding_capital_retention_ratio=retention_ratio
    )
    
    # 验证计算逻辑：
    # total_cash = 1000000 * (1 - 0.3) = 700000
    # available_cash = 700000 / 10 = 70000
    # target_value = 700000 * 0.05 = 35000
    # estimated_cost ≈ 35000 * 0.0013 = 45.5
    # target_value + cost = 35045.5 < 70000，所以不调整
    # buy_shares = floor(35000 / 20 / 100) * 100 = 1700
    
    assert shares == 1700


def test_estimate_pending_buy_shares_high_retention_ratio(runner):
    """测试高保留比例的情况"""
    # 设置账户现金
    runner.account.state.cash = 100000.0
    
    # 参数
    ts_code = "000001.SZ"
    price = 10.0
    target_weight = 0.2
    total_pending_count = 5
    retention_ratio = 0.8  # 保留80%（只用20%）
    
    # 估算股数
    shares = runner._estimate_pending_buy_shares(
        ts_code=ts_code,
        price=price,
        target_weight=target_weight,
        total_pending_count=total_pending_count,
        pendding_capital_retention_ratio=retention_ratio
    )
    
    # 验证计算逻辑：
    # total_cash = 100000 * (1 - 0.8) = 20000
    # available_cash = 20000 / 5 = 4000
    # target_value = 20000 * 0.2 = 4000
    # estimated_cost ≈ 4000 * 0.0013 = 5.2
    # target_value + cost = 4005.2 > 4000，需要调整
    # target_value = 4000 - 5.2 = 3994.8
    # buy_shares = floor(3994.8 / 10 / 100) * 100 = 300
    
    assert shares == 300


def test_estimate_pending_buy_shares_rounding(runner):
    """测试100股取整的情况"""
    # 设置账户现金
    runner.account.state.cash = 100000.0
    
    # 参数（调整使得结果不是整百倍数）
    ts_code = "000001.SZ"
    price = 33.33  # 不规则价格
    target_weight = 0.1
    total_pending_count = 5
    retention_ratio = 0.3
    
    # 估算股数
    shares = runner._estimate_pending_buy_shares(
        ts_code=ts_code,
        price=price,
        target_weight=target_weight,
        total_pending_count=total_pending_count,
        pendding_capital_retention_ratio=retention_ratio
    )
    
    # 验证计算逻辑：
    # total_cash = 100000 * (1 - 0.3) = 70000
    # available_cash = 70000 / 5 = 14000
    # target_value = 70000 * 0.1 = 7000
    # estimated_cost ≈ 7000 * 0.0013 = 9.1
    # target_value + cost = 7009.1 < 14000，所以不调整
    # buy_shares = floor(7000 / 33.33 / 100) * 100 = 200
    
    assert shares == 200
    # 确保是100的倍数
    assert shares % 100 == 0
