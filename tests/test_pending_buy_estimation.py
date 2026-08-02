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
    """测试正常情况下的股数估算（组合价值口径）"""
    # 设置账户现金
    runner.account.state.cash = 500000.0

    # 参数
    price = 10.0
    target_weight = 0.1  # 10%
    current_total_value = 500000.0

    # 估算股数
    shares, reason = runner._analyze_pending_buy_shares_backtest_style(
        ts_code="000001.SZ",
        price=price,
        target_weight=target_weight,
        current_total_value=current_total_value,
    )

    # 组合价值口径：target_value = 500000 * 0.1 = 50000
    # buy_shares = floor(50000 / 10 / 100) * 100 = 5000
    # 现金充足，直接可买入
    assert shares == 5000
    assert reason == "可买入"


def test_estimate_pending_buy_shares_cash_limit(runner):
    """测试现金受限时按剩余现金缩量（组合价值口径）"""
    # 设置账户现金
    runner.account.state.cash = 100000.0

    # 参数（目标金额超出可用现金）
    price = 10.0
    target_weight = 0.9  # 90%
    current_total_value = 200000.0

    # 估算股数
    shares, reason = runner._analyze_pending_buy_shares_backtest_style(
        ts_code="000001.SZ",
        price=price,
        target_weight=target_weight,
        current_total_value=current_total_value,
    )

    # target_value = 200000 * 0.9 = 180000 → 18000 股，金额 180000+费用 > 现金 100000
    # 按剩余现金缩量：lot((100000 - 234) / 10) = 9900 股，金额 99000+费用 <= 现金
    assert shares == 9900
    assert "缩量" in reason


def test_estimate_pending_buy_shares_less_than_one_lot(runner):
    """测试不足一手的情况（组合价值口径）"""
    # 设置账户现金
    runner.account.state.cash = 5000.0

    # 参数
    price = 100.0  # 高价股
    target_weight = 0.01
    current_total_value = 5000.0

    # 估算股数
    shares, reason = runner._analyze_pending_buy_shares_backtest_style(
        ts_code="600000.SH",
        price=price,
        target_weight=target_weight,
        current_total_value=current_total_value,
    )

    # target_value = 5000 * 0.01 = 50 < 一手(100股*100=10000) → 不足一手
    assert shares == 0
    assert reason == "目标金额不足一手"


def test_estimate_pending_buy_shares_zero_price(runner):
    """测试价格为0的异常情况"""
    runner.account.state.cash = 100000.0

    shares, reason = runner._analyze_pending_buy_shares_backtest_style(
        ts_code="000001.SZ",
        price=0.0,
        target_weight=0.1,
        current_total_value=100000.0,
    )

    assert shares == 0
    assert reason == "无有效价格"


def test_estimate_pending_buy_shares_zero_weight(runner):
    """测试目标权重为0的异常情况"""
    runner.account.state.cash = 100000.0

    shares, reason = runner._analyze_pending_buy_shares_backtest_style(
        ts_code="000001.SZ",
        price=10.0,
        target_weight=0.0,
        current_total_value=100000.0,
    )

    assert shares == 0
    assert reason == "槽位权重<=0"


def test_estimate_pending_buy_shares_multiple_targets(runner):
    """测试组合总资产口径（目标权重决定金额，与执行一致）"""
    # 设置账户现金
    runner.account.state.cash = 1000000.0

    # 参数
    price = 20.0
    target_weight = 0.05  # 5%
    current_total_value = 1000000.0

    # 估算股数
    shares, reason = runner._analyze_pending_buy_shares_backtest_style(
        ts_code="000001.SZ",
        price=price,
        target_weight=target_weight,
        current_total_value=current_total_value,
    )

    # target_value = 1000000 * 0.05 = 50000 → floor(50000 / 20 / 100) * 100 = 2500
    assert shares == 2500
    assert reason == "可买入"


def test_estimate_pending_buy_shares_cash_not_enough(runner):
    """测试现金严重不足（缩量后仍不足一手）"""
    # 设置账户现金
    runner.account.state.cash = 1000.0

    # 参数
    price = 10.0
    target_weight = 0.5
    current_total_value = 20000.0

    # 估算股数
    shares, reason = runner._analyze_pending_buy_shares_backtest_style(
        ts_code="000001.SZ",
        price=price,
        target_weight=target_weight,
        current_total_value=current_total_value,
    )

    # target_value = 10000 → 1000 股，金额 10000+费用 > 现金 1000
    # 缩量后 lot((1000 - 15)/10) = 0 < 一手 → 不足一手
    assert shares == 0
    assert reason == "现金不足(缩量后不足一手)"


def test_estimate_pending_buy_shares_rounding(runner):
    """测试100股取整的情况（组合价值口径）"""
    # 设置账户现金
    runner.account.state.cash = 100000.0

    # 参数（不规则价格）
    price = 33.33
    target_weight = 0.1
    current_total_value = 100000.0

    # 估算股数
    shares, reason = runner._analyze_pending_buy_shares_backtest_style(
        ts_code="000001.SZ",
        price=price,
        target_weight=target_weight,
        current_total_value=current_total_value,
    )

    # target_value = 10000 → floor(10000 / 33.33 / 100) * 100 = 300
    assert shares == 300
    assert reason == "可买入"
    # 确保是100的倍数
    assert shares % 100 == 0
