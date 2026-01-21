"""测试纸面交易模块"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.common.cost import CostModel
from src.lazybull.paper import (
    AccountState,
    Fill,
    NAVRecord,
    Order,
    PaperAccount,
    PaperBroker,
    PaperStorage,
    Position,
    TargetWeight,
)


@pytest.fixture
def temp_storage():
    """临时存储目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        yield storage


@pytest.fixture
def sample_account():
    """示例账户"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        yield account


@pytest.fixture
def sample_prices():
    """示例价格"""
    return {
        '000001.SZ': 10.0,
        '000002.SZ': 20.0,
        '600000.SH': 15.0
    }


def test_position_model():
    """测试持仓数据模型"""
    pos = Position(
        ts_code='000001.SZ',
        shares=100,
        buy_price=10.0,
        buy_cost=15.0,
        buy_date='20260121'
    )
    
    assert pos.ts_code == '000001.SZ'
    assert pos.shares == 100
    assert pos.buy_price == 10.0
    assert pos.buy_cost == 15.0
    assert pos.buy_date == '20260121'


def test_order_model():
    """测试订单数据模型"""
    order = Order(
        ts_code='000001.SZ',
        action='buy',
        shares=100,
        price=10.0,
        target_weight=0.5,
        current_weight=0.0,
        reason='新建仓位'
    )
    
    assert order.ts_code == '000001.SZ'
    assert order.action == 'buy'
    assert order.shares == 100
    assert order.price == 10.0
    assert order.target_weight == 0.5
    assert order.current_weight == 0.0


def test_fill_model():
    """测试成交记录数据模型"""
    fill = Fill(
        trade_date='20260121',
        ts_code='000001.SZ',
        action='buy',
        shares=100,
        price=10.0,
        amount=1000.0,
        commission=5.0,
        stamp_tax=0.0,
        slippage=0.5,
        total_cost=5.5,
        reason='新建仓位'
    )
    
    assert fill.trade_date == '20260121'
    assert fill.ts_code == '000001.SZ'
    assert fill.action == 'buy'
    assert fill.shares == 100
    assert fill.total_cost == 5.5


def test_target_weight_model():
    """测试目标权重数据模型"""
    target = TargetWeight(
        ts_code='000001.SZ',
        target_weight=0.5,
        reason='信号生成'
    )
    
    assert target.ts_code == '000001.SZ'
    assert target.target_weight == 0.5
    assert target.reason == '信号生成'


def test_account_state_model(sample_prices):
    """测试账户状态数据模型"""
    state = AccountState(
        cash=50000.0,
        positions={
            '000001.SZ': Position(
                ts_code='000001.SZ',
                shares=1000,
                buy_price=10.0,
                buy_cost=15.0,
                buy_date='20260121'
            )
        },
        last_update='20260121'
    )
    
    # 测试持仓市值计算
    position_value = state.get_position_value(sample_prices)
    assert position_value == 10000.0  # 1000 shares * 10.0
    
    # 测试总资产计算
    total_value = state.get_total_value(sample_prices)
    assert total_value == 60000.0  # 50000 cash + 10000 position
    
    # 测试持仓权重计算
    weight = state.get_position_weight('000001.SZ', sample_prices)
    assert abs(weight - 10000.0/60000.0) < 1e-6


def test_storage_save_and_load_pending(temp_storage):
    """测试存储和读取待执行目标"""
    targets = [
        TargetWeight(ts_code='000001.SZ', target_weight=0.5, reason='信号生成'),
        TargetWeight(ts_code='000002.SZ', target_weight=0.3, reason='信号生成'),
    ]
    
    # 保存
    temp_storage.save_pending_weights('20260121', targets)
    
    # 读取
    loaded_targets = temp_storage.load_pending_weights('20260121')
    
    assert loaded_targets is not None
    assert len(loaded_targets) == 2
    assert loaded_targets[0].ts_code == '000001.SZ'
    assert loaded_targets[0].target_weight == 0.5


def test_storage_save_and_load_account_state(temp_storage):
    """测试存储和读取账户状态"""
    state = AccountState(
        cash=50000.0,
        positions={
            '000001.SZ': Position(
                ts_code='000001.SZ',
                shares=1000,
                buy_price=10.0,
                buy_cost=15.0,
                buy_date='20260121'
            )
        },
        last_update='20260121'
    )
    
    # 保存
    temp_storage.save_account_state(state)
    
    # 读取
    loaded_state = temp_storage.load_account_state()
    
    assert loaded_state is not None
    assert loaded_state.cash == 50000.0
    assert '000001.SZ' in loaded_state.positions
    assert loaded_state.positions['000001.SZ'].shares == 1000


def test_storage_append_trade(temp_storage):
    """测试追加成交记录"""
    fill = Fill(
        trade_date='20260121',
        ts_code='000001.SZ',
        action='buy',
        shares=100,
        price=10.0,
        amount=1000.0,
        commission=5.0,
        stamp_tax=0.0,
        slippage=0.5,
        total_cost=5.5,
        reason='新建仓位'
    )
    
    # 追加
    temp_storage.append_trade(fill)
    
    # 读取
    trades_df = temp_storage.load_all_trades()
    
    assert trades_df is not None
    assert len(trades_df) == 1
    assert trades_df.iloc[0]['ts_code'] == '000001.SZ'
    assert trades_df.iloc[0]['action'] == 'buy'


def test_storage_append_nav(temp_storage):
    """测试追加净值记录"""
    nav_record = NAVRecord(
        trade_date='20260121',
        cash=50000.0,
        position_value=50000.0,
        total_value=100000.0,
        nav=1.0
    )
    
    # 追加
    temp_storage.append_nav(nav_record)
    
    # 读取
    nav_df = temp_storage.load_all_nav()
    
    assert nav_df is not None
    assert len(nav_df) == 1
    assert nav_df.iloc[0]['trade_date'] == '20260121'
    assert nav_df.iloc[0]['nav'] == 1.0


def test_account_initialization(sample_account):
    """测试账户初始化"""
    assert sample_account.get_cash() == 100000.0
    assert len(sample_account.get_positions()) == 0


def test_account_update_cash(sample_account):
    """测试更新现金"""
    sample_account.update_cash(-10000.0)
    assert sample_account.get_cash() == 90000.0
    
    sample_account.update_cash(5000.0)
    assert sample_account.get_cash() == 95000.0


def test_account_add_position(sample_account, sample_prices):
    """测试增加持仓"""
    sample_account.add_position(
        ts_code='000001.SZ',
        shares=1000,
        buy_price=10.0,
        buy_cost=15.0,
        buy_date='20260121'
    )
    
    pos = sample_account.get_position('000001.SZ')
    assert pos is not None
    assert pos.shares == 1000
    assert pos.buy_price == 10.0
    
    # 累加持仓
    sample_account.add_position(
        ts_code='000001.SZ',
        shares=500,
        buy_price=12.0,
        buy_cost=10.0,
        buy_date='20260122'
    )
    
    pos = sample_account.get_position('000001.SZ')
    assert pos.shares == 1500
    # 平均价格应该是 (1000*10 + 500*12) / 1500
    expected_avg_price = (1000 * 10.0 + 500 * 12.0) / 1500
    assert abs(pos.buy_price - expected_avg_price) < 0.01


def test_account_reduce_position(sample_account):
    """测试减少持仓"""
    sample_account.add_position(
        ts_code='000001.SZ',
        shares=1000,
        buy_price=10.0,
        buy_cost=15.0,
        buy_date='20260121'
    )
    
    # 部分卖出
    sample_account.reduce_position('000001.SZ', 300)
    pos = sample_account.get_position('000001.SZ')
    assert pos is not None
    assert pos.shares == 700
    
    # 全部卖出
    sample_account.reduce_position('000001.SZ', 700)
    pos = sample_account.get_position('000001.SZ')
    assert pos is None


def test_account_get_position_weight(sample_account, sample_prices):
    """测试计算持仓权重"""
    # 添加持仓
    sample_account.add_position(
        ts_code='000001.SZ',
        shares=1000,
        buy_price=10.0,
        buy_cost=15.0,
        buy_date='20260121'
    )
    
    # 总资产 = 100000 - 10000 - 15 = 89985
    # 实际总资产 = 89985 + 1000*10 = 99985
    # 权重 = 10000 / 99985
    sample_account.update_cash(-10015.0)  # 买入成本
    
    weight = sample_account.get_position_weight('000001.SZ', sample_prices)
    expected_weight = 10000.0 / (89985.0 + 10000.0)
    assert abs(weight - expected_weight) < 1e-6


def test_broker_generate_orders_new_position():
    """测试生成订单：新建仓位"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        broker = PaperBroker(account, storage=storage)
        
        targets = [
            TargetWeight(ts_code='000001.SZ', target_weight=0.5, reason='新建仓位'),
        ]
        
        prices = {'000001.SZ': 10.0}
        
        orders = broker.generate_orders(targets, prices, '20260121')
        
        # 应该生成买入订单
        assert len(orders) == 1
        assert orders[0].action == 'buy'
        assert orders[0].ts_code == '000001.SZ'
        assert orders[0].shares > 0


def test_broker_generate_orders_reduce_position():
    """测试生成订单：减仓"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        
        # 先建立持仓
        account.add_position(
            ts_code='000001.SZ',
            shares=5000,
            buy_price=10.0,
            buy_cost=15.0,
            buy_date='20260120'
        )
        account.update_cash(-50015.0)
        
        broker = PaperBroker(account, storage=storage)
        
        # 目标权重降低
        targets = [
            TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='减仓'),
        ]
        
        prices = {'000001.SZ': 10.0}
        
        orders = broker.generate_orders(targets, prices, '20260121')
        
        # 应该生成卖出订单
        assert len(orders) == 1
        assert orders[0].action == 'sell'
        assert orders[0].ts_code == '000001.SZ'


def test_broker_execute_buy_order():
    """测试执行买入订单"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        broker = PaperBroker(account, cost_model=CostModel(), storage=storage)
        
        order = Order(
            ts_code='000001.SZ',
            action='buy',
            shares=1000,
            price=10.0,
            target_weight=0.1,
            current_weight=0.0,
            reason='新建仓位'
        )
        
        fills = broker.execute_orders([order], '20260121', 'close', 'close')
        
        # 应该成功执行
        assert len(fills) == 1
        assert fills[0].action == 'buy'
        assert fills[0].shares == 1000
        
        # 检查账户状态
        pos = account.get_position('000001.SZ')
        assert pos is not None
        assert pos.shares == 1000


def test_broker_execute_sell_order():
    """测试执行卖出订单"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        
        # 先建立持仓
        account.add_position(
            ts_code='000001.SZ',
            shares=1000,
            buy_price=10.0,
            buy_cost=15.0,
            buy_date='20260120'
        )
        account.update_cash(-10015.0)
        
        broker = PaperBroker(account, cost_model=CostModel(), storage=storage)
        
        order = Order(
            ts_code='000001.SZ',
            action='sell',
            shares=500,
            price=12.0,
            target_weight=0.05,
            current_weight=0.1,
            reason='减仓'
        )
        
        fills = broker.execute_orders([order], '20260121', 'close', 'close')
        
        # 应该成功执行
        assert len(fills) == 1
        assert fills[0].action == 'sell'
        assert fills[0].shares == 500
        
        # 检查账户状态
        pos = account.get_position('000001.SZ')
        assert pos is not None
        assert pos.shares == 500


def test_storage_pending_weights_not_exist(temp_storage):
    """测试读取不存在的待执行目标"""
    result = temp_storage.load_pending_weights('20991231')
    assert result is None


def test_storage_account_state_not_exist(temp_storage):
    """测试读取不存在的账户状态"""
    result = temp_storage.load_account_state()
    assert result is None


def test_storage_trades_not_exist(temp_storage):
    """测试读取不存在的成交记录"""
    result = temp_storage.load_all_trades()
    assert result is None


def test_storage_nav_not_exist(temp_storage):
    """测试读取不存在的净值记录"""
    result = temp_storage.load_all_nav()
    assert result is None
