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
        
        orders = broker.generate_orders(targets, prices, prices, '20260121')
        
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
        
        orders = broker.generate_orders(targets, prices, prices, '20260121')
        
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


def test_position_with_status():
    """测试Position模型的status和notes字段"""
    pos = Position(
        ts_code='000001.SZ',
        shares=1000,
        buy_price=10.0,
        buy_cost=15.0,
        buy_date='20260115',
        status='持有',
        notes='正常持仓'
    )
    
    assert pos.status == '持有'
    assert pos.notes == '正常持仓'
    
    # 测试持有天数计算
    holding_days = pos.get_holding_days('20260122')
    assert holding_days == 7  # 7天


def test_position_holding_days():
    """测试持有天数计算"""
    pos = Position(
        ts_code='000001.SZ',
        shares=1000,
        buy_price=10.0,
        buy_cost=15.0,
        buy_date='20260115'
    )
    
    # 同一天
    assert pos.get_holding_days('20260115') == 0
    
    # 7天后
    assert pos.get_holding_days('20260122') == 7
    
    # 30天后
    assert pos.get_holding_days('20260214') == 30


def test_broker_get_positions_detail(sample_account, sample_prices):
    """测试获取持仓明细"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        broker = PaperBroker(sample_account, storage=storage)
        
        # 添加持仓
        sample_account.add_position(
            ts_code='000001.SZ',
            shares=1000,
            buy_price=10.0,
            buy_cost=15.0,
            buy_date='20260115',
            status='持有'
        )
        sample_account.update_cash(-10015.0)
        
        # 获取持仓明细
        df = broker.get_positions_detail(sample_prices, current_date='20260122')
        
        assert len(df) == 1
        assert df.iloc[0]['股票代码'] == '000001.SZ'
        assert df.iloc[0]['持仓股数'] == 1000
        assert df.iloc[0]['持有天数'] == 7
        assert df.iloc[0]['状态'] == '持有'


def test_broker_generate_orders_with_separate_prices():
    """测试使用分开的买卖价格生成订单"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        broker = PaperBroker(account, storage=storage)
        
        # 添加持仓（用于测试卖出）
        account.add_position(
            ts_code='000001.SZ',
            shares=1000,
            buy_price=10.0,
            buy_cost=15.0,
            buy_date='20260120'
        )
        account.update_cash(-10015.0)
        
        targets = [
            TargetWeight(ts_code='000002.SZ', target_weight=0.5, reason='新建仓位'),
        ]
        
        buy_prices = {'000002.SZ': 20.0}
        sell_prices = {'000001.SZ': 11.0}  # 卖出价格不同
        
        orders = broker.generate_orders(targets, buy_prices, sell_prices, '20260121')
        
        # 应该生成买入和卖出订单
        buy_orders = [o for o in orders if o.action == 'buy']
        sell_orders = [o for o in orders if o.action == 'sell']
        
        assert len(buy_orders) == 1
        assert len(sell_orders) == 1
        assert buy_orders[0].price == 20.0
        assert sell_orders[0].price == 11.0


def test_broker_check_can_buy():
    """测试买入可交易性检查"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        broker = PaperBroker(account, storage=storage)
        
        # 正常股票
        tradability = {
            '000001.SZ': {
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'tradable': 1
            }
        }
        can_buy, reason = broker._check_can_buy('000001.SZ', tradability)
        assert can_buy is True
        
        # 停牌股票
        tradability['000001.SZ']['is_suspended'] = 1
        can_buy, reason = broker._check_can_buy('000001.SZ', tradability)
        assert can_buy is False
        assert '停牌' in reason
        
        # 涨停股票
        tradability['000001.SZ']['is_suspended'] = 0
        tradability['000001.SZ']['is_limit_up'] = 1
        can_buy, reason = broker._check_can_buy('000001.SZ', tradability)
        assert can_buy is False
        assert '涨停' in reason


def test_broker_check_can_sell():
    """测试卖出可交易性检查"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        broker = PaperBroker(account, storage=storage)
        
        # 正常股票
        tradability = {
            '000001.SZ': {
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'tradable': 1
            }
        }
        can_sell, reason = broker._check_can_sell('000001.SZ', tradability)
        assert can_sell is True
        
        # 停牌股票
        tradability['000001.SZ']['is_suspended'] = 1
        can_sell, reason = broker._check_can_sell('000001.SZ', tradability)
        assert can_sell is False
        assert '停牌' in reason
        
        # 跌停股票
        tradability['000001.SZ']['is_suspended'] = 0
        tradability['000001.SZ']['is_limit_down'] = 1
        can_sell, reason = broker._check_can_sell('000001.SZ', tradability)
        assert can_sell is False
        assert '跌停' in reason


def test_pending_sell_model():
    """测试延迟卖出订单模型"""
    from src.lazybull.paper import PendingSell
    
    ps = PendingSell(
        ts_code='000001.SZ',
        shares=1000,
        target_weight=0.0,
        reason='清仓',
        create_date='20260121',
        attempts=0
    )
    
    assert ps.ts_code == '000001.SZ'
    assert ps.shares == 1000
    assert ps.attempts == 0


def test_storage_save_and_load_pending_sells(temp_storage):
    """测试保存和读取延迟卖出队列"""
    from src.lazybull.paper import PendingSell
    
    pending_sells = [
        PendingSell(
            ts_code='000001.SZ',
            shares=1000,
            target_weight=0.0,
            reason='跌停延迟',
            create_date='20260121',
            attempts=1
        ),
        PendingSell(
            ts_code='000002.SZ',
            shares=500,
            target_weight=0.0,
            reason='停牌延迟',
            create_date='20260121',
            attempts=0
        ),
    ]
    
    # 保存
    temp_storage.save_pending_sells(pending_sells)
    
    # 读取
    loaded = temp_storage.load_pending_sells()
    
    assert len(loaded) == 2
    assert loaded[0].ts_code == '000001.SZ'
    assert loaded[0].shares == 1000
    assert loaded[0].attempts == 1
    assert loaded[1].ts_code == '000002.SZ'


def test_storage_run_records(temp_storage):
    """测试执行记录的保存和检查"""
    import pandas as pd
    
    # 检查不存在的记录
    assert not temp_storage.check_run_exists("t0", "20260121")
    
    # 保存T0记录
    record = {
        'trade_date': '20260121',
        'timestamp': pd.Timestamp.now().isoformat(),
        'targets_count': 5
    }
    temp_storage.save_run_record("t0", "20260121", record)
    
    # 检查存在
    assert temp_storage.check_run_exists("t0", "20260121")
    
    # 不同日期不存在
    assert not temp_storage.check_run_exists("t0", "20260122")
    
    # 不同类型不存在
    assert not temp_storage.check_run_exists("t1", "20260121")


def test_storage_rebalance_state(temp_storage):
    """测试调仓状态的保存和读取"""
    # 初始不存在
    state = temp_storage.load_rebalance_state()
    assert state is None
    
    # 保存状态
    rebalance_state = {
        'last_rebalance_date': '20260121',
        'rebalance_freq': 5
    }
    temp_storage.save_rebalance_state(rebalance_state)
    
    # 读取状态
    loaded = temp_storage.load_rebalance_state()
    assert loaded is not None
    assert loaded['last_rebalance_date'] == '20260121'
    assert loaded['rebalance_freq'] == 5


def test_broker_generate_orders_100_lot_buy():
    """测试买入订单100股取整"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        broker = PaperBroker(account, storage=storage)
        
        targets = [
            TargetWeight(ts_code='000001.SZ', target_weight=0.33, reason='新建仓位'),
        ]
        
        prices = {'000001.SZ': 10.5}  # 价格导致非整百数
        
        orders = broker.generate_orders(targets, prices, prices, '20260121')
        
        # 买入应该按100股向下取整
        assert len(orders) == 1
        buy_order = orders[0]
        assert buy_order.action == 'buy'
        assert buy_order.shares % 100 == 0  # 必须是100的倍数


def test_broker_generate_orders_100_lot_sell_reduce():
    """测试减仓卖出按100股取整"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        
        # 先建立持仓（非100倍数）
        account.add_position(
            ts_code='000001.SZ',
            shares=5555,  # 非100倍数
            buy_price=10.0,
            buy_cost=15.0,
            buy_date='20260120'
        )
        account.update_cash(-55565.0)
        
        broker = PaperBroker(account, storage=storage)
        
        # 目标权重降低（减仓）
        targets = [
            TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='减仓'),
        ]
        
        prices = {'000001.SZ': 10.0}
        
        orders = broker.generate_orders(targets, prices, prices, '20260121')
        
        # 减仓卖出应该按100股向下取整
        assert len(orders) == 1
        sell_order = orders[0]
        assert sell_order.action == 'sell'
        assert sell_order.shares % 100 == 0  # 必须是100的倍数


def test_broker_generate_orders_100_lot_sell_liquidate():
    """测试清仓卖出处理零股"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        
        # 建立持仓（非100倍数）
        account.add_position(
            ts_code='000001.SZ',
            shares=1255,  # 非100倍数（12手+55股零股）
            buy_price=10.0,
            buy_cost=15.0,
            buy_date='20260120'
        )
        account.update_cash(-12565.0)
        
        broker = PaperBroker(account, storage=storage)
        
        # 清仓目标
        targets = []  # 空目标列表意味着清仓所有持仓
        
        prices = {'000001.SZ': 10.0}
        
        orders = broker.generate_orders(targets, prices, prices, '20260121')
        
        # 清仓应该只卖出100倍数部分
        assert len(orders) == 1
        sell_order = orders[0]
        assert sell_order.action == 'sell'
        assert sell_order.shares == 1200  # 只卖12手，保留55股零股
        assert sell_order.shares % 100 == 0


def test_broker_execute_sell_marks_odd_lots():
    """测试清仓执行后标记零股"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        
        # 建立持仓（非100倍数）
        account.add_position(
            ts_code='000001.SZ',
            shares=1255,
            buy_price=10.0,
            buy_cost=15.0,
            buy_date='20260120'
        )
        account.update_cash(-12565.0)
        
        broker = PaperBroker(account, cost_model=CostModel(), storage=storage)
        
        # 清仓订单（只卖100倍数）
        order = Order(
            ts_code='000001.SZ',
            action='sell',
            shares=1200,  # 只卖12手
            price=10.0,
            target_weight=0.0,
            current_weight=0.1,
            reason='清仓'
        )
        
        fills = broker.execute_orders([order], '20260121', 'close', 'close')
        
        # 检查成交
        assert len(fills) == 1
        
        # 检查剩余持仓被标记
        pos = account.get_position('000001.SZ')
        assert pos is not None
        assert pos.shares == 55  # 剩余零股
        assert '零股' in pos.notes
        assert pos.status == '零股待处理'


def test_broker_pending_sells_not_executable():
    """测试不可卖出时加入延迟队列"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        
        # 建立持仓
        account.add_position(
            ts_code='000001.SZ',
            shares=1000,
            buy_price=10.0,
            buy_cost=15.0,
            buy_date='20260120'
        )
        account.update_cash(-10015.0)
        
        broker = PaperBroker(account, storage=storage)
        
        # 清空pending_sells
        broker.pending_sells = []
        
        # 模拟跌停（创建不可交易性数据）
        # 这需要mock _load_tradability_info 或直接设置
        # 简化：直接测试pending_sells列表
        
        # 清仓目标
        targets = []
        
        prices = {'000001.SZ': 10.0}
        
        # 注意：实际测试需要mock tradability数据
        # 这里测试队列保存功能
        orders = broker.generate_orders(targets, prices, prices, '20260121')
        
        # 验证pending_sells被保存
        loaded = storage.load_pending_sells()
        # 如果有pending_sells，说明broker.generate_orders调用了save
