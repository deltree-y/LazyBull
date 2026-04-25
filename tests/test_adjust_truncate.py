"""测试手工修正/回滚功能"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.paper import (
    AccountState,
    Fill,
    NAVRecord,
    PaperStorage,
    PendingBuy,
    PendingSell,
    Position,
    TradeInstruction,
)


@pytest.fixture
def temp_storage_with_data():
    """临时存储目录，包含测试数据"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        
        # 创建测试数据
        # 1. 成交记录
        trades_data = pd.DataFrame([
            {'trade_date': '20260205', 'ts_code': '000001.SZ', 'action': 'buy', 'shares': 100, 
             'price': 10.0, 'amount': 1000.0, 'commission': 5.0, 'stamp_tax': 0.0, 
             'slippage': 1.0, 'total_cost': 1006.0, 'reason': '测试买入'},
            {'trade_date': '20260208', 'ts_code': '000002.SZ', 'action': 'buy', 'shares': 200, 
             'price': 20.0, 'amount': 4000.0, 'commission': 10.0, 'stamp_tax': 0.0, 
             'slippage': 2.0, 'total_cost': 4012.0, 'reason': '测试买入'},
            {'trade_date': '20260210', 'ts_code': '000001.SZ', 'action': 'sell', 'shares': 50, 
             'price': 11.0, 'amount': 550.0, 'commission': 2.5, 'stamp_tax': 5.5, 
             'slippage': 0.5, 'total_cost': 8.5, 'reason': '测试卖出'},
        ])
        trades_file = storage.trades_path / "trades.parquet"
        trades_data.to_parquet(trades_file, index=False)
        
        # 2. 净值记录
        nav_data = pd.DataFrame([
            {'trade_date': '20260205', 'cash': 95000.0, 'position_value': 1000.0, 
             'total_value': 96000.0, 'nav': 0.96},
            {'trade_date': '20260208', 'cash': 91000.0, 'position_value': 5000.0, 
             'total_value': 96000.0, 'nav': 0.96},
            {'trade_date': '20260210', 'cash': 91500.0, 'position_value': 4500.0, 
             'total_value': 96000.0, 'nav': 0.96},
        ])
        nav_file = storage.nav_path / "nav.parquet"
        nav_data.to_parquet(nav_file, index=False)
        
        # 3. 运行记录
        for date in ['20260205', '20260208', '20260210']:
            t0_record = {'trade_date': date, 'timestamp': '2026-02-10T12:00:00'}
            storage.save_run_record('t0', date, t0_record)
            
            t1_record = {'trade_date': date, 'fills_count': 1}
            storage.save_run_record('t1', date, t1_record)
        
        # 4. 交易指令
        for date in ['20260206', '20260209', '20260211']:
            instructions = [
                TradeInstruction(
                    ts_code='000001.SZ',
                    action='buy',
                    shares=100,
                    price_type='close',
                    reason='测试指令',
                    source_date=date
                )
            ]
            storage.save_instructions(date, instructions)
        
        # 5. pending_buys 和 pending_sells
        storage.save_pending_buys([])
        storage.save_pending_sells([])
        
        # 6. rebalance_state
        rebalance_state = {
            'last_rebalance_date': '20260210',
            'rebalance_freq': 5
        }
        storage.save_rebalance_state(rebalance_state)
        
        yield storage


def test_truncate_since_basic(temp_storage_with_data):
    """测试基本的truncate_since功能"""
    storage = temp_storage_with_data
    
    # 执行清理：从 20260208 开始（包含）
    storage.truncate_since('20260208')
    
    # 验证成交记录
    trades = pd.read_parquet(storage.trades_path / "trades.parquet")
    assert len(trades) == 1
    assert trades.iloc[0]['trade_date'] == '20260205'
    
    # 验证净值记录
    nav = pd.read_parquet(storage.nav_path / "nav.parquet")
    assert len(nav) == 1
    assert nav.iloc[0]['trade_date'] == '20260205'
    
    # 验证运行记录
    t0_files = list(storage.runs_path.glob("t0_*.json"))
    t1_files = list(storage.runs_path.glob("t1_*.json"))
    assert len(t0_files) == 1  # 只剩 20260205
    assert len(t1_files) == 1
    
    # 验证交易指令
    inst_files = list(storage.instructions_path.glob("*.parquet"))
    assert len(inst_files) == 1  # 只剩 20260206


def test_truncate_since_rebalance_state_rollback(temp_storage_with_data):
    """测试rebalance_state回滚逻辑"""
    storage = temp_storage_with_data
    
    # 初始状态：last_rebalance_date = 20260210
    # 清理从 20260209 开始
    storage.truncate_since('20260209')
    
    # 应该回滚到 20260208（cut-off之前最近的t0记录）
    rebalance_state = storage.load_rebalance_state()
    assert rebalance_state is not None
    assert rebalance_state['last_rebalance_date'] == '20260208'


def test_truncate_since_rebalance_state_delete(temp_storage_with_data):
    """测试rebalance_state删除逻辑（无有效t0记录）"""
    storage = temp_storage_with_data
    
    # 清理所有数据（从最早日期开始）
    storage.truncate_since('20260205')
    
    # 应该删除 rebalance_state
    rebalance_state = storage.load_rebalance_state()
    assert rebalance_state is None


def test_truncate_since_pending_queues(temp_storage_with_data):
    """测试清空延迟队列"""
    storage = temp_storage_with_data
    
    # 添加一些延迟订单
    from src.lazybull.paper.models import PendingBuy, PendingSell
    
    pending_buys = [
        PendingBuy(
            ts_code='000001.SZ',
            target_weight=0.1,
            reason='测试',
            create_date='20260208',
            attempts=1
        )
    ]
    pending_sells = [
        PendingSell(
            ts_code='000002.SZ',
            shares=100,
            target_weight=0.0,
            reason='测试',
            create_date='20260208',
            attempts=1
        )
    ]
    storage.save_pending_buys(pending_buys)
    storage.save_pending_sells(pending_sells)
    
    # 执行清理
    storage.truncate_since('20260208')
    
    # 验证队列已清空
    assert storage.load_pending_buys() == []
    assert storage.load_pending_sells() == []


def test_adjust_delete_position():
    """测试删除持仓"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        
        # 创建账户状态
        positions = {
            '000001.SZ': Position(
                ts_code='000001.SZ',
                shares=1000,
                buy_price=10.0,
                buy_cost=10000.0,
                buy_date='20260205'
            )
        }
        account_state = AccountState(
            cash=50000.0,
            positions=positions,
            last_update='20260205'
        )
        storage.save_account_state(account_state)
        
        # 模拟删除持仓操作
        position = account_state.positions['000001.SZ']
        released_cash = position.shares * position.buy_price
        account_state.cash += released_cash
        del account_state.positions['000001.SZ']
        account_state.last_update = '20260208'
        
        storage.save_account_state(account_state)
        
        # 验证
        loaded_state = storage.load_account_state()
        assert loaded_state.cash == 60000.0  # 50000 + 10000
        assert '000001.SZ' not in loaded_state.positions
        assert loaded_state.last_update == '20260208'


def test_adjust_update_position():
    """测试更新持仓"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        
        # 创建账户状态
        positions = {
            '000001.SZ': Position(
                ts_code='000001.SZ',
                shares=1000,
                buy_price=10.0,
                buy_cost=10000.0,
                buy_date='20260205'
            )
        }
        account_state = AccountState(
            cash=50000.0,
            positions=positions,
            last_update='20260205'
        )
        storage.save_account_state(account_state)
        
        # 模拟更新持仓操作
        position = account_state.positions['000001.SZ']
        old_cost = position.shares * position.buy_price  # 10000
        new_shares = 800
        new_price = 12.0
        new_cost = new_shares * new_price  # 9600
        delta_cash = old_cost - new_cost  # 400
        
        account_state.cash += delta_cash
        position.shares = new_shares
        position.buy_price = new_price
        position.buy_cost = new_cost
        account_state.last_update = '20260208'
        
        storage.save_account_state(account_state)
        
        # 验证
        loaded_state = storage.load_account_state()
        assert loaded_state.cash == 50400.0  # 50000 + 400
        assert loaded_state.positions['000001.SZ'].shares == 800
        assert loaded_state.positions['000001.SZ'].buy_price == 12.0
        assert loaded_state.positions['000001.SZ'].buy_cost == 9600.0


def test_adjust_add_shares():
    """测试加仓"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        
        # 创建账户状态
        positions = {
            '000001.SZ': Position(
                ts_code='000001.SZ',
                shares=1000,
                buy_price=10.0,
                buy_cost=10000.0,
                buy_date='20260205'
            )
        }
        account_state = AccountState(
            cash=50000.0,
            positions=positions,
            last_update='20260205'
        )
        storage.save_account_state(account_state)
        
        # 模拟加仓操作
        position = account_state.positions['000001.SZ']
        add_shares = 500
        add_price = 12.0
        add_cost = add_shares * add_price  # 6000
        
        # 扣减现金
        account_state.cash -= add_cost
        
        # 加权更新买入价格
        old_shares = position.shares
        old_price = position.buy_price
        new_total_shares = old_shares + add_shares
        new_buy_price = (old_price * old_shares + add_price * add_shares) / new_total_shares
        
        position.shares = new_total_shares
        position.buy_price = new_buy_price
        position.buy_cost = new_total_shares * new_buy_price
        account_state.last_update = '20260208'
        
        storage.save_account_state(account_state)
        
        # 验证
        loaded_state = storage.load_account_state()
        assert loaded_state.cash == 44000.0  # 50000 - 6000
        assert loaded_state.positions['000001.SZ'].shares == 1500
        # 加权平均价格 = (10*1000 + 12*500) / 1500 = 16000/1500 = 10.666...
        assert abs(loaded_state.positions['000001.SZ'].buy_price - 10.666666666666666) < 0.0001


def test_adjust_cash():
    """测试设置现金"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        
        # 创建账户状态
        account_state = AccountState(
            cash=50000.0,
            positions={},
            last_update='20260205'
        )
        storage.save_account_state(account_state)
        
        # 模拟设置现金操作
        account_state.cash = 100000.0
        account_state.last_update = '20260208'
        
        storage.save_account_state(account_state)
        
        # 验证
        loaded_state = storage.load_account_state()
        assert loaded_state.cash == 100000.0
        assert loaded_state.last_update == '20260208'


def test_truncate_since_empty_data():
    """测试清理空数据（边界情况）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        
        # 不创建任何数据，直接清理
        storage.truncate_since('20260208')
        
        # 应该不会报错，正常返回


def test_truncate_since_no_cutoff_data():
    """测试清理不存在的日期（边界情况）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        
        # 创建一些早于 cut-off 的数据
        trades_data = pd.DataFrame([
            {'trade_date': '20260205', 'ts_code': '000001.SZ', 'action': 'buy', 'shares': 100, 
             'price': 10.0, 'amount': 1000.0, 'commission': 5.0, 'stamp_tax': 0.0, 
             'slippage': 1.0, 'total_cost': 1006.0, 'reason': '测试买入'},
        ])
        trades_file = storage.trades_path / "trades.parquet"
        trades_data.to_parquet(trades_file, index=False)
        
        # 清理未来日期（不应该删除任何数据）
        storage.truncate_since('20260220')
        
        # 验证数据仍然存在
        trades = pd.read_parquet(storage.trades_path / "trades.parquet")
        assert len(trades) == 1


# ==================== reset_t0 测试 ====================


@pytest.fixture
def temp_storage_for_reset_t0():
    """临时存储目录，模拟T0已执行、T1也已执行的完整场景"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)

        # 更早的一次完整T0+T1循环（20260205）
        storage.save_run_record('t0', '20260205', {
            'trade_date': '20260205', 't1_date': '20260206',
        })
        storage.save_run_record('t1', '20260206', {
            'trade_date': '20260206', 'fills_count': 2,
        })

        # 最新的T0（20260210），T1日期为20260211
        t0_record = {
            'trade_date': '20260210',
            't1_date': '20260211',
            'top_n': 5,
            'timestamp': '2026-02-10T15:30:00',
        }
        storage.save_run_record('t0', '20260210', t0_record)

        # T1已执行
        storage.save_run_record('t1', '20260211', {
            'trade_date': '20260211', 'fills_count': 3,
        })

        # T1的交易指令
        instructions = [
            TradeInstruction(
                ts_code='000001.SZ', action='buy', shares=100,
                price_type='close', reason='信号买入', source_date='20260210',
            ),
        ]
        storage.save_instructions('20260211', instructions)

        # 成交记录（跨越两个周期）
        trades_data = pd.DataFrame([
            {'trade_date': '20260206', 'ts_code': '000001.SZ', 'action': 'buy',
             'shares': 100, 'price': 10.0, 'amount': 1000.0, 'commission': 5.0,
             'stamp_tax': 0.0, 'slippage': 1.0, 'total_cost': 1006.0, 'reason': '买入'},
            {'trade_date': '20260211', 'ts_code': '000002.SZ', 'action': 'buy',
             'shares': 200, 'price': 20.0, 'amount': 4000.0, 'commission': 10.0,
             'stamp_tax': 0.0, 'slippage': 2.0, 'total_cost': 4012.0, 'reason': '买入'},
        ])
        trades_file = storage.trades_path / "trades.parquet"
        trades_data.to_parquet(trades_file, index=False)

        # 净值记录
        nav_data = pd.DataFrame([
            {'trade_date': '20260206', 'cash': 95000.0, 'position_value': 1000.0,
             'total_value': 96000.0, 'nav': 0.96},
            {'trade_date': '20260211', 'cash': 91000.0, 'position_value': 5000.0,
             'total_value': 96000.0, 'nav': 0.96},
        ])
        nav_file = storage.nav_path / "nav.parquet"
        nav_data.to_parquet(nav_file, index=False)

        # 账户状态（last_update 为 T1 执行日期）
        account_state = AccountState(
            cash=91000.0,
            positions={
                '000001.SZ': Position(
                    ts_code='000001.SZ', shares=100,
                    buy_price=10.0, buy_cost=1006.0, buy_date='20260206',
                ),
            },
            last_update='20260211',
        )
        storage.save_account_state(account_state)

        # 延迟订单
        pending_buys = [
            PendingBuy(
                ts_code='000003.SZ', target_weight=0.1,
                reason='补位买入', create_date='20260210',
            ),
        ]
        storage.save_pending_buys(pending_buys)
        pending_sells = [
            PendingSell(
                ts_code='000004.SZ', shares=100, target_weight=0.0,
                reason='停牌延迟', create_date='20260210',
            ),
        ]
        storage.save_pending_sells(pending_sells)

        # 调仓状态
        storage.save_rebalance_state({
            'last_rebalance_date': '20260210',
            'rebalance_freq': 5,
        })

        yield storage


def test_reset_t0_full_rollback(temp_storage_for_reset_t0):
    """测试 reset_t0 清空所有交易数据，恢复为新账户状态"""
    storage = temp_storage_for_reset_t0

    # 先保存一个 config 以提供 initial_capital
    storage.save_config({'initial_capital': 100000.0})

    stats = storage.reset_t0()

    assert stats['t0_date'] == '20260210'

    # 所有运行记录已删除
    assert not storage.check_run_exists('t0', '20260210')
    assert not storage.check_run_exists('t1', '20260211')
    assert not storage.check_run_exists('t0', '20260205')
    assert not storage.check_run_exists('t1', '20260206')

    # 交易指令已删除
    assert storage.load_instructions('20260211') is None

    # 延迟订单已清空
    assert storage.load_pending_buys() == []
    assert storage.load_pending_sells() == []

    # 账户状态已重置为新账户
    account = storage.load_account_state()
    assert account.last_update == ''
    assert account.cash == 100000.0
    assert len(account.positions) == 0

    # 成交记录已清空
    trades_file = storage.trades_path / "trades.parquet"
    assert not trades_file.exists()

    # 净值记录已清空
    nav_file = storage.nav_path / "nav.parquet"
    assert not nav_file.exists()

    # config.yaml 仍然存在
    config = storage.load_config()
    assert config is not None
    assert config['initial_capital'] == 100000.0


def test_reset_t0_no_records():
    """测试 reset_t0 无任何T0运行记录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)

        stats = storage.reset_t0()

        assert stats['t0_date'] is None


def test_find_latest_t0():
    """测试 find_latest_t0 找到最新的T0日期"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)

        # 无记录时返回None
        assert storage.find_latest_t0() is None

        # 添加多个T0记录
        for date in ['20260205', '20260210', '20260208']:
            storage.save_run_record('t0', date, {'trade_date': date})

        # 应返回最新的
        assert storage.find_latest_t0() == '20260210'


def test_reset_t0_account_reset_to_initial():
    """测试 reset_t0 重置账户为初始状态"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)

        # 保存配置
        storage.save_config({'initial_capital': 200000.0})

        # 设置账户状态（模拟已交易过）
        account = AccountState(
            cash=80000.0,
            positions={
                '000001.SZ': Position(
                    ts_code='000001.SZ', shares=100,
                    buy_price=10.0, buy_cost=5.0, buy_date='20260206',
                ),
            },
            last_update='20260211',
        )
        storage.save_account_state(account)

        # T0记录
        storage.save_run_record('t0', '20260210', {
            'trade_date': '20260210', 't1_date': '20260211',
        })

        storage.reset_t0()

        updated_account = storage.load_account_state()
        assert updated_account.last_update == ''
        assert updated_account.cash == 200000.0
        assert len(updated_account.positions) == 0
