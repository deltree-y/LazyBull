"""测试买入失败补位机制

测试场景：
1. 买入失败时生成 PendingBuy 记录
2. 重试 PendingBuy 时检查可交易性
3. 超过最大尝试次数后移除
4. 同日不重复推进 attempts
5. 生成补位目标应用一手约束
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.lazybull.common.trading_config import TradingConfig
from src.lazybull.paper import PaperStorage
from src.lazybull.paper.broker import PaperBroker
from src.lazybull.paper.account import PaperAccount
from src.lazybull.paper.models import (
    Fill,
    Order,
    PendingBuy,
    PendingSell,
    TargetWeight,
    TradeInstruction,
    normalize_trade_reason,
)
from src.lazybull.paper.runner import PaperTradingRunner
from src.lazybull.signals.ml_signal import MLSignal


@pytest.fixture
def temp_storage_dir():
    """创建临时存储目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def storage(temp_storage_dir):
    """创建存储实例"""
    return PaperStorage(root_path=temp_storage_dir, verbose=False)


@pytest.fixture
def account(storage):
    """创建账户实例"""
    return PaperAccount(initial_capital=500000.0, storage=storage, verbose=False)


@pytest.fixture
def broker(account, storage):
    """创建经纪实例"""
    return PaperBroker(account=account, storage=storage, verbose=False)


def test_pending_buy_save_and_load(storage):
    """测试 PendingBuy 的保存和加载"""
    # 创建测试数据
    pending_buys = [
        PendingBuy(
            ts_code='000001.SZ',
            target_weight=0.1,
            reason='补位-涨停',
            create_date='20260121',
            attempts=1,
            last_attempt_date='20260122',
            original_signal_date='20260120'
        ),
        PendingBuy(
            ts_code='000002.SZ',
            target_weight=0.15,
            reason='补位-停牌',
            create_date='20260121',
            attempts=2,
            last_attempt_date='20260123',
            original_signal_date='20260120'
        ),
    ]
    
    # 保存
    storage.save_pending_buys(pending_buys)
    
    # 加载
    loaded = storage.load_pending_buys()
    
    # 验证
    assert len(loaded) == 2
    assert loaded[0].ts_code == '000001.SZ'
    assert loaded[0].target_weight == 0.1
    assert loaded[0].reason == '补位-涨停'
    assert loaded[0].attempts == 1
    assert loaded[0].last_attempt_date == '20260122'
    assert loaded[1].ts_code == '000002.SZ'
    assert loaded[1].attempts == 2


def test_pending_buy_empty_load(storage):
    """测试加载不存在的 PendingBuy 文件"""
    loaded = storage.load_pending_buys()
    assert loaded == []


def test_broker_tracks_failed_buy_targets(broker):
    """测试 broker 跟踪买入失败的目标（指令驱动路径）"""
    # 创建买入指令（000002.SZ 无价格，应该失败）
    instructions = [
        TradeInstruction(
            ts_code='000001.SZ', action='buy', shares=1000, price_type='close',
            reason='信号生成', source_date='20260120', target_weight=0.2,
        ),
        TradeInstruction(
            ts_code='000002.SZ', action='buy', shares=1000, price_type='close',
            reason='信号生成', source_date='20260120', target_weight=0.2,
        ),
    ]

    # Mock 价格和可交易性数据
    buy_prices = {'000001.SZ': 10.0}  # 000002.SZ 无价格，应该失败
    sell_prices = {}

    # Mock 可交易性信息
    with patch.object(broker, '_load_tradability_info') as mock_tradability:
        mock_tradability.return_value = {
            '000001.SZ': {
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'tradable': 1
            },
        }

        # 执行指令
        broker.execute_instructions(instructions, buy_prices, sell_prices, '20260121')

        # 获取失败目标
        failed_targets = broker.get_failed_buy_targets()

        # 验证：000002.SZ 应该因为无价格数据而失败
        assert len(failed_targets) == 1
        assert failed_targets[0].ts_code == '000002.SZ'
        assert '无价格数据' in failed_targets[0].reason


def test_broker_tracks_limit_up_as_failed(broker):
    """测试 broker 将涨停标记为买入失败（指令驱动路径）"""
    # 创建买入指令
    instructions = [
        TradeInstruction(
            ts_code='000001.SZ', action='buy', shares=1000, price_type='close',
            reason='信号生成', source_date='20260120', target_weight=0.2,
        ),
    ]

    # Mock 价格
    buy_prices = {'000001.SZ': 10.0}
    sell_prices = {}

    # Mock 可交易性信息（涨停）
    with patch.object(broker, '_load_tradability_info') as mock_tradability:
        mock_tradability.return_value = {
            '000001.SZ': {
                'is_suspended': 0,
                'is_limit_up': 1,  # 涨停
                'is_limit_down': 0,
                'tradable': 1
            },
        }

        # 执行指令
        broker.execute_instructions(instructions, buy_prices, sell_prices, '20260121')

        # 获取失败目标
        failed_targets = broker.get_failed_buy_targets()

        # 验证
        assert len(failed_targets) == 1
        assert failed_targets[0].ts_code == '000001.SZ'
        assert '涨停' in failed_targets[0].reason


def test_execute_instructions_does_not_open_new_slot_when_sell_fails(broker):
    """T1 卖出失败时，不应继续新开仓占用超额槽位。"""

    broker.account.add_position(
        ts_code='000001.SZ',
        shares=1000,
        buy_price=10.0,
        buy_cost=0.0,
        buy_date='20260120',
        buy_pnl_price=10.0,
    )

    instructions = [
        TradeInstruction(
            ts_code='000001.SZ',
            action='sell',
            shares=1000,
            price_type='close',
            reason='持有期到期',
            source_date='20260120',
            target_weight=0.0,
        ),
        TradeInstruction(
            ts_code='000002.SZ',
            action='buy',
            shares=1000,
            price_type='close',
            reason='信号生成',
            source_date='20260120',
            target_weight=0.2,
            original_signal_date='20260120',
            desired_position_count=1,
        ),
    ]

    buy_prices = {'000002.SZ': 10.0}
    sell_prices = {'000001.SZ': 10.0}

    with patch.object(broker, '_load_tradability_info') as mock_tradability:
        mock_tradability.return_value = {
            '000001.SZ': {
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 1,
                'tradable': 1,
            },
            '000002.SZ': {
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'tradable': 1,
            },
        }

        fills = broker.execute_instructions(
            instructions=instructions,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
            trade_date='20260121',
        )

    assert fills == []
    assert broker.account.get_position('000001.SZ') is not None
    assert broker.account.get_position('000002.SZ') is None
    failed_targets = broker.get_failed_buy_targets()
    assert len(failed_targets) == 1
    assert failed_targets[0].ts_code == '000002.SZ'
    assert '无可用空槽' in failed_targets[0].reason
    assert failed_targets[0].original_signal_date == '20260120'


def test_run_t0_stagger_uses_overall_top_n_as_desired_position_count():
    """分批调仓下，T0 生成买入指令应传递总 top_n 作为槽位上限。"""
    runner = PaperTradingRunner.__new__(PaperTradingRunner)

    runner.position_sizing = "equal"
    runner.kelly_vol_window = 60
    runner.kelly_max_leverage = 1.0
    runner.verbose = False
    runner.signal = MagicMock()
    runner.client = MagicMock()
    runner.storage = MagicMock()
    runner.loader = MagicMock()
    runner.paper_storage = MagicMock()
    runner.account = MagicMock()
    runner.feature_builder = MagicMock()
    runner.cleaner = MagicMock()
    runner.missing_factors = []

    runner._correct_trade_date = MagicMock(return_value="20260120")
    runner._check_rebalance_day = MagicMock(return_value=(True, 1))
    runner._get_next_trade_date = MagicMock(return_value="20260121")
    runner._build_rebalance_sell_instructions = MagicMock(return_value=[])
    runner._ensure_strategy_state = MagicMock(return_value={})
    runner._save_strategy_state = MagicMock()

    runner.paper_storage.check_run_exists.return_value = False
    runner.paper_storage.load_instructions.return_value = []
    runner.paper_storage.load_rebalance_state.return_value = {}

    runner.loader.load_clean_daily.return_value = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"],
            "close": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    runner.loader.load_clean_trade_cal.return_value = pd.DataFrame(
        {"cal_date": ["20260120"], "is_open": [1]}
    )
    runner.account.get_total_value.return_value = 500000.0

    targets = [
        TargetWeight(ts_code=f"00000{i}.SZ", target_weight=0.2, reason="信号生成")
        for i in range(1, 6)
    ]
    runner._generate_signals = MagicMock(return_value=targets)

    runner._generate_instructions = MagicMock(
        return_value=[
            TradeInstruction(
                ts_code="000001.SZ",
                action="buy",
                shares=100,
                price_type="close",
                reason="信号生成",
                source_date="20260120",
                target_weight=0.2,
            )
        ]
    )

    config = TradingConfig(
        buy_price="close",
        sell_price="close",
        universe="mainboard",
        top_n=20,
        rebalance_freq=20,
        stagger_tranches=4,
    )

    runner.run_t0(trade_date="20260120", trading_config=config)

    assert runner._generate_instructions.called
    assert runner._generate_instructions.call_args.kwargs["desired_position_count"] == 20


def test_generate_signals_respects_top_n_even_with_trading_config(monkeypatch):
    """_generate_signals 传入 trading_config 时也应遵循函数参数 top_n。"""
    runner = PaperTradingRunner.__new__(PaperTradingRunner)

    runner.position_sizing = "equal"
    runner.signal = MagicMock()
    runner.signal.generate_ranked = MagicMock()
    runner.storage = MagicMock()
    runner.loader = MagicMock()
    runner.feature_builder = MagicMock()
    runner.cleaner = MagicMock()
    runner.client = MagicMock()
    runner.account = MagicMock()
    runner._save_strategy_state = MagicMock()

    monkeypatch.setattr(
        "src.lazybull.paper.runner.signals.ensure_features_for_date",
        lambda *args, **kwargs: (True, [], ""),
    )

    runner.loader.load_clean_stock_basic.return_value = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "name": ["A", "B", "C"],
        }
    )
    runner.loader.load_clean_daily_by_date.return_value = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "close": [10.0, 11.0, 12.0],
        }
    )
    runner.storage.load_cs_train_day.return_value = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "f1": [1.0, 2.0, 3.0],
        }
    )
    runner.account.get_positions.return_value = {}

    universe = MagicMock()
    universe.get_stocks.return_value = ["000001.SZ", "000002.SZ", "000003.SZ"]
    runner._create_universe = MagicMock(return_value=universe)

    runner._generate_ranked_with_lot_constraint = MagicMock(
        return_value=({"000001.SZ": 0.9, "000002.SZ": 0.8}, {"target_n": 5})
    )
    runner._normalize_signals = MagicMock(return_value={"000001.SZ": 0.5, "000002.SZ": 0.5})
    runner._enhance_target_info = MagicMock(
        return_value=[
            TargetWeight(ts_code="000001.SZ", target_weight=0.5, reason="信号生成"),
            TargetWeight(ts_code="000002.SZ", target_weight=0.5, reason="信号生成"),
        ]
    )
    runner._print_t0_targets = MagicMock()

    config = TradingConfig(
        buy_price="close",
        sell_price="close",
        universe="mainboard",
        top_n=20,
        rebalance_freq=20,
        stagger_tranches=4,
    )

    targets = runner._generate_signals(
        trade_date="20260120",
        top_n=5,
        trading_config=config,
    )

    assert len(targets) == 2
    assert runner._generate_ranked_with_lot_constraint.called
    assert runner._generate_ranked_with_lot_constraint.call_args.args[4] == 5


def test_retry_pending_buys_success(broker):
    """测试成功重试 PendingBuy"""
    # 添加一个 pending buy
    pending_buy = PendingBuy(
        ts_code='000001.SZ',
        target_weight=0.2,
        reason='补位-涨停',
        create_date='20260121',
        attempts=1,
        last_attempt_date='20260122',
        original_signal_date='20260120'
    )
    broker.pending_buys = [pending_buy]
    
    # Mock 数据加载
    with patch('src.lazybull.data.DataLoader') as MockLoader, \
         patch('src.lazybull.data.Storage') as MockStorage:
        
        # Mock daily_data
        mock_loader = MockLoader.return_value
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'close': [10.0],
        })
        mock_loader.load_clean_daily_by_date.return_value = daily_data
        
        # Mock 可交易性（可以买入）
        with patch.object(broker, '_load_tradability_info') as mock_tradability:
            mock_tradability.return_value = {
                '000001.SZ': {
                    'is_suspended': 0,
                    'is_limit_up': 0,
                    'is_limit_down': 0,
                    'tradable': 1
                },
            }
            
            # 重试
            fills, remaining = broker.retry_pending_buys('20260123', 'close')
            
            # 验证：应该成功买入
            assert len(fills) == 1
            assert fills[0].ts_code == '000001.SZ'
            assert fills[0].action == 'buy'
            assert len(remaining) == 0


def test_retry_pending_buys_still_limit_up(broker):
    """测试重试时仍然涨停的情况"""
    # 添加一个 pending buy
    pending_buy = PendingBuy(
        ts_code='000001.SZ',
        target_weight=0.2,
        reason='补位-涨停',
        create_date='20260121',
        attempts=1,
        last_attempt_date='20260122',
        original_signal_date='20260120'
    )
    broker.pending_buys = [pending_buy]
    
    # Mock 数据加载
    with patch('src.lazybull.data.DataLoader') as MockLoader, \
         patch('src.lazybull.data.Storage') as MockStorage:
        
        # Mock daily_data
        mock_loader = MockLoader.return_value
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'close': [11.0],
        })
        mock_loader.load_clean_daily_by_date.return_value = daily_data
        
        # Mock 可交易性（仍然涨停）
        with patch.object(broker, '_load_tradability_info') as mock_tradability:
            mock_tradability.return_value = {
                '000001.SZ': {
                    'is_suspended': 0,
                    'is_limit_up': 1,  # 仍然涨停
                    'is_limit_down': 0,
                    'tradable': 1
                },
            }
            
            # 重试
            fills, remaining = broker.retry_pending_buys('20260123', 'close')
            
            # 验证：应该保留在队列中，attempts 增加
            assert len(fills) == 0
            assert len(remaining) == 1
            assert remaining[0].attempts == 2
            assert remaining[0].last_attempt_date == '20260123'


def test_retry_pending_buys_max_attempts_exceeded(broker):
    """测试超过最大尝试次数"""
    # 添加一个已经尝试 5 次的 pending buy
    pending_buy = PendingBuy(
        ts_code='000001.SZ',
        target_weight=0.2,
        reason='补位-涨停',
        create_date='20260121',
        attempts=5,
        last_attempt_date='20260127',
        original_signal_date='20260120'
    )
    broker.pending_buys = [pending_buy]
    
    # Mock 数据加载
    with patch('src.lazybull.data.DataLoader') as MockLoader, \
         patch('src.lazybull.data.Storage') as MockStorage:
        
        # Mock daily_data
        mock_loader = MockLoader.return_value
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'close': [12.0],
        })
        mock_loader.load_clean_daily_by_date.return_value = daily_data
        
        # Mock 可交易性（可以买入，但已过期）
        with patch.object(broker, '_load_tradability_info') as mock_tradability:
            mock_tradability.return_value = {
                '000001.SZ': {
                    'is_suspended': 0,
                    'is_limit_up': 0,
                    'is_limit_down': 0,
                    'tradable': 1
                },
            }
            
            # 重试（第6次，应该移除）
            fills, remaining = broker.retry_pending_buys('20260128', 'close', max_attempts=5)
            
            # 验证：应该被移除
            assert len(fills) == 0
            assert len(remaining) == 0


def test_retry_pending_buys_same_day_no_increment(broker):
    """测试同日重试不增加 attempts"""
    # 添加一个 pending buy（last_attempt_date 是今天）
    pending_buy = PendingBuy(
        ts_code='000001.SZ',
        target_weight=0.2,
        reason='补位-涨停',
        create_date='20260121',
        attempts=2,
        last_attempt_date='20260123',  # 今天
        original_signal_date='20260120'
    )
    broker.pending_buys = [pending_buy]
    
    # Mock 数据加载
    with patch('src.lazybull.data.DataLoader') as MockLoader, \
         patch('src.lazybull.data.Storage') as MockStorage:
        
        # Mock daily_data
        mock_loader = MockLoader.return_value
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'close': [11.0],
        })
        mock_loader.load_clean_daily_by_date.return_value = daily_data
        
        # Mock 可交易性（涨停）
        with patch.object(broker, '_load_tradability_info') as mock_tradability:
            mock_tradability.return_value = {
                '000001.SZ': {
                    'is_suspended': 0,
                    'is_limit_up': 1,
                    'is_limit_down': 0,
                    'tradable': 1
                },
            }
            
            # 重试（同一天）
            fills, remaining = broker.retry_pending_buys('20260123', 'close')
            
            # 验证：attempts 不应增加
            assert len(remaining) == 1
            assert remaining[0].attempts == 2  # 仍然是2，没有增加


def test_clear_failed_buy_targets(broker):
    """测试清空失败买入目标"""
    broker._failed_buy_targets = [
        TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='涨停'),
    ]
    
    broker.clear_failed_buy_targets()
    
    assert len(broker.get_failed_buy_targets()) == 0


def test_generate_replacement_targets_respects_failed_count_with_trading_config(monkeypatch):
    """测试传入完整 trading_config 时，补位数量仍受失败数限制。"""

    class DummySignal:
        def __init__(self):
            self.top_n = None
            self.model_version = None

        def update_model_version(self, new_version):
            self.model_version = new_version

        def generate_ranked(self, *args, **kwargs):
            return []

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.lazybull.paper.runner.TushareClient'):
            runner = PaperTradingRunner(
                initial_capital=500000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        runner.signal = DummySignal()

        stock_basic = pd.DataFrame(
            {
                'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ'],
                'name': ['测试1', '测试2', '测试3', '测试4'],
                'market': ['主板', '主板', '主板', '主板'],
                'list_date': ['20200101', '20200101', '20200101', '20200101'],
            }
        )
        daily_data = pd.DataFrame(
            {
                'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ'],
                'close': [10.0, 11.0, 12.0, 13.0],
            }
        )
        signal_data = pd.DataFrame(
            {
                'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ'],
                'feature_a': [1.0, 2.0, 3.0, 4.0],
            }
        )

        monkeypatch.setattr(
            'src.lazybull.paper.runner.replacement.ensure_features_for_date',
            lambda *args, **kwargs: (True, [], ""),
        )
        runner.loader.load_clean_stock_basic = Mock(return_value=stock_basic)
        runner.loader.load_clean_daily_by_date = Mock(return_value=daily_data)
        runner.storage.load_cs_train_day = Mock(return_value=signal_data)

        mock_universe = Mock()
        mock_universe.get_stocks.return_value = stock_basic['ts_code'].tolist()
        runner._create_universe = Mock(return_value=mock_universe)
        runner._print_replacement_targets = Mock()

        seen = {}

        def fake_generate_ranked(*args, **kwargs):
            seen['top_n'] = args[4]
            return (
                {
                    '000001.SZ': 0.9,
                    '000002.SZ': 0.8,
                    '000003.SZ': 0.7,
                    '000004.SZ': 0.6,
                },
                {},
            )

        runner._generate_ranked_with_lot_constraint = Mock(side_effect=fake_generate_ranked)

        trading_config = TradingConfig(
            top_n=20,
            model_version=12962,
            position_sizing='equal',
        )

        targets = runner.generate_replacement_targets(
            trade_date='20260325',
            failed_count=2,
            universe_type='mainboard',
            model_version=12962,
            buy_price_type='close',
            trading_config=trading_config,
        )

        assert seen['top_n'] == 2
        assert runner.signal.top_n == 2
        assert len(targets) == 2
        assert [target.ts_code for target in targets] == ['000001.SZ', '000002.SZ']


def test_execute_pending_buys_uses_limited_candidate_pool_by_previous_day(monkeypatch):
    """补位应基于上一交易日重算候选池，并按槽位逐个匹配买入。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.lazybull.paper.runner.TushareClient'):
            runner = PaperTradingRunner(
                initial_capital=500000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        class DummySignal:
            def generate_ranked(self, *args, **kwargs):
                # 故意返回包含已持仓和可买候选，验证过滤与逐槽位分配
                return [
                    ('000001.SZ', 0.99),  # 已持仓，应跳过
                    ('000002.SZ', 0.95),
                    ('000003.SZ', 0.90),
                    ('000004.SZ', 0.85),
                    ('000005.SZ', 0.80),
                ]

        runner.signal = DummySignal()

        # 预置一个已持仓，验证候选池会排除
        runner.account.add_position(
            ts_code='000001.SZ',
            shares=1000,
            buy_price=10.0,
            buy_cost=10.0,
            buy_date='20260120',
            buy_pnl_price=10.0,
        )

        # 两个槽位 -> 有限候选池大小应为 4
        pending_buys = [
            PendingBuy(
                ts_code='SLOT_A',
                target_weight=0.05,
                reason='补位-槽位A',
                create_date='20260121',
                attempts=1,
                last_attempt_date='',
                original_signal_date='20260120',
            ),
            PendingBuy(
                ts_code='SLOT_B',
                target_weight=0.05,
                reason='补位-槽位B',
                create_date='20260121',
                attempts=1,
                last_attempt_date='',
                original_signal_date='20260120',
            ),
        ]

        # 上一交易日与特征数据
        monkeypatch.setattr(runner, '_get_prev_trade_date', lambda _: '20260122')
        runner.storage.load_cs_train_day = Mock(
            return_value=pd.DataFrame({'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']})
        )
        runner.loader.load_clean_stock_basic = Mock(return_value=pd.DataFrame({'dummy': [1]}))
        mock_universe = Mock()
        mock_universe.get_stocks.return_value = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
        runner._create_universe = Mock(return_value=mock_universe)

        prev_daily = pd.DataFrame(
            {
                'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ'],
                'trade_date': ['20260122'] * 5,
            }
        )
        trade_daily = pd.DataFrame(
            {
                'ts_code': ['000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ'],
                'trade_date': ['20260123'] * 4,
                'is_suspended': [0, 0, 0, 0],
                'is_limit_up': [0, 0, 0, 0],
                'is_limit_down': [0, 0, 0, 0],
            }
        )

        def _mock_load_daily(date: str):
            if date == '20260122':
                return prev_daily
            if date == '20260123':
                return trade_daily
            return pd.DataFrame()

        runner.loader.load_clean_daily_by_date = Mock(side_effect=_mock_load_daily)

        # 当日可买价格
        buy_prices = {
            '000002.SZ': 10.0,
            '000003.SZ': 10.0,
            '000004.SZ': 10.0,
            '000005.SZ': 10.0,
        }

        # 可交易性全部放行
        runner.broker._load_tradability_info = Mock(return_value={})

        fills = runner._execute_pending_buys(
            pending_buys=pending_buys,
            buy_prices=buy_prices,
            trade_date='20260123',
            buy_price_type='close',
        )

        # 两个槽位应各补齐一只，不重复且来自重算候选池前部
        assert len(fills) == 2
        bought_codes = [f.ts_code for f in fills]
        assert bought_codes == ['000002.SZ', '000003.SZ']
        # 股数应按回测口径估算：current_total_value(约510000) * 0.05 / 10 -> 2500 股
        assert [f.shares for f in fills] == [2500, 2500]

        # 成功后补位队列应清空
        remaining = runner.paper_storage.load_pending_buys()
        assert len(remaining) == 0


def test_execute_pending_buys_prefers_persisted_t0_ranked_candidates(monkeypatch):
    """补位应优先复用 T0 持久化的 ranked_candidates 顺序。"""

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.lazybull.paper.runner.TushareClient'):
            runner = PaperTradingRunner(
                initial_capital=500000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        class DummySignal:
            def generate_ranked(self, *args, **kwargs):
                raise AssertionError('不应在已有 T0 候选时重算 ranked_candidates')

        runner.signal = DummySignal()
        runner.paper_storage.save_ranked_candidates(
            [('000003.SZ', 0.99), ('000002.SZ', 0.95), ('000004.SZ', 0.90)],
            '20260122',
        )

        pending_buys = [
            PendingBuy(
                ts_code='SLOT_A',
                target_weight=0.05,
                reason='补位-槽位A',
                create_date='20260122',
                attempts=0,
                last_attempt_date='',
                original_signal_date='20260122',
            )
        ]

        monkeypatch.setattr(runner, '_get_prev_trade_date', lambda _: '20260122')
        runner.loader.load_clean_daily_by_date = Mock(return_value=pd.DataFrame())

        with patch('src.lazybull.paper.runner.execution.is_tradeable', return_value=(True, '可交易')):
            fills = runner._execute_pending_buys(
                pending_buys=pending_buys,
                buy_prices={'000002.SZ': 10.0, '000003.SZ': 10.0, '000004.SZ': 10.0},
                trade_date='20260123',
                buy_price_type='close',
            )

        assert len(fills) == 1
        assert fills[0].ts_code == '000003.SZ'




def test_normalize_trade_reason_collapses_repeated_replenishment_tags():
    """测试补位原因归一化会折叠重复前后缀。"""
    reason = (
        '补位槽位-补位槽位-信号生成 (权重=0.0500)'
        '（无可用空槽）（无可用空槽）'
    )

    normalized = normalize_trade_reason(
        reason,
        ensure_replenishment_prefix=True,
        append_suffix='（无可用空槽）',
    )

    assert normalized == '补位槽位-信号生成 (权重=0.0500)（无可用空槽）'
def test_execute_pending_buys_skip_tiny_buy_value_by_ratio():
    """补位路径在现金缩量后仍应拦截过小市值买入。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.lazybull.paper.runner.TushareClient'):
            runner = PaperTradingRunner(
                initial_capital=1_000_000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        # 配置最小买入后市值阈值：总资产/top_n*ratio = 1_000_400/20*0.2 ≈ 10004
        runner.paper_storage.save_config({"top_n": 20, "min_buy_value_ratio": 0.2})

        # 构造“总资产高、现金低”的场景，触发缩量到一手后再走阈值拦截
        runner.account.add_position(
            ts_code='600000.SH',
            shares=100000,
            buy_price=10.0,
            buy_cost=0.0,
            buy_date='20260120',
            buy_pnl_price=10.0,
        )
        runner.account.state.cash = 400.0
        runner.account.save_state()

        pending_buys = [
            PendingBuy(
                ts_code='601916.SH',
                target_weight=0.05,
                reason='补位-信号生成',
                create_date='20260121',
                attempts=0,
                last_attempt_date='',
                original_signal_date='20260120',
            )
        ]

        buy_prices = {
            '600000.SH': 10.0,
            '601916.SH': 2.46,
        }
        runner.loader.load_clean_daily_by_date = Mock(return_value=pd.DataFrame())

        with patch('src.lazybull.paper.runner.execution.is_tradeable', return_value=(True, '可交易')):
            fills = runner._execute_pending_buys(
                pending_buys=pending_buys,
                buy_prices=buy_prices,
                trade_date='20260123',
                buy_price_type='close',
            )

        # 缩量后可买一手(100股，约246元)，但低于阈值，应被拦截并保留在补位队列
        assert fills == []
        assert runner.account.get_position('601916.SH') is None

        remaining = runner.paper_storage.load_pending_buys()
        assert len(remaining) == 1
        assert remaining[0].ts_code == '601916.SH'
        assert remaining[0].attempts == 1
        assert remaining[0].last_attempt_date == '20260123'
