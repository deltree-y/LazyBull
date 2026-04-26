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
from src.lazybull.paper.models import Fill, Order, PendingBuy, PendingSell, TargetWeight
from src.lazybull.paper.runner import PaperTradingRunner
from src.lazybull.signals.ml_signal import MLSignal, SignalConfidenceGateState


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
    """测试 broker 跟踪买入失败的目标"""
    # 创建目标
    targets = [
        TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='信号生成'),
        TargetWeight(ts_code='000002.SZ', target_weight=0.2, reason='信号生成'),
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
        
        # 生成订单
        orders = broker.generate_orders(targets, buy_prices, sell_prices, '20260121')
        
        # 获取失败目标
        failed_targets = broker.get_failed_buy_targets()
        
        # 验证：000002.SZ 应该因为无价格数据而失败
        assert len(failed_targets) == 1
        assert failed_targets[0].ts_code == '000002.SZ'
        assert '无价格数据' in failed_targets[0].reason


def test_broker_tracks_limit_up_as_failed(broker):
    """测试 broker 将涨停标记为买入失败"""
    # 创建目标
    targets = [
        TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='信号生成'),
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
        
        # 生成订单
        orders = broker.generate_orders(targets, buy_prices, sell_prices, '20260121')
        
        # 获取失败目标
        failed_targets = broker.get_failed_buy_targets()
        
        # 验证
        assert len(failed_targets) == 1
        assert failed_targets[0].ts_code == '000001.SZ'
        assert '涨停' in failed_targets[0].reason


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
            'src.lazybull.paper.runner.ensure_features_for_date',
            lambda *args, **kwargs: (True, []),
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


def test_generate_signals_preserves_gate_exposure_with_weight_cap(monkeypatch):
    """测试存在单股上限时，门控降仓不会被后续归一化抹掉。"""

    class DummySignal:
        def __init__(self):
            self.top_n = None

        def generate_ranked(self, *args, **kwargs):
            return []

        def apply_confidence_gate_to_weights(self, signals, confidence_state=None, **kwargs):
            exposure = getattr(confidence_state, 'exposure', 1.0)
            return {stock: weight * exposure for stock, weight in signals.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.lazybull.paper.runner.TushareClient'):
            runner = PaperTradingRunner(
                initial_capital=500000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        runner.signal = DummySignal()

        stock_codes = [f'{i:06d}.SZ' for i in range(10)]
        stock_basic = pd.DataFrame(
            {
                'ts_code': stock_codes,
                'name': [f'测试{i}' for i in range(10)],
                'market': ['主板'] * 10,
                'list_date': ['20200101'] * 10,
            }
        )
        daily_data = pd.DataFrame(
            {
                'ts_code': stock_codes,
                'close': [10.0 + i for i in range(10)],
            }
        )
        signal_data = pd.DataFrame(
            {
                'ts_code': stock_codes,
                'feature_a': [float(i) for i in range(10)],
            }
        )

        monkeypatch.setattr(
            'src.lazybull.paper.runner.ensure_features_for_date',
            lambda *args, **kwargs: (True, []),
        )
        runner.loader.load_clean_stock_basic = Mock(return_value=stock_basic)
        runner.loader.load_clean_daily_by_date = Mock(return_value=daily_data)
        runner.storage.load_cs_train_day = Mock(return_value=signal_data)

        mock_universe = Mock()
        mock_universe.get_stocks.return_value = stock_codes
        runner._create_universe = Mock(return_value=mock_universe)
        gate_state = SignalConfidenceGateState(enabled=True, exposure=0.5, reason='测试半仓')
        runner._generate_ranked_with_lot_constraint = Mock(
            return_value=({code: 1.0 for code in stock_codes}, {'confidence_gate_state': gate_state})
        )
        runner._print_t0_targets = Mock()

        targets = runner._generate_signals(
            trade_date='20260324',
            top_n=10,
            buy_price_type='close',
            max_weight_per_stock=0.15,
        )

        assert len(targets) == 10
        assert abs(sum(target.target_weight for target in targets) - 0.5) < 1e-9
        assert all(abs(target.target_weight - 0.05) < 1e-9 for target in targets)


def test_confidence_gate_full_pass_emits_log_when_emit_log_true():
    """测试门控满仓通过时，paper 路径也会打印门控结果。"""
    signal = MLSignal(top_n=5, signal_gate_mode='composite', verbose=False)
    state = SignalConfidenceGateState(enabled=True, exposure=1.0, reason='测试满仓通过')

    with patch('src.lazybull.signals.ml_signal.logger.info') as mock_info:
        result = signal.apply_confidence_gate_to_weights(
            {'000001.SZ': 0.2, '000002.SZ': 0.3},
            confidence_state=state,
            date='20260324',
            emit_log=True,
        )

    assert result == {'000001.SZ': 0.2, '000002.SZ': 0.3}
    mock_info.assert_called_once_with('信号置信度门控: 20260324, 测试满仓通过，满仓通过')
