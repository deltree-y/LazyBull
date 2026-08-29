"""测试纸面交易T0等权策略一手可买约束功能"""

import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.lazybull.common.trading_config import TradingConfig
from src.lazybull.paper import PaperAccount, PaperStorage, TargetWeight
from src.lazybull.paper.runner import PaperTradingRunner
from src.lazybull.signals.ml_signal import MLSignal


@pytest.fixture
def mock_runner():
    """创建模拟的PaperTradingRunner实例"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock TushareClient to avoid token requirement
        with patch('src.lazybull.paper.runner.TushareClient'):
            runner = PaperTradingRunner(
                initial_capital=100000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False
            )
            yield runner


def test_equal_weight_lot_constraint_basic(mock_runner):
    """测试等权策略一手可买约束基本功能"""
    # 模拟数据
    date = pd.Timestamp('20260201')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '600000.SH', '600001.SH']
    top_n = 3
    buy_price_type = 'close'
    
    # 模拟特征数据
    signal_data = pd.DataFrame({
        'ts_code': stocks,
        'feature1': [1, 2, 3, 4, 5]
    })
    
    # 模拟日线数据：000001价格很高(1200元)，按100000/3=33333.33元分配，只能买27股（不足1手）
    # 000002价格50元，可买666股（6手）
    # 000003价格30元，可买1111股（11手）
    # 600000价格25元，可买1333股（13手）
    # 600001价格20元，可买1666股（16手）
    daily_data = pd.DataFrame({
        'ts_code': stocks,
        'close': [1200.0, 50.0, 30.0, 25.0, 20.0],
        'open': [1190.0, 49.0, 29.0, 24.5, 19.5]
    })
    
    # 模拟 MLSignal.generate_ranked 返回排序候选（按照价格从高到低排序作为示例）
    ranked_candidates = [
        ('000001.SZ', 0.9),
        ('000002.SZ', 0.8),
        ('000003.SZ', 0.7),
        ('600000.SH', 0.6),
        ('600001.SH', 0.5)
    ]
    
    # 设置mock signal
    mock_runner.signal = MagicMock(spec=MLSignal)
    mock_runner.signal.generate_ranked.return_value = ranked_candidates
    
    # 调用测试方法
    result = mock_runner._generate_ranked_with_lot_constraint(
        date, stocks, signal_data, daily_data, top_n, buy_price_type
    )
    
    # 验证结果
    # 000001.SZ应该被跳过（价格太高），选择后续的3只
    assert '000001.SZ' not in result
    assert '000002.SZ' in result
    assert '000003.SZ' in result
    assert '600000.SH' in result

    # 验证返回原始分数（与 ranked_candidates 一致）
    assert len(result) == 3
    assert abs(result['000002.SZ'] - 0.8) < 1e-6
    assert abs(result['000003.SZ'] - 0.7) < 1e-6
    assert abs(result['600000.SH'] - 0.6) < 1e-6


def test_equal_weight_lot_constraint_insufficient_candidates(mock_runner):
    """测试候选股票不足top_n的情况"""
    date = pd.Timestamp('20260201')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
    top_n = 5  # 要求5只，但候选只有3只
    buy_price_type = 'close'
    
    signal_data = pd.DataFrame({
        'ts_code': stocks,
        'feature1': [1, 2, 3]
    })
    
    # 所有股票价格都合理
    daily_data = pd.DataFrame({
        'ts_code': stocks,
        'close': [50.0, 30.0, 25.0],
        'open': [49.0, 29.0, 24.5]
    })
    
    ranked_candidates = [
        ('000001.SZ', 0.9),
        ('000002.SZ', 0.8),
        ('000003.SZ', 0.7)
    ]
    
    mock_runner.signal = MagicMock(spec=MLSignal)
    mock_runner.signal.generate_ranked.return_value = ranked_candidates
    
    result = mock_runner._generate_ranked_with_lot_constraint(
        date, stocks, signal_data, daily_data, top_n, buy_price_type
    )
    
    # 应该返回所有3只可买股票，原始分数为正数
    assert len(result) == 3
    for score in result.values():
        assert score > 0


def test_equal_weight_lot_constraint_all_too_expensive(mock_runner):
    """测试所有股票都太贵无法买1手的情况"""
    date = pd.Timestamp('20260201')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
    top_n = 3
    buy_price_type = 'close'
    
    signal_data = pd.DataFrame({
        'ts_code': stocks,
        'feature1': [1, 2, 3]
    })
    
    # 所有股票价格都非常高（100000/3=33333，price>333都买不到100股）
    daily_data = pd.DataFrame({
        'ts_code': stocks,
        'close': [5000.0, 4000.0, 3500.0],
        'open': [4900.0, 3900.0, 3400.0]
    })
    
    ranked_candidates = [
        ('000001.SZ', 0.9),
        ('000002.SZ', 0.8),
        ('000003.SZ', 0.7)
    ]
    
    mock_runner.signal = MagicMock(spec=MLSignal)
    mock_runner.signal.generate_ranked.return_value = ranked_candidates
    
    result = mock_runner._generate_ranked_with_lot_constraint(
        date, stocks, signal_data, daily_data, top_n, buy_price_type
    )
    
    # 应该返回空字典
    assert len(result) == 0


def test_equal_weight_lot_constraint_with_open_price(mock_runner):
    """测试使用open价格类型的情况"""
    date = pd.Timestamp('20260201')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
    top_n = 2
    buy_price_type = 'open'  # 使用开盘价
    
    signal_data = pd.DataFrame({
        'ts_code': stocks,
        'feature1': [1, 2, 3]
    })
    
    # open价格：000001太贵(1200)，000002和000003可以买
    daily_data = pd.DataFrame({
        'ts_code': stocks,
        'close': [1000.0, 50.0, 30.0],  # close不应该被使用
        'open': [1200.0, 48.0, 28.0]    # 使用open
    })
    
    ranked_candidates = [
        ('000001.SZ', 0.9),
        ('000002.SZ', 0.8),
        ('000003.SZ', 0.7)
    ]
    
    mock_runner.signal = MagicMock(spec=MLSignal)
    mock_runner.signal.generate_ranked.return_value = ranked_candidates
    
    result = mock_runner._generate_ranked_with_lot_constraint(
        date, stocks, signal_data, daily_data, top_n, buy_price_type
    )
    
    # 应该跳过000001，选择000002和000003
    assert '000001.SZ' not in result
    assert '000002.SZ' in result
    assert '000003.SZ' in result
    assert len(result) == 2


def test_equal_weight_lot_constraint_missing_price_data(mock_runner):
    """测试缺失价格数据的情况"""
    date = pd.Timestamp('20260201')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ']
    top_n = 3
    buy_price_type = 'close'
    
    signal_data = pd.DataFrame({
        'ts_code': stocks,
        'feature1': [1, 2, 3, 4]
    })
    
    # 000002缺失价格数据
    daily_data = pd.DataFrame({
        'ts_code': ['000001.SZ', '000003.SZ', '000004.SZ'],
        'close': [50.0, 30.0, 25.0],
        'open': [49.0, 29.0, 24.5]
    })
    
    ranked_candidates = [
        ('000001.SZ', 0.9),
        ('000002.SZ', 0.8),  # 缺失价格，应该被跳过
        ('000003.SZ', 0.7),
        ('000004.SZ', 0.6)
    ]
    
    mock_runner.signal = MagicMock(spec=MLSignal)
    mock_runner.signal.generate_ranked.return_value = ranked_candidates
    
    result = mock_runner._generate_ranked_with_lot_constraint(
        date, stocks, signal_data, daily_data, top_n, buy_price_type
    )
    
    # 000002应该被跳过，选择其他3只
    assert '000002.SZ' not in result
    assert '000001.SZ' in result
    assert '000003.SZ' in result
    assert '000004.SZ' in result
    assert len(result) == 3


def test_equal_weight_lot_constraint_boundary_case(mock_runner):
    """测试边界情况：刚好够买100股"""
    date = pd.Timestamp('20260201')
    stocks = ['000001.SZ', '000002.SZ']
    top_n = 2
    buy_price_type = 'close'
    
    signal_data = pd.DataFrame({
        'ts_code': stocks,
        'feature1': [1, 2]
    })
    
    # 000001: 100000/2=50000, price=500, 可买100股（刚好1手）
    # 000002: 100000/2=50000, price=501, 可买99股（不足1手）
    daily_data = pd.DataFrame({
        'ts_code': stocks,
        'close': [500.0, 501.0],
        'open': [498.0, 499.0]
    })
    
    ranked_candidates = [
        ('000001.SZ', 0.9),
        ('000002.SZ', 0.8)
    ]
    
    mock_runner.signal = MagicMock(spec=MLSignal)
    mock_runner.signal.generate_ranked.return_value = ranked_candidates
    
    result = mock_runner._generate_ranked_with_lot_constraint(
        date, stocks, signal_data, daily_data, top_n, buy_price_type
    )
    
    # 000001刚好够，000002不够
    assert '000001.SZ' in result
    assert '000002.SZ' not in result
    assert len(result) == 1


def test_stagger_lot_constraint_uses_portfolio_slot_budget(mock_runner):
    """分批预筛必须使用总组合单槽预算，避免放行最终买不到一手的股票。"""
    date = pd.Timestamp('20260201')
    stocks = ['000001.SZ', '000002.SZ']
    daily_data = pd.DataFrame(
        {
            'ts_code': stocks,
            'close': [150.0, 20.0],
            'open': [150.0, 20.0],
        }
    )
    mock_runner.signal = MagicMock(spec=MLSignal)
    mock_runner.signal.generate_ranked.return_value = [
        ('000001.SZ', 0.9),
        ('000002.SZ', 0.8),
    ]

    result = mock_runner._generate_ranked_with_lot_constraint(
        date,
        stocks,
        pd.DataFrame({'ts_code': stocks, 'feature1': [1, 2]}),
        daily_data,
        top_n=1,
        buy_price_type='close',
        trading_config=TradingConfig(top_n=20, stagger_tranches=4),
    )

    assert result == {'000002.SZ': 0.8}


def test_lot_constraint_applies_capital_retention_ratio(mock_runner):
    """一手预筛应使用扣除现金保留后的真实单槽预算。"""
    date = pd.Timestamp('20260201')
    stocks = ['000001.SZ', '000002.SZ']
    daily_data = pd.DataFrame(
        {
            'ts_code': stocks,
            'close': [45.0, 40.0],
            'open': [45.0, 40.0],
        }
    )
    mock_runner.signal = MagicMock(spec=MLSignal)
    mock_runner.signal.generate_ranked.return_value = [
        ('000001.SZ', 0.9),
        ('000002.SZ', 0.8),
    ]
    mock_runner._get_cost_setting = MagicMock(return_value=0.2)

    result = mock_runner._generate_ranked_with_lot_constraint(
        date,
        stocks,
        pd.DataFrame({'ts_code': stocks, 'feature1': [1, 2]}),
        daily_data,
        top_n=1,
        buy_price_type='close',
        trading_config=TradingConfig(top_n=20, stagger_tranches=4),
    )

    assert result == {'000002.SZ': 0.8}


def test_equal_weight_lot_constraint_excludes_existing_positions(
    mock_runner,
):
    """已持仓股票应被完全排除，不生成补差买单目标。"""
    date = pd.Timestamp('20260201')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ']
    top_n = 3
    buy_price_type = 'close'

    signal_data = pd.DataFrame(
        {
            'ts_code': stocks,
            'feature1': [1, 2, 3, 4],
        }
    )

    daily_data = pd.DataFrame(
        {
            'ts_code': stocks,
            'close': [10.0, 10.0, 10.0, 10.0],
            'open': [10.0, 10.0, 10.0, 10.0],
        }
    )

    # 排名前两只恰好是已有持仓
    ranked_candidates = [
        ('000001.SZ', 0.95),
        ('000002.SZ', 0.90),
        ('000003.SZ', 0.85),
        ('000004.SZ', 0.80),
    ]

    mock_runner.signal = MagicMock(spec=MLSignal)
    mock_runner.signal.generate_ranked.return_value = ranked_candidates

    result = mock_runner._generate_ranked_with_lot_constraint(
        date,
        stocks,
        signal_data,
        daily_data,
        top_n,
        buy_price_type,
        existing_positions={'000001.SZ', '000002.SZ'},
    )

    # 已持仓必须被排除，目标仅来自非持仓股票
    assert '000001.SZ' not in result
    assert '000002.SZ' not in result
    assert '000003.SZ' in result
    assert '000004.SZ' in result
    assert len(result) == 2
