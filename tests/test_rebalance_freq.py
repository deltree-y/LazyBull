"""测试自定义调仓频率功能"""

import tempfile

import pandas as pd
import pytest

from src.lazybull.backtest import BacktestEngine
from src.lazybull.common.cost import CostModel
from src.lazybull.common.trading_config import TradingConfig
from src.lazybull.paper.runner import PaperTradingRunner
from src.lazybull.signals.base import Signal
from src.lazybull.universe.base import Universe


class MockUniverse(Universe):
    """模拟股票池"""
    
    def __init__(self, stocks):
        self.stocks = stocks
    
    def get_stocks(self, date):
        return self.stocks


class MockSignal(Signal):
    """模拟信号生成器"""
    
    def generate(self, date, universe, data):
        # 等权分配
        n = len(universe)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {stock: weight for stock in universe}


@pytest.fixture
def mock_trading_dates():
    """创建模拟交易日列表"""
    dates = pd.date_range('2023-01-01', periods=50, freq='B')  # 50个交易日
    return [pd.Timestamp(d) for d in dates]


@pytest.fixture
def mock_price_data():
    """创建模拟价格数据"""
    dates = pd.date_range('2023-01-01', periods=50, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    
    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': 10.0,
                'close_adj': 10.0
            })
    
    return pd.DataFrame(data)


def test_rebalance_freq_integer(mock_trading_dates, mock_price_data):
    """测试整数调仓频率（每5个交易日调仓一次）"""
    universe = MockUniverse(['000001.SZ', '000002.SZ'])
    signal = MockSignal()
    
    # 使用整数调仓频率
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        cost_model=CostModel(),
        rebalance_freq=5,  # 每5个交易日调仓一次
        verbose=False
    )
    
    # 验证调仓日期数量（返回 Dict[date, tranche_idx]）
    rebalance_dict = engine._get_rebalance_dates(mock_trading_dates)
    rebalance_dates = sorted(rebalance_dict.keys())
    expected_count = (len(mock_trading_dates) + 4) // 5  # 向上取整
    assert len(rebalance_dates) == expected_count

    # 验证第一个调仓日是第一个交易日
    assert rebalance_dates[0] == mock_trading_dates[0]

    # 验证调仓间隔
    for i in range(1, len(rebalance_dates)):
        # 找到两个调仓日在原列表中的索引
        idx1 = mock_trading_dates.index(rebalance_dates[i-1])
        idx2 = mock_trading_dates.index(rebalance_dates[i])
        # 间隔应该是5个交易日（或最后一次小于5）
        assert idx2 - idx1 == 5 or i == len(rebalance_dates) - 1


def test_rebalance_freq_1day(mock_trading_dates):
    """测试每1个交易日调仓（相当于日频）"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq=1,  # 每1个交易日调仓
        verbose=False
    )
    
    rebalance_dates = engine._get_rebalance_dates(mock_trading_dates)
    assert len(rebalance_dates) == len(mock_trading_dates)


def test_rebalance_freq_5days(mock_trading_dates):
    """测试每5个交易日调仓（相当于周频）"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq=5,  # 每5个交易日调仓
        verbose=False
    )
    
    rebalance_dates = engine._get_rebalance_dates(mock_trading_dates)
    # 50个交易日，每5天调仓一次，应该有10次调仓
    assert len(rebalance_dates) == 10


def test_rebalance_freq_20days(mock_trading_dates):
    """测试每20个交易日调仓（相当于月频）"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq=20,  # 每20个交易日调仓
        verbose=False
    )
    
    rebalance_dates = engine._get_rebalance_dates(mock_trading_dates)
    # 50个交易日，每20天调仓一次，应该有3次调仓 (0, 20, 40)
    assert len(rebalance_dates) == 3


def test_stagger_tranches_are_evenly_distributed_for_non_divisible_period(
    mock_trading_dates,
):
    """20日分3批时应使用0/7/13偏移，避免尾部形成8日空档。"""
    engine = BacktestEngine(
        universe=MockUniverse(['000001.SZ']),
        signal=MockSignal(),
        rebalance_freq=20,
        stagger_tranches=3,
        verbose=False,
    )

    schedule = engine._get_rebalance_dates(mock_trading_dates)
    scheduled_indices = [
        mock_trading_dates.index(date) for date in sorted(schedule.keys())
    ]
    tranche_indices = [schedule[date] for date in sorted(schedule.keys())]

    assert scheduled_indices[:6] == [0, 7, 13, 20, 27, 33]
    assert tranche_indices[:6] == [0, 1, 2, 0, 1, 2]
    assert [
        current - previous
        for previous, current in zip(scheduled_indices[:5], scheduled_indices[1:6])
    ] == [7, 6, 7, 7, 6]


def test_trading_config_rejects_invalid_stagger_boundaries():
    """统一配置应在运行前拒绝会产生空批次或排期覆盖的参数。"""
    with pytest.raises(ValueError, match="不能超过 top_n"):
        TradingConfig(top_n=2, rebalance_freq=20, stagger_tranches=3)
    with pytest.raises(ValueError, match="不能超过 rebalance_freq"):
        TradingConfig(top_n=20, rebalance_freq=2, stagger_tranches=3)


def test_trading_config_rejects_infeasible_stock_weight_cap():
    """单股上限乘目标持仓数不足 100% 时应明确失败。"""
    with pytest.raises(ValueError, match="无法构成满仓组合"):
        TradingConfig(top_n=5, max_weight_per_stock=0.15)


def test_backtest_rejects_stagger_tranches_above_frequency():
    with pytest.raises(ValueError, match="不能超过调仓频率"):
        BacktestEngine(
            universe=MockUniverse(['000001.SZ']),
            signal=MockSignal(),
            rebalance_freq=2,
            stagger_tranches=3,
            verbose=False,
        )


def test_paper_stagger_catches_up_earliest_missed_tranche():
    """纸面交易漏跑计划日后应先补最早未履行批次。"""
    runner = PaperTradingRunner.__new__(PaperTradingRunner)
    dates = [date.strftime('%Y%m%d') for date in pd.bdate_range('2026-01-05', periods=40)]
    runner._get_open_trade_dates = lambda: dates
    state = {
        'last_rebalance_date': dates[0],
        'last_scheduled_rebalance_date': dates[0],
        'tranche_anchor_date': dates[0],
        'rebalance_freq': 20,
        'stagger_tranches': 3,
    }

    assert runner._check_staggered_rebalance_day(dates[8], 20, 3, state) == (True, 1)
    assert runner._resolved_rebalance_plan_date == dates[7]


def test_paper_stagger_resets_cycle_when_configuration_changes():
    """K 或调仓周期变化时应从当前成功 T0 重建周期。"""
    runner = PaperTradingRunner.__new__(PaperTradingRunner)
    runner.paper_storage = type(
        'Storage',
        (),
        {
            'load_rebalance_state': lambda self: {
                'last_rebalance_date': '20260105',
                'rebalance_freq': 20,
                'stagger_tranches': 1,
            }
        },
    )()

    assert runner._check_rebalance_day('20260112', 20, stagger_tranches=2) == (True, 0)
    assert runner._resolved_rebalance_plan_date == '20260112'


def test_paper_stagger_recovers_from_invalid_legacy_state_config():
    """旧状态中的空 freq/K 不应阻断当前交易日重建周期。"""
    runner = PaperTradingRunner.__new__(PaperTradingRunner)
    runner.paper_storage = type(
        'Storage',
        (),
        {
            'load_rebalance_state': lambda self: {
                'last_rebalance_date': '20260105',
                'rebalance_freq': '',
                'stagger_tranches': None,
            }
        },
    )()

    assert runner._check_rebalance_day('20260112', 20, stagger_tranches=2) == (True, 0)
    assert runner._resolved_rebalance_plan_date == '20260112'


def test_rebalance_freq_invalid_integer():
    """测试无效的整数调仓频率（负数或零）"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    with pytest.raises(ValueError, match="调仓频率必须为正整数"):
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            rebalance_freq=0,
            verbose=False
        )
    
    with pytest.raises(ValueError, match="调仓频率必须为正整数"):
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            rebalance_freq=-5,
            verbose=False
        )


def test_rebalance_freq_invalid_type():
    """测试无效的调仓频率类型（字符串）"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    with pytest.raises(TypeError, match="调仓频率必须为整数类型"):
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            rebalance_freq="20",
            verbose=False
        )

    with pytest.raises(TypeError, match="调仓频率必须为整数类型"):
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            rebalance_freq=[5],
            verbose=False
        )


def test_rebalance_freq_float():
    """测试浮点数调仓频率应抛出类型错误"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    with pytest.raises(TypeError, match="调仓频率必须为整数类型"):
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            rebalance_freq=5.5,
            verbose=False
        )


def test_holding_period_auto_integer():
    """测试整数调仓频率时自动设置持有期"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq=7,  # 每7天调仓
        verbose=False
    )
    
    # 持有期应该等于调仓频率
    assert engine.holding_period == 7


def test_holding_period_manual_override():
    """测试手动设置持有期覆盖自动计算"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq=5,
        holding_period=10,  # 手动设置持有期
        verbose=False
    )
    
    # 持有期应该使用手动设置的值
    assert engine.holding_period == 10
