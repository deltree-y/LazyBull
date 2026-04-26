"""测试数据确保和 T0 打印增强功能"""

import importlib
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.lazybull.data import DataCleaner, DataLoader, Storage, TushareClient
from src.lazybull.data.ensure import (
    ensure_basic_data,
    ensure_clean_data_for_date,
    ensure_raw_data_for_date,
)
import src.lazybull.features.ensure as ensure_module
from src.lazybull.features import FeatureBuilder, ensure_features_for_date
from src.lazybull.paper import PaperAccount, PaperStorage, TargetWeight
from src.lazybull.paper.runner import PaperTradingRunner


@pytest.fixture
def mock_client():
    """模拟 TushareClient"""
    client = Mock(spec=TushareClient)
    
    # 模拟交易日历
    trade_cal = pd.DataFrame({
        'exchange': ['SSE'] * 5,
        'cal_date': ['20250120', '20250121', '20250122', '20250123', '20250124'],
        'is_open': [1, 1, 1, 1, 1],
    })
    client.get_trade_cal.return_value = trade_cal
    
    # 模拟股票基本信息
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'name': ['测试股票1', '测试股票2'],
        'list_date': ['20200101', '20200101'],
        'market': ['主板', '主板'],
    })
    client.get_stock_basic.return_value = stock_basic
    
    # 模拟日线数据
    daily = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'trade_date': ['20250121', '20250121'],
        'open': [10.0, 20.0],
        'high': [11.0, 21.0],
        'low': [9.0, 19.0],
        'close': [10.5, 20.5],
        'vol': [1000000, 2000000],
        'amount': [10000000, 40000000],
        'pct_chg': [5.0, 2.5],
        'pre_close': [10.0, 20.0],
    })
    client.get_daily.return_value = daily
    
    # 模拟复权因子
    adj_factor = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'trade_date': ['20250121', '20250121'],
        'adj_factor': [1.0, 1.0],
    })
    client.get_adj_factor.return_value = adj_factor
    
    # 模拟停复牌和涨跌停（空数据）
    client.get_suspend_d.return_value = pd.DataFrame()
    client.get_stk_limit.return_value = pd.DataFrame()
    
    return client


@pytest.fixture
def temp_storage():
    """临时存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        yield storage


def test_ensure_raw_data_for_date(mock_client, temp_storage):
    """测试确保 raw 数据存在"""
    trade_date = '20250121'
    
    # 首次调用应该下载数据
    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date)
    assert result is True
    
    # 再次调用应该跳过下载（数据已存在）
    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is True


def test_ensure_basic_data(mock_client, temp_storage):
    """测试确保基础数据存在"""
    end_date = '20250121'
    
    # 首次调用应该下载数据
    result = ensure_basic_data(mock_client, temp_storage, end_date)
    assert result is True
    
    # 验证数据已保存
    trade_cal = temp_storage.load_raw("trade_cal")
    assert trade_cal is not None
    assert len(trade_cal) > 0
    
    stock_basic = temp_storage.load_raw("stock_basic")
    assert stock_basic is not None
    assert len(stock_basic) > 0


def test_ensure_clean_data_for_date(mock_client, temp_storage):
    """测试确保 clean 数据存在"""
    trade_date = '20250121'
    loader = DataLoader(temp_storage)
    cleaner = DataCleaner()
    
    # 确保基础数据存在
    ensure_basic_data(mock_client, temp_storage, trade_date)
    
    # 确保 clean 数据
    result = ensure_clean_data_for_date(
        temp_storage, loader, cleaner, mock_client, trade_date
    )
    assert result is True
    
    # 验证 clean 数据已保存
    assert temp_storage.is_data_exists("clean", "daily", trade_date)


def test_print_t0_targets():
    """测试 T0 打印信息"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 runner
        runner = PaperTradingRunner(
            initial_capital=500000.0,
            data_root=tmpdir,
            paper_root=tmpdir
        )
        
        # 创建测试数据
        targets = [
            TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='测试信号1'),
            TargetWeight(ts_code='000002.SZ', target_weight=0.3, reason='测试信号2'),
        ]
        
        stock_basic = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'name': ['测试股票1', '测试股票2'],
        })
        
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'close': [10.5, 20.5],
        })
        
        # 调用打印方法（不应抛出异常）
        try:
            runner._print_t0_targets(targets, stock_basic, daily_data)
            success = True
        except Exception as e:
            print(f"打印失败: {e}")
            success = False
        
        assert success is True


def test_enhance_target_info():
    """测试增强目标信息"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 runner
        runner = PaperTradingRunner(
            initial_capital=500000.0,
            data_root=tmpdir,
            paper_root=tmpdir
        )
        
        # 创建测试数据
        signal_dict = {
            '000001.SZ': 0.2,
            '000002.SZ': 0.3,
        }
        
        stock_basic = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'name': ['测试股票1', '测试股票2'],
        })
        
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'close': [10.5, 20.5],
        })
        
        # 调用增强方法
        targets = runner._enhance_target_info(
            signal_dict, stock_basic, daily_data, '20250121'
        )
        
        # 验证结果
        assert len(targets) == 2
        assert targets[0].ts_code == '000001.SZ'
        assert targets[0].target_weight == 0.2
        assert '权重=0.2000' in targets[0].reason


def test_generate_instructions_missing_capital_retention_ratio_uses_default(monkeypatch):
    """测试缺少 capital_retention_ratio 时仍可生成 T0 指令。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.lazybull.paper.runner.TushareClient'):
            runner = PaperTradingRunner(
                initial_capital=100000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        monkeypatch.setattr('src.lazybull.paper.runner.get_cost_settings', lambda: {})

        instructions = runner._generate_instructions(
            targets=[TargetWeight(ts_code='000001.SZ', target_weight=0.5, reason='测试信号')],
            buy_price_type='close',
            sell_price_type='open',
            current_prices={'000001.SZ': 10.0},
            source_date='20260325',
        )

        assert len(instructions) == 1
        assert instructions[0].action == 'buy'
        assert instructions[0].shares == 5000


def test_correct_trade_date_supports_next_with_last_trade_date(monkeypatch):
    """测试 next 会解析为上次执行日后的下一个交易日。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.lazybull.paper.runner.TushareClient'):
            runner = PaperTradingRunner(
                initial_capital=100000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        monkeypatch.setattr(
            runner.loader,
            'load_clean_trade_cal',
            lambda: pd.DataFrame(
                {
                    'cal_date': ['20260325', '20260326', '20260327'],
                    'is_open': [1, 1, 1],
                }
            ),
        )
        monkeypatch.setattr(runner.paper_storage, 'load_last_trade_date', lambda: '20260325')

        assert runner._correct_trade_date('next') == '20260326'


def test_load_factor_data_only_builds_trade_date_output(monkeypatch):
    """测试 _load_factor_data 只为目标交易日构建因子查询表输出。"""
    trade_date = '20260422'
    trading_dates = ['20260418', '20260421', trade_date]
    stub_df = pd.DataFrame({'ts_code': ['000001.SZ']})
    captured_dates = {}

    for attr in [
        '_MIN_FINA_RECORDS',
        '_MIN_HOLDER_RECORDS',
        '_MIN_FORECAST_RECORDS',
        '_MIN_EXPRESS_RECORDS',
        '_MIN_REPORT_RC_RECORDS',
    ]:
        monkeypatch.setattr(ensure_module, attr, 1)

    monkeypatch.setattr(
        ensure_module,
        '_try_ensure_historical_margin',
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        ensure_module,
        '_try_ensure_historical_cyq_perf',
        lambda *args, **kwargs: stub_df,
    )
    monkeypatch.setattr(
        ensure_module,
        '_try_ensure_historical_fund_portfolio',
        lambda *args, **kwargs: stub_df,
    )
    monkeypatch.setattr(
        ensure_module,
        '_try_ensure_historical_moneyflow_hsgt',
        lambda *args, **kwargs: stub_df,
    )
    monkeypatch.setattr(
        ensure_module,
        '_try_ensure_historical_top_list',
        lambda *args, **kwargs: stub_df,
    )

    def _record_builder(name):
        def _builder(_df, output_dates, *args, **kwargs):
            captured_dates[name] = list(output_dates)
            return {trade_date: stub_df}

        return _builder

    builder_targets = [
        ('src.lazybull.factors.fundamental', 'build_fundamental_lookup_by_date', 'fundamental'),
        ('src.lazybull.factors.margin', 'build_margin_lookup_by_date', 'margin'),
        ('src.lazybull.factors.holder', 'build_holder_lookup_by_date', 'holder'),
        ('src.lazybull.factors.earnings', 'build_earnings_lookup_by_date', 'earnings'),
        ('src.lazybull.factors.cyq_perf', 'build_cyq_perf_lookup_by_date', 'cyq_perf'),
        ('src.lazybull.factors.express', 'build_express_lookup_by_date', 'express'),
        ('src.lazybull.factors.fund_portfolio', 'build_fund_portfolio_lookup_by_date', 'fund_portfolio'),
        ('src.lazybull.factors.north_flow', 'build_north_flow_lookup_by_date', 'north_flow'),
        ('src.lazybull.factors.lhb', 'build_lhb_lookup_by_date', 'lhb'),
        ('src.lazybull.factors.consensus', 'build_consensus_lookup_by_date', 'consensus'),
    ]
    expected_names = {item[2] for item in builder_targets}

    for module_path, attr_name, name in builder_targets:
        monkeypatch.setattr(
            importlib.import_module(module_path),
            attr_name,
            _record_builder(name),
        )

    class StubLoader:
        def load_fina_indicator(self):
            return stub_df

        def load_margin_detail(self, start_date, end_date):
            return stub_df

        def load_stk_holdernumber(self):
            return stub_df

        def load_forecast(self):
            return stub_df

        def load_express(self):
            return stub_df

        def load_report_rc(self):
            return stub_df

    result = ensure_module._load_factor_data(
        loader=StubLoader(),
        client=Mock(),
        storage=Mock(),
        trade_date=trade_date,
        trading_dates_str=trading_dates,
        start_date='20260401',
        end_date=trade_date,
    )

    assert set(captured_dates.keys()) == expected_names
    assert result[-1] == []
    for output_dates in captured_dates.values():
        assert output_dates == [trade_date]


def test_try_ensure_historical_fund_portfolio_builds_and_reuses_agg_cache(
    monkeypatch, temp_storage
):
    """测试基金持仓历史补齐会写入并复用季度聚合缓存。"""
    period = "20260331"
    raw_df = pd.DataFrame(
        [
            {
                "ts_code": "000001.OF",
                "symbol": "000001",
                "ann_date": "20260420",
                "end_date": period,
                "stk_float_ratio": 1.2,
                "mkv": 1000,
                "amount": 100,
            },
            {
                "ts_code": "000002.OF",
                "symbol": "000001",
                "ann_date": "20260421",
                "end_date": period,
                "stk_float_ratio": 0.8,
                "mkv": 900,
                "amount": 90,
            },
        ]
    )
    temp_storage.save_raw_by_date(raw_df, "fund_portfolio", period)

    monkeypatch.setattr(ensure_module, "_generate_quarter_periods", lambda *_args: [period])
    monkeypatch.setattr(ensure_module, "_query_with_pagination", lambda *args, **kwargs: pd.DataFrame())

    result = ensure_module._try_ensure_historical_fund_portfolio(
        client=Mock(),
        storage=temp_storage,
        trading_dates_str=["20260422"],
    )

    assert result is not None
    assert temp_storage.is_data_exists("raw", "fund_portfolio_agg", period)

    original_load = temp_storage.load_raw_by_date

    def guarded_load(name, trade_date, format="parquet", columns=None):
        if name == "fund_portfolio":
            raise AssertionError("存在聚合缓存时不应再回读原始季度明细")
        return original_load(name, trade_date, format=format, columns=columns)

    monkeypatch.setattr(temp_storage, "load_raw_by_date", guarded_load)

    cached = ensure_module._try_ensure_historical_fund_portfolio(
        client=Mock(),
        storage=temp_storage,
        trading_dates_str=["20260422"],
    )

    assert cached is not None
    assert len(cached) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
