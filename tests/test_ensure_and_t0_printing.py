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


def test_generate_instructions_keeps_target_order_for_buys(monkeypatch):
    """测试 _generate_instructions 生成买单顺序与 targets 顺序一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.lazybull.paper.runner.TushareClient'):
            runner = PaperTradingRunner(
                initial_capital=100000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        monkeypatch.setattr('src.lazybull.paper.runner.get_cost_settings', lambda: {})

        targets = [
            TargetWeight(ts_code='000003.SZ', target_weight=0.34, reason='r3'),
            TargetWeight(ts_code='000001.SZ', target_weight=0.33, reason='r1'),
            TargetWeight(ts_code='000002.SZ', target_weight=0.33, reason='r2'),
        ]
        current_prices = {
            '000001.SZ': 10.0,
            '000002.SZ': 10.0,
            '000003.SZ': 10.0,
        }

        instructions = runner._generate_instructions(
            targets=targets,
            buy_price_type='close',
            sell_price_type='open',
            current_prices=current_prices,
            source_date='20260325',
        )

        buy_codes = [inst.ts_code for inst in instructions if inst.action == 'buy']
        assert buy_codes == ['000003.SZ', '000001.SZ', '000002.SZ']


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
        '_MIN_CASHFLOW_RECORDS',
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
        ('src.lazybull.factors.cashflow_quality', 'build_cashflow_quality_lookup_by_date', 'cashflow'),
        ('src.lazybull.factors.consensus_revision', 'build_consensus_revision_lookup_by_date', 'consensus_revision'),
    ]
    expected_names = {item[2] for item in builder_targets}

    for module_path, attr_name, name in builder_targets:
        monkeypatch.setattr(
            importlib.import_module(module_path),
            attr_name,
            _record_builder(name),
        )

    class StubLoader:
        def load_fina_indicator(self, start_date=None, end_date=None):
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

        def load_cashflow(self, start_date=None, end_date=None):
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


def test_ensure_features_aligns_build_window_and_precompute(monkeypatch):
    """测试 ensure_features_for_date 对齐离线构建窗口并调用 precompute。"""
    trade_date = "20250121"
    trade_dt = pd.Timestamp(trade_date)

    expected_start = (
        trade_dt - pd.DateOffset(months=ensure_module.FEATURE_DATA_HISTORY_MONTHS)
    ).strftime("%Y%m%d")
    expected_end = (
        trade_dt + pd.DateOffset(months=ensure_module.FEATURE_DATA_FUTURE_MONTHS)
    ).strftime("%Y%m%d")

    trade_cal = pd.DataFrame(
        {
            "cal_date": pd.date_range("2024-05-01", "2025-03-31", freq="B"),
            "is_open": 1,
        }
    )
    stock_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["测试股票"],
            "list_date": ["20200101"],
            "market": ["主板"],
        }
    )
    daily_df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [trade_date],
            "close": [10.0],
            "close_adj": [10.0],
            "open": [10.0],
            "open_adj": [10.0],
            "vol": [10000],
            "amount": [100000],
            "pct_chg": [0.0],
            "pre_close": [10.0],
        }
    )

    storage = Mock(spec=Storage)
    storage.is_feature_exists.return_value = False

    loader = Mock(spec=DataLoader)
    loader.load_clean_trade_cal.return_value = trade_cal
    loader.load_clean_stock_basic.return_value = stock_basic
    loader.load_clean_daily.return_value = daily_df
    loader.load_clean_daily_basic.return_value = pd.DataFrame()
    # 覆盖“moneyflow 缺失仅告警”路径
    loader.load_clean_moneyflow.return_value = None
    loader.load_shenwan_industry.return_value = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "sw_l1": ["测试行业"],
            "sw_l1_code": ["801000"],
        }
    )

    builder = Mock(spec=FeatureBuilder)
    builder.build_features_for_day.return_value = pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": ["000001.SZ"],
            "close": [10.0],
        }
    )

    monkeypatch.setattr(ensure_module, "ensure_basic_data", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        ensure_module,
        "ensure_clean_data_for_date",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        ensure_module,
        "_ensure_historical_clean_data",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        ensure_module,
        "_load_factor_data",
        lambda *args, **kwargs: (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            [],
        ),
    )

    success, missing = ensure_features_for_date(
        storage=storage,
        loader=loader,
        builder=builder,
        cleaner=Mock(spec=DataCleaner),
        client=Mock(spec=TushareClient),
        trade_date=trade_date,
        force=False,
    )

    assert success is True
    assert missing == []
    loader.load_clean_daily.assert_called_once_with(expected_start, expected_end)
    loader.load_clean_daily_basic.assert_called_once_with(expected_start, expected_end)
    loader.load_clean_moneyflow.assert_called_once_with(expected_start, expected_end)
    builder.precompute_daily_adj.assert_called_once()
    storage.save_cs_train_day.assert_called_once()


def test_incremental_catchup_by_calendar_date_covers_weekend_announcements(temp_storage):
    """测试公告类增量会按自然日补齐（含周末），而非只查交易日单点。"""
    existing = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20260331",
                "ann_date": "20260410",
                "revenue": 1.0,
            }
        ]
    )
    temp_storage.save_raw(existing, "express", is_force=True)

    queried_dates = []

    def _fetch_by_date(ann_date: str) -> pd.DataFrame:
        queried_dates.append(ann_date)
        if ann_date == "20260411":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000002.SZ",
                        "end_date": "20260331",
                        "ann_date": "20260411",
                        "revenue": 2.0,
                    }
                ]
            )
        if ann_date == "20260413":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000003.SZ",
                        "end_date": "20260331",
                        "ann_date": "20260413",
                        "revenue": 3.0,
                    }
                ]
            )
        return pd.DataFrame()

    result = ensure_module._incremental_catchup_by_calendar_date(
        storage=temp_storage,
        dataset_name="express",
        existing_df=temp_storage.load_raw("express"),
        trade_date="20260413",
        date_col="ann_date",
        dedup_cols=["ts_code", "end_date", "ann_date"],
        fetch_by_date=_fetch_by_date,
    )

    assert queried_dates == ["20260411", "20260412", "20260413"]
    assert result is not None
    ann_dates = result["ann_date"].astype(str).str.replace("-", "").str[:8]
    assert ann_dates.max() == "20260413"
    assert (ann_dates == "20260411").any()


@pytest.mark.parametrize(
    "func_name,dataset_name,threshold_attr,date_col",
    [
        ("_try_download_fina_indicator", "fina_indicator", "_MIN_FINA_RECORDS", "ann_date"),
        (
            "_try_download_stk_holdernumber",
            "stk_holdernumber",
            "_MIN_HOLDER_RECORDS",
            "ann_date",
        ),
        ("_try_download_forecast", "forecast", "_MIN_FORECAST_RECORDS", "ann_date"),
        ("_try_download_express", "express", "_MIN_EXPRESS_RECORDS", "ann_date"),
        ("_try_download_report_rc", "report_rc", "_MIN_REPORT_RC_RECORDS", "report_date"),
    ],
)
def test_incremental_download_functions_delegate_to_range_catchup(
    monkeypatch,
    temp_storage,
    func_name,
    dataset_name,
    threshold_attr,
    date_col,
):
    """测试公告/研报类增量下载统一走“日期区间补齐”入口。"""
    if date_col == "report_date":
        existing = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20260410",
                    "org_name": "测试机构",
                    "quarter": "2026Q1",
                }
            ]
        )
    else:
        existing = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20260410",
                    "end_date": "20260331",
                }
            ]
        )

    temp_storage.save_raw(existing, dataset_name, is_force=True)
    monkeypatch.setattr(ensure_module, threshold_attr, 1)

    captured = {}

    def _fake_catchup(**kwargs):
        captured.update(kwargs)
        return kwargs["existing_df"]

    monkeypatch.setattr(ensure_module, "_incremental_catchup_by_calendar_date", _fake_catchup)

    client = Mock(spec=TushareClient)
    func = getattr(ensure_module, func_name)
    result = func(client=client, storage=temp_storage, trade_date="20260430")

    assert result is not None
    assert captured["dataset_name"] == dataset_name
    assert captured["date_col"] == date_col
    assert captured["trade_date"] == "20260430"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
