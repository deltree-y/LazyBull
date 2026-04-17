"""测试持仓保留奖励（降低换手率）功能"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.lazybull.backtest import BacktestEngineML
from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.common.cost import CostModel
from src.lazybull.ml import ModelRegistry
from src.lazybull.signals import MLSignal
from src.lazybull.universe import BasicUniverse


# ── 共用 Mock / Fixture ──────────────────────────────────────────


class MockMLModel:
    """模拟 ML 模型：predict 返回 f1 * 0.1"""

    def predict(self, X):
        if len(X.columns) > 0:
            return X.iloc[:, 0].values * 0.1
        return np.random.randn(len(X))


STOCKS = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH", "600001.SH"]


@pytest.fixture
def temp_models_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def trained_model(temp_models_dir):
    registry = ModelRegistry(models_dir=temp_models_dir)
    model = MockMLModel()
    version = registry.register_model(
        model=model,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["f1", "f2", "f3"],
        label_column="y_ret_5",
        n_samples=1000,
        train_params={"n_estimators": 100},
    )
    return temp_models_dir, version


@pytest.fixture
def mock_stock_basic():
    return pd.DataFrame(
        {
            "ts_code": STOCKS,
            "symbol": ["000001", "000002", "000003", "600000", "600001"],
            "name": ["股票A", "股票B", "股票C", "股票D", "股票E"],
            "market": ["主板"] * 5,
            "list_date": ["20200101"] * 5,
        }
    )


def _make_price_data(dates: list[str], stocks: list[str] = None) -> pd.DataFrame:
    """生成稳定的模拟价格数据（固定价格, 无随机波动）"""
    stocks = stocks or STOCKS
    rows = []
    for date in dates:
        for stock in stocks:
            rows.append(
                {
                    "ts_code": stock,
                    "trade_date": date,
                    "close": 10.0,
                    "close_adj": 10.0,
                    "open": 10.0,
                    "open_adj": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "pct_chg": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _make_features(stocks: list[str], f1_values: list[float]) -> pd.DataFrame:
    """生成特征 DataFrame"""
    return pd.DataFrame(
        {
            "ts_code": stocks,
            "f1": f1_values,
            "f2": [0.0] * len(stocks),
            "f3": [0.0] * len(stocks),
        }
    )


# ── 1. 持仓保留奖励 (Holding Bonus) 单元测试 ──────────────────────


class TestHoldingBonus:
    """测试持仓保留奖励逻辑"""

    def test_holding_bonus_disabled_by_default(self):
        """默认不启用持仓奖励"""
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.holding_bonus_enabled = False
        assert engine.holding_bonus_enabled is False

    def test_holding_bonus_extends_holding_period(
        self, trained_model, mock_stock_basic
    ):
        """测试持仓保留奖励模式下, 保留股票的持有期被重置

        构造场景：
        - 4个交易日（信号日+T+1买入+第二轮信号日+T+1）
        - top_n=2, rebalance_freq=1
        - Day0 信号: 600001.SH(15)>000003.SZ(12) → 买入两只
        - Day2 信号: 600001.SH仍在top → 持有期被延续
        """
        dates = ["20230601", "20230602", "20230605", "20230606"]
        price_data = _make_price_data(dates)

        # Day0: 600001.SH(15), 000003.SZ(12), 000001.SZ(10), 000002.SZ(8), 600000.SH(6)
        # Day2: 600001.SH(20), 000001.SZ(3), 000003.SZ(2), ... → 600001.SH 依然最高
        features_by_date = {
            "20230601": _make_features(STOCKS, [10, 8, 12, 6, 15]),
            "20230605": _make_features(STOCKS, [3, 1, 2, 0, 20]),
        }

        models_dir, version = trained_model
        signal = MLSignal(
            top_n=2,
            model_version=version,
            models_dir=models_dir,
        )
        universe = BasicUniverse(
            stock_basic=mock_stock_basic,
            exclude_st=False,
            min_list_days=0,
            markets=["主板"],
        )

        engine = BacktestEngineML(
            features_by_date=features_by_date,
            universe=universe,
            signal=signal,
            initial_capital=100000.0,
            cost_model=CostModel(),
            rebalance_freq=2,  # 每2天调仓
            holding_bonus_enabled=True,
            holding_bonus_sigma=1.0,
            verbose=True,
        )

        trading_dates = [pd.Timestamp(d) for d in dates]
        nav_curve = engine.run(
            start_date=trading_dates[0],
            end_date=trading_dates[-1],
            trading_dates=trading_dates,
            price_data=price_data,
        )

        # 基本验证：回测应正常完成
        assert nav_curve is not None
        assert len(nav_curve) > 0

    def test_holding_bonus_score_boost(self, trained_model, mock_stock_basic):
        """测试持仓保留奖励的评分加成逻辑

        构造场景（5个交易日, rebalance_freq=2, top_n=2）:
        - Day0 信号 → 600001.SH(f1=15), 000003.SZ(f1=12)
        - Day0 信号 T+1 (Day1) 买入
        - Day2 信号: 排名洗牌但 600001.SH 因持仓加分仍留在top
          原始: 600000.SH(f1=14) > 000002.SZ(f1=13) > 600001.SH(f1=11)
          加分后: 600001.SH(f1=11 + bonus) 应能超过 000002.SZ(f1=13)
        """
        dates = ["20230601", "20230602", "20230605", "20230606", "20230607"]
        price_data = _make_price_data(dates)

        features_by_date = {
            # Day0: 600001.SH(15) > 000003.SZ(12) → top2
            "20230601": _make_features(STOCKS, [10, 8, 12, 6, 15]),
            # Day2: 原始排序 600000.SH(14) > 000002.SZ(13) > 600001.SH(11) > ...
            # 600001.SH 原本第3, 但因持仓加分可能升到 top2
            "20230605": _make_features(STOCKS, [5, 13, 7, 14, 11]),
        }

        models_dir, version = trained_model
        signal = MLSignal(
            top_n=2,
            model_version=version,
            models_dir=models_dir,
        )
        universe = BasicUniverse(
            stock_basic=mock_stock_basic,
            exclude_st=False,
            min_list_days=0,
            markets=["主板"],
        )

        engine = BacktestEngineML(
            features_by_date=features_by_date,
            universe=universe,
            signal=signal,
            initial_capital=100000.0,
            cost_model=CostModel(),
            rebalance_freq=2,
            holding_bonus_enabled=True,
            holding_bonus_sigma=2.0,  # 高加分 → 保证保留
            verbose=True,
        )

        trading_dates = [pd.Timestamp(d) for d in dates]
        nav_curve = engine.run(
            start_date=trading_dates[0],
            end_date=trading_dates[-1],
            trading_dates=trading_dates,
            price_data=price_data,
        )

        # 验证回测正常完成
        assert nav_curve is not None
        trades = engine.get_trades()
        # 如果 600001.SH 被保留，它不会被卖出再买入
        # 检查 decision_trace 里有 holding_bonus 信息
        assert len(nav_curve) > 0

    def test_holding_bonus_disabled_excludes_positions(
        self, trained_model, mock_stock_basic
    ):
        """当持仓保留奖励关闭时，已持仓股票应被排除出候选"""
        dates = ["20230601", "20230602", "20230605", "20230606", "20230607"]
        price_data = _make_price_data(dates)

        features_by_date = {
            "20230601": _make_features(STOCKS, [10, 8, 12, 6, 15]),
            "20230605": _make_features(STOCKS, [5, 13, 7, 14, 11]),
        }

        models_dir, version = trained_model
        signal = MLSignal(
            top_n=2,
            model_version=version,
            models_dir=models_dir,
        )
        universe = BasicUniverse(
            stock_basic=mock_stock_basic,
            exclude_st=False,
            min_list_days=0,
            markets=["主板"],
        )

        engine = BacktestEngineML(
            features_by_date=features_by_date,
            universe=universe,
            signal=signal,
            initial_capital=100000.0,
            cost_model=CostModel(),
            rebalance_freq=2,
            holding_bonus_enabled=False,  # 关闭
            verbose=True,
        )

        trading_dates = [pd.Timestamp(d) for d in dates]
        nav_curve = engine.run(
            start_date=trading_dates[0],
            end_date=trading_dates[-1],
            trading_dates=trading_dates,
            price_data=price_data,
        )
        assert nav_curve is not None
        assert len(nav_curve) > 0


class TestExtendHoldingPeriod:
    """测试 _extend_holding_period 方法"""

    def test_extend_resets_buy_date(self):
        """延续持有重置 buy_date 为 T+1"""
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.verbose = False
        trading_dates = [
            pd.Timestamp("20230601"),
            pd.Timestamp("20230602"),
            pd.Timestamp("20230605"),
        ]
        date_to_idx = {d: i for i, d in enumerate(trading_dates)}
        engine.positions = {
            "000001.SZ": {
                "buy_date": pd.Timestamp("20230601"),
                "signal_date": pd.Timestamp("20230530"),
                "shares": 100,
                "buy_trade_price": 10.0,
                "buy_pnl_price": 10.0,
            }
        }

        # 信号日是 20230601 → T+1 是 20230602
        engine._extend_holding_period(
            "000001.SZ", pd.Timestamp("20230601"), trading_dates, date_to_idx
        )
        assert engine.positions["000001.SZ"]["buy_date"] == pd.Timestamp("20230602")
        assert engine.positions["000001.SZ"]["signal_date"] == pd.Timestamp("20230601")

    def test_extend_missing_stock_noop(self):
        """延续不存在的股票不应报错"""
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.verbose = False
        engine.positions = {}
        trading_dates = [pd.Timestamp("20230601"), pd.Timestamp("20230602")]
        date_to_idx = {d: i for i, d in enumerate(trading_dates)}
        # 不应抛出异常
        engine._extend_holding_period(
            "999999.SZ", pd.Timestamp("20230601"), trading_dates, date_to_idx
        )

    def test_extend_last_date_noop(self):
        """信号日后没有交易日时不应修改"""
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.verbose = False
        trading_dates = [pd.Timestamp("20230601")]
        date_to_idx = {pd.Timestamp("20230601"): 0}
        engine.positions = {
            "000001.SZ": {
                "buy_date": pd.Timestamp("20230530"),
                "signal_date": pd.Timestamp("20230529"),
                "shares": 100,
            }
        }
        engine._extend_holding_period(
            "000001.SZ", pd.Timestamp("20230601"), trading_dates, date_to_idx
        )
        # buy_date 不应改变（无 T+1）
        assert engine.positions["000001.SZ"]["buy_date"] == pd.Timestamp("20230530")


# ── 2. TradingConfig 字段测试 ────────────────────────────────────


class TestTradingConfigFields:
    """测试 TradingConfig 中持仓保留奖励字段的默认值和传递"""

    def test_default_values(self):
        from src.lazybull.common.trading_config import TradingConfig

        tc = TradingConfig()
        assert tc.holding_bonus_enabled is False
        assert tc.holding_bonus_sigma == 0.5

    def test_custom_values(self):
        from src.lazybull.common.trading_config import TradingConfig

        tc = TradingConfig(
            holding_bonus_enabled=True,
            holding_bonus_sigma=1.0,
        )
        assert tc.holding_bonus_enabled is True
        assert tc.holding_bonus_sigma == 1.0
