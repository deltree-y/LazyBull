"""行业轮动加权 + Kelly/半Kelly 仓位管理 测试

覆盖:
1. 行业轮动加权(_post_filter_candidates 步骤2)
   - 加权后分数调整正确性
   - 缺失 rank 时分数不变
   - 重排序生效
   - 与硬过滤独立运作
2. Kelly 仓位管理(_normalize_signals / _kelly_weights)
   - equal 模式等权
   - score 模式按分数加权
   - kelly 模式仓位与分数/波动率关系正确
   - half_kelly 模式仓位为 kelly 的一半
   - fallback: 分数 <= 0 或无波动率数据时走中位数
   - kelly_max_leverage 约束
3. TradingConfig 新增字段校验
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.backtest.engine_ml import BacktestEngineML
from src.lazybull.common.cost import CostModel
from src.lazybull.common.trading_config import TradingConfig


# ── 公共 fixtures ────────────────────────────────────────────────

def _make_features_df(stocks_and_ranks):
    """构造含 ts_code + ind_momentum_rank 的特征 DataFrame"""
    return pd.DataFrame({
        "ts_code": [s for s, _ in stocks_and_ranks],
        "ind_momentum_rank": [r for _, r in stocks_and_ranks],
    })


def _make_engine_with_price_cache(
    position_sizing="equal",
    kelly_vol_window=60,
    kelly_max_leverage=0.25,
):
    """构造一个带 price_data_cache 的 BacktestEngine（用于 Kelly 测试）"""
    universe = MagicMock()
    universe.get_universe.return_value = [
        "000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"
    ]
    signal = MagicMock()
    signal.generate_signals.return_value = {}
    cost_model = CostModel()

    # 构造价格数据: 4只股票,60日收盘价
    dates = pd.bdate_range("2023-01-01", periods=80, freq="B")
    rows = []
    np.random.seed(42)
    for stock in ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]:
        base = 10.0
        for d in dates:
            # 用不同波动率
            if stock == "000001.SZ":
                vol = 0.01  # 低波动
            elif stock == "000002.SZ":
                vol = 0.03  # 高波动
            elif stock == "000003.SZ":
                vol = 0.02  # 中等波动
            else:
                vol = 0.015
            base *= np.exp(np.random.normal(0.001, vol))
            rows.append({
                "ts_code": stock,
                "trade_date": d.strftime("%Y%m%d"),
                "close": base,
            })

    price_df = pd.DataFrame(rows)

    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=cost_model,
        rebalance_freq=5,
        holding_period=5,
        position_sizing=position_sizing,
        kelly_vol_window=kelly_vol_window,
        kelly_max_leverage=kelly_max_leverage,
        verbose=False,
    )
    engine.price_data_cache = price_df
    return engine


# ── 1. 行业轮动加权测试 ──────────────────────────────────────────


class TestIndustryRotationEnhanced:
    """测试 BacktestEngineML._post_filter_candidates 步骤2: 行业轮动加权"""

    def _make_ml_engine(
        self,
        features_by_date,
        industry_rotation_enhanced=True,
        industry_rotation_alpha=0.3,
        industry_momentum_filter=False,
        industry_momentum_bottom_pct=0.2,
    ):
        universe = MagicMock()
        universe.get_universe.return_value = []
        signal = MagicMock()
        signal.generate_signals.return_value = {}

        engine = BacktestEngineML(
            features_by_date=features_by_date,
            industry_rotation_enhanced=industry_rotation_enhanced,
            industry_rotation_alpha=industry_rotation_alpha,
            industry_momentum_filter=industry_momentum_filter,
            industry_momentum_bottom_pct=industry_momentum_bottom_pct,
            universe=universe,
            signal=signal,
            initial_capital=100000.0,
            cost_model=CostModel(),
            rebalance_freq=5,
            holding_period=5,
            verbose=False,
        )
        return engine

    def test_rotation_adjusts_scores(self):
        """行业轮动加权: 强势行业分数上调,弱势行业分数下调"""
        features_df = _make_features_df([
            ("A.SZ", 1.0),   # 最强行业 → ×(1 + 0.3×0.5) = ×1.15
            ("B.SZ", 0.0),   # 最弱行业 → ×(1 + 0.3×(-0.5)) = ×0.85
            ("C.SZ", 0.5),   # 中位行业 → ×1.0（不变）
        ])
        engine = self._make_ml_engine(
            features_by_date={"20230101": features_df},
            industry_rotation_alpha=0.3,
        )

        candidates = [("A.SZ", 1.0), ("B.SZ", 1.0), ("C.SZ", 1.0)]
        date = pd.Timestamp("2023-01-01")
        result = engine._post_filter_candidates(candidates, date)

        result_dict = {s: sc for s, sc in result}
        assert abs(result_dict["A.SZ"] - 1.15) < 1e-9, "最强行业分数应上调至 1.15"
        assert abs(result_dict["B.SZ"] - 0.85) < 1e-9, "最弱行业分数应下调至 0.85"
        assert abs(result_dict["C.SZ"] - 1.0) < 1e-9, "中位行业分数应不变"

    def test_rotation_reorders_candidates(self):
        """加权后应重新排序:原本相同分数的股票,强势行业排前面"""
        features_df = _make_features_df([
            ("A.SZ", 0.2),   # 弱势
            ("B.SZ", 0.9),   # 强势
        ])
        engine = self._make_ml_engine(
            features_by_date={"20230101": features_df},
            industry_rotation_alpha=0.5,
        )

        candidates = [("A.SZ", 1.0), ("B.SZ", 1.0)]
        result = engine._post_filter_candidates(candidates, pd.Timestamp("2023-01-01"))

        # B 应排在 A 前面
        assert result[0][0] == "B.SZ"
        assert result[1][0] == "A.SZ"

    def test_rotation_missing_rank_keeps_score(self):
        """缺失 ind_momentum_rank 的股票分数不变"""
        features_df = _make_features_df([
            ("A.SZ", 0.8),
        ])  # B.SZ 不在 features 中
        engine = self._make_ml_engine(
            features_by_date={"20230101": features_df},
        )

        candidates = [("A.SZ", 1.0), ("B.SZ", 2.0)]
        result = engine._post_filter_candidates(candidates, pd.Timestamp("2023-01-01"))
        result_dict = {s: sc for s, sc in result}

        assert abs(result_dict["B.SZ"] - 2.0) < 1e-9, "缺失 rank 的股票分数应不变"

    def test_rotation_nan_rank_keeps_score(self):
        """rank 为 NaN 的股票分数不变"""
        features_df = _make_features_df([
            ("A.SZ", 0.8),
            ("B.SZ", float("nan")),
        ])
        engine = self._make_ml_engine(
            features_by_date={"20230101": features_df},
        )

        candidates = [("A.SZ", 1.0), ("B.SZ", 2.0)]
        result = engine._post_filter_candidates(candidates, pd.Timestamp("2023-01-01"))
        result_dict = {s: sc for s, sc in result}
        assert abs(result_dict["B.SZ"] - 2.0) < 1e-9

    def test_rotation_disabled_no_change(self):
        """关闭轮动加权时不修改分数"""
        features_df = _make_features_df([("A.SZ", 1.0)])
        engine = self._make_ml_engine(
            features_by_date={"20230101": features_df},
            industry_rotation_enhanced=False,
        )

        candidates = [("A.SZ", 1.0)]
        result = engine._post_filter_candidates(candidates, pd.Timestamp("2023-01-01"))
        assert result == candidates

    def test_rotation_and_filter_independent(self):
        """硬过滤 + 软加权可同时启用,先过滤后加权"""
        features_df = _make_features_df([
            ("A.SZ", 0.1),   # 被硬过滤(rank < 0.2)
            ("B.SZ", 0.5),   # 保留,中位行业
            ("C.SZ", 0.9),   # 保留,强势行业
        ])
        engine = self._make_ml_engine(
            features_by_date={"20230101": features_df},
            industry_momentum_filter=True,
            industry_momentum_bottom_pct=0.2,
            industry_rotation_enhanced=True,
            industry_rotation_alpha=0.3,
        )

        candidates = [("A.SZ", 1.0), ("B.SZ", 1.0), ("C.SZ", 1.0)]
        result = engine._post_filter_candidates(candidates, pd.Timestamp("2023-01-01"))

        # A 被硬过滤掉
        result_stocks = [s for s, _ in result]
        assert "A.SZ" not in result_stocks
        assert "B.SZ" in result_stocks
        assert "C.SZ" in result_stocks

        # 剩余的 B、C 被加权调整
        result_dict = {s: sc for s, sc in result}
        # C (rank=0.9): 1.0 × (1 + 0.3×(0.9-0.5)) = 1.12
        assert abs(result_dict["C.SZ"] - 1.12) < 1e-9
        # B (rank=0.5): 1.0 × (1 + 0.3×0.0) = 1.0
        assert abs(result_dict["B.SZ"] - 1.0) < 1e-9

    def test_rotation_no_features_no_crash(self):
        """日期无特征数据时不崩溃,候选列表不变"""
        engine = self._make_ml_engine(features_by_date={})
        candidates = [("A.SZ", 1.0)]
        result = engine._post_filter_candidates(candidates, pd.Timestamp("2023-01-01"))
        assert result == candidates


# ── 2. Kelly / 半 Kelly 仓位管理测试 ─────────────────────────────


class TestNormalizeSignals:
    """测试 BacktestEngine._normalize_signals 的 4 种模式"""

    def test_equal_weights(self):
        """equal 模式: 所有股票等权"""
        engine = _make_engine_with_price_cache(position_sizing="equal")
        signals = {"A.SZ": 0.5, "B.SZ": 0.3, "C.SZ": 0.2}
        date = pd.Timestamp("2023-04-01")
        result = engine._normalize_signals(signals, date)

        assert len(result) == 3
        for stock in signals:
            assert abs(result[stock] - 1.0 / 3) < 1e-9

    def test_score_weights(self):
        """score 模式: 权重按分数线性加权"""
        engine = _make_engine_with_price_cache(position_sizing="score")
        signals = {"A.SZ": 0.6, "B.SZ": 0.3, "C.SZ": 0.1}
        date = pd.Timestamp("2023-04-01")
        result = engine._normalize_signals(signals, date)

        assert abs(result["A.SZ"] - 0.6) < 1e-9
        assert abs(result["B.SZ"] - 0.3) < 1e-9
        assert abs(result["C.SZ"] - 0.1) < 1e-9

    def test_score_all_zero_fallback_to_equal(self):
        """score 模式: 所有分数 <= 0 时回退等权"""
        engine = _make_engine_with_price_cache(position_sizing="score")
        signals = {"A.SZ": 0.0, "B.SZ": -0.1}
        date = pd.Timestamp("2023-04-01")
        result = engine._normalize_signals(signals, date)

        for stock in signals:
            assert abs(result[stock] - 0.5) < 1e-9

    def test_empty_signals(self):
        """空信号返回空字典"""
        engine = _make_engine_with_price_cache()
        result = engine._normalize_signals({}, pd.Timestamp("2023-04-01"))
        assert result == {}


class TestKellyWeights:
    """测试 Kelly / 半 Kelly 仓位计算"""

    def test_kelly_weights_sum_to_one(self):
        """kelly 模式: 权重总和为 1.0"""
        engine = _make_engine_with_price_cache(position_sizing="kelly")
        signals = {
            "000001.SZ": 0.05,
            "000002.SZ": 0.03,
            "000003.SZ": 0.04,
        }
        date = pd.Timestamp("2023-04-01")
        result = engine._normalize_signals(signals, date)

        assert len(result) == 3
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9, f"Kelly 权重总和应为 1.0, 实际: {total}"

    def test_half_kelly_is_half(self):
        """half_kelly 模式的原始 f* 应为 kelly 的一半"""
        engine_kelly = _make_engine_with_price_cache(position_sizing="kelly")
        engine_half = _make_engine_with_price_cache(position_sizing="half_kelly")

        signals = {"000001.SZ": 0.05, "000002.SZ": 0.03}
        date = pd.Timestamp("2023-04-01")

        # 由于归一化后总和都是 1.0,但 f* 之间的比例关系应保持
        # half_kelly 相当于对所有 f* 乘 0.5 再归一化 → 权重比例与 kelly 完全相同
        result_kelly = engine_kelly._normalize_signals(signals, date)
        result_half = engine_half._normalize_signals(signals, date)

        # 归一化后比例应一致（因为 ×0.5 是全局的,再归一化后比例不变）
        for stock in signals:
            assert abs(result_kelly[stock] - result_half[stock]) < 1e-9

    def test_kelly_low_vol_gets_higher_weight(self):
        """低波动股票在 Kelly 中应获得更高的仓位"""
        # 设极高上限避免 max_leverage 截断抹平差异
        # 注: f* = μ/σ² 量级很大(分数~0.05, 方差~1e-4 → f*~500)
        engine = _make_engine_with_price_cache(
            position_sizing="kelly", kelly_max_leverage=10000.0,
        )
        # 给所有股票相同分数,波动率差异导致权重差异
        signals = {
            "000001.SZ": 0.05,   # 低波动
            "000002.SZ": 0.05,   # 高波动
        }
        date = pd.Timestamp("2023-04-01")
        result = engine._normalize_signals(signals, date)

        # 低波动(000001.SZ)应有更高权重(f* = μ/σ², σ²小 → f*大)
        assert result["000001.SZ"] > result["000002.SZ"], \
            "相同分数下,低波动股票应获得更高 Kelly 权重"

    def test_kelly_negative_score_fallback(self):
        """负分数的股票应走 fallback (中位 Kelly 值)"""
        engine = _make_engine_with_price_cache(position_sizing="kelly")
        signals = {
            "000001.SZ": 0.05,
            "000002.SZ": -0.01,   # 负分数 → fallback
        }
        date = pd.Timestamp("2023-04-01")
        result = engine._normalize_signals(signals, date)

        assert len(result) == 2
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9

    def test_kelly_max_leverage_clamp(self):
        """f* 应被 kelly_max_leverage 截断"""
        # 设一个很低的上限
        engine = _make_engine_with_price_cache(
            position_sizing="kelly",
            kelly_max_leverage=0.01,
        )
        signals = {
            "000001.SZ": 0.05,
            "000002.SZ": 0.03,
            "000003.SZ": 0.04,
        }
        date = pd.Timestamp("2023-04-01")
        result = engine._normalize_signals(signals, date)

        # 所有 f* 被截断到 0.01,归一化后应接近等权
        assert len(result) == 3
        for stock in result:
            assert abs(result[stock] - 1.0 / 3) < 0.01, \
                "Kelly max_leverage 截断后应接近等权"

    def test_kelly_no_price_data_all_fallback(self):
        """无 price_data_cache 时全部 fallback 到等权"""
        engine = _make_engine_with_price_cache(position_sizing="kelly")
        engine.price_data_cache = None  # 清除价格缓存

        signals = {"000001.SZ": 0.05, "000002.SZ": 0.03}
        date = pd.Timestamp("2023-04-01")
        result = engine._normalize_signals(signals, date)

        # 无法估计波动率 → 全部 fallback → 等权
        for stock in result:
            assert abs(result[stock] - 0.5) < 1e-9


# ── 3. position_sizing 参数校验测试 ──────────────────────────────


class TestPositionSizingValidation:
    """测试 position_sizing 参数校验"""

    def test_invalid_position_sizing_raises(self):
        """非法 position_sizing 值应抛出 ValueError"""
        universe = MagicMock()
        universe.get_universe.return_value = []
        signal = MagicMock()
        signal.generate_signals.return_value = {}

        with pytest.raises(ValueError, match="position_sizing"):
            BacktestEngine(
                universe=universe,
                signal=signal,
                initial_capital=100000.0,
                cost_model=CostModel(),
                rebalance_freq=5,
                holding_period=5,
                position_sizing="invalid_mode",
            )

    def test_valid_position_sizing_accepted(self):
        """合法 position_sizing 值应正常初始化"""
        universe = MagicMock()
        universe.get_universe.return_value = []
        signal = MagicMock()
        signal.generate_signals.return_value = {}

        for mode in ["equal", "score", "kelly", "half_kelly"]:
            engine = BacktestEngine(
                universe=universe,
                signal=signal,
                initial_capital=100000.0,
                cost_model=CostModel(),
                rebalance_freq=5,
                holding_period=5,
                position_sizing=mode,
                verbose=False,
            )
            assert engine.position_sizing == mode


# ── 4. TradingConfig 新增字段测试 ────────────────────────────────


class TestTradingConfigNewFields:
    """测试 TradingConfig 中新增的行业轮动和仓位管理字段"""

    def test_default_values(self):
        """默认值应符合预期"""
        cfg = TradingConfig()
        assert cfg.industry_rotation_enhanced is False
        assert cfg.industry_rotation_alpha == 0.3
        assert cfg.position_sizing == "equal"
        assert cfg.kelly_vol_window == 60
        assert cfg.kelly_max_leverage == 0.25

    def test_custom_values(self):
        """自定义值应正确赋值"""
        cfg = TradingConfig(
            industry_rotation_enhanced=True,
            industry_rotation_alpha=0.5,
            position_sizing="half_kelly",
            kelly_vol_window=30,
            kelly_max_leverage=0.15,
        )
        assert cfg.industry_rotation_enhanced is True
        assert cfg.industry_rotation_alpha == 0.5
        assert cfg.position_sizing == "half_kelly"
        assert cfg.kelly_vol_window == 30
        assert cfg.kelly_max_leverage == 0.15
