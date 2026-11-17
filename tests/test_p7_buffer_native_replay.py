"""原生历史驱动的合成等价与精确证据序列测试。"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from scripts.v2_p7.buffer_native_replay import (
    FrozenRankSignal,
    RecordNativeMixin,
    canonical,
    compare_exact,
    initialize_driver,
    load_prices,
    result_payload,
)
from scripts.v2_p7.buffer_native_validation import validate_account
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
from src.lazybull.v2.core.decision.weighting import estimate_stock_variance
from src.lazybull.v2.core.execution.price_index import prepare_price_index
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML


class SyntheticUniverse:
    def get_stocks(self, date, quote_data=None):
        return [f"{i:06}.SZ" for i in range(1, 26)]


class SyntheticStorage:
    def load_raw_by_date(self, dataset, trade_date):
        assert dataset == "suspend"
        return pd.DataFrame(columns=["ts_code", "suspend_type"])


class RecordedNative(RecordNativeMixin, NativeMergeBacktestEngineML):
    pass


def fixture_inputs():
    dates = list(pd.bdate_range("2024-01-02", periods=65))
    rows, ranked = [], {}
    codes = SyntheticUniverse().get_stocks(None)
    for t, date in enumerate(dates):
        day = date.strftime("%Y%m%d")
        shift = (t // 20) * 6 % 25
        order = codes[shift:] + codes[:shift]
        ranked[day] = [(code, 1.0 - rank / 100.0) for rank, code in enumerate(order)]
        for i, code in enumerate(codes):
            close = 10 + i / 10 + t / 100 + np.sin(t + i) / 100
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": day,
                    "open": close - 0.01,
                    "close": close,
                    "open_adj": close - 0.01,
                    "close_adj": close,
                    "is_st": False,
                    "is_suspended": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "tradable": True,
                }
            )
    return dates, pd.DataFrame(rows), ranked


def make_engine(ranked):
    return RecordedNative(
        portfolio_merger=SingleSleevePortfolioMerger(),
        universe=SyntheticUniverse(),
        signal=FrozenRankSignal(ranked),
        features_by_date={k: pd.DataFrame({"x": [1]}) for k in ranked},
        initial_capital=1000000,
        cost_model=CostModel(0.0001954, 5.0, 0.0005, 0.0005),
        rebalance_freq=20,
        stagger_tranches=2,
        holding_period=20,
        sell_timing="open",
        verbose=False,
        position_sizing="half_kelly",
        kelly_vol_window=60,
        kelly_max_leverage=0.2,
        max_weight_per_stock=0.15,
        min_buy_value_ratio=0.2,
        data_storage=SyntheticStorage(),
        completion_window_days=5,
    )


def test_manual_shared_index_driver_is_bitwise_native_with_categorical_cache():
    dates, prices, ranked = fixture_inputs()
    logger.disable("src.lazybull")
    try:
        native = make_engine(ranked)
        native.run(dates[0], dates[-1], dates, prices)
        encoded = prices.copy()
        encoded["trade_date"] = pd.Categorical(
            prices.trade_date, categories=sorted(prices.trade_date.unique()), ordered=True
        )
        encoded["ts_code"] = pd.Categorical(prices.ts_code)
        manual = make_engine(ranked)
        initialize_driver(manual, dates, encoded, prepare_price_index(encoded))
        for date in dates:
            manual.run_step(date, dates, encoded)
        check = compare_exact(result_payload(native), result_payload(manual))
        assert check["all_equal"], check
        assert validate_account(result_payload(native), initial_capital=1000000)["valid"]
        assert validate_account(result_payload(manual), initial_capital=1000000)["valid"]
        assert len(manual.trades) > 20
    finally:
        logger.enable("src.lazybull")


def test_exact_comparison_catches_one_ulp_and_row_order():
    left = {
        key: [{"v": 1.0}, {"v": 2.0}]
        for key in ("daily", "states", "trades", "attribution", "holdings_snapshot", "ending")
    }
    right = copy.deepcopy(left)
    assert compare_exact(left, right)["all_equal"]
    right["daily"][0]["v"] = np.nextafter(1.0, 2.0)
    assert not compare_exact(left, right)["all_equal"]
    right = copy.deepcopy(left)
    right["trades"].reverse()
    assert not compare_exact(left, right)["all_equal"]


def test_canonical_preserves_nonfinite_identity_without_invalid_json():
    assert canonical({"x": np.nan}) != canonical({"x": None})
    assert canonical({"x": np.float64(1.25)}) == canonical({"x": 1.25})
    with pytest.raises(TypeError):
        canonical(object())


def test_warmup_uses_actual_stock_rows_across_suspension(tmp_path):
    dates = list(pd.bdate_range("2023-01-02", periods=102))
    days = [date.strftime("%Y%m%d") for date in dates]
    folder = tmp_path / "data/clean/daily"
    folder.mkdir(parents=True)
    rows = []
    for i, (date, day) in enumerate(zip(dates, days)):
        current = [{"ts_code": "A", "trade_date": day, "close_adj": 10 + i / 100}]
        # B在近期连续缺行；其60个实际样本必须追溯到更早日期。
        if not 70 <= i < 90:
            current.append(
                {"ts_code": "B", "trade_date": day, "close_adj": 20 + np.sin(i / 3) + i / 100}
            )
        frame = pd.DataFrame(current)
        frame.to_parquet(folder / f"{date:%Y-%m-%d}.parquet", index=False)
        rows.extend(current)
    original = pd.DataFrame(rows)
    clipped, identity = load_prices(tmp_path, days, warmup_cutoff=days[100])
    assert len(identity) == 102
    assert len(clipped[(clipped.ts_code == "B") & (clipped.trade_date < days[100])]) == 60
    for code in ("A", "B"):
        assert estimate_stock_variance(original, code, dates[100], 60) == estimate_stock_variance(
            clipped, code, dates[100], 60
        )


def test_independent_group_preserves_corporate_block_but_never_swallows_engine_error(
    tmp_path, monkeypatch
):
    from scripts.v2_p7 import run_buffer_native as runner
    from scripts.v2_p7.t4.corporate import CorporateError

    (tmp_path / "history").mkdir()

    def corporate_block(*args, **kwargs):
        raise CorporateError("合成受限股事件")

    monkeypatch.setattr(runner, "run_group", corporate_block)
    blocked = runner.run_independent_group(tmp_path, tmp_path, "history")
    assert blocked["status"] == "blocked_corporate"
    assert blocked["economic_comparison_published"] is False
    assert (tmp_path / "history/blocked.json").is_file()

    def engine_error(*args, **kwargs):
        raise ValueError("机械/实现错误必须中止")

    monkeypatch.setattr(runner, "run_group", engine_error)
    with pytest.raises(ValueError, match="机械/实现"):
        runner.run_independent_group(tmp_path, tmp_path, "new_window")
