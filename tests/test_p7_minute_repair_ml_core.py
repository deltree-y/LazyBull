"""LR5M-ML-10-v1 核心合成反例：时段、特征、标签、切分、母域。

全部使用合成数据与隔离夹具，不访问真实 token、生产数据或外部 API。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.minute_repair_ml import domain, features, labels, sessions, spec, splits


def make_slots(
    close_path: dict[int, float],
    *,
    high_at: int | None = None,
    low_at: int | None = None,
    quantity: float = 1000.0,
    amount: float = 10000.0,
) -> pd.DataFrame:
    """构造 1..48 槽位表：默认 H=C+0.01、L=C-0.01，可按槽位放大边界。"""
    slots = sessions._empty_slots()
    last = 10.0
    for k in range(1, 49):
        if k in close_path:
            last = float(close_path[k])
        slots.loc[k, ["C", "O", "H", "L", "Q", "A"]] = [
            last,
            last,
            last + 0.01,
            last - 0.01,
            quantity,
            amount,
        ]
    if high_at is not None:
        slots.loc[high_at, "H"] = float(slots["H"].max()) + 0.5
    if low_at is not None:
        slots.loc[low_at, "L"] = float(slots["L"].min()) - 0.5
    return slots


def path_up_then_down() -> pd.DataFrame:
    """先冲高后探底：高点在 k=20，低点在 k=40，收盘回到 10.10。"""
    closes = {k: 10.0 + 0.4 * (k - 1) / 19 for k in range(1, 21)}
    closes.update({k: 10.4 - 0.6 * (k - 20) / 20 for k in range(21, 41)})
    closes.update({k: 9.8 + 0.3 * (k - 40) / 7 for k in range(41, 48)})
    slots = make_slots(closes)
    slots["H"] = slots["C"]
    slots["L"] = slots["C"]
    slots.loc[20, "H"] = 10.4
    slots.loc[40, "L"] = 9.8
    return slots


def path_down_then_up() -> pd.DataFrame:
    """先探底后冲高：低点在 k=3，高点在 k=40，收盘同为 10.10。"""
    closes = {1: 10.0, 2: 9.9, 3: 9.8}
    closes.update({k: 9.8 + 0.6 * (k - 3) / 37 for k in range(4, 41)})
    closes.update({k: 10.4 - 0.3 * (k - 40) / 7 for k in range(41, 48)})
    slots = make_slots(closes)
    slots["H"] = slots["C"]
    slots["L"] = slots["C"]
    slots.loc[3, "L"] = 9.8
    slots.loc[40, "H"] = 10.4
    return slots


class TestSessions:
    def test_end_convention_grid(self) -> None:
        assert sessions.classify_time("2026-11-13 09:35:00", "end") == (sessions.CORE, 1)
        assert sessions.classify_time("2026-11-13 11:30:00", "end") == (sessions.CORE, 24)
        assert sessions.classify_time("2026-11-13 13:05:00", "end") == (sessions.CORE, 25)
        assert sessions.classify_time("2026-11-13 15:00:00", "end") == (sessions.CORE, 48)

    def test_auction_and_lunch_are_not_core(self) -> None:
        assert sessions.classify_time("2026-11-13 09:30:00", "end") == (sessions.AUCTION, None)
        assert sessions.classify_time("2026-11-13 11:35:00", "end") == (sessions.LUNCH, None)
        assert sessions.classify_time("2026-11-13 12:30:00", "end") == (sessions.LUNCH, None)
        assert sessions.classify_time("2026-11-13 09:32:00", "end") == (sessions.OFF_GRID, None)

    def test_start_convention_grid(self) -> None:
        assert sessions.classify_time("2026-11-13 09:30:00", "start") == (sessions.CORE, 1)
        assert sessions.classify_time("2026-11-13 11:25:00", "start") == (sessions.CORE, 24)
        assert sessions.classify_time("2026-11-13 13:00:00", "start") == (sessions.CORE, 25)
        assert sessions.classify_time("2026-11-13 14:55:00", "start") == (sessions.CORE, 48)
        assert sessions.classify_time("2026-11-13 15:00:00", "start") == (sessions.OFF_GRID, None)

    def test_session_complete_requires_k1_to_k47(self) -> None:
        slots = make_slots({k: 10.0 for k in range(1, 49)})
        complete, missing = sessions.session_complete(slots)
        assert complete and missing == []
        slots.loc[20, ["C", "H", "L", "Q", "A"]] = [np.nan] * 5
        complete, missing = sessions.session_complete(slots)
        assert not complete and missing == [20]

    def test_closing_auction_regime_boundary(self) -> None:
        assert sessions.closing_auction_regime("20180817") == "continuous_to_close"
        assert sessions.closing_auction_regime("20180820") == "sse_closing_auction"

    def test_slot_labels(self) -> None:
        assert sessions.slot_label(1) == "09:35:00"
        assert sessions.slot_label(24) == "11:30:00"
        assert sessions.slot_label(25) == "13:05:00"
        assert sessions.slot_label(48) == "15:00:00"

    def test_normalize_day_separates_auction(self) -> None:
        rows = [
            {
                "ts_code": "600000.SH",
                "trade_time": "2026-11-13 09:30:00",
                "open": 10.0,
                "close": 10.0,
                "high": 10.0,
                "low": 10.0,
                "vol": 500.0,
                "amount": 5000.0,
            },
            {
                "ts_code": "600000.SH",
                "trade_time": "2026-11-13 09:35:00",
                "open": 10.0,
                "close": 10.1,
                "high": 10.1,
                "low": 10.0,
                "vol": 100.0,
                "amount": 1000.0,
            },
        ]
        normalized = sessions.normalize_day(pd.DataFrame(rows), convention="end")
        assert normalized["counts"][sessions.AUCTION] == 1
        assert normalized["auction"]["close"] == pytest.approx(10.0)
        assert normalized["slots"].loc[1, "C"] == pytest.approx(10.1)


class TestFeatures:
    def test_path_order_differs_with_identical_daily_ohlc(self) -> None:
        up_down = path_up_then_down()
        down_up = path_down_then_up()
        assert up_down["H"][2:48].max() == pytest.approx(down_up["H"][2:48].max())
        assert up_down["L"][2:48].min() == pytest.approx(down_up["L"][2:48].min())
        left = features.day_metrics(up_down)
        right = features.day_metrics(down_up)
        assert left["m_low_time"] == pytest.approx((40 - 2) / 45)
        assert right["m_low_time"] == pytest.approx((3 - 2) / 45)
        assert left["m_path_dd"] > right["m_path_dd"]

    def test_lunch_gap_not_in_rv(self) -> None:
        closes = {k: 10.0 for k in range(1, 25)}
        closes.update({k: 11.0 for k in range(25, 48)})
        metrics = features.day_metrics(make_slots(closes))
        assert metrics["m_am_ret"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["m_pm_ret"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["m_rv"] == pytest.approx(0.0, abs=1e-12)

    def test_zero_volume_slot_is_valid_observation(self) -> None:
        slots = make_slots({k: 10.0 + 0.001 * k for k in range(1, 49)})
        slots.loc[10, ["Q", "A"]] = 0.0
        complete, _missing = sessions.session_complete(slots)
        row = features.minute_feature_row(slots, [])
        assert complete and row["session_complete"]
        assert np.isfinite(row["m_rv"])
        assert np.isfinite(row["m_amount_hhi"])

    def test_missing_slot_flags_incomplete_and_not_informed(self) -> None:
        slots = make_slots({k: 10.0 + 0.001 * k for k in range(1, 49)})
        slots.loc[30, ["C", "H", "L", "Q", "A"]] = [np.nan] * 5
        row = features.minute_feature_row(slots, [])
        assert not row["session_complete"]
        assert row["minute_status"] == "session_incomplete"
        assert row["minute_informed"] is False
        assert row["missing_slots"] == "30"

    def test_zero_denominators_return_nan(self) -> None:
        slots = make_slots({k: 10.0 for k in range(1, 49)})
        for k in range(2, 48):
            slots.loc[k, "L"] = 10.0
            slots.loc[k, "H"] = 10.0
        metrics = features.day_metrics(slots)
        assert np.isnan(metrics["m_repair_fraction"])
        assert np.isnan(metrics["m_down_var_share"])
        assert np.isnan(metrics["m_core_amount_surprise"])

    def test_cross_day_requires_16_valid_history(self) -> None:
        today = {"q_late": 0.4, "A_core": 2000.0}
        short = [{"q_late": 0.3, "A_core": 1000.0}] * 15
        assert np.isnan(features.cross_day_features(today, short)["m_late_amount_surprise"])
        full = [{"q_late": 0.3, "A_core": 1000.0}] * 16
        out = features.cross_day_features(today, full)
        assert out["m_late_amount_surprise"] == pytest.approx(0.1)
        assert out["m_core_amount_surprise"] == pytest.approx(np.log(2.0))

    def test_feature_order_is_24_columns(self) -> None:
        assert len(spec.FEATURES) == 24
        assert spec.FEATURES[:6] == spec.DAILY_FEATURES
        assert spec.FEATURES[6:] == spec.MINUTE_FEATURES
        assert spec.MINUTE_FEATURES[0] == "m_am_ret"
        assert spec.MINUTE_FEATURES[-1] == "m_amount_hhi"

    def test_repair_fraction_real_out_of_range_raises(self) -> None:
        slots = make_slots({k: 10.0 for k in range(1, 49)}, high_at=10, low_at=30)
        slots.loc[47, "C"] = 20.0
        with pytest.raises(features.FeatureError):
            features.day_metrics(slots)


class TestLabels:
    def calendar(self) -> list[str]:
        days = [f"201904{day:02d}" for day in range(1, 31)]
        days += [f"201905{day:02d}" for day in range(1, 21)]
        return days

    def prices(self, calendar: list[str]) -> dict[str, pd.DataFrame]:
        close = pd.DataFrame({"AAA": np.linspace(10.0, 12.0, len(calendar))}, index=calendar)
        zeros = pd.DataFrame(0.0, index=calendar, columns=["AAA"])
        return {
            "close_adj": close,
            "open_adj": close.copy(),
            "is_suspended": zeros,
            "is_limit_up": zeros,
            "is_limit_down": zeros,
            "tradable": pd.DataFrame(1.0, index=calendar, columns=["AAA"]),
        }

    def test_horizon_uses_calendar_positions(self) -> None:
        calendar = self.calendar()
        prices = self.prices(calendar)
        raw, book = labels.build_labels(prices, calendar)
        expected = prices["open_adj"].iloc[11, 0] / prices["close_adj"].iloc[1, 0] - 1
        assert raw.iloc[0, 0] == pytest.approx(expected)
        assert book["label_status"].iloc[0, 0] == labels.STATUS_CODE["valid"]

    def test_tail_is_right_censored(self) -> None:
        calendar = self.calendar()
        _raw, book = labels.build_labels(self.prices(calendar), calendar)
        assert book["label_status"].iloc[-1, 0] == labels.STATUS_CODE["right_censored"]

    def test_industry_median_uses_full_base_domain(self) -> None:
        dates = ["20200102", "20200103"]
        codes = [f"S{i:02d}" for i in range(6)]
        raw = pd.DataFrame([[0.10, 0.20, 0.30, 0.40, 0.50, 0.90]] * 2, index=dates, columns=codes)
        status = pd.DataFrame(labels.STATUS_CODE["valid"], index=dates, columns=codes)
        base = np.ones((2, 6), dtype=bool)
        buckets = np.ones((2, 6), dtype=np.int16)
        neutral, median = labels.industry_neutralize(raw, status, base, buckets, min_members=6)
        assert median[0, 0] == pytest.approx(0.35)
        assert neutral[0, 5] == pytest.approx(0.55)
        short = np.zeros((2, 6), dtype=bool)
        short[:, :4] = True
        _neutral_short, median_short = labels.industry_neutralize(
            raw, status, short, buckets, min_members=5
        )
        assert np.isnan(median_short).all()

    def test_entry_and_exit_suspension_are_distinct(self) -> None:
        calendar = self.calendar()
        prices = self.prices(calendar)
        prices["is_suspended"].iloc[1, 0] = 1.0
        _raw, book = labels.build_labels(prices, calendar)
        assert book["label_status"].iloc[0, 0] == labels.STATUS_CODE["entry_suspended"]
        prices2 = self.prices(calendar)
        prices2["is_suspended"].iloc[11, 0] = 1.0
        _raw2, book2 = labels.build_labels(prices2, calendar)
        assert book2["label_status"].iloc[0, 0] == labels.STATUS_CODE["exit_suspended"]


class TestSplits:
    def calendar(self) -> list[str]:
        return [day.strftime("%Y%m%d") for day in pd.bdate_range("2015-01-05", "2026-10-29")]

    def test_sixteen_folds_and_train_window(self) -> None:
        calendar = self.calendar()
        folds = splits.build_folds(calendar, "20261029")
        splits.assert_fold_count(folds)
        first = folds[0]
        assert first["C"] >= "20190401"
        assert first["L"] < first["V"] < first["C"]
        months = (
            pd.Timestamp(f"{first['V'][:4]}-{first['V'][4:6]}-{first['V'][6:]}")
            - pd.Timestamp(f"{first['L'][:4]}-{first['L'][4:6]}-{first['L'][6:]}")
        ).days / 30.44
        assert months == pytest.approx(30.0, abs=1.5)
        assert folds[-1]["oos_last"] == "20261029"

    def test_sampling_phase_is_global(self) -> None:
        calendar = self.calendar()
        sample = splits.sampled_dates(calendar)
        picked = [day for day in calendar if day in sample][:5]
        positions = [calendar.index(day) for day in picked]
        assert positions == [positions[0] + 5 * i for i in range(5)]

    def test_purge_is_strict(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date": ["20200101", "20200102"],
                "label_end_date": ["20200110", "20200111"],
                "label_status": ["valid", "valid"],
            }
        )
        assert splits.purge_mask(frame, "20200111").tolist() == [True, False]
        train, _val = splits.split_masks(
            frame,
            {
                "train_days": ["20200101"],
                "val_days": ["20200102"],
                "V": "20200111",
                "C": "20200120",
            },
        )
        assert bool(train.iloc[0]) is True
        assert not bool(train.iloc[1])

    def test_oos_and_deploy_masks(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date": ["20200110", "20200111", "20200112"],
                "label_end_date": ["20200111", "20200112", "20200113"],
                "label_status": ["valid", "valid", "valid"],
            }
        )
        fold = {
            "train_days": ["20200110"],
            "val_days": ["20200111"],
            "oos_days": ["20200112"],
            "V": "20200110",
            "C": "20200112",
        }
        deploy = splits.deploy_mask(frame, fold)
        assert deploy.tolist() == [True, False, False]
        assert splits.oos_mask(frame, fold).tolist() == [False, False, True]


class TestDomain:
    DATES = [day.strftime("%Y%m%d") for day in pd.bdate_range("2019-11-01", "2020-01-03")]

    def build(self, limit_up: bool = False):
        dates = self.DATES
        codes = ["600000.SH", "600001.SH", "600002.SH", "000001.SZ", "600003.SH"]
        close = pd.DataFrame(10.0, index=dates, columns=codes)
        prices = {
            "close_adj": close,
            "open_adj": close.copy(),
            "amount": pd.DataFrame(1e6, index=dates, columns=codes),
            "is_st": pd.DataFrame(0.0, index=dates, columns=codes),
            "is_suspended": pd.DataFrame(0.0, index=dates, columns=codes),
            "is_limit_up": pd.DataFrame(0.0, index=dates, columns=codes),
            "is_limit_down": pd.DataFrame(0.0, index=dates, columns=codes),
            "tradable": pd.DataFrame(1.0, index=dates, columns=codes),
        }
        if limit_up:
            prices["is_limit_up"].iloc[0, 0] = 1.0
        basic = {
            "total_mv": pd.DataFrame(
                [[5e6, 1e7, 1.5e8, 1e7, 1e7]] * len(dates), index=dates, columns=codes
            )
        }
        stock_basic = pd.DataFrame(
            {
                "ts_code": codes,
                "market": ["主板"] * 5,
                "list_date": ["20000101"] * 5,
            }
        )
        pit = pd.DataFrame(
            {
                "ts_code": codes,
                "l1_code": ["410000", "410000", "410000", "480000", "99999999"],
                "l1_name": ["电子", "电子", "电子", "银行", None],
                "in_date": ["20100101"] * 5,
                "out_date": [None] * 5,
            }
        )
        calendar = [day.strftime("%Y%m%d") for day in pd.bdate_range("2016-01-01", "2020-01-03")]
        result = domain.build_domain(
            prices,
            basic,
            stock_basic,
            pit,
            listing_calendar=calendar,
            l1_names=domain.industry_l1_names(None, pit),
        )
        return result, codes

    def test_limit_up_at_t_not_excluded(self) -> None:
        result, codes = self.build(limit_up=True)
        last = len(result.dates) - 1
        assert result.eligible[last, codes.index("600000.SH")]
        assert result.limit_up[0, codes.index("600000.SH")]

    def test_market_cap_bounds_inclusive(self) -> None:
        result, codes = self.build()
        last = len(result.dates) - 1
        assert result.eligible[last, codes.index("600000.SH")]  # 500 亿端点
        assert not result.eligible[last, codes.index("600002.SH")]  # 15000 亿超上限
        assert result.reason[last, codes.index("600002.SH")] == domain.REASONS.index(
            "market_cap_out_of_range"
        )

    def test_financial_and_unknown_industry_excluded(self) -> None:
        result, codes = self.build()
        assert not result.eligible[0, codes.index("000001.SZ")]
        assert not result.eligible[0, codes.index("600003.SH")]
        assert result.reason[0, codes.index("000001.SZ")] == domain.REASONS.index(
            "financial_industry"
        )
        assert result.reason[0, codes.index("600003.SH")] == domain.REASONS.index(
            "industry_classification_unknown"
        )

    def test_delisted_stock_kept_in_historical_domain(self) -> None:
        result, codes = self.build()
        index = codes.index("600001.SH")
        # 前 15 个交易日不足 20 日成交额窗口；其后必须持续合格（此后退市也不倒删历史）
        assert result.eligible[15:, index].all()

    def test_amount20_missing_when_all_nan(self) -> None:
        dates = self.DATES
        codes = ["600000.SH"]
        close = pd.DataFrame(10.0, index=dates, columns=codes)
        prices = {
            "close_adj": close,
            "open_adj": close.copy(),
            "amount": pd.DataFrame(np.nan, index=dates, columns=codes),
            "is_st": pd.DataFrame(0.0, index=dates, columns=codes),
            "is_suspended": pd.DataFrame(0.0, index=dates, columns=codes),
            "is_limit_up": pd.DataFrame(0.0, index=dates, columns=codes),
            "is_limit_down": pd.DataFrame(0.0, index=dates, columns=codes),
            "tradable": pd.DataFrame(1.0, index=dates, columns=codes),
        }
        basic = {"total_mv": pd.DataFrame(1e7, index=dates, columns=codes)}
        stock_basic = pd.DataFrame(
            {"ts_code": codes, "market": ["主板"], "list_date": ["20000101"]}
        )
        pit = pd.DataFrame(
            {
                "ts_code": codes,
                "l1_code": ["410000"],
                "l1_name": ["电子"],
                "in_date": ["20100101"],
                "out_date": [None],
            }
        )
        calendar = [day.strftime("%Y%m%d") for day in pd.bdate_range("2016-01-01", "2020-01-03")]
        result = domain.build_domain(
            prices,
            basic,
            stock_basic,
            pit,
            listing_calendar=calendar,
            l1_names=domain.industry_l1_names(None, pit),
        )
        assert result.reason[-1, 0] == domain.REASONS.index("amount20_missing")
