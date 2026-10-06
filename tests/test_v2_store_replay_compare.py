# -*- coding: utf-8 -*-
"""v2 重放对账器测试（tmp_path 合成双分区，不触碰真实 data/）。

覆盖：完全一致（零差异）、行集差集、列差（仅单侧列登记，键列噪声源头剔除）、
超容差数值差、atol 边界、NaN==NaN 语义、单侧 NaN 计数、全 NaN 型差异 max|Δ| 显式归零、
非数值列精确相等、归因字段（D-04 命中/派生连锁/待调查）、over_gate 判定
（占比与 max|Δ| 双判据）、R3-04 全窗口分母（含零差异日 / 列部分日不可比）、
样例上限、missing 日登记。
"""

import pandas as pd
import pytest

from src.lazybull.v2.store.replay_compare import (
    GATE_MAX_ABS_DIFF,
    attribute_column,
    compare_partitions,
)

_CODES = ["600000.SH", "000001.SZ", "000002.SZ"]


def _write_day(root, date: str, df: pd.DataFrame) -> None:
    part = root / "features" / "cs_train"
    part.mkdir(parents=True, exist_ok=True)
    df.to_parquet(part / f"{date}.parquet", index=False)


def _day(
    date: str, close: list, name: list | None = None, extra: dict | None = None
) -> pd.DataFrame:
    n = len(close)
    data = {
        "trade_date": [date] * n,
        "ts_code": _CODES[:n],
        "close": close,
        "sw_industry": (name or ["银行"] * n)[:n],
    }
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


@pytest.fixture
def two_roots(tmp_path):
    replay = tmp_path / "replay"
    cs = tmp_path / "cs"
    return replay, cs


class TestIdentical:
    def test_zero_diff(self, two_roots):
        replay, cs = two_roots
        df = _day("20240102", [10.0, 20.0, 30.0], extra={"fund_count": [1.0, None, 3.0]})
        _write_day(replay, "20240102", df)
        _write_day(cs, "20240102", df)
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        assert report["summary"]["columns_with_diff"] == 0
        assert report["days"]["compared"] == 1
        assert report["days"]["row_set_diff_days"] == []


class TestDifferences:
    def test_row_set_diff(self, two_roots):
        replay, cs = two_roots
        _write_day(replay, "20240102", _day("20240102", [10.0, 20.0]))  # 少一股
        _write_day(cs, "20240102", _day("20240102", [10.0, 20.0, 30.0]))
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        diff_days = report["days"]["row_set_diff_days"]
        assert len(diff_days) == 1
        assert diff_days[0]["n_only_cs_train"] == 1
        assert diff_days[0]["sample_only_cs_train"] == ["000002.SZ"]

    def test_col_only_one_side(self, two_roots):
        replay, cs = two_roots
        _write_day(replay, "20240102", _day("20240102", [10.0, 20.0, 30.0]))
        _write_day(
            cs, "20240102", _day("20240102", [10.0, 20.0, 30.0], extra={"new_col": [1, 2, 3]})
        )
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        assert report["cols_only_cs_train"] == ["new_col"]
        assert report["cols_only_replay"] == []

    def test_key_columns_excluded_from_cols_only(self, two_roots):
        """键列噪声源头剔除：一侧缺 trade_date 列也不登记进 cols_only_*。"""
        replay, cs = two_roots
        df_replay = pd.DataFrame(
            {"ts_code": _CODES, "close": [10.0, 20.0, 30.0], "sw_industry": ["银行"] * 3}
        )
        _write_day(replay, "20240102", df_replay)  # 无 trade_date 列
        _write_day(cs, "20240102", _day("20240102", [10.0, 20.0, 30.0]))
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        assert report["cols_only_replay"] == []
        assert report["cols_only_cs_train"] == []  # trade_date 被剔除，不登记

    def test_numeric_over_tolerance(self, two_roots):
        replay, cs = two_roots
        _write_day(replay, "20240102", _day("20240102", [10.0, 20.0, 30.0]))
        _write_day(cs, "20240102", _day("20240102", [10.0, 20.5, 30.0]))  # 1 行超容差
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        close = report["columns"]["close"]
        assert close["over_rows"] == 1
        assert close["max_abs_diff"] == pytest.approx(0.5)
        assert close["over_gate"] is True  # max|Δ|=0.5 > 0.05
        assert close["attribution"] == "待调查"
        assert close["samples"][0]["ts_code"] == "000001.SZ"

    def test_numeric_within_atol_passes(self, two_roots):
        replay, cs = two_roots
        _write_day(replay, "20240102", _day("20240102", [10.0, 20.0, 30.0]))
        _write_day(cs, "20240102", _day("20240102", [10.0, 20.0 + 5e-7, 30.0]))
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        assert report["summary"]["columns_with_diff"] == 0

    def test_nan_semantics(self, two_roots):
        replay, cs = two_roots
        _write_day(
            replay,
            "20240102",
            _day("20240102", [10.0, None, 30.0], extra={"fund_count": [1, 2, 3]}),
        )
        _write_day(
            cs, "20240102", _day("20240102", [10.0, None, 31.0], extra={"fund_count": [1, None, 3]})
        )
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        close = report["columns"]["close"]  # 000002 30 vs 31；000001 双 NaN 一致
        assert close["over_rows"] == 1
        fund = report["columns"]["fund_count"]  # 000001 单侧 NaN ⇒ 计数 1
        assert fund["over_rows"] == 1
        assert fund["max_abs_diff"] == 0.0  # NaN 型差异无数值差
        assert fund["attribution"].startswith("D-04③")

    def test_all_nan_type_diff_max_abs_zero(self, two_roots):
        """over 行全为 NaN 型差异 ⇒ max|Δ| 显式 0.0（不依赖 pandas .max() 的 nan 侥幸）。"""
        replay, cs = two_roots
        _write_day(
            replay,
            "20240102",
            _day("20240102", [10.0, 20.0, 30.0], extra={"fund_count": [1.0, 2.0, 3.0]}),
        )
        _write_day(
            cs,
            "20240102",
            _day("20240102", [10.0, 20.0, 30.0], extra={"fund_count": [None, None, None]}),
        )
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        fund = report["columns"]["fund_count"]
        assert fund["over_rows"] == 3
        assert fund["max_abs_diff"] == 0.0
        assert not pd.isna(fund["max_abs_diff"])

    def test_nonnumeric_equality(self, two_roots):
        replay, cs = two_roots
        _write_day(replay, "20240102", _day("20240102", [10.0, 20.0, 30.0]))
        _write_day(
            cs, "20240102", _day("20240102", [10.0, 20.0, 30.0], name=["银行", "银行", "医药"])
        )
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        assert report["columns"]["sw_industry"]["over_rows"] == 1
        assert report["columns"]["sw_industry"]["max_abs_diff"] == 0.0

    def test_missing_day_registered(self, two_roots):
        replay, cs = two_roots
        _write_day(cs, "20240102", _day("20240102", [10.0, 20.0, 30.0]))
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        assert report["days"]["missing"][0]["date"] == "20240102"
        assert report["days"]["compared"] == 0

    def test_sample_cap(self, two_roots):
        replay, cs = two_roots
        codes = [f"00000{i}.SZ" for i in range(8)]
        base = pd.DataFrame({"trade_date": "20240102", "ts_code": codes, "close": range(8)})
        _write_day(replay, "20240102", base)
        _write_day(cs, "20240102", base.assign(close=[x + 10 for x in range(8)]))
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        assert len(report["columns"]["close"]["samples"]) == 5  # 上限 5

    def test_share_gate_without_max(self, two_roots):
        """max|Δ| 小但占比超 1e-4 ⇒ over_gate（占比判据独立生效）。"""
        replay, cs = two_roots
        codes = [f"{600000 + i}.SH" for i in range(2000)]
        base = pd.DataFrame({"trade_date": "20240102", "ts_code": codes, "close": [10.0] * 2000})
        perturbed = base.copy()
        perturbed.loc[0, "close"] = 10.0 + 1e-3  # 1/2000 = 5e-4 > 1e-4，max < 0.05
        _write_day(replay, "20240102", base)
        _write_day(cs, "20240102", perturbed)
        report = compare_partitions(replay, cs, ["20240102"], n_jobs=1)
        close = report["columns"]["close"]
        assert close["over_share"] == pytest.approx(5e-4)
        assert close["max_abs_diff"] < GATE_MAX_ABS_DIFF
        assert close["over_gate"] is True


class TestFullWindowDenominator:
    """R3-04 整改：over_share 分母 = 全窗口该列实际可比日的可比行数合计（含零差异日）。"""

    def test_over_share_uses_full_window(self, two_roots):
        """反例：20 日 × 5000 行仅 1 日 1 格差 0.01 ⇒ 正确占比 1e-5（过 1e-4 门）；
        旧口径分母只算差异日 ⇒ 2e-4 误报超门。"""
        replay, cs = two_roots
        dates = [f"202401{d:02d}" for d in range(1, 21)]  # 20240101~20240120
        codes = [f"{600000 + i}.SH" for i in range(5000)]
        for date in dates:
            base = pd.DataFrame({"trade_date": date, "ts_code": codes, "close": [10.0] * 5000})
            _write_day(replay, date, base)
            cs_day = base.copy()
            if date == "20240110":
                cs_day.loc[0, "close"] = 10.01  # 单格差 0.01
            _write_day(cs, date, cs_day)
        report = compare_partitions(replay, cs, dates, n_jobs=1)
        close = report["columns"]["close"]
        assert close["over_rows"] == 1
        assert close["total_rows"] == 20 * 5000  # 全窗口分母（含 19 个零差异日）
        assert close["over_share"] == pytest.approx(1e-5)
        assert close["over_gate"] is False  # 1e-5 ≤ 1e-4 且 max|Δ|=0.01 ≤ 0.05

    def test_denominator_counts_only_comparable_days(self, two_roots):
        """列在部分日不可比（不在当日 common_cols）⇒ 分母只计实际可比日。"""
        replay, cs = two_roots
        extra = {"extra_col": [1.0, 2.0, 3.0]}
        _write_day(replay, "20240102", _day("20240102", [10.0, 20.0, 30.0], extra=extra))
        _write_day(
            cs,
            "20240102",
            _day("20240102", [10.0, 20.0, 30.0], extra={"extra_col": [1.0, 2.5, 3.0]}),
        )
        # 第 2 日：仅 replay 侧有 extra_col ⇒ 当日该列不可比
        _write_day(replay, "20240103", _day("20240103", [10.0, 20.0, 30.0], extra=extra))
        _write_day(cs, "20240103", _day("20240103", [10.0, 20.0, 30.0]))
        report = compare_partitions(replay, cs, ["20240102", "20240103"], n_jobs=1)
        col = report["columns"]["extra_col"]
        assert col["over_rows"] == 1
        assert col["total_rows"] == 3  # 只计 20240102 可比日
        assert col["over_share"] == pytest.approx(1 / 3)


class TestAttribution:
    def test_d04_direct_hits(self):
        assert attribute_column("unlock_ratio").startswith("D-04①")
        assert attribute_column("lhb_net_amount").startswith("D-04①")
        assert attribute_column("block_discount_days_10d").startswith("D-04①")
        assert attribute_column("downside_corr_20").startswith("D-04②")
        assert attribute_column("dividend_payout_ratio").startswith("D-04③")
        assert attribute_column("fund_hold_ratio").startswith("D-04③")

    def test_derived_follows_parent(self):
        assert "派生连锁" in attribute_column("zscore_dividend_payout_ratio")
        assert "派生连锁" in attribute_column("zscore_ocf_to_profit_sz")
        assert attribute_column("zscore_pb") == "待调查"

    def test_unknown_is_pending(self):
        assert attribute_column("some_mystery_col") == "待调查"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
