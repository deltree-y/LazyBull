# -*- coding: utf-8 -*-
"""分红政策质量因子测试：PIT 契约（防前视）、送转调整、缺失语义、handler 展开、下载去重。"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.lazybull.data import DataLoader
from src.lazybull.data.dividend_raw import (
    _deduplicate_dividend,
    _load_dividend_coverage,
    download_dividend_full,
)
from src.lazybull.data.loader_dividend import DividendLoaderMixin
from src.lazybull.data.storage import Storage
from src.lazybull.factors.dividend import (
    DIVIDEND_FRESHNESS_COL,
    DIVIDEND_HIST_MISSING_COL,
    DIVIDEND_POLICY_SCHEMA_VERSION,
    DIVIDEND_POLICY_VERSION_COL,
    build_dividend_lookup_by_date,
)
from src.lazybull.features.ensure.downloads import _try_download_dividend
from src.lazybull.features.ensure.schema import (
    _REQUIRED_FACTOR_COLS,
    OPTIONAL_FACTOR_GROUP_DIVIDEND_POLICY,
    _check_features_schema,
)
from src.lazybull.features.factor_handlers import DividendPolicyFactorHandler
from src.lazybull.ml.train_core.constants import DIVIDEND_POLICY_FEATURE_COLUMNS

_TRADING_DATES = [
    "20240102",
    "20240103",
    "20240104",
    "20240105",
    "20240108",
    "20240109",
    "20240110",
    "20240111",
    "20240112",
    "20240115",
    "20240116",
    "20240117",
]


def _event_row(
    ts_code="000001.SZ",
    end_date="20231231",
    ann_date="20240401",
    imp_ann_date="20240601",
    ex_date="20240610",
    cash_div_tax=0.5,
    stk_div=0.0,
    div_proc="实施",
    base_share=10000.0,
):
    return {
        "ts_code": ts_code,
        "end_date": end_date,
        "ann_date": ann_date,
        "imp_ann_date": imp_ann_date,
        "ex_date": ex_date,
        "record_date": None,
        "pay_date": None,
        "div_proc": div_proc,
        "cash_div": 0.45,
        "cash_div_tax": cash_div_tax,
        "stk_div": stk_div,
        "stk_bo_rate": None,
        "stk_co_rate": None,
        "base_date": None,
        "base_share": base_share,
    }


# ═══════════════════════════════════════════════════════════════
# PIT：ex_date 前不可见 / 非实施行剔除
# ═══════════════════════════════════════════════════════════════


def test_lookup_ex_date_pit_dual_date_semantics():
    """状态因子 ex_date 前不可见；事件因子 imp_ann_date 公告后即合法可见。

    事件 ex_date=20240610、imp_ann_date=20240601：
      - 公告前（imp_ann > T）：无任何可见事件，lookup 无该股行；
      - 公告后除息前（imp_ann <= T < ex_date）：输出行，days_to_ex 有值，
        状态因子（continuity 等）为 NaN（该年度分红尚未落地）；
      - 除息后（ex_date <= T）：状态因子可见。
    """
    raw = pd.DataFrame([_event_row(ex_date="20240610", imp_ann_date="20240601")])
    dates = ["20240531", "20240607", "20240610", "20240611"]
    lookup = build_dividend_lookup_by_date(raw, dates, list_date_map={"000001.SZ": "20150101"})
    # 公告前不可见
    assert "20240531" not in lookup
    # 公告后、除息前：事件因子可见，状态因子 NaN
    assert "20240607" in lookup
    row_pre = lookup["20240607"].iloc[0]
    assert row_pre["dividend_days_to_ex_date"] == pytest.approx(3)
    assert row_pre["dividend_recent_imp_ann_10d"] >= 1
    assert np.isnan(row_pre["dividend_continuity_5y"])
    assert np.isnan(row_pre[DIVIDEND_FRESHNESS_COL])
    assert np.isnan(row_pre[DIVIDEND_HIST_MISSING_COL])

    # 首次分红尚未除息，不得提前改写“从未分红”状态
    features = pd.DataFrame({"ts_code": ["000001.SZ"], "list_days": [3000]})
    current = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})
    handled = DividendPolicyFactorHandler().apply(features, lookup["20240607"], "20240607", current)
    assert handled[DIVIDEND_HIST_MISSING_COL].iloc[0] == 1.0
    assert handled["dividend_continuity_5y"].iloc[0] == 0.0
    assert handled["dividend_days_to_ex_date"].iloc[0] == pytest.approx(3.0)
    # 除息当日及之后：状态因子可见
    assert "20240610" in lookup
    row = lookup["20240610"].iloc[0]
    assert row[DIVIDEND_FRESHNESS_COL] == 0
    assert not np.isnan(row["dividend_continuity_5y"])


def test_lookup_excludes_non_executed_rows():
    """预案/决案行不参与因子；仅实施行生效。"""
    raw = pd.DataFrame(
        [
            _event_row(div_proc="预案", ex_date=None, imp_ann_date=None),
            _event_row(div_proc="实施", ex_date="20240610", imp_ann_date="20240601"),
        ]
    )
    raw.loc[0, "ex_date"] = None
    lookup = build_dividend_lookup_by_date(raw, ["20240610", "20240611"])
    assert "20240611" in lookup
    assert len(lookup["20240611"]) == 1


def test_lookup_empty_input():
    assert build_dividend_lookup_by_date(None, ["20240101"]) == {}
    assert build_dividend_lookup_by_date(pd.DataFrame(), ["20240101"]) == {}


def test_batch_lookup_vectorizes_by_stock_and_date_chunk(monkeypatch):
    """批量 lookup 调用量应为股票数×日期块数，而不是股票数×日期数。"""
    import src.lazybull.factors.dividend as dividend_module

    raw = pd.DataFrame(
        [
            _event_row(ts_code="000001.SZ", ex_date="20230610", imp_ann_date="20230601"),
            _event_row(ts_code="000002.SZ", ex_date="20230610", imp_ann_date="20230601"),
        ]
    )
    dates = pd.bdate_range("20240102", periods=65).strftime("%Y%m%d").tolist()
    original = dividend_module._stock_rows_for_dates
    calls = []

    def _counting_vectorized(*args, **kwargs):
        calls.append(len(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(dividend_module, "_stock_rows_for_dates", _counting_vectorized)
    lookup = build_dividend_lookup_by_date(raw, dates)

    assert len(lookup) == 65
    assert calls == [64, 64, 1, 1]


def test_dense_event_features_enter_training_without_zscore():
    """稀有事件的横截面 zscore 常因零方差全 NaN，训练应使用原始稠密编码。"""
    assert "dividend_days_to_ex_date" in DIVIDEND_POLICY_FEATURE_COLUMNS
    assert "dividend_recent_imp_ann_10d" in DIVIDEND_POLICY_FEATURE_COLUMNS
    assert "zscore_dividend_days_to_ex_date" not in DIVIDEND_POLICY_FEATURE_COLUMNS
    assert "zscore_dividend_recent_imp_ann_10d" not in DIVIDEND_POLICY_FEATURE_COLUMNS


# ═══════════════════════════════════════════════════════════════
# 送转调整（每股口径）与增长率
# ═══════════════════════════════════════════════════════════════


def test_stk_div_adjustment_and_growth():
    """送转后历史每股分红按当前股本口径（前复权式）缩小；窗口为成熟财年。

    2020 财年每股派 1.0 且 10 送 10（stk_div=1.0），2021 财年每股派 1.0 无送转：
    当前股本口径 base：2020 = 1.0×G_before(1) = 1.0；2021 = 1.0×G_before(2) = 2.0。
    T=20240610 → 成熟财年 Y=2022（2023 财年 2024-09-01 才成熟），窗口 [2018, 2022]：
      continuity = 2/5（2020/2021 正分红）；
      growth_3y = 2022(0) vs 2019(0) → 分母 0 → NaN。
    """
    raw = pd.DataFrame(
        [
            _event_row(
                end_date="20201231",
                ex_date="20210610",
                imp_ann_date="20210601",
                cash_div_tax=1.0,
                stk_div=1.0,
            ),
            _event_row(
                end_date="20211231",
                ex_date="20220610",
                imp_ann_date="20220601",
                cash_div_tax=1.0,
                stk_div=0.0,
            ),
        ]
    )
    dates = ["20240610"]
    lookup = build_dividend_lookup_by_date(raw, dates, list_date_map={"000001.SZ": "20150101"})
    assert "20240610" in lookup
    row = lookup["20240610"].iloc[0]
    assert row["dividend_continuity_5y"] == pytest.approx(2 / 5)
    assert np.isnan(row["dividend_growth_3y"])


def test_future_stk_div_does_not_affect_earlier_cross_section():
    """审查反例：未来送转不得影响历史截面（PIT 截断）。

    2024 年送转（stk_div=1.0）在 T=20240110 尚未发生：当日近 12 月累计
    每股现金（yield 分子）不因未来送转被放大。
    """
    raw = pd.DataFrame(
        [
            _event_row(
                end_date="20221231",
                ex_date="20230701",
                imp_ann_date="20230620",
                cash_div_tax=1.0,
                stk_div=0.0,
            ),
            _event_row(
                end_date="20231231",
                ex_date="20240610",
                imp_ann_date="20240601",
                cash_div_tax=0.0,
                stk_div=1.0,  # 未来送转（T=20240110 时未发生）
            ),
        ]
    )
    dates = ["20240110", "20240611"]
    lookup = build_dividend_lookup_by_date(raw, dates, list_date_map={"000001.SZ": "20150101"})
    row_before = lookup["20240110"].iloc[0]
    row_after = lookup["20240611"].iloc[0]
    # 当前股本口径为前复权式：T=0110 时未来送转（ex=0610）未发生，G(T)=1 → 分子=1.0
    assert row_before["dividend_cash_12m_adj"] == pytest.approx(1.0)
    # 旧现金仍在 365 天窗口内；纯送转落地后 G(T)=2，历史每股分红折半
    assert row_after["dividend_cash_12m_adj"] == pytest.approx(0.5)


def test_missing_imp_ann_date_falls_back_to_ann_date():
    """审查反例：缺 imp_ann_date 的实施行回退 ann_date 后仍可见，不被删除。"""
    raw = pd.DataFrame(
        [
            _event_row(
                end_date="20221231",
                ex_date="20230610",
                imp_ann_date=None,
                ann_date="20230601",
                cash_div_tax=1.0,
            ),
        ]
    )
    lookup = build_dividend_lookup_by_date(raw, ["20230605", "20230611"])
    # 公告（回退 ann_date=0601）后、除息（0610）前：事件因子可见
    assert "20230605" in lookup
    assert lookup["20230605"].iloc[0]["dividend_days_to_ex_date"] == pytest.approx(5)
    assert "20230611" in lookup


def test_window_does_not_jump_across_year_boundary():
    """审查反例：无新公告时，跨年初因子不跳变。

    2022 财年分红 ex=20230610；2023 财年无分红。
    T=20240102 与 T=20231229 的成熟财年均为 Y=2022 → 连续性相同。
    """
    raw = pd.DataFrame(
        [
            _event_row(
                end_date="20221231",
                ex_date="20230610",
                imp_ann_date="20230601",
                cash_div_tax=1.0,
            ),
        ]
    )
    lookup = build_dividend_lookup_by_date(
        raw, ["20231229", "20240102"], list_date_map={"000001.SZ": "20150101"}
    )
    assert lookup["20231229"].iloc[0]["dividend_continuity_5y"] == pytest.approx(
        lookup["20240102"].iloc[0]["dividend_continuity_5y"]
    )
    assert lookup["20240102"].iloc[0]["dividend_growth_3y"] == pytest.approx(
        lookup["20231229"].iloc[0]["dividend_growth_3y"]
    )


def test_recent_imp_ann_window_uses_full_calendar_for_single_day():
    """审查反例：单日推理（仅传 T）时近 10 交易日窗口不退化，与批量一致。"""
    raw = pd.DataFrame(
        [
            _event_row(
                end_date="20221231",
                ex_date="20230610",
                imp_ann_date="20240103",
                cash_div_tax=1.0,
            ),
        ]
    )
    calendar = _TRADING_DATES  # 完整预热日历
    target = "20240110"
    single = build_dividend_lookup_by_date(
        raw, [target], list_date_map={"000001.SZ": "20150101"}, calendar_dates=calendar
    )
    batch = build_dividend_lookup_by_date(raw, calendar, list_date_map={"000001.SZ": "20150101"})
    assert single[target].iloc[0]["dividend_recent_imp_ann_10d"] == pytest.approx(
        batch[target].iloc[0]["dividend_recent_imp_ann_10d"]
    )
    assert batch[target].iloc[0]["dividend_recent_imp_ann_10d"] >= 1


def test_dividend_suspension_enters_window_as_zero():
    """审查反例：停发年份作为 0 进入成熟财年窗口，连续性不会永久保持 1.0。

    2018-2022 连续分红后停发（2023-2025 无实施记录）。
    T=20260301 → 成熟财年 Y=2024 → 窗口 [2020, 2024]：
    2020/2021/2022 正分红、2023/2024 停发=0 → continuity = 3/5。
    """
    rows = [
        _event_row(
            end_date=f"{y}1231",
            ex_date=f"{y + 1}0610",
            imp_ann_date=f"{y + 1}0601",
            cash_div_tax=1.0,
        )
        for y in range(2018, 2023)
    ]
    raw = pd.DataFrame(rows)
    lookup = build_dividend_lookup_by_date(
        raw, ["20260301"], list_date_map={"000001.SZ": "20150101"}
    )
    row = lookup["20260301"].iloc[0]
    assert row["dividend_continuity_5y"] == pytest.approx(3 / 5)


def test_executed_dividend_updates_annual_state_before_maturity_cutoff():
    """已实施正分红在 ex_date 立即进入年度状态，缺失年份仍到成熟后才记零。"""
    raw = pd.DataFrame(
        [
            _event_row(
                end_date="20211231",
                ex_date="20220610",
                imp_ann_date="20220601",
                cash_div_tax=1.0,
            ),
            _event_row(
                end_date="20221231",
                ex_date="20230610",
                imp_ann_date="20230601",
                cash_div_tax=1.0,
            ),
        ]
    )
    lookup = build_dividend_lookup_by_date(
        raw,
        ["20230609", "20230611"],
        list_date_map={"000001.SZ": "20150101"},
    )
    assert lookup["20230609"].iloc[0]["dividend_continuity_5y"] == pytest.approx(1 / 5)
    assert lookup["20230611"].iloc[0]["dividend_continuity_5y"] == pytest.approx(2 / 5)


def test_payout_uses_attributable_profit_and_revision_availability_date():
    """支付率只用归母净利润，未来修订仅在 f_ann_date 当日后生效。"""
    raw = pd.DataFrame(
        [
            _event_row(
                end_date="20221231",
                ex_date="20230610",
                imp_ann_date="20230601",
                cash_div_tax=0.5,
                base_share=10000.0,
            )
        ]
    )
    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["20230430", "20230430"],
            "f_ann_date": ["20230430", "20230701"],
            "end_date": ["20221231", "20221231"],
            "report_type": [1, 1],
            "update_flag": [1, 1],
            "n_income_attr_p": [100_000_000.0, 200_000_000.0],
            "net_profit": [25_000_000.0, 25_000_000.0],
        }
    )

    lookup = build_dividend_lookup_by_date(
        raw,
        ["20230630", "20230703"],
        income_raw=income,
        list_date_map={"000001.SZ": "20150101"},
    )

    assert lookup["20230630"].iloc[0]["dividend_payout_ratio"] == pytest.approx(0.5)
    assert lookup["20230703"].iloc[0]["dividend_payout_ratio"] == pytest.approx(0.25)


def test_same_fiscal_year_future_event_does_not_change_history():
    """审查反例：同财年未来第二次分红不得改变第一次分红后的历史截面。

    2022 财年两次实施：A ex=20230610、B ex=20231210。
    T=20231002（财年已成熟，A 可见、B 未发生）时，加入 B 前后因子值应一致
    （旧实现按年度 max ex 聚合会把年度可见性推迟到 B，污染历史截面）。
    """
    base_rows = [
        _event_row(
            end_date="20211231",
            ex_date="20220610",
            imp_ann_date="20220601",
            cash_div_tax=1.0,
        ),
        _event_row(
            end_date="20221231",
            ex_date="20230610",
            imp_ann_date="20230601",
            cash_div_tax=1.0,
        ),
    ]
    with_b = base_rows + [
        _event_row(
            end_date="20221231",
            ex_date="20231210",
            imp_ann_date="20231201",
            cash_div_tax=1.0,
        ),
    ]
    t_probe = "20231002"
    lookup_without = build_dividend_lookup_by_date(
        pd.DataFrame(base_rows), [t_probe], list_date_map={"000001.SZ": "20150101"}
    )
    lookup_with = build_dividend_lookup_by_date(
        pd.DataFrame(with_b), [t_probe], list_date_map={"000001.SZ": "20150101"}
    )
    row_wo = lookup_without[t_probe].iloc[0]
    row_w = lookup_with[t_probe].iloc[0]
    for col in (
        "dividend_continuity_5y",
        "dividend_stability_5y",
        "dividend_growth_3y",
        "dividend_growth_5y",
        "dividend_payout_ratio",
        "dividend_cash_12m_adj",
    ):
        left = row_wo[col]
        right = row_w[col]
        if np.isnan(left):
            assert np.isnan(right), f"{col} 因未来事件改变（NaN 不一致）"
        else:
            assert left == pytest.approx(right), f"{col} 因未来事件改变"


# ═══════════════════════════════════════════════════════════════
# 事件因子：days_to_ex_date 公告可见性 / recent_imp_ann 纯回看
# ═══════════════════════════════════════════════════════════════


def test_days_to_ex_date_requires_prior_announcement():
    """未来 ex_date 仅在实施公告日 <= T 时可见；未公告 → NaN。"""
    raw = pd.DataFrame(
        [
            _event_row(
                ex_date="20240115",
                imp_ann_date="20240105",
                end_date="20231231",
            ),
        ]
    )
    # 已公告（imp=0105 <= T=0110）→ days = 5
    lookup = build_dividend_lookup_by_date(raw, _TRADING_DATES)
    assert "20240110" in lookup
    assert lookup["20240110"].iloc[0]["dividend_days_to_ex_date"] == pytest.approx(5)
    # 公告前（T=0102 < imp=0105）→ 不可见
    assert "20240102" not in lookup

    raw2 = pd.DataFrame(
        [
            _event_row(
                ex_date="20240115",
                imp_ann_date="20240111",
                end_date="20231231",
            ),
        ]
    )
    lookup2 = build_dividend_lookup_by_date(raw2, _TRADING_DATES)
    # T=0110 时事件未公告（imp=0111 > T），lookup 无该股行 → days_to_ex NaN
    assert "20240110" not in lookup2


def test_recent_imp_ann_10d_lookback_only():
    """公告只影响发布日及之后 10 个交易日；发布前与滑出窗口后均为 0。

    公告 imp_ann_date=20240110：
      - 20240102~20240109（发布前）因子不存在（无可见事件 → 无行，handler 填 0）；
      - 20240110~20240123 窗口内 ≥1；
      - 20240124 及之后滑出窗口归 0（有历史事件仍输出行，计数=0）。
    """
    raw = pd.DataFrame(
        [
            _event_row(
                ex_date="20240210",
                imp_ann_date="20240110",
                end_date="20231231",
            ),
        ]
    )
    dates = [
        "20240102",
        "20240103",
        "20240104",
        "20240105",
        "20240108",
        "20240109",
        "20240110",
        "20240111",
        "20240112",
        "20240115",
        "20240116",
        "20240117",
        "20240118",
        "20240119",
        "20240122",
        "20240123",
        "20240124",
    ]
    lookup = build_dividend_lookup_by_date(raw, dates)
    # 发布前：无可见事件（ex=0210 尚未发生），lookup 无该股行
    for d in dates[:6]:
        assert d not in lookup
    # 发布日（盘后可见）与窗口内：计数 >= 1
    assert lookup["20240110"].iloc[0]["dividend_recent_imp_ann_10d"] >= 1
    assert lookup["20240123"].iloc[0]["dividend_recent_imp_ann_10d"] >= 1
    # 滑出窗口（T-9 = 0111 > 0110）：计数归 0
    assert lookup["20240124"].iloc[0]["dividend_recent_imp_ann_10d"] == 0


# ═══════════════════════════════════════════════════════════════
# handler：缺失语义展开 + 哨兵恒写
# ═══════════════════════════════════════════════════════════════


def _features_with_list_days(ts_codes, list_days_values):
    return pd.DataFrame({"ts_code": ts_codes, "list_days": list_days_values})


def _current_data(ts_codes, close_values):
    return pd.DataFrame({"ts_code": ts_codes, "close": close_values})


def test_handler_missing_semantics_expansion():
    """无分红历史：上市成熟 → continuity=0/hist_missing=1；未成熟 → NaN。哨兵恒写。"""
    handler = DividendPolicyFactorHandler()
    features = _features_with_list_days(["000001.SZ", "000002.SZ", "600000.SH"], [1000, 1000, 100])
    current = _current_data(["000001.SZ", "000002.SZ", "600000.SH"], [10.0, 20.0, 5.0])
    # lookup 仅有 000001（近期有分红）：
    data = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "dividend_continuity_5y": [1.0],
            "dividend_stability_5y": [0.9],
            "dividend_growth_3y": [0.5],
            "dividend_growth_5y": [0.4],
            "dividend_payout_ratio": [0.3],
            "dividend_cash_12m_adj": [0.6],
            "dividend_days_to_ex_date": [5.0],
            "dividend_recent_imp_ann_10d": [1.0],
            DIVIDEND_FRESHNESS_COL: [10.0],
            DIVIDEND_HIST_MISSING_COL: [0.0],
        }
    )
    result = handler.apply(features, data, "20240610", current)

    # 哨兵恒写全截面
    assert (result[DIVIDEND_POLICY_VERSION_COL] == DIVIDEND_POLICY_SCHEMA_VERSION).all()
    # 命中行透传
    assert result["dividend_continuity_5y"].iloc[0] == pytest.approx(1.0)
    assert result[DIVIDEND_HIST_MISSING_COL].iloc[0] == 0.0
    # yield = 0.6 / 10.0
    assert result["dividend_yield_hist_12m"].iloc[0] == pytest.approx(0.06)
    # 成熟未命中 → continuity=0 / hist_missing=1 / yield=0 / recent=0
    assert result["dividend_continuity_5y"].iloc[1] == 0.0
    assert result[DIVIDEND_HIST_MISSING_COL].iloc[1] == 1.0
    assert result["dividend_yield_hist_12m"].iloc[1] == 0.0
    assert result["dividend_days_to_ex_date"].iloc[1] == 31.0
    assert result["dividend_recent_imp_ann_10d"].iloc[1] == 0.0
    # 状态因子未命中 → NaN
    assert np.isnan(result["dividend_stability_5y"].iloc[1])
    assert np.isnan(result["dividend_payout_ratio"].iloc[1])
    # 未成熟（上市 <365 天）→ 全 NaN（含 continuity/hist_missing）
    assert np.isnan(result["dividend_continuity_5y"].iloc[2])
    assert np.isnan(result[DIVIDEND_HIST_MISSING_COL].iloc[2])
    assert np.isnan(result["dividend_yield_hist_12m"].iloc[2])
    assert np.isnan(result["dividend_days_to_ex_date"].iloc[2])


def test_handler_empty_data_writes_sentinel():
    """空数据（无分红数据）时全 NaN + 哨兵恒写（保证启用开关后 schema 稳定）。"""
    handler = DividendPolicyFactorHandler()
    features = _features_with_list_days(["000001.SZ"], [1000])
    current = _current_data(["000001.SZ"], [10.0])
    result = handler.apply(features, pd.DataFrame(), "20240610", current)
    assert (result[DIVIDEND_POLICY_VERSION_COL] == DIVIDEND_POLICY_SCHEMA_VERSION).all()
    assert np.isnan(result["dividend_stability_5y"].iloc[0])
    assert result["dividend_continuity_5y"].iloc[0] == 0.0


# ═══════════════════════════════════════════════════════════════
# 下载去重：div_proc 必入键
# ═══════════════════════════════════════════════════════════════


def test_deduplicate_dividend_keeps_div_proc_rows():
    """同一分红方案的预案/实施行按 div_proc 区分保留，不互相覆盖。"""
    df = pd.DataFrame(
        [
            _event_row(div_proc="预案", ex_date=None, imp_ann_date=None),
            _event_row(div_proc="实施", ex_date="20240610", imp_ann_date="20240601"),
        ]
    )
    result = _deduplicate_dividend(df)
    assert len(result) == 2
    assert set(result["div_proc"]) == {"预案", "实施"}


class _DividendClientStub:
    """按股票返回预设结果的 dividend 下载测试桩。"""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = []

    def query(self, api_name: str, ts_code: str) -> pd.DataFrame:
        assert api_name == "dividend"
        self.calls.append(ts_code)
        response = self.responses[ts_code]
        if isinstance(response, Exception):
            raise response
        return response.copy()


def test_force_replaces_successful_stock_and_retries_failed_stock(tmp_path):
    """force 成功股票整体替换，失败股票保留旧行且下次非 force 自动重试。"""
    storage = Storage(str(tmp_path))
    old = pd.DataFrame(
        [
            _event_row(ts_code="000001.SZ", ann_date="20200401", cash_div_tax=0.5),
            _event_row(ts_code="000002.SZ", ann_date="20200401", cash_div_tax=0.6),
        ]
    )
    storage.save_raw_by_date(old, "dividend", "2020-12-31")
    new_first = pd.DataFrame(
        [_event_row(ts_code="000001.SZ", ann_date="20200501", cash_div_tax=0.8)]
    )
    client = _DividendClientStub(
        {
            "000001.SZ": new_first,
            "000002.SZ": RuntimeError("模拟网络失败"),
        }
    )

    result = download_dividend_full(
        client,
        storage,
        pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]}),
        concurrency=1,
        force=True,
    )

    first_rows = result[result["ts_code"] == "000001.SZ"]
    second_rows = result[result["ts_code"] == "000002.SZ"]
    assert first_rows["ann_date"].tolist() == ["20200501"]
    assert second_rows["ann_date"].tolist() == ["20200401"]
    assert _load_dividend_coverage(storage) == {
        "000001.SZ": "data",
        "000002.SZ": "failed",
    }

    new_second = pd.DataFrame(
        [_event_row(ts_code="000002.SZ", ann_date="20200601", cash_div_tax=0.9)]
    )
    retry_client = _DividendClientStub({"000002.SZ": new_second})
    retried = download_dividend_full(
        retry_client,
        storage,
        pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]}),
        concurrency=1,
    )
    assert retry_client.calls == ["000002.SZ"]
    assert retried.loc[retried["ts_code"] == "000002.SZ", "ann_date"].tolist() == ["20200601"]
    assert _load_dividend_coverage(storage)["000002.SZ"] == "data"


def test_successful_empty_stock_is_persisted_and_old_partition_removed(tmp_path):
    """成功空结果记为 empty；force 时删除该股票旧行，后续不重复请求。"""
    storage = Storage(str(tmp_path))
    storage.save_raw_by_date(
        pd.DataFrame([_event_row(ts_code="000003.SZ", ann_date="20200401")]),
        "dividend",
        "2020-12-31",
    )
    client = _DividendClientStub({"000003.SZ": pd.DataFrame()})
    stock_basic = pd.DataFrame({"ts_code": ["000003.SZ"]})

    first = download_dividend_full(
        client,
        storage,
        stock_basic,
        concurrency=1,
        force=True,
    )
    second = download_dividend_full(client, storage, stock_basic, concurrency=1)

    assert first.empty
    assert second.empty
    assert client.calls == ["000003.SZ"]
    assert storage.list_partitions("raw", "dividend") == []
    assert _load_dividend_coverage(storage) == {"000003.SZ": "empty"}


def test_download_ignores_invalid_ann_date_and_suppresses_concat_warning(tmp_path, monkeypatch):
    """缺失公告日不生成非法分区，合并时不泄漏 pandas FutureWarning。"""
    storage = Storage(str(tmp_path))
    client = _DividendClientStub(
        {
            "000001.SZ": pd.DataFrame([_event_row(ts_code="000001.SZ")]),
            "000002.SZ": pd.DataFrame([_event_row(ts_code="000002.SZ", ann_date=None)]),
        }
    )
    original_concat = pd.concat

    def _warning_concat(*args, **kwargs):
        warnings.warn(
            "The behavior of DataFrame concatenation with empty or all-NA entries "
            "is deprecated.",
            FutureWarning,
        )
        return original_concat(*args, **kwargs)

    monkeypatch.setattr("src.lazybull.data.dividend_raw.pd.concat", _warning_concat)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = download_dividend_full(
            client,
            storage,
            pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]}),
            concurrency=1,
        )

    leaked = [warning for warning in caught if issubclass(warning.category, FutureWarning)]
    assert not leaked
    assert result["ts_code"].tolist() == ["000001.SZ"]
    assert storage.list_partitions("raw", "dividend") == ["2024-12-31"]


def test_ensure_reuses_preloaded_dividend_partitions(tmp_path, monkeypatch):
    """纸面增量入口不得在 factor loader 已加载全量后再次扫描年分区。"""
    storage = Storage(str(tmp_path))
    storage.save_raw(pd.DataFrame({"ts_code": ["000001.SZ"]}), "stock_basic", is_force=True)
    storage.save_raw_by_date(
        pd.DataFrame([_event_row(ts_code="000001.SZ", ann_date="20240401")]),
        "dividend",
        "2024-12-31",
    )
    existing = DataLoader(storage).load_dividend()
    assert existing is not None

    original_load = storage.load_raw_by_date
    partition_reads = []

    def _counting_load(dataset_name, partition):
        if dataset_name == "dividend":
            partition_reads.append(partition)
        return original_load(dataset_name, partition)

    class _NoNetworkClient:
        def query(self, *args, **kwargs):
            raise AssertionError("完整覆盖迁移不应触发按股查询")

        def get_dividend(self, **kwargs):
            raise AssertionError("本地日期已覆盖时不应触发单日查询")

    monkeypatch.setattr(storage, "load_raw_by_date", _counting_load)

    result = _try_download_dividend(
        _NoNetworkClient(),
        storage,
        "20240401",
        existing_df=existing,
    )

    assert result is existing
    assert partition_reads == []


def test_ensure_runs_daily_increment_for_complete_low_row_count_coverage(tmp_path):
    """逐股覆盖已完成时，即使历史行数很少也必须继续推进自然日日增量。"""
    storage = Storage(str(tmp_path))
    storage.save_raw(pd.DataFrame({"ts_code": ["000001.SZ"]}), "stock_basic", is_force=True)
    storage.save_raw_by_date(
        pd.DataFrame([_event_row(ts_code="000001.SZ", ann_date="20240401")]),
        "dividend",
        "2024-12-31",
    )
    existing = DataLoader(storage).load_dividend()

    class _IncrementClient:
        def __init__(self):
            self.calls = []

        def query(self, *args, **kwargs):
            raise AssertionError("已有分区应迁移为逐股完整覆盖，不应重复按股查询")

        def get_dividend(self, **kwargs):
            self.calls.append(kwargs)
            return pd.DataFrame()

    client = _IncrementClient()
    result = _try_download_dividend(client, storage, "20240402", existing_df=existing)

    assert result is existing
    assert client.calls == [
        {"ann_date": "20240402"},
        {"imp_ann_date": "20240402"},
    ]
    assert storage.load_sync_watermark("dividend") == "20240402"


def test_load_dividend_range_reads_annual_partition(tmp_path):
    """上半年范围查询按年份读取 YYYY-12-31 分区，再按 ann_date 精确过滤。"""
    storage = Storage(str(tmp_path))
    storage.save_raw_by_date(
        pd.DataFrame([_event_row(ann_date="20240315")]),
        "dividend",
        "2024-12-31",
    )
    storage.save_raw_by_date(
        pd.DataFrame([_event_row(ann_date="20250315")]),
        "dividend",
        "2025-12-31",
    )

    class _Loader(DividendLoaderMixin):
        def __init__(self, data_storage: Storage):
            self.storage = data_storage

    result = _Loader(storage).load_dividend("20240101", "20240630")

    assert result is not None
    assert result["ann_date"].tolist() == ["20240315"]


def test_ensure_schema_rejects_old_dividend_sentinel_value(tmp_path):
    """列名相同但哨兵值过期的缓存必须重建，当前版本值才可复用。"""
    storage = Storage(str(tmp_path))
    cache_dir = storage.features_path / "cs_infer"
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / "20240610.parquet"
    frame = pd.DataFrame({column: [0.0] for column in _REQUIRED_FACTOR_COLS})
    frame[DIVIDEND_POLICY_VERSION_COL] = DIVIDEND_POLICY_SCHEMA_VERSION - 1
    frame.to_parquet(file_path, index=False)

    required_groups = {OPTIONAL_FACTOR_GROUP_DIVIDEND_POLICY}
    assert not _check_features_schema(
        storage,
        "20240610",
        subdir="cs_infer",
        required_optional_groups=required_groups,
    )

    frame[DIVIDEND_POLICY_VERSION_COL] = DIVIDEND_POLICY_SCHEMA_VERSION
    frame.to_parquet(file_path, index=False)
    assert _check_features_schema(
        storage,
        "20240610",
        subdir="cs_infer",
        required_optional_groups=required_groups,
    )
