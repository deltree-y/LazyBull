#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""新增因子接线回归测试。"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.lazybull.data.loader import DataLoader
from src.lazybull.data.tushare_client import FINA_INDICATOR_DEFAULT_FIELDS, TushareClient
from src.lazybull.factors.cashflow_quality import build_cashflow_quality_lookup_by_date
from src.lazybull.factors.fundamental import build_fundamental_lookup_by_date
from src.lazybull.features.builder import FeatureBuilder
from src.lazybull.features.ensure import (
    _REQUIRED_FACTOR_COLS,
    _try_download_cashflow,
    _try_download_fina_indicator,
)
from src.lazybull.features.factor_handlers import (
    CashflowQualityFactorHandler,
    ConsensusRevisionFactorHandler,
)
from src.lazybull.features.neutralization import (
    apply_industry_neutralization,
    apply_size_neutralization,
)


class _StubStorage:
    """最小存储桩：仅覆盖 DataLoader 测试需要的方法。"""

    def __init__(self, payload):
        self._payload = payload

    def load_raw(self, name):
        return self._payload.get(name)


def test_required_factor_cols_include_new_cashflow_and_consensus_revision_fields():
    assert "cashflow_freshness_days" in _REQUIRED_FACTOR_COLS
    # 稳定列名、当前值 v3：旧现金流语义缓存缺列或版本不符时自动重建
    assert "cashflow_quality_schema_v2" in _REQUIRED_FACTOR_COLS
    assert "cons_revision_freshness_days" in _REQUIRED_FACTOR_COLS
    assert "consensus_freshness_days" in _REQUIRED_FACTOR_COLS
    assert "grossprofit_margin" in _REQUIRED_FACTOR_COLS
    assert "q_gr_yoy" in _REQUIRED_FACTOR_COLS
    assert "int_to_talcap" in _REQUIRED_FACTOR_COLS
    # v0.95.0 新增 lhb_cont_on_list: 旧 cs_infer 缓存缺列时由 ensure 自动重建
    assert "lhb_cont_on_list" in _REQUIRED_FACTOR_COLS
    assert "lhb_cont_up_days_5" in _REQUIRED_FACTOR_COLS
    assert "lhb_cont_up_days_20" in _REQUIRED_FACTOR_COLS
    # v0.95.4 筹码胜率 5 列齐备: 旧 4 列缓存缺 weight_avg_bias 时由 ensure 自动重建
    assert "winner_rate" in _REQUIRED_FACTOR_COLS
    assert "weight_avg_bias" in _REQUIRED_FACTOR_COLS


def test_cashflow_quality_handler_writes_sentinel_for_full_cross_section():
    """哨兵列对当日全截面恒写版本号（含无现金流数据的股票），训练入口据此拦截旧分区。"""
    from src.lazybull.factors.cashflow_quality import (
        CASHFLOW_QUALITY_SCHEMA_VERSION,
        CASHFLOW_QUALITY_VERSION_COL,
    )

    handler = CashflowQualityFactorHandler()
    features = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]})
    # 仅 000001.SZ 当日有现金流数据
    data = pd.DataFrame({"ts_code": ["000001.SZ"], "ocf": [1.0]})

    result = handler.apply(features, data, "20240401", None)

    sentinel = result[CASHFLOW_QUALITY_VERSION_COL]
    assert sentinel.tolist() == [CASHFLOW_QUALITY_SCHEMA_VERSION] * 2

    # 无数据日同样全截面写版本号
    empty_result = handler.apply(features, pd.DataFrame(), "20240401", None)
    assert (
        empty_result[CASHFLOW_QUALITY_VERSION_COL].tolist() == [CASHFLOW_QUALITY_SCHEMA_VERSION] * 2
    )


def test_consensus_revision_handler_writes_sentinel_for_full_cross_section():
    """哨兵列对当日全截面恒写版本号（含无修正数据的股票），保证训练入口可拦截旧分区。"""
    from src.lazybull.factors.consensus_revision import CONSENSUS_REVISION_VERSION_COL

    handler = ConsensusRevisionFactorHandler()
    features = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]})
    # 仅 000001.SZ 当日有修正数据
    data = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "cons_analyst_count_chg": [0.5],
            CONSENSUS_REVISION_VERSION_COL: [2],
        }
    )

    result = handler.apply(features, data, "20240401", None)

    sentinel = result[CONSENSUS_REVISION_VERSION_COL]
    assert sentinel.tolist() == [2, 2]

    # 无数据日同样全截面写版本号
    empty_result = handler.apply(features, pd.DataFrame(), "20240401", None)
    assert empty_result[CONSENSUS_REVISION_VERSION_COL].tolist() == [2, 2]


def test_loader_cashflow_normalizes_date_columns():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["2024-05-01"],
            "end_date": ["2024-03-31"],
            "f_ann_date": ["2024-05-02"],
        }
    )
    loader = DataLoader(storage=_StubStorage({"cashflow": raw}))

    df = loader.load_cashflow()

    assert df is not None
    assert df.loc[0, "ann_date"] == "20240501"
    assert df.loc[0, "end_date"] == "20240331"
    assert df.loc[0, "f_ann_date"] == "20240502"


def test_cashflow_quality_industry_neutralization_outputs_zscore_fcf_yield():
    builder = FeatureBuilder(require_label=False)
    df = pd.DataFrame(
        {
            "ts_code": [f"{i:06d}.SZ" for i in range(6)],
            "sw_industry": ["银行"] * 6,
            "tradable": [1] * 6,
            "fcf_yield": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )

    result = builder._apply_industry_neutralization(df)

    assert "zscore_fcf_yield" in result.columns
    assert result["zscore_fcf_yield"].notna().all()


def test_cashflow_fcf_yield_converts_total_mv_from_ten_thousand_yuan():
    features = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "total_mv": [10000.0],
        }
    )
    cashflow = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "fcf": [10000000.0],
        }
    )

    result = CashflowQualityFactorHandler().apply(
        features,
        cashflow,
        "20240506",
        features,
    )

    assert result["fcf_yield"].iloc[0] == 0.1


def test_cashflow_quality_uses_tushare_capex_field_and_ttm_caliber():
    """TTM 口径 + TuShare 字段：capex 字段名与 TTM 推导均按公式展开。"""
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 3 + ["000002.SZ"] * 3,
            "ann_date": ["20230428"] * 6,
            "f_ann_date": ["20230428"] * 6,
            "end_date": ["20230331", "20231231", "20240331"] * 2,
            "n_cashflow_act": [1e8, 4e8, 1.5e8, -5e7, -8e7, -6e7],
            "c_pay_acq_const_fiolta": [2e7, 1e8, 4e7, 1e7, 3e7, 2e7],
            "c_fr_sale_sg": [5e8, 2e9, 6e8, 1e9, 3e9, 8e8],
            "net_profit": [8e7, 3e8, 1e8, 2e7, 5e7, 3e7],
            "free_cashflow": [8e7, 3e8, 1.1e8, -6e7, -1.1e8, -8e7],
        }
    )

    lookup = build_cashflow_quality_lookup_by_date(raw, ["20240506"])
    result = lookup["20240506"].set_index("ts_code")

    # TTM(Q1_2024) = cum(Q1_2024) - cum(Q1_2023) + cum(Q4_2023)
    assert result.loc["000001.SZ", "ocf"] == pytest.approx(1.5e8 - 1e8 + 4e8)
    assert result.loc["000001.SZ", "fcf"] == pytest.approx(1.1e8 - 8e7 + 3e8)
    assert result.loc["000001.SZ", "capex_to_ocf"] == pytest.approx(
        (4e7 - 2e7 + 1e8) / (1.5e8 - 1e8 + 4e8)
    )
    assert result.loc["000001.SZ", "ocf_to_revenue"] == pytest.approx(
        (1.5e8 - 1e8 + 4e8) / (6e8 - 5e8 + 2e9)
    )
    # 负 OCF 股票：TTM 同样成立，capex_to_ocf 保持符号（分母为负）
    ttm_ocf_neg = -6e7 - (-5e7) + (-8e7)
    assert result.loc["000002.SZ", "ocf"] == pytest.approx(ttm_ocf_neg)
    assert result.loc["000002.SZ", "capex_to_ocf"] == pytest.approx((2e7 - 1e7 + 3e7) / ttm_ocf_neg)


def test_cashflow_quality_revision_visible_only_after_f_ann_date():
    """审计回归：修订版本只在 f_ann_date 之后可见，不得回填到原始公告日。"""
    raw = pd.DataFrame(
        {
            "ts_code": ["002111.SZ", "002111.SZ"],
            "ann_date": ["20210420", "20210420"],
            "f_ann_date": ["20210420", "20230705"],
            "end_date": ["20201231", "20201231"],
            "n_cashflow_act": [8.0e8, 7.8e8],
            "c_pay_acq_const_fiolta": [1.2e8, 1.2e8],
            "c_fr_sale_sg": [3.0e9, 3.0e9],
            "net_profit": [4.0e8, 4.0e8],
            "free_cashflow": [6.8e8, 6.6e8],
        }
    )

    before = build_cashflow_quality_lookup_by_date(raw, ["20210420"])["20210420"]
    after = build_cashflow_quality_lookup_by_date(raw, ["20230705"])["20230705"]

    # 原始公告日：仅原始版本可见（Q4 年报 TTM=当年累计）
    assert before.loc[0, "ocf"] == pytest.approx(8.0e8)
    assert before.loc[0, "cashflow_freshness_days"] == 0
    # 修订公告日：可见修订版本
    assert after.loc[0, "ocf"] == pytest.approx(7.8e8)
    assert after.loc[0, "cashflow_freshness_days"] == 0


def test_cashflow_quality_same_version_prefers_latest_update_flag_regardless_of_order():
    """同一可用日的重复版本必须按 TuShare update_flag=1 最新语义确定去重。"""
    rows = [
        {
            "ts_code": "000001.SZ",
            "ann_date": "20240430",
            "f_ann_date": "20240430",
            "end_date": "20231231",
            "update_flag": "1",
            "n_cashflow_act": 1.0e8,
            "free_cashflow": 1.0e7,
        },
        {
            "ts_code": "000001.SZ",
            "ann_date": "20240430",
            "f_ann_date": "20240430",
            "end_date": "20231231",
            "update_flag": "0",
            "n_cashflow_act": 1.0e8,
            "free_cashflow": 2.0e7,
        },
    ]

    forward = build_cashflow_quality_lookup_by_date(pd.DataFrame(rows), ["20240506"])["20240506"]
    reversed_result = build_cashflow_quality_lookup_by_date(
        pd.DataFrame(list(reversed(rows))), ["20240506"]
    )["20240506"]

    assert forward.loc[0, "fcf"] == pytest.approx(1.0e7)
    assert reversed_result.loc[0, "fcf"] == pytest.approx(1.0e7)


def test_cashflow_quality_dependency_revision_recomputes_latest_ttm():
    """去年同季度晚到修订必须触发当前报告期 TTM 重算，不能冻结旧依赖值。"""
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 4,
            "ann_date": ["20230430", "20240330", "20240430", "20230430"],
            "f_ann_date": ["20230430", "20240330", "20240430", "20240515"],
            "end_date": ["20230331", "20231231", "20240331", "20230331"],
            "n_cashflow_act": [1.0e7, 1.0e8, 2.0e7, 1.5e7],
            "free_cashflow": [1.0e7, 1.0e8, 2.0e7, 1.5e7],
        }
    )

    lookup = build_cashflow_quality_lookup_by_date(raw, ["20240514", "20240515"])

    assert lookup["20240514"].loc[0, "ocf"] == pytest.approx(1.1e8)
    assert lookup["20240515"].loc[0, "ocf"] == pytest.approx(1.05e8)
    assert lookup["20240515"].loc[0, "cashflow_freshness_days"] == 0


def test_cashflow_quality_ratio_denominator_floor_and_clip():
    """经济尺度下限 + 有界裁剪：极端比值被裁剪，低于下限的分母置 NaN。"""
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "ann_date": ["20240430", "20240430"],
            "f_ann_date": ["20240430", "20240430"],
            "end_date": ["20231231", "20231231"],
            "n_cashflow_act": [1e8, 1e3],  # 000002 OCF 低于经济尺度下限
            "c_pay_acq_const_fiolta": [5e11, 1e7],  # 极端 capex
            "c_fr_sale_sg": [1e9, 1e9],
            "net_profit": [1e8, 1e8],
            "free_cashflow": [1e8, 1e8],
        }
    )

    result = build_cashflow_quality_lookup_by_date(raw, ["20240506"])["20240506"]
    result = result.set_index("ts_code")

    # 5e11 / 1e8 = 5000 -> 裁剪到上限 50
    assert result.loc["000001.SZ", "capex_to_ocf"] == 50.0
    # 分母 OCF 低于经济尺度下限 -> NaN
    assert pd.isna(result.loc["000002.SZ", "capex_to_ocf"])


def test_cashflow_quality_late_old_period_correction_does_not_replace_latest_period():
    """无 f_ann_date 时回退 ann_date，晚发的旧报告期修正不覆盖已公告的新报告期。"""
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 6,
            "ann_date": [
                "20230501",  # Q2_2023 历史（TTM 基准）
                "20240501",  # Q1_2024
                "20240430",  # Q4_2023 历史（TTM 基准）
                "20240801",  # Q2_2024 初版
                "20240815",  # Q2_2024 修订
                "20240901",  # 晚发的 Q1_2024 修正
            ],
            "end_date": [
                "20230630",
                "20240331",
                "20231231",
                "20240630",
                "20240630",
                "20240331",
            ],
            "n_cashflow_act": [50.0, 100.0, 300.0, 200.0, 220.0, 999.0],
            "c_pay_acq_const_fiolta": [5.0, 20.0, 30.0, 30.0, 35.0, 40.0],
        }
    )

    result = build_cashflow_quality_lookup_by_date(raw, ["20240902"])["20240902"]

    # 报告期优先：当日可见公告中选最新报告期（20240630）的最新版本（20240815），
    # 晚发的 20240331 修正（999）不覆盖新报告期。
    assert result.loc[0, "ocf"] == 220.0 - 50.0 + 300.0
    assert result.loc[0, "cashflow_freshness_days"] == 18


def test_cashflow_capex_reaches_all_training_features():
    codes = [f"{index:06d}.SZ" for index in range(40)]
    # Q4 年报行：TTM 退化为当年累计，单行即可得到非空因子（无需历史季度）
    raw = pd.DataFrame(
        {
            "ts_code": codes,
            "ann_date": ["20240501"] * 40,
            "f_ann_date": ["20240501"] * 40,
            "end_date": ["20231231"] * 40,
            "n_cashflow_act": [1e8 + index * 7e6 for index in range(40)],
            "c_pay_acq_const_fiolta": [1e7 + (index % 7) * 3e6 for index in range(40)],
            "c_fr_sale_sg": [1e9 + index * 1e7 for index in range(40)],
            "net_profit": [2e8 + index * 1e7 for index in range(40)],
            "free_cashflow": [8e7 + index * 1e7 for index in range(40)],
        }
    )
    lookup = build_cashflow_quality_lookup_by_date(raw, ["20240506"])
    features = pd.DataFrame(
        {
            "ts_code": codes,
            # 总市值（万元）：100 亿元起，满足 fcf_yield 分母经济尺度下限（1 亿元）
            "total_mv": [100000.0 + index * 10000 for index in range(40)],
            "log_total_mv": [float(index) for index in range(40)],
            "sw_industry": ["银行" if index % 2 == 0 else "电子" for index in range(40)],
            "tradable": [1] * 40,
        }
    )
    factor_values = CashflowQualityFactorHandler().apply(
        features,
        lookup["20240506"],
        "20240506",
        features,
    )
    enriched = features.assign(**factor_values)
    industry_neutralized = apply_industry_neutralization(
        enriched,
        horizons=[5, 10, 20],
        lookback_windows=[5, 10, 20],
    )
    result = apply_size_neutralization(industry_neutralized)

    expected_columns = [
        "zscore_capex_to_ocf",
        "zscore_capex_to_ocf_sz",
        "zscore_fcf_yield",
        "zscore_fcf_yield_sz",
    ]
    for column in expected_columns:
        assert result[column].notna().all()
        assert result[column].nunique() > 1


def test_cashflow_client_default_fields_use_tushare_capex_name():
    client = object.__new__(TushareClient)

    with patch.object(client, "query", return_value=pd.DataFrame()) as query_mock:
        client.get_cashflow(ts_code="000001.SZ")
        client.get_cashflow_by_period("20240331")

    assert len(query_mock.call_args_list) == 2
    for call in query_mock.call_args_list:
        fields = call.kwargs["fields"]
        assert "c_pay_acq_const_fiolta" in fields.split(",")
        assert "c_pay_for_assets" not in fields.split(",")


def test_fina_indicator_lookup_maps_proxy_columns():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240501"],
            "end_date": ["20240331"],
            "roe_waa": [12.0],
            "q_ocf_to_sales": [0.37],
            "int_to_talcap": [5.2],
            "inv_turn": [1.8],
        }
    )

    lookup = build_fundamental_lookup_by_date(raw, ["20240506"])
    day_df = lookup["20240506"]

    assert day_df.loc[0, "cf_sales"] == 0.37
    assert day_df.loc[0, "int_to_talcap"] == 5.2
    assert day_df.loc[0, "inv_turn"] == 1.8


def test_builder_backfills_fundamental_proxy_features_before_neutralization():
    builder = FeatureBuilder(require_label=False)
    df = pd.DataFrame(
        {
            "ts_code": [f"{i:06d}.SZ" for i in range(6)],
            "sw_industry": ["银行"] * 6,
            "tradable": [1] * 6,
            "q_ocf_to_sales": [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
            "ocf_to_profit": [0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
            "int_to_talcap": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    backfilled = builder._backfill_fundamental_proxy_features(df)
    result = builder._apply_industry_neutralization(backfilled)

    assert backfilled["cf_sales"].tolist() == df["q_ocf_to_sales"].tolist()
    assert backfilled["cf_nm"].tolist() == df["ocf_to_profit"].tolist()
    assert "zscore_cf_sales" in result.columns
    assert "zscore_cf_nm" in result.columns
    assert "zscore_int_to_talcap" in result.columns
    assert result["zscore_cf_sales"].notna().all()
    assert result["zscore_cf_nm"].notna().all()
    assert result["zscore_int_to_talcap"].notna().all()


def test_try_download_fina_indicator_full_download_uses_explicit_fields():
    class _DummyStorage:
        def list_partitions(self, layer, name):
            assert layer == "raw"
            assert name == "fina_indicator"
            return []

    captured = {}

    def _fake_bulk_download_by_period(
        client,
        storage,
        dataset_name,
        api_name,
        dedup_cols,
        fields=None,
        start_year=2012,
        partition_by_period=False,
    ):
        captured["dataset_name"] = dataset_name
        captured["api_name"] = api_name
        captured["fields"] = fields
        captured["partition_by_period"] = partition_by_period
        return None

    with (
        patch(
            "src.lazybull.features.ensure.downloads._bulk_download_by_period",
            side_effect=_fake_bulk_download_by_period,
        ),
        patch(
            "src.lazybull.features.ensure.downloads.DataLoader.load_fina_indicator",
            return_value="ok",
        ),
    ):
        result = _try_download_fina_indicator(
            client=object(),
            storage=_DummyStorage(),
            trade_date="20250331",
        )

    assert result == "ok"
    assert captured["dataset_name"] == "fina_indicator"
    assert captured["api_name"] == "fina_indicator_vip"
    assert captured["fields"] == FINA_INDICATOR_DEFAULT_FIELDS
    assert captured["partition_by_period"] is True


def test_try_download_cashflow_full_download_uses_quarter_partitions():
    class _DummyStorage:
        def list_partitions(self, layer, name):
            assert layer == "raw"
            assert name == "cashflow"
            return []

    captured = {}

    def _fake_bulk_download_by_period(
        client,
        storage,
        dataset_name,
        api_name,
        dedup_cols,
        fields=None,
        start_year=2012,
        partition_by_period=False,
    ):
        captured["dataset_name"] = dataset_name
        captured["api_name"] = api_name
        captured["fields"] = fields
        captured["partition_by_period"] = partition_by_period
        return None

    with (
        patch(
            "src.lazybull.features.ensure.downloads._bulk_download_by_period",
            side_effect=_fake_bulk_download_by_period,
        ),
        patch(
            "src.lazybull.features.ensure.downloads.DataLoader.load_cashflow",
            return_value="ok",
        ),
    ):
        result = _try_download_cashflow(
            client=object(),
            storage=_DummyStorage(),
            trade_date="20250331",
        )

    assert result == "ok"
    assert captured["dataset_name"] == "cashflow"
    assert captured["api_name"] == "cashflow_vip"
    assert captured["fields"] is None
    assert captured["partition_by_period"] is True


def test_try_download_fina_indicator_existing_schema_triggers_period_refresh():
    class _DummyStorage:
        def list_partitions(self, layer, name):
            assert layer == "raw"
            assert name == "fina_indicator"
            return ["2024-03-31"]

        def load_raw_by_date(self, name, trade_date, format="parquet", columns=None):
            assert name == "fina_indicator"
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"] * 1000,
                    "ann_date": ["20240501"] * 1000,
                    "end_date": ["20240331"] * 1000,
                    "roe_waa": [12.0] * 1000,
                }
            )

    events = []

    def _fake_refresh_existing_period_rows(**kwargs):
        events.append(("refresh", kwargs["fields"]))
        assert kwargs["partition_date_col"] == "end_date"
        assert kwargs["partition_mode"] == "quarter"
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20240501"],
                "end_date": ["20240331"],
                "roe_waa": [12.0],
                "q_gr_yoy": [5.0],
                "q_ocf_to_sales": [0.2],
                "int_to_talcap": [3.0],
                "inv_turn": [1.1],
            }
        )

    def _fake_incremental_catchup_by_calendar_date(**kwargs):
        events.append(("incremental", kwargs["existing_df"].columns.tolist()))
        assert kwargs["partition_date_col"] == "end_date"
        assert kwargs["partition_mode"] == "quarter"
        return kwargs["existing_df"]

    refreshed_df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240501"],
            "end_date": ["20240331"],
            "roe_waa": [12.0],
            "q_gr_yoy": [5.0],
            "q_ocf_to_sales": [0.2],
            "int_to_talcap": [3.0],
            "inv_turn": [1.1],
        }
    )

    with (
        patch(
            "src.lazybull.features.ensure.downloads._refresh_existing_period_rows",
            side_effect=_fake_refresh_existing_period_rows,
        ),
        patch(
            "src.lazybull.features.ensure.downloads._incremental_catchup_by_calendar_date",
            side_effect=_fake_incremental_catchup_by_calendar_date,
        ),
        patch(
            "src.lazybull.features.ensure.downloads.DataLoader.load_fina_indicator",
            return_value=refreshed_df,
        ),
    ):
        result = _try_download_fina_indicator(
            client=object(),
            storage=_DummyStorage(),
            trade_date="20250331",
        )

    assert list(result.columns) == [
        "ts_code",
        "ann_date",
        "end_date",
        "roe_waa",
        "q_gr_yoy",
        "q_ocf_to_sales",
        "int_to_talcap",
        "inv_turn",
    ]
    assert events[0] == ("refresh", FINA_INDICATOR_DEFAULT_FIELDS)
    assert events[1][0] == "incremental"
    assert {"q_gr_yoy", "q_ocf_to_sales", "int_to_talcap", "inv_turn"}.issubset(events[1][1])
