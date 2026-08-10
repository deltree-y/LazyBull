#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""新增因子接线回归测试。"""

from unittest.mock import patch

import pandas as pd

from src.lazybull.data.loader import DataLoader
from src.lazybull.data.tushare_client import FINA_INDICATOR_DEFAULT_FIELDS
from src.lazybull.features.builder import FeatureBuilder
from src.lazybull.features.ensure import (
    _REQUIRED_FACTOR_COLS,
    _try_download_cashflow,
    _try_download_fina_indicator,
)
from src.lazybull.factors.fundamental import build_fundamental_lookup_by_date


class _StubStorage:
    """最小存储桩：仅覆盖 DataLoader 测试需要的方法。"""

    def __init__(self, payload):
        self._payload = payload

    def load_raw(self, name):
        return self._payload.get(name)


def test_required_factor_cols_include_new_cashflow_and_consensus_revision_fields():
    assert "cashflow_freshness_days" in _REQUIRED_FACTOR_COLS
    assert "cons_revision_freshness_days" in _REQUIRED_FACTOR_COLS
    assert "consensus_freshness_days" in _REQUIRED_FACTOR_COLS
    assert "grossprofit_margin" in _REQUIRED_FACTOR_COLS
    assert "q_gr_yoy" in _REQUIRED_FACTOR_COLS
    assert "int_to_talcap" in _REQUIRED_FACTOR_COLS
    # v0.95.0 新增 lhb_cont_on_list: 旧 cs_infer 缓存缺列时由 ensure 自动重建
    assert "lhb_cont_on_list" in _REQUIRED_FACTOR_COLS
    assert "lhb_cont_up_days_5" in _REQUIRED_FACTOR_COLS
    assert "lhb_cont_up_days_20" in _REQUIRED_FACTOR_COLS


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

    with patch(
        "src.lazybull.features.ensure.downloads._bulk_download_by_period",
        side_effect=_fake_bulk_download_by_period,
    ), patch(
        "src.lazybull.features.ensure.downloads.DataLoader.load_fina_indicator",
        return_value="ok",
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

    with patch(
        "src.lazybull.features.ensure.downloads._bulk_download_by_period",
        side_effect=_fake_bulk_download_by_period,
    ), patch(
        "src.lazybull.features.ensure.downloads.DataLoader.load_cashflow",
        return_value="ok",
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

    with patch(
        "src.lazybull.features.ensure.downloads._refresh_existing_period_rows",
        side_effect=_fake_refresh_existing_period_rows,
    ), patch(
        "src.lazybull.features.ensure.downloads._incremental_catchup_by_calendar_date",
        side_effect=_fake_incremental_catchup_by_calendar_date,
    ), patch(
        "src.lazybull.features.ensure.downloads.DataLoader.load_fina_indicator",
        return_value=refreshed_df,
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
