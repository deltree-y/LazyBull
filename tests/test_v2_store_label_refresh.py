# -*- coding: utf-8 -*-
"""P3 pre T3 标签专用刷新、精确身份、审计恰一次与分级恢复合成测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lazybull.features.labels import compute_forward_returns
from src.lazybull.features.neutralization import apply_industry_neutralization
from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.store.column_groups import LABEL_TABLES, PANEL_GROUPS
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.label_refresh import (
    LabelRefreshRecoveryError,
    plan_label_refreshes,
    refresh_labels_for_day,
    validate_label_day_readiness,
)
from src.lazybull.v2.store.panel_builder import bootstrap_manifest
from src.lazybull.v2.store.store_identity import (
    collect_file_identities,
    frame_content_digest,
    producer_identity,
    verify_file_identities,
)

_DAY = "20260703"
_CALENDAR = pd.bdate_range("2026-06-01", "2026-08-31").strftime("%Y%m%d").tolist()
_CODES = [f"00000{i}.SZ" for i in range(6)]
_PRODUCER = {"producer_digest": "producer-a", "environment": {"python": "test"}}


def _panel_frame(date: str = _DAY) -> pd.DataFrame:
    n = len(_CODES)
    return pd.DataFrame(
        {
            "trade_date": [date] * n,
            "ts_code": _CODES,
            "sw_industry": ["行业A"] * 3 + ["行业B"] * 3,
            "sw_industry_code": ["A"] * 3 + ["B"] * 3,
            "sw_l1_code": ["L1"] * n,
            "sw_l2_code": ["A"] * 3 + ["B"] * 3,
            "tradable": [1] * n,
        }
    )


def _write_panel(store: PanelDataStore, panel: pd.DataFrame) -> None:
    date = panel["trade_date"].iloc[0]
    for group in PANEL_GROUPS:
        if group == "core":
            frame = panel.copy()
        else:
            column = next(c for c in PANEL_GROUPS[group] if c not in {"trade_date", "ts_code"})
            frame = panel[["trade_date", "ts_code"]].assign(**{column: 1.0})
        store.append_features(TradeDate.from_str(date), group, frame)


def _forming_tables(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        name: panel[["ts_code", "trade_date"]].assign(
            label_value=np.nan, neu_label_value=np.nan, maturity_status="forming"
        )[["ts_code", "trade_date", "label_value", "neu_label_value", "maturity_status"]]
        for name in LABEL_TABLES
    }


def _daily_frames() -> dict[str, pd.DataFrame]:
    result = {}
    for date_index, date in enumerate(_CALENDAR):
        codes = [*_CODES, "999999.SZ"]
        base = np.arange(len(codes), dtype=float) + 10.0 + date_index / 1000
        result[date] = pd.DataFrame(
            {
                "trade_date": [date] * len(codes),
                "ts_code": codes,
                "close_adj": base,
                "open_adj": base * 1.02,
            }
        )
    return result


def _make_store(tmp_path: Path, *, forming: bool = True, date: str = _DAY):
    store = PanelDataStore(tmp_path)
    bootstrap_manifest(store)
    panel = _panel_frame(date)
    _write_panel(store, panel)
    if forming:
        for name, table in _forming_tables(panel).items():
            store.append_labels(name, table)
    return store, panel, _daily_frames()


def _refresh(store, daily, _date=_DAY, **overrides):
    kwargs = {
        "trading_dates": _CALENDAR,
        "daily_adj_dict": daily,
        "input_watermark": "20260803",
        "shenwan_level": "l2",
        "producer": _PRODUCER,
        "run_id": "run-a",
    }
    kwargs.update(overrides)
    return refresh_labels_for_day(store, _date, **kwargs)


def _audit_counts(store) -> tuple[int, int]:
    return len(store.manifest.label_refresh_intents()), len(store.manifest.label_refreshes())


def test_refresh_matches_legacy_order_exact_and_no_day_build(tmp_path, monkeypatch):
    store, panel, daily = _make_store(tmp_path)
    monkeypatch.setattr(
        "src.lazybull.v2.store.panel_builder.build_features_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("禁止整日重建")),
    )
    panel_bytes = {
        path: path.read_bytes()
        for path in (tmp_path / "features" / "panel" / _DAY).glob("*.parquet")
    }
    result = _refresh(store, daily)
    assert result.status == "refreshed"
    assert result.actual_status == "sealed"
    assert result.buy_date == "20260706" and result.endpoint_date == "20260803"
    assert _audit_counts(store) == (1, 1)

    current_idx = _CALENDAR.index(_DAY)
    raw = compute_forward_returns(
        panel[["trade_date", "ts_code"]],
        _DAY,
        _CALENDAR,
        current_idx,
        [5, 10, 20],
        daily_adj_dict=daily,
    )
    wide = panel.merge(raw, on=["trade_date", "ts_code"], validate="one_to_one")
    wide = apply_industry_neutralization(wide, [5, 10, 20], [5, 10, 20], "l2")
    for name, (raw_col, neu_col) in LABEL_TABLES.items():
        actual = pd.read_parquet(tmp_path / "labels" / name / f"{_DAY}.parquet")
        expected = wide.sort_values("ts_code").reset_index(drop=True)
        np.testing.assert_allclose(actual["label_value"], expected[raw_col], rtol=0, atol=0)
        np.testing.assert_allclose(actual["neu_label_value"], expected[neu_col], rtol=0, atol=0)
        assert actual["maturity_status"].eq("sealed").all()
    assert validate_label_day_readiness(store, _DAY)["ready"] is True
    for path, payload in panel_bytes.items():
        assert path.read_bytes() == payload


def test_exact_endpoint_missing_and_invalid_buy_price_never_use_t22(tmp_path):
    store, panel, daily = _make_store(tmp_path)
    endpoints = {
        "B": _CALENDAR[_CALENDAR.index(_DAY) + 1],
        "E5": _CALENDAR[_CALENDAR.index(_DAY) + 6],
        "E10": _CALENDAR[_CALENDAR.index(_DAY) + 11],
        "E20": _CALENDAR[_CALENDAR.index(_DAY) + 21],
    }
    daily[endpoints["E20"]] = daily[endpoints["E20"]].query("ts_code != @_CODES[0]").copy()
    daily[endpoints["B"]].loc[daily[endpoints["B"]]["ts_code"] == _CODES[1], "close_adj"] = 0
    t22 = _CALENDAR[_CALENDAR.index(_DAY) + 22]
    daily[t22] = daily[t22].assign(open_adj=1e9)
    result = _refresh(store, daily)

    y20 = pd.read_parquet(tmp_path / "labels" / "y_ret_20" / f"{_DAY}.parquet")
    y5 = pd.read_parquet(tmp_path / "labels" / "y_ret_5" / f"{_DAY}.parquet")
    first = y20["ts_code"] == _CODES[0]
    second = y20["ts_code"] == _CODES[1]
    assert y20.loc[first, "label_value"].isna().all()
    assert y20.loc[second, "label_value"].isna().all()
    assert y5.loc[y5["ts_code"] == _CODES[0], "label_value"].notna().all()
    assert not (y20["label_value"] > 1).any()
    assert result.cross_seam_impact["y_ret_5"]["affected_rows"] == 1
    assert result.cross_seam_impact["y_ret_10"]["affected_rows"] == 1


def test_event_key_ignores_run_id_and_unrelated_rows_but_not_relevant_price(tmp_path):
    store, _, daily = _make_store(tmp_path)
    first = _refresh(store, daily)
    original_files = {
        name: (tmp_path / "labels" / name / f"{_DAY}.parquet").read_bytes() for name in LABEL_TABLES
    }
    original_manifest = store.manifest.content_fingerprint()
    changed = copy.deepcopy(daily)
    changed[_CALENDAR[1]].loc[:, "close_adj"] += 100
    e20 = first.endpoint_date
    changed[e20].loc[changed[e20]["ts_code"] == "999999.SZ", "open_adj"] += 100
    second = _refresh(store, changed, run_id="another-run")

    assert second.status == "no_op" and second.event_key == first.event_key
    assert _audit_counts(store) == (1, 1)
    assert store.manifest.content_fingerprint() == original_manifest
    for name, payload in original_files.items():
        assert (tmp_path / "labels" / name / f"{_DAY}.parquet").read_bytes() == payload

    relevant = copy.deepcopy(changed)
    relevant[e20].loc[relevant[e20]["ts_code"] == _CODES[0], "open_adj"] += 1
    with pytest.raises(LabelRefreshRecoveryError, match="sealed"):
        _refresh(store, relevant)
    assert _audit_counts(store) == (1, 1)
    with pytest.raises(LabelRefreshRecoveryError, match="sealed"):
        _refresh(store, changed, producer={"producer_digest": "producer-b"})
    assert _audit_counts(store) == (1, 1)


def test_dry_run_has_no_side_effects(tmp_path):
    store, _, daily = _make_store(tmp_path)
    before = store.manifest.content_fingerprint()
    result = _refresh(store, daily, dry_run=True)
    assert result.status == "planned"
    assert result.expected_status == "sealed" and result.actual_status == "forming"
    assert _audit_counts(store) == (0, 0)
    assert store.manifest.content_fingerprint() == before


def test_dry_run_current_status_distinguishes_partial_missing_and_other_states(tmp_path):
    from src.lazybull.v2.store import label_refresh as label_refresh_module

    store, panel, daily = _make_store(tmp_path, forming=False)
    assert label_refresh_module._current_status(store, _DAY) == "missing"
    assert _refresh(store, daily, dry_run=True).actual_status == "missing"

    tables = _forming_tables(panel)
    store.append_labels("y_ret_5", tables["y_ret_5"])
    assert label_refresh_module._current_status(store, _DAY) == "partial_missing"
    assert _refresh(store, daily, dry_run=True).actual_status == "partial_missing"
    assert _audit_counts(store) == (0, 0)

    for name in ("y_ret_10", "y_ret_20"):
        store.append_labels(name, tables[name])
    assert label_refresh_module._current_status(store, _DAY) == "forming"

    sealed = {name: table.assign(maturity_status="sealed") for name, table in tables.items()}
    store.append_labels("y_ret_5", sealed["y_ret_5"])
    assert label_refresh_module._current_status(store, _DAY) == "mixed"
    for name in ("y_ret_10", "y_ret_20"):
        store.append_labels(name, sealed[name])
    assert label_refresh_module._current_status(store, _DAY) == "sealed"


def test_forming_consumer_readiness_requires_calendar_and_watermark(tmp_path):
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(ValueError, match="完整日历和实际输入水位"):
        validate_label_day_readiness(store, _DAY, consumer=True)
    with pytest.raises(RuntimeError, match="E20 已到水位"):
        validate_label_day_readiness(
            store,
            _DAY,
            consumer=True,
            trading_dates=_CALENDAR,
            input_watermark="20260803",
        )
    ready = validate_label_day_readiness(
        store,
        _DAY,
        consumer=True,
        trading_dates=_CALENDAR,
        input_watermark=_DAY,
    )
    assert ready["ready"] is True and ready["status"] == "forming"


def test_missing_completion_is_recovered_exactly_once(tmp_path, monkeypatch):
    store, _, daily = _make_store(tmp_path)

    def crash_completion(record):
        raise RuntimeError("模拟完成审计前崩溃")

    monkeypatch.setattr(store._wal, "append_label_refresh_completion", crash_completion)
    with pytest.raises(RuntimeError, match="完成审计前崩溃"):
        _refresh(store, daily)
    monkeypatch.undo()
    assert _audit_counts(store) == (1, 0)
    assert validate_label_day_readiness(store, _DAY, consumer=False)["ready"] is True
    with pytest.raises(RuntimeError, match="完成审计"):
        validate_label_day_readiness(store, _DAY, consumer=True)

    recovered = _refresh(store, daily)
    assert recovered.status == "recovered" and recovered.recovered is True
    assert _audit_counts(store) == (1, 1)
    assert _refresh(store, daily).status == "no_op"
    assert _audit_counts(store) == (1, 1)


def test_manifest_save_crash_replays_partition_and_audit_wal_exactly_once(tmp_path, monkeypatch):
    from src.lazybull.v2.store.manifest import Manifest

    store, _, daily = _make_store(tmp_path)

    def crash_save(self):
        raise RuntimeError("模拟 manifest.save 崩溃")

    monkeypatch.setattr(Manifest, "save", crash_save)
    with pytest.raises(RuntimeError, match="manifest.save 崩溃"):
        _refresh(store, daily)
    monkeypatch.undo()
    assert store.wal_path.exists() and store.wal_path.stat().st_size > 0

    reopened = PanelDataStore(tmp_path)
    assert _audit_counts(reopened) == (1, 1)
    assert validate_label_day_readiness(reopened, _DAY)["ready"] is True
    assert not reopened.wal_path.exists()
    assert _refresh(reopened, daily).status == "no_op"
    assert _audit_counts(reopened) == (1, 1)


def test_replace_before_partition_wal_requires_controlled_roll_forward(tmp_path, monkeypatch):
    store, _, daily = _make_store(tmp_path)

    def crash_partition(*args):
        raise RuntimeError("模拟 replace 后 WAL 前崩溃")

    monkeypatch.setattr(store._wal, "register_label_partition", crash_partition)
    with pytest.raises(RuntimeError, match="replace 后 WAL 前崩溃"):
        _refresh(store, daily)
    monkeypatch.undo()
    reopened = PanelDataStore(tmp_path)
    with pytest.raises(LabelRefreshRecoveryError, match="--recover-event") as exc_info:
        _refresh(reopened, daily)
    event_key = str(exc_info.value).split("--recover-event ", 1)[1].split()[0]
    plan = _refresh(reopened, daily, recover_event_key=event_key, dry_run=True)
    assert plan.status == "recovery_planned"
    assert _audit_counts(reopened) == (1, 0)
    recovered = _refresh(reopened, daily, recover_event_key=event_key)
    assert recovered.status == "recovered"
    assert _audit_counts(reopened) == (1, 1)
    assert validate_label_day_readiness(reopened, _DAY)["ready"] is True


def test_stale_prepare_is_superseded_append_only(tmp_path, monkeypatch):
    store, _, daily = _make_store(tmp_path)

    def crash_before_replace(*args):
        raise RuntimeError("模拟最终 replace 前崩溃")

    monkeypatch.setattr(store, "append_labels", crash_before_replace)
    with pytest.raises(RuntimeError, match="最终 replace 前崩溃"):
        _refresh(store, daily)
    monkeypatch.undo()
    old_prepare = store.manifest.label_refresh_intents()[0]

    result = _refresh(store, daily, producer={"producer_digest": "producer-new"})
    intents = store.manifest.label_refresh_intents()
    assert intents[0] == old_prepare
    superseded = [r for r in intents if r["record_type"] == "superseded"]
    assert len(superseded) == 1
    assert superseded[0]["event_key"] == old_prepare["event_key"]
    assert superseded[0]["superseded_by"] == result.event_key
    assert len(store.manifest.label_refreshes()) == 1
    assert result.orphan_intents[0]["event_key"] == old_prepare["event_key"]
    assert _refresh(store, daily, producer={"producer_digest": "producer-new"}).status == "no_op"
    assert len(store.manifest.label_refresh_intents()) == 3


def test_consecutive_producer_failures_terminate_prepares_once_and_replay_without_append(
    tmp_path, monkeypatch
):
    store, _, daily = _make_store(tmp_path)

    def crash_before_replace(*args):
        raise RuntimeError("模拟连续最终 replace 前崩溃")

    monkeypatch.setattr(store, "append_labels", crash_before_replace)
    prepare_keys = []
    for producer in ("producer-a", "producer-b", "producer-c"):
        with pytest.raises(RuntimeError, match="连续最终 replace 前崩溃"):
            _refresh(store, daily, producer={"producer_digest": producer})
        prepares = [
            record
            for record in store.manifest.label_refresh_intents()
            if record["record_type"] == "prepare"
        ]
        assert len(prepares) == len(prepare_keys) + 1
        prepare_keys.append(prepares[-1]["event_key"])
    monkeypatch.undo()

    intents = store.manifest.label_refresh_intents()
    superseded = [record for record in intents if record["record_type"] == "superseded"]
    assert len(intents) == 5
    assert [record["event_key"] for record in superseded] == prepare_keys[:2]
    assert [record["superseded_by"] for record in superseded] == prepare_keys[1:]
    assert _audit_counts(store) == (5, 0)

    recovered = _refresh(store, daily, producer={"producer_digest": "producer-c"})
    assert recovered.status == "recovered"
    assert _audit_counts(store) == (5, 1)
    assert _refresh(store, daily, producer={"producer_digest": "producer-c"}).status == "no_op"
    assert _audit_counts(store) == (5, 1)


@pytest.mark.parametrize("field", ["input_digest", "producer_digest", "target_status"])
def test_intent_identity_semantic_conflict_fails_before_wal(tmp_path, monkeypatch, field):
    from src.lazybull.v2.store import label_refresh as label_refresh_module

    store, _, daily = _make_store(tmp_path)

    def crash_before_replace(*args):
        raise RuntimeError("模拟身份冲突前置崩溃")

    monkeypatch.setattr(store, "append_labels", crash_before_replace)
    with pytest.raises(RuntimeError, match="身份冲突前置崩溃"):
        _refresh(store, daily)
    monkeypatch.undo()
    prepare = store.manifest.label_refresh_intents()[0]
    manifest_before = store.manifest.content_fingerprint()
    wal_before = store.wal_path.read_bytes() if store.wal_path.exists() else None

    with store.write_lock():
        with pytest.raises(ValueError, match="身份冲突"):
            label_refresh_module._append_intent_once(store, {**prepare, field: "conflict"})
        assert (
            label_refresh_module._append_intent_once(store, {**prepare, "run_id": "other"}) is False
        )

    assert store.manifest.content_fingerprint() == manifest_before
    wal_after = store.wal_path.read_bytes() if store.wal_path.exists() else None
    assert wal_after == wal_before


def test_existing_label_key_conflict_fails_before_new_intent(tmp_path):
    store, _, daily = _make_store(tmp_path)
    path = tmp_path / "labels" / "y_ret_5" / f"{_DAY}.parquet"
    frame = pd.read_parquet(path)
    extra = frame.iloc[[0]].assign(ts_code="extra.SZ")
    pd.concat([frame, extra], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="键集冲突"):
        _refresh(store, daily)
    assert _audit_counts(store) == (0, 0)


def test_all_y20_nan_is_rejected_before_intent(tmp_path):
    store, _, daily = _make_store(tmp_path)
    e20 = "20260803"
    daily[e20] = daily[e20].query("ts_code == '999999.SZ'").copy()
    with pytest.raises(RuntimeError, match="非空行数为 0"):
        _refresh(store, daily)
    assert _audit_counts(store) == (0, 0)


def test_planner_excludes_history_and_immature_days(tmp_path):
    store, _, _ = _make_store(tmp_path)
    next_day = _CALENDAR[_CALENDAR.index(_DAY) + 1]
    plans = plan_label_refreshes(
        store, ["20260702", _DAY, next_day], _CALENDAR, input_watermark="20260803"
    )
    assert [plan.trade_date for plan in plans] == [_DAY]
    assert plans[0].buy_date == "20260706" and plans[0].endpoint_date == "20260803"
    earlier_watermark = _CALENDAR[_CALENDAR.index("20260803") - 1]
    assert plan_label_refreshes(store, [_DAY], _CALENDAR, earlier_watermark) == []


def test_planner_later_only_rejects_incomplete_earlier_day(tmp_path):
    store, _, _ = _make_store(tmp_path)
    later = _CALENDAR[_CALENDAR.index(_DAY) + 1]
    with pytest.raises(RuntimeError, match=rf"乱序：{_DAY}.*candidate"):
        plan_label_refreshes(
            store,
            [later],
            _CALENDAR,
            input_watermark="20260804",
            validate_panel=False,
        )


def test_planner_supports_cli_empty_immature_and_controlled_start(tmp_path):
    store, _, _ = _make_store(tmp_path)
    later = _CALENDAR[_CALENDAR.index(_DAY) + 1]
    assert plan_label_refreshes(store, [], _CALENDAR, "20260804") == []

    immature_watermark = _CALENDAR[_CALENDAR.index("20260803") - 1]
    assert plan_label_refreshes(store, [later], _CALENDAR, immature_watermark) == []

    plans = plan_label_refreshes(
        store,
        [later],
        _CALENDAR,
        input_watermark="20260804",
        start_date=later,
        validate_panel=False,
    )
    assert [plan.trade_date for plan in plans] == [later]


def test_planner_completed_earlier_day_may_be_omitted(tmp_path):
    store, _, daily = _make_store(tmp_path)
    _refresh(store, daily)
    later = _CALENDAR[_CALENDAR.index(_DAY) + 1]
    plans = plan_label_refreshes(
        store,
        [later],
        _CALENDAR,
        input_watermark="20260804",
        validate_panel=False,
    )
    assert [plan.trade_date for plan in plans] == [later]


def test_historical_shadow_uses_frozen_row_domain_and_production_planner_excludes_it(tmp_path):
    store, _, daily = _make_store(tmp_path, date="20260702")
    result = _refresh(store, daily, _date="20260702", input_watermark="20260731")
    assert result.row_domain == "label_filtered"
    assert result.endpoint_date == "20260731"
    assert plan_label_refreshes(store, ["20260702"], _CALENDAR, "20260731") == []


def test_long_holiday_calendar_gap_uses_trading_index_not_natural_days(tmp_path):
    store, _, daily = _make_store(tmp_path)
    calendar = [d for d in _CALENDAR if d not in {"20260707", "20260708", "20260709"}]
    endpoint = calendar[calendar.index(_DAY) + 21]
    result = _refresh(store, daily, trading_dates=calendar, input_watermark=endpoint)
    assert endpoint != "20260803"
    assert result.endpoint_date == endpoint


def test_planner_rejects_sealed_later_day_before_earlier_completion(tmp_path):
    store, _, _ = _make_store(tmp_path)
    later = _panel_frame("20260706")
    _write_panel(store, later)
    for name, table in _forming_tables(later).items():
        sealed = table.assign(label_value=0.01, neu_label_value=0.01, maturity_status="sealed")
        store.append_labels(name, sealed)
    with pytest.raises(RuntimeError, match="乱序"):
        plan_label_refreshes(store, [_DAY, "20260706"], _CALENDAR, "20260804")


def test_completed_noop_rejects_current_file_fingerprint_drift(tmp_path):
    store, _, daily = _make_store(tmp_path)
    _refresh(store, daily)
    path = tmp_path / "labels" / "y_ret_5" / f"{_DAY}.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "label_value"] = frame.loc[0, "label_value"] + 0.01
    frame.to_parquet(path, index=False)
    with pytest.raises(LabelRefreshRecoveryError, match="登记不一致"):
        _refresh(store, daily)
    assert _audit_counts(store) == (1, 1)


@pytest.mark.parametrize("case", ["missing", "empty", "missing_table", "empty_value"])
def test_noop_and_readiness_reject_invalid_completion_outputs(tmp_path, case):
    store, _, daily = _make_store(tmp_path)
    _refresh(store, daily)
    label_paths = [tmp_path / "labels" / name / f"{_DAY}.parquet" for name in LABEL_TABLES]
    labels_before = {path: path.read_bytes() for path in label_paths}
    snapshot = store.manifest.snapshot()
    completion = snapshot["label_refreshes"][0]
    if case == "missing":
        del completion["output_fingerprints"]
    elif case == "empty":
        completion["output_fingerprints"] = {}
    elif case == "missing_table":
        del completion["output_fingerprints"]["y_ret_10"]
    else:
        completion["output_fingerprints"]["y_ret_20"] = ""
    manifest_path = store.manifest.path
    manifest_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_before = manifest_path.read_bytes()
    reopened = PanelDataStore(tmp_path)

    with pytest.raises((TypeError, ValueError), match="output_fingerprints"):
        _refresh(reopened, daily)
    with pytest.raises(RuntimeError, match="唯一完成审计"):
        validate_label_day_readiness(reopened, _DAY)

    assert _audit_counts(reopened) == (1, 1)
    assert manifest_path.read_bytes() == manifest_before
    assert not reopened.wal_path.exists()
    for path, payload in labels_before.items():
        assert path.read_bytes() == payload


def test_missing_whole_buy_endpoint_is_rejected_before_intent(tmp_path):
    store, _, daily = _make_store(tmp_path)
    del daily["20260706"]
    with pytest.raises(ValueError, match="必需端点"):
        _refresh(store, daily)
    assert _audit_counts(store) == (0, 0)


def test_file_and_frame_identity_detect_content_changes(tmp_path):
    frame = pd.DataFrame({"ts_code": ["a", "b"], "value": [1.0, np.nan]})
    assert frame_content_digest(frame) == frame_content_digest(frame.copy())
    assert frame_content_digest(frame) != frame_content_digest(frame.assign(value=[1.0, 2.0]))
    path = tmp_path / "input.bin"
    path.write_bytes(b"before")
    records = collect_file_identities([path])
    assert verify_file_identities(records) == []
    path.write_bytes(b"after")
    assert verify_file_identities(records) == [str(path.resolve()).replace("\\", "/")]


@pytest.mark.parametrize(
    "direct_dependency",
    [
        "src/lazybull/common/date_utils.py",
        "src/lazybull/features/labels.py",
        "src/lazybull/v2/common/types.py",
        "src/lazybull/v2/store/write_lock.py",
    ],
)
def test_producer_digest_ignores_unrelated_file_but_tracks_closure_and_environment(
    tmp_path, monkeypatch, direct_dependency
):
    from src.lazybull.v2.store import store_identity as identity

    root = tmp_path / "repo"
    for relative in identity._PRODUCER_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("base", encoding="utf-8")
    first = producer_identity(root)
    (root / "scripts" / "unrelated.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "unrelated.py").write_text("changed", encoding="utf-8")
    assert producer_identity(root)["producer_digest"] == first["producer_digest"]

    direct_path = root / direct_dependency
    direct_path.write_text("changed", encoding="utf-8")
    assert producer_identity(root)["producer_digest"] != first["producer_digest"]
    direct_path.write_text("base", encoding="utf-8")
    assert producer_identity(root)["producer_digest"] == first["producer_digest"]

    monkeypatch.setattr(identity, "environment_identity", lambda: {"python": "another"})
    assert producer_identity(root)["producer_digest"] != first["producer_digest"]
