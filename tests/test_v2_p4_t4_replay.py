# -*- coding: utf-8 -*-
"""v2 P4 T4 paper_replay 隔离测试；全部使用 tmp_path 合成数据。"""

from __future__ import annotations

import json
import socket

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p4.t4_compare_hosts import (
    canonical_nav,
    canonical_positions,
    canonical_trades,
    collect_shared_inputs,
    compare_nav,
    compare_positions,
    compare_trades,
    run_backtest_reference,
    write_backtest_batch,
)
from scripts.v2_p4.t4_gate_runner import _config_for_runs, _no_network, capture_data_state
from scripts.v2_p4.t05b_idempotency_probe import CallRecorder
from scripts.v2_p4.t05b_spike_runner import _DeterministicSignal, synthetic_cost_settings
from src.lazybull.data import DataLoader, Storage
from src.lazybull.universe import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.types import OrderSide, TradeDate
from src.lazybull.v2.core.accounting.engine_state import (
    dumps_blob,
    loads_blob,
    restore_state,
    state_from_blob,
)
from src.lazybull.v2.core.execution.run_loop import _resolve_step_calendar
from src.lazybull.v2.evidence.runs_loader import load_runs_batch
from src.lazybull.v2.evidence.runs_schema import HOST_MODES, validate_host_mode
from src.lazybull.v2.hosts.paper.config_adapter import to_engine_kwargs, to_trading_config
from src.lazybull.v2.hosts.paper.host import PaperHost
from src.lazybull.v2.hosts.paper.replay import (
    PaperReplayError,
    ReplayRowDomainError,
    assert_single_day_row_domain,
    resolve_replay_window,
    run_paper_replay,
    write_paper_replay_batch,
)
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import bootstrap_manifest

_CODES = [f"0000{i:02d}.SZ" for i in range(8)]
_CALENDAR_STRS = pd.bdate_range("2026-07-03", periods=12).strftime("%Y%m%d").tolist()
_CALENDAR = [pd.Timestamp(day) for day in _CALENDAR_STRS]
_FLAT = {
    "model_version": 7,
    "top_n": 2,
    "rebalance_freq": 5,
    "stagger_tranches": 1,
    "exclude_st": True,
    "min_list_days": 365,
    "position_sizing": "equal",
    "buy_price": "close",
    "sell_price": "open",
    "initial_capital": 1_000_000.0,
    "min_buy_value_ratio": 0.0,
    "horizon": 20,
    "universe": "mainboard",
    "engine": "v2",
}


@pytest.fixture(autouse=True)
def _costs():
    with synthetic_cost_settings():
        yield


def _signal_plan():
    return {
        day: (
            [(_CODES[0], 0.9), (_CODES[1], 0.8)]
            if (index // 5) % 2 == 0
            else [(_CODES[2], 0.95), (_CODES[3], 0.85)]
        )
        for index, day in enumerate(_CALENDAR_STRS)
    }


def _make_host(tmp_path, run_root, *, replay=True, available_through=None):
    data_root = tmp_path / "data"
    panel = PanelDataStore(data_root)
    bootstrap_manifest(panel)
    storage = Storage(root_path=str(data_root))
    daily_dir = storage.clean_path / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    for day in _CALENDAR_STRS:
        if available_through is not None and day > available_through:
            continue
        n = len(_CODES)
        feature = pd.DataFrame(
            {
                "trade_date": [day] * n,
                "ts_code": _CODES,
                "ret_1": np.linspace(0.1, 0.17, n),
                "ret_5": np.linspace(0.2, 0.27, n),
                "bp": np.linspace(0.01, 0.08, n),
                "amount_ma20": np.full(n, 60000.0),
                "total_mv": np.full(n, 800000.0),
                "sw_l1_code": ["801780"] * n,
            }
        )
        panel.append_features(TradeDate.from_str(day), "core", feature)
        closes = [10.0 * (1 + 0.1 * index) for index in range(n)]
        price = pd.DataFrame(
            {
                "trade_date": [day] * n,
                "ts_code": _CODES,
                "close": closes,
                "close_adj": closes,
                "open": closes,
                "open_adj": closes,
                "is_suspended": [0] * n,
                "is_limit_up": [0] * n,
                "is_limit_down": [0] * n,
                "vol": np.full(n, 1000.0),
                "pct_chg": np.full(n, 0.0),
                "is_st": [0] * n,
                "list_days": [1000] * n,
                "tradable": [1] * n,
            }
        )
        price.to_parquet(daily_dir / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False)
    config = to_trading_config(_FLAT)
    stock_basic = pd.DataFrame(
        {
            "ts_code": _CODES,
            "name": [f"股票{index}" for index in range(len(_CODES))],
            "market": ["主板"] * len(_CODES),
            "list_date": ["20200101"] * len(_CODES),
        }
    )
    universe = BasicUniverse(
        stock_basic, exclude_st=True, min_list_days=365, markets=["主板"], verbose=False
    )
    plan = _signal_plan()
    return PaperHost(
        store=PaperStateStore(root_path=str(run_root / "paper_state")),
        trading_config=config,
        universe=universe,
        signal=_DeterministicSignal(2, plan, plan[_CALENDAR_STRS[0]]),
        cost_model=CostModel(),
        stock_basic=stock_basic,
        data_storage=storage,
        panel_store=panel,
        trading_calendar=list(_CALENDAR),
        engine_kwargs=to_engine_kwargs(_FLAT, config),
        feature_columns=["ret_1", "ret_5", "bp"],
        train_params={},
        data_loader=DataLoader(storage=storage, verbose=False),
        schedule_mode="replay" if replay else "live",
        replay_calendar=list(_CALENDAR) if replay else None,
        verbose=False,
    )


def _state_fields(host):
    return loads_blob(host.store.load_engine_state_blob())["fields"]


def _limit_up_day1(tmp_path, code):
    path = tmp_path / "data" / "clean" / "daily" / "2026-07-06.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["ts_code"] == code, "is_limit_up"] = 1
    frame.to_parquet(path, index=False)


def test_live_t0_freezes_instruction_shares_and_t1_executes_them(tmp_path, monkeypatch):
    monkeypatch.setitem(_FLAT, "initial_capital", 2_000_000.0)
    run_root = tmp_path / "run"
    host = _make_host(tmp_path, run_root, replay=False)
    host.run_t0(_CALENDAR[0])

    next_day = _CALENDAR_STRS[1]
    frame = host.store.load_instructions_frame(next_day)
    buys = frame[frame["action"] == "buy"]
    assert not buys.empty
    expected = {str(row.ts_code): int(row.shares) for row in buys.itertuples()}
    pending = _state_fields(host)["pending_signals"][_CALENDAR[0]]
    assert expected == {
        str(code): int(shares) for code, shares in pending["instruction_shares"].items()
    }

    host.run_t1(_CALENDAR[1])
    filled = {
        str(trade["ts_code"]): int(trade["shares"])
        for trade in host.last_day_result.fills
        if trade.get("action") == "buy"
    }
    assert set(filled) == set(expected)
    # 现金/手续费不足时允许按整手缩量；但必须以固化股数为上限，
    # 不能重新按权重重算出更大的偏差。
    assert all(shares <= expected[code] for code, shares in filled.items())
    assert any(shares == expected[code] for code, shares in filled.items())


def test_live_t1_failure_creates_t0_completion_and_t2_execution(tmp_path):
    run_root = tmp_path / "run"
    host = _make_host(tmp_path, run_root, replay=False)
    _limit_up_day1(tmp_path, _CODES[0])
    host.run_t0(_CALENDAR[0])
    host.run_t1(_CALENDAR[1])

    completion = host.store.load_instructions_frame(_CALENDAR_STRS[2])
    buys = completion[completion["action"] == "buy"]
    assert not buys.empty
    assert all("补位槽位-信号生成" in str(reason) for reason in buys["reason"])

    host.run_t1(_CALENDAR[2])
    filled = [
        trade for trade in host.last_day_result.fills if trade.get("action") == "buy"
    ]
    assert filled
    assert {str(trade["ts_code"]) for trade in filled} == set(buys["ts_code"].astype(str))


def test_live_t2_price_drift_keeps_instruction_shares(tmp_path):
    run_root = tmp_path / "run"
    host = _make_host(tmp_path, run_root, replay=False)
    host.run_t0(_CALENDAR[0])
    frame = host.store.load_instructions_frame(_CALENDAR_STRS[1])
    expected = {
        str(row.ts_code): int(row.shares)
        for row in frame[frame["action"] == "buy"].itertuples()
    }
    path = tmp_path / "data" / "clean" / "daily" / "2026-07-06.parquet"
    prices = pd.read_parquet(path)
    prices.loc[:, ["close", "close_adj", "open", "open_adj"]] *= 1.5
    prices.to_parquet(path, index=False)
    host.run_t1(_CALENDAR[1])
    filled = {
        str(trade["ts_code"]): int(trade["shares"])
        for trade in host.last_day_result.fills
        if trade.get("action") == "buy"
    }
    assert set(filled) == set(expected)
    assert all(shares <= expected[code] for code, shares in filled.items())
    assert any(shares == expected[code] for code, shares in filled.items())


def test_residual_pending_stops_at_five_attempts():
    from src.lazybull.v2.core.execution.buy_execution import _build_residual_pending

    class _Engine:
        positions = {}
        unfilled_slots = {
            "20260717": {
                "attempts": 6,
                "original_signal_date": "20260717",
                "completion_meta": {"A": {"attempts": 6, "original_signal_date": "20260717"}},
                "unfilled_slot_weights": [{"stock": "A", "weight": 0.1}],
            }
        }

    pending, rows, _tranche, meta = _build_residual_pending(_Engine(), [("B", 0.9)])
    assert pending == {}
    assert rows == []
    assert meta == {}


def test_replay_step_disk_and_continuous_reference_exact(tmp_path):
    run_root = tmp_path / "run"
    host = _make_host(tmp_path, run_root)
    result = run_paper_replay(
        host,
        run_root=run_root,
        start_date=_CALENDAR_STRS[0],
        end_date=_CALENDAR_STRS[-1],
        frozen_through=_CALENDAR_STRS[-1],
    )
    assert len(result["_checkpoints"]) == len(_CALENDAR)
    assert len(result["_row_domain_assertions"]) == len(_CALENDAR)
    assert all(row["run_t1_fills_reconciled"] for row in result["_paper_replay_days"])
    assert len(result["_trades"]) > 0
    features, prices, _ = collect_shared_inputs(host, host.replay_calendar)
    reference = run_backtest_reference(
        host, features_by_date=features, price_data=prices, window=host.replay_calendar
    )
    trades, _ = compare_trades(
        canonical_trades(result["_trades"]),
        canonical_trades(reference["_trades"]),
        selected_count=2,
        allow_tie_swaps=False,
    )
    nav, _ = compare_nav(
        canonical_nav(result["_nav_curve"]), canonical_nav(reference["_nav_curve"])
    )
    positions = compare_positions(
        canonical_positions(result["_holdings_snapshot"]),
        canonical_positions(reference["_holdings_snapshot"]),
    )
    assert trades["passed"] and nav["passed"] and positions["passed"]


def test_continuous_step_calendar_default_keeps_window():
    window = list(_CALENDAR[1:7])
    assert _resolve_step_calendar(window, None) is window


def test_continuous_step_calendar_keeps_future_and_window_origin():
    window = list(_CALENDAR[1:7])
    assert _resolve_step_calendar(window, list(_CALENDAR)) == list(_CALENDAR[1:])


@pytest.mark.parametrize(
    "calendar",
    [
        list(_CALENDAR[:3]) + list(_CALENDAR[2:]),
        list(reversed(_CALENDAR)),
        list(_CALENDAR[:3]) + list(_CALENDAR[4:]),
    ],
    ids=["duplicate", "unsorted", "missing_window_day"],
)
def test_continuous_step_calendar_rejects_invalid_calendar(calendar):
    with pytest.raises(ValueError, match="完整执行日历"):
        _resolve_step_calendar(list(_CALENDAR[1:7]), calendar)


@pytest.mark.parametrize("start_index,window_days", [(0, 11), (1, 6)])
def test_replay_last_signal_day_keeps_next_day_plan_without_future_data(
    tmp_path, start_index, window_days
):
    """末日仅知次交易日日历时仍保留真实买入计划，窗口起点不受历史日历偏移影响。"""
    run_root = tmp_path / "run"
    window = list(_CALENDAR[start_index : start_index + window_days])
    last_day = window[-1]
    last_date = last_day.strftime("%Y%m%d")
    next_day = _CALENDAR[start_index + window_days]
    next_date = next_day.strftime("%Y%m%d")
    host = _make_host(tmp_path, run_root, available_through=last_date)
    host.replay_calendar = window

    # 未来仅有日历：行情与 panel 分区均不存在，不能依赖未来价格构造末日计划。
    future_price = host.data_storage.clean_path / "daily" / f"{next_day:%Y-%m-%d}.parquet"
    assert not future_price.exists()
    assert host.panel_store.manifest.partition_meta(f"panel/{next_date}/core.parquet") is None
    result = run_paper_replay(
        host,
        run_root=run_root,
        start_date=window[0].strftime("%Y%m%d"),
        end_date=last_date,
        frozen_through=last_date,
    )

    checkpoint_data = (run_root / "checkpoints" / f"{last_date}.bin").read_bytes()
    checkpoint = loads_blob(checkpoint_data)
    fields = checkpoint["fields"]
    expected_codes = {code for code, _ in _signal_plan()[last_date]}
    assert set(fields["pending_signals"][last_day]["signals"]) == expected_codes
    assert fields["_last_signal_date"] == last_day
    assert set(fields["_signal_dates"]) == set(window[::5])
    first_checkpoint = loads_blob(
        (run_root / "checkpoints" / f"{window[0]:%Y%m%d}.bin").read_bytes()
    )
    assert first_checkpoint["fields"]["_cycle_anchor_idx"] == 0
    assert fields["_cycle_anchor_idx"] == len(window) - 1

    orders = host.last_day_result.orders
    buy_orders = [order for order in orders if order.side is OrderSide.BUY]
    assert {order.ts_code.value for order in buy_orders} == expected_codes
    assert all(order.target_shares.value > 0 for order in buy_orders)
    instructions = host.store.load_instructions_frame(next_date)
    assert instructions is not None
    buys = instructions[instructions["action"] == "buy"]
    assert set(buys["ts_code"]) == expected_codes
    assert set(buys["source_date"]) == {last_date}
    assert set(buys["price_type"]) == {"close"}
    assert (buys["shares"] > 0).all()

    # 权威快照与检查点均保留末日计划；恢复后仍未执行下一日买入。
    assert dumps_blob(checkpoint) == checkpoint_data
    assert host.store.load_engine_state_blob() == checkpoint_data
    restored = host._new_engine({})
    restore_state(restored, state_from_blob(checkpoint))
    assert restored.pending_signals == fields["pending_signals"]
    nav_dates = pd.to_datetime(result["_nav_curve"]["date"]).tolist()
    assert nav_dates == window
    trade_dates = set(pd.to_datetime(result["_trades"]["date"]))
    assert trade_dates <= set(window)
    assert next_day not in trade_dates
    assert len(result["_checkpoints"]) == len(window)

    # 连续参照使用相同完整日历；仅延长索引范围，不驱动窗口外日期。
    features, prices, _ = collect_shared_inputs(host, window)
    reference = run_backtest_reference(
        host, features_by_date=features, price_data=prices, window=window
    )
    trades, _ = compare_trades(
        canonical_trades(result["_trades"]),
        canonical_trades(reference["_trades"]),
        selected_count=2,
        allow_tie_swaps=False,
    )
    nav, _ = compare_nav(
        canonical_nav(result["_nav_curve"]), canonical_nav(reference["_nav_curve"])
    )
    positions = compare_positions(
        canonical_positions(result["_holdings_snapshot"]),
        canonical_positions(reference["_holdings_snapshot"]),
    )
    assert trades["passed"] and nav["passed"] and positions["passed"]


def test_replay_checkpoint_preserves_full_trade_history(tmp_path):
    run_root = tmp_path / "run"
    host = _make_host(tmp_path, run_root)
    result = run_paper_replay(
        host,
        run_root=run_root,
        start_date=_CALENDAR_STRS[0],
        end_date=_CALENDAR_STRS[-1],
        frozen_through=_CALENDAR_STRS[-1],
    )
    fields = loads_blob(host.store.load_engine_state_blob())["fields"]
    assert len(fields["trades"]) == len(result["_trades"])
    assert len(fields["portfolio_values"]) == len(_CALENDAR)


def test_replay_window_rejects_label_filtered_and_calendar_mismatch(tmp_path):
    host = _make_host(tmp_path, tmp_path / "run")
    with pytest.raises(ReplayRowDomainError):
        resolve_replay_window(
            host, start_date="20260702", end_date="20260731", frozen_through="20261026"
        )
    with pytest.raises(PaperReplayError, match="不一致"):
        resolve_replay_window(
            host, start_date="20260703", end_date="20260706", frozen_through="20261026"
        )


def test_replay_rejects_uncontained_run_root(tmp_path):
    host = _make_host(tmp_path, tmp_path / "run")
    with pytest.raises(PaperReplayError, match="paper_state"):
        run_paper_replay(
            host,
            run_root=tmp_path / "elsewhere",
            start_date="20260703",
            end_date="20260731",
            frozen_through="20261026",
        )


def test_single_day_row_domain_detects_manifest_tamper(tmp_path):
    host = _make_host(tmp_path, tmp_path / "run")
    key = f"panel/{_CALENDAR_STRS[0]}/core.parquet"
    host.panel_store.manifest._data["partitions"][key]["row_domain"] = "label_filtered"
    with pytest.raises(ReplayRowDomainError, match="实际行域"):
        assert_single_day_row_domain(host.panel_store, _CALENDAR_STRS[0])


def test_host_mode_closed_enum():
    assert HOST_MODES == {"backtest", "paper_replay", "paper_shadow"}
    for value in sorted(HOST_MODES):
        assert validate_host_mode(value) == value
    for value in (None, "", "legacy", 1):
        with pytest.raises(ValueError, match="host_mode"):
            validate_host_mode(value)


def test_paper_replay_single_fold_write_readback_and_invalid_mode(tmp_path):
    run_root = tmp_path / "run"
    host = _make_host(tmp_path, run_root)
    result = run_paper_replay(
        host,
        run_root=run_root,
        start_date=_CALENDAR_STRS[0],
        end_date=_CALENDAR_STRS[-1],
        frozen_through=_CALENDAR_STRS[-1],
    )
    out = write_paper_replay_batch(
        result,
        batch_id="paper_batch",
        runs_root=tmp_path / "runs",
        config={"bt_top_n": 2},
        code_state={"git_commit": "synthetic", "git_dirty": False},
        data_state={"data_state_id": "synthetic_ds"},
        model_version=7,
        train_start="20200101",
        train_end="20260702",
        frozen_through=_CALENDAR_STRS[-1],
    )
    batch = load_runs_batch(out)
    assert batch.batch_meta["host_mode"] == "paper_replay"
    assert list(batch.folds) == [0]
    notes = batch.batch_meta["meta_notes"]
    assert json.loads(notes["replay_window"])["trading_days"] == len(_CALENDAR)
    meta_path = out / "batch_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["host_mode"] = "invalid"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="host_mode"):
        load_runs_batch(out)


def test_replay_window_rejects_after_frozen_through(tmp_path):
    host = _make_host(tmp_path, tmp_path / "run")
    with pytest.raises(ReplayRowDomainError, match="须 ⊆"):
        resolve_replay_window(
            host, start_date="20260703", end_date="20260731", frozen_through="20260730"
        )


def test_capture_data_state_uses_local_watermarks(tmp_path):
    host = _make_host(tmp_path, tmp_path / "run")
    state = capture_data_state(host.panel_store.root)
    assert state["sources"]["clean_daily"] == _CALENDAR_STRS[-1]
    assert state["sources"]["panel"] == _CALENDAR_STRS[-1]
    assert state["panel"]["row_domain_segments"]["structural_from"] == "20260703"
    assert len(state["data_state_id"]) == 16


def test_no_network_guard_fail_fast_and_records(tmp_path):
    recorder = CallRecorder()
    client = socket.socket()
    try:
        with _no_network(recorder):
            with pytest.raises(RuntimeError, match="禁网守卫"):
                client.connect(("127.0.0.1", 9))
    finally:
        client.close()
    assert len(recorder.events) == 1
    assert recorder.events[0]["kind"] == "network"


def test_config_for_runs_contains_execution_identity(tmp_path):
    host = _make_host(tmp_path, tmp_path / "run")
    config = _config_for_runs(host, enable_risk_budget=True)
    assert config["buy_price"] == "close" and config["sell_price"] == "open"
    assert config["initial_capital"] == 1_000_000.0
    assert config["enable_risk_budget"] is True


def test_backtest_writer_default_host_mode_unchanged(tmp_path):
    run_root = tmp_path / "run"
    host = _make_host(tmp_path, run_root)
    result = run_paper_replay(
        host,
        run_root=run_root,
        start_date=_CALENDAR_STRS[0],
        end_date=_CALENDAR_STRS[-1],
        frozen_through=_CALENDAR_STRS[-1],
    )
    features, prices, _ = collect_shared_inputs(host, host.replay_calendar)
    reference = run_backtest_reference(
        host, features_by_date=features, price_data=prices, window=host.replay_calendar
    )
    out = write_backtest_batch(
        reference,
        host=host,
        batch_id="backtest_batch",
        runs_root=tmp_path / "runs",
        config={"bt_top_n": 2},
        code_state={"git_commit": "synthetic", "git_dirty": False},
        data_state={"data_state_id": "synthetic_ds"},
        baseline_ref="paper_batch",
    )
    batch = load_runs_batch(out)
    assert batch.batch_meta["host_mode"] == "backtest"
    assert list(batch.folds) == [0]


def test_empty_trade_frames_compare_exact():
    empty = pd.DataFrame(
        columns=[
            "trade_date",
            "signal_date",
            "ts_code",
            "action",
            "price",
            "shares",
            "amount",
            "cost",
            "sell_type",
        ]
    )
    result, detail = compare_trades(empty, empty, selected_count=20, allow_tie_swaps=True)
    assert result["passed"] and result["compared_rows"] == 0
    assert result["tie_swap_count"] == 0 and detail.empty
