"""经营月报A基准的合成缓存、连续账户和资本独立性验证。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.v2_p7 import operating_monthly_baseline as baseline
from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook, NativeCorporateMixin
from scripts.v2_p7.buffer_native_replay import (
    FrozenRankSignal,
    RecordNativeMixin,
    compare_exact,
    initialize_driver,
    result_payload,
    sha256,
)
from scripts.v2_p7.run_buffer_native import cash_anchor
from scripts.v2_p7.t4.corporate import CorporateError
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
from src.lazybull.v2.core.execution.price_index import prepare_price_index
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML


def _stitch_inputs() -> tuple:
    days = [d.strftime("%Y%m%d") for d in pd.bdate_range("2026-01-01", periods=9)]
    old = {d: [("A", 1.0)] for d in days[:4]}
    new = {d: [("B", 1.0)] for d in days[6:]}
    gap = {d: [("A", 1.0)] for d in days[4:6]}
    return (
        old,
        {"signal_days": list(old), "calendar": days[:6]},
        new,
        {"days": list(new)},
        days,
        {
            "start": days[2],
            "end": days[-1],
            "gap": gap,
            "gap_identity": {"days": list(gap), "model_version": 24021},
            "warmup_start": days[0],
        },
    )


def test_stitch_keeps_gap_rankings_and_single_cash_clock() -> None:
    old, oi, new, ni, days, options = _stitch_inputs()
    result = baseline.stitch_rankings(old, oi, new, ni, days, **options)
    assert result.anchor == days[1]
    assert result.calendar == tuple(days[2:])
    assert result.price_days == tuple(days)
    assert set(result.ranked) == set(days[2:])
    assert result.identity["gap_days"] == days[4:6]
    assert result.identity["clock_reset_at_model_seam"] is False
    assert result.identity["not_virgin_holdout"] is True
    assert old == {d: [("A", 1.0)] for d in days[:4]}


@pytest.mark.parametrize(
    "fault", ["missing_gap", "wrong_model", "duplicate_code", "unsorted_score", "missing_anchor"]
)
def test_stitch_rejects_undeclared_gap_or_rank_drift(fault: str) -> None:
    old, oi, new, ni, days, options = _stitch_inputs()
    if fault == "missing_gap":
        options["gap"].pop(days[4])
    elif fault == "wrong_model":
        options["gap_identity"]["model_version"] = 24022
    elif fault == "duplicate_code":
        old[days[0]] = [("A", 1.0), ("A", 0.5)]
    elif fault == "unsorted_score":
        options["gap"][days[4]] = [("A", 0.1), ("B", 0.8)]
    else:
        options["start"] = days[0]
    with pytest.raises(ValueError):
        baseline.stitch_rankings(old, oi, new, ni, days, **options)


def _recent_cache(root: Path) -> Path:
    cache = root / "recent"
    cache.mkdir()
    (root / "source.py").write_text("# 合成源\n", encoding="utf-8")
    (root / "input.bin").write_bytes(b"frozen-input")
    days = ["20260203", "20260204"]
    audit = {
        "days": days,
        "model_version": 24022,
        "input_files": [{"path": "input.bin", "sha256": sha256(root / "input.bin")}],
        "absent_input_paths": ["absent.parquet"],
    }
    validation = {
        "complete": True,
        "input_files_verified_after": True,
        "days": 2,
        "ranked_rows": 1,
        "day_records": [{"day": days[0], "ranked_rows": 1}, {"day": days[1], "ranked_rows": 0}],
        "empty_days": [{"day": days[1], "empty_reason": "kernel_refused"}],
    }
    for name, value in [("audit.json", audit), ("validation.json", validation)]:
        (cache / name).write_text(json.dumps(value), encoding="utf-8")
    pd.DataFrame([{"trade_date": days[0], "rank": 1, "ts_code": "A", "score": 1.0}]).to_parquet(
        cache / "ranked.parquet", index=False
    )
    manifest = {
        "schema": "p7_buffer_native_rankings/v1",
        "complete": True,
        "model_version": 24022,
        "spec_sha256": baseline.SPEC_SHA,
        "source_code": {"path": "source.py", "sha256": sha256(root / "source.py")},
        "files": {
            name: sha256(cache / name)
            for name in ("audit.json", "validation.json", "ranked.parquet")
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache


def test_recent_cache_preserves_registered_empty_day_without_inference(tmp_path: Path) -> None:
    cache = _recent_cache(tmp_path)
    ranks, identity = baseline.read_recent_rankings(tmp_path, cache.relative_to(tmp_path))
    assert ranks == {"20260203": [("A", 1.0)], "20260204": []}
    assert identity["empty_days"] == ["20260204"]
    assert identity["input_files_verified_now"]


@pytest.mark.parametrize(
    "fault", ["input_drift", "source_drift", "appeared_input", "ranked_drift", "unregistered_empty"]
)
def test_recent_cache_detects_input_and_declared_empty_drift(tmp_path: Path, fault: str) -> None:
    cache = _recent_cache(tmp_path)
    if fault in {"input_drift", "source_drift", "appeared_input"}:
        name = {
            "input_drift": "input.bin",
            "source_drift": "source.py",
            "appeared_input": "absent.parquet",
        }[fault]
        (tmp_path / name).write_bytes(b"changed")
    elif fault == "ranked_drift":
        (cache / "ranked.parquet").write_bytes(b"changed")
    else:
        path = cache / "validation.json"
        value = json.loads(path.read_text())
        value["empty_days"] = []
        path.write_text(json.dumps(value))
        manifest = json.loads((cache / "manifest.json").read_text())
        manifest["files"]["validation.json"] = sha256(path)
        (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        baseline.read_recent_rankings(tmp_path, cache.relative_to(tmp_path))


def test_gap_provider_does_not_mutate_holdout_dates_or_model(monkeypatch) -> None:
    from scripts.v2_p7 import buffer_native_inputs as original

    provider = object.__new__(baseline.GapRankedProvider)
    provider.allowed_days = frozenset(["20260106"])
    frame = pd.DataFrame({"trade_date": ["20260106"], "ts_code": ["A"], "x": [1]})
    provider._features = lambda day: frame
    provider.loader = SimpleNamespace(load_clean_daily_by_date=lambda day, auto_ensure: frame)
    provider.universe = SimpleNamespace(get_stocks=lambda date, quote_data: ["A"])
    provider.signal = SimpleNamespace(generate_ranked=lambda date, codes, data: [("A", 2.0)])
    provider.day_records = []
    assert provider("20260106") == [("A", 2.0)]
    assert provider.day_records[0]["model_version"] == 24021
    assert original.START == "20260203"
    assert original.END == "20261029"
    assert original.MODEL_VERSION == 24022
    with pytest.raises(ValueError, match="冻结接缝"):
        provider("20260203")


def test_gap_cache_reuse_never_infers_again_and_checks_input_identity(
    tmp_path, monkeypatch
) -> None:
    days = ["20260106", "20260107"]
    (tmp_path / "source.bin").write_bytes(b"frozen")
    calls = []

    class Provider:
        def __init__(self, root, allowed):
            self.day_records = []

        def __call__(self, day):
            calls.append(day)
            rows = [("A", 1.0)] if day == days[0] else []
            self.day_records.append({"day": day, "ranked_rows": len(rows)})
            return rows

    def identity(root, allowed, provider):
        return {
            "schema": "p7_operating_monthly_gap_inputs/v1",
            "model_version": 24021,
            "days": list(allowed),
            "input_files": {"source.bin": sha256(root / "source.bin")},
            "absent_input_paths": [],
        }

    monkeypatch.setattr(baseline, "GapRankedProvider", Provider)
    monkeypatch.setattr(baseline, "_gap_identity", identity)
    cache = tmp_path / "new_gap"
    built, original = baseline.build_gap_rankings(tmp_path, cache, days=days, spec_sha256="a" * 64)
    inherited, current = baseline.read_gap_rankings(tmp_path, cache, spec_sha256="a" * 64)
    assert built == inherited == {days[0]: [("A", 1.0)], days[1]: []}
    assert original == current
    assert calls == days
    (tmp_path / "source.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="输入发生变化"):
        baseline.read_gap_rankings(tmp_path, cache, spec_sha256="a" * 64)


def _synthetic_account_inputs(tmp_path: Path) -> tuple:
    dates = list(pd.bdate_range("2024-01-02", periods=51))
    days = [d.strftime("%Y%m%d") for d in dates]
    codes = [f"{i:06}.SZ" for i in range(1, 26)]
    ranks = {d: [(c, 1.0 - i / 100) for i, c in enumerate(codes)] for d in days[1:]}
    prices = pd.DataFrame(
        [
            {
                "trade_date": d,
                "ts_code": code,
                "open": 123.47 + i * 0.37 + t * 0.007,
                "close": 123.48 + i * 0.37 + t * 0.007,
                "open_adj": 123.47 + i * 0.37 + t * 0.007,
                "close_adj": 123.48 + i * 0.37 + t * 0.007,
                "is_st": False,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
                "tradable": True,
            }
            for t, d in enumerate(days)
            for i, code in enumerate(codes)
        ]
    )
    dividend = {
        "ts_code": codes[0],
        "end_date": "20231231",
        "ann_date": days[0],
        "div_proc": "实施",
        "stk_div": 0.2,
        "cash_div": 1.0,
        "cash_div_tax": 1.25,
        "record_date": days[10],
        "ex_date": days[11],
        "pay_date": days[18],
        "imp_ann_date": days[0],
        "div_listdate": days[15],
    }
    pd.DataFrame([dividend]).to_parquet(tmp_path / "2023-12-31.parquet", index=False)
    book = NativeCorporateBook(calendar=days, dividend_dir=tmp_path)
    inputs = baseline.BaselineInputs(ranks, tuple(days[1:]), days[0], tuple(days), {})
    return inputs, prices, book, codes


def test_separate_capitals_run_native_cash_shares_and_corporate_book(tmp_path, monkeypatch) -> None:
    inputs, prices, book, codes = _synthetic_account_inputs(tmp_path)
    requested = []

    def kwargs(root, ranked, capital, current_config):
        requested.append((capital, current_config))
        return {
            "portfolio_merger": SingleSleevePortfolioMerger(),
            "universe": SimpleNamespace(get_stocks=lambda date, quote_data=None: codes),
            "signal": FrozenRankSignal(ranked),
            "features_by_date": {d: pd.DataFrame({"x": [1]}) for d in ranked},
            "initial_capital": float(capital),
            "cost_model": CostModel(0.0001954, 5.0, 0.0005, 0.0005),
            "rebalance_freq": 20,
            "holding_period": 20,
            "stagger_tranches": 2,
            "sell_timing": "open",
            "verbose": False,
            "position_sizing": "equal",
            "max_weight_per_stock": 0.15,
            "min_buy_value_ratio": 0.0,
            "enable_pending_order": True,
            "enable_position_completion": True,
            "completion_window_days": 5,
            "data_storage": SimpleNamespace(
                load_raw_by_date=lambda dataset, trade_date: pd.DataFrame(
                    columns=["ts_code", "suspend_type"]
                )
            ),
        }

    monkeypatch.setattr(baseline, "engine_kwargs", kwargs)
    index = prepare_price_index(prices)
    large = baseline.run_baseline_account(
        tmp_path,
        inputs=inputs,
        prices=prices,
        indexes=index,
        capital=1_000_000,
        corporate_book=book,
    )
    small = baseline.run_baseline_account(
        tmp_path, inputs=inputs, prices=prices, indexes=index, capital=800_000, corporate_book=book
    )
    assert requested == [(1_000_000, False), (800_000, False)]
    assert large["validation"]["valid"] and small["validation"]["valid"]
    assert large["daily"][0]["trade_date"] == small["daily"][0]["trade_date"] == inputs.anchor
    assert large["daily"][0]["cash"] == 1_000_000
    assert small["daily"][0]["cash"] == 800_000
    assert large["corporate"]["lock_count"] == small["corporate"]["lock_count"] == 1
    assert large["corporate"]["unlock_count"] == small["corporate"]["unlock_count"] == 1
    assert large["corporate"]["pay_count"] == small["corporate"]["pay_count"] == 1
    assert any(row["receivable"] > 0 for row in small["daily"])
    large_buy = {row["stock"]: row["shares"] for row in large["trades"] if row["action"] == "buy"}
    small_buy = {row["stock"]: row["shares"] for row in small["trades"] if row["action"] == "buy"}
    assert any(small_buy[c] != 0.8 * large_buy[c] for c in set(large_buy) & set(small_buy))
    assert small["baseline_identity"]["capital_scaled_from_other_nav"] is False
    assert small["baseline_identity"]["cross_sleeve_netting"] is False
    with pytest.raises(ValueError, match="固定half-Kelly"):
        baseline.run_baseline_account(
            tmp_path,
            inputs=inputs,
            prices=prices,
            indexes=index,
            capital=800_000,
            current_config=True,
            corporate_book=book,
        )


def _cash_rows(capital: float) -> list[dict]:
    return [
        {
            "trade_date": day,
            "total_value": capital,
            "cash": capital,
            "market_value": 0.0,
            "receivable": 0.0,
            "fees": 0.0,
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "nav": 1.0,
        }
        for day in ["20240101", "20240102"]
    ]


def test_combination_adds_actual_independent_assets_and_rejects_intersection() -> None:
    a, b = _cash_rows(800_000), _cash_rows(200_000)
    a[1].update(cash=799_990.0, market_value=20.0, total_value=800_010.0, fees=10.0)
    b[1].update(cash=199_980.0, receivable=15.0, total_value=199_995.0, fees=20.0)
    old = copy.deepcopy((a, b))
    result = baseline.combine_independent_daily(a, b, total_capital=1_000_000)
    assert result[1]["nav"] == 1.000005
    assert result[1]["receivable"] == 15
    assert result[1]["fees"] == 30
    assert (a, b) == old
    with pytest.raises(ValueError, match="等长"):
        baseline.combine_independent_daily(a, b[:1], total_capital=1_000_000)
    b[1]["trade_date"] = "20240103"
    with pytest.raises(ValueError, match="日期不一致"):
        baseline.combine_independent_daily(a, b, total_capital=1_000_000)


def test_price_warmup_counts_actual_rows_and_reports_short_histories(tmp_path, monkeypatch) -> None:
    inputs = baseline.BaselineInputs(
        {"20240102": [("A", 1.0), ("B", 0.5)]},
        ("20240102",),
        "20240101",
        ("20180102", "20240101", "20240102"),
        {},
    )
    table = pd.DataFrame(
        {
            "ts_code": ["A"] * 60 + ["B"] * 3 + ["A", "B"],
            "trade_date": ["20231229"] * 63 + ["20240102"] * 2,
        }
    )
    monkeypatch.setattr(
        baseline, "load_prices", lambda root, days, warmup_cutoff: (table, {"synthetic": "sha"})
    )
    _, info = baseline.load_baseline_prices(tmp_path, inputs)
    assert info["ranked_union_insufficient_count"] == 1
    assert info["first_day_ranked_insufficient_count"] == 1
    assert info["ranked_union_insufficient_60_at_start"] == [{"ts_code": "B", "pre_start_rows": 3}]
    assert info["native_variance_fallback_unchanged"] is True


def _lock_engine_kwargs(inputs, codes) -> dict:
    return {
        "portfolio_merger": SingleSleevePortfolioMerger(),
        "universe": SimpleNamespace(get_stocks=lambda date, quote_data=None: codes),
        "signal": FrozenRankSignal(inputs.ranked),
        "features_by_date": {d: pd.DataFrame({"x": [1]}) for d in inputs.ranked},
        "initial_capital": 1_000_000.0,
        "cost_model": CostModel(0.0001954, 5.0, 0.0005, 0.0005),
        "rebalance_freq": 20,
        "holding_period": 20,
        "stagger_tranches": 2,
        "sell_timing": "open",
        "verbose": False,
        "position_sizing": "equal",
        "max_weight_per_stock": 0.15,
        "min_buy_value_ratio": 0.0,
        "enable_pending_order": True,
        "enable_position_completion": True,
        "completion_window_days": 5,
        "data_storage": SimpleNamespace(
            load_raw_by_date=lambda dataset, trade_date: pd.DataFrame(
                columns=["ts_code", "suspend_type"]
            )
        ),
    }


def _drive_lock_engine(engine, inputs, prices, on_day=None) -> dict:
    dates = [pd.Timestamp(day) for day in inputs.calendar]
    initialize_driver(engine, dates, prices, prepare_price_index(prices))
    day, state = cash_anchor(inputs.anchor, int(engine.initial_capital))
    engine.native_corporate.before_day(engine, pd.Timestamp(inputs.anchor))
    engine.native_corporate.after_day(engine, pd.Timestamp(inputs.anchor))
    engine.research_daily.append(day)
    engine.research_states.append(state)
    pending = {}
    for date in dates:
        engine.run_step(date, dates, prices)
        if on_day is not None:
            on_day(engine, date)
        pending[date.strftime("%Y%m%d")] = set(engine.pending_order_manager.pending_orders)
    return pending


def test_locked_whole_sell_stays_pending_through_positive_rank_and_listing_limit_down(
    tmp_path,
) -> None:
    inputs, prices, _, codes = _synthetic_account_inputs(tmp_path)
    days = [inputs.anchor, *inputs.calendar]
    path = tmp_path / "2023-12-31.parquet"
    event = pd.read_parquet(path)
    event["div_listdate"] = days[30]
    event["stk_div"] = 0.3
    event.to_parquet(path, index=False)
    book = NativeCorporateBook(calendar=days, dividend_dir=tmp_path)
    code = codes[0]
    prices.loc[prices.trade_date.eq(days[30]) & prices.ts_code.eq(code), "is_limit_down"] = True
    engine = baseline.OperatingBaselineEngine(
        **_lock_engine_kwargs(inputs, codes), corporate_book=book
    )
    pending = _drive_lock_engine(engine, inputs, prices)
    result = baseline.baseline_result_payload(engine)
    assert baseline.validate_account(result, initial_capital=1_000_000)["valid"]
    sold = [row for row in engine.trades if row["stock"] == code and row["action"] == "sell"]
    assert sold[0]["date"] == pd.Timestamp(days[31])
    assert sold[0]["sell_timing"] == "open"
    first_defer = engine.locked_sell_deferrals[0]
    assert first_defer["ts_code"] == code
    assert first_defer["locked_shares"] > 0
    assert sold[0]["shares"] == first_defer["requested_shares"]
    first_index = days.index(first_defer["trade_date"])
    for day in days[first_index:31]:
        assert (code, "sell") in pending[day]
        assert inputs.ranked[day][0][0] == code
    assert (code, "sell") not in pending[days[31]]
    for state in engine.research_states[first_index:31]:
        assert state["positions"][code]["shares"] == first_defer["requested_shares"]
    assert engine.native_corporate.payload()["blocked_sell_count"] == 0
    assert engine.native_corporate.payload()["held_issue_count"] == 0
    assert result["locked_sell_deferrals"]
    assert code not in result["locked_sell_pending_intents"]


def test_no_lock_at_sell_preserves_old_bridge_all_account_tables(tmp_path) -> None:
    class Original(RecordNativeMixin, NativeCorporateMixin, NativeMergeBacktestEngineML):
        pass

    inputs, prices, book, codes = _synthetic_account_inputs(tmp_path)
    old = Original(**_lock_engine_kwargs(inputs, codes), corporate_book=book)
    new = baseline.OperatingBaselineEngine(
        **_lock_engine_kwargs(inputs, codes), corporate_book=book
    )
    _drive_lock_engine(old, inputs, prices)
    _drive_lock_engine(new, inputs, prices)
    assert compare_exact(result_payload(old), result_payload(new))["all_equal"]
    assert new.locked_sell_deferrals == []
    assert old.native_corporate.payload() == new.native_corporate.payload()


def test_deferred_intent_keeps_original_reason_and_partial_request_remains_strict(tmp_path) -> None:
    inputs, prices, _, codes = _synthetic_account_inputs(tmp_path)
    days = [inputs.anchor, *inputs.calendar]
    path = tmp_path / "2023-12-31.parquet"
    event = pd.read_parquet(path)
    event["div_listdate"] = days[30]
    event.to_parquet(path, index=False)
    book = NativeCorporateBook(calendar=days, dividend_dir=tmp_path)
    engine = baseline.OperatingBaselineEngine(
        **_lock_engine_kwargs(inputs, codes), corporate_book=book
    )
    code = codes[0]

    def on_day(active, date):
        if date == pd.Timestamp(days[12]):
            before = (active.current_capital, len(active.trades), copy.deepcopy(active.positions))
            active._sell_stock_direct(date, code, "stop_loss", "原始合成退出原因", "fixed")
            assert before == (active.current_capital, len(active.trades), active.positions)

    _drive_lock_engine(engine, inputs, prices, on_day)
    sold = [row for row in engine.trades if row["stock"] == code and row["action"] == "sell"]
    assert sold[0]["sell_type"] == "stop_loss"
    assert sold[0]["sell_reason"] == "原始合成退出原因"
    assert sold[0]["trigger_type"] == "fixed"
    assert sold[0]["date"] == pd.Timestamp(days[30])

    # 独立极小构造：部分请求超过可卖股时仍由旧桥报错，绝不转成整仓卖单。
    tiny = object.__new__(baseline.OperatingBaselineEngine)
    tiny.positions = {"X": {"shares": 1000}}
    tiny.trades = []
    tiny.locked_sell_intents = {}
    tiny.locked_sell_deferrals = []
    tiny.pending_order_manager = SimpleNamespace(pending_orders={})
    tiny.native_corporate = SimpleNamespace(
        locked_shares=lambda stock: 900,
        check_sell=lambda *args: (_ for _ in ()).throw(CorporateError("超过可卖")),
    )
    with pytest.raises(CorporateError, match="超过可卖"):
        tiny._sell_stock_direct(pd.Timestamp(days[12]), "X", fraction=0.5)
    assert tiny.pending_order_manager.pending_orders == {}
    assert tiny.locked_sell_deferrals == []
