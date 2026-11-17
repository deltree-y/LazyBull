"""涤纶独立入口的合成封包、真实两槽接线和失败留证；不读生产数据。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p7 import run_hog as runner
from scripts.v2_p7.operating_monthly_analysis import analyze_capital, cash_daily
from scripts.v2_p7.operating_monthly_corporate import prepare_research_dividends

DAYS = [
    "20190329",
    "20190401",
    "20190402",
    "20190502",
    "20190503",
    "20260202",
    "20260203",
    "20260204",
    "20261029",
]


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def digest(path: Path) -> str:
    return runner.sha256(path)


def frozen_source(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    base = root / "research"
    spec_path = root / "spec.md"
    spec_path.write_text("合成冻结规则", encoding="utf-8")
    monkeypatch.setattr(runner, "SPEC_SHA", digest(spec_path))
    write(
        base / "prereg/spec.json",
        {
            "spec_path": "spec.md",
            "sha256": digest(spec_path),
            "capital": list(runner.CAPITALS),
            "codes": list(runner.CODES),
        },
    )
    records = [
        {
            "source_id": str(index),
            "observation_date": day,
            "publication_date": day,
            "available_date": day[:6] + "21",
            "hog": price,
            "feed": 2.0,
            "quality_usable": True,
            "series_id": "synthetic",
            "unit": "CNY/kg",
        }
        for index, (day, price) in enumerate(
            zip(["20181220", "20190120", "20190220", "20190320"], [6.0, 6.2, 6.4, 8.0])
        )
    ]
    records_path, selection_path = (
        base / "input_prices/records.json",
        base / "input_prices/selection.json",
    )
    write(records_path, records)
    write(selection_path, {"reason": "synthetic"})
    record_name, select_name = (
        records_path.relative_to(root).as_posix(),
        selection_path.relative_to(root).as_posix(),
    )
    manifest = {
        "schema": "p7_hog_price_inputs_v1",
        "records_file": record_name,
        "records_sha256": digest(records_path),
        "spec_sha256": digest(spec_path),
        "frozen_before_stock_returns": True,
        "source_selection_file": select_name,
        "files": {record_name: digest(records_path), select_name: digest(selection_path)},
    }
    path = base / "input_prices/source_manifest.json"
    write(path, manifest)
    return base, digest(path)


def synthetic_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": day,
                "ts_code": code,
                "open": 10.0,
                "close": 10.0,
                "is_suspended": 0,
                "is_limit_up": 0,
                "is_limit_down": 0,
                "is_st": 0,
                "tradable": 1,
                "amount": 5000.0,
            }
            for day in DAYS
            for code in runner.CODES
        ]
    )


def inherited_a(root: Path) -> tuple[Path, str]:
    prior = root / "prior"
    quote_files = {}
    quotes = synthetic_quotes()
    for day in DAYS:
        path = root / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        quotes.loc[quotes.trade_date.eq(day)].to_parquet(path, index=False)
        quote_files[path.relative_to(root).as_posix()] = digest(path)
    calendar = root / "data/clean/trade_cal.parquet"
    availability_days = [
        "20190320",
        "20190321",
        "20190322",
        "20190325",
        "20190326",
        "20190327",
        "20190328",
        *DAYS,
    ]
    pd.DataFrame(
        {
            "exchange": ["SSE"] * len(availability_days),
            "is_open": [1] * len(availability_days),
            "cal_date": availability_days,
        }
    ).to_parquet(calendar, index=False)
    raw = root / "data/raw/dividend"
    raw.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": runner.CODES[0],
                "end_date": "20181231",
                "ann_date": "20190301",
                "imp_ann_date": "20190301",
                "div_proc": "预案",
                "cash_div": 0.1,
                "cash_div_tax": 0.125,
                "stk_div": 0.0,
                "record_date": "20190502",
                "ex_date": "20190503",
                "pay_date": "20190503",
                "div_listdate": "20190503",
            }
        ]
    ).to_parquet(raw / "2019-12-31.parquet", index=False)
    write(prior / "source_identity.json", {})
    write(prior / "price_identity.json", {"files": quote_files})
    write(
        prior / "dividend_identity.json",
        {
            (raw / "2019-12-31.parquet")
            .relative_to(root)
            .as_posix(): digest(raw / "2019-12-31.parquet")
        },
    )
    write(prior / "dividend_resolutions.json", [])
    write(
        prior / "baseline_inputs.json",
        {
            "anchor": runner.ANCHOR,
            "start": runner.START,
            "end": runner.END,
            "calendar_file": {
                "path": calendar.relative_to(root).as_posix(),
                "sha256": digest(calendar),
            },
        },
    )
    prepare_research_dividends(
        raw_dir=raw,
        output_dir=prior / "corporate_input",
        calendar=DAYS,
        codes=runner.CODES,
        resolutions=[],
    )
    for capital in runner.CAPITALS:
        for name, initial in (("A100", capital), ("A80", capital * 4 // 5)):
            folder = prior / str(capital) / name
            write(folder / "daily.json", cash_daily(DAYS, initial))
            write(
                folder / "baseline_identity.json",
                {
                    "initial_capital": initial,
                    "capital_scaled_from_other_nav": False,
                    "cross_sleeve_borrowing": False,
                    "cross_sleeve_netting": False,
                },
            )
            write(folder / "validation.json", {"all_passed": True})
    seal_prior(prior)
    return prior, digest(prior / "manifest.json")


def seal_prior(prior: Path) -> None:
    files = {
        p.relative_to(prior).as_posix(): digest(p)
        for p in prior.rglob("*")
        if p.is_file() and p.name not in ("manifest.json", "validation.json")
    }
    # 子目录的 validation 以及公司行为 manifest 均是账户证据。
    for p in prior.rglob("*.json"):
        if p.parent != prior:
            files[p.relative_to(prior).as_posix()] = digest(p)
    write(prior / "manifest.json", {"files": files})
    write(
        prior / "validation.json",
        {"complete": True, "manifest_sha256": digest(prior / "manifest.json")},
    )


def test_frozen_source_validates_then_rejects_mutated_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, sha = frozen_source(tmp_path, monkeypatch)
    assert len(runner.load_frozen_inputs(base, tmp_path, sha, lambda: None)[1]) == 4
    write(base / "input_prices/records.json", [])
    with pytest.raises(ValueError, match="SHA"):
        runner.load_frozen_inputs(base, tmp_path, sha, lambda: None)


def test_bad_source_rejected_before_any_stock_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, _ = frozen_source(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "BASE", base)
    monkeypatch.setattr(
        pd, "read_parquet", lambda *args, **kwargs: pytest.fail("坏封包不得读任何行情")
    )
    with pytest.raises(ValueError, match="manifest SHA"):
        runner.run("run_bad", "0" * 64)
    assert not (base / "run_bad").exists()


def test_first_month_delay_uses_verified_pre_window_calendar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, source_sha = frozen_source(tmp_path, monkeypatch)
    prior, prior_sha = inherited_a(tmp_path)
    _, records, _ = runner.load_frozen_inputs(base, tmp_path, source_sha, lambda: None)
    _, calendar, identity, _ = runner.load_inherited_a(
        prior, tmp_path, lambda: None, expected_sha=prior_sha
    )
    delayed = runner.build_signals(
        records,
        calendar[1:],
        delay_trading_days=5,
        availability_calendar=identity["availability_calendar"],
    )
    first = delayed.iloc[0]
    assert first.signal_date == "20190401"
    assert first.availability_cutoff == "20190325"
    assert first.input_known and first.eligible


def test_quotes_two_codes_sparse_retained_and_amount_units(tmp_path: Path) -> None:
    inherited_a(tmp_path)
    path = tmp_path / "data/clean/daily/2019-04-01.parquet"
    frame = pd.read_parquet(path).iloc[:1]
    frame.to_parquet(path, index=False)
    quotes, identity = runner.load_quotes(tmp_path, DAYS, lambda: None)
    assert len(quotes) == len(DAYS) * 2 - 1
    assert identity["missing_code_days"] == [{"trade_date": "20190401", "ts_code": runner.CODES[1]}]
    trades = pd.DataFrame(
        [{"trade_date": "20190402", "ts_code": runner.CODES[0], "amount": 10000.0}]
    )
    capacity = runner.capacity_readings(trades, quotes)
    assert capacity.fraction_of_daily_amount.iloc[0] == pytest.approx(0.002)


def setup_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, list]:
    base, source_sha = frozen_source(tmp_path, monkeypatch)
    prior, prior_sha = inherited_a(tmp_path)
    for name, value in (
        ("ROOT", tmp_path),
        ("BASE", base),
        ("PRIOR", prior),
        ("PRIOR_SHA", prior_sha),
    ):
        monkeypatch.setattr(runner, name, value)

    def snapshot(output: Path, root: Path) -> dict:
        (output / "methods").mkdir()
        return {"spec.md": digest(root / "spec.md")}

    monkeypatch.setattr(runner, "_snapshot", snapshot)
    # CI只测接线；生产入口不传bootstrap=False，继承分析默认2000次。
    monkeypatch.setattr(
        runner, "analyze_capital", lambda **kwargs: analyze_capital(**kwargs, bootstrap=False)
    )
    calls, original = [], runner.run_polyester_account

    def account(**kwargs: object) -> dict:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(runner, "run_polyester_account", account)
    return base, source_sha, calls


def test_account_failure_retains_failure_without_success_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, source_sha, _ = setup_run(tmp_path, monkeypatch)

    def fail(**kwargs: object) -> dict:
        raise ValueError("合成账户失败")

    monkeypatch.setattr(runner, "run_polyester_account", fail)
    with pytest.raises(ValueError, match="合成账户失败"):
        runner.run("run_failure", source_sha)
    failure = json.loads((base / "run_failure/failure.json").read_text(encoding="utf-8"))
    assert not failure["economic_success_claimed"]
    assert not (base / "run_failure/manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("codes", ["002493.SZ", "000703.SZ"]),
        ("codes", ["002714.SZ", "002714.SZ", "300498.SZ"]),
        ("capital", [1000000, 3000000]),
    ],
)
def test_wrong_stock_or_capital_spec_rejected_before_stock_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: list
) -> None:
    base, sha = frozen_source(tmp_path, monkeypatch)
    path = base / "prereg/spec.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    write(path, {**spec, field: value})
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "BASE", base)
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("坏规格不得读行情"))
    with pytest.raises(ValueError, match="股票或资本"):
        runner.run("run_bad_spec", sha)
    assert not (base / "run_bad_spec").exists()


def test_full_synthetic_run_six_actual_b_accounts_and_independent_sum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, source_sha, calls = setup_run(tmp_path, monkeypatch)
    summary = runner.run("run_test", source_sha)
    assert len(calls) == 12
    assert [call["capital"] for call in calls] == [200000] * 6 + [400000] * 6
    assert [call["dividend_haircut"] for call in calls[:6]] == [0, 0, 0, 0, 0.2, 0.2]
    assert all(call["codes"] == ("002714.SZ", "300498.SZ") for call in calls)
    assert calls[1]["signals"].eligible.all()
    assert calls[2]["signals"].eligible.all()
    assert calls[3]["signals"].eligible.iloc[0]
    assert calls[0]["signals"].eligible.iloc[0]
    assert not calls[0]["signals"].eligible.iloc[-1]
    output = base / "run_test"
    valid = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert valid["complete"] and valid["inherited_domains_unchanged"]
    assert summary["formal_admission"] == "not_evaluated"
    assert summary["future_success_probability"] is None
    assert summary["machine_budget_seconds"] == 19 * 60
    assert "final_identity_verification" in summary["timings_seconds"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for relative, sha in manifest["files"].items():
        assert digest(output / relative) == sha
    assert not (output / "1000000/B_source_strict").exists()
    for capital in runner.CAPITALS:
        folder = output / str(capital)
        assert (folder / "B_signal/capacity.parquet").exists()
        assumptions = json.loads((folder / "B_signal/assumptions.json").read_text("utf-8"))
        assert assumptions["slot_count"] == 2
        account = pd.read_parquet(folder / "B_signal/daily.parquet")
        equal = pd.read_parquet(folder / "B_equal/daily.parquet")
        trades = pd.read_parquet(folder / "B_signal/trades.parquet")
        combined = pd.read_parquet(folder / "A80_B_signal_daily.parquet")
        cash_control = pd.read_parquet(folder / "A80_B_cash_daily.parquet")
        assert set(trades.ts_code) == set(runner.CODES)
        assert trades.trade_date.min() == "20190402"
        assert account.cash.ge(0).all() and equal.cash.ge(0).all()
        assert account.market_value.iloc[-1] == 0
        assert equal.market_value.iloc[-1] > 0
        pd.testing.assert_series_equal(
            combined.total_value,
            account.total_value + capital * 4 // 5,
            check_names=False,
        )
        assert cash_control.total_value.eq(capital).all()
        assert cash_control.nav.eq(1).all()


@pytest.mark.parametrize("mutation", ["raw_domain", "spec", "source", "a_domain"])
def test_final_identity_rejects_changes_after_account_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    base, source_sha, _ = setup_run(tmp_path, monkeypatch)
    original, changed = runner.run_polyester_account, False

    def account(**kwargs: object) -> dict:
        nonlocal changed
        result = original(**kwargs)
        if not changed:
            if mutation == "raw_domain":
                raw = tmp_path / "data/raw/dividend"
                (raw / "2020-12-31.parquet").write_bytes((raw / "2019-12-31.parquet").read_bytes())
            elif mutation == "spec":
                path = base / "prereg/spec.json"
                value = json.loads(path.read_text("utf-8"))
                write(path, {**value, "unexpected_after_start": True})
            elif mutation == "source":
                write(base / "input_prices/selection.json", {"reason": "changed"})
            else:
                write(tmp_path / "prior/1000000/A100/extra.json", {})
            changed = True
        return result

    monkeypatch.setattr(runner, "run_polyester_account", account)
    with pytest.raises(ValueError, match="分区域|机器规格|SHA|文件域"):
        runner.run("run_changed", source_sha)
    output = base / "run_changed"
    assert (output / "failure.json").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / "validation.json").exists()


def test_budget_guard_stops_before_stock_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, source_sha = frozen_source(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "BASE", base)
    monkeypatch.setattr(runner, "MAX_MACHINE_SECONDS", -1)
    monkeypatch.setattr(
        pd, "read_parquet", lambda *args, **kwargs: pytest.fail("预算耗尽不得读行情")
    )
    with pytest.raises(TimeoutError, match="19 分钟"):
        runner.run("run_timeout", source_sha)
    assert not (base / "run_timeout").exists()


def test_snapshot_keeps_hog_and_shared_methods_without_changing_old_globals(tmp_path: Path) -> None:
    names = (
        "run_polyester.py",
        "polyester_account.py",
        "hog_policy.py",
        "run_hog.py",
        "build_hog_inputs.py",
    )
    for name in names:
        write(tmp_path / "scripts/v2_p7" / name, {"synthetic_method": name})
    write(tmp_path / "configs/base.yaml", {})
    output = tmp_path / "output"
    output.mkdir()
    old_codes = runner.inherited_runner.CODES
    identities = runner._snapshot(output, tmp_path)
    for name in names:
        assert identities[f"scripts/v2_p7/{name}"] == digest(output / "methods" / name)
    assert runner.inherited_runner.CODES == old_codes
    assert runner.CODES == ("002714.SZ", "300498.SZ")
