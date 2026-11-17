"""涤纶独立入口的合成封包、真实两槽接线和失败留证；不读生产数据。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p7 import run_polyester as runner
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
            "available_date": day,
            "poy": price,
            "pta": 5000.0,
            "meg": 4000.0,
            "quality_usable": True,
            "strict_usable": index != 3,
            "series_id": "synthetic",
            "unit": "CNY/t",
        }
        for index, (day, price) in enumerate(
            zip(["20181220", "20190120", "20190220", "20190320"], [6000, 6100, 6200, 7000])
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
        "schema": "p7_polyester_price_inputs_v1",
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


def test_strict_source_only_tightens_quality_without_mutating_records() -> None:
    records = [
        {"source_id": "main_only", "quality_usable": True, "strict_usable": False, "poy": 7000},
        {"source_id": "both", "quality_usable": True, "strict_usable": True, "poy": 6000},
        {"source_id": "unusable", "quality_usable": False, "strict_usable": True, "poy": None},
    ]
    result = runner.strict_records(records)
    assert [row["quality_usable"] for row in result] == [False, True, False]
    assert records[0]["quality_usable"] is True
    assert [row["poy"] for row in result] == [7000, 6000, None]


@pytest.mark.parametrize("flag", [None, 0, 1, "false"])
def test_strict_source_requires_explicit_boolean(flag: object) -> None:
    with pytest.raises(ValueError, match="strict_usable"):
        runner.strict_records(
            [{"source_id": "ambiguous", "quality_usable": True, "strict_usable": flag}]
        )


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


def test_inherited_accounts_same_axis_actual_capital_and_hashes(tmp_path: Path) -> None:
    prior, sha = inherited_a(tmp_path)
    accounts, days, identity, resolutions = runner.load_inherited_a(
        prior, tmp_path, lambda: None, expected_sha=sha
    )
    assert days == DAYS and accounts[2000000]["A80"][0]["cash"] == 1600000
    assert identity["not_reexecuted"] and resolutions == []
    assert identity["availability_calendar"][0] == "20190320"
    write(prior / "1000000/A100/daily.json", [])
    with pytest.raises(ValueError, match="SHA"):
        runner.load_inherited_a(prior, tmp_path, lambda: None, expected_sha=sha)


def test_inherited_account_extra_file_rejected(tmp_path: Path) -> None:
    prior, sha = inherited_a(tmp_path)
    write(prior / "1000000/A100/extra.json", {})
    with pytest.raises(ValueError, match="文件域"):
        runner.load_inherited_a(prior, tmp_path, lambda: None, expected_sha=sha)


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


def test_inherited_misaligned_calendar_rejected_even_with_resealed_hash(tmp_path: Path) -> None:
    prior, _ = inherited_a(tmp_path)
    path = prior / "1000000/A80/daily.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    write(path, rows[:-1])
    seal_prior(prior)
    with pytest.raises(ValueError, match="日轴"):
        runner.load_inherited_a(
            prior, tmp_path, lambda: None, expected_sha=digest(prior / "manifest.json")
        )


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


def test_capacity_zero_denominator_is_unknown_and_empty_trades_work() -> None:
    quotes = synthetic_quotes()
    quotes["amount"] = 0.0
    trades = pd.DataFrame([{"trade_date": DAYS[1], "ts_code": runner.CODES[0], "amount": 100.0}])
    result = runner.capacity_readings(trades, quotes)
    assert not result.capacity_denominator_known.iloc[0]
    assert pd.isna(result.fraction_of_daily_amount.iloc[0])
    assert runner.capacity_readings(trades.iloc[:0], quotes).empty


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


def test_full_synthetic_run_seven_actual_b_accounts_per_capital(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, source_sha, calls = setup_run(tmp_path, monkeypatch)
    summary = runner.run("run_test", source_sha)
    assert len(calls) == 14
    assert [call["capital"] for call in calls] == [200000] * 7 + [400000] * 7
    assert [call["dividend_haircut"] for call in calls[:7]] == [0, 0, 0, 0, 0, 0.2, 0.2]
    assert calls[1]["signals"].eligible.all()
    # 完整可用时间日历含窗前交易日，首月延迟不被误判成缺失。
    assert calls[3]["signals"].eligible.iloc[0]
    assert not calls[4]["signals"].eligible.any()
    assert calls[0]["signals"].eligible.iloc[0]
    output = base / "run_test"
    result = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert result["complete"] and summary["formal_admission"] == "not_evaluated"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for relative, sha in manifest["files"].items():
        assert digest(output / relative) == sha
    assert (output / "1000000/B_signal/capacity.parquet").exists()
    assert (output / "1000000/B_source_strict/capacity.parquet").exists()
    strict_signals = pd.read_parquet(output / "strict_signals.parquet")
    assert not strict_signals.input_known.iloc[0]
    assert summary["source_strict_signal_summary"]["known_months"] == 0
    assumptions = json.loads(
        (output / "1000000/B_signal/assumptions.json").read_text(encoding="utf-8")
    )
    assert assumptions["slot_count"] == 2


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
