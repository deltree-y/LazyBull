"""涤纶加工价差单规则经济预筛；独立证据入口，不接生产。

必须显式提供最终报价 manifest SHA；先核冻结输入，再读取行情。
复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook  # noqa: E402
from scripts.v2_p7.buffer_native_replay import sha256, write_json  # noqa: E402
from scripts.v2_p7.buffer_path_metrics import arm_metrics  # noqa: E402
from scripts.v2_p7.operating_monthly_analysis import (  # noqa: E402
    analyze_capital,
    cash_daily,
    normalize_monthly_daily,
    validate_path,
)
from scripts.v2_p7.operating_monthly_corporate import prepare_research_dividends  # noqa: E402
from scripts.v2_p7.polyester_account import run_polyester_account  # noqa: E402
from scripts.v2_p7.polyester_policy import (  # noqa: E402
    CODES,
    build_signals,
    signal_summary,
    validate_records,
)

BASE = ROOT / "data/reports/p7_polyester_economic_20261116"
PRIOR = ROOT / "data/reports/p7_operating_monthly_economic_20261116/run_r3"
SPEC_SHA = "0fc4445c8d707a3fd1564b6a889a5af918b2a7718377c2df64d46d2f2beecff4"
PRIOR_SHA = "541933bbbf3ecc2bc96a5bb572bdc8fb16cedde6b369c302b3501b4c19d5fd81"
ANCHOR, START, END = "20190329", "20190401", "20261029"
CAPITALS = (1000000, 2000000)
QUOTE_COLUMNS = [
    "trade_date",
    "ts_code",
    "open",
    "close",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "is_st",
    "tradable",
    "amount",
]
VARIANTS = (
    ("signal", 0.0),
    ("equal", 0.0),
    ("unknown_on", 0.0),
    ("delay5", 0.0),
    ("source_strict", 0.0),
    ("signal_tax_stress", 0.2),
    ("equal_tax_stress", 0.2),
)


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def strict_records(records: list[dict]) -> list[dict]:
    """严格来源场景仅收紧可用标记，不改原记录、数值或日期。"""
    result = []
    for row in records:
        if not isinstance(row.get("strict_usable"), bool):
            raise ValueError(f"报价 strict_usable 必须显式为布尔：{row['source_id']}")
        result.append({**row, "quality_usable": row["quality_usable"] and row["strict_usable"]})
    return result


def _safe_file(root: Path, relative: str) -> Path:
    """封包只能引用根内文件，拒绝绝对路径和越界引用。"""
    candidate = Path(relative)
    path = (root / candidate).resolve()
    if candidate.is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"封包路径非法或不存在：{relative}")
    return path


def _verify_files(root: Path, files: dict, guard: Callable[[], None]) -> None:
    if not isinstance(files, dict) or not files:
        raise ValueError("文件身份清单不能为空")
    for index, (relative, digest) in enumerate(files.items(), 1):
        guard()
        if sha256(_safe_file(root, relative)) != digest:
            raise ValueError(f"文件 SHA 不符：{relative}")
        if len(files) > 500 and index % 500 == 0:
            print(f"文件身份核验 {index}/{len(files)}", flush=True)


def load_frozen_inputs(
    base: Path, root: Path, expected_source_sha: str, guard: Callable[[], None]
) -> tuple[dict, list[dict], dict]:
    """行情读取前的硬门；缺失报价允许，但冻结身份和原件不允许漂移。"""
    spec = _read(base / "prereg/spec.json")
    if spec["sha256"] != SPEC_SHA or sha256(_safe_file(root, spec["spec_path"])) != SPEC_SHA:
        raise ValueError("涤纶预登记 SHA 不符")
    if set(spec["codes"]) != set(CODES) or tuple(spec["capital"]) != CAPITALS:
        raise ValueError("预登记固定股票或资本不符")
    path = base / "input_prices/source_manifest.json"
    if len(expected_source_sha) != 64 or sha256(path) != expected_source_sha:
        raise ValueError("最终报价 manifest SHA 不符")
    manifest = _read(path)
    if manifest["schema"] != "p7_polyester_price_inputs_v1":
        raise ValueError("报价封包 schema 不符")
    if manifest["spec_sha256"] != SPEC_SHA or manifest["frozen_before_stock_returns"] is not True:
        raise ValueError("报价未按预登记在收益读取前冻结")
    _verify_files(root, manifest["files"], guard)
    records_path = _safe_file(root, manifest["records_file"])
    if records_path != (base / "input_prices/records.json").resolve():
        raise ValueError("records 不是本轮独立输入")
    if manifest["files"].get(manifest["records_file"]) != manifest["records_sha256"]:
        raise ValueError("records 未进入一致的源文件清单")
    if sha256(records_path) != manifest["records_sha256"]:
        raise ValueError("报价 records SHA 不符")
    selection = manifest["source_selection_file"]
    if selection not in manifest["files"]:
        raise ValueError("来源选择裁决缺少冻结身份")
    records = validate_records(_read(records_path))
    strict_records(records)  # 行情读取前拒绝未登记或含歧义的严格来源标记。
    return spec, records, manifest


def load_inherited_a(
    prior: Path, root: Path, guard: Callable[[], None], *, expected_sha: str = PRIOR_SHA
) -> tuple[dict, list[str], dict, list[dict]]:
    """逐文件核四个原始 A 账户及输入身份，复用历史结果，不重算 A。"""
    if sha256(prior / "manifest.json") != expected_sha:
        raise ValueError("继承 A 的 manifest SHA 不符")
    manifest, validation = _read(prior / "manifest.json"), _read(prior / "validation.json")
    if validation.get("manifest_sha256") != expected_sha or validation.get("complete") is not True:
        raise ValueError("继承 A 没有完整封存验证")
    files = manifest["files"]
    support = (
        "source_identity.json",
        "price_identity.json",
        "dividend_identity.json",
        "baseline_inputs.json",
        "corporate_input/manifest.json",
        "dividend_resolutions.json",
    )
    selected = {name: files[name] for name in support}
    for capital in CAPITALS:
        for name in ("A100", "A80"):
            prefix = f"{capital}/{name}/"
            declared = {key: digest for key, digest in files.items() if key.startswith(prefix)}
            actual = {
                p.relative_to(prior).as_posix() for p in (prior / prefix).rglob("*") if p.is_file()
            }
            if not declared or set(declared) != actual:
                raise ValueError(f"A 目录与 manifest 文件域不符：{prefix}")
            selected.update(declared)
    selected.update({key: digest for key, digest in files.items() if key.startswith("methods/")})
    selected.update(
        {
            key: digest
            for key, digest in files.items()
            if key.startswith("corporate_input/dividend/")
        }
    )
    _verify_files(prior, selected, guard)
    old_source = _read(prior / "source_identity.json")
    for relative, digest in old_source.items():
        if relative.startswith("scripts/v2_p7/") and "/t4/" not in relative:
            snapshot = f"methods/{Path(relative).name}"
            if files.get(snapshot) != digest:
                raise ValueError(f"继承 A 方法快照身份不符：{relative}")
    prices, dividends = _read(prior / "price_identity.json"), _read(
        prior / "dividend_identity.json"
    )
    _verify_files(root, prices["files"], guard)
    _verify_files(root, dividends, guard)
    current_dividends = {
        p.relative_to(root).as_posix() for p in (root / "data/raw/dividend").glob("*.parquet")
    }
    if current_dividends != set(dividends):
        raise ValueError("原始红利分区域较冻结 A 变化")
    baseline = _read(prior / "baseline_inputs.json")
    if (baseline["anchor"], baseline["start"], baseline["end"]) != (ANCHOR, START, END):
        raise ValueError("继承 A 窗口不符")
    cal_identity = baseline["calendar_file"]
    _verify_files(root, {cal_identity["path"]: cal_identity["sha256"]}, guard)
    table = pd.read_parquet(
        root / cal_identity["path"], columns=["exchange", "is_open", "cal_date"]
    )
    days = table.loc[table.exchange.eq("SSE") & table.is_open.astype(int).eq(1), "cal_date"].astype(
        str
    )
    availability_calendar = sorted(set(days.loc[days.ge("20180101") & days.le(END)]))
    calendar = sorted(set(days.loc[days.ge(ANCHOR) & days.le(END)]))
    if calendar[:2] != [ANCHOR, START] or calendar[-1] != END:
        raise ValueError("原始交易日历不含冻结锚点及完整窗口")
    accounts = {}
    for capital in CAPITALS:
        accounts[capital] = {}
        for name, initial in (("A100", capital), ("A80", capital * 4 // 5)):
            folder = prior / str(capital) / name
            identity, valid = _read(folder / "baseline_identity.json"), _read(
                folder / "validation.json"
            )
            if identity["initial_capital"] != initial or valid.get("all_passed") is not True:
                raise ValueError(f"A 资本或验证不符：{capital}/{name}")
            if (
                identity["capital_scaled_from_other_nav"]
                or identity["cross_sleeve_borrowing"]
                or identity["cross_sleeve_netting"]
            ):
                raise ValueError("继承 A 不是独立真实资本账户")
            rows = _read(folder / "daily.json")
            frame = validate_path(rows)
            if frame.trade_date.tolist() != calendar or frame.total_value.iloc[0] != initial:
                raise ValueError(f"继承 A 日轴或初始现金不符：{capital}/{name}")
            accounts[capital][name] = rows
    identity = {
        "reference": str(prior.resolve()),
        "manifest_sha256": expected_sha,
        "checked_files": selected,
        "source_identity": old_source,
        "price_identity_sha256": files["price_identity.json"],
        "dividend_identity_sha256": files["dividend_identity.json"],
        "current_price_files": prices["files"],
        "current_dividend_files": dividends,
        "calendar_file": cal_identity,
        "availability_calendar": availability_calendar,
        "not_reexecuted": True,
        "not_virgin_holdout": True,
    }
    return accounts, calendar, identity, _read(prior / "dividend_resolutions.json")


def load_quotes(
    root: Path, calendar: list[str], guard: Callable[[], None]
) -> tuple[pd.DataFrame, dict]:
    """只读两股与执行必要列；稀疏行情保留缺失，不删除账户日期。"""
    frames, files = [], {}
    for index, day in enumerate(calendar, 1):
        guard()
        name = f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        path = root / "data/clean/daily" / name
        if not path.is_file():
            raise ValueError(f"研究交易日缺整个 clean 分区：{day}")
        digest = sha256(path)
        frame = pd.read_parquet(
            path, columns=QUOTE_COLUMNS, filters=[("ts_code", "in", list(CODES))]
        )
        if sha256(path) != digest:
            raise ValueError(f"行情读取期间分区变化：{day}")
        frame["trade_date"] = frame.trade_date.astype(str)
        if not frame.trade_date.eq(day).all() or not frame.ts_code.isin(CODES).all():
            raise ValueError(f"行情分区日期或股票域不符：{day}")
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"行情主键重复：{day}")
        files[path.relative_to(root).as_posix()] = digest
        frames.append(frame)
        if index % 250 == 0:
            print(f"两股执行行情 {index}/{len(calendar)}", flush=True)
    quotes = pd.concat(frames, ignore_index=True)
    observed = set(zip(quotes.trade_date, quotes.ts_code))
    missing = [
        {"trade_date": day, "ts_code": code}
        for day in calendar
        for code in CODES
        if (day, code) not in observed
    ]
    return quotes, {
        "files": files,
        "columns": QUOTE_COLUMNS,
        "codes": list(CODES),
        "missing_code_days": missing,
        "amount_unit": "千元",
        "flags_recomputed": False,
    }


def capacity_readings(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """逐笔相对当日成交额；TuShare clean amount 沿数据契约为千元。"""
    result = trades.rename(columns={"amount": "trade_amount_yuan"}).merge(
        quotes[["trade_date", "ts_code", "amount"]].rename(
            columns={"amount": "market_amount_thousand_yuan"}
        ),
        on=["trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    market = pd.to_numeric(result.market_amount_thousand_yuan, errors="coerce") * 1000.0
    result["market_amount_yuan"] = market
    result["fraction_of_daily_amount"] = result.trade_amount_yuan / market.where(market.gt(0))
    result["capacity_denominator_known"] = market.gt(0) & market.map(math.isfinite)
    return result


def _save_account(output: Path, account: dict) -> None:
    output.mkdir()
    for name, value in account.items():
        if isinstance(value, pd.DataFrame):
            value.to_parquet(output / f"{name}.parquet", index=False)
        else:
            write_json(output / f"{name}.json", value)


def _snapshot(output: Path, root: Path) -> dict:
    methods = output / "methods"
    methods.mkdir()
    files = {}
    for pattern in (
        "polyester*.py",
        "run_polyester.py",
        "operating_monthly_*.py",
        "buffer_native_*.py",
        "buffer_path_metrics.py",
    ):
        for path in sorted((root / "scripts/v2_p7").glob(pattern)):
            shutil.copyfile(path, methods / path.name)
            files[path.relative_to(root).as_posix()] = sha256(path)
    for folder in (
        "src/lazybull/v2",
        "src/lazybull/common",
        "src/lazybull/trading",
        "scripts/v2_p7/t4",
    ):
        for path in sorted((root / folder).rglob("*.py")):
            files[path.relative_to(root).as_posix()] = sha256(path)
    files["configs/base.yaml"] = sha256(root / "configs/base.yaml")
    return files


def path_diagnostics(paths: dict, b_accounts: dict) -> dict:
    """补逐年、回撤窗口和同日相关；年化复用既有指标实现。"""
    yearly, windows, historical = {}, {}, {}
    for name, rows in paths.items():
        frame = pd.DataFrame(rows)
        historical_part = frame.loc[frame.trade_date.le("20251231")]
        if len(historical_part) > 1:
            historical[name] = arm_metrics(historical_part)
        yearly[name] = {}
        for year in sorted(set(frame.trade_date.str[:4])):
            indexes = frame.index[frame.trade_date.str.startswith(year)].tolist()
            start, end = max(0, indexes[0] - 1), indexes[-1] + 1
            part = frame.iloc[start:end].copy()
            if len(part) < 2:
                continue
            for key in ("fees", "buy_amount", "sell_amount"):
                part.loc[part.index[0], key] = 0.0
            yearly[name][year] = arm_metrics(part)
        drawdown = frame.nav / frame.nav.cummax() - 1
        trough = int(drawdown.idxmin())
        peak = int(frame.nav.iloc[: trough + 1].idxmax())
        windows[name] = {
            "peak": frame.trade_date.iloc[peak],
            "trough": frame.trade_date.iloc[trough],
            "drawdown": float(drawdown.iloc[trough]),
        }
    a_returns = pd.DataFrame(paths["A100"]).nav.pct_change()
    correlations = {}
    for name, rows in b_accounts.items():
        returns = pd.DataFrame(rows).nav.pct_change()
        value = a_returns.corr(returns) if returns.std() > 0 and a_returns.std() > 0 else None
        correlations[name] = float(value) if value is not None and math.isfinite(value) else None
    return {
        "yearly": yearly,
        "max_drawdown_windows": windows,
        "b_return_correlation_with_a100": correlations,
        "historical_through_2025": historical,
        "inherited_early_window_ends": "20260202",
    }


def main_differences(result: dict) -> dict:
    """主场景相对现金/等权的同窗差；沿负值 MaxDD，正差意味着回撤较浅。"""
    metrics = result["metrics"]
    return {
        other: {
            period: {
                key: metrics["A80_B_signal"][period][key] - metrics[other][period][key]
                for key in ("cagr", "max_drawdown", "total_return", "fees_yuan")
            }
            for period in ("full", "early", "late")
        }
        for other in ("A100", "A80_B_cash", "A80_B_equal")
    }


def run(run_id: str, source_manifest_sha256: str) -> dict:
    """单次固定全部资本与场景；独立落盘，失败留证，不自动续跑。"""
    if not run_id.startswith("run_") or not run_id.replace("_", "").isalnum():
        raise ValueError("run-id 必须为 run_ 前缀简单目录名")
    started, timings = time.perf_counter(), {}

    def guard() -> None:
        if time.perf_counter() - started > 60 * 60:
            raise TimeoutError("本轮达到 60 分钟机器保护线，保留证据并停止")

    spec, records, manifest = load_frozen_inputs(BASE, ROOT, source_manifest_sha256, guard)
    output = BASE / run_id
    output.mkdir(exist_ok=False)
    try:
        write_json(output / "prereg.json", spec)
        write_json(output / "source_manifest.json", manifest)
        source_identity = _snapshot(output, ROOT)
        write_json(output / "source_identity.json", source_identity)
        mark = time.perf_counter()
        accounts, calendar, inherited, resolutions = load_inherited_a(
            PRIOR, ROOT, guard, expected_sha=PRIOR_SHA
        )
        write_json(output / "inherited_a_identity.json", inherited)
        write_json(output / "dividend_resolutions.json", resolutions)
        signals = build_signals(
            records, calendar[1:], availability_calendar=inherited["availability_calendar"]
        )
        delayed = build_signals(
            records,
            calendar[1:],
            delay_trading_days=5,
            availability_calendar=inherited["availability_calendar"],
        )
        strict = build_signals(
            strict_records(records),
            calendar[1:],
            availability_calendar=inherited["availability_calendar"],
        )
        signals.to_parquet(output / "signals.parquet", index=False)
        delayed.to_parquet(output / "signals_delay5.parquet", index=False)
        strict.to_parquet(output / "strict_signals.parquet", index=False)
        write_json(
            output / "signal_summary.json",
            {
                "main": signal_summary(signals),
                "delay5": signal_summary(delayed),
                "source_strict": signal_summary(strict),
            },
        )
        print(f"冻结输入和继承 A 核验通过；{signal_summary(signals)}", flush=True)
        quotes, price_identity = load_quotes(ROOT, calendar, guard)
        write_json(output / "price_identity.json", price_identity)
        quotes.to_parquet(output / "b_quotes.parquet", index=False)
        corporate = prepare_research_dividends(
            raw_dir=ROOT / "data/raw/dividend",
            output_dir=output / "corporate_input",
            calendar=calendar,
            codes=CODES,
            resolutions=resolutions,
        )
        if corporate["manifest"]["audit_after"]["unresolved_problem_count"]:
            raise ValueError("两股公司行为仍有未解决事件，不能执行账户")
        old_corporate = _read(PRIOR / "corporate_input/manifest.json")
        old_outputs = {
            Path(row["output_path"]).name: row["output_sha256"] for row in old_corporate["files"]
        }
        new_outputs = {
            Path(row["output_path"]).name: row["output_sha256"]
            for row in corporate["manifest"]["files"]
        }
        if old_outputs != new_outputs:
            raise ValueError("本轮红利研究副本与继承 A 不同")
        book = NativeCorporateBook(calendar=calendar, dividend_dir=Path(corporate["dividend_dir"]))
        timings["input_preparation"] = time.perf_counter() - mark
        results = {}
        for capital in CAPITALS:
            folder = output / str(capital)
            folder.mkdir()
            b_accounts = {"cash": cash_daily(calendar, capital // 5)}
            for name, haircut in VARIANTS:
                guard()
                mark = time.perf_counter()
                selected = (
                    {"delay5": delayed, "source_strict": strict}
                    .get(name, signals)[["signal_date", "ts_code", "eligible"]]
                    .copy()
                )
                if name.startswith("equal"):
                    selected["eligible"] = True
                elif name == "unknown_on":
                    selected["eligible"] = signals.unknown_on
                account = run_polyester_account(
                    calendar=calendar,
                    quotes=quotes,
                    signals=selected,
                    capital=capital // 5,
                    corporate_book=book,
                    arm_name=name,
                    codes=CODES,
                    dividend_haircut=haircut,
                )
                _save_account(folder / f"B_{name}", account)
                capacity_readings(account["trades"], quotes).to_parquet(
                    folder / f"B_{name}/capacity.parquet", index=False
                )
                b_accounts[name] = normalize_monthly_daily(account["daily"])
                timings[f"B_{name}_{capital}"] = time.perf_counter() - mark
                print(
                    f"{capital}/B_{name} 完成，累计 {time.perf_counter()-started:.1f}s", flush=True
                )
            guard()
            mark = time.perf_counter()
            result, paths = analyze_capital(
                capital=capital,
                a_full=accounts[capital]["A100"],
                a_partial=accounts[capital]["A80"],
                b_accounts=b_accounts,
            )
            result["diagnostics"] = path_diagnostics(paths, b_accounts)
            result["main_differences_vs_controls"] = main_differences(result)
            for name, rows in paths.items():
                pd.DataFrame(rows).to_parquet(folder / f"{name}_daily.parquet", index=False)
            write_json(folder / "analysis.json", result)
            results[str(capital)] = result
            timings[f"analysis_{capital}"] = time.perf_counter() - mark
            print(f"资本 {capital} 分析完成，累计 {time.perf_counter()-started:.1f}s", flush=True)
        guard()
        load_frozen_inputs(BASE, ROOT, source_manifest_sha256, guard)
        _verify_files(ROOT, source_identity, guard)
        _verify_files(ROOT, price_identity["files"], guard)
        _verify_files(PRIOR, inherited["checked_files"], guard)
        if sha256(PRIOR / "manifest.json") != PRIOR_SHA:
            raise ValueError("运行期间继承 A manifest 变化")
        _verify_files(ROOT, inherited["current_price_files"], guard)
        _verify_files(ROOT, inherited["current_dividend_files"], guard)
        calendar_identity = inherited["calendar_file"]
        _verify_files(ROOT, {calendar_identity["path"]: calendar_identity["sha256"]}, guard)
        summary = {
            "results": results,
            "signal_summary": signal_summary(signals),
            "source_strict_signal_summary": signal_summary(strict),
            "spec_sha256": SPEC_SHA,
            "price_source_manifest_sha256": source_manifest_sha256,
            "inherited_a_manifest_sha256": PRIOR_SHA,
            "timings_seconds": timings,
            "total_machine_seconds": time.perf_counter() - started,
            "data_purchase_yuan": 0,
            "formal_admission": "not_evaluated",
            "future_success_probability": None,
        }
        write_json(output / "summary.json", summary)
        files = {
            path.relative_to(output).as_posix(): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        write_json(output / "manifest.json", {"files": files, "spec_sha256": SPEC_SHA})
        write_json(
            output / "validation.json",
            {
                "complete": True,
                "source_unchanged": True,
                "inherited_a_verified": True,
                "formal_admission": "not_evaluated",
                "manifest_sha256": sha256(output / "manifest.json"),
            },
        )
        return summary
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "timings_seconds": timings,
                "economic_success_claimed": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()
    logger.remove()
    run(args.run_id, args.source_manifest_sha256)


if __name__ == "__main__":
    main()
