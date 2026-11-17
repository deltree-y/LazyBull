"""生猪饲料比单规则的独立资金账研究入口；不接生产。

先核验冻结报价、规格与来源，再读取股票输入。沿用已封存 A 及真实两槽账户。
复杂度豁免：阶段专用研究工具（CLAUDE §7.8）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from scripts.v2_p7 import run_polyester as inherited_runner  # noqa: E402
from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook  # noqa: E402
from scripts.v2_p7.buffer_native_replay import sha256, write_json  # noqa: E402
from scripts.v2_p7.hog_policy import (  # noqa: E402
    CODES,
    build_signals,
    signal_summary,
    validate_records,
)
from scripts.v2_p7.operating_monthly_analysis import (  # noqa: E402
    analyze_capital,
    cash_daily,
    normalize_monthly_daily,
)
from scripts.v2_p7.operating_monthly_corporate import prepare_research_dividends  # noqa: E402
from scripts.v2_p7.polyester_account import run_polyester_account  # noqa: E402
from scripts.v2_p7.run_polyester import (  # noqa: E402
    _read,
    _safe_file,
    _save_account,
    _verify_files,
    capacity_readings,
    load_inherited_a,
    main_differences,
    path_diagnostics,
)

BASE = ROOT / "data/reports/p7_hog_economic_20261116"
PRIOR = ROOT / "data/reports/p7_operating_monthly_economic_20261116/run_r3"
SPEC_SHA = "aa1251047b9c66af6d7c197c655189a52f93d84dae1bd2a39b59d5cce997905c"
PRIOR_SHA = "541933bbbf3ecc2bc96a5bb572bdc8fb16cedde6b369c302b3501b4c19d5fd81"
ANCHOR, START, END = "20190329", "20190401", "20261029"
CAPITALS = (1000000, 2000000)
MAX_MACHINE_SECONDS = 19 * 60
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
    ("signal_tax_stress", 0.2),
    ("equal_tax_stress", 0.2),
)


def load_frozen_inputs(
    base: Path, root: Path, expected_source_sha: str, guard: Callable[[], None]
) -> tuple[dict, list[dict], dict]:
    """行情读取前验证本轮独立封包；不得以运行时文件重新冻结规格。"""
    guard()
    spec = _read(base / "prereg/spec.json")
    if spec["sha256"] != SPEC_SHA or sha256(_safe_file(root, spec["spec_path"])) != SPEC_SHA:
        raise ValueError("生猪预登记 SHA 不符")
    if tuple(spec["codes"]) != CODES or tuple(spec["capital"]) != CAPITALS:
        raise ValueError("预登记固定股票或资本不符")
    path = base / "input_prices/source_manifest.json"
    if len(expected_source_sha) != 64 or sha256(path) != expected_source_sha:
        raise ValueError("生猪报价 manifest SHA 不符")
    manifest = _read(path)
    if manifest["schema"] != "p7_hog_price_inputs_v1":
        raise ValueError("生猪报价封包 schema 不符")
    if manifest["spec_sha256"] != SPEC_SHA or manifest["frozen_before_stock_returns"] is not True:
        raise ValueError("报价未在预登记及行情读取前冻结")
    _verify_files(root, manifest["files"], guard)
    records_path = _safe_file(root, manifest["records_file"])
    if records_path != (base / "input_prices/records.json").resolve():
        raise ValueError("records 不是本轮独立输入")
    if manifest["files"].get(manifest["records_file"]) != manifest["records_sha256"]:
        raise ValueError("records 未纳入一致的源文件清单")
    if sha256(records_path) != manifest["records_sha256"]:
        raise ValueError("生猪 records SHA 不符")
    if manifest["source_selection_file"] not in manifest["files"]:
        raise ValueError("来源选择裁决缺少冻结身份")
    records = validate_records(_read(records_path))
    return spec, records, manifest


def load_quotes(
    root: Path, calendar: list[str], guard: Callable[[], None]
) -> tuple[pd.DataFrame, dict]:
    """固定读取生猪两股执行必要列；复用 clean 标记，不补算交易状态。"""
    frames, files = [], {}
    for index, day in enumerate(calendar, 1):
        guard()
        name = f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        path = root / "data/clean/daily" / name
        if not path.is_file():
            raise ValueError(f"研究交易日缺少 clean 分区：{day}")
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
            print(f"生猪执行行情 {index}/{len(calendar)}", flush=True)
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


def _snapshot(output: Path, root: Path) -> dict:
    """沿用公共依赖快照，另封存本轮生猪方法；不改旧模块全局。"""
    files = inherited_runner._snapshot(output, root)
    methods = output / "methods"
    for pattern in ("hog*.py", "run_hog.py", "build_hog_inputs.py"):
        for path in sorted((root / "scripts/v2_p7").glob(pattern)):
            shutil.copyfile(path, methods / path.name)
            files[path.relative_to(root).as_posix()] = sha256(path)
    for relative, digest in files.items():
        if relative.startswith("scripts/v2_p7/") and "/t4/" not in relative:
            if sha256(methods / Path(relative).name) != digest:
                raise ValueError(f"方法快照与源码身份不符：{relative}")
    return files


def _check_end_identity(
    *,
    source_manifest_sha256: str,
    spec_file_sha256: str,
    source_identity: dict,
    price_identity: dict,
    inherited: dict,
    guard: Callable[[], None],
) -> None:
    """收尾重核来源、源码、A 的完整文件域与行情分红身份。"""
    load_frozen_inputs(BASE, ROOT, source_manifest_sha256, guard)
    if sha256(BASE / "prereg/spec.json") != spec_file_sha256:
        raise ValueError("运行期间机器规格文件变化")
    _verify_files(ROOT, source_identity, guard)
    _verify_files(ROOT, price_identity["files"], guard)
    # 使用同一继承检查器重查目录域，不能只哈希原清单而漏掉新增分红分区。
    _, _, current, _ = load_inherited_a(PRIOR, ROOT, guard, expected_sha=PRIOR_SHA)
    if current != inherited:
        raise ValueError("运行期间继承 A 身份变化")


def run(run_id: str, source_manifest_sha256: str) -> dict:
    """一次运行全部冻结场景，不依据结果选参；失败留证且不声明完成。"""
    if not run_id.startswith("run_") or not run_id.replace("_", "").isalnum():
        raise ValueError("run-id 必须为 run_ 前缀单目录名")
    started, timings = time.perf_counter(), {}

    def guard() -> None:
        if time.perf_counter() - started > MAX_MACHINE_SECONDS:
            raise TimeoutError("本批达到 19 分钟机器预算保护线，保留证据并停止")

    spec_file_sha = sha256(BASE / "prereg/spec.json")
    spec, records, manifest = load_frozen_inputs(BASE, ROOT, source_manifest_sha256, guard)
    if sha256(BASE / "prereg/spec.json") != spec_file_sha:
        raise ValueError("机器规格文件读取期间变化")
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
        signals.to_parquet(output / "signals.parquet", index=False)
        delayed.to_parquet(output / "signals_delay5.parquet", index=False)
        write_json(
            output / "signal_summary.json",
            {"main": signal_summary(signals), "delay5": signal_summary(delayed)},
        )
        print(f"生猪报价与继承 A 核验通过：{signal_summary(signals)}", flush=True)
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
            raise ValueError("生猪公司行为含未裁定记录，不能执行账户")
        old_corporate = _read(PRIOR / "corporate_input/manifest.json")
        old_outputs = {
            Path(row["output_path"]).name: row["output_sha256"] for row in old_corporate["files"]
        }
        new_outputs = {
            Path(row["output_path"]).name: row["output_sha256"]
            for row in corporate["manifest"]["files"]
        }
        if old_outputs != new_outputs:
            raise ValueError("本轮全域研究分红与继承 A 不同")
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
                selected = (delayed if name == "delay5" else signals)[
                    ["signal_date", "ts_code", "eligible"]
                ].copy()
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
                guard()
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
            guard()
            result["diagnostics"] = path_diagnostics(paths, b_accounts)
            result["main_differences_vs_controls"] = main_differences(result)
            for name, rows in paths.items():
                pd.DataFrame(rows).to_parquet(folder / f"{name}_daily.parquet", index=False)
            write_json(folder / "analysis.json", result)
            results[str(capital)] = result
            timings[f"analysis_{capital}"] = time.perf_counter() - mark
            print(f"资本 {capital} 分析完成，累计 {time.perf_counter()-started:.1f}s", flush=True)
        mark = time.perf_counter()
        _check_end_identity(
            source_manifest_sha256=source_manifest_sha256,
            spec_file_sha256=spec_file_sha,
            source_identity=source_identity,
            price_identity=price_identity,
            inherited=inherited,
            guard=guard,
        )
        timings["final_identity_verification"] = time.perf_counter() - mark
        guard()
        summary = {
            "results": results,
            "signal_summary": signal_summary(signals),
            "spec_sha256": SPEC_SHA,
            "machine_spec_sha256": spec_file_sha,
            "price_source_manifest_sha256": source_manifest_sha256,
            "inherited_a_manifest_sha256": PRIOR_SHA,
            "timings_seconds": timings,
            "total_machine_seconds": time.perf_counter() - started,
            "machine_budget_seconds": MAX_MACHINE_SECONDS,
            "data_purchase_yuan": 0,
            "formal_admission": "not_evaluated",
            "future_success_probability": None,
        }
        write_json(output / "summary.json", summary)
        files = {}
        for path in sorted(output.rglob("*")):
            guard()
            if path.is_file():
                files[path.relative_to(output).as_posix()] = sha256(path)
        write_json(output / "manifest.json", {"files": files, "spec_sha256": SPEC_SHA})
        write_json(
            output / "validation.json",
            {
                "complete": True,
                "source_unchanged": True,
                "inherited_a_verified": True,
                "inherited_domains_unchanged": True,
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
