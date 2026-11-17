"""快递经营月报单规则经济预筛；仅写独立研究证据目录。

运行：python scripts/v2_p7/run_operating_monthly.py --run-id run_r1
复杂度豁免：阶段专用研究工具，不入生产链路。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook  # noqa: E402
from scripts.v2_p7.buffer_native_replay import sha256, write_json  # noqa: E402
from scripts.v2_p7.operating_monthly_account import run_monthly_account  # noqa: E402
from scripts.v2_p7.operating_monthly_analysis import (  # noqa: E402
    analyze_capital,
    cash_daily,
    normalize_monthly_daily,
)
from scripts.v2_p7.operating_monthly_baseline import (  # noqa: E402
    load_baseline_inputs,
    load_baseline_prices,
    run_baseline_account,
)
from scripts.v2_p7.operating_monthly_corporate import prepare_research_dividends  # noqa: E402
from scripts.v2_p7.operating_monthly_policy import (  # noqa: E402
    CODES,
    build_signals,
    signal_summary,
)
from src.lazybull.v2.core.execution.price_index import prepare_price_index  # noqa: E402

BASE = ROOT / "data/reports/p7_operating_monthly_economic_20261116"


def validate_prereg(base: Path) -> dict:
    """读数前核对冻结正文，拒绝静默改写规则。"""
    spec = json.loads((base / "prereg/prereg.json").read_text(encoding="utf-8"))
    actual = sha256(ROOT / spec["spec_path"])
    if actual != spec["spec_sha256"]:
        raise ValueError("预登记正文SHA不符，禁止继续")
    return spec


def load_frozen_announcements(base: Path) -> tuple[list[dict], str]:
    """只消费通过完整性自验且记录哈希一致的源包。"""
    source = base / "input_announcements"
    manifest = json.loads((source / "source_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((source / "validation.json").read_text(encoding="utf-8"))
    records_path = source / "records.json"
    digest = sha256(records_path)
    if not manifest["all_catalogs_complete"] or not validation["all_passed"]:
        raise ValueError("公告源包完整性/数值自验不通过")
    if manifest["records_sha256"] != digest:
        raise ValueError("公告源包records SHA不符")
    return json.loads(records_path.read_text(encoding="utf-8")), digest


def _save_b(output: Path, result: dict) -> None:
    output.mkdir(exist_ok=False)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_parquet(output / f"{key}.parquet", index=False)
        else:
            write_json(output / f"{key}.json", value)


def _snapshot(output: Path) -> dict:
    """记录本轮入口及直接研究依赖；生产实现以git与文件哈希留痕。"""
    methods = output / "methods"
    methods.mkdir()
    identities = {}
    for pattern in (
        "operating_monthly_*.py",
        "run_operating_monthly.py",
        "buffer_native_*.py",
        "buffer_path_metrics.py",
        "run_buffer_native.py",
    ):
        for source in sorted((ROOT / "scripts/v2_p7").glob(pattern)):
            shutil.copyfile(source, methods / source.name)
            identities[source.relative_to(ROOT).as_posix()] = sha256(source)
    for folder in (
        "src/lazybull/v2/core",
        "src/lazybull/v2/hosts/native_engine",
        "src/lazybull/v2/common",
        "src/lazybull/common",
        "src/lazybull/trading",
    ):
        for source in sorted((ROOT / folder).rglob("*.py")):
            identities[source.relative_to(ROOT).as_posix()] = sha256(source)
    for source in (ROOT / "scripts/v2_p7/t4").rglob("*.py"):
        identities[source.relative_to(ROOT).as_posix()] = sha256(source)
    for relative in ("src/lazybull/v2/common/chain_metrics.py", "configs/base.yaml"):
        identities[relative] = sha256(ROOT / relative)
    return identities


def _compare_completed_pre_resolution(output: Path) -> dict:
    """输入消歧不影响未持有该事件的既有完整账户，逐位比对而不看收益择优。"""
    prior = BASE / "run_r2/1000000"
    for name in ("A100", "A80"):
        old = json.loads((prior / name / "daily.json").read_text(encoding="utf-8"))
        new = json.loads((output / "1000000" / name / "daily.json").read_text(encoding="utf-8"))
        pd.testing.assert_frame_equal(pd.DataFrame(old), pd.DataFrame(new), check_exact=True)
    old_b = pd.read_parquet(prior / "B_signal/daily.parquet")
    new_b = pd.read_parquet(output / "1000000/B_signal/daily.parquet")
    pd.testing.assert_frame_equal(old_b, new_b, check_exact=True)
    return {"reference": "run_r2", "exact": True, "accounts": ["A100", "A80", "B_signal"]}


def run(run_id: str) -> dict:
    """一次运行全部固定账户，超预算中止并留检查点，不拆分绕过审批。"""
    if not run_id.replace("_", "").isalnum() or not run_id.startswith("run_"):
        raise ValueError("run-id必须为run_前缀的简单目录名")
    spec = validate_prereg(BASE)
    records_path = BASE / "input_announcements/records.json"
    records, records_sha = load_frozen_announcements(BASE)
    amendment = json.loads((BASE / "prereg/execution_amendment.json").read_text(encoding="utf-8"))
    if sha256(ROOT / amendment["path"]) != amendment["sha256"]:
        raise ValueError("实施补充SHA不符")
    dividend_amendment = json.loads(
        (BASE / "prereg/dividend_resolution.json").read_text(encoding="utf-8")
    )
    if sha256(ROOT / dividend_amendment["path"]) != dividend_amendment["sha256"]:
        raise ValueError("分红输入修订SHA不符")
    resolutions_path = BASE / "corporate_resolution/resolutions_v2.json"
    resolutions_sha = sha256(resolutions_path)
    resolutions = json.loads(resolutions_path.read_text(encoding="utf-8"))
    output = BASE / run_id
    output.mkdir(exist_ok=False)
    start, timings = time.perf_counter(), {}

    def guard() -> None:
        if time.perf_counter() - start > 19 * 60:
            raise TimeoutError("完整账户批达到19分钟保护线；需先报告重估，禁止自动续跑")

    write_json(output / "prereg.json", spec)
    write_json(output / "execution_amendment.json", amendment)
    write_json(output / "dividend_resolution.json", dividend_amendment)
    write_json(output / "dividend_resolutions.json", resolutions)
    source_identity = _snapshot(output)
    write_json(output / "source_identity.json", source_identity)
    try:
        mark = time.perf_counter()
        gap = BASE / "baseline_inputs" / run_id.replace("run_", "gap_", 1)
        inputs = load_baseline_inputs(
            ROOT,
            gap_output=gap,
            spec_sha256=spec["spec_sha256"],
            budget_guard=guard,
        )
        original_gap = BASE / "baseline_inputs/gap_r1/ranked.parquet"
        if gap.name != "gap_r1":
            old_ranked = pd.read_parquet(original_gap)
            new_ranked = pd.read_parquet(gap / "ranked.parquet")
            pd.testing.assert_frame_equal(old_ranked, new_ranked, check_exact=True)
            write_json(
                output / "gap_equality.json",
                {
                    "exact": True,
                    "rows": len(new_ranked),
                    "old_sha256": sha256(original_gap),
                    "new_sha256": sha256(gap / "ranked.parquet"),
                },
            )
        write_json(output / "baseline_inputs.json", inputs.identity)
        signals = build_signals(records, inputs.calendar)
        signals.to_parquet(output / "signals.parquet", index=False)
        write_json(output / "signal_summary.json", signal_summary(signals))
        prices, price_identity = load_baseline_prices(ROOT, inputs)
        write_json(output / "price_identity.json", price_identity)
        calendar = [inputs.anchor, *inputs.calendar]
        quotes = prices.loc[
            prices.ts_code.astype(str).isin(CODES) & prices.trade_date.astype(str).isin(calendar)
        ].copy()
        quotes.to_parquet(output / "b_quotes.parquet", index=False)
        indexes = prepare_price_index(prices)
        corporate_input = prepare_research_dividends(
            raw_dir=ROOT / "data/raw/dividend",
            output_dir=output / "corporate_input",
            calendar=calendar,
            codes=CODES,
            resolutions=resolutions,
        )
        book = NativeCorporateBook(
            calendar=calendar, dividend_dir=Path(corporate_input["dividend_dir"])
        )
        write_json(
            output / "dividend_identity.json",
            {
                p.relative_to(ROOT).as_posix(): sha256(p)
                for p in sorted((ROOT / "data/raw/dividend").glob("*.parquet"))
            },
        )
        timings["input_preparation"] = time.perf_counter() - mark
        print(f"输入完成 {timings['input_preparation']:.1f}s；日历 {len(calendar)} 点", flush=True)
        results = {}
        for capital in spec["capital"]:
            guard()
            capital_dir = output / str(capital)
            capital_dir.mkdir()
            mark = time.perf_counter()
            a_full = run_baseline_account(
                ROOT,
                inputs=inputs,
                prices=prices,
                indexes=indexes,
                capital=capital,
                corporate_book=book,
                output=capital_dir / "A100",
                budget_guard=guard,
                progress=lambda n, total: print(f"A100/{capital} {n}/{total}", flush=True),
            )
            a_partial = run_baseline_account(
                ROOT,
                inputs=inputs,
                prices=prices,
                indexes=indexes,
                capital=int(capital * 0.8),
                corporate_book=book,
                output=capital_dir / "A80",
                budget_guard=guard,
                progress=lambda n, total: print(f"A80/{capital} {n}/{total}", flush=True),
            )
            timings[f"a_accounts_{capital}"] = time.perf_counter() - mark
            b_accounts = {"cash": cash_daily(calendar, int(capital * 0.2))}
            variants = (
                ("signal", 0),
                ("equal", 0),
                ("unknown_on", 0),
                ("signal_tax_stress", 0.2),
                ("equal_tax_stress", 0.2),
            )
            mark = time.perf_counter()
            for name, haircut in variants:
                guard()
                selected = signals[["signal_date", "ts_code", "eligible"]].copy()
                if name.startswith("equal"):
                    selected["eligible"] = True
                elif name == "unknown_on":
                    selected["eligible"] = signals.unknown_on
                account = run_monthly_account(
                    calendar=calendar,
                    quotes=quotes,
                    signals=selected,
                    capital=int(capital * 0.2),
                    corporate_book=book,
                    arm_name=name,
                    codes=CODES,
                    dividend_haircut=haircut,
                )
                _save_b(capital_dir / f"B_{name}", account)
                b_accounts[name] = normalize_monthly_daily(account["daily"])
            timings[f"b_accounts_{capital}"] = time.perf_counter() - mark
            mark = time.perf_counter()
            result, paths = analyze_capital(
                capital=capital,
                a_full=a_full["daily"],
                a_partial=a_partial["daily"],
                b_accounts=b_accounts,
            )
            for name, rows in paths.items():
                pd.DataFrame(rows).to_parquet(capital_dir / f"{name}_daily.parquet", index=False)
            write_json(capital_dir / "analysis.json", result)
            results[str(capital)] = result
            timings[f"analysis_{capital}"] = time.perf_counter() - mark
            print(f"资本 {capital} 已完成，累计 {time.perf_counter()-start:.1f}s", flush=True)
        guard()
        write_json(
            output / "pre_resolution_equality.json", _compare_completed_pre_resolution(output)
        )
        summary = {
            "results": results,
            "signal_summary": signal_summary(signals),
            "records_sha256": records_sha,
            "dividend_resolutions_sha256": resolutions_sha,
            "spec_sha256": spec["spec_sha256"],
            "timings_seconds": timings,
            "total_machine_seconds": time.perf_counter() - start,
            "data_purchase_yuan": 0,
            "formal_admission": "not_evaluated",
        }
        write_json(output / "summary.json", summary)
        if sha256(records_path) != records_sha:
            raise ValueError("运行期间公告记录变化")
        if sha256(resolutions_path) != resolutions_sha:
            raise ValueError("运行期间分红修订事实变化")
        changed = [
            path for path, digest in source_identity.items() if sha256(ROOT / path) != digest
        ]
        if changed:
            raise ValueError(f"运行期间代码变化：{changed}")
        files = {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        write_json(output / "manifest.json", {"files": files, "spec_sha256": spec["spec_sha256"]})
        write_json(
            output / "validation.json",
            {
                "complete": True,
                "source_unchanged": True,
                "account_identity_checked": True,
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
                "elapsed_seconds": time.perf_counter() - start,
                "timings_seconds": timings,
                "economic_success_claimed": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    logger.remove()
    run(args.run_id)


if __name__ == "__main__":
    main()
