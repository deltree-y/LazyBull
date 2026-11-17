"""固定续持的原生机械门、经济账户及候选未用历史窗口复验。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from scripts.v2_p7.buffer_native_corporate import (  # noqa: E402
    NativeCorporateBook,
    NativeCorporateMixin,
)
from scripts.v2_p7.buffer_native_engine import BufferNativeEngine  # noqa: E402
from scripts.v2_p7.buffer_native_inputs import build_holdout_rankings  # noqa: E402
from scripts.v2_p7.buffer_native_replay import (  # noqa: E402
    SPEC_SHA,
    RecordNativeMixin,
    compare_exact,
    engine_kwargs,
    initialize_driver,
    load_prices,
    read_historical_rankings,
    result_payload,
    sha256,
    write_json,
)
from scripts.v2_p7.buffer_native_validation import validate_account  # noqa: E402
from scripts.v2_p7.buffer_path_metrics import compare_accounts  # noqa: E402
from scripts.v2_p7.t4.corporate import CorporateError  # noqa: E402
from src.lazybull.v2.core.execution.price_index import prepare_price_index  # noqa: E402
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML  # noqa: E402

TIME_LIMIT_SECONDS = 45 * 60
CAPITALS = (1000000, 2000000)


class RecordedNative(RecordNativeMixin, NativeMergeBacktestEngineML):
    """未改变原生决策/成交的只读取证基准。"""


class RecordedResearch(RecordNativeMixin, NativeCorporateMixin, BufferNativeEngine):
    """公司行为共同适配与独立续持决策，记录层始终只读。"""


def dependencies(root: Path, spec: Path) -> dict[str, str]:
    """冻结实际依赖的代码、停牌/公司行为/行业元数据，不打印任何配置内容。"""
    paths = {spec, root / "configs/base.yaml", root / "data/paper/config.yaml"}
    paths.update((root / "src/lazybull").rglob("*.py"))
    paths.update((root / "scripts/v2_p7").rglob("*.py"))
    paths.update((root / "scripts/v2_p3").glob("*.py"))
    paths.update((root / "tests").glob("test_p7_buffer_native*.py"))
    for folder in ("dividend", "suspend"):
        paths.update((root / "data/raw" / folder).rglob("*.parquet"))
    for name in (
        "data/clean/stock_basic.parquet",
        "data/clean/trade_cal.parquet",
        "data/raw/shenwan_industry.parquet",
    ):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"缺必需元数据：{name}")
        paths.add(path)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def budget_check(started: float) -> None:
    """超时保留现场，拒绝发布删减窗口的胜出结论。"""
    if time.perf_counter() - started > TIME_LIMIT_SECONDS:
        raise TimeoutError("本轮正式准备与执行超过45机器分钟预算")


def cash_anchor(day: str, capital: int) -> tuple[dict, dict]:
    """无交易现金锚点不触发提前调仓，次日是窗口第一个信号日。"""
    daily = {
        "trade_date": day,
        "nav": 1.0,
        "total_value": float(capital),
        "cash": float(capital),
        "market_value": 0.0,
        "receivable": 0.0,
        "fees": 0.0,
        "buy_amount": 0.0,
        "sell_amount": 0.0,
        "positions": 0,
    }
    state = {
        "trade_date": day,
        "cash": float(capital),
        "positions": {},
        "pending_signals": {},
        "pending_condition_sells": {},
        "pending_stop_loss_sells": {},
        "unfilled_slots": {},
    }
    return daily, state


def execute_account(
    root: Path,
    folder: Path,
    *,
    ranked: dict,
    calendar: list[str],
    prices: pd.DataFrame,
    indexes: Any,
    capital: int,
    current_config: bool,
    kind: str,
    corporate_book: NativeCorporateBook,
    started: float,
    anchor: str | None,
) -> dict:
    """按完整日历顺序执行一账户；异常现场仍保存，但不发布其绩效。"""
    mark = time.perf_counter()
    folder.mkdir()
    kwargs = engine_kwargs(root, ranked, capital, current_config)
    if kind == "native_raw":
        engine = RecordedNative(**kwargs)
    else:
        engine = RecordedResearch(
            **kwargs,
            buffer_enabled=kind == "buffer_economic",
            rank_provider=lambda date: list(ranked.get(date.strftime("%Y%m%d"), [])),
            corporate_book=corporate_book if kind.endswith("economic") else None,
        )
    dates = [pd.Timestamp(day) for day in calendar]
    initialize_driver(engine, dates, prices, indexes)
    if anchor:
        daily, state = cash_anchor(anchor, capital)
        if hasattr(engine, "native_corporate"):
            engine.native_corporate.before_day(engine, pd.Timestamp(anchor))
            engine.native_corporate.after_day(engine, pd.Timestamp(anchor))
        engine.research_daily.append(daily)
        engine.research_states.append(state)
    print(f"START {folder.name} days={len(dates)}", flush=True)
    try:
        for count, date in enumerate(dates, 1):
            budget_check(started)
            engine.run_step(date, dates, prices)
            if count % 100 == 0:
                print(
                    f"PROGRESS {folder.name} {count}/{len(dates)} "
                    f"elapsed={time.perf_counter()-mark:.1f}s",
                    flush=True,
                )
    except Exception as error:
        payload = result_payload(engine)
        write_json(folder / "failed_checkpoint.json", payload)
        write_json(
            folder / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed_days": len(engine.research_daily),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    result = result_payload(engine)
    if hasattr(engine, "export_buffer_state"):
        result["buffer_state"] = engine.export_buffer_state()
    for key, value in result.items():
        write_json(folder / f"{key}.json", value)
    validation = validate_account(result, initial_capital=capital)
    write_json(folder / "validation.json", validation)
    timing = {
        "seconds": time.perf_counter() - mark,
        "days": len(calendar),
        "trade_rows": len(result["trades"]),
        "kind": kind,
        "current_config": current_config,
    }
    write_json(folder / "timing.json", timing)
    if not validation["valid"]:
        raise ValueError(f"账户守恒验证失败：{folder.name} {validation}")
    print(f"DONE {folder.name} {timing['seconds']:.3f}s", flush=True)
    return result


def run_group(
    root: Path,
    output: Path,
    name: str,
    *,
    ranked: dict,
    calendar: list[str],
    price_days: list[str],
    current_config: bool,
    started: float,
    anchor: str | None = None,
) -> dict:
    """先同配置raw机械门，再两档经济对照；不以结果改变规格。"""
    group = output / name
    group.mkdir()
    mark = time.perf_counter()
    prices, price_identity = load_prices(
        root, price_days, warmup_cutoff=calendar[0] if anchor else None
    )
    write_json(group / "price_identity.json", price_identity)
    indexes = prepare_price_index(prices)
    corporate = NativeCorporateBook(
        calendar=([anchor] if anchor else []) + calendar, dividend_dir=root / "data/raw/dividend"
    )
    write_json(
        group / "setup.json",
        {
            "calendar": calendar,
            "anchor": anchor,
            "price_days": price_days,
            "preparation_seconds": time.perf_counter() - mark,
            "position_sizing": "kelly" if current_config else "half_kelly",
            "max_per_industry": 5 if current_config else None,
            "buy_price": "close",
            "sell_price": "open",
            "completion_window_days": 5,
            "continuous_only": True,
            "live_t0_instruction_shares_equivalence": False,
        },
    )
    common = dict(
        root=root,
        ranked=ranked,
        calendar=calendar,
        prices=prices,
        indexes=indexes,
        current_config=current_config,
        corporate_book=corporate,
        started=started,
        anchor=anchor,
    )
    left = execute_account(
        folder=group / "native_raw_1000000", capital=1000000, kind="native_raw", **common
    )
    right = execute_account(
        folder=group / "research_off_raw_1000000",
        capital=1000000,
        kind="research_off_raw",
        **common,
    )
    mechanical = compare_exact(left, right)
    write_json(group / "mechanical.json", mechanical)
    if not mechanical["all_equal"]:
        raise ValueError(f"{name}关闭分支与原生机械门不一致")
    del left, right
    comparisons = {}
    for capital in CAPITALS:
        base = execute_account(
            folder=group / f"base_economic_{capital}",
            capital=capital,
            kind="base_economic",
            **common,
        )
        arm = execute_account(
            folder=group / f"buffer_economic_{capital}",
            capital=capital,
            kind="buffer_economic",
            **common,
        )
        comparison = compare_accounts(base["daily"], arm["daily"])
        comparison["evidence_class"] = "historical_continuous_replay_only"
        comparison["current_config_descriptive_only"] = current_config
        write_json(group / f"comparison_{capital}.json", comparison)
        comparisons[str(capital)] = comparison
        del base, arm
    write_json(
        group / "complete.json", {"seconds": time.perf_counter() - mark, "mechanical_exact": True}
    )
    del prices, indexes
    gc.collect()
    return {"mechanical": mechanical, "comparisons": comparisons}


def run_independent_group(root: Path, output: Path, name: str, **kwargs: Any) -> dict:
    """保留公司行为数据阻断，继续同一预登记中的独立窗口；不吞实现/机械错误。"""
    try:
        return run_group(root, output, name, **kwargs)
    except CorporateError as error:
        folder = output / name
        result = {
            "status": "blocked_corporate",
            "message": str(error),
            "economic_comparison_published": False,
        }
        mechanical_path = folder / "mechanical.json"
        if mechanical_path.is_file():
            result["mechanical"] = json.loads(mechanical_path.read_text(encoding="utf-8"))
        write_json(folder / "blocked.json", result)
        print(f"GROUP_BLOCKED {name}: {error}", flush=True)
        return result


def run(root: Path, output: Path, spec: Path) -> dict:
    """一次性冻结全流程，所有输出新建；最终核查输入未被其它任务改写。"""
    started = time.perf_counter()
    if sha256(spec) != SPEC_SHA:
        raise ValueError("预登记字节不符")
    output.mkdir(parents=True, exist_ok=False)
    methods = output / "methods"
    methods.mkdir()
    shutil.copyfile(spec, methods / spec.name)
    for pattern in ("buffer_native*.py", "run_buffer_native.py"):
        for path in (root / "scripts/v2_p7").glob(pattern):
            shutil.copyfile(path, methods / path.name)
    for path in (root / "tests").glob("test_p7_buffer_native*.py"):
        shutil.copyfile(path, methods / path.name)
    identity = dependencies(root, spec)
    write_json(output / "dependency_identity.json", identity)
    try:
        mark = time.perf_counter()
        historical, inherited = read_historical_rankings(root)
        write_json(output / "historical_input.json", inherited)
        print(f"HISTORICAL_INPUT {time.perf_counter()-mark:.3f}s", flush=True)
        mark = time.perf_counter()
        holdout = build_holdout_rankings(root, output / "holdout_inputs", spec_sha256=SPEC_SHA)
        print(f"HOLDOUT_INPUT {time.perf_counter()-mark:.3f}s", flush=True)
        groups = {}
        groups["historical_main"] = run_independent_group(
            root,
            output,
            "historical_main",
            ranked=historical,
            calendar=inherited["calendar"],
            price_days=inherited["calendar"],
            current_config=False,
            started=started,
        )
        # 新窗每股固定前60条可得行情预热；停牌股按实际行而非日历槽取尾部。
        prior = [day for day in inherited["calendar"] if day < "20260203"]
        days = sorted(holdout)
        if len(days) != 176 or days[0] != "20260203" or days[-1] != "20261029":
            raise ValueError("新增窗口未完整覆盖冻结176日")
        for current in (False, True):
            name = "holdout_current_descriptive" if current else "holdout_main"
            groups[name] = run_independent_group(
                root,
                output,
                name,
                ranked=holdout,
                calendar=days,
                price_days=prior + days,
                current_config=current,
                started=started,
                anchor="20260202",
            )
        if dependencies(root, spec) != identity:
            raise ValueError("运行期间方法或元数据改变，不发布完整结果")
        # 精确核对全部实际价格分区，模型/特征缓存已各自独立持有SHA与前后护栏。
        for group in output.glob("*/price_identity.json"):
            saved = json.loads(group.read_text(encoding="utf-8"))
            for name, digest in saved.items():
                if sha256(root / name) != digest:
                    raise ValueError(f"运行期间行情改变：{name}")
        blocked = [
            name for name, value in groups.items() if value.get("status") == "blocked_corporate"
        ]
        summary = {
            "completed": not blocked,
            "execution_finished": True,
            "blocked_groups": blocked,
            "seconds": time.perf_counter() - started,
            "data_cost_yuan": 0,
            "spec_sha256": SPEC_SHA,
            "groups": groups,
        }
        write_json(output / "summary.json", summary)
        files = {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        write_json(
            output / "manifest.json",
            {
                "completed": not blocked,
                "execution_finished": True,
                "spec_sha256": SPEC_SHA,
                "outputs": files,
            },
        )
        print(f"FINISHED seconds={summary['seconds']:.3f} blocked_groups={blocked}", flush=True)
        return summary
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "completed": False,
                "seconds": time.perf_counter() - started,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "economic_success_claimed": False,
            },
        )
        print(f"FAILED {type(error).__name__}: {error}", flush=True)
        raise


def main() -> None:
    """命令入口；禁自动覆盖旧run。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spec", type=Path, default=ROOT / "docs/experiments/p7_buffer_native_prereg_20261116.md"
    )
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    run(args.repo_root.resolve(), args.output.resolve(), args.spec.resolve())


if __name__ == "__main__":
    main()
