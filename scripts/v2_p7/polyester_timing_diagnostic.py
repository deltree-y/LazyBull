"""涤纶冻结路径的报价时点与损益差诊断；不生成新策略或重跑账户。

阶段研究工具：只读 run_r1 和原报价封包，结果写入新的诊断目录。
损益差按实际金额望远镜相加；月度归因是两条既有路径的会计差，不是删月反事实。
复杂度豁免：阶段专用只读研究诊断，不入生产链路（CLAUDE §7.8）；
若未来接入长期生产，将信号反查、会计归因、报告封装分别拆分后重新验收。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/reports/p7_polyester_economic_20261116"
OUTPUT = ROOT / "data/reports/p7_polyester_timing_20261116/delay_attribution"
RUN_SHA = "f45dcf2baa9fb663ff9b5d14f6846819481cb0c6f4e3b0b197fd3d6b1f0d2063"
INPUT_SHA = "cc805d1be93995a6f6d63a1ad3f815a74dec202084e7ebc89d7fb887b19e9e8f"
CRASH_START, CRASH_END = "20230508", "20240205"
CODES = ("601233.SH", "603225.SH")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _scalar(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_scalar(item) for item in value]
    return None if pd.isna(value) else value


def _near(left: float, right: float, context: str, *, tolerance: float = 1e-8) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{context} 不闭合：{left} != {right}")


def collapse_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """先验两股共享行业信号一致，再压成每月一条，不把两股当独立观察。"""
    columns = [name for name in frame.columns if name != "ts_code"]
    result = []
    for day, group in frame.groupby("signal_date", sort=True):
        if sorted(group.ts_code.tolist()) != sorted(CODES):
            raise ValueError(f"{day} 信号槽数或股票域不符")
        payloads = [{name: _scalar(row[name]) for name in columns} for _, row in group.iterrows()]
        if payloads[0] != payloads[1]:
            raise ValueError(f"{day} 两股行业信号不一致")
        result.append(payloads[0])
    return pd.DataFrame(result)


def _spread(row: dict) -> float:
    return row["poy"] - 0.855 * row["pta"] - 0.335 * row["meg"]


def source_legs(signals: pd.DataFrame, records: list[dict], arm: str) -> pd.DataFrame:
    """只反查已保存 source_ids，核四点算术；不重新选点或产生新信号。"""
    lookup = {row["source_id"]: row for row in records}
    if len(lookup) != len(records):
        raise ValueError("报价 source_id 重复")
    result = []
    for _, signal in signals.iterrows():
        selected = []
        if len(signal.source_ids) != 4 or len(signal.months) != 4:
            raise ValueError(f"{signal.signal_date} 保存的四点域不符")
        for lag, (source_id, month) in enumerate(zip(signal.source_ids, signal.months)):
            row = lookup.get(source_id) if source_id else None
            if source_id and row is None:
                raise ValueError(f"{signal.signal_date} 无法反查来源 {source_id}")
            selected.append(row)
            if row:
                if (
                    row["observation_date"][:6] != month
                    or row["available_date"] >= signal.availability_cutoff
                ):
                    raise ValueError(f"{signal.signal_date}/{source_id} 月份或可用时点不符")
                if not row["quality_usable"] or (
                    arm == "source_strict" and not row["strict_usable"]
                ):
                    raise ValueError(f"{signal.signal_date}/{source_id} 消费不可用报价")
            item = {
                "arm": arm,
                "signal_date": signal.signal_date,
                "signal_month": signal.signal_date[:6],
                "lag_month": lag,
                "quote_month": month,
                "availability_cutoff": signal.availability_cutoff,
                "source_id": source_id,
                "missing": row is None,
            }
            for name in (
                "observation_date",
                "publication_date",
                "available_date",
                "series_id",
                "source_kind",
                "publication_basis",
                "source_sha256",
                "strict_usable",
                "poy",
                "pta",
                "meg",
            ):
                item[name] = row.get(name) if row else None
            item["spread"] = _spread(row) if row else None
            result.append(item)
        known = (
            all(row is not None for row in selected)
            and len({row["series_id"] for row in selected if row}) == 1
        )
        if known != bool(signal.input_known):
            raise ValueError(f"{signal.signal_date} 保存的 known 与四点域不符")
        current = _spread(selected[0]) if selected[0] else None
        if current is not None:
            _near(current, signal.spread, f"{signal.signal_date} 价差")
        elif pd.notna(signal.spread):
            raise ValueError(f"{signal.signal_date} 缺当月点却有价差")
        average = sum(_spread(row) for row in selected[1:]) / 3 if known else None
        if known:
            _near(average, signal.prior_three_mean, f"{signal.signal_date} 前三月均值")
        if bool(signal.eligible) != bool(known and current > 0 and current > average):
            raise ValueError(f"{signal.signal_date} 保存决定与四点算术不符")
    return pd.DataFrame(result)


def _category(left: pd.Series, right: pd.Series) -> str:
    if left.input_known and right.input_known:
        if left.eligible == right.eligible:
            return "both_known_same_decision"
        return "both_known_main_on_delay_off" if left.eligible else "both_known_main_off_delay_on"
    if left.input_known:
        return "main_known_delay_unknown"
    if right.input_known:
        return "main_unknown_delay_known"
    return "both_unknown"


def compare_signals(
    main: pd.DataFrame, delay: pd.DataFrame, strict: pd.DataFrame, legs: pd.DataFrame
) -> pd.DataFrame:
    """按月并排四点与决定，known 反转和 unknown 变化分别命名。"""
    expected = main.signal_date.tolist()
    if delay.signal_date.tolist() != expected or strict.signal_date.tolist() != expected:
        raise ValueError("三条保存信号的月份域不同")
    result = []
    fields = (
        "input_known",
        "eligible",
        "reason",
        "spread",
        "prior_three_mean",
        "availability_cutoff",
        "source_ids",
    )
    for index in range(len(main)):
        left, right, source_strict = main.iloc[index], delay.iloc[index], strict.iloc[index]
        row = {
            "signal_date": left.signal_date,
            "month": left.signal_date[:6],
            "category": _category(left, right),
            "decision_changed": bool(left.eligible != right.eligible),
            "changed_lags": [
                lag for lag in range(4) if left.source_ids[lag] != right.source_ids[lag]
            ],
            "main_strict_decision_changed": bool(left.eligible != source_strict.eligible),
        }
        for name, signal in (("main", left), ("delay5", right), ("source_strict", source_strict)):
            row.update({f"{name}_{field}": _scalar(signal[field]) for field in fields})
            match = legs.loc[
                legs.arm.eq(name) & legs.signal_date.eq(left.signal_date) & legs.lag_month.eq(0)
            ].iloc[0]
            row.update(
                {
                    f"{name}_{field}": _scalar(match[field])
                    for field in (
                        "observation_date",
                        "publication_date",
                        "available_date",
                        "series_id",
                    )
                }
            )
        result.append(row)
    return pd.DataFrame(result)


def wealth_difference(
    main: pd.DataFrame, delay: pd.DataFrame, total_capital: int
) -> tuple[pd.DataFrame, dict]:
    """同资本无外部流量路径的金额差；不能将其解释为独立月份策略收益。"""
    days = main.trade_date.astype(str).tolist()
    if days != sorted(set(days)) or days != delay.trade_date.astype(str).tolist():
        raise ValueError("损益归因日轴不同或重复")
    if len(days) < 2:
        raise ValueError("损益归因缺少初始点")
    initial = main.total_value.iloc[0]
    _near(initial, delay.total_value.iloc[0], "相同初始现金")
    _near(initial, total_capital / 5, "实际 B 初始资本")
    result = pd.DataFrame({"trade_date": days})
    for prefix, frame in (("main", main), ("delay5", delay)):
        residual = (
            frame.total_value
            - frame.cash
            - frame.market_value
            - frame.receivable
            + frame.dividend_haircut_liability
        )
        if residual.abs().max() > 1e-6:
            raise ValueError(f"{prefix} 资产恒等式不成立")
        for name in (
            "total_value",
            "cash",
            "market_value",
            "receivable",
            "fees",
            "buy_amount",
            "sell_amount",
            "holding_count",
        ):
            result[f"{prefix}_{name}"] = frame[name].to_numpy()
        result[f"{prefix}_net_pnl"] = frame.total_value.diff().fillna(0).to_numpy()
        result[f"{prefix}_gross_pnl"] = result[f"{prefix}_net_pnl"] + frame.fees.to_numpy()
        result[f"{prefix}_exposure"] = (frame.market_value / frame.total_value).to_numpy()
    result["net_pnl_difference"] = result.main_net_pnl - result.delay5_net_pnl
    result["gross_pnl_difference"] = result.main_gross_pnl - result.delay5_gross_pnl
    result["fee_contribution"] = result.delay5_fees - result.main_fees
    result["wealth_difference"] = result.main_total_value - result.delay5_total_value
    result["cumulative_contribution"] = result.net_pnl_difference.cumsum()
    result["total_capital_return_contribution_pp"] = result.net_pnl_difference / total_capital * 100
    _near(
        result.net_pnl_difference.sum(),
        result.wealth_difference.iloc[-1],
        "逐日损益差望远镜",
        tolerance=1e-6,
    )
    if (result.cumulative_contribution - result.wealth_difference).abs().max() > 1e-6:
        raise ValueError("累计损益差和逐日资产差不一致")
    if (
        result.net_pnl_difference - result.gross_pnl_difference - result.fee_contribution
    ).abs().max() > 1e-6:
        raise ValueError("费用拆解不闭合")
    return result, {
        "initial_b_capital": initial,
        "ending_wealth_difference_yuan": float(result.wealth_difference.iloc[-1]),
        "net_pnl_difference_yuan": float(result.net_pnl_difference.sum()),
        "gross_pnl_difference_yuan": float(result.gross_pnl_difference.sum()),
        "fee_contribution_yuan": float(result.fee_contribution.sum()),
        "total_capital_terminal_return_difference_pp": float(
            result.total_capital_return_contribution_pp.sum()
        ),
        "path_dependent_not_month_deletion_counterfactual": True,
    }


def aggregate_difference(daily: pd.DataFrame, period: int) -> pd.DataFrame:
    """自然月/年金额归并，首期用保存锚点而不重置资金。"""
    work = daily.assign(period=daily.trade_date.str[:period])
    result = []
    for key, group in work.groupby("period", sort=True):
        row = {
            "period": key,
            "start_date": group.trade_date.iloc[0],
            "end_date": group.trade_date.iloc[-1],
            "days": len(group),
        }
        for field in (
            "net_pnl_difference",
            "gross_pnl_difference",
            "fee_contribution",
            "total_capital_return_contribution_pp",
            "main_net_pnl",
            "delay5_net_pnl",
            "main_fees",
            "delay5_fees",
            "main_buy_amount",
            "delay5_buy_amount",
            "main_sell_amount",
            "delay5_sell_amount",
        ):
            row[field] = float(group[field].sum())
        for prefix in ("main", "delay5"):
            row[f"{prefix}_mean_exposure"] = float(group[f"{prefix}_exposure"].mean())
            row[f"{prefix}_held_days"] = int(group[f"{prefix}_holding_count"].gt(0).sum())
        row["ending_wealth_difference"] = float(group.wealth_difference.iloc[-1])
        result.append(row)
    return pd.DataFrame(result)


def transaction_context(
    trades: pd.DataFrame, holdings: pd.DataFrame, daily: pd.DataFrame, arm: str
) -> pd.DataFrame:
    """给旧成交补上前日持仓证据；小额补仓与初次建仓分开，混合日不武断分类。"""
    previous = dict(zip(daily.trade_date.iloc[1:], daily.trade_date.iloc[:-1]))
    if holdings.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("日终持仓主键重复")
    held = holdings.set_index(["trade_date", "ts_code"]).shares.to_dict()
    result = []
    for _, trade in trades.iterrows():
        row = trade.to_dict()
        prior = previous.get(trade.trade_date)
        prior_shares = int(held.get((prior, trade.ts_code), 0))
        same_day_sold = trades.loc[
            trades.trade_date.eq(trade.trade_date)
            & trades.ts_code.eq(trade.ts_code)
            & trades.side.eq("sell")
        ]
        kind = "sell"
        if trade.side == "buy":
            kind = (
                "same_day_sell_and_buy_requires_sequence"
                if len(same_day_sold)
                else ("existing_position_topup" if prior_shares > 0 else "new_position")
            )
        row.update(
            {
                "arm": arm,
                "signal_month": str(trade.signal_date)[:6],
                "execution_month": str(trade.trade_date)[:6],
                "prior_close_date": prior,
                "prior_close_shares": prior_shares,
                "transaction_role": kind,
                "prior_shares_not_adjusted_for_intraday_corporate": True,
            }
        )
        result.append(row)
    return pd.DataFrame(result)


def planned_execution_comparison(main: pd.DataFrame, delay: pd.DataFrame) -> pd.DataFrame:
    """比较已保存计划执行日；信号截断压力不等于把成交统一后推五日。"""
    columns = ["trade_date", "ts_code", "execution_date", "eligible", "notional"]
    left, right = (frame.loc[frame.kind.eq("planned"), columns] for frame in (main, delay))
    result = left.merge(
        right,
        on=["trade_date", "ts_code"],
        how="outer",
        suffixes=("_main", "_delay5"),
        indicator=True,
        validate="one_to_one",
    )
    if not result["_merge"].eq("both").all():
        raise ValueError("两臂计划行域不一致")
    result["same_execution_date"] = result.execution_date_main.eq(result.execution_date_delay5)
    return result.drop(columns="_merge")


def _verify(root: Path, files: dict[str, str]) -> int:
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or _sha(path) != expected:
            raise ValueError(f"只读输入 SHA/路径不符：{relative}")
    return len(files)


def _save_frame(root: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(root / f"{name}.parquet", index=False)
    view = frame.copy()
    for column in view.columns:
        if view[column].map(lambda value: isinstance(value, (list, dict))).any():
            view[column] = view[column].map(
                lambda value: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
            )
    view.to_csv(root / f"{name}.csv", index=False, encoding="utf-8-sig")


def _brief(output: Path, summary: dict) -> None:
    """从完整会计归因表生成可读摘要，不选择月份重跑。"""
    monthly = pd.read_parquet(output / "1000000/monthly_attribution.parquet")
    selected = pd.concat(
        [monthly.nlargest(6, "net_pnl_difference"), monthly.nsmallest(4, "net_pnl_difference")]
    ).drop_duplicates("period")
    lines = [
        "# 涤纶五日信息压力：冻结路径诊断",
        "",
        "本诊断在经济结果已知后实施，只读取旧 run_r1，不产生新策略或删月反事实。金额正号表示主场景优于 delay5。",
        "",
        f"91 个信号月中决定变化 {summary['decision_changes']} 月，"
        f"报价点变化 {summary['source_point_changes']} 月。分类：{summary['categories']}。",
        "",
        "五日压力仍用每月首个交易日 T 决策，并保持 T+1 计划执行日。"
        "它把选择报价的可用截止点回退五个交易日，所以通常改用前一个自然月更早的报价；"
        "这会改变 S，也可能造成输入未知。它不是保持同一份信号再将订单统一延迟五日。",
        "",
        "逐日贡献 = (E_main,t−E_main,t−1)−(E_delay,t−E_delay,t−1)。"
        "自然月和年度贡献求和精确闭合到末日资产差；除以初始总资本得到可加的终值收益百分点，"
        "不能当作 CAGR 的月度贡献。价格、红利与送股都在实际净资产内，费用另外显式拆解。",
        "",
        "| 总资本 | 主减延迟末值差（元） | 费用贡献（元） | 回撤窗口增量差（元） | 同计划执行日行数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for capital, row in summary["capital"].items():
        lines.append(
            f"| {capital} | {row['ending_wealth_difference_yuan']:.2f} | "
            f"{row['fee_contribution_yuan']:.2f} | {row['crash']['window_pnl_difference']:.2f} | "
            f"{row['same_planned_execution_dates']}/{row['planned_rows']} |"
        )
    lines.extend(
        [
            "",
            "100 万元档贡献较大的正负月份（完整月份另见 monthly_attribution；排序仅用于解释）：",
            "",
            "| 月份 | 主减延迟净损益（元） | 当前月信号分类 | 主/延迟平均暴露 |",
            "|---|---:|---|---|",
        ]
    )
    for _, row in selected.iterrows():
        lines.append(
            f"| {row.period} | {row.net_pnl_difference:.2f} | {row.category} | "
            f"{row.main_mean_exposure:.3f}/{row.delay5_mean_exposure:.3f} |"
        )
    lines.extend(
        [
            "",
            "20230508 收盘至 20240205 收盘另见 "
            "crash_daily/crash_monthly/crash_trades/crash_holdings。"
            "窗口增量不计峰值当日已形成的资产差；期初与期末差均单独保留。",
            "",
            "同月决定相同也可能贡献损益差，因为两臂此前买入量、持仓、现金、费用及未成交义务不同。"
            "按当前月信号类别归并是路径标签，不是对该信号独立因果效应的估计。"
            "不得把差额最大的月份删除后宣称策略改善。",
            "",
            "trade_context 依据前收盘持仓及同日买卖，将新建仓和已有仓位补仓分开；"
            "同日买卖混合保留需逐笔核对标签。前收盘股数未冒称除权后的精确开盘股数。"
            "小额补仓不能自动归为新仓容量失败。",
            "",
            f"程序机器耗时：{summary['machine_seconds']:.3f} 秒；新增策略运行 0；数据采购 0 元。",
        ]
    )
    (output / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path = OUTPUT) -> dict:
    """验证旧输入、读取旧路径、计算诊断、冻结新方法；没有账户执行入口。"""
    start = time.perf_counter()
    run_root = BASE / "run_r1"
    if (
        _sha(run_root / "manifest.json") != RUN_SHA
        or _sha(BASE / "input_prices/source_manifest.json") != INPUT_SHA
    ):
        raise ValueError("原经济封包身份变化")
    run_manifest, source_manifest = _read(run_root / "manifest.json"), _read(
        BASE / "input_prices/source_manifest.json"
    )
    checks = {
        "run_files": _verify(run_root, run_manifest["files"]),
        "source_files": _verify(ROOT, source_manifest["files"]),
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        methods = output / "methods"
        methods.mkdir()
        for path in (Path(__file__), ROOT / "tests/test_p7_polyester_timing_diagnostic.py"):
            shutil.copyfile(path, methods / path.name)
        scope = ROOT / "docs/experiments/p7_polyester_timing_scope_20261116.md"
        identity = {
            "run_manifest_sha256": RUN_SHA,
            "input_manifest_sha256": INPUT_SHA,
            "scope_sha256": _sha(scope),
            "method_sha256": _sha(Path(__file__)),
            "test_sha256": _sha(ROOT / "tests/test_p7_polyester_timing_diagnostic.py"),
            "checks": checks,
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "post_result_diagnostic": True,
            "new_strategy_runs": 0,
            "independent_review": False,
        }
        _write(output / "identity.json", identity)
        records = _read(BASE / "input_prices/records.json")
        signals = {
            name: collapse_signals(pd.read_parquet(run_root / path))
            for name, path in (
                ("main", "signals.parquet"),
                ("delay5", "signals_delay5.parquet"),
                ("source_strict", "strict_signals.parquet"),
            )
        }
        legs = pd.concat(
            [source_legs(frame, records, name) for name, frame in signals.items()],
            ignore_index=True,
        )
        comparison = compare_signals(
            signals["main"], signals["delay5"], signals["source_strict"], legs
        )
        _save_frame(output, "signal_source_legs", legs)
        _save_frame(output, "monthly_signal_comparison", comparison)
        summaries = {}
        for capital in (1000000, 2000000):
            target = output / str(capital)
            target.mkdir()
            frames = {
                name: pd.read_parquet(run_root / str(capital) / folder / "daily.parquet")
                for name, folder in (("main", "B_signal"), ("delay5", "B_delay5"))
            }
            daily, summary = wealth_difference(frames["main"], frames["delay5"], capital)
            monthly = aggregate_difference(daily, 6).merge(
                comparison, left_on="period", right_on="month", how="left", validate="one_to_one"
            )
            yearly = aggregate_difference(daily, 4)
            _save_frame(target, "daily_wealth_difference", daily)
            _save_frame(target, "monthly_attribution", monthly)
            _save_frame(target, "yearly_attribution", yearly)
            groups = (
                monthly.assign(category=monthly.category.fillna("anchor_no_signal"))
                .groupby("category")
                .agg(
                    months=("period", "size"),
                    net_pnl_difference=("net_pnl_difference", "sum"),
                    fee_contribution=("fee_contribution", "sum"),
                )
                .reset_index()
            )
            _save_frame(target, "attribution_by_current_month_signal_category", groups)
            contexts, decisions, holding_frames = [], {}, []
            for name, folder in (("main", "B_signal"), ("delay5", "B_delay5")):
                account = run_root / str(capital) / folder
                trades, holdings = pd.read_parquet(account / "trades.parquet"), pd.read_parquet(
                    account / "holdings.parquet"
                )
                contexts.append(transaction_context(trades, holdings, frames[name], name))
                holding_frames.append(holdings.assign(arm=name))
                decisions[name] = pd.read_parquet(account / "decisions.parquet")
            trades = pd.concat(contexts, ignore_index=True)
            holdings = pd.concat(holding_frames, ignore_index=True)
            plans = planned_execution_comparison(decisions["main"], decisions["delay5"])
            _save_frame(target, "trade_context", trades)
            _save_frame(target, "holdings_comparison", holdings)
            _save_frame(target, "planned_execution_comparison", plans)
            _save_frame(
                target,
                "trade_counts_by_role",
                trades.groupby(["arm", "transaction_role"])
                .agg(
                    trades=("shares", "size"),
                    amount=("amount", "sum"),
                    minimum_amount=("amount", "min"),
                )
                .reset_index(),
            )
            crash = daily.loc[
                daily.trade_date.gt(CRASH_START) & daily.trade_date.le(CRASH_END)
            ].copy()
            peak = daily.loc[daily.trade_date.eq(CRASH_START)].iloc[0]
            end = daily.loc[daily.trade_date.eq(CRASH_END)].iloc[0]
            crash["window_cumulative_difference"] = crash.net_pnl_difference.cumsum()
            _near(
                crash.net_pnl_difference.sum(),
                end.wealth_difference - peak.wealth_difference,
                "回撤窗口损益差",
                tolerance=1e-6,
            )
            _save_frame(target, "crash_daily", crash)
            _save_frame(
                target,
                "crash_monthly",
                aggregate_difference(crash, 6).merge(
                    comparison, left_on="period", right_on="month", how="left"
                ),
            )
            _save_frame(
                target,
                "crash_trades",
                trades.loc[trades.trade_date.ge(CRASH_START) & trades.trade_date.le(CRASH_END)],
            )
            _save_frame(
                target,
                "crash_holdings",
                holdings.loc[
                    holdings.trade_date.ge(CRASH_START) & holdings.trade_date.le(CRASH_END)
                ],
            )
            combined_main = pd.read_parquet(run_root / str(capital) / "A80_B_signal_daily.parquet")
            combined_delay = pd.read_parquet(run_root / str(capital) / "A80_B_delay5_daily.parquet")
            if (
                combined_main.trade_date.tolist() != daily.trade_date.tolist()
                or combined_delay.trade_date.tolist() != daily.trade_date.tolist()
            ):
                raise ValueError("组合路径日历与 B 路径不同")
            if (
                (combined_main.total_value - combined_delay.total_value) - daily.wealth_difference
            ).abs().max() > 1e-6:
                raise ValueError("相同 A80 相消后组合金额差与 B 金额差不等")
            summary.update(
                {
                    "planned_rows": len(plans),
                    "same_planned_execution_dates": int(plans.same_execution_date.sum()),
                    "transaction_role_counts": trades.groupby(["arm", "transaction_role"])
                    .size()
                    .to_dict(),
                    "crash": {
                        "peak_close": CRASH_START,
                        "trough_close": CRASH_END,
                        "increment_start_excludes_peak_close": True,
                        "wealth_difference_at_peak": float(peak.wealth_difference),
                        "wealth_difference_at_trough": float(end.wealth_difference),
                        "window_pnl_difference": float(crash.net_pnl_difference.sum()),
                        "main_window_pnl": float(crash.main_net_pnl.sum()),
                        "delay5_window_pnl": float(crash.delay5_net_pnl.sum()),
                    },
                }
            )
            summary["transaction_role_counts"] = {
                "/".join(key): value for key, value in summary["transaction_role_counts"].items()
            }
            _write(target / "summary.json", summary)
            summaries[str(capital)] = summary
        _verify(run_root, run_manifest["files"])
        _verify(ROOT, source_manifest["files"])
        summary = {
            "months": len(comparison),
            "categories": comparison.category.value_counts().to_dict(),
            "decision_changes": int(comparison.decision_changed.sum()),
            "source_point_changes": int(comparison.changed_lags.map(bool).sum()),
            "capital": summaries,
            "limitations": [
                "两条完整既有路径的金额归因，存在持仓、资本和未成交义务路径依赖",
                "按自然月归并不等于该月信号的独立因果效果，更不等于删除该月后重跑",
                "五日信息截断保持T/T+1计划日不变，会选择同自然月更早报价，并非纯执行延迟",
                "税费前损益包含价格与公司行为影响，未伪装成纯价格效果",
            ],
            "machine_seconds": time.perf_counter() - start,
            "new_strategy_runs": 0,
            "data_cost_cny": 0,
        }
        _write(output / "summary.json", summary)
        _brief(output, summary)
        _write(
            output / "manifest.json",
            {
                "files": {
                    path.relative_to(output).as_posix(): _sha(path)
                    for path in sorted(output.rglob("*"))
                    if path.is_file()
                },
                "read_only_old_run": True,
            },
        )
        return summary
    except Exception as error:
        _write(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.perf_counter() - start,
                "new_strategy_runs": 0,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
