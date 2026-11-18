"""R5 诊断编排：主表、目标核对、分段、自举、相关与四个固定压力。

压力只诊断，不择优；每个压力只改一项，并与主口径并列登记。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from scripts.v2_p7.defensive_ml import b_native, evaluation, replay_stage, spec, universe

ARMS_OF_INTEREST = ("A100", "A80_CASH20", "A80_DEF20", "A80_LOWVOL20")


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def load_arm_series(campaign_dir: Path, capital: int) -> dict[str, pd.Series]:
    """读入四臂逐日总值（A 两臂 + DEF/LOWVOL 两臂）。"""
    accounts = Path(campaign_dir) / "accounts" / str(capital)
    def_arms = pd.read_parquet(accounts / "DEF_arms.parquet")
    low_arms = pd.read_parquet(accounts / "LOWVOL_arms.parquet")
    series = {
        "A100": def_arms["A100"].astype(float),
        "A80_CASH20": def_arms["A80_CASH20"].astype(float),
        "A80_DEF20": def_arms["A80_DEF20"].astype(float),
        "A80_LOWVOL20": low_arms["A80_LOWVOL20"].astype(float),
    }
    for name, values in series.items():
        values.index = [str(day) for day in values.index]
    return series


def sleeve_series(campaign_dir: Path, capital: int, label: str) -> pd.Series:
    """单袖自身净值（用于 B/LOWVOL 自身比较）。"""
    path = Path(campaign_dir) / "accounts" / str(capital) / f"{label}_daily.parquet"
    frame = pd.read_parquet(path)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.set_index("trade_date")["total_value"].astype(float)


def _stress_configs() -> dict[str, dict[str, Any]]:
    return {
        "slippage_doubled": {"stress_slippage": float(spec.STRESS["slippage_doubled"])},
        "rank_lag_1d": {"rank_lag": int(spec.STRESS["rank_lag_trading_days"])},
        "cash_discount_20pct": {"cash_discount": float(spec.STRESS["cash_equity_discount"])},
    }


def run_diagnostics(
    root: Path,
    campaign_dir: Path,
    *,
    log: Callable[[str], None] | None = None,
    run_stress: bool = True,
) -> dict[str, Any]:
    """汇总四臂读数与压力；返回可直接落盘的机器摘要。"""
    emit = _emit(log)
    output: dict[str, Any] = {"main": {}, "stress": {}, "bootstrap": {}, "correlation": {}}
    for capital in spec.TARGETS["capital_scenarios"]:
        arms = load_arm_series(campaign_dir, capital)
        table = evaluation.arm_metrics_table(arms)
        b_series = sleeve_series(campaign_dir, capital, "DEF")
        low_series = sleeve_series(campaign_dir, capital, "LOWVOL")
        output["main"][str(capital)] = {
            "arms": table.to_dict("records"),
            "target_vs_a100": evaluation.target_check(
                arms["A80_DEF20"], arms["A100"], arms["A80_CASH20"]
            ),
            "target_vs_cash": evaluation.target_check(
                arms["A80_DEF20"], arms["A80_CASH20"], arms["A80_CASH20"]
            ),
            "lowvol_target_vs_a100": evaluation.target_check(
                arms["A80_LOWVOL20"], arms["A100"], arms["A80_CASH20"]
            ),
            "segments": evaluation.segment_table(
                arms, [list(seg) for seg in spec.WINDOW["segments"]]
            ).to_dict("records"),
            "sleeve_only": {
                "def": evaluation.chain_metrics(b_series),
                "lowvol": evaluation.chain_metrics(low_series),
            },
        }
        output["correlation"][str(capital)] = evaluation.correlation_table(arms).to_dict()
        output["bootstrap"][str(capital)] = {
            "def_vs_a100": evaluation.bootstrap_deltas(arms, "A100", "A80_DEF20"),
            "def_vs_cash": evaluation.bootstrap_deltas(arms, "A80_CASH20", "A80_DEF20"),
        }
        emit(f"capital {capital} 主表完成")

    # 审计与零回收都必须在压力重放**之前**读取主口径账户，否则会读到压力覆盖后的账本
    pit = universe.industry_pit_frame(root)
    names = universe.industry_l1_names(universe.industry_code_table_frame(root), pit)
    audit = engine_audit(campaign_dir, pit, names)
    output["engine_audit"] = audit
    errors: list[str] = []
    if audit["any_day_over_20_slots"]:
        errors.append("存在单日持仓超过 20 槽")
    if not audit["both_sleeves_same_first_buy"]:
        errors.append("两袖首次买入日不一致（排期不同源）")
    for capital, entry in audit["per_capital"].items():
        for label, item in entry.items():
            industry = item.get("industry", {})
            if industry.get("days_over_industry_cap", 0) > 0:
                errors.append(
                    f"{capital}/{label} 行业上限超限 {industry['days_over_industry_cap']} 天"
                    f"（单桶最大 {industry['max_industry_positions']} 只）"
                )
    output["engineering_errors"] = errors
    output["zero_recovery"] = zero_recovery_check(campaign_dir)
    if run_stress:
        output["stress"] = _run_stress(root, campaign_dir, emit)
    return output


def _run_stress(
    root: Path, campaign_dir: Path, emit: Callable[[str], None]
) -> dict[str, Any]:
    """逐压力重放 B 袖并与冻结 A 重新组合。"""
    stress: dict[str, Any] = {}
    for name, options in _stress_configs().items():
        entry: dict[str, Any] = {}
        try:
            result = replay_stage.run_replay_stage(
                root, campaign_dir, log=emit, sleeve_kind="def", **options
            )
        except Exception as exc:  # noqa: BLE001 - 未实现即登记，不伪造结果
            stress[name] = {"status": "not_run", "reason": f"{type(exc).__name__}: {exc}"}
            emit(f"压力 {name} 未运行：{exc}")
            continue
        for capital in spec.TARGETS["capital_scenarios"]:
            sleeve = result["sleeve"][str(capital)]["daily"]
            series = sleeve.set_index("trade_date")["total_value"].astype(float)
            base = load_arm_series(campaign_dir, capital)
            combined = base["A80_CASH20"] - (float(capital) * 0.2) + series
            entry[str(capital)] = evaluation.target_check(
                combined, base["A100"], base["A80_CASH20"]
            )
        entry["config"] = {k: str(v) for k, v in options.items()}
        stress[name] = entry
        emit(f"压力 {name} 完成")
    return stress


def _industry_concentration(
    path: Path, pit: pd.DataFrame | None, l1_names: dict[str, str] | None
) -> dict[str, Any]:
    """F03-c 补充取证：逐日行业持仓计数（按当日 PIT 一级行业）。"""
    if pit is None or l1_names is None or not path.exists():
        return {"status": "not_available"}
    holdings = pd.read_parquet(path)
    if not len(holdings):
        return {"status": "empty"}
    code_col = next(
        (c for c in holdings.columns if c in ("ts_code", "stock", "code")), None
    )
    date_col = next((c for c in ("trade_date", "date") if c in holdings.columns), None)
    if code_col is None or date_col is None:
        return {"status": "not_available", "columns": [str(c) for c in holdings.columns][:8]}
    holdings[date_col] = pd.to_datetime(holdings[date_col].astype(str)).dt.strftime("%Y%m%d")
    # 口径（用户裁决 A）：行业归属以**信号日**（选股决策日）的 PIT 为准，
    # 与引擎在信号日刷新的 industry_mapping 一致；成交日归属会引入成交日才有的信息。
    if "signal_date" in holdings.columns:
        buy_col = "signal_date"
    elif "buy_date" in holdings.columns:
        buy_col = "buy_date"
    else:
        buy_col = date_col
    holdings[buy_col] = pd.to_datetime(holdings[buy_col].astype(str)).dt.strftime("%Y%m%d")
    day_maps: dict[str, dict[str, str]] = {}
    worst = 0
    over = 0
    for day, group in holdings.groupby(date_col):
        counts: dict[str, int] = {}
        for code, buy_day in zip(group[code_col].astype(str), group[buy_col].astype(str)):
            # 与引擎一致：按**买入日**行业归属，区间到期不改变已持仓的桶
            if buy_day not in day_maps:
                day_maps[buy_day] = b_native.industry_mapping_for_day(pit, l1_names, buy_day)
            name = day_maps[buy_day].get(code, "未知行业")
            counts[name] = counts.get(name, 0) + 1
        peak = max(counts.values())
        worst = max(worst, peak)
        if peak > spec.PORTFOLIO["max_per_industry"]:
            over += 1
    return {
        "status": "ok",
        "max_industry_positions": int(worst),
        "days_over_industry_cap": int(over),
        "days": int(holdings[date_col].nunique()),
    }


def engine_audit(
    campaign_dir: Path,
    pit: pd.DataFrame | None = None,
    l1_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """F03-a/c 取证：两袖首次买入日、逐日槽位上限、调仓事件成交分布、行业集中度。"""
    audit: dict[str, Any] = {}
    for capital in spec.TARGETS["capital_scenarios"]:
        accounts = Path(campaign_dir) / "accounts" / str(capital)
        entry: dict[str, Any] = {}
        for label in ("DEF", "LOWVOL"):
            daily_path = accounts / f"{label}_daily.parquet"
            trade_path = accounts / f"{label}_trades.parquet"
            if not daily_path.exists():
                continue
            daily = pd.read_parquet(daily_path)
            positions = daily["positions"].astype(int)
            buys = pd.Series(dtype=object)
            if trade_path.exists():
                trades = pd.read_parquet(trade_path)
                trades["date"] = pd.to_datetime(trades["date"]).dt.strftime("%Y%m%d")
                buys = trades[trades["action"] == "buy"].groupby("date").size()
            entry[label] = {
                "first_trade_date": str(buys.index.min()) if len(buys) else None,
                "first_buy_date": str(buys.index.min()) if len(buys) else None,
                "rebalance_events": int(len(buys)),
                "orders_per_event_min": int(buys.min()) if len(buys) else None,
                "orders_per_event_max": int(buys.max()) if len(buys) else None,
                "max_positions": int(positions.max()),
                "days_over_20_positions": int((positions > 20).sum()),
                "days_empty": int((positions == 0).sum()),
                "median_positions": float(positions.median()),
                "industry": _industry_concentration(
                    accounts / f"{label}_holdings.parquet", pit, l1_names
                ),
            }
        audit[str(capital)] = entry
    same_start = all(
        entry["DEF"]["first_buy_date"] == entry["LOWVOL"]["first_buy_date"]
        for entry in audit.values()
        if "DEF" in entry and "LOWVOL" in entry
    )
    over_slots = any(
        item["days_over_20_positions"] > 0 for entry in audit.values() for item in entry.values()
    )
    return {
        "per_capital": audit,
        "both_sleeves_same_first_buy": bool(same_start),
        "any_day_over_20_slots": bool(over_slots),
    }


def zero_recovery_check(campaign_dir: Path) -> dict[str, Any]:
    """未知永久缺价压力：检查期末是否有仍按陈旧价估值的持仓。"""
    rows = []
    for capital in spec.TARGETS["capital_scenarios"]:
        path = Path(campaign_dir) / "accounts" / str(capital) / "DEF_daily.parquet"
        frame = pd.read_parquet(path)
        last = frame.iloc[-1]
        rows.append(
            {
                "capital": int(capital),
                "stale_market_value": float(last.get("stale_market_value", 0.0)),
                "missing_price_count": int(last.get("missing_price_count", 0)),
                "total_value": float(last["total_value"]),
            }
        )
    applicable = any(row["missing_price_count"] > 0 for row in rows)
    return {
        "applicable": bool(applicable),
        "rows": rows,
        "note": "无未知永久缺价持仓时不适用（不得伪造已运行）",
    }
