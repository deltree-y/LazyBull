# -*- coding: utf-8 -*-
"""P7 T2.5 原始意图级预检（S1~S5 分字段计数 + 额度/召回/重叠/相关性）。

边界
====
- 本模块只做**估计**：``estimated_intent_coverage=true``，不宣称实际成交、
  恢复覆盖、独立策略分散化或经济结论；真实成交归 T4/T5/T6；
- 意向数量口径 = 目标级等权预算（``袖初资 / top_n``）按 T+1 开盘价整手估算，
  并**显式登记**为估计口径；T5 的配额/借用语义尚未冻结，本模块只报分量
  （``cash_open`` / ``buy_need`` / ``raw_demand`` / 缺口），不引入未冻结的 ρ；
- S1/S2/S3（反向净额）与 S4/S5（同向合并）**分字段计数**，禁止混报。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.v2_p7.t25.params import ARMS

#: 状态机场景（``docs/contracts/netting_freeze.md`` §3）
SCENARIOS: Tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")
REVERSE_SCENARIOS: Tuple[str, ...] = ("S1", "S2", "S3")
MERGE_SCENARIOS: Tuple[str, ...] = ("S4", "S5")

INTENT_NOTE = (
    "estimated_intent_coverage=true：意向股数 = 信号日 T 的冻结 sizing 权重"
    "（half_kelly + 0.15 上限）× 执行日开盘已知 NAV（现金+卖出回款+存量持仓×open）"
    "按 T+1 开盘价整手估计，非实际成交；不宣称覆盖、恢复或分散化结论。"
)

#: 机器可读的数量口径标签（须与 `params.sizing_basis` 一致）
QUANTITY_BASIS = "half_kelly_signal_day_weight_at_t1_open_nav"


def build_intents(
    plans: pd.DataFrame,
    price_lookup: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """把计划帧拆成逐（日, 袖, 股票）的买入与卖出意向。

    Args:
        plans: 两袖计划帧（含 ``execution_date`` / ``due_date`` / ``shares``）
        price_lookup: ``day -> {ts_code: (open, close)}`` 估计价

    Returns:
        ``(buys, sells)``；列 = ``arm, trade_date, ts_code, shares, notional``
    """
    if plans.empty:
        empty = pd.DataFrame(columns=["arm", "trade_date", "ts_code", "shares", "notional"])
        return empty.copy(), empty.copy()
    frame = plans.copy()
    frame["shares"] = frame["shares"].fillna(0).astype(int)
    buy_rows = [
        {
            "arm": str(row.arm),
            "trade_date": str(row.execution_date),
            "ts_code": str(row.ts_code),
            "shares": int(row.shares),
            "notional": float(row.shares) * float(row.buy_price or 0.0),
        }
        for row in frame.itertuples(index=False)
        if int(row.shares) > 0
    ]
    sell_rows = []
    for row in frame.itertuples(index=False):
        due = str(row.due_date or "")
        if not due or int(row.shares) <= 0:
            continue
        pair = (price_lookup(due) or {}).get(str(row.ts_code))
        price = float(pair[0] or 0.0) if pair else 0.0
        if price <= 0 and pair:
            price = float(pair[1] or 0.0)
        sell_rows.append(
            {
                "arm": str(row.arm),
                "trade_date": due,
                "ts_code": str(row.ts_code),
                "shares": int(row.shares),
                "notional": float(row.shares) * price,
            }
        )
    buys = pd.DataFrame(buy_rows, columns=["arm", "trade_date", "ts_code", "shares", "notional"])
    sells = pd.DataFrame(sell_rows, columns=["arm", "trade_date", "ts_code", "shares", "notional"])
    return buys, sells


def classify_intents(
    buys: pd.DataFrame, sells: pd.DataFrame, arms: Sequence[str] = ARMS
) -> List[Dict[str, Any]]:
    """逐（执行日, 股票）按状态机表归类（S1~S5；单袖动作单独登记）。

    ``arms`` 只决定配对的两个袖名（默认冻结的 ``v20``/``v10``）；只读扫描可传入候选名。
    """
    if buys.empty and sells.empty:
        return []
    left, right = str(arms[0]), str(arms[1])
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for side, frame in (("buy", buys), ("sell", sells)):
        for row in frame.itertuples(index=False):
            key = (str(row.trade_date), str(row.ts_code))
            slot = records.setdefault(key, {"trade_date": key[0], "ts_code": key[1]})
            slot[f"{row.arm}_{side}_shares"] = int(slot.get(f"{row.arm}_{side}_shares", 0)) + int(
                row.shares
            )
            slot[f"{row.arm}_{side}_notional"] = float(
                slot.get(f"{row.arm}_{side}_notional", 0.0)
            ) + float(row.notional)
    results: List[Dict[str, Any]] = []
    for key in sorted(records):
        slot = records[key]
        v20_buy = int(slot.get(f"{left}_buy_shares", 0))
        v20_sell = int(slot.get(f"{left}_sell_shares", 0))
        v10_buy = int(slot.get(f"{right}_buy_shares", 0))
        v10_sell = int(slot.get(f"{right}_sell_shares", 0))
        is_reverse = bool((v20_sell and v10_buy) or (v10_sell and v20_buy))
        reverse_sell = (v20_sell + v10_sell) if is_reverse else 0
        reverse_buy = (v20_buy + v10_buy) if is_reverse else 0
        scenario = _classify(v20_buy, v20_sell, v10_buy, v10_sell)
        results.append(
            {
                "trade_date": key[0],
                "ts_code": key[1],
                "scenario": scenario,
                "v20_buy_shares": v20_buy,
                "v20_sell_shares": v20_sell,
                "v10_buy_shares": v10_buy,
                "v10_sell_shares": v10_sell,
                "reverse_sell_shares": reverse_sell,
                "reverse_buy_shares": reverse_buy,
                "offset_shares": min(reverse_sell, reverse_buy),
                "net_shares": abs(reverse_sell - reverse_buy),
                "notional": float(
                    sum(
                        float(slot.get(f"{arm}_{side}_notional", 0.0))
                        for arm in arms
                        for side in ("buy", "sell")
                    )
                ),
            }
        )
    return results


def _classify(v20_buy: int, v20_sell: int, v10_buy: int, v10_sell: int) -> str:
    """状态机归类：S1 等量反向、S2 卖大于买、S3 买大于卖、S4/S5 同向合并、S6 单袖。"""
    reverse = (v20_sell and v10_buy) or (v10_sell and v20_buy)
    if reverse:
        sell = v20_sell + v10_sell
        buy = v20_buy + v10_buy
        if sell == buy:
            return "S1"
        return "S2" if sell > buy else "S3"
    if v20_buy and v10_buy:
        return "S4"
    if v20_sell and v10_sell:
        return "S5"
    if v20_buy or v10_buy or v20_sell or v10_sell:
        return "S6"
    return "S6"


def summarize_scenarios(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """按场景分字段计数（事件数 / 股数 / 名义额 / 抵消 / 净额）。"""
    summary = {
        scenario: {"events": 0, "shares": 0, "notional": 0.0, "offset_shares": 0, "net_shares": 0}
        for scenario in SCENARIOS + ("S6",)
    }
    by_day: Dict[str, Dict[str, int]] = {}
    for item in records:
        scenario = str(item["scenario"])
        bucket = summary[scenario]
        bucket["events"] += 1
        bucket["shares"] += int(
            item["v20_buy_shares"]
            + item["v20_sell_shares"]
            + item["v10_buy_shares"]
            + item["v10_sell_shares"]
        )
        bucket["notional"] += float(item["notional"])
        bucket["offset_shares"] += int(item["offset_shares"])
        bucket["net_shares"] += int(item["net_shares"])
        day_bucket = by_day.setdefault(str(item["trade_date"]), {})
        day_bucket[scenario] = day_bucket.get(scenario, 0) + 1
    for bucket in summary.values():
        bucket["notional"] = round(bucket["notional"], 2)
    reverse_events = sum(summary[scenario]["events"] for scenario in REVERSE_SCENARIOS)
    merge_events = sum(summary[scenario]["events"] for scenario in MERGE_SCENARIOS)
    return {
        "scenarios": summary,
        "days": len(by_day),
        "reverse_events": reverse_events,
        "reverse_shares": sum(summary[scenario]["offset_shares"] for scenario in REVERSE_SCENARIOS),
        "merge_events": merge_events,
        "merge_shares": sum(summary[scenario]["shares"] for scenario in MERGE_SCENARIOS),
        "day_scenario_counts": dict(sorted(by_day.items())),
    }


def funding_diagnostics(
    daily_frames: Mapping[str, pd.DataFrame],
    calendar: Sequence[str],
) -> Dict[str, Any]:
    """额度缺口 / 额度不紧绑定日 / 召回需求（1 日借还台账估计）。

    口径：``funding_gap = max(0, buy_need - cash_open)``；缺口先由当日到期卖出回款
    覆盖（引擎先卖后买），余量才视为需要外部额度（借用方向）；被借方在有借出余额
    的次日自身出现余量缺口即登记一次**召回需求**。ρ 未在 T2.5 冻结，故不引入。
    """
    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        frame = daily_frames.get(arm)
        if frame is None or frame.empty:
            per_arm[arm] = {"days": 0}
            continue
        buys = frame[frame["buy_need"] > 0]
        covered = buys[buys["funding_gap"] == 0]
        residual = (buys["funding_gap"] - buys["sell_proceeds"]).clip(lower=0.0)
        per_arm[arm] = {
            "days": int(len(frame)),
            "buy_days": int(len(buys)),
            "quota_not_binding_days": int(len(covered)),
            "funding_gap_days": int((buys["funding_gap"] > 0).sum()),
            "funding_gap_total": round(float(buys["funding_gap"].sum()), 2),
            "funding_gap_max": round(float(buys["funding_gap"].max()), 2) if len(buys) else 0.0,
            "residual_gap_days": int((residual > 0).sum()),
            "residual_gap_total": round(float(residual.sum()), 2),
            "raw_demand_mean": round(float(frame["raw_demand"].mean()), 2),
        }

    recall = _recall_ledger(daily_frames, calendar)
    return {
        "per_arm": per_arm,
        "recall": recall,
        "quota_rho_note": "ρ 未在 T2.5 冻结；此处只报 G/MV/D 分量与现金口径缺口，不构造配额。",
    }


def _recall_ledger(
    daily_frames: Mapping[str, pd.DataFrame], calendar: Sequence[str]
) -> Dict[str, Any]:
    """简化 1 日借还台账：估计双向缺口与召回需求（不是 T5 的借用会计）。"""
    frames = {arm: daily_frames[arm].set_index("trade_date") for arm in ARMS}
    lent: Dict[str, float] = {arm: 0.0 for arm in ARMS}
    borrow_events = 0
    borrow_amount = 0.0
    recall_events = 0
    recall_amount = 0.0
    days = 0
    for day in calendar:
        if any(day not in frames[arm].index for arm in ARMS):
            continue
        days += 1
        gap: Dict[str, float] = {}
        idle: Dict[str, float] = {}
        for arm in ARMS:
            row = frames[arm].loc[day]
            buy_need = float(row["buy_need"])
            gap[arm] = (
                max(0.0, float(row["funding_gap"]) - float(row["sell_proceeds"]))
                if buy_need > 0
                else 0.0
            )
            idle[arm] = max(0.0, float(row["cash_open"]) - buy_need)
        # ① 昨日借出余额 + 今日自身缺口 ⇒ 召回需求（T+1 归还口径）
        for arm in ARMS:
            if gap[arm] > 0 and lent[arm] > 0:
                recall_events += 1
                recall_amount += min(gap[arm], lent[arm])
        # ② 今日新借出（1 日寿命，次日必须归还）
        new_lent: Dict[str, float] = {arm: 0.0 for arm in ARMS}
        for arm in ARMS:
            other = next(item for item in ARMS if item != arm)
            if gap[arm] > 0 and idle[other] > 0:
                amount = min(gap[arm], idle[other])
                borrow_events += 1
                borrow_amount += amount
                new_lent[other] += amount
        lent = new_lent
    return {
        "days": days,
        "borrow_events": borrow_events,
        "borrow_amount": round(borrow_amount, 2),
        "recall_events": recall_events,
        "recall_amount": round(recall_amount, 2),
        "note": "1 日借还估计（T+1 归还），不代表 T5 的真实借用/召回会计。",
    }


def sleeve_equity_series(
    holdings: pd.DataFrame,
    daily_frames: Mapping[str, pd.DataFrame],
    closes_lookup: Any,
    calendar: Sequence[str],
) -> Dict[str, List[float]]:
    """逐袖估计净值序列 = 日终现金 + Σ 计划股数 × 当日收盘（估计口径）。"""
    shares_by_day: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}
    for arm in ARMS:
        frame = holdings[holdings["arm"] == arm]
        grouped: Dict[str, List[Tuple[str, int]]] = {}
        for row in frame.itertuples(index=False):
            grouped.setdefault(str(row.trade_date), []).append((str(row.ts_code), int(row.shares)))
        shares_by_day[arm] = grouped
    series: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    for day in calendar:
        closes = closes_lookup(day) or {}
        for arm in ARMS:
            frame = daily_frames[arm]
            match = frame[frame["trade_date"] == day]
            if match.empty:
                series[arm].append(float("nan"))
                continue
            row = match.iloc[0]
            cash_close = (
                float(row["cash_open"]) - float(row["buy_need"]) + float(row["sell_proceeds"])
            )
            market_value = 0.0
            for code, shares in shares_by_day[arm].get(day, []):
                price = float(closes.get(code, 0.0) or 0.0)
                market_value += shares * price
            series[arm].append(cash_close + market_value)
    return series


def sleeve_correlation(series: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    """两袖逐日收益相关（Pearson / Spearman），仅作同源副本诊断。"""
    left = np.asarray(series.get("v20", []), dtype=np.float64)
    right = np.asarray(series.get("v10", []), dtype=np.float64)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return {"days": 0, "pearson": None, "spearman": None}
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return {"days": int(mask.sum()), "pearson": None, "spearman": None}
    left_ret = np.diff(left[mask]) / np.where(left[mask][:-1] == 0, np.nan, left[mask][:-1])
    right_ret = np.diff(right[mask]) / np.where(right[mask][:-1] == 0, np.nan, right[mask][:-1])
    valid = np.isfinite(left_ret) & np.isfinite(right_ret)
    if int(valid.sum()) < 3:
        return {"days": int(valid.sum()), "pearson": None, "spearman": None}
    left_ret = left_ret[valid]
    right_ret = right_ret[valid]
    pearson = float(np.corrcoef(left_ret, right_ret)[0, 1])
    left_rank = pd.Series(left_ret).rank().to_numpy()
    right_rank = pd.Series(right_ret).rank().to_numpy()
    spearman = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {
        "days": int(valid.sum()),
        "pearson": pearson,
        "spearman": spearman,
        "v20_daily_return_std": float(np.std(left_ret, ddof=1)),
        "v10_daily_return_std": float(np.std(right_ret, ddof=1)),
    }


def intent_consistency(
    plans: pd.DataFrame,
    buys: pd.DataFrame,
    sells: pd.DataFrame,
    holdings: pd.DataFrame,
) -> Dict[str, Any]:
    """意向与持仓集合口径一致性自检（同一持仓登记账派生）。"""
    plan_shares = int(plans["shares"].fillna(0).sum()) if not plans.empty else 0
    buy_shares = int(buys["shares"].sum()) if not buys.empty else 0
    sell_shares = int(sells["shares"].sum()) if not sells.empty else 0
    holdings_shares = int(holdings["shares"].sum()) if not holdings.empty else 0
    return {
        "plan_shares": plan_shares,
        "buy_intent_shares": buy_shares,
        "sell_intent_shares": sell_shares,
        "holding_day_shares": holdings_shares,
        "buy_equals_plans": bool(buy_shares == plan_shares),
        "sell_not_above_plans": bool(sell_shares <= plan_shares),
    }


def natural_observations(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """自然必达项观测登记（为 0 即机械栏 ``incomplete/not_observed``）。"""
    counts = {scenario: 0 for scenario in SCENARIOS}
    for item in records:
        scenario = str(item["scenario"])
        if scenario in counts:
            counts[scenario] += 1
    missing = [scenario for scenario, value in counts.items() if value == 0]
    return {
        "counts": counts,
        "missing_scenarios": missing,
        "status": "observed" if not missing else "incomplete/not_observed",
        "note": "自然意图为 0 项不得换相位/换窗口/调配额补足。",
    }


def build_precheck(
    plans: pd.DataFrame,
    holdings: pd.DataFrame,
    daily_frames: Mapping[str, pd.DataFrame],
    calendar: Sequence[str],
    price_lookup: Any,
    closes_lookup: Any,
) -> Dict[str, Any]:
    """组装原始意图级预检（S1~S5 + 缺口/召回/重叠/相关性）。"""
    from scripts.v2_p7.t25.target_layer import holdings_overlap, target_overlap

    buys, sells = build_intents(plans, price_lookup)
    records = classify_intents(buys, sells)
    scenarios = summarize_scenarios(records)
    equity = sleeve_equity_series(holdings, daily_frames, closes_lookup, calendar)
    payload = {
        "schema": "p7_t25_intent_precheck/v1",
        "estimated_intent_coverage": True,
        "note": INTENT_NOTE,
        "quantity_basis": QUANTITY_BASIS,
        "intent_days": int(len({str(item["trade_date"]) for item in records})),
        "scenarios": scenarios,
        "natural_observations": natural_observations(records),
        "funding": funding_diagnostics(daily_frames, calendar),
        "overlap": {
            "targets": target_overlap(plans),
            "holdings": holdings_overlap(holdings),
        },
        "sleeve_correlation": sleeve_correlation(equity),
        "consistency": intent_consistency(plans, buys, sells, holdings),
    }
    return payload
