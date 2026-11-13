# -*- coding: utf-8 -*-
"""P7 T6 绩效与风险量：收益 / 回撤 / 稳定性 / 链式指标 / 费用与换手。

唯一入口
========
- **链式指标**：``src.lazybull.v2.common.chain_metrics.calculate_chain_metrics``
  （``total_return`` / ``cagr`` / ``max_drawdown`` / ``volatility`` / ``sharpe``）；
- **自然日复合年化**：``src.lazybull.v2.hosts.paper.performance.calculate_annualized_return``
  （返回百分数；**不复制公式**）；
- **NAV**：``calculate_nav``（已由 ``daily`` 逐日落地）。

回撤事件表（描述量）
==================
``drawdown_episodes`` 只做**事件刻画**（起止日、深度、持续/恢复期、区间内费用与成交额），
不作判据；**MaxDD 判据数值只取** ``calculate_chain_metrics``，并以「事件表极值 ==
chain_metrics.max_drawdown」在 G4 互证。

未复用 ``scripts/ana/analyze_drawdown.py`` 的原因（登记）
=====================================================
该函数把段阈值硬编码为 ``-10%``、要求 WF 的 ``split_index`` 并耦合绘图/报表，
无法给出 Top-10 事件表与区间费用分解；本模块只补事件刻画，不复制任何收益/回撤公式。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.t6.params import T6Params
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics
from src.lazybull.v2.hosts.paper.performance import calculate_annualized_return

METRICS_SCHEMA = "p7_t6_metrics/v1"

#: 回撤 Top-N
TOP_EPISODES = 10

#: 指标字段（chain_metrics 输出）
CHAIN_FIELDS = ("total_return", "cagr", "max_drawdown", "volatility", "sharpe", "trading_days")


def build_metrics(
    *,
    params: T6Params,
    frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> Dict[str, Any]:
    """两臂指标 + 配对差 + 分年/分月序列 + 费用换手归因。"""
    arms = {name: _arm_metrics(params, frames[name]) for name in ("vvh", "vvs")}
    paired = _paired(arms, physical_capital=float(params.physical_capital))
    return {
        "schema": METRICS_SCHEMA,
        "definitions": {
            "nav": "NAV_t = Σ_i(cash+MV)_i,t / PC（calculate_nav）",
            "total_return": "链式净值首末比 − 1（calculate_chain_metrics）",
            "annualized_natural_days_pct": (
                "自然日复合年化（paper/performance.calculate_annualized_return，百分数）"
            ),
            "cagr": "252 交易日年化（calculate_chain_metrics.cagr；与自然日口径分列，不混用）",
            "max_drawdown": "链式净值最大回撤（calculate_chain_metrics.max_drawdown，负值）",
            "volatility": "日收益年化波动（calculate_chain_metrics.volatility）",
            "turnover": "双边换手 = Σ 市场腿成交名义 / PC（内部抵消不计）",
            "fee_ratio": "累计费用 / PC",
            "paired": "Δ = VV-S − VV-H",
            "trading_days": (
                "链式指标交易日数由 calculate_chain_metrics 的 same_split 掩码决定"
                "（本窗口 1,731 = 1,744 个日收益 − 13 个折界日）；"
                "折界日不重置净值，属上游口径，已在 docs/reports/p7_t6_closure_20261113.md 登记"
            ),
            "monthly_annual": (
                "分月/分年序列是**收益率**（描述量，不可相加、非资金贡献）；"
                "C5 的集中度按 |月度收益率差| 计，不与 ΔTR 直接可比"
            ),
        },
        "arms": arms,
        "paired": paired,
        "triple": {
            "total_return": _pair_item(arms, "total_return"),
            "max_drawdown": _pair_item(arms, "max_drawdown"),
            "volatility": _pair_item(arms, "volatility"),
        },
        "by_year_paired": _by_year_paired(arms),
        "monthly_paired": _monthly_paired(arms),
    }


# ────────────────────────── 单臂 ──────────────────────────


def _arm_metrics(
    params: T6Params, frame: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    daily = frame["daily"]
    nav = pd.Series([float(row["nav"]) for row in daily], name="nav")
    dates = [str(row["trade_date"]) for row in daily]
    split = pd.Series([int(row["split_index"]) for row in daily])
    chain = calculate_chain_metrics(
        pd.DataFrame({"trade_date": dates, "nav": nav, "split_index": split})
    )
    episodes = drawdown_episodes(
        nav=nav,
        dates=dates,
        daily=daily,
        physical_capital=float(params.physical_capital),
    )
    monthly = monthly_returns(nav, dates)
    annual = annual_returns(nav, dates)
    fees = _fee_metrics(params, frame["trades"], daily)
    return {
        "days": len(daily),
        "window": [dates[0], dates[-1]],
        "nav_start": float(nav.iloc[0]),
        "nav_end": float(nav.iloc[-1]),
        "total_return": float(chain["total_return"]),
        "annualized_natural_days_pct": calculate_annualized_return(
            float(params.physical_capital),
            float(daily[-1]["total_value"]),
            dates[-1],
            dates[0],
        ),
        "chain_metrics": {key: chain[key] for key in CHAIN_FIELDS},
        "drawdown": {
            "max_drawdown": float(chain["max_drawdown"]),
            "episodes": episodes[:TOP_EPISODES],
            "episode_count": len(episodes),
            "episode_series_max": (
                float(min(item["depth"] for item in episodes)) if episodes else 0.0
            ),
            "unrecovered_at_end": sum(1 for item in episodes if item["in_progress"]),
        },
        "monthly": _distribution(monthly, label="month"),
        "annual": _distribution(annual, label="year"),
        "fees": fees,
        "turnover": _turnover_metrics(params, frame["trades"], daily),
        "credit": _credit_metrics(daily),
        "risk": _risk_counts(daily),
    }


def _fee_metrics(
    params: T6Params,
    trades: Sequence[Mapping[str, Any]],
    daily: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    pc = float(params.physical_capital)
    total = float(sum(float(trade.get("fee") or 0.0) for trade in trades))
    # 期末资产已扣除累计费用 ⇒ 净利润 = 前 − 后；毛利润须把费用加回（C6-04）
    net_profit = float(daily[-1]["total_value"]) - pc
    gross_profit = float(net_profit + total)
    by_year: Dict[str, float] = {}
    for row in daily:
        year = str(row["trade_date"])[:4]
        by_year[year] = by_year.get(year, 0.0) + float(row["fee_day"])
    return {
        "total": total,
        "over_pc": total / pc,
        "net_profit": net_profit,
        "gross_profit": gross_profit,
        # 费用占毛利比（§4.3 口径）：分母为正才算；净利非正不影响该比例
        "over_gross_profit": (total / gross_profit) if gross_profit > 0.0 else None,
        "over_net_profit": (total / net_profit) if net_profit > 0.0 else None,
        "by_year": {key: float(value) for key, value in sorted(by_year.items())},
        "trades": int(len(trades)),
        "buy_fee": float(
            sum(float(t.get("fee") or 0.0) for t in trades if t.get("kind") == "market_buy")
        ),
        "sell_fee": float(
            sum(float(t.get("fee") or 0.0) for t in trades if t.get("kind") == "market_sell")
        ),
    }


def _turnover_metrics(
    params: T6Params,
    trades: Sequence[Mapping[str, Any]],
    daily: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    pc = float(params.physical_capital)
    market = float(sum(float(trade.get("market_notional") or 0.0) for trade in trades))
    offset = float(
        sum(
            float(int(trade.get("shares") or 0) * float(trade.get("price") or 0.0))
            for trade in trades
            if str(trade.get("kind")) not in ("market_buy", "market_sell")
        )
    )
    by_year: Dict[str, float] = {}
    for row in daily:
        year = str(row["trade_date"])[:4]
        by_year[year] = by_year.get(year, 0.0) + float(row["market_notional"])
    return {
        "market_notional": market,
        "offset_notional": offset,
        "turnover": market / pc,
        "offset_turnover": offset / pc,
        "by_year": {key: float(value / pc) for key, value in sorted(by_year.items())},
        "buy_notional": float(
            sum(
                float(t.get("market_notional") or 0.0)
                for t in trades
                if t.get("kind") == "market_buy"
            )
        ),
        "sell_notional": float(
            sum(
                float(t.get("market_notional") or 0.0)
                for t in trades
                if t.get("kind") == "market_sell"
            )
        ),
    }


def _credit_metrics(daily: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    borrow_days = [row for row in daily if float(row["borrow_outstanding"]) > 0.0]
    grant_days = [row for row in daily if float(row["granted_total"]) > 0.0]
    return {
        "granted_total": float(sum(float(row["granted_total"]) for row in daily)),
        "returned_total": float(sum(float(row["returned_total"]) for row in daily)),
        "deadlocked_total": float(sum(float(row["deadlocked_total"]) for row in daily)),
        "late_recovery_total": float(sum(float(row["late_recovery_total"]) for row in daily)),
        "grant_days": len(grant_days),
        "borrow_days": len(borrow_days),
        "max_borrow_outstanding": float(
            max((float(row["borrow_outstanding"]) for row in daily), default=0.0)
        ),
        "max_lent_outstanding": float(
            max((float(row["lent_outstanding"]) for row in daily), default=0.0)
        ),
        "recall_days": sum(1 for row in daily if int(row["recall_due"]) > 0),
        "right_censored_days": sum(1 for row in daily if int(row["right_censored"]) > 0),
    }


def _risk_counts(daily: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "min_credit_sum": float(min(float(row["credit_sum"]) for row in daily)),
        "max_credit_sum": float(max(float(row["credit_sum"]) for row in daily)),
        "min_gate": float(
            min(
                (float(row["gate_min"]) for row in daily if row["gate_min"] is not None),
                default=0.0,
            )
        ),
        "identity_failure_days": sum(1 for row in daily if not bool(row["identity_ok"])),
        "cash_identity_max_abs_diff": float(
            max(abs(float(row["cash_sum"]) - float(row["physical_cash"])) for row in daily)
        ),
    }


# ────────────────────────── 回撤事件 ──────────────────────────


def drawdown_episodes(
    *,
    nav: pd.Series,
    dates: Sequence[str],
    daily: Sequence[Mapping[str, Any]],
    physical_capital: float,
) -> List[Dict[str, Any]]:
    """回撤事件表（按深度升序）：峰值 → 谷底 → 恢复（未恢复标 ``in_progress``）。"""
    values = nav.to_numpy(dtype=float)
    peak = np.maximum.accumulate(values)
    drawdown = values / peak - 1.0
    fee_cum = np.array([float(row["fee_cum"]) for row in daily], dtype=float)
    notional = np.array([float(row["market_notional"]) for row in daily], dtype=float)
    pc = float(physical_capital)
    episodes: List[Dict[str, Any]] = []
    start: Optional[int] = None
    for pos, value in enumerate(drawdown):
        if value < 0.0 and start is None:
            # 峰值 = 回撤开始前的最后一个 dd==0 的位置
            start = pos - 1 if pos > 0 else 0
            while start > 0 and drawdown[start] < 0.0:
                start -= 1
        if value == 0.0 and start is not None:
            episodes.append(_episode(start, pos, values, drawdown, dates, fee_cum, notional, pc))
            start = None
    if start is not None:
        episodes.append(
            _episode(
                start,
                len(values) - 1,
                values,
                drawdown,
                dates,
                fee_cum,
                notional,
                pc,
                open_end=True,
            )
        )
    episodes.sort(key=lambda item: item["depth"])
    return episodes


def _episode(
    start: int,
    end: int,
    values: np.ndarray,
    drawdown: np.ndarray,
    dates: Sequence[str],
    fee_cum: np.ndarray,
    notional: np.ndarray,
    physical_capital: float,
    *,
    open_end: bool = False,
) -> Dict[str, Any]:
    window = drawdown[start : end + 1]
    trough = start + int(np.argmin(window))
    fee_to_trough = float(fee_cum[trough] - (fee_cum[start] if start > 0 else 0.0))
    fee_to_end = float(fee_cum[end] - (fee_cum[start] if start > 0 else 0.0))
    notional_to_trough = float(notional[start : trough + 1].sum())
    notional_to_end = float(notional[start : end + 1].sum())
    depth = float(drawdown[trough])
    peak_value = float(values[start]) * physical_capital
    trough_value = float(values[trough]) * physical_capital
    loss_amount = float(peak_value - trough_value)
    return {
        "start_date": str(dates[start]),
        "trough_date": str(dates[trough]),
        "end_date": "" if open_end else str(dates[end]),
        "depth": depth,
        "peak_nav": float(values[start]),
        "trough_nav": float(values[trough]),
        "drawdown_days": int(trough - start),
        "recovery_days": int(-1 if open_end else end - trough),
        "total_days": int(end - start),
        "in_progress": bool(open_end),
        "fees_to_trough": fee_to_trough,
        "fees_to_end": fee_to_end,
        # 费用/初资（独立字段；C6-03 后不再与 depth 混用）
        "fee_pp_to_trough": fee_to_trough / physical_capital,
        # 峰谷资产损失金额与费用占**该损失**的比例（同一基数，C6-03）
        "loss_amount": loss_amount,
        "fee_share_of_loss": (float(fee_to_trough / loss_amount) if loss_amount > 0.0 else None),
        "notional_to_trough": notional_to_trough,
        "notional_to_end": notional_to_end,
    }


# ────────────────────────── 分月 / 分年 ──────────────────────────


def monthly_returns(nav: pd.Series, dates: Sequence[str]) -> pd.Series:
    """月末净值序列的月度收益（首月以期内首个净值为基期）。"""
    frame = pd.DataFrame({"date": list(dates), "nav": nav.to_numpy(dtype=float)})
    frame["month"] = frame["date"].str[:6]
    last = frame.groupby("month")["nav"].last()
    returns = last.pct_change(fill_method=None)
    if len(last) > 0:
        first_month = last.index[0]
        base = float(frame.loc[frame["month"] == first_month, "nav"].iloc[0])
        returns.loc[first_month] = float(last.loc[first_month]) / base - 1.0
    return returns.dropna()


def annual_returns(nav: pd.Series, dates: Sequence[str]) -> pd.Series:
    """年末净值序列的年度收益（首年以期内首个净值为基期）。"""
    frame = pd.DataFrame({"date": list(dates), "nav": nav.to_numpy(dtype=float)})
    frame["year"] = frame["date"].str[:4]
    last = frame.groupby("year")["nav"].last()
    returns = last.pct_change(fill_method=None)
    if len(last) > 0:
        first_year = last.index[0]
        base = float(frame.loc[frame["year"] == first_year, "nav"].iloc[0])
        returns.loc[first_year] = float(last.loc[first_year]) / base - 1.0
    return returns.dropna()


def _distribution(series: pd.Series, *, label: str) -> Dict[str, Any]:
    values = series.astype(float)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()) if len(values) else None,
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "skew": float(values.skew()) if len(values) > 2 else None,
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
        "worst": (
            {"period": str(values.idxmin()), "value": float(values.min())} if len(values) else None
        ),
        "best": (
            {"period": str(values.idxmax()), "value": float(values.max())} if len(values) else None
        ),
        "series": {str(key): float(value) for key, value in values.items()},
        "label": label,
    }


# ────────────────────────── 配对 ──────────────────────────


def _metric_value(arm_metrics: Mapping[str, Any], key: str) -> float:
    """按指标名取单臂值（顶层 / ``chain_metrics`` / ``drawdown`` 三处统一入口）。"""
    if key in arm_metrics:
        return float(arm_metrics[key])
    for section in ("chain_metrics", "drawdown"):
        nested = arm_metrics.get(section) or {}
        if key in nested:
            return float(nested[key])
    raise KeyError(f"指标 {key!r} 不在单臂载荷内（顶层/chain_metrics/drawdown）")


def _pair_item(arms: Mapping[str, Mapping[str, Any]], key: str) -> Dict[str, Any]:
    control = _metric_value(arms["vvh"], key)
    treatment = _metric_value(arms["vvs"], key)
    return {
        "control": control,
        "treatment": treatment,
        "delta": treatment - control,
        "relative_delta": ((treatment - control) / abs(control)) if control else None,
    }


def _paired(arms: Mapping[str, Mapping[str, Any]], *, physical_capital: float) -> Dict[str, Any]:
    pc = float(physical_capital)
    paired = {
        key: _pair_item(arms, key)
        for key in ("total_return", "max_drawdown", "volatility", "nav_end")
    }
    paired["cagr"] = {
        "control": float(arms["vvh"]["chain_metrics"]["cagr"]),
        "treatment": float(arms["vvs"]["chain_metrics"]["cagr"]),
    }
    paired["cagr"]["delta"] = paired["cagr"]["treatment"] - paired["cagr"]["control"]
    paired["sharpe"] = {
        "control": float(arms["vvh"]["chain_metrics"]["sharpe"] or 0.0),
        "treatment": float(arms["vvs"]["chain_metrics"]["sharpe"] or 0.0),
    }
    paired["sharpe"]["delta"] = paired["sharpe"]["treatment"] - paired["sharpe"]["control"]
    fee_delta = float(arms["vvs"]["fees"]["total"]) - float(arms["vvh"]["fees"]["total"])
    turnover_delta = float(arms["vvs"]["turnover"]["turnover"]) - float(
        arms["vvh"]["turnover"]["turnover"]
    )
    paired["fees"] = {
        "control": float(arms["vvh"]["fees"]["total"]),
        "treatment": float(arms["vvs"]["fees"]["total"]),
        "delta": fee_delta,
    }
    paired["turnover"] = {
        "control": float(arms["vvh"]["turnover"]["turnover"]),
        "treatment": float(arms["vvs"]["turnover"]["turnover"]),
        "delta": turnover_delta,
    }
    fee_over_pc = fee_delta / pc
    paired["attribution"] = {
        "physical_capital": pc,
        "delta_tr": paired["total_return"]["delta"],
        "delta_fee_over_pc": fee_over_pc,
        "delta_tr_fee": -fee_over_pc,
        "delta_tr_gross": paired["total_return"]["delta"] + fee_over_pc,
        "delta_turnover": turnover_delta,
        "same_sign_turnover_fee": bool(turnover_delta * fee_delta >= 0.0),
    }
    return paired


def _by_year_paired(arms: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    control = arms["vvh"]["annual"]["series"]
    treatment = arms["vvs"]["annual"]["series"]
    for year in sorted(set(control) | set(treatment)):
        left = float(control.get(year, 0.0))
        right = float(treatment.get(year, 0.0))
        rows.append(
            {
                "year": int(year),
                "control": left,
                "treatment": right,
                "delta": right - left,
                "fee_control": float(arms["vvh"]["fees"]["by_year"].get(year, 0.0)),
                "fee_treatment": float(arms["vvs"]["fees"]["by_year"].get(year, 0.0)),
                "turnover_control": float(arms["vvh"]["turnover"]["by_year"].get(year, 0.0)),
                "turnover_treatment": float(arms["vvs"]["turnover"]["by_year"].get(year, 0.0)),
            }
        )
    return rows


def _monthly_paired(arms: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    control = arms["vvh"]["monthly"]["series"]
    treatment = arms["vvs"]["monthly"]["series"]
    rows: List[Dict[str, Any]] = []
    for month in sorted(set(control) | set(treatment)):
        left = float(control.get(month, 0.0))
        right = float(treatment.get(month, 0.0))
        rows.append(
            {"month": str(month), "control": left, "treatment": right, "delta": right - left}
        )
    return rows
