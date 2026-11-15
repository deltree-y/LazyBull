# -*- coding: utf-8 -*-
"""型 II（风险调整改善）判据组件（F29 草案 v5）。

主门 ``ΔSharpe`` 用配对 10 日循环块自举的**下尾** ``Q_{α_c/2}``；护栏 ``ΔMDD`` 同为下尾；
资格门为完整冻结窗平均仓位 ≥10% 且市场最差 20% 交易日平均仓位 ≥20%；
约束为非危机期净收益（相对 A 腿 + 现金腿）≥0。E1/Q1/E2/E4-4 不适用。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.f2 import (
    MARKET_COVERAGE_MIN,
    MIN_MARKET_DAYS,
    MIN_NONCRISIS_DAYS,
    POSITION_MIN,
    WORST_DAY_MIN_DAYS,
    WORST_DAY_POSITION_MIN,
    WORST_DAY_SHARE,
)
from scripts.v2_p7.f2.formal import block_bootstrap_delta_maxdd, block_bootstrap_sharpe_delta

#: 型 II 不适用（豁免）的组件
NOT_APPLICABLE = ("E1", "E2", "Q1", "E4_4_utilization")


def _component(
    estimable: bool, passed: Optional[bool], *, calibrated: bool = False, **extra: Any
) -> Dict[str, Any]:
    """组件构造器；新增统计组件默认 ``calibrated=False``（未校准前只记 frozen）。"""
    payload: Dict[str, Any] = {
        "estimable": bool(estimable),
        "pass": passed,
        "calibrated": bool(calibrated),
    }
    payload.update(extra)
    return payload


def delta_sharpe_gate(
    a_returns: np.ndarray, combined_returns: np.ndarray, *, alpha_c: float
) -> Dict[str, Any]:
    """主门：``ΔSharpe`` 下尾 > 0。"""
    result = block_bootstrap_sharpe_delta(a_returns, combined_returns, tail=alpha_c / 2.0)
    result.pop("distribution", None)
    if result.get("lower") is None:
        return _component(False, None, reason=result.get("reason"), **result)
    return _component(True, bool(float(result["lower"]) > 0.0), **result)


def delta_maxdd_gate(
    a_returns: np.ndarray, combined_returns: np.ndarray, *, alpha_c: float
) -> Dict[str, Any]:
    """护栏：``ΔMDD`` 下尾 ≥ 0（新增护栏，不是 ES3 的 −1/−3 分段）。"""
    result = block_bootstrap_delta_maxdd(a_returns, combined_returns, tail=alpha_c / 2.0)
    if result.get("lower") is None:
        return _component(False, None, reason="insufficient_sample", **result)
    result.pop("distribution", None)
    return _component(True, bool(float(result["lower"]) >= 0.0), **result)


def _window_days(calendar: Sequence[str], window: Mapping[str, Any]) -> List[str]:
    """完整冻结窗日期集来自**冻结市场日历**，不由候选记录反推（C03）。"""
    start = str(window.get("start_date", "")).replace("-", "")
    end = str(window.get("end_date", "")).replace("-", "")
    return sorted(d for d in calendar if start <= d <= end)


def _single_window_position(
    aligned: pd.DataFrame,
    lagged_pos: pd.Series,
    calendar: Sequence[str],
    original: set,
    window: Mapping[str, Any],
    market_returns: pd.Series,
    *,
    min_position: float,
    worst_share: float,
    worst_min: float,
) -> Dict[str, Any]:
    """单窗：完整日历轴上判定缺记录/无效读数，仓位取**前一交易日**收盘。"""
    idx = _window_days(calendar, window)
    if not idx:
        return {"estimable": False, "pass": None, "reason": "empty_window"}
    # 已知空仓须有明确零持仓记录；缺记录是缺证据，不得推断为空仓（V01）
    missing = [day for day in idx if day not in original]
    if missing:
        return {
            "estimable": False,
            "pass": None,
            "reason": "missing_position_rows",
            "missing_days": len(missing),
        }
    sub = aligned.reindex(idx)
    nav = pd.to_numeric(sub.get("nav"), errors="coerce")
    mv = pd.to_numeric(sub.get("market_value"), errors="coerce")
    bad = ~np.isfinite(nav) | ~np.isfinite(mv) | (nav <= 0.0) | (mv < 0.0)
    if bool(bad.any()):
        return {
            "estimable": False,
            "pass": None,
            "reason": "invalid_position_row",
            "invalid_days": int(bad.sum()),
        }
    # 窗内滞后的有限性（Inf 不是 NaN，不能只查 isna）
    # 窗前一日：先判**存在性**，再判**有效性**（存在但值非法 ≠ 缺失）
    pos_in_cal = list(calendar).index(idx[0])
    if pos_in_cal == 0:
        return {"estimable": False, "pass": None, "reason": "missing_prior_position"}
    prev_day = calendar[pos_in_cal - 1]
    if prev_day not in original:
        return {"estimable": False, "pass": None, "reason": "missing_prior_position"}
    prev_nav = pd.to_numeric(
        pd.Series([aligned.loc[prev_day, "nav"]]), errors="coerce"
    ).iloc[0]
    prev_mv = pd.to_numeric(
        pd.Series([aligned.loc[prev_day, "market_value"]]), errors="coerce"
    ).iloc[0]
    if (
        not np.isfinite(prev_nav)
        or not np.isfinite(prev_mv)
        or prev_nav <= 0.0
        or prev_mv < 0.0
    ):
        return {"estimable": False, "pass": None, "reason": "invalid_prior_position"}
    pos = lagged_pos.reindex(idx)
    if bool(pos.isna().any()):
        return {
            "estimable": False,
            "pass": None,
            "reason": "missing_prior_position",
            "missing_days": int(pos.isna().sum()),
        }
    if bool((~np.isfinite(pos)).any()):
        return {
            "estimable": False,
            "pass": None,
            "reason": "invalid_prior_position",
            "invalid_days": int((~np.isfinite(pos)).sum()),
        }
    pos = pos.astype(float)
    mean_pos = float(pos.mean())
    market = market_returns.copy()
    market.index = market.index.astype(str)
    window_market = pd.to_numeric(market.reindex(idx), errors="coerce").dropna()
    coverage = float(len(window_market)) / float(len(idx))
    need = max(int(math.ceil(len(window_market) * worst_share)), WORST_DAY_MIN_DAYS)
    if coverage < MARKET_COVERAGE_MIN:
        return {
            "estimable": False,
            "pass": None,
            "reason": "market_coverage_insufficient",
            "window_days": len(idx),
            "market_days": int(len(window_market)),
            "coverage": coverage,
        }
    if len(window_market) < need or len(window_market) < MIN_MARKET_DAYS:
        return {
            "estimable": False,
            "pass": None,
            "reason": "insufficient_market_days",
            "window_days": len(idx),
            "market_days": int(len(window_market)),
            "min_market_days": MIN_MARKET_DAYS,
            "mean_position": mean_pos,
        }
    worst_days = list(window_market.nsmallest(need).index)
    worst_pos = float(pos.reindex(worst_days).astype(float).mean())
    return {
        "estimable": True,
        "pass": bool(mean_pos >= min_position and worst_pos >= worst_min),
        "mean_position": mean_pos,
        "worst_day_position": worst_pos,
        "window_days": int(len(idx)),
        "market_coverage": coverage,
        "worst_day_days": int(need),
    }


def position_gate(
    sleeve_frame: pd.DataFrame,
    windows: Sequence[Mapping[str, Any]],
    market_returns: pd.Series,
    *,
    min_position: float = POSITION_MIN,
    worst_share: float = WORST_DAY_SHARE,
    worst_min: float = WORST_DAY_POSITION_MIN,
) -> Dict[str, Any]:
    """资格门：**每个冻结窗**平均仓位 ≥10% 且市场最差 20% 交易日 ≥20%。

    轴：先按**冻结交易日历**排序对齐（保留空洞），在该轴上求前一交易日持仓，再截窗；
    缺记录 / 无效读数 / 缺前日 / 市场覆盖不足一律 `not_estimable`（V02）。
    聚合：已确认的逐窗 `fail` 优先上提，不被另一窗缺证据遮蔽（V03）。
    """
    frame = sleeve_frame.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    frame = frame.sort_index()
    original = set(frame.index)
    calendar = sorted(set(market_returns.index.astype(str)))  # 冻结交易日轴
    aligned = frame.reindex(calendar)
    pos = pd.to_numeric(aligned["market_value"], errors="coerce") / pd.to_numeric(
        aligned["nav"], errors="coerce"
    )
    lagged_pos = pos.shift(1)
    lines: Dict[str, Any] = {}
    for window in windows:
        key = f"{window.get('start_date')}~{window.get('end_date')}"
        lines[key] = _single_window_position(
            aligned,
            lagged_pos,
            calendar,
            original,
            window,
            market_returns,
            min_position=min_position,
            worst_share=worst_share,
            worst_min=worst_min,
        )
    if not lines:
        return _component(False, None, reason="empty_window")
    hard_fail = [
        k for k, v in lines.items() if bool(v.get("estimable")) and v.get("pass") is False
    ]
    all_estimable = all(bool(v.get("estimable")) for v in lines.values())
    if hard_fail:
        estimable, passed = True, False
    elif not all_estimable:
        estimable, passed = False, None
    else:
        estimable, passed = True, True
    payload = _component(
        estimable,
        passed,
        lines=lines,
        windows=len(lines),
        failed_windows=hard_fail,
        not_estimable_windows=[
            k for k, v in lines.items() if not bool(v.get("estimable"))
        ],
        min_position=float(min_position),
        worst_day_min=float(worst_min),
        domain="full_window",
        rule=(
            "每个冻结窗：完整窗平均仓位 ≥10% ∧ 市场最差 20% 交易日平均仓位 ≥20%"
            "（袖内 NAV、前一交易日收盘、空仓日计入分母；已确认 fail 优先）"
        ),
    )
    if not estimable:
        payload["reason"] = next(
            (v.get("reason") for v in lines.values() if not v["estimable"]), None
        )
    return payload


def noncrisis_cagr(values: np.ndarray, trading_days: int) -> Optional[float]:
    """**非危机期年化的唯一入口**：有效交易日口径（``trading_days/252``）。

    冻结配置：零息、不再平衡、非危机区间首尾相接。与 ``nav_metrics`` 的自然日跨度
    **不可互替**（R04：文本已删除"必须调用 nav_metrics"的分支，只保留本口径）。
    """
    if values.size < MIN_NONCRISIS_DAYS:
        return None
    years = float(trading_days) / 252.0
    if years <= 0.0:
        return None
    end = float(values[-1])
    if end <= 0.0:
        return None
    return float(end ** (1.0 / years) - 1.0)


def noncrisis_gate(
    a_leg_levels: pd.Series,
    combined_levels: pd.Series,
    crash_windows: Sequence[Mapping[str, Any]],
    *,
    calendar: Optional[Sequence[str]] = None,
    min_days: int = MIN_NONCRISIS_DAYS,
    cash_rate: float = 0.0,
    rebalance: bool = False,
    cash_budget: Optional[float] = None,
) -> Dict[str, Any]:
    """约束：非危机期 ``CAGR(A腿+B) − CAGR(A腿+现金)`` ≥ 0。

    现金腿由组合起点与 A 腿起点之差**内部推导**（不外部传入，避免额度不一致导致不公平比较）。
    先在**完整时间轴**算日收益，再按冻结窗掩码取反集（禁止先删 NAV 日期再桥接）。
    """
    if float(cash_rate) != 0.0 or bool(rebalance):
        return _component(
            False,
            None,
            reason="unsupported_cash_config",
            detail="仅支持零息、不再平衡；其他约定须扩展共享接口后另行预登记",
        )
    a_leg = pd.to_numeric(a_leg_levels, errors="coerce")
    combined = pd.to_numeric(combined_levels, errors="coerce")
    both = pd.concat({"a_leg": a_leg, "combined": combined}, axis=1)
    if bool(both.isna().any().any()):
        # 缺行不得桥接（dropna 会压缩时间轴、把跨窗收益接回来）
        return _component(False, None, reason="missing_levels")
    if calendar is None:
        return _component(False, None, reason="calendar_required")
    # 两条路径**同时**缺同一交易日时 concat 不会产生 NaN，必须用冻结日历查空洞（W01-A）
    start, end = both.index.min(), both.index.max()
    span = [str(d) for d in calendar if start <= str(d) <= end]
    if not span:
        return _component(False, None, reason="calendar_mismatch")
    missing = [day for day in span if day not in both.index]
    if missing:
        return _component(
            False, None, reason="missing_calendar_days", missing_days=len(missing)
        )
    both = both.reindex(span)
    if len(both) < min_days + 2:
        return _component(False, None, reason="insufficient_levels")
    budget = (
        float(cash_budget)
        if cash_budget is not None
        else float(both["combined"].iloc[0] - both["a_leg"].iloc[0])
    )
    frame = pd.DataFrame(
        {
            "a_leg": both["a_leg"],
            "combined": both["combined"],
            "cash": both["a_leg"] + budget,
        }
    )
    frame.index = frame.index.astype(str)
    frame = frame.dropna(how="any")
    if len(frame) < min_days + 2:
        return _component(False, None, reason="insufficient_levels")
    returns = frame.pct_change().dropna()
    masked = np.zeros(len(returns), dtype=bool)
    for window in crash_windows:
        start = str(window.get("start_date", "")).replace("-", "")
        end = str(window.get("end_date", "")).replace("-", "")
        if not start or not end:
            continue
        masked |= (returns.index >= start) & (returns.index <= end)
    sub = returns.loc[~masked]
    if len(sub) < min_days:
        return _component(False, None, reason="insufficient_noncrisis_days", n=int(len(sub)))
    nav_b = np.cumprod(1.0 + sub["combined"].to_numpy(dtype=float))
    nav_c = np.cumprod(1.0 + sub["cash"].to_numpy(dtype=float))
    cagr_b = noncrisis_cagr(nav_b, len(sub))
    cagr_c = noncrisis_cagr(nav_c, len(sub))
    if cagr_b is None or cagr_c is None:
        return _component(False, None, reason="cagr_undefined", n=int(len(sub)))
    delta_pp = (cagr_b - cagr_c) * 100.0
    return _component(
        True,
        bool(delta_pp >= 0.0),
        n=int(len(sub)),
        noncrisis_days=int(len(sub)),
        cash_budget=budget,
        cagr_combined=float(cagr_b),
        cagr_cash=float(cagr_c),
        delta_cagr_pp=float(delta_pp),
        rule=(
            "完整时间轴日收益 → 冻结窗掩码取反集 → 两条路径各自复合年化后相减 ≥0；"
            "对照 = A 腿 + 现金腿（不是纯现金）"
        ),
    )


def _e4_for_risk_adjusted(e4: Mapping[str, Any]) -> Dict[str, Any]:
    """型 II 的 E4：剔除 E4-4（利用率不适用），其余否决线保留。"""
    vetoes: List[str] = [v for v in e4.get("vetoes", []) if v != "E4_4_utilization"]
    not_estimable: List[str] = [
        n for n in e4.get("not_estimable", []) if n != "E4_4_utilization"
    ]
    lines = dict(e4.get("lines") or {})
    return {
        "estimable": not bool(not_estimable),
        "pass": bool(not vetoes and not not_estimable),
        "vetoes": vetoes,
        "not_estimable": not_estimable,
        "lines": lines,
        "exempt": ["E4_4_utilization"],
        "rule": "型 II 保留 E4-1/2/3/5；E4-4 利用率记 not_applicable",
    }


def risk_adjusted_components(
    *,
    levels: pd.DataFrame,
    crash_windows: Sequence[Mapping[str, Any]],
    sleeve_frame: pd.DataFrame,
    market_returns: pd.Series,
    correlations: Mapping[str, Any],
    e4: Mapping[str, Any],
    alpha_c: float,
) -> Dict[str, Dict[str, Any]]:
    """组装型 II 的六个组件（delta_sharpe / delta_maxdd / correlation / E4 / position / noncrisis）。"""
    frame = levels.copy()
    frame.index = frame.index.astype(str)
    a_returns = np.diff(frame["a_only"].to_numpy(dtype=float)) / frame["a_only"].to_numpy(
        dtype=float
    )[:-1]
    combined_returns = np.diff(frame["combined"].to_numpy(dtype=float)) / frame[
        "combined"
    ].to_numpy(dtype=float)[:-1]
    return {
        "delta_sharpe": delta_sharpe_gate(a_returns, combined_returns, alpha_c=alpha_c),
        "delta_maxdd": delta_maxdd_gate(a_returns, combined_returns, alpha_c=alpha_c),
        "correlation": dict(correlations),
        "E4": _e4_for_risk_adjusted(e4),
        "position": position_gate(sleeve_frame, crash_windows, market_returns),
        "noncrisis": noncrisis_gate(
            frame["a_leg"],
            frame["combined"],
            crash_windows,
            calendar=list(market_returns.index.astype(str)),
        ),
    }
