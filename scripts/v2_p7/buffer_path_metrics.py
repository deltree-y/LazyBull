"""固定续持研究的配对路径读数；重抽样仅展示不确定性，不作准入裁判。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_SEED = 20261116


def validate_daily(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """校验连续账户的日期、初始现金点和有限正净值，不静默删行。"""
    frame = pd.DataFrame(rows).copy()
    required = {"trade_date", "nav", "cash", "total_value", "market_value", "receivable"}
    if missing := required - set(frame.columns):
        raise ValueError(f"路径缺字段：{sorted(missing)}")
    frame["trade_date"] = frame.trade_date.astype(str)
    days = frame.trade_date.tolist()
    if len(frame) < 2 or days != sorted(set(days)):
        raise ValueError("账户日期必须唯一升序且包含至少两个时点")
    values = frame[list(required - {"trade_date"})].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not frame.nav.gt(0).all():
        raise ValueError("账户路径包含非有限值或非正净值")
    if float(frame.nav.iloc[0]) != 1.0:
        raise ValueError("首个收盘点必须为明确的空现金初始NAV=1")
    if frame.cash.min() < -1e-8:
        raise ValueError("现金为负，资金可行条件不满足")
    return frame


def goals(delta_cagr: float, delta_mdd: float, base_mdd: float, arm_mdd: float) -> dict:
    """历史点估计与用户两项目标逐一比较，不输出正式准入。"""
    return {
        "return_goal": bool(delta_cagr >= 0.005 and delta_mdd >= -0.01),
        "drawdown_goal": bool(
            base_mdd < 0 and abs(arm_mdd) <= abs(base_mdd) * 0.75 and delta_cagr >= -0.02
        ),
    }


def annual_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """年度收益含跨年首日收益；首年以已登记的初始NAV为基期。"""
    result = []
    previous_nav = float(frame.nav.iloc[0])
    for year, part in frame.groupby(frame.trade_date.str[:4], sort=True):
        last = float(part.nav.iloc[-1])
        result.append(
            {
                "year": str(year),
                "return": last / previous_nav - 1.0,
                "fees": float(part.fees.sum()),
                "buy_amount": float(part.buy_amount.sum()),
                "sell_amount": float(part.sell_amount.sum()),
                "mean_exposure": float((part.market_value / part.total_value).mean()),
            }
        )
        previous_nav = last
    return result


def arm_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """所有CAGR/MDD复用唯一实现，费用与换手保留金额及分母。"""
    metrics = calculate_chain_metrics(frame[["nav"]])
    intervals = len(frame) - 1
    previous = frame.total_value.shift(1).fillna(frame.total_value.iloc[0])
    turnover = (frame.buy_amount + frame.sell_amount) / previous / 2
    return {
        **metrics,
        "start_date": str(frame.trade_date.iloc[0]),
        "end_date": str(frame.trade_date.iloc[-1]),
        "fees_yuan": float(frame.fees.sum()),
        "fee_intensity_per_252": float((frame.fees / previous).sum() * 252 / intervals),
        "one_way_turnover_per_252": float(turnover.sum() * 252 / intervals),
        "mean_exposure": float((frame.market_value / frame.total_value).mean()),
        "minimum_cash": float(frame.cash.min()),
        "ending_nav": float(frame.nav.iloc[-1]),
        "ending_receivable": float(frame.receivable.iloc[-1]),
        "ending_market_value": float(frame.market_value.iloc[-1]),
        "annual": annual_rows(frame),
    }


def _pair_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    """将配对日收益构成初始点为1的链，复用统一指标。"""
    base = calculate_chain_metrics(pd.DataFrame({"nav": np.r_[1.0, np.cumprod(1.0 + left)]}))
    arm = calculate_chain_metrics(pd.DataFrame({"nav": np.r_[1.0, np.cumprod(1.0 + right)]}))
    return (
        arm["cagr"] - base["cagr"],
        arm["max_drawdown"] - base["max_drawdown"],
        base["max_drawdown"],
        arm["max_drawdown"],
    )


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    block: int = BOOTSTRAP_BLOCK,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """同一循环分块索引重抽两臂，频率不是未来概率或已标定置信度。"""
    if len(left) != len(right) or len(left) < 2 or repeats <= 0 or block <= 0:
        raise ValueError("配对收益长度/重抽样参数非法")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("配对收益含非有限值")
    if np.any(left <= -1) or np.any(right <= -1):
        raise ValueError("日收益不能小于等于-100%")
    count = len(left)
    rng = np.random.default_rng(seed)
    values = []
    return_hits = drawdown_hits = 0
    for _ in range(repeats):
        starts = rng.integers(0, count, size=(count + block - 1) // block)
        indices = ((starts[:, None] + np.arange(block)) % count).ravel()[:count]
        row = _pair_metrics(left[indices], right[indices])
        hit = goals(*row)
        return_hits += int(hit["return_goal"])
        drawdown_hits += int(hit["drawdown_goal"])
        values.append(row[:2])
    quantiles = np.quantile(np.asarray(values), [0.025, 0.5, 0.975], axis=0)
    return {
        "kind": "exploratory_paired_circular_block_bootstrap",
        "repeats": repeats,
        "block": block,
        "seed": seed,
        "delta_cagr_quantiles": dict(zip(["q025", "q50", "q975"], quantiles[:, 0].tolist())),
        "delta_mdd_quantiles": dict(zip(["q025", "q50", "q975"], quantiles[:, 1].tolist())),
        "return_goal_frequency": return_hits / repeats,
        "drawdown_goal_frequency": drawdown_hits / repeats,
        "future_success_probability": None,
        "calibrated_for_admission": False,
    }


def compare_accounts(
    base_rows: list[dict[str, Any]], arm_rows: list[dict[str, Any]], *, bootstrap: bool = True
) -> dict[str, Any]:
    """先验证逐日同域，再输出真实路径、年度及探索性重抽样。"""
    base, arm = validate_daily(base_rows), validate_daily(arm_rows)
    if base.trade_date.tolist() != arm.trade_date.tolist():
        raise ValueError("两臂日期集合或顺序不同，禁止缩成有利交集")
    if float(base.total_value.iloc[0]) != float(arm.total_value.iloc[0]):
        raise ValueError("两臂初始资金不同")
    base_metrics, arm_result = arm_metrics(base), arm_metrics(arm)
    delta = {
        name: float(arm_result[name] - base_metrics[name])
        for name in [
            "cagr",
            "max_drawdown",
            "total_return",
            "fees_yuan",
            "fee_intensity_per_252",
            "one_way_turnover_per_252",
            "mean_exposure",
        ]
    }
    annual = [
        {
            "year": a["year"],
            "base_return": a["return"],
            "arm_return": b["return"],
            "delta_return": b["return"] - a["return"],
            "base_fees": a["fees"],
            "arm_fees": b["fees"],
        }
        for a, b in zip(base_metrics["annual"], arm_result["annual"], strict=True)
    ]
    result = {
        "base": base_metrics,
        "arm": arm_result,
        "delta": delta,
        "point_estimate_goals": goals(
            delta["cagr"],
            delta["max_drawdown"],
            base_metrics["max_drawdown"],
            arm_result["max_drawdown"],
        ),
        "annual_paired": annual,
        "formal_admission": "not_evaluated",
    }
    if bootstrap:
        result["uncertainty"] = paired_bootstrap(
            base.nav.pct_change(fill_method=None).iloc[1:].to_numpy(),
            arm.nav.pct_change(fill_method=None).iloc[1:].to_numpy(),
        )
    return result
