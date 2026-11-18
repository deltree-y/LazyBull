"""R5 读数：链式指标、四臂对比、固定分段与配对环形块自举。

指标唯一实现 = `v2/common/chain_metrics.calculate_chain_metrics`，本模块不自建年化公式。
自举只报描述性区间，不产出统计通过结论。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import spec
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics


def chain_metrics(values: pd.Series) -> dict[str, Any]:
    """按唯一实现算链式指标（nav 序列）。"""
    frame = pd.DataFrame({"nav": values.to_numpy(dtype=float)})
    metrics = calculate_chain_metrics(frame)
    mdd = metrics.get("max_drawdown")
    out = dict(metrics)
    out["mdd_loss"] = abs(mdd) if mdd is not None else None
    return out


def arm_metrics_table(arms: Mapping[str, pd.Series]) -> pd.DataFrame:
    """每个臂一行的主表（CAGR / signed MDD / 损失幅度 / Sharpe / 期末值）。"""
    rows = []
    for name, series in arms.items():
        metrics = chain_metrics(series)
        rows.append(
            {
                "arm": name,
                "start": float(series.iloc[0]),
                "end": float(series.iloc[-1]),
                "cagr": metrics.get("cagr"),
                "max_drawdown": metrics.get("max_drawdown"),
                "mdd_loss": metrics.get("mdd_loss"),
                "sharpe": metrics.get("sharpe"),
                "volatility": metrics.get("volatility"),
                "trading_days": metrics.get("trading_days"),
            }
        )
    return pd.DataFrame(rows)


def _diff(left: Any, right: Any) -> float:
    """双方都有值才相减，否则 NaN（禁止用 0 冒充缺失）。"""
    if left is None or right is None:
        return float("nan")
    return float(left) - float(right)


def relative_to(reference: pd.Series, candidate: pd.Series) -> dict[str, float]:
    """相对参照：cagr_delta 与 mdd_loss_delta 均为正数放大方向一致的差。"""
    base = chain_metrics(reference)
    cand = chain_metrics(candidate)
    return {
        "cagr_delta": _diff(cand.get("cagr"), base.get("cagr")),
        "mdd_loss_delta": _diff(cand.get("mdd_loss"), base.get("mdd_loss")),
        "dd_delta": _diff(cand.get("max_drawdown"), base.get("max_drawdown")),
        "sharpe_delta": _diff(cand.get("sharpe"), base.get("sharpe")),
    }


def target_check(candidate: pd.Series, baseline: pd.Series, cash_arm: pd.Series) -> dict[str, Any]:
    """主目标三项：相对 A100 的 CAGR ≥ +0.5pp、MDD 损失增幅 ≤ 1pp、且胜过现金臂。"""
    rel = relative_to(baseline, candidate)
    cash_cagr = chain_metrics(cash_arm).get("cagr")
    cand_cagr = chain_metrics(candidate).get("cagr")
    meets_cagr = rel["cagr_delta"] >= float(spec.TARGETS["cagr_delta_min"])
    meets_mdd = rel["mdd_loss_delta"] <= float(spec.TARGETS["mdd_loss_delta_max"])
    beats_cash = cand_cagr is not None and cash_cagr is not None and cand_cagr > cash_cagr
    return {
        **rel,
        "cagr_vs_cash": _diff(cand_cagr, cash_cagr),
        "meets_cagr": bool(meets_cagr) if np.isfinite(rel["cagr_delta"]) else False,
        "meets_mdd": bool(meets_mdd) if np.isfinite(rel["mdd_loss_delta"]) else False,
        "beats_cash": bool(beats_cash),
        "all_three": bool(
            meets_cagr and meets_mdd and beats_cash
        ),
    }


def segment_table(arms: Mapping[str, pd.Series], segments: Sequence[Sequence[str]]) -> pd.DataFrame:
    """固定分段指标（段首用前一日资产作基点，段内不重新配置资金）。"""
    rows = []
    for label, (start, end) in enumerate(segments, start=1):
        for name, series in arms.items():
            index = list(series.index)
            position = {day: i for i, day in enumerate(index)}
            first = min((day for day in index if day >= start), default=None)
            last = max((day for day in index if day <= end), default=None)
            if first is None or last is None:
                continue
            begin = position[first]
            base_value = float(series.iloc[begin - 1]) if begin > 0 else float(series.iloc[begin])
            window = series.loc[first:last]
            normalised = pd.concat(
                [pd.Series([base_value], index=[str(int(first) - 1)]), window]
            )
            metrics = chain_metrics(normalised)
            rows.append(
                {
                    "segment": int(label),
                    "start": first,
                    "end": last,
                    "arm": name,
                    "cagr": metrics.get("cagr"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "mdd_loss": metrics.get("mdd_loss"),
                    "sharpe": metrics.get("sharpe"),
                }
            )
    return pd.DataFrame(rows)


def year_segments(index: Sequence[str]) -> list[list[str]]:
    """按自然年切段（段首仍取前一日资产为基点，与固定分段同口径）。"""
    years = sorted({str(day)[:4] for day in index})
    return [[f"{year}0101", f"{year}1231"] for year in years]


def daily_returns(series: pd.Series) -> np.ndarray:
    """日收益（首日为 0，保证与资产序列等长）。"""
    values = series.to_numpy(dtype=float)
    returns = np.zeros_like(values)
    returns[1:] = values[1:] / values[:-1] - 1.0
    return returns


def _circular_blocks(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """环形块索引（同一抽样同时作用于全部比较臂）。"""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    offsets = np.arange(block)
    index = (starts[:, None] + offsets[None, :]).reshape(-1) % n
    return index[:n]


def _metrics_from_returns(returns: np.ndarray) -> dict[str, float]:
    # 必须保留期初基点 1.0：否则首个抽中收益会被链式指标当成起点而漏计
    nav = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    metrics = chain_metrics(pd.Series(nav))
    return {
        "cagr": float(metrics.get("cagr") or 0.0),
        "max_drawdown": float(metrics.get("max_drawdown") or 0.0),
        "mdd_loss": float(metrics.get("mdd_loss") or 0.0),
        "sharpe": float(metrics.get("sharpe") or 0.0),
    }


def bootstrap_deltas(
    arms: Mapping[str, pd.Series],
    comparisons: Sequence[tuple[str, str]],
    *,
    block: int | None = None,
    repetitions: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """配对环形块自举：每对比较的 ΔCAGR / ΔMDD损失 / ΔSharpe 分位。"""
    block = int(block or spec.BOOTSTRAP["block"])
    repetitions = int(repetitions or spec.BOOTSTRAP["repetitions"])
    seed = int(seed if seed is not None else spec.BOOTSTRAP["seed"])
    returns = {name: daily_returns(series) for name, series in arms.items()}
    lengths = {name: len(array) for name, array in returns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"臂长度不一致：{lengths}")
    n = len(next(iter(returns.values())))
    rng = np.random.default_rng(seed)
    records: dict[str, dict[str, list[float]]] = {
        f"{cand}_vs_{base}": {"cagr": [], "mdd_loss": [], "sharpe": []}
        for base, cand in comparisons
    }
    for _ in range(repetitions):
        index = _circular_blocks(n, block, rng)
        cache = {name: _metrics_from_returns(array[index]) for name, array in returns.items()}
        for base, cand in comparisons:
            key = f"{cand}_vs_{base}"
            records[key]["cagr"].append(cache[cand]["cagr"] - cache[base]["cagr"])
            records[key]["mdd_loss"].append(cache[cand]["mdd_loss"] - cache[base]["mdd_loss"])
            records[key]["sharpe"].append(cache[cand]["sharpe"] - cache[base]["sharpe"])
    result: dict[str, Any] = {
        "method": "circular_block",
        "block": block,
        "repetitions": repetitions,
        "seed": seed,
        "days": int(n),
    }
    for key, columns in records.items():
        result[key] = {}
        for metric, values in columns.items():
            array = np.asarray(values, dtype=float)
            result[key][metric] = {
                "p2_5": float(np.percentile(array, 2.5)),
                "p50": float(np.percentile(array, 50)),
                "p97_5": float(np.percentile(array, 97.5)),
                "share_positive": float(np.mean(array > 0)),
            }
    return result


def correlation_table(arms: Mapping[str, pd.Series]) -> pd.DataFrame:
    """全样本日收益相关（描述性，不替代 F29 三口径正式实现）。"""
    frame = pd.DataFrame({name: daily_returns(series) for name, series in arms.items()})
    return frame.corr()


def non_overlap_returns(series: pd.Series, horizon: int = 20) -> pd.Series:
    """20 日非重叠收益（自窗口首日起每 horizon 日取一个点）。"""
    values = series.to_numpy(dtype=float)
    points = list(range(0, len(values), int(horizon)))
    out = []
    for i in range(1, len(points)):
        start = points[i - 1]
        end = points[i]
        out.append(values[end] / values[start] - 1.0)
    return pd.Series(out, dtype=float)


def turnover_and_cost(daily: pd.DataFrame) -> dict[str, Any]:
    """换手与费用汇总（逐日账字段：buy_amount / sell_amount / fees / total_value）。"""
    if daily.empty:
        return {"turnover": None, "fees": None, "buy_turnover": None, "sell_turnover": None}
    values = daily["total_value"].astype(float)
    average = float(values.mean())
    buys = float(daily["buy_amount"].astype(float).sum()) if "buy_amount" in daily else 0.0
    sells = float(daily["sell_amount"].astype(float).sum()) if "sell_amount" in daily else 0.0
    fees = float(daily["fees"].astype(float).sum()) if "fees" in daily else 0.0
    years = max(len(daily) / 252.0, 1e-9)
    total_turnover = (buys + sells) / average if average > 0 else None
    return {
        "turnover": total_turnover,
        # 全窗累计换手 / 年数：与"单边年换手"口径一致，避免把 7.6 年累计当年度读数
        "turnover_annualized": total_turnover / years if total_turnover is not None else None,
        "buy_turnover": buys / average if average > 0 else None,
        "sell_turnover": sells / average if average > 0 else None,
        "fees": fees,
        "fee_share": fees / average if average > 0 else None,
        "years": years,
    }


def exposure_profile(daily: pd.DataFrame) -> dict[str, Any]:
    """实际股票仓位与平均持仓数。"""
    if daily.empty:
        return {"mean_exposure": None, "mean_positions": None}
    values = daily["total_value"].astype(float)
    exposure = (values - daily["cash"].astype(float)) / values
    return {
        "mean_exposure": float(exposure.mean()),
        "max_exposure": float(exposure.max()),
        "mean_positions": float(daily["positions"].astype(float).mean())
        if "positions" in daily
        else None,
    }
