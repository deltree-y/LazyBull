"""R5 读数：链式指标、四臂对比、固定分段与配对环形块自举。

指标唯一实现 = `v2/common/chain_metrics.calculate_chain_metrics`，本模块不自建年化公式。
自举只报描述性区间，不产出统计通过结论。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

from scripts.v2_p7.defensive_ml import spec


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
    """相对参照：dd_improvement = MDD_cand - MDD_ref（正=改善），CAGR 差为百分点差。"""
    base = chain_metrics(reference)
    cand = chain_metrics(candidate)
    return {
        "dd_improvement": _diff(cand.get("max_drawdown"), base.get("max_drawdown")),
        "cagr_delta": _diff(cand.get("cagr"), base.get("cagr")),
        "sharpe_delta": _diff(cand.get("sharpe"), base.get("sharpe")),
    }


def target_check(candidate: pd.Series, baseline: pd.Series, cash_arm: pd.Series) -> dict[str, Any]:
    """主目标三项：回撤改善 ≥3pp、CAGR 代价 ≤1pp、不输现金对照。"""
    rel = relative_to(baseline, candidate)
    cash_cagr = chain_metrics(cash_arm).get("cagr")
    cand_cagr = chain_metrics(candidate).get("cagr")
    meets_dd = rel["dd_improvement"] >= float(spec.TARGETS["dd_improvement_min"])
    meets_cagr = rel["cagr_delta"] >= float(spec.TARGETS["cagr_delta_min"])
    beats_cash = (
        cand_cagr is not None and cash_cagr is not None and cand_cagr >= cash_cagr
    )
    return {
        **rel,
        "cagr_vs_cash": _diff(cand_cagr, cash_cagr),
        "meets_dd": bool(meets_dd),
        "meets_cagr": bool(meets_cagr),
        "beats_cash": bool(beats_cash),
        "all_three": bool(meets_dd and meets_cagr and beats_cash),
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
                    "sharpe": metrics.get("sharpe"),
                }
            )
    return pd.DataFrame(rows)


def daily_returns(series: pd.Series) -> np.ndarray:
    """日收益（首日为 0，保证与资产序列等长）。"""
    values = series.to_numpy(dtype=float)
    returns = np.zeros_like(values)
    returns[1:] = values[1:] / values[:-1] - 1.0
    return returns


def _circular_blocks(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """环形块索引（同一抽样同时作用于所有臂）。"""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    offsets = np.arange(block)
    index = (starts[:, None] + offsets[None, :]).reshape(-1) % n
    return index[:n]


def _metrics_from_returns(returns: np.ndarray) -> dict[str, float]:
    # 必须保留期初基点 1.0：否则首个抽中收益会被链式指标当成起点而漏计
    nav = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    series = pd.Series(nav)
    metrics = chain_metrics(series)
    return {
        "cagr": float(metrics.get("cagr") or 0.0),
        "max_drawdown": float(metrics.get("max_drawdown") or 0.0),
        "sharpe": float(metrics.get("sharpe") or 0.0),
    }


def bootstrap_deltas(
    arms: Mapping[str, pd.Series],
    baseline: str,
    candidate: str,
    *,
    block: int | None = None,
    repetitions: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """配对环形块自举：ΔCAGR / ΔMDD / ΔSharpe 的 2.5/50/97.5 分位。"""
    block = int(block or spec.BOOTSTRAP["block"])
    repetitions = int(repetitions or spec.BOOTSTRAP["repetitions"])
    seed = int(seed if seed is not None else spec.BOOTSTRAP["seed"])
    returns = {name: daily_returns(series) for name, series in arms.items()}
    n = len(returns[baseline])
    for name, array in returns.items():
        if len(array) != n:
            raise ValueError(f"臂 {name} 长度 {len(array)} 与基准 {n} 不一致")
    rng = np.random.default_rng(seed)
    records = {"cagr": [], "max_drawdown": [], "sharpe": []}
    for _ in range(repetitions):
        index = _circular_blocks(n, block, rng)
        base = _metrics_from_returns(returns[baseline][index])
        cand = _metrics_from_returns(returns[candidate][index])
        records["cagr"].append(cand["cagr"] - base["cagr"])
        records["max_drawdown"].append(cand["max_drawdown"] - base["max_drawdown"])
        records["sharpe"].append(cand["sharpe"] - base["sharpe"])
    result: dict[str, Any] = {
        "block": block,
        "repetitions": repetitions,
        "seed": seed,
        "days": int(n),
    }
    for key, values in records.items():
        array = np.asarray(values, dtype=float)
        result[key] = {
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
