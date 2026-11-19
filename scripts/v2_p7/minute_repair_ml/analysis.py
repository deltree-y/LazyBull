"""比较、分段、相关与配对块自举（协议 §11.2/§11.3）。

链式指标统一调用 `v2/common/chain_metrics.py`，不另抄 CAGR/MDD 公式；
`MDD_loss` 是最大回撤损失的正数形式（底层返回负数，这里统一取相反数）。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import spec
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

ARMS: tuple[str, ...] = ("A100", "A80_CASH20", "A80_LR20", "A80_DAILY20")


class AnalysisError(RuntimeError):
    """分析层错误（缺臂、日期轴不一致）。"""


def _chain_metrics(nav: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame({"nav": nav.astype(float).to_numpy()})
    metrics = calculate_chain_metrics(
        frame,
        annual_trading_days=int(spec.METRICS["annual_trading_days"]),
        annual_risk_free_rate=float(spec.METRICS["annual_risk_free_rate"]),
    )
    metrics["mdd_loss"] = (
        float(-metrics["max_drawdown"]) if metrics["max_drawdown"] is not None else None
    )
    return metrics


def arm_metrics(combined: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """各臂绝对指标（净值从首日归一）。"""
    missing = [arm for arm in ARMS if arm not in combined.columns]
    if missing:
        raise AnalysisError(f"缺少臂：{missing}")
    out: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        series = combined[arm].astype(float)
        if series.isna().any() or (series <= 0).any():
            raise AnalysisError(f"臂 {arm} 存在缺失或非正净值")
        out[arm] = _chain_metrics(series / float(series.iloc[0]))
    return out


def target_table(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """四项差值与达标状态（只决定资源建议，不是正式准入门）。"""
    targets = spec.TARGETS
    cagr = {arm: metrics[arm]["cagr"] for arm in ARMS}
    mdd = {arm: metrics[arm]["mdd_loss"] for arm in ARMS}
    values = {
        "cagr_delta_vs_A100": cagr["A80_LR20"] - cagr["A100"],
        "mdd_loss_delta_vs_A100": mdd["A80_LR20"] - mdd["A100"],
        "cagr_delta_vs_cash": cagr["A80_LR20"] - cagr["A80_CASH20"],
        "minute_increment_vs_daily": cagr["A80_LR20"] - cagr["A80_DAILY20"],
    }
    checks = {
        "cagr_delta_ok": values["cagr_delta_vs_A100"] >= float(targets["cagr_delta_min"]),
        "mdd_loss_ok": values["mdd_loss_delta_vs_A100"] <= float(targets["mdd_loss_delta_max"]),
        "beats_cash_ok": values["cagr_delta_vs_cash"] > 0,
        "minute_increment_ok": values["minute_increment_vs_daily"]
        >= float(targets["minute_increment_min"]),
    }
    return {"values": values, "checks": checks, "all_met": all(checks.values())}


def daily_returns(combined: pd.DataFrame) -> pd.DataFrame:
    """各臂日收益（首日之后，避免把初始日当成收益）。"""
    values = combined[list(ARMS)].astype(float)
    returns = values.pct_change(fill_method=None).iloc[1:]
    if returns.isna().any().any():
        raise AnalysisError("组合日收益存在缺失（日期轴不完整）")
    return returns


def segment_metrics(combined: pd.DataFrame) -> dict[str, Any]:
    """三个固定子段（不按结果择优重组）。"""
    out: dict[str, Any] = {}
    for start, end in spec.WINDOW["segments"]:
        part = combined.loc[(combined.index >= str(start)) & (combined.index <= str(end))]
        label = f"{start}-{end}"
        if len(part) < 3:
            out[label] = {"days": int(len(part)), "metrics": None}
            continue
        out[label] = {
            "days": int(len(part)),
            "metrics": {arm: _chain_metrics(part[arm] / float(part[arm].iloc[0])) for arm in ARMS},
        }
    return out


def recent_segment_flag(segments: Mapping[str, Any]) -> dict[str, Any]:
    """最后固定子段：对现金 CAGR 为负且对 A100 回撤更差 → continue_limited 触发。"""
    keys = sorted(segments)
    if not keys:
        return {"triggered": False, "reason": "no_segment"}
    last = segments[keys[-1]]
    metrics = last.get("metrics")
    if not metrics:
        return {"triggered": False, "reason": "segment_too_short"}
    cash_delta = metrics["A80_LR20"]["cagr"] - metrics["A80_CASH20"]["cagr"]
    mdd_delta = metrics["A80_LR20"]["mdd_loss"] - metrics["A100"]["mdd_loss"]
    return {
        "triggered": bool(cash_delta < 0 and mdd_delta > 0),
        "cash_delta": float(cash_delta),
        "mdd_loss_delta": float(mdd_delta),
        "segment": keys[-1],
    }


def correlations(combined: pd.DataFrame) -> dict[str, Any]:
    """全样本日收益相关 + 非重叠 20 日收益相关（显式给出日期与样本数）。"""
    returns = daily_returns(combined)
    base = returns["A80_LR20"]
    daily = {
        arm: {
            "corr": float(base.corr(returns[arm])),
            "days": int(base.notna().sum()),
        }
        for arm in ARMS
        if arm != "A80_LR20"
    }
    values = combined[list(ARMS)].astype(float)
    block = values.iloc[::20]
    block_returns = block.pct_change(fill_method=None).iloc[1:]
    non_overlap = {
        arm: {
            "corr": float(block_returns["A80_LR20"].corr(block_returns[arm])),
            "blocks": int(block_returns["A80_LR20"].notna().sum()),
            "first": str(block_returns.index[0]) if len(block_returns) else None,
            "last": str(block_returns.index[-1]) if len(block_returns) else None,
        }
        for arm in ARMS
        if arm != "A80_LR20"
    }
    return {"daily": daily, "non_overlap_20d": non_overlap}


def paired_block_bootstrap(
    combined: pd.DataFrame,
    *,
    block: int | None = None,
    repetitions: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """配对循环块自举：同一重排同时作用于四臂，重建链式指标并取差值分位。"""
    length = int(block or spec.BOOTSTRAP["block"])
    runs = int(repetitions or spec.BOOTSTRAP["repetitions"])
    rng = np.random.default_rng(int(seed or spec.BOOTSTRAP["seed"]))
    returns = daily_returns(combined)
    values = returns.to_numpy(dtype=float)
    total = len(values)
    if total <= length + 1:
        return {"status": "insufficient_days", "days": int(total)}
    blocks = int(np.ceil(total / length))
    records = {
        key: []
        for key in (
            "cagr_delta_vs_A100",
            "mdd_loss_delta_vs_A100",
            "cagr_delta_vs_cash",
            "minute_increment_vs_daily",
        )
    }
    columns = list(ARMS)
    index = {arm: columns.index(arm) for arm in ARMS}
    for _ in range(runs):
        starts = rng.integers(0, total - length + 1, size=blocks)
        sample = np.concatenate([values[start : start + length] for start in starts])[:total]
        navs = {arm: pd.Series(np.cumprod(1.0 + sample[:, index[arm]])) for arm in ARMS}
        metrics = {arm: _chain_metrics(navs[arm]) for arm in ARMS}
        cagr = {arm: metrics[arm]["cagr"] for arm in ARMS}
        mdd = {arm: metrics[arm]["mdd_loss"] for arm in ARMS}
        records["cagr_delta_vs_A100"].append(cagr["A80_LR20"] - cagr["A100"])
        records["mdd_loss_delta_vs_A100"].append(mdd["A80_LR20"] - mdd["A100"])
        records["cagr_delta_vs_cash"].append(cagr["A80_LR20"] - cagr["A80_CASH20"])
        records["minute_increment_vs_daily"].append(cagr["A80_LR20"] - cagr["A80_DAILY20"])
    summary = {}
    for key, values_list in records.items():
        array = np.asarray(values_list, dtype=float)
        summary[key] = {
            "p2_5": float(np.percentile(array, 2.5)),
            "p50": float(np.percentile(array, 50)),
            "p97_5": float(np.percentile(array, 97.5)),
            "share_positive": float((array > 0).mean()),
        }
    return {
        "status": "ok",
        "block": length,
        "repetitions": runs,
        "seed": int(seed or spec.BOOTSTRAP["seed"]),
        "days": int(total),
        "summary": summary,
        "note": "描述性区间，不是正式 F29 检验；达标重排比例不是未来成功概率",
    }


def pressure_checks(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """压力情景判据（不含 0.20pp 分钟增量门）。"""
    returns = {
        "cagr_delta_vs_A100": metrics["A80_LR20"]["cagr"] - metrics["A100"]["cagr"],
        "mdd_loss_delta_vs_A100": metrics["A80_LR20"]["mdd_loss"] - metrics["A100"]["mdd_loss"],
        "cagr_delta_vs_cash": metrics["A80_LR20"]["cagr"] - metrics["A80_CASH20"]["cagr"],
    }
    checks = {
        "cagr_delta_non_negative": returns["cagr_delta_vs_A100"] >= 0,
        "mdd_loss_within_1pp": returns["mdd_loss_delta_vs_A100"]
        <= float(spec.TARGETS["mdd_loss_delta_max"]),
        "beats_cash": returns["cagr_delta_vs_cash"] > 0,
    }
    return {"values": returns, "checks": checks, "all_met": all(checks.values())}
