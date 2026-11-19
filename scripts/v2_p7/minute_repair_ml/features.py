"""唯一 24 列特征实现（协议 §6）：6 个日线控制列 + 18 个分钟列。

硬约束：
- 只使用 T 日及以前的信息；跨日分钟基准窗口为 T 之前 20 个交易日（不含 T）。
- 缺失返回 NaN 并给出原因，不填 0、不前填、不做全历史标准化或缩尾。
- 分母为零返回 NaN，除少数明确定义的空集合取 0 外不任意加极小数。
- 同槽重复、真实越界等属于数据错误，抛 :class:`FeatureError`，不静默吞掉。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import sessions, spec

CORE_SLOTS = list(range(2, 48))
AM_SLOTS = list(range(2, 25))
PM_SLOTS = list(range(26, 48))
J_SLOTS = AM_SLOTS + PM_SLOTS
LATE_SLOTS = list(range(42, 48))
AMOUNT_SCALE = 1e8
IMPACT_MIN_SLOTS = 5
HISTORY_WINDOW = 20
HISTORY_MIN_VALID = 16


class FeatureError(RuntimeError):
    """特征计算遇到不可解释的数据（真实越界、同槽重复等）。"""


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _ratio(numerator: float, denominator: float) -> float:
    """有限且分母 > 0 才返回比值，否则 NaN。"""
    if not (_finite(numerator) and _finite(denominator)) or denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _log_ratio(numerator: float, denominator: float) -> float:
    if not (_finite(numerator) and _finite(denominator)) or numerator <= 0 or denominator <= 0:
        return float("nan")
    return float(np.log(numerator / denominator))


def _slot_arrays(slots: pd.DataFrame) -> dict[str, np.ndarray]:
    """取出 1..48 槽的 C/H/L/Q/A（越界槽为 NaN）。"""
    out: dict[str, np.ndarray] = {}
    for name in ("C", "H", "L", "Q", "A"):
        values = np.full(49, np.nan, dtype=float)
        for k in range(1, 49):
            values[k] = float(slots.loc[k, name])
        out[name] = values
    return out


def _interval_returns(arrays: Mapping[str, np.ndarray]) -> dict[int, float]:
    """r_k = ln(C_k/C_{k-1})，仅 k ∈ J（不含午休跨段）。"""
    close = arrays["C"]
    out: dict[int, float] = {}
    for k in J_SLOTS:
        out[k] = _log_ratio(close[k], close[k - 1])
    return out


def _impact(returns: Mapping[int, float], amounts: np.ndarray, slots: Iterable[int]) -> float:
    """`imp(S) = ln(1 + 1e8 * mean(|r_k|/A_k))`，至少 5 根正成交额槽位。"""
    values: list[float] = []
    for k in slots:
        amount = float(amounts[k])
        ret = float(returns.get(k, np.nan))
        if _finite(amount) and amount > 0 and _finite(ret):
            values.append(abs(ret) / amount)
    if len(values) < IMPACT_MIN_SLOTS:
        return float("nan")
    return float(np.log1p(AMOUNT_SCALE * float(np.mean(values))))


def day_metrics(slots: pd.DataFrame) -> dict[str, Any]:
    """单日分钟指标（含跨日基准所需的核心量额）。

    Returns:
        dict: 18 列中除 `m_late_amount_surprise` / `m_core_amount_surprise` 外的 16 列，
        另含 `A_core`、`q_late`（供 T 之前 20 日基准使用）与 `reasons`。
    """
    arrays = _slot_arrays(slots)
    close, high, low, quantity, amount = (
        arrays["C"],
        arrays["H"],
        arrays["L"],
        arrays["Q"],
        arrays["A"],
    )
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    metrics["m_am_ret"] = _log_ratio(close[24], close[1])
    metrics["m_pm_ret"] = _log_ratio(close[47], close[25])

    core_high = np.nanmax(high[CORE_SLOTS]) if np.isfinite(high[CORE_SLOTS]).any() else np.nan
    core_low = np.nanmin(low[CORE_SLOTS]) if np.isfinite(low[CORE_SLOTS]).any() else np.nan
    metrics["m_repair_ret"] = _log_ratio(close[47], core_low)

    path_close = close[1:48]
    if np.isfinite(path_close).sum() >= 2:
        running = np.maximum.accumulate(np.where(np.isfinite(path_close), path_close, -np.inf))
        with np.errstate(invalid="ignore", divide="ignore"):
            drawdown = np.where(
                np.isfinite(path_close) & (path_close > 0) & np.isfinite(running) & (running > 0),
                np.log(running / path_close),
                0.0,
            )
        metrics["m_path_dd"] = float(np.max(drawdown))
    else:
        metrics["m_path_dd"] = float("nan")

    if _finite(core_high) and _finite(core_low) and core_high > core_low:
        raw = (close[47] - core_low) / (core_high - core_low)
        if not _finite(raw):
            metrics["m_repair_fraction"] = float("nan")
        elif -1e-9 <= raw <= 1 + 1e-9:
            metrics["m_repair_fraction"] = float(min(1.0, max(0.0, raw)))
        else:
            raise FeatureError(
                f"m_repair_fraction 真实越界：{raw!r}（C47={close[47]}, L={core_low}, H={core_high}）"
            )
    else:
        metrics["m_repair_fraction"] = float("nan")

    low_values = low[CORE_SLOTS]
    if np.isfinite(low_values).any() and _finite(core_low):
        hits = np.where(np.isclose(low_values, core_low, rtol=0.0, atol=1e-12))[0]
        if len(hits):
            metrics["m_low_time"] = float((CORE_SLOTS[int(hits[0])] - 2) / 45.0)
        else:
            metrics["m_low_time"] = float("nan")
    else:
        metrics["m_low_time"] = float("nan")

    returns = _interval_returns(arrays)
    finite_returns = np.array([ret for ret in returns.values() if _finite(ret)], dtype=float)
    if len(finite_returns):
        metrics["m_rv"] = float(np.sqrt(np.sum(finite_returns**2)))
        squares = np.sum(finite_returns**2)
        downs = np.sum(np.minimum(finite_returns, 0.0) ** 2)
        metrics["m_down_var_share"] = float(downs / squares) if squares > 0 else float("nan")
    else:
        metrics["m_rv"] = float("nan")
        metrics["m_down_var_share"] = float("nan")

    total_amount = (
        float(np.nansum(amount[J_SLOTS])) if np.isfinite(amount[J_SLOTS]).any() else np.nan
    )
    if _finite(total_amount) and total_amount > 0:
        negative = [
            k
            for k in J_SLOTS
            if _finite(returns.get(k, np.nan)) and returns[k] < 0 and _finite(amount[k])
        ]
        metrics["m_neg_amount_share"] = float(sum(amount[k] for k in negative) / total_amount)
        ranked = sorted(negative, key=lambda k: (returns[k], k))[:3]
        metrics["m_drop3_amount_share"] = float(sum(amount[k] for k in ranked) / total_amount)
    else:
        metrics["m_neg_amount_share"] = float("nan")
        metrics["m_drop3_amount_share"] = float("nan")

    metrics["m_impact_am"] = _impact(returns, amount, AM_SLOTS)
    metrics["m_impact_pm"] = _impact(returns, amount, PM_SLOTS)
    if _finite(metrics["m_impact_am"]) and _finite(metrics["m_impact_pm"]):
        metrics["m_impact_repair"] = float(metrics["m_impact_am"] - metrics["m_impact_pm"])
    else:
        metrics["m_impact_repair"] = float("nan")

    core_amount = (
        float(np.nansum(amount[CORE_SLOTS])) if np.isfinite(amount[CORE_SLOTS]).any() else np.nan
    )
    core_quantity = (
        float(np.nansum(quantity[CORE_SLOTS]))
        if np.isfinite(quantity[CORE_SLOTS]).any()
        else np.nan
    )
    if _finite(core_amount) and _finite(core_quantity) and core_amount > 0 and core_quantity > 0:
        metrics["m_close_vwap_gap"] = _log_ratio(close[47], core_amount / core_quantity)
    else:
        metrics["m_close_vwap_gap"] = float("nan")

    pm_amount = float(np.nansum(amount[25:48])) if np.isfinite(amount[25:48]).any() else np.nan
    metrics["m_pm_amount_share"] = _ratio(pm_amount, core_amount)
    if _finite(core_amount) and core_amount > 0:
        shares = np.where(np.isfinite(amount[CORE_SLOTS]), amount[CORE_SLOTS], 0.0) / core_amount
        metrics["m_amount_hhi"] = float(np.sum(shares**2))
    else:
        metrics["m_amount_hhi"] = float("nan")

    late_amount = (
        float(np.nansum(amount[LATE_SLOTS])) if np.isfinite(amount[LATE_SLOTS]).any() else np.nan
    )
    metrics["A_core"] = core_amount
    metrics["q_late"] = _ratio(late_amount, core_amount)
    metrics["reasons"] = reasons
    for name in spec.MINUTE_FEATURES:
        if name not in metrics:
            metrics[name] = float("nan")
    return metrics


def _median_or_nan(values: Sequence[float]) -> tuple[float, int]:
    """过去 20 日基准：至少 16 个有效观察才取中位数。"""
    finite = [float(v) for v in values if _finite(float(v))]
    if len(finite) < HISTORY_MIN_VALID:
        return float("nan"), len(finite)
    return float(np.median(finite)), len(finite)


def cross_day_features(
    today: Mapping[str, Any], history: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """跨日分钟特征（只用 T 之前的 20 个交易日，不含 T）。"""
    window = list(history)[-HISTORY_WINDOW:]
    late_median, late_n = _median_or_nan([item.get("q_late", np.nan) for item in window])
    core_median, core_n = _median_or_nan([item.get("A_core", np.nan) for item in window])
    out: dict[str, Any] = {
        "m_late_amount_surprise": float("nan"),
        "m_core_amount_surprise": float("nan"),
        "late_history_valid": late_n,
        "core_history_valid": core_n,
    }
    if _finite(today.get("q_late", np.nan)) and _finite(late_median):
        out["m_late_amount_surprise"] = float(today["q_late"] - late_median)
    if (
        _finite(today.get("A_core", np.nan))
        and today["A_core"] > 0
        and _finite(core_median)
        and core_median > 0
    ):
        out["m_core_amount_surprise"] = float(np.log(today["A_core"] / core_median))
    return out


def minute_feature_row(slots: pd.DataFrame, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """单股单日的 18 个分钟列 + 状态（`minute_informed` 判据在此实现）。"""
    complete, missing = sessions.session_complete(slots)
    metrics = {k: v for k, v in day_metrics(slots).items()}
    metrics.update(cross_day_features(metrics, history))
    finite_count = sum(
        1 for name in spec.MINUTE_FEATURES if _finite(float(metrics.get(name, np.nan)))
    )
    informed = bool(complete and finite_count >= spec.MIN_INFORMED_MINUTE_COLUMNS)
    status = "informed"
    if not complete:
        status = "session_incomplete"
    elif finite_count < spec.MIN_INFORMED_MINUTE_COLUMNS:
        status = "insufficient_minute_columns"
    row = {name: float(metrics.get(name, np.nan)) for name in spec.MINUTE_FEATURES}
    row.update(
        {
            "session_complete": bool(complete),
            "missing_slots": ",".join(str(k) for k in missing),
            "minute_finite_columns": int(finite_count),
            "minute_informed": informed,
            "minute_status": status,
            "A_core": float(metrics.get("A_core", np.nan)),
            "q_late": float(metrics.get("q_late", np.nan)),
        }
    )
    return row


def minute_series(
    day_frames: Mapping[str, pd.DataFrame],
    *,
    convention: str | None = None,
    trading_days: Sequence[str] | None = None,
) -> pd.DataFrame:
    """按交易日顺序计算单只股票的分钟特征时间序列。

    Args:
        day_frames: `{YYYYMMDD: raw rows}`；缺失日期不出现在映射中（日历不压缩）。
        convention: 供应商时间戳约定；缺省取冻结值。
        trading_days: 完整交易日序列；提供时按该序列滚动历史基准。

    Returns:
        DataFrame（index=trade_date）含 18 列 + 状态列 + `A_core`/`q_late`。
    """
    days = list(trading_days) if trading_days is not None else sorted(day_frames)
    history: deque[dict[str, Any]] = deque(maxlen=HISTORY_WINDOW)
    rows: list[dict[str, Any]] = []
    for day in days:
        frame = day_frames.get(str(day))
        if frame is None:
            # 当日缺分钟数据：保留行、全 NaN 并显式原因，历史窗口**不前填**
            row = {name: float("nan") for name in spec.MINUTE_FEATURES}
            row.update(
                {
                    "trade_date": str(day),
                    "session_complete": False,
                    "missing_slots": "all",
                    "minute_finite_columns": 0,
                    "minute_informed": False,
                    "minute_status": "no_minute_source",
                    "A_core": float("nan"),
                    "q_late": float("nan"),
                }
            )
            rows.append(row)
            history.append({"q_late": float("nan"), "A_core": float("nan")})
            continue
        normalized = sessions.normalize_day(frame, convention=convention, trade_date=str(day))
        slots = normalized["slots"]
        row = minute_feature_row(slots, list(history))
        row["trade_date"] = str(day)
        rows.append(row)
        history.append({"q_late": row["q_late"], "A_core": row["A_core"]})
    frame = pd.DataFrame(rows)
    columns = (
        ["trade_date"]
        + list(spec.MINUTE_FEATURES)
        + [
            "session_complete",
            "missing_slots",
            "minute_finite_columns",
            "minute_informed",
            "minute_status",
            "A_core",
            "q_late",
        ]
    )
    return frame[[name for name in columns if name in frame.columns]]


def daily_controls(
    close_adj: pd.DataFrame,
    amount: pd.DataFrame,
    turnover_rate: pd.DataFrame,
    total_mv: pd.DataFrame,
) -> pd.DataFrame:
    """6 个日线控制列（宽表 -> 长表，index=`(trade_date, ts_code)`）。

    口径（协议 §6.1/§6.2）：
    - `d_ret_1` / `d_ret_5`：真实端点存在才成立（停牌不压缩日历）；
    - `d_vol_20` / `d_turnover_20`：截至 T 的 20 个交易日，至少 16 个有效；
    - `d_amount_rel_20`：T 与前 20 个交易日（不含 T）均值之比，至少 16 个有效。
    """
    close = close_adj.astype(float)
    amount_yuan = amount.astype(float)  # 调用方负责先转成元
    returns = close / close.shift(1) - 1.0
    ret_1 = returns
    ret_5 = close / close.shift(5) - 1.0
    vol_20 = returns.rolling(20, min_periods=HISTORY_MIN_VALID).std(ddof=1)
    prior_amount = (
        amount_yuan.shift(1).rolling(HISTORY_WINDOW, min_periods=HISTORY_MIN_VALID).mean()
    )
    amount_rel = np.log(amount_yuan.where(amount_yuan > 0) / prior_amount.where(prior_amount > 0))
    turnover = (
        (turnover_rate.astype(float) / 100.0)
        .rolling(HISTORY_WINDOW, min_periods=HISTORY_MIN_VALID)
        .mean()
    )
    log_mv = np.log(total_mv.astype(float).where(total_mv.astype(float) > 0))

    frames = {
        "d_ret_1": ret_1,
        "d_ret_5": ret_5,
        "d_vol_20": vol_20,
        "d_amount_rel_20": amount_rel,
        "d_turnover_20": turnover,
        "d_log_mv": log_mv,
    }
    pieces = []
    for name, frame in frames.items():
        stacked = frame.stack(future_stack=True).rename(name)
        pieces.append(stacked)
    out = pd.concat(pieces, axis=1)
    out.index.names = ["trade_date", "ts_code"]
    return out.reset_index()


def feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """逐列覆盖（有限值占比）与分钟状态计数。"""
    rows = []
    for name in spec.FEATURES:
        series = pd.to_numeric(frame.get(name), errors="coerce")
        rows.append(
            {
                "feature": name,
                "finite": int(np.isfinite(series.to_numpy(dtype=float)).sum()),
                "rows": int(len(frame)),
                "share": (
                    float(np.isfinite(series.to_numpy(dtype=float)).mean()) if len(frame) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def assert_feature_columns(frame: pd.DataFrame, features: Sequence[str] = spec.FEATURES) -> None:
    """模型列白名单严格校验：缺整列即报错，不得自动减列后继续叫 v1。"""
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise FeatureError(f"特征矩阵缺列（禁止自动降列）：{missing}")
