# -*- coding: utf-8 -*-
"""P7 正式 T2/T3（IDX-IN300）：E1~E4、独立夏普区间、三口径相关上界与完整复合门。

判据（预登记冻结，``prereg.json`` 先行落盘）
==========================================
- **E1 信号层尺子**：逐事件毛残存 ``d_i = r_i − r_bench,b``（未成交按现金 0 收益），
  统计量 ``D = mean(d_i)``；组内伪事件随机化 10,000 次（seed 42），
  门槛 ``Q_{1−α/(核心格数·λ)}``（α=0.05、1 格、λ=1 ⇒ 0.95）；
- **E2 制度重排**：月度配对超额按 3 个月循环块重排 10,000 次，报告年化超额分布；
  下限 ≤0 视为结果脆弱（冻结），点估计 ≤0 且功效足够记 fail；
- **T2 组合级尺子**：配对 10 日块自举的 ``ΔMaxDD`` 分布，按 **ES3 既有规则**在
  2.5% 分位上下判（门槛常量不变，读数由点估计升级为区间下界）；
- **Q1 独立夏普**：候选独立日收益 10 日块自举 95% 下界 ``L_B``，与零假设流
  （组内伪事件 + 同一执行模拟器）绝对夏普 95 分位 ``Q_0.95(S^0)`` 比较；
  A/A 校准（模拟器 vs 实际账户）超容差 ⇒ 冻结；MDE 注入网格无 ≥80% 功效 ⇒ 冻结；
- **三口径相关**：全样本 / 10 日非重叠 / 崩盘段，97.5% 上界均须 < 0.3；
- **E4 五条否决线**：崩盘窗 ΔMaxDD≥0、相关上界、实际成交容量、月利用率中位数≥50%、
  坏消息侧与 PIT 可归因；
- **完整复合门** = 上述全部合取；``pass`` 另需用户组织的独立存活评审（本程序不替代）。

边界：本程序不接生产、不产生前向/日历验证结论；A 参照为历史净收益（H 级证据）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.v2_p7.f2 import (
    BLOCK_Q1,
    COMPONENT_QUANTILE,
    CORR_UPPER,
    DELTA_SHARPE_BLOCK,
    ES3_HARD_PP,
    ES3_TOL_PP,
    LAMBDA_FROZEN,
    MDE_GRID,
    MIN_POWER,
    MIN_SHARPE_BLOCKS,
    MIN_SHARPE_OBSERVATIONS,
    MIN_TICKET_CNY,
    NULL_PATHS,
    NULL_SEED,
    RESAMPLES,
    RF_ANNUAL,
    SEED_MAIN,
    SLEEVE_ENHANCEMENT,
    SLEEVE_RISK_ADJUSTED,
    SLEEVE_TYPES,
    TRADING_DAYS,
    UTIL_MEDIAN_MIN,
    VERDICT_FAIL,
    VERDICT_FROZEN,
    VERDICT_PASS,
)
from scripts.v2_p7.f2.panel import BatchPanel, simulate
from scripts.v2_p7.p4.metrics import corr_line, monthly_returns


# ────────────────────────── 基础统计 ──────────────────────────


def annualized_sharpe(returns: np.ndarray, rf: float = RF_ANNUAL) -> Optional[float]:
    """日收益年化夏普（rf 年化 / 252；零波动或样本不足返回 None）。"""
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 30:
        return None
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return None
    return float((values.mean() - rf / TRADING_DAYS) / std * math.sqrt(TRADING_DAYS))


def block_bootstrap_sharpe_lower(
    returns: np.ndarray,
    *,
    block: int = BLOCK_Q1,
    resamples: int = RESAMPLES,
    seed: int = SEED_MAIN,
    quantile: float = 0.025,
) -> Dict[str, Any]:
    """循环块自举的夏普分位（返回下界、点估计与有效样本数）。"""
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    point = annualized_sharpe(values)
    if n < 60 or point is None:
        return {"n": n, "point": point, "lower": None, "upper": None, "resamples": 0}
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / block))
    offsets = np.arange(block)
    sharpes: List[float] = []
    chunk = 500
    for _ in range(math.ceil(resamples / chunk)):
        size = min(chunk, resamples - len(sharpes))
        starts = rng.integers(0, n, size=(size, blocks))
        indices = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(size, -1)[:, :n]
        sample = values[indices]
        mean = sample.mean(axis=1)
        std = sample.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (mean - RF_ANNUAL / TRADING_DAYS) / std * math.sqrt(TRADING_DAYS)
        sharpes.extend(ratio[np.isfinite(ratio)].tolist())
    values_out = np.asarray(sharpes, dtype=float)
    if values_out.size == 0:
        return {"n": n, "point": point, "lower": None, "upper": None, "resamples": 0}
    return {
        "n": n,
        "point": float(point),
        "lower": float(np.quantile(values_out, quantile)),
        "upper": float(np.quantile(values_out, 1.0 - quantile)),
        "resamples": int(values_out.size),
    }


def block_bootstrap_delta_maxdd(
    a_only_returns: np.ndarray,
    combined_returns: np.ndarray,
    *,
    block: int = 10,
    resamples: int = RESAMPLES,
    seed: int = SEED_MAIN,
    tail: float = 0.025,
) -> Dict[str, Any]:
    """配对块自举的 ``ΔMaxDD = MaxDD(combined) − MaxDD(a_only)`` 分布（pp，改善为正）。

    ``tail`` 为下尾概率（默认 0.025 保持既有行为）；型 II 传 ``α_c/2``。
    """
    a = np.asarray(a_only_returns, dtype=float)
    c = np.asarray(combined_returns, dtype=float)
    if a.shape != c.shape:
        raise ValueError("配对收益长度不一致")
    mask = np.isfinite(a) & np.isfinite(c)
    a, c = a[mask], c[mask]
    n = int(a.size)
    if n < 60:
        return {"n": n, "point": None, "lower": None, "upper": None, "resamples": 0}
    point = (_maxdd(c) - _maxdd(a)) * 100.0
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / block))
    offsets = np.arange(block)
    deltas: List[float] = []
    chunk = 200
    for _ in range(math.ceil(resamples / chunk)):
        size = min(chunk, resamples - len(deltas))
        starts = rng.integers(0, n, size=(size, blocks))
        indices = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(size, -1)[:, :n]
        a_path = np.cumprod(1.0 + a[indices], axis=1)
        c_path = np.cumprod(1.0 + c[indices], axis=1)
        a_dd = (a_path / np.maximum.accumulate(a_path, axis=1) - 1.0).min(axis=1)
        c_dd = (c_path / np.maximum.accumulate(c_path, axis=1) - 1.0).min(axis=1)
        deltas.extend(((c_dd - a_dd) * 100.0).tolist())
    values = np.asarray(deltas, dtype=float)
    return {
        "n": n,
        "point": float(point),
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
        "tail": float(tail),
        "resamples": int(values.size),
        "distribution": values,
    }


def block_bootstrap_sharpe_delta(
    a_only_returns: np.ndarray,
    combined_returns: np.ndarray,
    *,
    block: int = DELTA_SHARPE_BLOCK,
    resamples: int = RESAMPLES,
    seed: int = SEED_MAIN,
    tail: float = 0.025,
) -> Dict[str, Any]:
    """型 II 主门：配对块自举的 ``ΔSharpe = Sharpe(combined) − Sharpe(a_only)``。

    下尾 = ``Q_tail``（用于"下界 > 0"），上尾 = ``Q_(1−tail)``。rf=0。
    """
    a = np.asarray(a_only_returns, dtype=float)
    c = np.asarray(combined_returns, dtype=float)
    if a.shape != c.shape:
        raise ValueError("收益长度不一致")
    mask = np.isfinite(a) & np.isfinite(c)
    a, c = a[mask], c[mask]
    n = int(a.size)
    empty = {"n": n, "point": None, "lower": None, "upper": None, "resamples": 0}
    if n < MIN_SHARPE_OBSERVATIONS or n // block < MIN_SHARPE_BLOCKS:
        return {**empty, "reason": "insufficient_sample"}
    a_std, c_std = a.std(ddof=1), c.std(ddof=1)
    if a_std <= 0.0 or c_std <= 0.0:
        return {**empty, "reason": "zero_volatility"}
    scale = math.sqrt(TRADING_DAYS)
    point = float((c.mean() / c_std - a.mean() / a_std) * scale)
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / block))
    offsets = np.arange(block)
    deltas: List[float] = []
    chunk = 200
    for _ in range(math.ceil(resamples / chunk)):
        size = min(chunk, resamples - len(deltas))
        starts = rng.integers(0, n, size=(size, blocks))
        indices = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(size, -1)[:, :n]
        a_path = a[indices]
        c_path = c[indices]
        a_mu, a_sd = a_path.mean(axis=1), a_path.std(axis=1, ddof=1)
        c_mu, c_sd = c_path.mean(axis=1), c_path.std(axis=1, ddof=1)
        valid = (a_sd > 0.0) & (c_sd > 0.0)
        sharpe = (c_mu[valid] / c_sd[valid] - a_mu[valid] / a_sd[valid]) * scale
        deltas.extend(sharpe.tolist())
    values = np.asarray(deltas, dtype=float)
    if values.size < 100:
        return {**empty, "reason": "degenerate_bootstrap"}
    return {
        "n": n,
        "blocks": int(n // block),
        "point": point,
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
        "tail": float(tail),
        "resamples": int(values.size),
        "distribution": values,
        "base_sharpe": float(a.mean() / a_std * scale),
        "combined_sharpe": float(c.mean() / c_std * scale),
        "rule": "配对 10 日循环块自举；下尾 Q_tail > 0 才通过（rf=0）",
    }


def _maxdd(returns: np.ndarray) -> float:
    path = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1.0).min())


def block_bootstrap_mean(
    values: np.ndarray,
    *,
    block: int,
    resamples: int = RESAMPLES,
    seed: int = SEED_MAIN,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> Dict[str, Any]:
    """循环块自举的均值分布（E2 制度重排）。"""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    n = int(data.size)
    if n < 12:
        return {"n": n, "point": None, "quantiles": None, "share_le_zero": None}
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / block))
    offsets = np.arange(block)
    means: List[float] = []
    chunk = 500
    for _ in range(math.ceil(resamples / chunk)):
        size = min(chunk, resamples - len(means))
        starts = rng.integers(0, n, size=(size, blocks))
        indices = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(size, -1)[:, :n]
        means.extend(data[indices].mean(axis=1).tolist())
    sample = np.asarray(means, dtype=float)
    return {
        "n": n,
        "point": float(data.mean()),
        "quantiles": {str(q): float(np.quantile(sample, q)) for q in quantiles},
        "share_le_zero": float((sample <= 0.0).mean()),
        "resamples": int(sample.size),
    }


# ────────────────────────── E1 信号层尺子 ──────────────────────────


def event_excess(
    panels: Mapping[str, BatchPanel], selection: Mapping[str, Sequence[str]]
) -> Dict[str, Any]:
    """逐事件毛残存 ``d_i = r_i − r_bench,b``（未成交按现金 0 收益计入）。"""
    rows: List[Dict[str, Any]] = []
    for notice_id in sorted(panels):
        panel = panels[notice_id]
        codes = list(selection.get(notice_id, ()))
        if not codes:
            continue
        for code in codes:
            pos = panel.codes.index(code) if code in panel.codes else -1
            if pos < 0:
                continue
            gross = _event_gross(panel, pos)
            rows.append(
                {
                    "notice_id": notice_id,
                    "ts_code": code,
                    "filled": gross is not None,
                    "gross_return": gross,
                    "bench_return": float(panel.bench_return),
                    "excess": (0.0 if gross is None else float(gross))
                    - float(panel.bench_return),
                }
            )
    excess = np.asarray([row["excess"] for row in rows], dtype=float)
    return {
        "events": len(rows),
        "filled": int(sum(1 for row in rows if row["filled"])),
        "mean_excess": float(excess.mean()) if excess.size else None,
        "rows": rows,
    }


def _event_gross(panel: BatchPanel, pos: int) -> Optional[float]:
    """单事件毛收益（未成交返回 None）：入场开盘 → 到期收盘/顺延开盘 + 公司行为。"""
    if not bool(panel.entry_tradable[pos]):
        return None
    entry = float(panel.open_raw[pos, 0])
    if not np.isfinite(entry) or entry <= 0.0:
        return None
    exit_price = None
    exit_pos = -1
    flow_pos = panel.days.index(panel.flow_date) if panel.flow_date in panel.days else -1
    for day_pos in range(max(flow_pos, 1), panel.n_days):
        if day_pos == flow_pos:
            if bool(panel.sellable_close[pos, day_pos]):
                exit_price = float(panel.close_raw[pos, day_pos])
                exit_pos = day_pos
                break
            continue
        if bool(panel.sell_open_ok[pos, day_pos]):
            exit_price = float(panel.open_raw[pos, day_pos])
            exit_pos = day_pos
            break
    if exit_price is None or not np.isfinite(exit_price) or exit_price <= 0.0:
        last = float(panel.close_raw[pos, :][np.isfinite(panel.close_raw[pos, :])][-1])
        exit_price = last if np.isfinite(last) and last > 0.0 else entry
        exit_pos = panel.n_days - 1
    ratio = 0.0
    cash = 0.0
    for event in panel.corp[pos]:
        if 0 < event.day_index <= exit_pos:
            ratio += event.share_ratio
            cash += event.cash_per_share
    total = exit_price * (1.0 + ratio) + cash
    return float(total / entry - 1.0)


def e1_component(
    panels: Mapping[str, BatchPanel],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED_MAIN,
    quantile: Optional[float] = None,
) -> Dict[str, Any]:
    """真实选择 vs 组内伪事件随机化（门槛 = 1−α_c/(格数·λ) 分位）。

    ``quantile`` 由预登记的 ``pool`` 驱动（``1 − 0.05/pool``）；不传则用冻结默认值。
    """
    real = event_excess(panels, {n: p.selected for n, p in panels.items()})
    if real["mean_excess"] is None:
        return {"estimable": False, "reason": "empty_event_set"}
    sizes = {notice: len(panel.selected) for notice, panel in panels.items()}
    rng = np.random.default_rng(seed)
    null = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selection: Dict[str, List[str]] = {}
        for notice_id, panel in panels.items():
            k = min(sizes[notice_id], panel.n_codes)
            if k <= 0:
                selection[notice_id] = []
                continue
            picked = rng.choice(panel.n_codes, size=k, replace=False)
            selection[notice_id] = [panel.codes[pos] for pos in picked]
        drawn = event_excess(panels, selection)
        null[index] = float(drawn["mean_excess"] or 0.0)
    q = COMPONENT_QUANTILE if quantile is None else float(quantile)
    threshold = float(np.quantile(null, q))
    statistic = float(real["mean_excess"])
    return {
        "estimable": True,
        "statistic": statistic,
        "threshold": threshold,
        "quantile": q,
        "alpha_component": float(1.0 - q) * (1 * LAMBDA_FROZEN),
        "lambda": LAMBDA_FROZEN,
        "pass": bool(statistic > threshold),
        "null": {
            "resamples": int(len(null)),
            "mean": float(null.mean()),
            "p05": float(np.quantile(null, 0.05)),
            "p50": float(np.quantile(null, 0.50)),
            "p95": float(np.quantile(null, 0.95)),
            "share_above_statistic": float((null >= statistic).mean()),
        },
        "events": real["events"],
        "filled": real["filled"],
        "event_rows": real["rows"],
        "power": _e1_power(null, statistic, threshold),
        "bench_rule": "逐批同域等权毛收益（P3 §3）；事件未成交按现金 0 收益计",
    }


def _e1_power(null: np.ndarray, statistic: float, threshold: float) -> Dict[str, Any]:
    """负向功效：真实统计量被注入位移 δ_R·σ(D^0) 后超过门槛的比率（δ_R 网格）。"""
    sigma = float(null.std(ddof=1)) if null.size > 2 else 0.0
    out: Dict[str, Any] = {"sigma": sigma, "grid": {}}
    for delta in (0.5, 1.0, 2.0, 4.0):
        shifted = float(statistic) + float(delta) * sigma
        out["grid"][str(delta)] = bool(shifted > threshold)
    out["detectable_any"] = bool(any(out["grid"].values()))
    return out


# ────────────────────────── Q1 独立夏普 ──────────────────────────


def q1_sharpe(
    panels: Mapping[str, BatchPanel],
    *,
    real_returns: np.ndarray,
    principal: float,
    calendar: Optional[Sequence[str]] = None,
    null_paths: int = NULL_PATHS,
    seed: int = NULL_SEED,
) -> Dict[str, Any]:
    """``L_B`` vs ``Q_0.95(S^0)``，附 A/A 校准与 MDE 注入。"""
    lower = block_bootstrap_sharpe_lower(real_returns)
    if lower["lower"] is None:
        return {"estimable": False, "reason": "insufficient_returns", "lower": lower}
    real_sim = simulate(
        panels, {n: p.selected for n, p in panels.items()}, principal=principal
    )
    sim_start = float(real_sim["nav"].iloc[0])
    if not _sim_start_ok(sim_start, principal):
        # 模拟器首行 = 首个交易日收盘 NAV；偏离期初本金 >20% 视为日历错位，禁止 A/A
        return {"estimable": False, "reason": "sim_start_mismatch", "lower": lower}
    real_sim_returns = _sim_returns(real_sim, calendar)
    real_sim_sharpe = annualized_sharpe(real_sim_returns)
    # 与 P4 ``table1.metrics.total_return`` 同口径：分母一律取**期初本金**
    # （模拟器首行已含首日损益，用 nav[0] 作分母会整段错位一个交易日）
    real_sim_total = float(real_sim["nav"].iloc[-1] / float(principal) - 1.0)
    actual_total = float(np.prod(1.0 + real_returns) - 1.0)
    aa_delta_return = float(real_sim_total - actual_total)
    aa_delta_sharpe = (
        None
        if real_sim_sharpe is None
        else float(real_sim_sharpe - float(lower["point"] or 0.0))
    )
    aa_ok = bool(
        abs(aa_delta_return) <= 0.02
        and aa_delta_sharpe is not None
        and abs(aa_delta_sharpe) <= 0.05
    )
    rng = np.random.default_rng(seed)
    sizes = {notice: len(panel.selected) for notice, panel in panels.items()}
    null_sharpes: List[float] = []
    for _ in range(int(null_paths)):
        selection: Dict[str, List[str]] = {}
        for notice_id, panel in panels.items():
            k = min(sizes[notice_id], panel.n_codes)
            if k <= 0:
                selection[notice_id] = []
                continue
            picked = rng.choice(panel.n_codes, size=k, replace=False)
            selection[notice_id] = [panel.codes[pos] for pos in picked]
        frame = simulate(panels, selection, principal=principal)
        rets = _sim_returns(frame, calendar)
        value = annualized_sharpe(rets)
        if value is not None:
            null_sharpes.append(float(value))
    if len(null_sharpes) < 30:
        return {"estimable": False, "reason": "null_stream_insufficient", "lower": lower}
    null = np.asarray(null_sharpes, dtype=float)
    threshold = float(np.quantile(null, 0.95))
    gate = bool(float(lower["lower"]) > threshold)
    mde = _q1_mde(real_returns, threshold)
    return {
        "estimable": True,
        "L_B": float(lower["lower"]),
        "sharpe_point": float(lower["point"]),
        "Q_095_null": threshold,
        "null": {
            "paths": int(null.size),
            "mean": float(null.mean()),
            "p50": float(np.quantile(null, 0.50)),
            "p95": threshold,
            "max": float(null.max()),
        },
        "pass": bool(gate and aa_ok and mde["power_ok"]),
        "gate_raw": gate,
        "aa_calibration": {
            "real_sim_sharpe": real_sim_sharpe,
            "real_account_sharpe": float(lower["point"]),
            "delta_sharpe": aa_delta_sharpe,
            "real_sim_total_return": real_sim_total,
            "real_account_total_return": actual_total,
            "delta_total_return": aa_delta_return,
            "tolerance": {"total_return": 0.02, "sharpe": 0.05},
            "ok": aa_ok,
        },
        "mde": mde,
        "rule": "L_B > Q_0.95(S^0)；A/A 校准与 MDE 功效任一不满足即冻结",
    }


def _sim_start_ok(sim_start: float, principal: float) -> bool:
    """模拟器首行 = 首个交易日收盘 NAV；与期初本金偏离 >20% 视为日历错位。"""
    if sim_start <= 0.0 or principal <= 0.0:
        return False
    return abs(float(sim_start) / float(principal) - 1.0) <= 0.2


def _sim_returns(frame: pd.DataFrame, calendar: Optional[Sequence[str]]) -> np.ndarray:
    """模拟器 NAV → 与候选账户**同一日历**的日收益（空窗前向填充，零收益保留）。"""
    series = frame.set_index("trade_date")["nav"].astype(float)
    if calendar is not None and len(calendar) > 0:
        index = pd.Index([str(day) for day in calendar])
        series = series.reindex(index.union(series.index)).sort_index().ffill().bfill()
        series = series.loc[index]
    return series.pct_change().dropna().to_numpy(dtype=float)


def _q1_mde(real_returns: np.ndarray, threshold: float) -> Dict[str, Any]:
    """注入 ``δ_S·σ_d/√252`` 后的检出率网格（≥80% 的最小 δ_S 即 MDE）。"""
    values = np.asarray(real_returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 60:
        return {"power_ok": False, "reason": "insufficient_returns", "grid": {}}
    sigma = float(values.std(ddof=1))
    grid: Dict[str, Any] = {}
    mde = None
    for delta in MDE_GRID:
        injected = values + float(delta) * sigma / math.sqrt(TRADING_DAYS)
        detected = 0
        for seed in range(1, 201):
            line = block_bootstrap_sharpe_lower(injected, seed=seed, resamples=200)
            if line["lower"] is not None and float(line["lower"]) > threshold:
                detected += 1
        rate = detected / 200.0
        grid[str(delta)] = float(rate)
        if mde is None and rate >= MIN_POWER:
            mde = float(delta)
    return {
        "sigma_daily": sigma,
        "grid": grid,
        "mde": mde,
        "power_ok": bool(mde is not None),
        "rule": "网格内无 ≥80% 检出率 ⇒ inconclusive_frozen（不得记 fail）",
    }


# ────────────────────────── E2 制度重排 ──────────────────────────


def e2_regime(
    candidate_levels: pd.Series,
    benchmark_levels: pd.Series,
    start_date: str,
    *,
    block_months: int = 3,
) -> Dict[str, Any]:
    """月度配对超额的制度重排分布（排列敏感性，不是新制度生成器）。"""
    candidate = monthly_returns(candidate_levels, start_date)
    benchmark = monthly_returns(benchmark_levels, start_date)
    joined = pd.concat([candidate.rename("c"), benchmark.rename("b")], axis=1, join="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < 12:
        return {"estimable": False, "reason": "insufficient_months", "months": int(len(joined))}
    excess = (joined["c"] - joined["b"]).to_numpy(dtype=float)
    stats = block_bootstrap_mean(excess, block=block_months)
    q = stats["quantiles"] or {}
    lower_pp = 12.0 * 100.0 * float(q.get("0.05", float("nan")))
    point_pp = 12.0 * 100.0 * float(stats["point"] or float("nan"))
    if point_pp <= 0.0:
        state = "fail"
    elif lower_pp > 0.0:
        state = "pass"
    else:
        state = "frozen"
    return {
        "estimable": True,
        "months": int(len(excess)),
        "monthly_mean_excess": float(stats["point"] or 0.0),
        "annualized_pp": point_pp,
        "annualized_lower_pp": lower_pp,
        "annualized_upper_pp": 12.0 * 100.0 * float(q.get("0.95", float("nan"))),
        "share_le_zero": stats["share_le_zero"],
        "state": state,
        "pass": bool(state == "pass"),
        "rule": (
            "同一 ES1 统计量（12×月度均值超额）在 3 个月循环块重排下的 5% 分位 >0 记 pass；"
            "点估计 ≤0 记 fail；区间跨 0 记 frozen（结果脆弱，不动 ES1 阈值）"
        ),
    }


# ────────────────────────── 三口径相关 ──────────────────────────


def three_correlations(
    a_nav: pd.Series,
    candidate_levels: pd.Series,
    crash_windows: Sequence[Mapping[str, Any]],
    *,
    common_start: str,
    common_end: str,
) -> Dict[str, Any]:
    """全样本 / 10 日非重叠 / 崩盘段条件相关（97.5% 上界，口径同 P3 表二）。"""
    a = pd.to_numeric(a_nav, errors="coerce").dropna()
    c = pd.to_numeric(candidate_levels, errors="coerce").dropna()
    common = a.index[(a.index >= common_start) & (a.index <= common_end) & a.index.isin(c.index)]
    if len(common) < 2:
        return {"estimable": False, "reason": "common_window_insufficient"}
    a_levels = a.loc[common]
    c_levels = c.reindex(common).ffill()
    a_ret = a_levels.pct_change().dropna()
    c_ret = c_levels.pct_change().dropna()
    joined = pd.concat([a_ret.rename("a"), c_ret.rename("c")], axis=1, join="inner").dropna()
    full = corr_line(joined["a"], joined["c"], block=10, min_observations=250)
    if len(joined) >= 40:
        from scripts.v2_p7.p4.metrics import compound_blocks

        a_blocks = compound_blocks(joined["a"].to_numpy(dtype=float), 10)
        c_blocks = compound_blocks(joined["c"].to_numpy(dtype=float), 10)
        rebalance = corr_line(
            pd.Series(a_blocks), pd.Series(c_blocks), block=1, min_observations=40
        )
    else:
        rebalance = {"estimable": False, "point": None, "upper_975": None, "n": 0}
    crash_a: List[float] = []
    crash_c: List[float] = []
    windows: Dict[str, Any] = {}
    for window in crash_windows:
        start = str(window["start_date"]).replace("-", "")
        end = str(window["end_date"]).replace("-", "")
        part = joined.loc[(joined.index >= start) & (joined.index <= end)]
        windows[f"{window.get('year')}_{start}_{end}"] = {"paired_days": int(len(part))}
        crash_a.extend(part["a"].tolist())
        crash_c.extend(part["c"].tolist())
    crash = corr_line(pd.Series(crash_a), pd.Series(crash_c), block=5, min_observations=60)
    crash["pairable_windows"] = int(sum(1 for item in windows.values() if item["paired_days"] > 0))
    if crash["pairable_windows"] < 2:
        crash["estimable"] = False
    lines = {"full_sample": full, "rebalance_10d": rebalance, "crash_segment": crash}
    points = [lines[key]["point"] for key in lines]
    uppers = [lines[key]["upper_975"] for key in lines]
    estimable = all(bool(lines[key]["estimable"]) for key in lines)
    finite_upper = [float(value) for value in uppers if value is not None and np.isfinite(value)]
    return {
        "estimable": bool(estimable),
        "common_window": [str(common[0]), str(common[-1])],
        "common_days": int(len(common)),
        "lines": lines,
        "windows": windows,
        "point_max": max((float(v) for v in points if v is not None), default=None),
        "upper_max": max(finite_upper) if finite_upper else None,
        "upper_bound": CORR_UPPER,
        "pass": bool(estimable and finite_upper and max(finite_upper) < CORR_UPPER),
        "rule": "三口径 97.5% 上界均须 < 0.3；任一不可估计即冻结（不补 0、不删窗口）",
    }


# ────────────────────────── E4 五条否决线 ──────────────────────────


def e4_vetoes(
    *,
    p4_root: Any,
    table3_levels: pd.DataFrame,
    crash_windows: Sequence[Mapping[str, Any]],
    correlations: Mapping[str, Any],
    negative_side_boundary: str,
) -> Dict[str, Any]:
    """E4-1~E4-5（逐条可复算；任一触发即 veto）。"""
    from pathlib import Path

    root = Path(p4_root)
    levels = table3_levels.copy()
    levels.index = levels.index.astype(str)
    crash_lines: List[Dict[str, Any]] = []
    for window in crash_windows:
        start = str(window["start_date"]).replace("-", "")
        end = str(window["end_date"]).replace("-", "")
        part = levels.loc[(levels.index >= start) & (levels.index <= end)]
        if len(part) < 2:
            crash_lines.append(
                {"window": [start, end], "pairable": False, "delta_maxdd_pp": None}
            )
            continue
        a_dd = _series_maxdd(part["a_only"])
        c_dd = _series_maxdd(part["combined"])
        crash_lines.append(
            {
                "window": [start, end],
                "pairable": True,
                "a_only_maxdd": float(a_dd),
                "combined_maxdd": float(c_dd),
                "delta_maxdd_pp": float((c_dd - a_dd) * 100.0),
                "pass": bool((c_dd - a_dd) >= 0.0),
            }
        )
    pairable = [row for row in crash_lines if row["pairable"]]
    e4_1_ok = bool(pairable) and all(bool(row["pass"]) for row in pairable)

    fills_path = root / "fills.csv"
    capacity: Dict[str, Any]
    if fills_path.exists():
        fills = pd.read_csv(fills_path, dtype={"ts_code": str})
        buys = fills.loc[fills["side"].eq("buy")].copy()
        notional = pd.to_numeric(buys["amount"], errors="coerce")
        capacity = {
            "buy_fills": int(len(buys)),
            "min_buy_notional": float(notional.min()) if len(buys) else None,
            "below_min_ticket": int((notional < MIN_TICKET_CNY).sum()) if len(buys) else 0,
            "min_ticket": MIN_TICKET_CNY,
        }
        capacity["pass"] = bool(
            len(buys) > 0 and capacity["below_min_ticket"] == 0
        )
    else:
        capacity = {"pass": None, "reason": "fills_missing"}

    util_path = root / "daily_utilization.parquet"
    utilization: Dict[str, Any]
    if util_path.exists():
        frame = pd.read_parquet(util_path)
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["month"] = frame["trade_date"].str.slice(0, 6)
        frame["ratio"] = frame["market_value"] / frame["nav"].replace(0.0, np.nan)
        monthly = frame.groupby("month")["ratio"].mean().dropna()
        median = float(monthly.median()) if len(monthly) else None
        utilization = {
            "months": int(len(monthly)),
            "zero_utilization_months": int((monthly <= 0.0).sum()),
            "median": median,
            "p25": float(monthly.quantile(0.25)) if len(monthly) else None,
            "p75": float(monthly.quantile(0.75)) if len(monthly) else None,
            "min_required": UTIL_MEDIAN_MIN,
            "pass": bool(median is not None and median >= UTIL_MEDIAN_MIN),
        }
    else:
        utilization = {"pass": None, "reason": "daily_utilization_missing"}

    pit: Dict[str, Any] = {"negative_side_boundary": negative_side_boundary}
    audit_path = root / "selection_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path, dtype={"ts_code": str})
        statuses = audit["status"].astype(str).value_counts().to_dict()
        unresolved = int(
            sum(value for key, value in statuses.items() if key.startswith("unresolved"))
        )
        pit.update({"statuses": statuses, "unresolved": unresolved})
        pit["pass"] = bool(unresolved == 0)
    else:
        pit["pass"] = None

    lines = {
        "E4_1_crash_drawdown": {"pass": e4_1_ok, "windows": crash_lines},
        "E4_2_correlation": {
            "pass": bool(correlations.get("pass")),
            "upper_max": correlations.get("upper_max"),
            "upper_bound": CORR_UPPER,
        },
        "E4_3_capacity": capacity,
        "E4_4_utilization": utilization,
        "E4_5_pit_negative_side": pit,
    }
    vetoes = [key for key, row in lines.items() if row.get("pass") is False]
    not_estimable = [key for key, row in lines.items() if row.get("pass") is None]
    return {
        "estimable": not bool(not_estimable),
        "lines": lines,
        "vetoes": vetoes,
        "not_estimable": not_estimable,
        "pass": bool(not vetoes and not not_estimable),
        "rule": "五条否决线全部未触发且全部可估计才算过；缺件/不可估计一律不得记 pass",
    }


def _series_maxdd(levels: pd.Series) -> float:
    values = pd.to_numeric(levels, errors="coerce").dropna().astype(float)
    return float((values / values.cummax() - 1.0).min())


def _cagr(levels: pd.Series) -> float:
    """净值序列 CAGR（自然日 / 365.25；与 P4 ``nav_metrics`` 同口径）。"""
    values = pd.to_numeric(levels, errors="coerce").dropna().astype(float)
    if len(values) < 2 or values.iloc[0] <= 0.0:
        return float("nan")
    dates = pd.to_datetime(values.index, format="%Y%m%d")
    days = max((dates[-1] - dates[0]).days, 1)
    return float((values.iloc[-1] / values.iloc[0]) ** (365.25 / days) - 1.0)


# ────────────────────────── 组合级尺子 / 复合门 ──────────────────────────


def t2_combo_ruler(
    table3_levels: pd.DataFrame,
) -> Dict[str, Any]:
    """配对块自举的 ``ΔMaxDD`` 分布，按 ES3 既有规则在 2.5% 分位上下判。"""
    levels = table3_levels.copy()
    levels.index = levels.index.astype(str)
    returns = levels[["a_only", "combined"]].pct_change().dropna()
    boot = block_bootstrap_delta_maxdd(
        returns["a_only"].to_numpy(dtype=float),
        returns["combined"].to_numpy(dtype=float),
    )
    point = boot["point"]
    lower = boot["lower"]
    if point is None or lower is None:
        return {"estimable": False, "reason": "bootstrap_insufficient", "bootstrap": boot}
    p_pp = (_cagr(levels["combined"]) - _cagr(levels["a_only"])) * 100.0
    if lower >= -ES3_TOL_PP:
        passed = True
    elif lower >= -ES3_HARD_PP:
        passed = bool(p_pp > 0.0)
    else:
        passed = False
    return {
        "estimable": True,
        "delta_maxdd_pp": float(point),
        "delta_maxdd_lower_pp": float(lower),
        "delta_maxdd_upper_pp": float(boot["upper"]),
        "delta_cagr_pp": float(p_pp),
        "observations": int(boot["n"]),
        "resamples": int(boot["resamples"]),
        "pass": bool(passed),
        "rule": (
            "同一 ES3 规则（门槛常量不变）作用在配对 ΔMaxDD 的 2.5% 分位："
            "≥−1.0pp 通过；−3.0~−1.0pp 需 ΔCAGR>0；<−3.0pp 失败"
        ),
    }


def _fail_e1(name: str, item: Mapping[str, Any]) -> Tuple[str, List[str]]:
    if bool((item.get("power") or {}).get("detectable_any")):
        return "fail", [name]
    return "frozen", ["E1_underpowered"]


def _fail_e2(name: str, item: Mapping[str, Any]) -> Tuple[str, List[str]]:
    if str(item.get("state") or "") == "fail":
        return "fail", [name]
    return "frozen", ["E2_fragile"]


def _fail_q1(name: str, item: Mapping[str, Any]) -> Tuple[str, List[str]]:
    aa_ok = bool((item.get("aa_calibration") or {}).get("ok"))
    power_ok = bool((item.get("mde") or {}).get("power_ok"))
    if aa_ok and power_ok:
        return "fail", [name]
    pairs = ((aa_ok, "Q1_aa_calibration"), (power_ok, "Q1_mde_power"))
    return "frozen", [label for ok, label in pairs if not ok]


def _fail_e4(name: str, item: Mapping[str, Any]) -> Tuple[str, List[str]]:
    if item.get("vetoes"):
        return "fail", [name]
    return "frozen", ["E4_not_estimable"]


def _fail_any(name: str, item: Mapping[str, Any]) -> Tuple[str, List[str]]:
    del item
    return "fail", [name]


#: 组件未通过时的归类规则（"经济失败" vs "证据不足"）
_FAIL_RULES = {"E1": _fail_e1, "E2": _fail_e2, "Q1": _fail_q1, "E4": _fail_e4}

#: 需要独立校准资格的新增组件（型 II）；缺字段/None/非 True 一律冻结（R03）
CALIBRATION_REQUIRED = frozenset({"delta_sharpe", "delta_maxdd", "position", "noncrisis"})


def _component_verdict(name: str, item: Mapping[str, Any]) -> Tuple[str, List[str]]:
    """单组件归类：``pass`` / ``fail``（功效足够的经济失败）/ ``frozen``（证据不足）。

    新增统计组件（型 II 的 ΔSharpe/ΔMDD/仓位/非危机期）在独立校准完成前
    ``calibrated=False`` ⇒ 只记 ``frozen``，不得据此判经济失败（D02）。
    """
    if not item.get("estimable", False):
        return "frozen", [name]
    if name in CALIBRATION_REQUIRED and item.get("calibrated") is not True:
        return "frozen", [f"{name}_uncalibrated"]
    if bool(item.get("pass")):
        return "pass", []
    return _FAIL_RULES.get(name, _fail_any)(name, item)


def composite_gate(
    components: Mapping[str, Mapping[str, Any]],
    sleeve_type: str = SLEEVE_ENHANCEMENT,
) -> Dict[str, Any]:
    """按袖型分派的复合门（F29 草案 v5）。

    - 型 I（``enhancement``）：FULL_GATE = E1 ∧ E2 ∧ T2 ∧ Q1 ∧ 相关 ∧ E4；
    - 型 II（``risk_adjusted``）：ΔSharpe ∧ ΔMDD ∧ 相关 ∧ E4 ∧ 仓位 ∧ 非危机期；
      E1/Q1/E2/E4-4 由调用方标 ``not_applicable``（不参与合取，也不计为通过）。

    三向 verdict 与优先级：有硬门 fail ⇒ fail；否则任一必需项 not_estimable ⇒
    ``inconclusive_frozen``；全部适用必需项通过 ⇒ pass。
    """
    if sleeve_type not in SLEEVE_TYPES:
        raise ValueError(f"未知袖型：{sleeve_type}")
    if sleeve_type == SLEEVE_RISK_ADJUSTED:
        required = ["delta_sharpe", "delta_maxdd", "correlation", "E4", "position", "noncrisis"]
        rule = (
            "型 II 风险调整改善：ΔSharpe 配对下尾 >0 ∧ ΔMDD 配对下尾 ≥0 ∧ 三口径相关 "
            "97.5% 上界 <0.3 ∧ 五条 E4（E4-4 不适用）∧ 完整窗仓位 ≥10% ∧ 最差 20% 交易日 "
            "仓位 ≥20% ∧ 非危机期净收益 ≥0；E1/Q1/E2/E4-4 记 not_applicable"
        )
    else:
        required = ["E1", "E2", "T2", "Q1", "correlation", "E4"]
        rule = (
            "型 I 收益增强：FULL_GATE = E1 组件筛查 ∧ E2 制度重排 ∧ T2 组合级尺子 ∧ "
            "Q1 独立夏普 ∧ 三口径相关 ∧ 五条 E4 否决线"
        )
    failed: List[str] = []
    frozen: List[str] = []
    applicable: List[str] = []
    invalid: List[str] = []
    for name in required:
        item = components.get(name) or {}
        if str(item.get("state") or "") == "not_applicable":
            # 必需组件不允许由读数自行声明豁免（白名单为空）；标了即契约违反（C01）
            invalid.append(name)
            continue
        applicable.append(name)
        state, labels = _component_verdict(name, item)
        if state == "fail":
            failed.extend(labels)
        elif state == "frozen":
            frozen.extend(labels)
    if invalid:
        frozen.extend(f"{name}_invalid_not_applicable" for name in invalid)
    if not applicable:
        # 全部必需组件被跳过不能成为成功路径（C01）
        verdict = VERDICT_FROZEN
        frozen.append("no_applicable_component")
    elif failed:
        verdict = VERDICT_FAIL
    elif frozen:
        verdict = VERDICT_FROZEN
    else:
        verdict = VERDICT_PASS
    return {
        "verdict": verdict,
        "sleeve_type": sleeve_type,
        "frozen_components": sorted(set(frozen)),
        "failed_components": sorted(set(failed)),
        "required": required,
        "applicable": applicable,
        "invalid_not_applicable": invalid,
        "not_applicable": [n for n in required if n not in applicable and n not in invalid],
        "rule": rule + "；功效足够的门不过 ⇒ fail；否则证据不足 ⇒ inconclusive_frozen；全过 ⇒ pass",
        "pending_independent_review": bool(verdict == VERDICT_PASS),
    }
