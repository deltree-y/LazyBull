"""固定 19 列特征（协议 §5）。

全部为宽表（日 × 股）向量化计算，逐列抽取到合格行，避免同时驻留 19 张矩阵。
窗口一律含 T、不超过 T；缺失日不插值、不填零；窗口内有效观测不足即 NaN。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from scripts.v2_p7.defensive_ml import spec

#: 窗口有效观测下限（80% 规则与协议的 96/120 契约）
MIN_OBS = {
    "w20": 16,
    "w60": 48,
    "w120": 96,
    "neg_beta": 20,
    "amihud": 16,
}

ZERO_EPS = 1e-12
VAR_FLOOR = 1e-12


def _ret_window(close: pd.DataFrame, span: int) -> pd.DataFrame:
    """P_T / P_(T-span) - 1，端点缺失即 NaN。"""
    past = close.shift(span)
    valid = close.notna() & past.notna() & (past > 0)
    return (close / past.where(valid) - 1.0).where(valid)


def rolling_mdd120(close: pd.DataFrame, window: int = 120, min_valid: int = 96) -> np.ndarray:
    """窗口内真实有效收盘路径的最大回撤损失幅度（非负）。

    逐日取窗口切片后做前缀最大值累积：峰必须先于谷，缺日不插值。
    """
    values = close.to_numpy(dtype=np.float64)
    total, width = values.shape
    out = np.full((total, width), np.nan, dtype=np.float64)
    for end in range(window - 1, total):
        block = values[end - window + 1: end + 1, :]
        valid = np.isfinite(block) & (block > 0)
        count = valid.sum(axis=0)
        if not count.any():
            continue
        filled = np.where(valid, block, -np.inf)
        running_max = np.maximum.accumulate(filled, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(valid, block / running_max, np.nan)
        lowest = np.full(width, np.nan, dtype=np.float64)
        selected = count > 0
        if selected.any():
            lowest[selected] = np.nanmin(ratio[:, selected], axis=0)
        out[end] = np.where(count >= min_valid, 1.0 - lowest, np.nan)
    return out


def _paired(returns: pd.DataFrame, market: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """配对有效样本：个股收益与市场收益同时可用。"""
    frame = market.reindex(returns.index)
    mask = returns.notna() & frame.notna().values.reshape(-1, 1)
    stock = returns.where(mask)
    bench = pd.DataFrame(
        np.where(mask, frame.to_numpy().reshape(-1, 1), np.nan),
        index=returns.index,
        columns=returns.columns,
    )
    return stock, bench


def _ols_beta(stock: pd.DataFrame, bench: pd.DataFrame, min_obs: int) -> dict[str, pd.DataFrame]:
    """窗口 OLS：beta / alpha / 市场方差（带截距等价口径）。"""
    mean_s = stock.rolling(120, min_periods=min_obs).mean()
    mean_b = bench.rolling(120, min_periods=min_obs).mean()
    mean_sb = (stock * bench).rolling(120, min_periods=min_obs).mean()
    mean_bb = (bench * bench).rolling(120, min_periods=min_obs).mean()
    cov = mean_sb - mean_s * mean_b
    var = mean_bb - mean_b * mean_b
    beta = (cov / var.where(var > VAR_FLOOR)).where(var > VAR_FLOOR)
    alpha = mean_s - beta * mean_b
    count = stock.rolling(120, min_periods=min_obs).count()
    return {
        "beta": beta,
        "alpha": alpha,
        "mean_stock": mean_s,
        "mean_bench": mean_b,
        "mean_cross": mean_sb,
        "mean_bench_sq": mean_bb,
        "count": count,
    }


def _idiovol(
    stock: pd.DataFrame,
    bench: pd.DataFrame,
    ols: dict[str, pd.DataFrame],
    min_obs: int,
) -> pd.DataFrame:
    """同一 120 日配对样本内、用**固定** alpha_T/beta_T 的残差样本标准差。

    SSE/n 展开（alpha、beta 取窗口内 OLS 解）：
    E[(r-alpha-beta*m)^2] = E[r^2] - 2aE[r] - 2bE[rm] + a^2 + 2abE[m] + b^2E[m^2]
    样本方差取 ddof=1；市场方差过小或样本不足返回 NaN。
    """
    alpha = ols["alpha"]
    beta = ols["beta"]
    mean_s = ols["mean_stock"]
    mean_b = ols["mean_bench"]
    mean_sb = ols["mean_cross"]
    mean_bb = ols["mean_bench_sq"]
    count = ols["count"]
    mean_ss = (stock * stock).rolling(120, min_periods=min_obs).mean()
    second = (
        mean_ss
        - 2.0 * alpha * mean_s
        - 2.0 * beta * mean_sb
        + alpha * alpha
        + 2.0 * alpha * beta * mean_b
        + beta * beta * mean_bb
    )
    var = (second * count / (count - 1.0)).where(count > 1)
    var = var.where(var >= 0.0)
    return np.sqrt(var)


def _amihud(returns: pd.DataFrame, amount_yuan: pd.DataFrame) -> pd.DataFrame:
    """|r| / 成交额 * 1e8，只用相邻收益有效且成交额为正的配对日。"""
    usable = returns.notna() & amount_yuan.notna() & (amount_yuan > 0)
    ratio = (returns.abs() / amount_yuan.where(usable)) * 1e8
    return ratio.where(usable).rolling(20, min_periods=MIN_OBS["amihud"]).mean()


def _amount_stats(amount_yuan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """20 日正成交额离散度与均值对数。"""
    positive = amount_yuan.where(amount_yuan > 0)
    mean = positive.rolling(20, min_periods=MIN_OBS["amihud"]).mean()
    std = positive.rolling(20, min_periods=MIN_OBS["amihud"]).std(ddof=1)
    cv = std / mean.where(mean > 0)
    log_mean = np.log(mean.where(mean > 0))
    return cv, log_mean


def feature_columns() -> tuple[str, ...]:
    """冻结列序（唯一来源 spec.FEATURES）。"""
    return spec.FEATURES


def compute_columns(
    prices: dict[str, pd.DataFrame],
    basic: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    market: pd.Series,
    *,
    with_mdd: bool = True,
) -> dict[str, object]:
    """按列计算（返回宽表/ndarray），供上层逐列抽取。"""
    close = prices["close_adj"]
    amount_yuan = prices["amount"].astype(float) * 1000.0
    total_mv_yuan = basic["total_mv"].astype(float) * 1e4
    turnover = basic["turnover_rate"].astype(float)

    stock, bench = _paired(returns, market)
    ols = _ols_beta(stock, bench, MIN_OBS["w120"])
    beta = ols["beta"]
    negative = bench < 0
    stock_neg = stock.where(negative)
    bench_neg = bench.where(negative)
    ols_neg = _ols_beta(stock_neg, bench_neg, MIN_OBS["neg_beta"])

    amount_cv, log_amount = _amount_stats(amount_yuan)
    zero_ratio = (returns.abs() <= ZERO_EPS).astype(float).where(returns.notna())
    downvol = np.sqrt(returns.clip(upper=0).pow(2).rolling(60, min_periods=MIN_OBS["w60"]).mean())
    mdd = rolling_mdd120(close) if with_mdd else None

    return {
        "def_ret20": _ret_window(close, 20),
        "def_ret60": _ret_window(close, 60),
        "def_ret120": _ret_window(close, 120),
        "def_vol20": returns.rolling(20, min_periods=MIN_OBS["w20"]).std(ddof=1),
        "def_vol60": returns.rolling(60, min_periods=MIN_OBS["w60"]).std(ddof=1),
        "def_downvol60": downvol,
        "def_mdd120": mdd,
        "def_beta120": beta,
        "def_downbeta120": ols_neg["beta"],
        "def_idiovol120": _idiovol(stock, bench, ols, MIN_OBS["w120"]),
        "def_amihud20": _amihud(returns, amount_yuan),
        "def_amount_cv20": amount_cv,
        "def_log_amount20": log_amount,
        "def_log_total_mv": np.log(total_mv_yuan.where(total_mv_yuan > 0)),
        "def_turnover20": turnover.rolling(20, min_periods=MIN_OBS["w20"]).mean() / 100.0,
        "def_zero_return60": zero_ratio.rolling(60, min_periods=MIN_OBS["w60"]).mean(),
    }


def cashflow_frame(
    lookup: dict[str, pd.DataFrame], dates: Iterable[str]
) -> pd.DataFrame:
    """把逐日现金流查询表压成长表（含 schema v3 列与 freshness）。"""
    chunks = []
    for day in dates:
        frame = lookup.get(day)
        if frame is None or len(frame) == 0:
            continue
        part = frame[["ts_code", "ocf_to_revenue", "capex_to_ocf", "cashflow_freshness_days"]]
        part = part.copy()
        part["trade_date"] = day
        chunks.append(part)
    if not chunks:
        return pd.DataFrame(
            columns=["trade_date", "ts_code", "ocf_to_revenue", "capex_to_ocf",
                     "cashflow_freshness_days"]
        )
    return pd.concat(chunks, ignore_index=True)


def attach_cashflow(
    base: pd.DataFrame, cashflow_long: pd.DataFrame, calendar: list[str]
) -> pd.DataFrame:
    """按「前一交易日」把现金流状态并入信号日（含 age 的自然日追加）。"""
    position = {day: i for i, day in enumerate(calendar)}
    prev_map = {day: calendar[position[day] - 1] for day in calendar if position[day] > 0}
    mapping = cashflow_long.rename(columns={"trade_date": "asof_date"})
    frame = base.copy()
    frame["asof_date"] = frame["trade_date"].map(prev_map)
    merged = frame.merge(mapping, on=["asof_date", "ts_code"], how="left")
    gap = (
        pd.to_datetime(merged["trade_date"], format="%Y%m%d")
        - pd.to_datetime(merged["asof_date"], format="%Y%m%d")
    ).dt.days
    merged["def_cashflow_age"] = merged["cashflow_freshness_days"] + gap
    merged["def_ocf_to_sales_cash"] = merged["ocf_to_revenue"]
    merged["def_capex_to_ocf"] = merged["capex_to_ocf"]
    return merged.drop(
        columns=["asof_date", "ocf_to_revenue", "capex_to_ocf", "cashflow_freshness_days"]
    )


def extract_feature_frame(
    columns: dict[str, object],
    eligible: np.ndarray,
    dates: list[str],
    codes: list[str],
) -> pd.DataFrame:
    """把宽表/矩阵按合格位置抽成长表（只保留指定列）。"""
    rows, cols = np.nonzero(eligible)
    date_array = np.asarray(dates, dtype=object)
    code_array = np.asarray(codes, dtype=object)
    frame = pd.DataFrame(
        {"trade_date": date_array[rows], "ts_code": code_array[cols]}
    )
    for name in spec.FEATURES[:16]:
        matrix = columns[name]
        values = matrix.to_numpy() if hasattr(matrix, "to_numpy") else np.asarray(matrix)
        frame[name] = values[rows, cols]
    return frame
