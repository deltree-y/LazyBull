"""风控因子批量预计算模块

将 22 个基于 daily_adj 历史窗口的风控因子（A 类下行风险 8 个、B 类波动结构 6 个、
D 类流动性 8 个）改为全周期一次性向量化 rolling 计算，替代原先每交易日的
「全量切片 + groupby.tail + pivot + 逐股循环」模式。

设计说明：
- 输入为长表 daily_adj（含 ts_code / trade_date / close_adj 等列），
  内部 pivot 成 (trade_date × ts_code) 宽矩阵后统一做 rolling 运算；
- 输出为与 daily_adj 行对齐的长表（ts_code / trade_date + 22 个因子列，float32）；
- 与逐日路径的语义差异：窗口按「最近 N 个交易日」对齐（停牌日为 NaN 并按
  min_periods 跳过），而非「该股最近 N 条观测」；对无停牌的股票两者完全一致；
- 公告类截面因子（pledge/unlock/block/short 共 8 个）不依赖历史窗口，
  仍由 factor_registry 逐日计算，不在本模块范围内。

调用方：features/builder.py 的 FeatureBuilder（首次构建时缓存整个周期结果）。
"""

import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

_EPS = 1e-8
_SQRT_252 = float(np.sqrt(252.0))
_LOG_2 = float(np.log(2.0))
# 分块列数：控制 sliding_window_view 排序等操作的峰值内存
_CHUNK_COLS = 400

# 本模块负责批量预计算的 22 个因子（与 downside/volatility/liquidity 模块注册名一致）
PRECOMPUTED_RISK_FACTOR_NAMES: List[str] = [
    # A 类：下行风险
    "downside_vol_20",
    "downside_corr_20",
    "var_95_20",
    "cvar_95_20",
    "max_drawdown_20",
    "drawdown_duration",
    "skewness_20",
    "kurtosis_20",
    # B 类：波动结构
    "parkinson_vol_20",
    "vol_of_vol_20",
    "vol_regime_percentile",
    "garch_persistence",
    "high_low_range_ratio",
    "gap_risk",
    # D 类：流动性
    "turnover_cv_20",
    "amount_cv_20",
    "amihud_illiq_20",
    "vol_ratio_5_20",
    "up_down_vol_ratio",
    "volume_climax_days",
    "turnover_percentile",
    "volume_price_divergence",
]


# ---------------------------------------------------------------------------
# 分块滑窗工具（用于无法直接以 pandas rolling 表达的因子）
# ---------------------------------------------------------------------------


def _rolling_window_chunked(
    arr: np.ndarray,
    window: int,
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    min_valid: int,
) -> np.ndarray:
    """对 (n_dates × n_stocks) 矩阵按列分块做滑窗计算。

    Args:
        arr: 二维数组，行=日期（升序），列=股票
        window: 滑窗长度（交易日数）
        fn: 回调 (windows, nvalid) -> 结果；windows 形状 (n_out, k, window)
        min_valid: 窗口内最少有效样本数，不足则置 NaN

    Returns:
        与 arr 同形状的结果矩阵，前 window-1 行为 NaN
    """
    n_dates, n_stocks = arr.shape
    out = np.full((n_dates, n_stocks), np.nan)
    if n_dates < window:
        return out
    for start in range(0, n_stocks, _CHUNK_COLS):
        chunk = np.ascontiguousarray(arr[:, start : start + _CHUNK_COLS], dtype=np.float64)
        sw = sliding_window_view(chunk, window, axis=0)  # (n_out, k, window)
        nvalid = np.sum(~np.isnan(sw), axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = fn(sw, nvalid)
        vals = np.where(nvalid >= min_valid, vals, np.nan)
        out[window - 1 :, start : start + _CHUNK_COLS] = vals
    return out


def _cvar_from_windows(sw: np.ndarray, nvalid: np.ndarray) -> np.ndarray:
    """CVaR 95%：窗口内收益 ≤ 5% 分位数（线性插值）的均值。"""
    w = sw.shape[2]
    s = np.sort(sw, axis=2)  # NaN 排在末尾
    # 与 pandas Series.quantile(0.05) 一致的线性插值
    pos = 0.05 * np.maximum(nvalid - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(nvalid - 1, 0))
    frac = pos - lo
    v_lo = np.take_along_axis(s, lo[..., None], axis=2)[..., 0]
    v_hi = np.take_along_axis(s, hi[..., None], axis=2)[..., 0]
    q = v_lo + frac * (v_hi - v_lo)
    # 排序后累计和 → 尾部（≤ q）均值
    cum = np.nancumsum(s, axis=2)
    m = np.sum(s <= q[..., None], axis=2)  # NaN <= q 为 False，天然排除
    m_idx = np.clip(m - 1, 0, w - 1)
    tail_sum = np.take_along_axis(cum, m_idx[..., None], axis=2)[..., 0]
    return np.where(m > 0, tail_sum / np.maximum(m, 1), q)


def _max_drawdown_from_windows(sw: np.ndarray, nvalid: np.ndarray) -> np.ndarray:
    """窗口内最大回撤：min(price / cummax(price) - 1)。"""
    cummax = np.fmax.accumulate(sw, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = sw / np.where(np.abs(cummax) < _EPS, np.nan, cummax) - 1.0
    return np.nanmin(dd, axis=2)


def _drawdown_duration_from_windows(sw: np.ndarray, nvalid: np.ndarray) -> np.ndarray:
    """自窗口内最高点以来的交易日数（NaN 视为 -inf 不参与峰值判定）。"""
    filled = np.where(np.isnan(sw), -np.inf, sw)
    peak_idx = filled.argmax(axis=2)
    return (sw.shape[2] - 1 - peak_idx).astype(np.float64)


def _days_since_vol_max_from_windows(sw: np.ndarray, nvalid: np.ndarray) -> np.ndarray:
    """距窗口内天量的交易日数（与原实现一致：NaN 填 0 后 argmax）。"""
    filled = np.where(np.isnan(sw), 0.0, sw)
    max_idx = filled.argmax(axis=2)
    return (sw.shape[2] - 1 - max_idx).astype(np.float64)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def precompute_risk_factors(daily_adj: pd.DataFrame) -> Optional[pd.DataFrame]:
    """全周期批量预计算 22 个风控因子。

    Args:
        daily_adj: 后复权日线长表，必须含 ts_code / trade_date / close_adj，
            可选 open_adj / high_adj / low_adj / vol / amount / turnover_rate /
            ret_1 / pre_close_adj（缺失列对应的因子输出 NaN）

    Returns:
        与 daily_adj 行对齐的长表（ts_code / trade_date + 22 个因子列，float32）；
        输入不合法时返回 None。
    """
    required = {"ts_code", "trade_date", "close_adj"}
    if daily_adj is None or len(daily_adj) == 0:
        logger.warning("风控因子预计算：daily_adj 为空，跳过")
        return None
    missing = required - set(daily_adj.columns)
    if missing:
        logger.warning(f"风控因子预计算：daily_adj 缺少必需列 {missing}，跳过")
        return None

    optional_cols = [
        "open_adj",
        "high_adj",
        "low_adj",
        "vol",
        "amount",
        "turnover_rate",
        "ret_1",
        "pre_close_adj",
    ]
    keep_cols = ["ts_code", "trade_date", "close_adj"] + [
        c for c in optional_cols if c in daily_adj.columns
    ]
    base = daily_adj[keep_cols]

    # 日收益 ret_1：优先复用已有列，其次 pre_close_adj 推导，最后按股 pct_change
    if "ret_1" not in base.columns:
        if "pre_close_adj" in base.columns:
            base = base.assign(ret_1=base["close_adj"] / base["pre_close_adj"] - 1)
        else:
            base = base.sort_values(["ts_code", "trade_date"])
            base = base.assign(
                ret_1=base.groupby("ts_code")["close_adj"].pct_change(fill_method=None)
            )

    pivot_cols = [
        c
        for c in [
            "ret_1",
            "close_adj",
            "open_adj",
            "high_adj",
            "low_adj",
            "vol",
            "amount",
            "turnover_rate",
        ]
        if c in base.columns
    ]
    wide_all = base.pivot(index="trade_date", columns="ts_code", values=pivot_cols)
    wide_all = wide_all.sort_index()

    def _w(col: str) -> Optional[pd.DataFrame]:
        return wide_all[col] if col in pivot_cols else None

    ret = _w("ret_1")
    close = _w("close_adj")
    open_ = _w("open_adj")
    high = _w("high_adj")
    low = _w("low_adj")
    vol = _w("vol")
    amount = _w("amount")
    turnover = _w("turnover_rate")

    idx = wide_all.index
    cols = wide_all[pivot_cols[0]].columns
    results: Dict[str, pd.DataFrame] = {}

    logger.info(
        f"风控因子批量预计算：{len(idx)} 个交易日 × {len(cols)} 只股票，"
        f"{len(PRECOMPUTED_RISK_FACTOR_NAMES)} 个因子..."
    )

    # 全 NaN 窗口（新股上市前的日期在宽矩阵中全为 NaN）会触发 numpy 的
    # "All-NaN slice encountered" 告警，结果本身是正确的 NaN，属预期行为，定向抑制
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )

        # ── A 类：下行风险 ──────────────────────────────────────
        if ret is not None:
            # A1 下行波动率：clip(upper=0) 保留 NaN，正收益归零
            results["downside_vol_20"] = ret.clip(upper=0.0).rolling(20, min_periods=5).std(ddof=0)

            # A2 下行相关性：市场下跌日的 stock_ret vs mkt_ret 滚动相关
            mkt_ret = ret.mean(axis=1)
            down_mask = mkt_ret < 0
            ret_down = ret.copy()
            ret_down.loc[~down_mask] = np.nan
            mkt_down = mkt_ret.where(down_mask)
            results["downside_corr_20"] = ret_down.rolling(20, min_periods=5).corr(mkt_down)

            # A3 历史 VaR 95%
            results["var_95_20"] = ret.rolling(20, min_periods=5).quantile(0.05)

            # A4 CVaR 95%（滑窗排序 + 尾部均值，与逐日 quantile 语义一致）
            results["cvar_95_20"] = pd.DataFrame(
                _rolling_window_chunked(ret.to_numpy(), 20, _cvar_from_windows, min_valid=10),
                index=idx,
                columns=cols,
            )

            # A7/A8 偏度、峰度
            results["skewness_20"] = ret.rolling(20, min_periods=5).skew()
            results["kurtosis_20"] = ret.rolling(20, min_periods=5).kurt()

        if close is not None:
            # A5 最大回撤（20 日）
            results["max_drawdown_20"] = pd.DataFrame(
                _rolling_window_chunked(
                    close.to_numpy(), 20, _max_drawdown_from_windows, min_valid=1
                ),
                index=idx,
                columns=cols,
            )
            # A6 回撤持续天数（60 日窗口内距峰值的交易日数）
            results["drawdown_duration"] = pd.DataFrame(
                _rolling_window_chunked(
                    close.to_numpy(), 60, _drawdown_duration_from_windows, min_valid=2
                ),
                index=idx,
                columns=cols,
            )

        # ── B 类：波动结构 ──────────────────────────────────────
        if high is not None and low is not None:
            hl_sq = np.log(high / low.clip(lower=_EPS)) ** 2
            n_hl = hl_sq.rolling(20, min_periods=5).count()
            results["parkinson_vol_20"] = (
                np.sqrt(hl_sq.rolling(20, min_periods=5).sum() / (4.0 * _LOG_2 * n_hl)) * _SQRT_252
            )
            if close is not None:
                results["high_low_range_ratio"] = (
                    ((high - low) / close.clip(lower=_EPS)).rolling(20, min_periods=5).mean()
                )

        if close is not None:
            daily_ret = close.pct_change(fill_method=None)
            cnt_close_80 = close.notna().rolling(80, min_periods=1).sum()

            # B2 波动率的波动率
            rolling_vol = daily_ret.rolling(20, min_periods=5).std() * _SQRT_252
            vol_of_vol = rolling_vol.rolling(60, min_periods=20).std()
            results["vol_of_vol_20"] = vol_of_vol.where(cnt_close_80 >= 40)

            # B3 波动率历史分位（252 日滚动 rank）
            rv_regime = daily_ret.rolling(20, min_periods=10).std() * _SQRT_252
            rank_min = rv_regime.rolling(252, min_periods=1).rank(method="min")
            rank_cnt = rv_regime.rolling(252, min_periods=1).count()
            pct = (rank_min - 1.0) / rank_cnt.where(rank_cnt > 0)
            cnt_close_252 = close.notna().rolling(252, min_periods=1).sum()
            results["vol_regime_percentile"] = pct.where(cnt_close_252 >= 60)

            # B4 GARCH 波动持续性：平方收益的一阶自相关（60 日）
            sq_ret = daily_ret**2
            persistence = sq_ret.rolling(60, min_periods=30).corr(sq_ret.shift(1))
            results["garch_persistence"] = persistence.where(cnt_close_80 >= 30)

        if open_ is not None and low is not None:
            # B6 向下跳空频率
            gap_down = (open_ < low.shift(1)).astype(float)
            gap_mean = gap_down.rolling(20, min_periods=1).mean()
            cnt_open_21 = open_.notna().rolling(21, min_periods=1).sum()
            results["gap_risk"] = gap_mean.where(cnt_open_21 >= 10)

        # ── D 类：流动性 ────────────────────────────────────────
        if turnover is not None:
            results["turnover_cv_20"] = turnover.rolling(20, min_periods=10).std() / (
                turnover.rolling(20, min_periods=10).mean() + _EPS
            )
            t_rank = turnover.rolling(252, min_periods=1).rank(method="min")
            t_cnt = turnover.rolling(252, min_periods=1).count()
            t_pct = (t_rank - 1.0) / t_cnt.where(t_cnt > 0)
            results["turnover_percentile"] = t_pct.where(t_cnt >= 60)

        if amount is not None:
            results["amount_cv_20"] = amount.rolling(20, min_periods=10).std() / (
                amount.rolling(20, min_periods=10).mean() + _EPS
            )

        if close is not None and amount is not None:
            daily_ret_c = close.pct_change(fill_method=None)
            cnt_close_20 = close.notna().rolling(20, min_periods=1).sum()
            illiq = (daily_ret_c.abs() / amount.clip(lower=_EPS)).rolling(
                20, min_periods=10
            ).mean() * 1e6
            results["amihud_illiq_20"] = illiq.where(cnt_close_20 >= 10)

        if vol is not None:
            vol_5 = vol.rolling(5, min_periods=1).mean()
            vol_20 = vol.rolling(20, min_periods=10).mean()
            results["vol_ratio_5_20"] = vol_5 / (vol_20 + _EPS)

            climax = pd.DataFrame(
                _rolling_window_chunked(
                    vol.to_numpy(), 20, _days_since_vol_max_from_windows, min_valid=1
                ),
                index=idx,
                columns=cols,
            )
            cnt_vol_60 = vol.notna().rolling(60, min_periods=1).sum()
            results["volume_climax_days"] = climax.where(cnt_vol_60 >= 10)

        if close is not None and vol is not None:
            daily_ret_v = close.pct_change(fill_method=None)
            cnt_close_20v = close.notna().rolling(20, min_periods=1).sum()
            up_vol = vol.where(daily_ret_v > 0).rolling(20, min_periods=1).mean()
            down_vol = vol.where(daily_ret_v < 0).rolling(20, min_periods=1).mean()
            results["up_down_vol_ratio"] = (up_vol / (down_vol + _EPS)).where(cnt_close_20v >= 10)

            results["volume_price_divergence"] = close.rolling(10, min_periods=8).corr(vol)

    # ── 展平回长表（与 daily_adj 行对齐）────────────────────
    date_pos = idx.get_indexer(base["trade_date"])
    code_pos = cols.get_indexer(base["ts_code"])
    flat_pos = date_pos * len(cols) + code_pos

    out = pd.DataFrame(
        {
            "ts_code": base["ts_code"].to_numpy(),
            "trade_date": base["trade_date"].to_numpy(),
        }
    )
    for name in PRECOMPUTED_RISK_FACTOR_NAMES:
        wide = results.get(name)
        if wide is None:
            out[name] = np.full(len(base), np.nan, dtype=np.float32)
        else:
            out[name] = wide.to_numpy().ravel()[flat_pos].astype(np.float32)

    logger.info(f"风控因子批量预计算完成：{len(out)} 条记录")
    return out


def build_risk_factor_cache_dict(long_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """将预计算长表按 trade_date 分组为 O(1) 查表字典。

    Returns:
        {trade_date: 当日截面 DataFrame（ts_code + 22 个因子列）}
    """
    return {
        d: g.drop(columns=["trade_date"]).reset_index(drop=True)
        for d, g in long_df.groupby("trade_date", sort=False)
    }
