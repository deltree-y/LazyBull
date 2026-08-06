# -*- coding: utf-8 -*-
"""风控公告类数据 → 日频截面查询表构建（pledge_stat / share_float / block_trade）。

三类原始数据均为低频公告/事件数据，无法直接参与逐日截面特征构建。
本模块将其转换为 {trade_date: DataFrame} 的日频查询表（与 factors/fundamental、
factors/consensus_revision 等 lookup builder 同模式），供 features 层因子处理器消费。

PIT 契约（与 scripts/raw_download/announcement_risk.py 保持一致）：
  - pledge_stat : PIT 按公告日（ann_date，缺失时回退 end_date）前向填充，
                  输出 pledge_ratio / pledge_freshness_days / pledge_ratio_prev；
  - share_float : float_ratio 为单持有人占总股本比例（百分比 0-100），先按
                  (ts_code, float_date) 聚合求和（同批解禁总比例），再按
                  ann_date <= T 且 float_date > T 取最近解禁日一条，
                  输出 days_to_unlock / unlock_ratio；
  - block_trade : 大宗交易当日可见，按近 10 个交易日聚合折价（未复权收盘价），
                  输出 block_discount_avg_10d / block_discount_days_10d。

所有输出列名与 factors/risk/announcement_factors.py 的消费列名严格一致。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# 大宗折价聚合窗口（交易日）
_BLOCK_TRADE_WINDOW = 10


def _norm_date_series(s: pd.Series) -> pd.Series:
    """将日期列统一为 YYYYMMDD 字符串（容错 20240101 / 2024-01-01 / datetime）。"""
    out = s.astype(str).str.strip().str.replace("-", "", regex=False).str[:8]
    return out


def _trade_ord(trading_dates: List[str]) -> np.ndarray:
    """交易日列表 → int64 序数数组（YYYYMMDD 仅用于排序/区间比较）。"""
    return np.array([int(d) for d in trading_dates], dtype=np.int64)


def _trade_dt(trading_dates: List[str]) -> np.ndarray:
    """交易日列表 → datetime64[ns] 数组（用于自然日差计算）。

    YYYYMMDD 不能直接做整数相减（20240301-20240210=91 但实际 20 天），
    必须经 datetime64 转自然日。
    """
    return pd.to_datetime(trading_dates, format="%Y%m%d").to_numpy(dtype="datetime64[ns]")


# ═══════════════════════════════════════════════════════════════
# 质押（pledge_stat）
# ═══════════════════════════════════════════════════════════════


def build_pledge_lookup_by_date(
    pledge_raw: Optional[pd.DataFrame],
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """基于 pledge_stat 原始数据构建质押日频查询表。

    PIT 语义：T 日可见最近一次 `ann_date <= T`（ann_date 缺失时用 end_date）的
    质押统计；`pledge_freshness_days` 为公告日到 T 的自然日差；
    `pledge_ratio_prev` 为该股上一条（更早）公告的质押率（供 delta-on-update）。

    Args:
        pledge_raw: pledge_stat 原始数据（ts_code, ann_date, end_date, pledge_ratio 等）
        trading_dates: 交易日列表（YYYYMMDD）

    Returns:
        {trade_date: DataFrame(ts_code, pledge_ratio, pledge_freshness_days, pledge_ratio_prev)}
    """
    if pledge_raw is None or len(pledge_raw) == 0:
        logger.warning("pledge_stat 数据为空，跳过质押日频查询表构建")
        return {}

    df = pledge_raw.copy()
    df["ts_code"] = df["ts_code"].astype(str)

    # PIT 日期列：优先公告日 ann_date，缺失回退统计期 end_date
    if "ann_date" in df.columns and df["ann_date"].notna().any():
        date_col = "ann_date"
        if "end_date" in df.columns:
            df[date_col] = df[date_col].fillna(df["end_date"])
    elif "end_date" in df.columns:
        date_col = "end_date"
    else:
        logger.warning("pledge_stat 缺少 ann_date/end_date，无法构建质押查询表")
        return {}

    df[date_col] = _norm_date_series(df[date_col])
    df = df[df[date_col].str.match(r"^\d{8}$", na=False)]
    if len(df) == 0:
        return {}

    if "pledge_ratio" not in df.columns:
        logger.warning("pledge_stat 缺少 pledge_ratio 列，无法构建质押查询表")
        return {}
    df["pledge_ratio"] = pd.to_numeric(df["pledge_ratio"], errors="coerce")

    df = df.sort_values(["ts_code", date_col])
    # 上一条（更早）公告的质押率：groupby 内 shift
    df["pledge_ratio_prev"] = df.groupby("ts_code")["pledge_ratio"].shift(1)

    trade_ord = _trade_ord(trading_dates)
    trade_dt = _trade_dt(trading_dates)
    lookup: Dict[str, List[dict]] = {}

    for ts_code, grp in df.groupby("ts_code", sort=False):
        ann_ord = grp[date_col].astype(np.int64).to_numpy()
        ann_dt = pd.to_datetime(grp[date_col], format="%Y%m%d").to_numpy(dtype="datetime64[ns]")
        ratio = grp["pledge_ratio"].to_numpy(dtype=float)
        prev = grp["pledge_ratio_prev"].to_numpy(dtype=float)

        first_idx = int(np.searchsorted(trade_ord, ann_ord[0], side="left"))
        if first_idx >= len(trade_ord):
            continue
        # 质押率在公告后持续有效，覆盖至数据末尾
        td_ord = trade_ord[first_idx:]
        pos = np.searchsorted(ann_ord, td_ord, side="right") - 1
        valid = pos >= 0
        if not valid.any():
            continue
        td_valid = np.arange(first_idx, len(trade_ord), dtype=np.int64)[valid]
        pos = pos[valid]
        for j in range(len(td_valid)):
            td = trading_dates[td_valid[j]]
            freshness = int((trade_dt[td_valid[j]] - ann_dt[pos[j]]) / np.timedelta64(1, "D"))
            lookup.setdefault(td, []).append(
                {
                    "ts_code": ts_code,
                    "pledge_ratio": ratio[pos[j]],
                    "pledge_freshness_days": freshness,
                    "pledge_ratio_prev": prev[pos[j]],
                }
            )

    if not lookup:
        return {}
    logger.info(
        f"质押日频查询表构建完成: {len(lookup)} 个交易日有数据, "
        f"覆盖 {sum(len(v) for v in lookup.values())} 条股票-日记录"
    )
    return {td: pd.DataFrame(rows) for td, rows in lookup.items()}


# ═══════════════════════════════════════════════════════════════
# 限售解禁（share_float）
# ═══════════════════════════════════════════════════════════════


def build_share_float_lookup_by_date(
    share_float_raw: Optional[pd.DataFrame],
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """基于 share_float 原始数据构建限售解禁日频查询表。

    PIT 语义：T 日仅可见 `ann_date <= T` 的公告；对每条公告，解禁日在
    `float_date`（可能在未来）。仅统计"已公告且未解禁"（float_date > T）的记录，
    取最近解禁日（float_date 最小）一条，输出 days_to_unlock / unlock_ratio。

    Args:
        share_float_raw: share_float 原始数据（ts_code, ann_date, float_date, float_ratio 等）
        trading_dates: 交易日列表（YYYYMMDD）

    Returns:
        {trade_date: DataFrame(ts_code, days_to_unlock, unlock_ratio)}
    """
    if share_float_raw is None or len(share_float_raw) == 0:
        logger.warning("share_float 数据为空，跳过解禁日频查询表构建")
        return {}

    df = share_float_raw.copy()
    df["ts_code"] = df["ts_code"].astype(str)
    if "ann_date" not in df.columns or "float_date" not in df.columns:
        logger.warning("share_float 缺少 ann_date/float_date，无法构建解禁查询表")
        return {}
    df["ann_date"] = _norm_date_series(df["ann_date"])
    df["float_date"] = _norm_date_series(df["float_date"])
    df = df[
        df["ann_date"].str.match(r"^\d{8}$", na=False)
        & df["float_date"].str.match(r"^\d{8}$", na=False)
    ]
    if len(df) == 0:
        return {}

    if "float_ratio" not in df.columns:
        logger.warning("share_float 缺少 float_ratio 列，无法构建解禁查询表")
        return {}
    df["float_ratio"] = pd.to_numeric(df["float_ratio"], errors="coerce")

    # ── 同批解禁聚合：float_ratio 为"单个持有人"占总股本比例（百分比 0-100），
    # 同一 (ts_code, float_date)（同一批解禁）有多条持有人记录，求和才是该批
    # 解禁总比例（实测中位数约 10%、最大 <100%）；公告日取该批最早公告。
    df = df.groupby(["ts_code", "float_date"], as_index=False).agg(
        ann_date=("ann_date", "min"), float_ratio=("float_ratio", "sum")
    )
    df = df.sort_values(["ts_code", "float_date"])

    trade_ord = _trade_ord(trading_dates)
    trade_dt = _trade_dt(trading_dates)
    lookup: Dict[str, List[dict]] = {}

    for ts_code, grp in df.groupby("ts_code", sort=False):
        ann_ord = grp["ann_date"].astype(np.int64).to_numpy()
        float_ord = grp["float_date"].astype(np.int64).to_numpy()
        float_dt = pd.to_datetime(grp["float_date"], format="%Y%m%d").to_numpy(
            dtype="datetime64[ns]"
        )
        ratio = grp["float_ratio"].to_numpy(dtype=float)

        first_idx = int(np.searchsorted(trade_ord, ann_ord.min(), side="left"))
        # 最新解禁日之后不再有未解禁事件
        last_idx = int(np.searchsorted(trade_ord, float_ord.max(), side="right"))
        if first_idx >= last_idx:
            continue

        for i in range(first_idx, last_idx):
            t = trade_ord[i]
            mask = (ann_ord <= t) & (float_ord > t)
            if not mask.any():
                continue
            idx = int(np.argmin(float_ord[mask]))
            row_idx = np.where(mask)[0][idx]
            lookup.setdefault(trading_dates[i], []).append(
                {
                    "ts_code": ts_code,
                    "days_to_unlock": int(
                        (float_dt[row_idx] - trade_dt[i]) / np.timedelta64(1, "D")
                    ),
                    "unlock_ratio": ratio[row_idx],
                }
            )

    if not lookup:
        return {}
    logger.info(
        f"解禁日频查询表构建完成: {len(lookup)} 个交易日有数据, "
        f"覆盖 {sum(len(v) for v in lookup.values())} 条股票-日记录"
    )
    return {td: pd.DataFrame(rows) for td, rows in lookup.items()}


# ═══════════════════════════════════════════════════════════════
# 大宗交易（block_trade）
# ═══════════════════════════════════════════════════════════════


def build_block_trade_lookup_by_date(
    block_trade_raw: Optional[pd.DataFrame],
    trading_dates: List[str],
    close_lookup: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, pd.DataFrame]:
    """基于 block_trade 原始数据构建大宗交易日频查询表。

    折价率 = (成交价 - 当日收盘价) / 当日收盘价（未复权口径，与成交价可比）。
    对每个交易日 T，聚合近 10 个交易日内该股全部大宗交易：
      - block_discount_avg_10d : 折价率均值（同日多笔先取均值）
      - block_discount_days_10d : 出现折价（成交价低于收盘价）的交易日数

    Args:
        block_trade_raw: block_trade 原始数据（ts_code, trade_date, price 等）
        trading_dates: 交易日列表（YYYYMMDD）
        close_lookup: {trade_date: {ts_code: 未复权收盘价}}，缺省时按无折价处理

    Returns:
        {trade_date: DataFrame(ts_code, block_discount_avg_10d, block_discount_days_10d)}
    """
    if block_trade_raw is None or len(block_trade_raw) == 0:
        logger.warning("block_trade 数据为空，跳过宗交易日频查询表构建")
        return {}

    df = block_trade_raw.copy()
    df["ts_code"] = df["ts_code"].astype(str)
    df["trade_date"] = _norm_date_series(df["trade_date"])
    df = df[df["trade_date"].str.match(r"^\d{8}$", na=False)]
    if len(df) == 0:
        return {}

    if "price" not in df.columns:
        logger.warning("block_trade 缺少 price 列，无法构建大宗交易查询表")
        return {}
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    # 计算每笔折价率（未复权收盘价）
    if close_lookup:
        df["close"] = [
            close_lookup.get(str(td), {}).get(ts, np.nan)
            for td, ts in zip(df["trade_date"], df["ts_code"])
        ]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    else:
        df["close"] = np.nan
    df["discount"] = np.where(df["close"] > 1e-8, (df["price"] - df["close"]) / df["close"], np.nan)
    df = df.dropna(subset=["discount"])
    if len(df) == 0:
        return {}

    # 同日多笔聚合：折价率均值 + 是否出现折价
    agg = df.groupby(["ts_code", "trade_date"], as_index=False).agg(
        discount_mean=("discount", "mean"), has_discount=("discount", lambda s: bool((s < 0).any()))
    )

    trade_ord = _trade_ord(trading_dates)
    lookup: Dict[str, List[dict]] = {}

    for ts_code, grp in agg.groupby("ts_code", sort=False):
        g_ord = grp["trade_date"].astype(np.int64).to_numpy()
        g_disc = grp["discount_mean"].to_numpy(dtype=float)
        g_has = grp["has_discount"].to_numpy(dtype=bool)

        first_idx = int(np.searchsorted(trade_ord, g_ord.min(), side="left"))
        # 窗口聚合延伸到最后一笔交易后 (窗口-1) 个交易日（折价影响持续存在）
        last_idx = int(
            np.searchsorted(trade_ord, g_ord.max(), side="right") + (_BLOCK_TRADE_WINDOW - 1)
        )
        last_idx = min(last_idx, len(trade_ord))
        if first_idx >= last_idx:
            continue

        for i in range(first_idx, last_idx):
            win_start = trade_ord[max(0, i - _BLOCK_TRADE_WINDOW + 1)]
            t = trade_ord[i]
            lo = int(np.searchsorted(g_ord, win_start, side="left"))
            hi = int(np.searchsorted(g_ord, t, side="right"))
            if lo >= hi:
                continue
            lookup.setdefault(trading_dates[i], []).append(
                {
                    "ts_code": ts_code,
                    "block_discount_avg_10d": float(np.mean(g_disc[lo:hi])),
                    "block_discount_days_10d": int(np.sum(g_has[lo:hi])),
                }
            )

    if not lookup:
        return {}
    logger.info(
        f"大宗交易日频查询表构建完成: {len(lookup)} 个交易日有数据, "
        f"覆盖 {sum(len(v) for v in lookup.values())} 条股票-日记录"
    )
    return {td: pd.DataFrame(rows) for td, rows in lookup.items()}
