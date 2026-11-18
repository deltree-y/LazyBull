"""宽表面板装载：交易日轴 + 价格/成交/市值/交易标记的日 × 股矩阵。

只读 clean 层，不重算 cleaner 语义（ST/涨跌停/停牌/可交易标记一律复用）。
缺分区或缺失值保留 NaN，**不压缩交易日轴、不插值、不填零**。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DAILY_COLUMNS: tuple[str, ...] = (
    "ts_code",
    "trade_date",
    "close_adj",
    "open_adj",
    "amount",
    "is_st",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "tradable",
)

BASIC_COLUMNS: tuple[str, ...] = ("ts_code", "trade_date", "total_mv", "turnover_rate")


class PanelError(RuntimeError):
    """面板装载错误（分区缺失、日期口径不一致）。"""


def load_calendar(root: Path) -> list[str]:
    """SSE 开市日历（YYYYMMDD 升序）。"""
    cal = pd.read_parquet(Path(root) / "data/clean/trade_cal.parquet")
    frame = cal[cal["exchange"] == "SSE"]
    days = frame.loc[frame["is_open"].astype(int) == 1, "cal_date"].astype(str)
    days = sorted(set(days.tolist()))
    if not days:
        raise PanelError("交易日历为空")
    return days


def norm_day(value: str) -> str:
    """统一日期口径为 YYYYMMDD（分区文件名带短横线，行内为 8 位串）。"""
    return str(value).replace("-", "").replace("/", "")


def _partition_map(directory: Path) -> dict[str, Path]:
    """分区文件名（YYYY-MM-DD）-> 路径。"""
    return {path.stem.replace("-", ""): path for path in sorted(directory.glob("*.parquet"))}


def _read_range(
    directory: Path, start: str, end: str, columns: Sequence[str]
) -> pd.DataFrame:
    """读取 [start, end] 内的分区（文件名为 YYYY-MM-DD）。"""
    if not directory.exists():
        raise PanelError(f"分区目录不存在：{directory}")
    mapping = _partition_map(directory)
    selected = [day for day in sorted(mapping) if start <= day <= end]
    if not selected:
        raise PanelError(f"[{start},{end}] 内无分区：{directory}")
    frames = [pd.read_parquet(mapping[day], columns=list(columns)) for day in selected]
    frame = pd.concat(frames, ignore_index=True)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def _to_wide(frame: pd.DataFrame, value: str, calendar: Sequence[str]) -> pd.DataFrame:
    """长表 -> 日 × 股宽表（索引=给定日历顺序）。"""
    wide = frame.pivot(index="trade_date", columns="ts_code", values=value)
    wide = wide.reindex(index=list(calendar))
    wide.index.name = "trade_date"
    return wide


def load_daily_panel(
    root: Path,
    start: str,
    end: str,
    calendar: Sequence[str] | None = None,
    columns: Iterable[str] = DAILY_COLUMNS,
) -> dict[str, pd.DataFrame]:
    """装载价格/成交/交易标记宽表。"""
    days = list(calendar) if calendar is not None else load_calendar(root)
    low, high = norm_day(start), norm_day(end)
    window = [day for day in days if low <= day <= high]
    if not window:
        raise PanelError(f"[{start},{end}] 不在日历内")
    long = _read_range(Path(root) / "data/clean/daily", window[0], window[-1], columns)
    out: dict[str, pd.DataFrame] = {}
    for name in columns:
        if name in ("ts_code", "trade_date"):
            continue
        out[name] = _to_wide(long, name, window)
    return out


def load_basic_panel(
    root: Path,
    start: str,
    end: str,
    calendar: Sequence[str] | None = None,
    columns: Iterable[str] = BASIC_COLUMNS,
) -> dict[str, pd.DataFrame]:
    """装载市值/换手率宽表（daily_basic）。"""
    days = list(calendar) if calendar is not None else load_calendar(root)
    low, high = norm_day(start), norm_day(end)
    window = [day for day in days if low <= day <= high]
    if not window:
        raise PanelError(f"[{start},{end}] 不在日历内")
    long = _read_range(Path(root) / "data/clean/daily_basic", window[0], window[-1], columns)
    out: dict[str, pd.DataFrame] = {}
    for name in columns:
        if name in ("ts_code", "trade_date"):
            continue
        out[name] = _to_wide(long, name, window)
    return out


def daily_returns(close_adj: pd.DataFrame) -> pd.DataFrame:
    """相邻交易日收益：仅两端都有真实有效价格时成立，其余 NaN。"""
    prev = close_adj.shift(1)
    valid = close_adj.notna() & prev.notna() & (prev > 0)
    ratio = close_adj / prev.where(valid)
    return (ratio - 1.0).where(valid)


def market_series(returns: pd.DataFrame, universe_mask: pd.DataFrame) -> pd.DataFrame:
    """市场基准 m_t：截面等权平均有效相邻日收益，附分母与有效率。

    Args:
        returns: 日 × 股收益矩阵。
        universe_mask: 参与平均的股票（当日已上市主板股票）。
    """
    masked = returns.where(universe_mask.reindex_like(returns).fillna(False))
    count = masked.notna().sum(axis=1)
    mean = masked.mean(axis=1)
    listed = universe_mask.sum(axis=1).replace(0, np.nan)
    return pd.DataFrame(
        {
            "mkt_ret": mean,
            "n_valid": count.astype(int),
            "n_listed": universe_mask.sum(axis=1).astype(int),
            "valid_ratio": (count / listed).astype(float),
        }
    )


def frame_stats(frame: pd.DataFrame) -> dict[str, object]:
    """宽表规模摘要（内存与覆盖率核对用）。"""
    return {
        "rows": int(frame.shape[0]),
        "cols": int(frame.shape[1]),
        "first": str(frame.index[0]),
        "last": str(frame.index[-1]),
        "cells": int(frame.shape[0] * frame.shape[1]),
        "nonnull_ratio": float(frame.notna().to_numpy().mean()) if frame.size else 0.0,
    }
