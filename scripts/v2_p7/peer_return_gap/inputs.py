# -*- coding: utf-8 -*-
"""PIT 输入装载与基本域判定（协议 §2）。

单位口径：``daily.amount`` 千元 → ×1000 = 元；``daily_basic.total_mv`` 万元 →
×10000 = 元。金额阈值全部以元写死，转换只在本模块发生一次。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import spec
from .common import ResearchError

DAILY_COLUMNS = (
    "ts_code",
    "trade_date",
    "open",
    "close",
    "amount",
    "close_adj",
    "is_st",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "tradable",
)
MAIN_CODE_PATTERN = r"^\d{6}\.(SZ|SH)$"


def daily_path(root: Path, day: str) -> Path:
    return root / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"


def basic_path(root: Path, day: str) -> Path:
    return root / "data/clean/daily_basic" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"


def load_daily(root: Path, day: str, columns: Iterable[str] = DAILY_COLUMNS) -> pd.DataFrame:
    path = daily_path(root, day)
    if not path.exists():
        raise ResearchError(f"clean/daily 缺分区：{day}")
    frame = pd.read_parquet(path, columns=list(columns))
    frame["trade_date"] = frame["trade_date"].astype(str)
    if frame["trade_date"].nunique() != 1 or frame["trade_date"].iloc[0] != day:
        raise ResearchError(f"{day} 分区 trade_date 不一致")
    return frame


def load_daily_concat(root: Path, days: Sequence[str]) -> pd.DataFrame:
    frames = [load_daily(root, day) for day in days]
    if not frames:
        return pd.DataFrame(columns=list(DAILY_COLUMNS))
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates(subset=["trade_date", "ts_code"], keep="first")


def load_total_mv(root: Path, days: Sequence[str]) -> pd.DataFrame:
    """当日总市值（元）；同键冲突保留全部并报错，不做任意保留。"""
    frames = []
    for day in days:
        path = basic_path(root, day)
        if not path.exists():
            raise ResearchError(f"clean/daily_basic 缺分区：{day}")
        frame = pd.read_parquet(path, columns=["ts_code", "trade_date", "total_mv"])
        frame["trade_date"] = frame["trade_date"].astype(str)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    duplicated = raw.duplicated(subset=["trade_date", "ts_code"], keep=False)
    if duplicated.any():
        sample = raw[duplicated].head(5).to_dict("records")
        raise ResearchError(f"daily_basic 主键重复：{sample}")
    return raw


def load_stock_basic(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "data/clean/stock_basic.parquet")
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["list_date"] = frame["list_date"].astype(str)
    return frame


def load_industry_map(root: Path, day: str) -> tuple[pd.Series, dict[str, Any]]:
    """申万一级 PIT 映射；同日多条不同 L1 记冲突（返回空值）。"""
    path = root / "data/clean/sw_industry_pit.parquet"
    if not path.exists():
        raise ResearchError("缺少 sw_industry_pit.parquet")
    frame = pd.read_parquet(
        path, columns=["ts_code", "in_date", "out_date", "l1_code", "l1_name"]
    )
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["in_date"] = frame["in_date"].astype(str)
    out = frame["out_date"].fillna("99999999").astype(str)
    active = frame[(frame["in_date"] <= day) & (out > day)]
    conflicts = active.groupby("ts_code")["l1_code"].nunique()
    bad = set(conflicts[conflicts > 1].index)
    mapped = (
        active[~active["ts_code"].isin(bad)]
        .drop_duplicates(subset=["ts_code"], keep="first")
        .set_index("ts_code")["l1_name"]
    )
    mapped = mapped[mapped.notna()]
    return mapped, {
        "date": day,
        "active_rows": int(len(active)),
        "mapped_codes": int(len(mapped)),
        "conflicts": sorted(bad)[:50],
        "conflict_count": int(len(bad)),
    }


class IndustryPit:
    """申万一级 PIT 映射的内存缓存（只读一次 13205 行，按日过滤）。"""

    def __init__(self, root: Path) -> None:
        path = root / "data/clean/sw_industry_pit.parquet"
        if not path.exists():
            raise ResearchError("缺少 sw_industry_pit.parquet")
        frame = pd.read_parquet(
            path, columns=["ts_code", "in_date", "out_date", "l1_code", "l1_name"]
        )
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["in_date"] = frame["in_date"].astype(str)
        # out_date 的空值是 Arrow NA（不是字符串 "nan"）：fillna 成一个极远日期，
        # 等价于协议「空结束日按开放区间」。
        frame["out_date"] = frame["out_date"].fillna("99999999").astype(str)
        self.frame = frame

    def at(self, day: str) -> tuple[pd.Series, dict[str, Any]]:
        frame = self.frame
        active = frame[(frame["in_date"] <= day) & (frame["out_date"] > day)]
        conflicts = active.groupby("ts_code")["l1_code"].nunique()
        bad = set(conflicts[conflicts > 1].index)
        mapped = (
            active[~active["ts_code"].isin(bad)]
            .drop_duplicates(subset=["ts_code"], keep="first")
            .set_index("ts_code")["l1_name"]
        )
        mapped = mapped[mapped.notna()]
        return mapped, {
            "date": day,
            "active_rows": int(len(active)),
            "mapped_codes": int(len(mapped)),
            "conflict_count": int(len(bad)),
            "conflicts": sorted(bad)[:50],
        }


def listing_days_by_code(
    basic: pd.DataFrame, calendar: Sequence[str], day: str
) -> pd.Series:
    """上市以来开市日数（严格用日历，不用 clean/daily 的自然日 list_days）。"""
    labels = np.asarray(list(calendar))
    day_index = int(np.searchsorted(labels, day, side="left"))
    list_index = np.searchsorted(labels, basic["list_date"].to_numpy(), side="right")
    values = day_index - list_index
    return pd.Series(values, index=basic["ts_code"].to_numpy(), dtype="float64")


def amount_statistics(
    window: pd.DataFrame, codes: Sequence[str]
) -> pd.DataFrame:
    """最近 20 日平均成交额（元）与实有成交日数；已知停牌记 0，资料不明记缺。"""
    frame = window[window["ts_code"].isin(set(codes))].copy()
    frame["amount_yuan"] = pd.to_numeric(frame["amount"], errors="coerce") * 1000.0
    suspended = pd.to_numeric(frame["is_suspended"], errors="coerce").fillna(1) > 0
    frame.loc[suspended, "amount_yuan"] = 0.0
    frame["known"] = frame["amount_yuan"].notna()
    grouped = frame.groupby("ts_code", sort=True)
    result = pd.DataFrame(
        {
            "mean_amount_yuan": grouped["amount_yuan"].mean(),
            "amount_known_days": grouped["known"].sum(),
            "amount_rows": grouped.size(),
        }
    )
    result = result.reindex(list(codes))
    result["mean_amount_yuan"] = result["mean_amount_yuan"].where(
        result["amount_known_days"] >= spec.AMOUNT_MIN_DAYS
    )
    return result


def base_domain(
    *,
    day: str,
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    calendar: Sequence[str],
    industry: pd.Series,
    total_mv_yuan: pd.Series | None,
    amount_stats: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """协议 §2 第 2 组基本域；每个排除原因单独计数。"""
    frame = daily.copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    reasons: dict[str, Any] = {}
    total = int(len(frame))
    keep = pd.Series(True, index=frame.index)

    def reject(mask: pd.Series, name: str) -> None:
        nonlocal keep
        hit = mask.fillna(False) & keep
        reasons[name] = int(hit.sum())
        keep &= ~hit

    basic_map = basic.set_index("ts_code")
    market = frame["ts_code"].map(basic_map["market"])
    codes_ok = frame["ts_code"].str.match(MAIN_CODE_PATTERN).fillna(False)
    mother = (
        codes_ok
        & (market == spec.MARKET_MAIN)
        & (pd.to_numeric(frame["is_st"], errors="coerce").fillna(1) <= 0)
        & (pd.to_numeric(frame["close"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(frame["close_adj"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(frame["open"], errors="coerce").fillna(0) > 0)
    )
    reject(~codes_ok, "code_format")
    reject(market != spec.MARKET_MAIN, "not_main_board")

    listed = frame["ts_code"].map(listing_days_by_code(basic, calendar, day))
    reject(listed < spec.MIN_LISTED_TRADING_DAYS, "listed_lt_252")

    reject(pd.to_numeric(frame["is_st"], errors="coerce").fillna(1) > 0, "st")
    reject(pd.to_numeric(frame["is_suspended"], errors="coerce").fillna(1) > 0, "suspended")
    reject(pd.to_numeric(frame["tradable"], errors="coerce").fillna(0) <= 0, "not_tradable")
    reject(pd.to_numeric(frame["is_limit_up"], errors="coerce").fillna(1) > 0, "limit_up")
    for column in ("close", "close_adj", "open"):
        reject(pd.to_numeric(frame[column], errors="coerce").fillna(0) <= 0, f"bad_{column}")

    mapped = frame["ts_code"].map(industry)
    reject(mapped.isna(), "unknown_industry")
    reject(mapped.isin(spec.EXCLUDED_L1), "financial_excluded")

    if total_mv_yuan is None:
        reasons["missing_total_mv"] = int(keep.sum())
        keep &= False
    else:
        mv = frame["ts_code"].map(total_mv_yuan)
        reject(mv.isna(), "missing_total_mv")
        reject((mv < spec.MV_MIN_YUAN) | (mv > spec.MV_MAX_YUAN), "mv_out_of_band")

    if amount_stats is None:
        reasons["liquidity_not_evaluated"] = int(keep.sum())
    else:
        mean_amount = frame["ts_code"].map(amount_stats["mean_amount_yuan"])
        known_days = frame["ts_code"].map(amount_stats["amount_known_days"])
        reject(known_days.fillna(0) < spec.AMOUNT_MIN_DAYS, "amount_too_few_days")
        reject(mean_amount.isna() | (mean_amount < spec.AMOUNT_MIN_YUAN), "amount_lt_50m")

    selected = frame[keep].copy()
    selected["l1_name"] = selected["ts_code"].map(industry)
    selected["total_mv_yuan"] = (
        selected["ts_code"].map(total_mv_yuan) if total_mv_yuan is not None else np.nan
    )
    summary = {
        "date": day,
        "rows": total,
        "mother_count": int(mother.sum()),
        "base_count": int(len(selected)),
        "excluded": reasons,
        "liquidity_evaluated": amount_stats is not None,
    }
    return selected, summary


def build_base_frame(
    root: Path,
    *,
    day: str,
    calendar: Sequence[str],
    basic: pd.DataFrame,
    liquidity_window: Sequence[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """单日基本域；流动性窗口给出时才评估成交额条件。"""
    daily = load_daily(root, day)
    industry, industry_meta = load_industry_map(root, day)
    mv_frame = load_total_mv(root, [day])
    total_mv = mv_frame.set_index("ts_code")["total_mv"].astype(float) * 10000.0
    amount_stats = None
    if liquidity_window is not None:
        window_days = [item for item in liquidity_window if item <= day][-spec.AMOUNT_WINDOW:]
        window = load_daily_concat(root, window_days)
        amount_stats = amount_statistics(window, daily["ts_code"].tolist())
    selected, summary = base_domain(
        day=day,
        daily=daily,
        basic=basic,
        calendar=calendar,
        industry=industry,
        total_mv_yuan=total_mv,
        amount_stats=amount_stats,
    )
    summary["industry"] = industry_meta
    summary["liquidity_window"] = list(liquidity_window or [])
    return selected, summary
