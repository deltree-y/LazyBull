# -*- coding: utf-8 -*-
"""P4 只读数据适配：事件、行情、分红、A 参照与交易日历。"""

from __future__ import annotations

import hashlib
import os
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path("d:/my_pro/LazyBull")
DATA = ROOT / "data"
EVENTS_CSV = (
    DATA
    / "paper_shadow/p7_idx_in300_feasibility/p7_idx_in300_pit_20261114_r1"
    / "regular_adjustment_events_with_calendar.csv"
)
#: A 参照：R01 整改后必须指向**新** T6 根（旧 r6 根只读保留）；默认仍为历史根，
#: 重取时以 ``P7_A_NAV`` / ``P7_A_MANIFEST`` 显式覆盖并登记进证据。


def _resolve(path: Path) -> Path:
    """相对路径一律相对仓库根解析（env 覆盖常传相对路径）。"""
    return path if path.is_absolute() else ROOT / path


A_NAV = _resolve(
    Path(
        os.environ.get(
            "P7_A_NAV",
            str(DATA / "paper_shadow/p7_testbed/p7_t6_20261113_r6/arms/vvh/daily_vvh.parquet"),
        )
    )
)
A_MANIFEST = _resolve(
    Path(
        os.environ.get(
            "P7_A_MANIFEST",
            str(DATA / "paper_shadow/p7_testbed/p7_t6_20261113_r6/manifest.json"),
        )
    )
)
DIVIDEND_DIR = DATA / "raw/dividend"
CLEAN_DAILY = DATA / "clean/daily"
CLEAN_BASIC = DATA / "clean/daily_basic"
CUTOFF = "20261029"
SAMPLE_START = "20111010"
MAINBOARD = ("000", "001", "002", "003", "600", "601", "603", "605")
DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "close",
    "pre_close",
    "amount",
    "open_adj",
    "close_adj",
    "is_st",
    "is_suspended",
    "is_limit_down",
]
BASIC_COLUMNS = ["ts_code", "trade_date", "circ_mv"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code6(values: pd.Series) -> pd.Series:
    return values.astype(str).str.slice(0, 6)


def date8(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)


def mainboard_mask(codes: pd.Series) -> pd.Series:
    return codes.astype(str).str.slice(0, 3).isin(MAINBOARD)


@dataclass(frozen=True, slots=True)
class Batch:
    notice_id: str
    signal_date: str
    entry_date: str
    flow_date: str


class P4Data:
    """按需读取，禁止未来日；行情日期来自 clean 分区本身。"""

    def __init__(self) -> None:
        self.dates = sorted(path.stem.replace("-", "") for path in CLEAN_DAILY.glob("*.parquet"))
        if not self.dates:
            raise RuntimeError("缺少 clean/daily 分区")
        self._date_to_path = {
            path.stem.replace("-", ""): path for path in CLEAN_DAILY.glob("*.parquet")
        }
        self._basic_to_path = {
            path.stem.replace("-", ""): path for path in CLEAN_BASIC.glob("*.parquet")
        }
        self._date_set = set(self.dates)
        self.cutoff = CUTOFF
        self.events_all, self.events, self.batches = self._load_events()
        self.dividends = self._load_dividends()
        self.a_nav = self._load_a_nav()

    def _load_events(self) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Batch, ...]]:
        frame = pd.read_csv(EVENTS_CSV, dtype=str).fillna("")
        for column in ("ann_date", "effective_date", "entry_date", "flow_date"):
            frame[column] = date8(frame[column])
        frame["ts_code"] = code6(frame["ts_code"])
        frame["direction"] = frame["direction"].str.lower()
        usable_batches = []
        for notice_id, group in frame.groupby("notice_id", sort=True):
            signal = str(group["ann_date"].iloc[0])
            entry = str(group["entry_date"].iloc[0])
            flow = str(group["flow_date"].iloc[0])
            if entry < SAMPLE_START or flow > CUTOFF or entry >= flow:
                continue
            if signal not in self._date_set or entry not in self._date_set:
                raise RuntimeError(f"批次 {notice_id} 的公告/入场日不在 clean 日历")
            usable_batches.append(Batch(notice_id, signal, entry, flow))
        ids = {item.notice_id for item in usable_batches}
        events = frame.loc[frame["notice_id"].isin(ids)].copy()
        if len(usable_batches) != 30:
            raise RuntimeError(f"可执行批次应为 30，实际 {len(usable_batches)}")
        return frame, events, tuple(usable_batches)

    def _load_dividends(self) -> pd.DataFrame:
        columns = [
            "ts_code",
            "end_date",
            "ann_date",
            "div_proc",
            "stk_div",
            "cash_div",
            "cash_div_tax",
            "record_date",
            "ex_date",
            "pay_date",
            "imp_ann_date",
        ]
        frames = [
            pd.read_parquet(path, columns=columns)
            for path in sorted(DIVIDEND_DIR.glob("*.parquet"))
        ]
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.loc[frame["div_proc"].astype(str).eq("实施")].copy()
        frame["ts_code"] = code6(frame["ts_code"])
        for column in (
            "end_date",
            "ann_date",
            "record_date",
            "ex_date",
            "pay_date",
            "imp_ann_date",
        ):
            frame[column] = date8(frame[column])
        frame = frame.loc[frame["ex_date"].str.len().eq(8)].copy()
        frame = frame.drop_duplicates(
            subset=["ts_code", "end_date", "div_proc", "ann_date"], keep="last"
        )
        frame["sort_key"] = (
            frame["imp_ann_date"].fillna("")
            + "|"
            + frame["ann_date"].fillna("")
            + "|"
            + frame["end_date"].fillna("")
        )
        frame = frame.sort_values(["ts_code", "ex_date", "sort_key"], kind="mergesort")
        frame = frame.groupby(["ts_code", "ex_date"], as_index=False).tail(1)
        return frame.reset_index(drop=True)

    def _load_a_nav(self) -> pd.Series:
        frame = pd.read_parquet(A_NAV, columns=["trade_date", "nav"])
        frame["trade_date"] = date8(frame["trade_date"])
        frame = frame.drop_duplicates("trade_date", keep="last").set_index("trade_date")
        values = pd.to_numeric(frame["nav"], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            raise RuntimeError("A 参照 NAV 存在非正/非有限值")
        return values.astype(float).sort_index()

    def read_daily(self, day: str) -> pd.DataFrame:
        """读取一个 clean 日全市场快照；返回统一 6 位代码。"""
        day = str(day)
        path = self._date_to_path.get(day)
        if path is None:
            return pd.DataFrame(columns=["code", *DAILY_COLUMNS])
        frame = pd.read_parquet(path, columns=DAILY_COLUMNS)
        frame["trade_date"] = date8(frame["trade_date"])
        frame["code"] = code6(frame["ts_code"])
        return frame.drop(columns=["ts_code"]).drop_duplicates("code", keep="last")

    def read_basic(self, day: str) -> pd.DataFrame:
        path = self._basic_to_path.get(str(day))
        if path is None:
            return pd.DataFrame(columns=["code", *BASIC_COLUMNS])
        frame = pd.read_parquet(path, columns=BASIC_COLUMNS)
        frame["trade_date"] = date8(frame["trade_date"])
        frame["code"] = code6(frame["ts_code"])
        return frame.drop(columns=["ts_code"]).drop_duplicates("code", keep="last")

    def read_codes(self, day: str, codes: list[str]) -> pd.DataFrame:
        frame = self.read_daily(day)
        if frame.empty:
            return frame
        return frame.loc[frame["code"].isin({str(code) for code in codes})].copy()

    def prev_trade_day(self, day: str) -> str:
        idx = bisect_left(self.dates, str(day))
        if idx <= 0:
            raise RuntimeError(f"没有早于 {day} 的交易日")
        return self.dates[idx - 1]

    def trade_days(self, start: str, end: str) -> list[str]:
        return [day for day in self.dates if str(start) <= day <= str(end)]

    def effective_trade_day(self, day: str) -> str | None:
        idx = bisect_left(self.dates, str(day))
        return self.dates[idx] if idx < len(self.dates) else None

    def batch_events(self, notice_id: str) -> pd.DataFrame:
        return self.events.loc[self.events["notice_id"].eq(str(notice_id))].copy()

    def all_a_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Batch, ...]]:
        """返回值仅为兼容审计调用；真实事件表仍在 events。"""
        return self.events_all, self.events, self.batches

    def input_hashes(self) -> dict[str, str]:
        return {
            "events": sha256(EVENTS_CSV),
            "a_nav": sha256(A_NAV),
            "a_manifest": sha256(A_MANIFEST),
            "clean_daily_first": sha256(
                Path(
                    CLEAN_DAILY
                    / f"{self.dates[0][:4]}-{self.dates[0][4:6]}-{self.dates[0][6:]}.parquet"
                )
            ),
        }
