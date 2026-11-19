"""标准化分钟存储：5 分钟槽位长表（年分区）+ 竞价旁路。

标准化层只做**时间边界与单位**的确定映射，不改经济值：
- `trade_date`：YYYYMMDD 字符串；
- `slot`：1..48（目标网格收盘时刻）；
- `C/H/L/O`：原始价格（不做复权，同日比值不需要复权）；
- `Q`：股；`A`：元。

09:30 竞价行单独落 `auction/`，**不复制进连续交易槽**。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import sessions

STANDARD_COLUMNS: tuple[str, ...] = ("trade_date", "ts_code", "slot", "O", "C", "H", "L", "Q", "A")
AUCTION_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "ts_code",
    "O",
    "C",
    "H",
    "L",
    "Q",
    "A",
    "trade_time",
)


class StoreError(RuntimeError):
    """标准化层错误（槽位冲突、越界、schema 不符）。"""


def standardize_day(
    frame: pd.DataFrame,
    trade_date: str,
    *,
    convention: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """把单股单日原始行标准化为 (槽位长表, 竞价长表, 问题清单)。"""
    if len(frame) == 0:
        return (
            pd.DataFrame(columns=list(STANDARD_COLUMNS)),
            pd.DataFrame(columns=list(AUCTION_COLUMNS)),
            [{"trade_date": str(trade_date), "reason": "empty_frame", "ts_code": None}],
        )
    codes = frame["ts_code"].astype(str).unique()
    if len(codes) != 1:
        raise StoreError(f"standardize_day 只接受单股票：{trade_date} {list(codes)[:3]}")
    code = str(codes[0])
    normalized = sessions.normalize_day(frame, convention=convention, trade_date=str(trade_date))
    slots = normalized["slots"]
    rows = []
    for k in range(1, 49):
        row = slots.loc[k]
        if not np.isfinite(row["C"]):
            continue
        rows.append(
            {
                "trade_date": str(trade_date),
                "ts_code": code,
                "slot": int(k),
                "O": float(row["O"]),
                "C": float(row["C"]),
                "H": float(row["H"]),
                "L": float(row["L"]),
                "Q": float(row["Q"]),
                "A": float(row["A"]),
            }
        )
    long_frame = pd.DataFrame(rows, columns=list(STANDARD_COLUMNS))
    auction = normalized["auction"]
    auction_frame = pd.DataFrame(
        (
            [
                {
                    "trade_date": str(trade_date),
                    "ts_code": code,
                    "O": auction["open"],
                    "C": auction["close"],
                    "H": auction["high"],
                    "L": auction["low"],
                    "Q": auction["vol"],
                    "A": auction["amount"],
                    "trade_time": auction["trade_time"],
                }
            ]
            if auction
            else []
        ),
        columns=list(AUCTION_COLUMNS),
    )
    problems = [
        {"trade_date": str(trade_date), "ts_code": code, **item} for item in normalized["problems"]
    ]
    return long_frame, auction_frame, problems


def year_path(directory: Path, year: str) -> Path:
    """标准化年分区文件路径。"""
    return Path(directory) / f"std_5m_year={year}.parquet"


def auction_year_path(directory: Path, year: str) -> Path:
    """竞价年分区文件路径。"""
    return Path(directory) / f"auction_year={year}.parquet"


def write_year(frame: pd.DataFrame, directory: Path, year: str) -> dict[str, Any]:
    """写标准化年分区（先写临时文件再改名，失败不留可用分区）。"""
    directory.mkdir(parents=True, exist_ok=True)
    target = year_path(directory, year)
    staging = target.with_suffix(".partial.parquet")
    frame.to_parquet(staging, index=False)
    staging.replace(target)
    return {"path": str(target), "rows": int(len(frame))}


def read_year(directory: Path, year: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """读标准化年分区（支持多块 part；缺失返回空表）。

    R1 按块贡献写 `std_5m_year=<Y>.<chunk>.parquet`，预热跨块会有重复行，
    读取时统一按 (trade_date, ts_code, slot) 去重，保证与单文件语义一致。
    """
    parts = sorted(Path(directory).glob(f"std_5m_year={year}*.parquet"))
    if not parts:
        return pd.DataFrame(columns=list(STANDARD_COLUMNS))
    frames = [pd.read_parquet(path, columns=list(columns) if columns else None) for path in parts]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["slot"] = frame["slot"].astype(int)
    if len(frames) > 1:
        subset = ["trade_date", "ts_code", "slot"]
        duplicated = frame.duplicated(subset=subset, keep=False)
        if duplicated.any():
            economic = ["O", "C", "H", "L", "Q", "A"]
            conflicting = 0
            for _, group in frame.loc[duplicated].groupby(subset, sort=False):
                if group[economic].drop_duplicates().shape[0] > 1:
                    conflicting += 1
            if conflicting:
                # 跨块同键异值不得 keep=first 静默消解：读取侧直接拒绝，交回下载侧裁定
                raise StoreError(f"跨块同键异值 {conflicting} 组，拒绝静默去重：{parts[0]}")
            frame = frame.drop_duplicates(subset=subset, keep="first")
    return frame.reset_index(drop=True)


def read_auction_year(directory: Path, year: str) -> pd.DataFrame:
    """读竞价旁路年分区（多块 part 拼接）。"""
    parts = sorted(Path(directory).glob(f"auction_year={year}*.parquet"))
    if not parts:
        return pd.DataFrame(columns=list(AUCTION_COLUMNS))
    frames = [pd.read_parquet(path) for path in parts]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame.reset_index(drop=True)


class MinuteStore:
    """按年惰性装载的标准化分钟存储（逐日切片为槽位宽表）。"""

    def __init__(self, directory: Path, convention: str | None = None) -> None:
        self.directory = Path(directory)
        self.convention = convention
        self._year: str | None = None
        self._frame: pd.DataFrame | None = None
        self._years_seen: set[str] = set()

    def _load(self, year: str) -> pd.DataFrame:
        if self._year == year and self._frame is not None:
            return self._frame
        frame = read_year(self.directory, year)
        self._year = year
        self._frame = frame
        self._years_seen.add(year)
        return frame

    @property
    def years_seen(self) -> list[str]:
        """已读年份（审计用）。"""
        return sorted(self._years_seen)

    def day_codes(self, day: str) -> set[str]:
        """某交易日有标准化行的股票集合。"""
        frame = self._load(str(day)[:4])
        if frame.empty:
            return set()
        subset = frame[frame["trade_date"] == str(day)]
        return set(subset["ts_code"].unique())

    def day_frames(self, day: str, codes: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
        """某日每只股票的槽位宽表（index=1..48，C/H/L/O/Q/A）。"""
        frame = self._load(str(day)[:4])
        if frame.empty:
            return {}
        subset = frame[frame["trade_date"] == str(day)]
        if codes is not None:
            wanted = set(str(code) for code in codes)
            subset = subset[subset["ts_code"].isin(wanted)]
        if subset.empty:
            return {}
        out: dict[str, pd.DataFrame] = {}
        for code, part in subset.groupby("ts_code", sort=False):
            slots = sessions._empty_slots()
            values = part.set_index("slot")[["C", "H", "L", "O", "Q", "A"]]
            slots.update(values)
            out[str(code)] = slots
        return out

    def coverage(self, day: str) -> dict[str, Any]:
        """单日覆盖摘要（有行股票数与槽位总数）。"""
        frame = self._load(str(day)[:4])
        subset = frame[frame["trade_date"] == str(day)] if not frame.empty else frame
        return {
            "trade_date": str(day),
            "codes": int(subset["ts_code"].nunique()) if len(subset) else 0,
            "rows": int(len(subset)),
        }
