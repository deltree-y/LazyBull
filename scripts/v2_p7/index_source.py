# -*- coding: utf-8 -*-
"""P7 崩盘门指数基座取数与冻结（只读外部源，快照落 P7 证据根）。

边界
====
- 只调 TuShare ``index_daily`` 读取指数收盘价，**不写生产 raw/clean**；冻结快照落
  ``<governance_dir>/specs/sources/``，供规格与 prereg 逐字节引用。
- 分页契约：``index_daily`` 单次最多 10000 条，按 ``limit/offset`` 翻页读到空为止，
  再按 ``trade_date`` 去重并断言唯一/单调/有限；缺交易日只登记不补齐。
- 指数为**价格指数**（不含分红再投），只用于识别市场崩盘窗，不参与收益计量。
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

INDEX_API = "index_daily"
INDEX_CODE = "000300.SH"
INDEX_START = "20050101"
PAGE_LIMIT = 10000
SNAPSHOT_COLUMNS = ("trade_date", "close")


class IndexSourceError(RuntimeError):
    """指数取数/归一/覆盖校验失败（一律 fail-closed）。"""


def fetch_index_daily(
    client: Any,
    ts_code: str = INDEX_CODE,
    start_date: str = INDEX_START,
    end_date: str = "",
    max_pages: int = 200,
) -> pd.DataFrame:
    """翻页读取指数日线：**以空页为终止**，offset 按实际返回行数推进。

    不能以「短页」结束：服务端实际单页上限可能低于请求的 ``limit``，
    短页之后仍可能有数据（CLAUDE.md 分页契约）。
    """
    frames = []
    offset = 0
    seen_dates: set = set()
    reached_empty = False
    for _ in range(max_pages):
        page = client.query(
            INDEX_API,
            fields="ts_code,trade_date,close",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            limit=PAGE_LIMIT,
            offset=offset,
        )
        if page is None or len(page) == 0:
            reached_empty = True
            break
        if "trade_date" not in page.columns:
            raise IndexSourceError("index_daily 返回缺少 trade_date，无法判断分页推进")
        dates = set(page["trade_date"].astype(str))
        if dates <= seen_dates:
            raise IndexSourceError("index_daily 分页未推进（本页主键全部重复）")
        seen_dates |= dates
        frames.append(page)
        offset += len(page)
    if not reached_empty:
        raise IndexSourceError(f"index_daily 翻页超过上限 {max_pages} 页，拒绝以不完整数据继续")
    if not frames:
        raise IndexSourceError(f"index_daily 未返回任何数据：{ts_code} {start_date}~{end_date}")
    return pd.concat(frames, ignore_index=True)


def verify_small_window(
    client: Any,
    reference: pd.DataFrame,
    ts_code: str = INDEX_CODE,
    start_date: str = "",
    end_date: str = "",
    fields: str = "ts_code,trade_date,close",
) -> Dict[str, Any]:
    """独立小窗口对账：单次取一小段，逐值与全量结果比对（防分页漏行）。"""
    probe = client.query(
        INDEX_API,
        fields=fields,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        limit=100,
        offset=0,
    )
    if probe is None or len(probe) == 0:
        raise IndexSourceError(f"小窗口对账未返回数据：{start_date}~{end_date}")
    expected = reference[
        (reference["trade_date"] >= start_date) & (reference["trade_date"] <= end_date)
    ]
    merged = probe.merge(
        expected,
        on="trade_date",
        how="outer",
        suffixes=("_probe", "_full"),
        indicator=True,
    )
    missing_in_probe = int((merged["_merge"] == "right_only").sum())
    missing_in_full = int((merged["_merge"] == "left_only").sum())
    mismatched = merged[
        (merged["_merge"] == "both")
        & ((merged["close_probe"].astype(float) - merged["close_full"].astype(float)).abs() > 1e-9)
    ]
    if missing_in_probe or missing_in_full or len(mismatched):
        raise IndexSourceError(
            f"小窗口对账失败：仅全量 {missing_in_probe} 行 / 仅小窗 {missing_in_full} 行 / "
            f"值不符 {len(mismatched)} 行"
        )
    return {
        "window": [start_date, end_date],
        "rows": int(len(probe)),
        "consistent": True,
    }


def normalize_index_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """归一为 ``trade_date/close`` 升序唯一序列。"""
    missing = [column for column in SNAPSHOT_COLUMNS if column not in raw.columns]
    if missing:
        raise IndexSourceError(f"指数行情缺列：{missing}")
    frame = raw.loc[:, list(SNAPSHOT_COLUMNS)].copy()
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 8)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if not frame["trade_date"].str.fullmatch(r"\d{8}").all():
        raise IndexSourceError("trade_date 存在非 YYYYMMDD 值")
    values = frame["close"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise IndexSourceError("close 存在 NaN 或 ±Inf（要求全部有限）")
    if (values <= 0).any():
        raise IndexSourceError("close 存在零或负值（要求严格为正）")
    duplicated = frame[frame["trade_date"].duplicated(keep=False)]
    if len(duplicated):
        grouped = duplicated.groupby("trade_date")["close"].agg(["min", "max"])
        conflicts = grouped[(grouped["max"] - grouped["min"]).abs() > 1e-9]
        if len(conflicts):
            raise IndexSourceError(f"跨页同 trade_date 收盘价冲突：{list(conflicts.index[:5])}")
    frame = frame.drop_duplicates(subset=["trade_date"], keep="first")
    frame = frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    if frame["trade_date"].duplicated().any():
        raise IndexSourceError("trade_date 去重后仍不唯一")
    if len(frame) < 2:
        raise IndexSourceError(f"指数样本不足：{len(frame)} 行")
    return frame


def write_snapshot(frame: pd.DataFrame, path: Path) -> str:
    """二进制写 LF CSV 快照，返回磁盘字节 SHA256（已存在即拒绝覆盖）。"""
    if path.exists():
        raise IndexSourceError(f"快照已存在，拒绝覆盖：{path}（换基座须新文件 + 新 prereg）")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["trade_date", "close"])
    for row in frame.itertuples(index=False):
        writer.writerow([row.trade_date, repr(float(row.close))])
    data = buffer.getvalue().encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_snapshot(path: Path) -> pd.DataFrame:
    """读取冻结快照（不重新取数，保证复算身份一致）。"""
    frame = pd.read_csv(path, dtype={"trade_date": str})
    return normalize_index_frame(frame)


def log_returns(frame: pd.DataFrame) -> Sequence[float]:
    """收盘价对数收益（首日无收益，长度 = 行数 − 1）。"""
    import numpy as np

    values = frame["close"].to_numpy(dtype=float)
    return np.log(values[1:] / values[:-1])


def coverage_report(frame: pd.DataFrame, calendar: Sequence[str]) -> Dict[str, Any]:
    """与交易日历比对，登记缺失交易日（不补齐）。"""
    dates = list(frame["trade_date"])
    present = set(dates)
    in_range = [day for day in calendar if dates[0] <= day <= dates[-1]]
    missing = [day for day in in_range if day not in present]
    return {
        "date_range": [dates[0], dates[-1]],
        "rows": len(dates),
        "calendar_days_in_range": len(in_range),
        "missing_calendar_days": len(missing),
        "missing_examples": missing[:5],
        "complete": not missing,
    }


def resolve_end_date(explicit: Optional[str], calendar: Sequence[str]) -> str:
    """结束日：显式给定优先，否则取日历末日。"""
    if explicit:
        return str(explicit)
    if not calendar:
        raise IndexSourceError("交易日历为空，无法确定结束日")
    return str(calendar[-1])
