"""历史时段映射、5 分钟槽位与竞价归属（协议 §5.4）。

目标语义：常规网格 48 槽，k=1..24 收盘时刻 09:35..11:30，k=25..48 收盘时刻 13:05..15:00。
供应商原始时间戳是区间起点还是终点，必须由 R0 核验后冻结；本模块同时实现两种映射，
默认取 `spec.SESSION_MAPPING["vendor_convention"]`，无法无歧义映射时抛
:class:`SessionMappingError`，**不得**偷偷切到 1 分钟聚合。

额外 09:30 竞价记录单独保留，只用于对账，不复制进连续交易槽。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from datetime import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import spec

MORNING_OPEN = time(9, 30)
MORNING_CLOSE = time(11, 30)
AFTERNOON_OPEN = time(13, 0)
AFTERNOON_CLOSE = time(15, 0)

#: 槽位分类
CORE = "core"
AUCTION = "auction"
LUNCH = "lunch"
OFF_GRID = "off_grid"


class SessionMappingError(RuntimeError):
    """时段映射失败（原始时间戳无法无歧义映射到目标网格）。"""


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def slot_end_time(k: int) -> time:
    """槽位 k 的收盘时刻（目标语义）。"""
    if not 1 <= int(k) <= 48:
        raise ValueError(f"槽位越界：{k}")
    minutes = (
        _minutes(MORNING_OPEN) + 5 * int(k)
        if k <= 24
        else _minutes(AFTERNOON_OPEN) + 5 * (int(k) - 24)
    )
    return time(minutes // 60, minutes % 60)


def slot_label(k: int) -> str:
    """槽位收市时刻字符串（HH:MM:SS）。"""
    moment = slot_end_time(k)
    return f"{moment.hour:02d}:{moment.minute:02d}:00"


def parse_trade_time(value: Any) -> time:
    """解析供应商 `trade_time` 的时分秒（接受字符串与 pandas 时间戳）。"""
    if isinstance(value, str):
        text = value.strip()
        if " " in text:
            text = text.split(" ", 1)[1]
        parts = text.split(":")
        if len(parts) < 2:
            raise SessionMappingError(f"无法解析 trade_time：{value!r}")
        hour, minute = int(parts[0]), int(parts[1])
        second = int(parts[2]) if len(parts) > 2 else 0
        return time(hour, minute, second)
    stamp = pd.Timestamp(value)
    return time(int(stamp.hour), int(stamp.minute), int(stamp.second))


def classify_time(value: Any, convention: str | None = None) -> tuple[str, int | None]:
    """把供应商时间戳映射到 (类别, 槽位)。

    约定（两种都必须精确落在 5 分钟网格上，否则 `off_grid`）：
    - `end`：时间戳 = 区间收盘时刻（09:35..11:30 / 13:05..15:00，另有 09:30 竞价行）；
    - `start`：时间戳 = 区间起始时刻（09:30..11:25 / 13:00..14:55，09:30 即 k=1）。
    """
    mode = convention or str(spec.SESSION_MAPPING["vendor_convention"])
    if mode not in ("end", "start"):
        raise SessionMappingError(f"未知时间戳约定：{mode!r}")
    moment = parse_trade_time(value)
    minutes = _minutes(moment)
    morning_open = _minutes(MORNING_OPEN)
    morning_close = _minutes(MORNING_CLOSE)
    afternoon_open = _minutes(AFTERNOON_OPEN)
    afternoon_close = _minutes(AFTERNOON_CLOSE)

    if mode == "end" and minutes == morning_open:
        return AUCTION, None
    if minutes < morning_open or minutes > afternoon_close:
        return OFF_GRID, None
    if morning_close < minutes < afternoon_open:
        return LUNCH, None

    in_morning = morning_open <= minutes <= morning_close
    in_afternoon = afternoon_open <= minutes <= afternoon_close
    if not (in_morning or in_afternoon):
        return OFF_GRID, None

    if mode == "end":
        offset = minutes - (morning_open if in_morning else afternoon_open)
        if offset <= 0 or offset % 5 != 0:
            return OFF_GRID, None
        k = offset // 5 if in_morning else 24 + offset // 5
    else:
        offset = minutes - (morning_open if in_morning else afternoon_open)
        if offset % 5 != 0:
            return OFF_GRID, None
        k = offset // 5 + 1 if in_morning else 25 + offset // 5
    if not 1 <= k <= 48:
        return OFF_GRID, None
    return CORE, k


def normalize_day(
    frame: pd.DataFrame,
    *,
    convention: str | None = None,
    trade_date: str | None = None,
) -> dict[str, Any]:
    """把单股单日原始分钟行整理成 48 槽网格 + 竞价行 + 问题清单。

    Returns:
        dict: `slots`(index=1..48 的 C/H/L/Q/A 表)、`auction`(单行或 None)、
        `problems`(list[dict])、`counts`(分类计数)。
    """
    if frame is None or len(frame) == 0:
        return {
            "slots": _empty_slots(),
            "auction": None,
            "problems": [{"reason": "empty_frame"}],
            "counts": {CORE: 0, AUCTION: 0, LUNCH: 0, OFF_GRID: 0},
        }
    rows: dict[int, dict[str, Any]] = {}
    auction: dict[str, Any] | None = None
    problems: list[dict[str, Any]] = []
    counts = {CORE: 0, AUCTION: 0, LUNCH: 0, OFF_GRID: 0}
    for record in frame.itertuples(index=False):
        raw_time = getattr(record, "trade_time")
        category, k = classify_time(raw_time, convention)
        counts[category] += 1
        payload = {
            "open": float(getattr(record, "open")),
            "close": float(getattr(record, "close")),
            "high": float(getattr(record, "high")),
            "low": float(getattr(record, "low")),
            "vol": float(getattr(record, "vol")),
            "amount": float(getattr(record, "amount")),
            "trade_time": str(raw_time),
        }
        if category == AUCTION:
            if auction is not None:
                problems.append({"reason": "duplicate_auction", "trade_time": str(raw_time)})
            else:
                auction = payload
            continue
        if category != CORE:
            continue
        if k in rows:
            problems.append({"reason": "duplicate_slot", "slot": k, "trade_time": str(raw_time)})
            # 同槽重复：保留先出现者并登记冲突，不做 keep=last
            continue
        rows[k] = payload
    slots = _empty_slots()
    for k, payload in rows.items():
        slots.loc[k, ["C", "H", "L", "O", "Q", "A"]] = [
            payload["close"],
            payload["high"],
            payload["low"],
            payload["open"],
            payload["vol"],
            payload["amount"],
        ]
        slots.loc[k, "trade_time"] = payload["trade_time"]
    return {
        "slots": slots,
        "auction": auction,
        "problems": problems,
        "counts": counts,
        "trade_date": trade_date,
    }


def _empty_slots() -> pd.DataFrame:
    """空槽位表（1..48，全部 NaN）。"""
    index = pd.RangeIndex(1, 49, name="slot")
    frame = pd.DataFrame(
        {
            "C": np.nan,
            "H": np.nan,
            "L": np.nan,
            "O": np.nan,
            "Q": np.nan,
            "A": np.nan,
            "trade_time": "",
        },
        index=index,
        dtype=object,
    )
    for name in ("C", "H", "L", "O", "Q", "A"):
        frame[name] = frame[name].astype(float)
    return frame


def session_complete(slots: pd.DataFrame) -> tuple[bool, list[int]]:
    """`session_complete`：k=1..47 所需价格及量额记录可解释（k=48 另作质量告警）。

    Returns:
        (是否完整, 缺失槽位列表)
    """
    missing: list[int] = []
    for k in range(1, 48):
        row = slots.loc[k]
        price_ok = (
            np.isfinite(row["C"])
            and float(row["C"]) > 0
            and np.isfinite(row["H"])
            and np.isfinite(row["L"])
        )
        volume_ok = (
            np.isfinite(row["Q"])
            and float(row["Q"]) >= 0
            and np.isfinite(row["A"])
            and float(row["A"]) >= 0
        )
        if not (price_ok and volume_ok):
            missing.append(k)
    return (not missing), missing


def closing_auction_regime(trade_date: str) -> str:
    """收盘制度版本（沪市 2018-08-20 起引入收盘集合竞价）。"""
    return (
        "sse_closing_auction"
        if str(trade_date) >= str(spec.MINUTE["sse_closing_auction_from"])
        else "continuous_to_close"
    )


def coverage_summary(days: Sequence[Any]) -> dict[str, Any]:
    """批量时段覆盖摘要（供质量报告）。"""
    rows = []
    for item in days:
        rows.append(dict(item))
    if not rows:
        return {"days": 0}
    frame = pd.DataFrame(rows)
    numeric = [name for name in frame.columns if frame[name].dtype.kind in "if"]
    return {
        "days": int(len(frame)),
        **{name: float(frame[name].mean()) for name in numeric},
    }
