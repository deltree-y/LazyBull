# -*- coding: utf-8 -*-
"""月末时钟（协议 §3）：T = 自然月最后交易日，F = 前一自然月最后交易日。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from . import spec
from .common import ResearchError


def load_calendar(root: Path) -> list[str]:
    """沪深共同开市日（SSE 日历；两所节假日一致，差异记入 R0 说明）。"""
    path = root / "data/clean/trade_cal.parquet"
    frame = pd.read_parquet(path, columns=["exchange", "cal_date", "is_open"])
    frame = frame[(frame["exchange"] == "SSE") & (frame["is_open"] == 1)]
    days = sorted({str(value) for value in frame["cal_date"]})
    if not days or days != sorted(set(days)):
        raise ResearchError("交易日历为空或未严格升序")
    return days


def calendar_payload(days: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"cal_date": list(days)})
    frame["is_open"] = 1
    frame["prev_open"] = frame["cal_date"].shift(1)
    return frame


def month_clock(
    days: Sequence[str], start: str, end: str, *, anchor: str | None = None
) -> list[dict[str, Any]]:
    """生成受理的信号月；首信号锚定 ``anchor``，末信号要求执行日在窗内。"""
    anchor = anchor or spec.ANCHOR_DATE
    labels = list(days)
    index = {day: position for position, day in enumerate(labels)}
    if start not in index:
        raise ResearchError(f"研究窗起点不是开市日：{start}")
    if end not in index:
        raise ResearchError(f"研究窗终点不是开市日：{end}")

    by_month: dict[str, list[str]] = {}
    for day in labels:
        by_month.setdefault(day[:6], []).append(day)
    months = sorted(by_month)
    rows: list[dict[str, Any]] = []
    for position, month in enumerate(months):
        last = by_month[month][-1]
        exec_position = index[last] + 1
        if last == anchor:
            pass
        elif last < anchor:
            continue
        if position == 0:
            continue
        exec_day = labels[exec_position] if exec_position < len(labels) else None
        if exec_day is None or exec_day > end:
            continue
        formation = by_month[months[position - 1]][-1]
        rows.append(
            {
                "month": month,
                "formation_date": formation,
                "signal_date": last,
                "exec_date": exec_day,
                "formation_index": index[formation],
                "signal_index": index[last],
                "exec_index": exec_position,
                "warmup_start_index": max(0, index[formation] - spec.CORR_WINDOW - 1),
            }
        )
    if not rows:
        raise ResearchError("时钟为空")
    return rows


def window_days(days: Sequence[str], start: str, end: str) -> list[str]:
    return [day for day in days if start <= day <= end]


def describe_clock(rows: Sequence[dict[str, Any]], days: Sequence[str]) -> dict[str, Any]:
    """时钟诊断：左窗与信号月零重叠、跨年、短月与右删失。"""
    checks = []
    for row in rows:
        f_index = int(row["formation_index"])
        t_index = int(row["signal_index"])
        left_start = max(0, f_index - spec.CORR_WINDOW + 1)
        left = days[left_start:f_index + 1]
        checks.append(
            {
                "month": row["month"],
                "formation_date": row["formation_date"],
                "signal_date": row["signal_date"],
                "exec_date": row["exec_date"],
                "left_window_start": left[0],
                "left_window_days": len(left),
                "left_signal_overlap": bool(set(left) & set(days[f_index + 1:t_index + 1])),
                "signal_month_days": t_index - f_index,
                "last_signal_accepted": t_index + 1 < len(days),
            }
        )
    dropped = [row["month"] for row in checks if row["left_signal_overlap"]]
    if dropped:
        raise ResearchError(f"左窗与信号月重叠：{dropped}")
    return {
        "months": len(checks),
        "first_month": checks[0]["month"],
        "last_month": checks[-1]["month"],
        "first_signal_date": checks[0]["signal_date"],
        "first_exec_date": checks[0]["exec_date"],
        "last_signal_date": checks[-1]["signal_date"],
        "last_exec_date": checks[-1]["exec_date"],
        "cross_year_months": [row["month"] for row in checks if row["month"][4:] == "12"],
        "short_months": {
            row["month"]: row["signal_month_days"]
            for row in checks
            if row["signal_month_days"] < 20
        },
        "checks": checks,
    }
