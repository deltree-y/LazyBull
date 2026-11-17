"""生猪饲料比的冻结月初单规则；只消费官方报价与日历，不读取股票收益。

复杂度豁免：阶段专用经济预筛，不入生产链路（CLAUDE §7.8）；
如转为生产政策，须先拆分输入契约、时间选择和决策模块。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

CODES = ("002714.SZ", "300498.SZ")
DATE_FIELDS = ("observation_date", "publication_date", "available_date")


def _day(value: Any) -> str:
    text = str(value)
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"报价日期不是YYYYMMDD：{value!r}")
    pd.to_datetime(text, format="%Y%m%d", errors="raise")
    return text


def validate_records(records: list[dict]) -> list[dict]:
    """核验冻结输入；不消费不合格报价，不覆盖原始行或上游质量判断。"""
    result, identities = [], set()
    for item in records:
        row = dict(item)
        key = row.get("source_id")
        if not isinstance(key, str) or not key.strip() or key in identities:
            raise ValueError(f"报价source_id为空或重复：{key!r}")
        identities.add(key)
        for field in DATE_FIELDS:
            row[field] = _day(row.get(field))
        if not row["observation_date"] <= row["publication_date"] <= row["available_date"]:
            raise ValueError(f"报价观测/公开/可用日期倒置：{key}")
        if not isinstance(row.get("quality_usable"), bool):
            raise ValueError(f"报价quality_usable必须是布尔：{key}")
        if row["quality_usable"]:
            series = row.get("series_id")
            if row.get("unit") != "CNY/kg" or not isinstance(series, str) or not series.strip():
                raise ValueError(f"报价缺少统一单位或序列身份：{key}")
            for field in ("hog", "feed"):
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{key}/{field}必须是数值")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{key}/{field}必须正且有限")
            ratio = row["hog"] / row["feed"]
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"{key}/ratio必须正且有限")
            row["ratio"] = ratio
        result.append(row)
    return result


def signal_dates(calendar: Sequence[str]) -> list[str]:
    """每月首个交易日；调用方传研究交易域，不含初始现金锚点。"""
    days = [_day(value) for value in calendar]
    if not days or days != sorted(set(days)):
        raise ValueError("信号日历必须非空、唯一升序")
    result, seen = [], set()
    for day in days:
        if day[:6] not in seen:
            result.append(day)
            seen.add(day[:6])
    return result


def _choose(records: list[dict], month: str, cutoff: str) -> dict | None:
    candidates = [
        row
        for row in records
        if row["quality_usable"]
        and row["observation_date"][:6] == month
        and row["available_date"] < cutoff
    ]
    return max(
        candidates,
        key=lambda row: (row["observation_date"], row["publication_date"], row["source_id"]),
        default=None,
    )


def _reason(eligible: bool, known: bool, complete: bool, current: dict | None) -> str:
    if eligible:
        return "ratio_improving"
    if known:
        return "ratio_not_improving"
    if complete:
        return "series_break"
    return "missing_current" if current is None else "missing_prior_month"


def build_signals(
    records: list[dict],
    calendar: Sequence[str],
    *,
    delay_trading_days: int = 0,
    availability_calendar: Sequence[str] | None = None,
) -> pd.DataFrame:
    """四个连续自然月比值改善；信息延迟改变可用截止，不改变决策月份。"""
    if type(delay_trading_days) is not int or delay_trading_days < 0:
        raise ValueError("信息延迟必须为非负整数交易日")
    records = validate_records(records)
    days = [_day(value) for value in calendar]
    clock = (
        [_day(value) for value in availability_calendar]
        if availability_calendar is not None
        else days
    )
    signal_dates(clock)
    if not set(days).issubset(clock):
        raise ValueError("可用时间日历必须包含全部研究交易日")
    output = []
    for day in signal_dates(days):
        index = clock.index(day) - delay_trading_days
        cutoff = clock[index] if index >= 0 else "00000000"
        period = pd.Period(day[:6], freq="M")
        months = [(period - offset).strftime("%Y%m") for offset in range(1, 5)]
        selected = [_choose(records, month, cutoff) for month in months]
        current = selected[0]
        complete = all(row is not None for row in selected)
        same_series = complete and len({row["series_id"] for row in selected}) == 1
        known = bool(complete and same_series)
        average = sum(row["ratio"] for row in selected[1:]) / 3 if known else None
        ratio = current["ratio"] if current else None
        eligible = bool(known and ratio > average)
        for code in CODES:
            output.append(
                {
                    "signal_date": day,
                    "ts_code": code,
                    "eligible": eligible,
                    "unknown_on": bool(eligible or not known),
                    "input_known": known,
                    "reason": _reason(eligible, known, complete, current),
                    "ratio": ratio,
                    "prior_three_mean": average,
                    "source_ids": [row["source_id"] if row else None for row in selected],
                    "months": months,
                    "availability_cutoff": cutoff,
                    "delay_trading_days": delay_trading_days,
                }
            )
    return pd.DataFrame(output)


def signal_summary(frame: pd.DataFrame) -> dict:
    """按月份披露缺失与决定；两股共享行业信息，不算两个独立观测。"""
    monthly = frame.drop_duplicates("signal_date")
    return {
        "months": len(monthly),
        "known_months": int(monthly.input_known.sum()),
        "eligible_months": int(monthly.eligible.sum()),
        "unknown_months": int((~monthly.input_known).sum()),
        "reason_counts": monthly.reason.value_counts().to_dict(),
        "independent_stock_observations_claimed": False,
    }
