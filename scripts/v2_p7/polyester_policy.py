"""涤纶价差的冻结月初单规则；只消费公开报价与日历，不读取股票收益。"""

from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

CODES = ("601233.SH", "603225.SH")
PTA_WEIGHT = 0.855
MEG_WEIGHT = 0.335


def _day(value: Any) -> str:
    text = str(value)
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"报价日期不是YYYYMMDD：{value!r}")
    pd.to_datetime(text, format="%Y%m%d", errors="raise")
    return text


def validate_records(records: list[dict]) -> list[dict]:
    """校验上游已封存的口径，不自行把未核实报价升级成可用。"""
    result, identities = [], set()
    for item in records:
        row = dict(item)
        key = row["source_id"]
        if not isinstance(key, str) or not key or key in identities:
            raise ValueError(f"报价source_id为空或重复：{key}")
        identities.add(key)
        for field in ("observation_date", "publication_date", "available_date"):
            row[field] = _day(row[field])
        if not row["observation_date"] <= row["publication_date"] <= row["available_date"]:
            raise ValueError(f"报价观测/公开/可用日期倒置：{key}")
        if not isinstance(row["quality_usable"], bool):
            raise ValueError(f"报价quality_usable必须是布尔：{key}")
        if row["quality_usable"]:
            if row["unit"] != "CNY/t" or not row.get("series_id"):
                raise ValueError(f"报价缺少统一单位或序列身份：{key}")
            for field in ("poy", "pta", "meg"):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{key}/{field}必须是数值")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{key}/{field}必须正且有限")
            row["spread"] = row["poy"] - PTA_WEIGHT * row["pta"] - MEG_WEIGHT * row["meg"]
        result.append(row)
    return result


def signal_dates(calendar: Sequence[str]) -> list[str]:
    """每月首个交易日；调用方传研究交易域，现金锚点不在此域。"""
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


def build_signals(
    records: list[dict],
    calendar: Sequence[str],
    *,
    delay_trading_days: int = 0,
    availability_calendar: Sequence[str] | None = None,
) -> pd.DataFrame:
    """保留全部月份和两股；缺失不开仓，另给未知全开情景所需标记。"""
    if type(delay_trading_days) is not int or delay_trading_days < 0:
        raise ValueError("信息延迟必须为非负整数交易日")
    records = validate_records(records)
    days = list(calendar)
    clock = list(availability_calendar) if availability_calendar is not None else days
    signal_dates(clock)
    if not set(days).issubset(clock):
        raise ValueError("可用时间日历必须包含全部研究信号日")
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
        average = sum(row["spread"] for row in selected[1:]) / 3 if known else None
        spread = current["spread"] if current else None
        eligible = bool(known and spread > 0 and spread > average)
        unknown_on = bool(eligible or (not known and (spread is None or spread > 0)))
        reason = (
            "spread_improving"
            if eligible
            else (
                "spread_not_improving"
                if known
                else (
                    "series_break"
                    if complete
                    else "missing_current" if current is None else "missing_prior_month"
                )
            )
        )
        for code in CODES:
            output.append(
                {
                    "signal_date": day,
                    "ts_code": code,
                    "eligible": eligible,
                    "unknown_on": unknown_on,
                    "input_known": known,
                    "reason": reason,
                    "spread": spread,
                    "prior_three_mean": average,
                    "source_ids": [row["source_id"] if row else None for row in selected],
                    "months": months,
                    "availability_cutoff": cutoff,
                    "delay_trading_days": delay_trading_days,
                }
            )
    return pd.DataFrame(output)


def signal_summary(frame: pd.DataFrame) -> dict:
    """按月份披露可判与缺失，不把同一行业信号的两股当独立观察。"""
    monthly = frame.drop_duplicates("signal_date")
    return {
        "months": len(monthly),
        "known_months": int(monthly.input_known.sum()),
        "eligible_months": int(monthly.eligible.sum()),
        "unknown_months": int((~monthly.input_known).sum()),
        "reason_counts": monthly.reason.value_counts().to_dict(),
        "independent_stock_observations_claimed": False,
    }
