"""快递经营月报的固定单规则；只读公告与日历，不接触价格/收益。"""

from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

CODES = ("002352.SZ", "002120.SZ", "002468.SZ", "600233.SH")


def _number(value: Any) -> float | None:
    """保留未知与零的区别。"""
    if value is None or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def signal_dates(calendar: Sequence[str]) -> list[str]:
    """每月20日后第一个交易日，不提前到当月20日。"""
    days = list(calendar)
    if days != sorted(set(days)):
        raise ValueError("信号日历必须唯一升序")
    result, seen = [], set()
    for day in days:
        pd.to_datetime(day, format="%Y%m%d", errors="raise")
        if int(day[6:]) > 20 and day[:6] not in seen:
            result.append(day)
            seen.add(day[:6])
    return result


def _choose_record(records: list[dict], code: str, day: str) -> tuple[dict | None, str]:
    previous = pd.Timestamp(day) - pd.offsets.MonthBegin(1)
    # MonthBegin(1)在月中回到本月月初，再减一天才是上一营业月。
    previous -= pd.Timedelta(days=1)
    matches = [
        row
        for row in records
        if row["ts_code"] == code
        and int(row["operating_year"]) == previous.year
        and int(row["month"]) == previous.month
    ]
    available = [row for row in matches if str(row["publication_date"]) < day]
    if not available:
        return None, "late_or_same_day" if matches else "no_previous_month_report"
    regular = [
        row
        for row in available
        if not any(word in row.get("title", "") for word in ("初步核算", "业绩预告"))
    ]
    chosen = max(
        regular or available,
        key=lambda row: (str(row["publication_date"]), str(row["announcement_id"])),
    )
    return chosen, "available"


def build_signals(records: list[dict], calendar: Sequence[str]) -> pd.DataFrame:
    """完整保留四股每月，包括未知/迟报；不按结果删行。"""
    rows = []
    for day in signal_dates(calendar):
        for code in CODES:
            source, availability = _choose_record(records, code, day)
            volume = _number(source.get("volume_yoy")) if source else None
            price = _number(source.get("canonical_price_yoy")) if source else None
            known = volume is not None and price is not None
            eligible = known and volume > 0 and price > 0
            unknown_on = (volume is None or volume > 0) and (price is None or price > 0)
            rows.append(
                {
                    "signal_date": day,
                    "ts_code": code,
                    "eligible": bool(eligible),
                    "unknown_on": bool(unknown_on),
                    "input_known": known,
                    "availability": availability,
                    "volume_yoy": volume,
                    "price_yoy": price,
                    "announcement_id": source["announcement_id"] if source else None,
                    "publication_date": source["publication_date"] if source else None,
                    "source_sha256": source["sha256"] if source else None,
                    "semantic_flags": source.get("semantic_flags", []) if source else [],
                    "reason": (
                        "positive_volume_and_price"
                        if eligible
                        else "unknown_input" if not known else "nonpositive_volume_or_price"
                    ),
                }
            )
    return pd.DataFrame(rows)


def signal_summary(frame: pd.DataFrame) -> dict:
    """覆盖率与方向以全部预定槽为分母，不把缺失称为经营恶化。"""
    return {
        "monthly_dates": int(frame.signal_date.nunique()),
        "slots": len(frame),
        "known_slots": int(frame.input_known.sum()),
        "eligible_slots": int(frame.eligible.sum()),
        "unknown_slots": int((~frame.input_known).sum()),
        "by_code": {
            code: {
                "slots": len(part),
                "known": int(part.input_known.sum()),
                "eligible": int(part.eligible.sum()),
                "unknown": int((~part.input_known).sum()),
            }
            for code, part in frame.groupby("ts_code", sort=True)
        },
        "availability": {str(k): int(v) for k, v in frame.availability.value_counts().items()},
    }
