"""分钟数据质量：栏位检查、单位换算、日线对账与异常调查线（协议 §5.5）。

约定：
- 异常调查线（量额相对误差 >0.5%、OHLC 价差 >1 个最小报价单位）只用于**调查分类**，
  不是删除股票或整库零容忍门；
- 缺行与零成交分开：真实零成交是合法观察，缺整根是数据缺口；
- 价格为零且无法确认语义的记录不当有效价格。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import sessions, spec

ROW_OK = "ok"
ANOMALY_VOLUME = "volume_rel_error"
ANOMALY_AMOUNT = "amount_rel_error"
ANOMALY_OHLC = "ohlc_tick_gap"
ANOMALY_ZERO_PRICE = "zero_price"
ANOMALY_MISSING = "missing_slots"


def validate_rows(frame: pd.DataFrame) -> dict[str, Any]:
    """基础栏位检查（返回计数，不修改数据）。"""
    if frame is None or len(frame) == 0:
        return {
            "rows": 0,
            "duplicates": 0,
            "time_conflicts": 0,
            "invalid_ohlc": 0,
            "negative_volume": 0,
            "zero_price": 0,
            "off_second": 0,
        }
    duplicates = int(frame.duplicated(subset=["ts_code", "trade_time"]).sum())
    open_px = pd.to_numeric(frame["open"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    vol = pd.to_numeric(frame["vol"], errors="coerce")
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    ohlc_bad = (high < low) | (close > high) | (close < low) | (open_px > high) | (open_px < low)
    seconds = frame["trade_time"].astype(str).str.slice(-2)
    return {
        "rows": int(len(frame)),
        "duplicates": duplicates,
        "invalid_ohlc": int((ohlc_bad & open_px.notna()).sum()),
        "negative_volume": int(((vol < 0) | (amount < 0)).sum()),
        "zero_price": int(((close <= 0) | (open_px <= 0)).sum()),
        "off_second": int((seconds != "00").sum()),
    }


def validate_standardized(frame: pd.DataFrame) -> dict[str, int]:
    """标准化槽位表的行质量（重复键、非法 OHLC、零价、负量额、越界槽）。"""
    duplicates = int(frame.duplicated(subset=["trade_date", "ts_code", "slot"]).sum())
    open_px = pd.to_numeric(frame["O"], errors="coerce")
    close = pd.to_numeric(frame["C"], errors="coerce")
    high = pd.to_numeric(frame["H"], errors="coerce")
    low = pd.to_numeric(frame["L"], errors="coerce")
    vol = pd.to_numeric(frame["Q"], errors="coerce")
    amount = pd.to_numeric(frame["A"], errors="coerce")
    ohlc_bad = (high < low) | (close > high) | (close < low) | (open_px > high) | (open_px < low)
    slots = pd.to_numeric(frame["slot"], errors="coerce")
    return {
        "duplicates": duplicates,
        "invalid_ohlc": int((ohlc_bad & open_px.notna()).sum()),
        "zero_price": int(((close <= 0) | (open_px <= 0)).sum()),
        "negative_volume": int(((vol < 0) | (amount < 0)).sum()),
        "off_second": 0,
        "bad_slot": int(((slots < 1) | (slots > int(spec.MINUTE["grid_slots"]))).sum()),
    }


def volume_in_shares(minute_vol: float) -> float:
    """分钟量 -> 股（分钟原生单位即股，保留函数以便单位证据集中一处）。"""
    return float(minute_vol) if spec.UNITS["minute_vol"] == "share" else float(minute_vol) * 100.0


def daily_volume_in_shares(daily_vol: float) -> float:
    """日线量（手）-> 股。"""
    return float(daily_vol) * float(spec.UNITS["daily_vol_to_share"])


def daily_amount_in_yuan(daily_amount: float) -> float:
    """日线额（千元）-> 元。"""
    return float(daily_amount) * float(spec.UNITS["daily_amount_to_yuan"])


def reconcile_day(
    minute_day: pd.DataFrame,
    daily_row: Mapping[str, Any],
    *,
    include_auction: bool = True,
) -> dict[str, Any]:
    """单股单日与日线对账（量额误差 + OHLC 边界差 + 异常分类）。"""
    result: dict[str, Any] = {
        "minute_rows": int(len(minute_day)),
        "flags": [],
        "reconcilable": False,
    }
    if minute_day is None or len(minute_day) == 0:
        result["flags"].append(ANOMALY_MISSING)
        return result
    frame = minute_day
    if not include_auction and "trade_time" in frame.columns:
        auction = f"{str(frame['trade_time'].iloc[0])[:10]} {spec.MINUTE['auction_time']}"
        frame = frame[frame["trade_time"].astype(str) != auction]
    minute_volume = float(pd.to_numeric(frame["vol"], errors="coerce").sum())
    minute_amount = float(pd.to_numeric(frame["amount"], errors="coerce").sum())
    daily_volume = daily_volume_in_shares(float(daily_row["vol"]))
    daily_amount = daily_amount_in_yuan(float(daily_row["amount"]))
    result["minute_volume"] = minute_volume
    result["daily_volume"] = daily_volume
    result["minute_amount"] = minute_amount
    result["daily_amount"] = daily_amount
    result["volume_rel_error"] = _rel_error(minute_volume, daily_volume)
    result["amount_rel_error"] = _rel_error(minute_amount, daily_amount)
    # 有符号误差：只看绝对值会把"偏高"和"偏低"混成一类（复核 2026-11-19 指出的口径缺陷）
    result["signed_volume_rel_error"] = _signed_error(minute_volume, daily_volume)
    result["signed_amount_rel_error"] = _signed_error(minute_amount, daily_amount)
    result["reconcilable"] = True
    threshold = float(spec.UNITS["minute_anomaly_volume_rel_error"])
    if result["volume_rel_error"] is not None and result["volume_rel_error"] > threshold:
        result["flags"].append(ANOMALY_VOLUME)
    if result["amount_rel_error"] is not None and result["amount_rel_error"] > threshold:
        result["flags"].append(ANOMALY_AMOUNT)
    minute_high = float(pd.to_numeric(frame["high"], errors="coerce").max())
    minute_low = float(pd.to_numeric(frame["low"], errors="coerce").min())
    result["high_gap"] = abs(minute_high - float(daily_row["high"]))
    result["low_gap"] = abs(minute_low - float(daily_row["low"]))
    tick = float(spec.UNITS["tick_size_yuan"])
    if result["high_gap"] > tick or result["low_gap"] > tick:
        result["flags"].append(ANOMALY_OHLC)
    return result


def _rel_error(minute_value: float, daily_value: float) -> float | None:
    if not np.isfinite(minute_value) or not np.isfinite(daily_value) or daily_value <= 0:
        return None
    return float(abs(minute_value - daily_value) / daily_value)


def _signed_error(minute_value: float, daily_value: float) -> float | None:
    """有符号相对误差：(分钟合计 − 日线) / 日线。"""
    if not np.isfinite(minute_value) or not np.isfinite(daily_value) or daily_value <= 0:
        return None
    return float((minute_value - daily_value) / daily_value)


def reconcile_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """把逐股对账记录汇总成表（异常分布与分年统计）。"""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(item) for item in records])
    if "flags" in frame.columns:
        frame["flag_text"] = frame["flags"].apply(
            lambda items: ",".join(items) if items else ROW_OK
        )
    return frame


def anomaly_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """异常调查线汇总（不是删除门）。"""
    if frame is None or frame.empty:
        return {"minutes_days": 0}
    volume = pd.to_numeric(frame.get("volume_rel_error"), errors="coerce")
    amount = pd.to_numeric(frame.get("amount_rel_error"), errors="coerce")
    return {
        "minute_days": int(len(frame)),
        "volume_error_median": float(volume.median()) if volume.notna().any() else None,
        "volume_error_p99": float(volume.quantile(0.99)) if volume.notna().any() else None,
        "amount_error_median": float(amount.median()) if amount.notna().any() else None,
        "amount_error_p99": float(amount.quantile(0.99)) if amount.notna().any() else None,
        "volume_over_line": int(
            (volume > float(spec.UNITS["minute_anomaly_volume_rel_error"])).sum()
        ),
        "amount_over_line": int(
            (amount > float(spec.UNITS["minute_anomaly_volume_rel_error"])).sum()
        ),
    }


def session_status(normalized: Mapping[str, Any]) -> dict[str, Any]:
    """标准化后的时段状态（供覆盖台账）。"""
    slots = normalized["slots"]
    complete, missing = sessions.session_complete(slots)
    return {
        "session_complete": bool(complete),
        "missing_slots": missing,
        "counts": normalized.get("counts", {}),
        "has_auction": normalized.get("auction") is not None,
    }


#: 协议 §5.5：核心槽为 k=1..47；k=48 缺失只作质量告警，不产生核心特征伪值
CORE_SLOTS: int = int(spec.MINUTE["core_slots"][1])
FULL_SLOTS: int = int(spec.MINUTE["grid_slots"])


def day_verdict(
    frame: pd.DataFrame | None,
    *,
    expected_slots: int | None = None,
    require_auction: bool = False,
    auction_present: bool | None = None,
) -> dict[str, Any]:
    """单股单日槽位/行质量裁定（不修改数据）。

    与 `sessions.session_complete` 同一口径：**核心要求 k=1..47 可解释**，
    k=48 缺失单独记 `tail_slot_missing` 告警（协议 §5.5），不当核心缺失。
    重复键、非法 OHLC、零价、负量额、越界槽任一命中即 `invalid_rows`。
    """
    expected = int(expected_slots or CORE_SLOTS)
    if frame is None or len(frame) == 0:
        return {
            "status": "no_rows",
            "rows": 0,
            "core_slots": 0,
            "missing_core": expected,
            "tail_slot_present": False,
        }
    slots = sorted(int(value) for value in frame["slot"].unique().tolist())
    core_present = [slot for slot in slots if 1 <= slot <= expected]
    missing_core = [slot for slot in range(1, expected + 1) if slot not in core_present]
    counters = validate_standardized(frame)
    status = "ok"
    if (
        counters["invalid_ohlc"]
        or counters["zero_price"]
        or counters["negative_volume"]
        or counters["duplicates"]
        or counters["bad_slot"]
    ):
        status = "invalid_rows"
    elif missing_core:
        status = "partial_slots"
    if require_auction and not auction_present and status == "ok":
        status = "missing_auction"
    warnings = []
    if FULL_SLOTS not in slots:
        warnings.append("tail_slot_missing")
    return {
        "status": status,
        "rows": int(len(frame)),
        "core_slots": len(core_present),
        "missing_core": len(missing_core),
        "missing_slots_sample": missing_core[:10],
        "tail_slot_present": FULL_SLOTS in slots,
        "warnings": warnings,
        "counters": counters,
    }


def reconciliation_disposition(check: Mapping[str, Any] | None) -> str:
    """对账处置：0.5% 是**调查线**（协议 §5.5），不是准入线。

    返回可解释处置或未解释标记：
    - `within_line`：量与额都在调查线内；
    - `ohlc_tick_rounding`：量额在线内，仅高低价差 1 个最小报价单位；
    - `volume_rounding`：量差 ≤1 手（100 股）；
    - `amount_rounding`：额差 ≤1 元或相对误差 ≤1e-6；
    - `unexplained_*`：超出上述机械解释，必须调查后才能发布；
    - `not_reconcilable`：缺日线或缺分钟行，无法对账。
    """
    if not check or not check.get("reconcilable"):
        return "not_reconcilable"
    line = float(spec.UNITS["minute_anomaly_volume_rel_error"])
    volume_rel = check.get("volume_rel_error")
    amount_rel = check.get("amount_rel_error")
    if volume_rel is None or amount_rel is None:
        return "not_reconcilable"
    volume_abs = abs(float(check.get("minute_volume", 0.0)) - float(check.get("daily_volume", 0.0)))
    amount_abs = abs(float(check.get("minute_amount", 0.0)) - float(check.get("daily_amount", 0.0)))
    if volume_rel <= line and amount_rel <= line:
        if check.get("high_gap", 0.0) > float(spec.UNITS["tick_size_yuan"]) or check.get(
            "low_gap", 0.0
        ) > float(spec.UNITS["tick_size_yuan"]):
            return "ohlc_tick_rounding"
        return "within_line"
    if volume_abs <= float(spec.UNITS["volume_rounding_shares"]) and amount_rel <= line:
        return "volume_rounding"
    if amount_abs <= float(spec.UNITS["amount_rounding_yuan"]) or amount_rel <= 1e-6:
        return "amount_rounding"
    if volume_rel > line and amount_rel > line:
        return "unexplained_both"
    return "unexplained_volume" if volume_rel > line else "unexplained_amount"


def classify_observation(record: Mapping[str, Any]) -> tuple[str, str | None]:
    """单 (股, 日) 观察的分类：返回 (bucket, reason)。

    bucket ∈ ok / accepted_exclusion / anomaly / unresolved_*
    规则：请求失败或未发出 → unresolved_failed；无行且停牌/未上市 → accepted_exclusion；
    无行且不可解释 → unresolved_empty；行存在但槽位/合法性不合格 → unresolved_incomplete；
    对账越线且**有机械解释**（整手/整元/1 个报价单位）→ anomaly（披露，不阻断）；
    越线且无法解释 → unresolved_anomaly（阻断，须调查）。
    """
    status = str(record.get("status"))
    if status in {"failed", "transport_error", "blocked_rate_limit", "not_attempted", "conflict"}:
        return "unresolved_failed", status
    if status == "empty" or status == "no_rows":
        if record.get("suspended"):
            return "accepted_exclusion", "suspended"
        if record.get("not_listed"):
            return "accepted_exclusion", "not_listed"
        return "unresolved_empty", "empty_without_suspension"
    quality_status = str(record.get("quality_status") or "ok")
    if quality_status != "ok":
        return "unresolved_incomplete", quality_status
    disposition = str(record.get("reconciliation") or "")
    if disposition.startswith("unexplained"):
        return "unresolved_anomaly", disposition
    if disposition in {"ohlc_tick_rounding", "volume_rounding", "amount_rounding"}:
        return "anomaly", disposition
    return "ok", None


def adjudicate_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """逐 (股, 交易日) 覆盖裁定（含预热日），区分已接受排除 / 异常 / 未解决。

    记录字段：`ts_code`、`trade_date`、`status`（请求状态）、`rows`、`slot_rows`、
    `quality_status`、`reconciliation`、`suspended`、`phase`（eligible/warmup）。
    """
    counts = {
        "ok": 0,
        "accepted_exclusion": 0,
        "anomaly": 0,
        "unresolved_empty": 0,
        "unresolved_failed": 0,
        "unresolved_incomplete": 0,
        "unresolved_anomaly": 0,
    }
    warmup = {"expected": 0, "ok": 0, "accepted_exclusion": 0, "unresolved": 0}
    unresolved: list[dict[str, Any]] = []
    for item in records:
        bucket, reason = classify_observation(item)
        counts[bucket] += 1
        if str(item.get("phase")) == "warmup":
            warmup["expected"] += 1
            if bucket == "ok":
                warmup["ok"] += 1
            elif bucket == "accepted_exclusion":
                warmup["accepted_exclusion"] += 1
            else:
                warmup["unresolved"] += 1
        if bucket.startswith("unresolved"):
            unresolved.append(
                {
                    "ts_code": item.get("ts_code"),
                    "trade_date": item.get("trade_date"),
                    "reason": reason,
                    "phase": item.get("phase"),
                }
            )
    total = len(records)
    accepted = counts["ok"] + counts["accepted_exclusion"] + counts["anomaly"]
    return {
        "expected_rows": total,
        "accepted_rows": accepted,
        "accepted_share": (accepted / total) if total else None,
        "counts": counts,
        "anomaly_count": counts["anomaly"],
        "warmup": warmup,
        "unresolved_count": len(unresolved),
        "unresolved_sample": unresolved[:50],
        "status": "ready" if not unresolved else "not_ready",
    }


def exclusion_ledger(domain_frame: pd.DataFrame) -> pd.DataFrame:
    """排除台账（逐原因行数与占比；保留的特征 NaN 不叫排除）。"""
    reason_columns = [name for name in domain_frame.columns if name.startswith("reason_")]
    total = int(domain_frame[reason_columns].to_numpy().sum())
    rows = []
    for name in reason_columns:
        count = int(domain_frame[name].sum())
        rows.append(
            {"reason": name.replace("reason_", ""), "rows": count, "share": count / max(1, total)}
        )
    return pd.DataFrame(rows)
