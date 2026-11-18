"""60 日端点收益标签与七态：valid / right_censored / entry_missing /
exit_suspended / exit_delisted / source_gap / unresolved。

标签 = `open_adj(T+61) / close_adj(T+1) - 1`（协议 §6.1），不复权净值、不扣成本。
禁止顺延端点、禁止用估值 carry 冒充开盘价、禁止把未知退市金额填成 -100%。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.v2_p7.defensive_ml import spec

STATUSES: tuple[str, ...] = (
    "valid",
    "right_censored",
    "entry_missing",
    "exit_suspended",
    "exit_delisted",
    "source_gap",
    "unresolved",
)
STATUS_CODE = {name: index for index, name in enumerate(STATUSES)}


def _last_observed(close: pd.DataFrame) -> np.ndarray:
    """每只股票最后一个有效收盘位置（无有效价格记 -1）。"""
    valid = close.notna().to_numpy()
    total = valid.shape[0]
    positions = np.arange(total).reshape(-1, 1)
    masked = np.where(valid, positions, -1)
    return masked.max(axis=0)


def build_labels(
    prices: dict[str, pd.DataFrame], calendar: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (标签宽表, 状态宽表)。索引=信号日，列=股票。"""
    close = prices["close_adj"]
    open_px = prices["open_adj"]
    entry_offset = int(spec.LABEL["entry_offset"])
    exit_offset = int(spec.LABEL["exit_offset"])
    total = len(calendar)

    entry_close = close.shift(-entry_offset)
    exit_open = open_px.shift(-exit_offset)
    entry_ok = entry_close.notna() & (entry_close > 0)
    exit_ok = exit_open.notna() & (exit_open > 0)
    valid = entry_ok & exit_ok
    value = (exit_open / entry_close.where(entry_ok) - 1.0).where(valid)

    suspended = prices["is_suspended"].astype(float)
    limit_down = prices["is_limit_down"].astype(float)
    limit_up = prices["is_limit_up"].astype(float)
    tradable = prices["tradable"].astype(float)
    exit_suspended = suspended.shift(-exit_offset) == 1.0
    entry_blocked = (limit_up.shift(-entry_offset) == 1.0) | (
        tradable.shift(-entry_offset) != 1.0
    )
    exit_blocked = (limit_down.shift(-exit_offset) == 1.0) | exit_suspended

    last_obs = _last_observed(close)
    rows = np.arange(total).reshape(-1, 1)
    exit_pos = rows + exit_offset
    censored = exit_pos >= total
    delisted = (exit_pos > last_obs.reshape(1, -1)) & (last_obs.reshape(1, -1) >= 0)

    # 优先级（后者覆盖前者）：unresolved < source_gap < exit_delisted <
    # exit_suspended < entry_missing < right_censored < valid
    status = np.full(value.shape, STATUS_CODE["unresolved"], dtype=np.uint8)
    status = np.where(~exit_ok.to_numpy(), STATUS_CODE["source_gap"], status)
    status = np.where(delisted, STATUS_CODE["exit_delisted"], status)
    status = np.where(exit_suspended.to_numpy(), STATUS_CODE["exit_suspended"], status)
    status = np.where(~entry_ok.to_numpy(), STATUS_CODE["entry_missing"], status)
    status = np.where(censored, STATUS_CODE["right_censored"], status)
    status = np.where(valid.to_numpy(), STATUS_CODE["valid"], status)

    shape_index = value.index
    shape_columns = value.columns
    book: dict[str, pd.DataFrame] = {
        "y_def_ret60": value,
        "label_status": pd.DataFrame(status, index=shape_index, columns=shape_columns),
        "entry_execution_blocked": pd.DataFrame(
            entry_blocked.to_numpy(), index=shape_index, columns=shape_columns
        ),
        "exit_execution_blocked": pd.DataFrame(
            exit_blocked.to_numpy(), index=shape_index, columns=shape_columns
        ),
        "exit_suspended": pd.DataFrame(
            exit_suspended.to_numpy(), index=shape_index, columns=shape_columns
        ),
        "exit_delisted": pd.DataFrame(
            delisted, index=shape_index, columns=shape_columns
        ),
    }
    return value, book


def status_table(status: pd.Series, label_end: pd.Series) -> pd.DataFrame:
    """长表状态（供数据集合成）。"""
    names = np.asarray(STATUSES, dtype=object)
    return pd.DataFrame(
        {
            "label_status": names[status.to_numpy().astype(int)],
            "label_end_date": label_end,
        }
    )


def extract_label_frame(
    book: dict[str, pd.DataFrame],
    eligible: np.ndarray,
    dates: list[str],
    codes: list[str],
    calendar: list[str],
) -> pd.DataFrame:
    """抽取合格行的标签与状态（含端点日期与两个执行阻塞位）。"""
    rows, cols = np.nonzero(eligible)
    out = pd.DataFrame(
        {
            "trade_date": np.asarray(dates, dtype=object)[rows],
            "ts_code": np.asarray(codes, dtype=object)[cols],
        }
    )
    out["y_def_ret60"] = book["y_def_ret60"].to_numpy()[rows, cols]
    status = book["label_status"].to_numpy().astype(np.uint8)
    out["label_status"] = np.asarray(STATUSES, dtype=object)[status[rows, cols]]
    position = {day: i for i, day in enumerate(calendar)}
    entry_offset = int(spec.LABEL["entry_offset"])
    exit_offset = int(spec.LABEL["exit_offset"])
    last_index = len(calendar) - 1

    def _planned(day: str, offset: int) -> str:
        """计划日：落在日历内即真实日期，超出则显式记未知（空串），不得截断成水位末日。"""
        index = position[day] + offset
        return calendar[index] if index <= last_index else ""

    out["entry_date"] = [_planned(d, entry_offset) for d in out["trade_date"]]
    out["label_end_date"] = [_planned(d, exit_offset) for d in out["trade_date"]]
    out["label_dates_in_calendar"] = out["label_end_date"].astype(str).str.len().eq(8)
    for name in ("entry_execution_blocked", "exit_execution_blocked"):
        out[name] = book[name].to_numpy()[rows, cols].astype(bool)
    return out


def maturity_frame(status_frame: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """逐日标签状态汇总（成熟比例与缺口原因）。"""
    status = status_frame.to_numpy().astype(np.uint8)
    total = status.shape[1]
    rows = []
    for i, day in enumerate(calendar):
        counts = np.bincount(status[i], minlength=len(STATUSES))
        rows.append(
            {
                "trade_date": day,
                "cells": int(total),
                **{name: int(counts[j]) for j, name in enumerate(STATUSES)},
            }
        )
    return pd.DataFrame(rows)
