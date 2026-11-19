"""10 日端点收益标签 + PIT 一级行业中性化 + 八态（协议 §7）。

标签 = `open_adj(calendar[T+11]) / close_adj(calendar[T+1]) - 1`：
- 端点用全市场交易日历定位，不按个股实际有行情的行数压缩；
- 行业基准只在**基础合格域**上取中位数（含无分钟数据的股票），每行业至少 5 只有效原始标签；
- 禁止顺延端点、禁止用估值 carry 冒充开盘价、禁止把未知退市回收填成 -100%。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import spec

STATUSES: tuple[str, ...] = (
    "valid",
    "right_censored",
    "entry_suspended",
    "exit_suspended",
    "exit_delisted",
    "source_gap",
    "industry_label_insufficient",
    "unresolved",
)
STATUS_CODE = {name: index for index, name in enumerate(STATUSES)}


def _last_observed(close: pd.DataFrame) -> np.ndarray:
    """每只股票最后一个有效收盘位置（无有效价格记 -1）。"""
    valid = close.notna().to_numpy()
    positions = np.arange(valid.shape[0]).reshape(-1, 1)
    return np.where(valid, positions, -1).max(axis=0)


def build_labels(
    prices: dict[str, pd.DataFrame], calendar: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """返回 (原始标签宽表, 状态与执行阻塞宽表)。索引=信号日，列=股票。"""
    close = prices["close_adj"]
    open_px = prices["open_adj"]
    entry_offset = int(spec.LABEL["entry_offset"])
    exit_offset = int(spec.LABEL["exit_offset"])
    total = len(calendar)

    entry_close = close.shift(-entry_offset)
    exit_open = open_px.shift(-exit_offset)
    entry_ok = entry_close.notna() & (entry_close > 0)
    exit_ok = exit_open.notna() & (exit_open > 0)

    suspended = prices["is_suspended"].astype(float)
    limit_down = prices["is_limit_down"].astype(float)
    limit_up = prices["is_limit_up"].astype(float)
    tradable = prices["tradable"].astype(float)

    entry_suspended = suspended.shift(-entry_offset) == 1.0
    exit_suspended = suspended.shift(-exit_offset) == 1.0
    entry_blocked = (limit_up.shift(-entry_offset) == 1.0) | (tradable.shift(-entry_offset) != 1.0)
    exit_blocked = (limit_down.shift(-exit_offset) == 1.0) | exit_suspended

    last_obs = _last_observed(close)
    rows = np.arange(total).reshape(-1, 1)
    exit_pos = rows + exit_offset
    censored = (exit_pos >= total) | (rows + entry_offset >= total)
    delisted = (exit_pos > last_obs.reshape(1, -1)) & (last_obs.reshape(1, -1) >= 0)

    # 有效标签 = 两端都有真实价格且两端都没有停牌（涨跌停只作执行旁路，不阻断价格标签）
    valid = entry_ok & exit_ok & (~entry_suspended) & (~exit_suspended) & (~delisted) & (~censored)
    value = (exit_open / entry_close.where(entry_ok) - 1.0).where(valid)

    status = np.full(value.shape, STATUS_CODE["unresolved"], dtype=np.uint8)
    # 入场端点真实停牌与"价格源缺口"分开登记；出场端点同样分开。
    # 停牌日 clean 层仍可能带上一笔收盘价，因此停牌判定**优先于**价格存在性。
    status = np.where(~entry_ok.to_numpy(), STATUS_CODE["source_gap"], status)
    status = np.where(entry_suspended.to_numpy(), STATUS_CODE["entry_suspended"], status)
    status = np.where(exit_suspended.to_numpy(), STATUS_CODE["exit_suspended"], status)
    status = np.where(delisted, STATUS_CODE["exit_delisted"], status)
    status = np.where(~exit_ok.to_numpy(), STATUS_CODE["source_gap"], status)
    status = np.where(censored, STATUS_CODE["right_censored"], status)
    status = np.where(valid.to_numpy(), STATUS_CODE["valid"], status)

    book = {
        "label_status": pd.DataFrame(status, index=value.index, columns=value.columns),
        "entry_execution_blocked": pd.DataFrame(
            entry_blocked.to_numpy(), index=value.index, columns=value.columns
        ),
        "exit_execution_blocked": pd.DataFrame(
            exit_blocked.to_numpy(), index=value.index, columns=value.columns
        ),
        "exit_suspended": pd.DataFrame(
            exit_suspended.to_numpy(), index=value.index, columns=value.columns
        ),
        "exit_delisted": pd.DataFrame(delisted, index=value.index, columns=value.columns),
    }
    return value, book


def industry_neutralize(
    raw: pd.DataFrame,
    status: pd.DataFrame,
    base_mask: np.ndarray,
    bucket_matrix: np.ndarray,
    *,
    min_members: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """在基础合格域上按 PIT 一级行业取中位数并中性化。

    Returns:
        (中性标签矩阵, 行业中位数矩阵)，NaN 表示不可算。
    """
    threshold = int(min_members or spec.LABEL["min_industry_members"])
    values = raw.to_numpy(dtype=float)
    base = np.asarray(base_mask, dtype=bool)
    valid = status.to_numpy() == STATUS_CODE["valid"]
    usable = base & valid
    neutral = np.full(values.shape, np.nan, dtype=np.float64)
    median_matrix = np.full(values.shape, np.nan, dtype=np.float64)

    for i in range(values.shape[0]):
        row_values = np.where(usable[i], values[i], np.nan)
        buckets = bucket_matrix[i]
        keep = np.isfinite(row_values)
        if not keep.any():
            continue
        frame = pd.DataFrame({"bucket": buckets[keep], "value": row_values[keep]})
        grouped = frame.groupby("bucket")["value"]
        medians = grouped.median()
        counts = grouped.size()
        accepted = medians[counts >= threshold]
        if len(accepted) == 0:
            continue
        lookup = accepted.to_dict()
        mapped = np.array([lookup.get(int(bucket), np.nan) for bucket in buckets], dtype=float)
        median_matrix[i] = mapped
        neutral[i] = values[i] - mapped
    return neutral, median_matrix


def extract_label_frame(
    raw: pd.DataFrame,
    status: pd.DataFrame,
    neutral: np.ndarray,
    median_matrix: np.ndarray,
    mask: np.ndarray,
    dates: Sequence[str],
    codes: Sequence[str],
    calendar: Sequence[str],
) -> pd.DataFrame:
    """抽取目标域行的标签、行业中位数与状态。"""
    rows, cols = np.nonzero(mask)
    date_values = np.asarray(list(dates), dtype=object)
    code_values = np.asarray(list(codes), dtype=object)
    out = pd.DataFrame({"trade_date": date_values[rows], "ts_code": code_values[cols]})
    raw_values = raw.to_numpy(dtype=np.float64)
    status_values = status.to_numpy().astype(np.uint8)
    out[spec.LABEL["raw_column"]] = raw_values[rows, cols]
    out["industry_median"] = median_matrix[rows, cols]
    out[spec.LABEL["column"]] = neutral[rows, cols]

    codes_status = status_values[rows, cols].copy()
    needs_industry = codes_status == STATUS_CODE["valid"]
    insufficient = needs_industry & ~np.isfinite(median_matrix[rows, cols])
    codes_status[insufficient] = STATUS_CODE["industry_label_insufficient"]
    out["label_status"] = np.asarray(STATUSES, dtype=object)[codes_status]

    position = {str(day): i for i, day in enumerate(calendar)}
    last_index = len(calendar) - 1
    entry_offset = int(spec.LABEL["entry_offset"])
    exit_offset = int(spec.LABEL["exit_offset"])

    def _planned(day: str, offset: int) -> str:
        index = position[str(day)] + offset
        return str(calendar[index]) if index <= last_index else ""

    out["entry_date"] = [_planned(str(d), entry_offset) for d in out["trade_date"]]
    out["label_end_date"] = [_planned(str(d), exit_offset) for d in out["trade_date"]]
    return out


def maturity_frame(status: pd.DataFrame, calendar: Sequence[str]) -> pd.DataFrame:
    """逐日标签状态汇总（成熟比例与缺口原因）。"""
    values = status.to_numpy().astype(np.uint8)
    total = values.shape[1]
    rows = []
    for i, day in enumerate(calendar):
        counts = np.bincount(values[i], minlength=len(STATUSES))
        rows.append(
            {
                "trade_date": str(day),
                "cells": int(total),
                **{name: int(counts[j]) for j, name in enumerate(STATUSES)},
            }
        )
    return pd.DataFrame(rows)


def pre_entry_return(prices: dict[str, pd.DataFrame], calendar: Sequence[str]) -> pd.DataFrame:
    """买入前不可捕获收益 `close_adj(T+1)/close_adj(T)-1`（协议 §7.3，仅报告用）。"""
    close = prices["close_adj"]
    return (close.shift(-1) / close - 1.0).where(close > 0)
