# -*- coding: utf-8 -*-
"""C1 预筛：月度分组残差与信号（唯一映射，见评审包 v3.1 §2）。

r3 修复（C1-PSR2-03）：
- L1/全市场回退使用**完整合格母截面**，允许重叠参照集合；
- 增加退市终点（最后可观测日）与完整月识别（截断月不生成信号）。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.v2_p7.t1.dataplane import Panel

MAIN_BOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")
FIN_L1_CODES = {"480000", "490000"}
FIN_L1_NAME_KEYWORDS = ("银行", "非银金融", "保险", "证券", "多元金融")
MV_MIN = 5.0e5
MV_MAX = 1.5e7
MIN_CELL = 5
MIN_INDUSTRY = 10
MIN_VALID_MONTHS = 8
FORMATION_MONTHS = 11


@dataclass(frozen=True)
class MonthlyPanel:
    days: Sequence[str]
    month_end_days: np.ndarray
    returns: np.ndarray
    residuals: np.ndarray
    signals: np.ndarray
    eligible: np.ndarray
    ref_keys: np.ndarray
    ref_types: np.ndarray
    l1_keys: np.ndarray
    amount_ma20: np.ndarray
    last_obs: np.ndarray


def _load_industry_codes(
    root: Path, stocks: Sequence[str], days: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str], Dict[str, str], Dict[str, int], Dict[str, int]]:
    path = Path(root) / "clean" / "sw_industry_pit.parquet"
    frame = pd.read_parquet(
        path, columns=["ts_code", "in_date", "out_date", "l2_code", "l1_code", "l1_name"]
    )
    code_index = {code: idx for idx, code in enumerate(stocks)}
    day_array = np.asarray([str(d) for d in days])
    n_days, n_stocks = len(days), len(stocks)
    l1_matrix = np.full((n_days, n_stocks), -1, dtype=np.int32)
    l2_matrix = np.full((n_days, n_stocks), -1, dtype=np.int32)
    l1_labels: Dict[str, int] = {}
    l2_labels: Dict[str, int] = {}
    l1_names: Dict[str, str] = {}
    l1_code_by_label: Dict[int, str] = {}
    l2_code_by_label: Dict[int, str] = {}
    for row in frame.itertuples(index=False):
        idx = code_index.get(str(row.ts_code))
        if idx is None:
            continue
        start = int(np.searchsorted(day_array, str(row.in_date), side="left"))
        end = int(np.searchsorted(day_array, str(row.out_date or "99999999"), side="left"))
        if end <= start:
            continue
        l1 = str(row.l1_code) if pd.notna(row.l1_code) else "-1"
        l2 = str(row.l2_code) if pd.notna(row.l2_code) else "-1"
        l1_id = l1_labels.setdefault(l1, len(l1_labels))
        l2_id = l2_labels.setdefault(l2, len(l2_labels))
        l1_code_by_label.setdefault(l1_id, l1)
        l2_code_by_label.setdefault(l2_id, l2)
        l1_matrix[start:end, idx] = l1_id
        l2_matrix[start:end, idx] = l2_id
        if l1 not in l1_names and isinstance(row.l1_name, str):
            l1_names[l1] = row.l1_name
    return l1_matrix, l2_matrix, l1_code_by_label, l1_names, l1_labels, l2_labels


def month_end_indices(days: Sequence[str], calendar_days: Sequence[str]) -> np.ndarray:
    """完整月的月末面板位置；截断月（日历月末晚于面板末日）排除。"""
    day_index = {str(d): i for i, d in enumerate(days)}
    cal = pd.Series([str(d) for d in calendar_days])
    frame = pd.DataFrame({"day": cal, "ym": pd.to_datetime(cal, format="%Y%m%d").dt.to_period("M")})
    last = frame.groupby("ym")["day"].max().sort_index()
    panel_last = str(days[-1])
    out: List[int] = []
    for day in last.tolist():
        if day > panel_last:
            continue
        idx = day_index.get(day)
        if idx is not None:
            out.append(idx)
    return np.asarray(out, dtype=np.int64)


def _main_board_mask(stocks: Sequence[str]) -> np.ndarray:
    return np.asarray([str(code)[:3] in MAIN_BOARD_PREFIXES for code in stocks], dtype=bool)


def _financial_mask(
    l1_labels: Dict[int, str], l1_names: Dict[str, str], n_labels: int
) -> np.ndarray:
    mask = np.zeros(n_labels, dtype=bool)
    for label, code in l1_labels.items():
        name = l1_names.get(code, "")
        if code in FIN_L1_CODES or any(keyword in name for keyword in FIN_L1_NAME_KEYWORDS):
            mask[label] = True
    return mask


def _size_tiers(mv: np.ndarray, eligible: np.ndarray, stocks: Sequence[str]) -> np.ndarray:
    tiers = np.full(len(mv), -1, dtype=np.int64)
    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        return tiers
    order = sorted(idx.tolist(), key=lambda i: (float(mv[i]), str(stocks[i])))
    n = len(order)
    for rank, i in enumerate(order):
        tiers[i] = min(2, int(rank * 3 / n))
    return tiers


def _reference_groups(
    l1: np.ndarray,
    l2: np.ndarray,
    tiers: np.ndarray,
    eligible: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 ref_type / ref_key。

    ref_type=2 → L2×规模（或 L2 全档合并）；ref_type=1 → 完整 L1；
    ref_type=0 → 完整全市场。L1/全市场参照允许与 L2 参照重叠。
    """
    ref_type = np.full(len(l1), -1, dtype=np.int64)
    ref_key = np.full(len(l1), -1, dtype=np.int64)
    for code in np.unique(l2[eligible]):
        members = np.flatnonzero(eligible & (l2 == code))
        if members.size < MIN_INDUSTRY:
            continue
        counts = np.bincount(tiers[members], minlength=3)
        if counts.min() < MIN_CELL:
            ref_type[members] = 2
            ref_key[members] = int(code) * 4 + 3
        else:
            ref_type[members] = 2
            ref_key[members] = int(code) * 4 + tiers[members].astype(np.int64)
    pending = eligible & (ref_type < 0)
    for code in np.unique(l1[pending]):
        l1_members = np.flatnonzero(eligible & (l1 == code))
        if l1_members.size < MIN_INDUSTRY:
            continue
        pending_members = np.flatnonzero(pending & (l1 == code))
        ref_type[pending_members] = 1
        ref_key[pending_members] = int(code)
    pending = eligible & (ref_type < 0)
    ref_type[pending] = 0
    ref_key[pending] = -1
    return ref_type, ref_key


def _mean_ex_self(ret: np.ndarray, eligible: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """按 keys 分组、剔除自身的等权均值；组内 <2 → NaN。"""
    out = np.full(len(ret), np.nan, dtype=np.float64)
    valid = eligible & np.isfinite(ret) & (keys >= 0)
    if not valid.any():
        return out
    g = keys[valid]
    r = ret[valid]
    _, inverse = np.unique(g, return_inverse=True)
    sums = np.bincount(inverse, weights=r)
    counts = np.bincount(inverse)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ex = (sums[inverse] - r) / (counts[inverse] - 1)
    ok = counts[inverse] >= 2
    vals = np.where(ok, r - mean_ex, np.nan)
    out[valid] = vals
    return out


def _residuals(
    ret: np.ndarray,
    eligible: np.ndarray,
    ref_type: np.ndarray,
    ref_key: np.ndarray,
    l1: np.ndarray,
) -> np.ndarray:
    """按 ref_type 分别使用 L2 / 完整 L1 / 完整全市场参照。"""
    resid = np.full(len(ret), np.nan, dtype=np.float64)
    # L2 参照
    mask2 = eligible & (ref_type == 2)
    if mask2.any():
        vals = _mean_ex_self(ret, mask2, ref_key)
        resid[mask2] = vals[mask2]
    # L1 参照：完整 L1 成员（含正常 L2 股）
    mask1 = eligible & (ref_type == 1)
    if mask1.any():
        vals = _mean_ex_self(ret, eligible, l1)
        resid[mask1] = vals[mask1]
    # 全市场参照
    mask0 = eligible & (ref_type == 0)
    if mask0.any():
        vals = _mean_ex_self(ret, eligible, np.zeros(len(ret), dtype=np.int64))
        resid[mask0] = vals[mask0]
    return resid


def build_monthly(
    panel: Panel,
    root: Path,
    *,
    calendar_days: Sequence[str] | None = None,
) -> MonthlyPanel:
    days = list(panel.days)
    stocks = list(panel.stocks)
    calendar = list(calendar_days) if calendar_days is not None else days
    l1_matrix, l2_matrix, l1_codes, l1_names, _, _ = _load_industry_codes(root, stocks, days)
    fin_mask = _financial_mask(l1_codes, l1_names, max(l1_codes.keys(), default=-1) + 1)
    main_mask = _main_board_mask(stocks)
    ends = month_end_indices(days, calendar)
    n_months, n_stocks = len(ends), len(stocks)
    last_obs = np.full(n_stocks, -1, dtype=np.int64)
    for idx in range(n_stocks):
        finite = np.flatnonzero(np.isfinite(panel.close_adj[:, idx]))
        if finite.size:
            last_obs[idx] = int(finite[-1])

    returns = np.full((n_months, n_stocks), np.nan, dtype=np.float64)
    residuals = np.full((n_months, n_stocks), np.nan, dtype=np.float64)
    signals = np.full((n_months, n_stocks), np.nan, dtype=np.float64)
    eligible_matrix = np.zeros((n_months, n_stocks), dtype=bool)
    ref_keys = np.full((n_months, n_stocks), -1, dtype=np.int64)
    ref_types = np.full((n_months, n_stocks), -1, dtype=np.int64)
    l1_keys = np.full((n_months, n_stocks), -1, dtype=np.int64)
    amount_ma20 = np.full((n_months, n_stocks), np.nan, dtype=np.float64)

    for j in range(1, n_months):
        prev, cur = int(ends[j - 1]), int(ends[j])
        with np.errstate(invalid="ignore", divide="ignore"):
            returns[j] = panel.close_adj[cur] / panel.close_adj[prev] - 1.0
        mv = panel.total_mv[prev]
        l1 = l1_matrix[prev]
        l2 = l2_matrix[prev]
        base = (
            main_mask
            & panel.listed[prev]
            & (prev <= last_obs)
            & (~panel.is_st[prev])
            & (~panel.suspended[prev])
            & np.isfinite(mv)
            & (mv >= MV_MIN)
            & (mv <= MV_MAX)
            & (l2 >= 0)
        )
        if fin_mask.size:
            base &= ~fin_mask[np.clip(l1, 0, fin_mask.size - 1)]
        tiers = _size_tiers(mv, base, stocks)
        ref_type, ref_key = _reference_groups(l1, l2, tiers, base)
        residuals[j] = _residuals(returns[j], base, ref_type, ref_key, l1)

        mv_sig = panel.total_mv[cur]
        l1_sig = l1_matrix[cur]
        l2_sig = l2_matrix[cur]
        sig_eligible = (
            main_mask
            & panel.listed[cur]
            & (cur <= last_obs)
            & (~panel.is_st[cur])
            & (~panel.suspended[cur])
            & np.isfinite(mv_sig)
            & (mv_sig >= MV_MIN)
            & (mv_sig <= MV_MAX)
            & (l2_sig >= 0)
        )
        if fin_mask.size:
            sig_eligible &= ~fin_mask[np.clip(l1_sig, 0, fin_mask.size - 1)]
        eligible_matrix[j] = sig_eligible
        ref_keys[j] = ref_key
        ref_types[j] = ref_type
        l1_keys[j] = l1
        lo = max(0, cur - 19)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            amount_ma20[j] = np.nanmean(panel.amount[lo : cur + 1], axis=0)

    for m in range(FORMATION_MONTHS, n_months):
        block = residuals[m - FORMATION_MONTHS : m]
        count = np.sum(np.isfinite(block), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.nanmean(block, axis=0)
                std = np.nanstd(block, axis=0, ddof=1)
                signal = mean / std
        signals[m] = np.where(
            (count >= MIN_VALID_MONTHS) & (std > 0) & np.isfinite(signal), signal, np.nan
        )

    return MonthlyPanel(
        days=days,
        month_end_days=ends,
        returns=returns,
        residuals=residuals,
        signals=signals,
        eligible=eligible_matrix,
        ref_keys=ref_keys,
        ref_types=ref_types,
        l1_keys=l1_keys,
        amount_ma20=amount_ma20,
        last_obs=last_obs,
    )


def top_codes(
    monthly: MonthlyPanel, stocks: Sequence[str], month: int, top_n: int = 6
) -> List[str]:
    if month < 0 or month >= monthly.signals.shape[0]:
        return []
    signal = monthly.signals[month]
    eligible = monthly.eligible[month] & np.isfinite(signal)
    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        return []
    order = sorted(
        idx.tolist(),
        key=lambda i: (
            -float(signal[i]),
            (
                -float(monthly.amount_ma20[month, i])
                if np.isfinite(monthly.amount_ma20[month, i])
                else np.inf
            ),
            str(stocks[i]),
        ),
    )
    return [str(stocks[i]) for i in order[:top_n]]
