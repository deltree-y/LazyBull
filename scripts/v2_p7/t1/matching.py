# -*- coding: utf-8 -*-
"""P7 T1 匹配组构造（冻结口径见设计 §6.4）。

冻结口径
========
- 对照池：信号日全市场**有完整基础行情**且当日没有任何 forecast/express/fina 公告记录的股票；
  「完整基础行情」在此固定为：未复权/复权开盘价与收盘价、前收、最高最低价全部有限且为正，
  当日总市值有限且为正，且过去 20 日收益与波动可算（不足 20 个有效观测即视为不完整）。
  该定义排除当日停牌（开盘价为 0）与新上市不足 20 日的股票，**不缩成 A 候选池**。
- 匹配变量：行业不一致（申万二级）、log 总市值、过去 20 日收益、过去 20 日波动，
  全部只用 T 日及以前信息。
- 标准化：连续变量按当日**对照池**的中位数/MAD 标准化并裁剪到 ``[-3, 3]``；
  MAD 为 0（或非有限）时该变量当日从距离中移除并计数，不默认各行业距离等价。
- 距离：行业指示 + 三个已裁剪标准化变量的绝对差之和。
- 每事件取 5 个不同对照，按 ``(距离, ts_code)`` 决胜；跨事件允许复用对照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.v2_p7.t1.dataplane import industry_at, listed_at

MATCH_VARIABLES: Tuple[str, ...] = ("log_total_mv", "ret20", "vol20")

#: 组状态（入口阶段：只有「已构造」与「不可解析」两种；成交状态由 execution 追加）
STATUS_OK = 0
STATUS_EVENT_COVARIATE_MISSING = 1
STATUS_INSUFFICIENT_CONTROLS = 2
STATUS_EVENT_PRICE_INVALID = 3

STATUS_REASONS: Dict[int, str] = {
    STATUS_OK: "matched",
    STATUS_EVENT_COVARIATE_MISSING: "event_covariate_missing",
    STATUS_INSUFFICIENT_CONTROLS: "insufficient_controls",
    STATUS_EVENT_PRICE_INVALID: "event_data_invalid",
}


@dataclass(frozen=True)
class MatchConfig:
    """匹配参数（全部来自设计冻结口径）。"""

    control_count: int = 5
    clip: float = 3.0
    mad_scale: float = 1.0
    mad_zero_epsilon: float = 1e-12
    variables: Tuple[str, ...] = MATCH_VARIABLES


@dataclass
class Standardization:
    """单日单个变量的稳健标准化结果。"""

    median: float
    mad: float
    dropped: bool
    z: np.ndarray


@dataclass
class DayMatch:
    """单日匹配结果（行顺序与输入事件顺序一致）。"""

    controls: np.ndarray
    distances: np.ndarray
    industry_match: np.ndarray
    industry_known: np.ndarray
    z_event: np.ndarray
    z_controls: np.ndarray
    medians: np.ndarray
    mads: np.ndarray
    dropped: np.ndarray
    n_candidates: int


@dataclass
class GroupTable:
    """六股匹配组账本（每行一个意图组）。"""

    group_id: np.ndarray
    source: np.ndarray
    event_stock: np.ndarray
    members: np.ndarray
    distances: np.ndarray
    industry_match: np.ndarray
    industry_known: np.ndarray
    z_event: np.ndarray
    z_controls: np.ndarray
    status: np.ndarray
    reason: List[str]
    knowledge_date: List[str]
    event_date: List[str]
    signal_pos: np.ndarray
    fingerprint: List[str]

    @property
    def size(self) -> int:
        return int(self.group_id.size)


@dataclass
class MatchingDiagnostics:
    """匹配质量与池构成诊断（全部只作报告）。"""

    candidates_per_day: List[int] = field(default_factory=list)
    events_per_day: List[int] = field(default_factory=list)
    mad_zero_days: Dict[str, int] = field(default_factory=dict)
    days_with_events: int = 0
    days_without_candidates: int = 0
    events_matched: int = 0
    events_insufficient_controls: int = 0
    events_covariate_missing: int = 0
    events_price_invalid: int = 0
    events_industry_unknown: int = 0

    def as_dict(self) -> Dict[str, Any]:
        candidates = np.asarray(self.candidates_per_day, dtype=float)
        events = np.asarray(self.events_per_day, dtype=float)
        return {
            "days_with_events": self.days_with_events,
            "days_without_candidates": self.days_without_candidates,
            "candidates_per_day": {
                "median": _safe_quantile(candidates, 0.5),
                "mean": float(candidates.mean()) if candidates.size else None,
                "min": float(candidates.min()) if candidates.size else None,
                "max": float(candidates.max()) if candidates.size else None,
            },
            "events_per_day": {
                "median": _safe_quantile(events, 0.5),
                "mean": float(events.mean()) if events.size else None,
                "max": float(events.max()) if events.size else None,
            },
            "mad_zero_days": dict(sorted(self.mad_zero_days.items())),
            "events_matched": self.events_matched,
            "events_insufficient_controls": self.events_insufficient_controls,
            "events_covariate_missing": self.events_covariate_missing,
            "events_price_invalid": self.events_price_invalid,
            "events_industry_unknown": self.events_industry_unknown,
        }


def _safe_quantile(values: np.ndarray, q: float) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, q))


def standardize_by_mad(
    event_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    clip: float = 3.0,
    mad_scale: float = 1.0,
    epsilon: float = 1e-12,
) -> Standardization:
    """单变量：用对照池中位数/MAD 标准化并裁剪；MAD=0 时标记移除。"""
    finite = candidate_values[np.isfinite(candidate_values)]
    if finite.size == 0:
        empty = np.full(event_values.shape, np.nan)
        return Standardization(float("nan"), float("nan"), True, empty)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)) * mad_scale)
    if not np.isfinite(mad) or mad <= epsilon:
        return Standardization(median, mad, True, np.full(event_values.shape, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.clip((event_values - median) / mad, -clip, clip)
    return Standardization(median, mad, False, z)


def _smallest_by_distance_and_key(row_distance: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    """按 (距离, 决胜键) 全局取前 k 个：同距离边界不因 argpartition 截断而丢键。

    `argpartition` 先定位第 k 小距离，再把「严格更近」与「恰好等于边界」分开处理：
    边界内的候选按冻结键全局排序取前若干，避免只在 k+1 子集内 lexsort 造成的错选。
    """
    if int(np.isfinite(row_distance).sum()) < k:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(row_distance, k - 1)[:k]
    boundary = float(np.max(row_distance[part]))
    less = np.flatnonzero(row_distance < boundary)
    equal = np.flatnonzero(row_distance == boundary)
    need = int(k - less.size)
    if equal.size <= need:
        chosen = np.concatenate([less, equal])
    else:
        order = np.argsort(keys[equal], kind="stable")[:need]
        chosen = np.concatenate([less, equal[order]])
    order = np.lexsort((keys[chosen], row_distance[chosen]))
    return chosen[order].astype(np.int64)


def _standardize_columns(
    *,
    event_covariates: np.ndarray,
    candidate_covariates: np.ndarray,
    config: MatchConfig,
    n_vars: int,
    z_event: np.ndarray,
    z_candidates: np.ndarray,
    medians: np.ndarray,
    mads: np.ndarray,
    dropped: np.ndarray,
) -> None:
    """逐协变量的 MAD 标准化（``match_one_day`` 主循环提取，就地写入输出缓冲）。"""
    for var in range(n_vars):
        result = standardize_by_mad(
            event_covariates[:, var],
            candidate_covariates[:, var],
            clip=config.clip,
            mad_scale=config.mad_scale,
            epsilon=config.mad_zero_epsilon,
        )
        medians[var] = result.median
        mads[var] = result.mad
        dropped[var] = result.dropped
        z_event[:, var] = result.z.astype(np.float32)
        if not result.dropped:
            candidate = standardize_by_mad(
                candidate_covariates[:, var],
                candidate_covariates[:, var],
                clip=config.clip,
                mad_scale=config.mad_scale,
                epsilon=config.mad_zero_epsilon,
            )
            z_candidates[:, var] = candidate.z


def _empty_day_match(n_events: int, n_vars: int, n_candidates: int, k: int) -> DayMatch:
    """空输入时的全 -1 / NaN 结果（``match_one_day`` 早退分支提取，行为不变）。"""
    return DayMatch(
        controls=np.full((n_events, k), -1, dtype=np.int64),
        distances=np.full((n_events, k), np.nan, dtype=np.float32),
        industry_match=np.zeros((n_events, k), dtype=bool),
        industry_known=np.zeros((n_events, k), dtype=bool),
        z_event=np.full((n_events, n_vars), np.nan, dtype=np.float32),
        z_controls=np.full((n_events, n_vars), np.nan, dtype=np.float32),
        medians=np.full(n_vars, np.nan),
        mads=np.full(n_vars, np.nan),
        dropped=np.zeros(n_vars, dtype=bool),
        n_candidates=n_candidates,
    )


def match_one_day(
    event_covariates: np.ndarray,
    candidate_covariates: np.ndarray,
    event_industry: np.ndarray,
    candidate_industry: np.ndarray,
    config: MatchConfig = MatchConfig(),
    candidate_keys: Optional[np.ndarray] = None,
    event_ids: Optional[np.ndarray] = None,
    candidate_ids: Optional[np.ndarray] = None,
) -> DayMatch:
    """单日匹配：返回每个事件的 5 个对照下标（不足则 -1）。

    ``candidate_keys`` 为决胜次序键（升序）；面板路径传入股票下标（已按 ts_code 升序），
    测试可直接传 ``arange``。``event_ids``/``candidate_ids`` 用于把**事件股自身**从
    当日对照池中剔除（公告日与信号日不总是同一天，事件股可能仍留在池内）。
    """
    n_events = int(event_covariates.shape[0])
    n_vars = int(event_covariates.shape[1])
    n_candidates = int(candidate_covariates.shape[0])
    k = int(config.control_count)
    keys = candidate_keys if candidate_keys is not None else np.arange(n_candidates)

    if n_events == 0 or n_candidates == 0:
        return _empty_day_match(n_events, n_vars, n_candidates, k)

    controls = np.full((n_events, k), -1, dtype=np.int64)
    distances = np.full((n_events, k), np.nan, dtype=np.float32)
    industry_match = np.zeros((n_events, k), dtype=bool)
    industry_known = np.zeros((n_events, k), dtype=bool)
    z_event = np.full((n_events, n_vars), np.nan, dtype=np.float32)
    z_controls = np.full((n_events, n_vars), np.nan, dtype=np.float32)
    medians = np.full(n_vars, np.nan)
    mads = np.full(n_vars, np.nan)
    dropped = np.zeros(n_vars, dtype=bool)

    z_candidates = np.full((n_candidates, n_vars), np.nan, dtype=np.float64)
    _standardize_columns(
        event_covariates=event_covariates,
        candidate_covariates=candidate_covariates,
        config=config,
        n_vars=n_vars,
        z_event=z_event,
        z_candidates=z_candidates,
        medians=medians,
        mads=mads,
        dropped=dropped,
    )

    both_known = (event_industry[:, None] >= 0) & (candidate_industry[None, :] >= 0)
    industry_gap = (both_known & (event_industry[:, None] != candidate_industry[None, :])).astype(
        np.float32
    )
    industry_same = (both_known & (event_industry[:, None] == candidate_industry[None, :])).astype(
        bool
    )
    distance = industry_gap.copy()
    for var in range(n_vars):
        if dropped[var]:
            continue
        distance += np.abs(z_event[:, var][:, None] - z_candidates[:, var][None, :]).astype(
            np.float32
        )

    # 资格前置（v2 协议 A）：行业未知的候选不入对照池。
    # 未知既不是"同行业"也不是"不同行业"，不得因免罚而获得选择优势。
    eligible = candidate_industry >= 0
    if not eligible.all():
        distance[:, ~eligible] = np.inf
    # 事件侧行业未知时行业项无定义，整行不可匹配（由调用方记为 insufficient_controls）
    if not np.all(event_industry >= 0):
        distance[event_industry < 0, :] = np.inf

    for row in range(n_events):
        row_distance = distance[row]
        if event_ids is not None and candidate_ids is not None:
            own = candidate_ids == event_ids[row]
            if own.any():
                row_distance = row_distance.copy()
                row_distance[own] = np.inf
        if int(np.isfinite(row_distance).sum()) < k:
            continue
        chosen = _smallest_by_distance_and_key(row_distance, keys, k)
        if chosen.size < k:
            continue
        controls[row] = chosen
        distances[row] = row_distance[chosen]
        industry_match[row] = industry_same[row][chosen].astype(bool)
        industry_known[row] = (both_known[row][chosen]).astype(bool)
        z_controls[row] = z_candidates[chosen].mean(axis=0)

    return DayMatch(
        controls=controls,
        distances=distances,
        industry_match=industry_match,
        industry_known=industry_known,
        z_event=z_event,
        z_controls=z_controls,
        medians=medians,
        mads=mads,
        dropped=dropped,
        n_candidates=n_candidates,
    )


def event_covariates(panel: Any, day_position: int, stock_indices: np.ndarray) -> np.ndarray:
    """事件侧协变量（允许当日停牌，但要求市值与 20 日历史可算）。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        mv = panel.total_mv[day_position, stock_indices].astype(np.float64)
        log_mv = np.where((mv > 0) & np.isfinite(mv), np.log(mv), np.nan)
    return np.column_stack(
        [
            log_mv,
            panel.ret20[day_position, stock_indices].astype(np.float64),
            panel.vol20[day_position, stock_indices].astype(np.float64),
        ]
    )


def candidate_mask(panel: Any, day_position: int) -> np.ndarray:
    """对照池掩码：完整基础行情 + 无任何三源公告记录。"""
    open_ok = np.isfinite(panel.open_u[day_position]) & (panel.open_u[day_position] > 0)
    close_ok = np.isfinite(panel.close_adj[day_position]) & (panel.close_adj[day_position] > 0)
    quote_ok = (
        np.isfinite(panel.high[day_position])
        & (panel.high[day_position] > 0)
        & np.isfinite(panel.low[day_position])
        & (panel.low[day_position] > 0)
        & np.isfinite(panel.pre_close[day_position])
        & (panel.pre_close[day_position] > 0)
    )
    mv = panel.total_mv[day_position]
    mv_ok = np.isfinite(mv) & (mv > 0)
    hist_ok = np.isfinite(panel.ret20[day_position]) & np.isfinite(panel.vol20[day_position])
    return open_ok & close_ok & quote_ok & mv_ok & hist_ok & (~panel.announce[day_position])


def candidate_covariates(panel: Any, day_position: int, candidates: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mv = panel.total_mv[day_position, candidates].astype(np.float64)
    return np.column_stack(
        [
            np.log(mv),
            panel.ret20[day_position, candidates].astype(np.float64),
            panel.vol20[day_position, candidates].astype(np.float64),
        ]
    )


def map_events_to_days(
    events: pd.DataFrame,
    panel: Any,
    *,
    evaluation_start_pos: int,
    horizon: int,
    entry_max_defer: int = 10,
    exit_max_defer: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """事件 → (信号交易日位置, 股票下标)；首个不早于公告日的交易日为信号日。

    冻结观察窗（设计 §6.1）：信号日之后必须能容纳「最长 10 日入场 + h 日持有 + 最长 10 日
    离场」。放不下的尾端事件登记为**右删失**，不得当作未成交（真实 0）或不可解析
    （T1 评审 R2）。
    """
    days = np.asarray(panel.days)
    ann = events["knowledge_date"].astype(str).to_numpy()
    positions = np.searchsorted(days, ann, side="left")
    in_range = positions < panel.n_days
    stats = {
        "events_in_file": int(len(events)),
        "events_beyond_cutoff_or_unmappable": int((~in_range).sum()),
        "events_before_evaluation_window": 0,
        "events_code_not_in_panel": 0,
        "events_right_censored_tail": 0,
    }
    frame = events.loc[in_range].copy()
    frame["signal_pos"] = positions[in_range].astype(np.int64)
    before = frame["signal_pos"] < int(evaluation_start_pos)
    stats["events_before_evaluation_window"] = int(before.sum())
    frame = frame[~before]
    required_last = (
        frame["signal_pos"] + int(horizon) + int(entry_max_defer) + int(exit_max_defer) + 1
    )
    censored = required_last > (panel.n_days - 1)
    stats["events_right_censored_tail"] = int(censored.sum())
    frame = frame[~censored]
    stock_position = frame["ts_code"].astype(str).map(panel.stock_index)
    unknown = stock_position.isna()
    stats["events_code_not_in_panel"] = int(unknown.sum())
    frame = frame[~unknown].copy()
    frame["event_stock"] = stock_position[~unknown].astype(np.int64)
    return frame.reset_index(drop=True), stats


def _match_day(
    panel: Any,
    source: str,
    day_position: int,
    ordered: pd.DataFrame,
    *,
    config: MatchConfig,
    diagnostics: MatchingDiagnostics,
    excluded: Dict[str, int],
    intent_domain: str,
) -> List[Dict[str, Any]]:
    """单日：构造当日全部匹配组记录（事件股自身不进对照池）。"""
    stocks = ordered["event_stock"].to_numpy(dtype=np.int64)
    diagnostics.days_with_events += 1
    diagnostics.events_per_day.append(int(len(stocks)))
    candidates = np.flatnonzero(candidate_mask(panel, day_position))
    diagnostics.candidates_per_day.append(int(candidates.size))
    if candidates.size == 0:
        diagnostics.days_without_candidates += 1

    observable = listed_at(panel, day_position, stocks)
    covars = event_covariates(panel, day_position, stocks)
    covars_ok = observable & np.isfinite(covars).all(axis=1)
    event_industry = industry_at(panel, day_position, stocks)
    day_match: Optional[DayMatch] = None
    if candidates.size and covars_ok.any():
        day_match = match_one_day(
            covars[covars_ok],
            candidate_covariates(panel, day_position, candidates),
            event_industry[covars_ok],
            industry_at(panel, day_position, candidates),
            config=config,
            candidate_keys=candidates,
            event_ids=stocks[covars_ok],
            candidate_ids=candidates,
        )
        for var_index, name in enumerate(config.variables):
            if day_match.dropped[var_index]:
                diagnostics.mad_zero_days[name] += 1

    rows: List[Dict[str, Any]] = []
    matched_index = 0
    for position, row in enumerate(ordered.itertuples(index=False)):
        if intent_domain == "listed" and not observable[position]:
            excluded["not_listed_at_signal"] += 1
            continue
        record: Dict[str, Any] = {
            "source": source,
            "event_stock": int(row.event_stock),
            "signal_pos": day_position,
            "knowledge_date": str(row.knowledge_date),
            "event_date": str(row.event_date),
            "fingerprint": str(row.fingerprint),
            "members": np.full(1 + config.control_count, -1, dtype=np.int64),
            "distances": np.full(config.control_count, np.nan, dtype=np.float32),
            "industry_match": np.zeros(config.control_count, dtype=bool),
            "industry_known": np.zeros(config.control_count, dtype=bool),
            "z_event": np.full(len(config.variables), np.nan, dtype=np.float32),
            "z_controls": np.full(len(config.variables), np.nan, dtype=np.float32),
            "status": STATUS_OK,
            "reason": STATUS_REASONS[STATUS_OK],
        }
        record["members"][0] = int(row.event_stock)
        if not covars_ok[position]:
            record["status"] = STATUS_EVENT_COVARIATE_MISSING
            record["reason"] = STATUS_REASONS[STATUS_EVENT_COVARIATE_MISSING]
            diagnostics.events_covariate_missing += 1
            excluded["event_covariate_missing"] += 1
            rows.append(record)
            continue
        if day_match is None or (day_match.controls[matched_index] < 0).any():
            record["status"] = STATUS_INSUFFICIENT_CONTROLS
            record["reason"] = STATUS_REASONS[STATUS_INSUFFICIENT_CONTROLS]
            diagnostics.events_insufficient_controls += 1
            if event_industry[position] < 0:
                diagnostics.events_industry_unknown += 1
        else:
            record["z_event"] = day_match.z_event[matched_index]
            record["members"][1:] = candidates[day_match.controls[matched_index]]
            record["distances"] = day_match.distances[matched_index]
            record["industry_match"] = day_match.industry_match[matched_index]
            record["industry_known"] = day_match.industry_known[matched_index]
            record["z_controls"] = day_match.z_controls[matched_index]
            diagnostics.events_matched += 1
        matched_index += 1
        rows.append(record)
    return rows


def build_groups(
    events: pd.DataFrame,
    panel: Any,
    *,
    source: str,
    evaluation_start_pos: int,
    config: MatchConfig = MatchConfig(),
    intent_domain: str = "listed",
    horizon: int = 10,
    entry_max_defer: int = 10,
    exit_max_defer: int = 10,
) -> Tuple[GroupTable, MatchingDiagnostics, Dict[str, Any]]:
    """按日构造六股匹配组（事件列 0 = 事件股，1..5 = 对照）。

    ``intent_domain``：
    - ``listed``（默认）：事件股在信号日必须**已上市**（IPO 前记录不计入意图域），停牌仍计入；
      这是匹配与共同执行的前置条件，属可核验的市场事实，不是经济筛选（T1 评审 R1）；
    - ``strict``：所有事件一律计入意图域，不可构造组改记 ``unresolved``。
    """
    mapped, map_stats = map_events_to_days(
        events,
        panel,
        evaluation_start_pos=evaluation_start_pos,
        horizon=int(horizon),
        entry_max_defer=int(entry_max_defer),
        exit_max_defer=int(exit_max_defer),
    )
    diagnostics = MatchingDiagnostics(
        mad_zero_days={name: 0 for name in config.variables},
    )
    excluded: Dict[str, int] = {"not_listed_at_signal": 0, "event_covariate_missing": 0}
    domain = {
        "source": source,
        "intent_domain": str(intent_domain),
        "horizon": int(horizon),
        **map_stats,
        "events_mappable": int(len(mapped)),
    }

    rows: List[Dict[str, Any]] = []
    for day_position, group in mapped.groupby("signal_pos", sort=True):
        ordered = group.sort_values(["ts_code", "fingerprint"], kind="mergesort").reset_index(
            drop=True
        )
        rows.extend(
            _match_day(
                panel,
                source,
                int(day_position),
                ordered,
                config=config,
                diagnostics=diagnostics,
                excluded=excluded,
                intent_domain=intent_domain,
            )
        )

    size = len(rows)
    table = GroupTable(
        group_id=np.arange(size, dtype=np.int64),
        source=np.asarray([row["source"] for row in rows], dtype=object),
        event_stock=np.asarray([row["event_stock"] for row in rows], dtype=np.int64),
        members=np.asarray([row["members"] for row in rows], dtype=np.int64).reshape(
            size, 1 + config.control_count
        ),
        distances=np.asarray([row["distances"] for row in rows], dtype=np.float32).reshape(
            size, config.control_count
        ),
        industry_match=np.asarray([row["industry_match"] for row in rows], dtype=bool).reshape(
            size, config.control_count
        ),
        industry_known=np.asarray([row["industry_known"] for row in rows], dtype=bool).reshape(
            size, config.control_count
        ),
        z_event=np.asarray([row["z_event"] for row in rows], dtype=np.float32).reshape(
            size, len(config.variables)
        ),
        z_controls=np.asarray([row["z_controls"] for row in rows], dtype=np.float32).reshape(
            size, len(config.variables)
        ),
        status=np.asarray([row["status"] for row in rows], dtype=np.int16),
        reason=[row["reason"] for row in rows],
        knowledge_date=[row["knowledge_date"] for row in rows],
        event_date=[row["event_date"] for row in rows],
        signal_pos=np.asarray([row["signal_pos"] for row in rows], dtype=np.int64),
        fingerprint=[row["fingerprint"] for row in rows],
    )
    domain["groups"] = int(size)
    domain["universe_excluded"] = {"not_listed_at_signal": int(excluded["not_listed_at_signal"])}
    domain["in_domain_reasons"] = {
        "event_covariate_missing": int(excluded["event_covariate_missing"]),
        "insufficient_controls": int(diagnostics.events_insufficient_controls),
    }
    domain["accounting_ok"] = bool(
        domain["events_in_file"]
        == domain["events_beyond_cutoff_or_unmappable"]
        + domain["events_before_evaluation_window"]
        + domain["events_code_not_in_panel"]
        + domain["events_right_censored_tail"]
        + domain["groups"]
        + domain["universe_excluded"]["not_listed_at_signal"]
    )
    domain["accounting_note"] = (
        "events_in_file = beyond_cutoff_or_unmappable + before_evaluation_window + "
        "code_not_in_panel + right_censored_tail + groups + universe_excluded(not_listed_at_signal)"
    )
    return table, diagnostics, domain


def matching_quality(table: GroupTable, config: MatchConfig = MatchConfig()) -> Dict[str, Any]:
    """匹配质量表：距离分布、行业一致率、MAD=0、标准化均值差、对照复用度。"""
    matched = table.status == STATUS_OK
    distances = table.distances[matched]
    flat = distances[np.isfinite(distances)]
    controls = table.members[matched][:, 1:]
    reuse_counts: Dict[int, int] = {}
    for value in controls.reshape(-1):
        if value < 0:
            continue
        reuse_counts[int(value)] = reuse_counts.get(int(value), 0) + 1
    reuse_values = np.asarray(list(reuse_counts.values()), dtype=float)
    if matched.any():
        same_pairs = int(table.industry_match[matched].sum())
        known_pairs = int(table.industry_known[matched].sum())
        total_pairs = int(matched.sum()) * int(config.control_count)
        industry_rate = (same_pairs / known_pairs) if known_pairs else None
        known_share = (known_pairs / total_pairs) if total_pairs else None
    else:
        industry_rate = None
        known_share = None
    smd = {}
    for var_index, name in enumerate(config.variables):
        difference = table.z_event[matched, var_index] - table.z_controls[matched, var_index]
        difference = difference[np.isfinite(difference)]
        smd[name] = float(difference.mean()) if difference.size else None
    return {
        "matched_groups": int(matched.sum()),
        "distance": {
            "mean": float(flat.mean()) if flat.size else None,
            "median": float(np.median(flat)) if flat.size else None,
            "p90": float(np.quantile(flat, 0.9)) if flat.size else None,
            "min": float(flat.min()) if flat.size else None,
            "max": float(flat.max()) if flat.size else None,
        },
        "industry_match_rate": industry_rate,
        "industry_known_pair_share": known_share,
        "standardized_mean_difference": smd,
        "control_reuse": {
            "unique_controls": int(len(reuse_counts)),
            "mean_reuse": float(reuse_values.mean()) if reuse_values.size else None,
            "max_reuse": int(reuse_values.max()) if reuse_values.size else None,
            "reused_share": (
                float((reuse_values > 1).sum() / reuse_values.size) if reuse_values.size else None
            ),
        },
    }
