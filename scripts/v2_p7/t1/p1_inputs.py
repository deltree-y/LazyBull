# -*- coding: utf-8 -*-
"""P-1 重开输入面：意图域（L1）、涨跌停/停牌矩阵、行业成员簿与基准腿。

口径（与 `docs/experiments/p7_p1_prereg_20261113.md` 一致）
=========================================================
- L1 = L0 扣除：信号日未上市 / 右删失（``t+1+10+h > cutoff``）/ T 日无 PIT 行业归属；
- 缺行（clean/daily 无该行）是**证据未知**，不等于“停牌可证”：必须用 `raw/suspend`
  与 `raw/stk_limit` 补证据；两者都缺 ⇒ 该分支不可覆盖（unbounded）；
- 基准名单 T 日冻结；建仓=事件实际入场日开盘；出场=事件实际退出日开盘（未平仓=截止日）；
- 基准腿为毛收益（不叠加虚拟费用），成员缺价用最后可得复权收盘并登记回退天数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.t1 import dataplane

EVENT_SOURCES: tuple[str, ...] = ("forecast", "express", "fina_report")
SOURCE_CODE: Mapping[str, int] = {name: index for index, name in enumerate(EVENT_SOURCES)}


@dataclass
class IntentTable:
    """L1 意图（事件股 / 信号日位置 / 源编码）。"""

    stock: np.ndarray
    signal_pos: np.ndarray
    source_code: np.ndarray
    fingerprint: list[str]
    knowledge: list[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.stock.size)


@dataclass
class Extras:
    """行情之外的三态证据：行存在性 / 涨跌停价 / 停牌记录。"""

    has_row: np.ndarray
    up_limit: np.ndarray
    down_limit: np.ndarray
    suspend_flag: np.ndarray
    stats: Dict[str, Any] = field(default_factory=dict)


def _next_trading_day(days: Sequence[str], value: str) -> str | None:
    """非交易日公告顺延诊断：返回 >= value 的第一个交易日（无则 None）。"""
    ordered = np.asarray([str(day) for day in days])
    position = int(np.searchsorted(ordered, str(value), side="left"))
    return str(ordered[position]) if position < ordered.size else None


def build_intents(
    events: Mapping[str, pd.DataFrame],
    panel: dataplane.Panel,
    *,
    evaluation_start: str,
    horizon: int,
    entry_max_defer: int,
    cutoff_pos: int,
    calendar_mode: str = "legacy",
) -> IntentTable:
    """构造事前合格域 L1（唯一分母），逐项登记扣除原因。

    ``calendar_mode="next_open"``：非交易日公告的信号日顺延到首个政策允许交易日，
    保留原 ``knowledge_date``；入场仍为信号日 + 1 个交易日。
    """
    start_pos = panel.day_position(evaluation_start)
    stocks: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    fingerprints: list[str] = []
    knowledges: list[str] = []
    stats: Dict[str, Any] = {
        "horizon": int(horizon),
        "entry_max_defer": int(entry_max_defer),
        "sources": {},
        "deductions": {
            "before_evaluation_window": 0,
            "shifted_from_non_trading_day": 0,
            "date_not_in_calendar": 0,
            "date_not_in_calendar_recoverable": 0,
            "code_not_in_panel": 0,
            "code_not_in_panel_samples": [],
            "not_listed_at_signal": 0,
            "industry_unknown": 0,
            "right_censored_tail": 0,
        },
    }

    industry_matrix = panel.industry_matrix
    if industry_matrix is None:
        raise ValueError("P-1 重开必须使用 PIT 行业矩阵（industry_source=pit）")

    for source in EVENT_SOURCES:
        frame = events[source]
        if frame is None or not len(frame):
            stats["sources"][source] = {"events": 0, "intents": 0}
            continue
        code = frame["ts_code"].astype(str)
        knowledge = frame["knowledge_date"].astype(str)
        signal_days = knowledge
        shifted = pd.Series(False, index=knowledge.index)
        if calendar_mode == "next_open":
            next_days = knowledge.map(lambda value: _next_trading_day(panel.days, value))
            shifted = next_days.notna() & (next_days != knowledge)
            signal_days = next_days.where(shifted, knowledge)
        code_pos = code.map(panel.stock_index)
        day_pos = signal_days.map(panel.day_index)
        code_valid = code_pos.notna()
        day_valid = day_pos.notna()
        start_str = str(evaluation_start)
        # 先按信号日时间扣除：早于评价窗口的不计入 L1
        before = signal_days < start_str
        invalid_date = (signal_days >= start_str) & (~day_valid)
        missing_code = (signal_days >= start_str) & day_valid & (~code_valid)
        stats["deductions"]["before_evaluation_window"] += int(before.sum())
        stats["deductions"]["date_not_in_calendar"] += int(invalid_date.sum())
        stats["deductions"]["code_not_in_panel"] += int(missing_code.sum())
        stats["deductions"]["shifted_from_non_trading_day"] += int(
            (shifted & (signal_days >= start_str)).sum()
        )
        if invalid_date.any():
            years = signal_days[invalid_date].str.slice(0, 4).value_counts().sort_index()
            by_year = stats["deductions"].setdefault("date_not_in_calendar_by_year", {})
            for year, count in years.items():
                key = str(year)
                by_year[key] = int(by_year.get(key, 0)) + int(count)
        if missing_code.any():
            samples = sorted(code[missing_code].unique().tolist())[:10]
            existing = stats["deductions"].setdefault("code_not_in_panel_samples", [])
            existing.extend(item for item in samples if item not in existing)
        valid = code_valid & day_valid

        kept = pd.DataFrame(
            {
                "code": code_pos[valid].astype(np.int64),
                "day": day_pos[valid].astype(np.int64),
                "source": np.full(int(valid.sum()), SOURCE_CODE[source], dtype=np.int8),
                "fingerprint": frame.loc[valid, "fingerprint"].astype(str).to_numpy(),
                "knowledge": knowledge[valid].to_numpy(),
            }
        )
        kept = kept[kept["day"] >= start_pos].reset_index(drop=True)
        if not len(kept):
            stats["sources"][source] = {"events": int(len(frame)), "intents": 0}
            continue

        listed_ok = np.asarray(panel.listed)[kept["day"].to_numpy(), kept["code"].to_numpy()]
        industry_ok = (
            np.asarray(industry_matrix)[kept["day"].to_numpy(), kept["code"].to_numpy()] >= 0
        )
        right_censored = kept["day"].to_numpy() + 1 + int(entry_max_defer) + int(horizon) > int(
            cutoff_pos
        )
        stats["deductions"]["not_listed_at_signal"] += int((~listed_ok).sum())
        stats["deductions"]["industry_unknown"] += int((listed_ok & ~industry_ok).sum())
        stats["deductions"]["right_censored_tail"] += int(
            (listed_ok & industry_ok & right_censored).sum()
        )
        keep = listed_ok & industry_ok & ~right_censored
        stocks.append(kept.loc[keep, "code"].to_numpy(dtype=np.int32))
        positions.append(kept.loc[keep, "day"].to_numpy(dtype=np.int32))
        sources.append(kept.loc[keep, "source"].to_numpy(dtype=np.int8))
        fingerprints.extend(kept.loc[keep, "fingerprint"].tolist())
        knowledges.extend(kept.loc[keep, "knowledge"].tolist())
        stats["sources"][source] = {
            "events": int(len(frame)),
            "intents": int(keep.sum()),
        }

    if stocks:
        stock = np.concatenate(stocks)
        signal_pos = np.concatenate(positions)
        source_code = np.concatenate(sources)
    else:
        stock = np.zeros(0, dtype=np.int32)
        signal_pos = np.zeros(0, dtype=np.int32)
        source_code = np.zeros(0, dtype=np.int8)

    order = np.lexsort((stock, source_code, signal_pos))
    stats["intent_total"] = int(order.size)
    stats["deductions"]["right_censored_total"] = stats["deductions"]["right_censored_tail"]
    return IntentTable(
        stock=stock[order],
        signal_pos=signal_pos[order],
        source_code=source_code[order],
        fingerprint=[fingerprints[index] for index in order],
        knowledge=[knowledges[index] for index in order],
        stats=stats,
    )


def _load_daily_matrix(
    root: Path,
    name: str,
    value_columns: Sequence[str],
    panel: dataplane.Panel,
    *,
    cutoff: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """把 raw 按日分区（ts_code + 数值列）填充为 day×stock 矩阵。"""
    paths = []
    for path in sorted((root / name).glob("*.parquet")):
        stem = path.stem.replace("-", "")[:8]
        if stem and stem <= str(cutoff) and stem in panel.day_index:
            paths.append((stem, path))
    shape = (panel.n_days, panel.n_stocks)
    first = np.full(shape, np.nan, dtype=np.float32)
    second = np.full(shape, np.nan, dtype=np.float32)
    files = 0
    for _, path in paths:
        frame = pd.read_parquet(path, columns=["ts_code", *value_columns])
        day = path.stem.replace("-", "")[:8]
        day_pos = int(panel.day_index[day])
        codes = frame["ts_code"].astype(str).map(panel.stock_index)
        keep = codes.notna()
        if not keep.any():
            continue
        stock_pos = codes[keep].to_numpy(dtype=np.int64)
        first[day_pos, stock_pos] = frame.loc[keep, value_columns[0]].to_numpy(dtype=np.float32)
        if len(value_columns) > 1:
            second[day_pos, stock_pos] = frame.loc[keep, value_columns[1]].to_numpy(
                dtype=np.float32
            )
        files += 1
    stats = {"name": name, "files": files, "partitions_in_window": len(paths)}
    return first, second, stats


def load_extras(root: Path, panel: dataplane.Panel, *, cutoff: str) -> Extras:
    """加载涨跌停价与停牌记录（raw/stk_limit、raw/suspend），缺失即 NaN/False。"""
    up_limit, down_limit, limit_stats = _load_daily_matrix(
        root, "stk_limit", ("up_limit", "down_limit"), panel, cutoff=cutoff
    )
    suspend = np.zeros((panel.n_days, panel.n_stocks), dtype=bool)
    suspend_stats = {"name": "suspend", "files": 0, "rows": 0}
    for path in sorted((root / "suspend").glob("*.parquet")):
        day = path.stem.replace("-", "")[:8]
        if not day or day > str(cutoff) or day not in panel.day_index:
            continue
        frame = pd.read_parquet(path, columns=["ts_code", "suspend_type"])
        day_pos = int(panel.day_index[day])
        codes = frame["ts_code"].astype(str).map(panel.stock_index)
        keep = codes.notna() & (frame["suspend_type"].astype(str).str.upper() == "S")
        if keep.any():
            suspend[day_pos, codes[keep].to_numpy(dtype=np.int64)] = True
            suspend_stats["rows"] += int(keep.sum())
        suspend_stats["files"] += 1
    has_row = (
        np.asarray(panel.has_row)
        if panel.has_row is not None
        else np.zeros((panel.n_days, panel.n_stocks), dtype=bool)
    )
    return Extras(
        has_row=has_row,
        up_limit=up_limit,
        down_limit=down_limit,
        suspend_flag=suspend,
        stats={
            "stk_limit": limit_stats,
            "suspend": suspend_stats,
            "has_row_cells": int(has_row.sum()),
            "up_limit_cells": int(np.isfinite(up_limit).sum()),
            "suspend_cells": int(suspend.sum()),
        },
    )


def build_industry_members(
    panel: dataplane.Panel, *, start_pos: int, require_listed: bool = False
) -> Dict[int, Dict[int, np.ndarray]]:
    """逐日按 PIT 行业分组（-1 不入组）；返回 day → {industry: 股票下标数组}。

    ``require_listed=True`` 时只保留该日 ``listed`` 为真的成员（修正批次基准资格）。
    """
    matrix = panel.industry_matrix
    if matrix is None:
        raise ValueError("P-1 重开必须使用 PIT 行业矩阵")
    listed = np.asarray(panel.listed)
    members: Dict[int, Dict[int, np.ndarray]] = {}
    for day in range(int(start_pos), panel.n_days):
        row = np.asarray(matrix[day])
        order = np.argsort(row, kind="stable")
        sorted_codes = row[order]
        cuts = np.flatnonzero(np.diff(sorted_codes) != 0) + 1
        groups = np.split(order, cuts)
        day_map: Dict[int, np.ndarray] = {}
        for group in groups:
            code = int(row[group[0]])
            if code < 0:
                continue
            index = group.astype(np.int32)
            if require_listed:
                index = index[listed[day, index]]
            if index.size:
                day_map[code] = index
        members[day] = day_map
    return members


@dataclass
class BenchmarkBook:
    """行业等权（PIT）毛基准：T 日冻名单、事件实际入场日建仓、同一 e→x 窗口。

    ``mode="fixed"``：T 日资格（listed）+ 开盘/收盘价源分离 + 不可估值成员保留固定份额；
    ``mode="legacy"``：r6 原口径（仅两端 valid 成员均值）。
    """

    panel: dataplane.Panel
    members: Dict[int, Dict[int, np.ndarray]]
    last_valid: np.ndarray
    open_adj: np.ndarray
    close_adj: np.ndarray
    mode: str = "legacy"

    @classmethod
    def build(
        cls,
        panel: dataplane.Panel,
        members: Dict[int, Dict[int, np.ndarray]],
        *,
        mode: str = "legacy",
    ) -> "BenchmarkBook":
        open_adj = np.asarray(panel.open_adj)
        close_adj = np.asarray(panel.close_adj)
        valid = np.isfinite(close_adj) & (close_adj > 0)
        index = np.arange(panel.n_days, dtype=np.int32)[:, None]
        last_valid = np.where(valid, index, -1).astype(np.int32)
        np.maximum.accumulate(last_valid, axis=0, out=last_valid)
        return cls(
            panel=panel,
            members=members,
            last_valid=last_valid,
            open_adj=open_adj,
            close_adj=close_adj,
            mode=str(mode),
        )

    def _price(self, day: int, stocks: np.ndarray, *, clock: str) -> tuple[np.ndarray, np.ndarray]:
        """成员复权价与回退天数。

        ``clock="open"``：当日 open → 前收盘 → 前收盘前最后可得收盘；
        ``clock="close"``：当日 close → 当日及之前最后可得收盘；
        ``clock="legacy"``：当日 open → 当日 close → 最后可得收盘（r6 原口径）。
        """
        day = int(day)
        open_price = self.open_adj[day, stocks]
        close_price = self.close_adj[day, stocks]
        last_idx = self.last_valid[day, stocks]
        open_ok = np.isfinite(open_price) & (open_price > 0)
        close_ok = np.isfinite(close_price) & (close_price > 0)
        has_last = last_idx >= 0
        last_price = np.where(has_last, self.close_adj[np.maximum(last_idx, 0), stocks], np.nan)
        last_ok = has_last & np.isfinite(last_price) & (last_price > 0)
        if clock == "open":
            prev_idx = (
                self.last_valid[day - 1, stocks]
                if day > 0
                else np.full(stocks.size, -1, dtype=np.int32)
            )
            prev_ok = prev_idx >= 0
            prev_price = np.where(prev_ok, self.close_adj[np.maximum(prev_idx, 0), stocks], np.nan)
            prev_ok = prev_ok & np.isfinite(prev_price) & (prev_price > 0)
            price = np.where(open_ok, open_price, np.where(prev_ok, prev_price, last_price))
            days_back = np.where(
                open_ok,
                0,
                np.where(prev_ok, 1, np.where(last_ok, day - last_idx, -1)),
            ).astype(np.int32)
            return price, days_back
        if clock == "close":
            price = np.where(close_ok, close_price, last_price)
            days_back = np.where(close_ok, 0, np.where(last_ok, day - last_idx, -1)).astype(
                np.int32
            )
            return price, days_back
        price = np.where(open_ok, open_price, np.where(close_ok, close_price, last_price))
        days_back = np.where(
            open_ok | close_ok,
            0,
            np.where(last_ok, day - last_idx, -1),
        ).astype(np.int32)
        return price, days_back

    def return_window(
        self,
        signal_day: int,
        industry: int,
        event_stock: int,
        entry: int,
        exit_: int,
        *,
        exit_mode: str = "open",
    ) -> Dict[str, Any]:
        """返回基准毛收益与覆盖读数；``fixed`` 下不可估值成员保留固定份额。"""
        day_map = self.members.get(int(signal_day), {})
        members = day_map.get(int(industry), np.zeros(0, dtype=np.int32))
        if members.size:
            members = members[members != int(event_stock)]
        if not members.size:
            return {
                "ret": float("nan"),
                "ret_ex_unvalued": float("nan"),
                "members": 0,
                "unvalued": 0,
                "unvalued_weight": 0.0,
                "max_fallback": 0,
                "fallback_weight": 0.0,
                "coverage": "unavailable",
            }
        clock = "legacy" if self.mode == "legacy" else "open"
        buy, buy_fb = self._price(entry, members, clock=clock)
        sell_clock = "legacy" if self.mode == "legacy" else exit_mode
        sell, sell_fb = self._price(exit_, members, clock=sell_clock)
        valid = np.isfinite(buy) & (buy > 0) & np.isfinite(sell) & (sell > 0)
        unvalued = int((~valid).sum())
        n_all = int(members.size)
        unvalued_weight = unvalued / n_all
        if not valid.any():
            return {
                "ret": float("nan"),
                "ret_ex_unvalued": float("nan"),
                "members": n_all,
                "unvalued": unvalued,
                "unvalued_weight": float(unvalued_weight),
                "max_fallback": 0,
                "fallback_weight": 0.0,
                "coverage": "unavailable",
            }
        with np.errstate(invalid="ignore", divide="ignore"):
            returns = sell[valid] / buy[valid] - 1.0
        fallback = np.where(valid, np.maximum(buy_fb, sell_fb), 0)
        fallback_weight = float((np.asarray(fallback) > 0).sum() / n_all)
        if self.mode == "fixed":
            ret = float(np.sum(returns) / n_all)
            coverage = "full" if unvalued == 0 and int(valid.sum()) >= 5 else "partial"
        else:
            ret = float(np.mean(returns))
            coverage = "full" if int(valid.sum()) >= 5 else "thin"
        return {
            "ret": ret,
            "ret_ex_unvalued": float(np.mean(returns)),
            "members": n_all,
            "unvalued": unvalued,
            "unvalued_weight": float(unvalued_weight),
            "max_fallback": int(fallback.max()) if fallback.size else 0,
            "fallback_weight": fallback_weight,
            "coverage": coverage,
        }

    def member_detail(
        self,
        signal_day: int,
        industry: int,
        event_stock: int,
        entry: int,
        exit_: int,
        *,
        exit_mode: str = "open",
    ) -> pd.DataFrame:
        """成员级明细（抽样用）：身份、入/出时点、价源、权重、回退与收益。"""
        day_map = self.members.get(int(signal_day), {})
        members = day_map.get(int(industry), np.zeros(0, dtype=np.int32))
        if members.size:
            members = members[members != int(event_stock)]
        if not members.size:
            return pd.DataFrame()
        clock = "legacy" if self.mode == "legacy" else "open"
        buy, buy_fb = self._price(entry, members, clock=clock)
        sell, sell_fb = self._price(
            exit_, members, clock=("legacy" if self.mode == "legacy" else exit_mode)
        )
        valid = np.isfinite(buy) & (buy > 0) & np.isfinite(sell) & (sell > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            returns = np.where(valid, sell / buy - 1.0, np.nan)
        return pd.DataFrame(
            {
                "ts_code": [self.panel.stocks[index] for index in members],
                "signal_day": int(signal_day),
                "industry": int(industry),
                "entry": int(entry),
                "exit": int(exit_),
                "buy_price": buy,
                "sell_price": sell,
                "buy_fallback": buy_fb,
                "sell_fallback": sell_fb,
                "weight": 1.0 / members.size,
                "ret": returns,
                "valued": valid,
            }
        )
