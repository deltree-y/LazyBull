"""历史动态母域与 PIT 资格（协议 §5.1）。

判定顺序：历史上市事实 → 主板 → 上市交易日年龄 → 历史金融行业 → 历史 ST →
当日清洗交易标记 → 价格有效性 → 总市值（含端点）→ 20 日平均成交额（含 T，≥16 有效）→
PIT 一级行业已知。

**T 日涨跌停不进入资格排除**（与 AD-ML 不同）：T+1 实际可成交性由执行器判断，
涨跌停只在旁路登记。金融身份未知（无一级代码或名称不可核）判 `industry_classification_unknown`，
不得当作非金融并入 UNKNOWN 桶。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import spec

#: 申万一级金融代码（480000 银行 / 490000 非银金融；SW2014 与 SW2021 同码）
FINANCIAL_L1_CODES: tuple[str, ...] = ("480000", "490000")
FINANCIAL_L1_NAMES: tuple[str, ...] = ("银行", "非银金融")
UNKNOWN_INDUSTRY = "99999999"

#: 首要原因码（顺序即判定顺序；0 = 合格）
REASONS: tuple[str, ...] = (
    "eligible",
    "board_unknown",
    "not_main_board",
    "list_age_lt_252",
    "industry_classification_unknown",
    "financial_industry",
    "st",
    "suspended_or_untradable",
    "price_invalid",
    "market_cap_missing",
    "market_cap_out_of_range",
    "amount20_missing",
    "amount20_below_min",
)

#: 原因位（bitmask，可并存）
FLAGS: tuple[str, ...] = REASONS[1:] + ("list_date_missing", "limit_up", "limit_down")


@dataclass
class DomainResult:
    """资格判定的宽表结果（numpy 便于统计与落盘）。"""

    dates: list[str]
    codes: list[str]
    eligible: np.ndarray
    reason: np.ndarray
    flags: np.ndarray
    limit_up: np.ndarray
    limit_down: np.ndarray
    condition_counts: dict[str, int] = field(default_factory=dict)

    def frame(self, array: np.ndarray) -> pd.DataFrame:
        """numpy -> 宽表（便于调试与测试）。"""
        return pd.DataFrame(array, index=self.dates, columns=self.codes)


def _flag(flags: np.ndarray, name: str, mask: np.ndarray) -> None:
    """置位原因位（不改变首要原因）。"""
    flags |= np.asarray(mask, dtype=bool).astype(np.uint16) << FLAGS.index(name)


def industry_l1_names(code_table: pd.DataFrame | None, pit: pd.DataFrame) -> dict[str, str]:
    """一级行业代码 -> 名称（可核映射，版本无关）。"""
    names: dict[str, str] = {"480000": FINANCIAL_L1_NAMES[0], "490000": FINANCIAL_L1_NAMES[1]}
    if code_table is not None and len(code_table):
        level = code_table["level"].astype(str).str.upper()
        l1 = code_table[level.str.startswith("L1")]
        for code, name in zip(l1["industry_code"].astype(str), l1["industry_name"].astype(str)):
            if isinstance(name, str) and name.strip():
                names[str(code)] = name.strip()
    for code, name in zip(pit["l1_code"].astype(str), pit["l1_name"]):
        if isinstance(name, str) and name.strip():
            names.setdefault(str(code), name.strip())
    return names


def normalize_industry_code(value: object) -> str:
    """行业代码唯一归一（空串/NaN/pd.NA 统一为 `99999999`）。"""
    if value is None:
        return UNKNOWN_INDUSTRY
    try:
        if pd.isna(value):
            return UNKNOWN_INDUSTRY
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in ("nan", "none", "<na>") else UNKNOWN_INDUSTRY


def _industry_masks(
    pit: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
    l1_names: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """行业 PIT -> (行业身份已知, 金融) 两个布尔矩阵。"""
    code_index = {code: j for j, code in enumerate(codes)}
    known = np.zeros((len(dates), len(codes)), dtype=bool)
    financial = np.zeros_like(known)
    ordered = np.array(sorted(dates))
    frame = pit.copy()
    frame["in_date"] = frame["in_date"].astype(str)
    frame["out_date"] = frame["out_date"].fillna("99999999").astype(str)
    for row in frame.itertuples(index=False):
        j = code_index.get(str(row.ts_code))
        if j is None:
            continue
        l1_code = normalize_industry_code(getattr(row, "l1_code", None))
        if l1_code == UNKNOWN_INDUSTRY:
            continue
        name = l1_names.get(l1_code)
        if name is None:
            continue
        i0 = int(np.searchsorted(ordered, str(row.in_date), side="left"))
        i1 = int(np.searchsorted(ordered, str(row.out_date), side="left"))
        if i1 <= i0:
            continue
        known[i0:i1, j] = True
        if name in FINANCIAL_L1_NAMES or l1_code in FINANCIAL_L1_CODES:
            financial[i0:i1, j] = True
    if len({str(day) for day in dates}) != len(dates):
        raise ValueError("交易日轴存在重复日期")
    return known, financial


def _list_age(
    stock_basic: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
    listing_calendar: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """上市交易日年龄矩阵与「上市日期缺失」位（按完整交易日历计数）。"""
    full = np.array(sorted(listing_calendar or dates))
    full_pos = {str(day): i for i, day in enumerate(full)}
    day_positions = np.array([full_pos[str(day)] for day in dates], dtype=np.int32)
    info = stock_basic.set_index("ts_code")
    age = np.full((len(dates), len(codes)), -1, dtype=np.int32)
    missing = np.zeros((len(dates), len(codes)), dtype=bool)
    for j, code in enumerate(codes):
        if code not in info.index:
            missing[:, j] = True
            continue
        raw = info.at[code, "list_date"]
        if raw is None or pd.isna(raw) or not str(raw).strip():
            missing[:, j] = True
            continue
        pos = int(np.searchsorted(full, str(raw), side="left"))
        age[:, j] = day_positions - pos
    return age, missing


def build_domain(
    prices: dict[str, pd.DataFrame],
    basic: dict[str, pd.DataFrame],
    stock_basic: pd.DataFrame,
    industry_pit: pd.DataFrame,
    *,
    min_list_days: int | None = None,
    min_mv_yuan: float | None = None,
    max_mv_yuan: float | None = None,
    min_amount20_yuan: float | None = None,
    amount_min_periods: int | None = None,
    listing_calendar: Sequence[str] | None = None,
    l1_names: dict[str, str] | None = None,
) -> DomainResult:
    """按协议 §5.1 逐日判定资格。"""
    universe = spec.UNIVERSE
    min_days = int(min_list_days if min_list_days is not None else universe["min_list_days"])
    min_mv = float(min_mv_yuan if min_mv_yuan is not None else universe["min_total_mv_yuan"])
    max_mv = float(max_mv_yuan if max_mv_yuan is not None else universe["max_total_mv_yuan"])
    min_amount = float(
        min_amount20_yuan if min_amount20_yuan is not None else universe["min_amount20_yuan"]
    )
    min_periods = int(
        amount_min_periods if amount_min_periods is not None else universe["amount_min_periods"]
    )

    close = prices["close_adj"]
    dates = [str(day) for day in close.index]
    codes = [str(code) for code in close.columns]
    shape = (len(dates), len(codes))
    flags = np.zeros(shape, dtype=np.uint16)
    reason = np.zeros(shape, dtype=np.uint8)

    in_basic = set(stock_basic["ts_code"].astype(str))
    board_unknown = np.array([c not in in_basic for c in codes])[None, :].repeat(len(dates), axis=0)
    market_lookup = dict(zip(stock_basic["ts_code"].astype(str), stock_basic["market"].astype(str)))
    main_board = np.array([market_lookup.get(c, "") in set(universe["markets"]) for c in codes])[
        None, :
    ].repeat(len(dates), axis=0)

    age, list_date_missing = _list_age(stock_basic, dates, codes, listing_calendar)
    names = l1_names if l1_names is not None else industry_l1_names(None, industry_pit)
    known, financial = _industry_masks(industry_pit, dates, codes, names)

    close_values = close.to_numpy(dtype=float)
    total_mv = basic["total_mv"].reindex(index=close.index, columns=close.columns).to_numpy(
        dtype=float
    ) * float(spec.UNITS["total_mv_to_yuan"])
    is_st = prices["is_st"].to_numpy(dtype=float) == 1.0
    suspended = prices["is_suspended"].to_numpy(dtype=float) == 1.0
    untradable = prices["tradable"].to_numpy(dtype=float) != 1.0
    limit_up = prices["is_limit_up"].to_numpy(dtype=float) == 1.0
    limit_down = prices["is_limit_down"].to_numpy(dtype=float) == 1.0

    amount_frame = prices["amount"].astype(float) * float(spec.UNITS["daily_amount_to_yuan"])
    amount20 = (
        amount_frame.rolling(int(universe["amount_window"]), min_periods=min_periods)
        .mean()
        .to_numpy()
    )

    conditions = [
        ("board_unknown", board_unknown),
        ("not_main_board", ~main_board),
        ("list_age_lt_252", age < min_days),
        ("industry_classification_unknown", ~known),
        ("financial_industry", financial),
        ("st", is_st),
        ("suspended_or_untradable", suspended | untradable),
        ("price_invalid", ~(np.isfinite(close_values) & (close_values > 0))),
        ("market_cap_missing", ~np.isfinite(total_mv)),
        (
            "market_cap_out_of_range",
            np.isfinite(total_mv) & ((total_mv < min_mv) | (total_mv > max_mv)),
        ),
        ("amount20_missing", ~np.isfinite(amount20)),
        ("amount20_below_min", np.isfinite(amount20) & (amount20 < min_amount)),
    ]

    eligible = np.ones(shape, dtype=bool)
    counts: dict[str, int] = {}
    for name, mask in conditions:
        mask = np.asarray(mask, dtype=bool)
        counts[name] = int(mask.sum())
        _flag(flags, name, mask)
        first = eligible & mask
        reason[first] = REASONS.index(name)
        eligible &= ~mask
    _flag(flags, "list_date_missing", list_date_missing)
    _flag(flags, "limit_up", limit_up)
    _flag(flags, "limit_down", limit_down)

    return DomainResult(
        dates=dates,
        codes=codes,
        eligible=eligible,
        reason=reason,
        flags=flags,
        limit_up=limit_up,
        limit_down=limit_down,
        condition_counts=counts,
    )


def daily_domain(
    result: DomainResult,
    minute_complete: np.ndarray,
    minute_informed: np.ndarray,
    min_scored: int | None = None,
    mother_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """逐日资格台账（母域/基础合格/时段完整/分钟可用/可评分）。"""
    threshold = int(min_scored if min_scored is not None else spec.UNIVERSE["min_scored_per_day"])
    mother = np.asarray(mother_mask, dtype=bool) if mother_mask is not None else None
    rows = []
    for i, day in enumerate(result.dates):
        counts = np.bincount(result.reason[i], minlength=len(REASONS))
        base = int(counts[0])
        eligible = result.reason[i] == 0
        complete = int((eligible & minute_complete[i]).sum())
        informed = int((eligible & minute_informed[i]).sum())
        mother_count = int(mother[i].sum()) if mother is not None else int(counts.sum())
        rows.append(
            {
                "trade_date": day,
                "mother_count": mother_count,
                "base_eligible_count": base,
                "session_complete_count": complete,
                "minute_informed_count": informed,
                "scored_count": informed,
                "min100_status": "ok" if informed >= threshold else "universe_below_minimum",
                **{
                    f"reason_{name}": int(counts[idx])
                    for idx, name in enumerate(REASONS[1:], start=1)
                },
            }
        )
    return pd.DataFrame(rows)


def coverage_gate(domain: pd.DataFrame) -> dict[str, object]:
    """协议 §5.6 广度门：不足 100 的计划评分日占比 >1% 或连续 5 日不足即 not_ready。"""
    below = (domain["scored_count"] < int(spec.UNIVERSE["min_scored_per_day"])).to_list()
    share = float(np.mean(below)) if below else 0.0
    longest = 0
    current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    status = "ready"
    if share > float(spec.UNIVERSE["below_min_share_max"]):
        status = "not_ready"
    if longest >= int(spec.UNIVERSE["below_min_max_streak"]):
        status = "not_ready"
    return {
        "days": int(len(domain)),
        "below_days": int(sum(below)),
        "below_share": share,
        "longest_below_streak": int(longest),
        "threshold_share": float(spec.UNIVERSE["below_min_share_max"]),
        "status": status,
    }


def reason_ledger(result: DomainResult) -> pd.DataFrame:
    """全期原因汇总（行数 + 占比），用于排除台账。"""
    total = int(result.reason.size)
    counts = np.bincount(result.reason.reshape(-1), minlength=len(REASONS))
    return pd.DataFrame(
        {"reason": list(REASONS), "rows": counts.astype(int), "share": counts / max(1, total)}
    )


def industry_bucket_matrix(
    industry_pit: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
    l1_names: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """行业桶矩阵（int16，0=无映射）：FINANCIAL / UNKNOWN / 各一级代码。"""
    code_index = {code: j for j, code in enumerate(codes)}
    ordered = np.array(sorted(dates))
    matrix = np.zeros((len(dates), len(codes)), dtype=np.int16)
    names: list[str] = []
    ids: dict[str, int] = {}
    frame = industry_pit.copy()
    frame["in_date"] = frame["in_date"].astype(str)
    frame["out_date"] = frame["out_date"].fillna("99999999").astype(str)
    resolved = l1_names if l1_names is not None else industry_l1_names(None, industry_pit)
    for row in frame.itertuples(index=False):
        j = code_index.get(str(row.ts_code))
        if j is None:
            continue
        i0 = int(np.searchsorted(ordered, str(row.in_date), side="left"))
        i1 = int(np.searchsorted(ordered, str(row.out_date), side="left"))
        if i1 <= i0:
            continue
        l1_code = normalize_industry_code(getattr(row, "l1_code", None))
        name = resolved.get(l1_code)
        if l1_code == UNKNOWN_INDUSTRY or name is None:
            key = "UNKNOWN"
        elif name in FINANCIAL_L1_NAMES or l1_code in FINANCIAL_L1_CODES:
            key = "FINANCIAL"
        else:
            key = l1_code
        if key not in ids:
            names.append(key)
            ids[key] = len(names)
        matrix[i0:i1, j] = ids[key]
    return matrix, names


def stock_basic_frame(root: Path) -> pd.DataFrame:
    """读取清洁层股票基础表。"""
    return pd.read_parquet(Path(root) / "data/clean/stock_basic.parquet")


def industry_pit_frame(root: Path) -> pd.DataFrame:
    """读取申万行业 PIT 表。"""
    return pd.read_parquet(Path(root) / "data/clean/sw_industry_pit.parquet")


def industry_code_table_frame(root: Path) -> pd.DataFrame | None:
    """读取申万代码表（缺失返回 None，由上层降级为 PIT 自带名称）。"""
    path = Path(root) / "data/clean/sw_industry_code_table.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
