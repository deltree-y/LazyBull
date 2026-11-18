"""逐日资格判定：主板 + 非金融 + 非 ST + 可交易 + 市值/成交额/价格历史。

顺序与协议 §4.2 一致：上市事实 → 主板 → 上市年龄 → 金融身份 → ST →
清洗交易标记 → 价格有效性 → 市值 → 成交额 → 价格历史。
每一行保留**全部**原因位（bitmask）与**首要**原因码。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

#: 申万一级金融代码（实测：480000 银行 / 490000 非银金融；SW2014 与 SW2021 同码）
FINANCIAL_L1_CODES: tuple[str, ...] = ("480000", "490000")

#: 申万一级金融名称（代码表口径，作为代码不可得时的兜底判定）
FINANCIAL_L1_NAMES: tuple[str, ...] = ("银行", "非银金融")

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
    "limit_up",
    "price_invalid",
    "market_cap_missing",
    "market_cap_out_of_range",
    "amount20_missing",
    "amount20_below_min",
    "insufficient_price_history",
)

#: 原因位（bitmask，可并存）；list_date_missing 不参与首要原因，只作附加标记
FLAGS: tuple[str, ...] = REASONS[1:] + ("list_date_missing",)


@dataclass
class UniverseResult:
    """资格判定的宽表结果（numpy 便于后续写盘与统计）。"""

    dates: list[str]
    codes: list[str]
    eligible: np.ndarray  # bool[D, C]
    reason: np.ndarray  # uint8[D, C]
    flags: np.ndarray  # uint16[D, C]
    n_valid_returns_120: np.ndarray  # int16[D, C]

    def frame(self, array: np.ndarray) -> pd.DataFrame:
        """numpy -> 宽表（便于调试与测试）。"""
        return pd.DataFrame(array, index=self.dates, columns=self.codes)


def _flag(flags: np.ndarray, name: str, mask: np.ndarray) -> None:
    """置位原因位（不改变首要原因）。"""
    flags |= (mask.astype(np.uint16) << FLAGS.index(name))


def industry_l1_names(
    code_table: pd.DataFrame | None, pit: pd.DataFrame
) -> dict[str, str]:
    """一级行业代码 -> 名称（可核映射，版本无关）。

    来源优先级：官方代码表 L1 行 > PIT 行自带 `l1_name`。两者都没有的旧码
    （如 440000/470000）保持未映射，调用方据此判定「金融身份未知」。
    """
    # 金融一级代码是权威常量（两版同码），名称缺失也必须能判定金融
    names: dict[str, str] = {"480000": FINANCIAL_L1_NAMES[0], "490000": FINANCIAL_L1_NAMES[1]}
    if code_table is not None and len(code_table):
        level = code_table["level"].astype(str).str.upper()
        l1 = code_table[level.str.startswith("L1")]
        for code, name in zip(
            l1["industry_code"].astype(str), l1["industry_name"].astype(str)
        ):
            if name.strip():
                names[str(code)] = name.strip()
    for code, name in zip(pit["l1_code"].astype(str), pit["l1_name"]):
        if isinstance(name, str) and name.strip():
            names.setdefault(str(code), name.strip())
    return names


def _industry_masks(
    pit: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
    l1_names: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """行业 PIT -> (行业身份已知, 金融) 两个布尔矩阵。

    硬规则（协议 §4.3）：
    - 一级代码有可核名称 ⇒ 身份已知；名称为金融族 ⇒ 金融；
    - 一级代码无可核名称（旧码/未映射）⇒ 身份未知，`known=False`，
      由 `industry_classification_unknown` 排除，**不得**当作非金融放进 UNKNOWN 桶。
    区间语义 `in_date <= T < out_date`；out_date 缺失视为开放区间。
    """
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
        name = l1_names.get(str(row.l1_code))
        if name is None:
            continue
        i0 = int(np.searchsorted(ordered, str(row.in_date), side="left"))
        i1 = int(np.searchsorted(ordered, str(row.out_date), side="left"))
        if i1 <= i0:
            continue
        known[i0:i1, j] = True
        if name in FINANCIAL_L1_NAMES or str(row.l1_code) in FINANCIAL_L1_CODES:
            financial[i0:i1, j] = True
    if len({str(day) for day in dates}) != len(dates):  # 日期唯一性守卫
        raise ValueError("交易日轴存在重复日期")
    return known, financial


def _list_age(
    stock_basic: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
    listing_calendar: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """上市交易日年龄矩阵与「上市日期缺失」位。

    年龄必须按**完整交易日历**计数：只用当前装载窗口会把窗口首日之后
    的股票误判为「上市不足 252 日」。
    """
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


def build_universe(
    prices: dict[str, pd.DataFrame],
    basic: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    stock_basic: pd.DataFrame,
    industry_pit: pd.DataFrame,
    *,
    min_list_days: int,
    min_mv_yuan: float,
    max_mv_yuan: float,
    min_amount20_yuan: float,
    min_valid_returns: int,
    amount_min_periods: int = 16,
    listing_calendar: Sequence[str] | None = None,
    l1_names: dict[str, str] | None = None,
) -> UniverseResult:
    """按协议 §4.2 逐日判定资格，返回宽表结果。"""
    dates = [str(day) for day in returns.index]
    codes = [str(code) for code in returns.columns]
    shape = (len(dates), len(codes))
    flags = np.zeros(shape, dtype=np.uint16)
    reason = np.zeros(shape, dtype=np.uint8)

    in_basic = set(stock_basic["ts_code"].astype(str))
    board_unknown = np.array([c not in in_basic for c in codes])[None, :].repeat(len(dates), axis=0)
    main_board_lookup = dict(
        zip(
            stock_basic["ts_code"].astype(str),
            stock_basic["market"].astype(str),
        )
    )
    main_board = np.array([main_board_lookup.get(c, "") == "主板" for c in codes])[
        None, :
    ].repeat(len(dates), axis=0)

    age, list_date_missing = _list_age(stock_basic, dates, codes, listing_calendar)
    names = l1_names if l1_names is not None else industry_l1_names(None, industry_pit)
    known, financial = _industry_masks(industry_pit, dates, codes, names)

    close = prices["close_adj"].to_numpy(dtype=float)
    total_mv = basic["total_mv"].to_numpy(dtype=float) * 1e4
    is_st = prices["is_st"].to_numpy(dtype=float) == 1.0
    suspended = prices["is_suspended"].to_numpy(dtype=float) == 1.0
    untradable = prices["tradable"].to_numpy(dtype=float) != 1.0
    limit_up = prices["is_limit_up"].to_numpy(dtype=float) == 1.0

    amount_frame = prices["amount"].astype(float) * 1000.0
    amount20 = amount_frame.rolling(20, min_periods=amount_min_periods).mean().to_numpy()
    valid_returns = returns.notna().astype(np.int16)
    n_valid = (
        pd.DataFrame(valid_returns.to_numpy(), index=returns.index, columns=returns.columns)
        .rolling(120, min_periods=120)
        .sum()
        .to_numpy()
    )
    n_valid_int = np.where(np.isnan(n_valid), -1, n_valid).astype(np.int16)

    conditions = [
        ("board_unknown", board_unknown),
        ("not_main_board", ~main_board),
        ("list_age_lt_252", age < min_list_days),
        ("industry_classification_unknown", ~known),
        ("financial_industry", financial),
        ("st", is_st),
        ("suspended_or_untradable", suspended | untradable),
        ("limit_up", limit_up),
        ("price_invalid", ~(np.isfinite(close) & (close > 0))),
        ("market_cap_missing", ~np.isfinite(total_mv)),
        (
            "market_cap_out_of_range",
            np.isfinite(total_mv) & ((total_mv < min_mv_yuan) | (total_mv > max_mv_yuan)),
        ),
        ("amount20_missing", ~np.isfinite(amount20)),
        ("amount20_below_min", np.isfinite(amount20) & (amount20 < min_amount20_yuan)),
        ("insufficient_price_history", n_valid_int < min_valid_returns),
    ]

    eligible = np.ones(shape, dtype=bool)
    counts: dict[str, int] = {}
    for name, mask in conditions:
        mask = np.asarray(mask, dtype=bool)
        counts[name] = int(mask.sum())
        _flag(flags, name, mask)
        # 首要原因 = 首个命中的失败位
        first = eligible & mask
        reason[first] = REASONS.index(name)
        eligible &= ~mask
    _flag(flags, "list_date_missing", list_date_missing)

    result = UniverseResult(
        dates=dates,
        codes=codes,
        eligible=eligible,
        reason=reason,
        flags=flags,
        n_valid_returns_120=n_valid_int,
    )
    result.condition_counts = counts  # type: ignore[attr-defined]
    return result


def daily_domain(result: UniverseResult) -> pd.DataFrame:
    """逐日资格台账：母截面数、合格数、各原因计数、是否满足 ≥100。"""
    rows = []
    for i, day in enumerate(result.dates):
        counts = np.bincount(result.reason[i], minlength=len(REASONS))
        row = {
            "trade_date": day,
            "mother_count": int(counts[0] + counts[1:].sum()),
            "eligible_count": int(counts[0]),
            "min100_status": "ok" if counts[0] >= 100 else "universe_below_minimum",
        }
        for idx, name in enumerate(REASONS[1:], start=1):
            row[f"reason_{name}"] = int(counts[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def reason_ledger(result: UniverseResult) -> pd.DataFrame:
    """全期原因汇总（行数 + 占比），用于排除台账。"""
    total = int(result.reason.size)
    counts = np.bincount(result.reason.reshape(-1), minlength=len(REASONS))
    return pd.DataFrame(
        {
            "reason": list(REASONS),
            "rows": counts.astype(int),
            "share": counts / max(1, total),
        }
    )


def industry_bucket_matrix(
    industry_pit: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
    l1_names: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """行业桶矩阵（int16，0=无映射）：FINANCIAL / UNKNOWN / 各一级代码。

    返回 (矩阵, 桶名列表)；桶名下标 = 矩阵值 - 1。
    """
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
        l1_code = str(row.l1_code)
        name = resolved.get(l1_code)
        if name is None:
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


def industry_lookup(
    industry_pit: pd.DataFrame,
    dates: Sequence[str],
    l1_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """给定日期的一级行业长表（供行业上限与诊断使用）。"""
    frame = industry_pit.copy()
    frame["in_date"] = frame["in_date"].astype(str)
    frame["out_date"] = frame["out_date"].fillna("99999999").astype(str)
    chunks = []
    for day in dates:
        active = frame[(frame["in_date"] <= day) & (frame["out_date"] > day)]
        if len(active) == 0:
            continue
        part = active[["ts_code", "l1_code", "l1_name"]].copy()
        part["trade_date"] = day
        chunks.append(part)
    if not chunks:
        return pd.DataFrame(columns=["trade_date", "ts_code", "l1_code", "l1_name"])
    merged = pd.concat(chunks, ignore_index=True)
    names = l1_names if l1_names is not None else industry_l1_names(None, industry_pit)

    def _bucket(code: object) -> str:
        key = str(code)
        name = names.get(key)
        if name is None:
            return "UNKNOWN"
        if name in FINANCIAL_L1_NAMES or key in FINANCIAL_L1_CODES:
            return "FINANCIAL"
        return key

    merged["industry_bucket"] = merged["l1_code"].map(_bucket)
    return merged


def stock_basic_frame(root: Path) -> pd.DataFrame:
    """读取清洁层股票基础表（列集已实测：无 list_status/delist_date）。"""
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
