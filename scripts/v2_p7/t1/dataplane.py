# -*- coding: utf-8 -*-
"""P7 T1 数据面：日历、行情面板、协变量与输入身份摘要（全部只读）。

设计要点
========
- 只读 `data/raw`、`data/clean` 与 `configs/`，不触发任何 ensure/下载；
- 日历先截断到共同数据截止（`20261029`）再构造任何位置，日历尾部不参与成熟判定；
- 面板按 (交易日, 股票) 展平为二维数组，价格用 float64（收益比值精度），
  标志位用 bool，市值/成交额用 float32（仅作匹配与构成诊断）；
- 20 日收益/波动只用 **T 日及以前** 信息；窗口内不允许缺观测（不足 20 个有效日 → NaN），
  由匹配阶段按「协变量缺失」显式归类，不补 0、不插值。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DAILY_FIELDS: Tuple[str, ...] = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "pre_close",
    "close",
    "close_adj",
    "open_adj",
    "is_st",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "amount",
)

TRADING_DAYS_PER_YEAR = 250.0


def normalize_date(values: Iterable[Any]) -> pd.Series:
    """任意日期列归一为 8 位字符串（非法 → NaN，保留输入索引）。"""
    if isinstance(values, pd.Series):
        text = values.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    else:
        text = pd.Series(list(values), dtype="object")
        text = text.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    return text.where(text.str.fullmatch(r"\d{8}"), np.nan)


def load_trading_days(data_root: Path, cutoff: str) -> List[str]:
    """SSE 开市日（升序）并截断到共同数据截止（含）。"""
    calendar = pd.read_parquet(Path(data_root) / "raw" / "trade_cal.parquet")
    opened = calendar[(calendar["exchange"] == "SSE") & (calendar["is_open"] == 1)]
    days = sorted(str(int(value)) for value in opened["cal_date"])
    bounded = [day for day in days if day <= str(cutoff)]
    if not bounded:
        raise ValueError(f"共同截止 {cutoff} 之前没有任何交易日")
    return bounded


def first_clean_daily_date(data_root: Path) -> str:
    """clean/daily 最早分区日期（面板索引下界，含 20 日协变量回看）。"""
    files = sorted((Path(data_root) / "clean" / "daily").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 clean/daily 分区：{Path(data_root) / 'clean' / 'daily'}")
    return files[0].stem.replace("-", "")[:8]


def metadata_digest(root: Path, *, max_files: int = 0) -> Dict[str, Any]:
    """目录身份摘要（相对路径 + 字节数 + mtime_ns 的稳定 SHA256）。

    大目录（clean 分区上万文件）逐字节哈希代价过高，因此使用**元数据清单摘要**：
    任何文件新增/删除/改写都会改变大小或 mtime，从而改变摘要；事件源等小体量输入
    另用逐文件内容 SHA 覆盖（见 `run_p1` 的 identity 段）。
    """
    root = Path(root)
    if not root.exists():
        return {"exists": False, "root": root.as_posix(), "files": 0, "bytes": 0, "sha256": None}
    digest = hashlib.sha256()
    files = 0
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        files += 1
        total += stat.st_size
        if max_files and files >= max_files:
            break
    return {
        "exists": True,
        "root": root.as_posix(),
        "files": files,
        "bytes": total,
        "digest_kind": "metadata(relpath+size+mtime_ns)",
        "sha256": digest.hexdigest(),
    }


@dataclass
class Panel:
    """按 (交易日位置, 股票下标) 展平的只读面板。"""

    days: List[str]
    day_index: Mapping[str, int]
    stocks: List[str]
    stock_index: Mapping[str, int]
    open_u: np.ndarray
    open_adj: np.ndarray
    close_adj: np.ndarray
    pre_close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    amount: np.ndarray
    total_mv: np.ndarray
    is_st: np.ndarray
    suspended: np.ndarray
    limit_up: np.ndarray
    limit_down: np.ndarray
    industry: np.ndarray
    ret20: np.ndarray
    vol20: np.ndarray
    announce: np.ndarray
    listed: np.ndarray
    industry_since: np.ndarray
    share_mult: np.ndarray
    share_unknown: np.ndarray
    industry_source: Dict[str, Any]
    load_stats: Dict[str, Any]
    industry_matrix: Optional[np.ndarray] = None

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def n_stocks(self) -> int:
        return len(self.stocks)

    def day_position(self, date: str) -> int:
        position = self.day_index.get(str(date))
        if position is None:
            raise KeyError(f"日期不在截断日历内：{date}")
        return int(position)

    def position_or_none(self, date: str) -> Optional[int]:
        return self.day_index.get(str(date))

    def trade_price_ok(self, positions: np.ndarray, members: np.ndarray) -> np.ndarray:
        """真实开盘价存在（未复权与复权价均有限且为正）。"""
        clipped = np.clip(positions, 0, self.n_days - 1)
        raw = self.open_u[clipped[:, None], members]
        adj = self.open_adj[clipped[:, None], members]
        return np.isfinite(raw) & (raw > 0) & np.isfinite(adj) & (adj > 0)


def _iter_year_chunks(paths: Sequence[Path], chunk_years: int = 1) -> Iterable[List[Path]]:
    """按年份把分区文件分块，控制读取峰值内存。"""
    bucket: Dict[str, List[Path]] = {}
    for path in paths:
        year = path.stem.replace("-", "")[:4]
        bucket.setdefault(year, []).append(path)
    years = sorted(bucket)
    for start in range(0, len(years), chunk_years):
        chunk: List[Path] = []
        for year in years[start : start + chunk_years]:
            chunk.extend(sorted(bucket[year]))
        yield chunk


def _partition_dates(paths: Sequence[Path], cutoff: str, start: str = "") -> List[Path]:
    """分区名（YYYY-MM-DD）落在 [start, cutoff] 的文件（只读、不打开文件）。"""
    kept: List[Path] = []
    for path in paths:
        stem = path.stem.replace("-", "")[:8]
        if not stem or stem > str(cutoff):
            continue
        if start and stem < str(start):
            continue
        kept.append(path)
    return kept


def _collect_stock_codes(paths: Sequence[Path]) -> List[str]:
    codes: set[str] = set()
    for path in paths:
        frame = pd.read_parquet(path, columns=["ts_code"])
        codes.update(frame["ts_code"].astype(str).unique().tolist())
    return sorted(codes)


def _fill_matrix(
    matrix: np.ndarray,
    day_positions: np.ndarray,
    stock_positions: np.ndarray,
    values: np.ndarray,
    dtype,
) -> None:
    if values.size:
        matrix[day_positions, stock_positions] = values.astype(dtype, copy=False)


def load_industry_mapping(
    data_root: Path,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """申万二级行业映射与**纳入分类日**（`in_date`，PIT 下界）。

    `raw/shenwan_industry.parquet` 是**当前快照**：一条记录代表「自 `in_date` 起纳入当前
    分类」。因此它只能证明 `t >= in_date` 时的行业归属，早于该日一律视为**未知**（-1），
    由匹配阶段单独登记覆盖率——不得把当前标签外推到 `in_date` 之前（T1 评审 R5）。
    """
    path = Path(data_root) / "raw" / "shenwan_industry.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺少申万行业映射：{path}")
    frame = pd.read_parquet(path)
    if "sw_l2" not in frame.columns:
        raise ValueError("申万行业映射缺少 sw_l2 列")
    if "in_date" not in frame.columns:
        raise ValueError("申万行业映射缺少 in_date（无法做 PIT 屏蔽）")
    codes = frame["sw_l2"].astype(str).str.strip()
    mapping = dict(zip(frame["ts_code"].astype(str), codes))
    since = dict(zip(frame["ts_code"].astype(str), normalize_date(frame["in_date"])))
    info = {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "level": "sw_l2",
        "columns": list(frame.columns),
        "pit_rule": "仅当 T >= in_date 时使用该标签；更早一律未知(-1)",
        "snapshot_caveat": "当前快照、无历史进出日期：PIT 覆盖不完整，按输入限制登记",
    }
    return mapping, since, info


def _industry_matrix(
    root: Path,
    stocks: Sequence[str],
    days: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """PIT 行业矩阵：逐 (交易日, 股票) 取申万二级代码（与快照模式同粒度）。

    区间语义 ``in_date <= T < out_date``；无区间覆盖 → -1。
    ``sw_version=UNMAPPED``（pre-2014 旧码）同样计入已知——同一交易日全市场共用一套码制，
    同码比较成立；跨版本比较一律禁止（版本切换只在日与日之间，不在日内）。
    """
    path = Path(root) / "clean" / "sw_industry_pit.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少行业 PIT 区间表：{path}（先跑 scripts/download_sw_industry_history.py）"
        )
    columns = ["ts_code", "in_date", "out_date", "l2_code", "sw_version"]
    frame = pd.read_parquet(path, columns=columns)
    code_index = {code: index for index, code in enumerate(stocks)}
    day_array = np.asarray([str(day) for day in days])
    matrix = np.full((len(days), len(stocks)), -1, dtype=np.int32)

    labels: Dict[str, int] = {}
    ts_codes = frame["ts_code"].astype(str).to_numpy()
    in_dates = frame["in_date"].astype(str).to_numpy()
    out_dates = frame["out_date"].fillna("99999999").astype(str).to_numpy()
    level_codes = frame["l2_code"].astype(str).to_numpy()

    matched = 0
    for row in range(len(frame)):
        index = code_index.get(ts_codes[row])
        if index is None:
            continue
        start = int(np.searchsorted(day_array, in_dates[row], side="left"))
        end = int(np.searchsorted(day_array, out_dates[row], side="left"))
        if end <= start:
            continue
        label = level_codes[row]
        if label not in labels:
            labels[label] = len(labels)
        matrix[start:end, index] = labels[label]
        matched += 1

    info = {
        "mode": "pit",
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "matched_rows": int(matched),
        "level": "sw_l2_code",
        "pit_rule": "in_date <= T < out_date；无区间覆盖 → -1",
        "version_policy": "UNMAPPED（pre-2014 旧码）计为已知；跨版本不比较",
        "versions": {str(k): int(v) for k, v in frame["sw_version"].value_counts().items()},
    }
    return matrix, info


def compute_covariates(
    close_adj: np.ndarray,
    lookback: int = 20,
    chunk: int = 512,
    *,
    ffill_limit: int = 20,
    min_obs: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """过去 20 日收益与日收益波动（仅用 T 日及以前）。

    缺失处理（PIT 口径，正式运行固定）：
    - 停牌/无成交日不产生新信息：收盘价按**最多 ``ffill_limit`` 个交易日**向前延续，
      超过该长度视为该股已不可连续观测，不再外推；
    - 20 日收益要求 T 与 T−20 的延续收盘都存在；
    - 20 日波动要求窗口内**有效日收益不少于 ``min_obs`` 个**（不足即 NaN，不插值）。
    """
    n_days, n_stocks = close_adj.shape
    ret20 = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
    vol20 = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
    if n_days <= lookback:
        return ret20, vol20
    filled_close = pd.DataFrame(close_adj).ffill(limit=int(ffill_limit)).to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret20[lookback:] = (filled_close[lookback:] / filled_close[:-lookback] - 1.0).astype(
            np.float32
        )
    for start in range(0, n_stocks, chunk):
        stop = min(start + chunk, n_stocks)
        block = filled_close[:, start:stop]
        with np.errstate(invalid="ignore", divide="ignore"):
            daily = np.full_like(block, np.nan)
            daily[1:] = block[1:] / block[:-1] - 1.0
        valid = np.isfinite(daily)
        filled = np.where(valid, daily, 0.0)
        zeros = np.zeros((1, block.shape[1]), dtype=np.float64)
        cum1 = np.vstack([zeros, np.cumsum(filled, axis=0)])
        cum2 = np.vstack([zeros, np.cumsum(filled * filled, axis=0)])
        cumn = np.vstack([zeros, np.cumsum(valid.astype(np.float64), axis=0)])
        for day in range(lookback, n_days):
            lo = day - lookback + 1
            count = cumn[day + 1] - cumn[lo]
            total = cum1[day + 1] - cum1[lo]
            total_sq = cum2[day + 1] - cum2[lo]
            enough = count >= int(min_obs)
            with np.errstate(invalid="ignore", divide="ignore"):
                variance = (total_sq - total * total / np.maximum(count, 1.0)) / np.maximum(
                    count - 1.0, 1.0
                )
            values = np.where(enough, np.sqrt(np.maximum(variance, 0.0)), np.nan)
            vol20[day, start:stop] = values.astype(np.float32)
    return ret20, vol20


def _allocate_arrays(n_days: int, n_stocks: int) -> Dict[str, np.ndarray]:
    """分配面板数组（价格 float64、市值/成交额 float32、标志位 bool）。"""
    shape = (n_days, n_stocks)
    return {
        "open_u": np.full(shape, np.nan, dtype=np.float64),
        "open_adj": np.full(shape, np.nan, dtype=np.float64),
        "close_adj": np.full(shape, np.nan, dtype=np.float64),
        "pre_close": np.full(shape, np.nan, dtype=np.float32),
        "high": np.full(shape, np.nan, dtype=np.float32),
        "low": np.full(shape, np.nan, dtype=np.float32),
        "amount": np.full(shape, np.nan, dtype=np.float32),
        "total_mv": np.full(shape, np.nan, dtype=np.float32),
        "is_st": np.zeros(shape, dtype=bool),
        "suspended": np.zeros(shape, dtype=bool),
        "limit_up": np.zeros(shape, dtype=bool),
        "limit_down": np.zeros(shape, dtype=bool),
        "announce": np.zeros(shape, dtype=bool),
    }


def _positions(
    frame: pd.DataFrame,
    date_column: str,
    day_lookup: pd.Series,
    stock_index: Mapping[str, int],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """把 frame 的 (日期, 股票) 映射为面板下标，并返回过滤后的 frame。"""
    normalized = normalize_date(frame[date_column])
    frame = frame.assign(_day=normalized).dropna(subset=["_day", "ts_code"])
    day_positions = frame["_day"].map(day_lookup)
    stock_positions = frame["ts_code"].astype(str).map(stock_index)
    keep = day_positions.notna() & stock_positions.notna()
    return (
        frame[keep],
        day_positions[keep].to_numpy(dtype=np.int64),
        stock_positions[keep].to_numpy(dtype=np.int64),
    )


def _fill_daily_files(
    arrays: Dict[str, np.ndarray],
    paths: Sequence[Path],
    day_lookup: pd.Series,
    stock_index: Mapping[str, int],
) -> int:
    """按年分块读取 clean/daily 并填入面板；返回读取行数。"""
    rows = 0
    for chunk in _iter_year_chunks(paths):
        frames = [pd.read_parquet(path, columns=list(DAILY_FIELDS)) for path in chunk]
        frame = pd.concat(frames, ignore_index=True)
        rows += len(frame)
        frame, d_pos, s_pos = _positions(frame, "trade_date", day_lookup, stock_index)
        for column in ("open", "open_adj", "close_adj", "pre_close", "high", "low", "amount"):
            matrix = arrays["open_u" if column == "open" else column]
            _fill_matrix(matrix, d_pos, s_pos, frame[column].to_numpy(), matrix.dtype)
        for column in ("is_st", "is_suspended", "is_limit_up", "is_limit_down"):
            target = {
                "is_st": "is_st",
                "is_suspended": "suspended",
                "is_limit_up": "limit_up",
                "is_limit_down": "limit_down",
            }[column]
            arrays[target][d_pos, s_pos] = frame[column].to_numpy().astype(bool)
        del frame, frames
    return rows


def _fill_basic_files(
    total_mv: np.ndarray,
    paths: Sequence[Path],
    day_lookup: pd.Series,
    stock_index: Mapping[str, int],
) -> int:
    rows = 0
    for chunk in _iter_year_chunks(paths):
        frames = [
            pd.read_parquet(path, columns=["ts_code", "trade_date", "total_mv"]) for path in chunk
        ]
        frame = pd.concat(frames, ignore_index=True)
        rows += len(frame)
        frame, d_pos, s_pos = _positions(frame, "trade_date", day_lookup, stock_index)
        _fill_matrix(
            total_mv,
            d_pos,
            s_pos,
            pd.to_numeric(frame["total_mv"], errors="coerce").to_numpy(),
            np.float32,
        )
        del frame, frames
    return rows


def _fill_announcements(
    announce: np.ndarray,
    announcements: Optional[pd.DataFrame],
    day_lookup: pd.Series,
    stock_index: Mapping[str, int],
) -> None:
    if announcements is None or not len(announcements):
        return
    frame, d_pos, s_pos = _positions(
        announcements.copy(), "knowledge_date", day_lookup, stock_index
    )
    announce[d_pos, s_pos] = True
    del frame


def industry_at(panel: "Panel", day: int, stocks: np.ndarray) -> np.ndarray:
    """T 日**当时可得**的行业代码（无覆盖 → -1，未知单独登记）。

    PIT 模式（``industry_matrix`` 非空）直接按行取；快照模式沿用「``T >= in_date``」下界判定。
    """
    if panel.industry_matrix is not None:
        return np.asarray(panel.industry_matrix[int(day), stocks], dtype=np.int32)
    since = np.asarray(panel.industry_since)[stocks]
    codes = panel.industry[stocks]
    valid = (since >= 0) & (since <= int(day))
    return np.where(valid, codes, -1).astype(np.int32)


def listed_at(panel: "Panel", day: int, stocks: np.ndarray) -> np.ndarray:
    """T 日是否已上市（以首次可观测日与 `list_date` 的较早者为界）。"""
    return np.asarray(panel.listed)[int(day), stocks].astype(bool)


def share_ratio(
    panel: "Panel", entry: np.ndarray, exit_: np.ndarray, members: np.ndarray
) -> np.ndarray:
    """持有期送转股倍数 = C[exit]/C[entry]（现金分红不改股数，不在此列）。"""
    mult = np.asarray(panel.share_mult)
    safe = np.where(members >= 0, members, 0)
    return mult[exit_, safe] / mult[entry, safe]


def share_chain_known(
    panel: "Panel", entry: np.ndarray, exit_: np.ndarray, members: np.ndarray
) -> np.ndarray:
    """持有期内是否存在无法解析的送转记录（`stk_div` 缺失）。"""
    unknown = np.asarray(panel.share_unknown)
    safe = np.where(members >= 0, members, 0)
    return (unknown[exit_, safe] - unknown[entry, safe]) == 0


def _industry_vector(
    root: Path,
    stocks: Sequence[str],
    day_lookup: pd.Series,
    n_days: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """行业代码与「纳入分类日」位置；早于该日视为未知。"""
    industry_map, since_map, industry_info = load_industry_mapping(root)
    codes: Dict[str, int] = {}
    industry = np.full(len(stocks), -1, dtype=np.int32)
    since = np.full(len(stocks), -1, dtype=np.int32)
    for index, code in enumerate(stocks):
        label = industry_map.get(code)
        if label is None or label == "" or label == "nan":
            continue
        if label not in codes:
            codes[label] = len(codes)
        industry[index] = codes[label]
        raw_since = since_map.get(code)
        if raw_since is None:
            continue
        position = int(np.searchsorted(np.asarray(sorted(day_lookup.index)), str(raw_since)))
        if position >= n_days:
            continue
        since[index] = position
    return industry, since, industry_info


def _listed_matrix(
    root: Path,
    stocks: Sequence[str],
    day_lookup: pd.Series,
    close_adj: np.ndarray,
    n_days: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """上市状态矩阵：`t >= 首次可观测日` 为已上市（停牌仍算已上市）。

    首次可观测日取 `raw/stock_basic.list_date` 与「clean 首个有限收盘」两者中**较早者**，
    避免上市首日停牌或 list_date 缺失造成漏判。此处只判定「是否已上市」，不判定
    「当日能否成交」——成交资格由执行阶段冻结条件负责（T1 评审 R1）。
    """
    basic_path = root / "raw" / "stock_basic.parquet"
    list_map: Dict[str, str] = {}
    if basic_path.exists():
        basic = pd.read_parquet(basic_path)
        if "list_date" in basic.columns:
            list_map = dict(zip(basic["ts_code"].astype(str), normalize_date(basic["list_date"])))
    first_quote = np.argmax(np.isfinite(close_adj), axis=0)
    has_quote = np.isfinite(close_adj).any(axis=0)
    ordered = np.asarray(sorted(day_lookup.index))
    listed = np.zeros((n_days, len(stocks)), dtype=bool)
    known = 0
    for index, code in enumerate(stocks):
        candidates: List[int] = []
        raw = list_map.get(code)
        if raw is not None and isinstance(raw, str):
            candidates.append(int(np.searchsorted(ordered, raw)))
        if has_quote[index]:
            candidates.append(int(first_quote[index]))
        if not candidates:
            continue
        start = min(candidates)
        if start >= n_days:
            continue
        listed[start:, index] = True
        known += 1
    stats = {
        "listed_known_stocks": int(known),
        "listed_source": "raw/stock_basic.list_date ∪ clean 首个有限收盘（取较早）",
    }
    return listed, stats


def _share_matrices(
    root: Path,
    day_lookup: pd.Series,
    stock_index: Mapping[str, int],
    n_days: int,
    n_stocks: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """持有期送转股倍数矩阵（累计）与不可解析计数矩阵。

    只使用 `div_proc == 实施` 且 `ex_date` 可解析的记录；`stk_div` 缺失时记未知计数，
    执行阶段据此把该组判为收益链不可解析（T1 评审 R4）。
    """
    paths = sorted((root / "raw" / "dividend").glob("*.parquet"))
    mult = np.ones((n_days, n_stocks), dtype=np.float64)
    unknown = np.zeros((n_days, n_stocks), dtype=np.int16)
    rows = 0
    if paths:
        ordered = np.asarray(sorted(day_lookup.index))
        for chunk in _iter_year_chunks(paths):
            frames = [
                pd.read_parquet(path, columns=["ts_code", "ex_date", "stk_div", "div_proc"])
                for path in chunk
            ]
            frame = pd.concat(frames, ignore_index=True)
            rows += len(frame)
            frame = frame[frame["div_proc"].astype(str).str.strip() == "实施"]
            ex = normalize_date(frame["ex_date"])
            frame = frame.assign(_ex=ex).dropna(subset=["_ex"])
            if frame.empty:
                continue
            positions = np.searchsorted(ordered, frame["_ex"].to_numpy(), side="left")
            stock_positions = frame["ts_code"].astype(str).map(stock_index)
            keep = (positions < n_days) & stock_positions.notna()
            if not keep.any():
                continue
            d_pos = positions[keep.to_numpy()]
            s_pos = stock_positions[keep].to_numpy(dtype=np.int64)
            values = pd.to_numeric(frame.loc[keep, "stk_div"], errors="coerce").to_numpy()
            finite = np.isfinite(values)
            np.multiply.at(mult, (d_pos[finite], s_pos[finite]), 1.0 + values[finite])
            np.add.at(unknown, (d_pos[~finite], s_pos[~finite]), 1)
            del frame, frames
    share_mult = np.cumprod(mult, axis=0).astype(np.float32)
    share_unknown = np.cumsum(unknown, axis=0, dtype=np.int32)
    del mult, unknown
    stats = {
        "dividend_rows": int(rows),
        "dividend_partitions": len(paths),
        "share_mult_max": float(np.nanmax(share_mult)) if share_mult.size else 1.0,
        "unknown_share_days": int((share_unknown > 0).sum()),
    }
    return share_mult, share_unknown, stats


def load_panel(
    data_root: Path,
    days: Sequence[str],
    *,
    cutoff: str,
    announcements: Optional[pd.DataFrame] = None,
    lookback: int = 20,
    panel_start: str = "",
    ffill_limit: int = 20,
    min_obs: int = 15,
    industry_source: str = "snapshot",
) -> Panel:
    """加载截止日内的行情/市值面板与协变量、公告覆盖标记。

    ``panel_start`` 仅供冒烟降配（只读更短的行情窗口），正式运行必须为空。
    ``industry_source``：``snapshot``（默认，r2 口径，逐值不变）/ ``pit``（区间表驱动，r3）。
    """
    root = Path(data_root)
    daily_paths = _partition_dates(
        sorted((root / "clean" / "daily").glob("*.parquet")), cutoff, panel_start
    )
    basic_paths = _partition_dates(
        sorted((root / "clean" / "daily_basic").glob("*.parquet")), cutoff, panel_start
    )
    if not daily_paths:
        raise FileNotFoundError(f"未找到 clean/daily 分区：{root / 'clean' / 'daily'}")
    if not basic_paths:
        raise FileNotFoundError(f"未找到 clean/daily_basic 分区：{root / 'clean' / 'daily_basic'}")

    days = list(days)
    day_index = {day: index for index, day in enumerate(days)}
    stocks = _collect_stock_codes(daily_paths)
    stock_index = {code: index for index, code in enumerate(stocks)}
    n_days, n_stocks = len(days), len(stocks)
    day_lookup = pd.Series(range(n_days), index=days)

    arrays = _allocate_arrays(n_days, n_stocks)
    rows_daily = _fill_daily_files(arrays, daily_paths, day_lookup, stock_index)
    rows_basic = _fill_basic_files(arrays["total_mv"], basic_paths, day_lookup, stock_index)
    _fill_announcements(arrays["announce"], announcements, day_lookup, stock_index)
    if industry_source == "pit":
        industry_matrix, industry_info = _industry_matrix(root, stocks, days)
        industry = industry_matrix[-1].copy()
        industry_since = np.full(n_stocks, -1, dtype=np.int32)
    elif industry_source == "snapshot":
        industry, industry_since, industry_info = _industry_vector(root, stocks, day_lookup, n_days)
        industry_matrix = None
    else:
        raise ValueError(f"未知行业来源：{industry_source}（可选 snapshot / pit）")

    close_adj = arrays["close_adj"]
    total_mv = (
        pd.DataFrame(arrays["total_mv"]).ffill(limit=int(ffill_limit)).to_numpy(dtype=np.float32)
    )
    ret20, vol20 = compute_covariates(
        close_adj, lookback=lookback, ffill_limit=ffill_limit, min_obs=min_obs
    )
    listed, listed_stats = _listed_matrix(root, stocks, day_lookup, close_adj, n_days)
    share_mult, share_unknown, share_stats = _share_matrices(
        root, day_lookup, stock_index, n_days, n_stocks
    )
    load_stats = {
        "rows_daily": int(rows_daily),
        "rows_daily_basic": int(rows_basic),
        "daily_partitions": len(daily_paths),
        "daily_basic_partitions": len(basic_paths),
        "stocks": n_stocks,
        "days": n_days,
        "industry_known_stocks": int((industry >= 0).sum()),
        "industry_since_known_stocks": int((industry_since >= 0).sum()),
        "industry_source_mode": industry_source,
        "industry_cells": int(industry_matrix.size) if industry_matrix is not None else None,
        "industry_cells_known": (
            int((industry_matrix >= 0).sum()) if industry_matrix is not None else None
        ),
        "listed": listed_stats,
        "shares": share_stats,
        "covariate_rule": {
            "lookback_days": int(lookback),
            "ffill_limit_trading_days": int(ffill_limit),
            "vol20_min_observations": int(min_obs),
            "note": "收盘价与总市值按最多 ffill_limit 个交易日向前延续；超限不外推",
        },
    }
    return Panel(
        days=days,
        day_index=day_index,
        stocks=stocks,
        stock_index=stock_index,
        open_u=arrays["open_u"],
        open_adj=arrays["open_adj"],
        close_adj=close_adj,
        pre_close=arrays["pre_close"],
        high=arrays["high"],
        low=arrays["low"],
        amount=arrays["amount"],
        total_mv=total_mv,
        is_st=arrays["is_st"],
        suspended=arrays["suspended"],
        limit_up=arrays["limit_up"],
        limit_down=arrays["limit_down"],
        industry=industry,
        ret20=ret20,
        vol20=vol20,
        announce=arrays["announce"],
        listed=listed,
        industry_since=industry_since,
        share_mult=share_mult,
        share_unknown=share_unknown,
        industry_source=industry_info,
        load_stats=load_stats,
        industry_matrix=industry_matrix,
    )


def watermark_payload(panel: Panel, cutoff: str) -> Dict[str, Any]:
    """面板水位（评价窗起止、截断与覆盖）留档。"""
    return {
        "common_data_cutoff": cutoff,
        "trading_days": panel.n_days,
        "first_day": panel.days[0] if panel.days else None,
        "last_day": panel.days[-1] if panel.days else None,
        "stocks": panel.n_stocks,
        "industry_known_stocks": panel.load_stats["industry_known_stocks"],
        "rows_daily": panel.load_stats["rows_daily"],
        "rows_daily_basic": panel.load_stats["rows_daily_basic"],
    }


def summarize_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
