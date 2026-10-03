# -*- coding: utf-8 -*-
"""v2 normalized 层（``data/normalized/``）只读转换器：raw → normalized。

口径：normalized = clean 的 v2 命名（数值路径零重设计）——逐日逻辑与
`src/lazybull/data/build_clean.py::build_clean_data` 的逐日段**完全同序**，且全部复用
`DataCleaner` 同一实现（clean_daily → clean_suspend/limit/stock_st →
add_tradable_universe_flag），禁止重算涨跌停/停牌/ST 标记（硬契约：标记只能在
cleaner 层生成，本层与旧 clean 层一样只做复用）。

与旧 clean 层的差异（仅存储形态，不涉及数值）：
- 落盘命名 ``data/normalized/daily/YYYYMMDD.parquet``（无横线，契约命名；
  旧 clean 是 ``YYYY-MM-DD.parquet``）；
- normalized 是**缓存**：允许覆盖重写（两阶段 tmp+replace 原子写），不做封存语义；
- 对账出口 `reconcile_daily_vs_clean`：normalized vs 旧 clean 逐日逐值比对。

依赖注入：loader / cleaner 由调用方构造（旧 DataLoader / DataCleaner），本模块不建全局单例。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from loguru import logger

__all__ = [
    "convert_calendar",
    "convert_stock_basic",
    "convert_daily",
    "reconcile_daily_vs_clean",
]

#: 数值比对容差（与验收闸门母截面块级口径一致）
RECONCILE_ATOL = 1e-6

_KEY = "ts_code"


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """原子写 parquet（tmp + replace；normalized 是缓存，允许覆盖）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    return path


def _read_raw_single(raw_root: Path, name: str) -> pd.DataFrame:
    """读 raw 单文件数据集（trade_cal / stock_basic），缺失显式报错。"""
    path = Path(raw_root) / f"{name}.parquet"
    if not path.exists():
        raise ValueError(f"缺少 raw 单文件数据集: {path}")
    return pd.read_parquet(path)


def convert_calendar(raw_root: Path | str, cleaner: Any, out_root: Path | str) -> Path:
    """转换交易日历（raw/trade_cal.parquet → normalized/trade_cal.parquet）。

    复用 `DataCleaner.clean_trade_cal` 同一实现；cleaner 由调用方注入。
    """
    raw = _read_raw_single(Path(raw_root), "trade_cal")
    cleaned = cleaner.clean_trade_cal(raw)
    out = _atomic_write_parquet(cleaned, Path(out_root) / "trade_cal.parquet")
    logger.info(f"normalized trade_cal 落盘: {out}（{len(cleaned)} 行）")
    return out


def convert_stock_basic(raw_root: Path | str, cleaner: Any, out_root: Path | str) -> Path:
    """转换股票基本信息（raw/stock_basic.parquet → normalized/stock_basic.parquet）。"""
    raw = _read_raw_single(Path(raw_root), "stock_basic")
    cleaned = cleaner.clean_stock_basic(raw)
    out = _atomic_write_parquet(cleaned, Path(out_root) / "stock_basic.parquet")
    logger.info(f"normalized stock_basic 落盘: {out}（{len(cleaned)} 行）")
    return out


def _load_stock_basic_clean(loader: Any, cleaner: Any) -> pd.DataFrame:
    """stock_basic 清洗（与 build_clean.py:51-58 同序：raw → clean_stock_basic）。"""
    raw = loader.storage.load_raw("stock_basic")
    if raw is None:
        raise ValueError("缺少 raw 层 stock_basic 数据（convert_daily 的可交易标记依赖它）")
    return cleaner.clean_stock_basic(raw)


def convert_daily(
    date: str,
    loader: Any,
    cleaner: Any,
    out_root: Path | str,
    min_list_days: int = 365,
) -> Path:
    """转换单日行情（raw 当日五表 → normalized/daily/YYYYMMDD.parquet）。

    与 `build_clean_data` 逐日段完全同序：clean_daily →（有则）clean_suspend /
    clean_limit / clean_stock_st → add_tradable_universe_flag → 落盘。

    Args:
        date: 交易日（YYYYMMDD）
        loader: 旧 DataLoader（经 ``loader.storage`` 读 raw 分区；调用方构造）
        cleaner: 旧 DataCleaner（调用方构造）
        out_root: normalized 根目录（如 ``data/normalized``）
        min_list_days: 最小上市天数（冻结口径 365）

    Returns:
        落盘文件路径。

    Raises:
        ValueError: 当日 daily / adj_factor / stock_basic raw 缺失（强制依赖）。
    """
    storage = loader.storage
    daily_raw = storage.load_raw_by_date("daily", date)
    if daily_raw is None or len(daily_raw) == 0:
        raise ValueError(f"未找到 raw 层 daily 数据: {date}")
    adj_factor_raw = storage.load_raw_by_date("adj_factor", date)
    if adj_factor_raw is None or len(adj_factor_raw) == 0:
        raise ValueError(f"未找到 raw 层复权因子: {date}（拒绝构建 normalized daily）")

    daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)

    suspend_clean = _clean_optional_daily(storage, cleaner, "suspend", date)
    limit_clean = _clean_optional_daily(storage, cleaner, "stk_limit", date)
    stock_st_clean = _clean_optional_daily(storage, cleaner, "stock_st", date)

    stock_basic_clean = _load_stock_basic_clean(loader, cleaner)
    daily_clean = cleaner.add_tradable_universe_flag(
        daily_clean,
        stock_basic_clean,
        stock_st_df=stock_st_clean,
        suspend_info_df=suspend_clean,
        limit_info_df=limit_clean,
        min_list_days=min_list_days,
    )
    out = _atomic_write_parquet(daily_clean, Path(out_root) / "daily" / f"{date}.parquet")
    logger.debug(f"normalized daily 落盘: {out}（{len(daily_clean)} 行）")
    return out


def _clean_optional_daily(
    storage: Any, cleaner: Any, name: str, date: str
) -> Optional[pd.DataFrame]:
    """清洗当日报表（缺失返回 None，与 build_clean.py 的可选语义一致）。"""
    raw = storage.load_raw_by_date(name, date)
    if raw is None or len(raw) == 0:
        return None
    clean_fn = {
        "suspend": cleaner.clean_suspend_info,
        "stk_limit": cleaner.clean_limit_info,
        "stock_st": cleaner.clean_stock_st,
    }[name]
    return clean_fn(raw)


def _compare_one_date(date: str, data_root: Path) -> dict[str, Any]:
    """单日 normalized vs clean 逐值比对（行集 + 列交集 + 数值 atol）。"""
    norm_path = data_root / "normalized" / "daily" / f"{date}.parquet"
    clean_path = data_root / "clean" / "daily" / f"{date[:4]}-{date[4:6]}-{date[6:8]}.parquet"
    report: dict[str, Any] = {
        "status": "pass",
        "note": None,
        "rows": 0,
        "max_abs_diff": 0.0,
        "rows_over_tolerance": 0,
        "mismatched_columns": [],
        "columns_only_in_normalized": [],
        "columns_only_in_clean": [],
    }
    if not norm_path.exists() or not clean_path.exists():
        report["status"] = "fail"
        report["note"] = f"分区缺失（normalized={norm_path.exists()}, clean={clean_path.exists()}）"
        return report
    norm_df = pd.read_parquet(norm_path)
    clean_df = pd.read_parquet(clean_path)
    return _compare_frames(norm_df, clean_df, report)


def _compare_frames(
    norm_df: pd.DataFrame, clean_df: pd.DataFrame, report: dict[str, Any]
) -> dict[str, Any]:
    """帧级比对实现（供 _compare_one_date 与测试复用）。"""
    report["rows"] = len(norm_df)
    norm_codes = set(norm_df[_KEY].astype(str))
    clean_codes = set(clean_df[_KEY].astype(str))
    if norm_codes != clean_codes or len(norm_df) != len(clean_df):
        report["status"] = "fail"
        report["note"] = (
            f"行集不一致: normalized {len(norm_df)} 行 / clean {len(clean_df)} 行，"
            f"ts_code 差集 ±{len(norm_codes ^ clean_codes)}"
        )
        return report
    report["columns_only_in_normalized"] = sorted(set(norm_df.columns) - set(clean_df.columns))
    report["columns_only_in_clean"] = sorted(set(clean_df.columns) - set(norm_df.columns))
    common = sorted(set(norm_df.columns) & set(clean_df.columns) - {_KEY})
    left = norm_df.set_index(_KEY).sort_index()
    right = clean_df.set_index(_KEY).sort_index()
    mismatched: list[str] = []
    for col in common:
        diff_info = _compare_column(left[col], right[col])
        if diff_info is not None:
            mismatched.append(col)
            report["max_abs_diff"] = max(report["max_abs_diff"], diff_info[0])
            report["rows_over_tolerance"] += diff_info[1]
    report["mismatched_columns"] = mismatched
    if mismatched:
        report["status"] = "fail"
    return report


def _compare_column(left: pd.Series, right: pd.Series) -> Optional[tuple[float, int]]:
    """单列比对：数值列 atol 口径，非数值列精确相等；不一致返回 (max_abs_diff, 超容差行数)。"""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        lval = pd.to_numeric(left, errors="coerce")
        rval = pd.to_numeric(right, errors="coerce")
        both_nan = lval.isna() & rval.isna()
        diff = (lval - rval).abs()
        over = (diff > RECONCILE_ATOL) & ~both_nan
        nan_mismatch = lval.isna() != rval.isna()
        bad = over | nan_mismatch
        if not bool(bad.any()):
            return None
        return float(diff[bad].max()), int(bad.sum())
    if bool((left.astype(str) != right.astype(str)).any()):
        return 0.0, int((left.astype(str) != right.astype(str)).sum())
    return None


def reconcile_daily_vs_clean(dates: Iterable[str], data_root: Path | str) -> dict[str, Any]:
    """normalized vs 旧 clean/daily 逐日逐值对账（同列交集、atol=1e-6、行集一致）。

    Args:
        dates: 交易日列表（YYYYMMDD）
        data_root: 数据根（含 normalized/ 与 clean/ 两个子目录）

    Returns:
        ``{"dates": {date: 单日报告}, "summary": {total/passed/failed/worst_max_abs_diff}}``。
    """
    root = Path(data_root)
    per_date: dict[str, Any] = {}
    worst = 0.0
    failed = 0
    for date in dates:
        report = _compare_one_date(str(date), root)
        per_date[str(date)] = report
        worst = max(worst, report["max_abs_diff"])
        if report["status"] != "pass":
            failed += 1
    summary = {
        "total": len(per_date),
        "passed": len(per_date) - failed,
        "failed": failed,
        "worst_max_abs_diff": worst,
    }
    logger.info(f"normalized vs clean 对账: {summary}")
    return {"dates": per_date, "summary": summary}
