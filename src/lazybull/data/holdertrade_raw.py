# -*- coding: utf-8 -*-
"""stk_holdertrade（股东增减持）raw 层：拉取、水位、按年分区落盘。

Phase 0 审计结论见 `docs/stk_holdertrade_pit_audit.md`，实现必须遵守：

- **单页 3000 行（per-request）**：不传 `limit/offset` 时**不报错**、只返回最新 3000 行
  ⇒ 必须分页读到返回空（触顶页继续翻），否则静默丢历史；
- **分页可取全**（实测 2024H1：分页 4559 行 == 逐月拼装逐行一致；2019 全年 19571 行）；
- **跨页会重复行**（≈1.8%），源内整行重复 24.7% ⇒ **全字段去重**（没有自然唯一键）；
  保留 `(ts_code, ann_date)` 多行聚合语义；
- **无 `begin_date`/`close_date`** ⇒ PIT 锚点只能是 `ann_date`；
- 稀疏（披露季每日中位 24 行 / 12 只股票）⇒ 消费侧需状态保留 + freshness 衰减（Phase 2）。

**落盘布局**：`data/raw/stk_holdertrade/YYYY-12-31.parquet`（按 `ann_date` 年份分区，
沿 `dividend` 既有模式）。选年分区而不是整体单文件/半年/季度：
① 全年实测 1.9 万行（2019）→ 年文件 <2MB，按季/半年会让分区过碎；
② "当前年度续传 + 合并"只需操作 1 个分区，与 `dividend` / `report_rc` 的年分区语义一致；
③ 复用 `Storage.list_partitions` / `load_raw_by_date` 既有能力，避免新造一套。
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from .storage import Storage
from .tushare_client import TushareClient

#: raw 数据集名（目录与文件前缀）
HOLDERTRADE_RAW_NAME = "stk_holdertrade"

#: 单页行数上限（Phase 0 实测：接口单次最多 3000 行）
HOLDERTRADE_PAGE_LIMIT = 3000

#: 增量续传回拉天数：从已有最大 ann_date 往前多拉几天再合并，
#: 防同日补录/更正（去重按全字段，重复拉取无副作用）
HOLDERTRADE_RESUME_OVERLAP_DAYS = 3


def _to_date(value: str) -> datetime:
    """解析 YYYYMMDD 字符串（非法输入明确报错，不做静默回退）。"""
    text = str(value).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"日期格式非法（应为 YYYYMMDD）: {value!r}")
    return datetime.strptime(text, "%Y%m%d")


def _norm_date_series(series: pd.Series) -> pd.Series:
    """把日期列归一化为 YYYYMMDD 字符串（保留 NaN）。"""
    text = series.astype(str).str.replace("-", "", regex=False).str.strip()
    return text.where(series.notna(), pd.NA)


def month_windows(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """按月切分下载窗口（单月窗口实测行数远小于单页上限，天然规避截断）。"""
    start, end = _to_date(start_date), _to_date(end_date)
    if start > end:
        raise ValueError(f"开始日期晚于结束日期: {start_date} > {end_date}")
    windows: List[Tuple[str, str]] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        month_end = (cursor + pd.offsets.MonthEnd(0)).to_pydatetime()
        window_end = min(month_end, end)
        windows.append((max(cursor, start).strftime("%Y%m%d"), window_end.strftime("%Y%m%d")))
        cursor = (cursor + pd.offsets.MonthBegin(1)).to_pydatetime()
    return windows


def deduplicate_holdertrade(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """规范 ann_date、剔除非法公告日、按**全字段**去重（无自然键）。

    该数据集没有自然唯一键（`ts_code+ann_date` 重复 67%），且 24.7% 为整行完全重复，
    因此去重口径就是"全字段完全相同视为同一行"；`(ts_code, ann_date)` 的多行明细
    （同一股东分笔、多个股东）必须保留，聚合在因子层做。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    work = df.copy()
    if "ann_date" not in work.columns:
        raise ValueError("holdertrade 数据缺少 ann_date，无法去重或按年分区")
    work["ann_date"] = _norm_date_series(work["ann_date"])
    valid = work["ann_date"].str.match(r"^\d{8}$", na=False)
    valid &= pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce").notna()
    invalid_count = int((~valid).sum())
    if invalid_count > 0:
        codes = (
            work.loc[~valid, "ts_code"].astype(str).unique().tolist()
            if "ts_code" in work.columns
            else []
        )
        logger.warning(
            f"[{HOLDERTRADE_RAW_NAME}] 忽略 {invalid_count} 条 ann_date 缺失或非法的记录，"
            f"股票示例: {', '.join(codes[:10])}" + (" ..." if len(codes) > 10 else "")
        )
        work = work.loc[valid].copy()
    if len(work) == 0:
        return work.reset_index(drop=True)
    before = len(work)
    work = work.drop_duplicates(ignore_index=True)
    dropped = before - len(work)
    if dropped:
        logger.debug(f"[{HOLDERTRADE_RAW_NAME}] 全字段去重移除 {dropped} 行（含跨页重复）")
    sort_cols = [c for c in ("ann_date", "ts_code") if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return work


def existing_holdertrade_df(storage: Storage) -> Optional[pd.DataFrame]:
    """枚举年分区加载已有数据（分区数据集不能用 `load_raw` 单文件读取）。"""
    frames: List[pd.DataFrame] = []
    for partition in storage.list_partitions("raw", HOLDERTRADE_RAW_NAME):
        df = storage.load_raw_by_date(HOLDERTRADE_RAW_NAME, partition)
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        return None
    return deduplicate_holdertrade(pd.concat(frames, ignore_index=True))


def holdertrade_latest_ann_date(storage: Storage) -> Optional[str]:
    """已有数据的最大公告日（增量水位）；无数据返回 None。

    只枚举分区并读取 `ann_date` 单列（从最新分区降序找第一个有有效公告日的分区），
    供纸面 ensure 的每日判定使用——避免为了取一个日期而全量加载数据集。
    """
    partitions = sorted(storage.list_partitions("raw", HOLDERTRADE_RAW_NAME), reverse=True)
    for partition in partitions:
        df = storage.load_raw_by_date(HOLDERTRADE_RAW_NAME, partition, columns=["ann_date"])
        if df is None or len(df) == 0 or "ann_date" not in df.columns:
            continue
        ann = df["ann_date"].astype(str)
        ann = ann[ann.str.match(r"^\d{8}$", na=False)]
        if len(ann):
            return str(ann.max())
    return None


def save_holdertrade_by_year(storage: Storage, df: pd.DataFrame) -> None:
    """按 `ann_date` 年份分区落盘（全量重写各年分区，年文件 <2MB 成本可控）。"""
    if df is None or len(df) == 0:
        return
    if "ann_date" not in df.columns:
        raise ValueError("holdertrade 数据缺少 ann_date，无法按年分区落盘")
    desired: set = set()
    work = df.copy()
    work["_year"] = work["ann_date"].astype(str).str[:4]
    for year, group in work.groupby("_year", sort=True):
        if not str(year).isdigit():
            raise ValueError(f"年份非法: {year!r}（ann_date 未规范化的数据禁止落盘）")
        partition = f"{year}-12-31"
        desired.add(partition)
        storage.save_raw_by_date(
            group.drop(columns=["_year"]).reset_index(drop=True),
            HOLDERTRADE_RAW_NAME,
            partition,
        )
    removed = 0
    for partition in storage.list_partitions("raw", HOLDERTRADE_RAW_NAME):
        if partition in desired:
            continue
        path = Path(storage.raw_path) / HOLDERTRADE_RAW_NAME / f"{partition}.parquet"
        if path.exists():
            path.unlink()
            removed += 1
    logger.info(
        f"[{HOLDERTRADE_RAW_NAME}] 已按 ann_date 年分区落盘 {len(df)} 条记录"
        + (f"，移除 {removed} 个旧分区" if removed else "")
    )


def load_holdertrade(
    storage: Storage, years: Optional[Sequence[str]] = None
) -> Optional[pd.DataFrame]:
    """加载 holdertrade（默认全部年份；可指定 `years=['2019','2020']`）。"""
    if years is not None:
        frames: List[pd.DataFrame] = []
        for year in years:
            partition = f"{year}-12-31"
            df = storage.load_raw_by_date(HOLDERTRADE_RAW_NAME, partition)
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return None
        return deduplicate_holdertrade(pd.concat(frames, ignore_index=True))
    return existing_holdertrade_df(storage)


def _resolve_incremental_start(
    storage: Storage, start_date: str, end_date: str, force: bool
) -> Optional[str]:
    """计算增量起点（回拉 overlap 天）；已覆盖则返回 None 表示无需下载。"""
    if force:
        return start_date
    latest = holdertrade_latest_ann_date(storage)
    if latest is None:
        return start_date
    if latest >= end_date:
        logger.info(f"[{HOLDERTRADE_RAW_NAME}] 已有数据覆盖至 {latest}（≥ {end_date}），无需增量")
        return None
    resume = _to_date(latest) - timedelta(days=HOLDERTRADE_RESUME_OVERLAP_DAYS)
    return max(resume.strftime("%Y%m%d"), start_date)


def download_holdertrade(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> Dict[str, object]:
    """按月窗口分页下载 stk_holdertrade，合并已有数据并按年分区落盘。

    Returns:
        摘要 dict：``rows_new`` / ``rows_total`` / ``windows`` / ``latest_ann_date`` / ``resumed_from``
    """
    effective_start = _resolve_incremental_start(storage, start_date, end_date, force)
    if effective_start is None:
        existing = existing_holdertrade_df(storage)
        return {
            "rows_new": 0,
            "rows_total": 0 if existing is None else len(existing),
            "windows": 0,
            "latest_ann_date": holdertrade_latest_ann_date(storage),
            "resumed_from": None,
        }

    windows = month_windows(effective_start, end_date)
    logger.info(
        f"[{HOLDERTRADE_RAW_NAME}] 下载 {effective_start}~{end_date}，"
        f"{len(windows)} 个月窗口（单页上限 {HOLDERTRADE_PAGE_LIMIT}，分页读满）"
    )
    frames: List[pd.DataFrame] = []
    for window_start, window_end in windows:
        df = client._query_with_pagination(
            HOLDERTRADE_RAW_NAME,
            page_limit=HOLDERTRADE_PAGE_LIMIT,
            start_date=window_start,
            end_date=window_end,
        )
        if df is not None and len(df) > 0:
            frames.append(df)
    new_df = deduplicate_holdertrade(
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )

    existing = None if force else existing_holdertrade_df(storage)
    merged_frames = [f for f in (existing, new_df) if f is not None and len(f) > 0]
    if not merged_frames:
        logger.warning(f"[{HOLDERTRADE_RAW_NAME}] 未获取到任何数据（窗口数 {len(windows)}）")
        return {
            "rows_new": 0,
            "rows_total": 0,
            "windows": len(windows),
            "latest_ann_date": None,
            "resumed_from": effective_start,
        }
    merged = deduplicate_holdertrade(pd.concat(merged_frames, ignore_index=True))
    save_holdertrade_by_year(storage, merged)
    summary = {
        "rows_new": len(new_df),
        "rows_total": len(merged),
        "windows": len(windows),
        "latest_ann_date": str(merged["ann_date"].max()),
        "resumed_from": effective_start,
    }
    logger.info(
        f"[{HOLDERTRADE_RAW_NAME}] 完成: 新增 {summary['rows_new']} 行，"
        f"合计 {summary['rows_total']} 行，最新公告日 {summary['latest_ann_date']}"
    )
    return summary


def iter_year_partitions(df: pd.DataFrame) -> Iterator[Tuple[str, pd.DataFrame]]:
    """按 `ann_date` 年份切分（供测试与审计核对分区内容）。"""
    work = df.copy()
    work["_year"] = work["ann_date"].astype(str).str[:4]
    for year, group in work.groupby("_year", sort=True):
        yield str(year), group.drop(columns=["_year"]).reset_index(drop=True)
