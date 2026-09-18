# -*- coding: utf-8 -*-
"""repurchase（股票回购）raw 层：拉取、水位、按年分区落盘。

Phase 0 审计结论见 `docs/repurchase_pit_audit.md`，实现必须遵守：

- **单页 2000 行（per-request）**：不传 `limit/offset` 时**不报错**、只返回最新 2000 行
  （实测 2022 全年真实 6,034 行，不翻页丢 67%）⇒ 必须分页读到返回空（触顶页继续翻）；
- **分页可取全**（实测：分页合计 6,034 行 == 逐月独立拉取合计，按全字段排序逐行 0 差异）；
- **跨页重复 0.31%**、源内整行重复 0.30% ⇒ **全字段去重**（无自然唯一键：
  `ts_code+ann_date` 重复 2.70%、`+proc` 2.40%）；
- **无 `begin_date`**，`end_date` 是进度报告期 ⇒ PIT 锚点只能是 `ann_date`（缺失 0%、100% 合法）；
- `proc` 为**多阶段状态机**（预案/股东大会通过/实施/完成/停止）⇒ 进度类因子需计划级版本化（Phase 2 再定）；
- 稀疏（每日行数中位 16，覆盖 41% 交易日）⇒ 消费侧需窗口聚合 + 窗口外显式填 0（Phase 2）。

**落盘布局（方案阶段已定，见审计文档 §5.1）**：`data/raw/repurchase/YYYY-12-31.parquet`
（按 `ann_date` 年份分区，沿 `dividend` / `stk_holdertrade` 既有模式）。选年分区而不是整体单文件/半年/季度：
① 全年实测 6k~15k 行 → 年文件 <2MB，按季/半年会让分区过碎；
② "当前年度续传 + 合并"只需操作 1 个分区，与 `dividend` / `stk_holdertrade` 年分区语义一致；
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
REPURCHASE_RAW_NAME = "repurchase"

#: 单页行数上限（Phase 0 实测：接口单次最多 2000 行，超限不报错）
REPURCHASE_PAGE_LIMIT = 2000

#: 增量续传回拉天数：从已有最大 ann_date 往前多拉几天再合并，
#: 防同日补录/更正（去重按全字段，重复拉取无副作用）
REPURCHASE_RESUME_OVERLAP_DAYS = 3


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


def deduplicate_repurchase(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """规范 ann_date、剔除非法公告日、按**全字段**去重（无自然键）。

    该数据集没有自然唯一键（`ts_code+ann_date` 重复 2.70%），去重口径 = "全字段完全相同视为同一行"；
    `(ts_code, ann_date, proc)` 的多行明细（不同阶段/多次进度披露）必须保留，聚合在因子层做。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    work = df.copy()
    if "ann_date" not in work.columns:
        raise ValueError("repurchase 数据缺少 ann_date，无法去重或按年分区")
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
            f"[{REPURCHASE_RAW_NAME}] 忽略 {invalid_count} 条 ann_date 缺失或非法的记录，"
            f"股票示例: {', '.join(codes[:10])}" + (" ..." if len(codes) > 10 else "")
        )
        work = work.loc[valid].copy()
    if len(work) == 0:
        return work.reset_index(drop=True)
    before = len(work)
    work = work.drop_duplicates(ignore_index=True)
    dropped = before - len(work)
    if dropped:
        logger.debug(f"[{REPURCHASE_RAW_NAME}] 全字段去重移除 {dropped} 行（含跨页重复）")
    sort_cols = [c for c in ("ann_date", "ts_code", "proc") if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return work


def existing_repurchase_df(storage: Storage) -> Optional[pd.DataFrame]:
    """枚举年分区加载已有数据（分区数据集不能用 `load_raw` 单文件读取）。"""
    frames: List[pd.DataFrame] = []
    for partition in storage.list_partitions("raw", REPURCHASE_RAW_NAME):
        df = storage.load_raw_by_date(REPURCHASE_RAW_NAME, partition)
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        return None
    return deduplicate_repurchase(pd.concat(frames, ignore_index=True))


def repurchase_latest_ann_date(storage: Storage) -> Optional[str]:
    """已有数据的最大公告日（增量水位）；无数据返回 None。

    只枚举分区并读取 `ann_date` 单列（从最新分区降序找第一个有有效公告日的分区），
    供纸面 ensure 的每日判定使用——避免为了取一个日期而全量加载数据集。
    """
    partitions = sorted(storage.list_partitions("raw", REPURCHASE_RAW_NAME), reverse=True)
    for partition in partitions:
        df = storage.load_raw_by_date(REPURCHASE_RAW_NAME, partition, columns=["ann_date"])
        if df is None or len(df) == 0 or "ann_date" not in df.columns:
            continue
        ann = df["ann_date"].astype(str)
        ann = ann[ann.str.match(r"^\d{8}$", na=False)]
        if len(ann):
            return str(ann.max())
    return None


def save_repurchase_by_year(storage: Storage, df: pd.DataFrame) -> None:
    """按 `ann_date` 年份分区落盘（全量重写各年分区，年文件 <2MB 成本可控）。"""
    if df is None or len(df) == 0:
        return
    if "ann_date" not in df.columns:
        raise ValueError("repurchase 数据缺少 ann_date，无法按年分区落盘")
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
            REPURCHASE_RAW_NAME,
            partition,
        )
    removed = 0
    for partition in storage.list_partitions("raw", REPURCHASE_RAW_NAME):
        if partition in desired:
            continue
        path = Path(storage.raw_path) / REPURCHASE_RAW_NAME / f"{partition}.parquet"
        if path.exists():
            path.unlink()
            removed += 1
    logger.info(
        f"[{REPURCHASE_RAW_NAME}] 已按 ann_date 年分区落盘 {len(df)} 条记录"
        + (f"，移除 {removed} 个旧分区" if removed else "")
    )


def load_repurchase(
    storage: Storage, years: Optional[Sequence[str]] = None
) -> Optional[pd.DataFrame]:
    """加载 repurchase（默认全部年份；可指定 `years=['2020','2021']`）。"""
    if years is not None:
        frames: List[pd.DataFrame] = []
        for year in years:
            partition = f"{year}-12-31"
            df = storage.load_raw_by_date(REPURCHASE_RAW_NAME, partition)
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return None
        return deduplicate_repurchase(pd.concat(frames, ignore_index=True))
    return existing_repurchase_df(storage)


def _resolve_incremental_start(
    storage: Storage, start_date: str, end_date: str, force: bool
) -> Optional[str]:
    """计算增量起点（回拉 overlap 天）；已覆盖则返回 None 表示无需下载。"""
    if force:
        return start_date
    latest = repurchase_latest_ann_date(storage)
    if latest is None:
        return start_date
    if latest >= end_date:
        logger.info(f"[{REPURCHASE_RAW_NAME}] 已有数据覆盖至 {latest}（≥ {end_date}），无需增量")
        return None
    resume = _to_date(latest) - timedelta(days=REPURCHASE_RESUME_OVERLAP_DAYS)
    return max(resume.strftime("%Y%m%d"), start_date)


def download_repurchase(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> Dict[str, object]:
    """按月窗口分页下载 repurchase，合并已有数据并按年分区落盘。

    Returns:
        摘要 dict：``rows_new`` / ``rows_total`` / ``windows`` / ``latest_ann_date`` / ``resumed_from``
    """
    effective_start = _resolve_incremental_start(storage, start_date, end_date, force)
    if effective_start is None:
        existing = existing_repurchase_df(storage)
        return {
            "rows_new": 0,
            "rows_total": 0 if existing is None else len(existing),
            "windows": 0,
            "latest_ann_date": repurchase_latest_ann_date(storage),
            "resumed_from": None,
        }

    windows = month_windows(effective_start, end_date)
    logger.info(
        f"[{REPURCHASE_RAW_NAME}] 下载 {effective_start}~{end_date}，"
        f"{len(windows)} 个月窗口（单页上限 {REPURCHASE_PAGE_LIMIT}，分页读满）"
    )
    frames: List[pd.DataFrame] = []
    for window_start, window_end in windows:
        df = client._query_with_pagination(
            REPURCHASE_RAW_NAME,
            page_limit=REPURCHASE_PAGE_LIMIT,
            start_date=window_start,
            end_date=window_end,
        )
        if df is not None and len(df) > 0:
            frames.append(df)
    new_df = deduplicate_repurchase(
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )

    existing = None if force else existing_repurchase_df(storage)
    merged_frames = [f for f in (existing, new_df) if f is not None and len(f) > 0]
    if not merged_frames:
        logger.warning(f"[{REPURCHASE_RAW_NAME}] 未获取到任何数据（窗口数 {len(windows)}）")
        return {
            "rows_new": 0,
            "rows_total": 0,
            "windows": len(windows),
            "latest_ann_date": None,
            "resumed_from": effective_start,
        }
    merged = deduplicate_repurchase(pd.concat(merged_frames, ignore_index=True))
    save_repurchase_by_year(storage, merged)
    summary = {
        "rows_new": len(new_df),
        "rows_total": len(merged),
        "windows": len(windows),
        "latest_ann_date": str(merged["ann_date"].max()),
        "resumed_from": effective_start,
    }
    logger.info(
        f"[{REPURCHASE_RAW_NAME}] 完成: 新增 {summary['rows_new']} 行，"
        f"合计 {summary['rows_total']} 行，最新公告日 {summary['latest_ann_date']}"
    )
    return summary


def iter_year_partitions(df: pd.DataFrame) -> Iterator[Tuple[str, pd.DataFrame]]:
    """按 `ann_date` 年份切分（供测试与审计核对分区内容）。"""
    work = df.copy()
    work["_year"] = work["ann_date"].astype(str).str[:4]
    for year, group in work.groupby("_year", sort=True):
        yield str(year), group.drop(columns=["_year"]).reset_index(drop=True)
