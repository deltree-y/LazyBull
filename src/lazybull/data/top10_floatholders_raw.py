# -*- coding: utf-8 -*-
"""top10_floatholders（十大流通股东）raw 层：拉取、水位、按年分区落盘。

Phase 0 审计结论见 `docs/top10_floatholders_pit_audit.md`，实现必须遵守：

- **单页 6000 行（per-request）**：`limit=10000` 仍只回 6000 且**不报错**（静默截断）
  ⇒ 必须 `offset` 翻页读到空（触顶页继续翻，`max_pages` 兜底告警）；
- **分页可取全**（实测 `period=20231231` 分页读满 54,890 行 / 10 页；与单股独立拉取按键对齐后逐列 0 差异）；
- **跨页重复 0**（与前两族不同，但仍按契约执行全字段去重，防未来抖动）；
- **禁止使用 `start_date/end_date` 参数**（语义不透明，疑似按报告期过滤）⇒ 增量一律按 `period`（报告期）；
- **PIT 锚点 = `ann_date`**（缺失/非法 0%）；同期多版本极罕见（≤0.02%）⇒ **不做版本状态机**，
  消费侧按 `ann_date <= T` 过滤即可；
- 唯一键 = `(ts_code, end_date, ann_date, holder_name)`；行数 =10 占 96%、<10 占 3%、>10 占 1%
  ⇒ 分析口径为"取该版本全部行、按 `hold_amount` 降序取前 N"，**禁止**假设"恰好 10 行"；
- 覆盖接近全市场（每期 ~5,600 只）⇒ **不是稀疏事件族**，无需 0 填充设计。

**落盘布局（方案阶段已定，见审计文档 §5.1）**：`data/raw/top10_floatholders/YYYY-12-31.parquet`
（按 `end_date` 报告期年份分区）。选年分区：① 每年 4 期 × ~5.5 万行 ≈ 22 万行、年文件 3~6 MB，
按季分区会让分区过碎；② 与 `dividend` / `stk_holdertrade` / `repurchase` 年分区语义一致；
③ 复用 `Storage.list_partitions` / `load_raw_by_date` 既有能力。

**增量刷新口径**（审计 §5.1）：同一报告期在 1~2 个月内陆续披露 ⇒ 刷新必须**回拉最近 2 个已存报告期**
再叠加水位之后的新报告期；水位 = 分区内 max(`end_date`)。
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from .storage import Storage
from .tushare_client import TushareClient

#: raw 数据集名（目录与文件前缀）
TOP10FH_RAW_NAME = "top10_floatholders"

#: 单页行数上限（Phase 0 实测：limit=10000 仍回 6000，超限静默截断）
TOP10FH_PAGE_LIMIT = 6000

#: 增量刷新回拉的报告期数（同一报告期在披露窗口内会持续补充）
TOP10FH_RESUME_OVERLAP_PERIODS = 2

#: 单调可用键（实测重复 0）；行数可能 <10 或 >10，禁止按"第几行=第几名"解释
TOP10FH_KEY_COLS = ["ts_code", "end_date", "ann_date", "holder_name"]

#: 报告期候选（季末）
_QUARTER_ENDS = ((3, 31), (6, 30), (9, 30), (12, 31))


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


def quarter_ends(start_date: str, end_date: str) -> List[str]:
    """枚举区间内的报告期（季末：0331/0630/0930/1231），升序返回。

    `top10_floatholders` 只能按 `period`（报告期）批量拉取（审计 §3：`start_date/end_date`
    语义不透明、`ann_date` 过滤跨报告期），故下载窗口由报告期枚举替代日期窗口。
    """
    start, end = _to_date(start_date), _to_date(end_date)
    if start > end:
        raise ValueError(f"开始日期晚于结束日期: {start_date} > {end_date}")
    periods: List[str] = []
    for year in range(start.year, end.year + 1):
        for month, day in _QUARTER_ENDS:
            stamp = f"{year}{month:02d}{day:02d}"
            if start.strftime("%Y%m%d") <= stamp <= end.strftime("%Y%m%d"):
                periods.append(stamp)
    return periods


def deduplicate_top10fh(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """规范日期列、剔除非法公告日、按**全字段**去重。

    去重口径 = "全字段完全相同视为同一行"（实测 0 重复）；**必须保留**
    `(ts_code, end_date)` 下的多行股东明细（每股正常 ~10 行）。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    work = df.copy()
    for col in ("ann_date", "end_date"):
        if col not in work.columns:
            raise ValueError(f"top10_floatholders 数据缺少 {col}，无法去重或按年分区")
    work["ann_date"] = _norm_date_series(work["ann_date"])
    work["end_date"] = _norm_date_series(work["end_date"])
    valid = work["ann_date"].str.match(r"^\d{8}$", na=False) & work["end_date"].str.match(
        r"^\d{8}$", na=False
    )
    valid &= pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce").notna()
    valid &= pd.to_datetime(work["end_date"], format="%Y%m%d", errors="coerce").notna()
    invalid_count = int((~valid).sum())
    if invalid_count > 0:
        codes = (
            work.loc[~valid, "ts_code"].astype(str).unique().tolist()
            if "ts_code" in work.columns
            else []
        )
        logger.warning(
            f"[{TOP10FH_RAW_NAME}] 忽略 {invalid_count} 条 ann_date/end_date 缺失或非法的记录，"
            f"股票示例: {', '.join(codes[:10])}" + (" ..." if len(codes) > 10 else "")
        )
        work = work.loc[valid].copy()
    if len(work) == 0:
        return work.reset_index(drop=True)
    before = len(work)
    work = work.drop_duplicates(ignore_index=True)
    dropped = before - len(work)
    if dropped:
        logger.debug(f"[{TOP10FH_RAW_NAME}] 全字段去重移除 {dropped} 行（含跨页重复）")
    sort_cols = [c for c in ("end_date", "ts_code", "ann_date", "holder_name") if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return work


def existing_top10fh_df(storage: Storage) -> Optional[pd.DataFrame]:
    """枚举年分区加载已有数据（分区数据集不能用 `load_raw` 单文件读取）。"""
    frames: List[pd.DataFrame] = []
    for partition in storage.list_partitions("raw", TOP10FH_RAW_NAME):
        df = storage.load_raw_by_date(TOP10FH_RAW_NAME, partition)
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        return None
    return deduplicate_top10fh(pd.concat(frames, ignore_index=True))


def top10fh_latest_end_date(storage: Storage) -> Optional[str]:
    """已有数据的最大报告期（增量水位）；无数据返回 None。

    只枚举分区并读取 `end_date` 单列（从最新分区降序找第一个有有效报告期的分区），
    避免为了取一个日期而全量加载数据集。
    """
    partitions = sorted(storage.list_partitions("raw", TOP10FH_RAW_NAME), reverse=True)
    for partition in partitions:
        df = storage.load_raw_by_date(TOP10FH_RAW_NAME, partition, columns=["end_date"])
        if df is None or len(df) == 0 or "end_date" not in df.columns:
            continue
        end = df["end_date"].astype(str)
        end = end[end.str.match(r"^\d{8}$", na=False)]
        if len(end):
            return str(end.max())
    return None


def save_top10fh_by_year(storage: Storage, df: pd.DataFrame) -> None:
    """按 `end_date`（报告期）年份分区落盘（全量重写各年分区，年文件 3~6 MB 成本可控）。"""
    if df is None or len(df) == 0:
        return
    if "end_date" not in df.columns:
        raise ValueError("top10_floatholders 数据缺少 end_date，无法按年分区落盘")
    desired: set = set()
    work = df.copy()
    work["_year"] = work["end_date"].astype(str).str[:4]
    for year, group in work.groupby("_year", sort=True):
        if not str(year).isdigit():
            raise ValueError(f"年份非法: {year!r}（end_date 未规范化的数据禁止落盘）")
        partition = f"{year}-12-31"
        desired.add(partition)
        storage.save_raw_by_date(
            group.drop(columns=["_year"]).reset_index(drop=True),
            TOP10FH_RAW_NAME,
            partition,
        )
    removed = 0
    for partition in storage.list_partitions("raw", TOP10FH_RAW_NAME):
        if partition in desired:
            continue
        path = Path(storage.raw_path) / TOP10FH_RAW_NAME / f"{partition}.parquet"
        if path.exists():
            path.unlink()
            removed += 1
    logger.info(
        f"[{TOP10FH_RAW_NAME}] 已按 end_date 年分区落盘 {len(df)} 条记录"
        + (f"，移除 {removed} 个旧分区" if removed else "")
    )


def load_top10fh(storage: Storage, years: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
    """加载 top10_floatholders（默认全部年份；可指定 `years=['2020','2021']`）。"""
    if years is not None:
        frames: List[pd.DataFrame] = []
        for year in years:
            partition = f"{year}-12-31"
            df = storage.load_raw_by_date(TOP10FH_RAW_NAME, partition)
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return None
        return deduplicate_top10fh(pd.concat(frames, ignore_index=True))
    return existing_top10fh_df(storage)


def iter_year_partitions(storage: Storage) -> Iterator[str]:
    """遍历已有年分区（升序），供体检/审计工具使用。"""
    yield from sorted(storage.list_partitions("raw", TOP10FH_RAW_NAME))


def _stored_periods(storage: Storage) -> List[str]:
    """已存报告期（降序）；只读 `end_date` 单列，不加载全量数据。"""
    periods: set = set()
    for partition in storage.list_partitions("raw", TOP10FH_RAW_NAME):
        df = storage.load_raw_by_date(TOP10FH_RAW_NAME, partition, columns=["end_date"])
        if df is None or len(df) == 0 or "end_date" not in df.columns:
            continue
        end = df["end_date"].astype(str)
        periods.update(end[end.str.match(r"^\d{8}$", na=False)].unique().tolist())
    return sorted(periods, reverse=True)


def resolve_incremental_periods(
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> Tuple[List[str], Optional[str]]:
    """解析本次需要拉取的报告期（升序）与续传起点。

    规则（审计 §5.1）：`force` 时拉区间内全部报告期；否则拉
    「**最近 `TOP10FH_RESUME_OVERLAP_PERIODS` 个已存报告期**」（同一报告期在披露窗口内会持续补充数据）
    ∪「**区间内尚无数据的报告期**」（缺口补齐——本数据集按报告期批量拉取，缺一期就是永久空洞，
    不能像按日期分区那样只从水位往后续传）。

    Returns:
        (periods, resumed_from)：`resumed_from` 为回拉下界（无存量时为 None）。
    """
    all_periods = quarter_ends(start_date, end_date)
    if force or not all_periods:
        return all_periods, None
    stored = _stored_periods(storage)
    if not stored:
        return all_periods, None
    stored_set = set(stored)
    all_set = set(all_periods)
    overlap = set(stored[: max(int(TOP10FH_RESUME_OVERLAP_PERIODS), 1)]) & all_set
    periods = [p for p in all_periods if (p not in stored_set) or (p in overlap)]
    if not periods:
        return [], None
    floor = min(overlap) if overlap else None
    missing = [p for p in periods if p not in stored_set]
    logger.info(
        f"[{TOP10FH_RAW_NAME}] 增量续传: 已存 {len(stored)} 个报告期（最新 {stored[0]}），"
        f"回拉 {len(overlap)} 期 + 补齐缺口 {len(missing)} 期 ⇒ 本次拉取 {len(periods)} 期"
    )
    return periods, floor


def download_top10fh(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> Dict[str, object]:
    """按报告期分页下载 top10_floatholders，合并已有数据并按年分区落盘。

    Returns:
        摘要 dict：``rows_new`` / ``rows_total`` / ``periods`` / ``latest_end_date`` / ``resumed_from``
    """
    periods, resumed_from = resolve_incremental_periods(storage, start_date, end_date, force)
    if not periods:
        existing = existing_top10fh_df(storage)
        return {
            "rows_new": 0,
            "rows_total": 0 if existing is None else len(existing),
            "periods": 0,
            "latest_end_date": top10fh_latest_end_date(storage),
            "resumed_from": None,
        }

    logger.info(
        f"[{TOP10FH_RAW_NAME}] 下载 {len(periods)} 个报告期（{periods[0]}~{periods[-1]}），"
        f"单页上限 {TOP10FH_PAGE_LIMIT}，分页读满"
    )
    frames: List[pd.DataFrame] = []
    for period in periods:
        df = client._query_with_pagination(
            TOP10FH_RAW_NAME, page_limit=TOP10FH_PAGE_LIMIT, period=period
        )
        if df is not None and len(df) > 0:
            frames.append(df)
    new_df = deduplicate_top10fh(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())

    existing = None if force else existing_top10fh_df(storage)
    merged_frames = [f for f in (existing, new_df) if f is not None and len(f) > 0]
    if not merged_frames:
        logger.warning(f"[{TOP10FH_RAW_NAME}] 未获取到任何数据（报告期数 {len(periods)}）")
        return {
            "rows_new": 0,
            "rows_total": 0,
            "periods": len(periods),
            "latest_end_date": None,
            "resumed_from": resumed_from,
        }
    merged = deduplicate_top10fh(pd.concat(merged_frames, ignore_index=True))
    save_top10fh_by_year(storage, merged)
    summary = {
        "rows_new": len(new_df),
        "rows_total": len(merged),
        "periods": len(periods),
        "latest_end_date": str(merged["end_date"].max()),
        "resumed_from": resumed_from,
    }
    logger.info(
        f"[{TOP10FH_RAW_NAME}] 完成: 新增 {summary['rows_new']} 行，累计 {summary['rows_total']} 行，"
        f"最新报告期 {summary['latest_end_date']}"
    )
    return summary
