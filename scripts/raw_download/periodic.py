# -*- coding: utf-8 -*-
"""raw_download 子包：按季度/年月批量下载与分页查询。"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.data import Storage, TushareClient

from .core import ERROR_COLLECTOR, ProgressTracker


def _to_int_date(s: str) -> int:
    """修复 #6: 强制把 YYYYMMDD 字符串转 int 比较, 杜绝字典序陷阱。"""
    s = str(s).strip().replace("-", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"非法日期格式 (需 YYYYMMDD): {s!r}")
    return int(s)


def _generate_quarter_periods(start_date: str, end_date: str) -> List[str]:
    """生成日期范围内覆盖的所有季度末 YYYYMMDD 列表 (数值比较, 修复 #6)。"""
    quarter_ends = ["0331", "0630", "0930", "1231"]
    start_int = _to_int_date(start_date)
    end_int = _to_int_date(end_date)
    start_year = start_int // 10000
    end_year = end_int // 10000
    periods = []
    for year in range(start_year, end_year + 1):
        for qe in quarter_ends:
            p = f"{year}{qe}"
            if start_int <= int(p) <= end_int:
                periods.append(p)
    return periods


def _generate_month_periods(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """生成按月切分的 (start, end) 列表。"""
    import calendar
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    periods: List[Tuple[str, str]] = []
    current = start.replace(day=1)
    while current <= end:
        month_start = current.strftime("%Y%m%d")
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end_dt = current.replace(day=last_day)
        if month_end_dt > end:
            month_end_dt = end
        periods.append((month_start, month_end_dt.strftime("%Y%m%d")))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
    return periods


def _save_merged(
    storage: Storage,
    dataset_name: str,
    new_dfs: List["pd.DataFrame"],
    existing_df: Optional["pd.DataFrame"],
    dedup_cols: List[str],
    sort_cols: Optional[List[str]] = None,
) -> None:
    """合并新旧数据并保存。

    修复 #9: 合并前按 sort_cols 排序, dedup 的 keep="last" 才有明确语义。
    """
    new_dfs = [d for d in new_dfs if d is not None and len(d) > 0]
    if not new_dfs:
        result = existing_df if existing_df is not None else pd.DataFrame()
    else:
        result = pd.concat(new_dfs, ignore_index=True)
        if existing_df is not None and len(existing_df) > 0:
            result = pd.concat([existing_df, result], ignore_index=True)

    if dedup_cols and len(result) > 0:
        # 先按 sort_cols (如 ann_date) 升序, 然后 keep="last" 保留最新
        if sort_cols:
            cols_present = [c for c in sort_cols if c in result.columns]
            if cols_present:
                result = result.sort_values(cols_present, kind="stable")
        result = result.drop_duplicates(subset=dedup_cols, keep="last")
        result = result.reset_index(drop=True)

    storage.save_raw(result, dataset_name, is_force=True)
    logger.info(f"[{dataset_name}] 已保存: {len(result)} 条记录")


def _query_with_pagination(
    client: TushareClient,
    api_name: str,
    page_limit: int = 50000,
    fields: Optional[str] = None,
    max_pages: int = 1000,
    **kwargs,
) -> "pd.DataFrame":
    """翻页获取全量数据。

    修复: 改用 client.query 走令牌桶限频与限流重试，避免原先直接 client.pro.query
    绕过限频，在并发下触发 TuShare 限流后整段失败重下；同时移除 probe 探测
    （每整页多一次额外请求，且对不支持 offset 的接口存在死循环风险），
    恢复 len<page_limit 作为终止条件，并以 max_pages 兜底防死循环。
    """
    all_pages: List["pd.DataFrame"] = []
    offset = 0

    for _ in range(max_pages):
        df = client.query(
            api_name,
            fields=fields or "",
            limit=page_limit,
            offset=offset,
            **kwargs,
        )
        if df is None or len(df) == 0:
            break
        all_pages.append(df)
        if len(df) < page_limit:
            break
        offset += page_limit
        logger.debug(f"  [{api_name}] 分页: offset={offset}")
    else:
        logger.warning(
            f"[{api_name}] 达到最大分页数 {max_pages}，数据可能仍未取完，请检查是否支持 offset 分页"
        )

    if not all_pages:
        return pd.DataFrame()
    return pd.concat(all_pages, ignore_index=True)


def download_by_period(
    client: TushareClient,
    storage: Storage,
    dataset_name: str,
    api_name: str,
    start_date: str,
    end_date: str,
    dedup_cols: List[str],
    fields: Optional[str] = None,
    force: bool = False,
    page_limit: int = 50000,
    partition_by_period: bool = False,
    sort_cols: Optional[List[str]] = None,
) -> None:
    """按报告期(period)批量下载数据。"""
    periods = _generate_quarter_periods(start_date, end_date)
    if not periods:
        logger.warning(f"[{dataset_name}] 区间内无有效季度")
        return

    existing_df = None
    existing_periods: Set[str] = set()
    if not force:
        if partition_by_period:
            for p in periods:
                if storage.is_data_exists("raw", dataset_name, p):
                    existing_periods.add(p)
            if existing_periods:
                logger.info(f"[{dataset_name}] 已有 {len(existing_periods)} 个季度分区")
        else:
            existing_df = storage.load_raw(dataset_name)
            if existing_df is not None and len(existing_df) > 0:
                if "end_date" in existing_df.columns:
                    existing_periods = set(
                        existing_df["end_date"].astype(str)
                        .str.replace("-", "").str[:8].unique()
                    )
                logger.info(f"[{dataset_name}] 已有 {len(existing_periods)} 个季度数据")

    periods_to_download = [p for p in periods if p not in existing_periods]
    if not periods_to_download:
        logger.info(f"[{dataset_name}] 全部季度已存在, 跳过。如需重下加 --force")
        return

    logger.info(
        f"[{dataset_name}] 按季度下载 {len(periods_to_download)} 个 "
        f"({periods_to_download[0]}~{periods_to_download[-1]})"
    )

    tracker = ProgressTracker(len(periods_to_download), label=dataset_name, log_every=4)
    all_dfs: List["pd.DataFrame"] = []
    success = empty = 0

    for period in periods_to_download:
        try:
            df = _query_with_pagination(
                client, api_name, page_limit=page_limit,
                fields=fields, period=period,
            )
            if df is not None and len(df) > 0:
                if partition_by_period:
                    storage.save_raw_by_date(df, dataset_name, period)
                else:
                    all_dfs.append(df)
                success += 1
                logger.info(f"  [{dataset_name}] {period}: {len(df)} 条")
            else:
                empty += 1
        except Exception as e:
            ERROR_COLLECTOR.add(dataset_name, f"period={period}", str(e))
        tracker.tick(extra_info=f"ok={success} empty={empty}")

    if not partition_by_period and all_dfs:
        _save_merged(
            storage, dataset_name, all_dfs, existing_df,
            dedup_cols, sort_cols=sort_cols or ["ann_date", "end_date"],
        )

    logger.info(f"[{dataset_name}] 完成: 成功={success} 空={empty}")