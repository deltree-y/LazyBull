# -*- coding: utf-8 -*-
"""ensure 子包：批量下载、分页查询与合并保存。"""

import time
from typing import List, Optional, Set

import pandas as pd
from loguru import logger

from ...data import Storage, TushareClient
from .incremental import _drop_duplicates_keep_updated


def _generate_quarter_periods(start_year: int, end_year: int) -> List[str]:
    """生成从 start_year 到 end_year 的所有季度末日期"""
    quarter_ends = ["0331", "0630", "0930", "1231"]
    return [f"{y}{q}" for y in range(start_year, end_year + 1) for q in quarter_ends]


def _query_with_pagination(
    client: TushareClient,
    api_name: str,
    page_limit: int = 50000,
    fields: Optional[str] = None,
    max_pages: int = 1000,
    **kwargs,
) -> pd.DataFrame:
    """带分页的 API 调用，自动检测并翻页获取全量数据。"""
    all_pages: List[pd.DataFrame] = []
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
    else:
        raise RuntimeError(f"[{api_name}] 达到最大分页数 {max_pages}，无法确认数据已完整下载")
    if not all_pages:
        return pd.DataFrame()
    return pd.concat(all_pages, ignore_index=True)


def _bulk_download_by_period(
    client: TushareClient,
    storage: Storage,
    dataset_name: str,
    api_name: str,
    dedup_cols: List[str],
    fields: Optional[str] = None,
    start_year: int = 2012,
    partition_by_period: bool = False,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    """按报告期(period)批量下载全量数据（自动分页）

    适用于 fina_indicator_vip, forecast_vip, express_vip, fund_portfolio。
    每季度1次 API 调用，替代逐股下载。
    当单季度数据超过上限时自动通过 offset 分页获取全量。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        dataset_name: 数据集名称
        api_name: TuShare API 名称
        dedup_cols: 去重列
        fields: 返回字段（部分 API 需要）
        start_year: 起始年份
        partition_by_period: 按季度独立分区落盘
        force: 忽略已有数据全量重下（数据不足残缺恢复用，默认 False 走断点续传）

    Returns:
        下载并保存后的完整 DataFrame，或 None
    """
    import datetime as _dt

    current_year = _dt.datetime.now().year
    periods = _generate_quarter_periods(start_year, current_year)

    # 断点续传：跳过已有季度；force=True 时忽略已有数据全量重下
    existing_df = None
    existing_periods: Set[str] = set()
    if not force:
        if partition_by_period:
            existing_periods = {
                partition.replace("-", "")
                for partition in storage.list_partitions("raw", dataset_name)
            }
        else:
            existing_df = storage.load_raw(dataset_name)
            if existing_df is not None and len(existing_df) > 0:
                if "end_date" in existing_df.columns:
                    existing_periods = set(
                        existing_df["end_date"].astype(str).str.replace("-", "").str[:8].unique()
                    )

    periods_to_download = [p for p in periods if p not in existing_periods]
    if not periods_to_download:
        return existing_df

    logger.info(f"[{dataset_name}] 按季度批量下载: {len(periods_to_download)} 个季度")

    all_dfs: List[pd.DataFrame] = []
    success = empty = errors = 0
    failed_periods: List[str] = []
    t0 = time.time()

    for period in periods_to_download:
        try:
            df = _query_with_pagination(
                client,
                api_name,
                fields=fields,
                period=period,
            )
            if df is not None and len(df) > 0:
                if partition_by_period:
                    storage.save_raw_by_date(
                        _drop_duplicates_keep_updated(df, dedup_cols),
                        dataset_name,
                        period,
                    )
                else:
                    all_dfs.append(df)
                success += 1
            else:
                empty += 1
        except Exception as exc:
            errors += 1
            failed_periods.append(period)
            log = logger.warning if force else logger.debug
            log(f"[{dataset_name}] {period} 失败: {exc}")

    if not partition_by_period and all_dfs:
        existing_df = _save_merged_bulk(storage, dataset_name, all_dfs, existing_df, dedup_cols)

    elapsed_total = time.time() - t0
    logger.info(
        f"[{dataset_name}] 全量下载完成: 成功={success} 空={empty} 失败={errors} "
        f"耗时={elapsed_total:.0f}秒"
    )
    if force and failed_periods:
        failed_preview = ", ".join(failed_periods[:5])
        if len(failed_periods) > 5:
            failed_preview += f" 等 {len(failed_periods)} 个季度"
        raise RuntimeError(
            f"[{dataset_name}] 强制全量下载失败: {errors}/{len(periods_to_download)} "
            f"个季度异常 ({failed_preview})"
        )
    return existing_df


def _bulk_download_stk_holdernumber(
    client: TushareClient,
    storage: Storage,
    dedup_cols: Optional[List[str]] = None,
    start_year: int = 2012,
) -> Optional[pd.DataFrame]:
    """按月批量下载股东人数全量数据（单次限3000条）

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        dedup_cols: 去重列
        start_year: 起始年份

    Returns:
        下载并保存后的完整 DataFrame，或 None
    """
    import calendar
    import datetime as _dt

    if dedup_cols is None:
        dedup_cols = ["ts_code", "end_date"]

    current = _dt.datetime.now()
    # 生成月范围
    month_ranges = []
    dt = _dt.datetime(start_year, 1, 1)
    while dt <= current:
        m_start = dt.strftime("%Y%m%d")
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        m_end_dt = dt.replace(day=last_day)
        if m_end_dt > current:
            m_end_dt = current
        m_end = m_end_dt.strftime("%Y%m%d")
        month_ranges.append((m_start, m_end))
        if dt.month == 12:
            dt = dt.replace(year=dt.year + 1, month=1, day=1)
        else:
            dt = dt.replace(month=dt.month + 1, day=1)

    existing_df = storage.load_raw("stk_holdernumber")
    logger.info(f"[stk_holdernumber] 按月批量下载: {len(month_ranges)} 个月")

    all_dfs: List[pd.DataFrame] = []
    success = empty = errors = 0
    t0 = time.time()

    for i, (m_start, m_end) in enumerate(month_ranges, 1):
        try:
            df = client.get_stk_holdernumber(start_date=m_start, end_date=m_end)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success += 1
            else:
                empty += 1
        except Exception as e:
            errors += 1
            logger.debug(f"[stk_holdernumber] {m_start}~{m_end} 失败: {e}")

        if i % 24 == 0 or i == len(month_ranges):
            logger.info(
                f"[stk_holdernumber] [{i}/{len(month_ranges)}] "
                f"成功={success} 空={empty} 失败={errors}"
            )

    if all_dfs:
        existing_df = _save_merged_bulk(
            storage, "stk_holdernumber", all_dfs, existing_df, dedup_cols
        )

    elapsed_total = time.time() - t0
    logger.info(
        f"[stk_holdernumber] 全量下载完成: 成功={success} 空={empty} 失败={errors} "
        f"耗时={elapsed_total:.0f}秒"
    )
    return existing_df


def _save_merged_bulk(
    storage: Storage,
    dataset_name: str,
    new_dfs: List[pd.DataFrame],
    existing_df: Optional[pd.DataFrame],
    dedup_cols: Optional[List[str]],
) -> pd.DataFrame:
    """合并新旧数据并保存（批量下载中间/最终保存用）"""
    result = pd.concat(new_dfs, ignore_index=True)
    if existing_df is not None and len(existing_df) > 0:
        result = pd.concat([existing_df, result], ignore_index=True)
    if dedup_cols:
        result = _drop_duplicates_keep_updated(result, dedup_cols)
    storage.save_raw(result, dataset_name, is_force=True)
    logger.info(f"[{dataset_name}] 已保存: {len(result)} 条记录")
    return result
