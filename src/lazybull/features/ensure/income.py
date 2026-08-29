# -*- coding: utf-8 -*-
"""纸面交易利润表自动补齐。"""

from typing import List, Optional

import pandas as pd
from loguru import logger

from ...data import DataLoader, Storage, TushareClient
from ...data.financial_statement_versions import (
    INCOME_VERSION_DEDUP_COLS,
    fill_actual_announcement_date,
)
from ...data.tushare_client import INCOME_DEFAULT_FIELDS
from .bulk import _bulk_download_by_period, _query_with_pagination
from .concat_utils import _concat_no_warning
from .constants import _MIN_INCOME_RECORDS
from .incremental import (
    _drop_duplicates_keep_updated,
    _incremental_catchup_by_calendar_date,
    _load_all_partitions,
)

_INCOME_PAGE_LIMIT = 5000


def _normalize_income_rows(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """限定合并报表并补齐实际公告日，保证版本键完整。"""
    if df is None or len(df) == 0:
        return df
    work = df.copy()
    if "report_type" in work.columns:
        work = work[pd.to_numeric(work["report_type"], errors="coerce").eq(1)]
    if "f_ann_date" not in work.columns:
        if "ann_date" not in work.columns:
            raise ValueError("income 数据同时缺少 f_ann_date 与 ann_date")
        work["f_ann_date"] = work["ann_date"]
    return fill_actual_announcement_date(work)


def _fetch_income_day(client: TushareClient, date_value: str) -> Optional[pd.DataFrame]:
    """同时按实际公告日与公告日查询，覆盖修订和缺失 f_ann_date 的行。"""
    parts: List[pd.DataFrame] = []
    for date_col in ("f_ann_date", "ann_date"):
        day_df = _query_with_pagination(
            client,
            "income_vip",
            page_limit=_INCOME_PAGE_LIMIT,
            fields=INCOME_DEFAULT_FIELDS,
            report_type="1",
            **{date_col: date_value},
        )
        normalized = _normalize_income_rows(day_df)
        if normalized is not None and len(normalized) > 0:
            parts.append(normalized)
    if not parts:
        return None
    return _drop_duplicates_keep_updated(
        _concat_no_warning(parts),
        list(INCOME_VERSION_DEDUP_COLS),
    )


def _try_download_income(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
    existing_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """补齐合并利润表；增量水位按 f_ann_date 连续成功前缀推进。"""
    existing = existing_df if existing_df is not None else _load_all_partitions(storage, "income")
    existing = _normalize_income_rows(existing)
    if existing is not None and len(existing) >= _MIN_INCOME_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="income",
                existing_df=existing,
                trade_date=trade_date,
                date_col="f_ann_date",
                dedup_cols=list(INCOME_VERSION_DEDUP_COLS),
                fetch_by_date=lambda date_value: _fetch_income_day(client, date_value),
                partition_date_col="end_date",
                partition_mode="quarter",
            )
        except Exception as exc:
            logger.warning(f"增量下载 income 失败: {exc}")
            return existing

    count = len(existing) if existing is not None else 0
    logger.info(f"利润表数据不足 (当前 {count} 条)，启动按季度批量下载...")
    _bulk_download_by_period(
        client,
        storage,
        dataset_name="income",
        api_name="income_vip",
        dedup_cols=list(INCOME_VERSION_DEDUP_COLS),
        fields=INCOME_DEFAULT_FIELDS,
        partition_by_period=True,
        page_limit=_INCOME_PAGE_LIMIT,
        query_kwargs={"report_type": "1"},
    )
    return DataLoader(storage).load_income(start_date=trade_date, end_date=trade_date)
