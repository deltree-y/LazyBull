# -*- coding: utf-8 -*-
"""利润表 income_vip 下载。"""

from datetime import datetime

from loguru import logger

from src.lazybull.data import Storage, TushareClient
from src.lazybull.data.financial_statement_versions import (
    INCOME_VERSION_DEDUP_COLS,
    INCOME_VERSIONED_RAW_WATERMARK,
)
from src.lazybull.data.tushare_client import INCOME_DEFAULT_FIELDS

from .periodic import download_by_period

INCOME_PAGE_LIMIT = 5000


def download_income(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """按季度下载合并利润表，保留 f_ann_date 修订版本。"""
    existing_periods = {
        str(period).replace("-", "")[:8] for period in storage.list_partitions("raw", "income")
    }
    covers_existing_history = not existing_periods or all(
        start_date <= period <= end_date for period in existing_periods
    )
    migration_pending = storage.load_sync_watermark(INCOME_VERSIONED_RAW_WATERMARK) is None
    effective_force = force or (migration_pending and covers_existing_history)
    if effective_force and not force and existing_periods:
        logger.warning("[income] 检测到版本化 raw 尚未完成迁移，将重下请求范围内全部季度")

    completed = download_by_period(
        client,
        storage,
        dataset_name="income",
        api_name="income_vip",
        start_date=start_date,
        end_date=end_date,
        dedup_cols=list(INCOME_VERSION_DEDUP_COLS),
        fields=INCOME_DEFAULT_FIELDS,
        force=effective_force,
        page_limit=INCOME_PAGE_LIMIT,
        partition_by_period=True,
        sort_cols=["end_date", "f_ann_date"],
        query_kwargs={"report_type": "1"},
    )
    if completed and covers_existing_history and effective_force:
        storage.save_sync_watermark(
            INCOME_VERSIONED_RAW_WATERMARK,
            datetime.now().strftime("%Y%m%d"),
        )
