# -*- coding: utf-8 -*-
"""ensure 子包：因子按需下载（增量优先，数据不足则批量全量）。"""

from typing import List, Optional

import pandas as pd
from loguru import logger

from ...data import DataLoader, Storage, TushareClient
from ...data.report_rc import REPORT_RC_ROW_KEY_COLUMNS, query_report_rc_adaptive
from ...data.tushare_client import FINA_INDICATOR_DEFAULT_FIELDS
from .bulk import _bulk_download_by_period, _bulk_download_stk_holdernumber, _query_with_pagination
from .constants import (
    _FINA_REQUIRED_RAW_COLS,
    _MIN_CASHFLOW_RECORDS,
    _MIN_EXPRESS_RECORDS,
    _MIN_FINA_RECORDS,
    _MIN_FORECAST_RECORDS,
    _MIN_HOLDER_RECORDS,
    _MIN_REPORT_RC_RECORDS,
)
from .historical import _refresh_existing_period_rows
from .incremental import (
    _append_and_save_partitioned,
    _incremental_catchup_by_calendar_date,
    _load_all_partitions,
)


def _try_download_fina_indicator(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载财务指标数据

    数据充足（>= 阈值）：按公告日区间补齐增量公告。
    数据不足或不存在：逐股全量下载全部历史财务指标。
    """
    existing = _load_all_partitions(storage, "fina_indicator")

    if existing is not None and len(existing) >= _MIN_FINA_RECORDS:
        missing_schema_cols = [c for c in _FINA_REQUIRED_RAW_COLS if c not in existing.columns]
        if missing_schema_cols:
            logger.info(
                "财务指标数据缺少关键列，先执行历史 schema 回补: " + ", ".join(missing_schema_cols)
            )
            repaired = _refresh_existing_period_rows(
                client=client,
                storage=storage,
                dataset_name="fina_indicator",
                api_name="fina_indicator_vip",
                existing_df=existing,
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fields=FINA_INDICATOR_DEFAULT_FIELDS,
                partition_date_col="end_date",
                partition_mode="quarter",
            )
            if repaired is not None:
                existing = repaired

        # 增量模式：数据量充足，按公告日区间补齐（路由写入对应季度分区）
        try:
            _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="fina_indicator",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=lambda d: client.get_fina_indicator_by_date(ann_date=d),
                partition_date_col="end_date",
                partition_mode="quarter",
            )
        except Exception as e:
            logger.warning(f"增量下载 fina_indicator 失败: {e}")
        # 统一返回目标交易日窗口数据（与正常加载路径形态一致）
        return DataLoader(storage).load_fina_indicator(start_date=trade_date, end_date=trade_date)

    # 全量下载（首次或数据不足）— 按季度批量
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"财务指标数据不足 (当前 {cnt} 条, 阈值 {_MIN_FINA_RECORDS})，" f"启动按季度批量下载..."
    )
    _bulk_download_by_period(
        client,
        storage,
        dataset_name="fina_indicator",
        api_name="fina_indicator_vip",
        dedup_cols=["ts_code", "end_date", "ann_date"],
        fields=FINA_INDICATOR_DEFAULT_FIELDS,
        partition_by_period=True,
    )
    return DataLoader(storage).load_fina_indicator(start_date=trade_date, end_date=trade_date)


def _try_download_cashflow(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载现金流量表数据

    数据充足（>= 阈值）：按公告日区间补齐增量。
    数据不足或不存在：按季度批量下载全量。
    """
    existing = _load_all_partitions(storage, "cashflow")

    if existing is not None and len(existing) >= _MIN_CASHFLOW_RECORDS:
        try:
            _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="cashflow",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=lambda d: client.query("cashflow_vip", ann_date=d),
                partition_date_col="end_date",
                partition_mode="quarter",
            )
        except Exception as e:
            logger.warning(f"增量下载 cashflow 失败: {e}")
        # 统一返回目标交易日窗口数据（与正常加载路径形态一致）
        return DataLoader(storage).load_cashflow(start_date=trade_date, end_date=trade_date)

    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"现金流量表数据不足 (当前 {cnt} 条, 阈值 {_MIN_CASHFLOW_RECORDS})，"
        f"启动按季度批量下载..."
    )
    _bulk_download_by_period(
        client,
        storage,
        dataset_name="cashflow",
        api_name="cashflow_vip",
        dedup_cols=["ts_code", "end_date", "ann_date"],
        fields=None,
        partition_by_period=True,
    )
    return DataLoader(storage).load_cashflow(start_date=trade_date, end_date=trade_date)


def _try_download_stk_holdernumber(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载股东人数数据

    数据充足（>= 阈值）：按公告日区间补齐增量。
    数据不足或不存在：逐股全量下载。
    """
    existing = storage.load_raw("stk_holdernumber")

    if existing is not None and len(existing) >= _MIN_HOLDER_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="stk_holdernumber",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date"],
                fetch_by_date=lambda d: client.get_stk_holdernumber(ann_date=d),
            )
        except Exception as e:
            logger.warning(f"增量下载 stk_holdernumber 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）— 按月批量
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"股东人数数据不足 (当前 {cnt} 条, 阈值 {_MIN_HOLDER_RECORDS})，" f"启动按月批量下载..."
    )
    return _bulk_download_stk_holdernumber(
        client,
        storage,
        dedup_cols=["ts_code", "end_date"],
    )


def _try_download_forecast(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载业绩预告数据（按季度 end_date 分区存储）

    数据充足（>= 阈值）：按公告日区间补齐增量, 路由写入对应季度分区。
    数据不足或不存在：按季度批量全量下载。
    """
    existing = _load_all_partitions(storage, "forecast")

    if existing is not None and len(existing) >= _MIN_FORECAST_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="forecast",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=lambda d: client.get_forecast_by_date(ann_date=d),
                partition_date_col="end_date",
                partition_mode="quarter",
            )
        except Exception as e:
            logger.warning(f"增量下载 forecast 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）— 按季度批量, 每季度独立分区
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"业绩预告数据不足 (当前 {cnt} 条, 阈值 {_MIN_FORECAST_RECORDS})，" f"启动按季度批量下载..."
    )
    _bulk_download_by_period(
        client,
        storage,
        dataset_name="forecast",
        api_name="forecast_vip",
        dedup_cols=["ts_code", "end_date", "ann_date"],
        partition_by_period=True,
    )
    return _load_all_partitions(storage, "forecast")


def _try_download_express(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载业绩快报数据（按季度 end_date 分区存储）

    数据充足（>= 阈值）：按公告日区间补齐增量快报，路由写入对应季度分区。
    数据不足或不存在：按季度批量全量下载。
    """
    existing = _load_all_partitions(storage, "express")
    # 旧单文件仍存在（含"部分分区 + 旧单文件"混合态）时先迁移合并，避免漏读旧数据；
    # 迁移不可用（空文件/缺分区列等异常旧文件）时保留已有分区数据，不遮蔽
    if storage.load_raw("express") is not None:
        migrated = storage.migrate_raw_single_file_to_partitions(
            "express",
            partition_date_col="end_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
        )
        if migrated is not None:
            existing = migrated

    if existing is not None and len(existing) >= _MIN_EXPRESS_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="express",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=lambda d: client.get_express_vip(ann_date=d),
                partition_date_col="end_date",
                partition_mode="quarter",
            )
        except Exception as e:
            logger.warning(f"增量下载 express 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）— 按季度批量，每季度独立分区
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"业绩快报数据不足 (当前 {cnt} 条, 阈值 {_MIN_EXPRESS_RECORDS})，启动按季度批量下载..."
    )
    _bulk_download_by_period(
        client,
        storage,
        dataset_name="express",
        api_name="express_vip",
        dedup_cols=["ts_code", "end_date", "ann_date"],
        partition_by_period=True,
        # 数据不足为异常态（正常全量应有数万条）：忽略已有残缺季度全量重下补齐
        force=True,
    )
    rebuilt = _load_all_partitions(storage, "express")
    rebuilt_count = len(rebuilt) if rebuilt is not None else 0
    if rebuilt_count < _MIN_EXPRESS_RECORDS:
        raise RuntimeError(
            "express 强制全量重建后数据仍不足: "
            f"当前 {rebuilt_count} 条, 最低 {_MIN_EXPRESS_RECORDS} 条"
        )
    return rebuilt


def _try_download_report_rc(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载一致预期研报数据（按年 report_date 分区存储）

    数据充足（>= 阈值）: 按 report_date 区间补齐增量研报, 路由写入对应年份分区。
    数据不足或不存在: 按年份批量回溯下载（report_rc 每次返回 2000 条, 需分页）。
    """
    existing = _load_all_partitions(storage, "report_rc")

    if existing is not None and len(existing) >= _MIN_REPORT_RC_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="report_rc",
                existing_df=existing,
                trade_date=trade_date,
                date_col="report_date",
                dedup_cols=list(REPORT_RC_ROW_KEY_COLUMNS),
                fetch_by_date=lambda d: _query_with_pagination(
                    client,
                    "report_rc",
                    page_limit=2000,
                    report_date=d,
                ),
                partition_date_col="report_date",
                partition_mode="year",
            )
        except Exception as e:
            logger.warning(f"增量下载 report_rc 失败: {e}")
            return existing

    # 全量下载 — 按年分页, 每年独立分区
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"一致预期数据不足 (当前 {cnt} 条, 阈值 {_MIN_REPORT_RC_RECORDS})，" f"启动按年批量下载..."
    )
    target_date = str(trade_date).strip().replace("-", "")[:8]
    if len(target_date) != 8 or not target_date.isdigit():
        raise ValueError(f"report_rc 回补目标日期无效: {trade_date}")
    target_year = int(target_date[:4])
    all_pages: List[pd.DataFrame] = []
    for year in range(target_year - 5, target_year + 1):
        end_date = min(f"{year}1231", target_date)
        try:
            df = query_report_rc_adaptive(
                lambda start_date, end_date: _query_with_pagination(
                    client,
                    "report_rc",
                    page_limit=2000,
                    start_date=start_date,
                    end_date=end_date,
                ),
                start_date=f"{year}0101",
                end_date=end_date,
            )
            if df is not None and len(df) > 0:
                all_pages.append(df)
                logger.info(f"  report_rc {year} 年: {len(df)} 条")
        except Exception as e:
            logger.warning(f"  report_rc {year} 年下载失败: {e}")
    if not all_pages:
        return existing
    merged = pd.concat(all_pages, ignore_index=True)
    result = _append_and_save_partitioned(
        storage,
        "report_rc",
        merged,
        dedup_cols=list(REPORT_RC_ROW_KEY_COLUMNS),
        partition_date_col="report_date",
        partition_mode="year",
    )
    logger.info(f"一致预期全量下载完成: 总计 {len(result) if result is not None else 0} 条")
    return result
