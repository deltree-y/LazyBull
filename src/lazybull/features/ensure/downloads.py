# -*- coding: utf-8 -*-
"""ensure 子包：因子按需下载（增量优先，数据不足则批量全量）。"""

from typing import List, Optional

import pandas as pd
from loguru import logger

from ...data import DataLoader, Storage, TushareClient
from ...data.tushare_client import FINA_INDICATOR_DEFAULT_FIELDS
from .bulk import _bulk_download_by_period, _bulk_download_stk_holdernumber
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
from .incremental import _incremental_catchup_by_calendar_date


def _try_download_fina_indicator(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载财务指标数据

    数据充足（>= 阈值）：按公告日区间补齐增量公告。
    数据不足或不存在：逐股全量下载全部历史财务指标。
    """
    existing = storage.load_raw("fina_indicator")

    if existing is not None and len(existing) >= _MIN_FINA_RECORDS:
        missing_schema_cols = [c for c in _FINA_REQUIRED_RAW_COLS if c not in existing.columns]
        if missing_schema_cols:
            logger.info(
                "财务指标数据缺少关键列，先执行历史 schema 回补: "
                + ", ".join(missing_schema_cols)
            )
            repaired = _refresh_existing_period_rows(
                client=client,
                storage=storage,
                dataset_name="fina_indicator",
                api_name="fina_indicator_vip",
                existing_df=existing,
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fields=FINA_INDICATOR_DEFAULT_FIELDS,
            )
            if repaired is not None:
                existing = repaired

        # 增量模式：数据量充足，按公告日区间补齐
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="fina_indicator",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=lambda d: client.get_fina_indicator_by_date(ann_date=d),
            )
        except Exception as e:
            logger.warning(f"增量下载 fina_indicator 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）— 按季度批量
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"财务指标数据不足 (当前 {cnt} 条, 阈值 {_MIN_FINA_RECORDS})，"
        f"启动按季度批量下载..."
    )
    _bulk_download_by_period(
        client, storage,
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
    existing = storage.load_raw("cashflow")

    if existing is not None and len(existing) >= _MIN_CASHFLOW_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="cashflow",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=lambda d: client.query("cashflow_vip", ann_date=d),
            )
        except Exception as e:
            logger.warning(f"增量下载 cashflow 失败: {e}")
            return existing

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
        f"股东人数数据不足 (当前 {cnt} 条, 阈值 {_MIN_HOLDER_RECORDS})，"
        f"启动按月批量下载..."
    )
    return _bulk_download_stk_holdernumber(
        client, storage,
        dedup_cols=["ts_code", "end_date"],
    )


def _try_download_forecast(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载业绩预告数据

    数据充足（>= 阈值）：按公告日区间补齐增量。
    数据不足或不存在：逐股全量下载。
    """
    existing = storage.load_raw("forecast")

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
            )
        except Exception as e:
            logger.warning(f"增量下载 forecast 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）— 按季度批量
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"业绩预告数据不足 (当前 {cnt} 条, 阈值 {_MIN_FORECAST_RECORDS})，"
        f"启动按季度批量下载..."
    )
    return _bulk_download_by_period(
        client, storage,
        dataset_name="forecast",
        api_name="forecast_vip",
        dedup_cols=["ts_code", "end_date", "ann_date"],
    )


def _try_download_express(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载业绩快报数据

    数据充足：按公告日区间补齐增量快报。
    数据不足：逐股全量下载。
    """
    existing = storage.load_raw("express")

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
            )
        except Exception as e:
            logger.warning(f"增量下载 express 失败: {e}")
            return existing

    # 全量下载 — 按季度批量
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"业绩快报数据不足 (当前 {cnt} 条, 阈值 {_MIN_EXPRESS_RECORDS})，"
        f"启动按季度批量下载..."
    )
    return _bulk_download_by_period(
        client, storage,
        dataset_name="express",
        api_name="express_vip",
        dedup_cols=["ts_code", "end_date", "ann_date"],
    )


def _try_download_report_rc(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载一致预期研报数据

    数据充足（>= 阈值）: 按 report_date 区间补齐增量研报。
    数据不足或不存在: 按年份批量回溯下载（report_rc 每次返回 2000 条, 需分页）。
    """
    existing = storage.load_raw("report_rc")

    if existing is not None and len(existing) >= _MIN_REPORT_RC_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="report_rc",
                existing_df=existing,
                trade_date=trade_date,
                date_col="report_date",
                dedup_cols=["ts_code", "report_date", "org_name", "quarter"],
                fetch_by_date=lambda d: client.get_report_rc(report_date=d),
            )
        except Exception as e:
            logger.warning(f"增量下载 report_rc 失败: {e}")
            return existing

    # 全量下载 — 按年分页
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"一致预期数据不足 (当前 {cnt} 条, 阈值 {_MIN_REPORT_RC_RECORDS})，"
        f"启动按年批量下载..."
    )
    import datetime as _dt
    current_year = _dt.datetime.now().year
    all_pages: List[pd.DataFrame] = []
    for year in range(current_year - 5, current_year + 1):
        try:
            df = _query_with_pagination(
                client,
                "report_rc",
                start_date=f"{year}0101",
                end_date=f"{year}1231",
            )
            if df is not None and len(df) > 0:
                all_pages.append(df)
                logger.info(f"  report_rc {year} 年: {len(df)} 条")
        except Exception as e:
            logger.warning(f"  report_rc {year} 年下载失败: {e}")
    if not all_pages:
        return existing
    merged = pd.concat(all_pages, ignore_index=True)
    result = _append_and_save_raw(
        storage, "report_rc", merged,
        dedup_cols=["ts_code", "report_date", "org_name", "quarter"],
    )
    logger.info(f"一致预期全量下载完成: 总计 {len(result)} 条")
    return result