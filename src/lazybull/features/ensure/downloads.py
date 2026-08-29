# -*- coding: utf-8 -*-
"""ensure 子包：因子按需下载（增量优先，数据不足则批量全量）。"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from ...common.config import get_tushare_settings
from ...data import DataLoader, Storage, TushareClient
from ...data.financial_statement_versions import (
    CASHFLOW_VERSION_DEDUP_COLS,
    CASHFLOW_VERSIONED_RAW_WATERMARK,
)
from ...data.report_rc import REPORT_RC_ROW_KEY_COLUMNS, query_report_rc_adaptive
from ...data.tushare_client import FINA_INDICATOR_DEFAULT_FIELDS
from .bulk import _bulk_download_by_period, _bulk_download_stk_holdernumber, _query_with_pagination
from .concat_utils import _concat_no_warning
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
    _drop_duplicates_keep_updated,
    _incremental_catchup_by_calendar_date,
    _load_all_partitions,
    _normalize_date_str,
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


# cashflow 版本化去重键：含 f_ann_date（实际公告日），修订版本按版本保留，
# 避免弱键 (ts_code, end_date, ann_date) 把未来修订回填到旧公告日（前视污染）。
_CASHFLOW_VERSION_DEDUP_COLS = list(CASHFLOW_VERSION_DEDUP_COLS)

# 晚到修订刷新：ann_date 增量只会前向查询，ann_date 不变的晚到修订
# （f_ann_date > ann_date）永远收不到；按报告期重查最近 N 个季度补齐。
_CASHFLOW_REVISION_REFRESH_PERIODS = 8
_CASHFLOW_REVISION_REFRESH_WATERMARK = "cashflow_revision_refresh"
_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK = CASHFLOW_VERSIONED_RAW_WATERMARK
_CASHFLOW_REVISION_FULL_REFRESH_INTERVAL_DAYS = 90


def _recent_quarter_periods(trade_date: str, count: int) -> List[str]:
    """返回截至 trade_date 最近 count 个季度末日期（YYYYMMDD）。"""
    dt = pd.to_datetime(trade_date, format="%Y%m%d")
    year = dt.year
    quarter_ends = ["0331", "0630", "0930", "1231"]
    # 最近一个已结束季度（1~3 月属于上一年 Q4）
    q_index = ((dt.month - 1) // 3 - 1) % 4
    if dt.month <= 3:
        year -= 1
    periods: List[str] = []
    for _ in range(count):
        periods.append(f"{year}{quarter_ends[q_index]}")
        q_index -= 1
        if q_index < 0:
            q_index = 3
            year -= 1
    return periods


def _refresh_one_cashflow_period(
    client: TushareClient,
    storage: Storage,
    period: str,
) -> Tuple[str, bool, int]:
    """刷新单个报告期的现金流修订版本，返回 (period, 是否成功, 合并后行数)。

    各报告期读写各自独立的分区文件，可安全并发（客户端令牌桶限频线程安全，
    存储按目标文件派生临时名原子替换）；失败仅影响该期，由调用方汇总后
    决定水位是否推进（部分失败不推进）。
    """
    try:
        existing = storage.load_raw_by_date("cashflow", period)
        df = _query_with_pagination(client, "cashflow_vip", page_limit=6400, period=period)
        if df is None or len(df) == 0:
            # 已存在的历史分区不应被空响应视为成功，否则迁移水位会永久越过缺口。
            success = existing is None or len(existing) == 0
            if not success:
                logger.warning(f"[cashflow] 修订刷新 {period} 返回空数据，保留原分区并重试")
            return period, success, 0
        if existing is not None and len(existing) > 0:
            merged = _concat_no_warning([existing, df])
        else:
            merged = df
        merged = _drop_duplicates_keep_updated(merged, _CASHFLOW_VERSION_DEDUP_COLS)
        storage.save_raw_by_date(merged, "cashflow", period)
        return period, True, len(merged)
    except Exception as e:
        logger.warning(f"[cashflow] 修订刷新 {period} 失败: {e}")
        return period, False, 0


def _refresh_cashflow_revisions_if_due(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> None:
    """刷新现金流版本：近期按日，全历史首次及每 90 天复查。

    以版本化键合并（同报告期多版本共存），PIT 因子层按 f_ann_date 选择当日
    可见版本。首次升级会重查全部现有分区，修复旧弱键已丢版本的 raw；之后
    近 8 季度每日刷新，全历史每 90 天复查。各水位仅在对应范围全部成功后推进。
    """
    target = _normalize_date_str(trade_date)
    if target is None:
        return

    daily_watermark = storage.load_sync_watermark(_CASHFLOW_REVISION_REFRESH_WATERMARK)
    daily_due = daily_watermark is None or daily_watermark < target

    full_watermark = storage.load_sync_watermark(_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK)
    try:
        full_due = (
            full_watermark is None
            or (
                pd.to_datetime(target, format="%Y%m%d")
                - pd.to_datetime(full_watermark, format="%Y%m%d")
            ).days
            >= _CASHFLOW_REVISION_FULL_REFRESH_INTERVAL_DAYS
        )
    except (TypeError, ValueError):
        logger.warning(f"[cashflow] 全历史刷新水位无效，将重新刷新: {full_watermark!r}")
        full_due = True

    if not daily_due and not full_due:
        return

    recent_periods = (
        _recent_quarter_periods(target, _CASHFLOW_REVISION_REFRESH_PERIODS) if daily_due else []
    )
    historical_periods: List[str] = []
    if full_due:
        try:
            raw_partitions = storage.list_partitions("raw", "cashflow")
            partition_values = list(raw_partitions)
        except (AttributeError, NotImplementedError, TypeError):
            # 测试桩或兼容存储可能不支持分区枚举；近期刷新仍可继续，
            # 但不提交全历史水位，待真实 Storage 下次补齐。
            partition_values = []
        historical_periods = sorted(
            {
                normalized
                for partition in partition_values
                if (normalized := _normalize_date_str(partition)) is not None
            },
            reverse=True,
        )

    periods = list(dict.fromkeys(recent_periods + historical_periods))
    workers = get_tushare_settings()["download_concurrency"]
    t0 = time.time()
    results: List[Tuple[str, bool, int]] = []

    # 各季度分区文件相互独立，可安全并发；逐季结果统一汇总后决定水位推进。
    if workers <= 1 or len(periods) <= 1:
        # 串行路径（与 scripts/raw_download 的 _run_concurrent 降级口径一致，便于排障）
        for idx, period in enumerate(periods, 1):
            results.append(_refresh_one_cashflow_period(client, storage, period))
            if idx % 20 == 0 or idx == len(periods):
                logger.info(f"[cashflow] 修订刷新进度 {idx}/{len(periods)} (串行)")
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cashflow-refresh") as pool:
            futures = [
                pool.submit(_refresh_one_cashflow_period, client, storage, period)
                for period in periods
            ]
            for idx, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if idx % 20 == 0 or idx == len(periods):
                    logger.info(f"[cashflow] 修订刷新进度 {idx}/{len(periods)} (并发={workers})")

    period_success: Dict[str, bool] = {}
    refreshed_rows = 0
    for period, success, rows in results:
        period_success[period] = success
        refreshed_rows += rows

    recent_success = all(period_success.get(period, False) for period in recent_periods)
    if daily_due and recent_success:
        storage.save_sync_watermark(_CASHFLOW_REVISION_REFRESH_WATERMARK, target)
    elif daily_due:
        logger.warning(
            f"[cashflow] 近期修订刷新部分失败，下次运行重试: "
            f"{sum(period_success.get(p, False) for p in recent_periods)}/{len(recent_periods)}"
        )

    historical_success = all(period_success.get(period, False) for period in historical_periods)
    if full_due and historical_periods and historical_success:
        storage.save_sync_watermark(_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK, target)
    elif full_due and historical_periods:
        logger.warning(
            f"[cashflow] 全历史版本刷新部分失败，下次运行重试: "
            f"{sum(period_success.get(p, False) for p in historical_periods)}/"
            f"{len(historical_periods)}"
        )

    logger.info(
        f"[cashflow] 修订刷新完成: 查询 {len(periods)} 个季度, "
        f"分区累计 {refreshed_rows} 行, full_refresh={full_due}, "
        f"耗时={time.time() - t0:.0f}秒"
    )


def _try_download_cashflow(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载现金流量表数据

    数据充足（>= 阈值）：按公告日区间补齐增量（版本化去重）。
    数据不足或不存在：按季度批量下载全量（分页粒度 6400，版本化去重）。
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
                dedup_cols=_CASHFLOW_VERSION_DEDUP_COLS,
                fetch_by_date=lambda d: _query_with_pagination(
                    client,
                    "cashflow_vip",
                    page_limit=6400,
                    ann_date=d,
                ),
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
        dedup_cols=_CASHFLOW_VERSION_DEDUP_COLS,
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
    merged = _concat_no_warning(all_pages)
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


def _try_download_dividend(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
    existing_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """下载分红送股数据（dividend，2000 积分）。

    先按逐股覆盖状态补齐全历史，再按自然日增量推进同步水位；历史总行数
    只反映分红事件数量，不能作为是否执行日增量的完整性或时效性判据。
    """
    from ...data.dividend_raw import (
        DIVIDEND_DEDUP_COLS,
        download_dividend_full,
    )

    existing = existing_df if existing_df is not None else _load_all_partitions(storage, "dividend")

    # 先补未覆盖股票（按股全历史）：已覆盖股票 O(分区扫描) 跳过，失败股票
    # 保持 failed 状态、成功空结果记为 empty，避免失败被编码为“不分红”，
    # 也避免真实不分红股票反复查询。
    stock_basic = storage.load_raw("stock_basic")
    try:
        if stock_basic is not None and len(stock_basic) > 0:
            existing = download_dividend_full(
                client,
                storage,
                stock_basic,
                existing_df=existing,
            )
    except Exception as e:
        logger.warning(f"dividend 未覆盖股票补齐失败: {e}")

    if existing is None or len(existing) == 0:
        logger.info("分红送股全历史补齐后仍无数据，跳过日期增量")
        return existing

    # 单日增量需同时查询 ann_date 与 imp_ann_date：TuShare dividend 表实施
    # 状态更新不产生新的 ann_date（预案公告日保持不变），只查 ann_date 会漏掉
    # 当天才发布实施公告的记录。两路结果合并去重（落盘去重键含 div_proc）。
    def _fetch_dividend_day(d: str) -> Optional[pd.DataFrame]:
        parts: List[pd.DataFrame] = []
        for kwarg in ({"ann_date": d}, {"imp_ann_date": d}):
            day_df = client.get_dividend(**kwarg)
            if day_df is not None and len(day_df) > 0:
                parts.append(day_df)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return _concat_no_warning(parts)

    try:
        existing = _incremental_catchup_by_calendar_date(
            storage=storage,
            dataset_name="dividend",
            existing_df=existing,
            trade_date=trade_date,
            date_col="ann_date",
            dedup_cols=list(DIVIDEND_DEDUP_COLS),
            fetch_by_date=_fetch_dividend_day,
            partition_date_col="ann_date",
            partition_mode="year",
        )
    except Exception as e:
        logger.warning(f"增量下载 dividend 失败: {e}")
    # 统一返回全量数据（因子 lookup 需覆盖历史与未来已公告事件）
    return existing
