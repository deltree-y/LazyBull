# -*- coding: utf-8 -*-
"""ensure 子包：历史 clean 数据补齐与旧 schema 回补。"""

from typing import List, Optional

import pandas as pd
from loguru import logger

from ...data import DataCleaner, DataLoader, Storage, TushareClient
from ...data.ensure import ensure_clean_data_for_date
from .bulk import _query_with_pagination, _save_merged_bulk
from .concat_utils import _concat_no_warning
from .constants import HISTORICAL_DATA_MONTHS, MAX_HISTORICAL_DAYS
from .incremental import (
    _append_and_save_partitioned,
    _drop_duplicates_keep_updated,
    _normalize_date_str,
)


def _ensure_historical_clean_data(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    client: TushareClient,
    trade_date: str,
    force: bool
) -> bool:
    """确保历史 clean 数据存在
    
    Features 构建需要历史数据来计算动量、均值等特征。
    这里按与 build_clean_features 一致的 warmup 窗口补齐历史 clean 数据。
    
    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        cleaner: DataCleaner 实例
        client: TushareClient 实例
        trade_date: 当前交易日期，格式 YYYYMMDD
        force: 是否强制重新构建
        
    Returns:
        是否成功（至少部分历史数据可用）
    """
    # 获取交易日历
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        logger.warning("无法加载交易日历，跳过历史数据检查")
        return False
    
    # 确保日期格式统一
    if 'cal_date' in trade_cal.columns:
        if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
            trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
    
    # 获取 warmup 窗口内的历史交易日
    start_dt = pd.to_datetime(trade_date, format='%Y%m%d') - pd.DateOffset(
        months=HISTORICAL_DATA_MONTHS
    )
    
    trading_dates = trade_cal[
        (trade_cal['cal_date'] >= start_dt) &
        (trade_cal['cal_date'] < pd.to_datetime(trade_date, format='%Y%m%d')) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    if not trading_dates:
        logger.warning("未找到历史交易日")
        return False
    
    # 转换为 YYYYMMDD 格式
    trading_dates_str = [
        d.strftime('%Y%m%d') if isinstance(d, pd.Timestamp) else d
        for d in trading_dates
    ]
    
    logger.info(f"检查 {len(trading_dates_str)} 个历史交易日的 clean 数据")
    
    # 检查并补齐缺失的历史数据（最多补齐最近的指定个交易日）
    missing_count = 0
    success_count = 0
    
    for hist_date in trading_dates_str[-MAX_HISTORICAL_DAYS:]:  # 最多检查最近指定个交易日
        # 检查 daily/daily_basic/moneyflow 任一缺失即需补齐
        daily_ok = storage.is_data_exists("clean", "daily", hist_date)
        daily_basic_ok = storage.is_data_exists("clean", "daily_basic", hist_date)
        moneyflow_ok = storage.is_data_exists("clean", "moneyflow", hist_date)
        if not (daily_ok and daily_basic_ok and moneyflow_ok):
            missing_count += 1
            # 尝试补齐（ensure_clean_data_for_date 内部会跳过已存在的数据集）
            if ensure_clean_data_for_date(
                storage, loader, cleaner, client, hist_date, force
            ):
                success_count += 1
    
    if missing_count > 0:
        logger.info(f"补齐了 {success_count}/{missing_count} 个历史交易日的 clean 数据")
    
    # 只要有部分数据可用就返回 True
    return True


def _merge_refreshed_rows(
    existing_df: pd.DataFrame,
    refreshed_df: pd.DataFrame,
    dedup_cols: List[str],
) -> pd.DataFrame:
    """按主键将刷新结果补回旧表，保留旧表中的其他列。"""
    existing = _drop_duplicates_keep_updated(existing_df, dedup_cols).set_index(dedup_cols)
    refreshed = _drop_duplicates_keep_updated(refreshed_df, dedup_cols).set_index(dedup_cols)

    full_index = existing.index.union(refreshed.index)
    existing = existing.reindex(full_index)
    refreshed = refreshed.reindex(full_index)

    for col in refreshed.columns:
        if col in existing.columns:
            existing[col] = refreshed[col].combine_first(existing[col])
        else:
            existing[col] = refreshed[col]

    return existing.reset_index()


def _refresh_existing_period_rows(
    client: TushareClient,
    storage: Storage,
    dataset_name: str,
    api_name: str,
    existing_df: Optional[pd.DataFrame],
    dedup_cols: List[str],
    fields: str,
    period_col: str = "end_date",
    partition_date_col: Optional[str] = None,
    partition_mode: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """对已有 period 数据按季度重拉，回补旧 schema 缺列。

    Args:
        partition_date_col / partition_mode: 若提供，回补结果路由写入对应分区
            （分区存储数据集，避免写入单文件后被分区遮蔽）；否则沿用单文件保存。
    """
    if existing_df is None or len(existing_df) == 0 or period_col not in existing_df.columns:
        return existing_df

    refresh_periods = sorted(
        {
            period
            for period in existing_df[period_col].map(_normalize_date_str).dropna().tolist()
            if period
        }
    )
    if not refresh_periods:
        return existing_df

    logger.info(f"[{dataset_name}] 检测到旧 schema，按季度回补 {len(refresh_periods)} 个 period")
    refreshed_dfs: List[pd.DataFrame] = []
    for period in refresh_periods:
        try:
            df = _query_with_pagination(client, api_name, fields=fields, period=period)
            if df is not None and len(df) > 0:
                refreshed_dfs.append(df)
        except Exception as e:
            logger.warning(f"[{dataset_name}] period={period} schema 回补失败: {e}")

    if not refreshed_dfs:
        logger.warning(f"[{dataset_name}] schema 回补未获取到任何数据，保留旧表")
        return existing_df

    refreshed_df = _concat_no_warning(refreshed_dfs)
    merged_df = _merge_refreshed_rows(existing_df, refreshed_df, dedup_cols)
    if partition_date_col and partition_mode:
        _append_and_save_partitioned(
            storage,
            dataset_name,
            merged_df,
            dedup_cols=dedup_cols,
            partition_date_col=partition_date_col,
            partition_mode=partition_mode,
        )
    else:
        storage.save_raw(merged_df, dataset_name, is_force=True)
    logger.info(f"[{dataset_name}] schema 回补完成: {len(merged_df)} 条记录")
    return merged_df