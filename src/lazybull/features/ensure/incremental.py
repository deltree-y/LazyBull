# -*- coding: utf-8 -*-
"""ensure 子包：按自然日增量补齐与日期工具（公告/事件类数据）。"""

from typing import Callable, List, Optional

import pandas as pd
from loguru import logger

from ...data import Storage


def _normalize_date_str(date_value: object) -> Optional[str]:
    """将日期值标准化为 YYYYMMDD 字符串。"""
    if pd.isna(date_value):
        return None
    date_str = str(date_value).strip()
    if not date_str:
        return None
    date_str = date_str.replace("-", "")[:8]
    if len(date_str) != 8 or not date_str.isdigit():
        return None
    return date_str


def _get_latest_date(df: Optional[pd.DataFrame], date_col: str) -> Optional[str]:
    """从 DataFrame 中提取指定日期列的最大日期（YYYYMMDD）。"""
    if df is None or len(df) == 0 or date_col not in df.columns:
        return None
    dates = df[date_col].map(_normalize_date_str).dropna()
    if dates.empty:
        return None
    return str(dates.max())


def _iter_calendar_dates(start_date: str, end_date: str) -> List[str]:
    """生成闭区间 [start_date, end_date] 的自然日列表（YYYYMMDD）。"""
    try:
        start_ts = pd.to_datetime(start_date, format="%Y%m%d")
        end_ts = pd.to_datetime(end_date, format="%Y%m%d")
    except Exception:
        return []
    if start_ts > end_ts:
        return []
    return [d.strftime("%Y%m%d") for d in pd.date_range(start_ts, end_ts, freq="D")]


def _incremental_catchup_by_calendar_date(
    storage: Storage,
    dataset_name: str,
    existing_df: Optional[pd.DataFrame],
    trade_date: str,
    date_col: str,
    dedup_cols: List[str],
    fetch_by_date: Callable[[str], Optional[pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    """按自然日补齐公告/事件类增量数据，避免只查单日导致漏数。

    适用于 ann_date/report_date 这类“可能在非交易日发布”的数据。
    """
    if existing_df is None or len(existing_df) == 0:
        return existing_df

    target_date = _normalize_date_str(trade_date)
    if target_date is None:
        logger.warning(f"[{dataset_name}] 无法解析 trade_date={trade_date}，跳过增量补齐")
        return existing_df

    latest_date = _get_latest_date(existing_df, date_col)
    if latest_date is None:
        logger.warning(
            f"[{dataset_name}] 本地数据缺少有效 {date_col}，无法执行区间补齐，保持现有数据"
        )
        return existing_df

    if latest_date >= target_date:
        logger.info(
            f"[{dataset_name}] 本地最新 {date_col}={latest_date}，已覆盖目标日期 {target_date}"
        )
        return existing_df

    start_date = (
        pd.to_datetime(latest_date, format="%Y%m%d") + pd.Timedelta(days=1)
    ).strftime("%Y%m%d")
    pending_dates = _iter_calendar_dates(start_date, target_date)
    if not pending_dates:
        return existing_df

    logger.info(
        f"[{dataset_name}] 区间增量补齐: {date_col} {start_date}~{target_date} "
        f"(共 {len(pending_dates)} 天)"
    )

    new_dfs: List[pd.DataFrame] = []
    success_days = 0
    empty_days = 0
    failed_days = 0

    for idx, cur_date in enumerate(pending_dates, 1):
        try:
            day_df = fetch_by_date(cur_date)
            if day_df is not None and len(day_df) > 0:
                new_dfs.append(day_df)
                success_days += 1
            else:
                empty_days += 1
        except Exception as e:
            failed_days += 1
            logger.warning(f"[{dataset_name}] {date_col}={cur_date} 增量下载失败: {e}")

        if idx % 30 == 0 or idx == len(pending_dates):
            logger.info(
                f"[{dataset_name}] 增量进度 {idx}/{len(pending_dates)} "
                f"(有数据={success_days}, 空={empty_days}, 失败={failed_days})"
            )

    if not new_dfs:
        logger.info(
            f"[{dataset_name}] 区间增量完成: 无新增记录 "
            f"(空={empty_days}, 失败={failed_days})"
        )
        return existing_df

    new_merged = pd.concat(new_dfs, ignore_index=True)
    result = _append_and_save_raw(
        storage,
        dataset_name,
        new_merged,
        dedup_cols=dedup_cols,
    )
    logger.info(
        f"[{dataset_name}] 区间增量完成: 新增 {len(new_merged)} 条, "
        f"总计 {len(result)} 条"
    )
    return result


def _append_and_save_raw(
    storage: Storage,
    dataset_name: str,
    new_df: pd.DataFrame,
    dedup_cols: List[str],
) -> pd.DataFrame:
    """将增量数据追加到已有单文件并去重保存

    Args:
        storage: Storage 实例
        dataset_name: 数据集名称（如 fina_indicator）
        new_df: 新下载的增量 DataFrame
        dedup_cols: 去重列

    Returns:
        合并后的完整 DataFrame
    """
    existing_df = storage.load_raw(dataset_name)
    if existing_df is not None and len(existing_df) > 0:
        result = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        result = new_df.copy()
    result = result.drop_duplicates(subset=dedup_cols, keep="last")
    storage.save_raw(result, dataset_name, is_force=True)
    return result