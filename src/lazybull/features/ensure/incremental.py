# -*- coding: utf-8 -*-
"""ensure 子包：按自然日增量补齐与日期工具（公告/事件类数据）。"""

from typing import Callable, List, Optional

import pandas as pd
from loguru import logger

from ...data import Storage
from ...data.financial_statement_versions import deduplicate_prefer_latest_update_flag
from ...data.report_rc import deduplicate_report_rc
from .concat_utils import _concat_no_warning


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


def _max_date(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """取两个 YYYYMMDD 日期字符串中较大者（任一为 None 返回另一个）。"""
    if a is None:
        return b
    if b is None:
        return a
    return a if a >= b else b


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
    partition_date_col: Optional[str] = None,
    partition_mode: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """按自然日补齐公告/事件类增量数据，避免只查单日导致漏数。

    适用于 ann_date/report_date 这类“可能在非交易日发布”的数据。

    Args:
        storage: Storage 实例
        dataset_name: 数据集名称
        existing_df: 本地已有数据 (用于判断最新日期与缺口)
        trade_date: 目标日期 YYYYMMDD
        date_col: 公告/事件日期列 (如 ann_date / report_date)
        dedup_cols: 去重列
        fetch_by_date: 按单日拉取数据的回调
        partition_date_col: 若提供, 增量按该列路由写入对应分区 (分区存储模式);
            否则沿用整文件追加 (单文件模式)
        partition_mode: 分区模式 ("quarter" / "year")，配合 partition_date_col
    """
    if existing_df is None or len(existing_df) == 0:
        return existing_df

    target_date = _normalize_date_str(trade_date)
    if target_date is None:
        logger.warning(f"[{dataset_name}] 无法解析 trade_date={trade_date}，跳过增量补齐")
        return existing_df

    watermark = storage.load_sync_watermark(dataset_name)
    if watermark is not None:
        # 水位 = 连续成功同步前缀。水位之后的区间可能存在失败日（数据未连续），
        # 不可用数据最新公告日越过水位，必须从水位之后逐日重查。
        latest_date = watermark
    else:
        # 无水位：以数据最新公告日为首个连续前缀
        latest_data = _get_latest_date(existing_df, date_col)
        if latest_data is None:
            logger.warning(
                f"[{dataset_name}] 本地数据缺少有效 {date_col} 且无同步水位，"
                "无法执行区间补齐，保持现有数据"
            )
            return existing_df
        latest_date = latest_data

    if latest_date >= target_date:
        logger.info(
            f"[{dataset_name}] 本地最新 {date_col}={latest_date}，已覆盖目标日期 {target_date}"
        )
        return existing_df

    start_date = (pd.to_datetime(latest_date, format="%Y%m%d") + pd.Timedelta(days=1)).strftime(
        "%Y%m%d"
    )
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
    last_success: Optional[str] = None

    for idx, cur_date in enumerate(pending_dates, 1):
        try:
            day_df = fetch_by_date(cur_date)
            if day_df is not None and len(day_df) > 0:
                new_dfs.append(day_df)
                success_days += 1
            else:
                empty_days += 1
            last_success = cur_date
        except Exception as e:
            failed_days += 1
            logger.warning(f"[{dataset_name}] {date_col}={cur_date} 增量下载失败: {e}")
            # 水位只代表连续成功前缀：遇到首个失败立即停止，
            # 失败日及之后留待下次从水位之后重试，保证失败日不被跳过。
            break

        if idx % 30 == 0 or idx == len(pending_dates):
            logger.info(
                f"[{dataset_name}] 增量进度 {idx}/{len(pending_dates)} "
                f"(有数据={success_days}, 空={empty_days}, 失败={failed_days})"
            )

    # 先落盘新数据，成功后再原子推进水位：
    # 避免"水位已提交但数据落盘失败"导致下次跳过该区间造成永久缺失。
    result = existing_df
    if new_dfs:
        new_merged = _concat_no_warning(new_dfs)
        if partition_date_col and partition_mode:
            result = _append_and_save_partitioned(
                storage,
                dataset_name,
                new_merged,
                dedup_cols=dedup_cols,
                partition_date_col=partition_date_col,
                partition_mode=partition_mode,
                existing_full_df=existing_df,
            )
        else:
            result = _append_and_save_raw(
                storage,
                dataset_name,
                new_merged,
                dedup_cols=dedup_cols,
            )
        # 若落盘抛异常，此处不会执行，水位保持原值，下次从原水位之后重查

    # 数据（或空日查询）成功推进后，水位推进到最后一个成功日（含空日）
    if last_success is not None:
        storage.save_sync_watermark(dataset_name, _max_date(watermark, last_success))

    logger.info(
        f"[{dataset_name}] 区间增量完成: 新增数据日={success_days}, 空={empty_days}, "
        f"失败={failed_days}, 水位推进到 {last_success or '未推进'}"
    )
    return result


def _partition_date_str(date_value: object, partition_mode: str) -> Optional[str]:
    """把日期值映射为分区文件名日期 YYYY-MM-DD。

    Args:
        date_value: 日期值 (YYYYMMDD / YYYY-MM-DD 字符串或 datetime)
        partition_mode:
            - "quarter": 原样映射 (报告期 end_date 本身即季度末, 与 fina_indicator 一致)
            - "year": 映射为该年 12-31 (report_date 按年聚合, 与 report_rc 按年下载一致)

    Returns:
        YYYY-MM-DD 字符串; 无法解析返回 None
    """
    norm = _normalize_date_str(date_value)
    if norm is None:
        return None
    if partition_mode == "quarter":
        return f"{norm[:4]}-{norm[4:6]}-{norm[6:8]}"
    if partition_mode == "year":
        return f"{norm[:4]}-12-31"
    raise ValueError(f"不支持的分区模式: {partition_mode!r}")


def _load_all_partitions(storage: Storage, dataset_name: str) -> Optional[pd.DataFrame]:
    """读取数据集全部分区并合并 (分区优先加载路径)。

    Args:
        storage: Storage 实例
        dataset_name: 数据集名称 (forecast / report_rc)

    Returns:
        合并后的完整 DataFrame; 无任何分区返回 None
    """
    partitions = storage.list_partitions("raw", dataset_name)
    if not partitions:
        return None
    dfs: List[pd.DataFrame] = []
    for p in partitions:
        df = storage.load_raw_by_date(dataset_name, p)
        if df is not None and len(df) > 0:
            dfs.append(df)
    if not dfs:
        return None
    return _concat_no_warning(dfs)


def _drop_duplicates_keep_updated(
    df: pd.DataFrame,
    dedup_cols: List[str],
) -> pd.DataFrame:
    """按键去重，同键冲突时优先保留 TuShare ``update_flag=1`` 最新行。

    cashflow 使用包含 f_ann_date 的版本键，因此不同可用日版本仍会完整保留；
    只有同一版本键内部的重复行按官方最新标志确定选择。
    """
    return deduplicate_prefer_latest_update_flag(df, dedup_cols)


def _append_and_save_partitioned(
    storage: Storage,
    dataset_name: str,
    new_df: pd.DataFrame,
    dedup_cols: List[str],
    partition_date_col: str,
    partition_mode: str,
    existing_full_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """把增量数据按分区日期列路由写入对应分区 (分区内去重), 返回合并后全量。

    forecast 按 end_date 季度分区、report_rc 按 report_date 年分区;
    同分区内按 dedup_cols 去重, 跨分区天然隔离 (分区键即去重键的一部分),
    避免整文件读-合并-重写 (O(全量) -> O(增量))。

    Args:
        storage: Storage 实例
        dataset_name: 数据集名称
        new_df: 新增增量数据
        dedup_cols: 分区内去重列
        partition_date_col: 分区依据的日期列 (forecast 用 end_date; report_rc 用 report_date)
        partition_mode: "quarter" 或 "year"
        existing_full_df: 调用方已加载的全量数据；提供时直接内存合并返回

    Returns:
        合并后的完整 DataFrame (已存在分区 + 新增)
    """
    if new_df is None or len(new_df) == 0 or partition_date_col not in new_df.columns:
        return _load_all_partitions(storage, dataset_name)

    work = new_df.copy()
    work["_partition_date"] = work[partition_date_col].map(
        lambda v: _partition_date_str(v, partition_mode)
    )
    work = work.dropna(subset=["_partition_date"])
    if len(work) == 0:
        return _load_all_partitions(storage, dataset_name)

    partition_count = work["_partition_date"].nunique()
    for part_date, part in work.groupby("_partition_date", sort=True):
        part = part.drop(columns=["_partition_date"])
        existing = storage.load_raw_by_date(dataset_name, part_date)
        if existing is not None and len(existing) > 0:
            merged = _concat_no_warning([existing, part])
        else:
            merged = part
        if dataset_name == "report_rc":
            merged = deduplicate_report_rc(
                merged,
                include_quarter=True,
                require_full_identity=True,
            )
        else:
            merged = _drop_duplicates_keep_updated(merged, dedup_cols)
        storage.save_raw_by_date(merged, dataset_name, part_date)

    logger.info(f"[{dataset_name}] 增量写入 {partition_count} 个分区")
    if existing_full_df is not None:
        result = _concat_no_warning([existing_full_df, new_df])
        if dataset_name == "report_rc":
            return deduplicate_report_rc(
                result,
                include_quarter=True,
                require_full_identity=True,
            )
        return _drop_duplicates_keep_updated(result, dedup_cols)
    return _load_all_partitions(storage, dataset_name)


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
        result = _concat_no_warning([existing_df, new_df])
    else:
        result = new_df.copy()
    result = _drop_duplicates_keep_updated(result, dedup_cols)
    storage.save_raw(result, dataset_name, is_force=True)
    return result
