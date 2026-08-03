# -*- coding: utf-8 -*-
"""raw_download 子包：另类数据下载 (股东人数/北向/龙虎榜/一致预期/现金流)。"""

import threading
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.data import Storage, TushareClient

from .core import ERROR_COLLECTOR, ProgressTracker, _run_concurrent
from .periodic import (
    _generate_month_periods,
    _query_with_pagination,
    _save_merged,
    _to_int_date,
    download_by_period,
)


def download_stk_holdernumber(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    dedup_cols: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """按月批量下载股东人数数据。"""
    if dedup_cols is None:
        dedup_cols = ["ts_code", "end_date"]

    month_ranges = _generate_month_periods(start_date, end_date)
    if not month_ranges:
        logger.warning("[stk_holdernumber] 日期范围无效")
        return

    existing_df = None
    latest_ann = None
    if not force:
        existing_df = storage.load_raw("stk_holdernumber")
        if existing_df is not None and len(existing_df) > 0:
            logger.info(f"[stk_holdernumber] 已有 {len(existing_df)} 条数据")
            if "ann_date" in existing_df.columns:
                ann_dates = (
                    existing_df["ann_date"]
                    .astype(str)
                    .str.replace("-", "")
                    .str[:8]
                )
                ann_dates = ann_dates[ann_dates.str.match(r"^\d{8}$", na=False)]
                if len(ann_dates) > 0:
                    latest_ann = ann_dates.max()

    if latest_ann is not None:
        # 断点续传：只下 ann_date 大于已有最大公告日的月份段，避免每次全量重下
        month_ranges = [mr for mr in month_ranges if mr[1] > latest_ann]
        if not month_ranges:
            logger.info(
                f"[stk_holdernumber] 已有数据覆盖至 {latest_ann}，无需增量。如需重下加 --force"
            )
            return
        logger.info(
            f"[stk_holdernumber] 断点续传：已有最新公告日 {latest_ann}，"
            f"待下 {len(month_ranges)} 个月"
        )

    logger.info(
        f"[stk_holdernumber] 按月下载: {len(month_ranges)} 月 "
        f"({month_ranges[0][0]}~{month_ranges[-1][1]})"
    )

    tracker = ProgressTracker(len(month_ranges), label="stk_holdernumber", log_every=12)
    all_dfs: List["pd.DataFrame"] = []
    success = empty = 0

    for m_start, m_end in month_ranges:
        try:
            # 单月分页拉取，规避 stk_holdernumber 单次 3000 条上限截断
            df = _query_with_pagination(
                client,
                "stk_holdernumber",
                page_limit=3000,
                start_date=m_start,
                end_date=m_end,
            )
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success += 1
            else:
                empty += 1
        except Exception as e:
            ERROR_COLLECTOR.add("stk_holdernumber", f"{m_start}~{m_end}", str(e))
        tracker.tick(extra_info=f"ok={success} empty={empty}")

    if all_dfs:
        _save_merged(
            storage, "stk_holdernumber", all_dfs, existing_df,
            dedup_cols, sort_cols=["ann_date", "end_date"],
        )

    logger.info(f"[stk_holdernumber] 完成: 成功={success} 空={empty}")


def download_moneyflow_hsgt(
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载北向资金 (按日分区)。

    修复 #3: 先检查哪些交易日尚未落盘, 仅针对缺失日期计算需要拉取的半年分段,
    避免"分段已全下、逐日写入才发现都跳过"的浪费。
    """
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    if not trading_dates:
        logger.warning("[moneyflow_hsgt] 区间无交易日, 跳过")
        return

    # 先筛待下载日期
    if force:
        pending_dates = list(trading_dates)
    else:
        pending_dates = [td for td in trading_dates if not storage.is_data_exists("raw", "moneyflow_hsgt", td)]

    skip = len(trading_dates) - len(pending_dates)
    logger.info(
        f"[moneyflow_hsgt] 共 {len(trading_dates)} 个交易日, 跳过 {skip}, 待下 {len(pending_dates)}"
    )
    if not pending_dates:
        return

    # 基于 pending_dates 的首末日决定拉取窗口, 按半年切分
    pd_start, pd_end = pending_dates[0], pending_dates[-1]
    months = _generate_month_periods(pd_start, pd_end)
    segments: List[Tuple[str, str]] = []
    i = 0
    while i < len(months):
        seg_start = months[i][0]
        j = min(i + 5, len(months) - 1)
        seg_end = months[j][1]
        segments.append((seg_start, seg_end))
        i = j + 1

    logger.info(f"[moneyflow_hsgt] 将拉取 {len(segments)} 个半年分段覆盖待下载日期")

    all_dfs: List["pd.DataFrame"] = []
    for s, e in segments:
        try:
            df = client.get_moneyflow_hsgt(start_date=s, end_date=e)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                logger.info(f"  [moneyflow_hsgt] {s}~{e}: {len(df)} 条")
        except Exception as ex:
            ERROR_COLLECTOR.add("moneyflow_hsgt", f"{s}~{e}", str(ex))

    if not all_dfs:
        logger.warning("[moneyflow_hsgt] 全部分段返回空")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    merged["trade_date"] = merged["trade_date"].astype(str)
    merged = merged.drop_duplicates(subset=["trade_date"], keep="last")

    tracker = ProgressTracker(len(pending_dates), label="moneyflow_hsgt_write", log_every=200)
    success = 0
    for td in pending_dates:
        sub = merged[merged["trade_date"] == td]
        if len(sub) > 0:
            storage.save_raw_by_date(sub, "moneyflow_hsgt", td)
            success += 1
        tracker.tick(extra_info=f"ok={success}")

    logger.info(f"[moneyflow_hsgt] 完成: 新下载={success} 跳过={skip}")


def download_top_list(
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载龙虎榜 (按日分区, 无数据存空占位避免重复下载)。"""
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    if not trading_dates:
        logger.warning("[top_list] 区间无交易日, 跳过")
        return

    pending = [td for td in trading_dates if force or not storage.is_data_exists("raw", "top_list", td)]
    skip = len(trading_dates) - len(pending)

    logger.info(f"[top_list] 共 {len(trading_dates)} 天, 跳过 {skip}, 待下 {len(pending)}")
    if not pending:
        return

    tracker = ProgressTracker(len(pending), label="top_list", log_every=100)
    counters = {"success": 0, "empty": 0}
    counter_lock = threading.Lock()

    def _worker(td: str) -> None:
        try:
            df = client.get_top_list(trade_date=td)
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, "top_list", td)
                with counter_lock:
                    counters["success"] += 1
            else:
                storage.save_raw_by_date(
                    pd.DataFrame(columns=[
                        "trade_date", "ts_code", "net_amount",
                        "net_rate", "amount_rate", "reason",
                    ]),
                    "top_list", td,
                )
                with counter_lock:
                    counters["empty"] += 1
        except Exception as e:
            ERROR_COLLECTOR.add("top_list", td, str(e))
        tracker.tick(extra_info=f"ok={counters['success']} empty={counters['empty']}")

    _run_concurrent(pending, _worker, label="top_list")

    logger.info(f"[top_list] 完成: 新下载={counters['success']} 空占位={counters['empty']}")


def download_report_rc(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载卖方研报一致预期 (按年分页增量)。

    修复 #8: --force 模式下丢弃 existing_df, 语义与其它函数一致 (强制重下不保留旧数据)。
    """
    existing_df = None
    existing_years: Set[str] = set()
    if not force:
        existing_df = storage.load_raw("report_rc")
        if existing_df is not None and len(existing_df) > 0 and "report_date" in existing_df.columns:
            existing_years = set(
                existing_df["report_date"].astype(str).str[:4].unique()
            )
            logger.info(f"[report_rc] 已有 {len(existing_df)} 条, 覆盖 {len(existing_years)} 年")

    start_year = _to_int_date(start_date) // 10000
    end_year = _to_int_date(end_date) // 10000
    years_to_download = [
        str(y) for y in range(start_year, end_year + 1)
        if force or str(y) not in existing_years
    ]

    if not years_to_download:
        logger.info("[report_rc] 全部年份已存在, 跳过。如需重下加 --force")
        return

    logger.info(f"[report_rc] 按年下载 {len(years_to_download)} 年 ({years_to_download[0]}~{years_to_download[-1]})")

    tracker = ProgressTracker(len(years_to_download), label="report_rc", log_every=1)
    all_dfs: List["pd.DataFrame"] = []
    success = empty = 0
    for y in years_to_download:
        y_start = max(f"{y}0101", start_date)
        y_end = min(f"{y}1231", end_date)
        try:
            # 按年分页拉取，规避 report_rc 单次 2000 条上限截断
            df = _query_with_pagination(
                client,
                "report_rc",
                page_limit=2000,
                start_date=y_start,
                end_date=y_end,
            )
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success += 1
                logger.info(f"  [report_rc] {y}: {len(df)} 条")
            else:
                empty += 1
        except Exception as e:
            ERROR_COLLECTOR.add("report_rc", f"year={y}", str(e))
        tracker.tick(extra_info=f"ok={success} empty={empty}")

    if all_dfs:
        _save_merged(
            storage, "report_rc", all_dfs,
            # 修复 #8: force 时 existing_df 传 None
            existing_df if not force else None,
            dedup_cols=["ts_code", "report_date", "org_name", "author_name"],
            sort_cols=["report_date"],
        )

    logger.info(f"[report_rc] 完成: 成功={success} 空={empty}")


def download_cashflow(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载现金流量表数据 (cashflow_vip, 5000积分)。按报告期批量下载全市场数据。"""
    download_by_period(
        client, storage,
        dataset_name="cashflow",
        api_name="cashflow_vip",
        start_date=start_date, end_date=end_date,
        dedup_cols=["ts_code", "end_date", "ann_date"],
        fields=None,
        force=force,
        partition_by_period=True,
        sort_cols=["end_date", "ann_date"],
    )