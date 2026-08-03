# -*- coding: utf-8 -*-
"""raw_download 子包：按日分区下载模板 (margin_detail / cyq_perf / stock_st)。"""

import threading
from typing import Callable, List, Optional

import pandas as pd
from loguru import logger

from src.lazybull.data import Storage, TushareClient

from .core import ERROR_COLLECTOR, ProgressTracker, _run_concurrent


def _download_by_trade_date(
    dataset_name: str,
    fetcher: Callable[[TushareClient, str], "pd.DataFrame"],
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """按交易日分区下载模板 (margin_detail, cyq_perf 等共用)。"""
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    if not trading_dates:
        logger.warning(f"[{dataset_name}] 区间无交易日, 跳过")
        return

    # 预筛
    pending = [td for td in trading_dates if force or not storage.is_data_exists("raw", dataset_name, td)]
    skip = len(trading_dates) - len(pending)

    logger.info(f"[{dataset_name}] 共 {len(trading_dates)} 天, 跳过 {skip}, 待下 {len(pending)}")
    if not pending:
        return

    tracker = ProgressTracker(len(pending), label=dataset_name, log_every=100)
    counters = {"success": 0, "empty": 0}
    counter_lock = threading.Lock()

    def _worker(td: str) -> None:
        try:
            df = fetcher(client, td)
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, dataset_name, td)
                with counter_lock:
                    counters["success"] += 1
            else:
                with counter_lock:
                    counters["empty"] += 1
        except Exception as e:
            ERROR_COLLECTOR.add(dataset_name, td, str(e))
        tracker.tick(extra_info=f"ok={counters['success']} empty={counters['empty']}")

    _run_concurrent(pending, _worker, label=dataset_name)

    logger.info(f"[{dataset_name}] 完成: 成功={counters['success']} 空={counters['empty']}")


def download_margin_detail(client, storage, trade_cal, start_date, end_date, force=False):
    _download_by_trade_date(
        "margin_detail",
        lambda c, d: c.query("margin_detail", trade_date=d),
        client, storage, trade_cal, start_date, end_date, force,
    )


def download_cyq_perf(client, storage, trade_cal, start_date, end_date, force=False):
    _download_by_trade_date(
        "cyq_perf",
        lambda c, d: c.get_cyq_perf(trade_date=d),
        client, storage, trade_cal, start_date, end_date, force,
    )


def download_stock_st(client, storage, trade_cal, start_date, end_date, force=False):
    _download_by_trade_date(
        "stock_st",
        lambda c, d: c.get_stock_st(trade_date=d),
        client, storage, trade_cal, start_date, end_date, force,
    )