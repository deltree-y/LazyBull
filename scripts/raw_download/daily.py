# -*- coding: utf-8 -*-
"""raw_download 子包：日线数据下载（按日分区原子性 + 并发）。"""

import threading
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.data import Storage, TushareClient

from .core import DAILY_SUBSETS, ERROR_COLLECTOR, ProgressTracker, _run_concurrent


def _fetch_stk_limit_paginated(client: TushareClient, trade_date: str) -> pd.DataFrame:
    """按日分页获取涨跌停价格（TuShare stk_limit 单次 limit 上限 6000，
    单日全市场含指数约 7400 条，超限期必须分页取全，否则被截断到 6000）。"""
    from .periodic import _query_with_pagination

    return _query_with_pagination(client, "stk_limit", page_limit=6000, trade_date=trade_date)


_DAILY_FETCHERS: Dict[str, Callable] = {
    "daily": lambda c, d: c.get_daily(trade_date=d),
    "daily_basic": lambda c, d: c.get_daily_basic(trade_date=d),
    "adj_factor": lambda c, d: c.get_adj_factor(trade_date=d),
    "suspend": lambda c, d: c.get_suspend_d(trade_date=d),
    "stk_limit": _fetch_stk_limit_paginated,
    "moneyflow": lambda c, d: c.get_moneyflow(trade_date=d),
    "stock_st": lambda c, d: c.get_stock_st(trade_date=d),
}


_DAILY_ALLOW_EMPTY = {"suspend", "stk_limit", "moneyflow", "adj_factor", "stock_st"}


def _pending_daily_subsets(
    storage: Storage, trade_date: str, force: bool, subsets: Optional[List[str]] = None
) -> List[str]:
    """返回当日还需要下载的子数据集名称（可限定 subsets 子集）。"""
    check = list(subsets) if subsets else list(DAILY_SUBSETS)
    if force:
        return check
    return [s for s in check if not storage.is_data_exists("raw", s, trade_date)]


def download_daily_data(
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False,
    subsets: Optional[List[str]] = None,
) -> None:
    """下载日线数据 (按日期分区, 原子性 + ETA 进度)。

    Args:
        subsets: 仅下载指定子数据集（如 ["stk_limit"]），默认全部日线子集。

    修复 #5: 单日 7 个接口原子性 —— 只要任一接口抛异常, 整天标记失败;
    已成功拉取的 DataFrame 不落盘, 下次重跑可重新尝试, 避免"半个日子"永久缺失。
    修复 #4: moneyflow 返回空时不再是 error 日志, 而是 raise 被记录到错误汇总。
    修复 #13: len(trading_dates)==0 时直接返回, 防止除零。
    """
    logger.info(f"下载日线数据 ({start_date}~{end_date})...")

    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    if not trading_dates:
        logger.warning(f"日期区间 {start_date}~{end_date} 内无交易日, 跳过日线下载")
        return

    logger.info(f"共 {len(trading_dates)} 个交易日需要检查")

    # 预筛: 已全部落盘的日期直接跳过
    pending: List[Tuple[str, List[str]]] = []
    skip_count = 0
    for td in trading_dates:
        subs = _pending_daily_subsets(storage, td, force, subsets=subsets)
        if not subs:
            skip_count += 1
        else:
            pending.append((td, subs))

    logger.info(f"跳过已存在: {skip_count} 天, 需要下载: {len(pending)} 天")
    if not pending:
        return

    tracker = ProgressTracker(len(pending), label="daily", log_every=20)
    total_rows: Dict[str, int] = {s: 0 for s in DAILY_SUBSETS}
    fail_days = 0
    # 并发下 total_rows / fail_days / Storage 写盘都需要线程保护
    stats_lock = threading.Lock()

    def _fetch_one_day(trade_date: str, subs: List[str]) -> None:
        """下载单个交易日的全部子接口 (原子性) 并落盘。"""
        nonlocal fail_days
        day_data: Dict[str, "pd.DataFrame"] = {}
        day_failed_reason: Optional[str] = None
        for sub in subs:
            try:
                df = _DAILY_FETCHERS[sub](client, trade_date)
                if df is None or len(df) == 0:
                    if sub in _DAILY_ALLOW_EMPTY:
                        day_data[sub] = pd.DataFrame()  # 占位, 不写盘
                    else:
                        day_failed_reason = f"{sub} 返回空 (强制依赖)"
                        break
                else:
                    day_data[sub] = df
            except Exception as e:
                day_failed_reason = f"{sub} 异常: {e}"
                break

        if day_failed_reason is not None:
            ERROR_COLLECTOR.add("daily", trade_date, day_failed_reason)
            with stats_lock:
                fail_days += 1
            tracker.tick(extra_info=f"fail={fail_days}")
            return

        # 全部接口都成功(或允许空) —— 统一落盘, 保证当日原子性
        # Storage.save_raw_by_date 对不同 (sub, trade_date) 路径写不同文件, 可并发
        for sub, df in day_data.items():
            if len(df) > 0:
                storage.save_raw_by_date(df, sub, trade_date)
                with stats_lock:
                    total_rows[sub] += len(df)

        tracker.tick(extra_info=f"fail={fail_days}")

    _run_concurrent(
        work_items=pending,
        worker=lambda item: _fetch_one_day(item[0], item[1]),
        label="daily",
    )

    logger.info("=" * 60)
    logger.info("日线数据下载完成")
    for sub in DAILY_SUBSETS:
        logger.info(f"  {sub:12s}: 新增 {total_rows[sub]} 条记录")
    logger.info(f"失败天数: {fail_days} (详见最终错误汇总)")
    logger.info("=" * 60)
