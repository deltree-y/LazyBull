# -*- coding: utf-8 -*-
"""raw_download 子包：基础数据下载 (trade_cal / stock_basic)。"""

from typing import Optional

import pandas as pd
from loguru import logger

from src.lazybull.data import Storage, TushareClient

from .core import ERROR_COLLECTOR


def download_basic_data(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> "pd.DataFrame":
    """下载 trade_cal 和 stock_basic。

    注意修复 #2: trade_cal 需要扩展日历窗口时, 必须合并旧数据后再保存，
    否则短窗口调用会截断历史。stock_basic 同时拉取 L/D/P 修复 #12 生存者偏差。
    """
    # 1. 交易日历: 按需合并新旧数据
    logger.info("检查交易日历...")
    existing_cal: Optional[pd.DataFrame] = None
    need_download = force
    if not force:
        existing_cal = storage.load_raw("trade_cal")
        if existing_cal is None or "cal_date" not in existing_cal.columns:
            need_download = True
        else:
            latest = str(existing_cal["cal_date"].astype(str).max()).replace("-", "")[:8]
            earliest = str(existing_cal["cal_date"].astype(str).min()).replace("-", "")[:8]
            # 任一端不覆盖则需要扩展
            if latest < end_date or earliest > start_date:
                need_download = True
                logger.info(
                    f"交易日历需要扩展: 现有 {earliest}~{latest}, "
                    f"目标 {start_date}~{end_date}"
                )
            else:
                logger.info(f"交易日历已覆盖 {earliest}~{latest}, 跳过")

    if need_download:
        # 为了安全, 拉取并集窗口 (min(现有起点, 目标起点) ~ max(现有终点, 目标终点))
        query_start = start_date
        query_end = end_date
        if existing_cal is not None and "cal_date" in existing_cal.columns:
            ex_min = str(existing_cal["cal_date"].astype(str).min()).replace("-", "")[:8]
            ex_max = str(existing_cal["cal_date"].astype(str).max()).replace("-", "")[:8]
            query_start = min(ex_min, start_date)
            query_end = max(ex_max, end_date)
        logger.info(f"下载交易日历 ({query_start}~{query_end})...")
        new_cal = client.get_trade_cal(
            start_date=query_start, end_date=query_end, exchange="SSE"
        )
        if existing_cal is not None and len(existing_cal) > 0:
            new_cal = pd.concat([existing_cal, new_cal], ignore_index=True)
            new_cal = new_cal.drop_duplicates(subset=["cal_date"], keep="last")
            new_cal = new_cal.sort_values("cal_date").reset_index(drop=True)
        storage.save_raw(new_cal, "trade_cal", is_force=True)
        logger.info(f"交易日历已保存: {len(new_cal)} 条")
        trade_cal = new_cal
    else:
        trade_cal = existing_cal

    # 2. 股票基本信息: 同时拉 L/D/P 消除生存者偏差 (#12)
    logger.info("检查股票基本信息...")
    if not force and storage.check_basic_data_freshness("stock_basic", end_date):
        logger.info("股票基本信息已存在, 跳过")
    else:
        logger.info("下载股票基本信息 (L+D+P)...")
        dfs = []
        for status in ("L", "D", "P"):
            try:
                df = client.get_stock_basic(list_status=status)
                if df is not None and len(df) > 0:
                    dfs.append(df)
                    logger.info(f"  list_status={status}: {len(df)} 条")
            except Exception as e:
                ERROR_COLLECTOR.add("stock_basic", f"list_status={status}", str(e))
        if not dfs:
            raise RuntimeError("stock_basic 三种状态全部下载失败, 无法继续")
        stock_basic = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["ts_code"])
        storage.save_raw(stock_basic, "stock_basic", is_force=True)
        logger.info(f"股票基本信息已保存: {len(stock_basic)} 条 (含退市/暂停)")

    return trade_cal