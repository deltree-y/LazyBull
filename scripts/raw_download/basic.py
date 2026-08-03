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
    # 1. 交易日历: 每次全量下载（不按 start/end 裁剪，保证日历完整），合并旧数据去重
    logger.info("下载交易日历 (全量)...")
    existing_cal = storage.load_raw("trade_cal")
    new_cal = client.get_trade_cal(exchange="SSE")
    if new_cal is None or len(new_cal) == 0:
        if existing_cal is not None and len(existing_cal) > 0:
            logger.warning("全量下载交易日历返回空，保留已有数据")
            trade_cal = existing_cal
        else:
            raise RuntimeError("交易日历下载失败且无历史数据可保留")
    else:
        if existing_cal is not None and len(existing_cal) > 0:
            new_cal = pd.concat([existing_cal, new_cal], ignore_index=True)
            new_cal = new_cal.drop_duplicates(subset=["cal_date"], keep="last")
            new_cal = new_cal.sort_values("cal_date").reset_index(drop=True)
        storage.save_raw(new_cal, "trade_cal", is_force=True)
        logger.info(f"交易日历已保存: {len(new_cal)} 条")
        trade_cal = new_cal

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