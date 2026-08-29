# -*- coding: utf-8 -*-
"""raw_download 子包：分红送股（dividend）下载薄包装。

核心下载逻辑在 src/lazybull/data/dividend_raw.py（与纸面 ensure 共享）。
"""

from typing import Optional

import pandas as pd
from loguru import logger

from src.lazybull.data import Storage, TushareClient
from src.lazybull.data.dividend_raw import download_dividend_full


def download_dividend(
    client: TushareClient,
    storage: Storage,
    stock_basic: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False,
    concurrency: Optional[int] = None,
) -> None:
    """下载分红送股数据（按股全历史查询 + ann_date 年分区落盘）。

    start_date/end_date 仅作日志展示；dividend 接口不支持区间查询，
    实际下载以 ts_code 为粒度覆盖全历史。
    """
    logger.info(f"[dividend] 下载区间 {start_date}~{end_date}（按股票全历史）")
    download_dividend_full(
        client,
        storage,
        stock_basic,
        concurrency=concurrency,
        force=force,
    )
    logger.info("[dividend] 完成")
