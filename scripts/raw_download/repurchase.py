# -*- coding: utf-8 -*-
"""raw_download 子包：股票回购（repurchase）下载薄包装。

核心下载逻辑在 `src/lazybull/data/repurchase_raw.py`（与纸面 ensure 共享）：
按月窗口分页读满（单页 2000）→ 全字段去重 → 按 `ann_date` 年份分区落盘 → 非法公告日剔除告警。
"""

from typing import Optional

from loguru import logger

from src.lazybull.data import Storage, TushareClient
from src.lazybull.data.repurchase_raw import download_repurchase


def download_repurchase_dataset(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载股票回购数据（按 ann_date 年分区落盘，增量续传）。

    Args:
        client: TuShare 客户端
        storage: Storage 实例
        start_date: 起始日期 YYYYMMDD（增量模式下会被已有水位收窄）
        end_date: 结束日期 YYYYMMDD
        force: 是否强制全量重下（忽略水位）
    """
    summary: Optional[dict] = download_repurchase(
        client, storage, start_date=start_date, end_date=end_date, force=force
    )
    logger.info(
        "[repurchase] 下载摘要: "
        + ", ".join(f"{key}={value}" for key, value in (summary or {}).items())
    )
