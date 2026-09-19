# -*- coding: utf-8 -*-
"""raw_download 子包：十大流通股东（top10_floatholders）下载薄包装。

核心下载逻辑在 `src/lazybull/data/top10_floatholders_raw.py`：
按**报告期**（季末）分页读满（单页 6000）→ 全字段去重 → 按 `end_date` 年份分区落盘 →
非法 `ann_date`/`end_date` 剔除告警。增量刷新回拉最近 2 个已存报告期（审计 §5.1）。
"""

from typing import Optional

from loguru import logger

from src.lazybull.data import Storage, TushareClient
from src.lazybull.data.top10_floatholders_raw import download_top10fh


def download_top10fh_dataset(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载十大流通股东数据（按 `end_date` 年分区落盘，增量续传）。

    Args:
        client: TuShare 客户端
        storage: Storage 实例
        start_date: 起始日期 YYYYMMDD（增量模式下会被已有水位收窄到报告期粒度）
        end_date: 结束日期 YYYYMMDD
        force: 是否强制全量重下（忽略水位）
    """
    summary: Optional[dict] = download_top10fh(
        client, storage, start_date=start_date, end_date=end_date, force=force
    )
    logger.info(
        "[top10_floatholders] 下载摘要: "
        + ", ".join(f"{key}={value}" for key, value in (summary or {}).items())
    )
