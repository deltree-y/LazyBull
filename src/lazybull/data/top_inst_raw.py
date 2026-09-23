# -*- coding: utf-8 -*-
"""top_inst（龙虎榜机构席位）raw 层：按 `trade_date` 年分区加载与去重防御。

数据契约（探索期下载，2026-09-23；见 `docs/top_inst_factor_health.md` §2 与
`docs/plans/top_inst_factor_plan.md`）：

- 存储：`data/raw/top_inst/YYYY-12-31.parquet`（按 **trade_date** 年分区，沿既有年分区模式）；
- 采集协议：逐交易日 + 单页 `limit=2000` + `offset` 翻页读到空（实测单日峰值 4,559 行 > 单页上限，
  不翻页必丢）；全字段去重；
- **raw 层只做日期规范化 + 全字段去重（防御）**：数据源的结构缺陷（`buy/sell` 列互换 /
  「席位 × side × reason」笛卡尔展开 / 新旧文本格式重复）统一由
  `factors/top_inst.py::clean_top_inst` 在**因子层**清洗——清洗口径是因子语义的一部分，
  不下沉到 raw 层（raw 保留源数据原貌，便于复盘与重洗）。
"""

from typing import List, Optional, Sequence

import pandas as pd
from loguru import logger

from .storage import Storage

#: raw 数据集名（目录与文件前缀）
TOP_INST_RAW_NAME = "top_inst"


def deduplicate_top_inst(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """日期规范化 + **全字段**去重（下载侧已做；加载侧防御一次，幂等）。

    注意：**不做** (ts_code, trade_date, exalter, side) 级去重——多榜单展开与新老文本格式
    重复的识别依赖金额锚，属于因子清洗口径（`clean_top_inst`），raw 层禁止擅自聚合。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    work = df.copy()
    if "trade_date" in work.columns:
        text = work["trade_date"].astype(str).str.replace("-", "", regex=False).str.strip()
        work["trade_date"] = text.where(text.str.match(r"^\d{8}$", na=False), pd.NA)
        illegal = int(work["trade_date"].isna().sum())
        if illegal:
            logger.warning(f"[{TOP_INST_RAW_NAME}] 剔除 {illegal} 行非法/缺失 trade_date")
            work = work[work["trade_date"].notna()]
    before = len(work)
    work = work.drop_duplicates(ignore_index=True)
    if before != len(work):
        logger.debug(f"[{TOP_INST_RAW_NAME}] 加载侧全字段去重移除 {before - len(work)} 行")
    if "trade_date" in work.columns:
        work = work.sort_values(
            [c for c in ("trade_date", "ts_code", "exalter", "side") if c in work.columns],
            kind="stable",
        ).reset_index(drop=True)
    return work


def existing_top_inst_df(storage: Storage) -> Optional[pd.DataFrame]:
    """枚举年分区加载已有数据（分区数据集不能用 `load_raw` 单文件读取）。"""
    frames: List[pd.DataFrame] = []
    for partition in storage.list_partitions("raw", TOP_INST_RAW_NAME):
        df = storage.load_raw_by_date(TOP_INST_RAW_NAME, partition)
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        return None
    return deduplicate_top_inst(pd.concat(frames, ignore_index=True))


def load_top_inst(
    storage: Storage, years: Optional[Sequence[str]] = None
) -> Optional[pd.DataFrame]:
    """加载 top_inst（默认全部年份；可指定 `years=['2020','2021']`）。"""
    if years is not None:
        frames: List[pd.DataFrame] = []
        for year in years:
            partition = f"{year}-12-31"
            df = storage.load_raw_by_date(TOP_INST_RAW_NAME, partition)
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return None
        return deduplicate_top_inst(pd.concat(frames, ignore_index=True))
    return existing_top_inst_df(storage)
