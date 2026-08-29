# -*- coding: utf-8 -*-
"""分红送股（dividend）数据加载 Mixin。

与 `DataLoader` 组合使用（`loader.py` 的 `class DataLoader(DividendLoaderMixin, ...)`），
保持"新增功能对应新增文件"。

分区存储约定：
  - dividend : 年分区（ann_date），对齐 report_rc/share_float（公告型事件数据）
"""

import warnings
from typing import List, Optional

import pandas as pd

from ..common.date_utils import normalize_series_to_yyyymmdd

_CONCAT_ALL_NA_WARNING = (
    r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
)


def _concat_no_all_na_warning(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """原样合并 DataFrame，仅屏蔽 pandas 的 empty/all-NA FutureWarning。"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=_CONCAT_ALL_NA_WARNING)
        return pd.concat(frames, ignore_index=True)


class DividendLoaderMixin:
    """为 DataLoader 提供分红送股原始数据加载方法。"""

    def load_dividend(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """加载分红送股数据（按 ann_date 年分区）。

        与 report_rc 一致：分区模式下必须按分区枚举加载（`load_raw` 只读
        单文件，对分区目录返回 None）；范围查询不能走按日分区的
        `load_raw_by_date_range`（会漏掉年份分区 `YYYY-12-31` 文件），
        改为按年枚举分区后按 ann_date 列过滤。

        Args:
            start_date: 开始日期 YYYYMMDD（可选，按 ann_date 过滤）
            end_date: 结束日期 YYYYMMDD（可选，按 ann_date 过滤）

        Returns:
            分红送股 DataFrame（ts_code, ann_date, ex_date, imp_ann_date,
            end_date, div_proc, cash_div, cash_div_tax, stk_div 等），无数据返回 None
        """
        partitions = self.storage.list_partitions("raw", "dividend")
        if not partitions:
            return None
        if start_date and end_date:
            start_year = int(str(start_date)[:4])
            end_year = int(str(end_date)[:4])
            partitions = [
                p for p in partitions if len(p) == 10 and start_year <= int(p[:4]) <= end_year
            ]
            if not partitions:
                return None
        frames: List[pd.DataFrame] = []
        for p in partitions:
            part = self.storage.load_raw_by_date("dividend", p)
            if part is not None and len(part) > 0:
                frames.append(part)
        if not frames:
            return None
        df = _concat_no_all_na_warning(frames)
        for col in ("ann_date", "ex_date", "imp_ann_date", "record_date", "end_date"):
            if col in df.columns:
                df[col] = normalize_series_to_yyyymmdd(df[col])
        if start_date and end_date and "ann_date" in df.columns:
            df = df[(df["ann_date"] >= str(start_date)[:8]) & (df["ann_date"] <= str(end_date)[:8])]
        if len(df) == 0:
            return None
        return df
