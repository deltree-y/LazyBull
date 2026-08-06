"""风控公告类数据加载 Mixin（pledge_stat / share_float / block_trade）。

与 `DataLoader` 组合使用（`loader.py` 的 `class DataLoader(AnnouncementRiskLoaderMixin)`），
避免向已较大的 loader.py 继续追加方法，保持"新增功能对应新增文件"。

各数据集分区存储约定：
  - pledge_stat : 季分区（end_date），对齐 fina_indicator
  - share_float : 年分区（ann_date），对齐 report_rc（PIT 按公告日）
  - block_trade : 日分区（trade_date），对齐 margin_detail
"""

from typing import Optional

import pandas as pd

from ..common.date_utils import normalize_series_to_yyyymmdd


class AnnouncementRiskLoaderMixin:
    """为 DataLoader 提供质押/解禁/大宗三类原始数据加载方法。"""

    # ── pledge_stat：季分区 ─────────────────────────────────

    def load_pledge_stat(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """加载股权质押统计（按 end_date 季分区）。

        Args:
            start_date: 开始日期 YYYYMMDD（可选）
            end_date: 结束日期 YYYYMMDD（可选）

        Returns:
            质押统计 DataFrame（ts_code, end_date, pledge_ratio 等），无数据返回 None
        """
        df = self.storage.load_raw_by_date_range("pledge_stat", start_date, end_date) \
            if start_date and end_date else self.storage.load_raw("pledge_stat")
        if df is None:
            return None
        if "end_date" in df.columns:
            df["end_date"] = normalize_series_to_yyyymmdd(df["end_date"])
        return df

    # ── share_float：年分区（PIT 按 ann_date）────────────────

    def load_share_float(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """加载限售解禁数据（按 ann_date 年分区）。

        按公告日（ann_date）过滤，保证 PIT：T 日仅能看到 ann_date <= T 的解禁公告，
        float_date 为未来解禁日（用于计算 days_to_unlock）。

        Args:
            start_date: 开始日期 YYYYMMDD（可选）
            end_date: 结束日期 YYYYMMDD（可选）

        Returns:
            限售解禁 DataFrame（ts_code, ann_date, float_date, float_share, float_ratio 等）
        """
        df = self.storage.load_raw_by_date_range("share_float", start_date, end_date) \
            if start_date and end_date else self.storage.load_raw("share_float")
        if df is None:
            return None
        for col in ("ann_date", "float_date"):
            if col in df.columns:
                df[col] = normalize_series_to_yyyymmdd(df[col])
        return df

    # ── block_trade：日分区 ──────────────────────────────────

    def load_block_trade(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """加载大宗交易数据（按 trade_date 日分区）。

        Args:
            start_date: 开始日期 YYYYMMDD（可选）
            end_date: 结束日期 YYYYMMDD（可选）

        Returns:
            大宗交易 DataFrame（trade_date, ts_code, price, vol, amount 等）
        """
        df = self.storage.load_raw_by_date_range("block_trade", start_date, end_date) \
            if start_date and end_date else self.storage.load_raw("block_trade")
        if df is None:
            return None
        if "trade_date" in df.columns:
            df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
        return df
