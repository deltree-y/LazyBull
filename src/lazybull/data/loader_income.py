# -*- coding: utf-8 -*-
"""利润表季度分区加载。"""

from typing import Optional

import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd


class IncomeLoaderMixin:
    """为 DataLoader 提供利润表加载能力。"""

    def load_income(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: int = 6,
    ) -> Optional[pd.DataFrame]:
        """加载 income 季度分区，默认覆盖分红政策五年窗口及前序财年。"""
        df = self._load_quarter_partitioned_raw(
            "income",
            start_date=start_date,
            end_date=end_date,
            lookback_years=lookback_years,
        )
        if df is None:
            df = self.storage.load_raw("income")
        if df is None:
            logger.warning("未找到利润表数据")
        else:
            for col in ("ann_date", "f_ann_date", "end_date"):
                if col in df.columns:
                    df[col] = normalize_series_to_yyyymmdd(df[col])
        return df
