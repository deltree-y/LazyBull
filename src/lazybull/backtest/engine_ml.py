"""ML 回测引擎

扩展 BacktestEngine 以支持 ML 信号的特征数据注入
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import to_trade_date_str
from .engine import BacktestEngine


class BacktestEngineML(BacktestEngine):
    """支持 ML 信号的回测引擎

    通过重写 _build_signal_data 方法注入特征数据，
    其他回测逻辑（信号过滤、回填、权重归一化等）复用父类实现。
    """

    def __init__(
        self,
        features_by_date: Dict[str, pd.DataFrame],
        **kwargs,
    ):
        """初始化 ML 回测引擎

        Args:
            features_by_date: 按日期组织的特征数据字典，键为日期字符串（YYYYMMDD），值为特征 DataFrame
            **kwargs: 其他参数传递给父类 BacktestEngine
        """
        super().__init__(**kwargs)
        self.features_by_date = features_by_date

        logger.info(
            f"ML 回测引擎初始化: 特征数据覆盖 {len(features_by_date)} 个交易日"
        )

    def _build_signal_data(self, date: pd.Timestamp) -> Optional[Dict]:
        """构建信号数据（注入 ML 特征）

        从 features_by_date 中获取当日特征数据。

        Args:
            date: 信号生成日期

        Returns:
            包含 "features" 键的数据字典，如果当日无特征数据则返回 None
        """
        # 转换日期格式
        date_str = date.strftime('%Y%m%d')

        # 获取特征数据
        features_df = self.features_by_date.get(date_str)

        if features_df is None or len(features_df) == 0:
            # 无特征数据，返回 None 让父类跳过该日期
            logger.warning(f"信号日 {date.date()} 没有特征数据，跳过")
            return None

        # 返回特征数据字典
        return {"features": features_df}

    # ── 候选过滤（已移除行业动量过滤 & 行业轮动加权）──────────────────

    def _post_filter_candidates(
        self, ranked_candidates: list, date: pd.Timestamp
    ) -> list:
        """对候选列表做行业维度的过滤和/或加权（当前为透传，无操作）。"""
        return ranked_candidates


    def _get_holding_features_row(
        self, date: pd.Timestamp, stock: str
    ) -> Optional[pd.Series]:
        """覆写基类 hook:从 features_by_date 读取持仓股票的截面特征行

        缺失时返回 None,scorer 会降级到中位分。
        """
        date_str = date.strftime('%Y%m%d')
        features_df = self.features_by_date.get(date_str)
        if features_df is None or features_df.empty:
            return None
        mask = features_df['ts_code'] == stock
        if not mask.any():
            return None
        return features_df.loc[mask].iloc[0]

    def _get_current_position_atr_stats(
        self, date: pd.Timestamp
    ) -> Optional[tuple[float, float, float]]:
        """获取当日持仓股票 atr_pct_14 的最小值、均值和最大值。"""
        if not self.positions:
            return None

        date_str = date.strftime('%Y%m%d')
        features_df = self.features_by_date.get(date_str)
        if features_df is None or 'atr_pct_14' not in features_df.columns:
            return None

        position_codes = set(self.positions.keys())
        atr_series = features_df.loc[
            features_df['ts_code'].isin(position_codes), 'atr_pct_14'
        ].dropna()
        atr_series = atr_series[atr_series > 0]
        if atr_series.empty:
            return None

        return (
            float(atr_series.min()),
            float(atr_series.mean()),
            float(atr_series.max()),
        )
