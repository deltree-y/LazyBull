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

    市场择时仓位管理（可选）：
    当 market_regime_enabled=True 时，在每个调仓日根据全市场状态动态调整仓位：
    - 熊市（mkt_ret_avg_20 < bear_threshold）：仓位降至 bear_exposure
    - 正常/牛市：保持满仓
    """

    def __init__(
        self,
        features_by_date: Dict[str, pd.DataFrame],
        market_regime_enabled: bool = False,
        market_regime_bear_threshold: float = -0.02,
        market_regime_bear_exposure: float = 0.3,
        **kwargs,
    ):
        """初始化 ML 回测引擎

        Args:
            features_by_date: 按日期组织的特征数据字典，键为日期字符串（YYYYMMDD），值为特征 DataFrame
            market_regime_enabled: 是否启用市场择时仓位管理，默认 False
            market_regime_bear_threshold: mkt_ret_avg_20 低于此值判定为熊市，默认 -0.02
            market_regime_bear_exposure: 熊市仓位系数（0~1），默认 0.3
            **kwargs: 其他参数传递给父类 BacktestEngine
        """
        super().__init__(**kwargs)
        self.features_by_date = features_by_date
        self.market_regime_enabled = market_regime_enabled
        self.market_regime_bear_threshold = market_regime_bear_threshold
        self.market_regime_bear_exposure = market_regime_bear_exposure

        regime_info = ""
        if market_regime_enabled:
            regime_info = (
                f", 市场择时=开启(bear_threshold={market_regime_bear_threshold}, "
                f"bear_exposure={market_regime_bear_exposure})"
            )
        logger.info(f"ML 回测引擎初始化: 特征数据覆盖 {len(features_by_date)} 个交易日{regime_info}")
    
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

    # ── 市场择时仓位管理 ──────────────────────────────────────────────

    def _get_market_regime_exposure(self, date: pd.Timestamp) -> float:
        """根据市场状态计算仓位系数

        利用 features_by_date 中已有的 mkt_ret_avg_20（过去 20 日全市场平均
        收益之和）和 mkt_adv_dec_ratio（60 日涨跌比均值）来判断熊市。

        Returns:
            仓位系数，1.0 = 满仓，< 1.0 = 降仓
        """
        date_str = date.strftime('%Y%m%d')
        features_df = self.features_by_date.get(date_str)
        if features_df is None or len(features_df) == 0:
            return 1.0

        # mkt_ret_avg_20 是广播到所有股票的同一值，取首行即可
        mkt_ret = np.nan
        if 'mkt_ret_avg_20' in features_df.columns:
            mkt_ret = features_df['mkt_ret_avg_20'].iloc[0]

        if np.isnan(mkt_ret):
            return 1.0

        if mkt_ret < self.market_regime_bear_threshold:
            return self.market_regime_bear_exposure

        return 1.0

    def _execute_pending_buys(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """执行买入前应用市场择时仓位缩放

        在父类执行买入之前，将 pending_signals 中的权重乘以市场仓位系数。
        原理与 ECT（权益曲线交易）相同：权重之和 < 1 → 剩余资金留作现金。
        """
        if self.market_regime_enabled:
            # 找到前一个交易日的信号（与父类逻辑一致）
            current_idx = date_to_idx.get(date)
            if current_idx is not None and current_idx > 0:
                signal_date = trading_dates[current_idx - 1]
                signal_data = self.pending_signals.get(signal_date)

                if signal_data is not None:
                    exposure = self._get_market_regime_exposure(signal_date)
                    if exposure < 1.0:
                        # 缩放信号权重
                        if isinstance(signal_data, dict) and 'signals' in signal_data:
                            signal_data['signals'] = {
                                stock: w * exposure
                                for stock, w in signal_data['signals'].items()
                            }
                        elif isinstance(signal_data, dict):
                            self.pending_signals[signal_date] = {
                                stock: w * exposure
                                for stock, w in signal_data.items()
                            }
                        if self.verbose:
                            logger.info(
                                f"  市场择时: {date.date()}, "
                                f"mkt_regime_exposure={exposure:.2f}, 仓位降至 {exposure*100:.0f}%"
                            )

        # 调用父类完成实际买入
        super()._execute_pending_buys(date, trading_dates, date_to_idx)
