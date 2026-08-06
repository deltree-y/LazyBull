# -*- coding: utf-8 -*-
"""FeatureBuilder 因子 mixin：高级因子/风控因子/窗口/价值红利/资金流/过滤委托。"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .static_core import _calculate_window_features_static
from .static_extra import (
    _add_advanced_factors_static,
    _add_filter_flags_static,
    _add_limit_flags_static,
    _add_moneyflow_features_static,
    _add_value_dividend_features_static,
    _apply_filters_static,
    _attach_risk_factors_static,
)


class FeatureFactorsMixin:
    """FeatureBuilder 因子 mixin。"""

    def _add_advanced_factors(
        self,
        features: pd.DataFrame,
        current_data: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int,
    ) -> pd.DataFrame:
        return _add_advanced_factors_static(
            features=features,
            current_data=current_data,
            trade_date=trade_date,
            trading_dates=trading_dates,
            current_idx=current_idx,
            lookback_windows=self.lookback_windows,
            tech_factor_cache_dict=self._tech_factor_cache_dict,
            _get_tech_factor_today_fn=lambda td: self._get_tech_factor_today(
                daily_adj, td, trading_dates
            ),
        )

    def _get_risk_factor_cache(self, daily_adj: pd.DataFrame) -> Optional[Dict[str, pd.DataFrame]]:
        """获取（首次调用时构建）风控因子批量预计算缓存。

        22 个基于 daily_adj 历史窗口的风控因子在首次调用时对整个周期
        一次性向量化计算并缓存（与技术因子缓存模式一致），之后每日 O(1) 查表。
        构建失败时返回 None，调用方回退到逐日滑窗路径。
        """
        if self._risk_factor_cache_dict is not None:
            return self._risk_factor_cache_dict
        if self._risk_cache_failed:
            return None
        try:
            from ...risk.precompute import (
                PRECOMPUTED_RISK_FACTOR_NAMES,
                build_risk_factor_cache_dict,
                precompute_risk_factors,
            )
        except ImportError:
            logger.debug("风控因子预计算模块不可用，回退逐日计算路径")
            self._risk_cache_failed = True
            return None
        try:
            logger.info("首次构建：批量预计算风控因子（缓存中）...")
            long_df = precompute_risk_factors(daily_adj)
            if long_df is None or len(long_df) == 0:
                logger.warning("风控因子批量预计算返回空，回退逐日计算路径")
                self._risk_cache_failed = True
                return None
            self._risk_factor_cache_dict = build_risk_factor_cache_dict(long_df)
            self._risk_factor_names = list(PRECOMPUTED_RISK_FACTOR_NAMES)
            return self._risk_factor_cache_dict
        except (ValueError, KeyError, TypeError, MemoryError) as e:
            logger.warning(f"风控因子批量预计算失败（{type(e).__name__}: {e}），回退逐日计算路径")
            self._risk_cache_failed = True
            return None

    def _add_risk_factors(
        self,
        features: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
    ) -> pd.DataFrame:
        """计算风控模型专属因子（下行风险/波动结构/流动性/公告类）。

        优先路径：22 个历史窗口因子读取批量预计算缓存（O(1) 查表合并），
        9 个公告类截面因子仍逐日计算（无历史窗口，开销极小）。
        预计算不可用时回退到旧的逐日滑窗计算路径。
        """
        try:
            from ...factors.risk.factor_registry import compute_all_risk_factors
        except ImportError:
            logger.debug("风控因子模块不可用，跳过")
            return features

        # 优先路径：批量预计算缓存查表
        cache_dict = self._get_risk_factor_cache(daily_adj)
        if cache_dict is not None and self._risk_factor_names:
            return _attach_risk_factors_static(
                features, trade_date, cache_dict, self._risk_factor_names
            )

        # 回退路径：切片到最长风控窗口（252 交易日 + 余量）逐日计算
        sliced = self._slice_by_trading_days(daily_adj, trading_dates, trade_date, warmup_days=260)
        if sliced is None or len(sliced) == 0:
            return features

        # 风控因子需要日收益列 ret_1（从后复权价格衍生）
        if "ret_1" not in sliced.columns and "pre_close_adj" in sliced.columns:
            sliced = sliced.copy()
            sliced["ret_1"] = sliced["close_adj"] / sliced["pre_close_adj"] - 1

        risk_cols = compute_all_risk_factors(
            df=features,
            daily_adj=sliced,
            market_state=None,
            trade_date=trade_date,
        )
        if risk_cols:
            new_df = pd.DataFrame(
                {
                    name: (s.values if isinstance(s, pd.Series) else s)
                    for name, s in risk_cols.items()
                },
                index=features.index,
            )
            features = pd.concat([features, new_df], axis=1)
            logger.debug(f"已添加 {len(risk_cols)} 个风控因子")
        return features

    def _calculate_window_features(
        self, hist_data: pd.DataFrame, current_data: pd.DataFrame, window: int
    ) -> pd.DataFrame:
        return _calculate_window_features_static(hist_data, current_data, window)

    def _add_value_dividend_features(
        self,
        features: pd.DataFrame,
        daily_basic_data: pd.DataFrame,
        trade_date: str,
    ) -> pd.DataFrame:
        return _add_value_dividend_features_static(features, daily_basic_data, trade_date)

    def _add_moneyflow_features(
        self,
        features: pd.DataFrame,
        moneyflow_data: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int,
    ) -> pd.DataFrame:
        return _add_moneyflow_features_static(
            features=features,
            moneyflow_data=moneyflow_data,
            trade_date=trade_date,
            trading_dates=trading_dates,
            current_idx=current_idx,
            trading_date_index=self._trading_date_index,
        )

    def _add_filter_flags(
        self,
        df: pd.DataFrame,
        stock_basic: pd.DataFrame,
        suspend_info: Optional[pd.DataFrame],
        trade_date: str,
    ) -> pd.DataFrame:
        return _add_filter_flags_static(df, stock_basic, suspend_info, trade_date)

    def _add_limit_flags(
        self,
        df: pd.DataFrame,
        daily_data: pd.DataFrame,
        limit_info: Optional[pd.DataFrame],
        trade_date: str,
    ) -> pd.DataFrame:
        return _add_limit_flags_static(df, daily_data, limit_info, trade_date)

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        result = _apply_filters_static(
            df,
            require_label=self.require_label,
            label_filter_mode=self.label_filter_mode,
            horizon=self.horizon,
            horizons=self.horizons,
            min_list_days=self.min_list_days,
        )
        return result