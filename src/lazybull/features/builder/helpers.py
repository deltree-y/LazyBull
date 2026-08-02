# -*- coding: utf-8 -*-
"""FeatureBuilder 工具 mixin：交易日/回看/复权/技术因子缓存/市场状态/中性化委托。"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ...factors import precompute_technical_factors
from .static_core import _backfill_fundamental_proxy_features_static
from .static_extra import _add_new_individual_features_static

_WARMUP_TRADING_DAYS = 120


class FeatureHelpersMixin:
    """FeatureBuilder 工具 mixin。"""

    @staticmethod
    def _backfill_fundamental_proxy_features(features: pd.DataFrame) -> pd.DataFrame:
        """用可稳定获取的字段回填基本面代理列（cf_sales、cf_nm）。"""
        return _backfill_fundamental_proxy_features_static(features)

    def _get_trading_dates(self, trade_cal: pd.DataFrame) -> List[str]:
        if self._trading_dates_cache is not None:
            return self._trading_dates_cache
        if "cal_date" in trade_cal.columns:
            if pd.api.types.is_datetime64_any_dtype(trade_cal["cal_date"]):
                trade_cal = trade_cal.copy()
                trade_cal["cal_date"] = trade_cal["cal_date"].dt.strftime("%Y%m%d")
            trading_dates = trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()
        else:
            logger.error("交易日历缺少 cal_date 字段")
            return []
        self._trading_dates_cache = sorted(trading_dates)
        self._trading_date_index = {d: i for i, d in enumerate(self._trading_dates_cache)}
        return self._trading_dates_cache

    def _get_lookback_dates(self, trade_date: str, n: int, trading_dates: List[str]) -> List[str]:
        if self._trading_date_index is not None:
            idx = self._trading_date_index.get(trade_date, -1)
            if idx == -1:
                return []
        else:
            if trade_date not in trading_dates:
                return []
            idx = trading_dates.index(trade_date)
        if idx < n:
            return []
        return trading_dates[idx - n : idx]

    def _slice_by_trading_days(
        self,
        daily_df: pd.DataFrame,
        trading_dates: List[str],
        anchor_trade_date: str,
        warmup_days: int = _WARMUP_TRADING_DAYS,
    ) -> pd.DataFrame:
        if daily_df is None or len(daily_df) == 0:
            return daily_df
        if anchor_trade_date not in trading_dates:
            return daily_df
        anchor_idx = trading_dates.index(anchor_trade_date)
        warmup_start_idx = max(0, anchor_idx - warmup_days)
        window_dates = set(trading_dates[warmup_start_idx:])
        return daily_df[daily_df["trade_date"].isin(window_dates)]

    def _calculate_adj_close(
        self, daily_data: pd.DataFrame, adj_factor: pd.DataFrame
    ) -> pd.DataFrame:
        daily_adj = daily_data.copy()
        if "close_adj" in daily_adj.columns:
            logger.info("数据已包含复权价格列，跳过复权计算")
            return daily_adj
        if pd.api.types.is_datetime64_any_dtype(daily_adj["trade_date"]):
            daily_adj["trade_date"] = daily_adj["trade_date"].dt.strftime("%Y%m%d")
        if pd.api.types.is_datetime64_any_dtype(adj_factor["trade_date"]):
            adj_factor = adj_factor.copy()
            adj_factor["trade_date"] = adj_factor["trade_date"].dt.strftime("%Y%m%d")
        daily_adj = daily_adj.merge(
            adj_factor[["ts_code", "trade_date", "adj_factor"]],
            on=["ts_code", "trade_date"],
            how="left",
        )
        daily_adj["adj_factor"] = pd.to_numeric(daily_adj["adj_factor"], errors="coerce")
        daily_adj = daily_adj.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        daily_adj["adj_factor"] = daily_adj.groupby("ts_code")["adj_factor"].ffill().bfill()
        daily_adj["close_adj"] = daily_adj["close"] * daily_adj["adj_factor"]
        if "open" in daily_adj.columns:
            daily_adj["open_adj"] = daily_adj["open"] * daily_adj["adj_factor"]
        if "high" in daily_adj.columns:
            daily_adj["high_adj"] = daily_adj["high"] * daily_adj["adj_factor"]
        if "low" in daily_adj.columns:
            daily_adj["low_adj"] = daily_adj["low"] * daily_adj["adj_factor"]
        missing_adj = daily_adj["adj_factor"].isna().sum()
        if missing_adj > 0:
            missing_codes = daily_adj.loc[daily_adj["adj_factor"].isna(), "ts_code"].unique()
            logger.warning(
                f"有 {missing_adj} 条记录缺少复权因子（涉及 {len(missing_codes)} 只股票），"
                "对应复权价保持为空，避免收益污染"
            )
        return daily_adj

    def _get_tech_factor_today(
        self,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if self._tech_factor_cache is None:
            logger.info("首次构建：批量预计算技术指标与波动率因子（缓存中）...")
            if trading_dates is not None:
                daily_adj_for_cache = self._slice_by_trading_days(
                    daily_adj, trading_dates, trade_date
                )
            else:
                daily_adj_for_cache = daily_adj
            self._tech_factor_cache = precompute_technical_factors(
                daily_adj=daily_adj_for_cache, vol_windows=self.lookback_windows
            )
            if (
                self._daily_adj_dict is not None
                and self._tech_factor_cache is not None
                and len(self._tech_factor_cache) > 0
            ):
                self._tech_factor_cache_dict = {
                    d: sub_df.reset_index(drop=True)
                    for d, sub_df in self._tech_factor_cache.groupby("trade_date", sort=False)
                }

        if self._tech_factor_cache is None or len(self._tech_factor_cache) == 0:
            return pd.DataFrame(columns=["ts_code", "trade_date"])

        if self._tech_factor_cache_dict is not None:
            return self._tech_factor_cache_dict.get(
                trade_date, pd.DataFrame(columns=["ts_code", "trade_date"])
            )
        return self._tech_factor_cache[self._tech_factor_cache["trade_date"] == trade_date]

    def _calculate_forward_returns(
        self,
        current_data: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int,
    ) -> pd.DataFrame:
        """已移出至 labels.py，保留委托以兼容直接调用的测试代码。"""
        from ..labels import compute_forward_returns

        return compute_forward_returns(
            current_data=current_data,
            trade_date=trade_date,
            trading_dates=trading_dates,
            current_idx=current_idx,
            horizons=self.horizons,
            daily_adj_dict=self._daily_adj_dict,
            daily_adj=daily_adj,
        )

    def _merge_shenwan_industry(
        self, features: pd.DataFrame, shenwan_industry: pd.DataFrame
    ) -> pd.DataFrame:
        """已移出至 industry_merge.py，保留委托。"""
        from ..industry_merge import merge_shenwan_industry

        return merge_shenwan_industry(features, shenwan_industry, self.shenwan_level, self.verbose)

    def _apply_industry_neutralization(self, features: pd.DataFrame) -> pd.DataFrame:
        """已移出至 neutralization.py，保留委托。"""
        from ..neutralization import apply_industry_neutralization

        return apply_industry_neutralization(
            features, self.horizons, self.lookback_windows, self.shenwan_level
        )

    def _apply_size_neutralization(
        self, result: pd.DataFrame, n_size_groups: int = 10
    ) -> pd.DataFrame:
        """已移出至 neutralization.py，保留委托。"""
        from ..neutralization import apply_size_neutralization

        return apply_size_neutralization(result, n_size_groups=n_size_groups)

    def _add_new_individual_features(self, result: pd.DataFrame) -> pd.DataFrame:
        """已改为模块级函数，保留委托。"""
        return _add_new_individual_features_static(result)

    def _add_market_state_features(
        self,
        result: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: list,
        current_idx: int,
        daily_basic_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        from ..market_state import add_market_state_features, precompute_market_state_cache

        if self._market_state_cache is None:
            self._market_state_cache = precompute_market_state_cache(
                daily_adj=daily_adj,
                trading_dates=trading_dates,
                trade_date=trade_date,
                daily_basic_data=daily_basic_data,
                tech_factor_cache=self._tech_factor_cache,
            )

        return add_market_state_features(
            result=result,
            daily_adj=daily_adj,
            trade_date=trade_date,
            trading_dates=trading_dates,
            current_idx=current_idx,
            daily_basic_data=daily_basic_data,
            market_state_cache=self._market_state_cache,
        )