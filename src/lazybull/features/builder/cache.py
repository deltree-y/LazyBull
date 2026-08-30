# -*- coding: utf-8 -*-
"""FeatureBuilder 缓存管理 mixin：缓存槽位/失效/复权预计算/因子注册表。"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ...common.config import get_shenwan_level, normalize_shenwan_level


class FeatureCacheMixin:
    """FeatureBuilder 缓存管理 mixin。"""

    def __init__(
        self,
        min_list_days: int = 365,
        horizon: Optional[int] = None,
        horizons: Optional[List[int]] = None,
        lookback_windows: Optional[List[int]] = None,
        require_label: bool = True,
        label_filter_mode: str = "all",
        shenwan_level: Optional[str] = None,
        verbose: bool = False,
    ):
        if label_filter_mode not in ("single", "all"):
            raise ValueError(
                f"label_filter_mode 必须是 'single' 或 'all'，传入：{label_filter_mode}"
            )
        self.min_list_days = min_list_days
        self.horizons = horizons or [5, 10, 20]
        if horizon is not None and horizon not in self.horizons:
            logger.warning(f"传入 horizon={horizon} 不在 horizons={self.horizons} 中，自动追加")
            self.horizons = sorted(set(self.horizons) | {horizon})
        self.horizon = horizon if horizon is not None else self.horizons[0]
        self.lookback_windows = lookback_windows or [5, 10, 20]
        self.require_label = require_label
        self.label_filter_mode = label_filter_mode
        self.shenwan_level = (
            normalize_shenwan_level(shenwan_level)
            if shenwan_level is not None
            else get_shenwan_level()
        )
        self.verbose = verbose

        # 缓存槽位
        self._market_state_cache: Optional[pd.DataFrame] = None
        self._tech_factor_cache: Optional[pd.DataFrame] = None
        self._tech_factor_cache_dict: Optional[Dict[str, pd.DataFrame]] = None
        self._trading_dates_cache: Optional[List[str]] = None
        self._trading_date_index: Optional[Dict[str, int]] = None
        self._daily_adj_precomputed: Optional[pd.DataFrame] = None
        self._daily_adj_dict: Optional[Dict[str, pd.DataFrame]] = None
        self._risk_factor_cache_dict: Optional[Dict[str, pd.DataFrame]] = None
        self._risk_factor_names: Optional[List[str]] = None
        self._risk_cache_failed: bool = False
        self._factor_registry = None  # 延迟初始化

        # 每日摘要用
        self._filter_stats: Dict = {}
        self._last_summary: str = ""

        if self.verbose:
            logger.info(
                f"特征构建器初始化: min_list_days={min_list_days}, "
                f"horizons={self.horizons}, lookback_windows={self.lookback_windows}, "
                f"require_label={require_label}, shenwan_level={self.shenwan_level}"
            )

    def clear_caches(self) -> None:
        cache_names = [
            "_market_state_cache",
            "_tech_factor_cache",
            "_tech_factor_cache_dict",
            "_trading_dates_cache",
            "_trading_date_index",
            "_daily_adj_precomputed",
            "_daily_adj_dict",
            "_risk_factor_cache_dict",
            "_risk_factor_names",
        ]
        cleared = []
        for name in cache_names:
            if getattr(self, name, None) is not None:
                setattr(self, name, None)
                cleared.append(name)
        self._risk_cache_failed = False
        if cleared:
            logger.debug(f"FeatureBuilder 缓存已释放: {', '.join(cleared)}")

    def _invalidate_precomputed_state(self) -> None:
        cache_names = [
            "_market_state_cache",
            "_tech_factor_cache",
            "_tech_factor_cache_dict",
            "_trading_dates_cache",
            "_trading_date_index",
            "_daily_adj_precomputed",
            "_daily_adj_dict",
            "_risk_factor_cache_dict",
            "_risk_factor_names",
        ]
        for name in cache_names:
            if getattr(self, name, None) is not None:
                setattr(self, name, None)
        self._risk_cache_failed = False

    def precompute_daily_adj(
        self,
        daily_data: pd.DataFrame,
        adj_factor: pd.DataFrame,
        daily_basic_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self._invalidate_precomputed_state()
        logger.info("预计算 daily_adj（含 pre_close_adj）并建立日期索引字典...")
        daily_adj = self._calculate_adj_close(daily_data, adj_factor)
        if (
            daily_basic_data is not None
            and len(daily_basic_data) > 0
            and {"ts_code", "trade_date", "turnover_rate"}.issubset(daily_basic_data.columns)
        ):
            turnover = (
                daily_basic_data[["ts_code", "trade_date", "turnover_rate"]]
                .drop_duplicates(["ts_code", "trade_date"], keep="last")
                .rename(columns={"turnover_rate": "_daily_basic_turnover_rate"})
            )
            daily_adj = daily_adj.merge(
                turnover,
                on=["ts_code", "trade_date"],
                how="left",
                validate="many_to_one",
            )
            if "turnover_rate" in daily_adj.columns:
                daily_adj["turnover_rate"] = daily_adj["turnover_rate"].fillna(
                    daily_adj["_daily_basic_turnover_rate"]
                )
                daily_adj.drop(columns=["_daily_basic_turnover_rate"], inplace=True)
            else:
                daily_adj.rename(
                    columns={"_daily_basic_turnover_rate": "turnover_rate"}, inplace=True
                )
        daily_adj = daily_adj.sort_values(["ts_code", "trade_date"])
        daily_adj["pre_close_adj"] = daily_adj.groupby("ts_code")["close_adj"].shift(1)
        self._daily_adj_precomputed = daily_adj
        self._daily_adj_dict = {
            d: sub_df.reset_index(drop=True)
            for d, sub_df in daily_adj.groupby("trade_date", sort=False)
        }
        logger.info(
            f"daily_adj 预计算完成：{len(daily_adj)} 条记录，"
            f"{len(self._daily_adj_dict)} 个交易日"
        )

    def _get_factor_registry(self):
        if self._factor_registry is None:
            from ..factor_handlers import create_factor_registry

            self._factor_registry = create_factor_registry()
        return self._factor_registry
