"""特征构建编排器 —— 精简后的 FeatureBuilder。

将原有的 26 个方法拆分到独立模块（labels/industry_merge/neutralization/
market_state/factor_handlers/context/parallel），FeatureBuilder 现为轻量编排器。

保留：缓存管理、价格工具、窗口特征、过滤标记、高级因子编排。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.config import get_shenwan_level, normalize_shenwan_level
from ..common.date_utils import normalize_date_columns, to_trade_date_str
from ..factors import (
    calculate_amplitude,
    calculate_shadows,
    calculate_industry_alpha_windows,
    calculate_acceleration,
    calculate_volume_burst,
    precompute_technical_factors,
)
from ..factors.normalization import cross_sectional_zscore

_WARMUP_TRADING_DAYS = 120


class FeatureBuilder:
    """特征构建编排器。

    负责缓存管理、预计算调度、将单日特征构建委托给独立模块。
    保持与旧版 API 完全兼容。
    """

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

    # ── 缓存管理 ──────────────────────────────────────────────

    def clear_caches(self) -> None:
        cache_names = [
            "_market_state_cache", "_tech_factor_cache",
            "_tech_factor_cache_dict", "_trading_dates_cache",
            "_trading_date_index", "_daily_adj_precomputed",
            "_daily_adj_dict",
        ]
        cleared = []
        for name in cache_names:
            if getattr(self, name, None) is not None:
                setattr(self, name, None)
                cleared.append(name)
        if cleared:
            logger.debug(f"FeatureBuilder 缓存已释放: {', '.join(cleared)}")

    def _invalidate_precomputed_state(self) -> None:
        cache_names = [
            "_market_state_cache", "_tech_factor_cache",
            "_tech_factor_cache_dict", "_trading_dates_cache",
            "_trading_date_index", "_daily_adj_precomputed",
            "_daily_adj_dict",
        ]
        for name in cache_names:
            if getattr(self, name, None) is not None:
                setattr(self, name, None)

    # ── 预计算 ────────────────────────────────────────────────

    def precompute_daily_adj(self, daily_data: pd.DataFrame, adj_factor: pd.DataFrame) -> None:
        self._invalidate_precomputed_state()
        logger.info("预计算 daily_adj（含 pre_close_adj）并建立日期索引字典...")
        daily_adj = self._calculate_adj_close(daily_data, adj_factor)
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

    # ── 主入口 ────────────────────────────────────────────────

    def build_features_for_day(
        self,
        trade_date: str,
        trade_cal: pd.DataFrame,
        daily_data: pd.DataFrame,
        adj_factor: pd.DataFrame,
        stock_basic: pd.DataFrame,
        daily_basic_data: Optional[pd.DataFrame] = None,
        moneyflow_data: Optional[pd.DataFrame] = None,
        suspend_info: Optional[pd.DataFrame] = None,
        limit_info: Optional[pd.DataFrame] = None,
        shenwan_industry: Optional[pd.DataFrame] = None,
        apply_industry_neutralization: bool = False,
        apply_size_neutralization: bool = False,
        fundamental_data: Optional[pd.DataFrame] = None,
        margin_data: Optional[pd.DataFrame] = None,
        holder_data: Optional[pd.DataFrame] = None,
        earnings_data: Optional[pd.DataFrame] = None,
        cyq_perf_data: Optional[pd.DataFrame] = None,
        express_data: Optional[pd.DataFrame] = None,
        fund_portfolio_data: Optional[pd.DataFrame] = None,
        north_flow_data: Optional[Dict[str, float]] = None,
        lhb_data: Optional[pd.DataFrame] = None,
        consensus_data: Optional[pd.DataFrame] = None,
        cashflow_data: Optional[pd.DataFrame] = None,
        consensus_revision_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """构建单个交易日的截面特征和标签（签名保持向后兼容）。"""

        from .context import FeatureContext

        ctx = FeatureContext(
            trade_date=trade_date,
            trade_cal=trade_cal,
            daily_data=daily_data,
            adj_factor=adj_factor,
            stock_basic=stock_basic,
            daily_basic_data=daily_basic_data,
            moneyflow_data=moneyflow_data,
            suspend_info=suspend_info,
            limit_info=limit_info,
            shenwan_industry=shenwan_industry,
            apply_industry_neutralization=apply_industry_neutralization,
            apply_size_neutralization=apply_size_neutralization,
            fundamental_data=fundamental_data,
            margin_data=margin_data,
            holder_data=holder_data,
            earnings_data=earnings_data,
            cyq_perf_data=cyq_perf_data,
            express_data=express_data,
            fund_portfolio_data=fund_portfolio_data,
            north_flow_data=north_flow_data,
            lhb_data=lhb_data,
            consensus_data=consensus_data,
            cashflow_data=cashflow_data,
            consensus_revision_data=consensus_revision_data,
            horizons=self.horizons,
            horizon=self.horizon,
            lookback_windows=self.lookback_windows,
            require_label=self.require_label,
            label_filter_mode=self.label_filter_mode,
            min_list_days=self.min_list_days,
            shenwan_level=self.shenwan_level,
            verbose=self.verbose,
        )
        return self._build_features(ctx)

    def _build_features(self, ctx: "FeatureContext") -> pd.DataFrame:
        """内部编排逻辑。"""
        # 1. 交易日序列
        trading_dates = self._get_trading_dates(ctx.trade_cal)
        if self._trading_date_index is not None:
            current_idx = self._trading_date_index.get(ctx.trade_date, -1)
            if current_idx == -1:
                logger.warning(f"{ctx.trade_date} 不是交易日，跳过")
                return pd.DataFrame()
        else:
            if ctx.trade_date not in trading_dates:
                logger.warning(f"{ctx.trade_date} 不是交易日，跳过")
                return pd.DataFrame()
            current_idx = trading_dates.index(ctx.trade_date)

        # 2. 复权价格
        if self._daily_adj_precomputed is not None:
            daily_adj = self._daily_adj_precomputed
        else:
            daily_adj = self._calculate_adj_close(ctx.daily_data, ctx.adj_factor)
            daily_adj = daily_adj.sort_values(["ts_code", "trade_date"])
            daily_adj["pre_close_adj"] = daily_adj.groupby("ts_code")["close_adj"].shift(1)

        # 3. 当日截面（O(1) 字典查表优化）
        if self._daily_adj_dict is not None:
            current_data = self._daily_adj_dict.get(ctx.trade_date, pd.DataFrame()).copy()
        else:
            current_data = daily_adj[daily_adj["trade_date"] == ctx.trade_date].copy()

        if len(current_data) == 0:
            logger.warning(f"{ctx.trade_date} 没有行情数据")
            return pd.DataFrame()

        # 4. 标签
        from .labels import compute_forward_returns

        labels = compute_forward_returns(
            current_data=current_data,
            trade_date=ctx.trade_date,
            trading_dates=trading_dates,
            current_idx=current_idx,
            horizons=self.horizons,
            daily_adj_dict=self._daily_adj_dict,
            daily_adj=daily_adj,
        )

        # 5. 基础特征
        features = _calculate_base_features(
            current_data=current_data,
            daily_adj_dict=self._daily_adj_dict,
            trade_date=ctx.trade_date,
            trading_dates=trading_dates,
            current_idx=current_idx,
            lookback_windows=self.lookback_windows,
            trading_date_index=self._trading_date_index,
            daily_basic_data=ctx.daily_basic_data,
            moneyflow_data=ctx.moneyflow_data,
            daily_adj=daily_adj,
        )

        # 6. 价值红利 + 资金流（FeatureBuilder 实例方法）
        if ctx.daily_basic_data is not None and len(ctx.daily_basic_data) > 0:
            features = self._add_value_dividend_features(
                features, ctx.daily_basic_data, ctx.trade_date
            )
        if ctx.moneyflow_data is not None and len(ctx.moneyflow_data) > 0:
            features = self._add_moneyflow_features(
                features, ctx.moneyflow_data, ctx.trade_date, trading_dates, current_idx
            )

        # 7. 因子处理器（替代原 11 个内联 if-else 块）
        features = self._get_factor_registry().apply_all(features, ctx, current_data)

        # 7.5 基本面代理回填（cf_sales、cf_nm 等列）
        features = self._backfill_fundamental_proxy_features(features)

        # 8. 行业合并
        if ctx.shenwan_industry is not None:
            from .industry_merge import merge_shenwan_industry

            features = merge_shenwan_industry(
                features, ctx.shenwan_industry, self.shenwan_level, self.verbose
            )

        # 9. 高级因子
        features = self._add_advanced_factors(
            features, current_data, daily_adj, ctx.trade_date, trading_dates, current_idx
        )

        # 10. 合并特征和标签
        result = features.merge(labels, on=["trade_date", "ts_code"], how="inner")

        # 11. 过滤标记
        result = self._add_filter_flags(result, ctx.stock_basic, ctx.suspend_info, ctx.trade_date)
        result = self._add_limit_flags(result, ctx.daily_data, ctx.limit_info, ctx.trade_date)
        result = self._apply_filters(result)

        # 12. 中性化
        if ctx.apply_industry_neutralization and ctx.shenwan_industry is not None:
            from .neutralization import apply_industry_neutralization

            result = apply_industry_neutralization(
                result, self.horizons, self.lookback_windows, self.shenwan_level
            )
        if ctx.apply_size_neutralization:
            from .neutralization import apply_size_neutralization

            result = apply_size_neutralization(result)

        # 13. 个股特征
        result = _add_new_individual_features_static(result)

        # 14. 市场状态
        result = self._add_market_state_features(
            result, daily_adj, ctx.trade_date, trading_dates, current_idx, ctx.daily_basic_data
        )

        # 每日摘要
        self._emit_daily_summary(result, ctx.trade_date)
        return result

    def _emit_daily_summary(self, result: pd.DataFrame, trade_date: str) -> None:
        _n_samples = len(result)
        _n_neu = len([c for c in result.columns if c.startswith("neu_")])
        _n_zscore = len(
            [c for c in result.columns if c.startswith("zscore_") and not c.endswith("_sz")]
        )
        _n_sz = len([c for c in result.columns if c.endswith("_sz")])
        _fs = getattr(self, "_filter_stats", {}) or {}
        _removed = _fs.get("original", 0) - _fs.get("result", 0) if _fs else 0
        _parts = [f"{trade_date} ✓ {_n_samples}样本"]
        if _removed > 0:
            _parts.append(f"剔除{_removed}")
        _feat_parts = []
        if _n_neu:
            _feat_parts.append(f"demean+{_n_neu}")
        if _n_zscore:
            _feat_parts.append(f"zscore+{_n_zscore}")
        if _n_sz:
            _feat_parts.append(f"size+{_n_sz}")
        if _feat_parts:
            _parts.append(" | ".join(_feat_parts))
        self._last_summary = "  ".join(_parts)
        logger.debug(self._last_summary)

    def _get_factor_registry(self):
        if self._factor_registry is None:
            from .factor_handlers import create_factor_registry

            self._factor_registry = create_factor_registry()
        return self._factor_registry

    # ── 基本面代理回填 ──────────────────────────────────────

    @staticmethod
    def _backfill_fundamental_proxy_features(features: pd.DataFrame) -> pd.DataFrame:
        """用可稳定获取的字段回填基本面代理列（cf_sales、cf_nm）。"""
        if "cf_sales" not in features.columns:
            features["cf_sales"] = np.nan
        if "q_ocf_to_sales" in features.columns:
            features["cf_sales"] = features["cf_sales"].combine_first(
                features["q_ocf_to_sales"]
            )
        if "ocf_to_revenue" in features.columns:
            features["cf_sales"] = features["cf_sales"].combine_first(
                features["ocf_to_revenue"]
            )

        if "cf_nm" not in features.columns:
            features["cf_nm"] = np.nan
        if "ocf_to_profit" in features.columns:
            features["cf_nm"] = features["cf_nm"].combine_first(
                features["ocf_to_profit"]
            )

        return features

    # ── 交易日历工具 ──────────────────────────────────────────

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

    def _get_lookback_dates(
        self, trade_date: str, n: int, trading_dates: List[str]
    ) -> List[str]:
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
                f"将使用原始价格"
            )
            daily_adj["close_adj"].fillna(daily_adj["close"], inplace=True)
            if "open_adj" in daily_adj.columns:
                daily_adj["open_adj"].fillna(daily_adj["open"], inplace=True)
            if "high_adj" in daily_adj.columns:
                daily_adj["high_adj"].fillna(daily_adj["high"], inplace=True)
            if "low_adj" in daily_adj.columns:
                daily_adj["low_adj"].fillna(daily_adj["low"], inplace=True)
        return daily_adj

    # ── 技术因子缓存 ──────────────────────────────────────────

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

    # ── 高级因子 ──────────────────────────────────────────────

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

    # ── 窗口特征 ──────────────────────────────────────────────

    def _calculate_window_features(
        self, hist_data: pd.DataFrame, current_data: pd.DataFrame, window: int
    ) -> pd.DataFrame:
        return _calculate_window_features_static(hist_data, current_data, window)

    # ── 价值红利 ──────────────────────────────────────────────

    def _add_value_dividend_features(
        self,
        features: pd.DataFrame,
        daily_basic_data: pd.DataFrame,
        trade_date: str,
    ) -> pd.DataFrame:
        from ..common.feature_utils import log1p_transform

        daily_basic_today = daily_basic_data[
            daily_basic_data["trade_date"] == trade_date
        ].copy()
        if len(daily_basic_today) == 0:
            logger.warning(f"{trade_date} 没有 daily_basic 数据，价值红利特征将为空")
            return features
        value_cols = [
            "ts_code", "pb", "pe_ttm", "ps_ttm", "dv_ttm",
            "total_mv", "circ_mv", "turnover_rate", "volume_ratio",
        ]
        existing_cols = ["ts_code"] + [
            c for c in value_cols[1:] if c in daily_basic_today.columns
        ]
        daily_basic_today = daily_basic_today[existing_cols].copy()
        features = features.merge(daily_basic_today, on="ts_code", how="left")
        if "dv_ttm" in features.columns:
            features["dv_ttm"] = features["dv_ttm"].fillna(0)
        if "pe_ttm" in features.columns:
            features["ep_ttm"] = np.where(
                (features["pe_ttm"].notna()) & (features["pe_ttm"] > 0),
                1.0 / features["pe_ttm"],
                np.nan,
            )
            features["is_loss"] = (
                (features["pe_ttm"].isna()) | (features["pe_ttm"] <= 0)
            ).astype(int)
        if "pb" in features.columns:
            features["bp"] = np.where(
                (features["pb"].notna()) & (features["pb"] > 0),
                1.0 / features["pb"],
                np.nan,
            )
        if "total_mv" in features.columns:
            features["log_total_mv"] = log1p_transform(features["total_mv"])
        if "circ_mv" in features.columns:
            features["log_circ_mv"] = log1p_transform(features["circ_mv"])
        return features

    # ── 资金流 ────────────────────────────────────────────────

    def _add_moneyflow_features(
        self,
        features: pd.DataFrame,
        moneyflow_data: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int,
    ) -> pd.DataFrame:
        from ..common.feature_utils import winsorize_series

        moneyflow_today = moneyflow_data[moneyflow_data["trade_date"] == trade_date].copy()
        if len(moneyflow_today) == 0:
            logger.warning(f"{trade_date} 没有 moneyflow 数据，资金流特征将为空")
            return features
        merge_cols = ["ts_code", "net_mf_amount"]
        merge_cols = [c for c in merge_cols if c in moneyflow_today.columns]
        if len(merge_cols) > 1:
            features = features.merge(moneyflow_today[merge_cols], on="ts_code", how="left")
        if (
            "buy_lg_amount" in moneyflow_today.columns
            and "sell_lg_amount" in moneyflow_today.columns
        ):
            moneyflow_today["lg_net_amount"] = (
                moneyflow_today["buy_lg_amount"] - moneyflow_today["sell_lg_amount"]
            )
            features = features.merge(
                moneyflow_today[["ts_code", "lg_net_amount"]], on="ts_code", how="left"
            )
        if (
            "buy_elg_amount" in moneyflow_today.columns
            and "sell_elg_amount" in moneyflow_today.columns
        ):
            moneyflow_today["elg_net_amount"] = (
                moneyflow_today["buy_elg_amount"] - moneyflow_today["sell_elg_amount"]
            )
            features = features.merge(
                moneyflow_today[["ts_code", "elg_net_amount"]], on="ts_code", how="left"
            )
            _total = (
                moneyflow_today["buy_elg_amount"] + moneyflow_today["sell_elg_amount"]
            )
            moneyflow_today["order_imbalance"] = np.where(
                _total > 1e-6,
                (
                    moneyflow_today["buy_elg_amount"]
                    - moneyflow_today["sell_elg_amount"]
                )
                / _total,
                np.nan,
            )
            features = features.merge(
                moneyflow_today[["ts_code", "order_imbalance"]], on="ts_code", how="left"
            )
        for window in [5, 20]:
            hist_dates = self._get_lookback_dates(trade_date, window, trading_dates)
            if not hist_dates:
                features[f"net_mf_amount_sum_{window}"] = np.nan
                features[f"net_mf_amount_mean_{window}"] = np.nan
                if "lg_net_amount" in features.columns:
                    features[f"lg_net_amount_sum_{window}"] = np.nan
                if "elg_net_amount" in features.columns:
                    features[f"elg_net_amount_sum_{window}"] = np.nan
                continue
            hist_moneyflow = moneyflow_data[
                moneyflow_data["trade_date"].isin(hist_dates)
            ].copy()
            if len(hist_moneyflow) == 0:
                continue
            if (
                "buy_lg_amount" in hist_moneyflow.columns
                and "sell_lg_amount" in hist_moneyflow.columns
            ):
                hist_moneyflow["lg_net_amount"] = (
                    hist_moneyflow["buy_lg_amount"] - hist_moneyflow["sell_lg_amount"]
                )
            if (
                "buy_elg_amount" in hist_moneyflow.columns
                and "sell_elg_amount" in hist_moneyflow.columns
            ):
                hist_moneyflow["elg_net_amount"] = (
                    hist_moneyflow["buy_elg_amount"] - hist_moneyflow["sell_elg_amount"]
                )
                # 历史订单失衡（用于滚动均值）
                _total = (
                    hist_moneyflow["buy_elg_amount"]
                    + hist_moneyflow["sell_elg_amount"]
                )
                hist_moneyflow["order_imbalance"] = np.where(
                    _total > 1e-6,
                    (
                        hist_moneyflow["buy_elg_amount"]
                        - hist_moneyflow["sell_elg_amount"]
                    )
                    / _total,
                    np.nan,
                )
            agg_dict = {}
            if "net_mf_amount" in hist_moneyflow.columns:
                agg_dict["net_mf_amount"] = ["sum", "mean"]
            if "lg_net_amount" in hist_moneyflow.columns:
                agg_dict["lg_net_amount"] = ["sum"]
            if "elg_net_amount" in hist_moneyflow.columns:
                agg_dict["elg_net_amount"] = ["sum"]
            if "order_imbalance" in hist_moneyflow.columns:
                agg_dict["order_imbalance"] = ["mean"]
            if not agg_dict:
                continue
            rolling_features = (
                hist_moneyflow.groupby("ts_code").agg(agg_dict).reset_index()
            )
            new_columns = ["ts_code"]
            for col in rolling_features.columns[1:]:
                if isinstance(col, tuple):
                    new_columns.append(f"{col[0]}_{col[1]}_{window}")
                else:
                    new_columns.append(col)
            rolling_features.columns = new_columns
            features = features.merge(rolling_features, on="ts_code", how="left")
        winsorize_cols = [
            c for c in features.columns if "net_amount" in c or "mf_amount" in c
        ]
        for col in winsorize_cols:
            if col in features.columns:
                features[col] = winsorize_series(features[col], limits=(0.01, 0.01))
        return features

    # ── 过滤 ──────────────────────────────────────────────────

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

    # ── 向后兼容委托（原私有方法已移出，保留薄封装供测试/旧调用方使用）──

    def _calculate_forward_returns(
        self,
        current_data: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int,
    ) -> pd.DataFrame:
        """已移出至 labels.py，保留委托以兼容直接调用的测试代码。"""
        from .labels import compute_forward_returns

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
        from .industry_merge import merge_shenwan_industry

        return merge_shenwan_industry(
            features, shenwan_industry, self.shenwan_level, self.verbose
        )

    def _apply_industry_neutralization(self, features: pd.DataFrame) -> pd.DataFrame:
        """已移出至 neutralization.py，保留委托。"""
        from .neutralization import apply_industry_neutralization

        return apply_industry_neutralization(
            features, self.horizons, self.lookback_windows, self.shenwan_level
        )

    def _apply_size_neutralization(
        self, result: pd.DataFrame, n_size_groups: int = 10
    ) -> pd.DataFrame:
        """已移出至 neutralization.py，保留委托。"""
        from .neutralization import apply_size_neutralization

        return apply_size_neutralization(result, n_size_groups=n_size_groups)

    def _add_new_individual_features(self, result: pd.DataFrame) -> pd.DataFrame:
        """已改为模块级函数，保留委托。"""
        return _add_new_individual_features_static(result)

    # ── 市场状态 ──────────────────────────────────────────────

    def _add_market_state_features(
        self,
        result: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: list,
        current_idx: int,
        daily_basic_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        from .market_state import add_market_state_features, precompute_market_state_cache

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


# ══════════════════════════════════════════════════════════════════
#  模块级静态函数（供 parallel.py / 测试使用，不依赖 FeatureBuilder 实例）
# ══════════════════════════════════════════════════════════════════


def _calculate_base_features(
    current_data: pd.DataFrame,
    daily_adj_dict: Optional[Dict[str, pd.DataFrame]],
    trade_date: str,
    trading_dates: List[str],
    current_idx: int,
    lookback_windows: List[int],
    trading_date_index: Optional[Dict[str, int]],
    daily_basic_data: Optional[pd.DataFrame] = None,
    moneyflow_data: Optional[pd.DataFrame] = None,
    daily_adj: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """模块级基础特征计算（无状态版本）。"""
    base_columns = ["trade_date", "ts_code", "vol", "amount"]
    clean_marker_columns = [
        "is_st", "is_suspended", "is_limit_up", "is_limit_down",
        "list_days", "tradable",
    ]
    columns_to_keep = base_columns.copy()
    for col in clean_marker_columns:
        if col in current_data.columns:
            columns_to_keep.append(col)

    features = current_data[columns_to_keep].copy()

    # ret_1
    features = features.merge(
        current_data[["ts_code", "pct_chg"]],
        on="ts_code", how="left", suffixes=("", "_dup"),
    )
    features.rename(columns={"pct_chg": "ret_1"}, inplace=True)
    features["ret_1"] = features["ret_1"] / 100.0

    # opening_strength
    if "open" in current_data.columns and "pre_close" in current_data.columns:
        _open = current_data[["ts_code", "open", "pre_close"]].copy()
        _open["opening_strength"] = np.where(
            _open["pre_close"] > 1e-6, _open["open"] / _open["pre_close"] - 1, np.nan,
        )
        features = features.merge(
            _open[["ts_code", "opening_strength"]], on="ts_code", how="left",
        )

    # intraday_vol_structure
    if all(c in current_data.columns for c in ["high", "open", "low"]):
        _hloc = current_data[["ts_code", "high", "open", "low"]].copy()
        _up = _hloc["high"] - _hloc["open"]
        _down = _hloc["open"] - _hloc["low"]
        _hloc["intraday_vol_structure"] = np.where(_down > 1e-6, _up / _down, np.nan)
        features = features.merge(
            _hloc[["ts_code", "intraday_vol_structure"]], on="ts_code", how="left",
        )

    # 回看特征
    for window in lookback_windows:
        hist_dates = _get_lookback_dates_static(
            trade_date, window, trading_dates, trading_date_index,
        )
        if not hist_dates:
            features[f"ret_{window}"] = np.nan
            features[f"vol_ratio_{window}"] = np.nan
            features[f"ma_deviation_{window}"] = np.nan
            continue
        if daily_adj_dict is not None:
            _frames = [daily_adj_dict[d] for d in hist_dates if d in daily_adj_dict]
            hist_data = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
        elif daily_adj is not None:
            hist_data = daily_adj[daily_adj["trade_date"].isin(hist_dates)]
        else:
            hist_data = pd.DataFrame()

        hist_features = _calculate_window_features_static(hist_data, current_data, window)
        features = features.merge(hist_features, on="ts_code", how="left")

    return features


def _get_lookback_dates_static(
    trade_date: str,
    n: int,
    trading_dates: List[str],
    trading_date_index: Optional[Dict[str, int]],
) -> List[str]:
    if trading_date_index is not None:
        idx = trading_date_index.get(trade_date, -1)
        if idx == -1:
            return []
    else:
        if trade_date not in trading_dates:
            return []
        idx = trading_dates.index(trade_date)
    if idx < n:
        return []
    return trading_dates[idx - n : idx]


def _calculate_window_features_static(
    hist_data: pd.DataFrame, current_data: pd.DataFrame, window: int
) -> pd.DataFrame:
    if len(hist_data) == 0:
        return pd.DataFrame(columns=["ts_code"])
    hist_data = hist_data.sort_values(["ts_code", "trade_date"])
    grouped = hist_data.groupby("ts_code", as_index=False)
    window_features = grouped.agg(
        {"close_adj": ["first", "last", "mean"], "vol": "mean", "amount": "mean"}
    )
    new_columns = []
    for col in window_features.columns:
        if col[0] == "ts_code":
            new_columns.append("ts_code")
        else:
            new_columns.append("_".join(col).strip("_"))
    window_features.columns = new_columns
    window_features = window_features.rename(
        columns={
            "close_adj_first": "first_close",
            "close_adj_last": "last_close",
            "close_adj_mean": "ma_close",
            "vol_mean": "mean_vol",
            "amount_mean": "mean_amount",
        }
    )
    window_features[f"ret_{window}"] = (
        window_features["last_close"] / window_features["first_close"]
    ) - 1
    current_vol_amount = current_data[["ts_code", "vol", "amount", "close_adj"]].copy()
    window_features = window_features.merge(current_vol_amount, on="ts_code", how="left")
    window_features[f"vol_ratio_{window}"] = np.where(
        window_features["mean_vol"] > 1e-6,
        window_features["vol"] / window_features["mean_vol"],
        np.nan,
    )
    window_features[f"ma_deviation_{window}"] = np.where(
        window_features["ma_close"] > 1e-6,
        (window_features["close_adj"] - window_features["ma_close"])
        / window_features["ma_close"],
        np.nan,
    )
    keep_cols = [
        "ts_code", f"ret_{window}", f"vol_ratio_{window}",
        f"ma_deviation_{window}", "mean_amount",
    ]
    window_features = window_features[keep_cols]
    window_features = window_features.rename(columns={"mean_amount": f"amount_ma{window}"})
    return window_features


def _add_advanced_factors_static(
    features: pd.DataFrame,
    current_data: pd.DataFrame,
    trade_date: str,
    trading_dates: List[str],
    current_idx: int,
    lookback_windows: List[int],
    tech_factor_cache_dict: Optional[Dict[str, pd.DataFrame]] = None,
    _get_tech_factor_today_fn=None,
) -> pd.DataFrame:
    """模块级高级因子计算（无状态版本）。"""
    result = features.copy()

    if all(
        col in current_data.columns
        for col in ["high_adj", "low_adj", "pre_close", "adj_factor"]
    ):
        amplitude_df = calculate_amplitude(current_data)
        result = result.merge(amplitude_df, on=["ts_code", "trade_date"], how="left")

    if all(
        col in current_data.columns
        for col in ["open_adj", "high_adj", "low_adj", "close_adj"]
    ):
        shadows_df = calculate_shadows(current_data)
        result = result.merge(shadows_df, on=["ts_code", "trade_date"], how="left")

    if "ret_1" in result.columns and current_idx >= max(lookback_windows):
        tech_today = None
        if tech_factor_cache_dict is not None:
            tech_today = tech_factor_cache_dict.get(
                trade_date, pd.DataFrame(columns=["ts_code", "trade_date"])
            )
        elif _get_tech_factor_today_fn is not None:
            tech_today = _get_tech_factor_today_fn(trade_date)
        if tech_today is not None and len(tech_today) > 0:
            vol_cols = [
                f"volatility_{w}"
                for w in lookback_windows
                if f"volatility_{w}" in tech_today.columns
            ]
            if vol_cols:
                result = result.merge(
                    tech_today[["ts_code", "trade_date"] + vol_cols],
                    on=["ts_code", "trade_date"],
                    how="left",
                )

    if all(f"ret_{w}" in result.columns for w in lookback_windows):
        industry_col = "sw_industry" if "sw_industry" in result.columns else None
        if industry_col is not None:
            industry_alpha_df = calculate_industry_alpha_windows(
                result, ret_windows=lookback_windows, industry_col=industry_col,
            )
            result = result.merge(
                industry_alpha_df, on=["ts_code", "trade_date"], how="left",
            )
            from ..factors.industry import calculate_industry_momentum_features

            ind_mom_df = calculate_industry_momentum_features(
                result, industry_col=industry_col, ret_col="ret_20",
            )
            result = result.merge(ind_mom_df, on=["ts_code", "trade_date"], how="left")

    if "ret_5" in result.columns and "ret_10" in result.columns:
        acceleration_df = calculate_acceleration(result)
        result = result.merge(acceleration_df, on=["ts_code", "trade_date"], how="left")

    vol_ratio_cols = [f"vol_ratio_{w}" for w in lookback_windows]
    if all(col in result.columns for col in vol_ratio_cols):
        vol_burst_df = calculate_volume_burst(result, vol_ratio_windows=lookback_windows)
        result = result.merge(vol_burst_df, on=["ts_code", "trade_date"], how="left")

    if current_idx >= 30:
        tech_today = None
        if tech_factor_cache_dict is not None:
            tech_today = tech_factor_cache_dict.get(
                trade_date, pd.DataFrame(columns=["ts_code", "trade_date"])
            )
        elif _get_tech_factor_today_fn is not None:
            tech_today = _get_tech_factor_today_fn(trade_date)
        if tech_today is not None and len(tech_today) > 0:
            tech_indicator_cols = [
                c
                for c in [
                    "rsi_14", "kdj_k", "kdj_d", "kdj_j",
                    "macd_dif", "macd_dea", "macd_hist",
                    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_pct",
                    "atr_14", "atr_pct_14",
                ]
                if c in tech_today.columns
            ]
            if tech_indicator_cols:
                result = result.merge(
                    tech_today[["ts_code", "trade_date"] + tech_indicator_cols],
                    on=["ts_code", "trade_date"],
                    how="left",
                )

    return result


def _add_filter_flags_static(
    df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    suspend_info: Optional[pd.DataFrame],
    trade_date: str,
) -> pd.DataFrame:
    result = df.copy()
    has_clean_flags = all(
        col in result.columns
        for col in ["is_st", "is_suspended", "tradable", "list_days"]
    )
    if has_clean_flags:
        logger.debug("数据已包含 clean 层过滤标记，直接复用")
        return result

    logger.info("clean 层标记不存在，开始计算过滤标记")
    stock_names = stock_basic[["ts_code", "name"]].copy()
    result = result.merge(stock_names, on="ts_code", how="left")
    result["is_st"] = (
        result["name"]
        .fillna("")
        .str.contains(r"^\*?S?\*?ST|退", case=False, regex=True)
        .astype(int)
    )
    stock_list_date = stock_basic[["ts_code", "list_date"]].copy()
    if pd.api.types.is_datetime64_any_dtype(stock_list_date["list_date"]):
        stock_list_date["list_date"] = stock_list_date["list_date"].dt.strftime("%Y%m%d")
    result = result.merge(
        stock_list_date, on="ts_code", how="left", suffixes=("", "_basic"),
    )
    try:
        trade_date_dt = pd.to_datetime(trade_date, format="%Y%m%d")
        result["list_date_dt"] = pd.to_datetime(
            result["list_date"], format="%Y%m%d", errors="coerce",
        )
        result["list_days"] = (trade_date_dt - result["list_date_dt"]).dt.days
        result.drop(columns=["list_date_dt"], inplace=True)
    except Exception as e:
        logger.warning(f"计算上市天数失败: {e}，使用默认值")
        result["list_days"] = 999

    if "vol" in result.columns:
        result["is_suspended"] = (result["vol"] <= 0).astype(int)
    else:
        result["is_suspended"] = 0

    if suspend_info is not None and len(suspend_info) > 0:
        if "suspend_date" in suspend_info.columns and "resume_date" in suspend_info.columns:
            suspend_info_normalized = normalize_date_columns(
                suspend_info, ["suspend_date", "resume_date"], to_str=True,
            )
            trade_date_str = to_trade_date_str(trade_date)
            suspend_today = suspend_info_normalized[
                (suspend_info_normalized["suspend_date"] <= trade_date_str)
                & (
                    (suspend_info_normalized["resume_date"] >= trade_date_str)
                    | (suspend_info_normalized["resume_date"].isna())
                )
            ]["ts_code"].unique()
            result.loc[result["ts_code"].isin(suspend_today), "is_suspended"] = 1
        elif (
            "trade_date" in suspend_info.columns
            and "suspend_type" in suspend_info.columns
        ):
            suspend_info_normalized = normalize_date_columns(
                suspend_info, ["trade_date"], to_str=True,
            )
            trade_date_str = to_trade_date_str(trade_date)
            suspend_today = suspend_info_normalized[
                (suspend_info_normalized["trade_date"] == trade_date_str)
                & (suspend_info_normalized["suspend_type"] == "S")
            ]["ts_code"].unique()
            result.loc[result["ts_code"].isin(suspend_today), "is_suspended"] = 1

    return result


def _add_limit_flags_static(
    df: pd.DataFrame,
    daily_data: pd.DataFrame,
    limit_info: Optional[pd.DataFrame],
    trade_date: str,
) -> pd.DataFrame:
    result = df.copy()
    has_clean_limit_flags = all(
        col in result.columns for col in ["is_limit_up", "is_limit_down"]
    )
    if has_clean_limit_flags:
        logger.debug("数据已包含 clean 层涨跌停标记，直接复用")
        return result

    logger.info("clean 层涨跌停标记不存在，开始计算")
    current_daily = daily_data[daily_data["trade_date"] == trade_date][
        ["ts_code", "close", "pct_chg"]
    ].copy()
    result = result.merge(
        current_daily, on="ts_code", how="left", suffixes=("", "_daily"),
    )
    result["is_limit_up"] = 0
    result["is_limit_down"] = 0

    non_st_mask = result["is_st"] == 0
    st_mask = result["is_st"] == 1
    kcb_mask = result["ts_code"].str.startswith("688")
    gem_mask = result["ts_code"].str.startswith("300") | result["ts_code"].str.startswith(
        "301"
    )
    reg_board_mask = (kcb_mask | gem_mask) & non_st_mask
    main_board_mask = ~(kcb_mask | gem_mask) & non_st_mask

    result.loc[reg_board_mask & (result["pct_chg"] >= 19.9), "is_limit_up"] = 1
    result.loc[reg_board_mask & (result["pct_chg"] <= -19.9), "is_limit_down"] = 1
    result.loc[main_board_mask & (result["pct_chg"] >= 9.9), "is_limit_up"] = 1
    result.loc[main_board_mask & (result["pct_chg"] <= -9.9), "is_limit_down"] = 1
    result.loc[st_mask & (result["pct_chg"] >= 4.9), "is_limit_up"] = 1
    result.loc[st_mask & (result["pct_chg"] <= -4.9), "is_limit_down"] = 1

    if limit_info is not None and len(limit_info) > 0:
        limit_today = limit_info[limit_info["trade_date"] == trade_date][
            ["ts_code", "up_limit", "down_limit"]
        ].copy()
        if len(limit_today) > 0:
            result = result.merge(
                limit_today, on="ts_code", how="left", suffixes=("", "_limit"),
            )
            result.loc[
                (result["close"] >= result["up_limit"] * 0.999), "is_limit_up"
            ] = 1
            result.loc[
                (result["close"] <= result["down_limit"] * 1.001), "is_limit_down"
            ] = 1
            result.drop(columns=["up_limit", "down_limit"], inplace=True, errors="ignore")

    result.drop(columns=["close", "pct_chg"], inplace=True, errors="ignore")
    return result


def _apply_filters_static(
    df: pd.DataFrame,
    require_label: bool = True,
    label_filter_mode: str = "all",
    horizon: int = 20,
    horizons: Optional[List[int]] = None,
    min_list_days: int = 365,
) -> pd.DataFrame:
    horizons = horizons or [5, 10, 20]

    filter_mask = (
        (df["is_st"] == 0)
        & (df["list_days"] >= min_list_days)
        & (df["is_suspended"] == 0)
    )

    if require_label:
        if label_filter_mode == "single":
            primary_col = f"y_ret_{horizon}"
            if primary_col in df.columns:
                filter_mask = filter_mask & df[primary_col].notna()
        else:
            for h in horizons:
                label_col = f"y_ret_{h}"
                if label_col in df.columns:
                    filter_mask = filter_mask & df[label_col].notna()

    return df[filter_mask].copy()


def _add_new_individual_features_static(result: pd.DataFrame) -> pd.DataFrame:
    # 去碎片化：上游多次逐列 merge/assign 导致 DataFrame 内部碎片，copy() 消除 PerformanceWarning
    result = result.copy()
    if "list_days" in result.columns:
        result["is_new_stock"] = (result["list_days"] < 365).astype(int)
    else:
        result["is_new_stock"] = 0

    if "circ_mv" in result.columns:
        result["size"] = result["circ_mv"]

    if "size" in result.columns and "sw_industry" in result.columns:
        result["_log1p_size"] = np.log1p(result["size"])
        result = cross_sectional_zscore(
            result,
            columns=["_log1p_size"],
            group_col="sw_industry",
            tradable_col="tradable",
            min_group_size=5,
            suffix="_z",
        )
        if "_log1p_size_z" in result.columns:
            result.rename(columns={"_log1p_size_z": "zscore_size"}, inplace=True)
        if "_log1p_size" in result.columns:
            result.drop(columns=["_log1p_size"], inplace=True)

    if "zscore_volatility_20" in result.columns and "zscore_size" in result.columns:
        result["spec_score"] = result["zscore_volatility_20"] * (-result["zscore_size"])
    else:
        result["spec_score"] = np.nan

    return result
