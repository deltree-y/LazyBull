# -*- coding: utf-8 -*-
"""FeatureBuilder 编排 mixin：单日特征构建主链路。"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .static_core import (
    _backfill_fundamental_proxy_features_static,
    _calculate_base_features,
)
from .static_extra import (
    _add_moneyflow_features_static,
    _add_new_individual_features_static,
    _add_value_dividend_features_static,
)


class FeatureOrchestrationMixin:
    """FeatureBuilder 单日构建编排 mixin。"""

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
        pledge_data: Optional[pd.DataFrame] = None,
        share_float_data: Optional[pd.DataFrame] = None,
        block_trade_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """构建单个交易日的截面特征和标签（签名保持向后兼容）。"""

        from ..context import FeatureContext

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
            pledge_data=pledge_data,
            share_float_data=share_float_data,
            block_trade_data=block_trade_data,
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
        from ..labels import compute_forward_returns

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
            features = _add_value_dividend_features_static(
                features=features,
                daily_basic_data=ctx.daily_basic_data,
                trade_date=ctx.trade_date,
            )
        if ctx.moneyflow_data is not None and len(ctx.moneyflow_data) > 0:
            features = _add_moneyflow_features_static(
                features=features,
                moneyflow_data=ctx.moneyflow_data,
                trade_date=ctx.trade_date,
                trading_dates=trading_dates,
                current_idx=current_idx,
                trading_date_index=self._trading_date_index,
            )

        # 7. 因子处理器（替代原 11 个内联 if-else 块）
        features = self._get_factor_registry().apply_all(features, ctx, current_data)

        # 7.5 基本面代理回填（cf_sales、cf_nm 等列）
        features = _backfill_fundamental_proxy_features_static(features)

        # 8. 行业合并
        if ctx.shenwan_industry is not None:
            from ..industry_merge import merge_shenwan_industry

            features = merge_shenwan_industry(
                features, ctx.shenwan_industry, self.shenwan_level, self.verbose
            )

        # 9. 高级因子
        features = self._add_advanced_factors(
            features, current_data, daily_adj, ctx.trade_date, trading_dates, current_idx
        )

        # 9.5 风控因子（独立模块，逻辑在 src/lazybull/risk/ 中维护）
        features = self._add_risk_factors(features, daily_adj, ctx.trade_date, trading_dates)

        # 10. 合并特征和标签
        result = features.merge(labels, on=["trade_date", "ts_code"], how="inner")

        # 11. 过滤标记
        result = self._add_filter_flags(result, ctx.stock_basic, ctx.suspend_info, ctx.trade_date)
        result = self._add_limit_flags(result, ctx.daily_data, ctx.limit_info, ctx.trade_date)
        result = self._apply_filters(result)

        # 12. 中性化
        if ctx.apply_industry_neutralization and ctx.shenwan_industry is not None:
            from ..neutralization import apply_industry_neutralization

            result = apply_industry_neutralization(
                result, self.horizons, self.lookback_windows, self.shenwan_level
            )
        if ctx.apply_size_neutralization:
            from ..neutralization import apply_size_neutralization

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
