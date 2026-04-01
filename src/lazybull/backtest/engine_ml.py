"""ML 回测引擎

扩展 BacktestEngine 以支持 ML 信号的特征数据注入
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import to_trade_date_str
from .engine import BacktestEngine


def _format_ma250_decision_log(
    date: pd.Timestamp,
    ma250_ratio: float,
    threshold: float,
    hard_stop_exposure: float,
    base_exposure: float,
    final_exposure: float,
    ma250_triggered: bool,
    atr_scaling_enabled: bool,
    atr_ratio: Optional[float] = None,
    mkt_atr: Optional[float] = None,
    mkt_atr_ma250: Optional[float] = None,
) -> str:
    """格式化 MA250 日志，突出控仓结果与 ATR 缩放计算式。"""
    if np.isnan(ma250_ratio):
        trigger_text = "ratio=NaN:状态未知"
    elif ma250_triggered:
        trigger_text = f"ratio={ma250_ratio:.3f}:触发控仓"
    else:
        trigger_text = f"ratio={ma250_ratio:.3f}:未触发控仓"

    has_valid_atr = (
        atr_scaling_enabled
        and atr_ratio is not None
        and mkt_atr is not None
        and mkt_atr_ma250 is not None
        and not np.isnan(atr_ratio)
        and not np.isnan(mkt_atr)
        and not np.isnan(mkt_atr_ma250)
        and mkt_atr > 0
    )
    if has_valid_atr:
        atr_text = (
            "ATR缩放=开启("
            f"scale=atr_ma250/atr_now={mkt_atr_ma250:.2%}/{mkt_atr:.2%}={atr_ratio:.1%}"
            ")"
        )
    elif atr_scaling_enabled:
        atr_text = "ATR缩放=开启(缺少有效ATR数据)"
    else:
        atr_text = "ATR缩放=关闭"

    return (
        f"  MA250: {date.date()}, "
        f"{trigger_text}, "
        f"{atr_text}, "
        f"base_after_ma250={base_exposure:.1%}, "
        f"final_after_atr={final_exposure:.1%}."
    )


class BacktestEngineML(BacktestEngine):
    """支持 ML 信号的回测引擎

    通过重写 _build_signal_data 方法注入特征数据，
    其他回测逻辑（信号过滤、回填、权重归一化等）复用父类实现。

    市场择时仓位管理（可选，market_regime_enabled=True）：
    支持 4 种模式（market_regime_mode）：
    - binary:     mkt_ret_avg_20 < threshold → bear_exposure，否则满仓（原有逻辑）
    - vol_target: exposure = target_vol / annualized_vol，波动越大仓位越低
    - trend:      基于 mkt_ma_trend（MA20/MA60）线性降仓，下行趋势自动减仓
    - combined:   vol_target 与 trend 取最小值（或相乘），双重保护

    MA250 硬条件（可选，market_regime_ma250_hard_stop=True）：
    当大盘累积收益曲线低于 250 日均线（mkt_ma250_ratio < threshold）时，
    强制将仓位降至 market_regime_ma250_exposure（默认 0.0，即完全空仓）。
    此条件优先级高于其他择时模式，作为系统性熊市的"否决性"保护。
    """

    def __init__(
        self,
        features_by_date: Dict[str, pd.DataFrame],
        market_regime_enabled: bool = False,
        market_regime_mode: str = "binary",
        market_regime_bear_threshold: float = -0.02,
        market_regime_bear_exposure: float = 0.3,
        market_regime_vol_target: float = 0.15,
        market_regime_trend_threshold: float = 1.0,
        market_regime_min_exposure: float = 0.2,
        market_regime_combine_method: str = "min",
        market_regime_trend_guard: bool = True,
        market_regime_drawdown_guard: bool = True,
        market_regime_drawdown_threshold: float = -0.08,
        industry_momentum_filter: bool = False,
        industry_momentum_bottom_pct: float = 0.2,
        market_regime_ma250_hard_stop: bool = False,
        market_regime_ma250_threshold: float = 1.0,
        market_regime_ma250_exposure: float = 0.0,
        market_regime_ma250_atr_scaling: bool = False,
        **kwargs,
    ):
        """初始化 ML 回测引擎

        Args:
            features_by_date: 按日期组织的特征数据字典，键为日期字符串（YYYYMMDD），值为特征 DataFrame
            market_regime_enabled: 是否启用市场择时仓位管理，默认 False
            market_regime_mode: 择时模式 binary|vol_target|trend|combined，默认 binary
            market_regime_bear_threshold: mkt_ret_avg_20 低于此值判定为熊市，默认 -0.02（仅 binary）
            market_regime_bear_exposure: 熊市仓位系数（0~1），默认 0.3（仅 binary）
            market_regime_vol_target: 年化波动率目标，默认 0.15（仅 vol_target/combined）
            market_regime_trend_threshold: mkt_ma_trend 低于此值开始降仓，默认 1.0（仅 trend/combined）
            market_regime_min_exposure: 最低仓位系数，默认 0.2（非 binary 模式的下限）
            market_regime_combine_method: combined 模式组合方式 min|multiply，默认 min
            market_regime_trend_guard: combined 模式下趋势保护开关，默认 True。
                开启时上行趋势（mkt_ma_trend >= threshold）强制满仓，避免高波动上涨被误杀
            market_regime_drawdown_guard: 回撤保护开关，默认 True。
                开启时当 mkt_drawdown_20 低于 drawdown_threshold 时停止降仓，
                避免急跌后在底部继续减仓踏空反弹
            market_regime_drawdown_threshold: 回撤保护阈值，默认 -0.08（-8%）。
                mkt_drawdown_20 低于此值时视为已充分下跌，不再继续降仓
            industry_momentum_filter: 是否启用行业动量过滤（剔除弱势行业股票），默认 False
            industry_momentum_bottom_pct: 剔除行业动量排名后 X% 的行业（0~1），默认 0.2
            market_regime_ma250_hard_stop: 是否启用 MA250 长周期硬条件，默认 False。
                开启后当 mkt_ma250_ratio < threshold 时强制仓位降至 ma250_exposure，
                优先级高于其他择时模式（system-level 否决条件）
            market_regime_ma250_threshold: MA250 硬条件触发阈值（大盘收益曲线/MA250），
                默认 1.0（即大盘跌破长期均线时触发）
            market_regime_ma250_exposure: MA250 硬条件触发后的仓位系数，默认 0.0（完全空仓）
            market_regime_ma250_atr_scaling: 是否在 MA250 模块中启用 ATR 动态仓位缩放，
                默认 False。开启后仓位 = base × MA(ATR,250)/CurrentATR，
                高波动降仓、低波动恢复（上限 1.0，下限 min_exposure）
            **kwargs: 其他参数传递给父类 BacktestEngine
        """
        super().__init__(**kwargs)
        self.features_by_date = features_by_date
        self.market_regime_enabled = market_regime_enabled
        self.market_regime_mode = market_regime_mode
        self.market_regime_bear_threshold = market_regime_bear_threshold
        self.market_regime_bear_exposure = market_regime_bear_exposure
        self.market_regime_vol_target = market_regime_vol_target
        self.market_regime_trend_threshold = market_regime_trend_threshold
        self.market_regime_min_exposure = market_regime_min_exposure
        self.market_regime_combine_method = market_regime_combine_method
        self.market_regime_trend_guard = market_regime_trend_guard
        self.market_regime_drawdown_guard = market_regime_drawdown_guard
        self.market_regime_drawdown_threshold = market_regime_drawdown_threshold
        self._last_regime_exposure = 1.0  # 上一次的仓位系数，用于检测变动
        self.industry_momentum_filter = industry_momentum_filter
        self.industry_momentum_bottom_pct = industry_momentum_bottom_pct
        self.market_regime_ma250_hard_stop = market_regime_ma250_hard_stop
        self.market_regime_ma250_threshold = market_regime_ma250_threshold
        self.market_regime_ma250_exposure = market_regime_ma250_exposure
        self.market_regime_ma250_atr_scaling = market_regime_ma250_atr_scaling

        # 校验：ATR 缩放依赖 MA250 硬条件
        if market_regime_ma250_atr_scaling and not market_regime_ma250_hard_stop:
            logger.warning(
                "ma250_atr_scaling=True 但 ma250_hard_stop=False，ATR 缩放不会生效"
            )

        regime_info = ""
        if market_regime_enabled:
            if market_regime_mode == "binary":
                regime_info = (
                    f", 市场择时=开启(mode=binary, bear_threshold={market_regime_bear_threshold}, "
                    f"bear_exposure={market_regime_bear_exposure})"
                )
            else:
                regime_info = (
                    f", 市场择时=开启(mode={market_regime_mode}, "
                    f"vol_target={market_regime_vol_target}, "
                    f"trend_threshold={market_regime_trend_threshold}, "
                    f"min_exposure={market_regime_min_exposure}, "
                    f"combine={market_regime_combine_method}, "
                    f"trend_guard={market_regime_trend_guard}, "
                    f"dd_guard={market_regime_drawdown_guard}, "
                    f"dd_threshold={market_regime_drawdown_threshold})"
                )
        ind_filter_info = ""
        if industry_momentum_filter:
            ind_filter_info = f", 行业动量过滤=开启(剔除后{industry_momentum_bottom_pct*100:.0f}%行业)"
        ma250_info = ""
        if market_regime_ma250_hard_stop:
            atr_tag = ", ATR缩放=开启" if market_regime_ma250_atr_scaling else ""
            ma250_info = (
                f", MA250硬条件=开启(threshold={market_regime_ma250_threshold}, "
                f"exposure={market_regime_ma250_exposure}{atr_tag})"
            )
        logger.info(
            f"ML 回测引擎初始化: 特征数据覆盖 {len(features_by_date)} 个交易日"
            f"{regime_info}{ind_filter_info}{ma250_info}"
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

    # ── 行业动量过滤 ────────────────────────────────────────────────

    def _post_filter_candidates(
        self, ranked_candidates: list, date: pd.Timestamp
    ) -> list:
        """剔除弱势行业的股票，剩余候选自动补位到 top_n

        利用 features_by_date 中的 ind_momentum_rank（行业动量百分位排名，0~1）
        过滤掉排名 < bottom_pct 的行业的所有股票。由于候选列表远多于 top_n，
        过滤后父类仍从前 N 个候选中选取，自动实现补位。
        """
        if not self.industry_momentum_filter:
            return ranked_candidates

        date_str = date.strftime('%Y%m%d')
        features_df = self.features_by_date.get(date_str)
        if features_df is None or 'ind_momentum_rank' not in features_df.columns:
            return ranked_candidates

        # 构建 {ts_code: ind_momentum_rank} 查找表
        rank_map = dict(zip(
            features_df['ts_code'],
            features_df['ind_momentum_rank'],
        ))

        threshold = self.industry_momentum_bottom_pct
        filtered = []
        removed = 0
        for stock, score in ranked_candidates:
            rank = rank_map.get(stock)
            if rank is not None and rank < threshold:
                removed += 1
                continue
            filtered.append((stock, score))

        if removed > 0 and self.verbose:
            logger.info(
                f"  行业动量过滤: {date.date()}, "
                f"剔除 {removed} 只弱势行业股票 (bottom {threshold*100:.0f}%)"
            )

        return filtered

    # ── ATR 功能 hook 覆写 ────────────────────────────────────────────

    def _build_position_extra_info(self, date: pd.Timestamp, stock: str) -> Dict:
        """买入时从 features_by_date 读取买入日 ATR%，用于 ATR 动态止损

        仅当 enable_profit_based_holding 且 use_atr_for_early_exit 同时开启时生效。
        直接使用预计算的 atr_pct_14（=atr_14/close_adj），无需在运行时查询价格。
        """
        if not (self.enable_profit_based_holding and self.use_atr_for_early_exit):
            return {}
        date_str = date.strftime('%Y%m%d')
        features_df = self.features_by_date.get(date_str)
        if features_df is None or 'atr_pct_14' not in features_df.columns:
            return {}
        row = features_df[features_df['ts_code'] == stock]
        if row.empty:
            return {}
        atr_pct = row['atr_pct_14'].iloc[0]
        if pd.isna(atr_pct) or atr_pct <= 0:
            return {}
        return {'buy_atr_pct': float(atr_pct)}

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

    # ── 市场择时仓位管理 ──────────────────────────────────────────────

    def _get_feature_scalar(self, features_df: pd.DataFrame, col: str) -> float:
        """从 features_df 取广播到所有行的标量值（首行），缺失返回 NaN"""
        if col not in features_df.columns:
            return np.nan
        val = features_df[col].iloc[0]
        return float(val) if not pd.isna(val) else np.nan

    def _apply_ma250_atr_scaling(
        self, base_exposure: float, features_df: pd.DataFrame
    ) -> float:
        """ATR 动态仓位缩放: B = clip(A * MA(ATR,250) / CurrentATR, min_exposure, 1.0)

        高波动时 ratio<1 → 降仓；低波动时 ratio>1 → 允许仓位恢复到满仓但不超过 1.0。
        无 ATR 数据时回退到 base_exposure。
        """
        mkt_atr = self._get_feature_scalar(features_df, 'mkt_atr_pct')
        mkt_atr_ma250 = self._get_feature_scalar(features_df, 'mkt_atr_pct_ma250')

        if np.isnan(mkt_atr) or np.isnan(mkt_atr_ma250) or mkt_atr <= 0:
            return base_exposure

        atr_ratio = mkt_atr_ma250 / mkt_atr
        exposure = base_exposure * atr_ratio
        return float(np.clip(exposure, self.market_regime_min_exposure, 1.0))

    def _get_market_regime_exposure(self, date: pd.Timestamp) -> float:
        """根据市场状态计算仓位系数

        按 market_regime_mode 分派到对应策略：
        - binary:     二值模式（原有逻辑）
        - vol_target: 波动率目标模式
        - trend:      趋势叠加模式
        - combined:   vol_target + trend 组合

        Returns:
            仓位系数，1.0 = 满仓，< 1.0 = 降仓
        """
        date_str = date.strftime('%Y%m%d')
        features_df = self.features_by_date.get(date_str)
        if features_df is None or len(features_df) == 0:
            return 1.0

        # MA250 模块（优先级最高，超越其他择时模式）
        if self.market_regime_ma250_hard_stop:
            ma250_ratio = self._get_feature_scalar(features_df, 'mkt_ma250_ratio')

            # 第一步：MA250 基准仓位 A
            if not np.isnan(ma250_ratio) and ma250_ratio < self.market_regime_ma250_threshold:
                base_exposure = self.market_regime_ma250_exposure
                ma250_triggered = True
            else:
                base_exposure = 1.0
                ma250_triggered = False

            # 第二步：ATR 动态缩放 B = A * MA(ATR,250) / CurrentATR
            atr_ratio = None
            mkt_atr = None
            mkt_atr_ma250 = None
            if self.market_regime_ma250_atr_scaling:
                mkt_atr = self._get_feature_scalar(features_df, 'mkt_atr_pct')
                mkt_atr_ma250 = self._get_feature_scalar(features_df, 'mkt_atr_pct_ma250')
                if (
                    not np.isnan(mkt_atr)
                    and not np.isnan(mkt_atr_ma250)
                    and mkt_atr > 0
                ):
                    atr_ratio = mkt_atr_ma250 / mkt_atr
                exposure = self._apply_ma250_atr_scaling(base_exposure, features_df)
            else:
                exposure = base_exposure

            # MA250 触发或 ATR 缩放导致降仓时，短路返回（不进入其他择时模式）
            if ma250_triggered or exposure < 1.0:
                if abs(exposure - self._last_regime_exposure) > 1e-6:
                    logger.warning(
                        _format_ma250_decision_log(
                            date=date,
                            ma250_ratio=ma250_ratio,
                            threshold=self.market_regime_ma250_threshold,
                            hard_stop_exposure=self.market_regime_ma250_exposure,
                            base_exposure=base_exposure,
                            final_exposure=exposure,
                            ma250_triggered=ma250_triggered,
                            atr_scaling_enabled=self.market_regime_ma250_atr_scaling,
                            atr_ratio=atr_ratio,
                            mkt_atr=mkt_atr,
                            mkt_atr_ma250=mkt_atr_ma250,
                        )
                    )
                return exposure

        # 若仅启用了 MA250 硬条件而未启用常规择时，此处直接返回满仓
        if not self.market_regime_enabled:
            return 1.0

        mode = self.market_regime_mode

        if mode == "binary":
            exposure = self._regime_binary(features_df)
        elif mode == "vol_target":
            exposure = self._regime_vol_target(features_df)
        elif mode == "trend":
            exposure = self._regime_trend(features_df)
        elif mode == "combined":
            exposure = self._regime_combined(features_df)
        else:
            logger.warning(f"未知 market_regime_mode={mode}，回退到 binary")
            exposure = self._regime_binary(features_df)

        # 回撤保护：已经大幅下跌时不再继续降仓，避免在底部减仓踏空反弹
        if self.market_regime_drawdown_guard and exposure < self._last_regime_exposure:
            drawdown = self._get_feature_scalar(features_df, 'mkt_drawdown_20')
            if not np.isnan(drawdown) and drawdown < self.market_regime_drawdown_threshold:
                logger.warning(
                    f"回撤保护触发: mkt_drawdown_20={drawdown:.1%} < {self.market_regime_drawdown_threshold:.0%}, "
                    f"阻止降仓 {self._last_regime_exposure:.0%} → {exposure:.0%}，维持 {self._last_regime_exposure:.0%}"
                )
                return self._last_regime_exposure

        return exposure

    def _regime_binary(self, features_df: pd.DataFrame) -> float:
        """二值模式：mkt_ret_avg_20 < threshold → bear_exposure，否则 1.0"""
        mkt_ret = self._get_feature_scalar(features_df, 'mkt_ret_avg_20')
        if np.isnan(mkt_ret):
            return 1.0
        if mkt_ret < self.market_regime_bear_threshold:
            return self.market_regime_bear_exposure
        return 1.0

    def _regime_vol_target(self, features_df: pd.DataFrame) -> float:
        """波动率目标模式：target_vol / realized_vol，clamp [min_exposure, 1.0]

        使用 mkt_ret_vol_20（近 20 日全市场日均收益时序标准差）年化后
        与目标波动率比较，波动越大仓位越低。
        """
        mkt_ret_vol = self._get_feature_scalar(features_df, 'mkt_ret_vol_20')
        if np.isnan(mkt_ret_vol) or mkt_ret_vol <= 0:
            return 1.0
        annualized_vol = mkt_ret_vol * np.sqrt(252)
        exposure = self.market_regime_vol_target / annualized_vol
        return float(np.clip(exposure, self.market_regime_min_exposure, 1.0))

    def _regime_trend(self, features_df: pd.DataFrame) -> float:
        """趋势叠加模式：基于 mkt_ma_trend 线性降仓

        mkt_ma_trend = MA20(cumret) / MA60(cumret)，>1 为上行趋势。
        当 mkt_ma_trend >= threshold 时满仓；低于 threshold 时线性缩放。
        """
        ma_trend = self._get_feature_scalar(features_df, 'mkt_ma_trend')
        if np.isnan(ma_trend):
            return 1.0
        if ma_trend >= self.market_regime_trend_threshold:
            return 1.0
        # 线性缩放：trend 越低于 threshold，exposure 越小
        exposure = ma_trend / self.market_regime_trend_threshold
        return float(np.clip(exposure, self.market_regime_min_exposure, 1.0))

    def _regime_combined(self, features_df: pd.DataFrame) -> float:
        """组合模式：同时考虑 vol_target 和 trend

        trend_guard=True（默认）时：上行趋势强制满仓，避免高波动上涨被 vol_target 误杀。
        仅当趋势向下时才启用 vol_target + trend 双重保护。

        combine_method="min" → 取两者中更保守的值（默认）
        combine_method="multiply" → 两者相乘（效果更强）
        """
        trend_exp = self._regime_trend(features_df)

        # 趋势保护：上行趋势时跳过 vol_target，直接满仓
        if self.market_regime_trend_guard and trend_exp >= 1.0:
            return 1.0

        vol_exp = self._regime_vol_target(features_df)
        if self.market_regime_combine_method == "multiply":
            combined = vol_exp * trend_exp
        else:  # "min"
            combined = min(vol_exp, trend_exp)
        return float(np.clip(combined, self.market_regime_min_exposure, 1.0))

    def _execute_pending_buys(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """执行买入前应用市场择时仓位缩放

        在父类执行买入之前，将 pending_signals 中的权重乘以市场仓位系数。
        原理与 ECT（权益曲线交易）相同：权重之和 < 1 → 剩余资金留作现金。
        """
        if self.market_regime_enabled or self.market_regime_ma250_hard_stop:
            # 找到前一个交易日的信号（与父类逻辑一致）
            current_idx = date_to_idx.get(date)
            if current_idx is not None and current_idx > 0:
                signal_date = trading_dates[current_idx - 1]
                signal_data = self.pending_signals.get(signal_date)

                if signal_data is not None:
                    exposure = self._get_market_regime_exposure(signal_date)

                    # 检测仓位变动并输出日志
                    prev = self._last_regime_exposure
                    if abs(exposure - prev) > 1e-6:
                        direction = "↓ 降仓" if exposure < prev else "↑ 加仓"
                        logger.warning(
                            f"  市场择时变动: {date.date()}, "
                            f"mode={self.market_regime_mode}, "
                            f"exposure {prev:.0%} → {exposure:.0%} ({direction})"
                        )
                    self._last_regime_exposure = exposure

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
                                f"mode={self.market_regime_mode}, "
                                f"exposure={exposure:.2f}, 仓位降至 {exposure*100:.0f}%"
                            )

        # 调用父类完成实际买入
        super()._execute_pending_buys(date, trading_dates, date_to_idx)
