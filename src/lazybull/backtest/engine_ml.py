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
        industry_rotation_enhanced: bool = False,
        industry_rotation_alpha: float = 0.3,
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
        self.industry_rotation_enhanced = industry_rotation_enhanced
        self.industry_rotation_alpha = industry_rotation_alpha
        self.market_regime_ma250_hard_stop = market_regime_ma250_hard_stop
        self.market_regime_ma250_threshold = market_regime_ma250_threshold
        self.market_regime_ma250_exposure = market_regime_ma250_exposure
        self.market_regime_ma250_atr_scaling = market_regime_ma250_atr_scaling
        self._last_market_regime_trace = self._build_default_market_trace()

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
        if industry_rotation_enhanced:
            ind_filter_info += f", 行业轮动加权=开启(alpha={industry_rotation_alpha})"
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

    def _build_default_market_trace(self) -> Dict:
        """构建市场层默认摘要。"""
        return {
            "market_layer_exposure": 1.0,
            "ma250": {
                "enabled": self.market_regime_ma250_hard_stop,
                "exposure": 1.0,
                "summary": "待执行日评估" if self.market_regime_ma250_hard_stop else "未启用",
            },
            "market_regime": {
                "enabled": self.market_regime_enabled,
                "exposure": 1.0,
                "summary": "待执行日评估" if self.market_regime_enabled else "未启用",
            },
        }

    def _initialize_decision_trace_for_signal(self, decision_trace: Dict) -> Dict:
        """补充市场层占位信息，供统一摘要使用。"""
        market_trace = self._build_default_market_trace()
        decision_trace["ma250"] = market_trace["ma250"]
        decision_trace["market_regime"] = market_trace["market_regime"]
        decision_trace["market_layer_exposure"] = market_trace["market_layer_exposure"]
        return decision_trace

    def _finalize_decision_trace_for_signal_day(
        self, decision_trace: Dict, signal_date: pd.Timestamp
    ) -> Dict:
        """在信号日预填市场择时摘要，避免统一摘要退回占位文案。"""
        if not (self.market_regime_enabled or self.market_regime_ma250_hard_stop):
            return decision_trace

        self._get_market_regime_exposure(signal_date)
        decision_trace["ma250"] = self._last_market_regime_trace["ma250"]
        decision_trace["market_regime"] = self._last_market_regime_trace["market_regime"]
        decision_trace["market_layer_exposure"] = self._last_market_regime_trace[
            "market_layer_exposure"
        ]
        final_target = decision_trace.get("final_target_exposure")
        market_layer = self._last_market_regime_trace["market_layer_exposure"]
        if final_target is not None:
            decision_trace["final_target_exposure"] = float(final_target) * float(market_layer)
        return decision_trace

    def _build_market_regime_summary(
        self,
        features_df: pd.DataFrame,
        exposure: float,
        drawdown_guard_triggered: bool = False,
        drawdown: Optional[float] = None,
    ) -> str:
        """根据当前模式构造市场择时摘要。"""
        mode = self.market_regime_mode
        if mode == "binary":
            mkt_ret = self._get_feature_scalar(features_df, "mkt_ret_avg_20")
            base = (
                f"mode=binary, mkt_ret_avg_20={mkt_ret:.2%}, "
                f"bear_threshold={self.market_regime_bear_threshold:.2%}, 市场层={exposure:.1%}"
                if not np.isnan(mkt_ret)
                else f"mode=binary, mkt_ret_avg_20缺失，市场层={exposure:.1%}"
            )
        elif mode == "vol_target":
            mkt_ret_vol = self._get_feature_scalar(features_df, "mkt_ret_vol_20")
            if np.isnan(mkt_ret_vol) or mkt_ret_vol <= 0:
                base = f"mode=vol_target, realized_vol缺失，市场层={exposure:.1%}"
            else:
                annualized_vol = mkt_ret_vol * np.sqrt(252)
                base = (
                    f"mode=vol_target, target_vol={self.market_regime_vol_target:.1%}, "
                    f"realized_vol={annualized_vol:.1%}, 市场层={exposure:.1%}"
                )
        elif mode == "trend":
            ma_trend = self._get_feature_scalar(features_df, "mkt_ma_trend")
            base = (
                f"mode=trend, mkt_ma_trend={ma_trend:.3f}, "
                f"threshold={self.market_regime_trend_threshold:.3f}, 市场层={exposure:.1%}"
                if not np.isnan(ma_trend)
                else f"mode=trend, mkt_ma_trend缺失，市场层={exposure:.1%}"
            )
        else:
            trend_exp = self._regime_trend(features_df)
            vol_exp = self._regime_vol_target(features_df)
            base = (
                f"mode=combined, vol_exp={vol_exp:.1%}, trend_exp={trend_exp:.1%}, "
                f"combine={self.market_regime_combine_method}, 市场层={exposure:.1%}"
            )

        if drawdown_guard_triggered and drawdown is not None and not np.isnan(drawdown):
            base += f"，回撤保护触发(mkt_drawdown_20={drawdown:.1%})"
        return base

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

    # ── 行业动量过滤 & 行业轮动加权 ──────────────────────────────────

    def _post_filter_candidates(
        self, ranked_candidates: list, date: pd.Timestamp
    ) -> list:
        """对候选列表做行业维度的过滤和/或加权

        两个独立开关:
        1. industry_momentum_filter: 硬过滤 — 剔除弱势行业(ind_momentum_rank < bottom_pct)
        2. industry_rotation_enhanced: 软加权 — 按行业动量排名对分数做乘性调整
           adjusted_score = score × (1 + alpha × (rank - 0.5))
           其中 rank ∈ [0,1],alpha 控制加权强度:
             - 最强行业(rank=1): 分数 × (1 + 0.5α)
             - 中位行业(rank=0.5): 分数不变
             - 最弱行业(rank=0): 分数 × (1 - 0.5α)
           调整后重新排序,弱势行业中的超强个股仍有机会入选。
        """
        date_str = date.strftime('%Y%m%d')
        features_df = self.features_by_date.get(date_str)

        # 步骤1: 行业动量硬过滤
        if self.industry_momentum_filter:
            if features_df is not None and 'ind_momentum_rank' in features_df.columns:
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
                ranked_candidates = filtered

        # 步骤2: 行业轮动加权
        if self.industry_rotation_enhanced:
            if features_df is not None and 'ind_momentum_rank' in features_df.columns:
                rank_map = dict(zip(
                    features_df['ts_code'],
                    features_df['ind_momentum_rank'],
                ))
                alpha = self.industry_rotation_alpha
                adjusted = []
                for stock, score in ranked_candidates:
                    rank = rank_map.get(stock)
                    if rank is not None and not np.isnan(rank):
                        # rank ∈ [0,1], 中位 0.5 → 不调整
                        multiplier = 1.0 + alpha * (rank - 0.5)
                        adjusted.append((stock, score * multiplier))
                    else:
                        adjusted.append((stock, score))
                # 重新按调整后分数降序排列
                adjusted.sort(key=lambda x: x[1], reverse=True)
                if self.verbose:
                    logger.info(
                        f"  行业轮动加权: {date.date()}, alpha={alpha}, "
                        f"候选 {len(adjusted)} 只"
                    )
                ranked_candidates = adjusted

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
        trace = self._build_default_market_trace()
        date_str = date.strftime('%Y%m%d')
        features_df = self.features_by_date.get(date_str)
        if features_df is None or len(features_df) == 0:
            if trace["ma250"]["enabled"]:
                trace["ma250"]["summary"] = "缺少特征数据，按100.0%处理"
            if trace["market_regime"]["enabled"]:
                trace["market_regime"]["summary"] = "缺少特征数据，按100.0%处理"
            self._last_market_regime_trace = trace
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

            if np.isnan(ma250_ratio):
                ratio_text = "ratio=NaN"
            else:
                comparator = "<" if ma250_triggered else ">="
                ratio_text = (
                    f"ratio={ma250_ratio:.3f} {comparator} {self.market_regime_ma250_threshold:.3f}"
                )

            if (
                self.market_regime_ma250_atr_scaling
                and atr_ratio is not None
                and mkt_atr is not None
                and mkt_atr_ma250 is not None
            ):
                atr_text = (
                    f"ATR缩放={mkt_atr_ma250:.2%}/{mkt_atr:.2%}={atr_ratio:.1%}"
                )
            elif self.market_regime_ma250_atr_scaling:
                atr_text = "ATR缩放=缺少有效ATR数据"
            else:
                atr_text = "ATR缩放=关闭"

            trace["ma250"] = {
                "enabled": True,
                "exposure": exposure,
                "summary": (
                    f"{ratio_text}，{'触发控仓' if ma250_triggered else '未触发硬条件'}，"
                    f"base_after_ma250={base_exposure:.1%}，{atr_text}，市场层={exposure:.1%}"
                ),
            }

            # MA250 触发或 ATR 缩放导致降仓时，短路返回（不进入其他择时模式）
            if ma250_triggered or exposure < 1.0:
                if self.market_regime_enabled:
                    trace["market_regime"] = {
                        "enabled": True,
                        "exposure": 1.0,
                        "summary": "跳过（MA250/ATR 已先行确定市场层仓位）",
                    }
                trace["market_layer_exposure"] = exposure
                self._last_market_regime_trace = trace
                return exposure
        else:
            trace["ma250"] = {
                "enabled": False,
                "exposure": 1.0,
                "summary": "未启用",
            }

        # 若仅启用了 MA250 硬条件而未启用常规择时，此处直接返回满仓
        if not self.market_regime_enabled:
            trace["market_regime"] = {
                "enabled": False,
                "exposure": 1.0,
                "summary": "未启用",
            }
            trace["market_layer_exposure"] = 1.0
            self._last_market_regime_trace = trace
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
        drawdown = np.nan
        drawdown_guard_triggered = False
        if self.market_regime_drawdown_guard and exposure < self._last_regime_exposure:
            drawdown = self._get_feature_scalar(features_df, 'mkt_drawdown_20')
            if not np.isnan(drawdown) and drawdown < self.market_regime_drawdown_threshold:
                drawdown_guard_triggered = True
                exposure = self._last_regime_exposure

        trace["market_regime"] = {
            "enabled": True,
            "exposure": exposure,
            "summary": self._build_market_regime_summary(
                features_df=features_df,
                exposure=exposure,
                drawdown_guard_triggered=drawdown_guard_triggered,
                drawdown=drawdown,
            ),
        }
        trace["market_layer_exposure"] = exposure
        self._last_market_regime_trace = trace

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

                    if isinstance(signal_data, dict) and "decision_trace" in signal_data:
                        signal_data["decision_trace"]["ma250"] = self._last_market_regime_trace[
                            "ma250"
                        ]
                        signal_data["decision_trace"]["market_regime"] = (
                            self._last_market_regime_trace["market_regime"]
                        )
                        signal_data["decision_trace"]["market_layer_exposure"] = (
                            self._last_market_regime_trace["market_layer_exposure"]
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

        # 调用父类完成实际买入
        super()._execute_pending_buys(date, trading_dates, date_to_idx)
