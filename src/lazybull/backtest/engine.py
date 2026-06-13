"""回测引擎"""

import bisect
import re
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ..common.cost import CostModel
from ..common.date_utils import to_trade_date_str
from ..common.trade_status import is_tradeable
from ..data.loader import DataLoader
from ..execution.pending_order import PendingOrderManager
from ..risk.equity_curve import EquityCurveConfig, EquityCurveMonitor
from ..risk.stop_loss import StopLossConfig, StopLossMonitor
from ..risk.weakness_exit import WeaknessExitConfig, WeaknessExitMonitor
from ..risk.stop_loss_checker import check_positions_stop_loss
from ..signals.base import Signal
from ..universe.base import Universe
from .holding_strength import HoldingStrengthScorer, HoldingStrengthWeights


def _format_rebalance_decision_summary(
    decision_trace: Dict,
    execution_date: Optional[pd.Timestamp] = None,
    tranche_tag: str = "",
) -> str:
    """格式化统一的调仓决策摘要日志。"""

    def _to_optional_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            if np.isnan(value):
                return None
        except TypeError:
            return None
        return float(value)

    def _compact_summary(summary: Optional[str]) -> str:
        if not summary:
            return "-"

        compact = str(summary).strip()
        compact = compact.replace("，", ", ")
        compact = compact.replace("达到阈值 ", "档=")
        compact = compact.replace("未达到首档阈值 ", "未达首档=")
        compact = compact.replace("目标仓位 ", "目标=")
        compact = compact.replace("base_after_ma250=", "base=")
        compact = compact.replace("final_after_atr=", "after_atr=")
        compact = re.sub(r",?\s*市场层=[0-9.]+%", "", compact)
        compact = re.sub(r"\s+", " ", compact)
        return compact

    def _fmt_exposure(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        try:
            if np.isnan(value):
                return "N/A"
        except TypeError:
            return "N/A"
        return f"{float(value):.1%}"

    signal_date = decision_trace.get("signal_date")
    signal_label = signal_date.date() if isinstance(signal_date, pd.Timestamp) else signal_date
    execution_label = (
        execution_date.date() if isinstance(execution_date, pd.Timestamp) else execution_date
    )
    candidate_count = decision_trace.get("candidate_count")
    target_n = decision_trace.get("target_n", 0)
    queued = bool(decision_trace.get("queued", execution_date is not None))

    signal_gate = decision_trace.get("signal_gate", {})
    ect = decision_trace.get("ect", {})
    ma250 = decision_trace.get("ma250", {})
    market_regime = decision_trace.get("market_regime", {})
    dynamic_topn = decision_trace.get("dynamic_topn", {})

    signal_gate_exposure = _to_optional_float(signal_gate.get("exposure", 1.0))
    signal_gate_quality_exposure = _to_optional_float(signal_gate.get("quality_exposure", 1.0))
    ect_exposure = _to_optional_float(ect.get("exposure", 1.0))
    ma250_exposure = _to_optional_float(ma250.get("exposure", 1.0))
    market_regime_exposure = _to_optional_float(market_regime.get("exposure", 1.0))
    market_layer_exposure = _to_optional_float(decision_trace.get("market_layer_exposure", 1.0))
    computed_exposure = None
    if (
        signal_gate_exposure is not None
        and ect_exposure is not None
        and market_layer_exposure is not None
    ):
        computed_exposure = signal_gate_exposure * ect_exposure * market_layer_exposure
    final_target_exposure = _to_optional_float(
        decision_trace.get(
            "final_target_exposure",
            computed_exposure,
        )
    )

    header = f"{tranche_tag}调仓决策摘要: 信号日 {signal_label}"
    if execution_label is not None:
        execution_text = execution_label
    else:
        execution_text = "-"

    candidate_text = candidate_count if candidate_count is not None else "N/A"

    # 动态 Top-N 显示
    if dynamic_topn.get("enabled") and dynamic_topn.get("reason"):
        base_n = dynamic_topn.get("base_n", target_n)
        effective_n = dynamic_topn.get("effective_n", target_n)
        topn_text = f"目标={effective_n}(基准={base_n}, {dynamic_topn['reason']})"
    else:
        topn_text = f"目标={target_n}"

    final_action = "入队" if queued else "不入队"

    # 构建门控行：综合显示 composite exposure + quality exposure + 最终门控系数
    gate_composite_exposure = signal_gate_exposure if signal_gate_exposure is not None else 1.0
    gate_quality_exposure = (
        signal_gate_quality_exposure if signal_gate_quality_exposure is not None else 1.0
    )
    gate_final_exposure = gate_composite_exposure * gate_quality_exposure

    if (
        final_target_exposure is not None
        and final_target_exposure <= 0
        and signal_gate_exposure is not None
        and signal_gate_exposure <= 0
    ):
        final_detail = "门控阻断, 不入队"
    else:
        final_detail = final_action
    gate_summary_text = _compact_summary(signal_gate.get("summary", "未启用"))
    # 质量监控相关字段（始终显示）
    quality_score = signal_gate.get("quality_score")
    quality_warmup = signal_gate.get("quality_warmup_remaining")
    if quality_score is not None:
        # 预热期中用特殊标记，正常期显示 hit_rate 和仓位系数
        if quality_warmup is not None and quality_warmup > 0:
            quality_text = f"质量=预热中(剩{quality_warmup}期)"
        else:
            quality_text = (
                f"质量=hit_rate={quality_score:.2f}, 系数={_fmt_exposure(gate_quality_exposure)}"
            )
        gate_text = (
            f"门控={_fmt_exposure(gate_final_exposure)}"
            f"[{gate_summary_text}"
            f" | {quality_text}]"
        )
    else:
        gate_text = (
            f"门控={_fmt_exposure(gate_final_exposure)}"
            f"[{gate_summary_text}]"
        )

    return (
        f"{header} | 执行={execution_text} | 候选={candidate_text} | {topn_text}"
        f" | {gate_text}"
        f" | ECT={_fmt_exposure(ect_exposure)}"
        f"[{_compact_summary(ect.get('summary', '未启用'))}]"
        f" | MA250/ATR={_fmt_exposure(ma250_exposure)}"
        f"[{_compact_summary(ma250.get('summary', '未启用'))}]"
        f" | 市场={_fmt_exposure(market_regime_exposure)}"
        f"[{_compact_summary(market_regime.get('summary', '未启用'))}]"
        f" | 最终={_fmt_exposure(final_target_exposure)}[{final_detail}]"
    )


def _format_buy_execution_stock_list(entries: List[Dict], include_reason: bool = False) -> str:
    """格式化调仓买入股票列表。"""

    if not entries:
        return "-"

    parts = []
    for entry in entries:
        stock = str(entry["stock"])
        reason = entry.get("reason")

        stock_text = stock
        if include_reason and reason:
            stock_text = f"{stock_text}({reason})"
        parts.append(stock_text)

    return ", ".join(parts)


def _sum_buy_execution_weights(entries: List[Dict]) -> float:
    """汇总调仓买入资金占比。"""

    return float(sum(float(entry.get("weight", 0.0)) for entry in entries))


def _format_buy_execution_summary(
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    planned_buys: List[Dict],
    successful_buys: List[Dict],
    failed_buys: List[Dict],
    inherited_position_count: int,
    inherited_position_weight: float,
    tranche_tag: str = "",
) -> str:
    """格式化调仓买入执行汇总日志。"""

    planned_weight = _sum_buy_execution_weights(planned_buys)
    successful_weight = _sum_buy_execution_weights(successful_buys)
    failed_weight = _sum_buy_execution_weights(failed_buys)

    return (
        f"\n"
        f"{tranche_tag}调仓买入汇总: 执行日 {execution_date.date()} | 信号日 {signal_date.date()} | "
        f"计划={len(planned_buys)} | 计划资金占比={planned_weight:.2%} | "
        f"继承上轮={inherited_position_count} | 继承资金占比={inherited_position_weight:.2%} | "
        f"成功={len(successful_buys)} | 失败={len(failed_buys)}\n"
        f"{tranche_tag}成功仓位: 数量={len(successful_buys)} | "
        f"股票=[{_format_buy_execution_stock_list(successful_buys)}] | "
        f"资金占比={successful_weight:.2%}\n"
        f"{tranche_tag}失败仓位: 数量={len(failed_buys)} | "
        f"股票=[{_format_buy_execution_stock_list(failed_buys, include_reason=True)}] | "
        f"资金占比={failed_weight:.2%}\n"
    )


class BacktestEngine:
    """回测引擎

    执行回测流程，生成净值曲线和交易记录

    交易规则：
    - T 日生成信号
    - T+1 日收盘价买入
    - 持有期到期卖出：T+n 日开盘价卖出（n 为持有期）
    - 条件卖出（亏损提前换出、整体止盈）：Tn 日检查 → Tn+1 日开盘价卖出
    - 卖出时机可配置：开盘价（默认）或收盘价
    """

    # 常量：每年交易日数量（用于年化波动率计算）
    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        universe: Universe,
        signal: Signal,
        initial_capital: float = 1000000.0,
        cost_model: Optional[CostModel] = None,
        rebalance_freq: int = 5,
        holding_period: Optional[int] = None,
        verbose: bool = True,
        enable_risk_budget: bool = False,
        vol_window: int = 20,
        vol_epsilon: float = 0.001,
        enable_pending_order: bool = True,
        max_retry_count: int = 5,
        max_retry_days: int = 10,
        stop_loss_config: Optional[StopLossConfig] = None,
        sell_timing: str = "open",
        enable_position_completion: bool = True,
        completion_window_days: int = 3,
        equity_curve_config: Optional[EquityCurveConfig] = None,
        data_storage=None,  # 新增：数据存储实例（用于读取 raw/suspend 数据）
        max_weight_per_stock: Optional[float] = None,  # 新增：单股最大权重
        max_per_industry: Optional[int] = None,  # 新增：单行业最大持仓数量
        stock_basic: Optional[pd.DataFrame] = None,  # 新增：股票基本信息（用于行业约束）
        stagger_tranches: int = 1,  # 分批调仓批次数（1=不分批）
        enable_profit_based_holding: bool = False,  # 是否启用盈亏动态持仓时长
        early_exit_loss_threshold: float = -0.05,  # 亏损提前换出阈值（达到持有期比例后生效）
        early_exit_holding_ratio: float = 0.6,  # 亏损提前换出最早触发时点（持有期比例）
        profit_extension_threshold: float = 0.05,  # 盈利延续持有阈值
        profit_extension_days: int = 5,  # 盈利延续持有额外天数（交易日）
        profit_extension_mode: str = "pnl",  # "pnl"=原浮盈单维判据 | "strength"=多维度强势度 | "disabled"=关闭
        profit_extension_strength_threshold: float = 0.6,  # strength 模式下的延续阈值 [0,1]
        profit_extension_strength_weights: Optional[Dict[str, float]] = None,  # 强势度 5 维度权重
        use_atr_for_early_exit: bool = False,  # 是否用个股 ATR 动态替代固定止损阈值
        atr_multiplier: float = 2.0,  # ATR 倍数（亏损超过 N×ATR% 时触发提前换出）
        early_exit_mode: str = "disabled",  # "disabled"=原硬卖 | "strength_veto"=二次确认门控
        early_exit_strength_protect_threshold: float = 0.55,  # strength >= 此值时否决卖出
        early_exit_max_reprieves: int = 2,  # 单只股票最多缓刑次数
        take_profit_threshold: Optional[
            float
        ] = None,  # 整体持仓止盈阈值（None=禁用，如0.15=整体浮盈15%止盈）
        take_profit_refill: bool = True,  # 整体止盈后是否触发自动补位买入
        time_stop_loss_enabled: bool = True,  # 是否启用时间止损（持仓超限未达盈利要求提前换出）
        time_stop_loss_days: int = 15,  # 时间止损触发最低持有天数（交易日）
        time_stop_loss_profit_ratio: float = -0.02,  # 时间止损利润阈值（当前盈亏低于此值触发）
        weakness_exit_enabled: bool = False,  # 是否启用表现弱势退出
        weakness_exit_threshold: float = 0.6,  # 弱势评分触发阈值
        weakness_exit_consecutive_days: int = 3,  # 需连续弱势天数
        weakness_exit_min_holding_days: int = 5,  # 最低持有天数
        weakness_exit_weights: str = "30,25,25,20",  # 4维度权重
        weakness_exit_industry_filter: bool = False,  # 弱势行业过滤
        weakness_exit_industry_bottom_pct: float = 0.3,  # 行业底部阈值
        signal_gate_quality_enabled: bool = False,  # 是否启用滚动模型质量监控
        signal_gate_quality_window: int = 5,  # 回看调仓周期数
        signal_gate_quality_threshold: float = 0.4,  # 最低滚动hit rate
        signal_gate_quality_halflife: int = 3,  # EWM半衰期
        signal_gate_dynamic_topn: bool = False,  # 是否启用动态Top-N
        signal_gate_topn_high_multiplier: float = 0.6,  # 高置信度缩减系数（<1）
        signal_gate_topn_low_multiplier: float = 1.5,  # 低置信度扩大系数（>1）
        holding_bonus_enabled: bool = False,  # 是否启用持仓保留奖励（降低换手率）
        holding_bonus_sigma: float = 0.5,  # 保留奖励幅度（截面分数标准差的倍数）
        position_sizing: str = "equal",  # 仓位管理: equal|score|kelly|half_kelly
        kelly_vol_window: int = 60,  # Kelly 波动率估计窗口（交易日）
        kelly_max_leverage: float = 0.25,  # 单只股票 Kelly 仓位上限（占总资产）
        enable_early_rebalance_on_empty: bool = True,  # 空仓时是否提前触发新一轮调仓
        min_buy_value_ratio: float = 0.0,  # 买入后最小持仓市值占平均仓位市值比例（0=关闭）
    ):
        """初始化回测引擎

        价格口径说明：
        - 成交价格（trade_price）：使用不复权 close/open，用于计算成交金额、持仓市值、可买入数量
        - 绩效价格（pnl_price）：使用后复权 close_adj/open_adj，用于计算收益率和绩效指标

        Args:
            universe: 股票池
            signal: 信号生成器
            initial_capital: 初始资金
            cost_model: 成本模型
            rebalance_freq: 调仓频率（交易日数），必须为正整数。例如：5表示每5个交易日调仓一次
            holding_period: 持有期（交易日），None 则自动根据调仓频率设置
            verbose: 是否输出详细日志（买入/卖出操作），默认True
            enable_risk_budget: 是否启用风险预算/波动率缩放，默认False（保持向后兼容）
            vol_window: 波动率计算窗口（交易日），默认20
            vol_epsilon: 波动率缩放的最小波动率，防止除零，默认0.001
            enable_pending_order: 是否启用延迟订单功能，默认True
            max_retry_count: 延迟订单最大重试次数，默认5次
            max_retry_days: 延迟订单最大延迟天数，默认10天
            stop_loss_config: 止损配置，None 表示不启用止损功能（默认）
            sell_timing: 卖出时机，'open' 表示开盘价卖出（默认），'close' 表示收盘价卖出
            enable_position_completion: 是否启用仓位补齐功能，默认True
            completion_window_days: 补齐窗口期（交易日），默认3天
            equity_curve_config: ECT（权益曲线交易）配置，None 表示不启用（默认）
            data_storage: 数据存储实例（用于读取 raw/suspend 数据），如不提供则在需要时创建
            max_weight_per_stock: 单个股票最大权重（0-1），None 表示不启用限权，启用后会在信号生成时对权重进行限制并归一化
            max_per_industry: 单个行业最大持仓数量，None 或 0 表示不启用行业约束
            stock_basic: 股票基本信息 DataFrame（用于行业约束），必须包含 ts_code 和 industry 列
            stagger_tranches: 分批调仓批次数，默认1（不分批）。设为K时将资金分成K份，
                每份错开 rebalance_freq/K 天调仓，降低单次调仓时点风险
            enable_profit_based_holding: 是否启用盈亏动态持仓时长，默认False。
                开启后在固定持有期基础上叠加两个规则：
                1. 亏损提前换出：持有达到持有期的 early_exit_holding_ratio 且亏损超过
                   early_exit_loss_threshold 时，提前换出（不等满持有期）
                2. 盈利延续持有：持有期满时若盈利超过 profit_extension_threshold，
                   允许延续持有 profit_extension_days 天（趋势跟踪）
            early_exit_loss_threshold: 亏损提前换出的盈亏率阈值，默认 -0.05（亏损5%）
            early_exit_holding_ratio: 亏损提前换出最早触发时点（占持有期比例），默认 0.6
            early_exit_mode: 亏损提前换出模式。"disabled"=原硬卖（默认），
                "strength_veto"=触发后用 HoldingStrengthScorer 二次确认，
                评分高于保护阈值时否决卖出（缓刑）
            early_exit_strength_protect_threshold: strength_veto 模式下的保护阈值，
                评分 >= 此值时否决卖出，默认 0.55
            early_exit_max_reprieves: strength_veto 模式下单只股票最多缓刑次数，
                防止无限拖延，默认 2
            profit_extension_threshold: 盈利延续持有的盈亏率阈值，默认 0.05（盈利5%）
            profit_extension_days: 盈利延续持有的额外天数（交易日），默认 5
            min_buy_value_ratio: 买入后最小持仓市值占“平均仓位市值”比例（0=关闭）。
                与纸面交易口径一致：阈值=总资产/目标持仓数*比例。
        """
        self.universe = universe
        self.signal = signal
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.data_storage = data_storage  # 保存数据存储实例
        self._suspend_calendar = None  # 停牌日历实例（延迟创建）

        # 组合构建约束参数
        self.max_weight_per_stock = max_weight_per_stock
        self.max_per_industry = (
            max_per_industry if max_per_industry and max_per_industry > 0 else None
        )
        self.stock_basic = stock_basic
        self.industry_mapping = None  # 延迟构建

        # 验证参数
        if max_weight_per_stock is not None:
            if max_weight_per_stock <= 0 or max_weight_per_stock > 1:
                raise ValueError(
                    f"max_weight_per_stock 必须在 (0, 1] 范围内，当前值: {max_weight_per_stock}"
                )

        if self.max_per_industry is not None:
            if stock_basic is None or stock_basic.empty:
                raise ValueError("启用行业约束时必须提供 stock_basic 数据")
            # 延迟导入以避免循环依赖
            from ..portfolio import load_industry_mapping

            loader = DataLoader(self.data_storage)  # 使用数据存储实例创建加载器
            shenwan_industry = loader.load_shenwan_industry()
            self.industry_mapping = load_industry_mapping(shenwan_industry, verbose=verbose)

        # 验证调仓频率
        if not isinstance(rebalance_freq, int):
            raise TypeError(f"调仓频率必须为整数类型，当前类型: {type(rebalance_freq).__name__}")
        if rebalance_freq <= 0:
            raise ValueError(f"调仓频率必须为正整数，当前值: {rebalance_freq}")

        # 验证卖出时机参数
        if sell_timing not in ["close", "open"]:
            raise ValueError(f"卖出时机参数必须为 'close' 或 'open'，当前值: {sell_timing}")

        self.rebalance_freq = rebalance_freq
        self.sell_timing = sell_timing
        self.verbose = verbose

        # 分批调仓参数
        if stagger_tranches < 1:
            raise ValueError(f"分批调仓批次数必须 >= 1，当前值: {stagger_tranches}")
        self.stagger_tranches = stagger_tranches

        # 盈亏动态持仓参数
        self.enable_profit_based_holding = enable_profit_based_holding
        self.early_exit_loss_threshold = early_exit_loss_threshold
        self.early_exit_holding_ratio = early_exit_holding_ratio
        self.profit_extension_threshold = profit_extension_threshold
        self.profit_extension_days = profit_extension_days
        self.use_atr_for_early_exit = use_atr_for_early_exit
        self.atr_multiplier = atr_multiplier
        # ── 亏损提前换出二次确认（strength_veto）──
        if early_exit_mode not in ("disabled", "strength_veto"):
            raise ValueError(
                f"early_exit_mode 必须为 disabled|strength_veto，当前值: {early_exit_mode}"
            )
        self.early_exit_mode = early_exit_mode
        self.early_exit_strength_protect_threshold = early_exit_strength_protect_threshold
        self.early_exit_max_reprieves = early_exit_max_reprieves
        self.early_exit_strength_scorer: Optional[HoldingStrengthScorer] = None
        self._early_exit_reprieve_counts: Dict[str, int] = {}
        if early_exit_mode == "strength_veto":
            # early_exit 专用权重：drawdown 归零（已知亏损，信息量低），
            # 侧重 ML 分数和动量（模型是否看好、趋势是否恢复）
            ee_weights = HoldingStrengthWeights(
                ml_score=0.35, momentum=0.30, technical=0.20,
                fund_flow=0.15, drawdown=0.00,
            )
            self.early_exit_strength_scorer = HoldingStrengthScorer(self, ee_weights)
            logger.info(
                f"亏损提前换出模式=strength_veto, "
                f"保护阈值={early_exit_strength_protect_threshold:.2f}, "
                f"最大缓刑次数={early_exit_max_reprieves}, "
                f"权重={ee_weights.normalize().as_dict()}"
            )
        # ── 盈利延续持有模式（多维度强势度评分）──
        if profit_extension_mode not in ("pnl", "strength", "disabled"):
            raise ValueError(
                f"profit_extension_mode 必须为 pnl|strength|disabled，当前值: {profit_extension_mode}"
            )
        self.profit_extension_mode = profit_extension_mode
        self.profit_extension_strength_threshold = profit_extension_strength_threshold
        self.profit_extension_strength_weights = profit_extension_strength_weights
        self.holding_strength_scorer: Optional[HoldingStrengthScorer] = None
        if profit_extension_mode == "strength":
            weights_obj = HoldingStrengthWeights.from_dict(profit_extension_strength_weights)
            self.holding_strength_scorer = HoldingStrengthScorer(self, weights_obj)
            logger.info(
                f"盈利延续持有模式=strength, 阈值={profit_extension_strength_threshold:.2f}, "
                f"权重={weights_obj.normalize().as_dict()}"
            )
        elif profit_extension_mode == "disabled":
            logger.info("盈利延续持有模式=disabled, 持有期满直接卖出")
        else:
            logger.info(
                f"盈利延续持有模式=pnl(原浮盈单维), 阈值={profit_extension_threshold:.2%}, "
                f"延续天数={profit_extension_days}"
            )
        # 整体持仓止盈参数
        self.take_profit_threshold = take_profit_threshold
        self.take_profit_refill = take_profit_refill
        # 时间止损参数
        self.time_stop_loss_enabled = time_stop_loss_enabled
        self.time_stop_loss_days = time_stop_loss_days
        self.time_stop_loss_profit_ratio = time_stop_loss_profit_ratio
        # 表现弱势退出
        self.weakness_exit_enabled = weakness_exit_enabled
        self.weakness_exit_monitor: Optional[WeaknessExitMonitor] = None
        if weakness_exit_enabled:
            w_parts = [float(x.strip()) / 100.0 for x in weakness_exit_weights.split(",")]
            if len(w_parts) != 4:
                w_parts = [0.30, 0.25, 0.25, 0.20]
            wc = WeaknessExitConfig(
                enabled=True,
                threshold=weakness_exit_threshold,
                consecutive_days=weakness_exit_consecutive_days,
                min_holding_days=weakness_exit_min_holding_days,
                weights=tuple(w_parts),
                industry_filter=weakness_exit_industry_filter,
                industry_bottom_pct=weakness_exit_industry_bottom_pct,
            )
            self.weakness_exit_monitor = WeaknessExitMonitor(wc)
        self.enable_early_rebalance_on_empty = enable_early_rebalance_on_empty
        self._last_ranked_candidates: list = []  # 最近一次调仓的候选排序列表（止盈补位用）
        self._last_signal_date: Optional[pd.Timestamp] = None  # 最近一次调仓日期
        self._last_rebalance_nav: Optional[float] = (
            None  # 上次调仓日组合净值（止盈基准 & 本调仓收益计算）
        )

        # 风险预算参数
        self.enable_risk_budget = enable_risk_budget
        self.vol_window = vol_window
        self.vol_epsilon = vol_epsilon

        # 延迟订单参数
        self.enable_pending_order = enable_pending_order
        self.pending_order_manager = None
        if enable_pending_order:
            self.pending_order_manager = PendingOrderManager(
                max_retry_count=max_retry_count,
                max_retry_days=max_retry_days,
                event_sink=self._record_pending_order_event,
            )

        # 仓位补齐参数
        self.enable_position_completion = enable_position_completion
        self.completion_window_days = completion_window_days

        # 止损配置
        self.stop_loss_config = stop_loss_config
        self.stop_loss_monitor = None
        if stop_loss_config and stop_loss_config.enabled:
            self.stop_loss_monitor = StopLossMonitor(stop_loss_config)

        # ECT 配置
        self.equity_curve_config = equity_curve_config
        self.equity_curve_monitor = None
        if equity_curve_config and equity_curve_config.enabled:
            self.equity_curve_monitor = EquityCurveMonitor(equity_curve_config)

        # 持有期逻辑：如果未指定，与调仓频率保持一致
        if holding_period is None:
            self.holding_period = self.rebalance_freq
        else:
            self.holding_period = holding_period  # 修复：应使用传入的 holding_period

        # 回测状态
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = (
            {}
        )  # {股票代码: {shares, buy_date, buy_trade_price, buy_pnl_price, buy_cost_cash}}
        self.pending_signals: Dict[pd.Timestamp, Dict] = {}  # {信号日期: {股票: 权重}}
        self.pending_stop_loss_sells: Dict[str, Dict] = (
            {}
        )  # {股票代码: {trigger_date, reason, trigger_type}} 待止损卖出队列
        self.pending_condition_sells: Dict[str, Dict] = (
            {}
        )  # {股票代码: {trigger_date, sell_type}} 待条件卖出队列（T0 触发、T1 执行）
        self._pending_take_profit_info: Optional[Dict] = None  # 止盈元数据（延迟到执行日处理）
        self._cycle_anchor_idx: int = 0  # 当前调仓周期起点 idx（用于 cycle_day 日志显示）
        self.portfolio_values: List[Dict] = []  # 组合价值历史
        self.trades: List[Dict] = []  # 交易记录

        # 仓位补齐状态跟踪
        # {调仓日期: {未成交股票列表, 目标数量, 候选列表, 剩余权重字典}}
        self.unfilled_slots: Dict[pd.Timestamp, Dict] = {}
        # 补齐统计
        self.completion_stats = {
            "total_unfilled": 0,  # 累计未满仓次数
            "total_completed": 0,  # 累计补齐成功次数
            "total_abandoned": 0,  # 累计放弃补齐次数
            "completion_attempts": 0,  # 累计补齐尝试次数
        }
        self._deferred_day_logs: List[Dict[str, str]] = []
        self._daily_warning_items: Dict[str, List[Dict]] = {}
        self.confidence_gate_history: List[Dict] = []  # 信号置信度门控历史

        # ── 滚动模型质量监控 ──
        self.signal_gate_quality_enabled = signal_gate_quality_enabled
        self.signal_gate_quality_window = signal_gate_quality_window
        self.signal_gate_quality_threshold = signal_gate_quality_threshold
        self.signal_gate_quality_halflife = signal_gate_quality_halflife
        # ── 动态 Top-N ──
        self.signal_gate_dynamic_topn = signal_gate_dynamic_topn
        self.signal_gate_topn_high_multiplier = signal_gate_topn_high_multiplier
        self.signal_gate_topn_low_multiplier = signal_gate_topn_low_multiplier
        # ── 换手率约束（持仓保留奖励）──
        self.holding_bonus_enabled = holding_bonus_enabled
        self.holding_bonus_sigma = holding_bonus_sigma
        # Kelly 仓位管理参数
        if position_sizing not in ("equal", "score", "kelly", "half_kelly"):
            raise ValueError(
                f"position_sizing 必须为 equal|score|kelly|half_kelly，"
                f"当前值: {position_sizing}"
            )
        self.position_sizing = position_sizing
        self.kelly_vol_window = kelly_vol_window
        self.kelly_max_leverage = kelly_max_leverage
        if min_buy_value_ratio < 0:
            raise ValueError(
                f"min_buy_value_ratio 必须 >= 0，当前值: {min_buy_value_ratio}"
            )
        self.min_buy_value_ratio = float(min_buy_value_ratio)
        self._normalize_log_count = 0  # 权重诊断日志计数，只打印前5次
        if position_sizing in ("kelly", "half_kelly"):
            logger.info(
                f"仓位管理模式={position_sizing}, 波动率窗口={kelly_vol_window}, "
                f"单股上限={kelly_max_leverage:.2f}"
            )
        self._prediction_quality_history: List[Dict] = []
        self._rolling_quality_score: float = 1.0  # 默认满分（预热期不干预）
        self._quality_warmup_remaining: int = signal_gate_quality_window  # 预热计数
        # 记录每次信号日的选股和预测均值，用于持仓结束后评估
        self._signal_tracking: Dict[str, Dict] = {}  # {信号日期str: {stocks, predicted_mean}}

        # 价格索引（在 run 时初始化）
        self.trade_price_index: Optional[pd.Series] = None  # 成交价格（不复权 close）
        self.pnl_price_index: Optional[pd.Series] = None  # 绩效价格（后复权 close_adj）
        self.trade_price_open_index: Optional[pd.Series] = None  # 开盘成交价格（不复权 open）
        self.pnl_price_open_index: Optional[pd.Series] = None  # 开盘绩效价格（后复权 open_adj）

        # 存储价格数据用于交易状态检查
        self.price_data_cache: Optional[pd.DataFrame] = None

        stagger_info = f", 分批调仓={self.stagger_tranches}批" if self.stagger_tranches > 1 else ""
        logger.info(
            f"回测引擎初始化完成: 初始资金={initial_capital}, "
            f"调仓频率={self.rebalance_freq}, 持有期={self.holding_period}天{stagger_info}, "
            f"卖出时机={self.sell_timing}, "
            f"风险预算={'启用' if enable_risk_budget else '禁用'}, "
            f"延迟订单={'启用' if enable_pending_order else '禁用'}, "
            f"仓位补齐={'启用' if enable_position_completion else '禁用'}, "
            f"最小买入阈值={'关闭' if self.min_buy_value_ratio <= 0 else f'{self.min_buy_value_ratio:.2f}'}, "
            f"补齐窗口={completion_window_days}天, "
            f"止损功能={'启用' if (stop_loss_config and stop_loss_config.enabled) else '禁用'}, "
            f"时间止损={'启用' if time_stop_loss_enabled else '禁用'}"
            f"({time_stop_loss_days}天, {time_stop_loss_profit_ratio:.0%})" if time_stop_loss_enabled else "", ", "
            f"弱势退出={'启用' if weakness_exit_enabled else '禁用'}"
            f"(阈值{weakness_exit_threshold:.0%})" if weakness_exit_enabled else "", ", "
            f"空仓提前调仓={'启用' if enable_early_rebalance_on_empty else '禁用'}, "
            f"详细日志={'开启' if verbose else '关闭'}"
        )
        sell_price_type = "开盘价" if self.sell_timing == "open" else "收盘价"
        logger.info(
            f"交易规则: T日生成信号 -> T+1日收盘价买入 -> 满{self.holding_period}天后T0生成卖出信号 -> 下一交易日{sell_price_type}卖出"
        )
        logger.info(f"价格口径: 成交使用不复权 close/open, 绩效使用后复权 close_adj/open_adj")

    def _get_suspend_calendar(self):
        """获取停牌日历实例（延迟创建）"""
        if self._suspend_calendar is None:
            from ..common.suspend_calendar import SuspendCalendar
            from ..data import Storage

            # 如果没有提供 data_storage，创建一个默认实例
            if self.data_storage is None:
                self.data_storage = Storage()

            self._suspend_calendar = SuspendCalendar(self.data_storage)

        return self._suspend_calendar

    def run(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """运行回测

        Args:
            start_date: 开始日期
            end_date: 结束日期
            trading_dates: 交易日列表
            price_data: 价格数据，需包含 ts_code, trade_date, close, close_adj（可选）

        Returns:
            净值曲线DataFrame
        """
        import time

        logger.info(f"开始回测: {start_date.date()} 至 {end_date.date()}")

        # 筛选回测期间的交易日
        trading_dates = [d for d in trading_dates if start_date <= d <= end_date]
        total_days = len(trading_dates)

        # 创建日期到索引的映射，优化查找效率
        date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}

        # 准备价格索引（使用 MultiIndex，替代嵌套字典）
        self._prepare_price_index(price_data)

        # 缓存价格数据用于交易状态检查
        self.price_data_cache = price_data

        # 获取调仓日期（信号生成日期）→ {日期: tranche_idx}
        signal_dates = self._get_rebalance_dates(trading_dates)

        if self.stagger_tranches > 1:
            logger.info(
                f"数据准备完成, 调仓日期共 {len(signal_dates)} 天"
                f"（{self.stagger_tranches} 批分批调仓）"
            )
        else:
            logger.info(f"数据准备完成, 调仓日期共 {len(signal_dates)} 天")

        # 记录开始时间
        start_time = time.time()
        deferred_sink_id = logger.add(
            self._collect_deferred_log,
            format="{message}",
            level="DEBUG",
            colorize=False,
            filter=lambda record: record["extra"].get("_defer_emit", False),
        )

        try:
            # 按日推进
            # _cycle_anchor_idx 是当前调仓周期的"第1天"在 trading_dates 中的 idx
            # 初始为 0（第一天即第1轮的第1天）；每次信号成功入队列时重置为信号日 idx
            # 这样门控连续阻断的空仓期不会推进 cycle_day
            self._cycle_anchor_idx = 0
            cycle_separator = (
                "\n================================================ 新一轮回测 ================================================="
            )
            for idx, date in enumerate(trading_dates):
                # 新一轮首日：输出分隔线（在所有业务日志之前）
                if idx == self._cycle_anchor_idx:
                    self._emit_immediate_log("INFO", cycle_separator)
                cycle_day = idx - self._cycle_anchor_idx + 1
                trade_start_idx = len(self.trades)
                self._deferred_day_logs = []
                self._reset_daily_warning_items()

                with logger.contextualize(_defer_emit=True):
                    # 处理延迟订单（先处理延迟订单，再处理新信号）
                    if self.enable_pending_order:
                        self._process_pending_orders(date)

                    # 检查止损（T 日检查，T+1 日执行卖出）
                    if self.stop_loss_monitor:
                        self._check_stop_loss(date, trading_dates, date_to_idx)

                    # 判断是否为信号生成日
                    if date in signal_dates:
                        tranche_idx = signal_dates[date]
                        self._generate_signal(
                            date,
                            trading_dates,
                            price_data,
                            date_to_idx,
                            tranche_idx=tranche_idx,
                        )
                        # 信号成功入队列 → 本日即为新周期第1天，更新 anchor 并输出分隔线
                        if date in self.pending_signals and idx != self._cycle_anchor_idx:
                            self._cycle_anchor_idx = idx
                            cycle_day = 1
                            self._emit_immediate_log("INFO", cycle_separator)

                        # 调仓日同步生成卖出信号：将当前非保护持仓排队到 T+1 卖出，
                        # 使卖出与买入在同一交易日执行，避免卖出滞后一天。
                        if date in self.pending_signals:
                            self._queue_rebalance_sells(date, trading_dates, date_to_idx)

                    # @2026/01/18: 改为先卖出再买入, 避免当天买入的股票被误判为达到持有期而卖出
                    # 执行止损卖出（Tn+1 执行）
                    if self.stop_loss_monitor:
                        self._execute_pending_stop_loss_sells(date, trading_dates, date_to_idx)

                    # 执行条件卖出（Tn+1 执行：亏损提前换出、整体止盈、持有期到期）
                    self._execute_pending_condition_sells(date, trading_dates, date_to_idx)

                    # 检查卖出条件并生成 T0 卖出信号
                    # - 持有期到期 / 盈利延续到期：写入 pending_condition_sells，Tn+1 执行
                    # - 亏损提前换出 / 整体止盈：写入 pending_condition_sells，Tn+1 执行
                    self._check_and_sell(date, trading_dates, date_to_idx)

                    # 执行待执行的买入操作（Tn+1）
                    self._execute_pending_buys(date, trading_dates, date_to_idx)

                    # 空仓提前调仓 / 盈利延续拖尾提前调仓：
                    # 场景 A（空仓）：持仓全部卖出，资金闲置 → 立即触发新一轮信号
                    # 场景 B（盈利延续拖尾）：cycle_day >= holding_period 但仍有残留持仓（通常为盈利延续）
                    #   → 若"残留持仓占比 + 新信号目标仓位 ≤ 100%"，则提前启动新一轮；否则继续等待
                    early_rebalance_guards_ok = (
                        self.enable_early_rebalance_on_empty
                        and not self.pending_signals
                        and not any(
                            slot_info.get("unfilled_count", 0) > 0
                            for slot_info in self.unfilled_slots.values()
                        )
                        and date not in signal_dates
                    )

                    is_empty_position = not self.positions
                    is_holding_period_exceeded = (
                        bool(self.positions) and cycle_day >= self.holding_period
                    )

                    if early_rebalance_guards_ok and (is_empty_position or is_holding_period_exceeded):
                        if not is_empty_position:
                            # 盈利延续拖尾场景：打印当前残留持仓占比
                            current_nav = self._calculate_portfolio_value(date)
                            residual_market_value = current_nav - self.current_capital
                            residual_ratio = (
                                residual_market_value / current_nav if current_nav > 0 else 0.0
                            )
                        else:
                            residual_ratio = 0.0

                        # 快照历史状态：提前调仓若未真正入队列则回滚，避免污染门控/质量计算基准
                        # 仅快照评估过程会追加的字段，保证启用/禁用该开关对正常调仓日的门控计算完全一致
                        gate_history_snapshot = self._snapshot_early_rebalance_state(date)

                        self._generate_signal(
                            date,
                            trading_dates,
                            price_data,
                            date_to_idx,
                            tranche_idx=0,
                        )

                        # 盈利延续拖尾场景：需额外校验 "残留仓位 + 新信号仓位 ≤ 100%"
                        # 若不满足，撤回本次信号，继续等待残留持仓到期
                        signal_accepted = date in self.pending_signals
                        if signal_accepted and is_holding_period_exceeded:
                            current_nav = self._calculate_portfolio_value(date)
                            residual_market_value = current_nav - self.current_capital
                            residual_ratio = (
                                residual_market_value / current_nav if current_nav > 0 else 0.0
                            )
                            new_signal_weight_sum = sum(
                                self.pending_signals[date].get("signals", {}).values()
                            )
                            combined_ratio = residual_ratio + new_signal_weight_sum
                            if combined_ratio > 1.0 + 1e-9:
                                # 超过上限，撤回信号
                                del self.pending_signals[date]
                                signal_accepted = False
                                self._record_early_rebalance_summary(
                                    "拖尾拒绝",
                                    f"残留{residual_ratio:.1%}+新信号{new_signal_weight_sum:.1%}"
                                    f"={combined_ratio:.1%}>100%",
                                )
                            else:
                                self._record_early_rebalance_summary(
                                    "拖尾通过",
                                    f"残留{residual_ratio:.1%}+新信号{new_signal_weight_sum:.1%}"
                                    f"={combined_ratio:.1%}",
                                )

                        # 信号未真正入队列（门控阻断或拖尾拒绝）→ 回滚历史快照，避免污染基准
                        if not signal_accepted:
                            self._restore_early_rebalance_state(date, gate_history_snapshot)
                            if is_empty_position:
                                self._record_early_rebalance_summary(
                                    "空仓未入队",
                                    "无持仓, 新信号未入队",
                                )

                        # 信号真正入队列后，才更新节奏并清理预定调仓日
                        if signal_accepted:
                            if is_empty_position:
                                self._record_early_rebalance_summary(
                                    "空仓触发",
                                    "无持仓, 新信号入队",
                                )
                            # 清除接下来一个持有期内的原预定调仓日，避免"刚买完又调仓"
                            next_rebalance_cutoff_idx = idx + self.holding_period
                            stale_dates = [
                                d
                                for d in list(signal_dates.keys())
                                if idx < date_to_idx.get(d, -1) <= next_rebalance_cutoff_idx
                            ]
                            for d in stale_dates:
                                del signal_dates[d]
                            if stale_dates:
                                logger.info(
                                    f"  已清除未来 {len(stale_dates)} 个预定调仓日（至 {stale_dates[-1].date()}），"
                                    f"避免重复调仓"
                                )
                            # 信号成功入队列 → 本日即为新周期第1天，更新 anchor 并输出分隔线
                            if idx != self._cycle_anchor_idx:
                                self._cycle_anchor_idx = idx
                                cycle_day = 1
                                self._emit_immediate_log("INFO", cycle_separator)

                    # 处理仓位补齐（在补齐窗口期内尝试补齐未满仓位）
                    if self.enable_position_completion:
                        self._process_position_completion(date, trading_dates, price_data, date_to_idx)

                    # 计算当日组合价值
                    portfolio_value = self._calculate_portfolio_value(date)

                trading_days = idx + 1
                buy_count, sell_count, trade_detail_logs = self._build_daily_trade_log(
                    date=date,
                    trade_start_idx=trade_start_idx,
                    date_to_idx=date_to_idx,
                )
                self._emit_daily_summary_log(
                    self._format_daily_progress_log(
                        date=date,
                        trading_days=trading_days,
                        total_days=total_days,
                        cycle_day=cycle_day,
                        portfolio_value=portfolio_value,
                        buy_count=buy_count,
                        sell_count=sell_count,
                    )
                )
                self._flush_deferred_day_logs(
                    predicate=lambda record: "调仓决策摘要:" in str(record.get("message", ""))
                )
                for trade_detail_log in trade_detail_logs:
                    self._emit_immediate_log("INFO", f"  {trade_detail_log}")
                signal_count_log = self._build_daily_signal_log(date)
                if signal_count_log:
                    self._emit_immediate_log("INFO", f"  {signal_count_log}")
                for warning_log in self._build_daily_warning_logs():
                    self._emit_immediate_log("INFO", f"  {warning_log}")
                self._flush_deferred_day_logs()

                self.portfolio_values.append(
                    {
                        "date": date,
                        "portfolio_value": portfolio_value,
                        "capital": self.current_capital,
                        "market_value": portfolio_value - self.current_capital,
                    }
                )
        finally:
            logger.remove(deferred_sink_id)
            self._deferred_day_logs = []

        # 生成净值曲线
        nav_df = self._generate_nav_curve()

        total_time = time.time() - start_time
        logger.info(
            f"回测完成: 共 {len(trading_dates)} 个交易日, {len(self.trades)} 笔交易, 总耗时 {total_time:.1f}秒"
        )

        # 输出延迟订单统计
        if self.enable_pending_order and self.pending_order_manager:
            stats = self.pending_order_manager.get_statistics()
            logger.info(
                f"延迟订单统计: 累计添加 {stats['total_added']}, "
                f"成功执行 {stats['total_succeeded']}, "
                f"过期放弃 {stats['total_expired']}, "
                f"剩余待处理 {stats['pending']}"
            )

        # 输出仓位补齐统计
        if self.enable_position_completion:
            logger.info(
                f"仓位补齐统计: 累计未满仓 {self.completion_stats['total_unfilled']} 次, "
                f"补齐成功 {self.completion_stats['total_completed']} 次, "
                f"补齐尝试 {self.completion_stats['completion_attempts']} 次, "
                f"放弃补齐 {self.completion_stats['total_abandoned']} 次"
            )

        return nav_df

    def _get_target_position_count(self) -> int:
        """获取组合当前期望的目标持仓数。"""
        target_n = getattr(self.signal, "top_n", None)
        if isinstance(target_n, int) and target_n > 0:
            return target_n * self.stagger_tranches
        return len(self.positions)

    def _collect_deferred_log(self, message) -> None:
        """收集单个交易日内暂缓输出的日志。"""
        record = message.record
        self._deferred_day_logs.append(
            {
                "level": record["level"].name,
                "message": record["message"],
            }
        )

    def _emit_immediate_log(self, level: str, message: str, colors: bool = False) -> None:
        """绕过单日缓冲，立即输出日志。"""
        bound_logger = logger.bind(_defer_emit=False)
        if colors:
            bound_logger.opt(colors=True).log(level.upper(), message)
            return
        bound_logger.log(level.upper(), message)

    def _emit_daily_summary_log(self, message: str) -> None:
        """输出每日顶格彩色总结行。"""
        self._emit_immediate_log(
            "INFO",
            f"<bold><cyan>{message}</cyan></bold>",
            colors=True,
        )

    def _normalize_deferred_log_message(self, message: str) -> Optional[str]:
        """将缓冲日志统一整理为两空格缩进格式。"""
        lines = [line.rstrip() for line in str(message).splitlines() if line.strip()]
        if not lines:
            return None
        return "\n".join(f"  {line.lstrip()}" for line in lines)

    def _flush_deferred_day_logs(
        self,
        predicate: Optional[Callable[[Dict[str, str]], bool]] = None,
    ) -> None:
        """在每日总结之后统一回放明细日志。"""
        remaining_logs: List[Dict[str, str]] = []
        for record in self._deferred_day_logs:
            if predicate is not None and not predicate(record):
                remaining_logs.append(record)
                continue
            normalized = self._normalize_deferred_log_message(record["message"])
            if normalized:
                self._emit_immediate_log(record["level"], normalized)
        self._deferred_day_logs = remaining_logs if predicate is not None else []

    @staticmethod
    def _format_compact_items(items: List[str], limit: int = 4) -> str:
        """压缩同类股票列表，避免单行过长。"""
        if not items:
            return "-"
        visible = items[:limit]
        suffix = f", ...+{len(items) - limit}" if len(items) > limit else ""
        return ", ".join(visible) + suffix

    @staticmethod
    def _format_trade_cash_wan(trade: Dict) -> str:
        """格式化买入现金支出（万元）。"""
        amount = float(trade.get("amount", 0.0) or 0.0)
        cost = float(trade.get("cost", 0.0) or 0.0)
        total_cash = max(amount + cost, 0.0)
        return f"{total_cash / 10000:.1f}w"

    def _build_daily_trade_log(
        self,
        date: pd.Timestamp,
        trade_start_idx: int,
        date_to_idx: Dict,
    ) -> Tuple[int, int, List[str]]:
        """构建当日实际成交摘要。"""
        day_trades = [
            trade for trade in self.trades[trade_start_idx:] if trade.get("date") == date
        ]
        if not day_trades:
            return 0, 0, []

        buy_items: List[str] = []
        sell_items: List[str] = []
        current_idx = date_to_idx.get(date)

        for trade in day_trades:
            stock = str(trade.get("stock", "-"))
            action = trade.get("action")
            if action == "buy":
                buy_items.append(f"{stock}({self._format_trade_cash_wan(trade)})")
                continue

            if action != "sell":
                continue

            buy_date = trade.get("buy_date")
            holding_days = 0
            if isinstance(buy_date, pd.Timestamp) and current_idx is not None:
                buy_idx = date_to_idx.get(buy_date)
                if buy_idx is not None:
                    holding_days = max(current_idx - buy_idx, 0)

            profit_pct = float(trade.get("pnl_profit_pct", 0.0) or 0.0)
            sell_items.append(f"{stock}({holding_days}d,{profit_pct:+.1%})")

        if not buy_items and not sell_items:
            return 0, 0, []

        lines = []
        if sell_items:
            lines.append(f"交易: 卖{len(sell_items)}[{', '.join(sell_items)}]")
        if buy_items:
            lines.append(f"交易: 买{len(buy_items)}[{', '.join(buy_items)}]")

        return len(buy_items), len(sell_items), lines

    def _format_profit_extension_summary(self, items: List[Dict]) -> str:
        """压缩盈利延续持有日志。"""
        details = []
        for item in items:
            score = item.get("score")
            score_suffix = f",{score:.2f}" if score is not None else ""
            details.append(
                f"{item['stock']}({item['holding_days']}d,{item['profit_rate']:+.1%}{score_suffix})"
            )
        return (
            f"盈利延续[{self.profit_extension_mode}] {len(items)}只: "
            f"{self._format_compact_items(details, limit=6)}"
        )

    def _format_completion_summary(
        self,
        success_items: List[Dict],
        delayed_slots: List[str],
        remaining_count: int,
    ) -> str:
        """压缩仓位补齐日志。"""
        parts = []
        if success_items:
            success_labels = []
            for item in success_items:
                slot = item["slot"]
                buy = item["buy"]
                success_labels.append(buy if slot == buy else f"{slot}→{buy}")
            parts.append(
                f"成功{len(success_items)}[{self._format_compact_items(success_labels)}]"
            )
        if delayed_slots:
            parts.append(f"延迟{len(delayed_slots)}[{self._format_compact_items(delayed_slots)}]")
        if remaining_count > 0:
            parts.append(f"待补{remaining_count}")
        return f"补齐: {' | '.join(parts)}"

    def _reset_daily_warning_items(self) -> None:
        """重置当日需汇总展示的压缩事件。"""
        self._daily_warning_items = {
            "early_rebalance": [],
            "profit_extension": [],
            "duplicate_buy": [],
            "position_unfilled": [],
            "completion_skipped": [],
            "completion_abandoned": [],
            "early_exit_trigger": [],
            "pending_order_added": [],
            "pending_order_success": [],
            "pending_order_expired": [],
            "take_profit": [],
            "time_stop_loss_trigger": [],
            "time_stop_loss_veto": [],
        }

    def _record_pending_order_event(self, event: Dict) -> None:
        """记录延迟订单事件，日终统一压缩显示。"""
        event_type = str(event.get("type", "")).lower()
        if event_type == "added":
            self._daily_warning_items.setdefault("pending_order_added", []).append(
                {
                    "stock": str(event.get("stock", "-")),
                    "action": str(event.get("action", "-")),
                    "reason": str(event.get("reason") or "-"),
                }
            )
            return

        if event_type == "success":
            self._daily_warning_items.setdefault("pending_order_success", []).append(
                {
                    "stock": str(event.get("stock", "-")),
                    "action": str(event.get("action", "-")),
                    "retry_count": int(event.get("retry_count", 0) or 0),
                    "delay_days": int(event.get("delay_days", 0) or 0),
                }
            )
            return

        if event_type in {"expired_retry", "expired_days"}:
            self._daily_warning_items.setdefault("pending_order_expired", []).append(
                {
                    "stock": str(event.get("stock", "-")),
                    "action": str(event.get("action", "-")),
                    "expire_type": event_type,
                    "retry_count": int(event.get("retry_count", 0) or 0),
                    "max_retry_count": int(event.get("max_retry_count", 0) or 0),
                    "delay_days": int(event.get("delay_days", 0) or 0),
                    "max_retry_days": int(event.get("max_retry_days", 0) or 0),
                }
            )

    def _record_completion_skip(self, label: str, detail: str) -> None:
        """记录补齐跳过原因，日终统一压缩显示。"""
        self._daily_warning_items.setdefault("completion_skipped", []).append(
            {"label": label, "detail": detail}
        )

    @staticmethod
    def _format_pending_order_group(
        items: List[Dict],
        action_label: str,
        item_formatter,
        count_prefix: str = "",
    ) -> str:
        """格式化延迟订单分组摘要。"""
        labels = [item_formatter(item) for item in items]
        prefix = f"{count_prefix}{action_label}{len(items)}"
        return f"{prefix}[{BacktestEngine._format_compact_items(labels, limit=6)}]"

    def _record_duplicate_buy_skip(self, stock: str, buy_date: pd.Timestamp) -> None:
        """记录重复买入跳过，日终统一压缩显示。"""
        self._daily_warning_items.setdefault("duplicate_buy", []).append(
            {"stock": stock, "buy_date": buy_date}
        )

    def _record_position_unfilled_summary(
        self,
        tranche_tag: str,
        target_n: int,
        actually_bought: int,
        unfilled_count: int,
        unfilled_stocks: List[str],
    ) -> None:
        """记录仓位未满摘要。"""
        self._daily_warning_items.setdefault("position_unfilled", []).append(
            {
                "tranche_tag": tranche_tag.strip(),
                "target_n": int(target_n),
                "actually_bought": int(actually_bought),
                "unfilled_count": int(unfilled_count),
                "unfilled_stocks": list(unfilled_stocks),
            }
        )

    def _record_completion_abandoned_summary(
        self,
        tranche_tag: str,
        original_signal_date: pd.Timestamp,
        attempts: int,
        unfilled_stocks: List[str],
    ) -> None:
        """记录补齐放弃摘要。"""
        self._daily_warning_items.setdefault("completion_abandoned", []).append(
            {
                "tranche_tag": tranche_tag.strip(),
                "original_signal_date": original_signal_date,
                "attempts": int(attempts),
                "unfilled_stocks": list(unfilled_stocks),
            }
        )

    def _record_early_exit_trigger(
        self, stock: str, holding_days: int, profit_rate: float
    ) -> None:
        """记录亏损提前换出触发。"""
        self._daily_warning_items.setdefault("early_exit_trigger", []).append(
            {
                "stock": stock,
                "holding_days": int(holding_days),
                "profit_rate": float(profit_rate),
            }
        )

    def _record_early_rebalance_summary(self, label: str, detail: str) -> None:
        """记录提前调仓相关事件，日终统一汇总。"""
        self._daily_warning_items.setdefault("early_rebalance", []).append(
            {"label": label, "detail": detail}
        )

    def _record_profit_extension_count(self, count: int) -> None:
        """记录当日盈利延续数量，供日终信号摘要使用。"""
        if count <= 0:
            return
        self._daily_warning_items.setdefault("profit_extension", []).append(
            {"count": int(count)}
        )

    def _record_take_profit_summary(
        self, profit_rate: float, threshold: float, n_positions: int
    ) -> None:
        """记录整体止盈触发，日终统一汇总。"""
        self._daily_warning_items.setdefault("take_profit", []).append(
            {
                "profit_rate": float(profit_rate),
                "threshold": float(threshold),
                "n_positions": int(n_positions),
            }
        )

    def _record_time_stop_loss_trigger(
        self, stock: str, holding_days: int, profit_rate: float
    ) -> None:
        """记录时间止损触发。"""
        self._daily_warning_items.setdefault("time_stop_loss_trigger", []).append(
            {
                "stock": stock,
                "holding_days": int(holding_days),
                "profit_rate": float(profit_rate),
            }
        )

    def _record_time_stop_loss_veto(
        self, stock: str, holding_days: int, profit_rate: float, score: float
    ) -> None:
        """记录时间止损被强势度否决。"""
        self._daily_warning_items.setdefault("time_stop_loss_veto", []).append(
            {
                "stock": stock,
                "holding_days": int(holding_days),
                "profit_rate": float(profit_rate),
                "score": float(score),
            }
        )

    def _build_daily_signal_log(self, date: pd.Timestamp) -> Optional[str]:
        """构建当日新生成买卖信号的数量摘要。"""

        def _format_groups(groups: List[Tuple[str, int]]) -> str:
            return ", ".join(f"{label}{count}" for label, count in groups if count > 0)

        buy_groups: List[Tuple[str, int]] = []
        signal_data = self.pending_signals.get(date)
        if isinstance(signal_data, dict):
            if "signals" in signal_data:
                buy_count = len(signal_data.get("signals", {}))
                if buy_count > 0:
                    buy_label = "调仓" if signal_data.get("decision_trace") else "补槽"
                    buy_groups.append((buy_label, buy_count))
            elif signal_data:
                buy_groups.append(("调仓", len(signal_data)))

        sell_label_map = {
            "holding_period": "持有期",
            "early_exit": "亏损换出",
            "time_stop_loss": "时间止损",
            "weakness_exit": "表现弱势",
            "take_profit": "止盈",
            "rebalance": "调仓",
        }
        stop_loss_label_map = {
            "drawdown": "回撤止损",
            "trailing_stop": "移动止损",
            "consecutive_limit_down": "连续跌停",
            "unknown": "止损",
        }

        sell_counts: Dict[str, int] = {}
        for info in self.pending_condition_sells.values():
            if info.get("trigger_date") != date:
                continue
            label = sell_label_map.get(str(info.get("sell_type") or ""), "条件卖出")
            sell_counts[label] = sell_counts.get(label, 0) + 1

        for info in self.pending_stop_loss_sells.values():
            if info.get("trigger_date") != date:
                continue
            label = stop_loss_label_map.get(str(info.get("trigger_type") or "unknown"), "止损")
            sell_counts[label] = sell_counts.get(label, 0) + 1

        sell_groups: List[Tuple[str, int]] = []
        for label in (
            "调仓",
            "持有期",
            "亏损换出",
            "时间止损",
            "表现弱势",
            "止盈",
            "回撤止损",
            "移动止损",
            "连续跌停",
            "止损",
            "条件卖出",
        ):
            count = sell_counts.get(label, 0)
            if count > 0:
                sell_groups.append((label, count))

        extension_count = sum(
            int(item.get("count", 0) or 0)
            for item in self._daily_warning_items.get("profit_extension", [])
        )

        if not buy_groups and not sell_groups and extension_count <= 0:
            return None

        parts = []
        if sell_groups:
            parts.append(f"卖[{_format_groups(sell_groups)}]")
        if buy_groups:
            parts.append(f"买[{_format_groups(buy_groups)}]")
        if extension_count > 0:
            parts.append(f"延续[{extension_count}]")

        return f"信号: {' | '.join(parts)}"

    def _build_daily_warning_logs(self) -> List[str]:
        """构建需在每日总结下展示的日级压缩摘要。"""
        lines: List[str] = []

        early_items = self._daily_warning_items.get("early_rebalance", [])
        if early_items:
            labels = [f"{item['label']}[{item['detail']}]" for item in early_items]
            lines.append(f"提前调仓: {self._format_compact_items(labels)}")

        duplicate_buy_items = self._daily_warning_items.get("duplicate_buy", [])
        if duplicate_buy_items:
            labels = [
                f"{item['stock']}({item['buy_date'].date()})"
                for item in duplicate_buy_items
            ]
            lines.append(
                f"重复买入跳过: {len(duplicate_buy_items)}只"
                f"[{self._format_compact_items(labels, limit=6)}]"
            )

        position_unfilled_items = self._daily_warning_items.get("position_unfilled", [])
        if position_unfilled_items:
            labels = []
            for item in position_unfilled_items:
                prefix = f"{item['tranche_tag']} " if item["tranche_tag"] else ""
                labels.append(
                    f"{prefix}目标{item['target_n']}/实买{item['actually_bought']}/待补"
                    f"{item['unfilled_count']}[{self._format_compact_items(item['unfilled_stocks'], limit=6)}]"
                    f"/{self.completion_window_days}天"
                )
            lines.append(f"仓位未满: {self._format_compact_items(labels, limit=3)}")

        completion_skipped_items = self._daily_warning_items.get("completion_skipped", [])
        if completion_skipped_items:
            groups = []
            ordered_labels = [
                "当日无行情",
                "前日无行情",
                "无数据",
                "无候选",
                "候选已持仓",
                "候选不可交易",
            ]
            for label in ordered_labels:
                matched = [
                    item for item in completion_skipped_items if item.get("label") == label
                ]
                if not matched:
                    continue
                details = [item["detail"] for item in matched]
                if label == "当日无行情":
                    groups.append(f"{label}{len(matched)}")
                else:
                    groups.append(
                        f"{label}{len(matched)}[{self._format_compact_items(details, limit=6)}]"
                    )
            if groups:
                lines.append(f"补齐跳过: {' | '.join(groups)}")

        completion_abandoned_items = self._daily_warning_items.get("completion_abandoned", [])
        if completion_abandoned_items:
            labels = []
            for item in completion_abandoned_items:
                prefix = f"{item['tranche_tag']} " if item["tranche_tag"] else ""
                labels.append(
                    f"{prefix}信号日{item['original_signal_date'].date()}/尝试{item['attempts']}次/"
                    f"剩{len(item['unfilled_stocks'])}[{self._format_compact_items(item['unfilled_stocks'], limit=6)}]"
                )
            lines.append(f"补齐放弃: {self._format_compact_items(labels, limit=3)}")

        early_exit_items = self._daily_warning_items.get("early_exit_trigger", [])
        if early_exit_items:
            labels = [
                f"{item['stock']}({item['holding_days']}d,{item['profit_rate']:+.1%})"
                for item in early_exit_items
            ]
            lines.append(
                f"亏损换出: 触发{len(early_exit_items)}"
                f"[{self._format_compact_items(labels, limit=6)}]"
            )

        pending_order_added_items = self._daily_warning_items.get("pending_order_added", [])
        if pending_order_added_items:
            groups = []
            for action in ("buy", "sell"):
                action_items = [
                    item for item in pending_order_added_items if item.get("action") == action
                ]
                if not action_items:
                    continue
                groups.append(
                    self._format_pending_order_group(
                        action_items,
                        action_label="买" if action == "buy" else "卖",
                        count_prefix="新增",
                        item_formatter=lambda item: f"{item['stock']}({item['reason']})",
                    )
                )
            if groups:
                lines.append(f"延迟订单: {' | '.join(groups)}")

        pending_order_success_items = self._daily_warning_items.get("pending_order_success", [])
        if pending_order_success_items:
            groups = []
            for action in ("buy", "sell"):
                action_items = [
                    item for item in pending_order_success_items if item.get("action") == action
                ]
                if not action_items:
                    continue
                groups.append(
                    self._format_pending_order_group(
                        action_items,
                        action_label="买" if action == "buy" else "卖",
                        count_prefix="成功",
                        item_formatter=lambda item: (
                            f"{item['stock']}(重{item['retry_count']},延{item['delay_days']}d)"
                        ),
                    )
                )
            if groups:
                lines.append(f"延迟订单成交: {' | '.join(groups)}")

        pending_order_expired_items = self._daily_warning_items.get("pending_order_expired", [])
        if pending_order_expired_items:
            groups = []
            for expire_type, label in (
                ("expired_retry", "超次"),
                ("expired_days", "超期"),
            ):
                for action in ("buy", "sell"):
                    action_items = [
                        item
                        for item in pending_order_expired_items
                        if item.get("expire_type") == expire_type and item.get("action") == action
                    ]
                    if not action_items:
                        continue
                    groups.append(
                        self._format_pending_order_group(
                            action_items,
                            action_label=("买" if action == "buy" else "卖"),
                            count_prefix=label,
                            item_formatter=lambda item: (
                                f"{item['stock']}(重{item['retry_count']}>{item['max_retry_count']})"
                                if item.get("expire_type") == "expired_retry"
                                else f"{item['stock']}(延{item['delay_days']}d>{item['max_retry_days']}d)"
                            ),
                        )
                    )
            if groups:
                lines.append(f"延迟订单放弃: {' | '.join(groups)}")

        take_profit_items = self._daily_warning_items.get("take_profit", [])
        if take_profit_items:
            details = [
                (
                    f"触发[{item['profit_rate']:+.1%}>={item['threshold']:+.1%}, "
                    f"{item['n_positions']}只次日卖出]"
                )
                for item in take_profit_items
            ]
            lines.append(f"整体止盈: {self._format_compact_items(details)}")

        time_trigger_items = self._daily_warning_items.get("time_stop_loss_trigger", [])
        time_veto_items = self._daily_warning_items.get("time_stop_loss_veto", [])
        if time_trigger_items or time_veto_items:
            parts = []
            if time_trigger_items:
                trigger_labels = [
                    f"{item['stock']}({item['holding_days']}d,{item['profit_rate']:+.1%})"
                    for item in time_trigger_items
                ]
                parts.append(
                    f"触发{len(time_trigger_items)}[{self._format_compact_items(trigger_labels)}]"
                )
            if time_veto_items:
                veto_labels = [
                    (
                        f"{item['stock']}({item['holding_days']}d,"
                        f"{item['profit_rate']:+.1%},{item['score']:.2f})"
                    )
                    for item in time_veto_items
                ]
                parts.append(
                    f"否决{len(time_veto_items)}[{self._format_compact_items(veto_labels)}]"
                )
            lines.append(f"时间止损: {' | '.join(parts)}")

        return lines

    def _calculate_current_exposure_pct(self, portfolio_value: float) -> float:
        """按当日组合市值计算股票仓位比例。"""
        if portfolio_value <= 0:
            return 0.0

        market_value = max(portfolio_value - self.current_capital, 0.0)
        exposure_pct = market_value / portfolio_value * 100
        return min(exposure_pct, 100.0)

    def _get_min_buy_value_threshold(self, date: pd.Timestamp) -> float:
        """计算最小买入后市值阈值（与纸面交易口径一致）。"""
        ratio = float(self.min_buy_value_ratio or 0.0)
        if ratio <= 0:
            return 0.0

        target_count = int(self._get_target_position_count() or 0)
        if target_count <= 0:
            return 0.0

        total_assets = float(self._calculate_portfolio_value(date))
        if total_assets <= 0:
            return 0.0

        avg_position_value = total_assets / float(target_count)
        return avg_position_value * ratio

    def _initialize_decision_trace_for_signal(self, decision_trace: Dict) -> Dict:
        """扩展点：子类可补充市场层占位信息。"""
        return decision_trace

    def _finalize_decision_trace_for_signal_day(
        self, decision_trace: Dict, signal_date: pd.Timestamp
    ) -> Dict:
        """扩展点：子类可在信号日补齐摘要所需状态。"""
        return decision_trace

    def _build_signal_decision_trace(
        self,
        date: pd.Timestamp,
        target_n: int,
        candidate_count: int,
        tranche_idx: int,
        confidence_gate_state=None,
    ) -> Dict:
        """构建调仓决策摘要所需的状态。"""
        gate_enabled = bool(
            confidence_gate_state is not None and getattr(confidence_gate_state, "enabled", False)
        )
        gate_exposure = (
            float(getattr(confidence_gate_state, "exposure", 1.0)) if gate_enabled else 1.0
        )
        gate_summary = (
            getattr(confidence_gate_state, "reason", "未启用") if gate_enabled else "未启用"
        )

        trace = {
            "signal_date": date,
            "target_n": target_n,
            "candidate_count": candidate_count,
            "tranche_idx": tranche_idx,
            "queued": False,
            "signal_gate": {
                "enabled": gate_enabled,
                "exposure": gate_exposure,
                "quality_exposure": 1.0,  # 由质量监控模块事后填入
                "summary": gate_summary,
            },
            "ect": {
                "enabled": self.equity_curve_monitor is not None,
                "exposure": 1.0,
                "summary": "待执行日评估" if self.equity_curve_monitor else "未启用",
            },
            "ma250": {
                "enabled": False,
                "exposure": 1.0,
                "summary": "未启用",
            },
            "market_regime": {
                "enabled": False,
                "exposure": 1.0,
                "summary": "未启用",
            },
            "market_layer_exposure": 1.0,
            "final_target_exposure": gate_exposure,
        }
        return self._initialize_decision_trace_for_signal(trace)

    def _mark_decision_trace_blocked(self, decision_trace: Dict) -> Dict:
        """标记该信号未进入待买队列。"""
        decision_trace["queued"] = False
        decision_trace["final_target_exposure"] = 0.0
        if decision_trace.get("ect", {}).get("enabled"):
            decision_trace["ect"]["exposure"] = None
            decision_trace["ect"]["summary"] = "未评估（信号门控已阻断）"
        if decision_trace.get("ma250", {}).get("enabled"):
            decision_trace["ma250"]["exposure"] = None
            decision_trace["ma250"]["summary"] = "未评估（信号门控已阻断）"
        if decision_trace.get("market_regime", {}).get("enabled"):
            decision_trace["market_regime"]["exposure"] = None
            decision_trace["market_regime"]["summary"] = "未评估（信号门控已阻断）"
        decision_trace["market_layer_exposure"] = None
        return decision_trace

    def _log_rebalance_decision_summary(
        self,
        decision_trace: Dict,
        execution_date: Optional[pd.Timestamp] = None,
        tranche_tag: str = "",
    ) -> None:
        """统一输出调仓决策摘要。"""
        logger.info(
            _format_rebalance_decision_summary(
                decision_trace=decision_trace,
                execution_date=execution_date,
                tranche_tag=tranche_tag,
            )
        )

    def _get_current_position_atr_stats(
        self, date: pd.Timestamp
    ) -> Optional[Tuple[float, float, float]]:
        """获取当日持仓 ATR% 统计（子类可覆写）。"""
        return None

    def _format_current_position_atr_stats(self, date: pd.Timestamp) -> str:
        """格式化当日持仓 ATR% 统计。"""
        atr_stats = self._get_current_position_atr_stats(date)
        if atr_stats is None:
            return "ATR:[N/A/N/A/N/A]"

        min_atr_pct, avg_atr_pct, max_atr_pct = atr_stats
        return f"ATR:[" f"{min_atr_pct:.2%}/{avg_atr_pct:.2%}/{max_atr_pct:.2%}]"

    def _format_daily_progress_log(
        self,
        date: pd.Timestamp,
        trading_days: int,
        total_days: int,
        cycle_day: int,
        portfolio_value: float,
        buy_count: int = 0,
        sell_count: int = 0,
    ) -> str:
        """格式化每日回测进度日志。"""
        total_return = (portfolio_value / self.initial_capital - 1) * 100
        # 简单年化收益率（不假设收益再投入）
        simple_annual = (total_return / 100) * (252 / trading_days) * 100 if trading_days > 0 else 0.0
        ann_return = simple_annual
        rebalance_return_str = (
            f"{(portfolio_value / self._last_rebalance_nav - 1) * 100:+.2f}%"
            if self._last_rebalance_nav and self._last_rebalance_nav > 0
            else "N/A"
        )
        target_position_count = self._get_target_position_count()
        current_exposure_pct = self._calculate_current_exposure_pct(portfolio_value)
        t_index = max(cycle_day - 1, 0)

        return (
            f"T{t_index}[{date.date()}]: {trading_days:0{len(str(total_days))}}/{total_days} 天"
            f" | 本轮[{cycle_day:0{len(str(self.rebalance_freq))}}/{self.rebalance_freq}]"
            f" | 持仓/仓位[{len(self.positions):0{len(str(target_position_count))}}/{target_position_count}]"
            f"/[{current_exposure_pct:.2f}%]"
            f" | 买/卖[{buy_count}/{sell_count}]"
            f" | 收益[本调仓/本轮/年化]=[{rebalance_return_str}/{total_return:+.2f}%/{ann_return:+.2f}%]"
        )

    def _build_nav_series(self, current_date: pd.Timestamp) -> Optional[pd.Series]:
        """构建用于 ECT 的历史 NAV 序列

        Args:
            current_date: 当前日期

        Returns:
            NAV Series (index=date, values=nav) 或 None
        """
        if not self.portfolio_values:
            return None

        # 从 portfolio_values 构建 DataFrame
        df = pd.DataFrame(self.portfolio_values)

        # 计算 NAV（相对于初始资金的净值）
        df["nav"] = df["portfolio_value"] / self.initial_capital

        # 转换为 Series
        nav_series = pd.Series(df["nav"].values, index=df["date"])

        return nav_series

    def _build_signal_data(self, date: pd.Timestamp) -> Optional[Dict]:
        """构建传递给信号生成器的额外数据（扩展点）

        子类可以重写此方法以注入特定数据（如 ML 特征）。

        Args:
            date: 信号生成日期

        Returns:
            数据字典，将与默认数据合并后传递给 signal.generate_ranked()
            返回 None 表示该日期无可用数据，将跳过信号生成
        """
        return {}

    def _post_filter_candidates(self, ranked_candidates: list, date: pd.Timestamp) -> list:
        """对排序候选列表做额外过滤（扩展点）

        Args:
            ranked_candidates: [(stock_code, score), ...] 已按分数降序排列
            date: 当前日期

        Returns:
            过滤后的候选列表
        """
        return ranked_candidates

    def _get_position_weight_for_planning(
        self,
        date: pd.Timestamp,
        stock: str,
        portfolio_value: Optional[float] = None,
    ) -> float:
        """获取指定持仓在规划日的组合权重。"""
        if stock not in self.positions:
            return 0.0

        total_value = float(portfolio_value or 0.0)
        if total_value <= 0:
            total_value = float(self._calculate_portfolio_value(date))
        if total_value <= 0:
            return 0.0

        info = self.positions[stock]
        shares = float(info.get("shares", 0) or 0)
        if shares <= 0:
            return 0.0

        trade_price = self._get_trade_price(date, stock)
        if trade_price is None:
            trade_price = info.get("last_known_price")
            if trade_price is None:
                trade_price = info.get("buy_trade_price", 0.0)
        else:
            info["last_known_price"] = trade_price

        if not trade_price:
            return 0.0

        return float(shares * trade_price / total_value)

    def _queue_condition_sell_refill_signal(
        self,
        date: pd.Timestamp,
        slot_weights: List[Dict[str, float]],
        price_data: pd.DataFrame,
        date_to_idx: Dict,
    ) -> None:
        """为持有期/盈利延续卖出生成 T0 买入计划，供下一交易日执行。"""
        if (
            not self.enable_position_completion
            or not slot_weights
            or date in self.pending_signals
            or price_data is None
            or price_data.empty
            or not hasattr(self.signal, "generate_ranked")
        ):
            return

        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx + 1 >= len(date_to_idx):
            return

        trade_date_str = to_trade_date_str(date)
        date_quote = price_data[price_data["trade_date"] == trade_date_str]
        stock_universe = self.universe.get_stocks(date, quote_data=date_quote)

        extra_data = self._build_signal_data(date)
        if extra_data is None:
            return

        signal_data = {}
        signal_data.update(extra_data)
        ranked_candidates = self.signal.generate_ranked(date, stock_universe, signal_data)
        if not ranked_candidates:
            return

        if self.max_per_industry is not None:
            from ..portfolio import apply_industry_constraint

            ranked_candidates = apply_industry_constraint(
                ranked_candidates,
                self.industry_mapping,
                max_per_industry=self.max_per_industry,
                target_n=len(ranked_candidates),
                verbose=self.verbose,
            )

        ranked_candidates = self._post_filter_candidates(ranked_candidates, date)
        existing_positions = set(self.positions.keys()) if self.positions else set()
        priority_candidates = [
            (stock, score)
            for stock, score in ranked_candidates
            if stock not in existing_positions
        ]
        if not priority_candidates:
            return

        normalized_slot_weights = []
        fallback_weight = 1.0 / max(len(slot_weights), 1)
        for slot in slot_weights:
            weight = float(slot.get("weight", 0.0) or 0.0)
            normalized_slot_weights.append(
                {
                    "stock": str(slot.get("stock", "")),
                    "weight": weight if weight > 0 else fallback_weight,
                }
            )

        planned_candidates = priority_candidates[: len(normalized_slot_weights)]
        if not planned_candidates:
            return

        signals = {}
        planned_slot_weights = []
        for slot_weight_info, (candidate_stock, _score) in zip(
            normalized_slot_weights, planned_candidates
        ):
            weight = float(slot_weight_info["weight"])
            signals[candidate_stock] = weight
            planned_slot_weights.append({"stock": candidate_stock, "weight": weight})

        desired_position_count = int(self._get_target_position_count() or len(self.positions))
        self.pending_signals[date] = {
            "signals": signals,
            "ranked_candidates": ranked_candidates,
            "priority_candidates": list(priority_candidates),
            "slot_weights": planned_slot_weights,
            "target_n": len(planned_slot_weights),
            "desired_position_count": desired_position_count,
            "tranche_idx": 0,
        }

        if self.verbose:
            logger.info(
                f"  持有期卖出补位计划: {date.date()} 生成 {len(planned_slot_weights)} 个待买槽位，"
                f"下一交易日按候选顺序执行"
            )

    def _extend_holding_period(
        self,
        stock: str,
        signal_date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        date_to_idx: Dict,
    ) -> None:
        """延续持有：重置持有期起点为 T+1（下一交易日），不产生交易成本。

        当持仓保留奖励启用时，仍在 Top-N 中的已持仓股票不会被卖出再买入，
        而是直接延续持有并重置持有期计时器。

        Args:
            stock: 股票代码
            signal_date: 信号生成日期（T 日）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if stock not in self.positions:
            return
        current_idx = date_to_idx.get(signal_date)
        if current_idx is None or current_idx + 1 >= len(trading_dates):
            return
        new_buy_date = trading_dates[current_idx + 1]  # T+1 作为新的持有期起点
        old_buy_date = self.positions[stock]["buy_date"]
        self.positions[stock]["buy_date"] = new_buy_date
        self.positions[stock]["signal_date"] = signal_date
        if self.verbose:
            logger.debug(
                f"  持仓延续: {stock} 持有期重置 "
                f"({old_buy_date.date()} → {new_buy_date.date()})"
            )

    def _get_holding_features_row(
        self, date: pd.Timestamp, stock: str
    ) -> Optional[pd.Series]:
        """持仓强势度评分数据源 hook

        基类无特征数据,返回 None。BacktestEngineML 子类会从 features_by_date
        读取对应股票的截面特征行并返回。
        """
        return None

    def _generate_signal(
        self,
        date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
        date_to_idx: Dict,
        tranche_idx: int = 0,
    ) -> None:
        """生成信号（在 T 日生成，T+1 日执行买入）

        新逻辑：生成排序候选列表，在 T+1 日过滤不可交易股票并回填，确保 top N 全部可交易。

        Args:
            date: 信号生成日期
            trading_dates: 交易日列表
            price_data: 价格数据，包含行情信息
            date_to_idx: 日期到索引的映射
            tranche_idx: 分批调仓的批次索引（0-based）
        """
        # 记录调仓日组合净值，用于止盈基准和"本调仓收益"计算
        self._last_rebalance_nav = self._calculate_portfolio_value(date)

        # ── 滚��质量监控：评估上一次信号的实际表现 ──
        if self.signal_gate_quality_enabled and self._signal_tracking:
            self._evaluate_expired_signal_quality(date, price_data)

        # 获取当日行情数据用于基础过滤（ST、停牌等基础过滤）
        trade_date_str = to_trade_date_str(date)
        date_quote = price_data[price_data["trade_date"] == trade_date_str]
        # 获取股票池（不过滤涨跌停，因为 T 日涨跌停不代表 T+1 日也涨跌停）
        # 但保留 ST、基本可交易性等过滤
        stock_universe = self.universe.get_stocks(date, quote_data=date_quote)

        # 调用扩展点获取额外数据（如 ML 特征）
        extra_data = self._build_signal_data(date)
        if extra_data is None:
            # None 表示该日期无可用数据，跳过信号生成
            if self.verbose:
                logger.warning(
                    f"信号日 {date.date()} 无可用数据（_build_signal_data 返回 None），跳过"
                )
            return

        # 合并默认数据和额外数据
        signal_data = {}
        signal_data.update(extra_data)

        # 生成排序后的候选列表（返回所有候选，不仅仅是 top N）
        ranked_candidates = self.signal.generate_ranked(date, stock_universe, signal_data)

        if not ranked_candidates:
            if self.verbose:
                logger.warning(f"信号日 {date.date()} 无候选")
            return

        # 获取 T+1 日（买入日）的行情数据
        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx + 1 >= len(trading_dates):
            # 没有 T+1 日，无法买入
            if self.verbose:
                logger.warning(f"信号日 {date.date()} 之后没有交易日，无法执行")
            return

        buy_date = trading_dates[current_idx + 1]
        buy_date_str = to_trade_date_str(buy_date)
        buy_date_quote = price_data[price_data["trade_date"] == buy_date_str]

        # 应用行业约束（如果启用）
        if self.max_per_industry is not None:
            # 延迟导入
            from ..portfolio import apply_industry_constraint

            ranked_candidates = apply_industry_constraint(
                ranked_candidates,
                self.industry_mapping,
                max_per_industry=self.max_per_industry,
                target_n=len(ranked_candidates),  # 保留所有候选，只改变顺序
                verbose=self.verbose,
            )

        # 扩展点：子类可覆盖此方法对候选列表做额外过滤
        ranked_candidates = self._post_filter_candidates(ranked_candidates, date)

        # ── 持仓处理：排除或奖励 ──
        existing_positions = set(self.positions.keys()) if self.positions else set()
        ranked_candidates_for_gate = ranked_candidates  # 用于置信度门控（不含奖励）

        if self.holding_bonus_enabled and existing_positions:
            # 换手率约束模式：对已持仓股票加分，不排除
            scores_array = np.array([s for _, s in ranked_candidates])
            score_std = float(np.std(scores_array)) if len(scores_array) > 1 else 0.0
            bonus = self.holding_bonus_sigma * score_std
            bonus_count = sum(1 for stock, _ in ranked_candidates if stock in existing_positions)

            # 含奖励的候选列表用于选股
            ranked_candidates_for_selection = sorted(
                [
                    (stock, score + bonus) if stock in existing_positions else (stock, score)
                    for stock, score in ranked_candidates
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            # 门控评估排除已持仓（不受 bonus 影响，保持原有评估逻辑）
            ranked_candidates_for_gate = [
                (stock, score) for stock, score in ranked_candidates
                if stock not in existing_positions
            ]
            if self.verbose and bonus_count > 0:
                logger.info(
                    f"  换手率约束: {bonus_count} 只持仓获得加分 "
                    f"(bonus={bonus:.4f}, sigma={self.holding_bonus_sigma}×std={score_std:.4f})"
                )
        elif existing_positions:
            # 原有逻辑：排除已持仓的股票
            ranked_candidates_for_selection = [
                (stock, score)
                for stock, score in ranked_candidates
                if stock not in existing_positions
            ]
            ranked_candidates_for_gate = ranked_candidates_for_selection
            if self.verbose:
                excluded = len(ranked_candidates) - len(ranked_candidates_for_selection)
                if excluded > 0:
                    logger.info(
                        f"  排除已持仓股票: {excluded} 只 "
                        f"(持仓 {len(existing_positions)} 只, "
                        f"候选从 {len(ranked_candidates)} 缩减到 "
                        f"{len(ranked_candidates_for_selection)})"
                    )
        else:
            ranked_candidates_for_selection = ranked_candidates

        confidence_gate_state = None
        if hasattr(self.signal, "evaluate_confidence_gate"):
            confidence_gate_state = self.signal.evaluate_confidence_gate(
                ranked_candidates_for_gate,
                date=date,
            )
            if getattr(confidence_gate_state, "enabled", False):
                self.confidence_gate_history.append(
                    {
                        "date": trade_date_str,
                        "tranche_idx": tranche_idx,
                        "score": confidence_gate_state.score,
                        "exposure": confidence_gate_state.exposure,
                        "candidate_count": confidence_gate_state.candidate_count,
                        "top_k": confidence_gate_state.top_k,
                        "top_mean": confidence_gate_state.top_mean,
                        "baseline_mean": confidence_gate_state.baseline_mean,
                        "score_std": confidence_gate_state.score_std,
                        "hit_threshold": confidence_gate_state.hit_threshold,
                        "reason": confidence_gate_state.reason,
                    }
                )

        # 从排序候选中选择 top N 股票
        # 始终仅基于 T0 排名生成次日买入计划；
        # T+1 的可交易性统一在执行阶段处理，避免在计划阶段引入前视过滤。
        signals = {}
        candidates_checked = 0
        filtered_reasons = {"停牌": 0, "涨停": 0, "跌停": 0}

        # 获取目标数量（从信号生成器获取）
        if hasattr(self.signal, "top_n"):
            base_n = self.signal.top_n
        else:
            base_n = len(ranked_candidates)

        # 动态 Top-N：根据门控置信度调整选股数量
        # 高置信度 → 集中（缩减），低置信度 → 分散（扩大）
        dynamic_topn_reason = None
        if (
            self.signal_gate_dynamic_topn
            and confidence_gate_state is not None
            and getattr(confidence_gate_state, "enabled", False)
        ):
            gate_exposure = getattr(confidence_gate_state, "exposure", 1.0)
            if gate_exposure >= 1.0:
                # 高置信度：集中持股
                target_n = max(3, int(round(base_n * self.signal_gate_topn_high_multiplier)))
                dynamic_topn_reason = (
                    f"高置信度(exposure={gate_exposure:.0%})→集中"
                    f"({base_n}×{self.signal_gate_topn_high_multiplier}={target_n}只)"
                )
            elif gate_exposure <= 0:
                # 门控阻断，target_n 保持 base_n（后续门控会清零信号）
                target_n = base_n
                dynamic_topn_reason = "门控阻断，top-N不生效"
            else:
                # 中低置信度：分散持股，按 exposure 线性插值
                # exposure 越低 → multiplier 越大（趋向 low_multiplier）
                multiplier = 1.0 + (1.0 - gate_exposure) * (
                    self.signal_gate_topn_low_multiplier - 1.0
                )
                target_n = min(
                    len(ranked_candidates_for_selection),
                    max(base_n, int(round(base_n * multiplier))),
                )
                dynamic_topn_reason = (
                    f"中低置信度(exposure={gate_exposure:.0%})→分散"
                    f"({base_n}×{multiplier:.2f}={target_n}只)"
                )
        else:
            target_n = base_n

        decision_trace = self._build_signal_decision_trace(
            date=date,
            target_n=target_n,
            candidate_count=len(ranked_candidates_for_selection),
            tranche_idx=tranche_idx,
            confidence_gate_state=confidence_gate_state,
        )
        # 将动态 Top-N 信息写入 decision_trace
        decision_trace["dynamic_topn"] = {
            "enabled": self.signal_gate_dynamic_topn,
            "base_n": base_n,
            "effective_n": target_n,
            "reason": dynamic_topn_reason,
        }
        decision_trace["holding_bonus"] = {
            "enabled": self.holding_bonus_enabled,
        }

        priority_candidates = list(ranked_candidates_for_selection)

        if self.holding_bonus_enabled and existing_positions:
            # 换手率约束模式：区分保留持仓和新买入
            held_kept = []
            for stock, score in priority_candidates[:target_n]:
                if stock in existing_positions:
                    held_kept.append(stock)
                else:
                    signals[stock] = score
                    candidates_checked += 1
            # 延续保留的持仓（重置持有期起点为 T+1）
            for stock in held_kept:
                self._extend_holding_period(stock, date, trading_dates, date_to_idx)
            if held_kept and self.verbose:
                logger.info(
                    f"  持仓保留: {len(held_kept)} 只仍在Top-{target_n}中，"
                    f"延续持有 ({', '.join(held_kept[:5])}"
                    f"{'...' if len(held_kept) > 5 else ''})"
                )
            decision_trace["holding_bonus"]["kept_count"] = len(held_kept)
            decision_trace["holding_bonus"]["kept_stocks"] = list(held_kept)
            decision_trace["holding_bonus"]["new_buy_count"] = len(signals)
        else:
            for stock, score in priority_candidates[:target_n]:
                signals[stock] = score
                candidates_checked += 1

        if not signals:
            if self.verbose:
                logger.warning(
                    f"信号日 {date.date()} 所有候选在 T+1 日 {buy_date.date()} 均不可交易，"
                    f"检查了 {candidates_checked} 个候选"
                )
            return

        # 归一化权重（子类可覆写 _normalize_signals 以实现 ATR 加权等策略）
        signals = self._normalize_signals(signals, date)

        # 应用权重限制（如果启用）
        if self.max_weight_per_stock is not None:
            from ..portfolio import cap_and_normalize_weights

            signals = cap_and_normalize_weights(
                signals, max_weight_per_stock=self.max_weight_per_stock, verbose=self.verbose
            )

            if not signals:
                if self.verbose:
                    logger.warning(f"信号日 {date.date()} 权重限制后无有效权重，跳过")
                return

        if confidence_gate_state is not None and hasattr(
            self.signal, "apply_confidence_gate_to_weights"
        ):
            signals = self.signal.apply_confidence_gate_to_weights(
                signals,
                confidence_state=confidence_gate_state,
                date=date,
                emit_log=False,
            )

            if not signals:
                decision_trace = self._mark_decision_trace_blocked(decision_trace)
                self._log_rebalance_decision_summary(decision_trace=decision_trace)
                return

        # ── 滚动模型质量监控：记录选股信息 + 应用质量仓位系数 ──
        if self.signal_gate_quality_enabled and signals:
            # 记录本次选股，用于后续持仓结束后评估
            predicted_mean = (
                confidence_gate_state.top_mean
                if confidence_gate_state is not None
                and not np.isnan(confidence_gate_state.top_mean)
                else 0.0
            )
            self._record_signal_for_quality_tracking(date, list(signals.keys()), predicted_mean)
            # 应用滚动质量仓位系数
            quality_exposure = self._get_rolling_quality_exposure()
            if quality_exposure < 1.0:
                signals = {stock: weight * quality_exposure for stock, weight in signals.items()}
            # 将质量系数写入 decision_trace，供摘要日志统一显示
            if "signal_gate" in decision_trace:
                decision_trace["signal_gate"]["quality_exposure"] = quality_exposure
                decision_trace["signal_gate"]["quality_score"] = self._rolling_quality_score
                decision_trace["signal_gate"]["quality_warmup_remaining"] = (
                    self._quality_warmup_remaining
                )
                # 同步更新 final_target_exposure（门控 × 质量）
                orig_gate = decision_trace["signal_gate"].get("exposure", 1.0)
                decision_trace["final_target_exposure"] = float(orig_gate) * float(quality_exposure)
            # 注入质量分数到门控状态
            if confidence_gate_state is not None:
                confidence_gate_state.rolling_quality = self._rolling_quality_score

        # 保存信号，待 T+1 执行
        # 同时保存 T0 候选优先级与槽位计划，供 T+1 顺位执行和后续补齐复用。
        slot_weights = [{"stock": stock, "weight": float(weight)} for stock, weight in signals.items()]
        self.pending_signals[date] = {
            "signals": signals,
            "ranked_candidates": ranked_candidates if self.enable_position_completion else [],
            "priority_candidates": list(priority_candidates),
            "slot_weights": slot_weights,
            "target_n": target_n,
            "desired_position_count": target_n,
            "tranche_idx": tranche_idx,
            "decision_trace": decision_trace,
        }

        # 保存最近一次调仓候选列表，供整体止盈补位使用
        self._last_ranked_candidates = list(ranked_candidates)
        self._last_signal_date = date
        decision_trace["queued"] = True
        decision_trace["final_target_exposure"] = float(sum(signals.values()))
        decision_trace = self._finalize_decision_trace_for_signal_day(
            decision_trace=decision_trace,
            signal_date=date,
        )
        self.pending_signals[date]["decision_trace"] = decision_trace

        tranche_tag = (
            f"[批次 {tranche_idx + 1}/{self.stagger_tranches}] "
            if self.stagger_tranches > 1
            else ""
        )
        self._log_rebalance_decision_summary(
            decision_trace=decision_trace,
            execution_date=buy_date,
            tranche_tag=tranche_tag,
        )

        # 分批调仓时始终打印信号生成汇总，便于确认各批次调度情况
        if self.verbose or self.stagger_tranches > 1:
            logger.info(
                f"  {tranche_tag}信号生成: {date.date()}, 选择 top {len(signals)}/{target_n} 股票（未检查 T+1 可交易性，将在买入时处理）, "
                f"候选总数 {len(priority_candidates)} 个"
            )

    def _execute_pending_buys(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """执行待执行的买入操作（T+1）

        同时跟踪未成交的槽位，如果启用补齐功能则记录到 unfilled_slots

        Args:
            date: 当前日期
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        # 查找前一个交易日的信号
        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx == 0:
            return

        signal_date = trading_dates[current_idx - 1]

        if signal_date not in self.pending_signals:
            return

        signal_data = self.pending_signals.pop(signal_date)

        # 兼容性处理：支持旧格式和新格式
        # 旧格式：signal_data = {stock: weight}
        # 新格式：signal_data = {
        #   'signals': {stock: weight}, 'priority_candidates': [...],
        #   'slot_weights': [...], 'ranked_candidates': [...], 'target_n': N
        # }
        if isinstance(signal_data, dict) and "signals" in signal_data:
            signals = signal_data["signals"]
            ranked_candidates = signal_data.get("ranked_candidates", [])
            priority_candidates = signal_data.get("priority_candidates", ranked_candidates)
            slot_weights = signal_data.get("slot_weights", [])
            target_n = signal_data.get("target_n", len(signals))
            desired_position_count = signal_data.get("desired_position_count")
            tranche_idx = signal_data.get("tranche_idx", 0)
            decision_trace = signal_data.get("decision_trace")
        else:
            signals = signal_data
            ranked_candidates = []
            priority_candidates = list(signals.items())
            slot_weights = [{"stock": stock, "weight": float(weight)} for stock, weight in signals.items()]
            target_n = len(signals)
            desired_position_count = None
            tranche_idx = 0
            decision_trace = None

        tranche_tag = (
            f"[批次 {tranche_idx + 1}/{self.stagger_tranches}] "
            if self.stagger_tranches > 1
            else ""
        )

        # 应用风险预算（波动率缩放）
        if self.enable_risk_budget:
            signals = self._apply_risk_budget(signals, date)

        # 应用 ECT 仓位系数
        ect_exposure = 1.0
        ect_reason = "未启用"
        if self.equity_curve_monitor:
            # 构建历史 NAV 序列
            nav_series = self._build_nav_series(date)

            if nav_series is not None and len(nav_series) > 0:
                # 计算 ECT 系数
                ect_exposure, ect_reason = self.equity_curve_monitor.calculate_exposure(
                    nav_series, current_date=to_trade_date_str(date)
                )

                if self.verbose and ect_exposure < 1.0:
                    logger.info(f"ECT 生效: {date.date()}, {ect_reason}")
                elif self.verbose:
                    logger.info(f"ECT 不生效: {date.date()}, {ect_reason}")

                # 将系数应用到所有目标权重
                if ect_exposure < 1.0:
                    signals = {stock: weight * ect_exposure for stock, weight in signals.items()}

                    if self.verbose:
                        logger.info(f"ECT 调整: 所有目标权重乘以系数 {ect_exposure:.2f}")

        # 执行日权重可能被风险预算/ECT 调整，按原槽位顺序同步更新。
        if slot_weights:
            ordered_slot_weights = []
            seen_slot_stocks = set()
            for slot in slot_weights:
                slot_stock = slot.get("stock")
                if slot_stock in signals and slot_stock not in seen_slot_stocks:
                    ordered_slot_weights.append(
                        {"stock": slot_stock, "weight": float(signals[slot_stock])}
                    )
                    seen_slot_stocks.add(slot_stock)
            for stock, weight in signals.items():
                if stock not in seen_slot_stocks:
                    ordered_slot_weights.append({"stock": stock, "weight": float(weight)})
        else:
            ordered_slot_weights = [
                {"stock": stock, "weight": float(weight)} for stock, weight in signals.items()
            ]
        slot_weights = ordered_slot_weights
        if not priority_candidates:
            priority_candidates = list(signals.items())

        if decision_trace is None:
            decision_trace = self._build_signal_decision_trace(
                date=signal_date,
                target_n=target_n,
                candidate_count=len(ranked_candidates) if ranked_candidates else len(signals),
                tranche_idx=tranche_idx,
                confidence_gate_state=None,
            )

        decision_trace["ect"] = {
            "enabled": self.equity_curve_monitor is not None,
            "exposure": ect_exposure,
            "summary": ect_reason if self.equity_curve_monitor else "未启用",
        }
        decision_trace["queued"] = True
        decision_trace["final_target_exposure"] = float(sum(signals.values()))

        # 计算当前组合市值
        portfolio_value = self._calculate_portfolio_value(date)
        current_value = portfolio_value

        # 分批调仓时，每个 tranche 只使用 1/K 的组合价值
        if self.stagger_tranches > 1:
            current_value = current_value / self.stagger_tranches

        planned_buys: List[Dict] = []
        successful_buys: List[Dict] = []
        failed_buys: List[Dict] = []
        holding_bonus_state = decision_trace.get("holding_bonus", {}) if decision_trace else {}
        inherited_stocks = list(holding_bonus_state.get("kept_stocks", []))
        inherited_position_count = int(
            holding_bonus_state.get("kept_count", len(inherited_stocks))
        )

        def _build_buy_detail(stock: str, target_value: float) -> Dict:
            actual_weight = float(target_value / portfolio_value) if portfolio_value > 0 else 0.0
            return {"stock": stock, "weight": actual_weight}

        def _get_position_weight(stock: str) -> float:
            if portfolio_value <= 0 or stock not in self.positions:
                return 0.0

            info = self.positions[stock]
            shares = info.get("shares", 0)
            trade_price = self._get_trade_price(date, stock)
            if trade_price is None:
                trade_price = info.get("last_known_price")
                if trade_price is None:
                    trade_price = info.get("buy_trade_price", 0.0)
            else:
                info["last_known_price"] = trade_price

            return float(shares * trade_price / portfolio_value) if trade_price else 0.0

        inherited_position_weight = float(sum(_get_position_weight(stock) for stock in inherited_stocks))

        def _record_buy_execution(buy_detail: Dict, stock: str, target_value: float) -> bool:
            trades_before = len(self.trades)

            self._buy_stock(date, stock, target_value, signal_date=signal_date)

            trade_executed = (
                len(self.trades) > trades_before
                and self.trades[-1].get("action") == "buy"
                and self.trades[-1].get("stock") == stock
                and self.trades[-1].get("date") == date
            )

            if trade_executed:
                successful_buys.append(buy_detail.copy())
                return True

            return False

        trade_date_str = to_trade_date_str(date)
        date_quote = (
            self.price_data_cache[self.price_data_cache["trade_date"] == trade_date_str]
            if self.price_data_cache is not None
            else pd.DataFrame()
        )
        blocked_tradeability_reasons: Dict[str, str] = {}
        bought_stock_set = set()
        remaining_unfilled_slots = []

        def _check_candidate_tradeable(candidate_stock: str) -> tuple:
            if candidate_stock in blocked_tradeability_reasons:
                return False, blocked_tradeability_reasons[candidate_stock]
            if date_quote.empty:
                blocked_tradeability_reasons[candidate_stock] = "无行情"
                return False, "无行情"

            tradeable, reason = is_tradeable(
                candidate_stock, trade_date_str, date_quote, action="buy"
            )
            if not tradeable:
                blocked_tradeability_reasons[candidate_stock] = reason
            return tradeable, reason

        if desired_position_count is None:
            desired_position_count = inherited_position_count + len(slot_weights)
        desired_position_count = int(desired_position_count or 0)
        available_slot_count = max(desired_position_count - len(self.positions), 0)
        planned_slot_weights = slot_weights[:available_slot_count]

        for slot_weight_info in planned_slot_weights:
            original_stock = slot_weight_info["stock"]
            weight = float(slot_weight_info["weight"])
            target_value = current_value * weight
            planned_buys.append(_build_buy_detail(original_stock, target_value))

            bought_for_slot = False
            last_reason = "候选耗尽"
            slot_failure_recorded = False

            for candidate_stock, _ in priority_candidates:
                if candidate_stock in self.positions or candidate_stock in bought_stock_set:
                    continue

                tradeable, reason = _check_candidate_tradeable(candidate_stock)
                if not tradeable:
                    last_reason = reason
                    if candidate_stock == original_stock:
                        failed_buys.append(
                            {
                                **_build_buy_detail(candidate_stock, target_value),
                                "reason": reason,
                            }
                        )
                        slot_failure_recorded = True
                    continue

                buy_detail = _build_buy_detail(candidate_stock, target_value)
                if _record_buy_execution(buy_detail, candidate_stock, target_value):
                    bought_stock_set.add(candidate_stock)
                    bought_for_slot = True
                    break

                last_reason = "未成交"
                if candidate_stock == original_stock:
                    failed_buys.append({**buy_detail, "reason": last_reason})
                    slot_failure_recorded = True

            if not bought_for_slot:
                remaining_unfilled_slots.append({"stock": original_stock, "weight": weight})
                if not slot_failure_recorded:
                    failed_buys.append(
                        {
                            **_build_buy_detail(original_stock, target_value),
                            "reason": last_reason,
                        }
                    )

        # 记录买入后的持仓数量
        actually_bought = len(successful_buys)

        # 同日顺延后仍未补满的空槽，才进入后续跨日补齐。
        if self.enable_position_completion and remaining_unfilled_slots and ranked_candidates:
            unfilled_count = len(remaining_unfilled_slots)
            unfilled_stocks = [slot["stock"] for slot in remaining_unfilled_slots]

            self.unfilled_slots[signal_date] = {
                "unfilled_count": unfilled_count,
                "unfilled_slot_weights": remaining_unfilled_slots,
                "target_n": len(planned_slot_weights),
                "ranked_candidates": ranked_candidates,
                "signal_date": signal_date,
                "first_attempt_date": date,
                "attempts": 0,
                "tranche_idx": tranche_idx,
            }

            self.completion_stats["total_unfilled"] += 1

            self._record_position_unfilled_summary(
                tranche_tag=tranche_tag,
                target_n=len(planned_slot_weights),
                actually_bought=actually_bought,
                unfilled_count=unfilled_count,
                unfilled_stocks=unfilled_stocks,
            )

        # 当日买入/卖出明细统一在每日总结下一行展示，这里不再单独输出买入执行日志。

    def _process_position_completion(
        self,
        date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
        date_to_idx: Dict,
    ) -> None:
        """处理仓位补齐逻辑

        在调仓日后的 T+1 至 T+completion_window_days 天内，尝试补齐未成交的槽位：
        1. 基于上一交易日 D-1 的数据重新生成候选股票（避免使用未来数据）
        2. 从候选中选择可用股票填补缺口，但使用调仓日 T 生成的槽位权重
        3. 检查当日 D 可交易性，不可交易则保留该槽位到下次补齐
        4. 超过补齐窗口则放弃

        Args:
            date: 当前日期（补齐买入日 D）
            trading_dates: 交易日列表
            price_data: 价格数据
            date_to_idx: 日期到索引的映射
        """
        if not self.unfilled_slots:
            return

        # 获取上一交易日（D-1）用于生成候选
        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx == 0:
            return

        prev_date = trading_dates[current_idx - 1]
        prev_date_str = to_trade_date_str(prev_date)
        prev_date_quote = price_data[price_data["trade_date"] == prev_date_str]

        # 获取当日（D）行情数据用于交易性检查
        trade_date_str = to_trade_date_str(date)
        date_quote = price_data[price_data["trade_date"] == trade_date_str]

        if date_quote.empty:
            if self.verbose:
                self._record_completion_skip("当日无行情", "当日无行情")
            return

        # 遍历所有未补齐的槽位
        completed_signal_dates = []
        completion_success_items: List[Dict[str, str]] = []
        completion_delayed_slots: List[str] = []

        for signal_date, slot_info in list(self.unfilled_slots.items()):
            first_attempt_date = slot_info["first_attempt_date"]
            unfilled_slot_weights = slot_info["unfilled_slot_weights"]
            target_n = slot_info["target_n"]
            attempts = slot_info["attempts"]
            original_signal_date = slot_info["signal_date"]  # T日
            completion_tranche_idx = slot_info.get("tranche_idx", 0)
            tranche_tag = (
                f"[批次 {completion_tranche_idx + 1}/{self.stagger_tranches}] "
                if self.stagger_tranches > 1
                else ""
            )

            # 计算已经过了多少个交易日（从 T+1 开始）
            first_attempt_idx = date_to_idx.get(first_attempt_date)

            if first_attempt_idx is None:
                continue

            days_elapsed = current_idx - first_attempt_idx

            # 在 T+1 日（首次尝试日）不进行补齐，从 T+2 日开始
            if days_elapsed == 0:
                continue

            # 检查是否超过补齐窗口（窗口从 T+1 开始，所以是 < completion_window_days）
            if days_elapsed >= self.completion_window_days:
                # 超过补齐窗口，放弃补齐
                unfilled_count = len(unfilled_slot_weights)
                unfilled_stocks = [slot["stock"] for slot in unfilled_slot_weights]
                self.completion_stats["total_abandoned"] += 1
                completed_signal_dates.append(signal_date)

                self._record_completion_abandoned_summary(
                    tranche_tag=tranche_tag,
                    original_signal_date=original_signal_date,
                    attempts=attempts,
                    unfilled_stocks=unfilled_stocks,
                )
                continue

            # 在补齐窗口内，尝试补齐
            # 使用 D-1 日的数据重新生成候选股票列表
            if prev_date_quote.empty:
                prefix = f"{tranche_tag.strip()} " if tranche_tag.strip() else ""
                self._record_completion_skip(
                    "前日无行情",
                    f"{prefix}信号日{original_signal_date.date()}",
                )
                continue

            # 获取 D-1 日的股票池
            stock_universe = self.universe.get_stocks(prev_date, quote_data=prev_date_quote)

            # 调用扩展点获取 D-1 日的额外数据
            extra_data = self._build_signal_data(prev_date)
            if extra_data is None:
                prefix = f"{tranche_tag.strip()} " if tranche_tag.strip() else ""
                self._record_completion_skip(
                    "无数据",
                    f"{prefix}信号日{original_signal_date.date()}",
                )
                continue

            # 合并默认数据和额外数据
            signal_data = {}
            signal_data.update(extra_data)

            # 使用 D-1 日的数据重新生成排序候选列表
            new_ranked_candidates = self.signal.generate_ranked(
                prev_date, stock_universe, signal_data
            )

            if not new_ranked_candidates:
                prefix = f"{tranche_tag.strip()} " if tranche_tag.strip() else ""
                self._record_completion_skip(
                    "无候选",
                    f"{prefix}信号日{original_signal_date.date()}",
                )
                continue

            # 从新的候选列表中选择可用股票，排除已持仓股票
            # 多取 buffer 以应对部分候选不可交易（停牌/涨跌停）的情况
            unfilled_count = len(unfilled_slot_weights)
            candidate_buffer = unfilled_count * 2
            stocks_to_try = []
            for stock, score in new_ranked_candidates:
                if stock not in self.positions:
                    stocks_to_try.append((stock, score))
                    if len(stocks_to_try) >= candidate_buffer:
                        break

            if not stocks_to_try:
                prefix = f"{tranche_tag.strip()} " if tranche_tag.strip() else ""
                self._record_completion_skip(
                    "候选已持仓",
                    f"{prefix}信号日{original_signal_date.date()}",
                )
                continue

            # 尝试按槽位补齐
            bought_stocks = []
            current_value = self._calculate_portfolio_value(date)
            # 分批调仓时，补齐预算也要按 tranche 比例分配，与正常买入路径一致
            if self.stagger_tranches > 1:
                current_value = current_value / self.stagger_tranches
            remaining_unfilled_slots = []
            bought_stock_set = set()  # 跟踪已买入的股票，避免重复买入
            untradeable_stocks = set()  # 当天不可交易的股票，跳过后续槽位的重复尝试

            # 逐个槽位尝试补齐
            for slot_weight_info in unfilled_slot_weights:
                original_stock = slot_weight_info["stock"]
                weight = slot_weight_info["weight"]

                # 尝试从有限的候选列表中买入（按顺序）
                bought_for_this_slot = False

                for stock, score in stocks_to_try:
                    # 跳过已买入或当天已确认不可交易的股票
                    if stock in bought_stock_set or stock in untradeable_stocks:
                        continue

                    # 检查是否可交易（在当日 D）
                    tradeable, reason = is_tradeable(
                        stock, trade_date_str, date_quote, action="buy"
                    )

                    if not tradeable:
                        untradeable_stocks.add(stock)
                        if self.verbose:
                            self._record_completion_skip(
                                "候选不可交易",
                                f"{stock}({reason})",
                            )
                        continue

                    # 可交易，尝试买入
                    target_value = current_value * weight
                    self._buy_stock(date, stock, target_value, signal_date=original_signal_date)

                    # 检查是否买入成功
                    if stock in self.positions:
                        bought_stocks.append(stock)
                        bought_stock_set.add(stock)  # 记录已买入
                        bought_for_this_slot = True
                        completion_success_items.append({"slot": original_stock, "buy": stock})

                        self.completion_stats["total_completed"] += 1

                        break

                # 如果该槽位未能补齐，保留到下次（会在下次重新生成有限候选继续尝试）
                if not bought_for_this_slot:
                    remaining_unfilled_slots.append(slot_weight_info)
                    completion_delayed_slots.append(original_stock)

            # 更新槽位信息
            slot_info["attempts"] += 1
            slot_info["unfilled_slot_weights"] = remaining_unfilled_slots
            self.completion_stats["completion_attempts"] += 1

            # 如果已经全部补齐，从待补齐列表中移除
            if not remaining_unfilled_slots:
                completed_signal_dates.append(signal_date)

        # 清理已完成或放弃的槽位
        for signal_date in completed_signal_dates:
            del self.unfilled_slots[signal_date]

        if completion_success_items or completion_delayed_slots:
            remaining_count = sum(
                len(slot_info.get("unfilled_slot_weights", []))
                for slot_info in self.unfilled_slots.values()
            )
            logger.info(
                self._format_completion_summary(
                    success_items=completion_success_items,
                    delayed_slots=completion_delayed_slots,
                    remaining_count=remaining_count,
                )
            )

    def _evaluate_profit_extension_for_stock(
        self,
        stock: str,
        date: pd.Timestamp,
        info: Dict,
        holding_days: int,
        profit_rate: float,
    ):
        """评估单只持仓是否应盈利延续保护（不卖出）。

        Args:
            stock: 股票代码
            date: 评估日期
            info: 持仓信息字典
            holding_days: 已持有交易日数
            profit_rate: 当前盈亏率（后复权口径）

        Returns:
            (should_extend: bool, log_detail: str)
        """
        if not self.enable_profit_based_holding:
            return False, ""

        if self.profit_extension_mode == "disabled":
            return False, "模式=disabled"

        within_extension_window = (
            holding_days < self.holding_period + self.profit_extension_days
        )

        if not within_extension_window:
            return False, (
                f"超过延续窗口(持有{holding_days}天, "
                f"上限{self.holding_period + self.profit_extension_days}天)"
            )

        if self.profit_extension_mode == "strength":
            if self.holding_strength_scorer is not None:
                breakdown = self.holding_strength_scorer.score(
                    stock=stock,
                    date=date,
                    position_info=info,
                    profit_rate=profit_rate,
                )
                if breakdown.total >= self.profit_extension_strength_threshold:
                    return True, (
                        f"强势度={breakdown.total:.3f} "
                        f">= 阈值={self.profit_extension_strength_threshold:.2f}"
                    )
                else:
                    return False, (
                        f"强势度={breakdown.total:.3f} "
                        f"< 阈值={self.profit_extension_strength_threshold:.2f}"
                    )
            return False, "强势度评分器缺失"

        # pnl 模式（默认）
        if profit_rate >= self.profit_extension_threshold:
            return True, (
                f"盈亏={profit_rate:.2%} "
                f">= 阈值={self.profit_extension_threshold:.2%}"
            )
        return False, (
            f"盈亏={profit_rate:.2%} "
            f"< 阈值={self.profit_extension_threshold:.2%}"
        )

    def _queue_rebalance_sells(
        self,
        date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        date_to_idx: Dict,
    ) -> None:
        """调仓日同步生成卖出信号：将当前非保护持仓排队到 T+1 卖出。

        使卖出与买入在同一交易日执行，消除调仓日卖出滞后一天的偏差。
        与 _check_and_sell 的职责互补：后者处理日常持有期到期/条件卖出，
        本方法仅在信号成功入队列的调仓日触发，针对即将到期的持仓提前排队。

        Args:
            date: 当前调仓日（信号生成日）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.positions:
            return

        current_idx = date_to_idx.get(date)
        if current_idx is None:
            return

        # 获取新信号中的股票（这些股票应保留，不卖出）
        signal_data = self.pending_signals.get(date, {})
        if isinstance(signal_data, dict) and "signals" in signal_data:
            new_signal_stocks = set(signal_data["signals"].keys())
        else:
            new_signal_stocks = set()

        sell_count = 0
        protected_count = 0

        for stock, info in list(self.positions.items()):
            # 跳过已在其他卖出队列中的持仓
            if stock in self.pending_condition_sells or stock in self.pending_stop_loss_sells:
                continue

            # 保留新信号中的股票（加仓/维持，不卖出）
            if stock in new_signal_stocks:
                continue

            buy_date = info.get("buy_date")
            buy_idx = date_to_idx.get(buy_date) if buy_date else None
            if buy_idx is None:
                continue

            holding_days = current_idx - buy_idx

            # 仅在持仓即将到期时提前触发（holding_period-1 天）
            # 若 holding_period=1，则 holding_days >= 0 全部触发
            if holding_days < max(0, self.holding_period - 1):
                continue

            # 评估盈利延续保护
            if self.enable_profit_based_holding:
                current_pnl_price = self._get_pnl_price(date, stock)
                if current_pnl_price is None:
                    continue
                buy_pnl_price = info.get("buy_pnl_price")
                if (
                    current_pnl_price
                    and buy_pnl_price
                    and not pd.isna(buy_pnl_price)
                    and buy_pnl_price > 0
                ):
                    profit_rate = (current_pnl_price - buy_pnl_price) / buy_pnl_price
                else:
                    profit_rate = 0.0

                should_extend, _detail = self._evaluate_profit_extension_for_stock(
                    stock=stock,
                    date=date,
                    info=info,
                    holding_days=holding_days,
                    profit_rate=profit_rate,
                )
                if should_extend:
                    protected_count += 1
                    continue

            # 排队到 T+1 卖出
            self.pending_condition_sells[stock] = {
                "trigger_date": date,
                "sell_type": "rebalance",
            }
            sell_count += 1

        if sell_count > 0 or protected_count > 0:
            parts = [f"调仓日同步卖出: {sell_count} 只排队到 T+1 卖出"]
            if protected_count > 0:
                parts.append(f"{protected_count} 只盈利延续保护")
            logger.info("，".join(parts))

    def _check_and_sell(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """检查卖出条件并生成 T0 卖出信号

        - 整体止盈、亏损提前换出：写入 pending_condition_sells 队列（Tn+1 执行）
        - 持有期到期、盈利延续到期：写入 pending_condition_sells 队列（Tn+1 执行）

        Args:
            date: 当前日期
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        profit_extension_items: List[Dict[str, Optional[float]]] = []
        holding_period_sell_slot_weights: List[Dict[str, float]] = []

        current_idx = date_to_idx.get(date)
        if current_idx is None:
            return

        portfolio_value_for_planning = self._calculate_portfolio_value(date)

        # 过滤已在待卖队列中的持仓（避免重复）
        positions_to_check = {
            stock: info
            for stock, info in self.positions.items()
            if stock not in self.pending_condition_sells
            and stock not in self.pending_stop_loss_sells
        }

        # ── 整体持仓止盈检查（Tn 检查，写入队列，Tn+1 执行）──────────
        # 盈亏基准：上次调仓日的组合净值（衡量"本轮调仓以来的盈利"）
        if (
            self.take_profit_threshold is not None
            and positions_to_check
            and self._last_rebalance_nav is not None
            and self._last_rebalance_nav > 0
        ):
            current_nav = self._calculate_portfolio_value(date)
            portfolio_profit_rate = (
                current_nav - self._last_rebalance_nav
            ) / self._last_rebalance_nav
            if portfolio_profit_rate >= self.take_profit_threshold:
                n_positions = len(positions_to_check)
                self._record_take_profit_summary(
                    profit_rate=portfolio_profit_rate,
                    threshold=self.take_profit_threshold,
                    n_positions=n_positions,
                )
                for stock in positions_to_check:
                    self.pending_condition_sells[stock] = {
                        "trigger_date": date,
                        "sell_type": "take_profit",
                    }
                # 暂存止盈元数据，延迟到执行日处理补位和NAV重置
                self._pending_take_profit_info = {
                    "trigger_date": date,
                    "n_positions": n_positions,
                    "ranked_candidates": list(self._last_ranked_candidates),
                    "signal_date": self._last_signal_date or date,
                }
                return  # 跳过后续逐只判断
        # ── 整体止盈检查结束 ──────────────────────────────────────────

        for stock, info in positions_to_check.items():
            buy_date = info["buy_date"]
            buy_idx = date_to_idx.get(buy_date)

            # 以实际买入日作为持有期起点，确保每只股票都持满 holding_period 个交易日
            # （原以 signal_date 为起点会导致补齐仓位实际持有天数不足，低估收益、高估换手率）
            anchor_idx = buy_idx
            if anchor_idx is None:
                signal_date = info.get("signal_date", buy_date)
                logger.warning(
                    f"股票 {stock} 买入日期 {buy_date}（信号日 {signal_date}）不在交易日映射中"
                )
                continue

            # 计算持有天数（交易日）
            holding_days = current_idx - anchor_idx

            # ── 表现弱势退出：纯价格表现评估，独立于 enable_profit_based_holding ──
            if (
                self.weakness_exit_monitor is not None
                and stock not in self.pending_condition_sells
            ):
                should_exit, weakness_detail = self._evaluate_weakness_exit(
                    stock=stock,
                    date=date,
                    position_info=info,
                    holding_days=holding_days,
                    profit_rate=0.0,
                    trading_dates=trading_dates,
                    date_to_idx=date_to_idx,
                )
                if should_exit:
                    self.pending_condition_sells[stock] = {
                        "trigger_date": date,
                        "sell_type": "weakness_exit",
                        "weakness_detail": weakness_detail,
                    }
                    continue

            if self.enable_profit_based_holding:
                # 计算当前盈亏率（使用后复权价格口径）
                current_pnl_price = self._get_pnl_price(date, stock)
                if current_pnl_price is None:
                    # 停牌或无价格数据时，无法评估当前盈亏，跳过亏损提前换出检查
                    # 注意：不能用 buy_trade_price 作为 fallback，因为它是不复权价格，
                    # 与后复权的 buy_pnl_price 混用会导致盈亏率严重失真
                    continue
                buy_pnl_price = info.get("buy_pnl_price")
                if (
                    current_pnl_price
                    and buy_pnl_price
                    and not pd.isna(buy_pnl_price)
                    and buy_pnl_price > 0
                ):
                    profit_rate = (current_pnl_price - buy_pnl_price) / buy_pnl_price
                else:
                    profit_rate = 0.0

                early_exit_holding = max(
                    1, int(self.holding_period * self.early_exit_holding_ratio)
                )

                if holding_days >= self.holding_period:
                    # ── 盈利延续持有决策 ─────────────────────────────
                    # 根据 profit_extension_mode 分派到不同判据:
                    #  - pnl:      原浮盈率 >= profit_extension_threshold(向后兼容)
                    #  - strength: 多维度强势度评分 >= profit_extension_strength_threshold
                    #  - disabled: 不延续,直接卖出
                    should_extend = False
                    extend_log_detail = ""
                    within_extension_window = (
                        holding_days < self.holding_period + self.profit_extension_days
                    )

                    if self.profit_extension_mode == "disabled":
                        should_extend = False
                    elif self.profit_extension_mode == "strength":
                        if within_extension_window and self.holding_strength_scorer is not None:
                            breakdown = self.holding_strength_scorer.score(
                                stock=stock,
                                date=date,
                                position_info=info,
                                profit_rate=profit_rate,
                            )
                            if breakdown.total >= self.profit_extension_strength_threshold:
                                should_extend = True
                                extend_log_detail = (
                                    f"强势度={breakdown.total:.3f} "
                                    f">= 阈值={self.profit_extension_strength_threshold:.2f}, "
                                    f"{breakdown.to_log_str()}"
                                )
                            else:
                                extend_log_detail = (
                                    f"强势度={breakdown.total:.3f} "
                                    f"< 阈值={self.profit_extension_strength_threshold:.2f}, "
                                    f"{breakdown.to_log_str()}"
                                )
                    else:  # pnl 模式(默认,向后兼容)
                        if (
                            profit_rate >= self.profit_extension_threshold
                            and within_extension_window
                        ):
                            should_extend = True
                            extend_log_detail = (
                                f"盈亏={profit_rate:.2%} "
                                f">= 阈值={self.profit_extension_threshold:.2%}"
                            )

                    if should_extend:
                        score = breakdown.total if self.profit_extension_mode == "strength" else None
                        profit_extension_items.append(
                            {
                                "stock": stock,
                                "holding_days": holding_days,
                                "profit_rate": profit_rate,
                                "score": score,
                            }
                        )
                        continue  # 延续持有，跳过卖出

                    # strength 模式下打印未延续的分项评分(便于归因)
                    if (
                        self.profit_extension_mode == "strength"
                        and extend_log_detail
                        and self.verbose
                    ):
                        logger.info(
                            f"  持有期满不延续[strength]: {stock} 持有{holding_days}天, "
                            f"{extend_log_detail}"
                        )

                    # 持有期到期（含延续到期）→ T0 生成卖出信号，T+1 执行
                    self.pending_condition_sells[stock] = {
                        "trigger_date": date,
                        "sell_type": "holding_period",
                    }
                    holding_period_sell_slot_weights.append(
                        {
                            "stock": stock,
                            "weight": self._get_position_weight_for_planning(
                                date,
                                stock,
                                portfolio_value=portfolio_value_for_planning,
                            ),
                        }
                    )
                else:
                    # 计算实际止损阈值（ATR 动态 or 固定）
                    threshold = self.early_exit_loss_threshold
                    threshold_desc = f"固定({threshold:.2%})"
                    if self.use_atr_for_early_exit:
                        buy_atr_pct = info.get("buy_atr_pct")
                        if buy_atr_pct is not None and not np.isnan(buy_atr_pct):
                            threshold = -self.atr_multiplier * buy_atr_pct
                            threshold_desc = f"ATR动态({threshold:.2%})"
                        else:
                            threshold_desc += "(ATR缺失,用固定)"

                    if holding_days >= early_exit_holding and profit_rate <= threshold:
                        # strength_veto 二次确认：评分高于保护阈值时否决卖出（缓刑）
                        if (
                            self.early_exit_mode == "strength_veto"
                            and self.early_exit_strength_scorer is not None
                        ):
                            reprieve_count = self._early_exit_reprieve_counts.get(
                                stock, 0
                            )
                            if reprieve_count < self.early_exit_max_reprieves:
                                breakdown = self.early_exit_strength_scorer.score(
                                    stock=stock,
                                    date=date,
                                    position_info=info,
                                    profit_rate=profit_rate,
                                )
                                if (
                                    breakdown.total
                                    >= self.early_exit_strength_protect_threshold
                                ):
                                    self._early_exit_reprieve_counts[stock] = (
                                        reprieve_count + 1
                                    )
                                    logger.warning(
                                        f"  亏损换出否决[strength_veto]: {stock} "
                                        f"持有{holding_days}天, "
                                        f"盈亏={profit_rate:.2%} <= {threshold_desc}, "
                                        f"但强势度={breakdown.total:.3f} >= "
                                        f"{self.early_exit_strength_protect_threshold:.2f}"
                                        f", 缓刑({reprieve_count + 1}/"
                                        f"{self.early_exit_max_reprieves}), "
                                        f"{breakdown.to_log_str()}"
                                    )
                                    continue  # 否决卖出，跳过

                        # 亏损提前换出 → 盘后发现，写入队列 Tn+1 执行
                        self._record_early_exit_trigger(
                            stock=stock,
                            holding_days=holding_days,
                            profit_rate=profit_rate,
                        )
                        self.pending_condition_sells[stock] = {
                            "trigger_date": date,
                            "sell_type": "early_exit",
                        }

                    # ── 时间止损：持仓超过指定天数后仍未达到最低盈利要求 → 提前换出 ──
                    if (
                        self.time_stop_loss_enabled
                        and stock not in self.pending_condition_sells
                        and holding_days >= self.time_stop_loss_days
                        and holding_days >= early_exit_holding
                    ):
                        if profit_rate < self.time_stop_loss_profit_ratio:
                            # strength_veto 二次确认（共用 early_exit_strength_scorer）
                            if (
                                self.early_exit_mode == "strength_veto"
                                and self.early_exit_strength_scorer is not None
                            ):
                                reprieve_count = self._early_exit_reprieve_counts.get(
                                    stock, 0
                                )
                                if reprieve_count < self.early_exit_max_reprieves:
                                    breakdown = self.early_exit_strength_scorer.score(
                                        stock=stock,
                                        date=date,
                                        position_info=info,
                                        profit_rate=profit_rate,
                                    )
                                    if (
                                        breakdown.total
                                        >= self.early_exit_strength_protect_threshold
                                    ):
                                        self._early_exit_reprieve_counts[stock] = (
                                            reprieve_count + 1
                                        )
                                        self._record_time_stop_loss_veto(
                                            stock=stock,
                                            holding_days=holding_days,
                                            profit_rate=profit_rate,
                                            score=breakdown.total,
                                        )
                                        continue  # 否决卖出

                            self._record_time_stop_loss_trigger(
                                stock=stock,
                                holding_days=holding_days,
                                profit_rate=profit_rate,
                            )
                            self.pending_condition_sells[stock] = {
                                "trigger_date": date,
                                "sell_type": "time_stop_loss",
                            }

            else:
                # 原始逻辑：达到持有期后在 T0 生成卖出信号，T+1 执行
                if holding_days >= self.holding_period:
                    self.pending_condition_sells[stock] = {
                        "trigger_date": date,
                        "sell_type": "holding_period",
                    }
                    # 与动态持仓分支保持一致：持有期卖出时同步生成次日补位计划，
                    # 避免出现 T+1 只卖不买导致空仓/短周期抖动。
                    holding_period_sell_slot_weights.append(
                        {
                            "stock": stock,
                            "weight": self._get_position_weight_for_planning(
                                date,
                                stock,
                                portfolio_value=portfolio_value_for_planning,
                            ),
                        }
                    )

        if profit_extension_items:
            self._record_profit_extension_count(len(profit_extension_items))
            logger.info(self._format_profit_extension_summary(profit_extension_items))

        if holding_period_sell_slot_weights:
            self._queue_condition_sell_refill_signal(
                date=date,
                slot_weights=holding_period_sell_slot_weights,
                price_data=self.price_data_cache,
                date_to_idx=date_to_idx,
            )

    def _evaluate_weakness_exit(
        self,
        stock: str,
        date: pd.Timestamp,
        position_info: dict,
        holding_days: int,
        profit_rate: float,
        trading_dates: list,
        date_to_idx: dict,
    ):
        """评估是否应因表现弱势而提前退出。

        使用 WeaknessExitMonitor 的 4 维纯价格表现评分。
        """
        if self.weakness_exit_monitor is None:
            return False, {}

        # 获取买入以来的后复权价格序列
        buy_date = position_info.get("buy_date")
        if buy_date is None:
            return False, {}

        buy_idx = date_to_idx.get(buy_date)
        current_idx = date_to_idx.get(date)
        if buy_idx is None or current_idx is None:
            return False, {}

        # 从 price_data_cache 提取价格序列
        pnl_prices = {}
        if self.price_data_cache is not None:
            stock_data = self.price_data_cache[
                self.price_data_cache["ts_code"] == stock
            ]
            for _, row in stock_data.iterrows():
                trade_date_str = to_trade_date_str(row["trade_date"])
                close_val = row.get("close_adj", row.get("close"))
                if pd.notna(close_val) and close_val > 0:
                    pnl_prices[trade_date_str] = float(close_val)

        if len(pnl_prices) < 2:
            return False, {}

        # 构建日期排序的价格 Series
        sorted_dates = sorted(pnl_prices.keys())
        price_values = [pnl_prices[d] for d in sorted_dates]
        price_series = pd.Series(price_values, index=sorted_dates)

        # 构建所有持仓的 profit_rate 字典
        all_profits = {}
        for s, info in self.positions.items():
            current_pnl = self._get_pnl_price(date, s)
            buy_pnl = info.get("buy_pnl_price")
            if current_pnl and buy_pnl and buy_pnl > 0:
                all_profits[s] = (current_pnl - buy_pnl) / buy_pnl
            else:
                all_profits[s] = 0.0

        # 可选行业动量排名
        industry_rank = None
        if (
            self.weakness_exit_monitor.config.industry_filter
            and hasattr(self, "_get_holding_features_row")
        ):
            try:
                features_row = self._get_holding_features_row(date, stock)
                if features_row is not None and "ind_momentum_rank" in features_row.index:
                    rank_val = features_row["ind_momentum_rank"]
                    if not pd.isna(rank_val):
                        industry_rank = float(rank_val)
            except Exception:
                pass

        weakness_score, breakdown = self.weakness_exit_monitor.evaluate(
            stock=stock,
            price_series=price_series,
            all_positions_profit=all_profits,
            holding_days=holding_days,
            industry_rank=industry_rank,
        )

        date_str = to_trade_date_str(date)
        consec = self.weakness_exit_monitor.update(stock, date_str, weakness_score)

        # 未达最低持有天数：仅跟踪不触发
        min_hold = self.weakness_exit_monitor.config.min_holding_days
        if holding_days < min_hold:
            return False, breakdown

        if weakness_score >= self.weakness_exit_monitor.config.threshold:
            if consec >= self.weakness_exit_monitor.config.consecutive_days:
                logger.warning(
                    f"  表现弱势退出: {stock} 持有{holding_days}天, "
                    f"评分={weakness_score:.3f}, 连弱{consec}天, "
                    f"排名={breakdown.get('pnl_rank', 0):.2f}, "
                    f"连跌={breakdown.get('down_streak', 0):.2f}, "
                    f"回撤={breakdown.get('drawdown', 0):.2f}, "
                    f"回升={breakdown.get('recovery', 0):.2f}"
                )
                return True, breakdown

        return False, breakdown

    def _execute_pending_condition_sells(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """执行待条件卖出操作（Tn+1 日执行）

        Args:
            date: 当前日期（执行日，Tn+1）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.pending_condition_sells:
            return

        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx == 0:
            return

        trigger_date = trading_dates[current_idx - 1]

        # 筛选前一交易日触发的条件卖出
        stocks_to_sell = [
            (stock, info)
            for stock, info in list(self.pending_condition_sells.items())
            if info["trigger_date"] == trigger_date
        ]
        if not stocks_to_sell:
            return

        # 执行卖出
        for stock, info in stocks_to_sell:
            if stock not in self.positions:
                self.pending_condition_sells.pop(stock, None)
                continue
            self._sell_stock(date, stock, sell_type=info["sell_type"])
            self.pending_condition_sells.pop(stock, None)

        # 处理延迟的止盈元数据（补位 + NAV 重置）
        if (
            self._pending_take_profit_info
            and self._pending_take_profit_info["trigger_date"] == trigger_date
        ):
            tp = self._pending_take_profit_info
            self._pending_take_profit_info = None
            # 重置调仓基准 NAV（卖出后的现金净值），使"本调仓"从 0 重新计量
            self._last_rebalance_nav = self._calculate_portfolio_value(date)
            # 写入补位 unfilled_slots
            n_positions = tp["n_positions"]
            if (
                self.take_profit_refill
                and self.enable_position_completion
                and tp["ranked_candidates"]
                and tp["trigger_date"] not in self.unfilled_slots
            ):
                weight = 1.0 / n_positions if n_positions > 0 else 0.0
                unfilled_slot_weights = [
                    {"stock": f"__tp_{i}__", "weight": weight, "filled": False}
                    for i in range(n_positions)
                ]
                self.unfilled_slots[tp["trigger_date"]] = {
                    "unfilled_count": n_positions,
                    "unfilled_slot_weights": unfilled_slot_weights,
                    "target_n": n_positions,
                    "ranked_candidates": list(tp["ranked_candidates"]),
                    "signal_date": tp["signal_date"],
                    "first_attempt_date": date,  # 执行日（Tn+1），补位从此日起算
                    "attempts": 0,
                    "tranche_idx": 0,
                }
                logger.info(
                    f"  整体止盈补位: 已写入 {n_positions} 个补位槽，自动补仓"
                    f"（候选池 {len(tp['ranked_candidates'])} 只）"
                )

        # 实际卖出明细统一在每日总结下一行展示，这里不再单独输出卖出执行日志。

    def _check_stop_loss(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """检查止损触发条件（T 日检查，生成 T+1 卖出信号）

        Args:
            date: 当前日期（检查日）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.stop_loss_monitor:
            return

        trade_date_str = to_trade_date_str(date)

        # 过滤掉已在待止损卖出队列中的持仓
        positions_to_check = {
            stock: info
            for stock, info in self.positions.items()
            if stock not in self.pending_stop_loss_sells
        }
        if not positions_to_check:
            return

        # 构建价格和跌停信息
        date_quote = self.price_data_cache[self.price_data_cache["trade_date"] == trade_date_str]
        prices: Dict[str, float] = {}
        limit_down_info: Dict[str, bool] = {}
        for stock in positions_to_check:
            price = self._get_trade_price(date, stock)
            if price is not None:
                prices[stock] = price
            if not date_quote.empty:
                stock_quote = date_quote[date_quote["ts_code"] == stock]
                if not stock_quote.empty and "is_limit_down" in stock_quote.columns:
                    limit_down_info[stock] = bool(stock_quote["is_limit_down"].iloc[0] == 1)

        # 获取停牌日历
        suspend_calendar = None
        try:
            suspend_calendar = self._get_suspend_calendar()
        except Exception as e:
            logger.warning(f"停牌日历初始化失败（{e}），将跳过停牌检查")

        # 调用公共止损检查
        actions = check_positions_stop_loss(
            positions=positions_to_check,
            stop_loss_monitor=self.stop_loss_monitor,
            prices=prices,
            limit_down_info=limit_down_info,
            suspend_calendar=suspend_calendar,
            trade_date=trade_date_str,
            verbose=self.verbose,
        )

        # 将结果转换为引擎内部的 pending_stop_loss_sells 格式
        for action in actions:
            self.pending_stop_loss_sells[action.ts_code] = {
                "trigger_date": date,
                "reason": action.reason,
                "trigger_type": action.trigger_type or "unknown",
            }
            if self.verbose:
                logger.warning(
                    f"  止损触发: {date.date()} {action.ts_code}, 原因: {action.reason}, "
                    f"将在下一交易日执行卖出"
                )

    def _execute_pending_stop_loss_sells(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """执行待止损卖出操作（T+1 日执行）

        Args:
            date: 当前日期（执行日，T+1）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.stop_loss_monitor or not self.pending_stop_loss_sells:
            return

        # 查找前一个交易日触发的止损
        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx == 0:
            return

        trigger_date = trading_dates[current_idx - 1]

        # 执行前一交易日触发的止损卖出
        stocks_to_sell = []
        for stock, info in list(self.pending_stop_loss_sells.items()):
            if info["trigger_date"] == trigger_date:
                stocks_to_sell.append((stock, info))

        if not stocks_to_sell:
            return

        # 执行卖出
        for stock, info in stocks_to_sell:
            # 检查股票是否还在持仓中（可能已被正常调仓卖出）
            if stock not in self.positions:
                # 从待卖出队列中移除
                self.pending_stop_loss_sells.pop(stock, None)
                continue

            # 执行止损卖出
            self._sell_stock(
                date,
                stock,
                sell_type="stop_loss",
                sell_reason=info["reason"],
                trigger_type=info["trigger_type"],
            )

            # 从待卖出队列中移除
            self.pending_stop_loss_sells.pop(stock, None)

        # 实际卖出明细统一在每日总结下一行展示，这里不再单独输出止损卖出执行日志。

    def _prepare_price_index(self, price_data: pd.DataFrame) -> None:
        """准备价格索引（使用 MultiIndex，替代嵌套字典）

        构建四套价格序列：
        - trade_price_index: 收盘成交价格（不复权 close）
        - pnl_price_index: 收盘绩效价格（后复权 close_adj）
        - trade_price_open_index: 开盘成交价格（不复权 open）
        - pnl_price_open_index: 开盘绩效价格（后复权 open_adj）

        Args:
            price_data: 价格数据，需包含 ts_code, trade_date, close, open（可选），close_adj（可选），open_adj（可选）
        """
        logger.info("开始准备价格索引...")

        # 检查必需列
        if "close" not in price_data.columns:
            raise ValueError("价格数据缺少 'close' 列，无法进行回测")

        # 转换日期列为 datetime（向量化操作，避免 iterrows）
        if not pd.api.types.is_datetime64_any_dtype(price_data["trade_date"]):
            # 创建副本以避免修改原始数据
            price_data = price_data.copy()
            price_data["trade_date"] = pd.to_datetime(price_data["trade_date"])

        # 构建收盘成交价格索引（不复权 close）
        trade_price_df = price_data[["trade_date", "ts_code", "close"]].copy()
        trade_price_df.set_index(["trade_date", "ts_code"], inplace=True)
        self.trade_price_index = trade_price_df["close"]

        # 构建收盘绩效价格索引（后复权 close_adj）
        if "close_adj" in price_data.columns:
            pnl_price_df = price_data[["trade_date", "ts_code", "close_adj"]].copy()
            pnl_price_df.set_index(["trade_date", "ts_code"], inplace=True)
            self.pnl_price_index = pnl_price_df["close_adj"]
            logger.info("价格索引构建完成: 收盘成交价格=close, 收盘绩效价格=close_adj")
        else:
            # 如果缺少 close_adj，回退到 close
            logger.warning(f"价格数据缺少 'close_adj' 列，绩效价格将使用 'close' 列（不复权）")
            self.pnl_price_index = self.trade_price_index.copy()
            logger.info("价格索引构建完成: 收盘成交价格=close, 收盘绩效价格=close（退化）")

        # 构建开盘成交价格索引（不复权 open）
        if "open" in price_data.columns:
            # 过滤掉NaN值，只保留有效的开盘价
            open_data = price_data[["trade_date", "ts_code", "open"]].copy()
            open_data = open_data[open_data["open"].notna()]

            if len(open_data) > 0:
                open_data.set_index(["trade_date", "ts_code"], inplace=True)
                self.trade_price_open_index = open_data["open"]
                logger.info(f"开盘价格索引构建完成: 开盘成交价格=open, 共{len(open_data)}条记录")
            else:
                logger.warning(f"价格数据的 'open' 列全部为NaN，开盘价格将使用收盘价格代替")
                self.trade_price_open_index = self.trade_price_index.copy()
        else:
            logger.warning(f"价格数据缺少 'open' 列，开盘价格将使用收盘价格代替")
            self.trade_price_open_index = self.trade_price_index.copy()

        # 构建开盘绩效价格索引（后复权 open_adj）
        if "open_adj" in price_data.columns:
            # 过滤掉NaN值，只保留有效的开盘绩效价格
            open_adj_data = price_data[["trade_date", "ts_code", "open_adj"]].copy()
            open_adj_data = open_adj_data[open_adj_data["open_adj"].notna()]

            if len(open_adj_data) > 0:
                open_adj_data.set_index(["trade_date", "ts_code"], inplace=True)
                self.pnl_price_open_index = open_adj_data["open_adj"]
                logger.info(
                    f"开盘绩效价格索引构建完成: 开盘绩效价格=open_adj, 共{len(open_adj_data)}条记录"
                )
            else:
                # 如果open_adj全部为NaN，尝试使用open
                if "open" in price_data.columns:
                    logger.warning(
                        f"价格数据的 'open_adj' 列全部为NaN，开盘绩效价格将使用 'open' 列（不复权）"
                    )
                    self.pnl_price_open_index = self.trade_price_open_index.copy()
                else:
                    logger.warning(
                        f"价格数据缺少 'open' 和 'open_adj' 列，开盘绩效价格将使用收盘绩效价格代替"
                    )
                    self.pnl_price_open_index = self.pnl_price_index.copy()
        else:
            # 如果缺少 open_adj，回退到 open 或 close_adj
            if "open" in price_data.columns:
                # 如果有 open 但没有 open_adj，使用 open
                logger.warning(
                    f"价格数据缺少 'open_adj' 列，开盘绩效价格将使用 'open' 列（不复权）"
                )
                self.pnl_price_open_index = self.trade_price_open_index.copy()
            else:
                # 如果连 open 都没有，使用 close_adj
                logger.warning(f"价格数据缺少 'open_adj' 列，开盘绩效价格将使用收盘绩效价格代替")
                self.pnl_price_open_index = self.pnl_price_index.copy()

    def _get_trade_price(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取收盘成交价格（不复权 close）

        Args:
            date: 日期
            stock: 股票代码

        Returns:
            成交价格，如果不存在则返回 None
        """
        try:
            return self.trade_price_index.loc[(date, stock)]
        except KeyError:
            return None

    def _get_pnl_price(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取收盘绩效价格（后复权 close_adj）

        Args:
            date: 日期
            stock: 股票代码

        Returns:
            绩效价格，如果不存在则返回 None
        """
        try:
            return self.pnl_price_index.loc[(date, stock)]
        except KeyError:
            return None

    def _get_trade_price_open(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取开盘成交价格（不复权 open）

        如果开盘价格不存在，返回 None。调用者应处理降级策略（如使用收盘价）。

        Args:
            date: 日期
            stock: 股票代码

        Returns:
            开盘成交价格，如果不存在则返回 None
        """
        try:
            return self.trade_price_open_index.loc[(date, stock)]
        except KeyError:
            return None

    def _get_pnl_price_open(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取开盘绩效价格（后复权 open_adj）

        如果开盘绩效价格不存在，返回 None。调用者应处理降级策略（如使用收盘绩效价格）。

        Args:
            date: 日期
            stock: 股票代码

        Returns:
            开盘绩效价格，如果不存在则返回 None
        """
        try:
            return self.pnl_price_open_index.loc[(date, stock)]
        except KeyError:
            return None

    def _calculate_volatility(self, stock: str, end_date: pd.Timestamp) -> float:
        """计算个股历史波动率（基于绩效价格，避免未来函数）

        使用 end_date 之前的 vol_window 个交易日的收益率计算波动率

        Args:
            stock: 股票代码
            end_date: 结束日期（不包含，只使用该日期之前的数据）

        Returns:
            年化波动率
        """
        try:
            # 获取该股票的所有绩效价格（按日期排序）
            stock_prices = self.pnl_price_index.xs(stock, level="ts_code").sort_index()

            # 筛选 end_date 之前的数据
            stock_prices = stock_prices[stock_prices.index < end_date]

            if len(stock_prices) < 2:
                return self.vol_epsilon

            # 取最近 vol_window 个交易日
            recent_prices = stock_prices.iloc[-self.vol_window :]

            if len(recent_prices) < 2:
                return self.vol_epsilon

            # 计算日收益率
            returns = recent_prices.pct_change().dropna()

            if len(returns) < 2:
                return self.vol_epsilon

            # 计算波动率（年化，假设每年252个交易日）
            vol = returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)

            # 确保波动率不低于 epsilon
            return max(vol, self.vol_epsilon)

        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"计算 {stock} 波动率时出错: {e}，使用默认值 {self.vol_epsilon}")
            return self.vol_epsilon

    def _apply_risk_budget(self, signals: Dict[str, float], date: pd.Timestamp) -> Dict[str, float]:
        """应用风险预算（波动率缩放）

        调整权重: adj_weight ∝ raw_weight / volatility
        然后归一化使权重和为1

        Args:
            signals: 原始信号 {stock: weight}
            date: 当前日期（买入日期）

        Returns:
            调整后的信号 {stock: adj_weight}
        """
        if not signals:
            return signals

        # 计算每只股票的波动率（使用 date 之前的数据）
        volatilities = {}
        for stock in signals:
            vol = self._calculate_volatility(stock, date)
            volatilities[stock] = vol

        # 计算调整后的权重: raw_weight / volatility
        adj_weights = {}
        for stock, weight in signals.items():
            adj_weights[stock] = weight / volatilities[stock]

        # 归一化
        total_adj_weight = sum(adj_weights.values())
        if total_adj_weight > 0:
            for stock in adj_weights:
                adj_weights[stock] /= total_adj_weight
        else:
            # 如果总权重为0，均分
            n = len(adj_weights)
            for stock in adj_weights:
                adj_weights[stock] = 1.0 / n if n > 0 else 0.0

        return adj_weights

    def _get_rebalance_dates(self, trading_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, int]:
        """获取调仓日期及对应的 tranche 索引

        Args:
            trading_dates: 交易日列表

        Returns:
            字典 {日期: tranche_idx}。stagger_tranches=1 时所有日期的 tranche 均为 0。
        """
        n = self.rebalance_freq
        if n <= 0:
            raise ValueError(f"调仓频率必须为正整数，当前值: {n}")

        if self.stagger_tranches <= 1:
            # 不分批：保持原有逻辑，所有调仓日 tranche=0
            return {trading_dates[i]: 0 for i in range(0, len(trading_dates), n)}

        # 分批调仓：K 个 tranche 各自错开 offset 天
        k = self.stagger_tranches
        offset = max(1, n // k)
        schedule = {}
        for t in range(k):
            start = t * offset
            for i in range(start, len(trading_dates), n):
                schedule[trading_dates[i]] = t
        return schedule

    def _process_pending_orders(self, date: pd.Timestamp) -> None:
        """处理延迟订单队列

        Args:
            date: 当前日期
        """
        if not self.pending_order_manager:
            return

        # 获取应重试的订单列表及已放弃的订单（仅 buy 订单会过期，sell 订单持续重试直至复牌）
        orders_to_retry, expired_orders = self.pending_order_manager.get_orders_to_retry(date)

        if not orders_to_retry:
            return

        # 获取当日行情数据
        trade_date_str = to_trade_date_str(date)
        date_quote = self.price_data_cache[self.price_data_cache["trade_date"] == trade_date_str]

        for order in orders_to_retry:
            # 检查是否可交易
            if date_quote.empty:
                # 当日行情数据为空，无法判断交易状态，继续延迟
                logger.warning(f"延迟订单 {order.stock} 在 {date.date()} 无行情数据，继续延迟")
                continue
            tradeable, reason = is_tradeable(
                order.stock, trade_date_str, date_quote, action=order.action
            )

            if tradeable:
                # 可交易，尝试执行
                if order.action == "buy":
                    self._buy_stock_direct(
                        date, order.stock, order.target_value, signal_date=order.signal_date
                    )
                    self.pending_order_manager.mark_success(date, order.stock, "buy")
                elif order.action == "sell":
                    self._sell_stock_direct(date, order.stock)
                    self.pending_order_manager.mark_success(date, order.stock, "sell")
            else:
                # 仍不可交易，更新延迟订单
                self.pending_order_manager.add_order(
                    stock=order.stock,
                    action=order.action,
                    current_date=date,
                    signal_date=order.signal_date,
                    target_value=order.target_value,
                    reason=reason,
                )

    def _buy_stock_with_status_check(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: Optional[pd.Timestamp] = None,
    ) -> None:
        """买入股票（带交易状态检查）

        如果启用延迟订单功能，会检查股票是否可交易（停牌、涨停）
        不可交易时加入延迟队列而非直接失败

        Args:
            date: 买入日期（T+1）
            stock: 股票代码
            target_value: 目标市值
            signal_date: 信号生成日期（用于延迟订单）
        """
        # 检查交易状态
        if self.enable_pending_order and self.price_data_cache is not None:
            trade_date_str = to_trade_date_str(date)
            date_quote = self.price_data_cache[
                self.price_data_cache["trade_date"] == trade_date_str
            ]
            if date_quote.empty:
                # 当日行情数据为空，无法判断交易状态，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action="buy",
                        current_date=date,
                        signal_date=signal_date or date,
                        target_value=target_value,
                        reason="无行情数据",
                    )
                if self.verbose:
                    logger.info(
                        f"买入延迟: {date.date()} {stock}, 原因: 无行情数据, "
                        f"目标市值: {target_value:.2f}"
                    )
                return
            tradeable, reason = is_tradeable(stock, trade_date_str, date_quote, action="buy")

            if not tradeable:
                # 不可交易，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action="buy",
                        current_date=date,
                        signal_date=signal_date or date,
                        target_value=target_value,
                        reason=reason,
                    )
                if self.verbose:
                    logger.info(
                        f"买入延迟: {date.date()} {stock}, 原因: {reason}, "
                        f"目标市值: {target_value:.2f}"
                    )
                return

        # 可交易，直接买入
        self._buy_stock_direct(date, stock, target_value, signal_date=signal_date)

    def _build_position_extra_info(self, date: pd.Timestamp, stock: str) -> Dict:
        """买入时附加额外元数据到 positions 字典（子类可覆写）

        默认返回空字典。engine_ml.py 覆写此方法以写入买入日 ATR 数据，
        供 _check_holding_periods 中的 ATR 动态止损使用。
        """
        return {}

    def _normalize_signals(self, signals: Dict[str, float], date: pd.Timestamp) -> Dict[str, float]:
        """将分数字典归一化为权重字典

        支持 4 种模式（由 self.position_sizing 控制）:
        - equal: 等权
        - score: 按预测分数线性加权
        - kelly: Kelly 最优仓位（基于分数 × 波动率估计）
        - half_kelly: 半 Kelly（Kelly 仓位的 50%，更保守）

        Kelly 公式: f* = μ / σ², 其中:
        - μ: 预期超额收益（用 ML 分数代理）
        - σ²: 收益率方差（从近期价格数据估计）
        最终 clip 到 [0, kelly_max_leverage], 再归一化总和为 1.0。
        """
        if not signals:
            return {}

        sizing = self.position_sizing

        if sizing == "equal":
            weight = 1.0 / len(signals)
            if self._normalize_log_count < 5:
                scores = list(signals.values())
                s_max, s_min = max(scores), min(scores)
                logger.info(
                    f"  [权重诊断 {self._normalize_log_count + 1}/5] equal, "
                    f"n={len(signals)}, 每只权重={weight:.4f}, "
                    f"分数范围=[{s_min:.4f}, {s_max:.4f}], 分数差距={s_max - s_min:.4f}"
                )
                self._normalize_log_count += 1
            return {stock: weight for stock in signals.keys()}

        elif sizing == "score":
            total_score = sum(signals.values())
            if total_score > 0:
                result = {stock: score / total_score for stock, score in signals.items()}
                if self._normalize_log_count < 5:
                    weights = sorted(result.values(), reverse=True)
                    w_max, w_min = weights[0], weights[-1]
                    eq_weight = 1.0 / len(result)
                    concentration = w_max / eq_weight
                    sample_stocks = sorted(result.items(), key=lambda x: x[1], reverse=True)[:3]
                    weights_str = ", ".join(
                        [f"{stock}: {weight:.4f}" for stock, weight in sample_stocks]
                    )
                    logger.info(
                        f"  [权重诊断 {self._normalize_log_count + 1}/5] score, "
                        f"n={len(result)}, 等权基准={eq_weight:.4f}, "
                        f"最高={w_max:.4f}({concentration:.1f}x), 最低={w_min:.4f}, "
                        f"示例（前3）: {weights_str}"
                    )
                    self._normalize_log_count += 1
                return result
            else:
                weight = 1.0 / len(signals)
                if self.verbose:
                    logger.warning(
                        f"所有分数 <= 0，回退到等权分配，每只股票权重 {weight:.4f}"
                    )
                return {stock: weight for stock in signals.keys()}

        elif sizing in ("kelly", "half_kelly"):
            return self._kelly_weights(signals, date, half=(sizing == "half_kelly"))

        else:
            weight = 1.0 / len(signals)
            return {stock: weight for stock in signals.keys()}

    def _kelly_weights(
        self,
        signals: Dict[str, float],
        date: pd.Timestamp,
        half: bool = False,
    ) -> Dict[str, float]:
        """计算 Kelly / 半 Kelly 仓位权重

        f* = score_rank / σ²，分数排名为主项，波动率负相关（低波动 → 更高权重）。

        对每只股票:
        1. score_rank = 分数百分位排名（0~1），量级稳定，避免原始分数与 σ² 量级不匹配
        2. f* = score_rank / σ²（分数高 + 波动低 → 权重高）
        3. 半 Kelly: 归一化后与等权混合（50% kelly + 50% 等权），比 kelly 更保守
        4. 归一化总和为 1.0，再按 kelly_max_leverage 做单股上限限制

        如果无法估计波动率，该股票仅用分数排名权重（等同 score 模式）。
        """
        n = len(signals)
        if n == 0:
            return {}

        # 计算分数百分位排名（0~1），分数最高的股票 rank=1，最低的 rank=1/n
        positive_stocks = {s: v for s, v in signals.items() if v > 0}
        if not positive_stocks:
            weight = 1.0 / n
            return {stock: weight for stock in signals}

        sorted_stocks = sorted(positive_stocks.items(), key=lambda x: x[1])
        m = len(sorted_stocks)
        score_ranks = {stock: (i + 1) / m for i, (stock, _) in enumerate(sorted_stocks)}

        # 计算每只股票的 1/σ²（无数据时用截面中位数）
        vol_adjusts = {}
        fallback_stocks = []
        for stock in positive_stocks:
            vol_sq = self._estimate_stock_variance(stock, date)
            if vol_sq is not None and vol_sq > 0:
                vol_adjusts[stock] = 1.0 / float(vol_sq)
            else:
                fallback_stocks.append(stock)

        if vol_adjusts:
            median_vol_adj = float(np.median(list(vol_adjusts.values())))
        else:
            median_vol_adj = 1.0
        for stock in fallback_stocks:
            vol_adjusts[stock] = median_vol_adj

        # f* = score_rank / σ²（标准 Kelly，低波动股获更高权重）
        raw_kelly = {
            stock: score_ranks[stock] * vol_adjusts[stock]
            for stock in positive_stocks
        }

        # fallback 股票（score <= 0）分配中位 kelly 值
        if raw_kelly:
            median_kelly = float(np.median(list(raw_kelly.values())))
        else:
            median_kelly = 1.0 / n
        for stock in signals:
            if stock not in raw_kelly:
                raw_kelly[stock] = median_kelly

        # 归一化总和为 1.0
        total = sum(raw_kelly.values())
        if total <= 0:
            weight = 1.0 / n
            return {stock: weight for stock in signals}
        kelly_weights = {stock: w / total for stock, w in raw_kelly.items()}

        # half_kelly: 50% kelly 权重 + 50% 等权，更保守，同时确保两种模式结果不同
        if half:
            eq_weight = 1.0 / n
            kelly_weights = {
                stock: 0.5 * w + 0.5 * eq_weight
                for stock, w in kelly_weights.items()
            }
            # 重新归一化（理论上和已为1，防浮点误差）
            total2 = sum(kelly_weights.values())
            if total2 > 0:
                kelly_weights = {s: w / total2 for s, w in kelly_weights.items()}

        result = kelly_weights

        # 按 kelly_max_leverage 做单股权重上限（迭代重归一化）
        if self.kelly_max_leverage < 1.0:
            for _ in range(10):
                capped = {s: min(w, self.kelly_max_leverage) for s, w in result.items()}
                cap_total = sum(capped.values())
                if cap_total <= 0:
                    break
                result = {s: w / cap_total for s, w in capped.items()}
                if all(w <= self.kelly_max_leverage + 1e-9 for w in result.values()):
                    break

        if self.verbose:
            mode_name = "half_kelly" if half else "kelly"
            sample = list(result.items())[:3]
            weights_str = ", ".join([f"{s}: {w:.4f}" for s, w in sample])
            logger.info(
                f"  权重方法: {mode_name}, 示例权重（前3只）: {weights_str}, "
                f"fallback={len(fallback_stocks)}只"
            )
        return result

    def _estimate_stock_variance(
        self, stock: str, date: pd.Timestamp
    ) -> Optional[float]:
        """估计股票近期收益率方差,供 Kelly 仓位计算使用

        从 price_data_cache 中取近 kelly_vol_window 日收盘价,计算日收益率方差。
        数据不足时返回 None。
        """
        if self.price_data_cache is None:
            return None

        date_str = to_trade_date_str(date)
        stock_data = self.price_data_cache[
            (self.price_data_cache["ts_code"] == stock)
            & (self.price_data_cache["trade_date"] <= date_str)
        ]

        if len(stock_data) < 20:
            return None

        # 取最近 kelly_vol_window 条
        stock_data = stock_data.sort_values("trade_date").tail(self.kelly_vol_window)

        # 优先使用后复权价格
        price_col = "close_adj" if "close_adj" in stock_data.columns else "close"
        prices = stock_data[price_col].values.astype(float)
        prices = prices[~np.isnan(prices)]
        if len(prices) < 10:
            return None

        log_returns = np.diff(np.log(prices))
        if len(log_returns) < 5:
            return None

        return float(np.var(log_returns))

    def _buy_stock_direct(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: Optional[pd.Timestamp] = None,
    ) -> None:
        """直接买入股票（不检查交易状态）

        内部使用，实际执行买入操作

        Args:
            date: 买入日期
            stock: 股票代码
            target_value: 目标市值
        """
        # 若已有持仓，跳过重复买入，避免覆盖持有期与成本基础导致计算错误
        if stock in self.positions:
            self._record_duplicate_buy_skip(
                stock=stock,
                buy_date=self.positions[stock]["buy_date"],
            )
            return

        # 获取成交价格（不复权 close）
        trade_price = self._get_trade_price(date, stock)
        if trade_price is None:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的成交价格，跳过买入")
            return

        # 获取绩效价格（后复权 close_adj）
        pnl_price = self._get_pnl_price(date, stock)
        if pnl_price is None:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的绩效价格，使用成交价格代替")
            pnl_price = trade_price

        # 按手买入（100股为一手）
        shares = int(target_value / trade_price / 100) * 100

        if shares == 0:
            return

        # 计算买入金额和成本（基于成交价格）
        amount = shares * trade_price
        cost = self.cost_model.calculate_buy_cost(amount)
        total_cost_cash = amount + cost  # 总现金支出（含手续费）

        if total_cost_cash > self.current_capital:
            # 资金不足，按可用资金买入
            # 确保有足够资金支付手续费
            if self.current_capital <= cost:
                # 资金不足以支付手续费，无法买入
                return

            shares = int((self.current_capital - cost) / trade_price / 100) * 100
            if shares == 0:
                return
            amount = shares * trade_price
            cost = self.cost_model.calculate_buy_cost(amount)
            total_cost_cash = amount + cost

        # 与纸面交易一致：过小仓位买入拦截（按买入后市值阈值）
        min_buy_value_threshold = self._get_min_buy_value_threshold(date)
        if min_buy_value_threshold > 0 and amount < min_buy_value_threshold:
            if self.verbose:
                logger.warning(
                    f"股票 {stock} 买入后市值 {amount:.2f} 低于阈值 "
                    f"{min_buy_value_threshold:.2f}（ratio={self.min_buy_value_ratio:.2f}），跳过买入"
                )
            return

        # 建立新持仓（记录买入的成交价格和绩效价格）
        self.positions[stock] = {
            "shares": shares,
            "buy_date": date,
            "signal_date": signal_date or date,
            "buy_trade_price": trade_price,  # 成交价格（不复权）
            "buy_pnl_price": pnl_price,  # 绩效价格（后复权）
            "buy_cost_cash": total_cost_cash,  # 总现金支出（含手续费）
        }
        # 子类可覆写 _build_position_extra_info 以附加额外元数据（如 ATR）
        extra = self._build_position_extra_info(date, stock)
        if extra:
            self.positions[stock].update(extra)

        self.current_capital -= total_cost_cash

        # 记录交易
        self.trades.append(
            {
                "date": date,
                "stock": stock,
                "action": "buy",
                "price": trade_price,  # 成交价格
                "shares": shares,
                "amount": amount,
                "cost": cost,
            }
        )

    def _buy_stock(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: Optional[pd.Timestamp] = None,
    ) -> None:
        """买入股票（在 T+1 日以收盘价买入）

        交易状态检查由调用方在执行阶段处理，这里只负责实际成交。

        Args:
            date: 买入日期（T+1）
            stock: 股票代码
            target_value: 目标市值
            signal_date: 触发本次买入计划的信号日期
        """
        # 直接买入，执行阶段的交易状态检查由上层调度负责。
        self._buy_stock_direct(date, stock, target_value, signal_date=signal_date)

    def _sell_stock(
        self,
        date: pd.Timestamp,
        stock: str,
        sell_type: str = "holding_period",
        sell_reason: Optional[str] = None,
        trigger_type: Optional[str] = None,
    ) -> None:
        """卖出股票（在 T+n 日以收盘价卖出）

        带交易状态检查的卖出方法。如果启用延迟订单功能，会检查股票是否可交易。

        Args:
            date: 卖出日期（T+n）
            stock: 股票代码
            sell_type: 卖出类型，'holding_period' 或 'stop_loss'
            sell_reason: 卖出原因描述（止损时使用）
            trigger_type: 触发类型（止损时使用）
        """
        self._sell_stock_with_status_check(date, stock, sell_type, sell_reason, trigger_type)

    def _sell_stock_with_status_check(
        self,
        date: pd.Timestamp,
        stock: str,
        sell_type: str = "holding_period",
        sell_reason: Optional[str] = None,
        trigger_type: Optional[str] = None,
    ) -> None:
        """卖出股票（带交易状态检查）

        如果启用延迟订单功能，会检查股票是否可交易（停牌或跌停）
        不可交易时加入延迟队列而非直接失败

        Args:
            date: 卖出日期
            stock: 股票代码
            sell_type: 卖出类型，'holding_period' 或 'stop_loss'
            sell_reason: 卖出原因描述（止损时使用）
            trigger_type: 触发类型（止损时使用）
        """
        # 检查交易状态
        if self.enable_pending_order and self.price_data_cache is not None:
            trade_date_str = to_trade_date_str(date)

            # 使用 SuspendCalendar 检查停牌状态
            is_suspended_flag = False
            suspend_calendar = None
            try:
                suspend_calendar = self._get_suspend_calendar()
                is_suspended_flag = suspend_calendar.is_suspended(stock, trade_date_str)
                if is_suspended_flag:
                    # 停牌，加入延迟队列
                    if self.pending_order_manager:
                        self.pending_order_manager.add_order(
                            stock=stock,
                            action="sell",
                            current_date=date,
                            signal_date=date,  # 卖出是基于持有期，用当前日期
                            target_value=None,
                            reason="停牌",
                        )
                    if self.verbose:
                        logger.info(f"  卖出延迟: {date.date()} {stock}, 原因: 停牌")
                    return
            except Exception as e:
                # 停牌数据加载失败，记录警告但继续检查（降级处理）
                logger.warning(f"停牌状态检查失败（{e}），继续检查其他交易状态")

            # 检查行情数据
            date_quote = self.price_data_cache[
                self.price_data_cache["trade_date"] == trade_date_str
            ]
            if date_quote.empty:
                # 当日行情数据为空，无法判断交易状态，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action="sell",
                        current_date=date,
                        signal_date=date,  # 卖出是基于持有期，用当前日期
                        target_value=None,
                        reason="无行情数据",
                    )
                if self.verbose:
                    logger.info(f"  卖出延迟: {date.date()} {stock}, 原因: 无行情数据")
                return

            # 检查跌停状态
            tradeable, reason = is_tradeable(stock, trade_date_str, date_quote, action="sell")

            if not tradeable:
                # 不可交易（跌停等），加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action="sell",
                        current_date=date,
                        signal_date=date,  # 卖出是基于持有期，用当前日期
                        target_value=None,
                        reason=reason,
                    )
                if self.verbose:
                    logger.info(f"  卖出延迟: {date.date()} {stock}, 原因: {reason}")
                return

        # 可交易，直接卖出
        self._sell_stock_direct(date, stock, sell_type, sell_reason, trigger_type)

    def _sell_stock_direct(
        self,
        date: pd.Timestamp,
        stock: str,
        sell_type: str = "holding_period",
        sell_reason: Optional[str] = None,
        trigger_type: Optional[str] = None,
    ) -> None:
        """直接卖出股票（不检查交易状态）

        现金流使用成交价格（trade_price）计算
        收益率使用绩效价格（pnl_price）计算

        根据 sell_timing 参数选择使用开盘价或收盘价：
        - sell_timing='close': 使用收盘价（默认）
        - sell_timing='open': 使用开盘价，如果开盘价不存在则降级到收盘价

        Args:
            date: 卖出日期（T+n）
            stock: 股票代码
            sell_type: 卖出类型，'holding_period' 或 'stop_loss'
            sell_reason: 卖出原因描述（止损时使用）
            trigger_type: 触发类型（止损时使用）
        """
        if stock not in self.positions or self.positions[stock]["shares"] == 0:
            return

        # 根据 sell_timing 参数选择价格
        if self.sell_timing == "open":
            # 尝试使用开盘价
            sell_trade_price = self._get_trade_price_open(date, stock)
            sell_pnl_price = self._get_pnl_price_open(date, stock)

            # 降级策略：如果开盘价不存在，使用收盘价
            if sell_trade_price is None:
                if self.verbose:
                    logger.warning(
                        f"股票 {stock} 在 {date.date()} 缺少开盘成交价格，" f"降级使用收盘价卖出"
                    )
                sell_trade_price = self._get_trade_price(date, stock)
                if sell_trade_price is None:
                    logger.warning(
                        f"无法获取 {stock} 在 {date.date()} 的成交价格（开盘/收盘），跳过卖出"
                    )
                    return

            if sell_pnl_price is None:
                # 开盘绩效价格不存在，尝试降级到收盘绩效价格
                sell_pnl_price = self._get_pnl_price(date, stock)
                if sell_pnl_price is None:
                    logger.warning(
                        f"无法获取 {stock} 在 {date.date()} 的绩效价格，使用成交价格代替"
                    )
                    sell_pnl_price = sell_trade_price
        else:
            # 使用收盘价（默认）
            sell_trade_price = self._get_trade_price(date, stock)
            if sell_trade_price is None:
                logger.warning(f"无法获取 {stock} 在 {date.date()} 的成交价格，跳过卖出")
                return

            sell_pnl_price = self._get_pnl_price(date, stock)
            if sell_pnl_price is None:
                logger.warning(f"无法获取 {stock} 在 {date.date()} 的绩效价格，使用成交价格代替")
                sell_pnl_price = sell_trade_price

        # 获取持仓信息
        shares = self.positions[stock]["shares"]
        buy_date = self.positions[stock]["buy_date"]
        buy_trade_price = self.positions[stock]["buy_trade_price"]
        buy_pnl_price = self.positions[stock]["buy_pnl_price"]
        buy_cost_cash = self.positions[stock]["buy_cost_cash"]

        # 计算现金流（基于成交价格）
        sell_amount = shares * sell_trade_price
        sell_cost = self.cost_model.calculate_sell_cost(sell_amount)
        sell_proceeds = sell_amount - sell_cost  # 卖出后实际到手金额

        # 计算收益率（基于绩效价格）
        pnl_buy_amount = shares * buy_pnl_price  # 绩效口径买入金额
        pnl_sell_amount = shares * sell_pnl_price  # 绩效口径卖出金额

        # 买入和卖出的手续费
        buy_amount = shares * buy_trade_price
        buy_cost = self.cost_model.calculate_buy_cost(buy_amount)
        total_cost = buy_cost + sell_cost  # 总手续费

        # 绩效收益（基于绩效价格，扣除手续费）
        # 收益 = 卖出金额 - 买入金额 - 总手续费
        pnl_profit_amount = pnl_sell_amount - pnl_buy_amount - total_cost
        # 收益率 = 收益 / (买入金额 + 买入手续费)
        # 买入成本是买入金额+买入手续费，这是投资者实际付出的成本
        pnl_profit_pct = (
            pnl_profit_amount / (pnl_buy_amount + buy_cost)
            if (pnl_buy_amount + buy_cost) > 0
            else 0
        )

        # 更新持仓和资金
        del self.positions[stock]
        self.current_capital += sell_proceeds

        # 清理亏损提前换出的缓刑计数
        self._early_exit_reprieve_counts.pop(stock, None)

        # 如果是止损卖出，清理止损监控器中的持仓状态
        if sell_type == "stop_loss" and self.stop_loss_monitor:
            self.stop_loss_monitor.remove_position(stock)

        # 清理弱势退出监控器中的持仓状态
        if self.weakness_exit_monitor is not None:
            self.weakness_exit_monitor.reset(stock)

        # 记录交易（包含绩效收益信息和卖出类型）
        trade_record = {
            "date": date,
            "stock": stock,
            "action": "sell",
            "price": sell_trade_price,  # 卖出成交价格
            "shares": shares,
            "amount": sell_amount,
            "cost": sell_cost,
            "buy_date": buy_date,
            "buy_price": buy_trade_price,  # 买入成交价格
            "buy_pnl_price": buy_pnl_price,  # 买入绩效价格
            "sell_pnl_price": sell_pnl_price,  # 卖出绩效价格
            "pnl_profit_amount": pnl_profit_amount,  # 绩效收益金额
            "pnl_profit_pct": pnl_profit_pct,  # 绩效收益率
            "sell_type": sell_type,  # 卖出类型
            "sell_timing": self.sell_timing,  # 新增：卖出时机（open/close）
        }

        # 如果是止损卖出，添加止损相关信息
        if sell_type == "stop_loss":
            trade_record["sell_reason"] = sell_reason
            trade_record["trigger_type"] = trigger_type

        self.trades.append(trade_record)

    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """计算组合市值（基于成交价格）

        Args:
            date: 计算日期

        Returns:
            组合总市值
        """
        market_value = 0.0

        for stock, info in self.positions.items():
            shares = info["shares"]
            trade_price = self._get_trade_price(date, stock)
            if trade_price is None:
                # 股票当日无价格（可能已退市/停牌），使用仓位中缓存的最后已知价格
                # 避免市值突降为 0 导致净值曲线出现虚假跳水
                trade_price = info.get("last_known_price")
                if trade_price is None:
                    # 兜底：使用买入价
                    trade_price = info.get("buy_trade_price", 0.0)
                    if trade_price > 0:
                        logger.warning(
                            f"股票 {stock} 在 {date.date()} 无价格数据，"
                            f"用买入价 {trade_price:.2f} 估值（可能已退市）"
                        )
            else:
                # 更新最后已知价格缓存
                info["last_known_price"] = trade_price

            market_value += shares * trade_price

        return self.current_capital + market_value

    def _generate_nav_curve(self) -> pd.DataFrame:
        """生成净值曲线

        Returns:
            净值曲线DataFrame
        """
        df = pd.DataFrame(self.portfolio_values)
        df["nav"] = df["portfolio_value"] / self.initial_capital
        df["return"] = df["nav"] - 1.0
        return df

    def get_trades(self) -> pd.DataFrame:
        """获取交易记录

        Returns:
            交易记录DataFrame
        """
        return pd.DataFrame(self.trades)

    def get_confidence_gate_stats(self) -> Dict[str, float]:
        """获取信号置信度门控统计。"""
        if not self.confidence_gate_history:
            return {
                "signal_days": 0,
                "blocked_days": 0,
                "block_rate": 0.0,
                "avg_exposure": 1.0,
                "avg_score": 0.0,
            }

        history_df = pd.DataFrame(self.confidence_gate_history)
        signal_days = int(len(history_df))
        blocked_days = int((history_df["exposure"] <= 0).sum())
        block_rate = blocked_days / signal_days if signal_days > 0 else 0.0
        avg_exposure = float(history_df["exposure"].mean()) if signal_days > 0 else 1.0
        avg_score = float(history_df["score"].mean()) if signal_days > 0 else 0.0

        return {
            "signal_days": signal_days,
            "blocked_days": blocked_days,
            "block_rate": block_rate,
            "avg_exposure": avg_exposure,
            "avg_score": avg_score,
        }

    # ── 提前调仓历史快照/回滚（避免污染门控和质量监控基准）──

    def _snapshot_early_rebalance_state(self, date: pd.Timestamp) -> Dict:
        """快照提前调仓可能污染的历史缓冲区状态（仅记录长度/标记），用于失败时回滚。

        返回的快照用于：提前调仓信号若未入 pending_signals（门控阻断或拖尾拒绝），
        需回滚 evaluate_confidence_gate 和 _record_signal_for_quality_tracking 追加的历史条目，
        确保"是否启用该开关"不会影响正常调仓日的门控/质量计算基准。
        """
        snapshot = {
            "confidence_gate_history_len": len(self.confidence_gate_history),
            "last_ranked_candidates": list(self._last_ranked_candidates),
            "last_signal_date": self._last_signal_date,
            # 完整深拷贝 _signal_tracking（_evaluate_expired_signal_quality 会删除过期键）
            "signal_tracking": {k: dict(v) for k, v in self._signal_tracking.items()},
            # _generate_signal 第 966 行会重置基准 NAV
            "last_rebalance_nav": getattr(self, "_last_rebalance_nav", None),
            # _evaluate_expired_signal_quality → _update_prediction_quality 会追加质量历史
            "prediction_quality_history": list(
                getattr(self, "_prediction_quality_history", [])
            ),
            "rolling_quality_score": getattr(self, "_rolling_quality_score", None),
            "quality_warmup_remaining": getattr(self, "_quality_warmup_remaining", None),
        }
        # 快照信号生成器侧的门控历史缓冲
        if hasattr(self.signal, "_separation_history"):
            snapshot["separation_history_len"] = len(self.signal._separation_history)
        if hasattr(self.signal, "_composite_score_history"):
            snapshot["composite_score_history_len"] = len(self.signal._composite_score_history)
        return snapshot

    def _restore_early_rebalance_state(self, date: pd.Timestamp, snapshot: Dict) -> None:
        """回滚提前调仓过程中追加的历史条目，使状态恢复到 _generate_signal 调用之前。"""
        # 回滚 confidence_gate_history
        target_len = snapshot["confidence_gate_history_len"]
        if len(self.confidence_gate_history) > target_len:
            del self.confidence_gate_history[target_len:]

        # 回滚信号生成器侧的门控历史缓冲
        if "separation_history_len" in snapshot and hasattr(
            self.signal, "_separation_history"
        ):
            target = snapshot["separation_history_len"]
            if len(self.signal._separation_history) > target:
                del self.signal._separation_history[target:]
        if "composite_score_history_len" in snapshot and hasattr(
            self.signal, "_composite_score_history"
        ):
            target = snapshot["composite_score_history_len"]
            if len(self.signal._composite_score_history) > target:
                del self.signal._composite_score_history[target:]

        # 完整还原 _signal_tracking（覆盖 _evaluate_expired_signal_quality 的删除和新增）
        self._signal_tracking = {k: dict(v) for k, v in snapshot["signal_tracking"].items()}

        # 还原 _last_rebalance_nav（_generate_signal 会无条件重置）
        if snapshot["last_rebalance_nav"] is not None:
            self._last_rebalance_nav = snapshot["last_rebalance_nav"]

        # 还原滚动质量监控状态（_update_prediction_quality 会修改）
        if snapshot["prediction_quality_history"] is not None and hasattr(
            self, "_prediction_quality_history"
        ):
            self._prediction_quality_history = list(snapshot["prediction_quality_history"])
        if snapshot["rolling_quality_score"] is not None and hasattr(
            self, "_rolling_quality_score"
        ):
            self._rolling_quality_score = snapshot["rolling_quality_score"]
        if snapshot["quality_warmup_remaining"] is not None and hasattr(
            self, "_quality_warmup_remaining"
        ):
            self._quality_warmup_remaining = snapshot["quality_warmup_remaining"]

        # 回滚止盈补位用的候选快照
        self._last_ranked_candidates = snapshot["last_ranked_candidates"]
        self._last_signal_date = snapshot["last_signal_date"]

    # ── 滚动模型质量监控 ──

    def _record_signal_for_quality_tracking(
        self, date: pd.Timestamp, selected_stocks: List[str], predicted_mean: float
    ) -> None:
        """记录本次调仓选中的股票，用于后续评估实际表现。"""
        if not self.signal_gate_quality_enabled:
            return
        date_str = date.strftime("%Y%m%d")
        self._signal_tracking[date_str] = {
            "stocks": list(selected_stocks),
            "predicted_mean": predicted_mean,
            "date": date,
        }

    def _update_prediction_quality(
        self,
        signal_date: pd.Timestamp,
        selected_stocks: List[str],
        price_data: pd.DataFrame,
        sell_date: pd.Timestamp,
    ) -> None:
        """一个调仓周期结束后，评估选股实际表现并更新滚动质量分数。

        Args:
            signal_date: 信号生成日期
            selected_stocks: 选中的股票列表
            price_data: 价格数据（需要包含 ts_code, close_adj 或 close）
            sell_date: 卖出日期
        """
        if not self.signal_gate_quality_enabled:
            return

        signal_date_str = signal_date.strftime("%Y%m%d")
        sell_date_str = sell_date.strftime("%Y%m%d")

        if price_data is None or price_data.empty or "trade_date" not in price_data.columns:
            return

        # 获取信号日后一个交易日（买入日）
        # price_data 的 index 是整数行号，通过 trade_date 列来查找日期位置
        unique_dates = sorted(price_data["trade_date"].unique())
        if not unique_dates:
            return
        signal_pos = bisect.bisect_left(unique_dates, signal_date_str)
        # 信号日T+1为买入日
        buy_pos = signal_pos + 1
        if buy_pos >= len(unique_dates):
            return
        buy_date_str = unique_dates[buy_pos]

        # 计算选中股票的收益率
        price_col = "close_adj" if "close_adj" in price_data.columns else "close"
        if "ts_code" not in price_data.columns:
            return

        buy_prices = price_data.loc[
            price_data["trade_date"] == buy_date_str, ["ts_code", price_col]
        ].set_index("ts_code")[price_col]
        sell_prices = price_data.loc[
            price_data["trade_date"] == sell_date_str, ["ts_code", price_col]
        ].set_index("ts_code")[price_col]

        if buy_prices.empty or sell_prices.empty:
            return

        # 选中股票收益率
        selected_returns = []
        for stock in selected_stocks:
            if stock in buy_prices.index and stock in sell_prices.index:
                bp = buy_prices[stock]
                sp = sell_prices[stock]
                if bp > 0:
                    selected_returns.append(sp / bp - 1.0)

        # 全市场收益率（计算中位数）
        common_stocks = buy_prices.index.intersection(sell_prices.index)
        all_returns = (sell_prices[common_stocks] / buy_prices[common_stocks] - 1.0).dropna()

        if len(selected_returns) == 0 or len(all_returns) == 0:
            return

        universe_median = float(all_returns.median())
        beat_count = sum(1 for r in selected_returns if r > universe_median)
        hit_rate = beat_count / len(selected_returns)

        self._prediction_quality_history.append(
            {
                "signal_date": signal_date_str,
                "sell_date": sell_date_str,
                "hit_rate": hit_rate,
                "selected_count": len(selected_returns),
                "selected_mean_return": float(np.mean(selected_returns)),
                "universe_median_return": universe_median,
                "beat_count": beat_count,
            }
        )

        # 更新滚动质量分数（EWM）
        if self._quality_warmup_remaining > 0:
            self._quality_warmup_remaining -= 1

        recent = self._prediction_quality_history[-self.signal_gate_quality_window :]
        if len(recent) > 0:
            hit_rates = [entry["hit_rate"] for entry in recent]
            hit_series = pd.Series(hit_rates)
            self._rolling_quality_score = float(
                hit_series.ewm(halflife=self.signal_gate_quality_halflife, min_periods=1)
                .mean()
                .iloc[-1]
            )

    def _evaluate_expired_signal_quality(
        self, current_date: pd.Timestamp, price_data: pd.DataFrame
    ) -> None:
        """在新调��时，评估已过期信号的实际选股表现。"""
        if not self._signal_tracking:
            return

        current_date_str = current_date.strftime("%Y%m%d")
        expired_keys = []

        for signal_date_str, tracking_info in self._signal_tracking.items():
            signal_date = tracking_info["date"]
            # 判断此信号是否已超过持有期
            # 保守处理：当前日期距离信���日已超过 holding_period，则认为已到期
            days_since = (current_date - signal_date).days
            if days_since < self.holding_period:
                continue

            # 评估此信号的实际表现
            self._update_prediction_quality(
                signal_date=signal_date,
                selected_stocks=tracking_info["stocks"],
                price_data=price_data,
                sell_date=current_date,
            )
            expired_keys.append(signal_date_str)

        # 清理已评估的信号
        for key in expired_keys:
            del self._signal_tracking[key]

    def _get_rolling_quality_exposure(self) -> float:
        """根据滚动模型质量计算仓位系数。

        Returns:
            仓位系数 (0.2~1.0)，预热期返回1.0
        """
        if not self.signal_gate_quality_enabled:
            return 1.0
        if self._quality_warmup_remaining > 0:
            return 1.0  # 预热期不干预
        if self._rolling_quality_score >= self.signal_gate_quality_threshold:
            return 1.0  # 模型表现正常
        # 模型表现低于阈值，按比例线性降仓
        return max(0.2, self._rolling_quality_score / self.signal_gate_quality_threshold)

    def get_prediction_quality_stats(self) -> Dict[str, float]:
        """获取滚动模型质量统计。"""
        if not self._prediction_quality_history:
            return {
                "quality_periods": 0,
                "avg_hit_rate": 0.0,
                "rolling_quality_score": self._rolling_quality_score,
                "quality_exposure": self._get_rolling_quality_exposure(),
                "warmup_remaining": self._quality_warmup_remaining,
            }
        hit_rates = [e["hit_rate"] for e in self._prediction_quality_history]
        return {
            "quality_periods": len(self._prediction_quality_history),
            "avg_hit_rate": float(np.mean(hit_rates)),
            "rolling_quality_score": self._rolling_quality_score,
            "quality_exposure": self._get_rolling_quality_exposure(),
            "warmup_remaining": self._quality_warmup_remaining,
        }
