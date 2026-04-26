"""统一策略参数配置模块

将 paper_trade.py / run_ml_backtest.py / bot_service.py 中重复定义的
策略参数（止损、ECT、组合约束、模型选择等）抽取为公共 dataclass + argparse 注册函数，
消除三套脚本之间的参数定义不一致。
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from ..risk.equity_curve import EquityCurveConfig
from ..risk.stop_loss import StopLossConfig


@dataclass
class TradingConfig:
    """统一策略参数"""

    # ── 模型 ──
    model_version: Optional[int] = None
    model_version_b: Optional[int] = None
    ensemble_weight_a: float = 0.5

    # ── 组合 ──
    top_n: int = 30
    signal_confidence_gate_enabled: bool = False
    signal_confidence_gate_top_k: int = 10
    signal_confidence_gate_thresholds: List[float] = field(default_factory=lambda: [0.8, 1.2, 1.6])
    signal_confidence_gate_exposure_levels: List[float] = field(
        default_factory=lambda: [0.3, 0.6, 1.0]
    )

    # ── 信号入口门控 v2（composite 模式替代旧置信度门控）──
    signal_gate_mode: str = "legacy"  # "legacy" 旧逻辑 | "composite" 新公式 | "disabled" 关闭
    signal_gate_cost_multiplier: float = 2.0  # 预测收益至少覆盖成本的倍数
    signal_gate_round_trip_cost: float = 0.003  # 往返交易成本估算（佣金+印花税+滑点）
    signal_gate_quality_enabled: bool = False  # 是否启用滚动模型质量监控
    signal_gate_quality_window: int = 5  # 回看调仓周期数
    signal_gate_quality_threshold: float = 0.4  # 最低滚动hit rate
    signal_gate_quality_halflife: int = 3  # EWM半衰期（调仓周期数）
    signal_gate_percentile_warmup: int = 20  # 百分位归一化预热期（调仓次数）
    signal_gate_dynamic_topn: bool = False  # 是否启用动态 Top-N（按置信度调整选股数量）
    signal_gate_topn_high_multiplier: float = 0.6  # 高置信度时缩减选股数量的系数（<1，集中持股）
    signal_gate_topn_low_multiplier: float = 1.5  # 低置信度时扩大选股数量的系数（>1，分散持股）

    # ── 换手率约束（持仓保留奖励）──
    holding_bonus_enabled: bool = False  # 是否启用持仓保留奖励（降低换手率）
    holding_bonus_sigma: float = 0.5  # 保留奖励幅度（截面分数标准差的倍数）

    # ── 盈利延续持有模式（强势度评分）──
    # pnl=原单一浮盈判据(向后兼容,默认) | strength=多维度强势度评分 | disabled=关闭延续机制
    profit_extension_mode: str = "pnl"
    profit_extension_strength_threshold: float = 0.6  # strength 模式延续阈值 [0, 1]
    profit_extension_strength_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "ml_score": 0.30,
            "momentum": 0.25,
            "technical": 0.15,
            "fund_flow": 0.15,
            "drawdown": 0.15,
        }
    )

    rebalance_freq: Optional[int] = 20
    stagger_tranches: int = 1
    max_per_industry: Optional[int] = None
    max_weight_per_stock: Optional[float] = None
    enable_early_rebalance_on_empty: bool = True  # 空仓时提前触发新一轮调仓

    # ── 股票池 ──
    exclude_st: bool = True
    min_list_days: int = 365

    # ── 止损 ──
    stop_loss_enabled: bool = False
    stop_loss_drawdown_pct: float = 30.0
    stop_loss_trailing_enabled: bool = False
    stop_loss_trailing_pct: float = 15.0
    stop_loss_consecutive_limit_down: int = 2

    # ── ECT ──
    equity_curve_enabled: bool = False
    equity_curve_drawdown_thresholds: List[float] = field(
        default_factory=lambda: [5.0, 10.0, 15.0, 20.0]
    )
    equity_curve_exposure_levels: List[float] = field(default_factory=lambda: [0.8, 0.6, 0.4, 0.2])
    equity_curve_ma_short: int = 5
    equity_curve_ma_long: int = 20
    equity_curve_recovery_mode: str = "gradual"
    equity_curve_recovery_step: float = 0.25
    equity_curve_recovery_delay_periods: int = 0

    # ── 市场择时仓位管理 ──
    market_regime_enabled: bool = False  # 是否启用市场择时
    market_regime_mode: str = "vol_target"  # binary|vol_target|trend|combined
    market_regime_bear_threshold: float = -0.03  # binary: mkt_ret_avg_20 低于此值判定熊市
    market_regime_bear_exposure: float = 0.3  # binary: 熊市仓位系数（0~1）
    market_regime_vol_target: float = 0.20  # vol_target/combined: 年化波动率目标
    market_regime_trend_threshold: float = 1.0  # trend/combined: mkt_ma_trend 降仓阈值
    market_regime_min_exposure: float = 0.2  # 非binary模式最低仓位下限
    market_regime_combine_method: str = "min"  # combined: min|multiply
    market_regime_trend_guard: bool = True  # combined: 上行趋势跳过vol降仓
    market_regime_drawdown_guard: bool = False  # 回撤保护：已大幅下跌时停止降仓
    market_regime_drawdown_threshold: float = -0.08  # 回撤保护阈值
    market_regime_ma250_hard_stop: bool = True  # MA250长周期硬条件
    market_regime_ma250_threshold: float = 1.0  # MA250触发阈值
    market_regime_ma250_exposure: float = 0.9  # MA250触发后仓位系数
    market_regime_ma250_atr_scaling: bool = True  # ATR动态仓位缩放

    # ── 行业动量过滤 & 行业轮动加权 ──
    industry_momentum_filter: bool = False  # 是否启用行业动量过滤（硬过滤弱势行业）
    industry_momentum_bottom_pct: float = 0.5  # 剔除排名后X%的行业
    industry_rotation_enhanced: bool = False  # 是否启用行业轮动加权（按行业动量排名调整分数）
    industry_rotation_alpha: float = 0.3  # 行业轮动加权强度(0=不调整,1=强调整)

    # ── 仓位管理模式 ──
    position_sizing: str = "equal"  # equal|score|kelly|half_kelly
    kelly_vol_window: int = 60  # Kelly 波动率估计窗口（交易日）
    kelly_max_leverage: float = 0.25  # 单只股票 Kelly 仓位上限（占总资产）

    # ── 盈亏动态持仓 ──
    enable_profit_based_holding: bool = True  # 是否启用盈亏动态持仓
    early_exit_loss_threshold: float = -0.07  # 亏损提前换出阈值
    early_exit_holding_ratio: float = 0.5  # 最早触发时点（占持有期比例）
    profit_extension_threshold: float = 0.1  # 盈利延续持有阈值（pnl模式）
    profit_extension_days: int = 2  # 盈利延续额外天数

    # ── 亏损提前换出二次确认（strength_veto 门控）──
    early_exit_mode: str = "disabled"  # disabled|strength_veto
    early_exit_strength_protect_threshold: float = 0.1  # strength_veto保护阈值
    early_exit_max_reprieves: int = 1  # 单只股票最多缓刑次数

    # ── ATR 动态止损 ──
    use_atr_for_early_exit: bool = False  # 用ATR替代固定阈值
    atr_multiplier: float = 2.0  # ATR 动态止损乘数

    # ── 整体止盈 ──
    take_profit_threshold: Optional[float] = None  # 整体浮盈阈值（None=禁用）
    take_profit_refill: bool = True  # 整体止盈后是否自动补位

    # ── 其他（仅 paper_trade 使用，backtest 不需要） ──
    buy_price: str = "close"
    sell_price: str = "open"
    initial_capital: float = 500000.0
    horizon: int = 5
    universe: str = "mainboard"

    # ─────────────────── 工厂方法 ───────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "TradingConfig":
        """从字典（如 PaperStorage.load_config()）构建 TradingConfig。

        忽略字典中不属于 TradingConfig 字段的键，
        对缺失的键使用 dataclass 默认值。
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        normalized = dict(d)
        if "position_sizing" not in normalized and "weight_method" in normalized:
            normalized["position_sizing"] = normalized["weight_method"]
        # 向后兼容旧字段名
        if "atr_multiplier" not in normalized and "early_exit_atr_multiplier" in normalized:
            normalized["atr_multiplier"] = normalized["early_exit_atr_multiplier"]
        filtered = {k: v for k, v in normalized.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_args(cls, args) -> "TradingConfig":
        """从 argparse Namespace 构建 TradingConfig。

        只取 TradingConfig 中定义的字段，忽略其余 CLI 参数。
        额外处理：将 profit_extension_strength_w_* 5 个独立 CLI 权重参数
        合并为 profit_extension_strength_weights 字典字段。
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        args_dict = vars(args)
        if "atr_multiplier" not in args_dict and "early_exit_atr_multiplier" in args_dict:
            args_dict["atr_multiplier"] = args_dict["early_exit_atr_multiplier"]

        d = {k: v for k, v in args_dict.items() if k in valid_keys}

        # 合并盈利延续持有强势度权重（5 个 CLI 参数 → dict 字段）
        weight_keys = (
            "profit_extension_strength_w_ml",
            "profit_extension_strength_w_momentum",
            "profit_extension_strength_w_technical",
            "profit_extension_strength_w_fund",
            "profit_extension_strength_w_drawdown",
        )
        if any(k in args_dict for k in weight_keys):
            d["profit_extension_strength_weights"] = {
                "ml_score": float(args_dict.get("profit_extension_strength_w_ml", 0.30)),
                "momentum": float(args_dict.get("profit_extension_strength_w_momentum", 0.25)),
                "technical": float(args_dict.get("profit_extension_strength_w_technical", 0.15)),
                "fund_flow": float(args_dict.get("profit_extension_strength_w_fund", 0.15)),
                "drawdown": float(args_dict.get("profit_extension_strength_w_drawdown", 0.15)),
            }
        return cls(**d)

    def to_dict(self) -> dict:
        """转为普通字典（用于持久化）"""
        return asdict(self)

    # ─────────────────── 配置对象构建 ───────────────────

    def create_stop_loss_config(self) -> Optional[StopLossConfig]:
        """构建 StopLossConfig（不启用时返回 None）"""
        if not self.stop_loss_enabled:
            return None
        return StopLossConfig(
            enabled=True,
            drawdown_pct=self.stop_loss_drawdown_pct,
            trailing_stop_enabled=self.stop_loss_trailing_enabled,
            trailing_stop_pct=self.stop_loss_trailing_pct,
            consecutive_limit_down_days=self.stop_loss_consecutive_limit_down,
        )

    def create_equity_curve_config(self) -> Optional[EquityCurveConfig]:
        """构建 EquityCurveConfig（不启用时返回 None）"""
        if not self.equity_curve_enabled:
            return None
        return EquityCurveConfig(
            enabled=True,
            drawdown_thresholds=self.equity_curve_drawdown_thresholds,
            exposure_levels=self.equity_curve_exposure_levels,
            ma_short_window=self.equity_curve_ma_short,
            ma_long_window=self.equity_curve_ma_long,
            recovery_mode=self.equity_curve_recovery_mode,
            recovery_step=self.equity_curve_recovery_step,
            recovery_delay_periods=self.equity_curve_recovery_delay_periods,
        )


# ─────────────────── argparse 注册函数 ───────────────────


def add_trading_args(parser, *, include_price: bool = False) -> None:
    """向 argparse parser 注册公共策略参数。

    Args:
        parser: argparse.ArgumentParser 或子 parser
        include_price: 是否注册 buy_price / sell_price / initial_capital / horizon / universe
                       （paper_trade 的 config 子命令需要，backtest 一般不需要）
    """
    # ── 模型 ──
    parser.add_argument(
        "--model-version", type=int, default=None, help="ML模型版本号（可选，默认最新版本）"
    )
    parser.add_argument(
        "--model-version-b",
        type=int,
        default=None,
        help="第二个模型版本号（用于集成），指定后自动启用双模型 Ensemble",
    )
    parser.add_argument(
        "--ensemble-weight-a",
        type=float,
        default=0.5,
        help="集成时模型A的排名权重，模型B权重为 1 - 该值，默认 0.5",
    )

    # ── 组合 ──
    parser.add_argument("--top-n", type=int, default=30, help="持仓股票数（默认：30）")
    parser.add_argument(
        "--signal-confidence-gate-enabled",
        action="store_true",
        default=False,
        help="启用信号置信度门控：低置信度时降仓或持币",
    )
    parser.add_argument(
        "--signal-confidence-gate-top-k",
        type=int,
        default=10,
        help="置信度评估使用的头部候选数量（默认：10）",
    )
    parser.add_argument(
        "--signal-confidence-gate-thresholds",
        type=float,
        nargs="+",
        default=[0.8, 1.2, 1.6],
        help="信号置信度阈值列表；低于首档时持币，默认：0.8 1.2 1.6",
    )
    parser.add_argument(
        "--signal-confidence-gate-exposure-levels",
        type=float,
        nargs="+",
        default=[0.3, 0.6, 1.0],
        help="各置信度阈值对应的仓位系数，默认：0.3 0.6 1.0",
    )

    # ── 信号入口门控 v2 ──
    parser.add_argument(
        "--signal-gate-mode",
        type=str,
        default="legacy",
        choices=["legacy", "composite", "disabled"],
        help="信号入口门控模式：legacy=旧公式, composite=新公式(成本+百分位), disabled=关闭",
    )
    parser.add_argument(
        "--signal-gate-cost-multiplier",
        type=float,
        default=2.0,
        help="composite模式：预测收益至少覆盖交易成本的倍数（默认：2.0）",
    )
    parser.add_argument(
        "--signal-gate-round-trip-cost",
        type=float,
        default=0.003,
        help="composite模式：往返交易成本估算（默认：0.003=0.3%%）",
    )
    parser.add_argument(
        "--signal-gate-quality-enabled",
        action="store_true",
        default=False,
        help="启用滚动模型质量监控：跟踪最近N期模型预测实际表现",
    )
    parser.add_argument(
        "--signal-gate-quality-window",
        type=int,
        default=5,
        help="滚动质量监控回看的调仓周期数（默认：5）",
    )
    parser.add_argument(
        "--signal-gate-quality-threshold",
        type=float,
        default=0.4,
        help="滚动质量监控最低hit rate（默认：0.4）",
    )
    parser.add_argument(
        "--signal-gate-quality-halflife",
        type=int,
        default=3,
        help="滚动质量EWM半衰期（调仓周期数，默认：3）",
    )
    parser.add_argument(
        "--signal-gate-percentile-warmup",
        type=int,
        default=20,
        help="百分位归一化预热期（调仓次数，默认：20）",
    )
    parser.add_argument(
        "--signal-gate-dynamic-topn",
        action="store_true",
        default=False,
        help="启用动态Top-N：高置信度时集中选股，低置信度时分散选股",
    )
    parser.add_argument(
        "--signal-gate-topn-high-multiplier",
        type=float,
        default=0.6,
        help="动态Top-N高置信度缩减系数（默认：0.6，即top_n×0.6）",
    )
    parser.add_argument(
        "--signal-gate-topn-low-multiplier",
        type=float,
        default=1.5,
        help="动态Top-N低置信度扩大系数（默认：1.5，即top_n×1.5）",
    )

    # ── 换手率约束（持仓保留奖励）──
    parser.add_argument(
        "--holding-bonus-enabled",
        action="store_true",
        default=False,
        help="启用持仓保留奖励：对已持仓股票给予分数加成，降低不必要换手",
    )
    parser.add_argument(
        "--holding-bonus-sigma",
        type=float,
        default=0.5,
        help="持仓保留奖励幅度，截面分数标准差的倍数（默认：0.5）",
    )

    # ── 盈利延续持有模式（强势度评分）──
    parser.add_argument(
        "--profit-extension-mode",
        type=str,
        default="pnl",
        choices=["pnl", "strength", "disabled"],
        help=(
            "盈利延续持有判据模式："
            "pnl=原单一浮盈判据(默认,兼容)；"
            "strength=多维度强势度评分(ML+动量+技术+资金+回撤)；"
            "disabled=关闭延续机制"
        ),
    )
    parser.add_argument(
        "--profit-extension-strength-threshold",
        type=float,
        default=0.6,
        help="strength 模式的延续阈值 [0,1]，高于此值才延续持有（默认：0.6）",
    )
    parser.add_argument(
        "--profit-extension-strength-w-ml",
        type=float,
        default=0.30,
        help="strength 模式 ML 分数维度权重（默认：0.30）",
    )
    parser.add_argument(
        "--profit-extension-strength-w-momentum",
        type=float,
        default=0.25,
        help="strength 模式 动量维度权重（默认：0.25）",
    )
    parser.add_argument(
        "--profit-extension-strength-w-technical",
        type=float,
        default=0.15,
        help="strength 模式 技术维度权重（默认：0.15）",
    )
    parser.add_argument(
        "--profit-extension-strength-w-fund",
        type=float,
        default=0.15,
        help="strength 模式 资金筹码维度权重（默认：0.15）",
    )
    parser.add_argument(
        "--profit-extension-strength-w-drawdown",
        type=float,
        default=0.15,
        help="strength 模式 回撤距离维度权重（默认：0.15）",
    )

    # ── 市场自适应 Top-N ──
    parser.add_argument(
        "--market-adaptive-topn-enabled",
        action="store_true",
        default=False,
        help="启用市场状态自适应选股数量：趋势市集中、震荡市分散",
    )
    parser.add_argument(
        "--market-adaptive-topn-bull-factor",
        type=float,
        default=0.7,
        help="趋势向上时集中系数（默认：0.7，即top_n×0.7）",
    )
    parser.add_argument(
        "--market-adaptive-topn-bear-factor",
        type=float,
        default=1.5,
        help="趋势向下/震荡时分散系数（默认：1.5，即top_n×1.5）",
    )

    parser.add_argument(
        "--rebalance-freq", type=int, default=20, help="调仓频率（交易日数），默认20"
    )
    parser.add_argument(
        "--stagger-tranches",
        type=int,
        default=1,
        help="分批调仓批次数（默认1=不分批）。设为K时资金分K份错开调仓，降低时点风险",
    )
    parser.add_argument(
        "--max-per-industry", type=int, default=None, help="单行业最大持仓数量（默认：不限制）"
    )
    parser.add_argument(
        "--max-weight-per-stock",
        type=float,
        default=None,
        help="单股最大权重，如 0.05 表示 5%%（默认：不限制）",
    )

    # ── 股票池 ──
    parser.add_argument(
        "--exclude-st", action="store_true", default=True, help="排除ST股票（默认：启用）"
    )
    parser.add_argument(
        "--no-exclude-st", action="store_false", dest="exclude_st", help="不排除ST股票"
    )
    parser.add_argument("--min-list-days", type=int, default=365, help="最少上市天数（默认：365）")

    # ── 止损 ──
    parser.add_argument(
        "--stop-loss-enabled", action="store_true", default=False, help="启用止损功能"
    )
    parser.add_argument(
        "--stop-loss-drawdown-pct", type=float, default=30.0, help="回撤止损百分比（默认：30.0）"
    )
    parser.add_argument(
        "--stop-loss-trailing-enabled", action="store_true", default=False, help="启用移动止损"
    )
    parser.add_argument(
        "--stop-loss-trailing-pct", type=float, default=15.0, help="移动止损百分比（默认：15.0）"
    )
    parser.add_argument(
        "--stop-loss-consecutive-limit-down",
        type=int,
        default=2,
        help="连续跌停触发天数（默认：2）",
    )

    # ── ECT ──
    parser.add_argument(
        "--equity-curve-enabled",
        action="store_true",
        default=False,
        help="启用权益曲线交易（ECT）功能",
    )
    parser.add_argument(
        "--equity-curve-drawdown-thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 15.0, 20.0],
        help="ECT 回撤阈值列表（百分比），默认：5.0 10.0 15.0 20.0",
    )
    parser.add_argument(
        "--equity-curve-exposure-levels",
        type=float,
        nargs="+",
        default=[0.8, 0.6, 0.4, 0.2],
        help="ECT 对应仓位系数列表，默认：0.8 0.6 0.4 0.2",
    )
    parser.add_argument(
        "--equity-curve-ma-short", type=int, default=5, help="ECT 短期均线窗口（默认：5）"
    )
    parser.add_argument(
        "--equity-curve-ma-long", type=int, default=20, help="ECT 长期均线窗口（默认：20）"
    )
    parser.add_argument(
        "--equity-curve-recovery-mode",
        type=str,
        default="gradual",
        choices=["gradual", "immediate"],
        help="ECT 恢复模式（默认：gradual）",
    )
    parser.add_argument(
        "--equity-curve-recovery-step",
        type=float,
        default=0.25,
        help="ECT 逐步恢复步长（默认：0.25）",
    )
    parser.add_argument(
        "--equity-curve-recovery-delay-periods",
        type=int,
        default=0,
        help="ECT 恢复等待周期数（默认：0）",
    )

    # ── 市场择时仓位管理 ──
    parser.add_argument(
        "--market-regime-enabled",
        action="store_true",
        default=False,
        help="启用市场择时仓位管理",
    )
    parser.add_argument(
        "--market-regime-mode",
        type=str,
        default="vol_target",
        choices=["binary", "vol_target", "trend", "combined"],
        help="市场择时模式（默认：vol_target）",
    )
    parser.add_argument(
        "--market-regime-bear-threshold",
        type=float,
        default=-0.03,
        help="binary模式：mkt_ret_avg_20低于此值判定熊市（默认：-0.03）",
    )
    parser.add_argument(
        "--market-regime-bear-exposure",
        type=float,
        default=0.3,
        help="binary模式：熊市仓位系数（默认：0.3）",
    )
    parser.add_argument(
        "--market-regime-vol-target",
        type=float,
        default=0.20,
        help="vol_target/combined模式：年化波动率目标（默认：0.20）",
    )
    parser.add_argument(
        "--market-regime-trend-threshold",
        type=float,
        default=1.0,
        help="trend/combined模式：mkt_ma_trend降仓阈值（默认：1.0）",
    )
    parser.add_argument(
        "--market-regime-min-exposure",
        type=float,
        default=0.2,
        help="非binary模式最低仓位下限（默认：0.2）",
    )
    parser.add_argument(
        "--market-regime-combine-method",
        type=str,
        default="min",
        choices=["min", "multiply"],
        help="combined模式组合方式（默认：min）",
    )
    parser.add_argument(
        "--market-regime-trend-guard",
        action="store_true",
        default=True,
        dest="market_regime_trend_guard",
        help="combined模式：上行趋势跳过vol降仓（默认：启用）",
    )
    parser.add_argument(
        "--no-market-regime-trend-guard",
        action="store_false",
        dest="market_regime_trend_guard",
        help="combined模式：关闭趋势保护",
    )
    parser.add_argument(
        "--market-regime-drawdown-guard",
        action="store_true",
        default=False,
        dest="market_regime_drawdown_guard",
        help="回撤保护：已大幅下跌时停止降仓",
    )
    parser.add_argument(
        "--no-market-regime-drawdown-guard",
        action="store_false",
        dest="market_regime_drawdown_guard",
        help="关闭回撤保护",
    )
    parser.add_argument(
        "--market-regime-drawdown-threshold",
        type=float,
        default=-0.08,
        help="回撤保护阈值（默认：-0.08）",
    )
    parser.add_argument(
        "--market-regime-ma250-hard-stop",
        action="store_true",
        default=True,
        dest="market_regime_ma250_hard_stop",
        help="启用MA250长周期硬条件（默认：启用）",
    )
    parser.add_argument(
        "--no-market-regime-ma250-hard-stop",
        action="store_false",
        dest="market_regime_ma250_hard_stop",
        help="关闭MA250硬条件",
    )
    parser.add_argument(
        "--market-regime-ma250-threshold",
        type=float,
        default=1.0,
        help="MA250触发阈值（默认：1.0）",
    )
    parser.add_argument(
        "--market-regime-ma250-exposure",
        type=float,
        default=0.9,
        help="MA250触发后仓位系数（默认：0.9）",
    )
    parser.add_argument(
        "--market-regime-ma250-atr-scaling",
        action="store_true",
        default=True,
        dest="market_regime_ma250_atr_scaling",
        help="MA250启用ATR动态仓位缩放（默认：启用）",
    )
    parser.add_argument(
        "--no-market-regime-ma250-atr-scaling",
        action="store_false",
        dest="market_regime_ma250_atr_scaling",
        help="关闭MA250 ATR缩放",
    )

    # ── 行业动量过滤 & 行业轮动加权 ──
    parser.add_argument(
        "--industry-momentum-filter",
        action="store_true",
        default=False,
        help="启用行业动量过滤：剔除弱势行业股票",
    )
    parser.add_argument(
        "--industry-momentum-bottom-pct",
        type=float,
        default=0.5,
        help="剔除行业动量排名后X%%的行业（默认：0.5）",
    )
    parser.add_argument(
        "--industry-rotation-enhanced",
        action="store_true",
        default=False,
        help="启用行业轮动加权：按行业动量排名对候选分数做乘性调整（强势加分、弱势扣分）",
    )
    parser.add_argument(
        "--industry-rotation-alpha",
        type=float,
        default=0.3,
        help="行业轮动加权强度（0=不调整, 1=强调整），默认 0.3",
    )

    # ── 仓位管理模式 ──
    parser.add_argument(
        "--position-sizing",
        type=str,
        default="equal",
        choices=["equal", "score", "kelly", "half_kelly"],
        help="仓位管理模式: equal=等权, score=按分数, kelly=Kelly最优, half_kelly=半Kelly(更保守)",
    )
    parser.add_argument(
        "--kelly-vol-window",
        type=int,
        default=60,
        help="Kelly 波动率估计窗口（交易日），默认 60",
    )
    parser.add_argument(
        "--kelly-max-leverage",
        type=float,
        default=0.25,
        help="Kelly 单只股票仓位上限（占总资产），默认 0.25",
    )

    # ── 盈亏动态持仓 ──
    parser.add_argument(
        "--enable-profit-based-holding",
        action="store_true",
        default=True,
        dest="enable_profit_based_holding",
        help="启用盈亏动态持仓（默认：启用）",
    )
    parser.add_argument(
        "--no-profit-based-holding",
        action="store_false",
        dest="enable_profit_based_holding",
        help="关闭盈亏动态持仓",
    )
    parser.add_argument(
        "--early-exit-loss-threshold",
        type=float,
        default=-0.07,
        help="亏损提前换出阈值（默认：-0.07）",
    )
    parser.add_argument(
        "--early-exit-holding-ratio",
        type=float,
        default=0.5,
        help="亏损提前换出最早触发时点，占持有期比例（默认：0.5）",
    )
    parser.add_argument(
        "--profit-extension-threshold",
        type=float,
        default=0.1,
        help="盈利延续持有阈值，pnl模式（默认：0.1）",
    )
    parser.add_argument(
        "--profit-extension-days",
        type=int,
        default=2,
        help="盈利延续额外交易日数（默认：2）",
    )

    # ── 亏损提前换出二次确认（strength_veto）──
    parser.add_argument(
        "--early-exit-mode",
        type=str,
        default="disabled",
        choices=["disabled", "strength_veto"],
        help="亏损提前换出模式：disabled=直接卖出, strength_veto=二次确认（默认：disabled）",
    )
    parser.add_argument(
        "--early-exit-strength-protect-threshold",
        type=float,
        default=0.1,
        help="strength_veto保护阈值 [0,1]（默认：0.1）",
    )
    parser.add_argument(
        "--early-exit-max-reprieves",
        type=int,
        default=1,
        help="单只股票最多缓刑次数（默认：1）",
    )

    # ── ATR 动态止损 ──
    parser.add_argument(
        "--use-atr-for-early-exit",
        action="store_true",
        default=False,
        help="用ATR动态阈值替代固定early_exit_loss_threshold",
    )
    parser.add_argument(
        "--atr-multiplier",
        "--early-exit-atr-multiplier",
        dest="atr_multiplier",
        type=float,
        default=2.0,
        help="ATR动态止损乘数（默认2.0）",
    )

    # ── 整体止盈 ──
    parser.add_argument(
        "--take-profit-threshold",
        type=float,
        default=None,
        help="整体持仓止盈阈值（如 0.15 表示整体浮盈15%%时清仓，默认禁用）",
    )
    parser.add_argument(
        "--no-take-profit-refill",
        dest="take_profit_refill",
        action="store_false",
        default=True,
        help="整体止盈后不触发补位买入（默认开启补位）",
    )
    parser.add_argument(
        "--no-early-rebalance-on-empty",
        dest="enable_early_rebalance_on_empty",
        action="store_false",
        default=True,
        help="禁用空仓/持有期拖尾时的提前调仓（默认启用）",
    )

    # ── paper_trade 专用 ──
    if include_price:
        parser.add_argument(
            "--buy-price",
            choices=["open", "close"],
            default="close",
            help="买入价格类型（默认：close）",
        )
        parser.add_argument(
            "--sell-price",
            choices=["open", "close"],
            default="open",
            help="卖出价格类型（默认：open）",
        )
        parser.add_argument(
            "--initial-capital", type=float, default=500000.0, help="初始资金（默认：500000）"
        )
        parser.add_argument(
            "--horizon", type=int, default=5, help="特征构建的预测周期（天数），默认5"
        )
        parser.add_argument(
            "--universe",
            choices=["mainboard", "all"],
            default="mainboard",
            help="股票池类型（默认：mainboard）",
        )
