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
    weight_method: str = "equal"
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
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_args(cls, args) -> "TradingConfig":
        """从 argparse Namespace 构建 TradingConfig。

        只取 TradingConfig 中定义的字段，忽略其余 CLI 参数。
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in vars(args).items() if k in valid_keys}
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
        "--weight-method",
        type=str,
        default="equal",
        choices=["equal", "score"],
        help="权重分配方法（默认：equal）",
    )
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
