"""统一策略参数配置模块

将 paper_trade.py / run_ml_backtest.py / bot_service.py 中重复定义的
策略参数（止损、ECT、组合约束、模型选择等）抽取为公共 dataclass + argparse 注册函数，
消除三套脚本之间的参数定义不一致。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from ..risk.stop_loss import StopLossConfig
from ..risk.equity_curve import EquityCurveConfig


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
    rebalance_freq: Optional[int] = 20
    max_per_industry: Optional[int] = None
    max_weight_per_stock: Optional[float] = None

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
    equity_curve_exposure_levels: List[float] = field(
        default_factory=lambda: [0.8, 0.6, 0.4, 0.2]
    )
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
    parser.add_argument("--model-version", type=int, default=None,
                        help="ML模型版本号（可选，默认最新版本）")
    parser.add_argument("--model-version-b", type=int, default=None,
                        help="第二个模型版本号（用于集成），指定后自动启用双模型 Ensemble")
    parser.add_argument("--ensemble-weight-a", type=float, default=0.5,
                        help="集成时模型A的排名权重，模型B权重为 1 - 该值，默认 0.5")

    # ── 组合 ──
    parser.add_argument("--top-n", type=int, default=30,
                        help="持仓股票数（默认：30）")
    parser.add_argument("--weight-method", type=str, default="equal",
                        choices=["equal", "score"],
                        help="权重分配方法（默认：equal）")
    parser.add_argument("--rebalance-freq", type=int, default=None,
                        help="调仓频率（交易日数）")
    parser.add_argument("--max-per-industry", type=int, default=None,
                        help="单行业最大持仓数量（默认：不限制）")
    parser.add_argument("--max-weight-per-stock", type=float, default=None,
                        help="单股最大权重，如 0.05 表示 5%%（默认：不限制）")

    # ── 股票池 ──
    parser.add_argument("--exclude-st", action="store_true", default=True,
                        help="排除ST股票（默认：启用）")
    parser.add_argument("--no-exclude-st", action="store_false", dest="exclude_st",
                        help="不排除ST股票")
    parser.add_argument("--min-list-days", type=int, default=365,
                        help="最少上市天数（默认：365）")

    # ── 止损 ──
    parser.add_argument("--stop-loss-enabled", action="store_true", default=False,
                        help="启用止损功能")
    parser.add_argument("--stop-loss-drawdown-pct", type=float, default=30.0,
                        help="回撤止损百分比（默认：30.0）")
    parser.add_argument("--stop-loss-trailing-enabled", action="store_true", default=False,
                        help="启用移动止损")
    parser.add_argument("--stop-loss-trailing-pct", type=float, default=15.0,
                        help="移动止损百分比（默认：15.0）")
    parser.add_argument("--stop-loss-consecutive-limit-down", type=int, default=2,
                        help="连续跌停触发天数（默认：2）")

    # ── ECT ──
    parser.add_argument("--equity-curve-enabled", action="store_true", default=False,
                        help="启用权益曲线交易（ECT）功能")
    parser.add_argument("--equity-curve-drawdown-thresholds", type=float, nargs="+",
                        default=[5.0, 10.0, 15.0, 20.0],
                        help="ECT 回撤阈值列表（百分比），默认：5.0 10.0 15.0 20.0")
    parser.add_argument("--equity-curve-exposure-levels", type=float, nargs="+",
                        default=[0.8, 0.6, 0.4, 0.2],
                        help="ECT 对应仓位系数列表，默认：0.8 0.6 0.4 0.2")
    parser.add_argument("--equity-curve-ma-short", type=int, default=5,
                        help="ECT 短期均线窗口（默认：5）")
    parser.add_argument("--equity-curve-ma-long", type=int, default=20,
                        help="ECT 长期均线窗口（默认：20）")
    parser.add_argument("--equity-curve-recovery-mode", type=str, default="gradual",
                        choices=["gradual", "immediate"],
                        help="ECT 恢复模式（默认：gradual）")
    parser.add_argument("--equity-curve-recovery-step", type=float, default=0.25,
                        help="ECT 逐步恢复步长（默认：0.25）")
    parser.add_argument("--equity-curve-recovery-delay-periods", type=int, default=0,
                        help="ECT 恢复等待周期数（默认：0）")

    # ── paper_trade 专用 ──
    if include_price:
        parser.add_argument("--buy-price", choices=["open", "close"], default="close",
                            help="买入价格类型（默认：close）")
        parser.add_argument("--sell-price", choices=["open", "close"], default="open",
                            help="卖出价格类型（默认：open）")
        parser.add_argument("--initial-capital", type=float, default=500000.0,
                            help="初始资金（默认：500000）")
        parser.add_argument("--horizon", type=int, default=20,
                            help="特征构建的预测周期（天数），默认20")
        parser.add_argument("--universe", choices=["mainboard", "all"], default="mainboard",
                            help="股票池类型（默认：mainboard）")
