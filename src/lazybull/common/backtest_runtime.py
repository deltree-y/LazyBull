"""回测运行时共享工具。

将 walk_forward / run_ml_backtest 的策略配置映射、信号创建、
BacktestEngineML 构造和滚动质量状态恢复统一到一个模块，
避免多个脚本各自维护一套参数透传逻辑。
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from ..backtest import BacktestEngineML
from ..common.cost import CostModel
from ..signals.ml_signal import MLSignal
from ..universe import BasicUniverse
from .config import get_models_root
from .signal_factory import create_signal
from .trading_config import TradingConfig


def infer_rebalance_freq_from_label(label_column: Optional[str], default: int = 20) -> int:
    """根据标签列名推断调仓频率。"""
    if not label_column:
        return default
    match = re.search(r"(\d+)", label_column)
    if match:
        return int(match.group(1))
    return default


def build_walk_forward_trading_config(args, *, model_version: int) -> TradingConfig:
    """将 walk_forward 参数映射为统一 TradingConfig。"""
    rebalance_freq = getattr(args, "bt_rebalance_freq", None)
    if rebalance_freq is None:
        rebalance_freq = infer_rebalance_freq_from_label(getattr(args, "label_column", None))

    return TradingConfig(
        model_version=model_version,
        top_n=getattr(args, "bt_top_n", 30),
        rebalance_freq=rebalance_freq,
        stagger_tranches=getattr(args, "stagger_tranches", 1),
        max_per_industry=getattr(args, "bt_max_per_industry", None),
        max_weight_per_stock=getattr(args, "bt_max_weight_per_stock", None),
        enable_early_rebalance_on_empty=getattr(args, "enable_early_rebalance_on_empty", True),
        exclude_st=getattr(args, "bt_exclude_st", True),
        min_list_days=getattr(args, "bt_min_list_days", 365),
        stop_loss_enabled=getattr(args, "bt_stop_loss_enabled", False),
        stop_loss_drawdown_pct=getattr(args, "bt_stop_loss_drawdown_pct", 30.0),
        stop_loss_trailing_enabled=getattr(args, "bt_stop_loss_trailing_enabled", False),
        stop_loss_trailing_pct=getattr(args, "bt_stop_loss_trailing_pct", 15.0),
        stop_loss_consecutive_limit_down=getattr(
            args, "bt_stop_loss_consecutive_limit_down", 2
        ),
        market_regime_enabled=getattr(args, "market_regime", False),
        market_regime_mode=getattr(args, "market_regime_mode", "binary"),
        market_regime_bear_threshold=getattr(args, "market_regime_bear_threshold", -0.02),
        market_regime_bear_exposure=getattr(args, "market_regime_bear_exposure", 0.3),
        market_regime_vol_target=getattr(args, "market_regime_vol_target", 0.15),
        market_regime_trend_threshold=getattr(args, "market_regime_trend_threshold", 1.0),
        market_regime_min_exposure=getattr(args, "market_regime_min_exposure", 0.2),
        market_regime_combine_method=getattr(args, "market_regime_combine_method", "min"),
        market_regime_trend_guard=getattr(args, "market_regime_trend_guard", True),
        market_regime_drawdown_guard=getattr(args, "market_regime_drawdown_guard", True),
        market_regime_drawdown_threshold=getattr(args, "market_regime_drawdown_threshold", -0.08),
        industry_momentum_filter=getattr(args, "industry_momentum_filter", False),
        industry_momentum_bottom_pct=getattr(args, "industry_momentum_bottom_pct", 0.2),
        industry_rotation_enhanced=getattr(args, "industry_rotation_enhanced", False),
        industry_rotation_alpha=getattr(args, "industry_rotation_alpha", 0.3),
        position_sizing=getattr(args, "position_sizing", "equal"),
        kelly_vol_window=getattr(args, "kelly_vol_window", 60),
        kelly_max_leverage=getattr(args, "kelly_max_leverage", 0.25),
        min_buy_value_ratio=getattr(args, "min_buy_value_ratio", 0.2),
        initial_capital=getattr(args, "bt_initial_capital", 1000000.0),
        sell_price=getattr(args, "bt_sell_timing", "open"),
    )


def create_or_reuse_signal(
    trading_config: TradingConfig,
    *,
    data_root: Optional[str] = None,
    persistent_signal: Optional[MLSignal] = None,
    verbose: bool = False,
):
    """创建或复用共享的 MLSignal。"""
    models_dir = get_models_root(str(Path(data_root) / "models") if data_root else None)

    if persistent_signal is not None:
        persistent_signal.top_n = trading_config.top_n
        if (
            trading_config.model_version_b is not None
            and hasattr(persistent_signal, "update_versions")
        ):
            persistent_signal.update_versions(
                trading_config.model_version,
                trading_config.model_version_b,
            )
        else:
            persistent_signal.update_model_version(trading_config.model_version)
        return persistent_signal

    return create_signal(trading_config, models_dir=models_dir, verbose=verbose)


def create_backtest_engine_from_config(
    *,
    trading_config: TradingConfig,
    universe: BasicUniverse,
    signal,
    features_by_date: dict,
    stock_basic: pd.DataFrame,
    data_storage,
    initial_capital: Optional[float] = None,
    sell_timing: Optional[str] = None,
    verbose: bool = False,
    completion_window_days: int = 5,
    enable_pending_order: bool = True,
    cost_model: Optional[CostModel] = None,
) -> BacktestEngineML:
    """根据统一 TradingConfig 构造 BacktestEngineML。"""
    return BacktestEngineML(
        universe=universe,
        signal=signal,
        features_by_date=features_by_date,
        initial_capital=(
            trading_config.initial_capital if initial_capital is None else initial_capital
        ),
        cost_model=cost_model or CostModel(),
        rebalance_freq=trading_config.rebalance_freq,
        stagger_tranches=trading_config.stagger_tranches,
        stop_loss_config=trading_config.create_stop_loss_config(),
        sell_timing=sell_timing or trading_config.sell_price,
        enable_pending_order=enable_pending_order,
        completion_window_days=completion_window_days,
        verbose=verbose,
        data_storage=data_storage,
        max_weight_per_stock=trading_config.max_weight_per_stock,
        max_per_industry=trading_config.max_per_industry,
        stock_basic=stock_basic,
        market_regime_enabled=trading_config.market_regime_enabled,
        market_regime_mode=trading_config.market_regime_mode,
        market_regime_bear_threshold=trading_config.market_regime_bear_threshold,
        market_regime_bear_exposure=trading_config.market_regime_bear_exposure,
        market_regime_vol_target=trading_config.market_regime_vol_target,
        market_regime_trend_threshold=trading_config.market_regime_trend_threshold,
        market_regime_min_exposure=trading_config.market_regime_min_exposure,
        market_regime_combine_method=trading_config.market_regime_combine_method,
        market_regime_trend_guard=trading_config.market_regime_trend_guard,
        market_regime_drawdown_guard=trading_config.market_regime_drawdown_guard,
        market_regime_drawdown_threshold=trading_config.market_regime_drawdown_threshold,
        industry_momentum_filter=trading_config.industry_momentum_filter,
        industry_momentum_bottom_pct=trading_config.industry_momentum_bottom_pct,
        industry_rotation_enhanced=trading_config.industry_rotation_enhanced,
        industry_rotation_alpha=trading_config.industry_rotation_alpha,
        position_sizing=trading_config.position_sizing,
        kelly_vol_window=trading_config.kelly_vol_window,
        kelly_max_leverage=trading_config.kelly_max_leverage,
        min_buy_value_ratio=trading_config.min_buy_value_ratio,
        enable_early_rebalance_on_empty=trading_config.enable_early_rebalance_on_empty,
    )



