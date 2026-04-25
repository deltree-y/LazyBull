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
        signal_confidence_gate_enabled=getattr(args, "signal_confidence_gate_enabled", False),
        signal_confidence_gate_top_k=getattr(args, "signal_confidence_gate_top_k", 10),
        signal_confidence_gate_thresholds=getattr(
            args, "signal_confidence_gate_thresholds", [0.8, 1.2, 1.6]
        ),
        signal_confidence_gate_exposure_levels=getattr(
            args, "signal_confidence_gate_exposure_levels", [0.3, 0.6, 1.0]
        ),
        signal_gate_mode=getattr(args, "signal_gate_mode", "legacy"),
        signal_gate_cost_multiplier=getattr(args, "signal_gate_cost_multiplier", 2.0),
        signal_gate_round_trip_cost=getattr(args, "signal_gate_round_trip_cost", 0.003),
        signal_gate_quality_enabled=getattr(args, "signal_gate_quality_enabled", False),
        signal_gate_quality_window=getattr(args, "signal_gate_quality_window", 5),
        signal_gate_quality_threshold=getattr(args, "signal_gate_quality_threshold", 0.4),
        signal_gate_quality_halflife=getattr(args, "signal_gate_quality_halflife", 3),
        signal_gate_percentile_warmup=getattr(args, "signal_gate_percentile_warmup", 20),
        signal_gate_dynamic_topn=getattr(args, "signal_gate_dynamic_topn", False),
        signal_gate_topn_high_multiplier=getattr(args, "signal_gate_topn_high_multiplier", 0.6),
        signal_gate_topn_low_multiplier=getattr(args, "signal_gate_topn_low_multiplier", 1.5),
        holding_bonus_enabled=getattr(args, "holding_bonus_enabled", False),
        holding_bonus_sigma=getattr(args, "holding_bonus_sigma", 0.5),
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
        equity_curve_enabled=getattr(args, "bt_equity_curve_enabled", False),
        equity_curve_drawdown_thresholds=getattr(
            args, "bt_equity_curve_drawdown_thresholds", [5.0, 10.0, 15.0, 20.0]
        ),
        equity_curve_exposure_levels=getattr(
            args, "bt_equity_curve_exposure_levels", [0.8, 0.6, 0.4, 0.2]
        ),
        equity_curve_ma_short=getattr(args, "bt_equity_curve_ma_short", 5),
        equity_curve_ma_long=getattr(args, "bt_equity_curve_ma_long", 20),
        equity_curve_recovery_mode=getattr(args, "bt_equity_curve_recovery_mode", "gradual"),
        equity_curve_recovery_step=getattr(args, "bt_equity_curve_recovery_step", 0.25),
        equity_curve_recovery_delay_periods=getattr(
            args, "bt_equity_curve_recovery_delay_periods", 0
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
        market_regime_ma250_hard_stop=getattr(args, "market_regime_ma250_hard_stop", False),
        market_regime_ma250_threshold=getattr(args, "market_regime_ma250_threshold", 1.0),
        market_regime_ma250_exposure=getattr(args, "market_regime_ma250_exposure", 0.0),
        market_regime_ma250_atr_scaling=getattr(
            args, "market_regime_ma250_atr_scaling", False
        ),
        industry_momentum_filter=getattr(args, "industry_momentum_filter", False),
        industry_momentum_bottom_pct=getattr(args, "industry_momentum_bottom_pct", 0.2),
        industry_rotation_enhanced=getattr(args, "industry_rotation_enhanced", False),
        industry_rotation_alpha=getattr(args, "industry_rotation_alpha", 0.3),
        position_sizing=getattr(args, "position_sizing", "equal"),
        kelly_vol_window=getattr(args, "kelly_vol_window", 60),
        kelly_max_leverage=getattr(args, "kelly_max_leverage", 0.25),
        enable_profit_based_holding=getattr(args, "enable_profit_based_holding", False),
        early_exit_loss_threshold=getattr(args, "early_exit_loss_threshold", -0.05),
        early_exit_holding_ratio=getattr(args, "early_exit_holding_ratio", 0.6),
        profit_extension_threshold=getattr(args, "profit_extension_threshold", 0.05),
        profit_extension_days=getattr(args, "profit_extension_days", 5),
        profit_extension_mode=getattr(args, "profit_extension_mode", "pnl"),
        profit_extension_strength_threshold=getattr(
            args, "profit_extension_strength_threshold", 0.6
        ),
        profit_extension_strength_weights=getattr(
            args, "profit_extension_strength_weights", None
        ),
        use_atr_for_early_exit=getattr(args, "use_atr_for_early_exit", False),
        atr_multiplier=getattr(args, "atr_multiplier", 2.0),
        early_exit_mode=getattr(args, "early_exit_mode", "disabled"),
        early_exit_strength_protect_threshold=getattr(
            args, "early_exit_strength_protect_threshold", 0.55
        ),
        early_exit_max_reprieves=getattr(args, "early_exit_max_reprieves", 2),
        take_profit_threshold=getattr(args, "take_profit_threshold", None),
        take_profit_refill=getattr(args, "take_profit_refill", True),
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
        equity_curve_config=trading_config.create_equity_curve_config(),
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
        market_regime_ma250_hard_stop=trading_config.market_regime_ma250_hard_stop,
        market_regime_ma250_threshold=trading_config.market_regime_ma250_threshold,
        market_regime_ma250_exposure=trading_config.market_regime_ma250_exposure,
        market_regime_ma250_atr_scaling=trading_config.market_regime_ma250_atr_scaling,
        industry_momentum_filter=trading_config.industry_momentum_filter,
        industry_momentum_bottom_pct=trading_config.industry_momentum_bottom_pct,
        industry_rotation_enhanced=trading_config.industry_rotation_enhanced,
        industry_rotation_alpha=trading_config.industry_rotation_alpha,
        position_sizing=trading_config.position_sizing,
        kelly_vol_window=trading_config.kelly_vol_window,
        kelly_max_leverage=trading_config.kelly_max_leverage,
        enable_profit_based_holding=trading_config.enable_profit_based_holding,
        early_exit_loss_threshold=trading_config.early_exit_loss_threshold,
        early_exit_holding_ratio=trading_config.early_exit_holding_ratio,
        profit_extension_threshold=trading_config.profit_extension_threshold,
        profit_extension_days=trading_config.profit_extension_days,
        profit_extension_mode=trading_config.profit_extension_mode,
        profit_extension_strength_threshold=trading_config.profit_extension_strength_threshold,
        profit_extension_strength_weights=trading_config.profit_extension_strength_weights,
        use_atr_for_early_exit=trading_config.use_atr_for_early_exit,
        atr_multiplier=trading_config.atr_multiplier,
        early_exit_mode=trading_config.early_exit_mode,
        early_exit_strength_protect_threshold=(
            trading_config.early_exit_strength_protect_threshold
        ),
        early_exit_max_reprieves=trading_config.early_exit_max_reprieves,
        take_profit_threshold=trading_config.take_profit_threshold,
        take_profit_refill=trading_config.take_profit_refill,
        enable_early_rebalance_on_empty=trading_config.enable_early_rebalance_on_empty,
        signal_gate_quality_enabled=trading_config.signal_gate_quality_enabled,
        signal_gate_quality_window=trading_config.signal_gate_quality_window,
        signal_gate_quality_threshold=trading_config.signal_gate_quality_threshold,
        signal_gate_quality_halflife=trading_config.signal_gate_quality_halflife,
        signal_gate_dynamic_topn=trading_config.signal_gate_dynamic_topn,
        signal_gate_topn_high_multiplier=trading_config.signal_gate_topn_high_multiplier,
        signal_gate_topn_low_multiplier=trading_config.signal_gate_topn_low_multiplier,
        holding_bonus_enabled=trading_config.holding_bonus_enabled,
        holding_bonus_sigma=trading_config.holding_bonus_sigma,
    )


def restore_signal_quality_state(engine: BacktestEngineML, signal: Optional[MLSignal]) -> None:
    """从持久化 signal 恢复滚动质量状态。"""
    if signal is None or not getattr(engine, "signal_gate_quality_enabled", False):
        return

    state = getattr(signal, "_persisted_quality_state", None)
    if state is None:
        return

    engine._prediction_quality_history = list(state["history"])
    engine._rolling_quality_score = state["score"]
    engine._quality_warmup_remaining = state["warmup_remaining"]
    logger.info(
        f"质量监控状态已恢复: {len(state['history'])}条历史, "
        f"score={state['score']:.3f}, warmup_remaining={state['warmup_remaining']}"
    )


def persist_signal_quality_state(engine: BacktestEngineML, signal: Optional[MLSignal]) -> None:
    """将滚动质量状态写回持久化 signal。"""
    if signal is None or not getattr(engine, "signal_gate_quality_enabled", False):
        return

    signal._persisted_quality_state = {
        "history": list(engine._prediction_quality_history),
        "score": engine._rolling_quality_score,
        "warmup_remaining": engine._quality_warmup_remaining,
    }
    logger.info(
        f"质量监控状态已保存: {len(engine._prediction_quality_history)}条历史, "
        f"score={engine._rolling_quality_score:.3f}"
    )
