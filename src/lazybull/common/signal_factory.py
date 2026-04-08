"""统一信号创建工厂

消除 paper_trade.py / run_ml_backtest.py / bot_service.py 中
重复的 单模型/双模型 ensemble 判断逻辑。
"""

from typing import Optional

from loguru import logger

from ..signals.base import Signal
from ..signals.ml_signal import MLSignal
from .trading_config import TradingConfig


def create_signal(
    config: TradingConfig,
    *,
    models_dir: str = "./data/models",
    verbose: bool = False,
) -> Signal:
    """根据 TradingConfig 创建 MLSignal。

    Args:
        config: 统一策略参数
        models_dir: 模型目录
        verbose: 是否输出详细日志

    Returns:
        Signal 实例
    """
    # 公共门控参数
    gate_kwargs = dict(
        signal_confidence_gate_enabled=config.signal_confidence_gate_enabled,
        signal_confidence_gate_top_k=config.signal_confidence_gate_top_k,
        signal_confidence_gate_thresholds=config.signal_confidence_gate_thresholds,
        signal_confidence_gate_exposure_levels=config.signal_confidence_gate_exposure_levels,
        signal_gate_mode=config.signal_gate_mode,
        signal_gate_cost_multiplier=config.signal_gate_cost_multiplier,
        signal_gate_round_trip_cost=config.signal_gate_round_trip_cost,
        signal_gate_percentile_warmup=config.signal_gate_percentile_warmup,
    )

    signal = MLSignal(
        top_n=config.top_n,
        model_version=config.model_version,
        models_dir=models_dir,
        weight_method=config.weight_method,
        verbose=verbose,
        **gate_kwargs,
    )
    return signal
