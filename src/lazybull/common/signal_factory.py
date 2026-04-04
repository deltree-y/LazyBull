"""统一信号创建工厂

消除 paper_trade.py / run_ml_backtest.py / bot_service.py 中
重复的 单模型/双模型 ensemble 判断逻辑。
"""

from typing import Optional

from loguru import logger

from .trading_config import TradingConfig
from ..signals.ml_signal import MLSignal, EnsembleMLSignal
from ..signals.base import Signal


def create_signal(
    config: TradingConfig,
    *,
    models_dir: str = "./data/models",
    verbose: bool = False,
) -> Signal:
    """根据 TradingConfig 创建 MLSignal 或 EnsembleMLSignal。

    Args:
        config: 统一策略参数
        models_dir: 模型目录
        verbose: 是否输出详细日志

    Returns:
        Signal 实例
    """
    if config.model_version_b is not None:
        signal = EnsembleMLSignal(
            model_version_a=config.model_version,
            model_version_b=config.model_version_b,
            ensemble_weight_a=config.ensemble_weight_a,
            top_n=config.top_n,
            models_dir=models_dir,
            weight_method=config.weight_method,
            signal_confidence_gate_enabled=config.signal_confidence_gate_enabled,
            signal_confidence_gate_top_k=config.signal_confidence_gate_top_k,
            signal_confidence_gate_thresholds=config.signal_confidence_gate_thresholds,
            signal_confidence_gate_exposure_levels=config.signal_confidence_gate_exposure_levels,
            verbose=verbose,
        )
        logger.info(
            f"使用双模型集成: model_a=v{config.model_version}, "
            f"model_b=v{config.model_version_b}, "
            f"weight_a={config.ensemble_weight_a}"
        )
    else:
        signal = MLSignal(
            top_n=config.top_n,
            model_version=config.model_version,
            models_dir=models_dir,
            weight_method=config.weight_method,
            signal_confidence_gate_enabled=config.signal_confidence_gate_enabled,
            signal_confidence_gate_top_k=config.signal_confidence_gate_top_k,
            signal_confidence_gate_thresholds=config.signal_confidence_gate_thresholds,
            signal_confidence_gate_exposure_levels=config.signal_confidence_gate_exposure_levels,
            verbose=verbose,
        )
    return signal
