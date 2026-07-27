"""统一信号创建工厂

消除 paper_trade.py / run_ml_backtest.py / bot_service.py 中
重复的 单模型/双模型 ensemble 判断逻辑。
"""

from typing import Optional

from loguru import logger

from .config import get_models_root
from ..signals.base import Signal
from ..signals.ensemble_signal import EnsembleSignal
from ..signals.ml_signal import MLSignal
from .trading_config import TradingConfig


def create_signal(
    config: TradingConfig,
    *,
    models_dir: Optional[str] = None,
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
    resolved_models_dir = models_dir or get_models_root()

    signal_a = MLSignal(
        top_n=config.top_n,
        model_version=config.model_version,
        models_dir=resolved_models_dir,
        verbose=verbose,
    )

    if config.model_version_b is None:
        return signal_a

    signal_b = MLSignal(
        top_n=config.top_n,
        model_version=config.model_version_b,
        models_dir=resolved_models_dir,
        verbose=verbose,
    )
    logger.info(
        f"创建双模型集成信号: A=v{config.model_version}, "
        f"B=v{config.model_version_b}, weight_a={config.ensemble_weight_a:.2f}"
    )
    return EnsembleSignal(
        signal_a,
        signal_b,
        weight_a=config.ensemble_weight_a,
    )
