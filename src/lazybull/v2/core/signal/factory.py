"""统一信号创建工厂

v2 迁移复制件（P2a-T6，行为冻结）：与旧模块 ``common/signal_factory.py``
双源并存至 v2 切换；除模块头与 import 改指外主体为复制：
- ``signals.{base,ml_signal,ensemble_signal}`` → 同包 v2 复制件（相对导入）；
- ``common.config`` → ``src.lazybull.common.config``（只读过渡依赖，同一
  模块对象同一单例，去除节点 P4 展平）；
- ``common.trading_config`` → ``src.lazybull.v2.common.trading_config``
  （T1 已迁，改指 v2 路径）。

消除 paper_trade.py / run_ml_backtest.py / bot_service.py 中
重复的 单模型/双模型 ensemble 判断逻辑。
"""

from typing import Optional

from loguru import logger

from src.lazybull.common.config import get_stock_selection_models_root
from src.lazybull.v2.common.trading_config import TradingConfig

from .base import Signal
from .ensemble_signal import EnsembleSignal
from .ml_signal import MLSignal


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
    resolved_models_dir = models_dir or get_stock_selection_models_root()

    # A5 已退役（P2a D6）：透传签名保留，λ>0 由 MLSignal 构造 fail-fast（默认 0=关闭）
    penalty_kwargs = {
        "downside_penalty": float(getattr(config, "downside_penalty", 0.0) or 0.0),
        "downside_penalty_column": str(
            getattr(config, "downside_penalty_column", "downside_vol_20")
        ),
    }

    signal_a = MLSignal(
        top_n=config.top_n,
        model_version=config.model_version,
        models_dir=resolved_models_dir,
        verbose=verbose,
        **penalty_kwargs,
    )

    if config.model_version_b is None:
        return signal_a

    signal_b = MLSignal(
        top_n=config.top_n,
        model_version=config.model_version_b,
        models_dir=resolved_models_dir,
        verbose=verbose,
        **penalty_kwargs,
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
