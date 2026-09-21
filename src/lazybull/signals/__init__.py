"""Signals模块初始化"""

from .base import EqualWeightSignal, FactorSignal, Signal
from .downside_penalty import (
    DOWNSIDE_PENALTY_COLUMNS,
    DOWNSIDE_PENALTY_GRID,
    apply_downside_penalty,
)
from .ensemble_signal import EnsembleSignal
from .ml_signal import MLSignal

__all__ = [
    "Signal",
    "EqualWeightSignal",
    "FactorSignal",
    "EnsembleSignal",
    "MLSignal",
    "DOWNSIDE_PENALTY_COLUMNS",
    "DOWNSIDE_PENALTY_GRID",
    "apply_downside_penalty",
]
