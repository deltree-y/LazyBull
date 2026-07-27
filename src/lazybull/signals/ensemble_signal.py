"""双模型加权集成信号。"""

from typing import Dict, List, Optional

import pandas as pd

from .base import Signal
from .ml_signal import MLSignal


class EnsembleSignal(Signal):
    """对两个 MLSignal 的分数做加权融合。"""

    def __init__(
        self,
        signal_a: MLSignal,
        signal_b: MLSignal,
        *,
        weight_a: float = 0.5,
    ):
        super().__init__("ensemble_ml")
        self.signal_a = signal_a
        self.signal_b = signal_b
        self.weight_a = float(weight_a)
        self.weight_b = 1.0 - self.weight_a
        self._top_n = signal_a.top_n
        self.model_version = signal_a.model_version
        self.model_version_b = signal_b.model_version

    @property
    def top_n(self) -> int:
        return self._top_n

    @top_n.setter
    def top_n(self, value: int) -> None:
        self._top_n = value
        self.signal_a.top_n = value
        self.signal_b.top_n = value

    def _combine_ranked_candidates(
        self,
        ranked_a: List[tuple],
        ranked_b: List[tuple],
    ) -> List[tuple]:
        score_map: Dict[str, float] = {}
        for ts_code, score in ranked_a:
            score_map[ts_code] = score_map.get(ts_code, 0.0) + self.weight_a * float(score)
        for ts_code, score in ranked_b:
            score_map[ts_code] = score_map.get(ts_code, 0.0) + self.weight_b * float(score)
        return sorted(score_map.items(), key=lambda item: item[1], reverse=True)

    def generate_ranked(self, date: pd.Timestamp, universe: List[str], data: Dict) -> List[tuple]:
        ranked_a = self.signal_a.generate_ranked(date, universe, data)
        ranked_b = self.signal_b.generate_ranked(date, universe, data)
        combined = self._combine_ranked_candidates(ranked_a, ranked_b)
        self._last_ranked_candidates = list(combined)
        return combined

    def generate(self, date: pd.Timestamp, universe: List[str], data: Dict) -> Dict[str, float]:
        ranked = self.generate_ranked(date, universe, data)[: self.top_n]
        if not ranked:
            return {}

        positive_scores = {ts_code: score for ts_code, score in ranked if score > 0}
        if not positive_scores:
            equal_weight = 1.0 / len(ranked)
            return {ts_code: equal_weight for ts_code, _ in ranked}

        total_score = sum(positive_scores.values())
        if total_score <= 0:
            equal_weight = 1.0 / len(ranked)
            return {ts_code: equal_weight for ts_code, _ in ranked}

        return {
            ts_code: positive_scores[ts_code] / total_score
            for ts_code in positive_scores
        }

    def update_versions(self, model_version_a: int, model_version_b: int) -> None:
        self.signal_a.update_model_version(model_version_a)
        self.signal_b.update_model_version(model_version_b)
        self.model_version = model_version_a
        self.model_version_b = model_version_b

    def update_model_version(self, new_version: int) -> None:
        self.signal_a.update_model_version(new_version)
        self.model_version = new_version
