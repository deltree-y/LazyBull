"""持仓风险控制模型

核心类：
  - PositionRiskConfig : 模型配置 dataclass（系数映射、特征名、校准指标）
  - PositionRiskModel  : 封装 XGBoost 三分类器，提供 predict/predict_batch
  - PositionRiskMonitor: 回测/纸面交易引擎插件，提供每日持仓评估

架构约束：Monitor 通过注入方式进入 engine，engine 仅保留 ~8 行调用胶水。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CLASS_REDUCE = 0
CLASS_HOLD = 1
CLASS_INCREASE = 2

CLASS_LABELS = {CLASS_REDUCE: "REDUCE", CLASS_HOLD: "HOLD", CLASS_INCREASE: "INCREASE"}

DEFAULT_COEFFICIENT_MAP = {
    CLASS_REDUCE: 0.5,
    CLASS_HOLD: 1.0,
    CLASS_INCREASE: 1.5,
}

# 注册表中风控模型的前缀
RISK_MODEL_PREFIX = "risk_pos_"


# ---------------------------------------------------------------------------
# PositionRiskConfig
# ---------------------------------------------------------------------------

@dataclass
class PositionRiskConfig:
    """风控模型配置。

    Attributes:
        model_version: 模型注册版本号
        feature_names: 模型使用的特征列名列表
        class_labels: 类别名映射
        coefficient_map: 各类别对应的仓位系数
        proba_threshold: REDUCE 触发提前退出的概率阈值
        calibration_f1: 校准段宏平均 F1
        calibration_monotonic: 校准段三类 forward return 是否单调
        extra: 额外元数据
    """
    model_version: int
    feature_names: List[str]
    class_labels: Dict[int, str] = field(default_factory=lambda: CLASS_LABELS)
    coefficient_map: Dict[int, float] = field(default_factory=lambda: DEFAULT_COEFFICIENT_MAP)
    proba_threshold: float = 0.6
    calibration_f1: Optional[float] = None
    calibration_monotonic: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_version': self.model_version,
            'feature_names': self.feature_names,
            'class_labels': {str(k): v for k, v in self.class_labels.items()},
            'coefficient_map': {str(k): v for k, v in self.coefficient_map.items()},
            'proba_threshold': self.proba_threshold,
            'calibration_f1': self.calibration_f1,
            'calibration_monotonic': self.calibration_monotonic,
            'extra': self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PositionRiskConfig':
        return cls(
            model_version=d['model_version'],
            feature_names=d['feature_names'],
            class_labels={int(k): v for k, v in d.get('class_labels', {}).items()},
            coefficient_map={int(k): v for k, v in d.get('coefficient_map', {}).items()},
            proba_threshold=d.get('proba_threshold', 0.6),
            calibration_f1=d.get('calibration_f1'),
            calibration_monotonic=d.get('calibration_monotonic'),
            extra=d.get('extra', {}),
        )


# ---------------------------------------------------------------------------
# PositionRiskResult
# ---------------------------------------------------------------------------

@dataclass
class PositionRiskResult:
    """单只持仓的风险评估结果。"""
    ts_code: str
    predicted_class: int
    coefficient: float
    proba: float               # 预测类别的置信度
    all_probas: np.ndarray     # 三分类概率 [P(REDUCE), P(HOLD), P(INCREASE)]

    @property
    def class_name(self) -> str:
        return CLASS_LABELS.get(self.predicted_class, "UNKNOWN")

    @property
    def should_exit(self) -> bool:
        """是否应触发提前退出。"""
        return self.predicted_class == CLASS_REDUCE

    @property
    def should_increase(self) -> bool:
        """是否可加仓。"""
        return self.predicted_class == CLASS_INCREASE


# ---------------------------------------------------------------------------
# PositionRiskModel
# ---------------------------------------------------------------------------

class PositionRiskModel:
    """持仓风控模型：封装 XGBoost 三分类器。

    使用方式：
        model = PositionRiskModel(config, xgb_classifier)
        result = model.predict_single(features_series)
        results = model.predict_batch(features_df)
    """

    def __init__(self, config: PositionRiskConfig, classifier: Any):
        """
        Args:
            config: 模型配置
            classifier: 已训练的 XGBClassifier (或兼容的 sklearn 分类器)
        """
        self.config = config
        self._clf = classifier

    # ── 预测 ──────────────────────────────────────────────

    def predict_single(self, features: pd.Series) -> PositionRiskResult:
        """对单只股票预测。

        Args:
            features: 包含所有 config.feature_names 的 Series

        Returns:
            PositionRiskResult
        """
        X = features[self.config.feature_names].values.reshape(1, -1)
        pred_class = int(self._clf.predict(X)[0])
        proba = self._clf.predict_proba(X)[0]
        return PositionRiskResult(
            ts_code=str(features.get('ts_code', '')),
            predicted_class=pred_class,
            coefficient=self.config.coefficient_map.get(pred_class, 1.0),
            proba=float(proba[pred_class]),
            all_probas=proba,
        )

    def predict_batch(self, features_df: pd.DataFrame) -> Dict[str, PositionRiskResult]:
        """批量预测。

        Args:
            features_df: DataFrame，每行一只股票，含 ts_code 和特征列

        Returns:
            {ts_code: PositionRiskResult} 字典
        """
        if len(features_df) == 0:
            return {}

        X = features_df[self.config.feature_names].values
        pred_classes = self._clf.predict(X)
        probas = self._clf.predict_proba(X)

        results = {}
        ts_codes = features_df.get('ts_code', features_df.index).values
        for i, ts_code in enumerate(ts_codes):
            cls = int(pred_classes[i])
            results[str(ts_code)] = PositionRiskResult(
                ts_code=str(ts_code),
                predicted_class=cls,
                coefficient=self.config.coefficient_map.get(cls, 1.0),
                proba=float(probas[i][cls]),
                all_probas=probas[i],
            )
        return results

    # ── 序列化 ────────────────────────────────────────────

    def get_classifier(self) -> Any:
        """获取内部的 XGBoost 分类器。"""
        return self._clf

    @property
    def feature_names(self) -> List[str]:
        return self.config.feature_names


# ---------------------------------------------------------------------------
# PositionRiskMonitor（引擎插件）
# ---------------------------------------------------------------------------

class PositionRiskMonitor:
    """风控模型引擎插件：每日评估持仓。

    通过注入方式集成到 BacktestEngine / PaperTradeRunner：
        engine.position_risk_monitor = PositionRiskMonitor(model)

    引擎内部仅需 ~8 行胶水代码：
        if self.position_risk_monitor:
            for pos in self.positions.values():
                if self.position_risk_monitor.should_exit_early(pos, date, features):
                    self._queue_condition_sell(pos, "risk_model_exit")
    """

    def __init__(
        self,
        model: PositionRiskModel,
        proba_threshold: Optional[float] = None,
    ):
        """
        Args:
            model: 已加载的 PositionRiskModel
            proba_threshold: REDUCE 触发提前退出的最低概率（可覆盖 config 中的值）
        """
        self.model = model
        self.proba_threshold = (
            proba_threshold if proba_threshold is not None
            else model.config.proba_threshold
        )
        # 每日评估缓存: {date: {ts_code: PositionRiskResult}}
        self._cache: Dict[str, Dict[str, PositionRiskResult]] = {}

    # ── 核心评估接口 ──────────────────────────────────────

    def evaluate_position(
        self,
        ts_code: str,
        date: str,
        features: pd.Series,
    ) -> PositionRiskResult:
        """评估单只持仓。

        Args:
            ts_code: 股票代码
            date: 评估日期
            features: 该股票的特征行（必须含模型所需全部列）

        Returns:
            PositionRiskResult
        """
        # 检查缓存
        if date in self._cache and ts_code in self._cache[date]:
            return self._cache[date][ts_code]

        # 检查特征完整性
        missing = set(self.model.feature_names) - set(features.index)
        if missing:
            logger.warning(
                f"风控模型评估 {ts_code} @ {date}: 缺失特征 {missing}，"
                f"默认返回 HOLD"
            )
            result = PositionRiskResult(
                ts_code=ts_code,
                predicted_class=CLASS_HOLD,
                coefficient=1.0,
                proba=0.5,
                all_probas=np.array([0.33, 0.34, 0.33]),
            )
        else:
            result = self.model.predict_single(features)

        # 缓存
        self._cache.setdefault(date, {})[ts_code] = result
        return result

    def evaluate_positions(
        self,
        positions: List[Dict],
        date: str,
        features_df: pd.DataFrame,
    ) -> Dict[str, PositionRiskResult]:
        """批量评估持仓。

        Args:
            positions: 持仓列表，每个含 ts_code
            date: 评估日期
            features_df: 当日特征截面

        Returns:
            {ts_code: PositionRiskResult}
        """
        ts_codes = [p['ts_code'] for p in positions]
        batch_features = features_df[features_df['ts_code'].isin(ts_codes)]
        batch_results = self.model.predict_batch(batch_features)

        # 对不在特征中的持仓，默认 HOLD
        results = {}
        for ts_code in ts_codes:
            if ts_code in batch_results:
                results[ts_code] = batch_results[ts_code]
            else:
                results[ts_code] = PositionRiskResult(
                    ts_code=ts_code,
                    predicted_class=CLASS_HOLD,
                    coefficient=1.0,
                    proba=0.5,
                    all_probas=np.array([0.33, 0.34, 0.33]),
                )

        # 缓存
        self._cache[date] = results
        return results

    # ── 决策接口（引擎调用的胶水方法）─────────────────────

    def should_exit_early(
        self,
        position: Dict,
        date: str,
        features: pd.Series,
    ) -> bool:
        """是否应触发提前退出。

        Args:
            position: 持仓字典（含 ts_code）
            date: 当前日期
            features: 该股票特征行

        Returns:
            True if class=REDUCE and proba > threshold
        """
        result = self.evaluate_position(position['ts_code'], date, features)
        return (
            result.predicted_class == CLASS_REDUCE
            and result.proba >= self.proba_threshold
        )

    def get_weight_multiplier(
        self,
        ts_code: str,
        date: str,
        features: pd.Series,
    ) -> float:
        """获取仓位调节系数。

        Args:
            ts_code: 股票代码
            date: 当前日期
            features: 该股票特征行

        Returns:
            {0.5, 1.0, 1.5} 之一
        """
        result = self.evaluate_position(ts_code, date, features)
        return result.coefficient

    # ── 缓存管理 ──────────────────────────────────────────

    def clear_cache(self, before_date: Optional[str] = None):
        """清理缓存。"""
        if before_date is None:
            self._cache.clear()
        else:
            self._cache = {
                d: v for d, v in self._cache.items() if d >= before_date
            }

    # ── 统计接口 ──────────────────────────────────────────

    def get_daily_summary(self, date: str) -> Dict[str, Any]:
        """获取某日的评估摘要。"""
        if date not in self._cache:
            return {}
        results = self._cache[date]
        classes = [r.predicted_class for r in results.values()]
        if not classes:
            return {}
        return {
            'date': date,
            'total': len(classes),
            'reduce_count': classes.count(CLASS_REDUCE),
            'hold_count': classes.count(CLASS_HOLD),
            'increase_count': classes.count(CLASS_INCREASE),
        }
