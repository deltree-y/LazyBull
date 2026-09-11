"""期末异常亏损模型：加载、输入契约与批量预测

方案 7.3：model.py 提供加载、输入契约、批量预测；不直接修改账户，
概率到退出意图的转换归 policy.py（后续阶段）。
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from .labels import TerminalLossLabelConfig
from .train import TerminalLossTrainConfig

MODEL_ARTIFACT_VERSION = 1


@dataclass
class TerminalLossModelConfig:
    """模型配置（随 artifact 持久化）。

    Attributes:
        task_id: 任务标识
        feature_names: 冻结特征清单（预测输入契约）
        artifact_version: artifact 结构版本
        train_config / label_config: 训练与标签配置快照
        metadata: 训练元数据（样本量、事件率、阶段日期等）
    """

    task_id: str
    feature_names: List[str]
    artifact_version: int = MODEL_ARTIFACT_VERSION
    train_config: TerminalLossTrainConfig = field(default_factory=TerminalLossTrainConfig)
    label_config: TerminalLossLabelConfig = field(default_factory=TerminalLossLabelConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TerminalLossModel:
    """期末异常亏损二分类模型封装。"""

    def __init__(self, config: TerminalLossModelConfig, classifier: Any) -> None:
        self.config = config
        self._clf = classifier

    # ── 预测 ──────────────────────────────────────────────

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """批量预测 p_loss（按 config.feature_names 严格对列）。

        Args:
            features_df: 每行一个持仓评估对象，须包含全部契约特征列；
                缺列必须报错，不静默缩减（方案 3.2 manifest 契约）

        Returns:
            与行序一致的正例概率数组
        """
        missing = [c for c in self.config.feature_names if c not in features_df.columns]
        if missing:
            raise ValueError(
                f"预测输入缺少特征列: {missing}；特征清单已冻结，" f"请检查特征母截面构建"
            )
        X = features_df[self.config.feature_names]
        return self._clf.predict_proba(X)[:, 1]

    # ── 序列化 ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """保存模型 artifact（classifier + config，joblib 单文件）。"""
        payload = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "config": asdict(self.config),
            "classifier": self._clf,
        }
        joblib.dump(payload, path)
        sidecar = path.rsplit(".", 1)[0] + ".json"
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(_json_safe(asdict(self.config)), f, ensure_ascii=False, indent=2)
        logger.info(f"terminal_loss 模型已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "TerminalLossModel":
        """加载模型 artifact，校验任务与结构版本。"""
        payload = joblib.load(path)
        version = payload.get("artifact_version")
        if version != MODEL_ARTIFACT_VERSION:
            raise ValueError(f"artifact 版本不符: 期望 {MODEL_ARTIFACT_VERSION}, 实际 {version}")
        raw = payload["config"]
        config = TerminalLossModelConfig(
            task_id=raw["task_id"],
            feature_names=raw["feature_names"],
            artifact_version=raw.get("artifact_version", MODEL_ARTIFACT_VERSION),
            train_config=TerminalLossTrainConfig(**raw.get("train_config", {})),
            label_config=TerminalLossLabelConfig(**raw.get("label_config", {})),
            metadata=raw.get("metadata", {}),
        )
        return cls(config, payload["classifier"])

    @property
    def feature_names(self) -> List[str]:
        return list(self.config.feature_names)


def _json_safe(value: Any) -> Any:
    """递归转换为 JSON 原生类型（sidecar 落盘用）。"""
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
