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
from .train import OBJECTIVE_RANK_PAIRWISE, TerminalLossTrainConfig

MODEL_ARTIFACT_VERSION = 1

#: 排序目标输出恢复排序分辨率的扰动尺度（严格单调、离散度 ≤ 2e-9）。
#: 背景（实测）：sklearn 1.9 的 ``IsotonicRegression`` 在大样本上会把拟合
#: 压缩成极少档位（合成实验 20 万样本 → ``transform`` 只有 10 个不同输出；
#: 生产 rank 折的 100 万行 Val → 133–254 个阈值 / 67–127 个输出档），直接
#: 输出会把模型完整的日内排序压成并列值——rank 臂实测每日只有 26–289 个
#: 不同 ``p_loss``（binary 臂 5800–9325），使门禁三个判据（全部只看排序）
#: 系统性偏低。这里在档位内按原始分数恢复严格排序，幅度远小于档间距
#: （实测 ≥1e-3），因此概率校准语义不变。
RANK_TIE_BREAK_EPS = 1e-9


def _restore_ranking_resolution(
    base: np.ndarray, scores: np.ndarray, eps: float = RANK_TIE_BREAK_EPS
) -> np.ndarray:
    """在校准档位内按原始排序分数恢复严格排序（v0.112.2）。

    ``scores`` 先压到 (-1, 1)，再映射到 [0, 1] 的极小扰动：

        p = base * (1 - 2ε) + ε * (1 + t) / 2,   t = s / (1 + |s|)

    - 同一档位内：不并列，按原始分数严格递增；
    - 不同档位间：档间距 g 的保持条件是 g > ε / (1 - 2ε) ≈ 1e-9（实测档间距
      ≥1e-3，远满足）→ 档位顺序不变；
    - 输出恒在 [0, 1]（不用 clip，避免边界档位重新并回并列）；
    - 同一输入分数 → 同一输出（不引入随机性）。
    """
    b = np.asarray(base, dtype=float)
    s = np.asarray(scores, dtype=float)
    t = s / (1.0 + np.abs(s))
    return b * (1.0 - 2.0 * eps) + eps * (1.0 + t) / 2.0


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
    """期末异常亏损模型封装。

    ``binary`` 目标：classifier 直接输出概率；
    ``rank_pairwise`` 目标：classifier 输出排序分数，必须经 Val 段 isotonic
    校准器映射回概率（缺校准器报错，不得静默当概率用）。
    """

    def __init__(
        self,
        config: TerminalLossModelConfig,
        classifier: Any,
        calibrator: Any = None,
    ) -> None:
        self.config = config
        self._clf = classifier
        self._calibrator = calibrator

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
        if self.config.train_config.objective == OBJECTIVE_RANK_PAIRWISE:
            if self._calibrator is None:
                raise ValueError(
                    "objective=rank_pairwise 的模型必须携带 Val 段 isotonic 校准器"
                    "（排序分数不是概率）；缺校准器不得静默当概率输出"
                )
            # 注意：XGBoost 返回 float32，而 isotonic 的输出 dtype 随输入——
            # float32 在 0.08 附近的量化步长约 7e-9，会盖过下面的 tie-break
            # 扰动（并把概率切成 ~7e-9 的网格冲掉排序）。先升到 float64。
            scores = np.asarray(self._clf.predict(X), dtype=np.float64)
            base = np.asarray(self._calibrator.transform(scores), dtype=np.float64)
            # 校准档位会合并大量并列值（sklearn isotonic 大样本压缩，见
            # RANK_TIE_BREAK_EPS 注释）：在档内按原始分数恢复严格排序，
            # 否则门禁的排序类判据（lift / auc_lift）会被系统性压低。
            return _restore_ranking_resolution(base, scores)
        return self._clf.predict_proba(X)[:, 1]

    # ── 序列化 ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """保存模型 artifact（classifier + config，joblib 单文件）。"""
        payload = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "config": asdict(self.config),
            "classifier": self._clf,
            "calibrator": self._calibrator,
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
        return cls(config, payload["classifier"], payload.get("calibrator"))

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
