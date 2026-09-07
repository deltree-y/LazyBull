"""期末异常亏损二分类训练与概率质量评估

实现 docs/plans/terminal_loss_risk_model_plan.md 3.6/3.7/8.1 契约：

- 早停指标使用内置 logloss（概率质量优先，服务后续校准），PR-AUC 仅
  记录展示不参与早停；如需自定义指标必须模块级可 pickle（早停指标契约）；
- 样本权重按方案第 6 节规则 1 传入（每行 1/期限网格大小）；
- 正则尺度决策 A（权重 1/20、正则参数保持原值：等效正则增强，且
  min_child_weight=1 使单一 (股票,日) 组无法独自成叶），策略标识落入
  元数据，禁止不登记；
- 概率质量报告按 h 与 h × σ 分位双维分组（方案 8.1：标签以 σ 归一化，
  高 σ 组界限天然宽松，p_loss 偏低必须被显式看见）；
- sigmoid（Platt）校准器在独立段拟合（方案 3.7），首轮仅离线研究。
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from xgboost import XGBClassifier

from .labels import TerminalLossLabelConfig

REG_SCALE_POLICY_A = "A_keep_regularization_with_1_over_grid_weights"


@dataclass(frozen=True)
class TerminalLossTrainConfig:
    """训练超参数（方案 3.6 保守研究起点）。

    Attributes:
        regularization_scale_policy: 正则尺度决策标识（A=权重 1/20 且正则
            参数保持原值；禁止无登记地混用其他尺度）
    """

    max_depth: int = 3
    learning_rate: float = 0.03
    n_estimators: int = 500
    early_stopping_rounds: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    scale_pos_weight: float = 1.0
    random_state: int = 42
    eval_metric: str = "logloss"
    device: str = "cuda"
    regularization_scale_policy: str = REG_SCALE_POLICY_A


@dataclass
class TerminalLossTrainResult:
    """训练结果与完整元数据（供模型注册与审计）。"""

    classifier: Any
    best_iteration: int
    train_config: TerminalLossTrainConfig
    label_config: TerminalLossLabelConfig
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        """导出 JSON 安全的元数据。"""
        meta = {
            "task_id": self.label_config.task_id,
            "feature_names": list(self.feature_names),
            "train_config": asdict(self.train_config),
            "label_config": asdict(self.label_config),
        }
        meta.update(self.metadata)
        return _json_safe(meta)


def train_terminal_loss_model(
    train_matrix: pd.DataFrame,
    es_matrix: pd.DataFrame,
    feature_names: List[str],
    train_config: Optional[TerminalLossTrainConfig] = None,
    label_config: Optional[TerminalLossLabelConfig] = None,
    stage_dates: Optional[Dict[str, Tuple[str, str]]] = None,
) -> TerminalLossTrainResult:
    """训练期末异常亏损二分类模型（Train 拟合 + ES 早停）。

    Args:
        train_matrix: 训练段矩阵（TERMINAL_LOSS_FEATURES + loss_label +
            sample_weight，由 build_training_matrix 产出）
        es_matrix: 早停段矩阵（同 schema）
        feature_names: 冻结特征清单（列缺失必须报错）
        train_config: 训练超参
        label_config: 标签配置（落入元数据）
        stage_dates: 各阶段 (start, end) 日期，落入元数据供审计

    Returns:
        TerminalLossTrainResult
    """
    cfg = train_config or TerminalLossTrainConfig()
    lcfg = label_config or TerminalLossLabelConfig()
    missing = [c for c in feature_names if c not in train_matrix.columns]
    if missing:
        raise ValueError(f"训练矩阵缺少特征列: {missing}")
    if train_matrix.empty or es_matrix.empty:
        raise ValueError("训练段或早停段为空，拒绝训练")

    X_train = train_matrix[feature_names]
    y_train = train_matrix["loss_label"].astype(int)
    w_train = train_matrix["sample_weight"].astype(float)
    X_es = es_matrix[feature_names]
    y_es = es_matrix["loss_label"].astype(int)

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric=cfg.eval_metric,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        early_stopping_rounds=cfg.early_stopping_rounds,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        scale_pos_weight=cfg.scale_pos_weight,
        random_state=cfg.random_state,
        tree_method="hist",
        device=cfg.device,
    )
    clf.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_es, y_es)],
        sample_weight_eval_set=[(es_matrix["sample_weight"].astype(float),)],
        verbose=False,
    )
    best_iteration = int(getattr(clf, "best_iteration", cfg.n_estimators) or cfg.n_estimators)

    meta: Dict[str, Any] = {
        "n_train": int(len(train_matrix)),
        "n_es": int(len(es_matrix)),
        "train_event_rate": float(y_train.mean()),
        "es_event_rate": float(y_es.mean()),
        "stage_dates": {k: list(v) for k, v in (stage_dates or {}).items()},
        "train_h_distribution": train_matrix["h"].value_counts().sort_index().to_dict(),
        "es_h_distribution": es_matrix["h"].value_counts().sort_index().to_dict(),
    }
    logger.info(
        f"terminal_loss 训练完成: train={len(train_matrix)} 行"
        f"(事件率 {y_train.mean():.4f}), es={len(es_matrix)} 行"
        f"(事件率 {y_es.mean():.4f}), best_iteration={best_iteration}"
    )
    return TerminalLossTrainResult(
        classifier=clf,
        best_iteration=best_iteration,
        train_config=cfg,
        label_config=lcfg,
        feature_names=list(feature_names),
        metadata=meta,
    )


# ── 概率质量评估（方案 8.1 报告门禁）──────────────────────────────


def evaluate_probability_quality(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    h_values: np.ndarray,
    sigma_values: Optional[np.ndarray] = None,
    sigma_bins: int = 4,
) -> Dict[str, Any]:
    """整体 + 分 h + 分 h × σ 分位的概率质量报告。

    Args:
        y_true: 0/1 标签
        p_pred: 预测概率
        h_values: 每行剩余期限
        sigma_values: 每行 sigma_daily_20（提供时输出 h × σ 双维校准表）
        sigma_bins: σ 分组数

    Returns:
        dict：logloss / brier / pr_auc / event_rate / 分 h 表 /
        calibration_by_h_sigma（DataFrame）
    """
    y_true = np.asarray(y_true, dtype=int)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), 1e-12, 1 - 1e-12)

    report: Dict[str, Any] = {
        "n": int(len(y_true)),
        "event_rate": float(y_true.mean()),
        "mean_pred": float(p_pred.mean()),
        "logloss": float(log_loss(y_true, p_pred, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_pred)),
        "pr_auc": float(average_precision_score(y_true, p_pred)),
    }
    by_h = []
    for h in np.unique(h_values):
        m = h_values == h
        by_h.append(
            {
                "h": int(h),
                "n": int(m.sum()),
                "event_rate": float(y_true[m].mean()),
                "mean_pred": float(p_pred[m].mean()),
                "brier": float(brier_score_loss(y_true[m], p_pred[m])),
            }
        )
    report["by_h"] = by_h

    if sigma_values is not None:
        sigma_values = np.asarray(sigma_values, dtype=float)
        rows = []
        for h in np.unique(h_values):
            m_h = h_values == h
            sig_h = sigma_values[m_h]
            valid = ~np.isnan(sig_h)
            if valid.sum() < sigma_bins:
                rows.append(
                    {"h": int(h), "sigma_bin": "all", "n": int(valid.sum()),
                     "event_rate": float(y_true[m_h][valid].mean()) if valid.any() else np.nan,
                     "mean_pred": float(p_pred[m_h][valid].mean()) if valid.any() else np.nan}
                )
                continue
            qs = np.nanquantile(sig_h, np.linspace(0, 1, sigma_bins + 1))
            bin_idx = np.digitize(sig_h, qs[1:-1], right=True)
            for b in range(sigma_bins):
                m_b = bin_idx == b
                if m_b.sum() == 0:
                    continue
                y_b = y_true[m_h][m_b]
                p_b = p_pred[m_h][m_b]
                rows.append(
                    {
                        "h": int(h),
                        "sigma_bin": f"q{b + 1}",
                        "n": int(m_b.sum()),
                        "event_rate": float(y_b.mean()),
                        "mean_pred": float(p_b.mean()),
                        "brier": float(brier_score_loss(y_b, p_b)),
                    }
                )
        report["calibration_by_h_sigma"] = pd.DataFrame(rows)
    return report


# ── sigmoid（Platt）校准器（方案 3.7）─────────────────────────────


class SigmoidCalibrator:
    """Platt sigmoid 校准：在独立校准段把模型分数映射为校准概率。

    仅正负事件与跨日覆盖足够时输出才可用于激活决策（调用方负责门禁）；
    本类不持有训练期数据，可安全序列化。
    """

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self.fitted_ = False

    def fit(self, raw_scores: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("校准段只有单一类别，拒绝拟合 sigmoid 校准器")
        self._lr.fit(raw_scores, y)
        self.fitted_ = True
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("校准器未拟合")
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(raw_scores)[:, 1]


def _json_safe(value: Any) -> Any:
    """递归转换为 JSON 原生类型。"""
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
