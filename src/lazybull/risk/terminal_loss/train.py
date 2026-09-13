"""期末异常亏损二分类训练与概率质量评估

实现 docs/plans/terminal_loss_risk_model_plan.md 3.6/3.7/8.1 契约：

- 早停段与评估段分离（方案 5.2，v0.109.0 修复）：早停只使用 Val 段
  （Train 尾部留出的内部验证段），ES 段只用于概率质量报告与门禁评估，
  **不参与早停**——把门禁指标算在早停选择段上会带乐观偏差；
- 早停指标：默认 `logloss`（概率质量优先，服务后续校准），可切
  `rank_ic_daily`（逐日截面 Spearman 均值，与门禁 lift 同向；实现复用
  `ml/train_core/eval.py::make_neg_rank_ic_daily`，模块级可 pickle 满足
  早停指标契约）；两种口径是不同签名，不得混组比较；
- 排序目标（rank_pairwise）的早停口径见 `_fit_rank_pairwise`：默认 `auc`
  由自实现回调 `ml/train_core/eval.py::ValPooledAUCStopping` 承担（v0.112.1
  ——XGBoost 内置 ranking AUC 的 O(n_g²) 成对展开会超过 int32 上限）；
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from xgboost import DMatrix, XGBClassifier, XGBRanker

from ...ml.train_core.eval import ValPooledAUCStopping, make_neg_rank_ic_daily
from .labels import TerminalLossLabelConfig

REG_SCALE_POLICY_A = "A_keep_regularization_with_1_over_grid_weights"

#: 逐日截面 Spearman RankIC 均值早停指标（与门禁 lift 同向）
EVAL_METRIC_RANK_IC_DAILY = "rank_ic_daily"

#: 排序目标（rank_pairwise）的早停指标（只对排序目标有效）
EVAL_METRIC_NDCG = "ndcg"

#: 排序目标的**默认**早停指标：Val 段池化 ROC AUC（全体行，不分组）。
#: 为什么用 auc 而不是逐日 RankIC（v0.112.0）：XGBRanker 下自定义 callable 不可用
#: ——带 qid 的 eval_set 传给 feval 的是**组级**数组（实测 300 行 Val / 22 日 →
#: 传入 22 个预测值），与逐行 RankIC 指标的行序契约冲突（IndexError）。AUC 与
#: 门禁第三判据 ``auc_lift = 2×AUC`` 同向且**基准率不变**，是 rank 目标下可用的
#: 最接近口径；``ndcg`` 保留供对照（Val 上极易饱和，实测 4/8 折在 ≤63 棵就停）。
#: v0.112.1 起 auc 早停由自实现回调（``ValPooledAUCStopping``）承担，**不再**把
#: ``eval_metric="auc"`` 交给 XGBoost 内置评估：内置 ranking AUC 对每个 query
#: group 做 O(n_g²) 成对展开并断言 Σ(n_g+2)(n_g-1)/2 < INT32_MAX，6 个月 Val 段
#: 实测 29.7 亿 > 21.47 亿（GPU 后端），8 折 rank 臂每折必然 XGBoostError。
EVAL_METRIC_AUC = "auc"

#: 支持的早停指标（未知取值必须明确失败，禁止静默回退）
SUPPORTED_EVAL_METRICS = (
    "logloss",
    EVAL_METRIC_RANK_IC_DAILY,
    EVAL_METRIC_NDCG,
    EVAL_METRIC_AUC,
)

#: 训练目标（v0.111.0，显式枚举）
#: - binary：XGBClassifier(objective=binary:logistic)，输出即概率；
#: - rank_pairwise：XGBRanker(rank:pairwise, qid=trade_date) 只学“当日截面排序”
#:   （单折 2023H2：daynorm 1.119→1.334、raw 1.232→1.472），再用 Val 段
#:   isotonic 映射回概率——阈值政策仍拿到概率语义；校准器随 artifact 落盘。
OBJECTIVE_BINARY = "binary"
OBJECTIVE_RANK_PAIRWISE = "rank_pairwise"
SUPPORTED_OBJECTIVES = (OBJECTIVE_BINARY, OBJECTIVE_RANK_PAIRWISE)

#: 目标 × 早停指标合法组合（非法组合必须报错，不静默回退）：
#: - binary：logloss（概率校准）/ rank_ic_daily（逐日截面排序）；
#: - rank_pairwise：auc（**默认**，自实现回调池化 AUC，与门禁第三判据
#:   auc_lift 同向、基准率不变）/ ndcg（内置列表口径，Val 上极易饱和，保留供对照）。
#: 自定义 callable（如逐日 RankIC）在 XGBRanker 下不可用：带 qid 的
#: eval_set 会把**组级**数组传给 feval，与逐行指标的行序契约冲突。
ALLOWED_METRICS: Dict[str, Tuple[str, ...]] = {
    OBJECTIVE_BINARY: ("logloss", EVAL_METRIC_RANK_IC_DAILY),
    OBJECTIVE_RANK_PAIRWISE: (EVAL_METRIC_AUC, EVAL_METRIC_NDCG),
}


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
    objective: str = OBJECTIVE_BINARY
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
    #: 排序目标的 Val 段 isotonic 校准器（binary 目标为 None）
    calibrator: Optional[Any] = None

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
    val_matrix: pd.DataFrame,
    feature_names: List[str],
    train_config: Optional[TerminalLossTrainConfig] = None,
    label_config: Optional[TerminalLossLabelConfig] = None,
    stage_dates: Optional[Dict[str, Tuple[str, str]]] = None,
) -> TerminalLossTrainResult:
    """训练期末异常亏损二分类模型（Train 拟合 + Val 早停，ES 仅评估）。

    Args:
        train_matrix: 训练段矩阵（TERMINAL_LOSS_FEATURES + loss_label +
            sample_weight，由 build_training_matrix 产出）
        val_matrix: 早停段（Val）矩阵（同 schema）：早停段必须与评估段（ES）
            分离，且满足 ``label_end_date < ES 段起点``（方案 5.2/5.3）
        feature_names: 冻结特征清单（列缺失必须报错）
        train_config: 训练超参
        label_config: 标签配置（落入元数据）
        stage_dates: 各阶段 (start, end) 日期，落入元数据供审计

    Returns:
        TerminalLossTrainResult
    """
    cfg = train_config or TerminalLossTrainConfig()
    lcfg = label_config or TerminalLossLabelConfig()
    if cfg.eval_metric not in SUPPORTED_EVAL_METRICS:
        raise ValueError(
            f"不支持的早停指标 {cfg.eval_metric!r}（可用: {list(SUPPORTED_EVAL_METRICS)}）"
        )
    missing = [c for c in feature_names if c not in train_matrix.columns]
    if missing:
        raise ValueError(f"训练矩阵缺少特征列: {missing}")
    if train_matrix.empty or val_matrix.empty:
        raise ValueError("训练段或早停段(Val)为空，拒绝训练")

    if cfg.objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"不支持的训练目标 {cfg.objective!r}（可用: {list(SUPPORTED_OBJECTIVES)}）"
        )
    allowed_metrics = ALLOWED_METRICS[cfg.objective]
    if cfg.eval_metric not in allowed_metrics:
        raise ValueError(
            f"objective={cfg.objective} 不允许 eval_metric={cfg.eval_metric!r}："
            f"可用 {list(allowed_metrics)}。排序目标的早停必须用排序口径"
            f"（auc / ndcg），二分类目标不能用列表口径 ndcg；逐日 RankIC 只对"
            f"二分类生效（XGBRanker 的 qid eval_set 不兼容逐行 callable）"
        )
    calibrator: Optional[Any] = None
    if cfg.objective == OBJECTIVE_RANK_PAIRWISE:
        clf, calibrator, best_iteration = _fit_rank_pairwise(
            train_matrix, val_matrix, feature_names, cfg
        )
    else:
        X_train = train_matrix[feature_names]
        y_train = train_matrix["loss_label"].astype(int)
        w_train = train_matrix["sample_weight"].astype(float)
        X_val = val_matrix[feature_names]
        y_val = val_matrix["loss_label"].astype(int)

        # 早停指标：rank_ic_daily 用 Val 段的 trade_date 分组（行序必须与 eval_set 一致）
        if cfg.eval_metric == EVAL_METRIC_RANK_IC_DAILY:
            if "trade_date" not in val_matrix.columns:
                raise ValueError("eval_metric=rank_ic_daily 需要 Val 矩阵包含 trade_date 列")
            eval_metric: Any = make_neg_rank_ic_daily(
                val_matrix["trade_date"].astype(str).to_numpy()
            )
            logger.info(
                f"早停指标: 逐日截面 Spearman RankIC 均值（Val 段 "
                f"{val_matrix['trade_date'].nunique()} 个交易日，与门禁 lift 同向）"
            )
        else:
            eval_metric = cfg.eval_metric

        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric=eval_metric,
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
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[(val_matrix["sample_weight"].astype(float),)],
            verbose=False,
        )
        best_iteration = int(getattr(clf, "best_iteration", cfg.n_estimators) or cfg.n_estimators)

    meta: Dict[str, Any] = {
        # best_iteration 必须落盘：WF 汇总（summarize_terminal_risk_wf）与早停健康度
        # 监控都从 sidecar 的 metadata.best_iteration 读取，缺登记即静默为空列
        "best_iteration": best_iteration,
        "n_train": int(len(train_matrix)),
        # n_val = 早停段（Val）行数；评估段（ES）行数 n_es 由调用方按评估矩阵登记
        # （早停段与评估段分离：门禁指标不得来自早停选择段，方案 5.2）
        "n_val": int(len(val_matrix)),
        "train_event_rate": float(train_matrix["loss_label"].mean()),
        "val_event_rate": float(val_matrix["loss_label"].mean()),
        "objective": cfg.objective,
        "calibration": ("isotonic_val" if cfg.objective == OBJECTIVE_RANK_PAIRWISE else "none"),
        "stage_dates": {k: list(v) for k, v in (stage_dates or {}).items()},
        "train_h_distribution": train_matrix["h"].value_counts().sort_index().to_dict(),
        "val_h_distribution": val_matrix["h"].value_counts().sort_index().to_dict(),
    }
    logger.info(
        f"terminal_loss 训练完成: objective={cfg.objective}, train={len(train_matrix)} 行"
        f"(事件率 {train_matrix['loss_label'].mean():.4f}), val={len(val_matrix)} 行"
        f"(事件率 {val_matrix['loss_label'].mean():.4f}), best_iteration={best_iteration}"
        f"（早停只用 Val 段，ES 段仅用于评估）"
    )
    return TerminalLossTrainResult(
        classifier=clf,
        best_iteration=best_iteration,
        train_config=cfg,
        label_config=lcfg,
        feature_names=list(feature_names),
        metadata=meta,
        calibrator=calibrator,
    )


# ── 排序目标训练路径（rank_pairwise）─────────────────────────────


def _qid(dates: pd.Series) -> np.ndarray:
    """trade_date → 连续整数分组号（XGBoost ranking 的 qid，要求已按分组排序）。"""
    _, inv = np.unique(dates.astype(str).to_numpy(), return_inverse=True)
    return inv.astype(int)


def _fit_rank_pairwise(
    train_matrix: pd.DataFrame,
    val_matrix: pd.DataFrame,
    feature_names: List[str],
    cfg: TerminalLossTrainConfig,
) -> Tuple[Any, Any, int]:
    """排序目标：XGBRanker(rank:pairwise, qid=trade_date) + Val isotonic 校准。

    为什么用排序目标（v0.111.0）：决策只需要“当日截面排序”，而池化 logloss
    还会花容量拟合跨日水平——该成分在样本外是 regime 赌注（σ→事件率符号会
    翻），学它既不可靠又挤占排序能力。单折实测 daynorm 1.119→1.334。

    早停口径（v0.112.1）：
        - ``auc``（默认）：**自实现** Val 段池化 ROC AUC 早停
          （``ml/train_core/eval.py::ValPooledAUCStopping``；after_iteration 只
          预测新增的一棵树并累加 margin，逐轮算 sklearn ROC AUC）。**不**把
          ``eval_metric="auc"`` 交给 XGBoost 内置评估：内置 ranking AUC 会对每个
          query group 做 O(n_g²) 成对展开并断言 Σ(n_g+2)(n_g-1)/2 < INT32_MAX，
          6 个月 Val 段（每个交易日一个 group × 每日数千行）实测 29.7 亿 >
          21.47 亿，GPU 后端每折必然 XGBoostError。
        - ``ndcg``：保留内置早停（排序后逐位置计算，无成对展开规模问题），但它在
          Val 上极易饱和——实测有 4/8 折在 ≤63 棵就停（2024H1 仅 4 棵），校准后
          分数大量并列（每日仅 22–240 个不同值 vs binary 的 5800–9325），仅供对照。
        逐行 callable（如逐日 RankIC）在 XGBRanker 下不可用（qid eval_set 传组级数组）。

    排序分数不是概率，因此必须在 **Val 段**（未经 ES）拟合 isotonic 回归映射
    回概率：阈值政策与概率质量报告都依赖概率语义。

    Returns:
        (ranker, calibrator, best_iteration)：best_iteration 为早停口径的停点

    Raises:
        ValueError: train/val 缺 trade_date，或早停指标不可用
    """
    for name, frame in (("train", train_matrix), ("val", val_matrix)):
        if "trade_date" not in frame.columns:
            raise ValueError(
                f"objective={OBJECTIVE_RANK_PAIRWISE} 需要{name}矩阵包含 trade_date"
                f"（qid 分组列），缺列不得静默回退到 binary"
            )
    tr = train_matrix.sort_values("trade_date", kind="stable")
    va = val_matrix.sort_values("trade_date", kind="stable")
    # scale_pos_weight 不透传：排序目标按同日内成对比较，正例权重参数对
    # rank:pairwise 无语义（传入会被静默忽略，不如显式不传）；min_child_weight
    # 仍按正则尺度策略 A 的不变量透传。
    ranker_kwargs: Dict[str, Any] = dict(
        objective="rank:pairwise",
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        random_state=cfg.random_state,
        tree_method="hist",
        device=cfg.device,
    )
    if cfg.eval_metric == EVAL_METRIC_AUC:
        # 自实现池化 AUC 早停（v0.112.1）：逐树 margin 增量 + sklearn ROC AUC，
        # 无 query group 成对展开，绕开 XGBoost 内置 ranking AUC 的 int32 上限。
        dval = DMatrix(va[feature_names], label=va["loss_label"].astype(float))
        stopper = ValPooledAUCStopping(
            dval, va["loss_label"].astype(int), cfg.early_stopping_rounds
        )
        ranker = XGBRanker(callbacks=[stopper], **ranker_kwargs)
        ranker.fit(
            tr[feature_names],
            tr["loss_label"].astype(int),
            qid=_qid(tr["trade_date"]),
            verbose=False,
        )
        best_iteration = stopper.best_iteration
    else:
        # ndcg：列表口径无成对展开规模问题，保留内置早停（带 qid 的 eval_set
        # 不能接逐行 callable，实测报错，故不接受其他自定义指标）。
        ranker = XGBRanker(
            eval_metric=cfg.eval_metric,
            early_stopping_rounds=cfg.early_stopping_rounds,
            **ranker_kwargs,
        )
        ranker.fit(
            tr[feature_names],
            tr["loss_label"].astype(int),
            qid=_qid(tr["trade_date"]),
            eval_set=[(va[feature_names], va["loss_label"].astype(int))],
            eval_qid=[_qid(va["trade_date"])],
            verbose=False,
        )
        best_iteration = int(
            getattr(ranker, "best_iteration", cfg.n_estimators) or cfg.n_estimators
        )
    # 物理裁剪到停点：XGBoost 官方推荐的取停点模型方式（``bst[0:best+1]``）。
    # 必须裁剪：auc 路径没有内置早停，wrapper 默认预测会退回全部树，把停点后
    # 多跑的 patience 棵树也算进去（校准器是在停点分数上拟合的）；裁剪后模型
    # 本体即停点模型，模型层 predict 无需任何 iteration_range 处理。
    # ndcg 路径内置早停未触发时 wrapper best_iteration 缺省为 n_estimators
    # （历史语义：标记“跑满”），必须夹紧到实际最后一棵树再切片，否则越界。
    n_trees = ranker.get_booster().num_boosted_rounds()
    best_iteration = min(best_iteration, n_trees - 1)
    ranker._Booster = ranker.get_booster()[0 : best_iteration + 1]
    if cfg.eval_metric == EVAL_METRIC_AUC:
        # 回调持有 Val DMatrix 与逐轮 margin：清引用，避免序列化携带大数据
        ranker.set_params(callbacks=None)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(ranker.predict(va[feature_names]), va["loss_label"].astype(float))
    logger.info(
        f"排序目标训练完成: rank:pairwise({cfg.eval_metric} 早停={best_iteration})，"
        f"Val 段 isotonic 校准已拟合（{len(va)} 行）"
    )
    return ranker, calibrator, best_iteration


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
                    {
                        "h": int(h),
                        "sigma_bin": "all",
                        "n": int(valid.sum()),
                        "event_rate": float(y_true[m_h][valid].mean()) if valid.any() else np.nan,
                        "mean_pred": float(p_pred[m_h][valid].mean()) if valid.any() else np.nan,
                    }
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
