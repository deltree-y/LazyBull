# -*- coding: utf-8 -*-
"""lgb：train_core 拆分模块。"""

from loguru import logger
from scipy.stats import mstats
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from typing import Optional
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import _format_feature_importance_compact
from .eval import _rank_ic_eval_lgb

def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str = "regression",
    skip_label_winsorize: bool = False,
    scale_pos_weight: Optional[float] = None,
    sample_weight: Optional[np.ndarray] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    num_leaves: Optional[int] = None,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    min_child_weight: int = 20,
    reg_alpha: float = 0.05,
    reg_lambda: float = 1.0,
    gamma: float = 0.1,
    early_stopping_rounds: Optional[int] = 200,
    early_stopping_metric: str = "auto",
) -> tuple:
    """训练 LightGBM 模型（支持回归和分类）

    参数签名与 train_xgboost_model() 基本一致，便于在训练脚本中通过 --algorithm 切换。
    gamma 参数映射为 LightGBM 的 min_split_gain。
    num_leaves 为 LightGBM 独有参数，不指定时默认 31。

    Returns:
        (model, train_params, train_metrics, val_metrics) 元组
    """
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError:
        logger.error("需要安装 lightgbm: pip install lightgbm")
        raise

    logger.info(f"开始训练 LightGBM 模型（任务类型: {task}）...")

    # 对回归标签进行 winsorize 处理
    if task == "regression" and not skip_label_winsorize:
        from scipy.stats import mstats

        y_train_processed = pd.Series(
            mstats.winsorize(y_train, limits=[0.01, 0.01]), index=y_train.index
        )
        logger.info("对回归标签进行 winsorize 处理（截断上下1%极端值），用于稳定训练")
    else:
        y_train_processed = y_train
        if task == "classification":
            logger.info("分类任务，跳过标签 winsorize 处理")
        elif skip_label_winsorize:
            logger.info("标签已在 cs_zscore 步骤中 winsorize，训练阶段跳过 winsorize")

    # 计算 scale_pos_weight（分类任务）
    computed_scale_pos_weight = None
    if task == "classification":
        pos_count = (y_train_processed == 1).sum()
        neg_count = (y_train_processed == 0).sum()

        if scale_pos_weight is None:
            if pos_count > 0:
                computed_scale_pos_weight = neg_count / pos_count
                logger.info(
                    f"自动计算 scale_pos_weight: {computed_scale_pos_weight:.4f} (负类={neg_count}, 正类={pos_count})"
                )
            else:
                logger.warning("训练集中无正类样本，无法计算 scale_pos_weight")
                computed_scale_pos_weight = 1.0
        else:
            computed_scale_pos_weight = scale_pos_weight
            logger.info(
                f"使用用户指定 scale_pos_weight: {computed_scale_pos_weight:.4f} (负类={neg_count}, 正类={pos_count})"
            )

    if sample_weight is not None:
        logger.info(f"使用样本权重（rank-weight），加权样本数={int((sample_weight > 1.0).sum())}")
    else:
        logger.info("未使用样本权重（rank-weight 未启用）")

    # num_leaves：LightGBM 的核心复杂度控制器
    # 如果未显式指定，使用默认值 31（LightGBM 官方默认值，适度复杂度）
    if num_leaves is None:
        num_leaves = 31

    # 准备训练参数（映射 XGBoost 参数到 LightGBM）
    train_params = {
        "objective": "regression" if task == "regression" else "binary",
        "metric": "mae" if task == "regression" else "auc",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "n_jobs": 10,
        "min_split_gain": gamma,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "min_child_weight": min_child_weight,
        "verbosity": -1,
    }

    if early_stopping_rounds:
        train_params["early_stopping_rounds"] = early_stopping_rounds

    logger.info(f"训练参数: {train_params}")

    # 创建模型
    if task == "regression":
        model = lgb.LGBMRegressor(**train_params)
    else:
        model = lgb.LGBMClassifier(**train_params)

    # 训练（LightGBM 使用 callbacks 进行早停）
    if early_stopping_rounds and len(X_val) > 0:
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=0),  # 静默
        ]
        es_metric_display = (
            early_stopping_metric
            if early_stopping_metric != "auto"
            else train_params.get("metric", "mae")
        )
        logger.info(f"使用早停机制（rounds={early_stopping_rounds}, metric={es_metric_display}）")

        fit_kwargs = {
            "sample_weight": sample_weight,
            "eval_set": [(X_val, y_val)],
            "callbacks": callbacks,
        }
        # 自定义早停 eval metric（rank_ic 替代 mae，尺度无关更稳定）
        if early_stopping_metric == "rank_ic" and task == "regression":
            fit_kwargs["eval_metric"] = _rank_ic_eval_lgb

        model.fit(X_train, y_train_processed, **fit_kwargs)
        logger.warning(f"模型训练完成（最佳迭代: {model.best_iteration_}）")
    elif len(X_val) > 0:
        callbacks = [lgb.log_evaluation(period=0)]
        logger.info(f"未使用早停机制，固定训练 {n_estimators} 棵树")
        model.fit(
            X_train,
            y_train_processed,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        logger.info(f"模型训练完成（固定 {n_estimators} 棵树）")
    else:
        logger.info(f"未使用早停机制，固定训练 {n_estimators} 棵树")
        model.fit(
            X_train,
            y_train_processed,
            sample_weight=sample_weight,
        )
        logger.info("模型训练完成（无验证集）")

    importance = model.feature_importances_
    feature_names = X_train.columns
    feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    logger.info(f"模型 Top-20 特征重要性:\n{_format_feature_importance_compact(feat_imp.head(20))}")

    # 计算训练集性能指标
    if task == "regression":
        y_train_pred = model.predict(X_train)
        y_train_eval = pd.Series(y_train_processed, index=y_train.index)
        train_mse = mean_squared_error(y_train_eval, y_train_pred)
        train_rmse = train_mse**0.5
        train_r2 = r2_score(y_train_eval, y_train_pred)
        train_ic = y_train_eval.corr(pd.Series(y_train_pred, index=y_train.index))

        train_metrics = {
            "mse": float(train_mse),
            "rmse": float(train_rmse),
            "r2": float(train_r2),
            "ic": float(train_ic),
        }

        logger.info(
            f"训练集性能: MSE={train_mse:.6f}, RMSE={train_rmse:.6f}, R2={train_r2:.4f}, IC={train_ic:.4f}"
        )
    else:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred_binary = model.predict(X_train)

        train_acc = accuracy_score(y_train, y_train_pred_binary)
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        train_precision = precision_score(y_train, y_train_pred_binary)
        train_recall = recall_score(y_train, y_train_pred_binary)

        train_metrics = {
            "accuracy": float(train_acc),
            "auc": float(train_auc),
            "precision": float(train_precision),
            "recall": float(train_recall),
        }

        logger.info(
            f"训练集性能: ACC={train_acc:.4f}, AUC={train_auc:.4f}, Precision={train_precision:.4f}, Recall={train_recall:.4f}"
        )

    # 计算验证集性能指标
    if len(X_val) > 0:
        if task == "regression":
            y_val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_rmse = val_mse**0.5
            val_r2 = r2_score(y_val, y_val_pred)
            val_ic = y_val.corr(pd.Series(y_val_pred, index=y_val.index))
            val_rank_ic, _ = spearmanr(y_val, y_val_pred)

            val_metrics = {
                "mse": float(val_mse),
                "rmse": float(val_rmse),
                "r2": float(val_r2),
                "ic": float(val_ic),
                "rank_ic": float(val_rank_ic),
            }

            logger.info("=" * 60)
            logger.info("验证集评估结果（回归任务）")
            logger.info("=" * 60)
            logger.info(f"验证集样本数: {len(X_val)}")
            logger.info(f"MSE（均方误差）: {val_mse:.6f}")
            logger.info(f"RMSE（均方根误差）: {val_rmse:.6f}")
            logger.info(f"R2（决定系数）: {val_r2:.4f}")
            logger.info(f"IC（信息系数）: {val_ic:.4f}")
            logger.info(f"RankIC（排序IC）: {val_rank_ic:.4f}  <- 选股策略关键指标")
            logger.info("=" * 60)
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

            y_val_pred_proba = model.predict_proba(X_val)[:, 1]
            y_val_pred_binary = model.predict(X_val)

            val_acc = accuracy_score(y_val, y_val_pred_binary)
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            val_precision = precision_score(y_val, y_val_pred_binary)
            val_recall = recall_score(y_val, y_val_pred_binary)

            val_metrics = {
                "accuracy": float(val_acc),
                "auc": float(val_auc),
                "precision": float(val_precision),
                "recall": float(val_recall),
            }

            logger.info("=" * 60)
            logger.info("验证集评估结果（分类任务）")
            logger.info("=" * 60)
            logger.info(f"验证集样本数: {len(X_val)}")
            logger.info(f"Accuracy（准确率）: {val_acc:.4f}")
            logger.info(f"AUC（ROC曲线下面积）: {val_auc:.4f}  <- 重要指标")
            logger.info(f"Precision（精确率）: {val_precision:.4f}")
            logger.info(f"Recall（召回率）: {val_recall:.4f}")
            logger.info("=" * 60)
    else:
        val_metrics = {}
        logger.warning("验证集为空，无法评估")

    # 添加 best_iteration 到 train_params
    if len(X_val) > 0 and hasattr(model, "best_iteration_"):
        train_params["best_iteration"] = int(model.best_iteration_)
    train_params["early_stopping_metric"] = early_stopping_metric

    return model, train_params, train_metrics, val_metrics
