# -*- coding: utf-8 -*-
"""xgb：train_core 拆分模块。"""

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
import numpy as np
import os
import pandas as pd
import xgboost as xgb

from .features import _format_feature_importance_compact
from .eval import make_neg_rank_ic_daily
from .eval import neg_rank_ic

def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str = "regression",
    skip_label_winsorize: bool = False,
    scale_pos_weight: Optional[float] = None,
    sample_weight: Optional[np.ndarray] = None,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    min_child_weight: int = 100,
    reg_alpha: float = 0.05,
    reg_lambda: float = 1.0,
    gamma: float = 0.1,
    objective_type: str = "mse",
    df_train_for_group: Optional[pd.DataFrame] = None,
    df_val_for_group: Optional[pd.DataFrame] = None,
    early_stopping_rounds: Optional[int] = 200,
    early_stopping_metric: str = "auto",
    min_best_iteration: int = 0,
) -> tuple:
    """训练 XGBoost 模型（支持回归、分类和排序学习）

    Args:
        task: 任务类型，"regression" 或 "classification"
        skip_label_winsorize: 是否跳过标签 winsorize（当 label_transform=cs_zscore 时为 True）
        scale_pos_weight: 正类权重（分类任务），None 表示自动计算为 neg/pos
        sample_weight: 样本权重数组（可选），用于 Top/Bottom K 强化训练精度，
                       由 build_rank_sample_weights() 生成；None 表示不使用样本权重
        X_train: 训练特征数据
        y_train: 训练标签数据
        X_val: 验证特征数据
        y_val: 验证标签数据
        n_estimators: 树的数量
        max_depth: 树的最大深度
        learning_rate: 学习率
        subsample: 样本采样比例
        colsample_bytree: 特征采样比例
        random_state: 随机种子
        min_child_weight: 叶节点最少样本权重和，防止过拟合，默认 100（金融数据建议 100-500）
        reg_alpha: L1 正则化系数，默认 0.05
        reg_lambda: L2 正则化系数，默认 1.0
        gamma: 节点分裂最小损失下降，默认 0.1
        objective_type: 目标函数类型，"mse"（回归，默认）或 "lambdarank"（排序学习，
                        直接优化股票排序而非预测收益绝对值，与 RankIC 评估指标对齐）
        df_train_for_group: 训练集 DataFrame（仅 lambdarank 需要，用于提取 trade_date 分组信息）
        df_val_for_group: 验证集 DataFrame（lambdarank 与 rank_ic_daily 早停指标需要，
                        用于提取 trade_date 分组信息，行序必须与 X_val 一致）
        min_best_iteration: best_iteration 下限监控阈值，默认 0（禁用）。
                        启用早停且 best_iteration 低于该值时仅告警并在 train_params 中
                        标记 best_iteration_floor_triggered，不改变模型行为（早停验证段
                        被极端事件主导时 best_iteration 会异常小，属诊断信号）

    Returns:
        (model, train_params, train_metrics, val_metrics) 元组
    """
    logger.info(f"开始训练 XGBoost 模型（任务类型: {task}）...")

    # 对回归标签进行 winsorize 处理（分类标签不需要，cs_zscore 标签也不需要）
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

    # 判断是否使用 LambdaRank（排序学习）
    use_lambdarank = task == "regression" and objective_type == "lambdarank"

    if use_lambdarank:
        if df_train_for_group is None:
            raise ValueError(
                "lambdarank 目标需要 df_train_for_group 参数（用于按 trade_date 分组）"
            )
        logger.info("使用 LambdaRank 排序学习目标（直接优化股票排序，与 RankIC 评估对齐）")

    # 统一确定 eval_metric：
    # - regression + rank_ic: 整段 Spearman 指标（可用于早停）
    # - regression + rank_ic_daily: 逐日截面 Spearman 均值指标（与 daily_rankic 评估口径一致，
    #   避免整段指标被单一事件期样本主导）
    # - 其余 regression: mae
    # - classification: auc
    if early_stopping_metric == "rank_ic_daily" and task == "regression" and len(X_val) > 0:
        if (
            df_val_for_group is None
            or len(df_val_for_group) != len(X_val)
            or "trade_date" not in df_val_for_group.columns
        ):
            raise ValueError(
                "early_stopping_metric=rank_ic_daily 需要 df_val_for_group 为与 X_val 行序一致"
                f"且包含 trade_date 列的 DataFrame（实际: "
                f"{'None' if df_val_for_group is None else f'len={len(df_val_for_group)}'}"
                f" vs X_val len={len(X_val)}）"
            )
        val_dates_for_metric = df_val_for_group["trade_date"].values
        xgb_eval_metric = make_neg_rank_ic_daily(val_dates_for_metric)
        logger.info(
            f"早停指标: 逐日截面 Spearman RankIC 均值（共 {len(set(val_dates_for_metric))} 个交易日，"
            "与 daily_rankic 评估口径一致）"
        )
    elif early_stopping_metric == "rank_ic_daily" and task == "regression":
        # 无验证集时早停后续会被禁用，metric 降级为 mae 保持既有语义
        xgb_eval_metric = "mae"
        logger.warning("rank_ic_daily 早停指标在无验证集时降级为 mae（早停将被禁用）")
    elif early_stopping_metric == "rank_ic" and task == "regression":
        xgb_eval_metric = neg_rank_ic
    else:
        xgb_eval_metric = "mae" if task == "regression" else "auc"

    # xgboost 在 sklearn 包装器 + callable eval_metric 路径下，
    # 会将 n_jobs 直接传给 ThreadPoolExecutor。这里确保 max_workers>0。
    def _resolve_xgb_n_jobs(n_jobs: int) -> int:
        cpu_count = os.cpu_count() or 1
        if n_jobs is None:
            return cpu_count
        try:
            n_jobs_int = int(n_jobs)
        except (TypeError, ValueError):
            return cpu_count
        if n_jobs_int == 0:
            return cpu_count
        if n_jobs_int < 0:
            # sklearn 语义：-1 表示使用全部 CPU，-2 表示保留 1 个核心，依此类推。
            return max(cpu_count + 1 + n_jobs_int, 1)
        return max(n_jobs_int, 1)

    resolved_n_jobs = _resolve_xgb_n_jobs(-1)

    # 准备训练参数
    if use_lambdarank:
        train_params = {
            "objective": "rank:pairwise",
            # 保持与早停配置一致：当用户指定 rank_ic 时，不强制改用 ndcg。
            "eval_metric": xgb_eval_metric,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": resolved_n_jobs,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "ndcg_exp_gain": False,          # ← 加这一行
        }
    else:
        train_params = {
            "objective": "reg:squarederror" if task == "regression" else "binary:logistic",
            "eval_metric": xgb_eval_metric,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": resolved_n_jobs,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
        }

    # 早停设置：early_stopping_rounds=None 或 0 表示禁用早停，使用固定 n_estimators
    if early_stopping_rounds:
        train_params["early_stopping_rounds"] = early_stopping_rounds

    # 分类任务添加 scale_pos_weight
    if task == "classification" and computed_scale_pos_weight is not None:
        train_params["scale_pos_weight"] = computed_scale_pos_weight

    # 日志中显示可读的 eval_metric 名称（callable 替换为字符串）
    log_params = {k: (v.__name__ if callable(v) else v) for k, v in train_params.items()}
    logger.info(f"训练参数: {log_params}")
    if early_stopping_rounds:
        es_metric_display = (
            early_stopping_metric
            if early_stopping_metric != "auto"
            else log_params.get("eval_metric", "mae")
        )
        logger.info(f"使用早停机制（rounds={early_stopping_rounds}, metric={es_metric_display}）")
    else:
        logger.info(f"未使用早停机制，固定训练 {n_estimators} 棵树")

    # LambdaRank 需要构造 qid（query group ID），每个 trade_date 为一个 query group
    # 同时需要将连续收益率标签转换为非负整数等级（XGBoost rank 要求）
    qid_train = None
    qid_val = None
    if use_lambdarank:
        # 按 trade_date 排序并构造 qid（同一天的股票属于同一组，组内进行排序优化）
        train_dates = df_train_for_group["trade_date"].values
        val_dates = (
            df_val_for_group["trade_date"].values
            if df_val_for_group is not None and len(df_val_for_group) > 0
            else np.array([])
        )

        # qid: 将日期映射为整数 group id
        unique_train_dates = sorted(set(train_dates))
        date_to_qid = {d: i for i, d in enumerate(unique_train_dates)}
        qid_train = np.array([date_to_qid[d] for d in train_dates])

        if len(val_dates) > 0:
            offset = len(unique_train_dates)
            unique_val_dates = sorted(set(val_dates))
            val_date_to_qid = {d: i + offset for i, d in enumerate(unique_val_dates)}
            qid_val = np.array([val_date_to_qid[d] for d in val_dates])

        logger.info(
            f"LambdaRank 分组: 训练集 {len(unique_train_dates)} 个交易日, "
            f"验证集 {len(unique_val_dates) if len(val_dates) > 0 else 0} 个交易日"
        )

        # 将连续收益率转换为按日截面排名等级 (0~31)
        # XGBoost rank:pairwise + NDCG 指数增益要求标签 <= 31
        # ~3000 只股票 / 32 级 ≈ 每级 ~94 只，粒度足够保留排序信息
        max_grade = 255 #31

        def _returns_to_grades(y: pd.Series, dates: np.ndarray) -> pd.Series:
            """按每日截面将连续收益率转为 0~max_grade 的整数等级"""
            grades = pd.Series(0, index=y.index, dtype=int)
            for date in sorted(set(dates)):
                mask = dates == date
                daily_y = y[mask]
                n = len(daily_y)
                if n <= 1:
                    grades[mask] = 0
                else:
                    pct_rank = daily_y.rank(method="average") / n  # (0, 1]
                    grades[mask] = (pct_rank * max_grade).clip(0, max_grade).astype(int)
            return grades

        y_train_processed = _returns_to_grades(y_train_processed, train_dates)
        logger.info(f"LambdaRank 标签转换完成: 连续收益 → 排名等级 (0~{max_grade})")
        logger.info(
            f"  等级范围: {y_train_processed.min()} ~ {y_train_processed.max()}, "
            f"均值: {y_train_processed.mean():.1f}"
        )

        # 验证集标签也需要转换
        if len(val_dates) > 0:
            y_val = _returns_to_grades(y_val, val_dates)

    # xgboost >= 2.1：构造函数携带 early_stopping_rounds 而 fit 未提供 eval_set 时，
    # 会自动从训练集切出 20% 作为早停验证集，导致实际训练样本悄然减少。
    # 无验证集时显式移除早停参数，保持"无验证集 = 固定 n_estimators"的既有语义。
    if train_params.get("early_stopping_rounds") and len(X_val) == 0:
        train_params.pop("early_stopping_rounds", None)
        logger.info("无验证集，已禁用早停（避免 xgboost 自动切分训练数据）")

    # 创建并训练模型
    if use_lambdarank:
        model = xgb.XGBRanker(**train_params)
    elif task == "regression":
        model = xgb.XGBRegressor(**train_params)
    else:
        model = xgb.XGBClassifier(**train_params)

    # 如果有验证集，使用早停机制
    if len(X_val) > 0:
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }
        if use_lambdarank:
            fit_kwargs["qid"] = qid_train
            if qid_val is not None:
                fit_kwargs["eval_qid"] = [qid_val]
        else:
            fit_kwargs["sample_weight"] = sample_weight

        model.fit(X_train, y_train_processed, **fit_kwargs)
        if early_stopping_rounds:
            logger.warning(f"模型训练完成（最佳迭代: {model.best_iteration}）")
        else:
            logger.info(f"模型训练完成（固定 {n_estimators} 棵树）")
    else:
        fit_kwargs = {"verbose": False}
        if use_lambdarank:
            fit_kwargs["qid"] = qid_train
        else:
            fit_kwargs["sample_weight"] = sample_weight

        model.fit(X_train, y_train_processed, **fit_kwargs)
        logger.info("模型训练完成（无验证集，未使用早停）")

    importance = model.feature_importances_
    feature_names = X_train.columns
    feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    logger.info(f"模型特征重要性（共 {len(feat_imp)} 个）:\n{_format_feature_importance_compact(feat_imp)}")

    # 计算训练集性能指标
    # 使用 y_train_processed（winsorize 后）与预测值比较，保持与训练目标一致
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
    # 注意：当使用 label_transform=cs_zscore 时，y_val 是截面 z-score 标准化后的标签（均值≈0，标准差≈1），
    #       val_mse/val_ic 等指标均在 z-score 空间计算，与 train_metrics（使用 y_train_processed，
    #       同样是处理后的标签）可比；但与真实收益单位的 val 逐日评估结果不可直接比较。
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
            logger.info(
                f"IC（信息系数）: {val_ic:.4f}  <- 重要指标（cs_zscore 模式下为 z-score 空间）"
            )
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
    # best_iteration 下限监控：早停验证段被极端事件主导时 best_iteration 会异常小，
    # 此处只告警与标记，不改变模型行为（回退轮数会改变全部下游对比语义，由调用方显式决策）。
    best_iteration_floor_triggered = False
    if len(X_val) > 0 and hasattr(model, "best_iteration"):
        best_it = int(model.best_iteration)
        train_params["best_iteration"] = best_it
        if early_stopping_rounds and min_best_iteration > 0 and best_it < min_best_iteration:
            best_iteration_floor_triggered = True
            val_es_range = "N/A"
            if df_val_for_group is not None and "trade_date" in df_val_for_group.columns:
                val_dates_series = df_val_for_group["trade_date"]
                val_es_range = f"{val_dates_series.min()} ~ {val_dates_series.max()}"
            logger.warning(
                f"best_iteration={best_it} 低于下限 {min_best_iteration}，"
                f"早停验证段 {val_es_range} 可能被极端事件主导或信号过弱，"
                "建议检查该窗口的 val_es 日期区间与市场事件重叠情况"
            )
    train_params["best_iteration_floor_triggered"] = best_iteration_floor_triggered
    train_params["early_stopping_metric"] = early_stopping_metric
    # 确保 eval_metric 可序列化（callable 替换为函数名）
    if callable(train_params.get("eval_metric")):
        train_params["eval_metric"] = train_params["eval_metric"].__name__

    return model, train_params, train_metrics, val_metrics
