#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练核心逻辑模块

从 train_ml_model.py 抽取的可复用训练函数，供训练脚本和 walk-forward 脚本共用。

功能：
- 加载特征数据
- 准备训练数据（切分训练集/验证集）
- 标签变换（截面 z-score、分类标签生成）
- 训练 XGBoost 模型
- 验证集逐日评估
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

try:
    import xgboost as xgb
except ImportError:
    logger.error("需要安装 xgboost: pip install xgboost")
    raise

from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml.eval_utils import (
    evaluate_predictions_by_date,
    summarize_daily_metrics,
    compute_diagnostic_statistics,
    print_diagnostic_report
)


def load_features_data(
    storage: Storage,
    loader: DataLoader,
    start_date: str,
    end_date: str
) -> tuple:
    """加载指定日期区间的特征数据
    
    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD
        
    Returns:
        (df, trade_days_count) 元组：合并后的特征 DataFrame 和交易日数量
    """
    logger.info(f"加载特征数据: {start_date} 至 {end_date}")
    
    # 获取交易日列表
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()
    
    trade_dates = trade_cal[
        (trade_cal['cal_date'] >= start_date) & 
        (trade_cal['cal_date'] <= end_date) & 
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    logger.info(f"共 {len(trade_dates)} 个交易日")
    
    # 加载每日特征数据
    all_features = []
    for trade_date in trade_dates:
        features = storage.load_cs_train_day(trade_date)
        if features is not None and len(features) > 0:
            all_features.append(features)
        else:
            logger.warning(f"日期 {trade_date} 没有特征数据")
    
    if not all_features:
        raise ValueError(f"指定日期区间内没有特征数据")
    
    # 合并所有数据
    df = pd.concat(all_features, ignore_index=True)
    logger.info(f"成功加载 {len(df)} 条样本")
    
    return df, len(trade_dates)


def prepare_training_data(df: pd.DataFrame, label_column: str = "y_ret_5", val_ratio: float = 0.2) -> tuple:
    """准备训练数据，并按时间切分训练集和验证集
    
    Args:
        df: 特征 DataFrame
        label_column: 标签列名
        val_ratio: 验证集比例，默认 0.2（最后 20% 的时间作为验证集）
        
    Returns:
        (X_train, y_train, X_val, y_val, feature_columns, df_train_split, df_val_split, data_stats) 元组
        data_stats 包含：samples_after_filter, val_start_date, val_end_date
    """
    logger.info("准备训练数据...")
    
    # 确认标签列存在
    if label_column not in df.columns:
        raise ValueError(f"标签列 {label_column} 不存在")
    
    # 定义需要排除的列（非特征列）
    # 标识列
    id_columns = ['ts_code', 'trade_date', 'name']
    # 标签列
    label_columns = [col for col in df.columns if col.startswith('y_')]
    # 过滤标记列（使用统一的列名，与clean层一致）
    filter_columns = ['is_st', 'is_suspended']
    # 其他非特征列
    other_exclude_columns = ['tradable', 'list_date', 'list_days', 'is_limit_up', 'is_limit_down', 'industry']
    # 临时过滤掉的列
    temp_test_exclude_columns = ['total_mv', 'circ_mv', 'log_circ_mv'] +\
                                ['kdj_k', 'kdj_d'] +\
                                ['bb_upper', 'bb_lower'] +\
                                ['macd_dif', 'macd_dea'] +\
                                ['ps_ttm', 'ep_ttm'] +\
                                ['amount_ma10', 'amount_ma20', 'volume_ratio', 'log_circ_mv', 'net_mf_amount_mean_5', 'net_mf_amount_mean_20', 'vol_burst_10', 'vol_burst_20', 'kdj_d', 'macd_dea', 'bb_upper', 'bb_lower']
    
    exclude_columns = id_columns + label_columns + filter_columns + other_exclude_columns + temp_test_exclude_columns
    
    # 获取特征列
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    logger.info(f"特征列数量: {len(feature_columns)}")
    logger.debug(f"特征列: {feature_columns[:10]}...")  # 只显示前10个
    
    # 过滤可训练样本（移除含有过滤标记的样本）
    mask = pd.Series([True] * len(df))
    for col in filter_columns:
        if col in df.columns:
            mask = mask & (~df[col].astype(bool))
    
    df_train = df[mask].copy()
    logger.info(f"过滤后样本数: {len(df_train)} / {len(df)}")
    samples_after_filter = len(df_train)
    
    # 移除标签为 NaN 的样本
    df_train = df_train.dropna(subset=[label_column])
    logger.info(f"移除标签 NaN 后样本数: {len(df_train)}")
    
    if len(df_train) == 0:
        raise ValueError("没有可用的训练样本")
    
    # 按时间切分训练集和验证集（避免未来信息泄漏）
    df_train = df_train.sort_values('trade_date')
    split_idx = int(len(df_train) * (1 - val_ratio))
    
    df_train_split = df_train.iloc[:split_idx]
    df_val_split = df_train.iloc[split_idx:]
    
    # 获取验证集的时间范围
    val_start_date = df_val_split['trade_date'].min() if len(df_val_split) > 0 else "N/A"
    val_end_date = df_val_split['trade_date'].max() if len(df_val_split) > 0 else "N/A"
    
    logger.info(f"训练集样本数: {len(df_train_split)}, 验证集样本数: {len(df_val_split)}")
    logger.info(f"验证集时间范围: {val_start_date} 至 {val_end_date}")
    
    # 准备训练集 X 和 y
    X_train = df_train_split[feature_columns].copy()
    y_train = df_train_split[label_column].copy()
    
    # 准备验证集 X 和 y
    X_val = df_val_split[feature_columns].copy()
    y_val = df_val_split[label_column].copy()
    
    # 处理特征中的缺失值（填充为0）
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    
    logger.info(f"训练数据准备完成: X_train shape={X_train.shape}, X_val shape={X_val.shape}")
    
    # 数据统计
    data_stats = {
        "samples_after_filter": samples_after_filter,
        "val_start_date": str(val_start_date),
        "val_end_date": str(val_end_date)
    }
    
    return X_train, y_train, X_val, y_val, feature_columns, df_train_split, df_val_split, data_stats


def transform_labels_cs_zscore(
    df: pd.DataFrame,
    label_column: str,
    winsorize_p: float = 0.01
) -> pd.DataFrame:
    """对标签进行截面 winsorize + zscore 变换
    
    仅在训练阶段生效，对每个 trade_date 的原始回归标签进行：
    1. 截面 winsorize（截断极端值）
    2. 截面 zscore（标准化：均值=0，标准差=1）
    
    Args:
        df: 训练数据 DataFrame
        label_column: 标签列名
        winsorize_p: winsorize 参数，默认 0.01（截断上下1%极端值）
        
    Returns:
        变换后的 DataFrame（标签列已替换为标准化后的值）
    """
    from src.lazybull.common.feature_utils import cross_sectional_zscore
    
    logger.info(f"对标签 {label_column} 进行截面 z-score 标准化...")
    logger.info(f"  winsorize 参数: {winsorize_p}")
    
    df_transformed = df.copy()
    
    # 按 trade_date 分组进行截面标准化
    df_transformed[label_column] = cross_sectional_zscore(
        df_transformed,
        value_col=label_column,
        group_col='trade_date',
        winsorize_limits=(winsorize_p, winsorize_p),
        ddof=0
    )
    
    # 统计标准化后的效果
    mean = df_transformed[label_column].mean()
    std = df_transformed[label_column].std()
    logger.info(f"标准化后: 均值={mean:.6f}, 标准差={std:.6f}")
    
    # 检查是否有 NaN（可能由于某天标准差为0）
    nan_count = df_transformed[label_column].isna().sum()
    if nan_count > 0:
        logger.warning(f"标准化后产生 {nan_count} 个 NaN（可能某天标准差为0），将被移除")
        df_transformed = df_transformed.dropna(subset=[label_column])
    
    return df_transformed


def generate_classification_labels(
    df: pd.DataFrame,
    label_column: str,
    pos_quantile: Optional[float] = None,
    pos_topk: Optional[int] = None
) -> pd.DataFrame:
    """生成分类标签（TopN 正类）
    
    按每个交易日截面，将原始标签按分位阈值或数量阈值转为 0/1 标签。
    
    Args:
        df: 训练数据 DataFrame
        label_column: 原始标签列名
        pos_quantile: 百分比阈值（例如 0.2 表示 Top20% 为正类）
        pos_topk: 数量阈值（例如 300 表示每个交易日收益最高的 300 只为正类）
        
    Returns:
        添加了二分类标签的 DataFrame（新增列 {label_column}_binary）
        
    Note:
        pos_quantile 和 pos_topk 二选一，pos_topk 优先级更高
        使用 rank(method='first') 确保 topk 数量严格等于 k（打散并列）
    """
    logger.info(f"生成分类标签（基于 {label_column}）...")
    
    if pos_quantile is None and pos_topk is None:
        raise ValueError("必须指定 pos_quantile 或 pos_topk 之一")
    
    if pos_topk is not None and pos_quantile is not None:
        logger.warning("同时指定了 pos_topk 和 pos_quantile，使用 pos_topk（优先级更高）")
    
    df_labeled = df.copy()
    binary_label_col = f"{label_column}_binary"
    
    # 初始化标签列为 NaN
    df_labeled[binary_label_col] = np.nan
    
    # 按 trade_date 分组，对每组的标签进行排名
    df_labeled['_rank'] = df_labeled.groupby('trade_date')[label_column].rank(
        method='first',
        ascending=False,
        na_option='keep'
    )
    
    if pos_topk is not None:
        # 数量模式：Top K（排名 <= K 为正类）
        df_labeled[binary_label_col] = (df_labeled['_rank'] <= pos_topk).astype(float)
        df_labeled.loc[df_labeled['_rank'].isna(), binary_label_col] = np.nan
    else:
        # 百分比模式：Top X%
        valid_counts = df_labeled.groupby('trade_date')['_rank'].transform('count')
        threshold_ranks = (valid_counts * pos_quantile).clip(lower=1).astype(int)
        df_labeled[binary_label_col] = (df_labeled['_rank'] <= threshold_ranks).astype(float)
        df_labeled.loc[df_labeled['_rank'].isna(), binary_label_col] = np.nan
    
    # 删除临时排名列
    df_labeled = df_labeled.drop(columns=['_rank'])
    
    # 统计正类比例
    total_valid = df_labeled[binary_label_col].notna().sum()
    pos_count = df_labeled[binary_label_col].sum()
    pos_ratio = pos_count / total_valid if total_valid > 0 else 0
    
    logger.info(f"分类标签生成完成:")
    logger.info(f"  模式: {'pos_topk=' + str(pos_topk) if pos_topk else 'pos_quantile=' + str(pos_quantile)}")
    logger.info(f"  正类样本数: {pos_count:.0f} / {total_valid:.0f} ({pos_ratio:.2%})")
    
    if pos_topk is not None:
        pos_counts_per_day = df_labeled.groupby('trade_date')[binary_label_col].sum()
        logger.debug(f"  各交易日正类数量统计: min={pos_counts_per_day.min():.0f}, max={pos_counts_per_day.max():.0f}, mean={pos_counts_per_day.mean():.1f}")
    
    return df_labeled


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str = "regression",
    skip_label_winsorize: bool = False,
    scale_pos_weight: Optional[float] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> tuple:
    """训练 XGBoost 模型（支持回归和分类）
    
    Args:
        task: 任务类型，"regression" 或 "classification"
        skip_label_winsorize: 是否跳过标签 winsorize（当 label_transform=cs_zscore 时为 True）
        scale_pos_weight: 正类权重（分类任务），None 表示自动计算为 neg/pos
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
        
    Returns:
        (model, train_params, train_metrics, val_metrics) 元组
    """
    logger.info(f"开始训练 XGBoost 模型（任务类型: {task}）...")
    
    # 对回归标签进行 winsorize 处理（分类标签不需要，cs_zscore 标签也不需要）
    if task == "regression" and not skip_label_winsorize:
        from scipy.stats import mstats
        y_train_processed = pd.Series(
            mstats.winsorize(y_train, limits=[0.01, 0.01]),
            index=y_train.index
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
                logger.info(f"自动计算 scale_pos_weight: {computed_scale_pos_weight:.4f} (负类={neg_count}, 正类={pos_count})")
            else:
                logger.warning("训练集中无正类样本，无法计算 scale_pos_weight")
                computed_scale_pos_weight = 1.0
        else:
            computed_scale_pos_weight = scale_pos_weight
            logger.info(f"使用用户指定 scale_pos_weight: {computed_scale_pos_weight:.4f} (负类={neg_count}, 正类={pos_count})")
    
    # 准备训练参数
    train_params = {
        "objective": "reg:squarederror" if task == "regression" else "binary:logistic",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "tree_method": "hist",
        "device": "cuda",
        "n_jobs": -1,
        "early_stopping_rounds": 30,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    
    # 分类任务添加 scale_pos_weight
    if task == "classification" and computed_scale_pos_weight is not None:
        train_params["scale_pos_weight"] = computed_scale_pos_weight
    
    logger.info(f"训练参数: {train_params}")
    logger.info("使用早停机制（early_stopping_rounds=30）")
    
    # 创建并训练模型
    if task == "regression":
        model = xgb.XGBRegressor(**train_params)
    else:
        model = xgb.XGBClassifier(**train_params)
    
    # 如果有验证集，使用早停机制
    if len(X_val) > 0:
        model.fit(
            X_train, y_train_processed,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        logger.info(f"模型训练完成（最佳迭代: {model.best_iteration}）")
    else:
        model.fit(X_train, y_train_processed)
        logger.info("模型训练完成（无验证集，未使用早停）")

    # 计算训练集性能指标
    if task == "regression":
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = train_mse ** 0.5
        train_r2 = r2_score(y_train, y_train_pred)
        train_ic = y_train.corr(pd.Series(y_train_pred, index=y_train.index))
        
        train_metrics = {
            "mse": float(train_mse),
            "rmse": float(train_rmse),
            "r2": float(train_r2),
            "ic": float(train_ic)
        }
        
        logger.info(f"训练集性能: MSE={train_mse:.6f}, RMSE={train_rmse:.6f}, R2={train_r2:.4f}, IC={train_ic:.4f}")
    else:
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
        
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
            "recall": float(train_recall)
        }
        
        logger.info(f"训练集性能: ACC={train_acc:.4f}, AUC={train_auc:.4f}, Precision={train_precision:.4f}, Recall={train_recall:.4f}")
    
    # 计算验证集性能指标
    if len(X_val) > 0:
        if task == "regression":
            y_val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_rmse = val_mse ** 0.5
            val_r2 = r2_score(y_val, y_val_pred)
            val_ic = y_val.corr(pd.Series(y_val_pred, index=y_val.index))
            val_rank_ic, _ = spearmanr(y_val, y_val_pred)
            
            val_metrics = {
                "mse": float(val_mse),
                "rmse": float(val_rmse),
                "r2": float(val_r2),
                "ic": float(val_ic),
                "rank_ic": float(val_rank_ic)
            }
            
            logger.info("=" * 60)
            logger.info("验证集评估结果（回归任务）")
            logger.info("=" * 60)
            logger.info(f"验证集样本数: {len(X_val)}")
            logger.info(f"MSE（均方误差）: {val_mse:.6f}")
            logger.info(f"RMSE（均方根误差）: {val_rmse:.6f}")
            logger.info(f"R2（决定系数）: {val_r2:.4f}")
            logger.info(f"IC（信息系数）: {val_ic:.4f}  <- 重要指标")
            logger.info(f"RankIC（排序IC）: {val_rank_ic:.4f}  <- 选股策略关键指标")
            logger.info("=" * 60)
        else:
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
            
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
                "recall": float(val_recall)
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
    if len(X_val) > 0 and hasattr(model, 'best_iteration'):
        train_params["best_iteration"] = int(model.best_iteration)
    
    return model, train_params, train_metrics, val_metrics


def evaluate_validation_daily(
    model,
    df_val: pd.DataFrame,
    feature_columns: List[str],
    original_return_col: str,
    task: str,
    topk_values: Optional[List[int]] = None
) -> Dict:
    """对验证集进行逐日评估（贴近交易场景）
    
    Args:
        model: 训练好的模型
        df_val: 验证集 DataFrame（包含 trade_date, ts_code, 特征列, 原始收益列）
        feature_columns: 特征列名列表
        original_return_col: 原始真实收益列名（如 y_ret_20）
        task: 任务类型
        topk_values: TopK 评估的 K 值列表
        
    Returns:
        逐日评估结果字典
    """
    if len(df_val) == 0:
        logger.warning("验证集为空，跳过逐日评估")
        return {}
    
    if original_return_col not in df_val.columns:
        logger.warning(f"验证集缺少原始收益列 {original_return_col}，跳过逐日评估")
        return {}
    
    if topk_values is None:
        topk_values = [30, 100, 300]
    
    logger.info("=" * 60)
    logger.info("验证集逐日评估（贴近交易场景）")
    logger.info("=" * 60)
    
    # 准备预测数据
    df_eval = df_val.copy()
    X_val_features = df_val[feature_columns].fillna(0)
    
    # 预测
    if task == "classification":
        y_pred_proba = model.predict_proba(X_val_features)[:, 1]
        df_eval['pred_score'] = y_pred_proba
    else:
        y_pred = model.predict(X_val_features)
        df_eval['pred_score'] = y_pred
    
    # 逐日评估
    daily_results = evaluate_predictions_by_date(
        df=df_eval,
        date_col='trade_date',
        prediction_col='pred_score',
        return_col=original_return_col,
        topk_values=topk_values
    )
    
    # 汇总统计
    summary = summarize_daily_metrics(daily_results)
    
    # 输出结果
    logger.info(f"评估天数: {len(daily_results)}")
    logger.info(f"逐日 RankIC 均值: {summary.get('RankIC_均值', np.nan):.4f}")
    logger.info(f"逐日 RankIC 标准差: {summary.get('RankIC_标准差', np.nan):.4f}")
    logger.info(f"逐日 RankIC IR: {summary.get('RankIC_IR', np.nan):.4f}")
    
    for k in topk_values:
        mean_key = f"Top{k}平均收益_均值"
        std_key = f"Top{k}平均收益_标准差"
        if mean_key in summary:
            logger.info(f"Top{k} 平均收益（跨日）: 均值={summary[mean_key]:.4f}, 标准差={summary[std_key]:.4f}")
    
    logger.info("=" * 60)
    
    # 计算并打印诊断统计
    diagnostics = compute_diagnostic_statistics(
        df=df_eval,
        date_col='trade_date',
        prediction_col='pred_score',
        return_col=original_return_col,
        topk_values=topk_values
    )
    
    print_diagnostic_report(diagnostics)
    
    # 返回汇总结果（包含诊断统计）
    result = {
        'daily_rankic_mean': summary.get('RankIC_均值', np.nan),
        'daily_rankic_std': summary.get('RankIC_标准差', np.nan),
        'daily_rankic_ir': summary.get('RankIC_IR', np.nan),
        **{f'top{k}_return_mean': summary.get(f"Top{k}平均收益_均值", np.nan) for k in topk_values},
        **{f'top{k}_return_std': summary.get(f"Top{k}平均收益_标准差", np.nan) for k in topk_values}
    }
    
    # 添加诊断统计
    result.update({f'diagnostic_{k}': v for k, v in diagnostics.items()})
    
    return result
