#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walk-forward 滚动训练脚本

功能：
- 按指定频率（季度/月度/半年度）滚动训练模型
- 每次训练使用固定窗口的历史数据（默认5年）
- 在紧随其后的样本外窗口进行评估（默认6个月）
- 生成多段 OOS 表现序列
- 复用 train_ml_model.py 的所有能力（训练、评估、模型注册、日志记录等）

使用示例：
    # 使用默认参数（季度滚动，5年训练窗口，6个月测试窗口）
    python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231
    
    # 指定按月度滚动
    python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 --step monthly
    
    # 自定义窗口大小
    python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 \
        --train-window-years 3 --test-window-months 3
    
    # 透传训练参数
    python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 \
        --task classification --pos-topk 300 --label y_ret_20
"""

import argparse
import sys
import traceback
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml import ModelRegistry
from src.lazybull.ml.train_core import (
    load_features_data,
    prepare_training_data,
    transform_labels_cs_zscore,
    generate_classification_labels,
    train_xgboost_model,
    train_lightgbm_model,
    evaluate_validation_daily,
    build_rank_sample_weights,
)
from src.lazybull.ml.walk_forward_utils import (
    generate_walk_forward_splits,
    print_splits_summary,
    WalkForwardSplit
)
from src.lazybull.ml.run_logger import (
    TrainingRunRecord,
    write_training_run_to_csv
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")
# test 期延伸到数据末尾时，标签列（如 y_ret_20）在最近 N 个交易日全为 NaN，concat 时触发此警告
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame concatenation with empty or all-NA entries.*")


def execute_split_training(
    split: WalkForwardSplit,
    wf_run_id: str,
    storage: Storage,
    loader: DataLoader,
    registry: ModelRegistry,
    args,
    topk_values: List[int]
) -> Dict:
    """执行单个 split 的训练
    
    Args:
        split: WalkForwardSplit 对象
        wf_run_id: walk-forward 运行ID
        storage: Storage 实例
        loader: DataLoader 实例
        registry: ModelRegistry 实例
        args: 命令行参数
        topk_values: TopK 评估值列表
        
    Returns:
        包含训练结果的字典
    """
    logger.info("=" * 80)
    logger.info(f"开始训练 Split {split.split_index}")
    logger.info(f"  训练区间: {split.train_start} 至 {split.train_end}")
    logger.info(f"  测试区间: {split.test_start} 至 {split.test_end}")
    logger.info("=" * 80)
    
    # 1. 加载训练数据
    df_train, train_days_count = load_features_data(
        storage, loader, split.train_start, split.train_end
    )
    total_train_samples = len(df_train)
    
    # 2. 加载测试数据
    df_test, test_days_count = load_features_data(
        storage, loader, split.test_start, split.test_end
    )
    total_test_samples = len(df_test)
    
    # 3. 应用标签变换
    if args.task == "classification":
        if args.pos_quantile is None and args.pos_topk is None:
            raise ValueError("分类任务必须指定 --pos-quantile 或 --pos-topk")
        
        df_train = generate_classification_labels(
            df_train,
            label_column=args.label_column,
            pos_quantile=args.pos_quantile,
            pos_topk=args.pos_topk
        )
        binary_label_col = f"{args.label_column}_binary"
        actual_label_column = binary_label_col
    else:
        # 回归任务：标签变换（cs_zscore 在切分后各自独立变换，避免数据泄露）
        actual_label_column = args.label_column

    # 4. 准备训练数据（按 trade_date 粒度切分训练集/验证集）
    # cs_zscore 回归任务：切分后对 train/val 各自独立变换，不共享截面统计量
    label_transform_fn = None
    if args.task == "regression" and args.label_transform == "cs_zscore":
        label_transform_fn = lambda d: transform_labels_cs_zscore(
            d, label_column=actual_label_column, winsorize_p=args.winsorize_p
        )
    X_train, y_train, X_val, y_val, feature_columns, df_train_split, df_val_split, data_stats, df_val_split_original = prepare_training_data(
        df_train, actual_label_column, val_ratio=args.val_ratio, label_transform_fn=label_transform_fn,
        enable_fundamental_features=args.enable_fundamental_features,
    )

    # 4.1. 构造样本权重（rank-weight：Top/Bottom K 增强）
    rank_sample_weight = None
    if args.rank_weight_enabled:
        rank_sample_weight = build_rank_sample_weights(
            df_train=df_train_split,
            label_column=actual_label_column,
            topk=args.rank_weight_topk,
            top_weight=args.rank_weight,
        )

    # 5. 训练模型
    skip_label_winsorize = (args.task == "regression" and args.label_transform == "cs_zscore")

    # 根据算法选择训练函数
    algorithm = getattr(args, 'algorithm', 'xgboost')
    train_fn = train_lightgbm_model if algorithm == "lightgbm" else train_xgboost_model

    # num_leaves 仅 LightGBM 使用
    extra_kwargs = {}
    if algorithm == "lightgbm":
        num_leaves_val = getattr(args, 'num_leaves', None)
        if num_leaves_val is not None:
            extra_kwargs["num_leaves"] = num_leaves_val

    # LambdaRank 需要传入 DataFrame 用于按 trade_date 构造 qid 分组
    if algorithm == "xgboost":
        objective_type = getattr(args, 'objective', 'mse')
        extra_kwargs["objective_type"] = objective_type
        if objective_type == "lambdarank":
            extra_kwargs["df_train_for_group"] = df_train_split
            extra_kwargs["df_val_for_group"] = df_val_split

    model, train_params, train_metrics, val_metrics = train_fn(
        X_train, y_train, X_val, y_val,
        task=args.task,
        skip_label_winsorize=skip_label_winsorize,
        scale_pos_weight=args.scale_pos_weight,
        sample_weight=rank_sample_weight,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        **extra_kwargs,
    )
    
    # 6. 验证集逐日评估（用于内部评估）
    # 使用变换前的原始 val df（df_val_split_original），确保 TopK 收益以真实收益单位展示，
    # 避免 cs_zscore 场景下 z-score 值（0~3）被误读为百分比收益
    val_daily_metrics = {}
    if len(df_val_split_original) > 0:
        original_return_col = args.label_column
        val_daily_metrics = evaluate_validation_daily(
            model=model,
            df_val=df_val_split_original,
            feature_columns=feature_columns,
            original_return_col=original_return_col,
            task=args.task,
            topk_values=topk_values
        )
    
    # 7. 样本外测试集评估（walk-forward 的核心）
    logger.info("=" * 60)
    logger.info("样本外测试集评估（OOS Evaluation）")
    logger.info("=" * 60)
    
    # 准备测试数据
    df_test_eval = df_test.copy()
    
    # 过滤测试集样本（与训练时一致：过滤 ST、停牌、涨停；跌停可买入，保留）
    filter_columns = ['is_st', 'is_suspended', 'is_limit_up']
    mask = pd.Series([True] * len(df_test_eval))
    for col in filter_columns:
        if col in df_test_eval.columns:
            mask = mask & (~df_test_eval[col].astype(bool))
    df_test_eval = df_test_eval[mask].copy()
    
    # 移除标签为 NaN 的样本
    if args.label_column in df_test_eval.columns:
        df_test_eval = df_test_eval.dropna(subset=[args.label_column])
    
    logger.info(f"测试集样本数（过滤后）: {len(df_test_eval)}")
    
    # 测试集预测
    _test_nan_rates = df_test_eval[feature_columns].isna().mean()
    _test_high_nan = _test_nan_rates[_test_nan_rates > 0.3]
    if len(_test_high_nan) > 0:
        logger.warning(
            f"测试集特征 NaN 比例 >30%（fillna(0) 可能使预测偏离训练分布）: "
            f"{_test_high_nan.round(3).to_dict()}"
        )
    X_test_features = df_test_eval[feature_columns].fillna(0)
    
    if args.task == "classification":
        y_test_pred_proba = model.predict_proba(X_test_features)[:, 1]
        df_test_eval['pred_score'] = y_test_pred_proba
    else:
        y_test_pred = model.predict(X_test_features)
        df_test_eval['pred_score'] = y_test_pred
    
    # 测试集逐日评估
    test_daily_metrics = evaluate_validation_daily(
        model=model,
        df_val=df_test_eval,
        feature_columns=feature_columns,
        original_return_col=args.label_column,
        task=args.task,
        topk_values=topk_values
    )
    
    # 8. 注册模型
    # 准备性能指标（包含训练集、验证集、测试集）
    performance_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "validation_daily": val_daily_metrics,
        "test": {},  # 测试集的聚合指标
        "test_daily": test_daily_metrics  # 测试集逐日评估
    }
    
    # 准备完整的训练参数
    full_train_params = train_params.copy()
    full_train_params.update({
        "algorithm": algorithm,
        "task": args.task,
        "label_transform": args.label_transform if args.task == "regression" else None,
        "winsorize_p": args.winsorize_p if args.label_transform == "cs_zscore" else None,
        "pos_quantile": args.pos_quantile if args.task == "classification" else None,
        "pos_topk": args.pos_topk if args.task == "classification" else None,
        "scale_pos_weight_manual": args.scale_pos_weight is not None
    })
    
    # 注册模型
    version = registry.register_model(
        model=model,
        model_type=f"{algorithm}_{args.task}_wf",
        train_start_date=split.train_start,
        train_end_date=split.train_end,
        feature_columns=feature_columns,
        label_column=args.label_column,
        n_samples=len(X_train) + len(X_val),
        train_params=full_train_params,
        performance_metrics=performance_metrics
    )
    
    logger.info(f"模型已注册: v{version}")
    
    # 9. 记录训练运行日志到CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 准备完整的数据统计
    complete_data_stats = {
        "trade_days_count": train_days_count,
        "total_samples": total_train_samples,
        "samples_after_filter": data_stats["samples_after_filter"],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "val_start_date": data_stats["val_start_date"],
        "val_end_date": data_stats["val_end_date"],
        "val_ratio": args.val_ratio
    }
    
    # 创建训练运行记录
    run_record = create_training_run_record_from_training_session(
        timestamp=timestamp,
        start_date=split.train_start,
        end_date=split.train_end,
        label_column=args.label_column,
        task=args.task,
        model_version=version,
        train_params=full_train_params,
        data_stats=complete_data_stats,
        performance_metrics=performance_metrics,
        wf_run_id=wf_run_id,
        split_index=split.split_index,
        step_frequency=args.step,
        test_start_date=split.test_start,
        test_end_date=split.test_end
    )
    
    # 写入CSV
    csv_path = args.run_log_csv if args.run_log_csv else f"{args.data_root}/models/ml_train_runs.csv"
    write_training_run_to_csv(run_record, csv_path)
    
    logger.info(f"训练运行日志已记录到: {csv_path}")
    
    # 返回结果
    result = {
        "split_index": split.split_index,
        "train_start": split.train_start,
        "train_end": split.train_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "model_version": version,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(df_test_eval),
        "best_iteration": train_params.get("best_iteration"),
        "val_rankic_ir": val_daily_metrics.get("daily_rankic_ir"),
        "test_daily_metrics": test_daily_metrics
    }
    
    return result


def create_training_run_record_from_training_session(
    timestamp: str,
    start_date: str,
    end_date: str,
    label_column: str,
    task: str,
    model_version: int,
    train_params: Dict,
    data_stats: Dict,
    performance_metrics: Dict,
    wf_run_id: Optional[str] = None,
    split_index: Optional[int] = None,
    step_frequency: Optional[str] = None,
    test_start_date: Optional[str] = None,
    test_end_date: Optional[str] = None
) -> TrainingRunRecord:
    """从训练会话创建训练运行记录
    
    支持 walk-forward 场景的扩展参数。
    """
    # 基本信息
    record = TrainingRunRecord(
        timestamp=timestamp,
        model_version=model_version,
        start_date=start_date,
        end_date=end_date,
        label_column=label_column,
        task=task
    )
    
    # Walk-forward 字段
    record.wf_run_id = wf_run_id
    record.split_index = split_index
    record.step_frequency = step_frequency
    record.test_start_date = test_start_date
    record.test_end_date = test_end_date
    
    # 训练配置
    record.label_transform = train_params.get("label_transform")
    record.winsorize_p = train_params.get("winsorize_p")
    record.pos_quantile = train_params.get("pos_quantile")
    record.pos_topk = train_params.get("pos_topk")
    record.scale_pos_weight = train_params.get("scale_pos_weight")
    record.scale_pos_weight_mode = "manual" if train_params.get("scale_pos_weight_manual") else "auto"
    
    # XGBoost超参数
    record.n_estimators = train_params.get("n_estimators", 0)
    record.max_depth = train_params.get("max_depth", 0)
    record.num_leaves = train_params.get("num_leaves", 0)
    record.learning_rate = train_params.get("learning_rate", 0.0)
    record.subsample = train_params.get("subsample", 0.0)
    record.colsample_bytree = train_params.get("colsample_bytree", 0.0)
    record.gamma = train_params.get("gamma", 0.0)
    record.reg_alpha = train_params.get("reg_alpha", 0.0)
    record.reg_lambda = train_params.get("reg_lambda", 0.0)
    record.early_stopping_rounds = train_params.get("early_stopping_rounds", 0)
    record.tree_method = train_params.get("tree_method", "")
    record.random_state = train_params.get("random_state", 0)
    record.n_jobs = train_params.get("n_jobs", 0)
    
    # 数据统计
    record.trade_days_count = data_stats.get("trade_days_count", 0)
    record.total_samples = data_stats.get("total_samples", 0)
    record.samples_after_filter = data_stats.get("samples_after_filter", 0)
    record.train_samples = data_stats.get("train_samples", 0)
    record.val_samples = data_stats.get("val_samples", 0)
    record.val_start_date = data_stats.get("val_start_date", "")
    record.val_end_date = data_stats.get("val_end_date", "")
    record.val_ratio = data_stats.get("val_ratio", 0.2)
    
    # 训练结果
    record.best_iteration = train_params.get("best_iteration")
    
    # 训练集评估指标
    train_metrics = performance_metrics.get("train", {})
    if task == "regression":
        record.train_mse = train_metrics.get("mse")
        record.train_rmse = train_metrics.get("rmse")
        record.train_r2 = train_metrics.get("r2")
        record.train_ic = train_metrics.get("ic")
    else:
        record.train_accuracy = train_metrics.get("accuracy")
        record.train_auc = train_metrics.get("auc")
        record.train_precision = train_metrics.get("precision")
        record.train_recall = train_metrics.get("recall")
    
    # 验证集评估指标
    val_metrics = performance_metrics.get("validation", {})
    if task == "regression":
        record.val_mse = val_metrics.get("mse")
        record.val_rmse = val_metrics.get("rmse")
        record.val_r2 = val_metrics.get("r2")
        record.val_ic = val_metrics.get("ic")
        record.val_rank_ic = val_metrics.get("rank_ic")
    else:
        record.val_accuracy = val_metrics.get("accuracy")
        record.val_auc = val_metrics.get("auc")
        record.val_precision = val_metrics.get("precision")
        record.val_recall = val_metrics.get("recall")
    
    # 验证集逐日评估
    val_daily_metrics = performance_metrics.get("validation_daily", {})
    record.val_daily_rankic_mean = val_daily_metrics.get("daily_rankic_mean")
    record.val_daily_rankic_std = val_daily_metrics.get("daily_rankic_std")
    record.val_daily_rankic_ir = val_daily_metrics.get("daily_rankic_ir")
    
    # 额外指标（包含测试集指标）
    additional_metrics = {}
    
    # 验证集TopK收益和诊断统计
    for key, value in val_daily_metrics.items():
        if key not in ["daily_rankic_mean", "daily_rankic_std", "daily_rankic_ir"]:
            additional_metrics[f"val_{key}"] = value
    
    # 测试集逐日评估指标（walk-forward 的核心）
    test_daily_metrics = performance_metrics.get("test_daily", {})
    for key, value in test_daily_metrics.items():
        additional_metrics[f"test_{key}"] = value
    
    record.additional_metrics = additional_metrics
    
    return record


def write_walk_forward_summary(
    results: List[Dict],
    output_path: str,
    args,
    wf_run_id: str
) -> None:
    """生成 walk-forward 汇总 CSV

    Args:
        results: 所有 split 的结果列表
        output_path: 输出CSV路径
        args: 命令行参数（用于写入训练参数列）
        wf_run_id: walk-forward 运行ID
    """
    if len(results) == 0:
        logger.warning("没有结果可以写入汇总文件")
        return

    logger.info(f"生成 walk-forward 汇总文件: {output_path}")

    # 训练参数（所有 split 共享，写入每行方便后续对比脚本独立使用）
    train_params_cols = {
        "wf_run_id": wf_run_id,
        "wf_start_date": args.wf_start_date,
        "wf_end_date": args.wf_end_date,
        "algorithm": args.algorithm,
        "step": args.step,
        "train_window_years": args.train_window_years,
        "test_window_months": args.test_window_months,
        "val_ratio": args.val_ratio,
        "label_column": args.label_column,
        "task": args.task,
        "label_transform": args.label_transform if args.task == "regression" else None,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "num_leaves": getattr(args, 'num_leaves', None),
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "rank_weight_enabled": args.rank_weight_enabled,
        "rank_weight_topk": args.rank_weight_topk,
        "rank_weight": args.rank_weight,
        "enable_fundamental": args.enable_fundamental_features,
    }

    # 提取每个 split 的关键指标
    summary_rows = []

    for result in results:
        row = {
            "split_index": result["split_index"],
            "train_start": result["train_start"],
            "train_end": result["train_end"],
            "test_start": result["test_start"],
            "test_end": result["test_end"],
            "model_version": result["model_version"],
            "train_samples": result["train_samples"],
            "val_samples": result["val_samples"],
            "test_samples": result["test_samples"],
            "best_iteration": result.get("best_iteration"),
            "val_rankic_ir": result.get("val_rankic_ir"),
        }

        # 添加测试集逐日评估指标
        test_daily = result.get("test_daily_metrics", {})
        row.update(test_daily)

        # 追加训练参数列（放在最后）
        row.update(train_params_cols)

        summary_rows.append(row)

    # 转为 DataFrame 并写入
    df_summary = pd.DataFrame(summary_rows)

    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"汇总文件已保存: {output_path}")
    logger.info(f"  共 {len(summary_rows)} 个切分")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Walk-forward 滚动训练")
    
    # Walk-forward 参数
    parser.add_argument(
        "--wf-start-date",
        type=str,
        required=True,
        help="walk-forward 起始日期，格式 YYYYMMDD"
    )
    parser.add_argument(
        "--wf-end-date",
        type=str,
        required=True,
        help="walk-forward 结束日期，格式 YYYYMMDD"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="quarterly",
        choices=["monthly", "quarterly", "semiannual"],
        help="滚动频率（monthly|quarterly|semiannual），默认 quarterly"
    )
    parser.add_argument(
        "--train-window-years",
        type=int,
        default=5,
        help="训练窗口年数，默认 5"
    )
    parser.add_argument(
        "--test-window-months",
        type=int,
        default=11,
        help="测试窗口月数，默认 11"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="训练数据内部验证集比例，默认 0.1"
    )
    
    # 数据参数
    parser.add_argument(
        "--label-column",
        type=str,
        default="y_ret_5",
        help="标签列名，默认 y_ret_5"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        choices=["y_ret_5", "y_ret_10", "y_ret_20", "neu_y_ret_5", "neu_y_ret_10", "neu_y_ret_20"],
        help="标签选择（y_ret_5|y_ret_10|y_ret_20|neu_y_ret_5|neu_y_ret_10|neu_y_ret_20），默认 y_ret_5。优先级高于 --label-column"
    )
    
    # 任务类型和标签变换参数
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="任务类型（regression|classification），默认 regression"
    )
    parser.add_argument(
        "--label-transform",
        type=str,
        default="raw",
        choices=["raw", "cs_zscore"],
        help="标签变换方式（raw|cs_zscore），默认 raw。仅对 regression 任务生效"
    )
    parser.add_argument(
        "--winsorize-p",
        type=float,
        default=0.01,
        help="winsorize 参数（截断比例），默认 0.01（截断上下1%%）。仅当 label-transform=cs_zscore 时生效"
    )
    parser.add_argument(
        "--pos-quantile",
        type=float,
        default=None,
        help="分类任务正类百分比阈值（例如 0.2 表示 Top20%%），与 pos-topk 二选一"
    )
    parser.add_argument(
        "--pos-topk",
        type=int,
        default=None,
        help="分类任务正类数量阈值（例如 300 表示每日 Top300），与 pos-quantile 二选一，优先级更高"
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="分类任务正类权重，None 表示自动计算为 neg/pos（默认）"
    )
    
    # 算法选择
    parser.add_argument(
        "--algorithm",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm"],
        help="训练算法（xgboost|lightgbm），默认 xgboost"
    )

    # 模型参数
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="树的数量，默认 200"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="树的最大深度，默认 5（金融数据噪声大不宜过深）"
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=None,
        help="LightGBM 叶子数，默认 31。仅 LightGBM 有效，XGBoost 忽略此参数"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="学习率，默认 0.05"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="样本采样比例，默认 0.8"
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="特征采样比例，默认 0.8"
    )
    parser.add_argument(
        "--min-child-weight",
        type=int,
        default=100,
        help="叶节点最少样本权重和，防止过拟合，默认 100（金融数据建议 100-500）"
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.05,
        help="L1 正则化系数，默认 0.05（建议范围 0.05-0.5）"
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0,
        help="L2 正则化系数，默认 1.0（建议范围 1.0-5.0）"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="节点分裂最小损失下降，默认 0.1（建议范围 0.0-1.0）"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子，默认 42"
    )

    # rank-weight 参数：Top/Bottom K 样本增强权重
    parser.add_argument(
        "--rank-weight-enabled",
        action="store_true",
        default=True,
        help="启用 Top/Bottom K 样本权重增强（默认开启）"
    )
    parser.add_argument(
        "--no-rank-weight",
        action="store_false",
        dest="rank_weight_enabled",
        help="禁用 rank-weight（覆盖 --rank-weight-enabled）"
    )
    parser.add_argument(
        "--rank-weight-topk",
        type=int,
        default=30,
        help="每日 Top/Bottom K 样本数，默认 30"
    )
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=5.0,
        help="Top/Bottom K 样本权重，默认 5.0"
    )

    # 目标函数
    parser.add_argument(
        "--objective",
        type=str,
        default="mse",
        choices=["mse", "lambdarank"],
        help="目标函数类型：mse（回归，默认）或 lambdarank（排序学习，直接优化股票排序）"
    )

    # 基本面因子
    parser.add_argument(
        "--enable-fundamental-features",
        action="store_true",
        help="启用基本面因子（ROE、营收增速等）作为训练特征"
    )

    # 其他参数
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="数据根目录，默认 ./data"
    )
    parser.add_argument(
        "--run-log-csv",
        type=str,
        default=None,
        help="训练运行日志CSV路径，默认为 {data_root}/models/ml_train_runs.csv"
    )
    parser.add_argument(
        "--wf-summary-csv",
        type=str,
        default=None,
        help="walk-forward 汇总CSV路径，默认为 {data_root}/walk_forward/walk_forward_summary.csv"
    )
    
    args = parser.parse_args()
    
    # 如果指定了 --label，则覆盖 --label-column
    if args.label is not None:
        args.label_column = args.label
    
    # 设置日志
    setup_logger()
    
    logger.info("=" * 80)
    logger.info("Walk-forward 滚动训练")
    logger.info("=" * 80)
    logger.info(f"Walk-forward 时间区间: {args.wf_start_date} 至 {args.wf_end_date}")
    logger.info(f"滚动频率: {args.step}")
    logger.info(f"训练窗口: {args.train_window_years} 年")
    logger.info(f"测试窗口: {args.test_window_months} 个月")
    logger.info(f"标签列: {args.label_column}")
    logger.info(f"任务类型: {args.task}")
    logger.info(f"数据目录: {args.data_root}")
    
    try:
        # 初始化组件
        storage = Storage(root_path=args.data_root)
        loader = DataLoader(storage)
        registry = ModelRegistry(models_dir=f"{args.data_root}/models")
        
        # 生成 walk-forward ID
        wf_run_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Walk-forward 运行ID: {wf_run_id}")
        
        # 1. 生成 walk-forward 切分
        trade_cal = loader.load_clean_trade_cal()
        if trade_cal is None:
            trade_cal = loader.load_trade_cal()
        
        splits = generate_walk_forward_splits(
            trade_cal=trade_cal,
            wf_start_date=args.wf_start_date,
            wf_end_date=args.wf_end_date,
            step_frequency=args.step,
            train_window_years=args.train_window_years,
            test_window_months=args.test_window_months
        )
        
        if len(splits) == 0:
            logger.error("未生成任何切分，请检查参数设置")
            sys.exit(1)
        
        print_splits_summary(splits)
        
        # 2. 执行每个 split 的训练
        results = []
        topk_values = [30, 100, 300]
        
        for split in splits:
            try:
                result = execute_split_training(
                    split=split,
                    wf_run_id=wf_run_id,
                    storage=storage,
                    loader=loader,
                    registry=registry,
                    args=args,
                    topk_values=topk_values
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Split {split.split_index} 训练失败: {e}")
                logger.error(traceback.format_exc())
                logger.warning("继续执行下一个 split...")
                continue
        
        # 3. 生成 walk-forward 汇总文件（统一输出到 raw/ 子目录）
        if len(results) > 0:
            if args.wf_summary_csv:
                summary_csv_path = args.wf_summary_csv
            else:
                summary_csv_path = f"{args.data_root}/walk_forward/raw/walk_forward_summary_{wf_run_id}.csv"

            write_walk_forward_summary(results, summary_csv_path, args, wf_run_id)
        else:
            logger.warning("没有成功完成的训练，跳过生成汇总文件")
        
        logger.info("=" * 80)
        logger.info("Walk-forward 滚动训练完成！")
        logger.info(f"  成功完成: {len(results)} / {len(splits)} 个切分")
        logger.info(f"  运行ID: {wf_run_id}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Walk-forward 训练失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
