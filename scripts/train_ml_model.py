#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost 模型训练脚本

功能：
- 读取指定日期区间的特征数据
- 训练 XGBoost 回归模型（默认标签为 neu_y_ret_20：行业中性化后的20日收益）
- 自动保存模型到 data/models 目录（使用 joblib）
- 自动递增版本号（v1, v2, ...）
- 记录训练元数据到 model_registry.json

使用示例：
    # 使用默认参数训练（默认标签为 neu_y_ret_20）
    python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231
    
    # 指定超参数
    python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
        --n-estimators 200 --max-depth 5 --learning-rate 0.05
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Dict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score

from src.lazybull.common.logger import setup_logger
from src.lazybull.common.feature_utils import (
    drop_high_correlation_features,
    analyze_feature_importance
)

from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml import ModelRegistry
from src.lazybull.ml.run_logger import (
    create_training_run_record_from_training_session,
    write_training_run_to_csv
)
from src.lazybull.ml.train_core import (
    load_features_data,
    prepare_training_data,
    transform_labels_cs_zscore,
    generate_classification_labels,
    train_xgboost_model,
    evaluate_validation_daily
)

try:
    import xgboost as xgb
except ImportError:
    logger.error("需要安装 xgboost: pip install xgboost")
    sys.exit(1)
import warnings
# 匹配告警信息中的关键字符串，设置为 ignore
warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练 XGBoost 模型")
    
    # 数据参数
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="训练开始日期，格式 YYYYMMDD"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="训练结束日期，格式 YYYYMMDD"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="neu_y_ret_20",
        help="标签列名，默认 neu_y_ret_20（行业中性化后的20日收益）"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        choices=["y_ret_5", "y_ret_10", "y_ret_20", "neu_y_ret_5", "neu_y_ret_10", "neu_y_ret_20"],
        help="标签选择，默认 neu_y_ret_20。优先级高于 --label-column"
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
    
    # 模型参数
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="树的数量，默认 200（建议范围：100-300）"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="树的最大深度，默认 8（建议范围：6-10）"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="学习率，默认 0.05（建议范围：0.01-0.1）"
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
        "--random-state",
        type=int,
        default=42,
        help="随机种子，默认 42"
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
        help="训练运行日志CSV路径，默认为 {data_root}/ml_train_runs.csv"
    )
    
    args = parser.parse_args()
    
    # 如果指定了 --label，则覆盖 --label-column
    if args.label is not None:
        args.label_column = args.label
    
    # 设置日志
    setup_logger()
    
    logger.info("=" * 60)
    logger.info("XGBoost 模型训练")
    logger.info("=" * 60)
    logger.info(f"训练日期区间: {args.start_date} 至 {args.end_date}")
    logger.info(f"标签列: {args.label_column}")
    logger.info(f"数据目录: {args.data_root}")
    
    try:
        # 初始化组件
        storage = Storage(root_path=args.data_root)
        loader = DataLoader(storage)
        registry = ModelRegistry(models_dir=f"{args.data_root}/models")
        
        # 1. 加载特征数据
        df, trade_days_count = load_features_data(storage, loader, args.start_date, args.end_date)
        total_samples = len(df)
        
        # 1.5. 应用标签变换（如果需要）
        if args.task == "classification":
            # 分类任务：生成二分类标签
            if args.pos_quantile is None and args.pos_topk is None:
                raise ValueError("分类任务必须指定 --pos-quantile 或 --pos-topk")
            
            df = generate_classification_labels(
                df,
                label_column=args.label_column,
                pos_quantile=args.pos_quantile,
                pos_topk=args.pos_topk
            )
            
            # 使用二分类标签作为训练标签
            binary_label_col = f"{args.label_column}_binary"
            actual_label_column = binary_label_col
        else:
            # 回归任务：应用标签变换
            if args.label_transform == "cs_zscore":
                df = transform_labels_cs_zscore(
                    df,
                    label_column=args.label_column,
                    winsorize_p=args.winsorize_p
                )
            actual_label_column = args.label_column
        
        # 2. 准备训练数据（包含验证集切分）
        X_train, y_train, X_val, y_val, feature_columns, df_train_split, df_val_split, data_stats = prepare_training_data(df, actual_label_column)

        # 2.6. 测试阶段：临时过滤掉一些高相关性特征，减少过拟合风险（后续可以改为参数控制）
        if False:  # 默认不启用，后续可以改为参数控制
            df_new = df.copy()
            df_new = df_new.drop('trade_date', axis=1)
            df_new = df_new.drop('ts_code', axis=1)
            df_new = df_new.drop('industry', axis=1)
            print(f"原始特征: {df_new.columns.tolist()}")
            redundant_features = drop_high_correlation_features(df_new, threshold=0.9)
            logger.warning(f"以下特征与其他特征高度相关，建议后续版本中删除: {redundant_features}")
            import seaborn as sns
            import matplotlib.pyplot as plt
            #plt.figure(figsize=(20, 15))
            #sns.heatmap(df_new.corr(), cmap='coolwarm', center=0)
            #plt.show()

            #shap_v = analyze_feature_importance(X_train, y_train)
            #logger.warning(f"SHAP 特征重要性分析结果: {shap_v}")
            exit()

        # 3. 训练模型
        # 当 label_transform=cs_zscore 时，标签已在 cs_zscore 步骤中 winsorize，训练时不再 winsorize
        skip_label_winsorize = (args.task == "regression" and args.label_transform == "cs_zscore")
        
        model, train_params, train_metrics, val_metrics = train_xgboost_model(
            X_train, y_train, X_val, y_val,
            task=args.task,
            skip_label_winsorize=skip_label_winsorize,
            scale_pos_weight=args.scale_pos_weight,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=args.random_state
        )
        
        # 4. 验证集逐日评估（贴近交易场景，特别是分类任务）
        daily_val_metrics = {}
        if len(df_val_split) > 0 and args.task == "classification":
            # 分类任务：基于原始收益列（如 y_ret_20）进行逐日评估
            original_return_col = args.label_column  # 使用原始标签列（去掉 _binary）
            daily_val_metrics = evaluate_validation_daily(
                model=model,
                df_val=df_val_split,
                feature_columns=feature_columns,
                original_return_col=original_return_col,
                task=args.task,
                topk_values=[30, 100, 300]
            )
        
        # 合并训练和验证指标
        performance_metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "validation_daily": daily_val_metrics  # 逐日评估结果
        }
        
        # 准备完整的训练参数（包含任务配置）
        full_train_params = train_params.copy()
        full_train_params.update({
            "task": args.task,
            "label_transform": args.label_transform if args.task == "regression" else None,
            "winsorize_p": args.winsorize_p if args.label_transform == "cs_zscore" else None,
            "pos_quantile": args.pos_quantile if args.task == "classification" else None,
            "pos_topk": args.pos_topk if args.task == "classification" else None,
            # 记录 scale_pos_weight 是否手动指定
            "scale_pos_weight_manual": args.scale_pos_weight is not None
        })
        
        # 4. 注册模型
        version = registry.register_model(
            model=model,
            model_type=f"xgboost_{args.task}",
            train_start_date=args.start_date,
            train_end_date=args.end_date,
            feature_columns=feature_columns,
            label_column=args.label_column,
            n_samples=len(X_train) + len(X_val),
            train_params=full_train_params,
            performance_metrics=performance_metrics
        )
        
        logger.info("=" * 60)
        logger.info(f"模型训练完成！版本: v{version}")
        logger.info(f"模型保存路径: {args.data_root}/models/")
        logger.info("=" * 60)
        
        # 5. 记录训练运行日志到CSV
        try:
            # 准备完整的数据统计
            complete_data_stats = {
                "trade_days_count": trade_days_count,
                "total_samples": total_samples,
                "samples_after_filter": data_stats["samples_after_filter"],
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "val_start_date": data_stats["val_start_date"],
                "val_end_date": data_stats["val_end_date"],
                "val_ratio": 0.2  # 默认值，如果需要可以改为参数
            }
            
            # 创建训练运行记录
            run_record = create_training_run_record_from_training_session(
                start_date=args.start_date,
                end_date=args.end_date,
                label_column=args.label_column,
                task=args.task,
                model_version=version,
                train_params=full_train_params,
                data_stats=complete_data_stats,
                performance_metrics=performance_metrics
            )
            
            # 确定CSV路径
            if args.run_log_csv is not None:
                csv_path = args.run_log_csv
            else:
                csv_path = f"{args.data_root}/models/ml_train_runs.csv"
            
            # 写入CSV
            write_training_run_to_csv(run_record, csv_path)
            
            logger.info(f"训练运行日志已记录到: {csv_path}")
        except Exception as e:
            logger.error(f"记录训练运行日志失败: {e}")
            logger.warning("训练已完成，但日志记录失败，不影响模型保存")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
