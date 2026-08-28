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
import gc
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

from src.lazybull.common.config import get_data_root, get_models_root
from src.lazybull.common.logger import setup_logger

from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml import ModelRegistry
from src.lazybull.ml.run_logger import (
    create_training_run_record_from_training_session,
    write_training_run_to_csv,
)
from src.lazybull.ml.train_core import (
    add_blended_return_label,
    attach_cons_revision_schema_version,
    attach_cashflow_quality_train_params,
    load_features_data,
    prepare_training_data,
    transform_labels_cs_zscore,
    generate_classification_labels,
    train_xgboost_model,
    train_lightgbm_model,
    evaluate_validation_daily,
    build_rank_sample_weights,
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
    parser.add_argument("--start-date", type=str, required=True, help="训练开始日期，格式 YYYYMMDD")
    parser.add_argument("--end-date", type=str, required=True, help="训练结束日期，格式 YYYYMMDD")
    parser.add_argument(
        "--label-column",
        type=str,
        default="neu_y_ret_20",
        help="标签列名，默认 neu_y_ret_20（行业中性化后的20日收益）",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        choices=["y_ret_5", "y_ret_10", "y_ret_20", "neu_y_ret_5", "neu_y_ret_10", "neu_y_ret_20"],
        help="标签选择，默认 neu_y_ret_20。优先级高于 --label-column",
    )
    parser.add_argument(
        "--neutral-label-blend-weight",
        type=float,
        default=0.0,
        help="原始收益在行业中性混合标签中的权重，范围 0~1，默认 0（保持原标签）",
    )

    # 任务类型和标签变换参数
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="任务类型（regression|classification），默认 regression",
    )
    parser.add_argument(
        "--label-transform",
        type=str,
        default="raw",
        choices=["raw", "cs_zscore"],
        help="标签变换方式（raw|cs_zscore），默认 raw。仅对 regression 任务生效",
    )
    parser.add_argument(
        "--winsorize-p",
        type=float,
        default=0.01,
        help="winsorize 参数（截断比例），默认 0.01（截断上下1%%）。仅当 label-transform=cs_zscore 时生效",
    )
    parser.add_argument(
        "--pos-quantile",
        type=float,
        default=None,
        help="分类任务正类百分比阈值（例如 0.2 表示 Top20%%），与 pos-topk 二选一",
    )
    parser.add_argument(
        "--pos-topk",
        type=int,
        default=None,
        help="分类任务正类数量阈值（例如 300 表示每日 Top300），与 pos-quantile 二选一，优先级更高",
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="分类任务正类权重，None 表示自动计算为 neg/pos（默认）",
    )

    # 模型参数
    parser.add_argument(
        "--n-estimators", type=int, default=200, help="树的数量，默认 200（建议范围：100-300）"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="树的最大深度，默认 5（建议范围：4-6，金融数据噪声大不宜过深）",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=None,
        help="LightGBM 叶子数，默认 31。仅 LightGBM 有效，XGBoost 忽略此参数",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.05, help="学习率，默认 0.05（建议范围：0.01-0.1）"
    )
    parser.add_argument("--subsample", type=float, default=0.8, help="样本采样比例，默认 0.8")
    parser.add_argument(
        "--colsample-bytree", type=float, default=0.8, help="特征采样比例，默认 0.8"
    )
    parser.add_argument(
        "--min-child-weight",
        type=int,
        default=100,
        help="叶节点最少样本权重和，防止过拟合，默认 100（金融数据建议 100-500）",
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.05,
        help="L1 正则化系数，默认 0.05（建议范围 0.05-0.5）",
    )
    parser.add_argument(
        "--reg-lambda", type=float, default=1.0, help="L2 正则化系数，默认 1.0（建议范围 1.0-5.0）"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="节点分裂最小损失下降，默认 0.1（建议范围 0.0-1.0）",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子，默认 42")

    # rank-weight 参数：Top/Bottom K 样本增强权重
    parser.add_argument(
        "--rank-weight-enabled",
        action="store_true",
        default=True,
        help="启用 Top/Bottom K 样本权重增强（默认开启）",
    )
    parser.add_argument(
        "--no-rank-weight",
        action="store_false",
        dest="rank_weight_enabled",
        help="禁用 rank-weight（覆盖 --rank-weight-enabled）",
    )
    parser.add_argument(
        "--rank-weight-topk", type=int, default=30, help="每日 Top/Bottom K 样本数，默认 30"
    )
    parser.add_argument(
        "--rank-weight", type=float, default=5.0, help="Top/Bottom K 样本权重，默认 5.0"
    )
    parser.add_argument(
        "--rank-weight-topk-weight-mode",
        type=str,
        default="linear_decay",
        choices=["linear_decay", "flat"],
        help="TopK 权重分配模式：linear_decay（默认）| flat（TopK 同权）",
    )

    # 算法选择
    parser.add_argument(
        "--algorithm",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm"],
        help="训练算法（xgboost|lightgbm），默认 xgboost",
    )

    # 目标函数
    parser.add_argument(
        "--objective",
        type=str,
        default="mse",
        choices=["mse", "lambdarank"],
        help="目标函数类型：mse（回归，默认）或 lambdarank（排序学习，直接优化股票排序）",
    )

    # 基本面因子
    parser.add_argument(
        "--enable-fundamental-features",
        action="store_true",
        help="启用基本面因子（ROE、营收增速等）作为训练特征",
    )

    # 另类数据因子
    parser.add_argument(
        "--enable-alt-features",
        action="store_true",
        help="启用另类数据因子（股东人数、业绩预告等）",
    )

    # 融资融券因子
    parser.add_argument(
        "--enable-margin-features",
        action="store_true",
        help="启用融资融券因子（融资余额变动、融券/融资比、净买入比等）",
    )

    # 筹码胜率因子
    parser.add_argument(
        "--enable-cyq-features",
        action="store_true",
        help="启用筹码胜率因子（winner_rate、成本偏离等）",
    )

    # 基金持仓因子
    parser.add_argument(
        "--enable-fund-features",
        action="store_true",
        help="启用基金持仓因子（持股比例、基金数量等）",
    )

    # 业绩快报因子
    parser.add_argument(
        "--enable-express-features",
        action="store_true",
        help="启用业绩快报因子（实际营收/净利润增速等）",
    )

    # 北向资金因子
    parser.add_argument(
        "--enable-north-features",
        action="store_true",
        default=False,
        help="启用北向资金因子（moneyflow_hsgt, 市场级广播）",
    )

    # 龙虎榜因子
    parser.add_argument(
        "--enable-lhb-features",
        action="store_true",
        default=False,
        help="启用龙虎榜因子（top_list, 稀疏数据未上榜填 0）",
    )

    # 一致预期因子
    parser.add_argument(
        "--enable-consensus-features",
        action="store_true",
        default=False,
        help="启用卖方一致预期因子（report_rc, 滚动 30/60/90 日聚合）",
    )

    parser.add_argument(
        "--enable-cashflow-quality-features",
        action="store_true",
        default=False,
        help="启用现金流质量因子（需先下载 cashflow 数据）",
    )

    parser.add_argument(
        "--enable-consensus-revision-features",
        action="store_true",
        default=False,
        help="启用一致预期修正因子（EPS修正加速度/分歧度等时序信号）",
    )

    parser.add_argument(
        "--feature-stability-filter",
        action="store_true",
        help="启用特征稳定性筛选（移除跨时期IC方向不一致的特征）",
    )

    parser.add_argument(
        "--factor-prune",
        action="store_true",
        help="启用因子精简（从 data/models/factor_exclude_list.json 加载排除列表）",
    )
    parser.add_argument(
        "--factor-exclude-file",
        type=str,
        default=None,
        help="因子精简清单路径；未指定时使用 data/models/factor_exclude_list.json",
    )

    parser.add_argument(
        "--freshness-strategy",
        type=str,
        default="state_keep_event_decay",
        choices=["state_keep_event_decay", "state_keep_event_no_decay", "drop_all"],
        help=(
            "freshness 处理策略：state_keep_event_decay=状态型保留+事件型衰减（默认），"
            "state_keep_event_no_decay=状态型保留+事件型不衰减（实验归因），"
            "drop_all=删除全部 freshness 特征"
        ),
    )
    parser.add_argument(
        "--event-freshness-half-life-days",
        type=float,
        default=45.0,
        help="事件型因子 freshness 衰减半衰期（天），默认 45",
    )

    # 其他参数
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录；未指定时使用 configs/base.yaml 中的 data.* 配置",
    )
    parser.add_argument(
        "--run-log-csv",
        type=str,
        default=None,
        help="训练运行日志CSV路径，默认为 {data_root}/ml_train_runs.csv",
    )

    args = parser.parse_args()

    # 如果指定了 --label，则覆盖 --label-column
    if args.label is not None:
        args.label_column = args.label
    if not 0.0 <= args.neutral_label_blend_weight <= 1.0:
        parser.error("--neutral-label-blend-weight 必须在 0~1 之间")
    if args.neutral_label_blend_weight > 0 and args.task != "regression":
        parser.error("--neutral-label-blend-weight 仅支持 regression 任务")
    if args.neutral_label_blend_weight > 0 and not args.label_column.startswith("neu_y_ret_"):
        parser.error("混合标签要求 --label 使用 neu_y_ret_N")

    # 设置日志
    setup_logger()

    logger.info("=" * 60)
    logger.info(f"{args.algorithm.upper()} 模型训练")
    logger.info("=" * 60)
    logger.info(f"训练算法: {args.algorithm}")
    logger.info(f"训练日期区间: {args.start_date} 至 {args.end_date}")
    logger.info(f"标签列: {args.label_column}")
    logger.info(f"行业中性标签混合权重: {args.neutral_label_blend_weight:.2f}")
    effective_data_root = args.data_root or get_data_root()
    logger.info(f"数据目录: {effective_data_root}")
    logger.info(
        f"rank-weight: {'已启用' if args.rank_weight_enabled else '已禁用'} "
        f"（topk={args.rank_weight_topk}, weight={args.rank_weight}, "
        f"topk_mode={getattr(args, 'rank_weight_topk_weight_mode', 'linear_decay')}）"
    )

    try:
        # 初始化组件
        storage = Storage(root_path=args.data_root)
        loader = DataLoader(storage)
        registry = ModelRegistry(
            models_dir=get_models_root(
                str(Path(args.data_root) / "models") if args.data_root else None
            )
        )

        # 1. 加载特征数据
        df, trade_days_count = load_features_data(storage, loader, args.start_date, args.end_date)
        total_samples = len(df)
        actual_label_column = add_blended_return_label(
            df,
            args.label_column,
            args.neutral_label_blend_weight,
        )

        # 1.5. 应用标签变换（如果需要）
        if args.task == "classification":
            # 分类任务：生成二分类标签
            if args.pos_quantile is None and args.pos_topk is None:
                raise ValueError("分类任务必须指定 --pos-quantile 或 --pos-topk")

            df = generate_classification_labels(
                df,
                label_column=args.label_column,
                pos_quantile=args.pos_quantile,
                pos_topk=args.pos_topk,
            )

            # 使用二分类标签作为训练标签
            binary_label_col = f"{actual_label_column}_binary"
            actual_label_column = binary_label_col

        # 2. 准备训练数据（按 trade_date 粒度切分训练集/验证集）
        # cs_zscore 回归任务：切分后对 train/val 各自独立变换，不共享截面统计量
        label_transform_fn = None
        if args.task == "regression" and args.label_transform == "cs_zscore":
            label_transform_fn = lambda d: transform_labels_cs_zscore(
                d, label_column=actual_label_column, winsorize_p=args.winsorize_p
            )
        (
            X_train,
            y_train,
            X_val,
            y_val,
            feature_columns,
            df_train_split,
            df_val_split,
            data_stats,
            df_val_split_original,
        ) = prepare_training_data(
            df,
            actual_label_column,
            label_transform_fn=label_transform_fn,
            enable_fundamental_features=args.enable_fundamental_features,
            enable_alt_features=args.enable_alt_features,
            enable_margin_features=args.enable_margin_features,
            enable_cyq_features=args.enable_cyq_features,
            enable_fund_features=args.enable_fund_features,
            enable_express_features=args.enable_express_features,
            enable_north_features=getattr(args, "enable_north_features", False),
            enable_lhb_features=getattr(args, "enable_lhb_features", False),
            enable_consensus_features=getattr(args, "enable_consensus_features", False),
            enable_cashflow_quality_features=getattr(
                args, "enable_cashflow_quality_features", False
            ),
            enable_consensus_revision_features=getattr(
                args, "enable_consensus_revision_features", False
            ),
            feature_stability_filter=args.feature_stability_filter,
            factor_prune=args.factor_prune,
            factor_exclude_file=getattr(args, "factor_exclude_file", None),
            freshness_strategy=getattr(args, "freshness_strategy", "state_keep_event_decay"),
            event_freshness_half_life_days=getattr(args, "event_freshness_half_life_days", 45.0),
        )

        # 原始 df 已不再需要，释放 ~3 GiB 内存
        del df
        gc.collect()

        # 3. 训练模型
        # 当 label_transform=cs_zscore 时，标签已在 cs_zscore 步骤中 winsorize，训练时不再 winsorize
        skip_label_winsorize = args.task == "regression" and args.label_transform == "cs_zscore"

        # 3.1. 构造样本权重（rank-weight：Top/Bottom K 增强）
        rank_sample_weight = None
        if args.rank_weight_enabled:
            rank_sample_weight = build_rank_sample_weights(
                df_train=df_train_split,
                label_column=actual_label_column,
                topk=args.rank_weight_topk,
                top_weight=args.rank_weight,
                topk_weight_mode=getattr(args, "rank_weight_topk_weight_mode", "linear_decay"),
            )

        # 根据算法选择训练函数
        train_fn = train_lightgbm_model if args.algorithm == "lightgbm" else train_xgboost_model

        # 构建训练参数（num_leaves 仅 LightGBM 使用）
        extra_kwargs = {}
        if args.algorithm == "lightgbm" and args.num_leaves is not None:
            extra_kwargs["num_leaves"] = args.num_leaves

        # LambdaRank 需要传入 DataFrame 用于按 trade_date 构造 qid 分组
        if args.algorithm == "xgboost":
            extra_kwargs["objective_type"] = args.objective
            if args.objective == "lambdarank":
                extra_kwargs["df_train_for_group"] = df_train_split
                extra_kwargs["df_val_for_group"] = df_val_split

        model, train_params, train_metrics, val_metrics = train_fn(
            X_train,
            y_train,
            X_val,
            y_val,
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

        # 4. 验证集逐日评估（贴近交易场景，特别是分类任务）
        daily_val_metrics = {}
        if len(df_val_split_original) > 0 and args.task == "classification":
            # 分类任务：基于原始收益列（如 y_ret_20）进行逐日评估
            # 使用变换前的原始 val df，确保收益单位不被 label_transform 污染
            original_return_col = args.label_column  # 使用原始标签列（去掉 _binary）
            daily_val_metrics = evaluate_validation_daily(
                model=model,
                df_val=df_val_split_original,
                feature_columns=feature_columns,
                original_return_col=original_return_col,
                task=args.task,
                topk_values=[30, 100, 300],
            )

        # 合并训练和验证指标
        performance_metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "validation_daily": daily_val_metrics,  # 逐日评估结果
        }

        # 准备完整的训练参数（包含任务配置）
        full_train_params = train_params.copy()
        attach_cons_revision_schema_version(
            full_train_params,
            getattr(args, "enable_consensus_revision_features", False),
        )
        attach_cashflow_quality_train_params(
            full_train_params,
            getattr(args, "enable_cashflow_quality_features", False),
            feature_columns=feature_columns,
        )
        full_train_params.update(
            {
                "algorithm": args.algorithm,
                "task": args.task,
                "label_transform": args.label_transform if args.task == "regression" else None,
                "neutral_label_blend_weight": args.neutral_label_blend_weight,
                "winsorize_p": args.winsorize_p if args.label_transform == "cs_zscore" else None,
                "pos_quantile": args.pos_quantile if args.task == "classification" else None,
                "pos_topk": args.pos_topk if args.task == "classification" else None,
                # 记录 scale_pos_weight 是否手动指定
                "scale_pos_weight_manual": args.scale_pos_weight is not None,
                # 记录 rank-weight 配置，便于回溯
                "rank_weight_enabled": args.rank_weight_enabled,
                "rank_weight_topk": args.rank_weight_topk if args.rank_weight_enabled else None,
                "rank_weight": args.rank_weight if args.rank_weight_enabled else None,
                "rank_weight_topk_weight_mode": (
                    getattr(args, "rank_weight_topk_weight_mode", "linear_decay")
                    if args.rank_weight_enabled
                    else None
                ),
                "enable_cashflow_quality_features": getattr(
                    args, "enable_cashflow_quality_features", False
                ),
                "enable_consensus_revision_features": getattr(
                    args, "enable_consensus_revision_features", False
                ),
                # 推理侧（MLSignal）按此复现事件型 freshness 衰减，必须与训练一致
                "freshness_strategy": getattr(args, "freshness_strategy", "state_keep_event_decay"),
                "event_freshness_half_life_days": getattr(
                    args, "event_freshness_half_life_days", 45.0
                ),
            }
        )

        # 4. 注册模型
        version = registry.register_model(
            model=model,
            model_type=f"{args.algorithm}_{args.task}",
            train_start_date=args.start_date,
            train_end_date=args.end_date,
            feature_columns=feature_columns,
            label_column=args.label_column,
            n_samples=len(X_train) + len(X_val),
            train_params=full_train_params,
            performance_metrics=performance_metrics,
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
                "val_ratio": 0.2,  # 默认值，如果需要可以改为参数
                "val_raw_start_date": data_stats.get(
                    "val_raw_start_date", data_stats["val_start_date"]
                ),
                "val_raw_end_date": data_stats.get("val_raw_end_date", data_stats["val_end_date"]),
                "val_raw_n_dates": data_stats.get("val_raw_n_dates", 0),
                "val_raw_samples": data_stats.get("val_raw_samples", 0),
                "val_es_start_date": data_stats.get(
                    "val_es_start_date", data_stats["val_start_date"]
                ),
                "val_es_end_date": data_stats.get("val_es_end_date", data_stats["val_end_date"]),
                "val_es_n_dates": data_stats.get("val_es_n_dates", 0),
                "val_es_samples": data_stats.get("val_es_samples", len(X_val)),
                "val_calib_start_date": data_stats.get("val_calib_start_date", "N/A"),
                "val_calib_end_date": data_stats.get("val_calib_end_date", "N/A"),
                "val_calib_n_dates": data_stats.get("val_calib_n_dates", 0),
                "val_calib_samples": data_stats.get("val_calib_samples", 0),
                "val_embargo_days": data_stats.get("val_embargo_days", 0),
                "val_embargo_days_applied": data_stats.get("val_embargo_days_applied", 0),
                "val_embargo_n_dates": data_stats.get("val_embargo_n_dates", 0),
                "val_embargo_samples": data_stats.get("val_embargo_samples", 0),
                "val_embargo_start_date": data_stats.get("val_embargo_start_date", "N/A"),
                "val_embargo_end_date": data_stats.get("val_embargo_end_date", "N/A"),
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
                performance_metrics=performance_metrics,
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
