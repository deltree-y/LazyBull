#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost 模型训练脚本

功能：
- 读取指定日期区间的特征数据
- 训练 XGBoost 回归模型（标签为 y_ret_5）
- 自动保存模型到 data/models 目录（使用 joblib）
- 自动递增版本号（v1, v2, ...）
- 记录训练元数据到 model_registry.json

使用示例：
    # 使用默认参数训练
    python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231
    
    # 指定超参数
    python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
        --n-estimators 200 --max-depth 5 --learning-rate 0.05
"""

import argparse
import sys
import traceback
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml import ModelRegistry

try:
    import xgboost as xgb
except ImportError:
    logger.error("需要安装 xgboost: pip install xgboost")
    sys.exit(1)


def load_features_data(
    storage: Storage,
    loader: DataLoader,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """加载指定日期区间的特征数据
    
    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD
        
    Returns:
        合并后的特征 DataFrame
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
    
    return df


def prepare_training_data(df: pd.DataFrame, label_column: str = "y_ret_5", val_ratio: float = 0.2) -> tuple:
    """准备训练数据，并按时间切分训练集和验证集
    
    Args:
        df: 特征 DataFrame
        label_column: 标签列名
        val_ratio: 验证集比例，默认 0.2（最后 20% 的时间作为验证集）
        
    Returns:
        (X_train, y_train, X_val, y_val, feature_columns) 元组
    """
    logger.info("准备训练数据...")
    
    # 确认标签列存在
    if label_column not in df.columns:
        raise ValueError(f"标签列 {label_column} 不存在")
    
    # 定义需要排除的列（非特征列）
    # 标识列
    id_columns = ['ts_code', 'trade_date', 'name']
    # 标签列
    label_columns = [label_column]
    # 过滤标记列（使用统一的列名，与clean层一致）
    filter_columns = ['is_st', 'is_suspended']
    # 其他非特征列
    other_exclude_columns = ['tradable', 'list_date', 'list_days', 'is_limit_up', 'is_limit_down']
    
    exclude_columns = id_columns + label_columns + filter_columns + other_exclude_columns
    
    # 获取特征列
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    logger.info(f"特征列数量: {len(feature_columns)}")
    logger.debug(f"特征列: {feature_columns[:10]}...")  # 只显示前10个
    
    # 过滤可训练样本（移除含有过滤标记的样本）
    # 注意：由于 FeatureBuilder 已经过滤了不符合条件的样本，这里理论上不需要再过滤
    # 但为了安全起见，还是检查一下这些列（如果存在）
    mask = pd.Series([True] * len(df))
    for col in filter_columns:
        if col in df.columns:
            mask = mask & (~df[col].astype(bool))
    
    df_train = df[mask].copy()
    logger.info(f"过滤后样本数: {len(df_train)} / {len(df)}")
    
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
    
    return X_train, y_train, X_val, y_val, feature_columns


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> tuple:
    """训练 XGBoost 模型（改进版本）
    
    改进点：
    1. 增加早停机制（early stopping）防止过拟合
    2. 优化默认超参数（更深的树和正则化）
    3. 添加更全面的评估指标（IC/RankIC）
    4. 对标签进行 winsorize 处理减少异常值影响
    
    Args:
        X_train: 训练特征数据
        y_train: 训练标签数据
        X_val: 验证特征数据
        y_val: 验证标签数据
        n_estimators: 树的数量（默认100，建议200-300）
        max_depth: 树的最大深度（默认6，建议6-10）
        learning_rate: 学习率（默认0.1，建议0.05-0.1）
        subsample: 样本采样比例
        colsample_bytree: 特征采样比例
        random_state: 随机种子
        
    Returns:
        (model, train_params, train_metrics, val_metrics) 元组
    """
    logger.info("开始训练 XGBoost 模型（改进版本）...")
    
    # 对标签进行 winsorize 处理（截断极端值，减少噪音）
    from scipy.stats import mstats
    y_train_winsorized = pd.Series(
        mstats.winsorize(y_train, limits=[0.01, 0.01]),  # 截断上下1%极端值
        index=y_train.index
    )
    logger.info("标签 winsorize 处理完成（截断上下1%极端值）")
    
    # 准备训练参数（增加正则化参数）
    train_params = {
        "objective": "reg:squarederror",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "tree_method": "hist",
        "n_jobs": -1,
        "early_stopping_rounds": 30,  # 早停机制参数
        # 增加正则化参数防止过拟合
        "gamma": 0.1,  # 分裂所需的最小损失减少
        "reg_alpha": 0.1,  # L1 正则化
        "reg_lambda": 1.0,  # L2 正则化
    }
    
    logger.info(f"训练参数: {train_params}")
    logger.info("使用早停机制（early_stopping_rounds=30）")
    
    # 创建并训练模型（使用早停）
    model = xgb.XGBRegressor(**train_params)
    
    # 如果有验证集，使用早停机制
    if len(X_val) > 0:
        model.fit(
            X_train, y_train_winsorized,
            eval_set=[(X_val, y_val)],
            verbose=False  # 不打印每轮训练信息
        )
        logger.info(f"模型训练完成（最佳迭代: {model.best_iteration}）")
    else:
        model.fit(X_train, y_train_winsorized)
        logger.info("模型训练完成（无验证集，未使用早停）")
    
    # 计算训练集性能指标
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)  # 注意：使用原始标签评估
    train_rmse = train_mse ** 0.5
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 计算训练集 IC（信息系数，衡量预测与真实值的相关性）
    train_ic = y_train.corr(pd.Series(y_train_pred, index=y_train.index))
    
    train_metrics = {
        "mse": float(train_mse),
        "rmse": float(train_rmse),
        "r2": float(train_r2),
        "ic": float(train_ic)
    }
    
    logger.info(f"训练集性能: MSE={train_mse:.6f}, RMSE={train_rmse:.6f}, R2={train_r2:.4f}, IC={train_ic:.4f}")
    
    # 计算验证集性能指标（包括 IC）
    if len(X_val) > 0:
        y_val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = val_mse ** 0.5
        val_r2 = r2_score(y_val, y_val_pred)
        
        # 计算验证集 IC（更重要的指标）
        val_ic = y_val.corr(pd.Series(y_val_pred, index=y_val.index))
        
        # 计算 RankIC（排序相关性，对选股策略更有意义）
        from scipy.stats import spearmanr
        val_rank_ic, _ = spearmanr(y_val, y_val_pred)
        
        val_metrics = {
            "mse": float(val_mse),
            "rmse": float(val_rmse),
            "r2": float(val_r2),
            "ic": float(val_ic),
            "rank_ic": float(val_rank_ic)
        }
        
        logger.info("=" * 60)
        logger.info("验证集评估结果")
        logger.info("=" * 60)
        logger.info(f"验证集样本数: {len(X_val)}")
        logger.info(f"MSE（均方误差）: {val_mse:.6f}")
        logger.info(f"RMSE（均方根误差）: {val_rmse:.6f}")
        logger.info(f"R2（决定系数）: {val_r2:.4f}")
        logger.info(f"IC（信息系数）: {val_ic:.4f}  <- 重要指标")
        logger.info(f"RankIC（排序IC）: {val_rank_ic:.4f}  <- 选股策略关键指标")
        logger.info("=" * 60)
        logger.info("提示：对于选股策略，IC 和 RankIC 比 R2 更重要")
        logger.info("     IC > 0.03 通常可认为有一定预测能力")
        logger.info("     RankIC > 0.05 说明排序能力较好")
        logger.info("=" * 60)
    else:
        val_metrics = {}
        logger.warning("验证集为空，无法评估")
    
    return model, train_params, train_metrics, val_metrics


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
        default="y_ret_5",
        help="标签列名，默认 y_ret_5"
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
    
    args = parser.parse_args()
    
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
        df = load_features_data(storage, loader, args.start_date, args.end_date)
        
        # 2. 准备训练数据（包含验证集切分）
        X_train, y_train, X_val, y_val, feature_columns = prepare_training_data(df, args.label_column)
        
        # 3. 训练模型
        model, train_params, train_metrics, val_metrics = train_xgboost_model(
            X_train, y_train, X_val, y_val,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=args.random_state
        )
        
        # 合并训练和验证指标
        performance_metrics = {
            "train": train_metrics,
            "validation": val_metrics
        }
        
        # 4. 注册模型
        version = registry.register_model(
            model=model,
            model_type="xgboost",
            train_start_date=args.start_date,
            train_end_date=args.end_date,
            feature_columns=feature_columns,
            label_column=args.label_column,
            n_samples=len(X_train) + len(X_val),
            train_params=train_params,
            performance_metrics=performance_metrics
        )
        
        logger.info("=" * 60)
        logger.info(f"模型训练完成！版本: v{version}")
        logger.info(f"模型保存路径: {args.data_root}/models/")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
