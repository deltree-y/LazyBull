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


def prepare_training_data(df: pd.DataFrame, label_column: str = "y_ret_5") -> tuple:
    """准备训练数据
    
    Args:
        df: 特征 DataFrame
        label_column: 标签列名
        
    Returns:
        (X, y, feature_columns) 元组
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
    # 过滤标记列（以 filter_ 开头）
    filter_columns = [col for col in df.columns if col.startswith('filter_')]
    # 可交易标记
    other_exclude_columns = ['is_tradable']
    
    exclude_columns = id_columns + label_columns + filter_columns + other_exclude_columns
    
    # 获取特征列
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    logger.info(f"特征列数量: {len(feature_columns)}")
    logger.debug(f"特征列: {feature_columns[:10]}...")  # 只显示前10个
    
    # 过滤可训练样本（移除含有过滤标记的样本）
    mask = True
    for col in filter_columns:
        mask = mask & (~df[col])
    
    df_train = df[mask].copy()
    logger.info(f"过滤后样本数: {len(df_train)} / {len(df)}")
    
    # 移除标签为 NaN 的样本
    df_train = df_train.dropna(subset=[label_column])
    logger.info(f"移除标签 NaN 后样本数: {len(df_train)}")
    
    if len(df_train) == 0:
        raise ValueError("没有可用的训练样本")
    
    # 准备 X 和 y
    X = df_train[feature_columns].copy()
    y = df_train[label_column].copy()
    
    # 处理特征中的缺失值（填充为0）
    X = X.fillna(0)
    
    logger.info(f"训练数据准备完成: X shape={X.shape}, y shape={y.shape}")
    
    return X, y, feature_columns


def train_xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> tuple:
    """训练 XGBoost 模型
    
    Args:
        X: 特征数据
        y: 标签数据
        n_estimators: 树的数量
        max_depth: 树的最大深度
        learning_rate: 学习率
        subsample: 样本采样比例
        colsample_bytree: 特征采样比例
        random_state: 随机种子
        
    Returns:
        (model, train_params, performance_metrics) 元组
    """
    logger.info("开始训练 XGBoost 模型...")
    
    # 准备训练参数
    train_params = {
        "objective": "reg:squarederror",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "tree_method": "hist",
        "n_jobs": -1
    }
    
    logger.info(f"训练参数: {train_params}")
    
    # 创建并训练模型
    model = xgb.XGBRegressor(**train_params)
    model.fit(X, y)
    
    logger.info("模型训练完成")
    
    # 计算性能指标
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y, y_pred)
    
    performance_metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2)
    }
    
    logger.info(f"训练集性能: MSE={mse:.6f}, RMSE={rmse:.6f}, R2={r2:.4f}")
    
    return model, train_params, performance_metrics


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
        default=100,
        help="树的数量，默认 100"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="树的最大深度，默认 6"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="学习率，默认 0.1"
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
        
        # 2. 准备训练数据
        X, y, feature_columns = prepare_training_data(df, args.label_column)
        
        # 3. 训练模型
        model, train_params, performance_metrics = train_xgboost_model(
            X, y,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=args.random_state
        )
        
        # 4. 注册模型
        version = registry.register_model(
            model=model,
            model_type="xgboost",
            train_start_date=args.start_date,
            train_end_date=args.end_date,
            feature_columns=feature_columns,
            label_column=args.label_column,
            n_samples=len(X),
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
