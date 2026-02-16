#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练运行日志模块

功能：
- 记录每次训练运行的参数、数据统计、评估指标到CSV文件
- 支持追加模式（不覆盖历史记录）
- 支持动态列扩展（新增字段时自动扩展表头）
"""

import csv
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
import pandas as pd
from loguru import logger


@dataclass
class TrainingRunRecord:
    """训练运行记录数据结构
    
    包含训练的所有关键信息：
    - 基本信息：时间戳、版本
    - 训练配置：日期区间、标签、任务类型、数据集划分
    - 标签变换：label_transform、winsorize_p
    - 分类任务配置：pos_quantile、pos_topk、scale_pos_weight及其模式
    - XGBoost超参数：n_estimators、max_depth、learning_rate等
    - 数据统计：交易日数、样本数、过滤统计
    - 训练结果：best_iteration
    - 评估指标：训练集/验证集的MSE、RMSE、R2、IC、RankIC、ACC、AUC等
    - 逐日评估：RankIC均值/标准差/IR、TopK收益统计
    - 诊断统计：全市场收益、样本数分布、TopK提升和分位数
    """
    
    # 基本信息
    timestamp: str = ""  # 训练时间戳
    model_version: Optional[int] = None  # 模型版本号
    
    # 训练配置
    start_date: str = ""  # 训练开始日期
    end_date: str = ""  # 训练结束日期
    label_column: str = ""  # 标签列名
    task: str = ""  # 任务类型：regression/classification
    
    # 标签变换配置
    label_transform: Optional[str] = None  # raw/cs_zscore（仅回归）
    winsorize_p: Optional[float] = None  # winsorize参数（仅cs_zscore）
    
    # 分类任务配置
    pos_quantile: Optional[float] = None  # 正类百分比阈值
    pos_topk: Optional[int] = None  # 正类数量阈值
    scale_pos_weight: Optional[float] = None  # 正类权重（实际使用值）
    scale_pos_weight_mode: Optional[str] = None  # auto/manual
    
    # XGBoost超参数
    n_estimators: int = 0
    max_depth: int = 0
    learning_rate: float = 0.0
    subsample: float = 0.0
    colsample_bytree: float = 0.0
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    early_stopping_rounds: int = 0
    tree_method: str = ""
    random_state: int = 0
    n_jobs: int = 0
    
    # 数据统计
    trade_days_count: int = 0  # 交易日数量
    total_samples: int = 0  # 总样本数（加载后）
    samples_after_filter: int = 0  # 过滤后样本数
    train_samples: int = 0  # 训练集样本数
    val_samples: int = 0  # 验证集样本数
    val_start_date: str = ""  # 验证集开始日期
    val_end_date: str = ""  # 验证集结束日期
    val_ratio: float = 0.2  # 验证集比例
    
    # 训练结果
    best_iteration: Optional[int] = None  # 最佳迭代次数
    
    # 训练集评估指标（回归）
    train_mse: Optional[float] = None
    train_rmse: Optional[float] = None
    train_r2: Optional[float] = None
    train_ic: Optional[float] = None
    
    # 训练集评估指标（分类）
    train_accuracy: Optional[float] = None
    train_auc: Optional[float] = None
    train_precision: Optional[float] = None
    train_recall: Optional[float] = None
    
    # 验证集评估指标（回归）
    val_mse: Optional[float] = None
    val_rmse: Optional[float] = None
    val_r2: Optional[float] = None
    val_ic: Optional[float] = None
    val_rank_ic: Optional[float] = None
    
    # 验证集评估指标（分类）
    val_accuracy: Optional[float] = None
    val_auc: Optional[float] = None
    val_precision: Optional[float] = None
    val_recall: Optional[float] = None
    
    # 验证集逐日评估（所有任务）
    val_daily_rankic_mean: Optional[float] = None
    val_daily_rankic_std: Optional[float] = None
    val_daily_rankic_ir: Optional[float] = None
    
    # TopK收益统计（动态生成：top30、top100、top300等）
    # 使用additional_metrics字段存储
    
    # 诊断统计（动态生成）
    # 全市场收益、每日样本数、TopK提升等
    # 使用additional_metrics字段存储
    
    # 额外指标（用于存储动态字段，如各种TopK统计）
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Walk-forward 相关字段
    wf_run_id: Optional[str] = None  # walk-forward 运行ID（UUID或可读字符串）
    split_index: Optional[int] = None  # 切分索引（在一次 walk-forward 运行中的序号）
    step_frequency: Optional[str] = None  # 滚动频率（monthly/quarterly/semiannual）
    test_start_date: Optional[str] = None  # 样本外测试开始日期
    test_end_date: Optional[str] = None  # 样本外测试结束日期
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（扁平化结构，便于写入CSV）"""
        result = {}
        
        # 基本字段
        for key, value in asdict(self).items():
            if key == 'additional_metrics':
                continue
            result[key] = value
        
        # 合并额外指标
        result.update(self.additional_metrics)
        
        return result


def write_training_run_to_csv(
    record: TrainingRunRecord,
    csv_path: str = "data/ml_train_runs.csv"
) -> None:
    """将训练运行记录追加写入CSV文件
    
    特性：
    - 文件不存在时自动创建并写入表头
    - 文件存在时追加新行
    - 支持动态列扩展：新增字段时自动扩展表头，旧行缺失字段留空
    
    Args:
        record: 训练运行记录对象
        csv_path: CSV文件路径，默认 data/ml_train_runs.csv
    """
    csv_path = Path(csv_path)
    
    # 确保目录存在
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换记录为字典
    record_dict = record.to_dict()
    
    # 如果文件不存在，创建并写入表头
    if not csv_path.exists():
        logger.info(f"创建训练运行日志文件: {csv_path}")
        
        # 写入表头和第一行数据
        df = pd.DataFrame([record_dict])
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"成功创建日志文件，写入第一条记录（共 {len(df.columns)} 列）")
    else:
        # 文件存在，追加模式
        logger.info(f"追加记录到现有日志文件: {csv_path}")
        
        # 读取现有数据（只读表头即可，用于检测列变化）
        existing_df = pd.read_csv(csv_path, nrows=0, encoding='utf-8-sig')
        existing_columns = set(existing_df.columns)
        new_columns = set(record_dict.keys())
        
        # 检测是否有新增列
        added_columns = new_columns - existing_columns
        if added_columns:
            logger.warning(f"检测到新增字段: {sorted(added_columns)}")
            logger.warning("将扩展表头以兼容新字段")
            
            # 读取完整的现有数据
            existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
            
            # 为新增列填充空值
            for col in added_columns:
                existing_df[col] = None
            
            # 确保新记录包含所有现有列
            all_columns = existing_columns | new_columns
            for col in existing_columns:
                if col not in record_dict:
                    record_dict[col] = None
            
            # 合并数据
            new_row_df = pd.DataFrame([record_dict])
            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            
            # 重新写入（覆盖，但包含所有历史数据）
            combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"成功扩展表头并追加记录（当前共 {len(combined_df)} 行，{len(combined_df.columns)} 列）")
        else:
            # 无新增列，直接追加
            # 确保新记录包含所有现有列（缺失字段补None）
            for col in existing_columns:
                if col not in record_dict:
                    record_dict[col] = None
            
            # 按现有列顺序排列
            ordered_record = {col: record_dict.get(col) for col in existing_df.columns}
            
            # 追加写入
            df = pd.DataFrame([ordered_record])
            df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            
            logger.info(f"成功追加记录到日志文件")


def create_training_run_record_from_training_session(
    start_date: str,
    end_date: str,
    label_column: str,
    task: str,
    model_version: Optional[int],
    train_params: Dict[str, Any],
    data_stats: Dict[str, Any],
    performance_metrics: Dict[str, Any]
) -> TrainingRunRecord:
    """从训练会话信息创建训练运行记录
    
    Args:
        start_date: 训练开始日期
        end_date: 训练结束日期
        label_column: 标签列名
        task: 任务类型
        model_version: 模型版本号
        train_params: 训练参数字典（包含所有超参数和任务配置）
        data_stats: 数据统计字典（样本数、日期等）
        performance_metrics: 性能指标字典（train/validation/validation_daily）
        
    Returns:
        TrainingRunRecord对象
    """
    # 当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 提取训练集和验证集指标
    train_metrics = performance_metrics.get("train", {})
    val_metrics = performance_metrics.get("validation", {})
    val_daily_metrics = performance_metrics.get("validation_daily", {})
    
    # 创建记录对象
    record = TrainingRunRecord(
        timestamp=timestamp,
        model_version=model_version,
        start_date=start_date,
        end_date=end_date,
        label_column=label_column,
        task=task,
        
        # 标签变换配置
        label_transform=train_params.get("label_transform"),
        winsorize_p=train_params.get("winsorize_p"),
        
        # 分类任务配置
        pos_quantile=train_params.get("pos_quantile"),
        pos_topk=train_params.get("pos_topk"),
        scale_pos_weight=train_params.get("scale_pos_weight"),
        scale_pos_weight_mode="manual" if train_params.get("scale_pos_weight_manual") else ("auto" if train_params.get("scale_pos_weight") is not None else None),
        
        # XGBoost超参数
        n_estimators=train_params.get("n_estimators", 0),
        max_depth=train_params.get("max_depth", 0),
        learning_rate=train_params.get("learning_rate", 0.0),
        subsample=train_params.get("subsample", 0.0),
        colsample_bytree=train_params.get("colsample_bytree", 0.0),
        gamma=train_params.get("gamma", 0.0),
        reg_alpha=train_params.get("reg_alpha", 0.0),
        reg_lambda=train_params.get("reg_lambda", 0.0),
        early_stopping_rounds=train_params.get("early_stopping_rounds", 0),
        tree_method=train_params.get("tree_method", ""),
        random_state=train_params.get("random_state", 0),
        n_jobs=train_params.get("n_jobs", 0),
        
        # 数据统计
        trade_days_count=data_stats.get("trade_days_count", 0),
        total_samples=data_stats.get("total_samples", 0),
        samples_after_filter=data_stats.get("samples_after_filter", 0),
        train_samples=data_stats.get("train_samples", 0),
        val_samples=data_stats.get("val_samples", 0),
        val_start_date=data_stats.get("val_start_date", ""),
        val_end_date=data_stats.get("val_end_date", ""),
        val_ratio=data_stats.get("val_ratio", 0.2),
        
        # 训练结果
        best_iteration=train_params.get("best_iteration"),
        
        # 训练集指标
        train_mse=train_metrics.get("mse"),
        train_rmse=train_metrics.get("rmse"),
        train_r2=train_metrics.get("r2"),
        train_ic=train_metrics.get("ic"),
        train_accuracy=train_metrics.get("accuracy"),
        train_auc=train_metrics.get("auc"),
        train_precision=train_metrics.get("precision"),
        train_recall=train_metrics.get("recall"),
        
        # 验证集指标
        val_mse=val_metrics.get("mse"),
        val_rmse=val_metrics.get("rmse"),
        val_r2=val_metrics.get("r2"),
        val_ic=val_metrics.get("ic"),
        val_rank_ic=val_metrics.get("rank_ic"),
        val_accuracy=val_metrics.get("accuracy"),
        val_auc=val_metrics.get("auc"),
        val_precision=val_metrics.get("precision"),
        val_recall=val_metrics.get("recall"),
        
        # 验证集逐日评估
        val_daily_rankic_mean=val_daily_metrics.get("daily_rankic_mean"),
        val_daily_rankic_std=val_daily_metrics.get("daily_rankic_std"),
        val_daily_rankic_ir=val_daily_metrics.get("daily_rankic_ir")
    )
    
    # 添加额外指标（TopK收益、诊断统计等）
    additional_metrics = {}
    
    # 提取所有TopK和诊断相关的指标
    for key, value in val_daily_metrics.items():
        if key.startswith("top") or key.startswith("diagnostic_"):
            additional_metrics[key] = value
    
    record.additional_metrics = additional_metrics
    
    return record
