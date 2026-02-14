"""集成测试：模拟训练脚本的日志记录流程"""

import tempfile
from pathlib import Path

import pandas as pd

from src.lazybull.ml.run_logger import (
    create_training_run_record_from_training_session,
    write_training_run_to_csv
)


def test_integration_regression_training():
    """集成测试：模拟回归任务训练并记录日志"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "ml_train_runs.csv"
        
        # 模拟训练参数（与实际训练脚本一致）
        train_params = {
            "task": "regression",
            "label_transform": "cs_zscore",
            "winsorize_p": 0.01,
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 30,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "best_iteration": 158,
            "scale_pos_weight_manual": False
        }
        
        # 模拟数据统计
        data_stats = {
            "trade_days_count": 244,
            "total_samples": 120000,
            "samples_after_filter": 115000,
            "train_samples": 92000,
            "val_samples": 23000,
            "val_start_date": "20231001",
            "val_end_date": "20231231",
            "val_ratio": 0.2
        }
        
        # 模拟性能指标
        performance_metrics = {
            "train": {
                "mse": 0.0123,
                "rmse": 0.1109,
                "r2": 0.2854,
                "ic": 0.0512
            },
            "validation": {
                "mse": 0.0145,
                "rmse": 0.1204,
                "r2": 0.2512,
                "ic": 0.0448,
                "rank_ic": 0.0521
            },
            "validation_daily": {
                "daily_rankic_mean": 0.0524,
                "daily_rankic_std": 0.1523,
                "daily_rankic_ir": 0.3442
            }
        }
        
        # 创建记录
        record = create_training_run_record_from_training_session(
            start_date="20230101",
            end_date="20231231",
            label_column="y_ret_5",
            task="regression",
            model_version=1,
            train_params=train_params,
            data_stats=data_stats,
            performance_metrics=performance_metrics
        )
        
        # 写入CSV
        write_training_run_to_csv(record, str(csv_path))
        
        # 验证
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        
        # 验证关键字段
        row = df.iloc[0]
        assert row["model_version"] == 1
        assert row["task"] == "regression"
        assert row["label_column"] == "y_ret_5"
        assert row["label_transform"] == "cs_zscore"
        assert row["n_estimators"] == 200
        assert row["best_iteration"] == 158
        assert abs(row["val_rank_ic"] - 0.0521) < 1e-6
        
        print("✅ 回归任务集成测试通过")
        print(f"   CSV 行数: {len(df)}")
        print(f"   CSV 列数: {len(df.columns)}")
        print(f"   模型版本: {row['model_version']}")
        print(f"   验证集 RankIC: {row['val_rank_ic']:.4f}")


def test_integration_classification_training():
    """集成测试：模拟分类任务训练并记录日志"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "ml_train_runs.csv"
        
        # 模拟训练参数
        train_params = {
            "task": "classification",
            "pos_topk": 300,
            "scale_pos_weight": 4.8333,
            "scale_pos_weight_manual": False,  # 自动计算
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 30,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "best_iteration": 142
        }
        
        # 模拟数据统计
        data_stats = {
            "trade_days_count": 244,
            "total_samples": 120000,
            "samples_after_filter": 115000,
            "train_samples": 92000,
            "val_samples": 23000,
            "val_start_date": "20231001",
            "val_end_date": "20231231",
            "val_ratio": 0.2
        }
        
        # 模拟性能指标（分类任务）
        performance_metrics = {
            "train": {
                "accuracy": 0.8645,
                "auc": 0.7821,
                "precision": 0.3125,
                "recall": 0.8124
            },
            "validation": {
                "accuracy": 0.8512,
                "auc": 0.7645,
                "precision": 0.2985,
                "recall": 0.7954
            },
            "validation_daily": {
                "daily_rankic_mean": 0.0612,
                "daily_rankic_std": 0.1421,
                "daily_rankic_ir": 0.4308,
                "top30_return_mean": 0.0032,
                "top30_return_std": 0.0215,
                "top100_return_mean": 0.0028,
                "top100_return_std": 0.0198,
                "top300_return_mean": 0.0025,
                "top300_return_std": 0.0184,
                "diagnostic_全市场收益_逐日均值的均值": 0.0006,
                "diagnostic_Top30_逐日均值_25分位": 0.0015
            }
        }
        
        # 创建记录
        record = create_training_run_record_from_training_session(
            start_date="20230101",
            end_date="20231231",
            label_column="y_ret_20",
            task="classification",
            model_version=5,
            train_params=train_params,
            data_stats=data_stats,
            performance_metrics=performance_metrics
        )
        
        # 写入CSV
        write_training_run_to_csv(record, str(csv_path))
        
        # 验证
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        
        # 验证关键字段
        row = df.iloc[0]
        assert row["model_version"] == 5
        assert row["task"] == "classification"
        assert row["label_column"] == "y_ret_20"
        assert row["pos_topk"] == 300
        assert abs(row["scale_pos_weight"] - 4.8333) < 1e-4
        assert row["scale_pos_weight_mode"] == "auto"
        assert abs(row["val_auc"] - 0.7645) < 1e-6
        assert abs(row["top30_return_mean"] - 0.0032) < 1e-6
        
        print("✅ 分类任务集成测试通过")
        print(f"   CSV 行数: {len(df)}")
        print(f"   CSV 列数: {len(df.columns)}")
        print(f"   模型版本: {row['model_version']}")
        print(f"   验证集 AUC: {row['val_auc']:.4f}")
        print(f"   Top30 收益: {row['top30_return_mean']:.6f}")


def test_integration_multiple_runs():
    """集成测试：模拟多次训练累积日志"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "ml_train_runs.csv"
        
        # 模拟 5 次不同配置的训练
        configs = [
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.05},
            {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.03},
            {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.05},
        ]
        
        for i, config in enumerate(configs):
            train_params = {
                "task": "regression",
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "learning_rate": config["learning_rate"],
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "early_stopping_rounds": 30,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1,
                "best_iteration": 150 + i * 5,
                "scale_pos_weight_manual": False
            }
            
            data_stats = {
                "trade_days_count": 244,
                "total_samples": 120000,
                "samples_after_filter": 115000,
                "train_samples": 92000,
                "val_samples": 23000,
                "val_start_date": "20231001",
                "val_end_date": "20231231",
                "val_ratio": 0.2
            }
            
            # 模拟不同的性能指标
            performance_metrics = {
                "train": {
                    "mse": 0.012 + i * 0.001,
                    "ic": 0.05 + i * 0.005
                },
                "validation": {
                    "mse": 0.014 + i * 0.001,
                    "ic": 0.045 + i * 0.005,
                    "rank_ic": 0.050 + i * 0.005
                },
                "validation_daily": {
                    "daily_rankic_mean": 0.052 + i * 0.003
                }
            }
            
            record = create_training_run_record_from_training_session(
                start_date="20230101",
                end_date="20231231",
                label_column="y_ret_5",
                task="regression",
                model_version=i + 1,
                train_params=train_params,
                data_stats=data_stats,
                performance_metrics=performance_metrics
            )
            
            write_training_run_to_csv(record, str(csv_path))
        
        # 验证
        df = pd.read_csv(csv_path)
        assert len(df) == 5
        
        print("✅ 多次训练累积日志测试通过")
        print(f"   总训练次数: {len(df)}")
        print(f"   CSV 列数: {len(df.columns)}")
        print("\n   模型对比:")
        print(df[['model_version', 'n_estimators', 'max_depth', 'learning_rate', 'val_rank_ic']].to_string(index=False))
        
        # 找出最佳模型
        best_idx = df['val_rank_ic'].idxmax()
        best_model = df.iloc[best_idx]
        print(f"\n   最佳模型: v{int(best_model['model_version'])}")
        print(f"   配置: n_estimators={int(best_model['n_estimators'])}, max_depth={int(best_model['max_depth'])}, lr={best_model['learning_rate']:.3f}")
        print(f"   验证集 RankIC: {best_model['val_rank_ic']:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("训练运行日志集成测试")
    print("=" * 60)
    print()
    
    test_integration_regression_training()
    print()
    
    test_integration_classification_training()
    print()
    
    test_integration_multiple_runs()
    print()
    
    print("=" * 60)
    print("✅ 所有集成测试通过！")
    print("=" * 60)
