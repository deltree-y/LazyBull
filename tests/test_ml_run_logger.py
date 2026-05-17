"""训练运行日志模块测试"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import src.lazybull.ml.run_logger as run_logger_module

from src.lazybull.ml.run_logger import (
    TrainingRunRecord,
    write_training_run_to_csv,
    create_training_run_record_from_training_session
)


@pytest.fixture
def temp_csv_dir():
    """创建临时CSV目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_train_params():
    """示例训练参数"""
    return {
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
        "best_iteration": 150
    }


@pytest.fixture
def sample_data_stats():
    """示例数据统计"""
    return {
        "trade_days_count": 200,
        "total_samples": 100000,
        "samples_after_filter": 95000,
        "train_samples": 76000,
        "val_samples": 19000,
        "val_start_date": "20231001",
        "val_end_date": "20231231",
        "val_ratio": 0.2
    }


@pytest.fixture
def sample_performance_metrics():
    """示例性能指标"""
    return {
        "train": {
            "mse": 0.01,
            "rmse": 0.1,
            "r2": 0.3,
            "ic": 0.05
        },
        "validation": {
            "mse": 0.012,
            "rmse": 0.11,
            "r2": 0.28,
            "ic": 0.045,
            "rank_ic": 0.048
        },
        "validation_daily": {
            "daily_rankic_mean": 0.05,
            "daily_rankic_std": 0.15,
            "daily_rankic_ir": 0.33,
            "top30_return_mean": 0.003,
            "top30_return_std": 0.02,
            "top100_return_mean": 0.0025,
            "top100_return_std": 0.018,
            "diagnostic_全市场收益_逐日均值的均值": 0.0005,
            "diagnostic_Top30_逐日均值_25分位": 0.001
        }
    }


def test_training_run_record_creation():
    """测试训练运行记录创建"""
    record = TrainingRunRecord(
        timestamp="2024-01-01 10:00:00",
        model_version=1,
        start_date="20230101",
        end_date="20231231",
        label_column="y_ret_5",
        task="regression",
        n_estimators=200,
        max_depth=8
    )
    
    assert record.timestamp == "2024-01-01 10:00:00"
    assert record.model_version == 1
    assert record.task == "regression"
    assert record.n_estimators == 200


def test_training_run_record_to_dict():
    """测试训练运行记录转字典"""
    record = TrainingRunRecord(
        timestamp="2024-01-01 10:00:00",
        model_version=1,
        start_date="20230101",
        end_date="20231231",
        label_column="y_ret_5",
        task="regression"
    )
    
    # 添加额外指标
    record.additional_metrics = {
        "top30_return_mean": 0.003,
        "diagnostic_test": 0.001
    }
    
    record_dict = record.to_dict()
    
    assert "timestamp" in record_dict
    assert "model_version" in record_dict
    assert "top30_return_mean" in record_dict
    assert "diagnostic_test" in record_dict
    assert "additional_metrics" not in record_dict  # 应该被扁平化


def test_write_training_run_to_csv_create_new(temp_csv_dir):
    """测试创建新CSV文件并写入第一条记录"""
    csv_path = Path(temp_csv_dir) / "test_runs.csv"
    
    record = TrainingRunRecord(
        timestamp="2024-01-01 10:00:00",
        model_version=1,
        start_date="20230101",
        end_date="20231231",
        label_column="y_ret_5",
        task="regression",
        n_estimators=200,
        train_mse=0.01
    )
    
    write_training_run_to_csv(record, str(csv_path))
    
    # 验证文件存在
    assert csv_path.exists()
    
    # 读取并验证内容
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["model_version"] == 1
    assert df.iloc[0]["task"] == "regression"
    assert df.iloc[0]["n_estimators"] == 200


def test_write_training_run_to_csv_append(temp_csv_dir):
    """测试追加记录到现有CSV"""
    csv_path = Path(temp_csv_dir) / "test_runs.csv"
    
    # 第一条记录
    record1 = TrainingRunRecord(
        timestamp="2024-01-01 10:00:00",
        model_version=1,
        start_date="20230101",
        end_date="20231231",
        label_column="y_ret_5",
        task="regression",
        n_estimators=200
    )
    write_training_run_to_csv(record1, str(csv_path))
    
    # 第二条记录
    record2 = TrainingRunRecord(
        timestamp="2024-01-02 10:00:00",
        model_version=2,
        start_date="20230101",
        end_date="20231231",
        label_column="y_ret_10",
        task="classification",
        n_estimators=300
    )
    write_training_run_to_csv(record2, str(csv_path))
    
    # 验证
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert df.iloc[0]["model_version"] == 1
    assert df.iloc[1]["model_version"] == 2
    assert df.iloc[0]["task"] == "regression"
    assert df.iloc[1]["task"] == "classification"


def test_write_training_run_to_csv_with_custom_path(temp_csv_dir):
    """测试使用自定义路径"""
    custom_dir = Path(temp_csv_dir) / "custom" / "subdir"
    csv_path = custom_dir / "my_runs.csv"
    
    record = TrainingRunRecord(
        timestamp="2024-01-01 10:00:00",
        model_version=1,
        task="regression"
    )
    
    write_training_run_to_csv(record, str(csv_path))
    
    # 验证目录和文件被创建
    assert custom_dir.exists()
    assert csv_path.exists()
    
    df = pd.read_csv(csv_path)
    assert len(df) == 1


def test_write_training_run_to_csv_column_expansion(temp_csv_dir):
    """测试列扩展兼容性：新增字段时自动扩展表头"""
    csv_path = Path(temp_csv_dir) / "test_runs.csv"
    
    # 第一条记录（只有基本字段）
    record1 = TrainingRunRecord(
        timestamp="2024-01-01 10:00:00",
        model_version=1,
        task="regression",
        n_estimators=200
    )
    write_training_run_to_csv(record1, str(csv_path))
    
    # 第二条记录（增加额外指标）
    record2 = TrainingRunRecord(
        timestamp="2024-01-02 10:00:00",
        model_version=2,
        task="regression",
        n_estimators=300
    )
    record2.additional_metrics = {
        "new_field_1": 0.123,
        "new_field_2": 0.456
    }
    write_training_run_to_csv(record2, str(csv_path))
    
    # 验证：表头扩展，旧记录新字段为空
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert "new_field_1" in df.columns
    assert "new_field_2" in df.columns
    assert pd.isna(df.iloc[0]["new_field_1"])  # 第一条记录的新字段应为NaN
    assert df.iloc[1]["new_field_1"] == 0.123  # 第二条记录有值
    
    # 第三条记录（不包含新字段）
    record3 = TrainingRunRecord(
        timestamp="2024-01-03 10:00:00",
        model_version=3,
        task="classification",
        n_estimators=250
    )
    write_training_run_to_csv(record3, str(csv_path))
    
    # 验证：第三条记录也能正常写入，新字段为空
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert pd.isna(df.iloc[2]["new_field_1"])


def test_write_training_run_to_csv_column_expansion_reads_full_csv_with_low_memory_false(
    temp_csv_dir, monkeypatch
):
    """列扩展时整表回读应关闭 low_memory，避免混合类型列触发 DtypeWarning。"""
    csv_path = Path(temp_csv_dir) / "test_runs.csv"

    record1 = TrainingRunRecord(
        timestamp="2024-01-01 10:00:00",
        model_version=1,
        task="regression",
        n_estimators=200,
    )
    write_training_run_to_csv(record1, str(csv_path))

    captured_kwargs = []
    original_read_csv = run_logger_module.pd.read_csv

    def _wrapped_read_csv(*args, **kwargs):
        captured_kwargs.append(kwargs.copy())
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(run_logger_module.pd, "read_csv", _wrapped_read_csv)

    record2 = TrainingRunRecord(
        timestamp="2024-01-02 10:00:00",
        model_version=2,
        task="regression",
        n_estimators=300,
    )
    record2.additional_metrics = {"mixed_column": "text-value"}
    write_training_run_to_csv(record2, str(csv_path))

    full_read_kwargs = [kwargs for kwargs in captured_kwargs if kwargs.get("nrows") is None]

    assert full_read_kwargs, "列扩展路径应触发一次整表回读"
    assert any(kwargs.get("low_memory") is False for kwargs in full_read_kwargs)


def test_create_training_run_record_from_training_session(
    sample_train_params,
    sample_data_stats,
    sample_performance_metrics
):
    """测试从训练会话创建记录"""
    record = create_training_run_record_from_training_session(
        start_date="20230101",
        end_date="20231231",
        label_column="y_ret_5",
        task="regression",
        model_version=1,
        train_params=sample_train_params,
        data_stats=sample_data_stats,
        performance_metrics=sample_performance_metrics
    )
    
    # 验证基本信息
    assert record.start_date == "20230101"
    assert record.end_date == "20231231"
    assert record.label_column == "y_ret_5"
    assert record.task == "regression"
    assert record.model_version == 1
    
    # 验证训练参数
    assert record.n_estimators == 200
    assert record.max_depth == 8
    assert record.learning_rate == 0.05
    assert record.label_transform == "cs_zscore"
    assert record.winsorize_p == 0.01
    assert record.best_iteration == 150
    
    # 验证数据统计
    assert record.trade_days_count == 200
    assert record.total_samples == 100000
    assert record.train_samples == 76000
    assert record.val_samples == 19000
    
    # 验证性能指标
    assert record.train_mse == 0.01
    assert record.train_ic == 0.05
    assert record.val_mse == 0.012
    assert record.val_rank_ic == 0.048
    
    # 验证逐日评估
    assert record.val_daily_rankic_mean == 0.05
    assert record.val_daily_rankic_ir == 0.33
    
    # 验证额外指标
    assert "top30_return_mean" in record.additional_metrics
    assert record.additional_metrics["top30_return_mean"] == 0.003
    assert "diagnostic_全市场收益_逐日均值的均值" in record.additional_metrics


def test_create_classification_record():
    """测试创建分类任务记录"""
    train_params = {
        "task": "classification",
        "pos_topk": 300,
        "scale_pos_weight": 5.0,
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
        "n_jobs": -1
    }
    
    data_stats = {
        "trade_days_count": 150,
        "total_samples": 80000,
        "samples_after_filter": 75000,
        "train_samples": 60000,
        "val_samples": 15000,
        "val_start_date": "20231101",
        "val_end_date": "20231231",
        "val_ratio": 0.2
    }
    
    performance_metrics = {
        "train": {
            "accuracy": 0.85,
            "auc": 0.75,
            "precision": 0.3,
            "recall": 0.8
        },
        "validation": {
            "accuracy": 0.83,
            "auc": 0.73,
            "precision": 0.28,
            "recall": 0.78
        },
        "validation_daily": {
            "daily_rankic_mean": 0.06,
            "daily_rankic_std": 0.12,
            "daily_rankic_ir": 0.5
        }
    }
    
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
    
    # 验证分类任务特定字段
    assert record.task == "classification"
    assert record.pos_topk == 300
    assert record.scale_pos_weight == 5.0
    assert record.scale_pos_weight_mode == "auto"
    
    # 验证分类指标
    assert record.train_accuracy == 0.85
    assert record.train_auc == 0.75
    assert record.val_accuracy == 0.83
    assert record.val_auc == 0.73


def test_full_workflow(temp_csv_dir):
    """测试完整工作流：创建记录 -> 写入CSV -> 验证"""
    csv_path = Path(temp_csv_dir) / "workflow_test.csv"
    
    # 模拟多次训练
    for i in range(3):
        train_params = {
            "task": "regression",
            "n_estimators": 200 + i * 50,
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
            "best_iteration": 150 + i * 10
        }
        
        data_stats = {
            "trade_days_count": 200,
            "total_samples": 100000,
            "samples_after_filter": 95000,
            "train_samples": 76000,
            "val_samples": 19000,
            "val_start_date": "20231001",
            "val_end_date": "20231231",
            "val_ratio": 0.2
        }
        
        performance_metrics = {
            "train": {"mse": 0.01 + i * 0.001, "ic": 0.05 + i * 0.01},
            "validation": {"mse": 0.012 + i * 0.001, "ic": 0.045 + i * 0.01},
            "validation_daily": {"daily_rankic_mean": 0.05 + i * 0.01}
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
    
    # 验证最终CSV
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert list(df["model_version"]) == [1, 2, 3]
    assert list(df["n_estimators"]) == [200, 250, 300]
    assert df.iloc[0]["train_mse"] == pytest.approx(0.01, rel=1e-6)
    assert df.iloc[2]["train_mse"] == pytest.approx(0.012, rel=1e-6)
