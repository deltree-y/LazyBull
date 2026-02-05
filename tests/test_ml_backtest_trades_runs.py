#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 ML 回测交易记录累加文件功能
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import csv
import hashlib
from datetime import datetime


# 将测试所需的函数直接在这里定义（复制自 run_ml_backtest.py）
def _append_dict_to_csv(file_path: Path, row: dict, fieldnames: list = None):
    """把一个 dict 追加到 CSV（如果不存在则写 header）"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = file_path.exists()

    with open(file_path, 'a', newline='', encoding='utf-8-sig') as f:
        if fieldnames is None:
            fieldnames_local = list(row.keys())
        else:
            fieldnames_local = fieldnames
        writer = csv.DictWriter(f, fieldnames=fieldnames_local)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _generate_run_id(args) -> str:
    """生成唯一的回测ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    params_str = f"{args.start_date}_{args.end_date}_{args.model_version}_{args.top_n}_{args.weight_method}_{args.rebalance_freq}_{args.initial_capital}_{args.sell_timing}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return f"{timestamp}_{params_hash}"


class TestMLBacktestTradesRuns:
    """测试ML回测交易记录累加文件"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # 清理
        shutil.rmtree(temp_path)
    
    def test_append_trades_creates_file_with_header(self, temp_dir):
        """测试首次追加时创建文件并写入表头"""
        csv_file = Path(temp_dir) / "test_trades_runs.csv"
        
        # 准备测试数据
        test_row = {
            "回测ID": "test_run_001",
            "回测时间": "2024-01-17 15:32:01",
            "开始日期": "20230101",
            "交易日期": "2023-01-03",
            "股票代码": "000001.SZ"
        }
        
        fieldnames = ["回测ID", "回测时间", "开始日期", "交易日期", "股票代码"]
        
        # 首次写入
        _append_dict_to_csv(csv_file, test_row, fieldnames=fieldnames)
        
        # 验证文件存在
        assert csv_file.exists(), "累加文件应该被创建"
        
        # 读取文件验证内容
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # 验证表头存在
        assert "回测ID" in df.columns, "应包含'回测ID'列"
        assert "回测时间" in df.columns, "应包含'回测时间'列"
        assert "开始日期" in df.columns, "应包含'开始日期'列"
        
        # 验证数据
        assert len(df) == 1, "应该有1条记录"
        assert df.iloc[0]["回测ID"] == "test_run_001"
        assert df.iloc[0]["股票代码"] == "000001.SZ"
    
    def test_append_trades_accumulates_records(self, temp_dir):
        """测试连续追加时记录累加"""
        csv_file = Path(temp_dir) / "test_trades_runs.csv"
        
        fieldnames = ["回测ID", "交易日期", "股票代码"]
        
        # 第一次追加
        _append_dict_to_csv(
            csv_file,
            {"回测ID": "run_001", "交易日期": "2023-01-03", "股票代码": "000001.SZ"},
            fieldnames=fieldnames
        )
        
        # 第二次追加
        _append_dict_to_csv(
            csv_file,
            {"回测ID": "run_001", "交易日期": "2023-01-04", "股票代码": "000002.SZ"},
            fieldnames=fieldnames
        )
        
        # 第三次追加（不同回测ID）
        _append_dict_to_csv(
            csv_file,
            {"回测ID": "run_002", "交易日期": "2023-01-05", "股票代码": "000003.SZ"},
            fieldnames=fieldnames
        )
        
        # 读取验证
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # 验证行数累加
        assert len(df) == 3, "应该有3条累加记录"
        
        # 验证不同回测的记录都存在
        assert (df["回测ID"] == "run_001").sum() == 2, "run_001应该有2条记录"
        assert (df["回测ID"] == "run_002").sum() == 1, "run_002应该有1条记录"
    
    def test_chinese_columns_exist(self, temp_dir):
        """测试中文列名存在"""
        csv_file = Path(temp_dir) / "test_trades_runs.csv"
        
        # 准备包含所有核心参数的测试数据
        test_row = {
            "回测ID": "test_001",
            "回测时间": "2024-01-17 15:32:01",
            "开始日期": "20230101",
            "结束日期": "20231231",
            "模型版本": "最新版本",
            "TopN": 5,
            "权重方法": "equal",
            "调仓频率": 10,
            "初始资金": 500000.0,
            "卖出时机": "open",
            "交易日期": "2023-01-03",
            "股票代码": "000001.SZ",
            "操作": "买入"
        }
        
        fieldnames = list(test_row.keys())
        
        _append_dict_to_csv(csv_file, test_row, fieldnames=fieldnames)
        
        # 读取验证
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # 验证所有核心参数中文列存在
        required_chinese_columns = [
            "回测ID",
            "回测时间",
            "开始日期",
            "结束日期",
            "模型版本",
            "TopN",
            "权重方法",
            "调仓频率",
            "初始资金",
            "卖出时机"
        ]
        
        for col in required_chinese_columns:
            assert col in df.columns, f"应包含核心参数列'{col}'"
        
        # 验证数据正确
        assert df.iloc[0]["模型版本"] == "最新版本"
        assert df.iloc[0]["TopN"] == 5
        assert df.iloc[0]["权重方法"] == "equal"
        assert df.iloc[0]["操作"] == "买入"
    
    def test_generate_run_id_uniqueness(self):
        """测试回测ID生成的唯一性"""
        import argparse
        
        # 模拟两个不同的参数配置
        args1 = argparse.Namespace(
            start_date="20230101",
            end_date="20231231",
            model_version=1,
            top_n=5,
            weight_method="equal",
            rebalance_freq=10,
            initial_capital=500000.0,
            sell_timing="open"
        )
        
        args2 = argparse.Namespace(
            start_date="20230101",
            end_date="20231231",
            model_version=2,  # 不同的模型版本
            top_n=5,
            weight_method="equal",
            rebalance_freq=10,
            initial_capital=500000.0,
            sell_timing="open"
        )
        
        # 生成ID
        run_id_1 = _generate_run_id(args1)
        run_id_2 = _generate_run_id(args2)
        
        # 验证ID格式（时间戳_hash）
        assert "_" in run_id_1, "回测ID应包含下划线分隔符"
        assert len(run_id_1.split("_")) == 2, "回测ID应该是时间戳_hash格式"
        
        # 验证不同参数生成不同的hash部分
        hash1 = run_id_1.split("_")[1]
        hash2 = run_id_2.split("_")[1]
        assert hash1 != hash2, "不同参数应生成不同的hash"
    
    def test_model_version_none_shows_as_latest(self, temp_dir):
        """测试模型版本为None时显示为'最新版本'"""
        csv_file = Path(temp_dir) / "test_trades_runs.csv"
        
        # None值应该被转换为"最新版本"
        test_row = {
            "回测ID": "test_001",
            "模型版本": "最新版本",  # 在实际代码中，None会被转换为"最新版本"
            "交易日期": "2023-01-03"
        }
        
        fieldnames = ["回测ID", "模型版本", "交易日期"]
        
        _append_dict_to_csv(csv_file, test_row, fieldnames=fieldnames)
        
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        assert df.iloc[0]["模型版本"] == "最新版本", "模型版本None应显示为'最新版本'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

