#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估面板功能演示脚本

演示如何使用评估面板功能评估 ML 信号的预测质量。
本脚本创建模拟数据和模型，运行评估面板并输出 CSV 文件。
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lazybull.ml import ModelRegistry
from src.lazybull.signals import MLSignal
from src.lazybull.universe import BasicUniverse


class MockMLModel:
    """模拟 ML 模型"""
    
    def predict(self, X):
        """基于第一个特征返回预测值"""
        if len(X.columns) > 0:
            return X.iloc[:, 0].values * 0.1
        return np.zeros(len(X))


def create_mock_data(n_dates=10, n_stocks=100):
    """创建模拟数据
    
    Args:
        n_dates: 交易日数量
        n_stocks: 每日股票数量
        
    Returns:
        features_by_date: 按日期组织的特征数据字典
        trading_dates: 交易日列表
        universe: 股票池
    """
    # 生成日期
    base_date = pd.Timestamp('20230101')
    trading_dates = [base_date + pd.Timedelta(days=i) for i in range(n_dates)]
    
    # 股票池
    stocks = [f'stock_{i:04d}' for i in range(n_stocks)]
    
    # 为每个日期生成特征数据
    features_by_date = {}
    for date in trading_dates:
        date_str = date.strftime('%Y%m%d')
        
        # 生成特征（f1 是主要预测特征，与收益正相关）
        f1 = np.random.randn(n_stocks) + 0.5  # 均值 0.5
        f2 = np.random.randn(n_stocks)
        f3 = np.random.randn(n_stocks)
        
        # 生成真实收益（与 f1 正相关 + 噪声）
        y_ret_5 = f1 * 0.02 + np.random.randn(n_stocks) * 0.01
        y_ret_10 = f1 * 0.04 + np.random.randn(n_stocks) * 0.015
        y_ret_20 = f1 * 0.08 + np.random.randn(n_stocks) * 0.02
        
        features_df = pd.DataFrame({
            'ts_code': stocks,
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'y_ret_5': y_ret_5,
            'y_ret_10': y_ret_10,
            'y_ret_20': y_ret_20,
        })
        
        features_by_date[date_str] = features_df
    
    return features_by_date, trading_dates, stocks


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("评估面板功能演示")
    logger.info("=" * 60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        models_dir = tmpdir / "models"
        output_dir = tmpdir / "reports"
        models_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        # 1. 创建并注册模型
        logger.info("创建模拟 ML 模型...")
        registry = ModelRegistry(models_dir=str(models_dir))
        model = MockMLModel()
        version = registry.register_model(
            model=model,
            model_type="xgboost",
            train_start_date="20230101",
            train_end_date="20231231",
            feature_columns=["f1", "f2", "f3"],
            label_column="y_ret_5",
            n_samples=10000,
            train_params={"n_estimators": 100}
        )
        logger.info(f"模型已注册: v{version}")
        
        # 2. 创建 ML 信号
        logger.info("创建 ML 信号...")
        signal = MLSignal(
            top_n=20,
            model_version=version,
            models_dir=str(models_dir),
            weight_method="equal",
            verbose=False
        )
        
        # 3. 创建模拟数据
        logger.info("创建模拟数据...")
        features_by_date, trading_dates, stocks = create_mock_data(n_dates=20, n_stocks=200)
        logger.info(f"创建了 {len(trading_dates)} 个交易日，每日 {len(stocks)} 只股票")
        
        # 4. 创建简单的股票池（所有股票都在池内）
        class SimpleUniverse:
            def __init__(self, stocks):
                self.stocks = stocks
            
            def get_stocks(self, date):
                return self.stocks
        
        universe = SimpleUniverse(stocks)
        
        # 5. 导入评估面板函数
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_ml_backtest", 
            project_root / "scripts" / "run_ml_backtest.py"
        )
        run_ml_backtest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_ml_backtest)
        
        export_evaluation_panel = run_ml_backtest.export_evaluation_panel
        
        # 6. 运行评估面板导出
        logger.info("运行评估面板导出...")
        
        # 创建一个简单的 args 对象
        class Args:
            start_date = "20230101"
            end_date = "20230120"
            model_version = version
            top_n = 20
            weight_method = "equal"
            rebalance_freq = 5
            initial_capital = 500000
            sell_timing = "open"
        
        args = Args()
        
        export_evaluation_panel(
            signal=signal,
            universe_obj=universe,
            features_by_date=features_by_date,
            trading_dates=trading_dates,
            label_column='y_ret_5',
            output_dir=str(output_dir),
            output_name='demo',
            n_groups=10,
            topk=20,
            args=args
        )
        
        # 7. 检查输出文件
        logger.info("=" * 60)
        logger.info("检查输出文件:")
        logger.info("=" * 60)
        
        daily_file = output_dir / "demo_eval_daily.csv"
        groups_file = output_dir / "demo_eval_groups.csv"
        summary_file = output_dir / "demo_eval_summary.csv"
        
        for file_path in [daily_file, groups_file, summary_file]:
            if file_path.exists():
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                logger.info(f"✓ {file_path.name}")
                logger.info(f"  - 行数: {len(df)}")
                logger.info(f"  - 列: {', '.join(df.columns.tolist())}")
                logger.info(f"  - 前3行预览:")
                print(df.head(3).to_string(index=False))
                logger.info("")
            else:
                logger.warning(f"✗ {file_path.name} 未生成")
        
        # 8. 显示汇总指标
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file, encoding='utf-8-sig')
            logger.info("=" * 60)
            logger.info("汇总指标:")
            logger.info("=" * 60)
            if len(summary_df) > 0:
                row = summary_df.iloc[-1]
                logger.info(f"评估天数: {row.get('评估天数', 'N/A')}")
                logger.info(f"RankIC均值: {row.get('RankIC均值', 'N/A'):.4f}" if pd.notna(row.get('RankIC均值')) else "RankIC均值: N/A")
                logger.info(f"RankIC标准差: {row.get('RankIC标准差', 'N/A'):.4f}" if pd.notna(row.get('RankIC标准差')) else "RankIC标准差: N/A")
                logger.info(f"RankIC_IR: {row.get('RankIC_IR', 'N/A'):.4f}" if pd.notna(row.get('RankIC_IR')) else "RankIC_IR: N/A")
                logger.info(f"TopK平均收益: {row.get('TopK平均收益', 'N/A'):.4f}" if pd.notna(row.get('TopK平均收益')) else "TopK平均收益: N/A")
                logger.info(f"多空平均收益: {row.get('多空平均收益', 'N/A'):.4f}" if pd.notna(row.get('多空平均收益')) else "多空平均收益: N/A")
        
        logger.info("=" * 60)
        logger.info("演示完成！")
        logger.info("=" * 60)
        logger.info(f"临时文件保存在: {tmpdir}")
        logger.info("（临时目录会在脚本结束后自动删除）")


if __name__ == "__main__":
    main()
