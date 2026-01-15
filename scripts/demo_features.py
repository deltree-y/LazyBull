#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示脚本：使用模拟数据展示特征构建流程
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage
from src.lazybull.features import FeatureBuilder


def create_mock_data():
    """创建模拟数据用于演示"""
    
    # 1. 交易日历 (20个交易日)
    dates = pd.date_range('2023-01-01', periods=20, freq='B')
    trade_cal = pd.DataFrame({
        'exchange': ['SSE'] * len(dates),
        'cal_date': dates.strftime('%Y%m%d').tolist(),
        'is_open': [1] * len(dates)
    })
    
    # 2. 股票基本信息
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH'],
        'name': ['平安银行', '万科A', '浦发银行', '贵州茅台'],
        'list_date': ['20100101', '20100101', '20100101', '20100101']
    })
    
    # 3. 日线行情
    stocks = stock_basic['ts_code'].tolist()
    daily_data = []
    
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        for i, stock in enumerate(stocks):
            base_price = 10.0 + i * 5
            # 模拟价格上涨趋势
            close = base_price * (1 + 0.001 * (dates.tolist().index(date)))
            pre_close = base_price * (1 + 0.001 * max(0, dates.tolist().index(date) - 1))
            pct_chg = ((close - pre_close) / pre_close) * 100 if pre_close > 0 else 0
            
            daily_data.append({
                'ts_code': stock,
                'trade_date': date_str,
                'close': close,
                'pre_close': pre_close,
                'pct_chg': pct_chg,
                'vol': 1000000 + i * 100000,
                'amount': (1000000 + i * 100000) * close
            })
    
    daily_data = pd.DataFrame(daily_data)
    
    # 4. 复权因子 (简化为1.0)
    adj_factor = daily_data[['ts_code', 'trade_date']].copy()
    adj_factor['adj_factor'] = 1.0
    
    return trade_cal, stock_basic, daily_data, adj_factor


def main():
    """主函数"""
    # 初始化日志
    setup_logger(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("特征构建演示")
    logger.info("=" * 60)
    
    # 创建模拟数据
    logger.info("创建模拟数据...")
    trade_cal, stock_basic, daily_data, adj_factor = create_mock_data()
    
    logger.info(f"交易日历: {len(trade_cal)} 条")
    logger.info(f"股票基本信息: {len(stock_basic)} 条")
    logger.info(f"日线行情: {len(daily_data)} 条")
    logger.info(f"复权因子: {len(adj_factor)} 条")
    
    # 初始化特征构建器
    builder = FeatureBuilder(
        min_list_days=10,  # 降低要求以适应模拟数据
        horizon=5
    )
    
    # 选择一个中间的交易日
    trade_date = trade_cal.iloc[10]['cal_date']
    logger.info(f"构建 {trade_date} 的特征...")
    
    # 构建特征
    features = builder.build_features_for_day(
        trade_date=trade_date,
        trade_cal=trade_cal,
        daily_data=daily_data,
        adj_factor=adj_factor,
        stock_basic=stock_basic
    )
    
    # 显示结果
    logger.info("=" * 60)
    logger.info("特征构建结果")
    logger.info("=" * 60)
    logger.info(f"样本数: {len(features)}")
    logger.info(f"特征列: {features.columns.tolist()}")
    logger.info("")
    logger.info("前3行数据:")
    logger.info("")
    
    # 选择部分列显示
    display_cols = [
        'trade_date', 'ts_code', 'name',
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        'y_ret_5',
        'is_st', 'suspend', 'limit_up', 'limit_down'
    ]
    
    # 过滤掉不存在的列
    display_cols = [col for col in display_cols if col in features.columns]
    
    print(features[display_cols].head(3).to_string(index=False))
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("特征统计信息")
    logger.info("=" * 60)
    
    # 数值特征统计
    numeric_features = [
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        'vol_ratio_5', 'vol_ratio_10', 'vol_ratio_20',
        'y_ret_5'
    ]
    numeric_features = [col for col in numeric_features if col in features.columns]
    
    print(features[numeric_features].describe().to_string())
    
    logger.info("")
    logger.info("=" * 60)
    
    # 保存示例特征
    storage = Storage()
    storage.save_cs_train_day(features, trade_date)
    logger.info(f"特征已保存到: {storage.features_path / 'cs_train' / f'{trade_date}.parquet'}")
    
    # 验证加载
    loaded_features = storage.load_cs_train_day(trade_date)
    logger.info(f"重新加载验证: {len(loaded_features)} 条记录")
    
    logger.info("=" * 60)
    logger.info("演示完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
