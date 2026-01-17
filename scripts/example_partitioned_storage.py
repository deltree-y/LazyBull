#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用按日分区存储的示例脚本

演示如何使用新的按日分区功能来拉取和存储数据
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient


def example_save_daily_partitioned():
    """示例：按日分区保存数据"""
    setup_logger(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("示例：使用按日分区存储")
    logger.info("=" * 60)
    
    # 初始化（现在默认使用partitioned存储）
    storage = Storage()
    client = TushareClient()
    
    # 示例1：拉取并保存单日数据
    logger.info("\n1. 拉取单日数据并按日期分区保存")
    trade_date = "20231201"
    
    # 拉取日线行情
    daily_df = client.get_daily(trade_date=trade_date)
    if len(daily_df) > 0:
        storage.save_raw_by_date(daily_df, "daily", trade_date)
        logger.info(f"已保存 {len(daily_df)} 条日线行情数据到 data/raw/daily/{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}.parquet")
    
    # 拉取每日指标
    daily_basic_df = client.get_daily_basic(trade_date=trade_date)
    if len(daily_basic_df) > 0:
        storage.save_raw_by_date(daily_basic_df, "daily_basic", trade_date)
        logger.info(f"已保存 {len(daily_basic_df)} 条每日指标数据")
    
    # 示例2：拉取一段时间的数据并按日分区保存
    logger.info("\n2. 拉取多日数据并逐日分区保存")
    
    # 获取交易日历
    trade_cal = client.get_trade_cal("20231201", "20231205", "SSE")
    trading_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
    
    for date in trading_dates[:3]:  # 仅示例前3天
        logger.info(f"处理 {date}...")
        
        # 拉取该日数据
        daily_df = client.get_daily(trade_date=date)
        if len(daily_df) > 0:
            storage.save_raw_by_date(daily_df, "daily", date)
            logger.info(f"  保存了 {len(daily_df)} 条记录")
    
    # 示例3：加载分区数据
    logger.info("\n3. 加载分区数据")
    
    # 加载单日数据
    single_day = storage.load_raw_by_date("daily", trade_date)
    if single_day is not None:
        logger.info(f"加载单日数据: {len(single_day)} 条记录")
    
    # 加载日期范围数据
    range_data = storage.load_raw_by_date_range("daily", "20231201", "20231205")
    if range_data is not None:
        logger.info(f"加载范围数据: {len(range_data)} 条记录")
    
    # 示例4：列出所有分区
    logger.info("\n4. 列出所有分区")
    partitions = storage.list_partitions("raw", "daily")
    logger.info(f"daily数据的分区日期: {partitions[:5]}...")  # 显示前5个
    
    logger.info("\n" + "=" * 60)
    logger.info("示例完成！")
    logger.info("=" * 60)


def example_migrate_from_monolithic():
    """示例：从单文件存储迁移到分区存储"""
    setup_logger(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("示例：数据迁移（单文件 -> 分区）")
    logger.info("=" * 60)
    
    storage = Storage()
    
    # 假设已有单文件存储的daily数据
    logger.info("\n1. 加载现有的单文件数据")
    daily_df = storage.load_raw("daily")
    
    if daily_df is None:
        logger.info("没有找到现有数据，跳过迁移")
        return
    
    logger.info(f"加载了 {len(daily_df)} 条记录")
    
    # 按日期分组并保存到分区
    logger.info("\n2. 按日期分区并保存")
    
    if 'trade_date' in daily_df.columns:
        # 确保日期格式
        import pandas as pd
        if not pd.api.types.is_datetime64_any_dtype(daily_df['trade_date']):
            daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'], format='%Y%m%d')
        
        # 按日期分组
        grouped = daily_df.groupby(daily_df['trade_date'].dt.strftime('%Y%m%d'))
        
        count = 0
        for date_str, group_df in grouped:
            storage.save_raw_by_date(group_df, "daily", date_str)
            count += 1
            if count % 10 == 0:
                logger.info(f"  已迁移 {count} 个分区...")
        
        logger.info(f"迁移完成！共 {count} 个分区")
        
        # 列出迁移后的分区
        partitions = storage.list_partitions("raw", "daily")
        logger.info(f"分区数量: {len(partitions)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("迁移完成！")
    logger.info("注意：迁移后可以删除原始单文件以释放空间")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="按日分区存储示例")
    parser.add_argument(
        "--mode",
        choices=["save", "migrate"],
        default="save",
        help="运行模式：save=保存示例，migrate=迁移示例"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "save":
            example_save_daily_partitioned()
        elif args.mode == "migrate":
            example_migrate_from_monolithic()
    except Exception as e:
        logger.exception(f"示例运行失败: {str(e)}")
        sys.exit(1)
