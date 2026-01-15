#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据拉取脚本
从TuShare拉取基础数据并保存到本地
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient


def pull_basic_data(client: TushareClient, storage: Storage):
    """拉取基础数据（交易日历、股票列表）
    
    这些数据不需要按日分区
    """
    # 1. 拉取交易日历（2020-2024）
    logger.info("拉取交易日历...")
    trade_cal = client.get_trade_cal(
        start_date="20200101",
        end_date="20241231",
        exchange="SSE"
    )
    storage.save_raw(trade_cal, "trade_cal")
    logger.info(f"交易日历拉取完成: {len(trade_cal)} 条记录")
    
    # 2. 拉取股票基本信息
    logger.info("拉取股票基本信息...")
    stock_basic = client.get_stock_basic(list_status="L")
    storage.save_raw(stock_basic, "stock_basic")
    logger.info(f"股票基本信息拉取完成: {len(stock_basic)} 条记录")
    
    return trade_cal


def pull_daily_data_monolithic(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str
):
    """使用单文件方式拉取日线数据（旧方式）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
    """
    logger.info(f"拉取日线行情（{start_date}-{end_date}）...")
    logger.info("使用单文件存储模式")
    
    # 拉取日线行情
    daily_data = client.get_daily(
        start_date=start_date,
        end_date=end_date
    )
    storage.save_raw(daily_data, "daily")
    logger.info(f"日线行情拉取完成: {len(daily_data)} 条记录")
    
    # 拉取每日指标（PE、PB等）
    logger.info(f"拉取每日指标（{start_date}-{end_date}）...")
    daily_basic = client.get_daily_basic(
        start_date=start_date,
        end_date=end_date
    )
    storage.save_raw(daily_basic, "daily_basic")
    logger.info(f"每日指标拉取完成: {len(daily_basic)} 条记录")


def pull_daily_data_partitioned(
    client: TushareClient,
    storage: Storage,
    trade_cal,
    start_date: str,
    end_date: str
):
    """使用按日分区方式拉取日线数据（新方式，推荐）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        trade_cal: 交易日历DataFrame
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
    """
    logger.info(f"拉取日线行情（{start_date}-{end_date}）...")
    logger.info("使用按日分区存储模式（推荐）")
    
    # 获取交易日列表
    trading_dates = trade_cal[
        (trade_cal['cal_date'] >= start_date) &
        (trade_cal['cal_date'] <= end_date) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    logger.info(f"共 {len(trading_dates)} 个交易日需要拉取")
    
    total_daily = 0
    total_basic = 0
    
    for i, trade_date in enumerate(trading_dates, 1):
        logger.info(f"[{i}/{len(trading_dates)}] 处理 {trade_date}...")
        
        try:
            # 拉取日线行情
            daily_data = client.get_daily(trade_date=trade_date)
            if len(daily_data) > 0:
                storage.save_raw_by_date(daily_data, "daily", trade_date)
                total_daily += len(daily_data)
                logger.debug(f"  日线: {len(daily_data)} 条")
            
            # 拉取每日指标
            daily_basic = client.get_daily_basic(trade_date=trade_date)
            if len(daily_basic) > 0:
                storage.save_raw_by_date(daily_basic, "daily_basic", trade_date)
                total_basic += len(daily_basic)
                logger.debug(f"  指标: {len(daily_basic)} 条")
                
        except Exception as e:
            logger.error(f"拉取 {trade_date} 数据失败: {str(e)}")
            continue
    
    logger.info(f"日线行情拉取完成: 共 {total_daily} 条记录")
    logger.info(f"每日指标拉取完成: 共 {total_basic} 条记录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从TuShare拉取数据")
    parser.add_argument(
        "--start-date",
        default="20230101",
        help="开始日期，格式YYYYMMDD，默认20230101"
    )
    parser.add_argument(
        "--end-date",
        default="20231231",
        help="结束日期，格式YYYYMMDD，默认20231231"
    )
    parser.add_argument(
        "--use-partitioning",
        action="store_true",
        help="使用按日分区存储（推荐，默认关闭以保持向后兼容）"
    )
    parser.add_argument(
        "--skip-basic",
        action="store_true",
        help="跳过基础数据（交易日历、股票列表）的拉取"
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()  # 确保配置已加载
    
    logger.info("=" * 60)
    logger.info("开始拉取数据")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} - {args.end_date}")
    logger.info(f"分区存储: {'是' if args.use_partitioning else '否'}")
    
    try:
        # 初始化客户端和存储
        client = TushareClient()
        storage = Storage(enable_partitioning=args.use_partitioning)
        
        # 拉取基础数据
        trade_cal = None
        if not args.skip_basic:
            trade_cal = pull_basic_data(client, storage)
        
        # 拉取日线数据
        if args.use_partitioning:
            # 如果需要交易日历但之前跳过了，现在加载
            if trade_cal is None:
                trade_cal = storage.load_raw("trade_cal")
                if trade_cal is None:
                    logger.error("无法加载交易日历，请先运行不带 --skip-basic 的命令")
                    sys.exit(1)
            
            pull_daily_data_partitioned(
                client, storage, trade_cal,
                args.start_date, args.end_date
            )
        else:
            pull_daily_data_monolithic(
                client, storage,
                args.start_date, args.end_date
            )
        
        logger.info("=" * 60)
        logger.info("数据拉取完成！")
        logger.info(f"数据保存位置: {storage.root_path}")
        if args.use_partitioning:
            logger.info("提示: 使用了按日分区存储，可以通过 list_partitions() 查看所有分区")
        logger.info("=" * 60)
        
    except (ValueError, ConnectionError, TimeoutError) as e:
        # TuShare相关错误（token、网络等）
        logger.error("=" * 60)
        logger.error("数据拉取失败")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("请按以下步骤配置TuShare token:")
        logger.error("1. 访问 https://tushare.pro/register 注册账号")
        logger.error("2. 获取token")
        logger.error("3. 创建 .env 文件（参考 .env.example）")
        logger.error("4. 在 .env 文件中设置: TS_TOKEN=your_token_here")
        logger.error("=" * 60)
        sys.exit(1)
        
    except Exception as e:
        logger.exception(f"数据拉取过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
