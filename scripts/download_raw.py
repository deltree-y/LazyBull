#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载原始数据脚本（仅下载raw层）

功能：
- 仅负责从TuShare拉取原始数据并保存到raw层（partitioned存储）
- 不触发clean或feature的构建
- 支持force参数强制重新下载已存在的数据
- trade_cal和stock_basic保存为单文件，其他数据按日期分区保存
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient

if TYPE_CHECKING:
    import pandas as pd


def download_basic_data(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False
) -> "pd.DataFrame":
    """下载基础数据（trade_cal和stock_basic）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新下载
        
    Returns:
        交易日历DataFrame
    """
    # 1. 下载交易日历
    logger.info("检查交易日历...")
    if not force and storage.check_basic_data_freshness("trade_cal", end_date):
        logger.info("交易日历数据已是最新，跳过下载")
        trade_cal = storage.load_raw("trade_cal")
    else:
        logger.info(f"下载交易日历（{start_date}-{end_date}）...")
        trade_cal = client.get_trade_cal(
            start_date=start_date,
            end_date=end_date,
            exchange="SSE"
        )
        storage.save_raw(trade_cal, "trade_cal", is_force=True)
        logger.info(f"交易日历下载完成: {len(trade_cal)} 条记录")
    
    # 2. 下载股票基本信息
    logger.info("检查股票基本信息...")
    if not force and storage.check_basic_data_freshness("stock_basic", end_date):
        logger.info("股票基本信息已存在，跳过下载")
    else:
        logger.info("下载股票基本信息...")
        stock_basic = client.get_stock_basic(list_status="L")
        storage.save_raw(stock_basic, "stock_basic", is_force=True)
        logger.info(f"股票基本信息下载完成: {len(stock_basic)} 条记录")
    
    return trade_cal


def download_daily_data(
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False
) -> None:
    """下载日线数据（按日期分区）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        trade_cal: 交易日历DataFrame
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新下载
    """
    import pandas as pd
    
    logger.info(f"下载日线数据（{start_date}-{end_date}）...")
    logger.info("使用按日分区存储模式")
    
    # 获取交易日列表
    trading_dates = trade_cal[
        (trade_cal['cal_date'] >= start_date) &
        (trade_cal['cal_date'] <= end_date) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    logger.info(f"共 {len(trading_dates)} 个交易日需要下载")
    
    total_daily = 0
    total_basic = 0
    skip_count = 0
    
    for i, trade_date in enumerate(trading_dates, 1):
        logger.info(f"[{i}/{len(trading_dates)}] ({i/len(trading_dates):.1%}) 处理 {trade_date}...")
        
        try:
            # 下载日线行情
            if not force and storage.is_data_exists("raw", "daily", trade_date):
                logger.info(f"  日线: 文件已存在，跳过下载")
                skip_count += 1
            else:
                daily_data = client.get_daily(trade_date=trade_date)
                if len(daily_data) > 0:
                    storage.save_raw_by_date(daily_data, "daily", trade_date)
                    total_daily += len(daily_data)
                    logger.info(f"  日线: 已保存 {len(daily_data)} 条记录")
            
            # 下载每日指标
            if not force and storage.is_data_exists("raw", "daily_basic", trade_date):
                logger.info(f"  指标: 文件已存在，跳过下载")
            else:
                daily_basic = client.get_daily_basic(trade_date=trade_date)
                if len(daily_basic) > 0:
                    storage.save_raw_by_date(daily_basic, "daily_basic", trade_date)
                    total_basic += len(daily_basic)
                    logger.info(f"  指标: 已保存 {len(daily_basic)} 条记录")

            # 下载复权因子
            if not force and storage.is_data_exists("raw", "adj_factor", trade_date):
                logger.info(f"  复权因子: 文件已存在，跳过下载")
            else:
                adj_factor = client.get_adj_factor(trade_date=trade_date)
                if len(adj_factor) > 0:
                    storage.save_raw_by_date(adj_factor, "adj_factor", trade_date)
                    logger.info(f"  复权因子: 已保存 {len(adj_factor)} 条记录")
                    
            # 下载停复牌信息
            if not force and storage.is_data_exists("raw", "suspend", trade_date):
                logger.info(f"  停复牌: 文件已存在，跳过下载")
            else:
                suspend = client.get_suspend_d(trade_date=trade_date)
                if len(suspend) > 0:
                    storage.save_raw_by_date(suspend, "suspend", trade_date)
                    logger.info(f"  停复牌: 已保存 {len(suspend)} 条记录")
                    
            # 下载涨跌停信息
            if not force and storage.is_data_exists("raw", "stk_limit", trade_date):
                logger.info(f"  涨跌停: 文件已存在，跳过下载")
            else:
                limit_up_down = client.get_stk_limit(trade_date=trade_date)
                if len(limit_up_down) > 0:
                    storage.save_raw_by_date(limit_up_down, "stk_limit", trade_date)
                    logger.info(f"  涨跌停: 已保存 {len(limit_up_down)} 条记录")
                    
        except Exception as e:
            logger.error(f"下载 {trade_date} 数据失败: {str(e)}")
            continue
    
    logger.info("=" * 60)
    logger.info("日线数据下载完成")
    logger.info("=" * 60)
    logger.info(f"新下载日线行情: {total_daily} 条记录")
    logger.info(f"新下载每日指标: {total_basic} 条记录")
    logger.info(f"跳过已存在: {skip_count} 个交易日")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载原始数据（仅raw层，不触发clean/feature构建）"
    )
    parser.add_argument(
        "--start-date",
        default="20200101",
        help="开始日期，格式YYYYMMDD（默认：20200101）"
    )
    parser.add_argument(
        "--end-date",
        default="20251231",
        help="结束日期，格式YYYYMMDD（默认：20251231）"
    )
    parser.add_argument(
        "--only-basic",
        action="store_true",
        help="仅下载基础数据（trade_cal和stock_basic）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载，即使文件已存在"
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()  # 确保配置已加载
    
    logger.info("=" * 60)
    logger.info("开始下载原始数据（raw层）")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} - {args.end_date}")
    logger.info(f"仅下载基础数据: {'是' if args.only_basic else '否'}")
    logger.info(f"强制重新下载: {'是' if args.force else '否'}")
    logger.info("=" * 60)
    
    try:
        # 初始化客户端和存储
        client = TushareClient()
        storage = Storage()
        
        # 下载基础数据
        trade_cal = download_basic_data(
            client, storage,
            args.start_date, args.end_date,
            force=args.force
        )
        
        if args.only_basic:
            logger.info("=" * 60)
            logger.info("仅下载基础数据，操作完成！")
            logger.info(f"数据保存位置: {storage.root_path}/raw")
            logger.info("=" * 60)
            sys.exit(0)
        
        # 下载日线数据
        download_daily_data(
            client, storage, trade_cal,
            args.start_date, args.end_date,
            force=args.force
        )
        
        logger.info("=" * 60)
        logger.info("原始数据下载完成！")
        logger.info(f"数据保存位置: {storage.root_path}/raw")
        logger.info("  - trade_cal.parquet (单文件)")
        logger.info("  - stock_basic.parquet (单文件)")
        logger.info("  - daily/{YYYY-MM-DD}.parquet (按日分区)")
        logger.info("  - daily_basic/{YYYY-MM-DD}.parquet (按日分区)")
        logger.info("  - adj_factor/{YYYY-MM-DD}.parquet (按日分区)")
        logger.info("  - suspend/{YYYY-MM-DD}.parquet (按日分区)")
        logger.info("  - stk_limit/{YYYY-MM-DD}.parquet (按日分区)")
        logger.info("=" * 60)
        logger.info("")
        logger.info("下一步操作提示：")
        logger.info("  1. 构建clean和feature: python scripts/build_clean_features.py")
        logger.info("  2. 仅构建feature: python scripts/build_features.py")
        logger.info("=" * 60)
        
    except (ValueError, ConnectionError, TimeoutError) as e:
        logger.error("=" * 60)
        logger.error("数据下载失败")
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
        logger.exception(f"数据下载过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
