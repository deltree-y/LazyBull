#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新基础数据脚本（trade_cal和stock_basic）

功能：
- 单独触发trade_cal和stock_basic的全量下载/更新
- trade_cal更新为最新全集
- stock_basic更新为最新全集
- 可用于手工或定时任务
- 支持force参数强制更新
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient


def update_trade_cal(
    client: TushareClient,
    storage: Storage,
    start_date: str = None,
    end_date: str = None,
    force: bool = False
) -> None:
    """更新交易日历（全量）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        start_date: 开始日期，格式YYYYMMDD（默认：5年前）
        end_date: 结束日期，格式YYYYMMDD（默认：2年后）
        force: 是否强制更新
    """
    logger.info("=" * 60)
    logger.info("更新交易日历")
    logger.info("=" * 60)
    
    # 设置默认日期范围（5年前到2年后，确保覆盖足够范围）
    if start_date is None:
        start_dt = datetime.now() - timedelta(days=5*365)
        start_date = start_dt.strftime('%Y%m%d')
    
    if end_date is None:
        end_dt = datetime.now() + timedelta(days=2*365)
        end_date = end_dt.strftime('%Y%m%d')
    
    logger.info(f"日期范围: {start_date} - {end_date}")
    
    # 检查是否需要更新
    if not force and storage.check_basic_data_freshness("trade_cal", end_date):
        logger.info("交易日历数据已是最新，无需更新")
        logger.info("提示：使用 --force 参数可强制更新")
        return
    
    # 下载全量交易日历
    logger.info("下载交易日历全集...")
    trade_cal = client.get_trade_cal(
        start_date=start_date,
        end_date=end_date,
        exchange="SSE"
    )
    
    if len(trade_cal) == 0:
        logger.warning("未获取到交易日历数据")
        return
    
    # 保存（覆盖原有数据）
    storage.save_raw(trade_cal, "trade_cal", is_force=True)
    logger.info(f"交易日历已更新: {len(trade_cal)} 条记录")
    
    # 显示数据范围
    if 'cal_date' in trade_cal.columns:
        min_date = trade_cal['cal_date'].min()
        max_date = trade_cal['cal_date'].max()
        trading_days = len(trade_cal[trade_cal['is_open'] == 1])
        logger.info(f"数据范围: {min_date} - {max_date}")
        logger.info(f"交易日数量: {trading_days}")
    
    logger.info("=" * 60)


def update_stock_basic(
    client: TushareClient,
    storage: Storage,
    force: bool = False
) -> None:
    """更新股票基本信息（全量）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        force: 是否强制更新
    """
    logger.info("=" * 60)
    logger.info("更新股票基本信息")
    logger.info("=" * 60)
    
    # 检查是否需要更新
    if not force:
        existing = storage.load_raw("stock_basic")
        if existing is not None:
            logger.info(f"stock_basic数据已存在，记录数: {len(existing)}")
            logger.info("提示：使用 --force 参数可强制更新")
            logger.info("提示：stock_basic建议定期（如每季度）更新一次")
            return
    
    # 下载全量股票列表（包括上市和退市）
    logger.info("下载股票基本信息全集...")
    
    # 获取上市股票
    stock_basic_listed = client.get_stock_basic(list_status="L")
    logger.info(f"上市股票: {len(stock_basic_listed)} 只")
    
    # 可选：获取退市股票（用于历史回测）
    try:
        stock_basic_delisted = client.get_stock_basic(list_status="D")
        logger.info(f"退市股票: {len(stock_basic_delisted)} 只")
        
        # 合并上市和退市股票
        import pandas as pd
        stock_basic = pd.concat([stock_basic_listed, stock_basic_delisted], ignore_index=True)
        logger.info(f"合并后总计: {len(stock_basic)} 只")
    except Exception as e:
        logger.warning(f"获取退市股票失败: {e}")
        logger.info("仅使用上市股票数据")
        stock_basic = stock_basic_listed
    
    if len(stock_basic) == 0:
        logger.warning("未获取到股票基本信息")
        return
    
    # 保存（覆盖原有数据）
    storage.save_raw(stock_basic, "stock_basic", is_force=True)
    logger.info(f"股票基本信息已更新: {len(stock_basic)} 条记录")
    
    # 显示统计信息
    if 'market' in stock_basic.columns:
        market_counts = stock_basic['market'].value_counts()
        logger.info("市场分布:")
        for market, count in market_counts.items():
            logger.info(f"  {market}: {count} 只")
    
    logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="更新基础数据（trade_cal和stock_basic的全集）"
    )
    parser.add_argument(
        "--trade-cal-start",
        help="交易日历开始日期，格式YYYYMMDD（默认：5年前）"
    )
    parser.add_argument(
        "--trade-cal-end",
        help="交易日历结束日期，格式YYYYMMDD（默认：2年后）"
    )
    parser.add_argument(
        "--only-trade-cal",
        action="store_true",
        help="仅更新交易日历"
    )
    parser.add_argument(
        "--only-stock-basic",
        action="store_true",
        help="仅更新股票基本信息"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制更新，即使数据已是最新"
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()  # 确保配置已加载
    
    logger.info("=" * 60)
    logger.info("更新基础数据（trade_cal和stock_basic）")
    logger.info("=" * 60)
    logger.info(f"仅更新trade_cal: {'是' if args.only_trade_cal else '否'}")
    logger.info(f"仅更新stock_basic: {'是' if args.only_stock_basic else '否'}")
    logger.info(f"强制更新: {'是' if args.force else '否'}")
    logger.info("=" * 60)
    
    try:
        # 初始化客户端和存储
        client = TushareClient()
        storage = Storage()
        
        # 更新trade_cal
        if not args.only_stock_basic:
            update_trade_cal(
                client, storage,
                start_date=args.trade_cal_start,
                end_date=args.trade_cal_end,
                force=args.force
            )
        
        # 更新stock_basic
        if not args.only_trade_cal:
            update_stock_basic(client, storage, force=args.force)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("基础数据更新完成！")
        logger.info(f"数据保存位置: {storage.root_path}/raw")
        logger.info("  - trade_cal.parquet (单文件)")
        logger.info("  - stock_basic.parquet (单文件)")
        logger.info("=" * 60)
        logger.info("")
        logger.info("更新策略说明：")
        logger.info("1. trade_cal: 建议每月更新一次，确保包含最新交易日")
        logger.info("2. stock_basic: 建议每季度更新一次，获取新上市/退市股票")
        logger.info("3. 可以在cron或定时任务中运行此脚本")
        logger.info("=" * 60)
        
    except (ValueError, ConnectionError, TimeoutError) as e:
        logger.error("=" * 60)
        logger.error("基础数据更新失败")
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
        logger.exception(f"更新过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
