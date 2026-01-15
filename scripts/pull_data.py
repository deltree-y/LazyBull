#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据拉取脚本
从TuShare拉取基础数据并保存到本地
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient


def main():
    """主函数"""
    # 初始化日志
    setup_logger(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("开始拉取数据")
    logger.info("=" * 60)
    
    try:
        # 初始化客户端和存储
        client = TushareClient()
        storage = Storage()
        
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
        
        # 3. 拉取日线行情（示例：仅拉取2023年数据，避免数据量过大）
        logger.info("拉取日线行情（2023年）...")
        logger.info("注意：全量数据拉取较慢，建议按需拉取特定时间段")
        
        # 这里只拉取少量数据作为示例
        daily_data = client.get_daily(
            start_date="20230101",
            end_date="20231231"
        )
        storage.save_raw(daily_data, "daily")
        logger.info(f"日线行情拉取完成: {len(daily_data)} 条记录")
        
        # 4. 拉取每日指标（PE、PB等）
        logger.info("拉取每日指标（2023年）...")
        daily_basic = client.get_daily_basic(
            start_date="20230101",
            end_date="20231231"
        )
        storage.save_raw(daily_basic, "daily_basic")
        logger.info(f"每日指标拉取完成: {len(daily_basic)} 条记录")
        
        logger.info("=" * 60)
        logger.info("数据拉取完成！")
        logger.info(f"数据保存位置: {storage.root_path}")
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
