#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纸面交易脚本

功能：
- T0 子命令：拉取数据 + 生成T1待执行目标
- T1 子命令：读取待执行目标 + 执行订单 + 打印明细

示例：
  python scripts/paper_trade.py t0 --trade-date 20260121 --buy-price close --universe mainboard
  python scripts/paper_trade.py t1 --trade-date 20260122 --buy-price open --sell-price close
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
from src.lazybull.paper import PaperTradingRunner


def run_t0(args):
    """运行T0工作流"""
    logger.info("=" * 80)
    logger.info("纸面交易 T0 工作流")
    logger.info("=" * 80)
    logger.info(f"交易日期: {args.trade_date}")
    logger.info(f"买入价格: {args.buy_price}")
    logger.info(f"股票池: {args.universe}")
    logger.info(f"持仓数: {args.top_n}")
    if args.model_version:
        logger.info(f"模型版本: {args.model_version}")
    logger.info("=" * 80)
    
    # 创建运行器
    runner = PaperTradingRunner(
        initial_capital=args.initial_capital
    )
    
    # 运行T0
    try:
        runner.run_t0(
            trade_date=args.trade_date,
            buy_price_type=args.buy_price,
            universe_type=args.universe,
            top_n=args.top_n,
            model_version=args.model_version
        )
        logger.info("T0 工作流完成！")
    except Exception as e:
        logger.exception(f"T0 工作流失败: {e}")
        sys.exit(1)


def run_t1(args):
    """运行T1工作流"""
    logger.info("=" * 80)
    logger.info("纸面交易 T1 工作流")
    logger.info("=" * 80)
    logger.info(f"交易日期: {args.trade_date}")
    logger.info(f"买入价格: {args.buy_price}")
    logger.info(f"卖出价格: {args.sell_price}")
    logger.info("=" * 80)
    
    # 创建运行器
    runner = PaperTradingRunner()
    
    # 运行T1
    try:
        runner.run_t1(
            trade_date=args.trade_date,
            buy_price_type=args.buy_price,
            sell_price_type=args.sell_price
        )
        logger.info("T1 工作流完成！")
    except Exception as e:
        logger.exception(f"T1 工作流失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="纸面交易命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # T0 子命令
    t0_parser = subparsers.add_parser(
        't0',
        help='T0 工作流：拉取数据 + 生成T1待执行目标'
    )
    t0_parser.add_argument(
        '--trade-date',
        required=True,
        help='交易日期，格式YYYYMMDD'
    )
    t0_parser.add_argument(
        '--buy-price',
        choices=['open', 'close'],
        default='close',
        help='买入价格类型（默认：close）'
    )
    t0_parser.add_argument(
        '--universe',
        choices=['mainboard', 'all'],
        default='mainboard',
        help='股票池类型（默认：mainboard，仅沪深主板）'
    )
    t0_parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='持仓股票数（默认：5）'
    )
    t0_parser.add_argument(
        '--model-version',
        type=int,
        help='ML模型版本（可选）'
    )
    t0_parser.add_argument(
        '--initial-capital',
        type=float,
        default=500000.0,
        help='初始资金（默认：500000）'
    )
    
    # T1 子命令
    t1_parser = subparsers.add_parser(
        't1',
        help='T1 工作流：读取待执行目标 + 执行订单 + 打印明细'
    )
    t1_parser.add_argument(
        '--trade-date',
        required=True,
        help='交易日期，格式YYYYMMDD'
    )
    t1_parser.add_argument(
        '--buy-price',
        choices=['open', 'close'],
        default='close',
        help='买入价格类型（默认：close）'
    )
    t1_parser.add_argument(
        '--sell-price',
        choices=['open', 'close'],
        default='close',
        help='卖出价格类型（默认：close，固定）'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()  # 确保配置已加载
    
    # 执行命令
    if args.command == 't0':
        run_t0(args)
    elif args.command == 't1':
        run_t1(args)


if __name__ == "__main__":
    main()
