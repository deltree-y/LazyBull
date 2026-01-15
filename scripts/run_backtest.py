#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测运行脚本
执行策略回测并生成报告
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.backtest import BacktestEngine, Reporter
from src.lazybull.common.cost import CostModel
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataLoader, Storage
from src.lazybull.signals import EqualWeightSignal
from src.lazybull.universe import BasicUniverse


def create_mock_data():
    """创建模拟数据用于演示
    
    Returns:
        (stock_basic, daily, trading_dates)
    """
    logger.info("使用模拟数据运行回测示例...")
    
    # 模拟股票基本信息
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600016.SH', '000858.SZ'],
        'symbol': ['000001', '000002', '600000', '600016', '000858'],
        'name': ['平安银行', '万科A', '浦发银行', '民生银行', '五粮液'],
        'market': ['主板', '主板', '主板', '主板', '主板'],
        'list_date': ['19910403', '19910129', '19991110', '20001219', '19980427']
    })
    
    # 模拟交易日
    trading_dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
    trading_dates = [d for d in trading_dates if d.month <= 12][:250]  # 取前250个交易日
    
    # 模拟日线数据（简单随机游走）
    daily_data = []
    for stock in stock_basic['ts_code']:
        price = 10.0
        for date in trading_dates:
            # 简单模拟价格波动
            import random
            price = price * (1 + random.uniform(-0.02, 0.02))
            daily_data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': round(price, 2),
                'open': round(price * 1.005, 2),
                'high': round(price * 1.01, 2),
                'low': round(price * 0.99, 2),
            })
    
    daily = pd.DataFrame(daily_data)
    
    return stock_basic, daily, trading_dates


def main():
    """主函数"""
    # 初始化日志
    setup_logger(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("开始回测")
    logger.info("=" * 60)
    
    try:
        # 尝试加载真实数据
        loader = DataLoader()
        stock_basic = loader.load_stock_basic()
        daily = loader.load_daily()
        trading_dates_list = loader.get_trading_dates('2023-01-01', '2023-12-31')
        
        # 如果没有真实数据，使用模拟数据
        use_mock = False
        if stock_basic is None or daily is None or not trading_dates_list:
            logger.warning("未找到真实数据，使用模拟数据运行回测示例")
            stock_basic, daily, trading_dates_list = create_mock_data()
            use_mock = True
        
        logger.info(f"股票池大小: {len(stock_basic)}")
        logger.info(f"日线数据: {len(daily)} 条")
        logger.info(f"交易日数: {len(trading_dates_list)}")
        
        # 初始化组件
        # 1. 股票池
        universe = BasicUniverse(
            stock_basic=stock_basic,
            exclude_st=True,
            min_list_days=252,
            markets=['主板']
        )
        
        # 2. 信号生成器（等权前5只）
        signal = EqualWeightSignal(top_n=5)
        
        # 3. 成本模型
        cost_model = CostModel(
            commission_rate=0.0003,
            min_commission=5.0,
            stamp_tax=0.001,
            slippage=0.001
        )
        
        # 4. 回测引擎
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            initial_capital=1000000.0,
            cost_model=cost_model,
            rebalance_freq="M"  # 月度调仓
        )
        
        # 运行回测
        trading_dates = [pd.to_datetime(d) for d in trading_dates_list]
        nav_curve = engine.run(
            start_date=trading_dates[0],
            end_date=trading_dates[-1],
            trading_dates=trading_dates,
            price_data=daily
        )
        
        # 生成报告
        reporter = Reporter()
        trades = engine.get_trades()
        
        report_name = "backtest_mock" if use_mock else "backtest_real"
        stats = reporter.generate_report(nav_curve, trades, output_name=report_name)
        
        logger.info("=" * 60)
        logger.info("回测完成！")
        logger.info("=" * 60)
        
        if use_mock:
            logger.info("")
            logger.info("提示：当前使用模拟数据运行。")
            logger.info("要使用真实数据回测，请:")
            logger.info("1. 配置 TS_TOKEN 环境变量")
            logger.info("2. 运行 python scripts/pull_data.py 拉取数据")
            logger.info("3. 再次运行本脚本")
        
    except Exception as e:
        logger.exception(f"回测过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
