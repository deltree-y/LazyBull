#!/usr/bin/env python3
"""
演示脚本：展示纸面交易持仓表格改进

此脚本创建一个临时账户来演示新的持仓表格和汇总统计格式
"""

import sys
import tempfile
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lazybull.paper import PaperAccount, PaperBroker, PaperStorage, NAVRecord
from loguru import logger

def demo_positions_display():
    """演示持仓表格显示"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建临时存储和账户
        storage = PaperStorage(tmpdir, verbose=False)
        account = PaperAccount(initial_capital=500000.0, storage=storage, verbose=False)
        broker = PaperBroker(account, storage=storage, verbose=False)
        
        # 保存配置（包含初始资金和账户起始日期）
        config = {
            'initial_capital': 500000.0,
            'account_start_date': '20260101',  # 设置账户起始日期
        }
        storage.save_config(config)
        
        # 保存一个 NAV 记录（作为账户起始）
        nav_record = NAVRecord(
            trade_date='20260101',
            cash=500000.0,
            position_value=0.0,
            total_value=500000.0,
            nav=1.0
        )
        storage.append_nav(nav_record)
        
        # 添加一些模拟持仓
        account.add_position(
            ts_code='000001.SZ',
            shares=10000,
            buy_price=10.50,
            buy_cost=105.0,  # 手续费
            buy_date='20260115',
            status='持有'
        )
        account.update_cash(-105105.0)
        
        account.add_position(
            ts_code='600000.SH',
            shares=5000,
            buy_price=8.30,
            buy_cost=41.5,
            buy_date='20260120',
            status='持有'
        )
        account.update_cash(-41541.5)
        
        account.add_position(
            ts_code='601398.SH',
            shares=20000,
            buy_price=5.50,
            buy_cost=110.0,
            buy_date='20260125',
            status='持有'
        )
        account.update_cash(-110110.0)
        
        # 准备当前价格（模拟盈利）
        current_prices = {
            '000001.SZ': 12.50,  # 盈利 19.0%
            '600000.SH': 8.80,   # 盈利 6.0%
            '601398.SH': 6.20,   # 盈利 12.7%
        }
        
        # 准备股票名称
        stock_names = {
            '000001.SZ': '平安银行',
            '600000.SH': '浦发银行',
            '601398.SH': '工商银行',
        }
        
        # 打印持仓表格
        print("\n" + "=" * 140)
        print("演示：改进后的持仓表格")
        print("=" * 140)
        broker.print_positions_summary(current_prices, current_date='20260131', stock_names=stock_names)
        
        print("\n\n")
        print("=" * 140)
        print("演示：缺少股票名称时的显示")
        print("=" * 140)
        # 不传入股票名称，展示 (na) 格式
        broker.print_positions_summary(current_prices, current_date='20260131')
        
        print("\n\n说明：")
        print("1. 股票代码现在显示为：ts_code(股票名称)")
        print("2. 表格不再显示'买入成本'列")
        print("3. '当前价格'列移到了'买入均价'前面")
        print("4. 汇总统计新增了'总盈亏百分比'和'年化收益率'")
        print("5. 年化收益率从配置的 account_start_date 或 NAV 记录最早日期开始计算")
        print()

if __name__ == "__main__":
    logger.remove()  # 移除默认 logger
    logger.add(sys.stderr, level="INFO")  # 只显示 INFO 及以上级别
    
    demo_positions_display()
