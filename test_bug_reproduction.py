"""测试脚本：复现多 horizon 标签计算错误的问题"""

import pandas as pd
import numpy as np
from src.lazybull.features import FeatureBuilder


def create_test_data():
    """创建模拟测试数据，模拟 600036.SH 的场景"""
    
    # 创建交易日历 - 从 20251220 到 20260210（足够多的交易日）
    # 模拟真实交易日历
    dates = [
        '20251220', '20251223', '20251224', '20251225', '20251226', '20251227',
        '20251230', '20251231',  # 这是我们要测试的日期
        '20260102', '20260103', '20260106', '20260107', '20260108', '20260109',  # t+5 应该是 20260109
        '20260112', '20260113', '20260114', '20260115', '20260116',  # t+10 应该是 20260116
        '20260119', '20260120', '20260121', '20260122', '20260123',
        '20260126', '20260127', '20260128', '20260129', '20260130',  # t+20 应该是 20260130
        '20260202', '20260203', '20260204', '20260205', '20260206'
    ]
    
    trade_cal = pd.DataFrame({
        'exchange': ['SSE'] * len(dates),
        'cal_date': dates,
        'is_open': [1] * len(dates)
    })
    
    # 创建股票基本信息
    stock_basic = pd.DataFrame({
        'ts_code': ['600036.SH'],
        'name': ['招商银行'],
        'list_date': ['20020101']
    })
    
    # 创建日线行情数据 - 使用问题中提到的真实价格
    # 20251231: 42.1
    # 20260109: 41.3 (应该是 t+5)
    # 20260116: 38.72 (应该是 t+10)
    # 20260130: 38.67 (应该是 t+20)
    
    prices = {
        '20251220': 42.5, '20251223': 42.4, '20251224': 42.3, '20251225': 42.2, 
        '20251226': 42.15, '20251227': 42.12, '20251230': 42.11, '20251231': 42.1,
        '20260102': 42.0, '20260103': 41.8, '20260106': 41.6, '20260107': 41.5,
        '20260108': 41.4, '20260109': 41.3,  # t+5
        '20260112': 40.8, '20260113': 40.5, '20260114': 39.9, '20260115': 39.2,
        '20260116': 38.72,  # t+10
        '20260119': 38.7, '20260120': 38.68, '20260121': 38.66, '20260122': 38.65,
        '20260123': 38.68, '20260126': 38.67, '20260127': 38.68, '20260128': 38.66,
        '20260129': 38.65, '20260130': 38.67,  # t+20
        '20260202': 38.70, '20260203': 38.72, '20260204': 38.75, '20260205': 38.80,
        '20260206': 38.85
    }
    
    data = []
    for date in dates:
        close = prices[date]
        data.append({
            'ts_code': '600036.SH',
            'trade_date': date,
            'close': close,
            'close_adj': close,  # 假设复权因子为1
            'pre_close': close * 1.01,  # 简化
            'pct_chg': -1.0,  # 简化
            'vol': 1000000,
            'amount': 1000000 * close,
            'is_st': 0,
            'is_suspended': 0,
            'is_limit_up': 0,
            'is_limit_down': 0,
            'list_days': 9000,
            'tradable': 1
        })
    
    daily_data = pd.DataFrame(data)
    
    # 创建复权因子（全为1）
    adj_factor = pd.DataFrame({
        'ts_code': ['600036.SH'] * len(dates),
        'trade_date': dates,
        'adj_factor': [1.0] * len(dates)
    })
    
    return trade_cal, stock_basic, daily_data, adj_factor


def test_bug_reproduction():
    """测试并展示bug"""
    
    print("=" * 80)
    print("测试多 horizon 标签计算")
    print("=" * 80)
    
    # 创建测试数据
    trade_cal, stock_basic, daily_data, adj_factor = create_test_data()
    
    # 创建特征构建器
    builder = FeatureBuilder(horizons=[5, 10, 20], require_label=False, verbose=True)
    
    # 构建 20251231 的特征
    trade_date = '20251231'
    features = builder.build_features_for_day(
        trade_date=trade_date,
        trade_cal=trade_cal,
        daily_data=daily_data,
        adj_factor=adj_factor,
        stock_basic=stock_basic
    )
    
    print(f"\n构建完成，共 {len(features)} 个样本")
    
    if len(features) > 0:
        row = features.iloc[0]
        print(f"\n股票: {row['ts_code']}")
        print(f"交易日期: {row['trade_date']}")
        
        # 显示标签
        print(f"\ny_ret_5: {row['y_ret_5']:.6f}")
        print(f"y_ret_10: {row['y_ret_10']:.6f}")
        print(f"y_ret_20: {row['y_ret_20']:.6f}")
        
        # 手工计算预期值
        price_t = 42.1  # 20251231
        price_t5 = 41.3  # 20260109 (第5个交易日后)
        price_t10 = 38.72  # 20260116 (第10个交易日后)
        price_t20 = 38.67  # 20260130 (第20个交易日后)
        
        expected_ret_5 = (price_t5 / price_t) - 1
        expected_ret_10 = (price_t10 / price_t) - 1
        expected_ret_20 = (price_t20 / price_t) - 1
        
        print(f"\n预期 y_ret_5: {expected_ret_5:.6f}")
        print(f"预期 y_ret_10: {expected_ret_10:.6f}")
        print(f"预期 y_ret_20: {expected_ret_20:.6f}")
        
        # 检查是否正确
        print(f"\ny_ret_5 正确: {abs(row['y_ret_5'] - expected_ret_5) < 1e-6}")
        print(f"y_ret_10 正确: {abs(row['y_ret_10'] - expected_ret_10) < 1e-6}")
        print(f"y_ret_20 正确: {abs(row['y_ret_20'] - expected_ret_20) < 1e-6}")
        
        # 显示交易日列表以供调试
        trading_dates = builder._get_trading_dates(trade_cal)
        idx = trading_dates.index(trade_date)
        print(f"\n当前日期在交易日列表中的索引: {idx}")
        print(f"t+5 日期 (index {idx+5}): {trading_dates[idx+5] if idx+5 < len(trading_dates) else 'N/A'}")
        print(f"t+10 日期 (index {idx+10}): {trading_dates[idx+10] if idx+10 < len(trading_dates) else 'N/A'}")
        print(f"t+20 日期 (index {idx+20}): {trading_dates[idx+20] if idx+20 < len(trading_dates) else 'N/A'}")
        
        # 检查是否有重复日期
        print(f"\n交易日列表长度: {len(trading_dates)}")
        print(f"唯一日期数量: {len(set(trading_dates))}")
        if len(trading_dates) != len(set(trading_dates)):
            print("警告：交易日列表包含重复日期！")
            from collections import Counter
            duplicates = [date for date, count in Counter(trading_dates).items() if count > 1]
            print(f"重复的日期: {duplicates}")


if __name__ == "__main__":
    test_bug_reproduction()
