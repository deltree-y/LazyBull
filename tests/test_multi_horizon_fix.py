"""测试多 horizon 标签计算修复

本测试文件专门用于验证修复后的多 horizon 标签计算逻辑，包括：
1. 重复交易日期的处理
2. 日期格式一致性
3. 真实场景模拟（600036.SH 案例）
4. 所有 horizon (5/10/20) 同时正确
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pytest

from lazybull.features.builder import FeatureBuilder


class TestMultiHorizonCalculationFix:
    """测试多 horizon 标签计算修复"""
    
    def test_duplicate_trading_dates_handling(self):
        """测试重复交易日期的处理（核心bug修复验证）"""
        
        # 创建包含重复日期的交易日历（模拟多个交易所的数据）
        dates = ['20230103', '20230104', '20230105', '20230106', '20230109',
                 '20230110', '20230111', '20230112', '20230113', '20230116',
                 '20230117', '20230118', '20230119', '20230120', '20230130',
                 '20230131', '20230201', '20230202', '20230203', '20230206']
        
        # 故意添加重复的日期（模拟bug场景）
        duplicate_dates = dates + dates  # 每个日期重复两次
        
        trade_cal = pd.DataFrame({
            'exchange': (['SSE'] * len(dates)) + (['SZSE'] * len(dates)),
            'cal_date': duplicate_dates,
            'is_open': [1] * len(duplicate_dates)
        })
        
        # 创建特征构建器
        builder = FeatureBuilder(horizons=[5, 10, 20])
        
        # 提取交易日列表
        trading_dates = builder._get_trading_dates(trade_cal)
        
        # 验证1：交易日列表应该去重
        assert len(trading_dates) == len(dates), \
            f"交易日列表应该去重，期望 {len(dates)} 个，实际 {len(trading_dates)} 个"
        
        # 验证2：交易日列表应该没有重复
        assert len(trading_dates) == len(set(trading_dates)), \
            "交易日列表不应包含重复日期"
        
        # 验证3：交易日列表应该排序
        assert trading_dates == sorted(trading_dates), \
            "交易日列表应该按时间顺序排序"
        
        print("✓ 重复交易日期处理测试通过")
    
    def test_real_case_600036_simulation(self):
        """测试真实案例：600036.SH 在 20251231 的标签计算"""
        
        # 创建模拟数据，复现问题场景
        dates = [
            '20251220', '20251223', '20251224', '20251225', '20251226', '20251227',
            '20251230', '20251231',  # t=0 测试日期
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
        
        stock_basic = pd.DataFrame({
            'ts_code': ['600036.SH'],
            'name': ['招商银行'],
            'list_date': ['20020101']
        })
        
        # 使用问题中提到的真实价格
        # 注意：重新核对交易日计数
        # 从 20251231 (t=0, index 7) 开始：
        # t+5: index 12 -> 20260108
        # t+10: index 17 -> 20260115  
        # t+20: index 27 -> 20260129
        prices = {
            '20251220': 42.5, '20251223': 42.4, '20251224': 42.3, '20251225': 42.2,
            '20251226': 42.15, '20251227': 42.12, '20251230': 42.11, '20251231': 42.1,  # t=0
            '20260102': 42.0, '20260103': 41.8, '20260106': 41.6, '20260107': 41.5,
            '20260108': 41.3,  # t+5 (修正：这才是第5个交易日后)
            '20260109': 41.2,  # t+6
            '20260112': 40.8, '20260113': 40.5, '20260114': 39.9, 
            '20260115': 38.72,  # t+10 (修正：这才是第10个交易日后)
            '20260116': 38.70,  # t+11
            '20260119': 38.7, '20260120': 38.68, '20260121': 38.66, '20260122': 38.65,
            '20260123': 38.68, '20260126': 38.67, '20260127': 38.68, '20260128': 38.66,
            '20260129': 38.67,  # t+20 (修正：这才是第20个交易日后)
            '20260130': 38.65,  # t+21
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
                'pre_close': close * 1.01,
                'pct_chg': -1.0,
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
        
        adj_factor = pd.DataFrame({
            'ts_code': ['600036.SH'] * len(dates),
            'trade_date': dates,
            'adj_factor': [1.0] * len(dates)
        })
        
        # 构建特征
        builder = FeatureBuilder(horizons=[5, 10, 20], require_label=False)
        
        trade_date = '20251231'
        features = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=trade_cal,
            daily_data=daily_data,
            adj_factor=adj_factor,
            stock_basic=stock_basic
        )
        
        # 验证结果
        assert len(features) > 0, "应该返回至少一个样本"
        
        row = features.iloc[0]
        
        # 计算预期值（修正后的日期）
        price_t = 42.1  # 20251231
        price_t5 = 41.3  # 20260108 (t+5)
        price_t10 = 38.72  # 20260115 (t+10)
        price_t20 = 38.67  # 20260129 (t+20)
        
        expected_ret_5 = (price_t5 / price_t) - 1
        expected_ret_10 = (price_t10 / price_t) - 1
        expected_ret_20 = (price_t20 / price_t) - 1
        
        # 验证标签计算的正确性
        assert abs(row['y_ret_5'] - expected_ret_5) < 1e-6, \
            f"y_ret_5 计算错误: 预期 {expected_ret_5:.6f}, 实际 {row['y_ret_5']:.6f}"
        
        assert abs(row['y_ret_10'] - expected_ret_10) < 1e-6, \
            f"y_ret_10 计算错误: 预期 {expected_ret_10:.6f}, 实际 {row['y_ret_10']:.6f}"
        
        assert abs(row['y_ret_20'] - expected_ret_20) < 1e-6, \
            f"y_ret_20 计算错误: 预期 {expected_ret_20:.6f}, 实际 {row['y_ret_20']:.6f}"
        
        # 验证 y_ret_10 和 y_ret_20 不应该几乎相等（这是原bug的表现）
        assert abs(row['y_ret_10'] - row['y_ret_20']) > 0.001, \
            f"y_ret_10 ({row['y_ret_10']:.6f}) 和 y_ret_20 ({row['y_ret_20']:.6f}) 不应该几乎相等"
        
        print("✓ 600036.SH 真实场景测试通过")
        print(f"  y_ret_5:  {row['y_ret_5']:.6f} (预期 {expected_ret_5:.6f})")
        print(f"  y_ret_10: {row['y_ret_10']:.6f} (预期 {expected_ret_10:.6f})")
        print(f"  y_ret_20: {row['y_ret_20']:.6f} (预期 {expected_ret_20:.6f})")
    
    def test_all_horizons_simultaneously_correct(self):
        """测试所有 horizon (5/10/20) 同时正确"""
        
        # 创建简单的线性价格序列以便验证
        num_days = 30
        dates = [f"2023{i+1:04d}" for i in range(num_days)]
        
        trade_cal = pd.DataFrame({
            'exchange': ['SSE'] * num_days,
            'cal_date': dates,
            'is_open': [1] * num_days
        })
        
        stock_basic = pd.DataFrame({
            'ts_code': ['TEST.SH'],
            'name': ['测试股票'],
            'list_date': ['20200101']
        })
        
        # 价格序列：每天增长1%
        data = []
        base_price = 100.0
        for i, date in enumerate(dates):
            price = base_price * (1.01 ** i)
            data.append({
                'ts_code': 'TEST.SH',
                'trade_date': date,
                'close': price,
                'close_adj': price,
                'pre_close': price / 1.01,
                'pct_chg': 1.0,
                'vol': 1000000,
                'amount': 1000000 * price,
                'is_st': 0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'list_days': 1000,
                'tradable': 1
            })
        
        daily_data = pd.DataFrame(data)
        
        adj_factor = pd.DataFrame({
            'ts_code': ['TEST.SH'] * num_days,
            'trade_date': dates,
            'adj_factor': [1.0] * num_days
        })
        
        # 构建特征
        builder = FeatureBuilder(horizons=[5, 10, 20], require_label=False)
        
        # 选择第2个交易日（确保有足够的未来数据）
        trade_date = dates[2]  # index=2
        
        features = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=trade_cal,
            daily_data=daily_data,
            adj_factor=adj_factor,
            stock_basic=stock_basic
        )
        
        assert len(features) > 0
        row = features.iloc[0]
        
        # 计算预期收益率
        # t=2: price = 100 * 1.01^2
        # t+5=7: price = 100 * 1.01^7
        # t+10=12: price = 100 * 1.01^12
        # t+20=22: price = 100 * 1.01^22
        
        expected_ret_5 = (1.01 ** 5) - 1  # ≈ 0.051
        expected_ret_10 = (1.01 ** 10) - 1  # ≈ 0.104
        expected_ret_20 = (1.01 ** 20) - 1  # ≈ 0.220
        
        # 验证
        assert abs(row['y_ret_5'] - expected_ret_5) < 1e-6, \
            f"y_ret_5 错误: {row['y_ret_5']:.6f} vs {expected_ret_5:.6f}"
        
        assert abs(row['y_ret_10'] - expected_ret_10) < 1e-6, \
            f"y_ret_10 错误: {row['y_ret_10']:.6f} vs {expected_ret_10:.6f}"
        
        assert abs(row['y_ret_20'] - expected_ret_20) < 1e-6, \
            f"y_ret_20 错误: {row['y_ret_20']:.6f} vs {expected_ret_20:.6f}"
        
        # 验证递增关系（因为是持续增长的价格序列）
        assert row['y_ret_5'] < row['y_ret_10'] < row['y_ret_20'], \
            "收益率应该随 horizon 递增"
        
        print("✓ 所有 horizon 同时正确性测试通过")
        print(f"  y_ret_5:  {row['y_ret_5']:.6f} (预期 {expected_ret_5:.6f})")
        print(f"  y_ret_10: {row['y_ret_10']:.6f} (预期 {expected_ret_10:.6f})")
        print(f"  y_ret_20: {row['y_ret_20']:.6f} (预期 {expected_ret_20:.6f})")
    
    def test_date_format_consistency(self):
        """测试日期格式一致性处理"""
        
        # 测试场景1：trade_cal 使用 datetime 格式
        dates_dt = pd.date_range('2023-01-03', periods=20, freq='B')
        trade_cal_dt = pd.DataFrame({
            'exchange': ['SSE'] * len(dates_dt),
            'cal_date': dates_dt,  # datetime 格式
            'is_open': [1] * len(dates_dt)
        })
        
        builder = FeatureBuilder()
        trading_dates_dt = builder._get_trading_dates(trade_cal_dt)
        
        # 应该转换为字符串格式
        assert all(isinstance(d, str) for d in trading_dates_dt), \
            "交易日期应该转换为字符串格式"
        
        # 应该是 YYYYMMDD 格式
        assert all(len(d) == 8 and d.isdigit() for d in trading_dates_dt), \
            "交易日期应该是 YYYYMMDD 格式"
        
        # 测试场景2：trade_cal 使用字符串格式
        dates_str = [d.strftime('%Y%m%d') for d in dates_dt]
        trade_cal_str = pd.DataFrame({
            'exchange': ['SSE'] * len(dates_str),
            'cal_date': dates_str,  # 字符串格式
            'is_open': [1] * len(dates_str)
        })
        
        trading_dates_str = builder._get_trading_dates(trade_cal_str)
        
        # 两种输入应该得到相同的结果
        assert trading_dates_dt == trading_dates_str, \
            "不同格式的输入应该产生相同的交易日列表"
        
        print("✓ 日期格式一致性测试通过")
    
    def test_unordered_trading_calendar(self):
        """测试乱序交易日历输入"""
        
        # 创建乱序的交易日历
        dates = ['20230110', '20230103', '20230106', '20230109', '20230104',
                 '20230105', '20230111', '20230112', '20230113', '20230116']
        
        trade_cal = pd.DataFrame({
            'exchange': ['SSE'] * len(dates),
            'cal_date': dates,
            'is_open': [1] * len(dates)
        })
        
        builder = FeatureBuilder()
        trading_dates = builder._get_trading_dates(trade_cal)
        
        # 验证输出应该是排序的
        assert trading_dates == sorted(trading_dates), \
            "交易日列表应该按时间顺序排序，即使输入是乱序的"
        
        # 验证没有遗漏日期
        assert len(trading_dates) == len(dates), \
            "所有交易日都应该被包含"
        
        print("✓ 乱序交易日历处理测试通过")
    
    def test_date_to_idx_mapping_performance(self):
        """测试日期到索引映射的性能优化"""
        
        # 创建大量交易日
        num_days = 1000
        dates = [f"2020{i+1:04d}" for i in range(num_days)]
        
        trade_cal = pd.DataFrame({
            'exchange': ['SSE'] * num_days,
            'cal_date': dates,
            'is_open': [1] * num_days
        })
        
        builder = FeatureBuilder()
        trading_dates = builder._get_trading_dates(trade_cal)
        
        # 构建日期到索引的映射
        date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
        
        # 验证映射的正确性
        for idx, date in enumerate(trading_dates):
            assert date_to_idx[date] == idx, \
                f"日期 {date} 的索引映射错误"
        
        # 验证映射可以快速查找
        test_date = dates[500]
        assert test_date in date_to_idx, \
            "映射应该包含所有交易日"
        
        assert date_to_idx[test_date] == trading_dates.index(test_date), \
            "映射的索引应该与 list.index() 结果一致"
        
        print("✓ 日期索引映射测试通过")


if __name__ == "__main__":
    # 运行所有测试
    import sys
    
    test_class = TestMultiHorizonCalculationFix()
    
    tests = [
        ('test_duplicate_trading_dates_handling', test_class.test_duplicate_trading_dates_handling),
        ('test_real_case_600036_simulation', test_class.test_real_case_600036_simulation),
        ('test_all_horizons_simultaneously_correct', test_class.test_all_horizons_simultaneously_correct),
        ('test_date_format_consistency', test_class.test_date_format_consistency),
        ('test_unordered_trading_calendar', test_class.test_unordered_trading_calendar),
        ('test_date_to_idx_mapping_performance', test_class.test_date_to_idx_mapping_performance),
    ]
    
    print("=" * 80)
    print("运行多 horizon 标签计算修复测试")
    print("=" * 80)
    
    failed = []
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        try:
            test_func()
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            failed.append((test_name, e))
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    if failed:
        print(f"测试结果: {len(tests) - len(failed)}/{len(tests)} 通过")
        print("\n失败的测试:")
        for test_name, error in failed:
            print(f"  - {test_name}: {error}")
        sys.exit(1)
    else:
        print(f"测试结果: 全部 {len(tests)} 个测试通过 ✓")
        sys.exit(0)
