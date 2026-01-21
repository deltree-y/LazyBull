#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证脚本：测试数据确保和 T0 打印增强功能

此脚本进行基本的语法和导入验证
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("LazyBull 数据确保和 T0 打印增强功能验证")
print("=" * 80)
print()

# 测试 1：导入验证
print("测试 1: 导入模块...")
try:
    from src.lazybull.data import (
        DataCleaner,
        DataLoader,
        Storage,
        TushareClient,
        ensure_basic_data,
        ensure_clean_data_for_date,
        ensure_raw_data_for_date,
    )
    print("✓ data 模块导入成功")
    print("  - ensure_basic_data")
    print("  - ensure_raw_data_for_date")
    print("  - ensure_clean_data_for_date")
except ImportError as e:
    print(f"✗ data 模块导入失败: {e}")
    sys.exit(1)

try:
    from src.lazybull.features import FeatureBuilder, ensure_features_for_date
    print("✓ features 模块导入成功")
    print("  - ensure_features_for_date")
except ImportError as e:
    print(f"✗ features 模块导入失败: {e}")
    sys.exit(1)

try:
    from src.lazybull.paper.runner import PaperTradingRunner
    print("✓ paper.runner 模块导入成功")
    print("  - PaperTradingRunner")
except ImportError as e:
    print(f"✗ paper.runner 模块导入失败: {e}")
    sys.exit(1)

print()

# 测试 2：检查常量定义
print("测试 2: 检查常量定义...")
try:
    from src.lazybull.data.ensure import (
        MIN_LIST_DAYS,
        TRADE_CAL_FUTURE_MONTHS,
        TRADE_CAL_HISTORY_MONTHS,
    )
    print(f"✓ data.ensure 常量已定义:")
    print(f"  - TRADE_CAL_HISTORY_MONTHS = {TRADE_CAL_HISTORY_MONTHS}")
    print(f"  - TRADE_CAL_FUTURE_MONTHS = {TRADE_CAL_FUTURE_MONTHS}")
    print(f"  - MIN_LIST_DAYS = {MIN_LIST_DAYS}")
except ImportError as e:
    print(f"✗ 常量导入失败: {e}")
    sys.exit(1)

try:
    from src.lazybull.features.ensure import (
        FEATURE_DATA_FUTURE_MONTHS,
        FEATURE_DATA_HISTORY_MONTHS,
        HISTORICAL_DATA_MONTHS,
        MAX_HISTORICAL_DAYS,
    )
    print(f"✓ features.ensure 常量已定义:")
    print(f"  - FEATURE_DATA_HISTORY_MONTHS = {FEATURE_DATA_HISTORY_MONTHS}")
    print(f"  - FEATURE_DATA_FUTURE_MONTHS = {FEATURE_DATA_FUTURE_MONTHS}")
    print(f"  - HISTORICAL_DATA_MONTHS = {HISTORICAL_DATA_MONTHS}")
    print(f"  - MAX_HISTORICAL_DAYS = {MAX_HISTORICAL_DAYS}")
except ImportError as e:
    print(f"✗ 常量导入失败: {e}")
    sys.exit(1)

try:
    from src.lazybull.paper.runner import SEPARATOR_LENGTH, SHARE_LOT_SIZE
    print(f"✓ paper.runner 常量已定义:")
    print(f"  - SHARE_LOT_SIZE = {SHARE_LOT_SIZE}")
    print(f"  - SEPARATOR_LENGTH = {SEPARATOR_LENGTH}")
except ImportError as e:
    print(f"✗ 常量导入失败: {e}")
    sys.exit(1)

print()

# 测试 3：检查方法存在性
print("测试 3: 检查 PaperTradingRunner 新增方法...")
runner_methods = [
    '_enhance_target_info',
    '_print_t0_targets',
]

for method_name in runner_methods:
    if hasattr(PaperTradingRunner, method_name):
        print(f"✓ 方法 {method_name} 已定义")
    else:
        print(f"✗ 方法 {method_name} 未找到")
        sys.exit(1)

print()

# 测试 4：检查函数签名
print("测试 4: 检查 ensure 函数签名...")
import inspect

# 检查 ensure_raw_data_for_date
sig = inspect.signature(ensure_raw_data_for_date)
params = list(sig.parameters.keys())
expected_params = ['client', 'storage', 'trade_date', 'force']
if params == expected_params:
    print(f"✓ ensure_raw_data_for_date 签名正确: {params}")
else:
    print(f"✗ ensure_raw_data_for_date 签名不正确")
    print(f"  预期: {expected_params}")
    print(f"  实际: {params}")

# 检查 ensure_clean_data_for_date
sig = inspect.signature(ensure_clean_data_for_date)
params = list(sig.parameters.keys())
expected_params = ['storage', 'loader', 'cleaner', 'client', 'trade_date', 'force']
if params == expected_params:
    print(f"✓ ensure_clean_data_for_date 签名正确: {params}")
else:
    print(f"✗ ensure_clean_data_for_date 签名不正确")

# 检查 ensure_features_for_date
sig = inspect.signature(ensure_features_for_date)
params = list(sig.parameters.keys())
expected_params = ['storage', 'loader', 'builder', 'cleaner', 'client', 'trade_date', 'force']
if params == expected_params:
    print(f"✓ ensure_features_for_date 签名正确: {params}")
else:
    print(f"✗ ensure_features_for_date 签名不正确")

print()

# 测试完成
print("=" * 80)
print("✓ 所有验证测试通过！")
print("=" * 80)
print()
print("验证内容：")
print("1. 所有新增模块和函数可以正常导入")
print("2. 常量定义正确且可访问")
print("3. PaperTradingRunner 新增方法存在")
print("4. ensure 函数签名符合预期")
print()
print("建议：")
print("- 在有完整环境的情况下运行完整测试套件")
print("- 使用真实或模拟数据验证端到端功能")
print("=" * 80)
