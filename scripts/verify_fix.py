#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证脚本：测试 CLI T1 指令驱动模式修复

此脚本用于手动验证修复效果，不依赖真实数据环境
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试能否正确导入相关模块"""
    print("=" * 80)
    print("测试 1: 导入模块")
    print("=" * 80)
    
    try:
        from src.lazybull.paper import PaperStorage, PaperTradingRunner
        from src.lazybull.paper.models import TradeInstruction
        print("✓ 导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_instruction_model():
    """测试 TradeInstruction 模型"""
    print("\n" + "=" * 80)
    print("测试 2: TradeInstruction 模型")
    print("=" * 80)
    
    try:
        from src.lazybull.paper.models import TradeInstruction
        
        # 创建测试指令
        instruction = TradeInstruction(
            ts_code='603115.SH',
            action='sell',
            shares=100,
            price_type='close',
            reason='减仓调整',
            source_date='20260202',
            target_weight=0.15
        )
        
        print(f"✓ 指令创建成功:")
        print(f"  - 股票代码: {instruction.ts_code}")
        print(f"  - 操作类型: {instruction.action}")
        print(f"  - 股数: {instruction.shares}")
        print(f"  - 价格类型: {instruction.price_type}")
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

def test_runner_signature():
    """测试 run_t0 方法签名是否包含 sell_price_type"""
    print("\n" + "=" * 80)
    print("测试 3: run_t0 方法签名")
    print("=" * 80)
    
    try:
        from src.lazybull.paper import PaperTradingRunner
        import inspect
        
        # 获取 run_t0 的签名
        sig = inspect.signature(PaperTradingRunner.run_t0)
        params = list(sig.parameters.keys())
        
        print(f"run_t0 方法参数: {', '.join(params)}")
        
        if 'sell_price_type' in params:
            print("✓ sell_price_type 参数已添加")
            return True
        else:
            print("✗ sell_price_type 参数未找到")
            return False
    except Exception as e:
        print(f"✗ 签名检查失败: {e}")
        return False

def test_cli_script_changes():
    """检查 CLI 脚本中的关键修改"""
    print("\n" + "=" * 80)
    print("测试 4: CLI 脚本修改检查")
    print("=" * 80)
    
    try:
        cli_file = project_root / "scripts" / "paper_trade.py"
        with open(cli_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            "load_instructions 调用": "load_instructions(trade_date)" in content,
            "execute_instructions 调用": "execute_instructions(" in content,
            "指令驱动模式日志": "【T1 指令驱动模式】" in content,
            "兼容模式日志": "【T1 兼容模式】" in content,
            "sell_price_type 参数传递": "sell_price_type=config['sell_price']" in content,
        }
        
        all_pass = True
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            print(f"{status} {check_name}: {'已添加' if result else '未找到'}")
            if not result:
                all_pass = False
        
        return all_pass
    except Exception as e:
        print(f"✗ CLI 脚本检查失败: {e}")
        return False

def test_runner_changes():
    """检查 runner 中的 sell_price_type 修复"""
    print("\n" + "=" * 80)
    print("测试 5: Runner 中 sell_price_type 修复检查")
    print("=" * 80)
    
    try:
        runner_file = project_root / "src" / "lazybull" / "paper" / "runner.py"
        with open(runner_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否还存在硬编码的 'close'
        # 注意：这个检查可能有误报，因为合法使用 'close' 的地方也会被检测到
        checks = {
            "run_t0 包含 sell_price_type 参数": "def run_t0(\n        self,\n        trade_date: str,\n        buy_price_type: str = 'close',\n        sell_price_type: str = 'close'," in content,
            "_generate_instructions 使用参数": "sell_price_type=sell_price_type," in content,
        }
        
        all_pass = True
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            print(f"{status} {check_name}: {'已修复' if result else '未找到'}")
            if not result:
                all_pass = False
        
        return all_pass
    except Exception as e:
        print(f"✗ Runner 检查失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("LazyBull 指令驱动模式修复验证")
    print("=" * 80)
    
    results = {
        "导入测试": test_imports(),
        "指令模型": test_instruction_model(),
        "run_t0 签名": test_runner_signature(),
        "CLI 脚本修改": test_cli_script_changes(),
        "Runner 修改": test_runner_changes(),
    }
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    all_pass = all(results.values())
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ 所有验证测试通过！")
        print("=" * 80)
        return 0
    else:
        print("✗ 部分测试失败，请检查上述详情")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
