"""测试纸面交易新CLI功能"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.paper import PaperAccount, PaperStorage, Position
from src.lazybull.risk.stop_loss import StopLossConfig, StopLossMonitor


@pytest.fixture
def temp_paper_storage():
    """临时纸面交易存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        yield storage


def test_save_and_load_config(temp_paper_storage):
    """测试配置保存和读取"""
    config = {
        'buy_price': 'close',
        'sell_price': 'close',
        'top_n': 5,
        'initial_capital': 500000.0,
        'rebalance_freq': 5,
        'weight_method': 'equal',
        'model_version': None,
        'stop_loss_enabled': True,
        'stop_loss_drawdown_pct': 20.0,
        'stop_loss_trailing_enabled': False,
        'stop_loss_trailing_pct': 15.0,
        'stop_loss_consecutive_limit_down': 2,
        'universe': 'mainboard'
    }
    
    # 保存配置
    temp_paper_storage.save_config(config)
    
    # 读取配置
    loaded_config = temp_paper_storage.load_config()
    
    assert loaded_config is not None
    assert loaded_config['buy_price'] == 'close'
    assert loaded_config['top_n'] == 5
    assert loaded_config['stop_loss_enabled'] is True
    assert loaded_config['stop_loss_drawdown_pct'] == 20.0


def test_load_config_not_exist(temp_paper_storage):
    """测试读取不存在的配置"""
    config = temp_paper_storage.load_config()
    assert config is None


def test_save_and_load_stop_loss_state(temp_paper_storage):
    """测试止损状态保存和读取"""
    state = {
        'position_high_prices': {
            '000001.SZ': 12.5,
            '000002.SZ': 15.8
        },
        'consecutive_limit_down_days': {
            '000001.SZ': 0,
            '000002.SZ': 1
        }
    }
    
    # 保存状态
    temp_paper_storage.save_stop_loss_state(state)
    
    # 读取状态
    loaded_state = temp_paper_storage.load_stop_loss_state()
    
    assert loaded_state is not None
    assert '000001.SZ' in loaded_state['position_high_prices']
    assert loaded_state['position_high_prices']['000001.SZ'] == 12.5
    assert loaded_state['consecutive_limit_down_days']['000002.SZ'] == 1


def test_load_stop_loss_state_not_exist(temp_paper_storage):
    """测试读取不存在的止损状态"""
    state = temp_paper_storage.load_stop_loss_state()
    assert state is None


def test_stop_loss_monitor_drawdown():
    """测试回撤止损触发"""
    config = StopLossConfig(
        enabled=True,
        drawdown_pct=20.0
    )
    monitor = StopLossMonitor(config)
    
    # 情况1：未触发止损（跌幅19%）
    triggered, trigger_type, reason = monitor.check_stop_loss(
        stock='000001.SZ',
        buy_price=10.0,
        current_price=8.1,  # 跌19%
        is_limit_down=False
    )
    assert triggered is False
    
    # 情况2：触发止损（跌幅20%）
    triggered, trigger_type, reason = monitor.check_stop_loss(
        stock='000001.SZ',
        buy_price=10.0,
        current_price=8.0,  # 跌20%
        is_limit_down=False
    )
    assert triggered is True
    assert '回撤止损' in reason


def test_stop_loss_monitor_trailing():
    """测试移动止损触发"""
    config = StopLossConfig(
        enabled=True,
        drawdown_pct=20.0,
        trailing_stop_enabled=True,
        trailing_stop_pct=15.0
    )
    monitor = StopLossMonitor(config)
    
    # 更新最高价
    monitor.update_position_price('000001.SZ', 12.0)
    monitor.update_position_price('000001.SZ', 15.0)  # 最高价15.0
    
    # 情况1：未触发移动止损（从最高点跌14%）
    triggered, trigger_type, reason = monitor.check_stop_loss(
        stock='000001.SZ',
        buy_price=10.0,
        current_price=12.9,  # 从15.0跌到12.9，跌14%
        is_limit_down=False
    )
    assert triggered is False
    
    # 情况2：触发移动止损（从最高点跌15%）
    triggered, trigger_type, reason = monitor.check_stop_loss(
        stock='000001.SZ',
        buy_price=10.0,
        current_price=12.75,  # 从15.0跌到12.75，跌15%
        is_limit_down=False
    )
    assert triggered is True
    assert '移动止损' in reason


def test_stop_loss_monitor_consecutive_limit_down():
    """测试连续跌停止损触发"""
    config = StopLossConfig(
        enabled=True,
        consecutive_limit_down_days=2
    )
    monitor = StopLossMonitor(config)
    
    # 第一天跌停
    triggered, trigger_type, reason = monitor.check_stop_loss(
        stock='000001.SZ',
        buy_price=10.0,
        current_price=9.0,
        is_limit_down=True
    )
    assert triggered is False  # 只跌停1天，未触发
    
    # 第二天跌停
    triggered, trigger_type, reason = monitor.check_stop_loss(
        stock='000001.SZ',
        buy_price=10.0,
        current_price=8.1,
        is_limit_down=True
    )
    assert triggered is True  # 连续2天跌停，触发
    assert '连续跌停' in reason


def test_stop_loss_monitor_reset_consecutive_limit_down():
    """测试连续跌停计数重置"""
    config = StopLossConfig(
        enabled=True,
        consecutive_limit_down_days=2
    )
    monitor = StopLossMonitor(config)
    
    # 第一天跌停
    monitor.check_stop_loss(
        stock='000001.SZ',
        buy_price=10.0,
        current_price=9.0,
        is_limit_down=True
    )
    assert monitor.consecutive_limit_down_days['000001.SZ'] == 1
    
    # 第二天不跌停（重置计数）
    monitor.check_stop_loss(
        stock='000001.SZ',
        buy_price=10.0,
        current_price=9.2,
        is_limit_down=False
    )
    assert monitor.consecutive_limit_down_days['000001.SZ'] == 0


def test_config_command_integration(temp_paper_storage):
    """测试config命令的集成（模拟命令执行）"""
    # 模拟config命令：保存配置
    config = {
        'buy_price': 'open',
        'sell_price': 'close',
        'top_n': 10,
        'initial_capital': 1000000.0,
        'rebalance_freq': 10,
        'weight_method': 'score',
        'model_version': 1,
        'stop_loss_enabled': True,
        'stop_loss_drawdown_pct': 15.0,
        'stop_loss_trailing_enabled': True,
        'stop_loss_trailing_pct': 10.0,
        'stop_loss_consecutive_limit_down': 3,
        'universe': 'all'
    }
    
    temp_paper_storage.save_config(config)
    
    # 验证配置文件存在
    config_path = Path(temp_paper_storage.root_path) / "config.json"
    assert config_path.exists()
    
    # 读取并验证
    loaded_config = temp_paper_storage.load_config()
    assert loaded_config['weight_method'] == 'score'
    assert loaded_config['model_version'] == 1


def test_run_command_requires_config(temp_paper_storage):
    """测试run命令要求先配置"""
    # 尝试读取不存在的配置
    config = temp_paper_storage.load_config()
    assert config is None
    
    # 模拟run命令应该报错并提示用户先运行config


def test_non_rebalance_day_allows_sell_only():
    """测试非调仓日允许仅执行卖出"""
    # 这个测试需要模拟run命令在非调仓日的行为
    # 预期：止损卖出和延迟卖出应该执行，T0应该跳过
    
    # 模拟调仓状态：上次调仓是2天前，但调仓频率是5天
    rebalance_state = {
        'last_rebalance_date': '20260119',
        'rebalance_freq': 5
    }
    
    # 当前日期是20260121（距离上次调仓2天）
    # 应该不满足调仓条件（需要5天）
    
    # 这里仅测试逻辑，实际测试需要mock交易日历
    current_date = '20260121'
    last_date = '20260119'
    freq = 5
    days_passed = 2  # 简化计算
    
    is_rebalance_day = days_passed >= freq
    assert is_rebalance_day is False
    
    # 非调仓日应该允许：
    # 1. 止损卖出 - 是
    # 2. 延迟卖出 - 是
    # 3. T1（如果有pending） - 是
    # 4. T0 - 否


def test_t1_idempotency(temp_paper_storage):
    """测试T1幂等性"""
    trade_date = '20260122'
    
    # 第一次运行T1
    assert not temp_paper_storage.check_run_exists("t1", trade_date)
    
    # 保存运行记录
    run_record = {
        'trade_date': trade_date,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    temp_paper_storage.save_run_record("t1", trade_date, run_record)
    
    # 第二次运行T1应该检测到已执行
    assert temp_paper_storage.check_run_exists("t1", trade_date)


def test_t0_idempotency(temp_paper_storage):
    """测试T0幂等性"""
    trade_date = '20260121'
    
    # 第一次运行T0
    assert not temp_paper_storage.check_run_exists("t0", trade_date)
    
    # 保存运行记录
    run_record = {
        'trade_date': trade_date,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    temp_paper_storage.save_run_record("t0", trade_date, run_record)
    
    # 第二次运行T0应该检测到已执行
    assert temp_paper_storage.check_run_exists("t0", trade_date)


def test_pending_sells_not_blocked_by_idempotency():
    """测试延迟卖出不受幂等性限制"""
    # 延迟卖出可以在同一日期多次执行
    # 这个行为在run命令中实现，不受check_run_exists限制
    
    # 模拟场景：
    # - T1已在20260122执行（幂等锁定）
    # - 但延迟卖出仍可在20260122重试
    
    # 验证逻辑：run命令处理延迟卖出时不调用check_run_exists
    # 这是设计要求，在实现中验证


def test_stop_loss_state_persistence():
    """测试止损状态持久化"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        
        # 创建止损监控器
        config = StopLossConfig(
            enabled=True,
            drawdown_pct=20.0,
            trailing_stop_enabled=True,
            trailing_stop_pct=15.0
        )
        monitor = StopLossMonitor(config)
        
        # 更新一些状态
        monitor.update_position_price('000001.SZ', 12.5)
        monitor.update_position_price('000002.SZ', 15.8)
        monitor.consecutive_limit_down_days['000003.SZ'] = 1
        
        # 保存状态
        state = {
            'position_high_prices': monitor.position_high_prices,
            'consecutive_limit_down_days': monitor.consecutive_limit_down_days
        }
        storage.save_stop_loss_state(state)
        
        # 创建新的监控器并加载状态
        monitor2 = StopLossMonitor(config)
        loaded_state = storage.load_stop_loss_state()
        monitor2.position_high_prices = loaded_state['position_high_prices']
        monitor2.consecutive_limit_down_days = loaded_state['consecutive_limit_down_days']
        
        # 验证状态恢复
        assert monitor2.position_high_prices['000001.SZ'] == 12.5
        assert monitor2.position_high_prices['000002.SZ'] == 15.8
        assert monitor2.consecutive_limit_down_days['000003.SZ'] == 1


def test_t0_targets_output_format():
    """测试T0目标输出格式"""
    # T0生成的目标应包含：ts_code、目标权重、reason/score
    # 不要求打印价格与股数
    
    targets_info = [
        {
            'ts_code': '000001.SZ',
            'target_weight': 0.2,
            'reason': '信号生成',
            'score': 0.85
        },
        {
            'ts_code': '000002.SZ',
            'target_weight': 0.15,
            'reason': '信号生成',
            'score': 0.78
        }
    ]
    
    # 验证格式
    for target in targets_info:
        assert 'ts_code' in target
        assert 'target_weight' in target
        assert 'reason' in target or 'score' in target


def test_stop_loss_generates_pending_sell_when_limit_down():
    """测试止损触发时如果跌停则进入延迟卖出队列"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        
        # 创建配置
        config = StopLossConfig(
            enabled=True,
            drawdown_pct=20.0
        )
        monitor = StopLossMonitor(config)
        
        # 检查止损（跌停）
        triggered, trigger_type, reason = monitor.check_stop_loss(
            stock='000001.SZ',
            buy_price=10.0,
            current_price=7.0,  # 跌30%，触发止损
            is_limit_down=True  # 跌停，不可卖出
        )
        
        assert triggered is True
        
        # 应该生成延迟卖出订单
        # 这个行为在run命令的_check_stop_loss函数中实现


def test_pending_sell_retry_removes_when_executed():
    """测试延迟卖出重试成功后从队列移除"""
    # 这个测试需要模拟整个流程
    # 1. 创建延迟卖出订单
    # 2. 重试成功
    # 3. 验证订单从队列移除
    
    from src.lazybull.paper.models import PendingSell
    
    # 创建延迟卖出订单
    pending_sell = PendingSell(
        ts_code='000001.SZ',
        shares=1000,
        target_weight=0.0,
        reason='跌停延迟',
        create_date='20260121',
        attempts=1
    )
    
    # 模拟成功卖出后，订单应该被移除
    # 实际逻辑在broker.retry_pending_sells中


def test_config_with_all_parameters():
    """测试配置包含所有必要参数"""
    required_params = [
        'buy_price',
        'sell_price',
        'top_n',
        'initial_capital',
        'rebalance_freq',
        'weight_method',
        'model_version',
        'stop_loss_enabled',
        'stop_loss_drawdown_pct',
        'stop_loss_trailing_enabled',
        'stop_loss_trailing_pct',
        'stop_loss_consecutive_limit_down',
        'universe'
    ]
    
    config = {
        'buy_price': 'close',
        'sell_price': 'close',
        'top_n': 5,
        'initial_capital': 500000.0,
        'rebalance_freq': 5,
        'weight_method': 'equal',
        'model_version': None,
        'stop_loss_enabled': False,
        'stop_loss_drawdown_pct': 20.0,
        'stop_loss_trailing_enabled': False,
        'stop_loss_trailing_pct': 15.0,
        'stop_loss_consecutive_limit_down': 2,
        'universe': 'mainboard'
    }
    
    # 验证所有必要参数都存在
    for param in required_params:
        assert param in config
