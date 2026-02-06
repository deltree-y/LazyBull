"""测试权益曲线交易（ECT）功能

验证 ECT 配置、监控器计算、回测集成等功能
"""

import pandas as pd
import numpy as np
import pytest

from src.lazybull.risk.equity_curve import (
    EquityCurveConfig,
    EquityCurveMonitor,
    create_equity_curve_config_from_dict
)


def test_equity_curve_config_creation():
    """测试 ECT 配置创建"""
    # 从字典创建配置
    config_dict = {
        'equity_curve_enabled': True,
        'equity_curve_drawdown_thresholds': [5.0, 10.0, 15.0],
        'equity_curve_exposure_levels': [0.8, 0.6, 0.4],
        'equity_curve_ma_short': 5,
        'equity_curve_ma_long': 20,
        'equity_curve_recovery_mode': 'gradual',
        'equity_curve_recovery_step': 0.1,
    }
    
    config = create_equity_curve_config_from_dict(config_dict)
    
    assert config.enabled is True
    assert config.drawdown_thresholds == [5.0, 10.0, 15.0]
    assert config.exposure_levels == [0.8, 0.6, 0.4]
    assert config.ma_short_window == 5
    assert config.ma_long_window == 20
    assert config.recovery_mode == 'gradual'
    assert config.recovery_step == 0.1


def test_equity_curve_config_validation():
    """测试 ECT 配置验证"""
    # 正常配置应该通过
    config = EquityCurveConfig(
        enabled=True,
        drawdown_thresholds=[5.0, 10.0, 15.0],
        exposure_levels=[0.8, 0.6, 0.4]
    )
    assert config.enabled is True
    
    # 回撤阈值和仓位系数长度不匹配
    with pytest.raises(ValueError):
        EquityCurveConfig(
            enabled=True,
            drawdown_thresholds=[5.0, 10.0],
            exposure_levels=[0.8, 0.6, 0.4]  # 长度不匹配
        )
    
    # 回撤阈值未递增
    with pytest.raises(ValueError):
        EquityCurveConfig(
            enabled=True,
            drawdown_thresholds=[5.0, 15.0, 10.0],  # 未递增
            exposure_levels=[0.8, 0.6, 0.4]
        )


def test_drawdown_calculation():
    """测试回撤计算和仓位系数"""
    config = EquityCurveConfig(
        enabled=True,
        drawdown_thresholds=[5.0, 10.0, 15.0],
        exposure_levels=[0.8, 0.6, 0.4],
        ma_short_window=3,
        ma_long_window=5
    )
    
    monitor = EquityCurveMonitor(config)
    
    # 创建模拟净值序列：从 1.0 到 1.2，然后下跌到 1.08（回撤10%）
    dates = pd.date_range('20240101', periods=20, freq='D')
    nav_values = [1.0, 1.05, 1.1, 1.15, 1.2, 1.18, 1.16, 1.14, 1.12, 1.10, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08]
    nav_series = pd.Series(nav_values, index=dates)
    
    # 计算仓位系数
    exposure, reason = monitor.calculate_exposure(nav_series, '20240120')
    
    # 回撤 10% (1.2 -> 1.08)，应触发第二档
    # 第二档对应 exposure_levels[1] = 0.6
    # 但还要考虑均线趋势
    # 由于净值在下跌，短期均线应该低于长期均线
    # 所以最终系数应该是 min(0.6, 0.5) = 0.5
    
    assert exposure <= 0.6  # 至少被回撤限制了
    assert "回撤" in reason
    assert "10" in reason or "1" in reason  # 应该提到回撤百分比


def test_ma_trend_calculation():
    """测试均线趋势计算"""
    config = EquityCurveConfig(
        enabled=True,
        drawdown_thresholds=[20.0],  # 高阈值，不触发回撤
        exposure_levels=[0.5],
        ma_short_window=3,
        ma_long_window=5,
        ma_exposure_on=1.0,
        ma_exposure_off=0.5
    )
    
    monitor = EquityCurveMonitor(config)
    
    # 创建上涨趋势的净值序列
    dates = pd.date_range('20240101', periods=10, freq='D')
    nav_values = [1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18]
    nav_series = pd.Series(nav_values, index=dates)
    
    exposure, reason = monitor.calculate_exposure(nav_series, '20240110')
    
    # 上涨趋势，短期均线应该高于长期均线
    # 回撤小，不触发回撤限制
    # 因此系数应该是 1.0
    assert exposure >= 0.9  # 允许一些浮动
    assert "均线趋势向上" in reason or "趋势" in reason


def test_recovery_mechanism():
    """测试恢复机制"""
    config = EquityCurveConfig(
        enabled=True,
        drawdown_thresholds=[10.0],
        exposure_levels=[0.5],
        ma_short_window=3,
        ma_long_window=5,
        recovery_mode='gradual',
        recovery_step=0.1,
        recovery_delay_periods=1
    )
    
    monitor = EquityCurveMonitor(config)
    
    # 第一次：大回撤，降仓
    dates1 = pd.date_range('20240101', periods=10, freq='D')
    nav_values1 = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.85, 0.85, 0.85, 0.85]
    nav_series1 = pd.Series(nav_values1, index=dates1)
    
    exposure1, reason1 = monitor.calculate_exposure(nav_series1, '20240110')
    
    # 应该降仓
    assert exposure1 < 1.0
    
    # 第二次：净值恢复，但应该逐步恢复
    dates2 = pd.date_range('20240101', periods=15, freq='D')
    nav_values2 = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.85, 0.85, 0.85, 0.85, 0.90, 0.95, 1.0, 1.05, 1.1]
    nav_series2 = pd.Series(nav_values2, index=dates2)
    
    exposure2, reason2 = monitor.calculate_exposure(nav_series2, '20240115')
    
    # 由于恢复机制，不应该立即满仓
    # 但应该比第一次高一些（如果过了等待期）
    assert exposure2 >= exposure1 or "恢复" in reason2


def test_ect_disabled():
    """测试 ECT 禁用时的行为"""
    config = EquityCurveConfig(enabled=False)
    monitor = EquityCurveMonitor(config)
    
    # 创建任意净值序列
    dates = pd.date_range('20240101', periods=10, freq='D')
    nav_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05]
    nav_series = pd.Series(nav_values, index=dates)
    
    exposure, reason = monitor.calculate_exposure(nav_series, '20240110')
    
    # ECT 禁用时应该返回 1.0
    assert exposure == 1.0
    assert "未启用" in reason


def test_empty_nav_history():
    """测试空 NAV 历史"""
    config = EquityCurveConfig(enabled=True)
    monitor = EquityCurveMonitor(config)
    
    # 空 Series
    nav_series = pd.Series(dtype=float)
    
    exposure, reason = monitor.calculate_exposure(nav_series, '20240110')
    
    # 空历史应该返回默认值 1.0
    assert exposure == 1.0
    assert "为空" in reason


def test_immediate_recovery_mode():
    """测试立即恢复模式"""
    config = EquityCurveConfig(
        enabled=True,
        drawdown_thresholds=[10.0],
        exposure_levels=[0.5],
        ma_short_window=3,
        ma_long_window=5,
        recovery_mode='immediate',  # 立即恢复
    )
    
    monitor = EquityCurveMonitor(config)
    
    # 第一次：大回撤
    dates1 = pd.date_range('20240101', periods=10, freq='D')
    nav_values1 = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.85, 0.85, 0.85, 0.85]
    nav_series1 = pd.Series(nav_values1, index=dates1)
    
    exposure1, _ = monitor.calculate_exposure(nav_series1, '20240110')
    assert exposure1 < 1.0
    
    # 第二次：净值完全恢复
    dates2 = pd.date_range('20240101', periods=15, freq='D')
    nav_values2 = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.85, 0.85, 0.85, 0.85, 0.95, 1.05, 1.15, 1.25, 1.3]
    nav_series2 = pd.Series(nav_values2, index=dates2)
    
    exposure2, _ = monitor.calculate_exposure(nav_series2, '20240115')
    
    # 立即恢复模式，应该接近满仓（取决于均线）
    # 至少应该比 gradual 模式恢复得快
    assert exposure2 > exposure1


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
