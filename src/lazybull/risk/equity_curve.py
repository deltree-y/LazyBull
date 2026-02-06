"""权益曲线交易（Equity Curve Trading, ECT）模块

基于账户盈亏曲线的仓位/风险管理功能。

功能：
- 计算权益曲线（NAV）的回撤
- 根据回撤分档控制仓位
- 计算 NAV 的短期/长期均线并判断趋势
- 输出 0.0~1.0 的仓位系数（exposure_multiplier）
- 风险解除后逐步恢复仓位
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from loguru import logger


class ECTSignalType(Enum):
    """ECT 信号类型"""
    DRAWDOWN = "drawdown"  # 回撤触发
    MA_TREND = "ma_trend"  # 均线趋势
    RECOVERY = "recovery"  # 恢复中


@dataclass
class EquityCurveConfig:
    """权益曲线交易配置"""
    # 基础开关
    enabled: bool = False
    
    # 回撤控制配置（分档）
    drawdown_thresholds: List[float] = None  # 回撤阈值列表（百分比），例如 [5.0, 10.0, 15.0]
    exposure_levels: List[float] = None  # 对应的仓位系数列表，例如 [0.8, 0.5, 0.2]
    
    # 均线趋势过滤配置
    ma_short_window: int = 5  # 短期均线窗口（交易日）
    ma_long_window: int = 20  # 长期均线窗口（交易日）
    ma_exposure_on: float = 1.0  # 均线趋势向上时的系数
    ma_exposure_off: float = 0.5  # 均线趋势向下时的系数
    
    # 恢复策略配置
    recovery_mode: str = "gradual"  # 恢复模式：gradual=逐步恢复，immediate=立即恢复
    recovery_step: float = 0.1  # 逐步恢复时每次增加的系数（每个调仓周期）
    recovery_delay_periods: int = 1  # 恢复前等待的调仓周期数
    
    # 限制
    min_exposure: float = 0.0  # 最小仓位系数
    max_exposure: float = 1.0  # 最大仓位系数
    
    def __post_init__(self):
        """验证配置"""
        # 设置默认回撤阈值和仓位系数
        if self.drawdown_thresholds is None:
            self.drawdown_thresholds = [5.0, 10.0, 15.0, 20.0]
        if self.exposure_levels is None:
            self.exposure_levels = [0.8, 0.6, 0.4, 0.2]
        
        # 验证回撤阈值和仓位系数长度一致
        if len(self.drawdown_thresholds) != len(self.exposure_levels):
            raise ValueError(
                f"回撤阈值和仓位系数数量不匹配: "
                f"drawdown_thresholds={len(self.drawdown_thresholds)}, "
                f"exposure_levels={len(self.exposure_levels)}"
            )
        
        # 验证回撤阈值递增
        for i in range(1, len(self.drawdown_thresholds)):
            if self.drawdown_thresholds[i] <= self.drawdown_thresholds[i-1]:
                raise ValueError(f"回撤阈值必须递增: {self.drawdown_thresholds}")
        
        # 验证仓位系数递减
        for i in range(1, len(self.exposure_levels)):
            if self.exposure_levels[i] >= self.exposure_levels[i-1]:
                raise ValueError(f"仓位系数必须递减: {self.exposure_levels}")


class EquityCurveMonitor:
    """权益曲线监控器
    
    负责监控权益曲线（NAV）变化，计算仓位系数
    """
    
    def __init__(self, config: EquityCurveConfig):
        """初始化权益曲线监控器
        
        Args:
            config: ECT 配置
        """
        self.config = config
        
        # 恢复状态跟踪
        self.is_recovering = False  # 是否处于恢复状态
        self.recovery_target = 1.0  # 恢复目标系数
        self.recovery_counter = 0  # 恢复计数器（已经过的调仓周期数）
        self.last_exposure = 1.0  # 上一次的仓位系数
        
        logger.info(
            f"权益曲线监控器初始化: enabled={config.enabled}, "
            f"drawdown_thresholds={config.drawdown_thresholds}, "
            f"exposure_levels={config.exposure_levels}, "
            f"ma_windows=({config.ma_short_window}, {config.ma_long_window}), "
            f"recovery_mode={config.recovery_mode}"
        )
    
    def calculate_exposure(
        self,
        nav_history: pd.Series,
        current_date: Optional[str] = None
    ) -> Tuple[float, str]:
        """计算当前的仓位系数
        
        Args:
            nav_history: 历史 NAV 序列，index 为日期，values 为净值
            current_date: 当前日期（用于日志），可选
            
        Returns:
            (exposure_multiplier, reason) 元组
            - exposure_multiplier: 0.0~1.0 的仓位系数
            - reason: 计算原因的中文描述
        """
        if not self.config.enabled:
            return 1.0, "ECT 未启用"
        
        if nav_history is None or len(nav_history) == 0:
            logger.warning("NAV 历史为空，返回默认仓位系数 1.0")
            return 1.0, "NAV 历史为空"
        
        # 确保 nav_history 按日期排序
        nav_history = nav_history.sort_index()
        current_nav = nav_history.iloc[-1]
        
        # 1. 计算回撤
        rolling_max = nav_history.expanding().max()
        current_max = rolling_max.iloc[-1]
        drawdown_pct = (current_nav - current_max) / current_max * 100  # 转为百分比
        
        # 2. 根据回撤确定基础仓位系数
        drawdown_exposure = self._calculate_drawdown_exposure(drawdown_pct)
        
        # 3. 计算均线趋势系数
        ma_exposure = self._calculate_ma_exposure(nav_history)
        
        # 4. 组合两个系数（取较小值，更保守）
        combined_exposure = min(drawdown_exposure, ma_exposure)
        
        # 5. 处理恢复逻辑
        final_exposure = self._apply_recovery_logic(combined_exposure)
        
        # 6. 应用限制
        final_exposure = max(self.config.min_exposure, min(self.config.max_exposure, final_exposure))
        
        # 7. 生成原因描述
        reason = self._generate_reason(
            drawdown_pct, drawdown_exposure, ma_exposure, 
            combined_exposure, final_exposure, current_date
        )
        
        # 8. 更新状态
        self.last_exposure = final_exposure
        
        return final_exposure, reason
    
    def _calculate_drawdown_exposure(self, drawdown_pct: float) -> float:
        """根据回撤计算仓位系数
        
        Args:
            drawdown_pct: 回撤百分比（负数表示回撤）
            
        Returns:
            仓位系数
        """
        # 回撤为负数，转为正数比较
        abs_drawdown = abs(drawdown_pct)
        
        # 根据分档确定仓位系数
        for i, threshold in enumerate(self.config.drawdown_thresholds):
            if abs_drawdown < threshold:
                # 未达到该档位阈值，返回上一档的系数
                if i == 0:
                    return 1.0  # 回撤小于第一档，满仓
                else:
                    return self.config.exposure_levels[i - 1]
        
        # 超过所有阈值，返回最低档系数
        return self.config.exposure_levels[-1]
    
    def _calculate_ma_exposure(self, nav_history: pd.Series) -> float:
        """根据均线趋势计算仓位系数
        
        Args:
            nav_history: NAV 历史序列
            
        Returns:
            仓位系数
        """
        # 如果数据不足，返回默认值
        if len(nav_history) < self.config.ma_long_window:
            return self.config.ma_exposure_on  # 数据不足时默认允许持仓
        
        # 计算短期和长期均线
        ma_short = nav_history.rolling(window=self.config.ma_short_window).mean().iloc[-1]
        ma_long = nav_history.rolling(window=self.config.ma_long_window).mean().iloc[-1]
        
        # 判断趋势
        if ma_short > ma_long:
            # 短期均线在长期均线上方，趋势向上
            return self.config.ma_exposure_on
        else:
            # 短期均线在长期均线下方，趋势向下
            return self.config.ma_exposure_off
    
    def _apply_recovery_logic(self, target_exposure: float) -> float:
        """应用恢复逻辑
        
        Args:
            target_exposure: 目标仓位系数
            
        Returns:
            应用恢复逻辑后的仓位系数
        """
        if self.config.recovery_mode == "immediate":
            # 立即恢复模式：直接使用目标系数
            self.is_recovering = False
            self.recovery_counter = 0
            return target_exposure
        
        # 逐步恢复模式
        if target_exposure < self.last_exposure:
            # 需要降仓，立即执行
            self.is_recovering = False
            self.recovery_counter = 0
            return target_exposure
        elif target_exposure > self.last_exposure:
            # 需要增仓，进入恢复状态
            if not self.is_recovering:
                # 首次进入恢复状态
                self.is_recovering = True
                self.recovery_target = target_exposure
                self.recovery_counter = 0
                return self.last_exposure  # 本次不增仓，等待
            else:
                # 已在恢复状态
                self.recovery_counter += 1
                
                # 检查是否已过等待期
                if self.recovery_counter < self.config.recovery_delay_periods:
                    return self.last_exposure  # 仍在等待期
                
                # 计算逐步增加后的仓位
                new_exposure = self.last_exposure + self.config.recovery_step
                
                if new_exposure >= target_exposure:
                    # 已达到目标，结束恢复
                    self.is_recovering = False
                    self.recovery_counter = 0
                    return target_exposure
                else:
                    # 继续恢复
                    return new_exposure
        else:
            # 目标与当前相同，保持
            return target_exposure
    
    def _generate_reason(
        self,
        drawdown_pct: float,
        drawdown_exposure: float,
        ma_exposure: float,
        combined_exposure: float,
        final_exposure: float,
        current_date: Optional[str] = None
    ) -> str:
        """生成原因描述
        
        Args:
            drawdown_pct: 回撤百分比
            drawdown_exposure: 回撤对应的仓位系数
            ma_exposure: 均线对应的仓位系数
            combined_exposure: 组合后的仓位系数
            final_exposure: 最终仓位系数
            current_date: 当前日期
            
        Returns:
            原因描述（中文）
        """
        date_str = f"[{current_date}] " if current_date else ""
        
        parts = []
        
        # 回撤信息
        abs_drawdown = abs(drawdown_pct)
        if abs_drawdown < self.config.drawdown_thresholds[0]:
            parts.append(f"回撤 {abs_drawdown:.2f}% (正常)")
        else:
            parts.append(f"回撤 {abs_drawdown:.2f}% (触发)")
        
        # 均线信息
        if ma_exposure == self.config.ma_exposure_on:
            parts.append(f"均线趋势向上")
        else:
            parts.append(f"均线趋势向下")
        
        # 恢复状态
        if self.is_recovering:
            parts.append(f"恢复中 ({self.recovery_counter}/{self.config.recovery_delay_periods}周期)")
        
        # 最终系数
        parts.append(f"系数={final_exposure:.2f}")
        
        reason = date_str + "ECT: " + ", ".join(parts)
        
        return reason
    
    def reset(self):
        """重置监控器状态"""
        self.is_recovering = False
        self.recovery_target = 1.0
        self.recovery_counter = 0
        self.last_exposure = 1.0


def create_equity_curve_config_from_dict(config_dict: Dict) -> EquityCurveConfig:
    """从配置字典创建 ECT 配置对象
    
    Args:
        config_dict: 配置字典，通常来自 YAML 配置文件或命令行参数
        
    Returns:
        EquityCurveConfig 对象
        
    示例配置：
        {
            'equity_curve_enabled': True,
            'equity_curve_drawdown_thresholds': [5.0, 10.0, 15.0],
            'equity_curve_exposure_levels': [0.8, 0.6, 0.4],
            'equity_curve_ma_short': 5,
            'equity_curve_ma_long': 20,
            'equity_curve_recovery_mode': 'gradual',
            'equity_curve_recovery_step': 0.1,
        }
    """
    return EquityCurveConfig(
        enabled=config_dict.get('equity_curve_enabled', False),
        drawdown_thresholds=config_dict.get('equity_curve_drawdown_thresholds', [5.0, 10.0, 15.0, 20.0]),
        exposure_levels=config_dict.get('equity_curve_exposure_levels', [0.8, 0.6, 0.4, 0.2]),
        ma_short_window=config_dict.get('equity_curve_ma_short', 5),
        ma_long_window=config_dict.get('equity_curve_ma_long', 20),
        ma_exposure_on=config_dict.get('equity_curve_ma_exposure_on', 1.0),
        ma_exposure_off=config_dict.get('equity_curve_ma_exposure_off', 0.5),
        recovery_mode=config_dict.get('equity_curve_recovery_mode', 'gradual'),
        recovery_step=config_dict.get('equity_curve_recovery_step', 0.1),
        recovery_delay_periods=config_dict.get('equity_curve_recovery_delay', 1),
        min_exposure=config_dict.get('equity_curve_min_exposure', 0.0),
        max_exposure=config_dict.get('equity_curve_max_exposure', 1.0)
    )
