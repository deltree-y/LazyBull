"""止损触发器模块

提供基于回撤、连续跌停等触发条件的止损功能。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from loguru import logger


class StopLossTriggerType(Enum):
    """止损触发类型"""
    DRAWDOWN = "drawdown"  # 回撤止损
    CONSECUTIVE_LIMIT_DOWN = "consecutive_limit_down"  # 连续跌停


@dataclass
class StopLossConfig:
    """止损配置"""
    enabled: bool = False
    # 回撤止损配置
    drawdown_pct: float = 20.0  # 从买入成本回撤超过N%触发止损，默认20%
    # 连续跌停配置
    consecutive_limit_down_days: int = 2  # 连续N天跌停触发止损，默认2天
    # 触发后处理策略
    post_trigger_action: str = "hold_cash"  # 触发后操作：hold_cash=持币，buy_alternative=补买备选


class StopLossMonitor:
    """止损监控器
    
    负责监控持仓的止损触发条件，生成止损信号
    """
    
    def __init__(self, config: StopLossConfig):
        """初始化止损监控器
        
        Args:
            config: 止损配置
        """
        self.config = config
        self.consecutive_limit_down_days: Dict[str, int] = {}  # 记录连续跌停天数
        
        logger.info(
            f"止损监控器初始化: enabled={config.enabled}, "
            f"drawdown_pct={config.drawdown_pct}%, "
            f"consecutive_limit_down_days={config.consecutive_limit_down_days}"
        )
    
    def check_stop_loss(
        self,
        stock: str,
        buy_price: float,
        current_price: float,
        is_limit_down: bool = False
    ) -> Tuple[bool, Optional[StopLossTriggerType], Optional[str]]:
        """检查是否触发止损
        
        Args:
            stock: 股票代码
            buy_price: 买入价格（成本价）
            current_price: 当前价格
            is_limit_down: 当日是否跌停
            
        Returns:
            (是否触发止损, 触发类型, 触发原因描述)
        """
        if not self.config.enabled:
            return False, None, None

        # 0. 买入价有效性检查
        if buy_price <= 0:
            logger.warning(f"  {stock} 买入价无效（{buy_price}），跳过止损检查")
            return False, None, None

        # 1. 检查回撤止损（从买入成本）
        drawdown_from_cost = (current_price - buy_price) / buy_price * 100
        if drawdown_from_cost <= -self.config.drawdown_pct:
            reason = f"回撤止损: 从买入价{buy_price:.2f}下跌至{current_price:.2f}，跌幅{-drawdown_from_cost:.2f}%"
            logger.warning(f"  {stock} 触发止损: {reason}")
            return True, StopLossTriggerType.DRAWDOWN, reason
        
        # 2. 检查连续跌停
        if is_limit_down:
            # 增加连续跌停计数
            self.consecutive_limit_down_days[stock] = self.consecutive_limit_down_days.get(stock, 0) + 1
            consecutive_days = self.consecutive_limit_down_days[stock]
            
            if consecutive_days >= self.config.consecutive_limit_down_days:
                reason = f"连续跌停止损: 连续{consecutive_days}天跌停"
                logger.warning(f"  {stock} 触发止损: {reason}")
                return True, StopLossTriggerType.CONSECUTIVE_LIMIT_DOWN, reason
        else:
            # 重置连续跌停计数
            self.consecutive_limit_down_days[stock] = 0
        
        return False, None, None
    
    def remove_position(self, stock: str):
        """移除持仓监控记录（卖出后调用）
        
        Args:
            stock: 股票代码
        """
        self.consecutive_limit_down_days.pop(stock, None)
    
    def reset(self):
        """重置所有监控记录"""
        self.consecutive_limit_down_days.clear()


def create_stop_loss_config_from_dict(config_dict: Dict) -> StopLossConfig:
    """从配置字典创建止损配置对象
    
    Args:
        config_dict: 配置字典，通常来自 YAML 配置文件
        
    Returns:
        StopLossConfig 对象
    """
    return StopLossConfig(
        enabled=config_dict.get('stop_loss_enabled', False),
        drawdown_pct=config_dict.get('stop_loss_drawdown_pct', 20.0),
        consecutive_limit_down_days=config_dict.get('stop_loss_consecutive_limit_down', 2),
        post_trigger_action=config_dict.get('stop_loss_post_action', 'hold_cash')
    )
