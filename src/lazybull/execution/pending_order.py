"""延迟订单管理模块

管理因涨跌停或停牌而无法立即执行的订单，支持延迟重试机制。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger


@dataclass
class PendingOrder:
    """延迟订单数据类
    
    Attributes:
        stock: 股票代码
        action: 操作类型 ('buy' 或 'sell')
        target_value: 目标市值（仅买入时有效）
        signal_date: 信号生成日期
        create_date: 订单创建日期（首次尝试日期）
        retry_count: 重试次数
        last_reason: 上次无法交易的原因
    """
    stock: str
    action: str  # 'buy' or 'sell'
    target_value: Optional[float]  # 仅买入时有效
    signal_date: pd.Timestamp
    create_date: pd.Timestamp
    retry_count: int = 0
    last_reason: Optional[str] = None
    
    def __repr__(self):
        return (
            f"PendingOrder(stock={self.stock}, action={self.action}, "
            f"target_value={self.target_value}, signal_date={self.signal_date.date()}, "
            f"create_date={self.create_date.date()}, retry_count={self.retry_count}, "
            f"last_reason={self.last_reason})"
        )


class PendingOrderManager:
    """延迟订单管理器
    
    管理因涨跌停或停牌而延迟的交易订单，支持：
    - 添加延迟订单
    - 重试延迟订单
    - 超时/超过重试次数后放弃订单
    - 查询和统计功能
    """
    
    def __init__(
        self,
        max_retry_count: int = 5,
        max_retry_days: int = 10
    ):
        """初始化延迟订单管理器
        
        Args:
            max_retry_count: 最大重试次数，默认5次
            max_retry_days: 最大延迟天数（交易日），默认10天
        """
        self.max_retry_count = max_retry_count
        self.max_retry_days = max_retry_days
        
        # 延迟订单队列：按股票代码和操作类型分组
        # key: (stock, action), value: PendingOrder
        self.pending_orders: Dict[Tuple[str, str], PendingOrder] = {}
        
        # 统计信息
        self.total_added = 0
        self.total_expired = 0
        self.total_succeeded = 0
        
        logger.info(
            f"延迟订单管理器初始化: 最大重试次数={max_retry_count}, "
            f"最大延迟天数={max_retry_days}"
        )
    
    def add_order(
        self,
        stock: str,
        action: str,
        current_date: pd.Timestamp,
        signal_date: pd.Timestamp,
        target_value: Optional[float] = None,
        reason: Optional[str] = None
    ) -> None:
        """添加延迟订单
        
        Args:
            stock: 股票代码
            action: 操作类型 ('buy' 或 'sell')
            current_date: 当前日期
            signal_date: 信号生成日期
            target_value: 目标市值（仅买入时需要）
            reason: 延迟原因（停牌/涨停/跌停）
        """
        key = (stock, action)
        
        if key in self.pending_orders:
            # 已存在，增加重试次数
            order = self.pending_orders[key]
            order.retry_count += 1
            order.last_reason = reason
            logger.debug(
                f"延迟订单更新: {stock} {action} "
                f"(重试次数: {order.retry_count}, 原因: {reason})"
            )
        else:
            # 新建订单
            order = PendingOrder(
                stock=stock,
                action=action,
                target_value=target_value,
                signal_date=signal_date,
                create_date=current_date,
                retry_count=1,
                last_reason=reason
            )
            self.pending_orders[key] = order
            self.total_added += 1
            logger.info(
                f"添加延迟订单: {stock} {action} "
                f"(信号日期: {signal_date.date()}, 原因: {reason})"
            )
    
    def get_orders_to_retry(self, current_date: pd.Timestamp) -> List[PendingOrder]:
        """获取当前应重试的订单列表
        
        检查所有延迟订单，过滤掉已超时或超过重试次数的订单。
        
        Args:
            current_date: 当前日期
            
        Returns:
            可重试的订单列表
        """
        orders_to_retry = []
        expired_keys = []
        
        for key, order in self.pending_orders.items():
            # 检查是否超过最大重试次数
            if order.retry_count > self.max_retry_count:
                logger.info(
                    f"延迟订单超过最大重试次数，放弃: {order.stock} {order.action} "
                    f"(重试次数: {order.retry_count}, 最大重试: {self.max_retry_count})"
                )
                expired_keys.append(key)
                self.total_expired += 1
                continue
            
            # 检查是否超过最大延迟天数（简化：使用自然日计算）
            days_elapsed = (current_date - order.create_date).days
            if days_elapsed > self.max_retry_days:
                logger.info(
                    f"延迟订单超过最大延迟天数，放弃: {order.stock} {order.action} "
                    f"(已延迟: {days_elapsed}天, 最大延迟: {self.max_retry_days}天)"
                )
                expired_keys.append(key)
                self.total_expired += 1
                continue
            
            orders_to_retry.append(order)
        
        # 移除过期订单
        for key in expired_keys:
            del self.pending_orders[key]
        
        return orders_to_retry
    
    def mark_success(self, success_date: pd.Timestamp, stock: str, action: str) -> None:
        """标记订单执行成功并移除
        
        Args:
            stock: 股票代码
            action: 操作类型
        """
        key = (stock, action)
        if key in self.pending_orders:
            order = self.pending_orders[key]
            logger.info(
                f"延迟订单执行成功: {stock} {action} "
                f"(重试次数: {order.retry_count}, "
                f"延迟天数: {(success_date - order.create_date).days})"
            )
            del self.pending_orders[key]
            self.total_succeeded += 1
    
    def remove_order(self, stock: str, action: str) -> None:
        """移除指定订单（手动放弃）
        
        Args:
            stock: 股票代码
            action: 操作类型
        """
        key = (stock, action)
        if key in self.pending_orders:
            del self.pending_orders[key]
            logger.debug(f"手动移除延迟订单: {stock} {action}")
    
    def has_order(self, stock: str, action: str) -> bool:
        """检查是否存在指定订单
        
        Args:
            stock: 股票代码
            action: 操作类型
            
        Returns:
            True 表示存在，False 表示不存在
        """
        return (stock, action) in self.pending_orders
    
    def get_pending_count(self) -> int:
        """获取当前延迟订单数量"""
        return len(self.pending_orders)
    
    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息
        
        Returns:
            包含统计信息的字典：
            {
                'pending': 当前待处理数,
                'total_added': 累计添加数,
                'total_succeeded': 累计成功数,
                'total_expired': 累计过期数
            }
        """
        return {
            'pending': len(self.pending_orders),
            'total_added': self.total_added,
            'total_succeeded': self.total_succeeded,
            'total_expired': self.total_expired
        }
    
    def clear_all(self) -> None:
        """清空所有延迟订单"""
        count = len(self.pending_orders)
        self.pending_orders.clear()
        if count > 0:
            logger.info(f"清空所有延迟订单，共 {count} 条")
    
    def get_all_orders(self) -> List[PendingOrder]:
        """获取所有延迟订单列表"""
        return list(self.pending_orders.values())
