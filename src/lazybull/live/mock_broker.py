"""模拟券商模块

提供MockBroker类，用于纸面交易模拟。仅在执行时（T+1）根据传入的价格模拟立即全部成交。
不支持部分成交、挂单、撤单等复杂场景，适合本地验证流程。
"""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .persistence import SimplePersistence


class MockBroker:
    """模拟券商类
    
    实现基本的买/卖功能，模拟立即全部成交。
    支持资金检查、佣金与滑点计算，调用持久化保存订单、持仓与账户。
    
    限制说明：
    - 不支持部分成交
    - 不支持挂单/撤单
    - 不支持市价单/限价单区分
    - 适合纸面交易验证，不适用于生产环境
    
    Attributes:
        persistence: 持久化模块
        cash: 可用现金
        positions: 持仓字典 {symbol: {"qty": int, "cost_price": float}}
        commission_rate: 佣金率（默认万3）
        slippage: 滑点率（默认0.1%）
    """
    
    def __init__(
        self,
        persistence: SimplePersistence,
        initial_cash: float = 500000.0,
        commission_rate: float = 0.0003,
        slippage: float = 0.001
    ):
        """初始化模拟券商
        
        Args:
            persistence: 持久化模块实例
            initial_cash: 初始资金
            commission_rate: 佣金率（默认万3，即0.0003）
            slippage: 滑点率（默认0.1%，即0.001）
        """
        self.persistence = persistence
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # 从持久化加载状态或使用初始值
        account = persistence.get_account()
        if account.get("cash", 0) > 0:
            self.cash = account["cash"]
            logger.info(f"从持久化加载账户，现金: {self.cash:.2f}")
        else:
            self.cash = initial_cash
            self.persistence.save_account(self.cash, self.cash)
            logger.info(f"初始化账户，初始资金: {self.cash:.2f}")
        
        self.positions = persistence.get_positions()
        logger.info(f"初始化完成，持仓数: {len(self.positions)}, "
                   f"佣金率: {commission_rate*10000:.1f}万分之, "
                   f"滑点: {slippage*100:.2f}%")
    
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        trade_date: str
    ) -> Dict[str, Any]:
        """下单（模拟立即全部成交）
        
        Args:
            symbol: 股票代码
            side: 买卖方向，'buy' 或 'sell'
            qty: 数量（股数）
            price: 价格（执行价，已含滑点）
            trade_date: 交易日期（YYYYMMDD格式）
        
        Returns:
            订单回执字典，包含：
            - local_order_id: 本地订单ID
            - status: 订单状态 ('filled' 或 'rejected')
            - symbol: 股票代码
            - side: 买卖方向
            - qty: 数量
            - filled_qty: 成交数量
            - price: 成交价格
            - trade_value: 成交金额（不含佣金）
            - commission: 佣金
            - reason: 拒单原因（仅当status='rejected'时）
        """
        order_id = f"MO{uuid.uuid4().hex[:12].upper()}"
        
        if side == "buy":
            return self._place_buy_order(order_id, symbol, qty, price, trade_date)
        elif side == "sell":
            return self._place_sell_order(order_id, symbol, qty, price, trade_date)
        else:
            raise ValueError(f"不支持的买卖方向: {side}")
    
    def _place_buy_order(
        self,
        order_id: str,
        symbol: str,
        qty: int,
        price: float,
        trade_date: str
    ) -> Dict[str, Any]:
        """买入订单
        
        Args:
            order_id: 订单ID
            symbol: 股票代码
            qty: 数量
            price: 价格
            trade_date: 交易日期
        
        Returns:
            订单回执
        """
        # 应用买入滑点（买入价格向上滑动）
        exec_price = price * (1 + self.slippage)
        trade_value = qty * exec_price
        commission = max(trade_value * self.commission_rate, 5.0)  # 最低5元
        total_cost = trade_value + commission
        
        # 检查资金是否足够
        if total_cost > self.cash:
            order = {
                "local_order_id": order_id,
                "status": "rejected",
                "symbol": symbol,
                "side": "buy",
                "qty": qty,
                "filled_qty": 0,
                "price": exec_price,
                "trade_value": 0.0,
                "commission": 0.0,
                "trade_date": trade_date,
                "create_time": datetime.now().isoformat(),
                "reason": f"资金不足，需要 {total_cost:.2f}, 可用 {self.cash:.2f}"
            }
            logger.warning(f"买入订单被拒: {symbol}, 原因: {order['reason']}")
            self.persistence.save_order(order)
            return order
        
        # 成交处理
        self.cash -= total_cost
        
        # 更新持仓
        if symbol in self.positions:
            old_qty = self.positions[symbol]["qty"]
            old_cost = self.positions[symbol]["cost_price"] * old_qty
            new_qty = old_qty + qty
            new_cost_price = (old_cost + trade_value) / new_qty
            self.positions[symbol] = {
                "qty": new_qty,
                "cost_price": new_cost_price
            }
        else:
            self.positions[symbol] = {
                "qty": qty,
                "cost_price": exec_price
            }
        
        # 保存订单和状态
        order = {
            "local_order_id": order_id,
            "status": "filled",
            "symbol": symbol,
            "side": "buy",
            "qty": qty,
            "filled_qty": qty,
            "price": exec_price,
            "trade_value": trade_value,
            "commission": commission,
            "trade_date": trade_date,
            "create_time": datetime.now().isoformat()
        }
        self.persistence.save_order(order)
        self._save_state()
        
        logger.info(f"买入成交: {symbol} {qty}股 @{exec_price:.2f}, "
                   f"金额: {trade_value:.2f}, 佣金: {commission:.2f}")
        return order
    
    def _place_sell_order(
        self,
        order_id: str,
        symbol: str,
        qty: int,
        price: float,
        trade_date: str
    ) -> Dict[str, Any]:
        """卖出订单
        
        Args:
            order_id: 订单ID
            symbol: 股票代码
            qty: 数量
            price: 价格
            trade_date: 交易日期
        
        Returns:
            订单回执
        """
        # 检查持仓是否足够
        if symbol not in self.positions or self.positions[symbol]["qty"] < qty:
            available = self.positions.get(symbol, {}).get("qty", 0)
            order = {
                "local_order_id": order_id,
                "status": "rejected",
                "symbol": symbol,
                "side": "sell",
                "qty": qty,
                "filled_qty": 0,
                "price": price,
                "trade_value": 0.0,
                "commission": 0.0,
                "trade_date": trade_date,
                "create_time": datetime.now().isoformat(),
                "reason": f"持仓不足，需要 {qty}, 可用 {available}"
            }
            logger.warning(f"卖出订单被拒: {symbol}, 原因: {order['reason']}")
            self.persistence.save_order(order)
            return order
        
        # 应用卖出滑点（卖出价格向下滑动）
        exec_price = price * (1 - self.slippage)
        trade_value = qty * exec_price
        commission = max(trade_value * self.commission_rate, 5.0)  # 最低5元
        stamp_duty = trade_value * 0.001  # 印花税千分之一
        total_fee = commission + stamp_duty
        net_proceeds = trade_value - total_fee
        
        # 成交处理
        self.cash += net_proceeds
        
        # 更新持仓
        self.positions[symbol]["qty"] -= qty
        if self.positions[symbol]["qty"] == 0:
            del self.positions[symbol]
        
        # 保存订单和状态
        order = {
            "local_order_id": order_id,
            "status": "filled",
            "symbol": symbol,
            "side": "sell",
            "qty": qty,
            "filled_qty": qty,
            "price": exec_price,
            "trade_value": trade_value,
            "commission": commission,
            "stamp_duty": stamp_duty,
            "trade_date": trade_date,
            "create_time": datetime.now().isoformat()
        }
        self.persistence.save_order(order)
        self._save_state()
        
        logger.info(f"卖出成交: {symbol} {qty}股 @{exec_price:.2f}, "
                   f"金额: {trade_value:.2f}, 佣金: {commission:.2f}, 印花税: {stamp_duty:.2f}")
        return order
    
    def _save_state(self):
        """保存当前状态到持久化"""
        total_value = self.cash
        for symbol, pos in self.positions.items():
            # 注意：这里只是简单保存成本价，实际市值需要用当前价格计算
            total_value += pos["qty"] * pos["cost_price"]
        
        self.persistence.save_positions(self.positions)
        self.persistence.save_account(self.cash, total_value)
    
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息
        
        Returns:
            账户信息字典，包含：
            - cash: 可用现金
            - positions: 持仓列表
            - total_value: 账户总值（基于成本价估算）
        """
        total_value = self.cash
        for symbol, pos in self.positions.items():
            total_value += pos["qty"] * pos["cost_price"]
        
        return {
            "cash": self.cash,
            "positions": self.positions.copy(),
            "total_value": total_value
        }
