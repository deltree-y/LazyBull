"""模拟券商实现（T+1 纸面交易）

仅用于 T+1 纸面验证，立即全部成交模式。
不支持部分成交、挂单、撤单等复杂逻辑。
生产环境请替换为真实 BrokerConnector。
"""

import uuid
from datetime import datetime
from typing import Dict, Optional

from loguru import logger

from .persistence import SimplePersistence
from ..common.cost import CostModel


class OrderResult:
    """订单执行结果"""
    
    def __init__(
        self,
        order_id: str,
        code: str,
        direction: str,
        shares: int,
        price: float,
        amount: float,
        cost: float,
        status: str,
        message: str = ""
    ):
        self.order_id = order_id
        self.code = code
        self.direction = direction
        self.shares = shares
        self.price = price
        self.amount = amount
        self.cost = cost
        self.status = status
        self.message = message
    
    def is_success(self) -> bool:
        """是否成功"""
        return self.status == "filled"
    
    def __repr__(self) -> str:
        return (
            f"OrderResult(order_id={self.order_id}, code={self.code}, "
            f"direction={self.direction}, shares={self.shares}, price={self.price:.2f}, "
            f"amount={self.amount:.2f}, cost={self.cost:.2f}, status={self.status})"
        )


class MockBroker:
    """模拟券商
    
    提供简化的纸面交易功能：
    - 立即全部成交（无部分成交）
    - 不支持挂单/撤单
    - 不检查涨跌停（假设可成交）
    - 简化的资金与持仓管理
    """
    
    def __init__(
        self,
        persistence: SimplePersistence,
        cost_model: Optional[CostModel] = None
    ):
        """初始化模拟券商
        
        Args:
            persistence: 持久化实例
            cost_model: 成本模型，不提供则使用默认
        """
        self.persistence = persistence
        self.cost_model = cost_model or self._default_cost_model()
        logger.info("MockBroker 初始化完成")
    
    def _default_cost_model(self) -> CostModel:
        """默认成本模型"""
        return CostModel(
            commission_rate=0.0003,  # 万三佣金
            min_commission=5.0,      # 最小 5 元
            stamp_tax=0.001,         # 千一印花税
            slippage=0.001           # 千一滑点
        )
    
    def _generate_order_id(self) -> str:
        """生成订单 ID"""
        return f"O{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:6]}"
    
    def _now_str(self) -> str:
        """当前时间字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def place_order(
        self,
        code: str,
        direction: str,
        shares: int,
        price: float
    ) -> OrderResult:
        """下单（立即成交）
        
        Args:
            code: 股票代码
            direction: 方向 "buy" 或 "sell"
            shares: 股数
            price: 价格（次日收盘价）
            
        Returns:
            订单结果
        """
        order_id = self._generate_order_id()
        
        # 参数校验
        if shares <= 0:
            return OrderResult(
                order_id=order_id,
                code=code,
                direction=direction,
                shares=shares,
                price=price,
                amount=0.0,
                cost=0.0,
                status="rejected",
                message="股数必须大于 0"
            )
        
        if price <= 0:
            return OrderResult(
                order_id=order_id,
                code=code,
                direction=direction,
                shares=shares,
                price=price,
                amount=0.0,
                cost=0.0,
                status="rejected",
                message="价格必须大于 0"
            )
        
        # 执行买入或卖出
        if direction == "buy":
            return self._execute_buy(order_id, code, shares, price)
        elif direction == "sell":
            return self._execute_sell(order_id, code, shares, price)
        else:
            return OrderResult(
                order_id=order_id,
                code=code,
                direction=direction,
                shares=shares,
                price=price,
                amount=0.0,
                cost=0.0,
                status="rejected",
                message=f"不支持的方向: {direction}"
            )
    
    def _execute_buy(
        self,
        order_id: str,
        code: str,
        shares: int,
        price: float
    ) -> OrderResult:
        """执行买入
        
        Args:
            order_id: 订单 ID
            code: 股票代码
            shares: 股数
            price: 价格
            
        Returns:
            订单结果
        """
        # 计算金额与成本
        amount = shares * price
        cost = self.cost_model.calculate_buy_cost(amount)
        total_need = amount + cost
        
        # 检查资金
        account = self.persistence.get_account()
        cash = account["cash"]
        
        if cash < total_need:
            return OrderResult(
                order_id=order_id,
                code=code,
                direction="buy",
                shares=shares,
                price=price,
                amount=amount,
                cost=cost,
                status="rejected",
                message=f"资金不足: 需要 {total_need:.2f}, 可用 {cash:.2f}"
            )
        
        # 更新现金
        new_cash = cash - total_need
        self.persistence.update_cash(new_cash)
        
        # 更新持仓
        position = self.persistence.get_position(code)
        if position:
            # 已有持仓，计算新的平均成本
            old_shares = position["shares"]
            old_cost = position["avg_cost"]
            new_shares = old_shares + shares
            new_avg_cost = (old_shares * old_cost + amount) / new_shares
            self.persistence.update_position(code, new_shares, new_avg_cost, price)
        else:
            # 新建持仓
            self.persistence.update_position(code, shares, price, price)
        
        # 记录订单
        order_record = {
            "order_id": order_id,
            "code": code,
            "direction": "buy",
            "shares": shares,
            "price": price,
            "amount": amount,
            "cost": cost,
            "status": "filled",
            "create_time": self._now_str(),
            "fill_time": self._now_str()
        }
        self.persistence.add_order(order_record)
        
        logger.info(
            f"买入成交: {code}, 股数={shares}, 价格={price:.2f}, "
            f"金额={amount:.2f}, 成本={cost:.2f}, 余额={new_cash:.2f}"
        )
        
        return OrderResult(
            order_id=order_id,
            code=code,
            direction="buy",
            shares=shares,
            price=price,
            amount=amount,
            cost=cost,
            status="filled",
            message="买入成功"
        )
    
    def _execute_sell(
        self,
        order_id: str,
        code: str,
        shares: int,
        price: float
    ) -> OrderResult:
        """执行卖出
        
        Args:
            order_id: 订单 ID
            code: 股票代码
            shares: 股数
            price: 价格
            
        Returns:
            订单结果
        """
        # 检查持仓
        position = self.persistence.get_position(code)
        if not position:
            return OrderResult(
                order_id=order_id,
                code=code,
                direction="sell",
                shares=shares,
                price=price,
                amount=0.0,
                cost=0.0,
                status="rejected",
                message=f"无持仓: {code}"
            )
        
        available_shares = position["shares"]
        if available_shares < shares:
            return OrderResult(
                order_id=order_id,
                code=code,
                direction="sell",
                shares=shares,
                price=price,
                amount=0.0,
                cost=0.0,
                status="rejected",
                message=f"持仓不足: 需要 {shares}, 可用 {available_shares}"
            )
        
        # 计算金额与成本
        amount = shares * price
        cost = self.cost_model.calculate_sell_cost(amount)
        total_receive = amount - cost
        
        # 更新现金
        account = self.persistence.get_account()
        cash = account["cash"]
        new_cash = cash + total_receive
        self.persistence.update_cash(new_cash)
        
        # 更新持仓
        new_shares = available_shares - shares
        if new_shares > 0:
            self.persistence.update_position(
                code,
                new_shares,
                position["avg_cost"],
                price
            )
        else:
            # 清空持仓
            self.persistence.update_position(code, 0, 0.0, price)
        
        # 记录订单
        order_record = {
            "order_id": order_id,
            "code": code,
            "direction": "sell",
            "shares": shares,
            "price": price,
            "amount": amount,
            "cost": cost,
            "status": "filled",
            "create_time": self._now_str(),
            "fill_time": self._now_str()
        }
        self.persistence.add_order(order_record)
        
        logger.info(
            f"卖出成交: {code}, 股数={shares}, 价格={price:.2f}, "
            f"金额={amount:.2f}, 成本={cost:.2f}, 余额={new_cash:.2f}"
        )
        
        return OrderResult(
            order_id=order_id,
            code=code,
            direction="sell",
            shares=shares,
            price=price,
            amount=amount,
            cost=cost,
            status="filled",
            message="卖出成功"
        )
    
    def get_account_info(self) -> Dict:
        """获取账户信息
        
        Returns:
            包含现金、持仓市值、总资产的字典
        """
        account = self.persistence.get_account()
        positions = self.persistence.get_positions()
        
        # 计算持仓市值
        position_value = 0.0
        for code, pos in positions.items():
            position_value += pos["shares"] * pos["last_price"]
        
        total_value = account["cash"] + position_value
        
        return {
            "cash": account["cash"],
            "position_value": position_value,
            "total_value": total_value,
            "initial_cash": account["initial_cash"],
            "positions": positions
        }
