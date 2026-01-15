"""交易成本模型"""

from typing import Optional


class CostModel:
    """交易成本模型
    
    包含佣金、印花税、滑点等成本计算
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_tax: float = 0.001,
        slippage: float = 0.001
    ):
        """初始化成本模型
        
        Args:
            commission_rate: 佣金费率（买卖双向）
            min_commission: 最小佣金（元）
            stamp_tax: 印花税（仅卖出）
            slippage: 滑点比率
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
    
    def calculate_commission(self, amount: float) -> float:
        """计算佣金
        
        Args:
            amount: 交易金额
            
        Returns:
            佣金金额
        """
        commission = amount * self.commission_rate
        return max(commission, self.min_commission)
    
    def calculate_stamp_tax(self, amount: float) -> float:
        """计算印花税（仅卖出）
        
        Args:
            amount: 交易金额
            
        Returns:
            印花税金额
        """
        return amount * self.stamp_tax
    
    def calculate_slippage(self, amount: float) -> float:
        """计算滑点成本
        
        Args:
            amount: 交易金额
            
        Returns:
            滑点成本
        """
        return amount * self.slippage
    
    def calculate_buy_cost(self, amount: float) -> float:
        """计算买入总成本
        
        Args:
            amount: 买入金额
            
        Returns:
            总成本（佣金 + 滑点）
        """
        commission = self.calculate_commission(amount)
        slippage = self.calculate_slippage(amount)
        return commission + slippage
    
    def calculate_sell_cost(self, amount: float) -> float:
        """计算卖出总成本
        
        Args:
            amount: 卖出金额
            
        Returns:
            总成本（佣金 + 印花税 + 滑点）
        """
        commission = self.calculate_commission(amount)
        stamp_tax = self.calculate_stamp_tax(amount)
        slippage = self.calculate_slippage(amount)
        return commission + stamp_tax + slippage
    
    def calculate_total_cost(self, buy_amount: float, sell_amount: float) -> float:
        """计算买卖双向总成本
        
        Args:
            buy_amount: 买入金额
            sell_amount: 卖出金额
            
        Returns:
            总成本
        """
        return self.calculate_buy_cost(buy_amount) + self.calculate_sell_cost(sell_amount)


def get_default_cost_model() -> CostModel:
    """获取默认成本模型"""
    return CostModel(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax=0.001,
        slippage=0.001
    )
