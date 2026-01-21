"""纸面交易数据模型"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class TargetWeight:
    """目标权重"""
    ts_code: str  # 股票代码
    target_weight: float  # 目标权重
    reason: str = "信号生成"  # 买卖原因


@dataclass
class Position:
    """持仓信息"""
    ts_code: str  # 股票代码
    shares: int  # 持仓股数
    buy_price: float  # 买入价格
    buy_cost: float  # 买入成本（含手续费）
    buy_date: str  # 买入日期 YYYYMMDD
    status: str = "持有"  # 持仓状态：持有、延迟卖出等
    notes: str = ""  # 备注信息
    
    def get_holding_days(self, current_date: str) -> int:
        """计算持有天数（自然日）
        
        Args:
            current_date: 当前日期 YYYYMMDD
            
        Returns:
            持有天数
        """
        try:
            import pandas as pd
            buy_dt = pd.to_datetime(self.buy_date, format='%Y%m%d')
            current_dt = pd.to_datetime(current_date, format='%Y%m%d')
            return (current_dt - buy_dt).days
        except Exception:
            return 0


@dataclass
class Order:
    """订单"""
    ts_code: str  # 股票代码
    action: str  # buy/sell
    shares: int  # 交易股数
    price: float  # 参考价格
    target_weight: float  # 目标权重
    current_weight: float  # 当前权重
    reason: str = "目标调仓"  # 交易原因


@dataclass
class Fill:
    """成交记录"""
    trade_date: str  # 交易日期 YYYYMMDD
    ts_code: str  # 股票代码
    action: str  # buy/sell
    shares: int  # 交易股数
    price: float  # 成交价格
    amount: float  # 成交金额
    commission: float  # 佣金
    stamp_tax: float  # 印花税
    slippage: float  # 滑点成本
    total_cost: float  # 总成本
    reason: str = "目标调仓"  # 交易原因


@dataclass
class AccountState:
    """账户状态"""
    cash: float  # 现金
    positions: dict = field(default_factory=dict)  # {ts_code: Position}
    last_update: str = ""  # 最后更新日期 YYYYMMDD
    
    def get_position_value(self, prices: dict) -> float:
        """计算持仓市值
        
        Args:
            prices: {ts_code: price} 价格字典
            
        Returns:
            持仓市值
        """
        total_value = 0.0
        for ts_code, pos in self.positions.items():
            if ts_code in prices:
                total_value += pos.shares * prices[ts_code]
        return total_value
    
    def get_total_value(self, prices: dict) -> float:
        """计算总资产
        
        Args:
            prices: {ts_code: price} 价格字典
            
        Returns:
            总资产
        """
        return self.cash + self.get_position_value(prices)
    
    def get_position_weight(self, ts_code: str, prices: dict) -> float:
        """计算持仓权重
        
        Args:
            ts_code: 股票代码
            prices: {ts_code: price} 价格字典
            
        Returns:
            持仓权重（0.0-1.0）
        """
        if ts_code not in self.positions:
            return 0.0
        
        total_value = self.get_total_value(prices)
        if total_value <= 0:
            return 0.0
        
        pos = self.positions[ts_code]
        if ts_code not in prices:
            return 0.0
        
        pos_value = pos.shares * prices[ts_code]
        return pos_value / total_value


@dataclass
class NAVRecord:
    """净值记录"""
    trade_date: str  # 交易日期 YYYYMMDD
    cash: float  # 现金
    position_value: float  # 持仓市值
    total_value: float  # 总资产
    nav: float  # 净值
