"""信号生成基类"""

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
from loguru import logger


class Signal(ABC):
    """信号基类
    
    定义信号生成的标准接口
    """
    
    def __init__(self, name: str = "base"):
        """初始化信号
        
        Args:
            name: 信号名称
        """
        self.name = name
    
    @abstractmethod
    def generate(self, date: pd.Timestamp, universe: List[str], data: Dict) -> Dict[str, float]:
        """生成信号
        
        Args:
            date: 当前日期
            universe: 股票池
            data: 数据字典，包含价格、因子等
            
        Returns:
            信号字典，{股票代码: 权重}
        """
        pass


class EqualWeightSignal(Signal):
    """等权信号
    
    对股票池中的前N只股票等权配置
    """
    
    def __init__(self, top_n: int = 30):
        """初始化等权信号
        
        Args:
            top_n: 持仓股票数量
        """
        super().__init__("equal_weight")
        self.top_n = top_n
    
    def generate(self, date: pd.Timestamp, universe: List[str], data: Dict) -> Dict[str, float]:
        """生成等权信号
        
        Args:
            date: 当前日期
            universe: 股票池
            data: 数据字典（此信号不使用）
            
        Returns:
            信号字典，前N只股票等权
        """
        # 简单选择前N只（实际应用中应该有排序逻辑）
        selected = universe[:self.top_n]
        
        if not selected:
            return {}
        
        # 等权分配
        weight = 1.0 / len(selected)
        signals = {stock: weight for stock in selected}
        
        logger.debug(f"信号 {self.name} 在 {date.date()} 生成 {len(signals)} 只股票")
        
        return signals


class FactorSignal(Signal):
    """因子信号
    
    基于因子打分生成信号（占位实现）
    """
    
    def __init__(self, top_n: int = 30, weight_method: str = "equal"):
        """初始化因子信号
        
        Args:
            top_n: 持仓股票数量
            weight_method: 权重方法，equal=等权，score=按得分加权
        """
        super().__init__("factor")
        self.top_n = top_n
        self.weight_method = weight_method
    
    def generate(self, date: pd.Timestamp, universe: List[str], data: Dict) -> Dict[str, float]:
        """生成因子信号
        
        Args:
            date: 当前日期
            universe: 股票池
            data: 数据字典，应包含因子数据
            
        Returns:
            信号字典
        """
        # TODO: 实现真实的因子打分逻辑
        # 这里简化为随机选择前N只
        selected = universe[:self.top_n]
        
        if not selected:
            return {}
        
        # 根据权重方法分配
        if self.weight_method == "equal":
            weight = 1.0 / len(selected)
            signals = {stock: weight for stock in selected}
        else:
            # TODO: 实现基于得分的加权
            weight = 1.0 / len(selected)
            signals = {stock: weight for stock in selected}
        
        logger.debug(f"信号 {self.name} 在 {date.date()} 生成 {len(signals)} 只股票")
        
        return signals
