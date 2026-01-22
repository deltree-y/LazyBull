"""纸面交易账户模块"""

from typing import Dict, Optional

import pandas as pd
from loguru import logger

from .models import AccountState, Position
from .storage import PaperStorage


class PaperAccount:
    """纸面交易账户
    
    管理现金、持仓和账户状态
    """
    
    def __init__(
        self,
        initial_capital: float = 500000.0,
        storage: Optional[PaperStorage] = None,
        verbose: bool = False,
    ):
        """初始化账户
        
        Args:
            initial_capital: 初始资金
            storage: 存储实例
            verbose: 是否输出详细日志
        """
        self.initial_capital = initial_capital
        self.storage = storage or PaperStorage(verbose=verbose)
        
        # 尝试从存储加载账户状态
        self.state = self.storage.load_account_state()
        
        if self.state is None:
            # 首次运行，创建新账户
            self.state = AccountState(
                cash=initial_capital,
                positions={},
                last_update="",
            )
            if verbose:
                logger.info(f"创建新纸面账户，初始资金: {initial_capital:,.2f}")
        else:
            if verbose:
                logger.info(f"加载已有账户状态，现金: {self.state.cash:,.2f}，持仓数: {len(self.state.positions)}")
    
    def get_cash(self) -> float:
        """获取现金"""
        return self.state.cash
    
    def get_positions(self) -> Dict[str, Position]:
        """获取持仓"""
        return self.state.positions
    
    def get_position(self, ts_code: str) -> Optional[Position]:
        """获取单个持仓
        
        Args:
            ts_code: 股票代码
            
        Returns:
            持仓，不存在返回None
        """
        return self.state.positions.get(ts_code)
    
    def get_position_value(self, prices: Dict[str, float]) -> float:
        """计算持仓市值
        
        Args:
            prices: {ts_code: price} 价格字典
            
        Returns:
            持仓市值
        """
        return self.state.get_position_value(prices)
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """计算总资产
        
        Args:
            prices: {ts_code: price} 价格字典
            
        Returns:
            总资产
        """
        return self.state.get_total_value(prices)
    
    def get_position_weight(self, ts_code: str, prices: Dict[str, float]) -> float:
        """计算持仓权重
        
        Args:
            ts_code: 股票代码
            prices: {ts_code: price} 价格字典
            
        Returns:
            持仓权重（0.0-1.0）
        """
        return self.state.get_position_weight(ts_code, prices)
    
    def update_cash(self, amount: float) -> None:
        """更新现金
        
        Args:
            amount: 增减金额（正数增加，负数减少）
        """
        self.state.cash += amount
        logger.debug(f"现金变动: {amount:+.2f}，当前现金: {self.state.cash:,.2f}")
    
    def add_position(
        self,
        ts_code: str,
        shares: int,
        buy_price: float,
        buy_cost: float,
        buy_date: str,
        status: str = "持有",
        notes: str = ""
    ) -> None:
        """增加持仓
        
        Args:
            ts_code: 股票代码
            shares: 股数
            buy_price: 买入价格
            buy_cost: 买入成本
            buy_date: 买入日期 YYYYMMDD
            status: 持仓状态
            notes: 备注信息
        """
        if ts_code in self.state.positions:
            # 已有持仓，累加
            pos = self.state.positions[ts_code]
            total_shares = pos.shares + shares
            total_cost = pos.buy_cost + buy_cost
            avg_price = (pos.buy_price * pos.shares + buy_price * shares) / total_shares
            
            self.state.positions[ts_code] = Position(
                ts_code=ts_code,
                shares=total_shares,
                buy_price=avg_price,
                buy_cost=total_cost,
                buy_date=buy_date,  # 更新为最新买入日期
                status=status,
                notes=notes
            )
            logger.debug(f"累加持仓 {ts_code}: {shares} 股，总持仓: {total_shares} 股")
        else:
            # 新建持仓
            self.state.positions[ts_code] = Position(
                ts_code=ts_code,
                shares=shares,
                buy_price=buy_price,
                buy_cost=buy_cost,
                buy_date=buy_date,
                status=status,
                notes=notes
            )
            logger.debug(f"新建持仓 {ts_code}: {shares} 股")
    
    def reduce_position(self, ts_code: str, shares: int) -> Optional[Position]:
        """减少持仓
        
        Args:
            ts_code: 股票代码
            shares: 股数
            
        Returns:
            减少后的持仓，如果全部卖出则返回None
        """
        if ts_code not in self.state.positions:
            logger.warning(f"尝试减少不存在的持仓: {ts_code}")
            return None
        
        pos = self.state.positions[ts_code]
        
        if shares >= pos.shares:
            # 全部卖出
            del self.state.positions[ts_code]
            logger.debug(f"清空持仓 {ts_code}")
            return None
        else:
            # 部分卖出
            remaining_shares = pos.shares - shares
            remaining_cost = pos.buy_cost * (remaining_shares / pos.shares)
            
            self.state.positions[ts_code] = Position(
                ts_code=ts_code,
                shares=remaining_shares,
                buy_price=pos.buy_price,
                buy_cost=remaining_cost,
                buy_date=pos.buy_date,
                status=pos.status,
                notes=pos.notes
            )
            logger.debug(f"减少持仓 {ts_code}: {shares} 股，剩余: {remaining_shares} 股")
            return self.state.positions[ts_code]
    
    def update_last_date(self, trade_date: str) -> None:
        """更新最后更新日期
        
        Args:
            trade_date: 交易日期 YYYYMMDD
        """
        self.state.last_update = trade_date
    
    def save_state(self) -> None:
        """保存账户状态"""
        self.storage.save_account_state(self.state)
        logger.info("账户状态已保存")
