"""回测引擎"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ..common.cost import CostModel
from ..signals.base import Signal
from ..universe.base import Universe


class BacktestEngine:
    """回测引擎
    
    执行回测流程，生成净值曲线和交易记录
    """
    
    def __init__(
        self,
        universe: Universe,
        signal: Signal,
        initial_capital: float = 1000000.0,
        cost_model: Optional[CostModel] = None,
        rebalance_freq: str = "M"
    ):
        """初始化回测引擎
        
        Args:
            universe: 股票池
            signal: 信号生成器
            initial_capital: 初始资金
            cost_model: 成本模型
            rebalance_freq: 调仓频率，D=日，W=周，M=月
        """
        self.universe = universe
        self.signal = signal
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.rebalance_freq = rebalance_freq
        
        # 回测状态
        self.current_capital = initial_capital
        self.positions: Dict[str, float] = {}  # {股票代码: 持仓数量}
        self.portfolio_values: List[Dict] = []  # 组合价值历史
        self.trades: List[Dict] = []  # 交易记录
        
        logger.info(
            f"回测引擎初始化完成: 初始资金={initial_capital}, "
            f"调仓频率={rebalance_freq}"
        )
    
    def run(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            trading_dates: 交易日列表
            price_data: 价格数据，包含ts_code, trade_date, close
            
        Returns:
            净值曲线DataFrame
        """
        logger.info(f"开始回测: {start_date.date()} 至 {end_date.date()}")
        
        # 筛选回测期间的交易日
        trading_dates = [d for d in trading_dates if start_date <= d <= end_date]
        
        # 准备价格数据字典，加速查询
        price_dict = self._prepare_price_dict(price_data)
        
        # 获取调仓日期
        rebalance_dates = self._get_rebalance_dates(trading_dates)
        
        # 按日推进
        for date in trading_dates:
            # 判断是否调仓日
            if date in rebalance_dates:
                self._rebalance(date, price_dict)
            
            # 计算当日组合价值
            portfolio_value = self._calculate_portfolio_value(date, price_dict)
            
            self.portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'capital': self.current_capital,
                'market_value': portfolio_value - self.current_capital
            })
        
        # 生成净值曲线
        nav_df = self._generate_nav_curve()
        
        logger.info(f"回测完成: 共 {len(trading_dates)} 个交易日, {len(self.trades)} 笔交易")
        
        return nav_df
    
    def _prepare_price_dict(self, price_data: pd.DataFrame) -> Dict:
        """准备价格字典
        
        Args:
            price_data: 价格数据
            
        Returns:
            {trade_date: {ts_code: close_price}}
        """
        price_dict = {}
        for _, row in price_data.iterrows():
            date = pd.to_datetime(row['trade_date'])
            if date not in price_dict:
                price_dict[date] = {}
            price_dict[date][row['ts_code']] = row['close']
        return price_dict
    
    def _get_rebalance_dates(self, trading_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
        """获取调仓日期
        
        Args:
            trading_dates: 交易日列表
            
        Returns:
            调仓日期列表
        """
        if self.rebalance_freq == "D":
            return trading_dates
        elif self.rebalance_freq == "W":
            # 每周最后一个交易日
            df = pd.DataFrame({'date': trading_dates})
            df['week'] = df['date'].dt.isocalendar().week
            df['year'] = df['date'].dt.year
            return df.groupby(['year', 'week'])['date'].last().tolist()
        elif self.rebalance_freq == "M":
            # 每月最后一个交易日
            df = pd.DataFrame({'date': trading_dates})
            df['month'] = df['date'].dt.to_period('M')
            return df.groupby('month')['date'].last().tolist()
        else:
            return [trading_dates[0]]  # 只在第一天调仓
    
    def _rebalance(self, date: pd.Timestamp, price_dict: Dict) -> None:
        """执行调仓
        
        Args:
            date: 调仓日期
            price_dict: 价格字典
        """
        # 获取股票池
        stock_universe = self.universe.get_stocks(date)
        
        # 生成信号
        signals = self.signal.generate(date, stock_universe, {})
        
        if not signals:
            logger.warning(f"调仓日 {date.date()} 无信号，跳过")
            return
        
        # 计算当前组合市值
        current_value = self._calculate_portfolio_value(date, price_dict)
        
        # 卖出不在信号中的持仓
        for stock in list(self.positions.keys()):
            if stock not in signals:
                self._sell_stock(date, stock, price_dict)
        
        # 买入信号中的股票
        for stock, weight in signals.items():
            target_value = current_value * weight
            self._buy_stock(date, stock, target_value, price_dict)
        
        logger.info(f"调仓完成: {date.date()}, 持仓 {len(self.positions)} 只股票")
    
    def _buy_stock(self, date: pd.Timestamp, stock: str, target_value: float, price_dict: Dict) -> None:
        """买入股票
        
        Args:
            date: 交易日期
            stock: 股票代码
            target_value: 目标市值
            price_dict: 价格字典
        """
        if date not in price_dict or stock not in price_dict[date]:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的价格，跳过买入")
            return
        
        price = price_dict[date][stock]
        shares = int(target_value / price / 100) * 100  # 按手买入
        
        if shares == 0:
            return
        
        amount = shares * price
        cost = self.cost_model.calculate_buy_cost(amount)
        total_cost = amount + cost
        
        if total_cost > self.current_capital:
            # 资金不足，按可用资金买入
            shares = int((self.current_capital - cost) / price / 100) * 100
            if shares == 0:
                return
            amount = shares * price
            cost = self.cost_model.calculate_buy_cost(amount)
            total_cost = amount + cost
        
        # 更新持仓和资金
        self.positions[stock] = self.positions.get(stock, 0) + shares
        self.current_capital -= total_cost
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock': stock,
            'action': 'buy',
            'price': price,
            'shares': shares,
            'amount': amount,
            'cost': cost
        })
    
    def _sell_stock(self, date: pd.Timestamp, stock: str, price_dict: Dict) -> None:
        """卖出股票
        
        Args:
            date: 交易日期
            stock: 股票代码
            price_dict: 价格字典
        """
        if stock not in self.positions or self.positions[stock] == 0:
            return
        
        if date not in price_dict or stock not in price_dict[date]:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的价格，跳过卖出")
            return
        
        price = price_dict[date][stock]
        shares = self.positions[stock]
        amount = shares * price
        cost = self.cost_model.calculate_sell_cost(amount)
        
        # 更新持仓和资金
        del self.positions[stock]
        self.current_capital += (amount - cost)
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock': stock,
            'action': 'sell',
            'price': price,
            'shares': shares,
            'amount': amount,
            'cost': cost
        })
    
    def _calculate_portfolio_value(self, date: pd.Timestamp, price_dict: Dict) -> float:
        """计算组合市值
        
        Args:
            date: 计算日期
            price_dict: 价格字典
            
        Returns:
            组合总市值
        """
        market_value = 0.0
        
        if date in price_dict:
            for stock, shares in self.positions.items():
                if stock in price_dict[date]:
                    market_value += shares * price_dict[date][stock]
        
        return self.current_capital + market_value
    
    def _generate_nav_curve(self) -> pd.DataFrame:
        """生成净值曲线
        
        Returns:
            净值曲线DataFrame
        """
        df = pd.DataFrame(self.portfolio_values)
        df['nav'] = df['portfolio_value'] / self.initial_capital
        df['return'] = df['nav'] - 1.0
        return df
    
    def get_trades(self) -> pd.DataFrame:
        """获取交易记录
        
        Returns:
            交易记录DataFrame
        """
        return pd.DataFrame(self.trades)
