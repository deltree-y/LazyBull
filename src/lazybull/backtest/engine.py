"""回测引擎"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from ..common.cost import CostModel
from ..signals.base import Signal
from ..universe.base import Universe


class BacktestEngine:
    """回测引擎
    
    执行回测流程，生成净值曲线和交易记录
    
    交易规则：
    - T 日生成信号
    - T+1 日收盘价买入
    - T+n 日收盘价卖出（n 为持有期，默认与调仓频率一致）
    """
    
    def __init__(
        self,
        universe: Universe,
        signal: Signal,
        initial_capital: float = 1000000.0,
        cost_model: Optional[CostModel] = None,
        rebalance_freq: str = "M",
        holding_period: Optional[int] = None
    ):
        """初始化回测引擎
        
        Args:
            universe: 股票池
            signal: 信号生成器
            initial_capital: 初始资金
            cost_model: 成本模型
            rebalance_freq: 调仓频率，D=日，W=周，M=月
            holding_period: 持有期（交易日），None 则自动根据调仓频率设置
        """
        self.universe = universe
        self.signal = signal
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.rebalance_freq = rebalance_freq
        
        # 设置持有期
        if holding_period is None:
            # 根据调仓频率自动设置持有期
            if rebalance_freq == "D":
                self.holding_period = 1
            elif rebalance_freq == "W":
                self.holding_period = 5
            else:  # M
                self.holding_period = 20
        else:
            self.holding_period = holding_period
        
        # 回测状态
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}  # {股票代码: {shares: 持仓数量, buy_date: 买入日期}}
        self.pending_signals: Dict[pd.Timestamp, Dict] = {}  # {信号日期: {股票: 权重}}
        self.portfolio_values: List[Dict] = []  # 组合价值历史
        self.trades: List[Dict] = []  # 交易记录
        
        logger.info(
            f"回测引擎初始化完成: 初始资金={initial_capital}, "
            f"调仓频率={rebalance_freq}, 持有期={self.holding_period}天"
        )
        logger.info(f"交易规则: T日生成信号 -> T+1日收盘价买入 -> T+{self.holding_period}日收盘价卖出")

    
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
        import time
        
        logger.info(f"开始回测: {start_date.date()} 至 {end_date.date()}")
        
        # 筛选回测期间的交易日
        trading_dates = [d for d in trading_dates if start_date <= d <= end_date]
        total_days = len(trading_dates)
        
        # 创建日期到索引的映射，优化查找效率
        date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
        
        # 准备价格数据字典，加速查询
        price_dict = self._prepare_price_dict(price_data)
        
        # 获取调仓日期（信号生成日期）
        signal_dates = self._get_rebalance_dates(trading_dates)
        
        # 记录开始时间
        start_time = time.time()
        
        # 使用 tqdm 显示进度条
        with tqdm(total=total_days, desc="回测进度", unit="天") as pbar:
            # 按日推进
            for idx, date in enumerate(trading_dates):
                # 判断是否为信号生成日
                if date in signal_dates:
                    self._generate_signal(date, trading_dates, price_dict, date_to_idx)
                
                # 执行待执行的买入操作（T+1）
                self._execute_pending_buys(date, trading_dates, price_dict, date_to_idx)
                
                # 检查并执行卖出操作（达到持有期）
                self._check_and_sell(date, trading_dates, price_dict, date_to_idx)
                
                # 计算当日组合价值
                portfolio_value = self._calculate_portfolio_value(date, price_dict)
                
                self.portfolio_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'capital': self.current_capital,
                    'market_value': portfolio_value - self.current_capital
                })
                
                # 更新进度条
                elapsed_time = time.time() - start_time
                pbar.set_postfix({
                    '当前日期': date.strftime('%Y-%m-%d'),
                    '净值': f"{portfolio_value/self.initial_capital:.4f}",
                    '已用时': f"{elapsed_time:.1f}秒"
                })
                pbar.update(1)
        
        # 生成净值曲线
        nav_df = self._generate_nav_curve()
        
        total_time = time.time() - start_time
        logger.info(f"回测完成: 共 {len(trading_dates)} 个交易日, {len(self.trades)} 笔交易, 总耗时 {total_time:.1f}秒")
        
        return nav_df
    
    def _generate_signal(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], price_dict: Dict, date_to_idx: Dict) -> None:
        """生成信号（在 T 日生成，T+1 日执行买入）
        
        Args:
            date: 信号生成日期
            trading_dates: 交易日列表
            price_dict: 价格字典
            date_to_idx: 日期到索引的映射
        """
        # 获取股票池
        stock_universe = self.universe.get_stocks(date)
        
        # 生成信号
        signals = self.signal.generate(date, stock_universe, {})
        
        if not signals:
            logger.warning(f"信号日 {date.date()} 无信号")
            return
        
        # 保存信号，待 T+1 执行
        self.pending_signals[date] = signals
        logger.info(f"信号生成: {date.date()}, 信号数 {len(signals)}")
    
    def _execute_pending_buys(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], price_dict: Dict, date_to_idx: Dict) -> None:
        """执行待执行的买入操作（T+1）
        
        Args:
            date: 当前日期
            trading_dates: 交易日列表
            price_dict: 价格字典
            date_to_idx: 日期到索引的映射
        """
        # 查找前一个交易日的信号
        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx == 0:
            return
        
        signal_date = trading_dates[current_idx - 1]
        
        if signal_date not in self.pending_signals:
            return
        
        signals = self.pending_signals.pop(signal_date)
        
        # 计算当前组合市值
        current_value = self._calculate_portfolio_value(date, price_dict)
        
        # 买入信号中的股票
        for stock, weight in signals.items():
            target_value = current_value * weight
            self._buy_stock(date, stock, target_value, price_dict)
        
        logger.info(f"买入执行: {date.date()}, 买入 {len(signals)} 只股票（信号日: {signal_date.date()}）")
    
    def _check_and_sell(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], price_dict: Dict, date_to_idx: Dict) -> None:
        """检查并执行卖出操作（达到持有期 T+n）
        
        Args:
            date: 当前日期
            trading_dates: 交易日列表
            price_dict: 价格字典
            date_to_idx: 日期到索引的映射
        """
        stocks_to_sell = []
        
        current_idx = date_to_idx.get(date)
        if current_idx is None:
            return
        
        for stock, info in self.positions.items():
            buy_date = info['buy_date']
            buy_idx = date_to_idx.get(buy_date)
            
            if buy_idx is None:
                logger.warning(f"股票 {stock} 买入日期 {buy_date} 不在交易日映射中")
                continue
            
            # 计算持有天数（交易日）
            holding_days = current_idx - buy_idx
            
            # 达到持有期，执行卖出
            if holding_days >= self.holding_period:
                stocks_to_sell.append(stock)
        
        # 执行卖出
        for stock in stocks_to_sell:
            self._sell_stock(date, stock, price_dict)
        
        if stocks_to_sell:
            logger.info(f"卖出执行: {date.date()}, 卖出 {len(stocks_to_sell)} 只股票（达到持有期）")
    
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
    
    
    def _buy_stock(self, date: pd.Timestamp, stock: str, target_value: float, price_dict: Dict) -> None:
        """买入股票（在 T+1 日以收盘价买入）
        
        Args:
            date: 买入日期（T+1）
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
        
        # 更新持仓和资金（记录买入日期）
        # 注意：在当前 T+n 卖出策略下，理论上不应该出现已有持仓的情况
        # 因为旧持仓应该在达到持有期后自动卖出
        # 如果出现已有持仓，说明可能是边界情况或配置问题
        if stock in self.positions:
            logger.warning(
                f"股票 {stock} 已有持仓（买入日期: {self.positions[stock]['buy_date']}），"
                f"新买入将覆盖旧持仓（可能配置有误）"
            )
        
        # 设置或覆盖持仓（记录买入价格和成本，用于计算收益）
        self.positions[stock] = {
            'shares': shares,
            'buy_date': date,
            'buy_price': price,
            'buy_cost': total_cost
        }
        
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
        """卖出股票（在 T+n 日以收盘价卖出）
        
        Args:
            date: 卖出日期（T+n）
            stock: 股票代码
            price_dict: 价格字典
        """
        if stock not in self.positions or self.positions[stock]['shares'] == 0:
            return
        
        if date not in price_dict or stock not in price_dict[date]:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的价格，跳过卖出")
            return
        
        price = price_dict[date][stock]
        shares = self.positions[stock]['shares']
        amount = shares * price
        cost = self.cost_model.calculate_sell_cost(amount)
        
        # 计算收益（基于 FIFO 原则：使用记录的买入成本）
        # 收益 = 卖出所得（扣除成本）- 买入成本
        buy_cost = self.positions[stock]['buy_cost']
        buy_price = self.positions[stock]['buy_price']
        sell_proceeds = amount - cost  # 卖出后实际到手金额
        profit_amount = sell_proceeds - buy_cost  # 绝对收益（已扣除买卖成本）
        profit_pct = profit_amount / buy_cost if buy_cost > 0 else 0  # 收益率
        
        # 更新持仓和资金
        del self.positions[stock]
        self.current_capital += sell_proceeds
        
        # 记录交易（包含收益信息）
        self.trades.append({
            'date': date,
            'stock': stock,
            'action': 'sell',
            'price': price,
            'shares': shares,
            'amount': amount,
            'cost': cost,
            'buy_price': buy_price,  # 买入价格
            'profit_amount': profit_amount,  # 单笔收益金额（已扣除成本）
            'profit_pct': profit_pct  # 单笔收益率
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
            for stock, info in self.positions.items():
                shares = info['shares']
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
