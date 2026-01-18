"""回测引擎"""

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ..common.cost import CostModel
from ..common.trade_status import is_tradeable
from ..execution.pending_order import PendingOrderManager
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
    
    # 常量：每年交易日数量（用于年化波动率计算）
    TRADING_DAYS_PER_YEAR = 252
    
    def __init__(
        self,
        universe: Universe,
        signal: Signal,
        initial_capital: float = 1000000.0,
        cost_model: Optional[CostModel] = None,
        rebalance_freq: str = "M",
        holding_period: Optional[int] = None,
        verbose: bool = True,
        price_type: str = "close",  # 保留以兼容旧代码，但不再使用
        enable_risk_budget: bool = False,
        vol_window: int = 20,
        vol_epsilon: float = 0.001,
        enable_pending_order: bool = True,
        max_retry_count: int = 5,
        max_retry_days: int = 10
    ):
        """初始化回测引擎
        
        价格口径说明：
        - 成交价格（trade_price）：使用不复权 close，用于计算成交金额、持仓市值、可买入数量
        - 绩效价格（pnl_price）：使用后复权 close_adj，用于计算收益率和绩效指标
        
        Args:
            universe: 股票池
            signal: 信号生成器
            initial_capital: 初始资金
            cost_model: 成本模型
            rebalance_freq: 调仓频率，D=日，W=周，M=月，或整数表示每N天调仓
            holding_period: 持有期（交易日），None 则自动根据调仓频率设置
            verbose: 是否输出详细日志（买入/卖出操作），默认True
            price_type: （已废弃，保留以兼容旧代码）价格类型，新版本中不再使用
            enable_risk_budget: 是否启用风险预算/波动率缩放，默认False（保持向后兼容）
            vol_window: 波动率计算窗口（交易日），默认20
            vol_epsilon: 波动率缩放的最小波动率，防止除零，默认0.001
            enable_pending_order: 是否启用延迟订单功能，默认True
            max_retry_count: 延迟订单最大重试次数，默认5次
            max_retry_days: 延迟订单最大延迟天数，默认10天
        """
        self.universe = universe
        self.signal = signal
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.rebalance_freq = rebalance_freq
        self.verbose = verbose
        self.price_type = price_type  # 保留以兼容旧代码
        
        # 风险预算参数
        self.enable_risk_budget = enable_risk_budget
        self.vol_window = vol_window
        self.vol_epsilon = vol_epsilon
        
        # 延迟订单参数
        self.enable_pending_order = enable_pending_order
        self.pending_order_manager = None
        if enable_pending_order:
            self.pending_order_manager = PendingOrderManager(
                max_retry_count=max_retry_count,
                max_retry_days=max_retry_days
            )
        
        # 设置持有期及调仓频率(目前二者保持一致)
        self.rebalance_freq = rebalance_freq
        if holding_period is None:
            self.holding_period = self.rebalance_freq
        else:
            self.holding_period = self.rebalance_freq
        
        
        # 回测状态
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}  # {股票代码: {shares, buy_date, buy_trade_price, buy_pnl_price, buy_cost_cash}}
        self.pending_signals: Dict[pd.Timestamp, Dict] = {}  # {信号日期: {股票: 权重}}
        self.portfolio_values: List[Dict] = []  # 组合价值历史
        self.trades: List[Dict] = []  # 交易记录
        
        # 价格索引（在 run 时初始化）
        self.trade_price_index: Optional[pd.Series] = None  # 成交价格（不复权 close）
        self.pnl_price_index: Optional[pd.Series] = None  # 绩效价格（后复权 close_adj）
        
        # 存储价格数据用于交易状态检查
        self.price_data_cache: Optional[pd.DataFrame] = None
        
        logger.info(
            f"回测引擎初始化完成: 初始资金={initial_capital}, "
            f"调仓频率={self.rebalance_freq}, 持有期={self.holding_period}天, "
            f"风险预算={'启用' if enable_risk_budget else '禁用'}, "
            f"延迟订单={'启用' if enable_pending_order else '禁用'}, "
            f"详细日志={'开启' if verbose else '关闭'}"
        )
        logger.info(f"交易规则: T日生成信号 -> T+1日收盘价买入 -> T+{self.holding_period}日收盘价卖出")
        logger.info(f"价格口径: 成交使用不复权 close, 绩效使用后复权 close_adj")

    
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
            price_data: 价格数据，需包含 ts_code, trade_date, close, close_adj（可选）
            
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
        
        # 准备价格索引（使用 MultiIndex，替代嵌套字典）
        self._prepare_price_index(price_data)
        
        # 缓存价格数据用于交易状态检查
        self.price_data_cache = price_data
        
        # 获取调仓日期（信号生成日期）
        signal_dates = self._get_rebalance_dates(trading_dates)

        logger.info(f"数据准备完成, 调仓日期共 {len(signal_dates)} 天")
        
        # 记录开始时间
        start_time = time.time()
        
        # 按日推进
        for idx, date in enumerate(trading_dates):
            # 处理延迟订单（先处理延迟订单，再处理新信号）
            if self.enable_pending_order:
                self._process_pending_orders(date)
            
            # 判断是否为信号生成日
            if date in signal_dates:
                self._generate_signal(date, trading_dates, price_data, date_to_idx)

            # @2026/01/18: 改为先卖出再买入, 避免当天买入的股票被误判为达到持有期而卖出
            # TODO: 更正确的做法应该是在持有期计算中排除当天买入的股票, 此部分还待优化
            # 检查并执行卖出操作（达到持有期）
            self._check_and_sell(date, trading_dates, date_to_idx)

            # 执行待执行的买入操作（T+1）
            self._execute_pending_buys(date, trading_dates, date_to_idx)
            
            # 计算当日组合价值
            portfolio_value = self._calculate_portfolio_value(date)
            
            self.portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'capital': self.current_capital,
                'market_value': portfolio_value - self.current_capital
            })
            
        # 生成净值曲线
        nav_df = self._generate_nav_curve()
        
        total_time = time.time() - start_time
        logger.info(f"回测完成: 共 {len(trading_dates)} 个交易日, {len(self.trades)} 笔交易, 总耗时 {total_time:.1f}秒")
        
        # 输出延迟订单统计
        if self.enable_pending_order and self.pending_order_manager:
            stats = self.pending_order_manager.get_statistics()
            logger.info(
                f"延迟订单统计: 累计添加 {stats['total_added']}, "
                f"成功执行 {stats['total_succeeded']}, "
                f"过期放弃 {stats['total_expired']}, "
                f"剩余待处理 {stats['pending']}"
            )
        
        return nav_df
    
    def _generate_signal(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], price_data: pd.DataFrame, date_to_idx: Dict) -> None:
        """生成信号（在 T 日生成，T+1 日执行买入）
        
        Args:
            date: 信号生成日期
            trading_dates: 交易日列表
            price_data: 价格数据，包含行情信息
            date_to_idx: 日期到索引的映射
        """
        # 获取当日行情数据用于过滤
        date_quote = price_data[price_data['trade_date'] == date]
        
        # 获取股票池（传入行情数据进行过滤）
        stock_universe = self.universe.get_stocks(date, quote_data=date_quote)
        
        # 生成信号
        signals = self.signal.generate(date, stock_universe, {})
        
        if not signals:
            if self.verbose:
                logger.warning(f"信号日 {date.date()} 无信号")
            return
        
        # 保存信号，待 T+1 执行
        self.pending_signals[date] = signals
        if self.verbose:
            logger.info(f"信号生成: {date.date()}, 信号数 {len(signals)}")
    
    def _execute_pending_buys(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict) -> None:
        """执行待执行的买入操作（T+1）
        
        Args:
            date: 当前日期
            trading_dates: 交易日列表
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
        
        # 应用风险预算（波动率缩放）
        if self.enable_risk_budget:
            signals = self._apply_risk_budget(signals, date)
        
        # 计算当前组合市值
        current_value = self._calculate_portfolio_value(date)
        
        # 买入信号中的股票
        for stock, weight in signals.items():
            target_value = current_value * weight
            self._buy_stock(date, stock, target_value)
        
        if self.verbose:
            logger.info(f"买入执行: {date.date()}, 买入 {len(signals)} 只股票（信号日: {signal_date.date()}）")
    
    def _check_and_sell(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict) -> None:
        """检查并执行卖出操作（达到持有期 T+n）
        
        Args:
            date: 当前日期
            trading_dates: 交易日列表
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
            self._sell_stock(date, stock)
        
        if stocks_to_sell and self.verbose:
            logger.info(f"卖出执行: {date.date()}, 卖出 {len(stocks_to_sell)} 只股票（达到持有期）")
    
    def _prepare_price_index(self, price_data: pd.DataFrame) -> None:
        """准备价格索引（使用 MultiIndex，替代嵌套字典）
        
        构建两套价格序列：
        - trade_price_index: 成交价格（不复权 close）
        - pnl_price_index: 绩效价格（后复权 close_adj）
        
        Args:
            price_data: 价格数据，需包含 ts_code, trade_date, close, close_adj（可选）
        """
        logger.info("开始准备价格索引...")
        
        # 检查必需列
        if 'close' not in price_data.columns:
            raise ValueError("价格数据缺少 'close' 列，无法进行回测")
        
        # 转换日期列为 datetime（向量化操作，避免 iterrows）
        if not pd.api.types.is_datetime64_any_dtype(price_data['trade_date']):
            # 创建副本以避免修改原始数据
            price_data = price_data.copy()
            price_data['trade_date'] = pd.to_datetime(price_data['trade_date'])
        
        # 构建成交价格索引（不复权 close）
        trade_price_df = price_data[['trade_date', 'ts_code', 'close']].copy()
        trade_price_df.set_index(['trade_date', 'ts_code'], inplace=True)
        self.trade_price_index = trade_price_df['close']
        
        # 构建绩效价格索引（后复权 close_adj）
        if 'close_adj' in price_data.columns:
            pnl_price_df = price_data[['trade_date', 'ts_code', 'close_adj']].copy()
            pnl_price_df.set_index(['trade_date', 'ts_code'], inplace=True)
            self.pnl_price_index = pnl_price_df['close_adj']
            logger.info("价格索引构建完成: 成交价格=close, 绩效价格=close_adj")
        else:
            # 如果缺少 close_adj，回退到 close
            logger.warning(f"价格数据缺少 'close_adj' 列，绩效价格将使用 'close' 列（不复权）")
            self.pnl_price_index = self.trade_price_index.copy()
            logger.info("价格索引构建完成: 成交价格=close, 绩效价格=close（退化）")
    
    def _get_trade_price(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取成交价格（不复权 close）
        
        Args:
            date: 日期
            stock: 股票代码
            
        Returns:
            成交价格，如果不存在则返回 None
        """
        try:
            return self.trade_price_index.loc[(date, stock)]
        except KeyError:
            return None
    
    def _get_pnl_price(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取绩效价格（后复权 close_adj）
        
        Args:
            date: 日期
            stock: 股票代码
            
        Returns:
            绩效价格，如果不存在则返回 None
        """
        try:
            return self.pnl_price_index.loc[(date, stock)]
        except KeyError:
            return None
    
    def _calculate_volatility(self, stock: str, end_date: pd.Timestamp) -> float:
        """计算个股历史波动率（基于绩效价格，避免未来函数）
        
        使用 end_date 之前的 vol_window 个交易日的收益率计算波动率
        
        Args:
            stock: 股票代码
            end_date: 结束日期（不包含，只使用该日期之前的数据）
            
        Returns:
            年化波动率
        """
        try:
            # 获取该股票的所有绩效价格（按日期排序）
            stock_prices = self.pnl_price_index.xs(stock, level='ts_code').sort_index()
            
            # 筛选 end_date 之前的数据
            stock_prices = stock_prices[stock_prices.index < end_date]
            
            if len(stock_prices) < 2:
                return self.vol_epsilon
            
            # 取最近 vol_window 个交易日
            recent_prices = stock_prices.iloc[-self.vol_window:]
            
            if len(recent_prices) < 2:
                return self.vol_epsilon
            
            # 计算日收益率
            returns = recent_prices.pct_change().dropna()
            
            if len(returns) < 2:
                return self.vol_epsilon
            
            # 计算波动率（年化，假设每年252个交易日）
            vol = returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            
            # 确保波动率不低于 epsilon
            return max(vol, self.vol_epsilon)
            
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"计算 {stock} 波动率时出错: {e}，使用默认值 {self.vol_epsilon}")
            return self.vol_epsilon
    
    def _apply_risk_budget(self, signals: Dict[str, float], date: pd.Timestamp) -> Dict[str, float]:
        """应用风险预算（波动率缩放）
        
        调整权重: adj_weight ∝ raw_weight / volatility
        然后归一化使权重和为1
        
        Args:
            signals: 原始信号 {stock: weight}
            date: 当前日期（买入日期）
            
        Returns:
            调整后的信号 {stock: adj_weight}
        """
        if not signals:
            return signals
        
        # 计算每只股票的波动率（使用 date 之前的数据）
        volatilities = {}
        for stock in signals:
            vol = self._calculate_volatility(stock, date)
            volatilities[stock] = vol
        
        # 计算调整后的权重: raw_weight / volatility
        adj_weights = {}
        for stock, weight in signals.items():
            adj_weights[stock] = weight / volatilities[stock]
        
        # 归一化
        total_adj_weight = sum(adj_weights.values())
        if total_adj_weight > 0:
            for stock in adj_weights:
                adj_weights[stock] /= total_adj_weight
        else:
            # 如果总权重为0，均分
            n = len(adj_weights)
            for stock in adj_weights:
                adj_weights[stock] = 1.0 / n if n > 0 else 0.0
        
        return adj_weights
    
    def _get_rebalance_dates(self, trading_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
        """获取调仓日期
        
        Args:
            trading_dates: 交易日列表
            
        Returns:
            调仓日期列表
        """
        #if self.rebalance_freq.isdecimal():
        #    self.rebalance_freq = int(self.rebalance_freq)
        # 支持整数天数
        if isinstance(self.rebalance_freq, int):
            # 每 N 个交易日调仓一次
            n = self.rebalance_freq
            if n <= 0:
                raise ValueError(f"调仓频率必须为正整数，当前值: {n}")
            return [trading_dates[i] for i in range(0, len(trading_dates), n)]
        
        # 支持字符串频率
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
            raise ValueError(
                f"不支持的调仓频率: {self.rebalance_freq}。"
                f"请使用 'D'（日）、'W'（周）、'M'（月）或正整数（每N天）"
            )
    
    
    
    def _process_pending_orders(self, date: pd.Timestamp) -> None:
        """处理延迟订单队列
        
        Args:
            date: 当前日期
        """
        if not self.pending_order_manager:
            return
        
        # 获取应重试的订单列表
        orders_to_retry = self.pending_order_manager.get_orders_to_retry(date)
        
        if not orders_to_retry:
            return
        
        # 获取当日行情数据
        date_quote = self.price_data_cache[self.price_data_cache['trade_date'] == date]
        trade_date_str = date.strftime('%Y%m%d')
        
        for order in orders_to_retry:
            # 检查是否可交易
            tradeable, reason = is_tradeable(
                order.stock, trade_date_str, date_quote, action=order.action
            )
            
            if tradeable:
                # 可交易，尝试执行
                if order.action == 'buy':
                    self._buy_stock_direct(date, order.stock, order.target_value)
                    self.pending_order_manager.mark_success(order.stock, 'buy')
                elif order.action == 'sell':
                    self._sell_stock_direct(date, order.stock)
                    self.pending_order_manager.mark_success(order.stock, 'sell')
            else:
                # 仍不可交易，更新延迟订单
                self.pending_order_manager.add_order(
                    stock=order.stock,
                    action=order.action,
                    current_date=date,
                    signal_date=order.signal_date,
                    target_value=order.target_value,
                    reason=reason
                )
    
    def _buy_stock_with_status_check(self, date: pd.Timestamp, stock: str, target_value: float, signal_date: Optional[pd.Timestamp] = None) -> None:
        """买入股票（带交易状态检查）
        
        如果启用延迟订单功能，会检查股票是否可交易（停牌、涨停）
        不可交易时加入延迟队列而非直接失败
        
        Args:
            date: 买入日期（T+1）
            stock: 股票代码
            target_value: 目标市值
            signal_date: 信号生成日期（用于延迟订单）
        """
        # 检查交易状态
        if self.enable_pending_order and self.price_data_cache is not None:
            date_quote = self.price_data_cache[self.price_data_cache['trade_date'] == date]
            trade_date_str = date.strftime('%Y%m%d')
            tradeable, reason = is_tradeable(stock, trade_date_str, date_quote, action='buy')
            
            if not tradeable:
                # 不可交易，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action='buy',
                        current_date=date,
                        signal_date=signal_date or date,
                        target_value=target_value,
                        reason=reason
                    )
                if self.verbose:
                    logger.info(
                        f"买入延迟: {date.date()} {stock}, 原因: {reason}, "
                        f"目标市值: {target_value:.2f}"
                    )
                return
        
        # 可交易，直接买入
        self._buy_stock_direct(date, stock, target_value)
    
    def _buy_stock_direct(self, date: pd.Timestamp, stock: str, target_value: float) -> None:
        """直接买入股票（不检查交易状态）
        
        内部使用，实际执行买入操作
        
        Args:
            date: 买入日期
            stock: 股票代码
            target_value: 目标市值
        """
        # 获取成交价格（不复权 close）
        trade_price = self._get_trade_price(date, stock)
        if trade_price is None:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的成交价格，跳过买入")
            return
        
        # 获取绩效价格（后复权 close_adj）
        pnl_price = self._get_pnl_price(date, stock)
        if pnl_price is None:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的绩效价格，使用成交价格代替")
            pnl_price = trade_price
        
        # 按手买入（100股为一手）
        shares = int(target_value / trade_price / 100) * 100
        
        if shares == 0:
            return
        
        # 计算买入金额和成本（基于成交价格）
        amount = shares * trade_price
        cost = self.cost_model.calculate_buy_cost(amount)
        total_cost_cash = amount + cost  # 总现金支出（含手续费）
        
        if total_cost_cash > self.current_capital:
            # 资金不足，按可用资金买入
            # 确保有足够资金支付手续费
            if self.current_capital <= cost:
                # 资金不足以支付手续费，无法买入
                return
            
            shares = int((self.current_capital - cost) / trade_price / 100) * 100
            if shares == 0:
                return
            amount = shares * trade_price
            cost = self.cost_model.calculate_buy_cost(amount)
            total_cost_cash = amount + cost
        
        # 更新持仓和资金
        # 注意：在当前 T+n 卖出策略下，理论上不应该出现已有持仓的情况
        # 因为旧持仓应该在达到持有期后自动卖出
        if stock in self.positions:
            logger.warning(
                f"股票 {stock} 已有持仓（买入日期: {self.positions[stock]['buy_date']}），"
                f"新买入将覆盖旧持仓（可能配置有误）"
            )
        
        # 设置或覆盖持仓（记录买入的成交价格和绩效价格）
        self.positions[stock] = {
            'shares': shares,
            'buy_date': date,
            'buy_trade_price': trade_price,  # 成交价格（不复权）
            'buy_pnl_price': pnl_price,      # 绩效价格（后复权）
            'buy_cost_cash': total_cost_cash  # 总现金支出（含手续费）
        }
        
        self.current_capital -= total_cost_cash
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock': stock,
            'action': 'buy',
            'price': trade_price,            # 成交价格
            'shares': shares,
            'amount': amount,
            'cost': cost
        })
    
    def _buy_stock(self, date: pd.Timestamp, stock: str, target_value: float, signal_date: Optional[pd.Timestamp] = None) -> None:
        """买入股票（在 T+1 日以收盘价买入）
        
        带交易状态检查的买入方法。如果启用延迟订单功能，会检查股票是否可交易。
        
        Args:
            date: 买入日期（T+1）
            stock: 股票代码
            target_value: 目标市值
            signal_date: 信号生成日期（用于延迟订单）
        """
        self._buy_stock_with_status_check(date, stock, target_value, signal_date)
    
    def _sell_stock_with_status_check(self, date: pd.Timestamp, stock: str) -> None:
        """卖出股票（带交易状态检查）
        
        如果启用延迟订单功能，会检查股票是否可交易（跌停）
        不可交易时加入延迟队列而非直接失败
        
        Args:
            date: 卖出日期
            stock: 股票代码
        """
        # 检查交易状态
        if self.enable_pending_order and self.price_data_cache is not None:
            date_quote = self.price_data_cache[self.price_data_cache['trade_date'] == date]
            trade_date_str = date.strftime('%Y%m%d')
            tradeable, reason = is_tradeable(stock, trade_date_str, date_quote, action='sell')
            
            if not tradeable:
                # 不可交易，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action='sell',
                        current_date=date,
                        signal_date=date,  # 卖出是基于持有期，用当前日期
                        target_value=None,
                        reason=reason
                    )
                if self.verbose:
                    logger.info(f"卖出延迟: {date.date()} {stock}, 原因: {reason}")
                return
        
        # 可交易，直接卖出
        self._sell_stock_direct(date, stock)
    
    def _sell_stock(self, date: pd.Timestamp, stock: str) -> None:
        """卖出股票（在 T+n 日以收盘价卖出）
        
        带交易状态检查的卖出方法。如果启用延迟订单功能，会检查股票是否可交易。
        
        Args:
            date: 卖出日期（T+n）
            stock: 股票代码
        """
        self._sell_stock_with_status_check(date, stock)
    
    def _sell_stock_direct(self, date: pd.Timestamp, stock: str) -> None:
        """直接卖出股票（不检查交易状态）
        
        现金流使用成交价格（trade_price）计算
        收益率使用绩效价格（pnl_price）计算
        
        Args:
            date: 卖出日期（T+n）
            stock: 股票代码
        """
        if stock not in self.positions or self.positions[stock]['shares'] == 0:
            return
        
        # 获取成交价格（不复权 close）
        sell_trade_price = self._get_trade_price(date, stock)
        if sell_trade_price is None:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的成交价格，跳过卖出")
            return
        
        # 获取绩效价格（后复权 close_adj）
        sell_pnl_price = self._get_pnl_price(date, stock)
        if sell_pnl_price is None:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的绩效价格，使用成交价格代替")
            sell_pnl_price = sell_trade_price
        
        # 获取持仓信息
        shares = self.positions[stock]['shares']
        buy_trade_price = self.positions[stock]['buy_trade_price']
        buy_pnl_price = self.positions[stock]['buy_pnl_price']
        buy_cost_cash = self.positions[stock]['buy_cost_cash']
        
        # 计算现金流（基于成交价格）
        sell_amount = shares * sell_trade_price
        sell_cost = self.cost_model.calculate_sell_cost(sell_amount)
        sell_proceeds = sell_amount - sell_cost  # 卖出后实际到手金额
        
        # 计算收益率（基于绩效价格）
        pnl_buy_amount = shares * buy_pnl_price  # 绩效口径买入金额
        pnl_sell_amount = shares * sell_pnl_price  # 绩效口径卖出金额
        
        # 买入和卖出的手续费
        buy_amount = shares * buy_trade_price
        buy_cost = self.cost_model.calculate_buy_cost(buy_amount)
        total_cost = buy_cost + sell_cost  # 总手续费
        
        # 绩效收益（基于绩效价格，扣除手续费）
        # 收益 = 卖出金额 - 买入金额 - 总手续费
        pnl_profit_amount = pnl_sell_amount - pnl_buy_amount - total_cost
        # 收益率 = 收益 / (买入金额 + 买入手续费)
        # 买入成本是买入金额+买入手续费，这是投资者实际付出的成本
        pnl_profit_pct = (
            pnl_profit_amount / (pnl_buy_amount + buy_cost) 
            if (pnl_buy_amount + buy_cost) > 0 else 0
        )
        
        # 更新持仓和资金
        del self.positions[stock]
        self.current_capital += sell_proceeds
        
        # 记录交易（包含绩效收益信息）
        self.trades.append({
            'date': date,
            'stock': stock,
            'action': 'sell',
            'price': sell_trade_price,       # 卖出成交价格
            'shares': shares,
            'amount': sell_amount,
            'cost': sell_cost,
            'buy_price': buy_trade_price,    # 买入成交价格
            'buy_pnl_price': buy_pnl_price,  # 买入绩效价格
            'sell_pnl_price': sell_pnl_price,  # 卖出绩效价格
            'pnl_profit_amount': pnl_profit_amount,  # 绩效收益金额
            'pnl_profit_pct': pnl_profit_pct  # 绩效收益率
        })
    
    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """计算组合市值（基于成交价格）
        
        Args:
            date: 计算日期
            
        Returns:
            组合总市值
        """
        market_value = 0.0
        
        for stock, info in self.positions.items():
            shares = info['shares']
            trade_price = self._get_trade_price(date, stock)
            if trade_price is not None:
                market_value += shares * trade_price
        
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
