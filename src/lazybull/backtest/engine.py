"""回测引擎"""

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ..common.cost import CostModel
from ..common.trade_status import is_tradeable
from ..common.date_utils import to_trade_date_str
from ..execution.pending_order import PendingOrderManager
from ..signals.base import Signal
from ..universe.base import Universe
from ..risk.stop_loss import StopLossConfig, StopLossMonitor


class BacktestEngine:
    """回测引擎
    
    执行回测流程，生成净值曲线和交易记录
    
    交易规则：
    - T 日生成信号
    - T+1 日收盘价买入
    - T+n 日卖出（n 为持有期，默认与调仓频率一致）
      - 卖出时机可配置：收盘价（默认）或开盘价
    """
    
    # 常量：每年交易日数量（用于年化波动率计算）
    TRADING_DAYS_PER_YEAR = 252
    
    def __init__(
        self,
        universe: Universe,
        signal: Signal,
        initial_capital: float = 1000000.0,
        cost_model: Optional[CostModel] = None,
        rebalance_freq: int = 5,
        holding_period: Optional[int] = None,
        verbose: bool = True,
        enable_risk_budget: bool = False,
        vol_window: int = 20,
        vol_epsilon: float = 0.001,
        enable_pending_order: bool = True,
        max_retry_count: int = 5,
        max_retry_days: int = 10,
        stop_loss_config: Optional[StopLossConfig] = None,
        sell_timing: str = 'close',
        enable_position_completion: bool = True,
        completion_window_days: int = 3
    ):
        """初始化回测引擎
        
        价格口径说明：
        - 成交价格（trade_price）：使用不复权 close/open，用于计算成交金额、持仓市值、可买入数量
        - 绩效价格（pnl_price）：使用后复权 close_adj/open_adj，用于计算收益率和绩效指标
        
        Args:
            universe: 股票池
            signal: 信号生成器
            initial_capital: 初始资金
            cost_model: 成本模型
            rebalance_freq: 调仓频率（交易日数），必须为正整数。例如：5表示每5个交易日调仓一次
            holding_period: 持有期（交易日），None 则自动根据调仓频率设置
            verbose: 是否输出详细日志（买入/卖出操作），默认True
            enable_risk_budget: 是否启用风险预算/波动率缩放，默认False（保持向后兼容）
            vol_window: 波动率计算窗口（交易日），默认20
            vol_epsilon: 波动率缩放的最小波动率，防止除零，默认0.001
            enable_pending_order: 是否启用延迟订单功能，默认True
            max_retry_count: 延迟订单最大重试次数，默认5次
            max_retry_days: 延迟订单最大延迟天数，默认10天
            stop_loss_config: 止损配置，None 表示不启用止损功能（默认）
            sell_timing: 卖出时机，'close' 表示 T+n 日收盘价卖出（默认），'open' 表示 T+n 日开盘价卖出
            enable_position_completion: 是否启用仓位补齐功能，默认True
            completion_window_days: 补齐窗口期（交易日），默认3天
        """
        self.universe = universe
        self.signal = signal
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        
        # 验证调仓频率
        if not isinstance(rebalance_freq, int):
            raise TypeError(f"调仓频率必须为整数类型，当前类型: {type(rebalance_freq).__name__}")
        if rebalance_freq <= 0:
            raise ValueError(f"调仓频率必须为正整数，当前值: {rebalance_freq}")
        
        # 验证卖出时机参数
        if sell_timing not in ['close', 'open']:
            raise ValueError(f"卖出时机参数必须为 'close' 或 'open'，当前值: {sell_timing}")
        
        self.rebalance_freq = rebalance_freq
        self.sell_timing = sell_timing
        self.verbose = verbose
        
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
        
        # 仓位补齐参数
        self.enable_position_completion = enable_position_completion
        self.completion_window_days = completion_window_days
        
        # 止损配置
        self.stop_loss_config = stop_loss_config
        self.stop_loss_monitor = None
        if stop_loss_config and stop_loss_config.enabled:
            self.stop_loss_monitor = StopLossMonitor(stop_loss_config)
        
        # 持有期逻辑：如果未指定，与调仓频率保持一致
        if holding_period is None:
            self.holding_period = self.rebalance_freq
        else:
            self.holding_period = holding_period  # 修复：应使用传入的 holding_period
        
        
        # 回测状态
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}  # {股票代码: {shares, buy_date, buy_trade_price, buy_pnl_price, buy_cost_cash}}
        self.pending_signals: Dict[pd.Timestamp, Dict] = {}  # {信号日期: {股票: 权重}}
        self.pending_stop_loss_sells: Dict[str, Dict] = {}  # {股票代码: {trigger_date, reason, trigger_type}} 待止损卖出队列
        self.portfolio_values: List[Dict] = []  # 组合价值历史
        self.trades: List[Dict] = []  # 交易记录
        
        # 仓位补齐状态跟踪
        # {调仓日期: {未成交股票列表, 目标数量, 候选列表, 剩余权重字典}}
        self.unfilled_slots: Dict[pd.Timestamp, Dict] = {}
        # 补齐统计
        self.completion_stats = {
            'total_unfilled': 0,      # 累计未满仓次数
            'total_completed': 0,     # 累计补齐成功次数
            'total_abandoned': 0,     # 累计放弃补齐次数
            'completion_attempts': 0  # 累计补齐尝试次数
        }
        
        # 价格索引（在 run 时初始化）
        self.trade_price_index: Optional[pd.Series] = None  # 成交价格（不复权 close）
        self.pnl_price_index: Optional[pd.Series] = None  # 绩效价格（后复权 close_adj）
        self.trade_price_open_index: Optional[pd.Series] = None  # 开盘成交价格（不复权 open）
        self.pnl_price_open_index: Optional[pd.Series] = None  # 开盘绩效价格（后复权 open_adj）
        
        # 存储价格数据用于交易状态检查
        self.price_data_cache: Optional[pd.DataFrame] = None
        
        logger.info(
            f"回测引擎初始化完成: 初始资金={initial_capital}, "
            f"调仓频率={self.rebalance_freq}, 持有期={self.holding_period}天, "
            f"卖出时机={self.sell_timing}, "
            f"风险预算={'启用' if enable_risk_budget else '禁用'}, "
            f"延迟订单={'启用' if enable_pending_order else '禁用'}, "
            f"仓位补齐={'启用' if enable_position_completion else '禁用'}, "
            f"补齐窗口={completion_window_days}天, "
            f"止损功能={'启用' if (stop_loss_config and stop_loss_config.enabled) else '禁用'}, "
            f"详细日志={'开启' if verbose else '关闭'}"
        )
        sell_price_type = "开盘价" if self.sell_timing == 'open' else "收盘价"
        logger.info(f"交易规则: T日生成信号 -> T+1日收盘价买入 -> T+{self.holding_period}日{sell_price_type}卖出")
        logger.info(f"价格口径: 成交使用不复权 close/open, 绩效使用后复权 close_adj/open_adj")

    
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
            logger.info(f"回测进度: {idx + 1}/{total_days} 天, 日期: {date.date()}")
            # 处理延迟订单（先处理延迟订单，再处理新信号）
            if self.enable_pending_order:
                self._process_pending_orders(date)
            
            # 检查止损（T 日检查，T+1 日执行卖出）
            if self.stop_loss_monitor:
                self._check_stop_loss(date, trading_dates, date_to_idx)
            
            # 判断是否为信号生成日
            if date in signal_dates:
                self._generate_signal(date, trading_dates, price_data, date_to_idx)

            # @2026/01/18: 改为先卖出再买入, 避免当天买入的股票被误判为达到持有期而卖出
            # TODO: 更正确的做法应该是在持有期计算中排除当天买入的股票, 此部分还待优化
            # 执行止损卖出（T+1 日执行）
            if self.stop_loss_monitor:
                self._execute_pending_stop_loss_sells(date, trading_dates, date_to_idx)
            
            # 检查并执行卖出操作（达到持有期）
            self._check_and_sell(date, trading_dates, date_to_idx)

            # 执行待执行的买入操作（T+1）
            self._execute_pending_buys(date, trading_dates, date_to_idx)
            
            # 处理仓位补齐（在补齐窗口期内尝试补齐未满仓位）
            if self.enable_position_completion:
                self._process_position_completion(date, trading_dates, price_data, date_to_idx)
            
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
        
        # 输出仓位补齐统计
        if self.enable_position_completion:
            logger.info(
                f"仓位补齐统计: 累计未满仓 {self.completion_stats['total_unfilled']} 次, "
                f"补齐成功 {self.completion_stats['total_completed']} 次, "
                f"补齐尝试 {self.completion_stats['completion_attempts']} 次, "
                f"放弃补齐 {self.completion_stats['total_abandoned']} 次"
            )
        
        return nav_df
    
    def _build_signal_data(self, date: pd.Timestamp) -> Optional[Dict]:
        """构建传递给信号生成器的额外数据（扩展点）
        
        子类可以重写此方法以注入特定数据（如 ML 特征）。
        
        Args:
            date: 信号生成日期
            
        Returns:
            数据字典，将与默认数据合并后传递给 signal.generate_ranked()
            返回 None 表示该日期无可用数据，将跳过信号生成
        """
        return {}
    
    def _generate_signal(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], price_data: pd.DataFrame, date_to_idx: Dict) -> None:
        """生成信号（在 T 日生成，T+1 日执行买入）
        
        新逻辑：生成排序候选列表，在 T+1 日过滤不可交易股票并回填，确保 top N 全部可交易。
        
        Args:
            date: 信号生成日期
            trading_dates: 交易日列表
            price_data: 价格数据，包含行情信息
            date_to_idx: 日期到索引的映射
        """
        # 获取当日行情数据用于基础过滤（ST、停牌等基础过滤）
        trade_date_str = to_trade_date_str(date)
        date_quote = price_data[price_data['trade_date'] == trade_date_str]
        # 获取股票池（不过滤涨跌停，因为 T 日涨跌停不代表 T+1 日也涨跌停）
        # 但保留 ST、基本可交易性等过滤
        stock_universe = self.universe.get_stocks(date, quote_data=date_quote)
        
        # 调用扩展点获取额外数据（如 ML 特征）
        extra_data = self._build_signal_data(date)
        if extra_data is None:
            # None 表示该日期无可用数据，跳过信号生成
            if self.verbose:
                logger.warning(f"信号日 {date.date()} 无可用数据（_build_signal_data 返回 None），跳过")
            return
        
        # 合并默认数据和额外数据
        signal_data = {}
        signal_data.update(extra_data)
        
        # 生成排序后的候选列表（返回所有候选，不仅仅是 top N）
        ranked_candidates = self.signal.generate_ranked(date, stock_universe, signal_data)
        
        if not ranked_candidates:
            if self.verbose:
                logger.warning(f"信号日 {date.date()} 无候选")
            return
        
        # 获取 T+1 日（买入日）的行情数据
        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx + 1 >= len(trading_dates):
            # 没有 T+1 日，无法买入
            if self.verbose:
                logger.warning(f"信号日 {date.date()} 之后没有交易日，无法执行")
            return
        
        buy_date = trading_dates[current_idx + 1]
        buy_date_str = to_trade_date_str(buy_date)
        buy_date_quote = price_data[price_data['trade_date'] == buy_date_str]
        
        # 从排序候选中选择 top N 股票
        # 当启用仓位补齐功能时，不在信号生成阶段过滤 T+1 的涨停/停牌，
        # 而是在 T+1 执行买入时处理失败，并在 T+2 等日期补齐
        signals = {}
        candidates_checked = 0
        filtered_reasons = {'停牌': 0, '涨停': 0, '跌停': 0}
        
        # 获取目标数量（从信号生成器获取）
        if hasattr(self.signal, 'top_n'):
            target_n = self.signal.top_n
        else:
            # 如果信号生成器没有 top_n 属性，则使用所有候选
            target_n = len(ranked_candidates)
        
        if self.enable_position_completion:
            # 启用补齐功能：直接选择 top N 股票，不检查 T+1 可交易性
            # 这样可以在 T+1 买入失败时触发补齐流程
            for stock, score in ranked_candidates[:target_n]:
                signals[stock] = score
                candidates_checked += 1
        else:
            # 未启用补齐功能：保留原有逻辑，在 T+1 过滤不可交易股票并回填
            for stock, score in ranked_candidates:
                candidates_checked += 1
                
                # 检查 T+1 日该股票是否可买入
                if buy_date_quote.empty:
                    # T+1 日行情数据为空，无法判断交易状态，跳过
                    filtered_reasons['停牌'] += 1
                    if self.verbose:
                        logger.warning(
                            f"信号日 {date.date()} 的候选股票 {stock} 在 T+1 日 {buy_date.date()} 无行情数据，"
                            f"假定不可买入，从候选中回填"
                        )
                    continue

                tradeable, reason = is_tradeable(stock, buy_date_str, buy_date_quote, action='buy')
                
                if tradeable:
                    # 可交易，加入信号
                    signals[stock] = score
                    
                    # 达到目标数量，停止
                    if len(signals) >= target_n:
                        break
                else:
                    # 不可交易，记录原因并继续检查下一个候选
                    filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1
                    if self.verbose:
                        logger.warning(
                            f"候选股票 {stock} 在 {buy_date.date()} 不可买入(原因: {reason})，"
                            f"从候选中顺延选择"
                        )
        
        if not signals:
            if self.verbose:
                logger.warning(
                    f"信号日 {date.date()} 所有候选在 T+1 日 {buy_date.date()} 均不可交易，"
                    f"检查了 {candidates_checked} 个候选"
                )
            return
        
        # 归一化权重（将分数转换为权重，使其和为 1）
        # 使用 getattr 保证向后兼容（某些 Mock 信号对象可能没有 weight_method 属性）
        weight_method = getattr(self.signal, 'weight_method', 'equal')
        if weight_method == "equal":
            # 等权
            weight = 1.0 / len(signals)
            signals = {stock: weight for stock in signals.keys()}
        else:
            # 按分数加权
            total_score = sum(signals.values())
            if total_score > 0:
                signals = {stock: score / total_score for stock, score in signals.items()}
            else:
                # 如果所有分数都是0或负数，使用等权
                weight = 1.0 / len(signals)
                signals = {stock: weight for stock in signals.keys()}
        
        # 保存信号，待 T+1 执行
        # 同时保存完整的排序候选列表用于补齐（如果启用补齐功能）
        self.pending_signals[date] = {
            'signals': signals,
            'ranked_candidates': ranked_candidates if self.enable_position_completion else [],
            'target_n': target_n
        }
        
        if self.verbose:
            if self.enable_position_completion:
                logger.info(
                    f"信号生成: {date.date()}, 选择 top {len(signals)}/{target_n} 股票（未检查 T+1 可交易性，将在买入时处理）, "
                    f"候选总数 {len(ranked_candidates)} 个"
                )
            else:
                logger.info(
                    f"信号生成: {date.date()}, 信号数 {len(signals)}/{target_n}, "
                    f"检查候选 {candidates_checked} 个, "
                    f"过滤: 停牌 {filtered_reasons.get('停牌', 0)}, "
                    f"涨停 {filtered_reasons.get('涨停', 0)}, "
                    f"跌停 {filtered_reasons.get('跌停', 0)}"
                )
    
    
    def _execute_pending_buys(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict) -> None:
        """执行待执行的买入操作（T+1）
        
        同时跟踪未成交的槽位，如果启用补齐功能则记录到 unfilled_slots
        
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
        
        signal_data = self.pending_signals.pop(signal_date)
        
        # 兼容性处理：支持旧格式和新格式
        # 旧格式（补齐功能禁用时）：signal_data = {stock: weight}
        # 新格式（补齐功能启用时）：signal_data = {'signals': {stock: weight}, 'ranked_candidates': [...], 'target_n': N}
        if isinstance(signal_data, dict) and 'signals' in signal_data:
            # 新格式
            signals = signal_data['signals']
            ranked_candidates = signal_data.get('ranked_candidates', [])
            target_n = signal_data.get('target_n', len(signals))
        else:
            # 旧格式兼容（当 enable_position_completion=False 或旧代码生成的信号）
            signals = signal_data
            ranked_candidates = []
            target_n = len(signals)
        
        # 应用风险预算（波动率缩放）
        if self.enable_risk_budget:
            signals = self._apply_risk_budget(signals, date)
        
        # 计算当前组合市值
        current_value = self._calculate_portfolio_value(date)
        
        # 记录买入前的持仓数量
        positions_before = len(self.positions)
        
        # 当启用补齐功能时，需要检查可交易性，因为信号生成时未检查 T+1 可交易性
        # 当未启用补齐功能时，信号生成时已经过滤，可以直接买入
        if self.enable_position_completion:
            # 获取当日行情数据用于交易性检查
            trade_date_str = to_trade_date_str(date)
            date_quote = self.price_data_cache[self.price_data_cache['trade_date'] == trade_date_str] if self.price_data_cache is not None else pd.DataFrame()
            
            # 买入信号中的股票，检查可交易性
            for stock, weight in signals.items():
                target_value = current_value * weight
                
                # 检查可交易性
                if not date_quote.empty:
                    tradeable, reason = is_tradeable(stock, trade_date_str, date_quote, action='buy')
                    
                    if not tradeable:
                        if self.verbose:
                            logger.info(
                                f"买入失败: {date.date()} {stock}, 原因: {reason}, "
                                f"权重 {weight:.4f}, 将在后续交易日补齐"
                            )
                        continue  # 跳过该股票，不买入
                
                # 可交易，执行买入
                self._buy_stock(date, stock, target_value, signal_date=signal_date)
        else:
            # 未启用补齐功能，直接买入（信号生成时已过滤）
            for stock, weight in signals.items():
                target_value = current_value * weight
                self._buy_stock(date, stock, target_value, signal_date=signal_date)
        
        # 记录买入后的持仓数量
        positions_after = len(self.positions)
        actually_bought = positions_after - positions_before
        
        # 如果启用补齐功能，检查是否有未成交的槽位
        # 修复：应该对比 target_n 而非 len(signals)，因为 signals 可能已经过滤或调整
        if self.enable_position_completion and actually_bought < target_n:
            # 找出未成交的股票
            unfilled_stocks = [stock for stock in signals.keys() if stock not in self.positions]
            
            if ranked_candidates:
                # 计算缺口槽位数量
                unfilled_count = target_n - actually_bought
                
                # 将 signals 的权重转换为槽位权重列表（按信号中的顺序）
                # 这样可以在补齐时为每个缺口槽位分配固定权重
                slot_weights = []
                for stock, weight in signals.items():
                    slot_weights.append({
                        'stock': stock,
                        'weight': weight,
                        'filled': stock in self.positions  # 标记是否已成交
                    })
                
                # 提取未成交槽位的权重
                unfilled_slot_weights = [slot for slot in slot_weights if not slot['filled']]
                
                # 记录未成交槽位信息，准备补齐
                self.unfilled_slots[signal_date] = {
                    'unfilled_count': unfilled_count,
                    'unfilled_slot_weights': unfilled_slot_weights,  # 保留原始权重序列
                    'target_n': target_n,
                    'ranked_candidates': ranked_candidates,
                    'signal_date': signal_date,  # 信号生成日（T日）
                    'first_attempt_date': date,  # T+1 日，第一次尝试买入的日期
                    'attempts': 0  # 补齐尝试次数
                }
                
                self.completion_stats['total_unfilled'] += 1
                
                if self.verbose:
                    logger.warning(
                        f"仓位未满: {date.date()}, 目标 {target_n} 只, 实际买入 {actually_bought} 只, "
                        f"缺口槽位 {unfilled_count} 个, 未成交股票: {unfilled_stocks}, "
                        f"将在接下来 {self.completion_window_days} 天内尝试补齐"
                    )
        
        if self.verbose:
            logger.info(f"买入执行: {date.date()}, 买入 {actually_bought} 只股票（信号日: {signal_date.date()}）")
    
    def _process_position_completion(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], price_data: pd.DataFrame, date_to_idx: Dict) -> None:
        """处理仓位补齐逻辑
        
        在调仓日后的 T+1 至 T+completion_window_days 天内，尝试补齐未成交的槽位：
        1. 基于上一交易日 D-1 的数据重新生成候选股票（避免使用未来数据）
        2. 从候选中选择可用股票填补缺口，但使用调仓日 T 生成的槽位权重
        3. 检查当日 D 可交易性，不可交易则保留该槽位到下次补齐
        4. 超过补齐窗口则放弃
        
        Args:
            date: 当前日期（补齐买入日 D）
            trading_dates: 交易日列表
            price_data: 价格数据
            date_to_idx: 日期到索引的映射
        """
        if not self.unfilled_slots:
            return
        
        # 获取上一交易日（D-1）用于生成候选
        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx == 0:
            return
        
        prev_date = trading_dates[current_idx - 1]
        prev_date_str = to_trade_date_str(prev_date)
        prev_date_quote = price_data[price_data['trade_date'] == prev_date_str]
        
        # 获取当日（D）行情数据用于交易性检查
        trade_date_str = to_trade_date_str(date)
        date_quote = price_data[price_data['trade_date'] == trade_date_str]
        
        if date_quote.empty:
            if self.verbose:
                logger.warning(f"补齐跳过: {date.date()}, 当日无行情数据")
            return
        
        # 遍历所有未补齐的槽位
        completed_signal_dates = []
        
        for signal_date, slot_info in list(self.unfilled_slots.items()):
            first_attempt_date = slot_info['first_attempt_date']
            unfilled_slot_weights = slot_info['unfilled_slot_weights']
            target_n = slot_info['target_n']
            attempts = slot_info['attempts']
            original_signal_date = slot_info['signal_date']  # T日
            
            # 计算已经过了多少个交易日（从 T+1 开始）
            first_attempt_idx = date_to_idx.get(first_attempt_date)
            
            if first_attempt_idx is None:
                continue
            
            days_elapsed = current_idx - first_attempt_idx
            
            # 在 T+1 日（首次尝试日）不进行补齐，从 T+2 日开始
            if days_elapsed == 0:
                continue
            
            # 检查是否超过补齐窗口（窗口从 T+1 开始，所以是 < completion_window_days）
            if days_elapsed >= self.completion_window_days:
                # 超过补齐窗口，放弃补齐
                unfilled_count = len(unfilled_slot_weights)
                unfilled_stocks_str = ', '.join([slot['stock'] for slot in unfilled_slot_weights])
                self.completion_stats['total_abandoned'] += 1
                completed_signal_dates.append(signal_date)
                
                logger.warning(
                    f"补齐放弃: {date.date()}, 信号日 {original_signal_date.date()}, "
                    f"已尝试 {attempts} 次补齐, 仍有 {unfilled_count} 个槽位未成交: {unfilled_stocks_str}, "
                    f"超过补齐窗口 {self.completion_window_days} 天，放弃补齐，对应权重持币"
                )
                continue
            
            # 在补齐窗口内，尝试补齐
            # 使用 D-1 日的数据重新生成候选股票列表
            if prev_date_quote.empty:
                logger.warning(
                    f"补齐跳过: {date.date()}, 信号日 {original_signal_date.date()}, "
                    f"上一交易日 {prev_date.date()} 无行情数据，无法生成候选"
                )
                continue
            
            # 获取 D-1 日的股票池
            stock_universe = self.universe.get_stocks(prev_date, quote_data=prev_date_quote)
            
            # 调用扩展点获取 D-1 日的额外数据
            extra_data = self._build_signal_data(prev_date)
            if extra_data is None:
                logger.warning(
                        f"补齐跳过: {date.date()}, 信号日 {original_signal_date.date()}, "
                        f"上一交易日 {prev_date.date()} 无可用数据（_build_signal_data 返回 None）"
                    )
                continue
            
            # 合并默认数据和额外数据
            signal_data = {}
            signal_data.update(extra_data)
            
            # 使用 D-1 日的数据重新生成排序候选列表
            new_ranked_candidates = self.signal.generate_ranked(prev_date, stock_universe, signal_data)
            
            if not new_ranked_candidates:
                logger.warning(
                    f"补齐跳过: {date.date()}, 信号日 {original_signal_date.date()}, "
                    f"基于 {prev_date.date()} 数据无候选股票"
                )
                continue
            
            # 从新的候选列表中选择可用股票，排除已持仓股票
            # 关键约束：只选择 unfilled_count 数量的候选（模拟实盘开盘前固定下单数量）
            unfilled_count = len(unfilled_slot_weights)
            stocks_to_try = []
            for stock, score in new_ranked_candidates:
                if stock not in self.positions:
                    stocks_to_try.append((stock, score))
                    # 限制候选数量为未成交数量
                    if len(stocks_to_try) >= unfilled_count:
                        break
            
            if not stocks_to_try:
                logger.warning(
                    f"补齐跳过: {date.date()}, 信号日 {original_signal_date.date()}, "
                    f"基于 {prev_date.date()} 数据生成的候选均已持仓"
                )
                continue
            
            # 尝试按槽位补齐
            bought_stocks = []
            current_value = self._calculate_portfolio_value(date)
            remaining_unfilled_slots = []
            bought_stock_set = set()  # 跟踪已买入的股票，避免重复买入
            
            # 逐个槽位尝试补齐
            for slot_weight_info in unfilled_slot_weights:
                original_stock = slot_weight_info['stock']
                weight = slot_weight_info['weight']
                
                # 尝试从有限的候选列表中买入（按顺序）
                bought_for_this_slot = False
                
                for stock, score in stocks_to_try:
                    # 跳过已买入的股票
                    if stock in bought_stock_set:
                        continue
                    
                    # 检查是否可交易（在当日 D）
                    tradeable, reason = is_tradeable(stock, trade_date_str, date_quote, action='buy')
                    
                    if not tradeable:
                        logger.info(
                            f"补齐失败: {date.date()} (基于 {prev_date.date()} 数据), "
                            f"槽位 {original_stock} (权重 {weight:.4f}) 尝试买入股票 {stock} 失败, 原因: {reason}"
                        )
                        continue
                    
                    # 可交易，尝试买入
                    target_value = current_value * weight
                    self._buy_stock(date, stock, target_value, signal_date=original_signal_date)
                    
                    # 检查是否买入成功
                    if stock in self.positions:
                        bought_stocks.append(stock)
                        bought_stock_set.add(stock)  # 记录已买入
                        bought_for_this_slot = True
                        
                        self.completion_stats['total_completed'] += 1
                        
                        logger.info(
                            f"补齐成功: {date.date()} (基于 {prev_date.date()} 数据), "
                            f"槽位 {original_stock} (权重 {weight:.4f}) 买入股票 {stock} 成功."
                            f"信号日 {original_signal_date.date()}, 目标市值 {target_value:.2f}, "
                            f"候选池大小 {len(stocks_to_try)-len(bought_stock_set)}/{unfilled_count}"
                        )
                        
                        break
                
                # 如果该槽位未能补齐，保留到下次（会在下次重新生成有限候选继续尝试）
                if not bought_for_this_slot:
                    remaining_unfilled_slots.append(slot_weight_info)
                    logger.info(
                        f"补齐延迟: {date.date()}, 槽位 {original_stock} (权重 {weight:.4f}) "
                        f"在有限候选池 {len(stocks_to_try)} 只中未找到可买入股票，保留到下次"
                    )
            
            # 更新槽位信息
            slot_info['attempts'] += 1
            slot_info['unfilled_slot_weights'] = remaining_unfilled_slots
            self.completion_stats['completion_attempts'] += 1
            
            # 如果已经全部补齐，从待补齐列表中移除
            if not remaining_unfilled_slots:
                completed_signal_dates.append(signal_date)
                logger.info(
                    f"补齐完成: {date.date()}, 信号日 {original_signal_date.date()}, "
                    f"本次补齐 {len(bought_stocks)} 只，仓位已满"
                )
        
        # 清理已完成或放弃的槽位
        for signal_date in completed_signal_dates:
            del self.unfilled_slots[signal_date]
    
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
            signal_date = info.get('signal_date', buy_date)
            buy_idx = date_to_idx.get(buy_date)
            signal_idx = date_to_idx.get(signal_date)
            
            # 优先使用信号日作为持有期起点，确保延迟成交的仓位与原批次同时卖出
            anchor_idx = signal_idx if signal_idx is not None else buy_idx
            if anchor_idx is None:
                logger.warning(f"股票 {stock} 买入/信号日期 {buy_date}/{signal_date} 不在交易日映射中")
                continue
            
            # 计算持有天数（交易日）
            holding_days = current_idx - anchor_idx
            
            # 达到持有期，执行卖出
            if holding_days >= self.holding_period:
                stocks_to_sell.append(stock)

        if stocks_to_sell and self.verbose:
            logger.info(f"卖出执行: {date.date()}, 尝试卖出 {len(stocks_to_sell)} 只股票（达到持有期）")

        # 执行卖出
        for stock in stocks_to_sell:
            self._sell_stock(date, stock, sell_type='holding_period')
        
    
    def _check_stop_loss(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict) -> None:
        """检查止损触发条件（T 日检查，生成 T+1 卖出信号）
        
        Args:
            date: 当前日期（检查日）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.stop_loss_monitor:
            return
        
        # 遍历所有持仓检查止损
        for stock, info in list(self.positions.items()):
            # 如果该股票已经在待止损卖出队列中，跳过（避免重复触发）
            if stock in self.pending_stop_loss_sells:
                continue
            
            # 获取当前价格
            current_price = self._get_trade_price(date, stock)
            if current_price is None:
                continue
            
            buy_price = info['buy_trade_price']
            
            # 获取当日行情数据判断是否跌停
            trade_date_str = to_trade_date_str(date)
            date_quote = self.price_data_cache[self.price_data_cache['trade_date'] == trade_date_str]
            is_limit_down = False
            if not date_quote.empty:
                stock_quote = date_quote[date_quote['ts_code'] == stock]
                if not stock_quote.empty and 'is_limit_down' in stock_quote.columns:
                    is_limit_down = stock_quote['is_limit_down'].iloc[0]
            
            # 检查止损触发
            triggered, trigger_type, reason = self.stop_loss_monitor.check_stop_loss(
                stock=stock,
                buy_price=buy_price,
                current_price=current_price,
                is_limit_down=is_limit_down
            )
            
            if triggered:
                # 止损触发，记录到待卖出队列（T+1 执行）
                self.pending_stop_loss_sells[stock] = {
                    'trigger_date': date,
                    'reason': reason,
                    'trigger_type': trigger_type.value if trigger_type else 'unknown'
                }
                
                if self.verbose:
                    logger.warning(
                        f"止损触发: {date.date()} {stock}, 原因: {reason}, "
                        f"将在下一交易日执行卖出"
                    )
    
    def _execute_pending_stop_loss_sells(self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict) -> None:
        """执行待止损卖出操作（T+1 日执行）
        
        Args:
            date: 当前日期（执行日，T+1）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.stop_loss_monitor or not self.pending_stop_loss_sells:
            return
        
        # 查找前一个交易日触发的止损
        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx == 0:
            return
        
        trigger_date = trading_dates[current_idx - 1]
        
        # 执行前一交易日触发的止损卖出
        stocks_to_sell = []
        for stock, info in list(self.pending_stop_loss_sells.items()):
            if info['trigger_date'] == trigger_date:
                stocks_to_sell.append((stock, info))
        
        if not stocks_to_sell:
            return
        
        # 执行卖出
        for stock, info in stocks_to_sell:
            # 检查股票是否还在持仓中（可能已被正常调仓卖出）
            if stock not in self.positions:
                # 从待卖出队列中移除
                self.pending_stop_loss_sells.pop(stock, None)
                continue
            
            # 执行止损卖出
            self._sell_stock(
                date, 
                stock, 
                sell_type='stop_loss',
                sell_reason=info['reason'],
                trigger_type=info['trigger_type']
            )
            
            # 从待卖出队列中移除
            self.pending_stop_loss_sells.pop(stock, None)
        
        if stocks_to_sell and self.verbose:
            logger.info(
                f"止损卖出执行: {date.date()}, 卖出 {len(stocks_to_sell)} 只股票 "
                f"（触发日: {trigger_date.date()}）"
            )
    
    def _prepare_price_index(self, price_data: pd.DataFrame) -> None:
        """准备价格索引（使用 MultiIndex，替代嵌套字典）
        
        构建四套价格序列：
        - trade_price_index: 收盘成交价格（不复权 close）
        - pnl_price_index: 收盘绩效价格（后复权 close_adj）
        - trade_price_open_index: 开盘成交价格（不复权 open）
        - pnl_price_open_index: 开盘绩效价格（后复权 open_adj）
        
        Args:
            price_data: 价格数据，需包含 ts_code, trade_date, close, open（可选），close_adj（可选），open_adj（可选）
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
        
        # 构建收盘成交价格索引（不复权 close）
        trade_price_df = price_data[['trade_date', 'ts_code', 'close']].copy()
        trade_price_df.set_index(['trade_date', 'ts_code'], inplace=True)
        self.trade_price_index = trade_price_df['close']
        
        # 构建收盘绩效价格索引（后复权 close_adj）
        if 'close_adj' in price_data.columns:
            pnl_price_df = price_data[['trade_date', 'ts_code', 'close_adj']].copy()
            pnl_price_df.set_index(['trade_date', 'ts_code'], inplace=True)
            self.pnl_price_index = pnl_price_df['close_adj']
            logger.info("价格索引构建完成: 收盘成交价格=close, 收盘绩效价格=close_adj")
        else:
            # 如果缺少 close_adj，回退到 close
            logger.warning(f"价格数据缺少 'close_adj' 列，绩效价格将使用 'close' 列（不复权）")
            self.pnl_price_index = self.trade_price_index.copy()
            logger.info("价格索引构建完成: 收盘成交价格=close, 收盘绩效价格=close（退化）")
        
        # 构建开盘成交价格索引（不复权 open）
        if 'open' in price_data.columns:
            # 过滤掉NaN值，只保留有效的开盘价
            open_data = price_data[['trade_date', 'ts_code', 'open']].copy()
            open_data = open_data[open_data['open'].notna()]
            
            if len(open_data) > 0:
                open_data.set_index(['trade_date', 'ts_code'], inplace=True)
                self.trade_price_open_index = open_data['open']
                logger.info(f"开盘价格索引构建完成: 开盘成交价格=open, 共{len(open_data)}条记录")
            else:
                logger.warning(f"价格数据的 'open' 列全部为NaN，开盘价格将使用收盘价格代替")
                self.trade_price_open_index = self.trade_price_index.copy()
        else:
            logger.warning(f"价格数据缺少 'open' 列，开盘价格将使用收盘价格代替")
            self.trade_price_open_index = self.trade_price_index.copy()
        
        # 构建开盘绩效价格索引（后复权 open_adj）
        if 'open_adj' in price_data.columns:
            # 过滤掉NaN值，只保留有效的开盘绩效价格
            open_adj_data = price_data[['trade_date', 'ts_code', 'open_adj']].copy()
            open_adj_data = open_adj_data[open_adj_data['open_adj'].notna()]
            
            if len(open_adj_data) > 0:
                open_adj_data.set_index(['trade_date', 'ts_code'], inplace=True)
                self.pnl_price_open_index = open_adj_data['open_adj']
                logger.info(f"开盘绩效价格索引构建完成: 开盘绩效价格=open_adj, 共{len(open_adj_data)}条记录")
            else:
                # 如果open_adj全部为NaN，尝试使用open
                if 'open' in price_data.columns:
                    logger.warning(f"价格数据的 'open_adj' 列全部为NaN，开盘绩效价格将使用 'open' 列（不复权）")
                    self.pnl_price_open_index = self.trade_price_open_index.copy()
                else:
                    logger.warning(f"价格数据缺少 'open' 和 'open_adj' 列，开盘绩效价格将使用收盘绩效价格代替")
                    self.pnl_price_open_index = self.pnl_price_index.copy()
        else:
            # 如果缺少 open_adj，回退到 open 或 close_adj
            if 'open' in price_data.columns:
                # 如果有 open 但没有 open_adj，使用 open
                logger.warning(f"价格数据缺少 'open_adj' 列，开盘绩效价格将使用 'open' 列（不复权）")
                self.pnl_price_open_index = self.trade_price_open_index.copy()
            else:
                # 如果连 open 都没有，使用 close_adj
                logger.warning(f"价格数据缺少 'open_adj' 列，开盘绩效价格将使用收盘绩效价格代替")
                self.pnl_price_open_index = self.pnl_price_index.copy()
    
    def _get_trade_price(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取收盘成交价格（不复权 close）
        
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
        """获取收盘绩效价格（后复权 close_adj）
        
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
    
    def _get_trade_price_open(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取开盘成交价格（不复权 open）
        
        如果开盘价格不存在，返回 None。调用者应处理降级策略（如使用收盘价）。
        
        Args:
            date: 日期
            stock: 股票代码
            
        Returns:
            开盘成交价格，如果不存在则返回 None
        """
        try:
            return self.trade_price_open_index.loc[(date, stock)]
        except KeyError:
            return None
    
    def _get_pnl_price_open(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取开盘绩效价格（后复权 open_adj）
        
        如果开盘绩效价格不存在，返回 None。调用者应处理降级策略（如使用收盘绩效价格）。
        
        Args:
            date: 日期
            stock: 股票代码
            
        Returns:
            开盘绩效价格，如果不存在则返回 None
        """
        try:
            return self.pnl_price_open_index.loc[(date, stock)]
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
        # 仅支持整数天数
        n = self.rebalance_freq
        if n <= 0:
            raise ValueError(f"调仓频率必须为正整数，当前值: {n}")
        return [trading_dates[i] for i in range(0, len(trading_dates), n)]
    
    
    
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
        trade_date_str = to_trade_date_str(date)
        date_quote = self.price_data_cache[self.price_data_cache['trade_date'] == trade_date_str]
        
        for order in orders_to_retry:
            # 检查是否可交易
            if date_quote.empty:
                # 当日行情数据为空，无法判断交易状态，继续延迟
                logger.warning(
                    f"延迟订单 {order.stock} 在 {date.date()} 无行情数据，继续延迟"
                )
                continue
            tradeable, reason = is_tradeable(
                order.stock, trade_date_str, date_quote, action=order.action
            )
            
            if tradeable:
                # 可交易，尝试执行
                if order.action == 'buy':
                    self._buy_stock_direct(
                        date, order.stock, order.target_value, signal_date=order.signal_date
                    )
                    self.pending_order_manager.mark_success(date, order.stock, 'buy')
                elif order.action == 'sell':
                    self._sell_stock_direct(date, order.stock)
                    self.pending_order_manager.mark_success(date, order.stock, 'sell')
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
            trade_date_str = to_trade_date_str(date)
            date_quote = self.price_data_cache[self.price_data_cache['trade_date'] == trade_date_str]
            if date_quote.empty:
                # 当日行情数据为空，无法判断交易状态，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action='buy',
                        current_date=date,
                        signal_date=signal_date or date,
                        target_value=target_value,
                        reason='无行情数据'
                    )
                if self.verbose:
                    logger.info(
                        f"买入延迟: {date.date()} {stock}, 原因: 无行情数据, "
                        f"目标市值: {target_value:.2f}"
                    )
                return
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
        self._buy_stock_direct(date, stock, target_value, signal_date=signal_date)
    
    def _buy_stock_direct(
        self, 
        date: pd.Timestamp, 
        stock: str, 
        target_value: float,
        signal_date: Optional[pd.Timestamp] = None
    ) -> None:
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
            'signal_date': signal_date or date,
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
        
        直接买入，不再进行交易状态检查，因为在信号生成阶段已经过滤了不可交易的股票。
        
        Args:
            date: 买入日期（T+1）
            stock: 股票代码
            target_value: 目标市值
            signal_date: 信号生成日期（保留参数以兼容，但不使用）
        """
        # 直接买入，不检查交易状态（已在信号生成时过滤）
        self._buy_stock_direct(date, stock, target_value, signal_date=signal_date)
    
    
    def _sell_stock(
        self, 
        date: pd.Timestamp, 
        stock: str,
        sell_type: str = 'holding_period',
        sell_reason: Optional[str] = None,
        trigger_type: Optional[str] = None
    ) -> None:
        """卖出股票（在 T+n 日以收盘价卖出）
        
        带交易状态检查的卖出方法。如果启用延迟订单功能，会检查股票是否可交易。
        
        Args:
            date: 卖出日期（T+n）
            stock: 股票代码
            sell_type: 卖出类型，'holding_period' 或 'stop_loss'
            sell_reason: 卖出原因描述（止损时使用）
            trigger_type: 触发类型（止损时使用）
        """
        self._sell_stock_with_status_check(date, stock, sell_type, sell_reason, trigger_type)
    
    def _sell_stock_with_status_check(
        self, 
        date: pd.Timestamp, 
        stock: str,
        sell_type: str = 'holding_period',
        sell_reason: Optional[str] = None,
        trigger_type: Optional[str] = None
    ) -> None:
        """卖出股票（带交易状态检查）
        
        如果启用延迟订单功能，会检查股票是否可交易（跌停）
        不可交易时加入延迟队列而非直接失败
        
        Args:
            date: 卖出日期
            stock: 股票代码
            sell_type: 卖出类型，'holding_period' 或 'stop_loss'
            sell_reason: 卖出原因描述（止损时使用）
            trigger_type: 触发类型（止损时使用）
        """
        # 检查交易状态
        if self.enable_pending_order and self.price_data_cache is not None:
            trade_date_str = to_trade_date_str(date)
            date_quote = self.price_data_cache[self.price_data_cache['trade_date'] == trade_date_str]
            if date_quote.empty:
                # 当日行情数据为空，无法判断交易状态，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action='sell',
                        current_date=date,
                        signal_date=date,  # 卖出是基于持有期，用当前日期
                        target_value=None,
                        reason='无行情数据'
                    )
                if self.verbose:
                    logger.info(f"卖出延迟: {date.date()} {stock}, 原因: 无行情数据")
                return
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
        self._sell_stock_direct(date, stock, sell_type, sell_reason, trigger_type)
    
    def _sell_stock_direct(
        self, 
        date: pd.Timestamp, 
        stock: str,
        sell_type: str = 'holding_period',
        sell_reason: Optional[str] = None,
        trigger_type: Optional[str] = None
    ) -> None:
        """直接卖出股票（不检查交易状态）
        
        现金流使用成交价格（trade_price）计算
        收益率使用绩效价格（pnl_price）计算
        
        根据 sell_timing 参数选择使用开盘价或收盘价：
        - sell_timing='close': 使用收盘价（默认）
        - sell_timing='open': 使用开盘价，如果开盘价不存在则降级到收盘价
        
        Args:
            date: 卖出日期（T+n）
            stock: 股票代码
            sell_type: 卖出类型，'holding_period' 或 'stop_loss'
            sell_reason: 卖出原因描述（止损时使用）
            trigger_type: 触发类型（止损时使用）
        """
        if stock not in self.positions or self.positions[stock]['shares'] == 0:
            return
        
        # 根据 sell_timing 参数选择价格
        if self.sell_timing == 'open':
            # 尝试使用开盘价
            sell_trade_price = self._get_trade_price_open(date, stock)
            sell_pnl_price = self._get_pnl_price_open(date, stock)
            
            # 降级策略：如果开盘价不存在，使用收盘价
            if sell_trade_price is None:
                if self.verbose:
                    logger.warning(
                        f"股票 {stock} 在 {date.date()} 缺少开盘成交价格，"
                        f"降级使用收盘价卖出"
                    )
                sell_trade_price = self._get_trade_price(date, stock)
                if sell_trade_price is None:
                    logger.warning(f"无法获取 {stock} 在 {date.date()} 的成交价格（开盘/收盘），跳过卖出")
                    return
            
            if sell_pnl_price is None:
                # 开盘绩效价格不存在，尝试降级到收盘绩效价格
                sell_pnl_price = self._get_pnl_price(date, stock)
                if sell_pnl_price is None:
                    logger.warning(f"无法获取 {stock} 在 {date.date()} 的绩效价格，使用成交价格代替")
                    sell_pnl_price = sell_trade_price
        else:
            # 使用收盘价（默认）
            sell_trade_price = self._get_trade_price(date, stock)
            if sell_trade_price is None:
                logger.warning(f"无法获取 {stock} 在 {date.date()} 的成交价格，跳过卖出")
                return
            
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
        
        # 如果是止损卖出，清理止损监控器中的持仓状态
        if sell_type == 'stop_loss' and self.stop_loss_monitor:
            self.stop_loss_monitor.remove_position(stock)
        
        # 记录交易（包含绩效收益信息和卖出类型）
        trade_record = {
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
            'pnl_profit_pct': pnl_profit_pct,  # 绩效收益率
            'sell_type': sell_type,  # 卖出类型
            'sell_timing': self.sell_timing  # 新增：卖出时机（open/close）
        }
        
        # 如果是止损卖出，添加止损相关信息
        if sell_type == 'stop_loss':
            trade_record['sell_reason'] = sell_reason
            trade_record['trigger_type'] = trigger_type
        
        self.trades.append(trade_record)
    
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
