"""回测引擎"""

import bisect
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ..common.cost import CostModel
from ..common.date_utils import to_trade_date_str
from ..data.loader import DataLoader
from ..execution.pending_order import PendingOrderManager
from ..risk.stop_loss import StopLossConfig, StopLossMonitor
from ..signals.base import Signal
from ..trading.sizing import (
    compute_kelly_weights,
    compute_min_buy_value_threshold,
    estimate_variance_from_prices,
)
from ..trading.stagger import (
    compute_tranche_schedule,
)
from ..trading.stagger import get_tranche_capital_fraction as _shared_tranche_capital_fraction
from ..trading.stagger import get_tranche_target_count as _shared_tranche_target_count
from ..universe.base import Universe
from .buy_execution import BacktestBuyExecutionMixin
from .pending_execution import BacktestPendingExecutionMixin
from .reporting import BacktestReportingMixin, _format_rebalance_decision_summary
from .run_loop import BacktestRunLoopMixin
from .sell_execution import BacktestSellExecutionMixin
from .signal_execution import BacktestSignalExecutionMixin


class BacktestEngine(
    BacktestReportingMixin,
    BacktestBuyExecutionMixin,
    BacktestSellExecutionMixin,
    BacktestSignalExecutionMixin,
    BacktestPendingExecutionMixin,
    BacktestRunLoopMixin,
):
    """回测引擎

    执行回测流程，生成净值曲线和交易记录

    交易规则：
    - T 日生成信号
    - T+1 日收盘价买入
    - 持有期到期卖出：T+n 日开盘价卖出（n 为持有期）
    - 条件卖出（亏损提前换出、整体止盈）：Tn 日检查 → Tn+1 日开盘价卖出
    - 卖出时机可配置：开盘价（默认）或收盘价
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
        sell_timing: str = "open",
        enable_position_completion: bool = True,
        completion_window_days: int = 3,
        data_storage=None,  # 新增：数据存储实例（用于读取 raw/suspend 数据）
        max_weight_per_stock: Optional[float] = None,  # 新增：单股最大权重
        max_per_industry: Optional[int] = None,  # 新增：单行业最大持仓数量
        stock_basic: Optional[pd.DataFrame] = None,  # 新增：股票基本信息（用于行业约束）
        stagger_tranches: int = 1,  # 分批调仓批次数（1=不分批）
        position_sizing: str = "equal",  # 仓位管理: equal|score|kelly|half_kelly
        kelly_vol_window: int = 60,  # Kelly 波动率估计窗口（交易日）
        kelly_max_leverage: float = 0.25,  # 单只股票 Kelly 仓位上限（占总资产）
        enable_early_rebalance_on_empty: bool = True,  # 空仓时是否提前触发新一轮调仓
        min_buy_value_ratio: float = 0.0,  # 买入后最小持仓市值占平均仓位市值比例（0=关闭）
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
            sell_timing: 卖出时机，'open' 表示开盘价卖出（默认），'close' 表示收盘价卖出
            enable_position_completion: 是否启用仓位补齐功能，默认True
            completion_window_days: 补齐窗口期（交易日），默认3天
            data_storage: 数据存储实例（用于读取 raw/suspend 数据），如不提供则在需要时创建
            max_weight_per_stock: 单个股票最大权重（0-1），None 表示不启用限权，启用后会在信号生成时对权重进行限制并归一化
            max_per_industry: 单个行业最大持仓数量，None 或 0 表示不启用行业约束
            stock_basic: 股票基本信息 DataFrame（用于行业约束），必须包含 ts_code 和 industry 列
            stagger_tranches: 分批调仓批次数，默认1（不分批）。设为K时将资金分成K份，
                每份错开 rebalance_freq/K 天调仓，降低单次调仓时点风险
            min_buy_value_ratio: 买入后最小持仓市值占“平均仓位市值”比例（0=关闭）。
                与纸面交易口径一致：阈值=总资产/目标持仓数*比例。
        """
        self.universe = universe
        self.signal = signal
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.data_storage = data_storage  # 保存数据存储实例
        self._suspend_calendar = None  # 停牌日历实例（延迟创建）

        # 组合构建约束参数
        self.max_weight_per_stock = max_weight_per_stock
        self.max_per_industry = (
            max_per_industry if max_per_industry and max_per_industry > 0 else None
        )
        self.stock_basic = stock_basic
        self.industry_mapping = None  # 延迟构建

        # 验证参数
        if max_weight_per_stock is not None:
            if max_weight_per_stock <= 0 or max_weight_per_stock > 1:
                raise ValueError(
                    f"max_weight_per_stock 必须在 (0, 1] 范围内，当前值: {max_weight_per_stock}"
                )

        if self.max_per_industry is not None:
            if stock_basic is None or stock_basic.empty:
                raise ValueError("启用行业约束时必须提供 stock_basic 数据")
            # 延迟导入以避免循环依赖
            from ..portfolio import load_industry_mapping

            loader = DataLoader(self.data_storage)  # 使用数据存储实例创建加载器
            shenwan_industry = loader.load_shenwan_industry()
            self.industry_mapping = load_industry_mapping(shenwan_industry, verbose=verbose)

        # 验证调仓频率
        if not isinstance(rebalance_freq, int):
            raise TypeError(f"调仓频率必须为整数类型，当前类型: {type(rebalance_freq).__name__}")
        if rebalance_freq <= 0:
            raise ValueError(f"调仓频率必须为正整数，当前值: {rebalance_freq}")

        # 验证卖出时机参数
        if sell_timing not in ["close", "open"]:
            raise ValueError(f"卖出时机参数必须为 'close' 或 'open'，当前值: {sell_timing}")

        self.rebalance_freq = rebalance_freq
        self.sell_timing = sell_timing
        self.verbose = verbose

        # 分批调仓参数
        if stagger_tranches < 1:
            raise ValueError(f"分批调仓批次数必须 >= 1，当前值: {stagger_tranches}")
        self.stagger_tranches = stagger_tranches

        self.enable_early_rebalance_on_empty = enable_early_rebalance_on_empty
        self._last_ranked_candidates: list = []  # 最近一次调仓的候选排序列表（止盈补位用）
        self._last_signal_date: Optional[pd.Timestamp] = None  # 最近一次调仓日期
        self._last_rebalance_nav: Optional[float] = (
            None  # 上次调仓日组合净值（止盈基准 & 本调仓收益计算）
        )

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
                max_retry_days=max_retry_days,
                event_sink=self._record_pending_order_event,
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
        self.positions: Dict[str, Dict] = (
            {}
        )  # {股票代码: {shares, buy_date, buy_trade_price, buy_pnl_price, buy_cost_cash}}
        self.pending_signals: Dict[pd.Timestamp, Dict] = {}  # {信号日期: {股票: 权重}}
        self.pending_stop_loss_sells: Dict[str, Dict] = (
            {}
        )  # {股票代码: {trigger_date, reason, trigger_type}} 待止损卖出队列
        self.pending_condition_sells: Dict[str, Dict] = (
            {}
        )  # {股票代码: {trigger_date, sell_type}} 待条件卖出队列（T0 触发、T1 执行）
        self._cycle_anchor_idx: int = 0  # 当前调仓周期起点 idx（用于 cycle_day 日志显示）
        self.portfolio_values: List[Dict] = []  # 组合价值历史
        self.trades: List[Dict] = []  # 交易记录
        self.execution_attribution_records: List[Dict] = []  # 信号槽位到实际成交的旁路记录

        # 仓位补齐状态跟踪
        # {调仓日期: {未成交股票列表, 目标数量, 候选列表, 剩余权重字典}}
        self.unfilled_slots: Dict[pd.Timestamp, Dict] = {}
        # 补齐统计
        self.completion_stats = {
            "total_unfilled": 0,  # 累计未满仓次数
            "total_completed": 0,  # 累计补齐成功次数
            "total_abandoned": 0,  # 累计放弃补齐次数
            "completion_attempts": 0,  # 累计补齐尝试次数
        }
        self._deferred_day_logs: List[Dict[str, str]] = []
        self._daily_warning_items: Dict[str, List[Dict]] = {}

        # Kelly 仓位管理参数
        if position_sizing not in ("equal", "score", "kelly", "half_kelly"):
            raise ValueError(
                f"position_sizing 必须为 equal|score|kelly|half_kelly，"
                f"当前值: {position_sizing}"
            )
        self.position_sizing = position_sizing
        self.kelly_vol_window = kelly_vol_window
        self.kelly_max_leverage = kelly_max_leverage
        if min_buy_value_ratio < 0:
            raise ValueError(f"min_buy_value_ratio 必须 >= 0，当前值: {min_buy_value_ratio}")
        self.min_buy_value_ratio = float(min_buy_value_ratio)
        self._normalize_log_count = 0  # 权重诊断日志计数，只打印前5次
        if position_sizing in ("kelly", "half_kelly"):
            logger.info(
                f"仓位管理模式={position_sizing}, 波动率窗口={kelly_vol_window}, "
                f"单股上限={kelly_max_leverage:.2f}"
            )

        # 价格索引（在 run 时初始化）
        self.trade_price_index: Optional[pd.Series] = None  # 成交价格（不复权 close）
        self.pnl_price_index: Optional[pd.Series] = None  # 绩效价格（后复权 close_adj）
        self.trade_price_open_index: Optional[pd.Series] = None  # 开盘成交价格（不复权 open）
        self.pnl_price_open_index: Optional[pd.Series] = None  # 开盘绩效价格（后复权 open_adj）

        # 存储价格数据用于交易状态检查
        self.price_data_cache: Optional[pd.DataFrame] = None

        stagger_info = f", 分批调仓={self.stagger_tranches}批" if self.stagger_tranches > 1 else ""
        logger.info(
            f"回测引擎初始化完成: 初始资金={initial_capital}, "
            f"调仓频率={self.rebalance_freq}, 持有期={self.holding_period}天{stagger_info}, "
            f"卖出时机={self.sell_timing}, "
            f"风险预算={'启用' if enable_risk_budget else '禁用'}, "
            f"延迟订单={'启用' if enable_pending_order else '禁用'}, "
            f"仓位补齐={'启用' if enable_position_completion else '禁用'}, "
            f"最小买入阈值={'关闭' if self.min_buy_value_ratio <= 0 else f'{self.min_buy_value_ratio:.2f}'}, "
            f"补齐窗口={completion_window_days}天, "
            f"止损功能={'启用' if (stop_loss_config and stop_loss_config.enabled) else '禁用'}, "
            f"空仓提前调仓={'启用' if enable_early_rebalance_on_empty else '禁用'}, "
            f"详细日志={'开启' if verbose else '关闭'}"
        )
        sell_price_type = "开盘价" if self.sell_timing == "open" else "收盘价"
        logger.info(
            f"交易规则: T日生成信号 -> T+1日收盘价买入 -> 满{self.holding_period}天后T0生成卖出信号 -> 下一交易日{sell_price_type}卖出"
        )
        logger.info(f"价格口径: 成交使用不复权 close/open, 绩效使用后复权 close_adj/open_adj")

    def _get_suspend_calendar(self):
        """获取停牌日历实例（延迟创建）"""
        if self._suspend_calendar is None:
            from ..common.suspend_calendar import SuspendCalendar
            from ..data import Storage

            # 如果没有提供 data_storage，创建一个默认实例
            if self.data_storage is None:
                self.data_storage = Storage()

            self._suspend_calendar = SuspendCalendar(self.data_storage)

        return self._suspend_calendar

    def _get_target_position_count(self) -> int:
        """获取组合当前期望的目标持仓数。"""
        target_n = getattr(self.signal, "top_n", None)
        if isinstance(target_n, int) and target_n > 0:
            return target_n
        return len(self.positions)

    def _get_tranche_target_count(
        self, tranche_idx: int, target_count: Optional[int] = None
    ) -> int:
        """获取当前批次应占用的目标持仓槽位数（委托 trading.stagger 共享实现）。"""
        if target_count is None:
            target_count = self._get_target_position_count()
        return _shared_tranche_target_count(tranche_idx, target_count, self.stagger_tranches)

    def _get_tranche_capital_fraction(self, tranche_idx: int) -> float:
        """获取当前批次占组合总资产的预算比例（委托 trading.stagger 共享实现）。"""
        target_count = self._get_target_position_count()
        return _shared_tranche_capital_fraction(tranche_idx, target_count, self.stagger_tranches)

    def _get_min_buy_value_threshold(self, date: pd.Timestamp) -> float:
        """计算最小买入后市值阈值（与纸面交易共用 trading.sizing 口径）。"""
        ratio = float(self.min_buy_value_ratio or 0.0)
        if ratio <= 0:
            return 0.0
        return compute_min_buy_value_threshold(
            total_assets=float(self._calculate_portfolio_value(date)),
            target_count=int(self._get_target_position_count() or 0),
            ratio=ratio,
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
        if "close" not in price_data.columns:
            raise ValueError("价格数据缺少 'close' 列，无法进行回测")

        # 转换日期列为 datetime（向量化操作，避免 iterrows）
        if not pd.api.types.is_datetime64_any_dtype(price_data["trade_date"]):
            # 创建副本以避免修改原始数据
            price_data = price_data.copy()
            price_data["trade_date"] = pd.to_datetime(price_data["trade_date"])

        # 构建收盘成交价格索引（不复权 close）
        trade_price_df = price_data[["trade_date", "ts_code", "close"]].copy()
        trade_price_df.set_index(["trade_date", "ts_code"], inplace=True)
        self.trade_price_index = trade_price_df["close"]

        # 构建收盘绩效价格索引（后复权 close_adj）
        if "close_adj" in price_data.columns:
            pnl_price_df = price_data[["trade_date", "ts_code", "close_adj"]].copy()
            pnl_price_df.set_index(["trade_date", "ts_code"], inplace=True)
            self.pnl_price_index = pnl_price_df["close_adj"]
            logger.info("价格索引构建完成: 收盘成交价格=close, 收盘绩效价格=close_adj")
        else:
            # 如果缺少 close_adj，回退到 close
            logger.warning(f"价格数据缺少 'close_adj' 列，绩效价格将使用 'close' 列（不复权）")
            self.pnl_price_index = self.trade_price_index.copy()
            logger.info("价格索引构建完成: 收盘成交价格=close, 收盘绩效价格=close（退化）")

        # 构建开盘成交价格索引（不复权 open）
        if "open" in price_data.columns:
            # 过滤掉NaN值，只保留有效的开盘价
            open_data = price_data[["trade_date", "ts_code", "open"]].copy()
            open_data = open_data[open_data["open"].notna()]

            if len(open_data) > 0:
                open_data.set_index(["trade_date", "ts_code"], inplace=True)
                self.trade_price_open_index = open_data["open"]
                logger.info(f"开盘价格索引构建完成: 开盘成交价格=open, 共{len(open_data)}条记录")
            else:
                logger.warning(f"价格数据的 'open' 列全部为NaN，开盘价格将使用收盘价格代替")
                self.trade_price_open_index = self.trade_price_index.copy()
        else:
            logger.warning(f"价格数据缺少 'open' 列，开盘价格将使用收盘价格代替")
            self.trade_price_open_index = self.trade_price_index.copy()

        # 构建开盘绩效价格索引（后复权 open_adj）
        if "open_adj" in price_data.columns:
            # 过滤掉NaN值，只保留有效的开盘绩效价格
            open_adj_data = price_data[["trade_date", "ts_code", "open_adj"]].copy()
            open_adj_data = open_adj_data[open_adj_data["open_adj"].notna()]

            if len(open_adj_data) > 0:
                open_adj_data.set_index(["trade_date", "ts_code"], inplace=True)
                self.pnl_price_open_index = open_adj_data["open_adj"]
                logger.info(
                    f"开盘绩效价格索引构建完成: 开盘绩效价格=open_adj, 共{len(open_adj_data)}条记录"
                )
            else:
                # 如果open_adj全部为NaN，尝试使用open
                if "open" in price_data.columns:
                    logger.warning(
                        f"价格数据的 'open_adj' 列全部为NaN，开盘绩效价格将使用 'open' 列（不复权）"
                    )
                    self.pnl_price_open_index = self.trade_price_open_index.copy()
                else:
                    logger.warning(
                        f"价格数据缺少 'open' 和 'open_adj' 列，开盘绩效价格将使用收盘绩效价格代替"
                    )
                    self.pnl_price_open_index = self.pnl_price_index.copy()
        else:
            # 如果缺少 open_adj，回退到 open 或 close_adj
            if "open" in price_data.columns:
                # 如果有 open 但没有 open_adj，使用 open
                logger.warning(
                    f"价格数据缺少 'open_adj' 列，开盘绩效价格将使用 'open' 列（不复权）"
                )
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
            stock_prices = self.pnl_price_index.xs(stock, level="ts_code").sort_index()

            # 筛选 end_date 之前的数据
            stock_prices = stock_prices[stock_prices.index < end_date]

            if len(stock_prices) < 2:
                return self.vol_epsilon

            # 取最近 vol_window 个交易日
            recent_prices = stock_prices.iloc[-self.vol_window :]

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

    def _get_rebalance_dates(self, trading_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, int]:
        """获取调仓日期及对应的 tranche 索引（委托 trading.stagger 共享实现）

        Args:
            trading_dates: 交易日列表

        Returns:
            字典 {日期: tranche_idx}。stagger_tranches=1 时所有日期的 tranche 均为 0。
        """
        return compute_tranche_schedule(trading_dates, self.rebalance_freq, self.stagger_tranches)

    def _normalize_signals(self, signals: Dict[str, float], date: pd.Timestamp) -> Dict[str, float]:
        """将分数字典归一化为权重字典

        支持 4 种模式（由 self.position_sizing 控制）:
        - equal: 等权
        - score: 按预测分数线性加权
        - kelly: Kelly 最优仓位（基于分数 × 波动率估计）
        - half_kelly: 半 Kelly（Kelly 仓位的 50%，更保守）

        Kelly 公式: f* = μ / σ², 其中:
        - μ: 预期超额收益（用 ML 分数代理）
        - σ²: 收益率方差（从近期价格数据估计）
        最终 clip 到 [0, kelly_max_leverage], 再归一化总和为 1.0。
        """
        if not signals:
            return {}

        sizing = self.position_sizing

        if sizing == "equal":
            weight = 1.0 / len(signals)
            if self._normalize_log_count < 5:
                scores = list(signals.values())
                s_max, s_min = max(scores), min(scores)
                logger.info(
                    f"  [权重诊断 {self._normalize_log_count + 1}/5] equal, "
                    f"n={len(signals)}, 每只权重={weight:.4f}, "
                    f"分数范围=[{s_min:.4f}, {s_max:.4f}], 分数差距={s_max - s_min:.4f}"
                )
                self._normalize_log_count += 1
            return {stock: weight for stock in signals.keys()}

        elif sizing == "score":
            total_score = sum(signals.values())
            if total_score > 0:
                result = {stock: score / total_score for stock, score in signals.items()}
                if self._normalize_log_count < 5:
                    weights = sorted(result.values(), reverse=True)
                    w_max, w_min = weights[0], weights[-1]
                    eq_weight = 1.0 / len(result)
                    concentration = w_max / eq_weight
                    sample_stocks = sorted(result.items(), key=lambda x: x[1], reverse=True)[:3]
                    weights_str = ", ".join(
                        [f"{stock}: {weight:.4f}" for stock, weight in sample_stocks]
                    )
                    logger.info(
                        f"  [权重诊断 {self._normalize_log_count + 1}/5] score, "
                        f"n={len(result)}, 等权基准={eq_weight:.4f}, "
                        f"最高={w_max:.4f}({concentration:.1f}x), 最低={w_min:.4f}, "
                        f"示例（前3）: {weights_str}"
                    )
                    self._normalize_log_count += 1
                return result
            else:
                weight = 1.0 / len(signals)
                if self.verbose:
                    logger.warning(f"所有分数 <= 0，回退到等权分配，每只股票权重 {weight:.4f}")
                return {stock: weight for stock in signals.keys()}

        elif sizing in ("kelly", "half_kelly"):
            return self._kelly_weights(signals, date, half=(sizing == "half_kelly"))

        else:
            weight = 1.0 / len(signals)
            return {stock: weight for stock in signals.keys()}

    def _kelly_weights(
        self,
        signals: Dict[str, float],
        date: pd.Timestamp,
        half: bool = False,
    ) -> Dict[str, float]:
        """计算 Kelly / 半 Kelly 仓位权重（委托 trading.sizing 共享实现）。"""
        result, fallback_count = compute_kelly_weights(
            signals,
            variance_fn=lambda stock: self._estimate_stock_variance(stock, date),
            half=half,
            max_leverage=self.kelly_max_leverage,
        )

        if self.verbose and result:
            mode_name = "half_kelly" if half else "kelly"
            sample = list(result.items())[:3]
            weights_str = ", ".join([f"{s}: {w:.4f}" for s, w in sample])
            logger.info(
                f"  权重方法: {mode_name}, 示例权重（前3只）: {weights_str}, "
                f"fallback={fallback_count}只"
            )
        return result

    def _estimate_stock_variance(self, stock: str, date: pd.Timestamp) -> Optional[float]:
        """估计股票近期收益率方差,供 Kelly 仓位计算使用

        从 price_data_cache 中取近 kelly_vol_window 日收盘价,计算日收益率方差。
        数据不足时返回 None。
        """
        if self.price_data_cache is None:
            return None

        date_str = to_trade_date_str(date)
        stock_data = self.price_data_cache[
            (self.price_data_cache["ts_code"] == stock)
            & (self.price_data_cache["trade_date"] <= date_str)
        ]

        if len(stock_data) < 20:
            return None

        # 取最近 kelly_vol_window 条
        stock_data = stock_data.sort_values("trade_date").tail(self.kelly_vol_window)

        # 优先使用后复权价格，方差口径统一由 trading.sizing 计算
        price_col = "close_adj" if "close_adj" in stock_data.columns else "close"
        return estimate_variance_from_prices(stock_data[price_col].values.astype(float))

    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """计算组合市值（基于成交价格）

        Args:
            date: 计算日期

        Returns:
            组合总市值
        """
        market_value = 0.0

        for stock, info in self.positions.items():
            shares = info["shares"]
            trade_price = self._get_trade_price(date, stock)
            if trade_price is None:
                # 股票当日无价格（可能已退市/停牌），使用仓位中缓存的最后已知价格
                # 避免市值突降为 0 导致净值曲线出现虚假跳水
                trade_price = info.get("last_known_price")
                if trade_price is None:
                    # 兜底：使用买入价
                    trade_price = info.get("buy_trade_price", 0.0)
                    if trade_price > 0:
                        logger.warning(
                            f"股票 {stock} 在 {date.date()} 无价格数据，"
                            f"用买入价 {trade_price:.2f} 估值（可能已退市）"
                        )
            else:
                # 更新最后已知价格缓存
                info["last_known_price"] = trade_price

            market_value += shares * trade_price

        return self.current_capital + market_value

    def _generate_nav_curve(self) -> pd.DataFrame:
        """生成净值曲线

        Returns:
            净值曲线DataFrame
        """
        df = pd.DataFrame(self.portfolio_values)
        df["nav"] = df["portfolio_value"] / self.initial_capital
        df["return"] = df["nav"] - 1.0
        return df

    def get_trades(self) -> pd.DataFrame:
        """获取交易记录

        Returns:
            交易记录DataFrame
        """
        return pd.DataFrame(self.trades)

    def get_execution_attribution(self) -> pd.DataFrame:
        """获取信号槽位到实际买入的归因记录。"""
        return pd.DataFrame(self.execution_attribution_records)

    # ── 提前调仓历史快照/回滚 ──

    def _snapshot_early_rebalance_state(self, date: pd.Timestamp) -> Dict:
        """快照提前调仓可能污染的状态，用于失败时回滚。"""
        snapshot = {
            "last_ranked_candidates": list(self._last_ranked_candidates),
            "last_signal_date": self._last_signal_date,
            "last_rebalance_nav": getattr(self, "_last_rebalance_nav", None),
        }
        return snapshot

    def _restore_early_rebalance_state(self, date: pd.Timestamp, snapshot: Dict) -> None:
        """回滚提前调仓过程中修改的状态。"""
        if snapshot["last_rebalance_nav"] is not None:
            self._last_rebalance_nav = snapshot["last_rebalance_nav"]

        self._last_ranked_candidates = snapshot["last_ranked_candidates"]
        self._last_signal_date = snapshot["last_signal_date"]
