"""回测引擎组装体（P2a-T5，行为冻结搬运件）。

来源：旧 ``backtest/engine.py`` 的 ``BacktestEngine``（T0 规划 §3.3）。本模块
是 7 个执行/记账 mixin 复制件的**唯一组装点**：MRO 保持旧清单相对顺序原样
（R6），``__init__`` 委托 ``core/accounting/state.py::init_engine_state``，
主类残部 23 个方法全部为 T4 函数级拆分件的薄委托包装（签名与旧方法逐字
一致）。

D6 摘除（exposure 政策族退役）：旧 MRO 中的 BacktestExposureOverrideMixin /
BacktestExposureTrimMixin / BacktestExposureReplenishMixin 不进新 MRO；
旧 ``__init__`` 中 5 行 exposure 状态（``exposure_table`` /
``exposure_stats`` / ``_exposure_missing_dates`` 赋值 + 2 个
``_init_exposure_*`` 调用）已在 ``init_engine_state`` 中摘除。旧签名本无
exposure 参数，天然 fail-fast（传 ``exposure_table=...`` 抛 TypeError，
访问 ``set_exposure_table`` 等抛 AttributeError）。

接线义务落实清单：
- ① ``_get_min_buy_value_threshold``：先短路 ``ratio <= 0 → 0.0``，仅在
  ratio > 0 时才求值 ``_calculate_portfolio_value(date)``（避免无谓触发
  ``last_known_price`` 缓存写回，污染估值路径）。
- ② ``__init__`` 传 ``pending_order_event_sink=self._record_pending_order_event``
  （禁止沿用默认 None）。
- ③ ``_snapshot_early_rebalance_state`` 取值时保留旧
  ``getattr(self, "_last_rebalance_nav", None)`` 防御。
- ④ 状态回写：``_get_suspend_calendar`` 回写 ``self._suspend_calendar`` 与
  ``self.data_storage``（组装体持有缓存）；``_normalize_signals`` 把
  ``normalize_signals`` 返回的计数回写 ``self._normalize_log_count``
  （旧方法内自增，T4 改为返回值；递增时机逐字保留）。

依赖处置（§3.5 登记表）：
- ``Universe`` 只读引用旧 ``universe.base``（永久沿用）；``DataLoader``
  只读过渡依赖已随 ``init_engine_state``（state.py）承载，本模块不再直接
  import（去除节点 P2b/P3）；
- ``Signal`` / ``CostModel`` / ``StopLossConfig`` 改指 v2 复制件
  （仅作 ``__init__`` 签名类型标注）。
"""

from typing import Dict, List, Optional

import pandas as pd

from src.lazybull.universe.base import Universe  # 只读引用，永久沿用（§3.5）
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.rules.stop_loss import StopLossConfig
from src.lazybull.v2.core.accounting.exports import (
    get_execution_attribution,
    get_trades,
)
from src.lazybull.v2.core.accounting.holdings_snapshot import (
    BacktestHoldingsSnapshotMixin,
)
from src.lazybull.v2.core.accounting.nav import (
    calculate_portfolio_value,
    generate_nav_curve,
    position_market_value,
)
from src.lazybull.v2.core.accounting.state import init_engine_state
from src.lazybull.v2.core.decision.params import (
    get_min_buy_value_threshold,
    get_target_position_count,
    get_tranche_capital_fraction,
    get_tranche_target_count,
)
from src.lazybull.v2.core.decision.weighting import (
    apply_risk_budget,
    calculate_volatility,
    estimate_stock_variance,
    kelly_weights,
    normalize_signals,
)
from src.lazybull.v2.core.execution.buy_execution import BacktestBuyExecutionMixin
from src.lazybull.v2.core.execution.engine_utils import (
    ensure_suspend_calendar,
    get_rebalance_dates,
    restore_early_rebalance_state,
    snapshot_early_rebalance_state,
)
from src.lazybull.v2.core.execution.pending_execution import (
    BacktestPendingExecutionMixin,
)
from src.lazybull.v2.core.execution.price_index import (
    PriceIndexes,
    get_pnl_price,
    get_pnl_price_open,
    get_trade_price,
    get_trade_price_open,
    prepare_price_index,
)
from src.lazybull.v2.core.execution.reporting import BacktestReportingMixin
from src.lazybull.v2.core.execution.run_loop import BacktestRunLoopMixin
from src.lazybull.v2.core.execution.sell_execution import BacktestSellExecutionMixin
from src.lazybull.v2.core.execution.signal_execution import (
    BacktestSignalExecutionMixin,
)
from src.lazybull.v2.core.signal.base import Signal


class BacktestEngine(
    BacktestReportingMixin,
    BacktestBuyExecutionMixin,
    BacktestSellExecutionMixin,
    BacktestSignalExecutionMixin,
    BacktestPendingExecutionMixin,
    BacktestHoldingsSnapshotMixin,
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
        # 状态初始化全量委托 init_engine_state（D6 已摘除 5 行 exposure 状态）；
        # 接线义务②：延迟订单事件回调接回 _record_pending_order_event（pending_execution mixin）
        init_engine_state(
            self,
            universe=universe,
            signal=signal,
            initial_capital=initial_capital,
            cost_model=cost_model,
            rebalance_freq=rebalance_freq,
            holding_period=holding_period,
            verbose=verbose,
            enable_risk_budget=enable_risk_budget,
            vol_window=vol_window,
            vol_epsilon=vol_epsilon,
            enable_pending_order=enable_pending_order,
            max_retry_count=max_retry_count,
            max_retry_days=max_retry_days,
            stop_loss_config=stop_loss_config,
            sell_timing=sell_timing,
            enable_position_completion=enable_position_completion,
            completion_window_days=completion_window_days,
            data_storage=data_storage,
            max_weight_per_stock=max_weight_per_stock,
            max_per_industry=max_per_industry,
            stock_basic=stock_basic,
            stagger_tranches=stagger_tranches,
            position_sizing=position_sizing,
            kelly_vol_window=kelly_vol_window,
            kelly_max_leverage=kelly_max_leverage,
            enable_early_rebalance_on_empty=enable_early_rebalance_on_empty,
            min_buy_value_ratio=min_buy_value_ratio,
            pending_order_event_sink=self._record_pending_order_event,
        )

    def _current_price_indexes(self) -> PriceIndexes:
        """由 4 个独立属性槽位现场组装 PriceIndexes（state.py 初始化，run 时填充）。"""
        return PriceIndexes(
            trade_price_index=self.trade_price_index,
            pnl_price_index=self.pnl_price_index,
            trade_price_open_index=self.trade_price_open_index,
            pnl_price_open_index=self.pnl_price_open_index,
        )

    def _get_suspend_calendar(self):
        """获取停牌日历实例（延迟创建，共用 common 构建函数）"""
        # 接线义务④：返回值回写 self._suspend_calendar 与 self.data_storage（组装体持有缓存）
        self._suspend_calendar, self.data_storage = ensure_suspend_calendar(
            self._suspend_calendar, self.data_storage
        )
        return self._suspend_calendar

    def _get_target_position_count(self) -> int:
        """获取组合当前期望的目标持仓数。"""
        return get_target_position_count(self.signal, self.positions)

    def _get_tranche_target_count(
        self, tranche_idx: int, target_count: Optional[int] = None
    ) -> int:
        """获取当前批次应占用的目标持仓槽位数（委托 common.rules.stagger 共享实现）。"""
        return get_tranche_target_count(
            tranche_idx, self.stagger_tranches, self.signal, self.positions, target_count
        )

    def _get_tranche_capital_fraction(self, tranche_idx: int) -> float:
        """获取当前批次占组合总资产的预算比例（委托 common.rules.stagger 共享实现）。"""
        return get_tranche_capital_fraction(
            tranche_idx, self.stagger_tranches, self.signal, self.positions
        )

    def _get_min_buy_value_threshold(self, date: pd.Timestamp) -> float:
        """计算最小买入后市值阈值（与纸面交易共用 core/decision/sizing 口径）。

        接线义务①：ratio <= 0 时先短路返回 0.0，不求值组合市值（避免触发
        ``last_known_price`` 缓存写回）。
        """
        ratio = float(self.min_buy_value_ratio or 0.0)
        if ratio <= 0:
            return 0.0
        return get_min_buy_value_threshold(
            ratio,
            float(self._calculate_portfolio_value(date)),
            int(self._get_target_position_count() or 0),
        )

    def _prepare_price_index(self, price_data: pd.DataFrame) -> None:
        """准备价格索引（使用 MultiIndex，替代嵌套字典）

        构建四套价格序列并拆包回写 4 个独立属性槽位：
        - trade_price_index: 收盘成交价格（不复权 close）
        - pnl_price_index: 收盘绩效价格（后复权 close_adj）
        - trade_price_open_index: 开盘成交价格（不复权 open）
        - pnl_price_open_index: 开盘绩效价格（后复权 open_adj）

        Args:
            price_data: 价格数据，需包含 ts_code, trade_date, close, open（可选），close_adj（可选），open_adj（可选）
        """
        indexes = prepare_price_index(price_data)
        self.trade_price_index = indexes.trade_price_index
        self.pnl_price_index = indexes.pnl_price_index
        self.trade_price_open_index = indexes.trade_price_open_index
        self.pnl_price_open_index = indexes.pnl_price_open_index

    def _get_trade_price(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取收盘成交价格（不复权 close）

        Args:
            date: 日期
            stock: 股票代码

        Returns:
            成交价格，如果不存在则返回 None
        """
        return get_trade_price(self._current_price_indexes(), date, stock)

    def _get_pnl_price(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取收盘绩效价格（后复权 close_adj）

        Args:
            date: 日期
            stock: 股票代码

        Returns:
            绩效价格，如果不存在则返回 None
        """
        return get_pnl_price(self._current_price_indexes(), date, stock)

    def _get_trade_price_open(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取开盘成交价格（不复权 open）

        如果开盘价格不存在，返回 None。调用者应处理降级策略（如使用收盘价）。

        Args:
            date: 日期
            stock: 股票代码

        Returns:
            开盘成交价格，如果不存在则返回 None
        """
        return get_trade_price_open(self._current_price_indexes(), date, stock)

    def _get_pnl_price_open(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """获取开盘绩效价格（后复权 open_adj）

        如果开盘绩效价格不存在，返回 None。调用者应处理降级策略（如使用收盘绩效价格）。

        Args:
            date: 日期
            stock: 股票代码

        Returns:
            开盘绩效价格，如果不存在则返回 None
        """
        return get_pnl_price_open(self._current_price_indexes(), date, stock)

    def _calculate_volatility(self, stock: str, end_date: pd.Timestamp) -> float:
        """计算个股历史波动率（基于绩效价格，避免未来函数）

        使用 end_date 之前的 vol_window 个交易日的收益率计算波动率

        Args:
            stock: 股票代码
            end_date: 结束日期（不包含，只使用该日期之前的数据）

        Returns:
            年化波动率
        """
        return calculate_volatility(
            self.pnl_price_index, stock, end_date, self.vol_window, self.vol_epsilon
        )

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
        return apply_risk_budget(
            signals, date, self.pnl_price_index, self.vol_window, self.vol_epsilon
        )

    def _get_rebalance_dates(self, trading_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, int]:
        """获取调仓日期及对应的 tranche 索引（委托 common.rules.stagger 共享实现）

        Args:
            trading_dates: 交易日列表

        Returns:
            字典 {日期: tranche_idx}。stagger_tranches=1 时所有日期的 tranche 均为 0。
        """
        return get_rebalance_dates(trading_dates, self.rebalance_freq, self.stagger_tranches)

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

        接线义务④：T4 函数改为返回更新后的日志计数，此处回写
        ``self._normalize_log_count``（旧方法内自增，递增时机逐字保留）。
        """
        weights, count = normalize_signals(
            signals,
            date,
            position_sizing=self.position_sizing,
            normalize_log_count=self._normalize_log_count,
            verbose=self.verbose,
            kelly_max_leverage=self.kelly_max_leverage,
            price_data_cache=self.price_data_cache,
            kelly_vol_window=self.kelly_vol_window,
        )
        self._normalize_log_count = count
        return weights

    def _kelly_weights(
        self,
        signals: Dict[str, float],
        date: pd.Timestamp,
        half: bool = False,
    ) -> Dict[str, float]:
        """计算 Kelly / 半 Kelly 仓位权重（委托 core/decision/sizing 共享实现）。"""
        return kelly_weights(
            signals,
            date,
            kelly_max_leverage=self.kelly_max_leverage,
            price_data_cache=self.price_data_cache,
            kelly_vol_window=self.kelly_vol_window,
            verbose=self.verbose,
            half=half,
        )

    def _estimate_stock_variance(self, stock: str, date: pd.Timestamp) -> Optional[float]:
        """估计股票近期收益率方差,供 Kelly 仓位计算使用

        从 price_data_cache 中取近 kelly_vol_window 日收盘价,计算日收益率方差。
        数据不足时返回 None。
        """
        return estimate_stock_variance(self.price_data_cache, stock, date, self.kelly_vol_window)

    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """计算组合市值（基于成交价格）

        Args:
            date: 计算日期

        Returns:
            组合总市值
        """
        return calculate_portfolio_value(
            self.positions, self.current_capital, self._current_price_indexes(), date
        )

    def _position_market_value(self, date: pd.Timestamp, stock: str) -> Optional[float]:
        """单只持仓的市值（价格口径与 `_calculate_portfolio_value` 完全一致：
        当日成交价 → 最后已知价 → 买入价兜底），非持仓或市值不可用返回 None。
        """
        return position_market_value(self.positions, self._current_price_indexes(), date, stock)

    def _generate_nav_curve(self) -> pd.DataFrame:
        """生成净值曲线

        Returns:
            净值曲线DataFrame
        """
        return generate_nav_curve(self.portfolio_values, self.initial_capital)

    def get_trades(self) -> pd.DataFrame:
        """获取交易记录

        Returns:
            交易记录DataFrame
        """
        return get_trades(self.trades)

    def get_execution_attribution(self) -> pd.DataFrame:
        """获取信号槽位到实际买入的归因记录。"""
        return get_execution_attribution(self.execution_attribution_records)

    # ── 提前调仓历史快照/回滚 ──

    def _snapshot_early_rebalance_state(self, date: pd.Timestamp) -> Dict:
        """快照提前调仓可能污染的状态，用于失败时回滚。

        接线义务③：``_last_rebalance_nav`` 取值保留旧
        ``getattr(self, "_last_rebalance_nav", None)`` 防御。
        """
        return snapshot_early_rebalance_state(
            self._last_ranked_candidates,
            self._last_signal_date,
            getattr(self, "_last_rebalance_nav", None),
        )

    def _restore_early_rebalance_state(self, date: pd.Timestamp, snapshot: Dict) -> None:
        """回滚提前调仓过程中修改的状态。

        回写约定（与旧方法逐字一致）：``_last_ranked_candidates`` /
        ``_last_signal_date`` 无条件回写；``_last_rebalance_nav`` 仅当快照
        值非 None 时回写（None 不回写）。
        """
        restored = restore_early_rebalance_state(snapshot)
        if "last_rebalance_nav" in restored:
            self._last_rebalance_nav = restored["last_rebalance_nav"]

        self._last_ranked_candidates = restored["last_ranked_candidates"]
        self._last_signal_date = restored["last_signal_date"]
