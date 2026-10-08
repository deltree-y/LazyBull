"""回测引擎状态初始化（P2a-T4，engine.py 函数级拆分件；D8 提取式拆分）。

来源：旧 ``backtest/engine.py`` ``BacktestEngine.__init__`` 的状态初始化
部分（T0 规划 §3.3）。旧 ``__init__`` 是复杂度基线存量超限键（行数 +
mccabe 20），按 D8「迁移即重构」做提取式拆分：主函数 ``init_engine_state``
仅留编排，各段提取为 helper；**操作顺序、默认值、校验异常类型与消息
逐字保留**。

D6 摘除：旧实现此处的 3 个 exposure mixin 关联初始化不迁——
``exposure_table`` / ``exposure_stats`` / ``_exposure_missing_dates``
状态字段与 ``_init_exposure_trim_state()`` / ``_init_exposure_replenish_state()``
2 处初始化调用（旧 engine.py:265-271）随退役组件一并摘除（B0 无系数表时
逐位等价；非默认值 fail-fast 由 T5 组装体兜底）。

依赖处置（§3.5 登记表）：
- ``CostModel`` 改指 v2 ``common.cost``；``StopLossConfig`` /
  ``StopLossMonitor`` 改指 v2 ``common.rules.stop_loss``；
  ``PendingOrderManager`` 改指 v2 ``core.execution.pending_order``（均 T1/T4 已迁）；
- ``Signal`` 改指 v2 ``core.signal.base``（T4 同批交付）；
- ``Universe`` 只读引用旧 ``universe.base``（永久沿用，§3.5）；
- ``DataLoader`` 只读过渡依赖旧 ``data.loader``（去除节点 P2b/P3 数据面切换，D2）；
- ``load_industry_mapping`` 改指 v2 ``core/decision/industry_constraint``
  （T3 已归位；原 T4 期的旧 ``portfolio`` 只读过渡依赖随之去除）。
"""

from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.lazybull.data.loader import DataLoader  # 只读过渡依赖（§3.5；去除节点 P2b/P3）
from src.lazybull.universe.base import Universe  # 永久沿用（§3.5）
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.rules.stop_loss import StopLossConfig, StopLossMonitor
from src.lazybull.v2.core.execution.pending_order import PendingOrderManager
from src.lazybull.v2.core.signal.base import Signal


def init_engine_state(
    engine: Any,
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
    data_storage=None,  # 数据存储实例（用于读取 raw/suspend 数据）
    max_weight_per_stock: Optional[float] = None,  # 单股最大权重
    max_per_industry: Optional[int] = None,  # 单行业最大持仓数量
    stock_basic: Optional[pd.DataFrame] = None,  # 股票基本信息（用于行业约束）
    stagger_tranches: int = 1,  # 分批调仓批次数（1=不分批）
    position_sizing: str = "equal",  # 仓位管理: equal|score|kelly|half_kelly
    kelly_vol_window: int = 60,  # Kelly 波动率估计窗口（交易日）
    kelly_max_leverage: float = 0.25,  # 单只股票 Kelly 仓位上限（占总资产）
    enable_early_rebalance_on_empty: bool = True,  # 空仓时是否提前触发新一轮调仓
    min_buy_value_ratio: float = 0.0,  # 买入后最小持仓市值占平均仓位市值比例（0=关闭）
    pending_order_event_sink: Optional[Callable[[Dict], None]] = None,  # 延迟订单事件回调
) -> None:
    """初始化回测引擎状态（旧 ``BacktestEngine.__init__`` 的状态初始化部分）。

    价格口径说明：
    - 成交价格（trade_price）：使用不复权 close/open，用于计算成交金额、持仓市值、可买入数量
    - 绩效价格（pnl_price）：使用后复权 close_adj/open_adj，用于计算收益率和绩效指标

    Args:
        engine: 目标引擎对象（组装体实例），属性直接写在其上
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
        position_sizing: 仓位管理模式 equal|score|kelly|half_kelly
        kelly_vol_window: Kelly 波动率估计窗口（交易日）
        kelly_max_leverage: 单只股票 Kelly 仓位上限（占总资产）
        enable_early_rebalance_on_empty: 空仓时是否提前触发新一轮调仓
        min_buy_value_ratio: 买入后最小持仓市值占“平均仓位市值”比例（0=关闭）。
            与纸面交易口径一致：阈值=总资产/目标持仓数*比例。
        pending_order_event_sink: 延迟订单事件回调（旧实现固定为
            ``self._record_pending_order_event``，由组装体接线）
    """
    _init_basics(engine, universe, signal, initial_capital, cost_model, data_storage)
    _init_constraints(engine, max_weight_per_stock, max_per_industry, stock_basic, verbose)
    _init_rebalance_timing(engine, rebalance_freq, sell_timing, verbose)
    _init_stagger(engine, stagger_tranches, rebalance_freq, signal, max_weight_per_stock)
    _init_early_rebalance_state(engine, enable_early_rebalance_on_empty)
    _init_risk_budget(engine, enable_risk_budget, vol_window, vol_epsilon)
    _init_pending_order(
        engine, enable_pending_order, max_retry_count, max_retry_days, pending_order_event_sink
    )
    _init_position_completion(engine, enable_position_completion, completion_window_days)
    _init_stop_loss(engine, stop_loss_config)
    _init_holding_period(engine, holding_period)
    _init_backtest_state(engine, initial_capital)
    _init_sizing(engine, position_sizing, kelly_vol_window, kelly_max_leverage, min_buy_value_ratio)
    _init_price_index_slots(engine)
    _log_init_summary(
        engine,
        initial_capital=initial_capital,
        enable_risk_budget=enable_risk_budget,
        enable_pending_order=enable_pending_order,
        enable_position_completion=enable_position_completion,
        completion_window_days=completion_window_days,
        stop_loss_config=stop_loss_config,
        enable_early_rebalance_on_empty=enable_early_rebalance_on_empty,
        verbose=verbose,
    )


def _init_basics(
    engine: Any,
    universe: Universe,
    signal: Signal,
    initial_capital: float,
    cost_model: Optional[CostModel],
    data_storage,
) -> None:
    """基础依赖与数据存储引用。"""
    engine.universe = universe
    engine.signal = signal
    engine.initial_capital = initial_capital
    engine.cost_model = cost_model or CostModel()
    engine.data_storage = data_storage  # 保存数据存储实例
    engine._suspend_calendar = None  # 停牌日历实例（延迟创建）


def _init_constraints(
    engine: Any,
    max_weight_per_stock: Optional[float],
    max_per_industry: Optional[int],
    stock_basic: Optional[pd.DataFrame],
    verbose: bool,
) -> None:
    """组合构建约束参数与校验（含行业映射构建）。"""
    # 组合构建约束参数
    engine.max_weight_per_stock = max_weight_per_stock
    engine.max_per_industry = (
        max_per_industry if max_per_industry and max_per_industry > 0 else None
    )
    engine.stock_basic = stock_basic
    engine.industry_mapping = None  # 延迟构建

    # 验证参数
    if max_weight_per_stock is not None:
        if max_weight_per_stock <= 0 or max_weight_per_stock > 1:
            raise ValueError(
                f"max_weight_per_stock 必须在 (0, 1] 范围内，当前值: {max_weight_per_stock}"
            )

    if engine.max_per_industry is not None:
        if stock_basic is None or stock_basic.empty:
            raise ValueError("启用行业约束时必须提供 stock_basic 数据")
        # 延迟导入以避免循环依赖
        # v2 core/decision/industry_constraint（T3 已归位改指）
        from src.lazybull.v2.core.decision.industry_constraint import load_industry_mapping

        loader = DataLoader(engine.data_storage)  # 使用数据存储实例创建加载器
        shenwan_industry = loader.load_shenwan_industry()
        engine.industry_mapping = load_industry_mapping(shenwan_industry, verbose=verbose)


def _init_rebalance_timing(
    engine: Any,
    rebalance_freq: int,
    sell_timing: str,
    verbose: bool,
) -> None:
    """调仓频率与卖出时机校验。"""
    # 验证调仓频率
    if not isinstance(rebalance_freq, int):
        raise TypeError(f"调仓频率必须为整数类型，当前类型: {type(rebalance_freq).__name__}")
    if rebalance_freq <= 0:
        raise ValueError(f"调仓频率必须为正整数，当前值: {rebalance_freq}")

    # 验证卖出时机参数
    if sell_timing not in ["close", "open"]:
        raise ValueError(f"卖出时机参数必须为 'close' 或 'open'，当前值: {sell_timing}")

    engine.rebalance_freq = rebalance_freq
    engine.sell_timing = sell_timing
    engine.verbose = verbose


def _init_stagger(
    engine: Any,
    stagger_tranches: int,
    rebalance_freq: int,
    signal: Signal,
    max_weight_per_stock: Optional[float],
) -> None:
    """分批调仓参数校验。"""
    # 分批调仓参数
    if not isinstance(stagger_tranches, int):
        raise TypeError(
            f"分批调仓批次数必须为整数类型，当前类型: {type(stagger_tranches).__name__}"
        )
    if stagger_tranches < 1:
        raise ValueError(f"分批调仓批次数必须 >= 1，当前值: {stagger_tranches}")
    if stagger_tranches > rebalance_freq:
        raise ValueError(
            "分批调仓批次数不能超过调仓频率，" f"当前值: {stagger_tranches} > {rebalance_freq}"
        )
    configured_top_n = getattr(signal, "top_n", None)
    if isinstance(configured_top_n, int) and configured_top_n > 0:
        if stagger_tranches > configured_top_n:
            raise ValueError(
                "分批调仓批次数不能超过目标持仓数，"
                f"当前值: {stagger_tranches} > {configured_top_n}"
            )
        if max_weight_per_stock is not None and max_weight_per_stock * configured_top_n < 1 - 1e-12:
            raise ValueError(
                "max_weight_per_stock 与目标持仓数无法构成满仓组合，需满足 "
                "max_weight_per_stock * top_n >= 1，当前值: "
                f"{max_weight_per_stock} * {configured_top_n}"
            )
    engine.stagger_tranches = stagger_tranches


def _init_early_rebalance_state(engine: Any, enable_early_rebalance_on_empty: bool) -> None:
    """空仓提前调仓开关与最近一次调仓状态。"""
    engine.enable_early_rebalance_on_empty = enable_early_rebalance_on_empty
    engine._last_ranked_candidates: list = []  # 最近一次调仓的候选排序列表（止盈补位用）
    engine._last_signal_date: Optional[pd.Timestamp] = None  # 最近一次调仓日期
    engine._last_rebalance_nav: Optional[float] = (
        None  # 上次调仓日组合净值（止盈基准 & 本调仓收益计算）
    )


def _init_risk_budget(
    engine: Any,
    enable_risk_budget: bool,
    vol_window: int,
    vol_epsilon: float,
) -> None:
    """风险预算参数。"""
    engine.enable_risk_budget = enable_risk_budget
    engine.vol_window = vol_window
    engine.vol_epsilon = vol_epsilon


def _init_pending_order(
    engine: Any,
    enable_pending_order: bool,
    max_retry_count: int,
    max_retry_days: int,
    event_sink: Optional[Callable[[Dict], None]],
) -> None:
    """延迟订单参数与管理器（事件回调由组装体接线）。"""
    engine.enable_pending_order = enable_pending_order
    engine.pending_order_manager = None
    if enable_pending_order:
        engine.pending_order_manager = PendingOrderManager(
            max_retry_count=max_retry_count,
            max_retry_days=max_retry_days,
            event_sink=event_sink,
        )


def _init_position_completion(
    engine: Any,
    enable_position_completion: bool,
    completion_window_days: int,
) -> None:
    """仓位补齐参数。"""
    engine.enable_position_completion = enable_position_completion
    engine.completion_window_days = completion_window_days


def _init_stop_loss(engine: Any, stop_loss_config: Optional[StopLossConfig]) -> None:
    """止损配置与监控器。"""
    engine.stop_loss_config = stop_loss_config
    engine.stop_loss_monitor = None
    if stop_loss_config and stop_loss_config.enabled:
        engine.stop_loss_monitor = StopLossMonitor(stop_loss_config)


def _init_holding_period(engine: Any, holding_period: Optional[int]) -> None:
    """持有期逻辑：如果未指定，与调仓频率保持一致。"""
    if holding_period is None:
        engine.holding_period = engine.rebalance_freq
    else:
        engine.holding_period = holding_period  # 修复：应使用传入的 holding_period


def _init_backtest_state(engine: Any, initial_capital: float) -> None:
    """回测运行状态容器（D6：exposure 关联字段与 2 处初始化调用已摘除）。"""
    # 回测状态
    engine.current_capital = initial_capital
    engine.positions: Dict[str, Dict] = (
        {}
    )  # {股票代码: {shares, buy_date, buy_trade_price, buy_pnl_price, buy_cost_cash}}
    engine.pending_signals: Dict[pd.Timestamp, Dict] = {}  # {信号日期: {股票: 权重}}
    engine.pending_stop_loss_sells: Dict[str, Dict] = (
        {}
    )  # {股票代码: {trigger_date, reason, trigger_type}} 待止损卖出队列
    engine.pending_condition_sells: Dict[str, Dict] = (
        {}
    )  # {股票代码: {trigger_date, sell_type}} 待条件卖出队列（T0 触发、T1 执行）
    engine._cycle_anchor_idx: int = 0  # 当前调仓周期起点 idx（用于 cycle_day 日志显示）
    engine.portfolio_values: List[Dict] = []  # 组合价值历史
    engine.trades: List[Dict] = []  # 交易记录
    engine.execution_attribution_records: List[Dict] = []  # 信号槽位到实际成交的旁路记录
    # 政策旁路（terminal_loss P2-1）：逐日持仓快照，只读记录不参与决策
    engine.holdings_snapshots: List[Dict] = []
    engine.record_holdings_snapshot: bool = False

    # 仓位补齐状态跟踪
    # {调仓日期: {未成交股票列表, 目标数量, 候选列表, 剩余权重字典}}
    engine.unfilled_slots: Dict[pd.Timestamp, Dict] = {}
    # 补齐统计
    engine.completion_stats = {
        "total_unfilled": 0,  # 累计未满仓次数
        "total_completed": 0,  # 累计补齐成功次数
        "total_abandoned": 0,  # 累计放弃补齐次数
        "completion_attempts": 0,  # 累计补齐尝试次数
    }
    engine._deferred_day_logs: List[Dict[str, str]] = []
    engine._daily_warning_items: Dict[str, List[Dict]] = {}


def _init_sizing(
    engine: Any,
    position_sizing: str,
    kelly_vol_window: int,
    kelly_max_leverage: float,
    min_buy_value_ratio: float,
) -> None:
    """Kelly 仓位管理参数与最小买入市值阈值校验。"""
    # Kelly 仓位管理参数
    if position_sizing not in ("equal", "score", "kelly", "half_kelly"):
        raise ValueError(
            f"position_sizing 必须为 equal|score|kelly|half_kelly，" f"当前值: {position_sizing}"
        )
    engine.position_sizing = position_sizing
    engine.kelly_vol_window = kelly_vol_window
    engine.kelly_max_leverage = kelly_max_leverage
    if min_buy_value_ratio < 0:
        raise ValueError(f"min_buy_value_ratio 必须 >= 0，当前值: {min_buy_value_ratio}")
    engine.min_buy_value_ratio = float(min_buy_value_ratio)
    engine._normalize_log_count = 0  # 权重诊断日志计数，只打印前5次
    if position_sizing in ("kelly", "half_kelly"):
        logger.info(
            f"仓位管理模式={position_sizing}, 波动率窗口={kelly_vol_window}, "
            f"单股上限={kelly_max_leverage:.2f}"
        )


def _init_price_index_slots(engine: Any) -> None:
    """价格索引槽位（在 run 时初始化）与价格数据缓存。"""
    # 价格索引（在 run 时初始化）
    engine.trade_price_index: Optional[pd.Series] = None  # 成交价格（不复权 close）
    engine.pnl_price_index: Optional[pd.Series] = None  # 绩效价格（后复权 close_adj）
    engine.trade_price_open_index: Optional[pd.Series] = None  # 开盘成交价格（不复权 open）
    engine.pnl_price_open_index: Optional[pd.Series] = None  # 开盘绩效价格（后复权 open_adj）

    # 存储价格数据用于交易状态检查
    engine.price_data_cache: Optional[pd.DataFrame] = None


def _log_init_summary(
    engine: Any,
    initial_capital: float,
    enable_risk_budget: bool,
    enable_pending_order: bool,
    enable_position_completion: bool,
    completion_window_days: int,
    stop_loss_config: Optional[StopLossConfig],
    enable_early_rebalance_on_empty: bool,
    verbose: bool,
) -> None:
    """初始化完成摘要日志（文案与旧实现逐字一致）。"""
    stagger_info = f", 分批调仓={engine.stagger_tranches}批" if engine.stagger_tranches > 1 else ""
    min_buy_text = (
        "关闭" if engine.min_buy_value_ratio <= 0 else f"{engine.min_buy_value_ratio:.2f}"
    )
    logger.info(
        f"回测引擎初始化完成: 初始资金={initial_capital}, "
        f"调仓频率={engine.rebalance_freq}, 持有期={engine.holding_period}天{stagger_info}, "
        f"卖出时机={engine.sell_timing}, "
        f"风险预算={'启用' if enable_risk_budget else '禁用'}, "
        f"延迟订单={'启用' if enable_pending_order else '禁用'}, "
        f"仓位补齐={'启用' if enable_position_completion else '禁用'}, "
        f"最小买入阈值={min_buy_text}, "
        f"补齐窗口={completion_window_days}天, "
        f"止损功能={'启用' if (stop_loss_config and stop_loss_config.enabled) else '禁用'}, "
        f"空仓提前调仓={'启用' if enable_early_rebalance_on_empty else '禁用'}, "
        f"详细日志={'开启' if verbose else '关闭'}"
    )
    sell_price_type = "开盘价" if engine.sell_timing == "open" else "收盘价"
    logger.info(
        f"交易规则: T日生成信号 -> T+1日收盘价买入 -> "
        f"持有{max(1, engine.holding_period - 1)}天后T0生成卖出信号 -> "
        f"下一交易日{sell_price_type}卖出（执行日恰为持有期满当天）"
    )
    logger.info("价格口径: 成交使用不复权 close/open, 绩效使用后复权 close_adj/open_adj")
