"""纸面交易公共运行时。"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from ..common.signal_factory import create_signal
from ..common.trading_config import TradingConfig
from ..data import DataLoader
from ..risk.equity_curve import EquityCurveMonitor, create_equity_curve_config_from_dict
from ..risk.stop_loss import StopLossConfig, StopLossMonitor
from ..risk.stop_loss_checker import check_positions_stop_loss
from .models import PendingBuy, PendingSell, TargetWeight, TradeInstruction, normalize_trade_reason
from .runner import PaperTradingRunner
from .storage import PaperStorage


@dataclass
class PaperTradeRuntimeContext:
    """纸面交易运行时上下文。"""

    storage: PaperStorage
    config: Dict[str, object]
    trading_config: TradingConfig
    runner: PaperTradingRunner


@dataclass
class PaperTradeExecutionResult:
    """纸面交易日执行结果。"""

    requested_date: str
    corrected_date: str
    storage: PaperStorage
    config: Dict[str, object]
    trading_config: TradingConfig
    runner: PaperTradingRunner
    stop_loss_actions: List[Dict[str, object]] = field(default_factory=list)
    early_exit_actions: List[Dict[str, object]] = field(default_factory=list)
    take_profit_actions: List[Dict[str, object]] = field(default_factory=list)
    pending_sell_actions: List[Dict[str, object]] = field(default_factory=list)
    t1_actions: List[Dict[str, object]] = field(default_factory=list)
    t0_targets: List[Dict[str, object]] = field(default_factory=list)
    t0_instructions: List[TradeInstruction] = field(default_factory=list)
    ect_exposure: float = 1.0
    ect_reason: str = "ECT 未启用"
    t0_status: str = "not_rebalance_day"
    protected_stocks: List[str] = field(default_factory=list)
    stock_names: Dict[str, str] = field(default_factory=dict)
    missing_factors: List[str] = field(default_factory=list)
    feature_error_detail: str = ""


def create_paper_trade_runtime(
    model_version_override: Optional[int] = None,
) -> PaperTradeRuntimeContext:
    """创建纸面交易公共运行时。"""
    storage = PaperStorage()
    config = storage.load_config()

    if config is None:
        raise RuntimeError(
            "未找到配置文件，请先编辑 data/paper/config.yaml 或运行 config 命令设置配置"
        )

    runtime_config = dict(config)
    if model_version_override is not None:
        runtime_config["model_version"] = model_version_override
    runtime_config.setdefault("horizon", 20)

    trading_config = TradingConfig.from_dict(runtime_config)
    signal = create_signal(trading_config)
    runner = PaperTradingRunner(
        signal=signal,
        initial_capital=float(runtime_config["initial_capital"]),
        position_sizing=str(runtime_config.get("position_sizing", "equal")),
        horizon=int(runtime_config["horizon"]),
    )
    return PaperTradeRuntimeContext(
        storage=storage,
        config=runtime_config,
        trading_config=trading_config,
        runner=runner,
    )


def execute_trade_workflow(
    trade_date: str,
    runtime: Optional[PaperTradeRuntimeContext] = None,
    model_version_override: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> PaperTradeExecutionResult:
    """执行纸面交易日工作流。"""

    def _report(step: str) -> None:
        if progress_callback is not None:
            progress_callback(step)

    context = runtime or create_paper_trade_runtime(model_version_override)
    storage = context.storage
    config = context.config
    trading_config = context.trading_config
    runner = context.runner

    _report("校正交易日")
    corrected_date = runner._correct_trade_date(trade_date)

    account_state = storage.load_account_state()
    if account_state and account_state.last_update and corrected_date < account_state.last_update:
        raise RuntimeError(
            f"日期回退：输入日期 {corrected_date} 早于账户最后更新日期 {account_state.last_update}，"
            "不允许回退执行"
        )

    stop_loss_config = trading_config.create_stop_loss_config() or StopLossConfig()
    stop_loss_monitor = StopLossMonitor(stop_loss_config)
    sl_state = storage.load_stop_loss_state()
    if sl_state:
        stop_loss_monitor.position_high_prices = sl_state.get("position_high_prices", {})
        stop_loss_monitor.consecutive_limit_down_days = sl_state.get(
            "consecutive_limit_down_days", {}
        )

    # T1 必须只执行上一交易日 T0 已落盘的指令，不能在执行日临时增删单。
    _report("执行 T1 指令")
    t1_actions = _execute_t1_if_pending(runner, corrected_date, config)

    # 恢复上一个 T0 生成的 ranked_candidates（如有），供后续 T0 规划使用
    _report("恢复排序候选")
    rc_loaded = storage.load_ranked_candidates()
    if isinstance(rc_loaded, tuple) and len(rc_loaded) == 2:
        ranked_candidates, signal_date = rc_loaded
        runner.signal._last_ranked_candidates = ranked_candidates
        runner.signal._last_signal_date = pd.Timestamp(signal_date)
        logger.info(f"已恢复 ranked_candidates: signal_date={signal_date}, count={len(ranked_candidates)}")
    elif rc_loaded:
        logger.warning(f"ranked_candidates 格式异常，跳过恢复: {type(rc_loaded)}")

    _report("止损检查")
    stop_loss_actions: List[Dict[str, object]] = []
    if bool(config["stop_loss_enabled"]):
        stop_loss_actions = _check_stop_loss(runner, stop_loss_monitor, corrected_date, config)
        storage.save_stop_loss_state(
            {
                "position_high_prices": stop_loss_monitor.position_high_prices,
                "consecutive_limit_down_days": stop_loss_monitor.consecutive_limit_down_days,
            }
        )
    else:
        logger.info("止损功能未启用，跳过")

    _report("亏损提前换出检查")
    early_exit_actions: List[Dict[str, object]] = []
    if bool(config.get("enable_profit_based_holding", False)):
        early_exit_actions = _check_early_exit(runner, corrected_date, config)
    else:
        logger.info("盈亏动态持仓未启用，跳过亏损提前换出")

    _report("整体止盈检查")
    take_profit_actions = _check_take_profit(runner, corrected_date, config)

    _report("规划次日卖出/补位指令")
    pending_sell_actions = _plan_next_day_retry_and_sell_instructions(
        runner=runner,
        trade_date=corrected_date,
        config=config,
        stop_loss_actions=stop_loss_actions,
        early_exit_actions=early_exit_actions,
        take_profit_actions=take_profit_actions,
    )

    # 持久化弱势退出监控状态（跨日连续计数）
    weakness_monitor = getattr(runner, "weakness_exit_monitor", None)
    if weakness_monitor is not None:
        storage.save_weakness_exit_state(weakness_monitor.get_state())

    _report("执行 T0")
    t0_targets, ect_exposure, ect_reason, t0_status, protected_stocks = _execute_t0_if_rebalance_day(
        runner, corrected_date, config
    )

    # 已将待重试队列转写为次日明确指令，避免 T1 再直接读取 pending_* 队列。
    if pending_sell_actions:
        runner.broker.pending_sells = []
        runner.broker.storage.save_pending_sells([])
    pending_buys = runner.paper_storage.load_pending_buys()
    if pending_buys:
        # 仅在当日已成功转写为明确指令后清空队列。
        t1_date = runner._get_next_trade_date(corrected_date)
        planned_next_day = runner.paper_storage.load_instructions(t1_date) if t1_date else []
        planned_buy_retries = [
            inst for inst in (planned_next_day or []) if inst.action == "buy" and inst.retry_attempt > 0
        ]
        if planned_buy_retries:
            runner.paper_storage.save_pending_buys([])

    # 仅在本日真实执行 T0 时保存 ranked_candidates，
    # 避免非调仓日因补位流程临时调用 generate_ranked 覆盖持久化候选池。
    # 注意：不再强依赖 _last_signal_date 类型/格式匹配，防止漏保存。
    if (
        t0_status == "success"
        and hasattr(runner.signal, "_last_ranked_candidates")
        and runner.signal._last_ranked_candidates
    ):
        logger.info("保存本轮 ranked_candidates 供下一个 T1 使用")
        storage.save_ranked_candidates(
            runner.signal._last_ranked_candidates,
            corrected_date,
        )

    _report("整理明日指令")
    t0_instructions: List[TradeInstruction] = []
    t1_date = runner._get_next_trade_date(corrected_date)
    if t1_date:
        t0_instructions = runner.paper_storage.load_instructions(t1_date) or []

    _report("加载股票名称")
    loader = DataLoader(runner.storage, verbose=False)
    stock_names = loader.build_stock_names_dict()
    storage.save_last_trade_date(corrected_date)

    return PaperTradeExecutionResult(
        requested_date=trade_date,
        corrected_date=corrected_date,
        storage=storage,
        config=config,
        trading_config=trading_config,
        runner=runner,
        stop_loss_actions=stop_loss_actions,
        early_exit_actions=early_exit_actions,
        take_profit_actions=take_profit_actions,
        pending_sell_actions=pending_sell_actions,
        t1_actions=t1_actions,
        t0_targets=t0_targets,
        t0_instructions=t0_instructions,
        ect_exposure=ect_exposure,
        ect_reason=ect_reason,
        t0_status=t0_status,
        protected_stocks=protected_stocks,
        stock_names=stock_names,
        missing_factors=list(runner.missing_factors),
        feature_error_detail=getattr(runner, "_last_feature_error", ""),
    )


def _check_stop_loss(
    runner: PaperTradingRunner,
    stop_loss_monitor: StopLossMonitor,
    trade_date: str,
    config: Dict[str, object],
) -> List[Dict[str, object]]:
    """检查止损触发。"""
    from ..common.suspend_calendar import SuspendCalendar

    del config
    actions: List[Dict[str, object]] = []
    positions = runner.account.get_positions()
    if not positions:
        logger.info("当前无持仓，跳过止损检查")
        return actions

    loader = DataLoader(runner.storage)
    daily_data = loader.load_clean_daily_by_date(trade_date)
    if daily_data is None or daily_data.empty:
        logger.warning(f"无法加载 {trade_date} 的价格数据，跳过止损检查")
        return actions

    prices = {}
    limit_down_info = {}
    for _, row in daily_data.iterrows():
        ts_code = row["ts_code"]
        prices[ts_code] = row.get("close", 0.0)
        limit_down_info[ts_code] = row.get("is_limit_down", 0) == 1

    suspend_calendar = SuspendCalendar(runner.storage)
    sl_actions = check_positions_stop_loss(
        positions=positions,
        stop_loss_monitor=stop_loss_monitor,
        prices=prices,
        limit_down_info=limit_down_info,
        suspend_calendar=suspend_calendar,
        trade_date=trade_date,
    )

    for sl_action in sl_actions:
        pos = positions.get(sl_action.ts_code)
        sell_shares = (pos.shares // 100) * 100 if pos else 0

        actions.append(
            {
                "ts_code": sl_action.ts_code,
                "shares": sell_shares,
                "reason": sl_action.reason,
                "can_execute": sl_action.can_execute,
            }
        )

    return actions


def _check_early_exit(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
) -> List[Dict[str, object]]:
    """检查亏损提前换出触发。"""
    actions = runner.evaluate_early_exit(trade_date, config)

    if not actions:
        logger.info("无持仓触发亏损提前换出")
        return actions

    logger.info(f"亏损提前换出检查完成：{len(actions)} 只股票触发")
    return actions


def _check_take_profit(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
) -> List[Dict[str, object]]:
    """检查整体止盈触发。"""
    threshold = config.get("take_profit_threshold")
    if threshold is None:
        logger.info("整体止盈未启用，跳过")
        return []

    positions = runner.account.get_positions()
    if not positions:
        logger.info("当前无持仓，跳过整体止盈检查")
        return []

    strategy_state = runner.paper_storage.load_strategy_state()
    last_rebalance_nav = strategy_state.get("last_rebalance_nav")
    if not last_rebalance_nav or last_rebalance_nav <= 0:
        logger.info("整体止盈基准净值缺失，跳过")
        return []

    buy_prices, sell_prices = runner._load_prices(
        trade_date,
        str(config["buy_price"]),
        str(config["sell_price"]),
    )
    all_prices = {**sell_prices, **buy_prices}
    if not all_prices:
        logger.warning("无法加载整体止盈所需价格数据，跳过")
        return []

    current_nav = runner.account.get_total_value(all_prices)
    profit_rate = (current_nav - last_rebalance_nav) / last_rebalance_nav
    if profit_rate < float(threshold):
        logger.info(f"整体止盈未触发: 本轮收益率={profit_rate:.2%}, 阈值={float(threshold):.2%}")
        return []

    existing_pending = set()
    actions = []
    for ts_code, pos in positions.items():
        if ts_code in existing_pending:
            continue

        sell_shares = (pos.shares // 100) * 100
        if sell_shares <= 0:
            continue

        reason = f"整体止盈: 本轮收益率={profit_rate:.2%} >= {float(threshold):.2%}"
        actions.append(
            {
                "ts_code": ts_code,
                "shares": sell_shares,
                "reason": reason,
                "can_execute": True,
            }
        )

    if not actions:
        logger.info("整体止盈无新增延迟卖出指令")
        return []

    strategy_state["pending_take_profit_trigger_date"] = trade_date
    if not bool(config.get("take_profit_refill", True)):
        strategy_state["take_profit_block_t0_date"] = runner._get_next_trade_date(trade_date)
    runner.paper_storage.save_strategy_state(strategy_state)
    logger.warning(
        f"整体止盈触发: 本轮收益率={profit_rate:.2%}, 已规划 {len(actions)} 条次日卖出指令"
    )
    return actions


def _merge_trade_instructions(
    existing: List[TradeInstruction],
    new_items: List[TradeInstruction],
) -> List[TradeInstruction]:
    """合并同一执行日的交易指令，按(action, ts_code)去重。"""
    merged: List[TradeInstruction] = []
    seen = set()
    for instruction in [*existing, *new_items]:
        key = (instruction.action, instruction.ts_code)
        if key in seen:
            continue
        seen.add(key)
        merged.append(instruction)
    return merged


def _save_next_day_instructions(
    runner: PaperTradingRunner,
    trade_date: str,
    instructions: List[TradeInstruction],
) -> None:
    """将 T0 规划结果写入下一交易日指令文件。"""
    if not instructions:
        return

    t1_date = runner._get_next_trade_date(trade_date)
    if not t1_date:
        logger.warning(f"{trade_date} 无下一交易日，跳过写入次日指令")
        return

    existing = runner.paper_storage.load_instructions(t1_date) or []
    merged = _merge_trade_instructions(existing, instructions)
    runner.paper_storage.save_instructions(t1_date, merged)


def _build_sell_instructions(
    actions: List[Dict[str, object]],
    trade_date: str,
    config: Dict[str, object],
    retry_attempt: int = 0,
) -> List[TradeInstruction]:
    """将卖出动作转为次日明确卖出指令。"""
    instructions: List[TradeInstruction] = []
    for action in actions:
        shares = int(action.get("shares", 0) or 0)
        if shares <= 0:
            continue
        instructions.append(
            TradeInstruction(
                ts_code=str(action["ts_code"]),
                action="sell",
                shares=shares,
                price_type=str(config["sell_price"]),
                reason=str(action["reason"]),
                source_date=trade_date,
                target_weight=0.0,
                retry_attempt=retry_attempt,
            )
        )
    return instructions


def _plan_pending_buy_retry_instructions(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
) -> List[TradeInstruction]:
    """将失败买入槽位在 T0 具体化为下一交易日买入指令。"""
    pending_buys = runner.paper_storage.load_pending_buys()
    if not pending_buys:
        return []

    failed_count = len(pending_buys)
    replacement_targets = runner.generate_replacement_targets(
        trade_date=trade_date,
        failed_count=failed_count,
        universe_type=str(config.get("universe", "mainboard")),
        model_version=config.get("model_version"),
        buy_price_type=str(config["buy_price"]),
        original_signal_date=str(getattr(pending_buys[0], "original_signal_date", trade_date) or trade_date),
        max_per_industry=config.get("max_per_industry"),
        exclude_st=bool(config.get("exclude_st", True)),
        min_list_days=int(config.get("min_list_days", 365)),
        trading_config=TradingConfig.from_dict(config),
    )
    if not replacement_targets:
        logger.info("无可用补位目标，保留 pending_buys 供后续 T0 继续规划")
        return []

    daily_data = runner.loader.load_clean_daily_by_date(trade_date)
    if daily_data is None or daily_data.empty:
        logger.warning(f"无法加载 {trade_date} 的价格数据，跳过补位指令规划")
        return []

    current_prices = {
        str(row["ts_code"]): float(row.get("close", 0.0) or 0.0)
        for _, row in daily_data.iterrows()
    }

    retry_attempt_by_code: Dict[str, int] = {}
    enriched_targets = []
    for target, pending_buy in zip(replacement_targets, pending_buys):
        target.target_weight = pending_buy.target_weight
        target.reason = normalize_trade_reason(
            pending_buy.reason,
            ensure_replenishment_prefix=True,
        )
        target.original_signal_date = pending_buy.original_signal_date or trade_date
        enriched_targets.append(target)
        retry_attempt_by_code[target.ts_code] = int(pending_buy.attempts)

    instructions = runner._generate_instructions(
        targets=enriched_targets,
        buy_price_type=str(config["buy_price"]),
        sell_price_type=str(config["sell_price"]),
        current_prices=current_prices,
        source_date=trade_date,
        desired_position_count=int(config.get("top_n", len(enriched_targets)) or len(enriched_targets)),
    )
    for instruction in instructions:
        instruction.retry_attempt = retry_attempt_by_code.get(instruction.ts_code, 0)

    if instructions:
        logger.info(f"已将 {len(instructions)} 个补位槽位具体化为次日买入指令")

    return instructions


def _plan_next_day_retry_and_sell_instructions(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
    stop_loss_actions: List[Dict[str, object]],
    early_exit_actions: List[Dict[str, object]],
    take_profit_actions: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """在 T0 统一规划下一交易日全部卖出/补位指令。"""
    planned_actions: List[Dict[str, object]] = []
    existing_sell_codes = set()
    instructions: List[TradeInstruction] = []

    retry_sell_actions: List[Dict[str, object]] = []
    for pending_sell in list(runner.broker.pending_sells):
        pos = runner.account.get_position(pending_sell.ts_code)
        if not pos or pos.shares <= 0:
            continue
        retry_sell_actions.append(
            {
                "ts_code": pending_sell.ts_code,
                "shares": min(int(pending_sell.shares), int(pos.shares)),
                "reason": str(pending_sell.reason),
                "status": "已转写为次日卖出指令",
            }
        )
        existing_sell_codes.add(pending_sell.ts_code)
        instructions.extend(
            _build_sell_instructions(
                [retry_sell_actions[-1]],
                trade_date=trade_date,
                config=config,
                retry_attempt=int(getattr(pending_sell, "attempts", 0) or 0),
            )
        )

    daily_sell_actions = [*stop_loss_actions, *early_exit_actions, *take_profit_actions]
    filtered_daily_actions = [
        action for action in daily_sell_actions if str(action.get("ts_code")) not in existing_sell_codes
    ]
    instructions.extend(_build_sell_instructions(filtered_daily_actions, trade_date, config))
    existing_sell_codes.update(str(action["ts_code"]) for action in filtered_daily_actions)

    holding_sell_actions: List[Dict[str, object]] = []
    # 持有期到期检查始终执行：enable_profit_based_holding=False 时仅做到期卖出，
    # enable_profit_based_holding=True 时才会评估盈利延续。
    _, holding_sell_actions = runner.evaluate_holding_period_actions(
        trade_date,
        config,
        exclude_stocks=existing_sell_codes,
    )
    instructions.extend(_build_sell_instructions(holding_sell_actions, trade_date, config))

    pending_buy_instructions = _plan_pending_buy_retry_instructions(runner, trade_date, config)
    instructions = _merge_trade_instructions(instructions, pending_buy_instructions)

    _save_next_day_instructions(runner, trade_date, instructions)

    planned_actions.extend(retry_sell_actions)
    planned_actions.extend(filtered_daily_actions)
    planned_actions.extend(holding_sell_actions)
    return planned_actions


def _process_pending_sells(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
) -> List[Dict[str, object]]:
    """处理延迟卖出队列。"""
    actions: List[Dict[str, object]] = []
    fills = runner.broker.retry_pending_sells(trade_date, str(config["sell_price"]))

    for pending_sell in runner.broker.pending_sells:
        actions.append(
            {
                "ts_code": pending_sell.ts_code,
                "shares": pending_sell.shares,
                "reason": pending_sell.reason,
                "status": f"不可卖出（尝试次数: {pending_sell.attempts}）",
            }
        )

    for fill in fills:
        actions.append(
            {
                "ts_code": fill.ts_code,
                "shares": fill.shares,
                "reason": fill.reason,
                "status": "已成交",
            }
        )

    if fills:
        runner.account.update_last_date(trade_date)
        runner.account.save_state()

        buy_prices, sell_prices = runner._load_prices(
            trade_date,
            str(config["buy_price"]),
            str(config["sell_price"]),
        )
        all_prices = {**sell_prices, **buy_prices}
        runner._record_nav(trade_date, all_prices)

        if any("整体止盈" in str(fill.reason or "") for fill in fills):
            strategy_state = runner.paper_storage.load_strategy_state()
            strategy_state["last_rebalance_nav"] = runner.account.get_total_value(all_prices)
            strategy_state["last_take_profit_date"] = trade_date
            strategy_state.pop("pending_take_profit_trigger_date", None)
            runner.paper_storage.save_strategy_state(strategy_state)

    logger.info(
        f"延迟卖出处理完成：成交 {len(fills)} 笔，剩余 {len(runner.broker.pending_sells)} 笔"
    )
    return actions


def _process_pending_buys(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
) -> List[Dict[str, object]]:
    """处理延迟买入队列。"""
    actions: List[Dict[str, object]] = []
    fills, remaining_buys = runner.broker.retry_pending_buys(trade_date, str(config["buy_price"]))

    for pending_buy in remaining_buys:
        actions.append(
            {
                "ts_code": pending_buy.ts_code,
                "target_weight": pending_buy.target_weight,
                "reason": pending_buy.reason,
                "status": f"不可买入（尝试次数: {pending_buy.attempts}/5）",
            }
        )

    for fill in fills:
        actions.append(
            {
                "ts_code": fill.ts_code,
                "target_weight": 0.0,
                "reason": fill.reason,
                "status": "已成交",
            }
        )

    if fills:
        runner.account.update_last_date(trade_date)
        runner.account.save_state()

        buy_prices, sell_prices = runner._load_prices(
            trade_date,
            str(config["buy_price"]),
            str(config["sell_price"]),
        )
        all_prices = {**sell_prices, **buy_prices}
        runner._record_nav(trade_date, all_prices)

    logger.info(f"延迟买入处理完成：成交 {len(fills)} 笔，剩余 {len(remaining_buys)} 笔")
    return actions


def _execute_t1_if_pending(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
) -> List[Dict[str, object]]:
    """执行 T1（仅执行 T0 已明确生成的交易指令）。"""
    actions: List[Dict[str, object]] = []

    if runner.paper_storage.check_run_exists("t1", trade_date):
        logger.info(f"T1 工作流已在 {trade_date} 执行过，跳过")
        return actions

    instructions = None
    inst_date = trade_date
    found = runner.paper_storage.find_pending_instructions(trade_date)
    if found:
        inst_date, instructions = found
        if inst_date != trade_date:
            source_date = instructions[0].source_date if instructions else inst_date
            try:
                trade_cal = runner.loader.load_clean_trade_cal()
                trade_dates_list = trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()
                src_idx = trade_dates_list.index(source_date)
                cur_idx = trade_dates_list.index(trade_date)
                gap = cur_idx - src_idx
                threshold = int(int(config["rebalance_freq"]) * 0.5)
                if gap >= threshold:
                    logger.warning(
                        f"发现 {inst_date} 的未执行指令（信号日 {source_date}），"
                        f"但距今已 {gap} 个交易日，超过阈值 {threshold}（rebalance_freq*0.5），"
                        "指令已过期，丢弃"
                    )
                    runner.paper_storage.save_run_record(
                        "t1",
                        inst_date,
                        {
                            "trade_date": inst_date,
                            "note": (f"指令过期丢弃（距信号日 {gap} 个交易日，阈值 {threshold}）"),
                            "expired": True,
                            "timestamp": pd.Timestamp.now().isoformat(),
                        },
                    )
                    instructions = None
                else:
                    logger.info(
                        f"发现 {inst_date} 的未执行指令（延迟 {gap} 个交易日），"
                        f"将在 {trade_date} 补充执行"
                    )
            except (ValueError, Exception) as exc:
                logger.warning(f"检查指令过期失败: {exc}，按原日期执行")

    if not instructions:
        logger.info(f"未找到 {trade_date} 的交易指令，跳过 T1")
        return actions

    if instructions:
        logger.info("=" * 80)
        logger.info(f"【T1 指令驱动】读取到 {len(instructions)} 条交易指令")
        logger.info("=" * 80)

    buy_prices, sell_prices = runner._load_prices(
        trade_date,
        str(config["buy_price"]),
        str(config["sell_price"]),
    )
    if not buy_prices and not sell_prices:
        logger.error("无法加载价格数据，跳过 T1")
        return actions

    fills_count = 0
    orders_count = 0
    logger.info("执行交易指令")
    fills = runner.broker.execute_instructions(
        instructions, buy_prices, sell_prices, trade_date
    )
    fills_count += len(fills) if fills else 0
    orders_count += len(instructions)
    for fill in fills:
        actions.append(
            {
                "ts_code": fill.ts_code,
                "action": fill.action,
                "shares": fill.shares,
                "reason": fill.reason,
            }
        )
    logger.info(f"指令执行完成：{len(instructions)} 条指令，{len(fills)} 笔成交")

    failed_buy_targets = runner.broker.get_failed_buy_targets()
    if failed_buy_targets:
        max_retry_attempt = max(
            [
                int(getattr(inst, "retry_attempt", 0) or 0)
                for inst in instructions
                if getattr(inst, "action", "") == "buy"
            ],
            default=0,
        )
        _handle_failed_buys(
            runner,
            trade_date,
            config,
            failed_buy_targets,
            attempt_count=max_retry_attempt,
        )

    if fills_count > 0:
        runner.account.update_last_date(trade_date)
        runner.account.save_state()
        all_prices = {**sell_prices, **buy_prices}
        runner._record_nav(trade_date, all_prices)

        if any("整体止盈" in str(fill.reason or "") for fill in fills):
            strategy_state = runner.paper_storage.load_strategy_state()
            strategy_state["last_rebalance_nav"] = runner.account.get_total_value(all_prices)
            strategy_state["last_take_profit_date"] = trade_date
            strategy_state.pop("pending_take_profit_trigger_date", None)
            runner.paper_storage.save_strategy_state(strategy_state)

    run_record = {
        "trade_date": trade_date,
        "buy_price_type": config["buy_price"],
        "sell_price_type": config["sell_price"],
        "instructions_count": len(instructions) if instructions else 0,
        "pending_buys_count": len(failed_buy_targets) if failed_buy_targets else 0,
        "orders_count": orders_count,
        "fills_count": fills_count,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    runner.paper_storage.save_run_record("t1", trade_date, run_record)
    if inst_date != trade_date and instructions:
        runner.paper_storage.save_run_record(
            "t1",
            inst_date,
            {**run_record, "note": f"指令延迟执行，实际执行日期 {trade_date}"},
        )

    logger.info(f"T1 执行完成：{len(actions)} 个订单")
    return actions


def _handle_failed_buys(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
    failed_buy_targets: List[TargetWeight],
    attempt_count: int,
) -> None:
    """处理买入失败并生成补位计划。

    对齐回测：补位队列保留"未成交槽位"的原始权重，
    不在 T1 预先重算替代股票，实际候选在下一交易日按 D-1 数据重算。
    """
    max_replenishment_attempts = 5

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"检测到 {len(failed_buy_targets)} 个买入失败目标")

    next_attempt = attempt_count + 1
    if next_attempt > max_replenishment_attempts:
        logger.warning(f"补位尝试次数已达上限 ({max_replenishment_attempts})，不再继续补位")
        logger.info("=" * 80)
        runner.broker.clear_failed_buy_targets()
        return

    logger.info(
        f"记录当日 {trade_date} 未成交槽位，下一交易日按回测口径执行补位（第 {next_attempt} 次尝试）"
    )
    logger.info("=" * 80)

    next_trade_date = runner._get_next_trade_date(trade_date)
    if next_trade_date:
        pending_buys = runner._build_pending_buys_from_failed_targets(
            failed_buy_targets,
            trade_date,
            attempts=next_attempt,
        )

        runner.paper_storage.save_pending_buys(pending_buys)
        logger.info(f"已保存 {len(pending_buys)} 个未成交槽位到补位队列（保留原始槽位权重）")
        logger.info(
            f"下一交易日 {next_trade_date} 将自动读取并执行补位买入（第 {next_attempt}/{max_replenishment_attempts} 次尝试）"
        )
        logger.info("补位买入不会触发现有持仓的卖出")
    else:
        logger.error("无法获取下一交易日，补位计划生成失败")

    runner.broker.clear_failed_buy_targets()


def _resolve_early_rebalance_context(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
    *,
    take_profit_block_t0_date: Optional[str],
) -> Tuple[bool, str, set]:
    """判断是否允许提前调仓，并返回触发模式与已确认的保护持仓。"""
    pending_instruction = runner.paper_storage.find_pending_instructions(trade_date)
    pending_buys = runner.paper_storage.load_pending_buys()
    current_positions = runner.account.get_positions()

    guards_ok = (
        bool(config.get("enable_early_rebalance_on_empty", True))
        and take_profit_block_t0_date != trade_date
        and not runner.broker.pending_sells
        and not pending_buys
        and pending_instruction is None
    )
    if not guards_ok:
        return False, "", set()

    if not current_positions:
        return True, "empty", set()

    if not bool(config.get("enable_profit_based_holding", False)):
        return False, "", set()
    if str(config.get("profit_extension_mode", "pnl")) == "disabled":
        return False, "", set()

    protected_stocks = set(runner.evaluate_profit_extension(trade_date, config))
    if protected_stocks:
        return True, "holding_tail", protected_stocks

    return False, "", set()


def _execute_t0_if_rebalance_day(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
) -> Tuple[List[Dict[str, object]], float, str, str, List[str]]:
    """执行 T0（如果是调仓日）。"""
    targets_info: List[Dict[str, object]] = []
    ect_exposure = 1.0
    ect_reason = "ECT 未启用"
    protected_stock_list: List[str] = []

    if runner.paper_storage.check_run_exists("t0", trade_date):
        logger.info(f"T0 工作流已在 {trade_date} 执行过，跳过")
        return targets_info, ect_exposure, ect_reason, "already_run", protected_stock_list

    trading_config = TradingConfig.from_dict(config)
    strategy_state = runner.paper_storage.load_strategy_state()
    take_profit_block_t0_date = strategy_state.get("take_profit_block_t0_date")
    if take_profit_block_t0_date == trade_date:
        logger.info("整体止盈且关闭自动补仓，本日不触发空仓提前调仓")
        strategy_state.pop("take_profit_block_t0_date", None)
        runner.paper_storage.save_strategy_state(strategy_state)

    allow_early_rebalance, early_rebalance_mode, precomputed_protected_stocks = (
        _resolve_early_rebalance_context(
            runner,
            trade_date,
            config,
            take_profit_block_t0_date=take_profit_block_t0_date,
        )
    )

    early_rebalance_triggered = False
    try:
        is_rebalance_day = runner._check_rebalance_day(trade_date, int(config["rebalance_freq"]))
    except RuntimeError as exc:
        if allow_early_rebalance:
            trigger_label = (
                "空仓提前调仓"
                if early_rebalance_mode == "empty"
                else "持有期拖尾提前调仓"
            )
            logger.warning(f"当前不是调仓日，但满足{trigger_label}条件：{exc}")
            is_rebalance_day = True
            early_rebalance_triggered = True
        else:
            logger.info(f"当前不是调仓日：{exc}")
            return targets_info, ect_exposure, ect_reason, "not_rebalance_day", protected_stock_list

    if not is_rebalance_day:
        if allow_early_rebalance:
            if early_rebalance_mode == "empty":
                logger.warning("非调仓日，但当前空仓，提前执行 T0")
            else:
                logger.warning("非调仓日，但当前仍有拖尾持仓，提前执行 T0")
            is_rebalance_day = True
            early_rebalance_triggered = True
        else:
            logger.info("非调仓日，跳过 T0")
            return targets_info, ect_exposure, ect_reason, "not_rebalance_day", protected_stock_list

    if early_rebalance_triggered:
        if early_rebalance_mode == "empty":
            logger.warning("空仓提前调仓触发，执行 T0")
        else:
            logger.warning(
                f"持有期拖尾提前调仓触发，执行 T0（盈利延续保护 {len(precomputed_protected_stocks)} 只）"
            )

    logger.info("当前是调仓日，执行 T0")

    if bool(config.get("equity_curve_enabled", False)):
        logger.info("-" * 80)
        logger.info("计算 ECT 仓位系数")
        logger.info("-" * 80)

        ect_config = create_equity_curve_config_from_dict(config)
        ect_monitor = EquityCurveMonitor(ect_config)
        nav_df = runner.paper_storage.load_all_nav()
        if nav_df is not None and len(nav_df) > 0:
            nav_series = nav_df.set_index("trade_date")["nav"]
            ect_exposure, ect_reason = ect_monitor.calculate_exposure(
                nav_series, current_date=trade_date
            )
            logger.info(f"ECT 计算结果: {ect_reason}")
            logger.info(f"ECT 仓位系数: {ect_exposure:.2f}")
        else:
            logger.warning("NAV 历史为空，使用默认系数 1.0")
            ect_exposure = 1.0
            ect_reason = "NAV 历史为空"

        logger.info("-" * 80)

    market_regime_exposure = 1.0
    market_regime_reason = "市场择时未启用"
    if bool(config.get("market_regime_enabled", False)) or bool(
        config.get("market_regime_ma250_hard_stop", False)
    ):
        logger.info("-" * 80)
        logger.info("计算市场择时仓位系数")
        logger.info("-" * 80)
        market_regime_exposure, market_regime_reason = runner.compute_market_regime_exposure(
            trade_date, config
        )
        logger.info(f"市场择时: {market_regime_reason}")
        logger.info(f"市场择时仓位系数: {market_regime_exposure:.2f}")
        logger.info("-" * 80)

    final_exposure = ect_exposure * market_regime_exposure
    if final_exposure < 1.0:
        logger.info(
            f"综合仓位系数: {final_exposure:.2f}"
            f" (ECT={ect_exposure:.2f} × 市场择时={market_regime_exposure:.2f})"
        )

    protected_stocks = set(precomputed_protected_stocks)
    if early_rebalance_triggered and early_rebalance_mode == "holding_tail" and protected_stocks:
        # 回测侧在生成信号后做 "残留占比 + 新信号权重 <= 100%" 校验。
        # 纸面交易在 signal_gate_mode=disabled 时，新信号权重约等于 100%，
        # 只要保护持仓残留占比 > 0 就必然超限。这里做前置短路，
        # 避免重复执行整段 T0 再撤回导致日志噪音。
        signal_gate_mode = str(config.get("signal_gate_mode", "disabled")).lower()
        if signal_gate_mode == "disabled":
            daily_data = runner.loader.load_clean_daily_by_date(trade_date)
            if daily_data is not None and not daily_data.empty:
                price_map: Dict[str, float] = {}
                for _, row in daily_data.iterrows():
                    price_map[str(row["ts_code"])] = float(row.get("close", 0.0))

                total_value = runner.account.get_total_value(price_map)
                residual_value = 0.0
                positions = runner.account.get_positions()
                for ts_code in protected_stocks:
                    pos = positions.get(ts_code)
                    price = price_map.get(ts_code, 0.0)
                    if pos is not None and price > 0:
                        residual_value += pos.shares * price

                residual_ratio = residual_value / total_value if total_value > 0 else 0.0
                if residual_ratio > 1e-9:
                    logger.info(
                        f"持有期拖尾提前调仓前置拒绝：signal_gate_mode=disabled 时，"
                        f"残留仓位 {residual_ratio:.1%} + 新信号仓位 100.0% 必然 > 100%，"
                        "与回测侧权重校验结论一致"
                    )
                    return (
                        targets_info,
                        ect_exposure,
                        ect_reason,
                        "not_rebalance_day",
                        [],
                    )
    if bool(config.get("enable_profit_based_holding", False)):
        logger.info("-" * 80)
        if str(config.get("profit_extension_mode", "pnl")) == "disabled":
            logger.info("盈利延续模式未启用，跳过")
        else:
            logger.info("计算盈利延续保护")
            if early_rebalance_mode == "holding_tail" and protected_stocks:
                logger.info("复用拖尾提前调仓阶段已确认的盈利延续保护")
            else:
                protected_stocks = set(runner.evaluate_profit_extension(trade_date, config))
            protected_stock_list = sorted(protected_stocks)
            if protected_stocks:
                logger.info(f"盈利延续保护: {len(protected_stocks)} 只股票 → {protected_stocks}")
            else:
                logger.info("无持仓满足盈利延续条件")
        logger.info("-" * 80)

    try:
        runner.run_t0(
            trade_date=trade_date,
            buy_price_type=str(config["buy_price"]),
            sell_price_type=str(config["sell_price"]),
            universe_type=str(config["universe"]),
            top_n=int(config["top_n"]),
            model_version=config.get("model_version"),
            rebalance_freq=int(config["rebalance_freq"]),
            max_per_industry=config.get("max_per_industry"),
            max_weight_per_stock=config.get("max_weight_per_stock"),
            exclude_st=bool(config.get("exclude_st", True)),
            min_list_days=int(config.get("min_list_days", 365)),
            industry_momentum_filter=bool(config.get("industry_momentum_filter", False)),
            industry_momentum_bottom_pct=float(config.get("industry_momentum_bottom_pct", 0.5)),
            holding_bonus_enabled=bool(config.get("holding_bonus_enabled", False)),
            holding_bonus_sigma=float(config.get("holding_bonus_sigma", 0.5)),
            trading_config=trading_config,
            force_rebalance=early_rebalance_triggered,
            protected_stocks=protected_stocks,
        )

        t1_date = runner._get_next_trade_date(trade_date)
        if t1_date:
            instructions = runner.paper_storage.load_instructions(t1_date)
            if instructions:
                # 持有期拖尾提前调仓权重校验（与回测侧 engine.py 一致）：
                # 生成信号后检查 "残留仓位占比 + 新信号仓位 ≤ 100%"，超限则撤回。
                if early_rebalance_triggered and early_rebalance_mode == "holding_tail":
                    positions = runner.account.get_positions()
                    daily_data = runner.loader.load_clean_daily_by_date(trade_date)
                    if daily_data is not None and not daily_data.empty and positions:
                        price_map: Dict[str, float] = {}
                        for _, row in daily_data.iterrows():
                            price_map[str(row["ts_code"])] = float(row.get("close", 0.0))

                        total_value = runner.account.get_total_value(price_map)

                        sell_codes = {
                            inst.ts_code for inst in instructions if inst.action == "sell"
                        }

                        residual_value = 0.0
                        for ts_code, pos in positions.items():
                            price = price_map.get(ts_code, 0.0)
                            if price > 0 and ts_code not in sell_codes:
                                residual_value += pos.shares * price

                        residual_ratio = residual_value / total_value if total_value > 0 else 0.0

                        new_buy_weight = sum(
                            inst.target_weight
                            for inst in instructions
                            if inst.action == "buy"
                        )

                        if residual_ratio + new_buy_weight > 1.0 + 1e-9:
                            logger.info(
                                f"持有期拖尾提前调仓撤回：残留仓位 {residual_ratio:.1%} + "
                                f"新信号仓位 {new_buy_weight:.1%} = "
                                f"{(residual_ratio + new_buy_weight):.1%} > 100%，"
                                f"与回测侧权重校验一致"
                            )
                            runner.paper_storage.save_instructions(t1_date, [])
                            return (
                                targets_info,
                                ect_exposure,
                                ect_reason,
                                "not_rebalance_day",
                                [],
                            )

                if final_exposure < 1.0:
                    logger.info(
                        f"应用综合仓位系数 {final_exposure:.2f} 到买入指令"
                        f" (ECT={ect_exposure:.2f}, 市场择时={market_regime_exposure:.2f})"
                    )
                    valid_instructions = []
                    for instruction in instructions:
                        if instruction.action == "buy":
                            original_shares = instruction.shares
                            instruction.shares = int(instruction.shares * final_exposure)
                            instruction.shares = (instruction.shares // 100) * 100
                            if instruction.shares == 0:
                                logger.warning(
                                    f"仓位调整后 {instruction.ts_code} 股数为0，"
                                    f"跳过该买入指令（原 {original_shares} 股）"
                                )
                                continue

                            if instruction.shares != original_shares:
                                instruction.reason = (
                                    f"{instruction.reason} (仓位调整:"
                                    f" {original_shares} -> {instruction.shares}股)"
                                )
                        valid_instructions.append(instruction)

                    runner.paper_storage.save_instructions(t1_date, valid_instructions)
                    logger.info(
                        f"已将仓位系数应用到买入指令：{len(valid_instructions)}/{len(instructions)} 条有效"
                    )

                for instruction in instructions:
                    if instruction.action == "buy":
                        targets_info.append(
                            {
                                "ts_code": instruction.ts_code,
                                "target_weight": instruction.target_weight,
                                "reason": instruction.reason,
                                "score": None,
                            }
                        )

        t0_status = "success" if targets_info else "no_targets"
        logger.info(f"T0 执行完成：生成 {len(targets_info)} 个目标")
    except Exception as exc:
        logger.error(f"T0 执行失败: {exc}")
        t0_status = f"error:{exc}"

    return targets_info, ect_exposure, ect_reason, t0_status, protected_stock_list
