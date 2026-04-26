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
from .models import PendingBuy, PendingSell, TradeInstruction
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
    stock_names: Dict[str, str] = field(default_factory=dict)
    missing_factors: List[str] = field(default_factory=list)


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
    if (
        bool(config.get("enable_profit_based_holding", False))
        and str(config.get("early_exit_mode", "disabled")) != "disabled"
    ):
        early_exit_actions = _check_early_exit(runner, corrected_date, config)
    else:
        logger.info("亏损提前换出未启用，跳过")

    _report("整体止盈检查")
    take_profit_actions = _check_take_profit(runner, corrected_date, config)

    _report("处理延迟卖出")
    pending_sell_actions = _process_pending_sells(runner, corrected_date, config)

    _report("执行 T1 指令")
    t1_actions = _execute_t1_if_pending(runner, corrected_date, config)

    _report("执行 T0")
    t0_targets, ect_exposure, ect_reason, t0_status = _execute_t0_if_rebalance_day(
        runner, corrected_date, config
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
        stock_names=stock_names,
        missing_factors=list(runner.missing_factors),
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

        if sl_action.is_limit_down:
            pending_sell = PendingSell(
                ts_code=sl_action.ts_code,
                shares=sell_shares,
                target_weight=0.0,
                reason=f"止损-{sl_action.reason}",
                create_date=trade_date,
                attempts=0,
            )
            runner.broker.pending_sells.append(pending_sell)
            runner.broker.storage.save_pending_sells(runner.broker.pending_sells)

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

    for action in actions:
        pending_sell = PendingSell(
            ts_code=str(action["ts_code"]),
            shares=int(action["shares"]),
            target_weight=0.0,
            reason=str(action["reason"]),
            create_date=trade_date,
            attempts=0,
        )
        runner.broker.pending_sells.append(pending_sell)
        logger.info(f"亏损提前换出 → 加入延迟卖出队列: {action['ts_code']} {action['shares']}股")

    runner.broker.storage.save_pending_sells(runner.broker.pending_sells)
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

    existing_pending = {sell.ts_code for sell in runner.broker.pending_sells}
    actions = []
    for ts_code, pos in positions.items():
        if ts_code in existing_pending:
            continue

        sell_shares = (pos.shares // 100) * 100
        if sell_shares <= 0:
            continue

        reason = f"整体止盈: 本轮收益率={profit_rate:.2%} >= {float(threshold):.2%}"
        runner.broker.pending_sells.append(
            PendingSell(
                ts_code=ts_code,
                shares=sell_shares,
                target_weight=0.0,
                reason=reason,
                create_date=trade_date,
                attempts=0,
            )
        )
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
    runner.broker.storage.save_pending_sells(runner.broker.pending_sells)
    logger.warning(
        f"整体止盈触发: 本轮收益率={profit_rate:.2%}, 已加入 {len(actions)} 条延迟卖出指令"
    )
    return actions


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
    """执行 T1（如果有交易指令或补位计划）。"""
    actions: List[Dict[str, object]] = []

    if runner.paper_storage.check_run_exists("t1", trade_date):
        pending_buys = runner.paper_storage.load_pending_buys()
        if not pending_buys:
            logger.info(f"T1 工作流已在 {trade_date} 执行过，跳过")
            return actions

        logger.info(f"T1 指令已执行，但有 {len(pending_buys)} 个补位计划待处理")
        buy_prices, sell_prices = runner._load_prices(
            trade_date,
            str(config["buy_price"]),
            str(config["sell_price"]),
        )
        if not buy_prices:
            logger.error("无法加载价格数据，跳过补位处理")
            return actions

        replenishment_fills = runner._execute_pending_buys(
            pending_buys,
            buy_prices,
            trade_date,
            str(config["buy_price"]),
        )
        if replenishment_fills:
            for fill in replenishment_fills:
                actions.append(
                    {
                        "ts_code": fill.ts_code,
                        "action": fill.action,
                        "shares": fill.shares,
                        "reason": fill.reason,
                    }
                )
            runner.account.update_last_date(trade_date)
            runner.account.save_state()
            all_prices = {**sell_prices, **buy_prices}
            runner._record_nav(trade_date, all_prices)

        new_failed = runner.broker.get_failed_buy_targets()
        if new_failed:
            max_attempt = max([pending_buy.attempts for pending_buy in pending_buys], default=0)
            _handle_failed_buys(runner, trade_date, config, new_failed, attempt_count=max_attempt)
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

    pending_buys = runner.paper_storage.load_pending_buys()
    if not instructions and not pending_buys:
        logger.info(f"未找到 {trade_date} 的交易指令或补位买入计划，跳过 T1")
        return actions

    if instructions:
        logger.info("=" * 80)
        logger.info(f"【T1 指令驱动】读取到 {len(instructions)} 条交易指令")
        logger.info("=" * 80)

    if pending_buys:
        logger.info(f"找到 {len(pending_buys)} 个补位买入计划（将在指令执行后处理）")

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

    if instructions:
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
        _handle_failed_buys(runner, trade_date, config, failed_buy_targets, attempt_count=0)

    if pending_buys:
        logger.info("执行补位买入计划")
        replenishment_fills = runner._execute_pending_buys(
            pending_buys,
            buy_prices,
            trade_date,
            str(config["buy_price"]),
        )
        if replenishment_fills:
            fills_count += len(replenishment_fills)
            orders_count += len(replenishment_fills)
            for fill in replenishment_fills:
                actions.append(
                    {
                        "ts_code": fill.ts_code,
                        "action": fill.action,
                        "shares": fill.shares,
                        "reason": fill.reason,
                    }
                )

        new_failed_buy_targets = runner.broker.get_failed_buy_targets()
        if new_failed_buy_targets:
            max_attempt = max([pending_buy.attempts for pending_buy in pending_buys], default=0)
            _handle_failed_buys(
                runner,
                trade_date,
                config,
                new_failed_buy_targets,
                attempt_count=max_attempt,
            )

    if fills_count > 0:
        runner.account.update_last_date(trade_date)
        runner.account.save_state()
        all_prices = {**sell_prices, **buy_prices}
        runner._record_nav(trade_date, all_prices)

    if instructions or pending_buys:
        run_record = {
            "trade_date": trade_date,
            "buy_price_type": config["buy_price"],
            "sell_price_type": config["sell_price"],
            "instructions_count": len(instructions) if instructions else 0,
            "pending_buys_count": len(pending_buys) if pending_buys else 0,
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
    failed_buy_targets: List[TradeInstruction],
    attempt_count: int,
) -> None:
    """处理买入失败并生成补位计划。"""
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
        f"基于当日 {trade_date} 数据重新生成下一交易日补位目标（第 {next_attempt} 次补位尝试）"
    )
    logger.info("=" * 80)

    next_trade_date = runner._get_next_trade_date(trade_date)
    if next_trade_date:
        trading_config = TradingConfig.from_dict(config)
        replacement_targets = runner.generate_replacement_targets(
            trade_date=trade_date,
            failed_count=len(failed_buy_targets),
            universe_type=str(config["universe"]),
            model_version=config.get("model_version"),
            buy_price_type=str(config["buy_price"]),
            original_signal_date=trade_date,
            max_per_industry=config.get("max_per_industry"),
            exclude_st=bool(config.get("exclude_st", True)),
            min_list_days=int(config.get("min_list_days", 365)),
            trading_config=trading_config,
        )

        if replacement_targets:
            pending_buys = []
            for target in replacement_targets:
                pending_buys.append(
                    PendingBuy(
                        ts_code=target.ts_code,
                        target_weight=target.target_weight,
                        reason=target.reason,
                        create_date=trade_date,
                        attempts=next_attempt,
                        last_attempt_date="",
                        original_signal_date=trade_date,
                    )
                )

            runner.paper_storage.save_pending_buys(pending_buys)
            logger.info(f"已生成 {len(replacement_targets)} 个补位目标，保存到独立的补位买入队列")
            logger.info(
                f"下一交易日 {next_trade_date} 将自动读取并执行补位买入（第 {next_attempt}/{max_replenishment_attempts} 次尝试）"
            )
            logger.info("补位买入不会触发现有持仓的卖出")
        else:
            logger.warning("无法生成补位目标，将原始失败目标保存为补位计划以待重试")
            fallback_pending = []
            for target in failed_buy_targets:
                fallback_pending.append(
                    PendingBuy(
                        ts_code=target.ts_code,
                        target_weight=target.target_weight,
                        reason=f"补位待重试-{target.reason}",
                        create_date=trade_date,
                        attempts=next_attempt,
                        last_attempt_date="",
                        original_signal_date=trade_date,
                    )
                )
            runner.paper_storage.save_pending_buys(fallback_pending)
            logger.info(f"已保存 {len(fallback_pending)} 个失败目标到补位队列，下次运行将重试")
    else:
        logger.error("无法获取下一交易日，补位计划生成失败")

    runner.broker.clear_failed_buy_targets()


def _execute_t0_if_rebalance_day(
    runner: PaperTradingRunner,
    trade_date: str,
    config: Dict[str, object],
) -> Tuple[List[Dict[str, object]], float, str, str]:
    """执行 T0（如果是调仓日）。"""
    targets_info: List[Dict[str, object]] = []
    ect_exposure = 1.0
    ect_reason = "ECT 未启用"

    if runner.paper_storage.check_run_exists("t0", trade_date):
        logger.info(f"T0 工作流已在 {trade_date} 执行过，跳过")
        return targets_info, ect_exposure, ect_reason, "already_run"

    trading_config = TradingConfig.from_dict(config)
    strategy_state = runner.paper_storage.load_strategy_state()
    take_profit_block_t0_date = strategy_state.get("take_profit_block_t0_date")
    if take_profit_block_t0_date == trade_date:
        logger.info("整体止盈且关闭自动补仓，本日不触发空仓提前调仓")
        strategy_state.pop("take_profit_block_t0_date", None)
        runner.paper_storage.save_strategy_state(strategy_state)

    pending_instruction = runner.paper_storage.find_pending_instructions(trade_date)
    pending_buys = runner.paper_storage.load_pending_buys()
    allow_early_rebalance = (
        bool(config.get("enable_early_rebalance_on_empty", True))
        and take_profit_block_t0_date != trade_date
        and not runner.account.get_positions()
        and not runner.broker.pending_sells
        and not pending_buys
        and pending_instruction is None
    )

    try:
        is_rebalance_day = runner._check_rebalance_day(trade_date, int(config["rebalance_freq"]))
    except RuntimeError as exc:
        if allow_early_rebalance:
            logger.warning(f"当前不是调仓日，但满足空仓提前调仓条件：{exc}")
            is_rebalance_day = True
        else:
            logger.info(f"当前不是调仓日：{exc}")
            return targets_info, ect_exposure, ect_reason, "not_rebalance_day"

    if not is_rebalance_day:
        if allow_early_rebalance:
            logger.warning("非调仓日，但当前空仓，提前执行 T0")
            is_rebalance_day = True
        else:
            logger.info("非调仓日，跳过 T0")
            return targets_info, ect_exposure, ect_reason, "not_rebalance_day"

    if allow_early_rebalance:
        logger.warning("空仓提前调仓触发，执行 T0")

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

    protected_stocks = set()
    if bool(config.get("enable_profit_based_holding", False)):
        logger.info("-" * 80)
        logger.info("计算盈利延续保护")
        logger.info("-" * 80)
        protected_stocks = runner.evaluate_profit_extension(trade_date, config)
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
            force_rebalance=allow_early_rebalance,
            protected_stocks=protected_stocks,
        )

        t1_date = runner._get_next_trade_date(trade_date)
        if t1_date:
            instructions = runner.paper_storage.load_instructions(t1_date)
            if instructions:
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

    return targets_info, ect_exposure, ect_reason, t0_status
