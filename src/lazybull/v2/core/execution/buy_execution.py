"""回测买入执行 mixin

v2 迁移复制件（P2a-T5，行为冻结）：与旧模块 ``backtest/buy_execution.py``
双源并存至 v2 切换；除模块头与 import 改指外，主体差异仅限：
- 退役 exposure 组件调用点摘除 4 处（逐位等价依据：B0 下乘子恒 1.0、
  回调守卫在实现内部）：
  1. 主买入预算的暴露乘子调用（旧 :123-124，``a *= 1.0`` ≡ 删除）
  2. 主买入成交后的预算折扣回补回调（旧 :170-174，仅删回调与其注释）
  3. 仓位补齐预算的暴露乘子调用（旧 :506-507，同 1 整删）
  4. 补齐成交后的预算折扣回补守卫块（旧 :545-554，整块删除）
- D8 提取式拆分（见文末登记）。

D8 提取式拆分（T5 落位时执行）：旧 ``_execute_pending_buys`` 圈复杂度 38、
``_process_position_completion`` 圈复杂度 28 超 mccabe 硬上限 12（旧路径基线
在册硬超限），复制到新路径 = 新增未豁免硬超限 ⇒ 按「提取模块级私有 helper +
主方法委托」拆分，公开签名、数值操作顺序、日志文本逐字保留：

- ``_execute_pending_buys`` 拆出：``_pop_pending_signal``（取信号）、
  ``_parse_pending_signal_data``（信号解析）、``_sync_slot_weights_with_signals``
  （槽位权重同步）、``_prepare_pending_buy_context``（风险预算/留痕/预算）、
  ``_execute_slot_matched_buys``（槽位匹配执行编排）、``_build_buy_detail``
  （买入明细）、``_get_position_weight``（持仓权重，含 last_known_price
  估值回退链 D4 逐字保留）、``_record_buy_execution``（成交记录）、
  ``_check_candidate_tradeable``（候选可交易检查）、``_on_slot_reject``
  （槽位拒绝回调）、``_find_t1_buy_trade``（成交倒查）、
  ``_record_execution_attribution``（执行归因）、
  ``_register_unfilled_slots``（未成交槽位登记）
- ``_process_position_completion`` 拆出：``_record_completion_skip_for_slot``
  （带批次前缀的跳过记录）、``_build_completion_stocks_to_try``（补齐候选构建）、
  ``_completion_buy_loop``（逐槽位补齐买入）、``_complete_slot_attempt``
  （单信号日补齐编排）
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.v2.common.date_utils import to_trade_date_str
from src.lazybull.v2.common.trade_status import is_tradeable
from src.lazybull.v2.core.decision.buy_plan import (
    REASON_ALREADY_BOUGHT,
    REASON_EXECUTION_FAILED,
    fill_slots_from_candidates,
)
from src.lazybull.v2.core.decision.sizing import compute_lot_shares


def _pop_pending_signal(
    engine,
    date: pd.Timestamp,
    trading_dates: List[pd.Timestamp],
    date_to_idx: Dict,
) -> Optional[Tuple[pd.Timestamp, Dict]]:
    """查找并弹出前一交易日信号（D8 提取件，顺序不变）；无信号返回 None。

    Args:
        engine: BacktestBuyExecutionMixin 实例
        date: 当前日期
        trading_dates: 交易日列表
        date_to_idx: 日期到索引的映射
    """
    # 查找前一个交易日的信号
    current_idx = date_to_idx.get(date)
    if current_idx is None or current_idx == 0:
        return None

    signal_date = trading_dates[current_idx - 1]

    if signal_date not in engine.pending_signals:
        return None

    signal_data = engine.pending_signals.pop(signal_date)
    return signal_date, signal_data


def _parse_pending_signal_data(engine, signal_data: Dict) -> Dict:
    """解析信号数据（兼容新旧格式；D8 提取件，顺序不变）。"""
    # 兼容性处理：支持旧格式和新格式
    # 旧格式：signal_data = {stock: weight}
    # 新格式：signal_data = {
    #   'signals': {stock: weight}, 'priority_candidates': [...],
    #   'slot_weights': [...], 'ranked_candidates': [...], 'target_n': N
    # }
    if isinstance(signal_data, dict) and "signals" in signal_data:
        signals = signal_data["signals"]
        ranked_candidates = signal_data.get("ranked_candidates", [])
        priority_candidates = signal_data.get("priority_candidates", ranked_candidates)
        slot_weights = signal_data.get("slot_weights", [])
        target_n = signal_data.get("target_n", len(signals))
        desired_position_count = signal_data.get("desired_position_count")
        tranche_idx = signal_data.get("tranche_idx", 0)
        decision_trace = signal_data.get("decision_trace")
    else:
        signals = signal_data
        ranked_candidates = []
        priority_candidates = list(signals.items())
        slot_weights = [
            {"stock": stock, "weight": float(weight)} for stock, weight in signals.items()
        ]
        target_n = len(signals)
        desired_position_count = None
        tranche_idx = 0
        decision_trace = None

    tranche_tag = (
        f"[批次 {tranche_idx + 1}/{engine.stagger_tranches}] "
        if engine.stagger_tranches > 1
        else ""
    )

    return {
        "signals": signals,
        "ranked_candidates": ranked_candidates,
        "priority_candidates": priority_candidates,
        "slot_weights": slot_weights,
        "target_n": target_n,
        "desired_position_count": desired_position_count,
        "tranche_idx": tranche_idx,
        "decision_trace": decision_trace,
        "tranche_tag": tranche_tag,
    }


def _sync_slot_weights_with_signals(signals: Dict, slot_weights: List[Dict]) -> List[Dict]:
    """执行日权重可能被风险预算调整，按原槽位顺序同步更新（D8 提取件，顺序不变）。"""
    if slot_weights:
        ordered_slot_weights = []
        seen_slot_stocks = set()
        for slot in slot_weights:
            slot_stock = slot.get("stock")
            if slot_stock in signals and slot_stock not in seen_slot_stocks:
                ordered_slot_weights.append(
                    {"stock": slot_stock, "weight": float(signals[slot_stock])}
                )
                seen_slot_stocks.add(slot_stock)
        for stock, weight in signals.items():
            if stock not in seen_slot_stocks:
                ordered_slot_weights.append({"stock": stock, "weight": float(weight)})
    else:
        ordered_slot_weights = [
            {"stock": stock, "weight": float(weight)} for stock, weight in signals.items()
        ]
    return ordered_slot_weights


def _prepare_pending_buy_context(
    engine, date: pd.Timestamp, signal_date: pd.Timestamp, parsed: Dict
) -> Dict:
    """风险预算、决策留痕与买入预算计算（D8 提取件，顺序不变）。"""
    signals = parsed["signals"]
    ranked_candidates = parsed["ranked_candidates"]
    priority_candidates = parsed["priority_candidates"]
    slot_weights = parsed["slot_weights"]
    target_n = parsed["target_n"]
    tranche_idx = parsed["tranche_idx"]
    decision_trace = parsed["decision_trace"]

    # 应用风险预算（波动率缩放）
    if engine.enable_risk_budget:
        signals = engine._apply_risk_budget(signals, date)

    slot_weights = _sync_slot_weights_with_signals(signals, slot_weights)
    if not priority_candidates:
        priority_candidates = list(signals.items())

    if decision_trace is None:
        decision_trace = engine._build_signal_decision_trace(
            date=signal_date,
            target_n=target_n,
            candidate_count=len(ranked_candidates) if ranked_candidates else len(signals),
            tranche_idx=tranche_idx,
        )

    decision_trace["queued"] = True
    decision_trace["final_target_exposure"] = float(sum(signals.values()))

    # 计算当前组合市值
    portfolio_value = engine._calculate_portfolio_value(date)
    current_value = portfolio_value

    # 分批调仓按本批槽位占总 TopN 的比例分配组合价值。
    if engine.stagger_tranches > 1:
        current_value *= engine._get_tranche_capital_fraction(tranche_idx)

    return {
        "signal_date": signal_date,
        "signals": signals,
        "ranked_candidates": ranked_candidates,
        "priority_candidates": priority_candidates,
        "slot_weights": slot_weights,
        "target_n": target_n,
        "desired_position_count": parsed["desired_position_count"],
        "tranche_idx": tranche_idx,
        "decision_trace": decision_trace,
        "tranche_tag": parsed["tranche_tag"],
        "portfolio_value": portfolio_value,
        "current_value": current_value,
    }


def _build_buy_detail(portfolio_value: float, stock: str, target_value: float) -> Dict:
    """买入明细（实际权重；D8 提取件，顺序不变）。"""
    actual_weight = float(target_value / portfolio_value) if portfolio_value > 0 else 0.0
    return {"stock": stock, "weight": actual_weight}


def _get_position_weight(engine, date: pd.Timestamp, portfolio_value: float, stock: str) -> float:
    """持仓权重（含 last_known_price 估值回退链；D8 提取件，顺序不变）。"""
    if portfolio_value <= 0 or stock not in engine.positions:
        return 0.0

    info = engine.positions[stock]
    shares = info.get("shares", 0)
    trade_price = engine._get_trade_price(date, stock)
    if trade_price is None:
        trade_price = info.get("last_known_price")
        if trade_price is None:
            trade_price = info.get("buy_trade_price", 0.0)
    else:
        info["last_known_price"] = trade_price

    return float(shares * trade_price / portfolio_value) if trade_price else 0.0


def _record_buy_execution(
    engine,
    date: pd.Timestamp,
    signal_date: pd.Timestamp,
    successful_buys: List[Dict],
    buy_detail: Dict,
    stock: str,
    target_value: float,
) -> bool:
    """执行买入并按成交结果登记（D8 提取件，顺序不变）。"""
    trades_before = len(engine.trades)

    engine._buy_stock(date, stock, target_value, signal_date=signal_date)

    trade_executed = (
        len(engine.trades) > trades_before
        and engine.trades[-1].get("action") == "buy"
        and engine.trades[-1].get("stock") == stock
        and engine.trades[-1].get("date") == date
    )

    if trade_executed:
        successful_buys.append(buy_detail.copy())
        return True

    return False


def _check_candidate_tradeable(
    blocked_tradeability_reasons: Dict[str, str],
    date_quote: pd.DataFrame,
    trade_date_str: str,
    candidate_stock: str,
) -> tuple:
    """候选可交易检查（带原因缓存；D8 提取件，顺序不变）。"""
    if candidate_stock in blocked_tradeability_reasons:
        return False, blocked_tradeability_reasons[candidate_stock]
    if date_quote.empty:
        blocked_tradeability_reasons[candidate_stock] = "无行情"
        return False, "无行情"

    tradeable, reason = is_tradeable(candidate_stock, trade_date_str, date_quote, action="buy")
    if not tradeable:
        blocked_tradeability_reasons[candidate_stock] = reason
    return tradeable, reason


def _on_slot_reject(
    slot_states: Dict[int, Dict],
    failed_buys: List[Dict],
    portfolio_value: float,
    current_value: float,
    slot: Dict,
    candidate_stock: str,
    reason: str,
) -> None:
    """槽位拒绝回调（失败原因记录；D8 提取件，顺序不变）。"""
    # 已持仓/当日已被其他槽位买入：与既有行为一致，静默跳过不计入失败原因
    if reason in ("__held__", REASON_ALREADY_BOUGHT):
        return
    if reason == REASON_EXECUTION_FAILED:
        reason = "未成交"
    state = slot_states.setdefault(id(slot), {"last_reason": "候选耗尽", "failure_recorded": False})
    state["last_reason"] = reason
    if candidate_stock == slot["stock"]:
        target_value = current_value * float(slot["weight"])
        failed_buys.append(
            {**_build_buy_detail(portfolio_value, candidate_stock, target_value), "reason": reason}
        )
        state["failure_recorded"] = True


def _execute_slot_matched_buys(engine, date: pd.Timestamp, ctx: Dict) -> Dict:
    """槽位匹配买入执行（D8 提取件，顺序不变；槽位匹配委托 buy_plan 共享骨架）。

    Returns:
        含 planned_slot_weights / planned_buys / successful_buys / failed_buys /
        match_result / slot_states 的执行结果字典
    """
    signal_date = ctx["signal_date"]
    slot_weights = ctx["slot_weights"]
    priority_candidates = ctx["priority_candidates"]
    portfolio_value = ctx["portfolio_value"]
    current_value = ctx["current_value"]
    desired_position_count = ctx["desired_position_count"]

    planned_buys: List[Dict] = []
    successful_buys: List[Dict] = []
    failed_buys: List[Dict] = []
    inherited_stocks: List[str] = []
    inherited_position_count = 0

    # 旧实现首算 inherited_position_weight 从不被读取（死绑定），但求值经
    # _get_position_weight 带 last_known_price 缓存写回副作用——行为冻结：
    # 保留调用、不绑定变量（T8 格式债清理，同 T5 偏差登记 4 先例）。
    float(
        sum(
            _get_position_weight(engine, date, portfolio_value, stock) for stock in inherited_stocks
        )
    )

    trade_date_str = to_trade_date_str(date)
    date_quote = (
        engine.price_data_cache[engine.price_data_cache["trade_date"] == trade_date_str]
        if engine.price_data_cache is not None
        else pd.DataFrame()
    )
    blocked_tradeability_reasons: Dict[str, str] = {}

    if desired_position_count is None:
        desired_position_count = inherited_position_count + len(slot_weights)
    desired_position_count = int(desired_position_count or 0)
    available_slot_count = max(desired_position_count - len(engine.positions), 0)
    planned_slot_weights = slot_weights[:available_slot_count]

    # 槽位匹配委托 trading.buy_plan 共享骨架，买入评估/执行/失败记录通过回调注入
    slot_states: Dict[int, Dict] = {}

    def _evaluate_candidate(candidate_stock: str, slot: Dict) -> tuple:
        if candidate_stock in engine.positions:
            return False, "__held__"
        return _check_candidate_tradeable(
            blocked_tradeability_reasons, date_quote, trade_date_str, candidate_stock
        )

    def _execute_buy(candidate_stock: str, slot: Dict) -> bool:
        target_value = current_value * float(slot["weight"])
        buy_detail = _build_buy_detail(portfolio_value, candidate_stock, target_value)
        return _record_buy_execution(
            engine, date, signal_date, successful_buys, buy_detail, candidate_stock, target_value
        )

    def _on_reject(slot: Dict, candidate_stock: str, reason: str) -> None:
        _on_slot_reject(
            slot_states,
            failed_buys,
            portfolio_value,
            current_value,
            slot,
            candidate_stock,
            reason,
        )

    for slot_weight_info in planned_slot_weights:
        planned_buys.append(
            _build_buy_detail(
                portfolio_value,
                slot_weight_info["stock"],
                current_value * float(slot_weight_info["weight"]),
            )
        )

    match_result = fill_slots_from_candidates(
        slots=planned_slot_weights,
        candidates=[candidate for candidate, _ in priority_candidates],
        evaluate_candidate=_evaluate_candidate,
        execute_buy=_execute_buy,
        on_reject=_on_reject,
    )

    return {
        "planned_slot_weights": planned_slot_weights,
        "planned_buys": planned_buys,
        "successful_buys": successful_buys,
        "failed_buys": failed_buys,
        "match_result": match_result,
        "slot_states": slot_states,
    }


def _find_t1_buy_trade(trades: List[Dict], date: pd.Timestamp, stock: Optional[str]) -> Dict:
    """倒序查找当日本股买入成交（D8 提取件，顺序不变）。"""
    return next(
        (
            item
            for item in reversed(trades)
            if item.get("action") == "buy"
            and item.get("date") == date
            and item.get("stock") == stock
        ),
        {},
    )


def _record_execution_attribution(engine, date: pd.Timestamp, ctx: Dict, run: Dict) -> None:
    """逐槽位登记执行归因记录（D8 提取件，顺序不变）。"""
    signal_date = ctx["signal_date"]
    tranche_idx = ctx["tranche_idx"]
    priority_candidates = ctx["priority_candidates"]
    planned_slot_weights = run["planned_slot_weights"]
    match_result = run["match_result"]
    slot_states = run["slot_states"]

    candidate_rank = {
        stock: rank for rank, (stock, _score) in enumerate(priority_candidates, start=1)
    }
    candidate_score = {stock: float(score) for stock, score in priority_candidates}
    filled_by_slot = {id(item["slot"]): item["stock"] for item in match_result.filled}
    for slot in planned_slot_weights:
        planned_stock = str(slot["stock"])
        actual_stock = filled_by_slot.get(id(slot))
        state = slot_states.setdefault(
            id(slot), {"last_reason": "候选耗尽", "failure_recorded": False}
        )
        trade = _find_t1_buy_trade(engine.trades, date, actual_stock)
        signal_price = engine._get_trade_price(signal_date, actual_stock) if actual_stock else None
        buy_price = trade.get("price")
        signal_to_buy_return = (
            float(buy_price / signal_price - 1.0)
            if buy_price is not None and signal_price not in (None, 0)
            else None
        )
        engine.execution_attribution_records.append(
            {
                "signal_date": signal_date,
                "ranking_date": signal_date,
                "execution_date": date,
                "execution_stage": "t1",
                "tranche_idx": tranche_idx,
                "planned_stock": planned_stock,
                "actual_stock": actual_stock,
                "planned_rank": candidate_rank.get(planned_stock),
                "actual_rank": candidate_rank.get(actual_stock),
                "pred_score": candidate_score.get(actual_stock),
                "target_weight": float(slot["weight"]),
                "status": "filled" if actual_stock else "unfilled",
                "reason": (
                    state["last_reason"]
                    if not actual_stock or actual_stock != planned_stock
                    else None
                ),
                "buy_price": buy_price,
                "signal_price": signal_price,
                "signal_to_buy_return": signal_to_buy_return,
            }
        )


def _register_unfilled_slots(engine, date: pd.Timestamp, ctx: Dict, run: Dict) -> None:
    """登记未成交槽位供跨日补齐（D8 提取件，顺序不变）。"""
    signal_date = ctx["signal_date"]
    ranked_candidates = ctx["ranked_candidates"]
    tranche_idx = ctx["tranche_idx"]
    tranche_tag = ctx["tranche_tag"]
    portfolio_value = ctx["portfolio_value"]
    current_value = ctx["current_value"]
    planned_slot_weights = run["planned_slot_weights"]
    match_result = run["match_result"]
    successful_buys = run["successful_buys"]
    failed_buys = run["failed_buys"]
    slot_states = run["slot_states"]

    remaining_unfilled_slots = []
    for slot in match_result.unfilled:
        weight = float(slot["weight"])
        remaining_unfilled_slots.append({"stock": slot["stock"], "weight": weight})
        state = slot_states.setdefault(
            id(slot), {"last_reason": "候选耗尽", "failure_recorded": False}
        )
        if not state["failure_recorded"]:
            failed_buys.append(
                {
                    **_build_buy_detail(portfolio_value, slot["stock"], current_value * weight),
                    "reason": state["last_reason"],
                }
            )

    # 记录买入后的持仓数量
    actually_bought = len(successful_buys)

    # 同日顺延后仍未补满的空槽，才进入后续跨日补齐。
    if engine.enable_position_completion and remaining_unfilled_slots and ranked_candidates:
        unfilled_count = len(remaining_unfilled_slots)
        unfilled_stocks = [slot["stock"] for slot in remaining_unfilled_slots]

        engine.unfilled_slots[signal_date] = {
            "unfilled_count": unfilled_count,
            "unfilled_slot_weights": remaining_unfilled_slots,
            "target_n": len(planned_slot_weights),
            "ranked_candidates": ranked_candidates,
            "signal_date": signal_date,
            "first_attempt_date": date,
            "attempts": 0,
            "tranche_idx": tranche_idx,
        }

        engine.completion_stats["total_unfilled"] += 1

        engine._record_position_unfilled_summary(
            tranche_tag=tranche_tag,
            target_n=len(planned_slot_weights),
            actually_bought=actually_bought,
            unfilled_count=unfilled_count,
            unfilled_stocks=unfilled_stocks,
        )


def _record_completion_skip_for_slot(
    engine, tranche_tag: str, reason: str, original_signal_date: pd.Timestamp
) -> None:
    """带批次前缀的补齐跳过记录（D8 提取件，四处同型合并，顺序不变）。"""
    prefix = f"{tranche_tag.strip()} " if tranche_tag.strip() else ""
    engine._record_completion_skip(
        reason,
        f"{prefix}信号日{original_signal_date.date()}",
    )


def _build_completion_stocks_to_try(
    positions: Dict, new_ranked_candidates: List[Tuple[str, float]], unfilled_count: int
) -> List[Tuple[str, float]]:
    """从新的候选列表中选择可用股票，排除已持仓股票（D8 提取件，顺序不变）。

    多取 buffer 以应对部分候选不可交易（停牌/涨跌停）的情况
    """
    candidate_buffer = unfilled_count * 2
    stocks_to_try = []
    for stock, score in new_ranked_candidates:
        if stock not in positions:
            stocks_to_try.append((stock, score))
            if len(stocks_to_try) >= candidate_buffer:
                break
    return stocks_to_try


def _completion_buy_loop(
    engine,
    date: pd.Timestamp,
    original_signal_date: pd.Timestamp,
    prev_date: pd.Timestamp,
    trade_date_str: str,
    date_quote: pd.DataFrame,
    unfilled_slot_weights: List[Dict],
    stocks_to_try: List[Tuple[str, float]],
    new_ranked_candidates: List[Tuple[str, float]],
    current_value: float,
    completion_success_items: List[Dict[str, str]],
    completion_delayed_slots: List[str],
) -> List[Dict]:
    """逐个槽位尝试补齐买入（D8 提取件，顺序不变）。

    Returns:
        本日仍未补齐的槽位列表（保留到下次补齐）
    """
    bought_stocks = []
    remaining_unfilled_slots = []
    bought_stock_set = set()  # 跟踪已买入的股票，避免重复买入
    untradeable_stocks = set()  # 当天不可交易的股票，跳过后续槽位的重复尝试

    # 逐个槽位尝试补齐
    for slot_weight_info in unfilled_slot_weights:
        original_stock = slot_weight_info["stock"]
        weight = slot_weight_info["weight"]

        # 尝试从有限的候选列表中买入（按顺序）
        bought_for_this_slot = False

        for stock, score in stocks_to_try:
            # 跳过已买入或当天已确认不可交易的股票
            if stock in bought_stock_set or stock in untradeable_stocks:
                continue

            # 检查是否可交易（在当日 D）
            tradeable, reason = is_tradeable(stock, trade_date_str, date_quote, action="buy")

            if not tradeable:
                untradeable_stocks.add(stock)
                if engine.verbose:
                    engine._record_completion_skip(
                        "候选不可交易",
                        f"{stock}({reason})",
                    )
                continue

            # 可交易，尝试买入
            target_value = current_value * weight
            engine._buy_stock(date, stock, target_value, signal_date=original_signal_date)

            # 检查是否买入成功
            if stock in engine.positions:
                bought_stocks.append(stock)
                bought_stock_set.add(stock)  # 记录已买入
                bought_for_this_slot = True
                completion_success_items.append({"slot": original_stock, "buy": stock})
                engine._update_completion_attribution(
                    original_signal_date=original_signal_date,
                    ranking_date=prev_date,
                    execution_date=date,
                    planned_stock=original_stock,
                    actual_stock=stock,
                    ranked_candidates=new_ranked_candidates,
                )

                engine.completion_stats["total_completed"] += 1

                break

        # 如果该槽位未能补齐，保留到下次（会在下次重新生成有限候选继续尝试）
        if not bought_for_this_slot:
            remaining_unfilled_slots.append(slot_weight_info)
            completion_delayed_slots.append(original_stock)

    return remaining_unfilled_slots


def _complete_slot_attempt(
    engine,
    date: pd.Timestamp,
    current_idx: int,
    prev_date: pd.Timestamp,
    prev_date_quote: pd.DataFrame,
    trade_date_str: str,
    date_quote: pd.DataFrame,
    date_to_idx: Dict,
    signal_date: pd.Timestamp,
    slot_info: Dict,
    completed_signal_dates: List,
    completion_success_items: List[Dict[str, str]],
    completion_delayed_slots: List[str],
) -> None:
    """单个信号日的补齐尝试（D8 提取件，顺序不变）。"""
    first_attempt_date = slot_info["first_attempt_date"]
    unfilled_slot_weights = slot_info["unfilled_slot_weights"]
    # （T8 清理：旧实现此处解包的 target_n 从不使用——死绑定删除，行为不变）
    attempts = slot_info["attempts"]
    original_signal_date = slot_info["signal_date"]  # T日
    completion_tranche_idx = slot_info.get("tranche_idx", 0)
    tranche_tag = (
        f"[批次 {completion_tranche_idx + 1}/{engine.stagger_tranches}] "
        if engine.stagger_tranches > 1
        else ""
    )

    # 计算已经过了多少个交易日（从 T+1 开始）
    first_attempt_idx = date_to_idx.get(first_attempt_date)

    if first_attempt_idx is None:
        return

    days_elapsed = current_idx - first_attempt_idx

    # 在 T+1 日（首次尝试日）不进行补齐，从 T+2 日开始
    if days_elapsed == 0:
        return

    # 检查是否超过补齐窗口（窗口从 T+1 开始，所以是 < completion_window_days）
    if days_elapsed >= engine.completion_window_days:
        # 超过补齐窗口，放弃补齐
        unfilled_count = len(unfilled_slot_weights)
        unfilled_stocks = [slot["stock"] for slot in unfilled_slot_weights]
        engine.completion_stats["total_abandoned"] += 1
        completed_signal_dates.append(signal_date)

        engine._record_completion_abandoned_summary(
            tranche_tag=tranche_tag,
            original_signal_date=original_signal_date,
            attempts=attempts,
            unfilled_stocks=unfilled_stocks,
        )
        return

    # 在补齐窗口内，尝试补齐
    # 使用 D-1 日的数据重新生成候选股票列表
    if prev_date_quote.empty:
        _record_completion_skip_for_slot(engine, tranche_tag, "前日无行情", original_signal_date)
        return

    # 获取 D-1 日的股票池
    stock_universe = engine.universe.get_stocks(prev_date, quote_data=prev_date_quote)

    # 调用扩展点获取 D-1 日的额外数据
    extra_data = engine._build_signal_data(prev_date)
    if extra_data is None:
        _record_completion_skip_for_slot(engine, tranche_tag, "无数据", original_signal_date)
        return

    # 合并默认数据和额外数据
    signal_data = {}
    signal_data.update(extra_data)

    # 使用 D-1 日的数据重新生成排序候选列表
    new_ranked_candidates = engine.signal.generate_ranked(prev_date, stock_universe, signal_data)

    if not new_ranked_candidates:
        _record_completion_skip_for_slot(engine, tranche_tag, "无候选", original_signal_date)
        return

    # 从新的候选列表中选择可用股票，排除已持仓股票
    # 多取 buffer 以应对部分候选不可交易（停牌/涨跌停）的情况
    unfilled_count = len(unfilled_slot_weights)
    stocks_to_try = _build_completion_stocks_to_try(
        engine.positions, new_ranked_candidates, unfilled_count
    )

    if not stocks_to_try:
        _record_completion_skip_for_slot(engine, tranche_tag, "候选已持仓", original_signal_date)
        return

    # 尝试按槽位补齐
    current_value = engine._calculate_portfolio_value(date)
    # 分批调仓时，补齐预算也按本批槽位比例分配，与正常买入路径一致。
    if engine.stagger_tranches > 1:
        current_value *= engine._get_tranche_capital_fraction(completion_tranche_idx)

    remaining_unfilled_slots = _completion_buy_loop(
        engine,
        date=date,
        original_signal_date=original_signal_date,
        prev_date=prev_date,
        trade_date_str=trade_date_str,
        date_quote=date_quote,
        unfilled_slot_weights=unfilled_slot_weights,
        stocks_to_try=stocks_to_try,
        new_ranked_candidates=new_ranked_candidates,
        current_value=current_value,
        completion_success_items=completion_success_items,
        completion_delayed_slots=completion_delayed_slots,
    )

    # 更新槽位信息
    slot_info["attempts"] += 1
    slot_info["unfilled_slot_weights"] = remaining_unfilled_slots
    engine.completion_stats["completion_attempts"] += 1

    # 如果已经全部补齐，从待补齐列表中移除
    if not remaining_unfilled_slots:
        completed_signal_dates.append(signal_date)


class BacktestBuyExecutionMixin:
    """提供回测买入执行与补齐相关实现。"""

    def _execute_pending_buys(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """执行待执行的买入操作（T+1）

        同时跟踪未成交的槽位，如果启用补齐功能则记录到 unfilled_slots

        Args:
            date: 当前日期
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        popped = _pop_pending_signal(self, date, trading_dates, date_to_idx)
        if popped is None:
            return
        signal_date, signal_data = popped
        parsed = _parse_pending_signal_data(self, signal_data)
        ctx = _prepare_pending_buy_context(self, date, signal_date, parsed)
        run = _execute_slot_matched_buys(self, date, ctx)
        _record_execution_attribution(self, date, ctx, run)
        _register_unfilled_slots(self, date, ctx, run)

        # 当日买入/卖出明细统一在每日总结下一行展示，这里不再单独输出买入执行日志。

    def _process_position_completion(
        self,
        date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
        date_to_idx: Dict,
    ) -> None:
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
        prev_date_quote = price_data[price_data["trade_date"] == prev_date_str]

        # 获取当日（D）行情数据用于交易性检查
        trade_date_str = to_trade_date_str(date)
        date_quote = price_data[price_data["trade_date"] == trade_date_str]

        if date_quote.empty:
            if self.verbose:
                self._record_completion_skip("当日无行情", "当日无行情")
            return

        # 遍历所有未补齐的槽位
        completed_signal_dates = []
        completion_success_items: List[Dict[str, str]] = []
        completion_delayed_slots: List[str] = []

        for signal_date, slot_info in list(self.unfilled_slots.items()):
            _complete_slot_attempt(
                self,
                date=date,
                current_idx=current_idx,
                prev_date=prev_date,
                prev_date_quote=prev_date_quote,
                trade_date_str=trade_date_str,
                date_quote=date_quote,
                date_to_idx=date_to_idx,
                signal_date=signal_date,
                slot_info=slot_info,
                completed_signal_dates=completed_signal_dates,
                completion_success_items=completion_success_items,
                completion_delayed_slots=completion_delayed_slots,
            )

        # 清理已完成或放弃的槽位
        for signal_date in completed_signal_dates:
            del self.unfilled_slots[signal_date]

        if completion_success_items or completion_delayed_slots:
            remaining_count = sum(
                len(slot_info.get("unfilled_slot_weights", []))
                for slot_info in self.unfilled_slots.values()
            )
            logger.info(
                self._format_completion_summary(
                    success_items=completion_success_items,
                    delayed_slots=completion_delayed_slots,
                    remaining_count=remaining_count,
                )
            )

    def _buy_stock_with_status_check(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: Optional[pd.Timestamp] = None,
    ) -> None:
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
            date_quote = self.price_data_cache[
                self.price_data_cache["trade_date"] == trade_date_str
            ]
            if date_quote.empty:
                # 当日行情数据为空，无法判断交易状态，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action="buy",
                        current_date=date,
                        signal_date=signal_date or date,
                        target_value=target_value,
                        reason="无行情数据",
                    )
                if self.verbose:
                    logger.info(
                        f"买入延迟: {date.date()} {stock}, 原因: 无行情数据, "
                        f"目标市值: {target_value:.2f}"
                    )
                return
            tradeable, reason = is_tradeable(stock, trade_date_str, date_quote, action="buy")

            if not tradeable:
                # 不可交易，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action="buy",
                        current_date=date,
                        signal_date=signal_date or date,
                        target_value=target_value,
                        reason=reason,
                    )
                if self.verbose:
                    logger.info(
                        f"买入延迟: {date.date()} {stock}, 原因: {reason}, "
                        f"目标市值: {target_value:.2f}"
                    )
                return

        # 可交易，直接买入
        self._buy_stock_direct(date, stock, target_value, signal_date=signal_date)

    def _build_position_extra_info(self, date: pd.Timestamp, stock: str) -> Dict:
        """买入时附加额外元数据到 positions 字典（子类可覆写）

        默认返回空字典。engine_ml.py 覆写此方法以写入买入日 ATR 数据，
        供 _check_holding_periods 中的 ATR 动态止损使用。
        """
        return {}

    def _buy_stock_direct(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: Optional[pd.Timestamp] = None,
    ) -> None:
        """直接买入股票（不检查交易状态）

        内部使用，实际执行买入操作

        Args:
            date: 买入日期
            stock: 股票代码
            target_value: 目标市值
        """
        # 若已有持仓，跳过重复买入，避免覆盖持有期与成本基础导致计算错误
        if stock in self.positions:
            self._record_duplicate_buy_skip(
                stock=stock,
                buy_date=self.positions[stock]["buy_date"],
            )
            return

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
        shares = compute_lot_shares(target_value, trade_price)

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

            shares = compute_lot_shares(self.current_capital - cost, trade_price)
            if shares == 0:
                return
            amount = shares * trade_price
            cost = self.cost_model.calculate_buy_cost(amount)
            total_cost_cash = amount + cost

        # 与纸面交易一致：过小仓位买入拦截（按买入后市值阈值）
        min_buy_value_threshold = self._get_min_buy_value_threshold(date)
        if min_buy_value_threshold > 0 and amount < min_buy_value_threshold:
            if self.verbose:
                logger.warning(
                    f"股票 {stock} 买入后市值 {amount:.2f} 低于阈值 "
                    f"{min_buy_value_threshold:.2f}（ratio={self.min_buy_value_ratio:.2f}），跳过买入"
                )
            return

        # 建立新持仓（记录买入的成交价格和绩效价格）
        self.positions[stock] = {
            "shares": shares,
            "buy_date": date,
            "signal_date": signal_date or date,
            "buy_trade_price": trade_price,  # 成交价格（不复权）
            "buy_pnl_price": pnl_price,  # 绩效价格（后复权）
            "buy_cost_cash": total_cost_cash,  # 总现金支出（含手续费）
        }
        # 子类可覆写 _build_position_extra_info 以附加额外元数据（如 ATR）
        extra = self._build_position_extra_info(date, stock)
        if extra:
            self.positions[stock].update(extra)

        self.current_capital -= total_cost_cash

        # 记录交易
        self.trades.append(
            {
                "date": date,
                "signal_date": signal_date or date,
                "stock": stock,
                "action": "buy",
                "price": trade_price,  # 成交价格
                "shares": shares,
                "amount": amount,
                "cost": cost,
            }
        )

    def _add_to_position(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        buy_type: str = "risk_replenish",
        buy_reason: Optional[str] = None,
    ) -> float:
        """对**已有持仓**按目标金额加仓（部分买入；与风控减仓 `fraction` 对称）

        与 `_buy_stock_direct` 的差异（因为 `_buy_stock_direct` 对已持仓股票直接跳过）：
        - 只对已有持仓生效（不存在持仓时返回 0，不建仓）；
        - **不重置买入日/持有期**（加仓不改变退出计划）；`buy_cost_cash` 累加（加权平均成本）；
        - 不应用最小买入门槛（`min_buy_value_ratio`）：比例加仓语义优先，
          金额不足一手的股票由 `compute_lot_shares` 自然跳过；
        - 资金不足时按可用现金缩量（与 `_buy_stock_direct` 同一处理）。

        Args:
            date: 买入日期（T+1 开盘/收盘，与全局买卖时点一致）
            stock: 股票代码
            target_value: 目标加仓金额（元）
            buy_type: 买入类型标记（落盘到成交记录，供归因使用）
            buy_reason: 买入原因描述

        Returns:
            实际现金支出（含手续费）；未成交返回 0.0。
        """
        if stock not in self.positions or self.positions[stock]["shares"] <= 0:
            return 0.0
        if target_value <= 0:
            return 0.0

        trade_price = self._get_trade_price(date, stock)
        if trade_price is None:
            return 0.0

        shares = compute_lot_shares(target_value, trade_price)
        if shares == 0:
            return 0.0

        amount = shares * trade_price
        cost = self.cost_model.calculate_buy_cost(amount)
        total_cost_cash = amount + cost
        if total_cost_cash > self.current_capital:
            if self.current_capital <= cost:
                return 0.0
            shares = compute_lot_shares(self.current_capital - cost, trade_price)
            if shares == 0:
                return 0.0
            amount = shares * trade_price
            cost = self.cost_model.calculate_buy_cost(amount)
            total_cost_cash = amount + cost

        position = self.positions[stock]
        position["shares"] = int(position["shares"]) + int(shares)
        # 加权平均成本：累加现金支出（原买入日/持有期不变）
        position["buy_cost_cash"] = float(position.get("buy_cost_cash", 0.0)) + total_cost_cash
        self.current_capital -= total_cost_cash

        self.trades.append(
            {
                "date": date,
                "signal_date": date,
                "stock": stock,
                "action": "buy",
                "price": trade_price,
                "shares": shares,
                "amount": amount,
                "cost": cost,
                "buy_type": buy_type,
                "buy_reason": buy_reason,
                "buy_date": position.get("buy_date"),
            }
        )
        return float(total_cost_cash)

    def _buy_stock(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: Optional[pd.Timestamp] = None,
    ) -> None:
        """买入股票（在 T+1 日以收盘价买入）

        交易状态检查由调用方在执行阶段处理，这里只负责实际成交。

        Args:
            date: 买入日期（T+1）
            stock: 股票代码
            target_value: 目标市值
            signal_date: 触发本次买入计划的信号日期
        """
        # 直接买入，执行阶段的交易状态检查由上层调度负责。
        self._buy_stock_direct(date, stock, target_value, signal_date=signal_date)

    def _update_completion_attribution(
        self,
        original_signal_date: pd.Timestamp,
        ranking_date: pd.Timestamp,
        execution_date: pd.Timestamp,
        planned_stock: str,
        actual_stock: str,
        ranked_candidates: List[Tuple[str, float]],
    ) -> None:
        """用补位成交结果更新原始未成交槽位。"""
        candidate_rank = {
            stock: rank for rank, (stock, _score) in enumerate(ranked_candidates, start=1)
        }
        candidate_score = {stock: float(score) for stock, score in ranked_candidates}
        trade = next(
            (
                item
                for item in reversed(self.trades)
                if item.get("action") == "buy"
                and item.get("date") == execution_date
                and item.get("stock") == actual_stock
            ),
            {},
        )
        signal_price = self._get_trade_price(ranking_date, actual_stock)
        buy_price = trade.get("price")
        signal_to_buy_return = (
            float(buy_price / signal_price - 1.0)
            if buy_price is not None and signal_price not in (None, 0)
            else None
        )
        for record in reversed(self.execution_attribution_records):
            if (
                record.get("signal_date") == original_signal_date
                and record.get("planned_stock") == planned_stock
                and record.get("status") == "unfilled"
            ):
                record.update(
                    {
                        "ranking_date": ranking_date,
                        "execution_date": execution_date,
                        "execution_stage": "completion",
                        "actual_stock": actual_stock,
                        "actual_rank": candidate_rank.get(actual_stock),
                        "pred_score": candidate_score.get(actual_stock),
                        "status": "filled",
                        "buy_price": buy_price,
                        "signal_price": signal_price,
                        "signal_to_buy_return": signal_to_buy_return,
                    }
                )
                return
