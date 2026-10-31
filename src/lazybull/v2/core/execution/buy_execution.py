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
from src.lazybull.v2.core.accounting.ledger import BookingEntry, make_fill_key
from src.lazybull.v2.core.decision.buy_plan import (
    REASON_ALREADY_BOUGHT,
    REASON_EXECUTION_FAILED,
    fill_slots_from_candidates,
)
from src.lazybull.v2.core.decision.sizing import (
    compute_lot_shares,
    compute_min_buy_value_threshold,
)
from src.lazybull.v2.core.execution.completion import (  # noqa: F401 - 兼容旧导入
    _build_completion_stocks_to_try,
    _complete_slot_attempt,
    _completion_buy_loop,
    _record_completion_skip_for_slot,
)


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
        instruction_shares = signal_data.get("instruction_shares") or {}
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
        instruction_shares = {}

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
        "instruction_shares": {
            str(code): int(shares) for code, shares in instruction_shares.items()
        },
        "instruction_reason": signal_data.get("instruction_reason") if isinstance(signal_data, dict) else None,
        "completion_meta": signal_data.get("completion_meta") or {} if isinstance(signal_data, dict) else {},
        "tranche_tag": tranche_tag,
    }


def _sync_slot_weights_with_signals(
    signals: Dict, slot_weights: List[Dict], instruction_shares: Dict[str, int]
) -> List[Dict]:
    """执行日权重可能被风险预算调整，按原槽位顺序同步更新（D8 提取件，顺序不变）。"""
    if slot_weights:
        ordered_slot_weights = []
        seen_slot_stocks = set()
        for slot in slot_weights:
            slot_stock = slot.get("stock")
            if slot_stock in signals and slot_stock not in seen_slot_stocks:
                ordered_slot_weights.append(
                    {
                        "stock": slot_stock,
                        "weight": float(signals[slot_stock]),
                        "instruction_shares": instruction_shares.get(slot_stock),
                    }
                )
                seen_slot_stocks.add(slot_stock)
        for stock, weight in signals.items():
            if stock not in seen_slot_stocks:
                ordered_slot_weights.append(
                    {
                        "stock": stock,
                        "weight": float(weight),
                        "instruction_shares": instruction_shares.get(stock),
                    }
                )
    else:
        ordered_slot_weights = [
            {
                "stock": stock,
                "weight": float(weight),
                "instruction_shares": instruction_shares.get(stock),
            }
            for stock, weight in signals.items()
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

    slot_weights = _sync_slot_weights_with_signals(
        signals, slot_weights, parsed.get("instruction_shares") or {}
    )
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
        "instruction_reason": parsed.get("instruction_reason"),
        "completion_meta": parsed.get("completion_meta") or {},
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
    requested_shares: Optional[int] = None,
) -> bool:
    """执行买入并按成交结果登记（D8 提取件，顺序不变）。"""
    trades_before = len(engine.trades)

    engine._buy_stock(
        date,
        stock,
        target_value,
        signal_date=signal_date,
        requested_shares=requested_shares,
    )

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
            # 原槽位若携带固化增量股数，则是“存量加仓”，不是重复建仓。
            if (
                candidate_stock == slot.get("stock")
                and slot.get("instruction_shares") is not None
            ):
                return True, ""
            return False, "__held__"
        return _check_candidate_tradeable(
            blocked_tradeability_reasons, date_quote, trade_date_str, candidate_stock
        )

    def _execute_buy(candidate_stock: str, slot: Dict) -> bool:
        target_value = current_value * float(slot["weight"])
        buy_detail = _build_buy_detail(portfolio_value, candidate_stock, target_value)
        # 只有原槽位候选使用 T0 固化股数；同日顺延到其他候选时仍按原槽位权重预算。
        requested_shares = (
            slot.get("instruction_shares")
            if candidate_stock == slot.get("stock")
            else None
        )
        return _record_buy_execution(
            engine,
            date,
            signal_date,
            successful_buys,
            buy_detail,
            candidate_stock,
            target_value,
            requested_shares=requested_shares,
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

        per_stock = {}
        for item in remaining_unfilled_slots:
            meta = (ctx.get("completion_meta") or {}).get(item["stock"], {})
            per_stock[item["stock"]] = {
                "original_signal_date": meta.get("original_signal_date") or signal_date,
                "attempts": int(meta.get("attempts") or 0) + 1,
            }
        attempts = max((item["attempts"] for item in per_stock.values()), default=1)
        original_signal_date = next(
            (item["original_signal_date"] for item in per_stock.values()), signal_date
        )
        engine.unfilled_slots[signal_date] = {
            "unfilled_count": unfilled_count,
            "unfilled_slot_weights": remaining_unfilled_slots,
            "target_n": len(planned_slot_weights),
            "ranked_candidates": ranked_candidates,
            "signal_date": signal_date,
            "original_signal_date": original_signal_date,
            "completion_meta": per_stock,
            "first_attempt_date": date,
            "attempts": attempts,
            "last_reason": ctx.get("instruction_reason"),
            "reason_count": str(ctx.get("instruction_reason") or "").count(
                "（买入后市值过小）"
            )
            + 1,
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


def _size_buy_shares(
    *,
    target_value: float,
    requested_shares: Optional[int],
    trade_price: float,
    capital: float,
    cost_model,
) -> Optional[tuple[int, float, float, float]]:
    """按固化股数/权重预算和可用现金计算本次买入股数（legacy 缩量顺序）。"""
    if requested_shares is not None:
        if int(requested_shares) <= 0:
            return None
        target_value = float(int(requested_shares)) * trade_price
    shares = compute_lot_shares(target_value, trade_price)
    if shares == 0:
        return None
    amount = shares * trade_price
    cost = cost_model.calculate_buy_cost(amount)
    total_cost_cash = amount + cost
    if total_cost_cash > capital:
        if capital <= cost:
            return None
        shares = compute_lot_shares(capital - cost, trade_price)
        if shares == 0:
            return None
        amount = shares * trade_price
        cost = cost_model.calculate_buy_cost(amount)
        total_cost_cash = amount + cost
    return shares, amount, cost, total_cost_cash


def _build_residual_pending(
    engine, ranked
) -> tuple[Dict[str, float], List[Dict], int, Dict[str, Dict]]:
    """从残余槽位和当日排名构建补位信号（原 plan_residual_completion 内层）。"""
    pending: Dict[str, float] = {}
    slot_rows: List[Dict] = []
    completion_meta: Dict[str, Dict] = {}
    tranche_idx = 0
    for signal_date, slot in list(engine.unfilled_slots.items()):
        remaining = []
        per_stock_meta = slot.get("completion_meta") or {}
        for item in slot.get("unfilled_slot_weights") or []:
            item_meta = per_stock_meta.get(item["stock"], {})
            if not item_meta:
                item_meta = {
                    "original_signal_date": slot.get("original_signal_date") or signal_date,
                    "attempts": int(slot.get("attempts") or 0),
                }
            if int(item_meta.get("attempts") or 0) > 5:
                continue
            candidate = next(
                (
                    stock
                    for stock, _score in ranked
                    if stock not in engine.positions
                    and stock not in pending
                    and stock != item["stock"]
                ),
                None,
            )
            if candidate is None:
                remaining.append(item)
                continue
            pending[candidate] = float(item["weight"])
            slot_rows.append({"stock": candidate, "weight": float(item["weight"])})
            completion_meta[candidate] = {
                "original_signal_date": item_meta.get("original_signal_date") or signal_date,
                "attempts": int(item_meta.get("attempts") or 0),
            }
            tranche_idx = int(slot.get("tranche_idx") or 0)
        if remaining:
            slot["unfilled_slot_weights"] = remaining
            slot["unfilled_count"] = len(remaining)
        else:
            engine.unfilled_slots.pop(signal_date, None)
    return pending, slot_rows, tranche_idx, completion_meta


def _residual_reason(engine, date: pd.Timestamp, pending: Dict[str, float], quote) -> str:
    """补位失败回执原因（保持 legacy 展示语义）。"""
    portfolio_value = engine._calculate_portfolio_value(date)
    min_value = engine._get_min_buy_value_threshold(date)
    if min_value <= 0:
        min_value = compute_min_buy_value_threshold(
            total_assets=portfolio_value,
            target_count=len(engine.positions) + len(pending),
            ratio=0.2,
        )
    first_stock, first_weight = next(iter(pending.items()))
    first_price = 0.0
    price_row = quote[quote["ts_code"] == first_stock]
    if not price_row.empty:
        first_price = float(price_row.iloc[0].get("close") or 0.0)
    planned_shares = compute_lot_shares(portfolio_value * first_weight, first_price)
    too_small = (
        min_value > 0 and planned_shares > 0 and planned_shares * first_price < min_value
    )
    retry_count = max(
        (int(slot.get("reason_count") or 1) for slot in engine.unfilled_slots.values()),
        default=0,
    )
    if not too_small:
        return "补位槽位-信号生成"
    return "补位槽位-信号生成" + "（买入后市值过小）" * (retry_count + 1)


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

    def plan_residual_completion(self, date: pd.Timestamp) -> None:
        """T0 收盘：把 T1 残余槽位具体化为下一交易日补位信号。

        不复用隔日内部直买路径；候选按当日数据重新生成，权重继承原槽位，
        由宿主投影固化下一交易日股数。
        """
        if not self.unfilled_slots or self.price_data_cache is None:
            return
        trade_date_str = to_trade_date_str(date)
        quote = self.price_data_cache[self.price_data_cache["trade_date"] == trade_date_str]
        if quote.empty:
            return
        universe = self.universe.get_stocks(date, quote_data=quote)
        data = self._build_signal_data(date)
        if data is None:
            return
        ranked = self.signal.generate_ranked(date, universe, data)
        if not ranked:
            return
        if pd.Timestamp(date) in self.pending_signals:
            return
        pending, slot_rows, tranche_idx, completion_meta = _build_residual_pending(self, ranked)
        if not pending:
            return
        self.pending_signals[pd.Timestamp(date)] = {
            "signals": pending,
            "ranked_candidates": list(ranked),
            "priority_candidates": list(ranked),
            "slot_weights": slot_rows,
            "target_n": len(self.positions) + len(pending),
            "desired_position_count": len(self.positions) + len(pending),
            "tranche_idx": tranche_idx,
            "completion_meta": completion_meta,
            "instruction_reason": _residual_reason(self, date, pending, quote),
        }

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
        requested_shares: Optional[int] = None,
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

        # 已有持仓：只有 T1 固化增量指令才加仓；普通重复买入仍跳过。
        if stock in self.positions:
            if requested_shares is not None and int(requested_shares) > 0:
                self._add_to_position(
                    date,
                    stock,
                    float(int(requested_shares)) * trade_price,
                    buy_type="instruction_add",
                    buy_reason="T1 固化指令加仓",
                )
                return
            self._record_duplicate_buy_skip(
                stock=stock,
                buy_date=self.positions[stock]["buy_date"],
            )
            return

        # 获取绩效价格（后复权 close_adj）
        pnl_price = self._get_pnl_price(date, stock)
        if pnl_price is None:
            logger.warning(f"无法获取 {stock} 在 {date.date()} 的绩效价格，使用成交价格代替")
            pnl_price = trade_price

        sized = _size_buy_shares(
            target_value=target_value,
            requested_shares=requested_shares,
            trade_price=trade_price,
            capital=self.current_capital,
            cost_model=self.cost_model,
        )
        if sized is None:
            return
        shares, amount, cost, total_cost_cash = sized

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
        # P4-T1（D2）：入账唯一入口 = Ledger.apply_fills——原位完成全部数值
        # 计算后构造入账指令，现金/持仓/成交日志的变更仅经账本落账
        new_position = {
            "shares": shares, "buy_date": date, "signal_date": signal_date or date,
            "buy_trade_price": trade_price,  # 成交价格（不复权）
            "buy_pnl_price": pnl_price,  # 绩效价格（后复权）
            "buy_cost_cash": total_cost_cash,  # 总现金支出（含手续费）
        }
        # 子类可覆写 _build_position_extra_info 以附加额外元数据（如 ATR）
        extra = self._build_position_extra_info(date, stock)
        if extra:
            new_position.update(extra)
        trade_record = {
            "date": date, "signal_date": signal_date or date, "stock": stock, "action": "buy",
            "price": trade_price,  # 成交价格
            "shares": shares, "amount": amount, "cost": cost,
        }
        self.ledger.apply_fills(
            self,
            [
                BookingEntry(
                    fill_key=make_fill_key(date, "buy", stock, len(self.trades)),
                    date=date,
                    stock=stock,
                    action="buy",
                    cash_delta=-total_cost_cash,
                    trade_record=trade_record,
                    new_position=new_position,
                )
            ],
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
        # P4-T1（D2）：入账唯一入口 = Ledger.apply_fills——加仓股数/成本增量
        # 原位计算后交账本落账（买入日/持有期不变，buy_cost_cash 加权累加）
        trade_record = {
            "date": date, "signal_date": date, "stock": stock, "action": "buy",
            "price": trade_price, "shares": shares, "amount": amount, "cost": cost,
            "buy_type": buy_type, "buy_reason": buy_reason,
            "buy_date": position.get("buy_date"),
        }
        self.ledger.apply_fills(
            self,
            [
                BookingEntry(
                    fill_key=make_fill_key(date, "buy", stock, len(self.trades)),
                    date=date,
                    stock=stock,
                    action="buy",
                    cash_delta=-total_cost_cash,
                    trade_record=trade_record,
                    add_shares=int(shares),
                    add_cost_cash=float(total_cost_cash),
                )
            ],
        )
        return float(total_cost_cash)

    def _buy_stock(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: Optional[pd.Timestamp] = None,
        requested_shares: Optional[int] = None,
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
        self._buy_stock_direct(
            date,
            stock,
            target_value,
            signal_date=signal_date,
            requested_shares=requested_shares,
        )

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