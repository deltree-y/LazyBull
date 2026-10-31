# -*- coding: utf-8 -*-
"""v2 补位规划辅助函数（从 buy_execution 拆出，行为不变）。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.v2.common.date_utils import to_trade_date_str
from src.lazybull.v2.common.trade_status import is_tradeable
from src.lazybull.v2.core.decision.sizing import compute_lot_shares


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
