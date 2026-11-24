# -*- coding: utf-8 -*-
"""回测买入执行的 pending 信号解析辅助（从 buy_execution.py 提取式拆分）。

``buy_execution.py`` 是 P2a-T5 行为冻结复制件，文件行数超硬上限 1200 ⇒
按既有 D8「提取模块级私有 helper」模式，把 pending 信号解析三件
（取信号 / 信号解析 / 槽位权重同步）整体搬出；函数体逐字保留，公开
签名、数值操作顺序、日志文本逐字不变，行为逐位一致。
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd


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
        "instruction_reason": (
            signal_data.get("instruction_reason") if isinstance(signal_data, dict) else None
        ),
        "completion_meta": (
            (signal_data.get("completion_meta") or {})
            if isinstance(signal_data, dict)
            else {}
        ),
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
