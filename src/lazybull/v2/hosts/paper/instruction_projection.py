# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主指令投影纯函数族（自 state_store.py 拆分，行为不变）。

拆分缘由（§7.4「新增功能对应新增文件」+ 复杂度门 1200 行文件硬上限，2026-10-28
T2 评审处置）：T0 产出 → 旧 ``instructions/<T+1>.parquet`` 布局的投影是纯函数族，
与 PaperStateStore 的落盘/锁/恢复职责正交。``state_store`` 经 import 重导出
``build_instruction_frame`` / ``INSTRUCTION_COLUMNS`` 等公开名，导入面不变。

共享小件：``_yyyymmdd``（日期值 ⇒ YYYYMMDD 文本）随本模块落定，state_store
单向 import（本模块不 import state_store，无环）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.common.constants import SHARE_LOT_SIZE
from src.lazybull.common.date_utils import calc_holding_trade_days
from src.lazybull.v2.core.decision.sizing import compute_lot_shares

__all__ = [
    "INSTRUCTION_COLUMNS",
    "DEFAULT_MAX_REPLACEMENT_ATTEMPTS",
    "build_instruction_frame",
]

#: instructions/<T+1>.parquet 列序契约（12 列；读侧兼容旧 10 列形态）
INSTRUCTION_COLUMNS: Tuple[str, ...] = (
    "ts_code",
    "action",
    "shares",
    "price_type",
    "reason",
    "source_date",
    "target_weight",
    "original_signal_date",
    "desired_position_count",
    "retry_attempt",
    "replacement_slot_code",
    "keep_buy_date",
)

#: 指令列缺省值（旧 10 列文件补列 / 规范化用）
_INSTRUCTION_COLUMN_DEFAULTS: Dict[str, Any] = {
    "ts_code": "",
    "action": "",
    "shares": 0,
    "price_type": "",
    "reason": "",
    "source_date": "",
    "target_weight": 0.0,
    "original_signal_date": "",
    "desired_position_count": 0,
    "retry_attempt": 0,
    "replacement_slot_code": "",
    "keep_buy_date": False,
}

#: 补位上限默认（对齐旧 MAX_REPLENISHMENT_ATTEMPTS=5 / PendingOrderManager 过期口径）
DEFAULT_MAX_REPLACEMENT_ATTEMPTS = 5

#: 条件卖出 sell_type ⇒ 旧链指令 reason 文本（entry 自带 reason 时优先）
_CONDITION_SELL_REASONS: Dict[str, str] = {"rebalance": "调仓卖出", "holding_period": "持有期到期"}


def _yyyymmdd(value: Any) -> str:
    """日期值 ⇒ YYYYMMDD 字符串（pd.Timestamp / datetime / str / None 防御；空 ⇒ ""）。"""
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return "" if pd.isna(value) else value.strftime("%Y%m%d")
    if hasattr(value, "strftime") and not isinstance(value, str):
        return value.strftime("%Y%m%d")
    return str(value).strip().replace("-", "").replace("/", "")


def _new_instruction_row(
    ts_code: str,
    action: str,
    shares: int,
    price_type: str,
    reason: str,
    source_date: str,
    *,
    target_weight: float = 0.0,
    original_signal_date: str = "",
    desired_position_count: int = 0,
    retry_attempt: int = 0,
    replacement_slot_code: str = "",
    keep_buy_date: bool = False,
) -> Dict[str, Any]:
    """构造 12 列指令行（keep_buy_date 恒 False 默认——暴露政策族默认关）。"""
    return {
        "ts_code": str(ts_code),
        "action": str(action),
        "shares": int(shares),
        "price_type": str(price_type),
        "reason": str(reason),
        "source_date": str(source_date),
        "target_weight": float(target_weight),
        "original_signal_date": str(original_signal_date),
        "desired_position_count": int(desired_position_count),
        "retry_attempt": int(retry_attempt),
        "replacement_slot_code": str(replacement_slot_code),
        "keep_buy_date": bool(keep_buy_date),
    }


def _position_lot_shares(positions: Dict[str, Dict[str, Any]], ts_code: str) -> int:
    """持仓整手可卖股数（不足一手 ⇒ 0；与旧链 ``(shares // 100) * 100`` 同口径）。"""
    info = positions.get(ts_code)
    if not info:
        return 0
    return int(info.get("shares", 0) or 0) // SHARE_LOT_SIZE * SHARE_LOT_SIZE


def _holding_period_reason(
    position: Optional[Dict[str, Any]], entry: Dict[str, Any], trade_dates: Optional[List[str]]
) -> Optional[str]:
    """持有期到期 reason（旧链口径 ``f"持有期到期: 持有{holding_days}天"``，holding_days=交易日数）。

    口径 = ``calc_holding_trade_days(buy_date, trigger_date, trade_dates)``（trigger_date = T0
    触发日，与旧链 T0 评估日一致；T2 评审处置补——门 5 输出级比较含 reason 字段）。
    ``buy_date``/``trade_dates`` 不可得 ⇒ None（回退静态文本，差异登记）。
    """
    if not trade_dates or not isinstance(position, dict):
        return None
    buy_date = _yyyymmdd(position.get("buy_date"))
    trigger = _yyyymmdd(entry.get("trigger_date"))
    if not buy_date or not trigger:
        return None
    holding_days = calc_holding_trade_days(buy_date, trigger, list(trade_dates))
    return f"持有期到期: 持有{holding_days}天"


def _rows_from_sell_queues(
    condition_sells: Dict[str, Dict[str, Any]],
    stop_loss_sells: Dict[str, Dict[str, Any]],
    positions: Dict[str, Dict[str, Any]],
    source_date: str,
    sell_price_type: str,
    trade_dates: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """条件卖出 + 止损卖出队列 → 卖出指令行（先卖后买的卖出段）。"""
    rows: List[Dict[str, Any]] = []
    for ts_code, entry in condition_sells.items():
        shares = _position_lot_shares(positions, ts_code)
        if shares <= 0:
            continue
        sell_type = str(entry.get("sell_type", "") or "")
        reason = entry.get("reason")
        if not reason and sell_type == "holding_period":
            reason = _holding_period_reason(positions.get(ts_code), entry, trade_dates)
        if not reason:
            reason = _CONDITION_SELL_REASONS.get(sell_type, sell_type or "条件卖出")
        rows.append(
            _new_instruction_row(ts_code, "sell", shares, sell_price_type, str(reason), source_date)
        )
    for ts_code, entry in stop_loss_sells.items():
        shares = _position_lot_shares(positions, ts_code)
        if shares <= 0:
            continue
        reason = str(entry.get("reason") or "止损卖出")
        rows.append(
            _new_instruction_row(ts_code, "sell", shares, sell_price_type, reason, source_date)
        )
    return rows


def _order_field(order: Any, name: str, default: Any = None) -> Any:
    """延迟订单字段读取（兼容 PendingOrder 对象与 dict 两形态）。"""
    if isinstance(order, dict):
        return order.get(name, default)
    return getattr(order, name, default)


def _rows_from_delayed_orders(
    delayed_orders: List[Any],
    positions: Dict[str, Dict[str, Any]],
    prices: Dict[str, float],
    source_date: str,
    sell_price_type: str,
    buy_price_type: str,
    max_replacement_attempts: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """当日延迟订单 → (卖出行列表, 补位买入行列表)（补位上限 = retry_count 超上限不再生成）。

    映射登记（不确定点见 state_store 模块 docstring）：补位买入 ``target_weight`` 恒 0.0、
    ``replacement_slot_code`` 恒 ``""``、``reason`` ← last_reason（缺省 "延迟订单"）、
    ``original_signal_date`` ← 订单 signal_date。
    """
    sell_rows: List[Dict[str, Any]] = []
    buy_rows: List[Dict[str, Any]] = []
    for order in delayed_orders:
        ts_code = str(_order_field(order, "stock", "") or "")
        action = str(_order_field(order, "action", "") or "")
        if not ts_code:
            continue
        retry = int(_order_field(order, "retry_count", 0) or 0)
        reason = str(_order_field(order, "last_reason", None) or "延迟订单")
        if action == "sell":
            shares = _position_lot_shares(positions, ts_code)
            if shares > 0:
                sell_rows.append(
                    _new_instruction_row(
                        ts_code,
                        "sell",
                        shares,
                        sell_price_type,
                        reason,
                        source_date,
                        retry_attempt=retry,
                    )
                )
            continue
        if action != "buy":
            continue
        if retry > max_replacement_attempts:
            logger.warning(
                f"补位买入超上限（retry={retry} > {max_replacement_attempts}），"
                f"不再生成补位指令: {ts_code}"
            )
            continue
        price = float(prices.get(ts_code, 0.0) or 0.0)
        target_value = float(_order_field(order, "target_value", 0.0) or 0.0)
        shares = compute_lot_shares(target_value, price, SHARE_LOT_SIZE)
        if shares <= 0:
            continue
        buy_rows.append(
            _new_instruction_row(
                ts_code,
                "buy",
                shares,
                buy_price_type,
                reason,
                source_date,
                original_signal_date=_yyyymmdd(_order_field(order, "signal_date", None)),
                retry_attempt=retry,
            )
        )
    return sell_rows, buy_rows


def _rows_from_buy_plan(
    buy_plan: Dict[str, Any],
    positions: Dict[str, Dict[str, Any]],
    prices: Dict[str, float],
    total_capital: float,
    capital_fraction: float,
    buy_price_type: str,
    source_date: str,
) -> List[Dict[str, Any]]:
    """买入计划（pending_signals 条目）→ 买入指令行（旧 ``_generate_instructions`` 口径）。

    口径：预算 = ``total_capital × capital_fraction``（宿主供给；旧链 = 账户总价值
    × (1 − 留存比) × 分批资金比例）；目标股数 = ``compute_lot_shares(预算 × 权重,
    价格)``；与现持仓差额再取整手 ⇒ 买入股数；指令 ``target_weight`` = 权重 ×
    capital_fraction（旧链组合权重口径）。
    """
    signals = buy_plan.get("signals") or {}
    if not isinstance(signals, dict) or not signals:
        return []
    desired_position_count = int(buy_plan.get("desired_position_count") or len(signals))
    budget = float(total_capital) * float(capital_fraction)
    rows: List[Dict[str, Any]] = []
    for ts_code, weight in signals.items():
        price = float(prices.get(ts_code, 0.0) or 0.0)
        if price <= 0:
            logger.warning(f"股票 {ts_code} 无价格数据，跳过生成买入指令投影")
            continue
        target_shares = compute_lot_shares(budget * float(weight), price, SHARE_LOT_SIZE)
        current_shares = int((positions.get(ts_code) or {}).get("shares", 0) or 0)
        shares = (target_shares - current_shares) // SHARE_LOT_SIZE * SHARE_LOT_SIZE
        if shares <= 0:
            continue
        rows.append(
            _new_instruction_row(
                ts_code,
                "buy",
                shares,
                buy_price_type,
                str(buy_plan.get("instruction_reason") or "信号生成"),
                source_date,
                target_weight=float(weight) * float(capital_fraction),
                original_signal_date=source_date,
                desired_position_count=desired_position_count,
            )
        )
    return rows


def _merge_instruction_rows(
    existing_rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """指令行合并：去重键 (action, ts_code)，**既有优先**（旧链合并语义复刻）。"""
    merged: List[Dict[str, Any]] = []
    seen = set()
    for row in [*existing_rows, *new_rows]:
        key = (row["action"], row["ts_code"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _normalize_instruction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """指令帧规范化为 12 列契约（读侧向后兼容旧 10 列形态：缺列补默认值）。"""
    out = frame.copy()
    for column in INSTRUCTION_COLUMNS:
        if column not in out.columns:
            out[column] = _INSTRUCTION_COLUMN_DEFAULTS[column]
    out = out.loc[:, list(INSTRUCTION_COLUMNS)]
    # NaN 防御（旧 10 列文件读出 / 合并路径的缺失值）
    out["replacement_slot_code"] = out["replacement_slot_code"].map(
        lambda value: str(value) if pd.notna(value) else ""
    )
    out["keep_buy_date"] = out["keep_buy_date"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    return out


def build_instruction_frame(
    *,
    signal_date: Any,
    buy_plan: Optional[Dict[str, Any]] = None,
    condition_sells: Optional[Dict[str, Dict[str, Any]]] = None,
    stop_loss_sells: Optional[Dict[str, Dict[str, Any]]] = None,
    delayed_orders: Optional[List[Any]] = None,
    positions: Optional[Dict[str, Dict[str, Any]]] = None,
    prices: Optional[Dict[str, float]] = None,
    total_capital: float = 0.0,
    capital_fraction: float = 1.0,
    buy_price_type: str = "close",
    sell_price_type: str = "open",
    max_replacement_attempts: int = DEFAULT_MAX_REPLACEMENT_ATTEMPTS,
    existing: Optional[pd.DataFrame] = None,
    trade_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """T0 产出 → T+1 指令帧（12 列；六语义见 state_store 模块 docstring，纯函数）。

    输入 = 引擎当日 T0 产出：``buy_plan``（``pending_signals[date]`` 条目，新格式含
    ``signals``/``desired_position_count``）+ ``condition_sells`` + ``stop_loss_sells``
    + ``delayed_orders``（当日延迟订单，PendingOrder 或 dict）；``signal_date`` ⇒ 指令
    ``source_date`` 列；``existing`` = 既有 T+1 指令帧（合并**既有优先**，去重键
    (action, ts_code)）；``trade_dates`` = 开市交易日列表（持有期到期 reason 的持有天数
    口径，旧链 ``calc_holding_trade_days`` 同源；缺省 ⇒ 回退静态文本，差异登记）。
    文件名 = T+1 执行日由 ``commit_day(instructions_date=...)`` 承载，本函数只产帧。
    返回 12 列 DataFrame（可能为空帧；空帧由 commit_day 跳过落盘）。
    """
    positions = positions or {}
    prices = prices or {}
    source_date = _yyyymmdd(signal_date)
    sell_rows = _rows_from_sell_queues(
        condition_sells or {},
        stop_loss_sells or {},
        positions,
        source_date,
        sell_price_type,
        trade_dates,
    )
    delayed_sell_rows, delayed_buy_rows = _rows_from_delayed_orders(
        list(delayed_orders or []),
        positions,
        prices,
        source_date,
        sell_price_type,
        buy_price_type,
        max_replacement_attempts,
    )
    buy_rows = (
        _rows_from_buy_plan(
            buy_plan,
            positions,
            prices,
            total_capital,
            capital_fraction,
            buy_price_type,
            source_date,
        )
        if buy_plan
        else []
    )
    # 先卖后买（卖出段 = 条件卖出 → 止损卖出 → 延迟卖出；买入段 = 计划买入 → 补位买入）
    new_rows = [*sell_rows, *delayed_sell_rows, *buy_rows, *delayed_buy_rows]
    existing_rows: List[Dict[str, Any]] = []
    if existing is not None:
        existing_rows = _normalize_instruction_frame(existing).to_dict("records")
    return pd.DataFrame(
        _merge_instruction_rows(existing_rows, new_rows), columns=list(INSTRUCTION_COLUMNS)
    )


def inject_instruction_shares(blob: Dict[str, Any], frame: Any) -> None:
    """把完整 T0 指令计划写入权威 pending；纯卖/空买也必须有行集。"""
    if frame is None or getattr(frame, "empty", True):
        return
    fields = blob.setdefault("fields", {})
    pending = fields.get("pending_signals") or {}
    keys = {str(pd.Timestamp(key).strftime("%Y%m%d")): key for key in pending}
    rows = frame.to_dict("records")
    signal_date = next(
        (
            row.get("original_signal_date") or row.get("source_date")
            for row in rows
            if row.get("original_signal_date") or row.get("source_date")
        ),
        None,
    )
    if signal_date is None:
        return
    key = keys.get(str(signal_date).replace("-", "")[:8])
    if key is None:
        key = pd.Timestamp(str(signal_date))
        pending[key] = {
            "signals": {},
            "ranked_candidates": [],
            "priority_candidates": [],
            "slot_weights": [],
            "target_n": 0,
            "desired_position_count": 0,
            "tranche_idx": 0,
        }
    entry = pending[key]
    for row in rows:
        if row.get("action") != "buy":
            continue
        shares = int(row.get("shares") or 0)
        if shares <= 0:
            continue
        entry.setdefault("instruction_shares", {})[str(row["ts_code"])] = shares
        entry["instruction_price_type"] = str(row.get("price_type") or "")
    # 完整行集（纯买/纯卖/混合）入权威计划，恢复直接返回，不再重算。
    entry["instruction_rows"] = json.loads(frame.to_json(orient="records", date_format="iso"))
    fields["pending_signals"] = pending


def resolve_t0_status(engine: Any, corrected_ts: Any, was_signal_day: bool) -> str:
    """T0 状态口径：成功 = pending_signals 含当日调仓计划（带 decision_trace）。"""
    entry = engine.pending_signals.get(corrected_ts)
    if isinstance(entry, dict) and "decision_trace" in entry:
        return "success"
    return "no_targets" if was_signal_day else "not_rebalance_day"
