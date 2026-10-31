# -*- coding: utf-8 -*-
"""纸面 v2 手工调整的纯状态规则；唯一实现供 CLI 与迁移验收共用。"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.lazybull.v2.core.accounting.engine_state import EngineState


def apply_adjustment(
    state: EngineState, *, command: str, trade_date: str, values: Mapping[str, Any]
) -> EngineState:
    """按 legacy 现金/持仓公式调整权威状态；不直接改任何投影文件。"""
    if command == "cash":
        state.current_capital = float(values["set"])
        return state
    code = str(values["ts_code"])
    if code not in state.positions:
        raise RuntimeError(f"持仓不存在: {code}")
    position = state.positions[code]
    if command == "delete-position":
        state.current_capital += int(position["shares"]) * float(position["buy_trade_price"])
        del state.positions[code]
        return state
    if command == "update-position":
        old_cost = int(position["shares"]) * float(position["buy_trade_price"])
        new_cost = int(values["shares"]) * float(values["buy_price"])
        state.current_capital += old_cost - new_cost
        position["shares"] = int(values["shares"])
        position["buy_trade_price"] = float(values["buy_price"])
        position["buy_cost_cash"] = new_cost
        return state
    if command == "add-shares":
        add_shares = int(values["shares"])
        add_cost = add_shares * float(values["price"])
        if add_cost > state.current_capital:
            raise RuntimeError("加仓现金不足")
        old_shares = int(position["shares"])
        new_shares = old_shares + add_shares
        old_price = float(position["buy_trade_price"])
        new_price = (old_price * old_shares + float(values["price"]) * add_shares) / new_shares
        state.current_capital -= add_cost
        # 与旧 CLI 一致：既有持仓加仓保留原买入日，不改变持有期到期时点。
        position.update(
            {
                "shares": new_shares,
                "buy_trade_price": new_price,
                "buy_cost_cash": new_shares * new_price,
            }
        )
        return state
    raise ValueError(f"未知 adjust 命令: {command}")
