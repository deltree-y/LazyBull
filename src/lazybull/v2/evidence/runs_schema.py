# -*- coding: utf-8 -*-
"""runs 产物契约列集（`docs/contracts/runs_artifact_contract.md`）的单一代码来源。

转换器（runs_convert）按本模块列集对齐目标产物（多列丢弃、缺列补 NA 并登记），
读入桥（runs_loader）按本模块列集做契约 §9.7 硬校验（缺列/多列报错）。
两侧共用本模块，禁止各自重写列集。
"""

from __future__ import annotations

from typing import Dict, List

#: chain_nav 契约列集（契约 §4）
CHAIN_NAV_CONTRACT_COLS: List[str] = ["date", "nav", "split_index"]

#: summary 必需列子集（契约 §3 标识/窗口/指标/血缘组；summary 全列随配置扩展，不做全等校验）
SUMMARY_REQUIRED_COLS: List[str] = [
    "batch_id", "split_index", "test_start", "test_end", "bt_total_return", "data_state_id",
]

#: trades 契约列集（契约 §5 + F2 补 4 列 + v2 新增 2 列）
TRADES_CONTRACT_COLS: List[str] = [
    "wf_run_id", "split_index", "model_version", "trade_date", "signal_date",
    "ts_code", "action", "price", "shares", "amount", "cost",
    "buy_date", "buy_price", "buy_pnl_price", "sell_pnl_price",
    "pnl_profit_amount", "pnl_profit_pct", "sell_type", "sell_timing",
    "sell_reason", "trigger_type", "buy_type", "buy_reason",  # F2 补（历史批源缺，转换补 NA）
    "lot_id", "tranche_idx",  # v2 新增（tranche_idx 可空）
]

#: attribution 契约列集（契约 §6，含 F2 补头部三列）
ATTR_CONTRACT_COLS: List[str] = [
    "wf_run_id", "split_index", "model_version", "signal_date", "ranking_date",
    "execution_date", "execution_stage", "tranche_idx", "planned_ts_code",
    "actual_ts_code", "planned_rank", "actual_rank", "pred_score", "target_weight",
    "status", "reason", "buy_price", "signal_price", "signal_to_buy_return",
]

#: topk_detail 契约列集（契约 §7.1）
TOPK_CONTRACT_COLS: List[str] = [
    "wf_run_id", "split_index", "test_start", "test_end", "model_version",
    "trade_date", "topk", "rank", "ts_code", "pred_score", "true_return",
    "score_column", "ml_score", "risk_score", "final_score",
]

#: 持仓快照契约列集（契约 §7，中文映射后的 ASCII 列）
SNAP_CONTRACT_COLS: List[str] = [
    "run_id", "split_index", "model_version", "trade_date", "ts_code",
    "shares", "market_value", "weight", "total_value", "buy_date",
    "signal_date", "held_days", "due_date", "remaining_days",
]

#: daily 契约列集（契约 §8；转换器重建产物）
DAILY_CONTRACT_COLS: List[str] = [
    "trade_date", "total_value", "cash", "market_value", "nav", "daily_return",
    "n_positions", "n_buys", "n_sells", "turnover_amount", "exposure_lambda",
]

#: 折内文件名 → 契约列集（loader §9.7 校验用）
FOLD_FILE_CONTRACT_COLS: Dict[str, List[str]] = {
    "trades": TRADES_CONTRACT_COLS,
    "attribution": ATTR_CONTRACT_COLS,
    "topk_detail": TOPK_CONTRACT_COLS,
    "holdings_snapshot": SNAP_CONTRACT_COLS,
    "daily": DAILY_CONTRACT_COLS,
}
