#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward OOS 回测薄组装（v2 侧）。

P3-T3 处置（§2.2 backtest 行）：**不整组搬运**——OOS 回测改指 ``v2/hosts/backtest``
（P2a 内核），本模块只保留参数映射薄组装。依赖方向合规（协议 §7：train 不含
hosts）——``replay.run_oos_backtest`` 由**宿主注入**（``OrchestratorHostConfig.
backtest_runner``），本模块仅做 kwargs 组装，train 侧零 hosts import。

退役口径：exposure_* 五参数 / downside_penalty* 不进入映射（退役家族，§17.6
守卫在编排入口拒绝）；v2 内核侧另有 D6 fail-fast 兜底。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class BacktestHostParams:
    """宿主回测参数组（§13.3.1 ④：不进训练协议，由宿主配置承载）。"""

    oos_backtest: bool = True
    oos_backtest_months: int = 0  # 0 = 编排器对齐 test_window_months(旧 runner 同口径)
    bt_top_n: int = 30
    bt_rebalance_freq: Optional[int] = None
    bt_initial_capital: float = 1000000.0
    bt_sell_timing: str = "open"
    bt_exclude_st: bool = True
    bt_min_list_days: int = 365
    bt_max_weight_per_stock: Optional[float] = None
    bt_max_per_industry: Optional[int] = None
    bt_stop_loss_enabled: bool = False
    bt_stop_loss_drawdown_pct: float = 30.0
    bt_stop_loss_consecutive_limit_down: int = 2
    position_sizing: str = "equal"
    kelly_vol_window: int = 60
    kelly_max_leverage: float = 0.25
    stagger_tranches: int = 1
    enable_early_rebalance_on_empty: bool = True


def build_oos_backtest_kwargs(
    bt: BacktestHostParams,
    *,
    model_version: int,
    bt_start: str,
    bt_end: str,
    storage: Any,
    loader: Any,
    trade_cal: pd.DataFrame,
    stock_basic: pd.DataFrame,
    label_column: str,
    data_root: Optional[str],
    persistent_signal: Any,
    split_index: int,
    stock_domain: str,
    runtime_lookups: Optional[Dict[str, Any]] = None,
    feature_source: str = "panel",
    store: Any = None,
    acceptance_purpose: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """旧 33 参调用形态 → v2 replay kwargs 的参数映射（逐键对应旧 runner 调用点）。

    ``runtime_lookups``:cs_train_verify 模式的四族数据查询表（键 =
    holdertrade_lookup / repurchase_lookup / top10fh_panel / top_inst_lookup）;
    panel 模式必须为空（replay 侧防双写校验）。
    ``feature_columns``:panel 模式列集收窄（P3-T3 P2-4，折模型实际特征列）。
    """
    lookups = dict(runtime_lookups or {})
    return dict(
        model_version=model_version,
        bt_start=bt_start,
        bt_end=bt_end,
        storage=storage,
        loader=loader,
        trade_cal=trade_cal,
        stock_basic=stock_basic,
        label_column=label_column,
        bt_top_n=bt.bt_top_n,
        bt_rebalance_freq=bt.bt_rebalance_freq,
        data_root=data_root,
        persistent_signal=persistent_signal,
        bt_exclude_st=bt.bt_exclude_st,
        bt_min_list_days=bt.bt_min_list_days,
        bt_sell_timing=bt.bt_sell_timing,
        bt_max_weight_per_stock=bt.bt_max_weight_per_stock,
        bt_max_per_industry=bt.bt_max_per_industry,
        bt_stop_loss_enabled=bt.bt_stop_loss_enabled,
        bt_stop_loss_drawdown_pct=bt.bt_stop_loss_drawdown_pct,
        bt_stop_loss_consecutive_limit_down=bt.bt_stop_loss_consecutive_limit_down,
        position_sizing=bt.position_sizing,
        kelly_vol_window=bt.kelly_vol_window,
        kelly_max_leverage=bt.kelly_max_leverage,
        stagger_tranches=bt.stagger_tranches,
        enable_early_rebalance_on_empty=bt.enable_early_rebalance_on_empty,
        initial_capital=bt.bt_initial_capital,
        split_num=split_index,
        stock_domain=stock_domain,
        holdertrade_lookup=lookups.get("holdertrade_lookup"),
        repurchase_lookup=lookups.get("repurchase_lookup"),
        top10fh_panel=lookups.get("top10fh_panel"),
        top_inst_lookup=lookups.get("top_inst_lookup"),
        feature_source=feature_source,
        store=store,
        acceptance_purpose=acceptance_purpose,
        feature_columns=feature_columns,
    )


def merge_backtest_metrics(result: Dict[str, Any], bt_metrics: Dict[str, Any]) -> None:
    """回测返回下划线键 pop 挂到 result（旧 runner :624-637 段逐字语义）。"""
    nav_curve = bt_metrics.pop("_nav_curve", None)
    if nav_curve is not None:
        result["_nav_curve"] = nav_curve
    trades = bt_metrics.pop("_trades", None)
    if trades is not None:
        result["_trades"] = trades
    execution_attribution = bt_metrics.pop("_execution_attribution", None)
    if execution_attribution is not None:
        result["_execution_attribution"] = execution_attribution
    holdings_snapshot = bt_metrics.pop("_holdings_snapshot", None)
    if holdings_snapshot is not None:
        result["_holdings_snapshot"] = holdings_snapshot
    result["bt_metrics"] = bt_metrics
