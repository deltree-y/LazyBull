"""v2 P4 T4 双宿主共享输入、连续参照与逐笔/NAV/持仓比较工具（阶段专用）。

连续参照由回测宿主的配置与引擎装配函数创建；paper_replay 只提供其真实逐日
装配的最终特征和行情，不提供、反投影或改写任何指令/成交。T3 证明性订单投影
在本通路中不可达。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.lazybull.v2.core.accounting.exports import get_execution_attribution, get_trades
from src.lazybull.v2.hosts.backtest.replay import (
    _assemble_replay_engine,
    _build_replay_trading_config,
    _compute_replay_metrics,
)
from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch
from src.lazybull.v2.hosts.paper.host import PaperHost
from src.lazybull.v2.hosts.paper.replay import execution_config_from_engine

TRADE_COLUMNS = [
    "trade_date",
    "signal_date",
    "ts_code",
    "action",
    "price",
    "shares",
    "amount",
    "cost",
    "sell_type",
]
POSITION_COLUMNS = [
    "trade_date",
    "ts_code",
    "shares",
    "market_value",
    "weight",
    "total_value",
    "buy_date",
    "signal_date",
    "held_days",
    "due_date",
    "remaining_days",
]


class _PersistentSignalAdapter:
    """为无版本更新接口的测试信号补齐回测宿主 persistent_signal 协议。"""

    def __init__(self, signal: Any) -> None:
        object.__setattr__(self, "_signal", signal)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._signal, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._signal, name, value)

    def update_model_version(self, version: Any) -> None:
        self._signal.model_version = version


def _scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    if isinstance(value, pd._libs.missing.NAType):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return _scalar(value.item())
        except (ValueError, TypeError):
            pass
    return value


def _records(frame: Optional[pd.DataFrame], columns: Sequence[str]) -> List[Tuple[Any, ...]]:
    if frame is None or frame.empty:
        return []
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"比较帧缺列: {missing}")
    return [
        tuple(_scalar(value) for value in row)
        for row in frame[list(columns)].itertuples(index=False, name=None)
    ]


def canonical_trades(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    """引擎原始或 runs 成交帧统一为门 2/门 3 九列比较口径。"""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    renamed = frame.rename(columns={"date": "trade_date", "stock": "ts_code"}).copy()
    # 引擎买入行天然不带卖出字段；与 runs_writer 的出口口径一致，缺列补 NA 后比较。
    for column in TRADE_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    for column in ("trade_date", "signal_date"):
        renamed[column] = renamed[column].map(
            lambda value: "" if _scalar(value) is None else str(_scalar(value)).replace("-", "")
        )
    return renamed[TRADE_COLUMNS]


def canonical_nav(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(columns={"date": "trade_date"})[["trade_date", "nav"]].copy()
    result["trade_date"] = result["trade_date"].map(
        lambda value: str(_scalar(value)).replace("-", "")
    )
    return result.sort_values("trade_date").reset_index(drop=True)


def canonical_positions(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)
    renamed = frame.rename(
        columns={
            "date": "trade_date",
            "portfolio_value": "total_value",
            "holding_days": "held_days",
            "planned_exit_date": "due_date",
            "remaining_intervals": "remaining_days",
        }
    ).copy()
    for column in ("trade_date", "buy_date", "signal_date", "due_date"):
        renamed[column] = renamed[column].map(
            lambda value: "" if _scalar(value) is None else str(_scalar(value)).replace("-", "")
        )
    return renamed[POSITION_COLUMNS].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def compare_trades(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    selected_count: int,
    allow_tie_swaps: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """逐笔九列比较；并列互换单列，门 3 超入选数 5% 必须判分辨率不足。"""
    left_rows, right_rows = _records(left, TRADE_COLUMNS), _records(right, TRADE_COLUMNS)
    max_len = max(len(left_rows), len(right_rows))
    detail_rows = []
    for index in range(max_len):
        lrow = left_rows[index] if index < len(left_rows) else None
        rrow = right_rows[index] if index < len(right_rows) else None
        detail = {"row_index": index, "ordered_equal": lrow == rrow}
        for column_index, column in enumerate(TRADE_COLUMNS):
            detail[f"left_{column}"] = lrow[column_index] if lrow is not None else None
            detail[f"right_{column}"] = rrow[column_index] if rrow is not None else None
        detail_rows.append(detail)
    ordered_identical = left_rows == right_rows
    multiset_equal = Counter(left_rows) == Counter(right_rows)
    positional_differences = sum(l != r for l, r in zip(left_rows, right_rows))
    tie_swaps = positional_differences if multiset_equal and not ordered_identical else 0
    threshold = float(selected_count) * 0.05
    resolution_insufficient = bool(allow_tie_swaps and tie_swaps > threshold)
    passed = multiset_equal and (
        not resolution_insufficient if allow_tie_swaps else ordered_identical
    )
    summary = {
        "passed": passed,
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "compared_rows": max_len,
        "ordered_identical_rows": sum(row["ordered_equal"] for row in detail_rows),
        "ordered_identical": ordered_identical,
        "multiset_equal": multiset_equal,
        "tie_swap_count": tie_swaps,
        "selected_count": int(selected_count),
        "tie_swap_threshold": threshold,
        "resolution_insufficient": resolution_insufficient,
        "difference_examples": [row for row in detail_rows if not row["ordered_equal"]][:10],
    }
    return summary, pd.DataFrame(detail_rows)


def compare_nav(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    merged = left.merge(
        right, on="trade_date", how="outer", suffixes=("_left", "_right"), indicator=True
    )
    merged["delta"] = merged["nav_left"] - merged["nav_right"]
    merged["abs_delta"] = merged["delta"].abs()
    finite = merged[["nav_left", "nav_right"]].notna().all(axis=1)
    max_delta = float(merged["abs_delta"].max()) if not merged.empty else 0.0
    dates_equal = set(left["trade_date"]) == set(right["trade_date"]) and len(left) == len(right)
    summary = {
        "passed": dates_equal and bool(finite.all()) and max_delta <= 1e-6,
        "rows_left": len(left),
        "rows_right": len(right),
        "dates_equal": dates_equal,
        "max_abs_delta": max_delta,
        "tolerance": 1e-6,
        "mismatch_dates": merged.loc[merged["_merge"] != "both", "trade_date"].tolist(),
    }
    return summary, merged.drop(columns=["_merge"])


def compare_positions(left: pd.DataFrame, right: pd.DataFrame) -> Dict[str, Any]:
    left_rows, right_rows = _records(left, POSITION_COLUMNS), _records(right, POSITION_COLUMNS)
    differences = [
        {"row_index": index, "left": lrow, "right": rrow}
        for index, (lrow, rrow) in enumerate(zip(left_rows, right_rows))
        if lrow != rrow
    ]
    return {
        "passed": left_rows == right_rows,
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "difference_examples": differences[:10],
    }


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("|".join(map(str, frame.dtypes)).encode("utf-8"))
    if not frame.empty:
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def collect_shared_inputs(
    host: PaperHost, window: Sequence[pd.Timestamp]
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, List[Dict[str, Any]]]:
    """重放 paper 的真实只读装配，提取连续参照所需的最终输入（非指令/成交）。"""
    features: Dict[str, pd.DataFrame] = {}
    price_parts: List[pd.DataFrame] = []
    receipts: List[Dict[str, Any]] = []
    for day in window:
        date = day.strftime("%Y%m%d")
        assembly = host._assemble(date)
        if not assembly.readiness.ok:
            raise RuntimeError(f"{date} 共享输入装配未就绪: {assembly.readiness.gaps}")
        host._assert_no_future_price(assembly.price_data, date)
        current = assembly.features_by_date.get(date)
        if current is None or current.empty:
            raise RuntimeError(f"{date} 共享输入缺当日特征截面")
        features[date] = current.copy()
        price_parts.append(assembly.price_data.copy())
        receipts.append(
            {
                "date": date,
                "feature_fingerprint": _frame_fingerprint(current),
                "price_fingerprint": _frame_fingerprint(assembly.price_data),
                "feature_rows": len(current),
                "price_rows": len(assembly.price_data),
            }
        )
    prices = pd.concat(price_parts, ignore_index=True)
    prices = prices.drop_duplicates(["trade_date", "ts_code"], keep="last")
    prices = prices.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return features, prices, receipts


def run_backtest_reference(
    host: PaperHost,
    *,
    features_by_date: Dict[str, pd.DataFrame],
    price_data: pd.DataFrame,
    window: Sequence[pd.Timestamp],
    enable_risk_budget: bool = False,
) -> Dict[str, Any]:
    """回测宿主连续 run 参照；输入仅限最终特征/行情，不接触 paper 指令或成交。"""
    config = host.trading_config
    effective = _build_replay_trading_config(
        model_version=int(config.model_version),
        label_column=f"neu_y_ret_{config.horizon}",
        bt_top_n=config.top_n,
        bt_rebalance_freq=config.rebalance_freq,
        bt_exclude_st=config.exclude_st,
        bt_min_list_days=config.min_list_days,
        bt_sell_timing=config.sell_price,
        bt_max_weight_per_stock=config.max_weight_per_stock,
        bt_max_per_industry=config.max_per_industry,
        bt_stop_loss_enabled=config.stop_loss_enabled,
        bt_stop_loss_drawdown_pct=config.stop_loss_drawdown_pct,
        bt_stop_loss_consecutive_limit_down=config.stop_loss_consecutive_limit_down,
        position_sizing=config.position_sizing,
        kelly_vol_window=config.kelly_vol_window,
        kelly_max_leverage=config.kelly_max_leverage,
        stagger_tranches=config.stagger_tranches,
        enable_early_rebalance_on_empty=config.enable_early_rebalance_on_empty,
        initial_capital=config.initial_capital,
    )
    engine = _assemble_replay_engine(
        effective,
        host.stock_basic,
        "main",
        str(host.panel_store.root),
        (
            host.signal
            if hasattr(host.signal, "update_model_version")
            else _PersistentSignalAdapter(host.signal)
        ),
        features_by_date,
        host.data_storage,
        config.initial_capital,
        config.sell_price,
        config.exclude_st,
        config.min_list_days,
        merger_mode=None,
        industry_reader=host.loader.load_shenwan_industry if config.max_per_industry else None,
    )
    engine.enable_risk_budget = bool(enable_risk_budget)
    start, end = window[0].strftime("%Y%m%d"), window[-1].strftime("%Y%m%d")
    nav = engine.run(
        window[0],
        window[-1],
        list(window),
        price_data,
        trading_dates_full=host.trading_calendar,
    )
    result = _compute_replay_metrics(nav, start, end, config.top_n, 0)
    result["_nav_curve"] = nav
    result["_trades"] = engine.get_trades()
    result["_execution_attribution"] = engine.get_execution_attribution()
    result["_holdings_snapshot"] = engine.get_holdings_snapshot()
    result["_execution_config"] = execution_config_from_engine(
        engine, model_version=config.model_version, buy_price=config.buy_price
    )
    return result


def model_training_window(host: PaperHost) -> Tuple[Optional[str], Optional[str]]:
    metadata = getattr(host.signal, "metadata", None) or {}
    return metadata.get("train_start_date"), metadata.get("train_end_date")


def write_backtest_batch(
    result: Mapping[str, Any],
    *,
    host: PaperHost,
    batch_id: str,
    runs_root: Path,
    config: Dict[str, object],
    code_state: Dict[str, object],
    data_state: Dict[str, object],
    baseline_ref: Optional[str],
) -> Path:
    train_start, train_end = model_training_window(host)
    fold = {
        "split_index": 0,
        "model_version": host.trading_config.model_version,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": result["bt_start"],
        "test_end": result["bt_end"],
        "metrics": {key: value for key, value in result.items() if not key.startswith("_")},
        "nav_curve": result["_nav_curve"],
        "trades": result["_trades"],
        "attribution": result["_execution_attribution"],
        "holdings_snapshot": result["_holdings_snapshot"],
    }
    return write_runs_batch(
        [fold],
        batch_id=batch_id,
        runs_root=runs_root,
        config=config,
        code_state=code_state,
        data_state=data_state,
        baseline_ref=baseline_ref,
        arms=("backtest",),
        source="v2_p4_backtest_reference",
        extra_meta_notes={"reference_inputs": "paper_daily_final_features_and_prices"},
    )
