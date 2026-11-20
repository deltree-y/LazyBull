"""R4—R6：三臂 B 袖原生账户、五臂合并、指标/压力/相关与报告。

复杂度豁免：P7 候选一阶段专用研究工具，不入生产链路（AGENTS §7.5）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook
from scripts.v2_p7.buffer_native_replay import (
    engine_kwargs,
    initialize_driver,
    result_payload,
)
from scripts.v2_p7.buffer_native_validation import validate_account
from scripts.v2_p7.defensive_ml.b_native import PITIndustrySleeve
from scripts.v2_p7.operating_monthly_baseline import combine_independent_daily

from . import spec

ARMS = ("PEER", "REV", "MATCH")


class PRGSleeve(PITIndustrySleeve):
    """原规格纠错的三处约束：月末替换、单笔 ≥V、无月中持有期卖出。

    - ``min_holding_days=0``：信号日把所有非目标持仓排到 T+1，不再等 ``holding_period-1``；
    - ``holding_period=99999``：引擎的到期卖出自此不触发；
    - ``min_trade_value``：目标金额低于协议 V 直接跳过，由 R5 逐笔复核兜底。
    """

    min_trade_value: float = 0.0
    min_trade_skips: int = 0

    def _queue_rebalance_sells(
        self, date: Any, trading_dates: Any, date_to_idx: Any
    ) -> None:
        from src.lazybull.trading.sell_rules import select_rebalance_sell_candidates

        if not self.positions:
            return
        current_idx = date_to_idx.get(date)
        if current_idx is None:
            return
        signal_data = self.pending_signals.get(date, {})
        if isinstance(signal_data, dict) and "signals" in signal_data:
            new_signal_stocks = set(signal_data["signals"].keys())
        else:
            new_signal_stocks = set()
        holding_days_map: dict[str, int | None] = {}
        for stock, info in list(self.positions.items()):
            buy_date = info.get("buy_date")
            buy_idx = date_to_idx.get(buy_date) if buy_date else None
            if buy_idx is None:
                continue
            holding_days_map[stock] = current_idx - buy_idx
        decision = select_rebalance_sell_candidates(
            holding_days_map,
            min_holding_days=0,
            target_codes=new_signal_stocks,
            queued_codes=set(self.pending_condition_sells)
            | set(self.pending_stop_loss_sells),
        )
        for stock in decision.sells:
            self.pending_condition_sells[stock] = {
                "trigger_date": date,
                "sell_type": "rebalance",
            }

    def _buy_stock_direct(
        self,
        date: Any,
        stock: str,
        target_value: float,
        signal_date: Any = None,
        requested_shares: int | None = None,
    ) -> None:
        if (
            requested_shares is None
            and self.min_trade_value > 0
            and float(target_value) < self.min_trade_value
        ):
            self.min_trade_skips += 1
            return
        return super()._buy_stock_direct(
            date, stock, target_value, signal_date, requested_shares
        )


def build_arm_rankings(
    plans: pd.DataFrame,
    matched: pd.DataFrame | None = None,
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """把三臂计划整理为 ``{arm: {signal_date: [(code, score)]}}``。

    PEER/MATCH 按 score 降序；REV 按 L 升序（``-L`` 降序）后取前六。
    同分用 ts_code 升序，保证确定性。
    """
    result: dict[str, dict[str, list[tuple[str, float]]]] = {arm: {} for arm in ARMS}
    for arm in ("PEER", "REV"):
        frame = plans[plans["arm"] == arm]
        for day, group in frame.groupby("signal_date", sort=True):
            key = "L" if arm == "REV" else "score"
            ordered = group.sort_values([key, "ts_code"], ascending=[True, True])
            if arm == "PEER":
                ordered = group.sort_values(["score", "ts_code"], ascending=[False, True])
            result[arm][str(day)] = [
                (str(row["ts_code"]), float(row[key]) if arm == "REV" else float(row["score"]))
                for _, row in ordered.head(spec.SLOTS).iterrows()
            ]
    peer_plans = plans[plans["arm"] == "PEER"].copy()
    pairs = matched
    if isinstance(pairs, pd.DataFrame) and len(pairs):
        lookup = {
            (str(row["signal_date"]), int(row["slot"])): str(row["match"])
            for _, row in pairs.iterrows()
            if row.get("match") is not None and pd.notna(row.get("match"))
        }
        for day, group in peer_plans.groupby("signal_date", sort=True):
            codes: list[tuple[str, float]] = []
            for _, row in group.sort_values("slot").iterrows():
                code = lookup.get((str(day), int(row["slot"])))
                if code:
                    codes.append((code, float(row["score"])))
            result["MATCH"][str(day)] = codes[: spec.SLOTS]
    return result


def _b_kwargs(
    root: Path,
    ranked: Mapping[str, list[tuple[str, float]]],
    capital: float,
    book: NativeCorporateBook,
) -> dict[str, Any]:
    """B 规格覆盖：单批/等权/行业 2/开盘卖/不补齐/不重试/关早调仓。"""
    kwargs = engine_kwargs(root, dict(ranked), int(capital), False)
    kwargs.update(
        {
            "rebalance_freq": 9999,
            "holding_period": 99999,
            "stagger_tranches": 1,
            "position_sizing": "equal",
            "max_weight_per_stock": float(spec.SLOT_WEIGHT),
            "max_per_industry": int(spec.SECTOR_CAP),
            "sell_timing": "open",
            "enable_pending_order": True,
            "max_retry_count": 0,
            "max_retry_days": 0,
            "enable_position_completion": False,
            "completion_window_days": 0,
            "enable_early_rebalance_on_empty": False,
            "enable_trailing_early_rebalance": False,
            "min_buy_value_ratio": 0.0,
            "corporate_book": book,
        }
    )
    return kwargs


def run_b_arm(
    *,
    root: Path,
    ranked: Mapping[str, list[tuple[str, float]]],
    calendar: Sequence[str],
    prices: pd.DataFrame,
    indexes: Any,
    capital: float,
    book: NativeCorporateBook,
    pit: pd.DataFrame,
    l1_names: Mapping[str, str],
) -> dict[str, Any]:
    """驱动原生引擎；信号日 = 月末 T，引擎在 T+1 开盘卖/收盘买。"""
    dates = [pd.Timestamp(str(day)) for day in calendar]
    engine = PRGSleeve(**_b_kwargs(root, ranked, capital, book))
    engine.min_trade_value = float(spec.MIN_NOTIONAL)
    engine.pit_frame = pit
    engine.pit_names = dict(l1_names)
    engine.pit_refresh_log = []
    engine.record_holdings_snapshot = True
    engine._hook_stats = {}
    initialize_driver(engine, dates, prices, indexes)
    engine._signal_dates = {
        pd.Timestamp(str(day)): 0 for day in ranked
    }
    engine._cycle_anchor_idx = 0
    for date in dates:
        engine.run_step(date, dates, prices)
    result = result_payload(engine)
    result["validation"] = validate_account(result, initial_capital=float(capital))
    result["signal_days"] = sorted(str(day) for day in ranked)
    result["engine_params"] = {
        name: getattr(engine, name, None)
        for name in (
            "rebalance_freq",
            "holding_period",
            "stagger_tranches",
            "position_sizing",
            "max_weight_per_stock",
            "max_per_industry",
            "max_retry_count",
            "max_retry_days",
            "enable_position_completion",
            "sell_timing",
            "min_buy_value_ratio",
        )
    }
    result["engine_params"]["min_trade_value"] = engine.min_trade_value
    result["engine_params"]["min_trade_skips"] = engine.min_trade_skips
    return result


def combine_five_arms(
    a100: Sequence[Mapping[str, Any]],
    a80: Sequence[Mapping[str, Any]],
    b_arms: Mapping[str, Sequence[Mapping[str, Any]]],
    total_capital: int,
) -> dict[str, list[dict[str, Any]]]:
    """A80 单账 + 各 B 臂逐日直接合计；A100 原样冻结。"""
    arms: dict[str, list[dict[str, Any]]] = {"A100": [dict(row) for row in a100]}
    for arm, rows in b_arms.items():
        try:
            arms[f"A80_{arm}"] = combine_independent_daily(
                a80, rows, total_capital=total_capital
            )
        except ValueError as error:
            left_days = [str(row["trade_date"]) for row in a80]
            right_days = [str(row["trade_date"]) for row in rows]
            mismatch = next(
                (
                    (index, left, right)
                    for index, (left, right) in enumerate(zip(left_days, right_days))
                    if left != right
                ),
                None,
            )
            detail = {
                "arm": arm,
                "left_len": len(left_days),
                "right_len": len(right_days),
                "left_head": left_days[:3],
                "right_head": right_days[:3],
                "left_tail": left_days[-3:],
                "right_tail": right_days[-3:],
                "first_mismatch": mismatch,
                "left_keys": sorted(a80[0].keys()),
                "right_keys": sorted(rows[0].keys()),
            }
            raise ValueError(f"{error} | {detail}") from error
    return arms


def metrics_from_daily(rows: Sequence[Mapping[str, Any]], capital: int) -> dict[str, Any]:
    """链式指标：与 paper/performance 同口径（自然日复合 + 正进行情 MDD）。"""
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    values = [float(row["total_value"]) for row in rows]
    days = [str(row["trade_date"]) for row in rows]
    chain = calculate_chain_metrics(pd.DataFrame({"nav": values}))
    cagr = chain["cagr"]
    mdd = chain["max_drawdown"]
    daily_ret = pd.Series(values).pct_change().dropna()
    sharpe = (
        float(daily_ret.mean() / daily_ret.std(ddof=1) * math.sqrt(252))
        if len(daily_ret) > 2 and daily_ret.std(ddof=1) > 0
        else float("nan")
    )
    total_return = values[-1] / capital - 1.0
    return {
        "capital": int(capital),
        "start": days[0],
        "end": days[-1],
        "rows": len(rows),
        "ending_value": round(values[-1], 6),
        "total_return": round(total_return, 10),
        "cagr": round(float(cagr), 10),
        "mdd_loss": round(abs(float(mdd)), 10),
        "sharpe": round(sharpe, 10) if math.isfinite(sharpe) else None,
        "fees": round(float(sum(float(row.get("fees", 0.0)) for row in rows)), 6),
    }


def yearly_metrics(
    rows: Sequence[Mapping[str, Any]], capital: int
) -> list[dict[str, Any]]:
    """分年 CAGR/MDD：每年以当年首个交易日的总资产为分母。"""
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    frame = pd.DataFrame(rows)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["year"] = frame["trade_date"].str[:4]
    result = []
    for year, group in frame.groupby("year", sort=True):
        chain = calculate_chain_metrics(
            pd.DataFrame({"nav": group["total_value"].astype(float).to_numpy()})
        )
        result.append(
            {
                "year": year,
                "cagr": round(float(chain["cagr"]), 10),
                "mdd_loss": round(abs(float(chain["max_drawdown"])), 10),
                "days": len(group),
            }
        )
    return result
