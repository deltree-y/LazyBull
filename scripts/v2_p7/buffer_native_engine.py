"""原生执行链上的固定 Top20 到期续持研究子类。

复杂度豁免：阶段专用原生路径验证，不入生产链路。
关闭时每个覆写方法立即调用 super，不改变生产输入、时钟或数值操作。
开启时只增加续期决策与原批名额/预算约束，真实买入日、成本与股数不因续持改写。
成交、同日候补、跨日补齐和延迟订单继续调用原生实现；买入恒定 close。

同日跨批退出按批次索引稳定顺序分组调用生产回填/买入/补齐，各组保留原批归属。
研究态 unfilled_slots 使用 (原信号日, 原批次) 键，调用原补齐方法期间还原日期键。
续期状态不属于生产 EngineState；export_buffer_state() 只作研究取证，不支持恢复续跑。
"""

# 复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from src.lazybull.trading.sell_rules import is_holding_period_exit_due
from src.lazybull.v2.core.decision.sizing import compute_lot_shares
from src.lazybull.v2.core.execution.signal_execution import (
    _filter_signal_candidates,
    _queue_signal_plan,
    _resolve_buy_date_quote,
)
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML

RankProvider = Callable[[pd.Timestamp], Sequence[tuple[str, float]]]


class BufferNativeError(RuntimeError):
    """研究子类的批次、生命周期或预算身份不可确定。"""


class BufferNativeEngine(NativeMergeBacktestEngineML):
    """固定 Top20/20 日重复续持；原生引擎标准 kwargs 原样传递。"""

    def __init__(
        self,
        *args: Any,
        buffer_enabled: bool = False,
        rank_provider: RankProvider | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.buffer_enabled = bool(buffer_enabled)
        self.rank_provider = rank_provider
        self.events: list[dict[str, Any]] = []
        self._buffer_meta: dict[str, dict[str, Any]] = {}
        self._buffer_ranks: dict[pd.Timestamp, list[tuple[str, float]]] = {}
        self._buffer_owner: dict[tuple[str, pd.Timestamp], int] = {}
        self._buffer_signal_batches: dict[pd.Timestamp, set[int]] = {}
        self._buffer_exec_batch: int | None = None
        if self.buffer_enabled:
            if rank_provider is None:
                raise BufferNativeError("开启续持必须提供冻结合法完整排名 rank_provider")
            if self.signal.top_n != 20 or self.stagger_tranches != 2 or self.holding_period != 20:
                raise BufferNativeError("固定研究只允许 Top20、两批、持有期20日")

    def _event(self, date: pd.Timestamp, kind: str, **values: Any) -> None:
        """研究留痕与生产日志/台账分离。"""
        self.events.append(
            {"trade_date": pd.Timestamp(date).strftime("%Y%m%d"), "kind": kind, **values}
        )

    def _ranked(self, date: pd.Timestamp) -> list[tuple[str, float]]:
        """同日只调用一次冻结提供者，不用前日排名填缺；空列表由提供者显式登记。"""
        date = pd.Timestamp(date)
        if date not in self._buffer_ranks:
            values = self.rank_provider(date)
            if values is None:
                raise BufferNativeError(f"{date} 排名缺失，不能当合法空排名")
            ranked = [(str(code), float(score)) for code, score in values]
            if len({code for code, _ in ranked}) != len(ranked):
                raise BufferNativeError(f"{date} 完整排名含重复股票")
            self._buffer_ranks[date] = ranked
        return self._buffer_ranks[date]

    @staticmethod
    def _life(info: Mapping[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
        """生命周期只认真实买入/原信号日期，送转和估值变化不创建新生命周期。"""
        return pd.Timestamp(info["buy_date"]), pd.Timestamp(
            info.get("signal_date", info["buy_date"])
        )

    def _sync_buffer_positions(self) -> None:
        """清理已卖生命周期；无法解释的外部持仓拒绝猜测批次。"""
        for code in list(self._buffer_meta):
            if code not in self.positions:
                self._buffer_meta.pop(code)
        for code, info in self.positions.items():
            if int(info.get("shares", 0)) <= 0:
                continue
            life = self._life(info)
            old = self._buffer_meta.get(code)
            if old and old["life"] == life:
                continue
            batch = self._buffer_owner.get((code, life[1]))
            if batch is None:
                batch = self._signal_dates.get(life[1])
            if batch not in (0, 1):
                raise BufferNativeError(f"{code} 生命周期 {life} 无原批归属")
            self._buffer_meta[code] = {"life": life, "tranche_idx": int(batch), "renewal_count": 0}

    def _has_sell_obligation(self, code: str) -> bool:
        """到期、止损和已延迟市场卖单都不能被后来排名撤销。"""
        manager = self.pending_order_manager
        return (
            code in self.pending_condition_sells
            or code in self.pending_stop_loss_sells
            or (manager is not None and (code, "sell") in manager.pending_orders)
        )

    def _evaluate_buffer_due(
        self, date: pd.Timestamp, date_to_idx: Mapping[pd.Timestamp, int]
    ) -> list[dict[str, Any]]:
        """两条卖出决策路径共用：只有正常到期才看 Top20，延期期间掉榜不提前退出。"""
        self._sync_buffer_positions()
        exits = []
        for code, info in list(self.positions.items()):
            if self._has_sell_obligation(code):
                continue
            meta = self._buffer_meta[code]
            buy_idx = date_to_idx.get(pd.Timestamp(info["buy_date"]))
            if buy_idx is None:
                raise BufferNativeError(f"{code} 真实买入日不在完整日历")
            held = date_to_idx[date] - buy_idx
            period = 20 * (int(meta["renewal_count"]) + 1)
            if not is_holding_period_exit_due(held, period):
                continue
            ranked = self._ranked(date)
            renew = code in {item[0] for item in ranked[:20]}
            if renew:
                meta["renewal_count"] += 1
            else:
                self.pending_condition_sells[code] = {
                    "trigger_date": date,
                    "sell_type": "holding_period",
                }
                exits.append(
                    {
                        "stock": code,
                        "tranche_idx": meta["tranche_idx"],
                        "weight": self._get_position_weight_for_planning(date, code),
                    }
                )
            self._event(
                date,
                "expiry",
                ts_code=code,
                holding_days=held,
                renew=renew,
                renewal_count=meta["renewal_count"],
                tranche_idx=meta["tranche_idx"],
                buy_date=pd.Timestamp(info["buy_date"]).strftime("%Y%m%d"),
                reason="renew_top20" if renew else "outside_top20_or_empty",
            )
        return exits

    def _generate_signal(
        self,
        date: pd.Timestamp,
        trading_dates: list[pd.Timestamp],
        price_data: pd.DataFrame,
        date_to_idx: dict,
        tranche_idx: int = 0,
    ) -> None:
        if not self.buffer_enabled:
            return super()._generate_signal(
                date, trading_dates, price_data, date_to_idx, tranche_idx
            )
        self._last_rebalance_nav = self._calculate_portfolio_value(date)
        ranked = self._ranked(date)
        self._evaluate_buffer_due(date, date_to_idx)
        if (
            not ranked
            or _resolve_buy_date_quote(self, date, trading_dates, price_data, date_to_idx) is None
        ):
            return
        ranked, eligible = _filter_signal_candidates(self, ranked, date, set(self.positions))
        retained = [
            code
            for code, meta in self._buffer_meta.items()
            if meta["tranche_idx"] == tranche_idx and not self._has_sell_obligation(code)
        ]
        target_n = max(0, 10 - len(retained))
        if target_n == 0:
            self._event(date, "retained_full_tranche", tranche_idx=tranche_idx)
            return
        picks = eligible[:target_n]
        if not picks:
            return
        scores = dict(ranked)
        combined = {code: float(scores.get(code, 0.0)) for code in retained}
        combined.update(picks)
        normalized = self._normalize_and_cap_signals(combined, date, tranche_idx)
        new_weights = {code: float(normalized.get(code, 0.0)) for code, _ in picks}
        retained_mv = sum(self._position_market_value(date, code) or 0.0 for code in retained)
        batch_nav = 0.5 * self._last_rebalance_nav
        room = max(0.0, batch_nav - retained_mv)
        nominal = sum(new_weights.values()) * batch_nav
        if nominal > room and nominal > 0:
            new_weights = {code: weight * room / nominal for code, weight in new_weights.items()}
        trace = self._build_signal_decision_trace(
            date=date, target_n=len(picks), candidate_count=len(eligible), tranche_idx=tranche_idx
        )
        _queue_signal_plan(
            self, date, new_weights, ranked, list(eligible), len(picks), 20, tranche_idx, trace
        )
        self._buffer_signal_batches.setdefault(date, set()).add(tranche_idx)
        self._event(
            date,
            "tranche_plan",
            tranche_idx=tranche_idx,
            retained=retained,
            retained_market_value=retained_mv,
            batch_budget=batch_nav,
            new_weight_sum=sum(new_weights.values()),
        )

    def _queue_rebalance_sells(
        self, date: pd.Timestamp, trading_dates: list[pd.Timestamp], date_to_idx: dict
    ) -> None:
        if not self.buffer_enabled:
            return super()._queue_rebalance_sells(date, trading_dates, date_to_idx)
        self._evaluate_buffer_due(date, date_to_idx)

    def _check_and_sell(
        self, date: pd.Timestamp, trading_dates: list[pd.Timestamp], date_to_idx: dict
    ) -> None:
        if not self.buffer_enabled:
            return super()._check_and_sell(date, trading_dates, date_to_idx)
        exits = self._evaluate_buffer_due(date, date_to_idx)
        if exits:
            self._queue_condition_sell_refill_signal(
                date, exits, self.price_data_cache, date_to_idx
            )

    def _queue_condition_sell_refill_signal(
        self,
        date: pd.Timestamp,
        slot_weights: list[dict],
        price_data: pd.DataFrame,
        date_to_idx: dict,
    ) -> None:
        if not self.buffer_enabled:
            return super()._queue_condition_sell_refill_signal(
                date, slot_weights, price_data, date_to_idx
            )
        if date in self.pending_signals:
            return
        groups = []
        for batch in (0, 1):
            slots = [
                slot
                for slot in slot_weights
                if self._buffer_meta[slot["stock"]]["tranche_idx"] == batch
            ]
            if not slots:
                continue
            super()._queue_condition_sell_refill_signal(date, slots, price_data, date_to_idx)
            plan = self.pending_signals.pop(date, None)
            if plan is not None:
                plan["tranche_idx"] = batch
                groups.append(plan)
                self._buffer_signal_batches.setdefault(date, set()).add(batch)
                self._event(
                    date,
                    "expiry_refill",
                    tranche_idx=batch,
                    exit_stocks=[slot["stock"] for slot in slots],
                )
        if groups:
            self.pending_signals[date] = dict(groups[0], buffer_groups=groups)

    def _execute_pending_buys(
        self, date: pd.Timestamp, trading_dates: list[pd.Timestamp], date_to_idx: dict
    ) -> None:
        if not self.buffer_enabled:
            return super()._execute_pending_buys(date, trading_dates, date_to_idx)
        idx = date_to_idx.get(date)
        if idx is None or idx == 0:
            return
        signal_date = trading_dates[idx - 1]
        payload = self.pending_signals.get(signal_date)
        if payload is None:
            return
        groups = payload.get("buffer_groups", [payload])
        saved_slots = self.unfilled_slots
        try:
            for group in sorted(groups, key=lambda item: item["tranche_idx"]):
                batch = int(group["tranche_idx"])
                self._buffer_exec_batch = batch
                self.pending_signals[signal_date] = group
                self.unfilled_slots = {}
                super()._execute_pending_buys(date, trading_dates, date_to_idx)
                for origin, value in self.unfilled_slots.items():
                    saved_slots[(origin, batch)] = value
        finally:
            self.unfilled_slots = saved_slots
            self._buffer_exec_batch = None

    def _process_position_completion(
        self,
        date: pd.Timestamp,
        trading_dates: list[pd.Timestamp],
        price_data: pd.DataFrame,
        date_to_idx: dict,
    ) -> None:
        if not self.buffer_enabled:
            return super()._process_position_completion(
                date, trading_dates, price_data, date_to_idx
            )
        groups = self.unfilled_slots
        try:
            for key in sorted(list(groups), key=lambda item: (item[0], item[1])):
                signal_date, batch = key
                self._buffer_exec_batch = int(batch)
                self.unfilled_slots = {signal_date: groups[key]}
                super()._process_position_completion(date, trading_dates, price_data, date_to_idx)
                remaining = self.unfilled_slots.get(signal_date)
                if remaining is None:
                    groups.pop(key)
                else:
                    groups[key] = remaining
        finally:
            self.unfilled_slots = groups
            self._buffer_exec_batch = None

    def _buy_stock(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: pd.Timestamp | None = None,
        requested_shares: int | None = None,
    ) -> None:
        if not self.buffer_enabled:
            return super()._buy_stock(date, stock, target_value, signal_date, requested_shares)
        if self._buffer_exec_batch is not None:
            self._buffer_owner[(stock, pd.Timestamp(signal_date or date))] = self._buffer_exec_batch
        return super()._buy_stock(date, stock, target_value, signal_date, requested_shares)

    def _buy_stock_direct(
        self,
        date: pd.Timestamp,
        stock: str,
        target_value: float,
        signal_date: pd.Timestamp | None = None,
        requested_shares: int | None = None,
    ) -> None:
        if not self.buffer_enabled:
            return super()._buy_stock_direct(
                date, stock, target_value, signal_date, requested_shares
            )
        self._sync_buffer_positions()
        origin = pd.Timestamp(signal_date or date)
        batch = self._buffer_exec_batch
        if batch is None:
            batch = self._buffer_owner.get((stock, origin))
        if batch is None:
            choices = self._buffer_signal_batches.get(origin, set())
            if len(choices) == 1:
                batch = next(iter(choices))
        if batch not in (0, 1):
            raise BufferNativeError(f"{date} {stock} 买入无唯一原批上下文")
        if stock in self.positions:
            self._event(date, "retained_no_add", ts_code=stock, tranche_idx=batch)
            return
        held = [code for code, meta in self._buffer_meta.items() if meta["tranche_idx"] == batch]
        if len(held) >= 10 or len(self.positions) >= 20:
            self._event(date, "tranche_slot_block", ts_code=stock, tranche_idx=batch)
            return
        nav = self._calculate_portfolio_value(date)
        occupied = sum(self._position_market_value(date, code) or 0.0 for code in held)
        remaining = max(0.0, nav * 0.5 - occupied)
        capped = min(float(target_value), remaining, nav * 0.15)
        if capped < float(target_value):
            self._event(
                date,
                "tranche_budget_cap",
                ts_code=stock,
                tranche_idx=batch,
                requested=float(target_value),
                capped=capped,
                occupied=occupied,
                nav=nav,
            )
        if requested_shares is not None:
            price = self._get_trade_price(date, stock)
            if price is None or price <= 0:
                return
            requested_shares = min(int(requested_shares), compute_lot_shares(capped, price))
        if capped <= 0:
            return
        super()._buy_stock_direct(date, stock, capped, signal_date, requested_shares)
        if stock in self.positions:
            self._buffer_owner[(stock, origin)] = int(batch)
            self._buffer_meta[stock] = {
                "life": self._life(self.positions[stock]),
                "tranche_idx": int(batch),
                "renewal_count": 0,
            }
            self._event(
                date,
                "buy_owner",
                ts_code=stock,
                tranche_idx=batch,
                signal_date=origin.strftime("%Y%m%d"),
                shares=int(self.positions[stock]["shares"]),
            )

    def run_step(
        self, date: pd.Timestamp, trading_dates_full: list[pd.Timestamp], price_data: pd.DataFrame
    ) -> None:
        if not self.buffer_enabled:
            return super().run_step(date, trading_dates_full, price_data)
        self._sync_buffer_positions()
        super().run_step(date, trading_dates_full, price_data)
        self._sync_buffer_positions()
        if self.current_capital < 0:
            raise BufferNativeError(f"{date} 续持账户出现负现金：{self.current_capital}")

    def export_buffer_state(self) -> dict[str, Any]:
        """JSON 可序列化的独立取证快照；不伪装成生产 EngineState 或续跑入口。"""
        return {
            "schema": "buffer_native/v1",
            "enabled": self.buffer_enabled,
            "positions": {
                code: {**meta, "life": [day.strftime("%Y%m%d") for day in meta["life"]]}
                for code, meta in sorted(self._buffer_meta.items())
            },
            "order_owners": [
                {"ts_code": code, "signal_date": day.strftime("%Y%m%d"), "tranche_idx": batch}
                for (code, day), batch in sorted(self._buffer_owner.items())
            ],
            "signal_batches": [
                {"signal_date": day.strftime("%Y%m%d"), "tranches": sorted(batches)}
                for day, batches in sorted(self._buffer_signal_batches.items())
            ],
            "events": deepcopy(self.events),
        }
