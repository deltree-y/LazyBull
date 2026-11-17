"""到期 Top20 续持的独立纯 A 研究账户，不接入生产执行链路。

复杂度豁免：阶段专用路径对照，不入生产链路。买入阻断当日取消且无候补，
卖出义务持续重试；这不是旧生产引擎的逐笔等价实现。T0 冻结预算，T+1
仅以真实开盘换算整手，费用、现金及公司行为由 P7 T4 组件统一记账。
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from scripts.v2_p7.buffer_path_policy import due_decision, plan_tranche
from scripts.v2_p7.t4.corporate import CorporateActionBook
from scripts.v2_p7.t4.dual import run_daily_identities
from scripts.v2_p7.t4.host import apply_corporate_actions
from scripts.v2_p7.t4.ledger import execute_plan
from scripts.v2_p7.t4.netting import Intent, assert_intent_conservation, plan_day
from scripts.v2_p7.t4.params import FROZEN_COST_SETTINGS
from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t4.state import PhysicalBook, SleeveState, stable_digest
from src.lazybull.trading.sell_rules import is_holding_period_exit_due
from src.lazybull.v2.common.cost import CostModel

SLEEVE = "value"
LOT_SIZE = 100
PERIOD = 20


class AccountError(RuntimeError):
    """研究账户违反输入、时点或资金不变量。"""


def run_account(inputs: Any, capital: float, buffer_enabled: bool, arm_name: str) -> dict[str, Any]:
    """返回六类逐行台账及 validation；首日现金 NAV=1 已在 daily 中。

    inputs 使用 ResearchInputs 接口；测试提供完全隔离的同接口合成数据。
    该函数不写文件、不访问网络、不自行计算收益/回撤指标。
    """
    return _Account(inputs, capital, buffer_enabled, arm_name).run()


class _Account:
    """单袖资金、真实 lot 与独立续持元信息；每次运行重新创建全部可变状态。"""

    def __init__(self, inputs: Any, capital: float, enabled: bool, name: str) -> None:
        if not math.isfinite(float(capital)) or float(capital) <= 0:
            raise AccountError(f"初资必须为正有限值：{capital!r}")
        self.inputs = inputs
        self.capital = float(capital)
        self.enabled = bool(enabled)
        self.name = str(name)
        self.days = tuple(inputs.calendar)
        if not self.days or tuple(sorted(set(self.days))) != self.days:
            raise AccountError("账户日历必须非空、严格递增且无重复")
        self.index = {day: idx for idx, day in enumerate(self.days)}
        self.signal_days = frozenset(inputs.signal_days)
        if not self.signal_days or not self.signal_days.issubset(self.index):
            raise AccountError("信号日必须非空且属于账户日历")
        if any(
            day not in self.signal_days or batch not in (0, 1)
            for day, batch in inputs.schedule.items()
        ):
            raise AccountError("批次排期必须属于信号域且批次为 0/1")
        missing = sorted(self.signal_days - set(inputs.ranked))
        if missing:
            raise AccountError(f"冻结合法排名缺日：{missing[:5]}")
        self.prices = PriceBook(inputs.loader)
        self.corporate = CorporateActionBook(
            calendar=self.days,
            dividend_dir=Path(inputs.params.data_root) / "raw" / "dividend",
            cutoff=self.days[-1],
        )
        self.cost = CostModel(**dict(FROZEN_COST_SETTINGS))
        self.sleeves = {SLEEVE: SleeveState(SLEEVE, self.capital, 0.0)}
        self.physical = PhysicalBook(self.capital)
        self.meta: dict[str, dict[str, Any]] = {}
        self.pending_sells: dict[str, dict[str, Any]] = {}
        self.pending_buys: list[dict[str, Any]] = []
        self.pending_batch_limits: dict[int, dict[str, Any]] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {
            key: [] for key in ("daily", "trades", "decisions", "holdings", "blocked", "corporate")
        }
        self.conservation_rows = 0
        self.identity_rows = 0
        self.seq = 0

    @property
    def state(self) -> SleeveState:
        """当前袖快照；实际股份唯一来自账本。"""
        return self.sleeves[SLEEVE]

    def run(self) -> dict[str, Any]:
        """逐日公司行为、执行、估值、T0 决策。"""
        for day in self.days:
            self.rows["corporate"].extend(
                apply_corporate_actions(
                    self.corporate,
                    day=day,
                    sleeves=self.sleeves,
                    physical=self.physical,
                    prices=self.prices,
                )
            )
            tracked = set(self.meta) | {row["ts_code"] for row in self.pending_buys}
            self.prices.begin_day(day, tracked=sorted(tracked))
            self._sell(day)
            self._buy(day)
            self._validate_state(day)
            report = run_daily_identities(
                trade_date=day,
                sleeves=self.sleeves,
                physical=self.physical,
                prices=self.prices,
                physical_initial=self.capital,
                sleeve_capital={SLEEVE: self.capital},
                cumulative_savings={SLEEVE: 0.0},
            )
            self.identity_rows += len(report.identities)
            if not report.ok or report.invalid_sleeves:
                raise AccountError(f"{day} 账本恒等式/权益无效：{report!r}")
            self._close(day, report.physical_total_value)
            self._plan(day, report.physical_total_value)
            self._holdings(day)
            self.rows["daily"][-1]["state_digest"] = stable_digest(
                {
                    "sleeve": asdict(self.state),
                    "physical": asdict(self.physical),
                    "meta": self.meta,
                    "pending_sells": self.pending_sells,
                    "pending_buys": self.pending_buys,
                    "pending_batch_limits": self.pending_batch_limits,
                }
            )
        last_holdings = [row for row in self.rows["holdings"] if row["trade_date"] == self.days[-1]]
        payload = self.corporate.payload()
        validation = {
            "valid": True,
            "arm_name": self.name,
            "capital": self.capital,
            "buffer_enabled": self.enabled,
            "stop_loss_enabled": False,
            "risk_overlay_enabled": False,
            "calendar_days": len(self.days),
            "initial_cash_point": self.days[0],
            "initial_cash_point_in_daily": True,
            "initial_nav_is_one": self.rows["daily"][0]["nav"] == 1.0,
            "identity_checks": self.identity_rows,
            "identity_violations": 0,
            "intent_conservation_rows": self.conservation_rows,
            "minimum_cash": min(row["cash"] for row in self.rows["daily"]),
            "maximum_holding_count": max(row["holding_count"] for row in self.rows["daily"]),
            "terminal_holdings": last_holdings,
            "terminal_pending_sells": self.pending_sells,
            "terminal_pending_buys": self.pending_buys,
            "terminal_receivables": [asdict(item) for item in self.state.receivables],
            "corporate_assumptions": {
                key: payload[key]
                for key in (
                    "cash_basis_rule",
                    "share_rule",
                    "stats",
                    "record_date_gap_count",
                )
            },
            "execution_difference": "买入阻断或不足额部分当日取消，无候补/追单；卖出每日重试",
            "production_bitwise_equivalent": False,
        }
        return {**self.rows, "validation": validation}

    def _execute(self, day: str, intents: Sequence[Intent]) -> None:
        """复用 T4 对实际可卖股数、费用、现金、整手及双账的执行；残差由本宿主管理。"""
        if not intents:
            return
        available = {
            SLEEVE: {lot.ts_code: lot.shares for lot in self.state.lots if lot.buy_date < day}
        }
        plan = plan_day(day, intents, available)
        outcome = execute_plan(
            plan=plan,
            sleeves=self.sleeves,
            physical=self.physical,
            prices=self.prices,
            cost_model=self.cost,
            lot_size=LOT_SIZE,
            seq_base=self.seq,
        )
        conserved = assert_intent_conservation(outcome.conservation_rows)
        if conserved["violations"]:
            raise AccountError(f"{day} 成交股数不守恒：{conserved}")
        self.conservation_rows += conserved["rows"]
        self.seq += len(outcome.trades) + 1
        self.sleeves = {SLEEVE: replace(outcome.sleeves[SLEEVE], residuals=())}
        self.physical = outcome.physical
        for trade in outcome.trades:
            code = str(trade["ts_code"])
            source = self.meta.get(code, {})
            self.rows["trades"].append(
                {
                    **trade,
                    "amount": int(trade["shares"]) * float(trade["price"]),
                    "tranche_idx": source.get("tranche_idx"),
                    "renewal_count": source.get("renewal_count", 0),
                }
            )
        for intent in intents:
            ledger = next(
                row for row in outcome.conservation_rows if row["intent_key"] == intent.intent_key
            )
            remaining = int(ledger["original"]) - int(ledger["filled"])
            if remaining:
                lookup = self.prices.execution_price(intent.ts_code, intent.side)
                reason = lookup.reason if not lookup.ok else "insufficient_cash"
                self._blocked(
                    day,
                    intent.ts_code,
                    intent.side,
                    reason,
                    remaining,
                    "retry" if intent.side == "sell" else "cancel",
                )

    def _sell(self, day: str) -> None:
        """卖出义务跨日不可撤销，除权后按真实持仓全卖；从不卖当日买入 lot。"""
        intents = []
        for code, order in sorted(self.pending_sells.items()):
            lots = [lot for lot in self.state.lots if lot.ts_code == code]
            if not lots:
                raise AccountError(f"{day} 待卖义务失去持仓：{code}")
            if order["signal_date"] >= day or any(lot.buy_date >= day for lot in lots):
                raise AccountError(f"{day} 卖出违反 T+1：{code}")
            intents.append(
                Intent(
                    intent_key=f"sell:{code}:{order['signal_date']}",
                    sleeve_id=SLEEVE,
                    ts_code=code,
                    side="sell",
                    shares=sum(lot.shares for lot in lots),
                    origin="expiry_sell",
                    signal_date=order["signal_date"],
                    due_date=order["due_date"],
                    model_version=order["model_version"],
                )
            )
        self._execute(day, intents)
        held = {lot.ts_code for lot in self.state.lots}
        for code in list(self.pending_sells):
            if code not in held:
                self.pending_sells.pop(code)
                self.meta.pop(code)

    def _buy(self, day: str) -> None:
        """按冻结候选顺序执行，仅用昨日预算与当前真实开盘，不做追单或候补。"""
        orders, self.pending_buys = self.pending_buys, []
        limits, self.pending_batch_limits = self.pending_batch_limits, {}
        spent: dict[int, float] = {}
        for order in orders:
            code, batch = str(order["ts_code"]), int(order["tranche_idx"])
            if order["signal_date"] >= day:
                raise AccountError(f"{day} 买入违反 T+1：{code}")
            if code in self.meta:
                raise AccountError(f"{day} 禁止同股追加第二批：{code}")
            if sum(item["tranche_idx"] == batch for item in self.meta.values()) >= 10:
                self._blocked(day, code, "buy", "tranche_slots_occupied", 0, "cancel")
                continue
            lookup = self.prices.execution_price(code, "buy")
            if not lookup.ok:
                self.prices.record_block(lookup.reason)
                self._blocked(day, code, "buy", lookup.reason, 0, "cancel")
                continue
            price = float(lookup.price)
            limit = limits[batch]
            # 实际卖成才释放 T0 占额；保留/受阻股继续占用昨日冻结市值。
            # 公司行为或当日价格波动不允许借用未来 close 重估开盘资金空间。
            occupied = sum(
                value for held, value in limit["holding_values"].items() if held in self.meta
            )
            remaining = max(0.0, limit["ceiling"] - occupied - spent.get(batch, 0.0))
            budget = min(float(order["budget"]), remaining)
            shares = int(budget / price / LOT_SIZE) * LOT_SIZE
            requested = int(float(order["budget"]) / price / LOT_SIZE) * LOT_SIZE
            if budget < float(order["budget"]):
                self._blocked(
                    day, code, "buy", "tranche_budget_occupied", requested - shares, "cancel"
                )
            if shares <= 0:
                if budget == float(order["budget"]):
                    self._blocked(day, code, "buy", "budget_below_lot", 0, "cancel")
                continue
            due_idx = self.index[day] + PERIOD
            due_date = self.days[due_idx] if due_idx < len(self.days) else ""
            self.meta[code] = {"tranche_idx": batch, "renewal_count": 0}
            self._execute(
                day,
                [
                    Intent(
                        intent_key=f"buy:{order['signal_date']}:{batch}:{code}",
                        sleeve_id=SLEEVE,
                        ts_code=code,
                        side="buy",
                        shares=shares,
                        origin="signal_buy",
                        signal_date=order["signal_date"],
                        due_date=due_date,
                        model_version=int(order["model_version"]),
                    )
                ],
            )
            if not self.state.shares_of(code):
                self.meta.pop(code)
            spent[batch] = spent.get(batch, 0.0) + self.state.shares_of(code) * price
            if spent[batch] > max(0.0, limit["ceiling"] - occupied) + 1e-8:
                raise AccountError(f"{day} 批次 {batch} 新买超过已释放的 T0 预算")

    def _blocked(
        self, day: str, code: str, side: str, reason: str, shares: int, action: str
    ) -> None:
        """逐单记阻断；无价时股数未知保留 0，不能假装已承诺可成交股数。"""
        self.rows["blocked"].append(
            {
                "trade_date": day,
                "ts_code": code,
                "side": side,
                "reason": reason,
                "unfilled_shares": int(shares),
                "handling": action,
            }
        )

    def _validate_state(self, day: str) -> None:
        """零借贷、非负现金、每股单 lot、每批十槽与元信息完整性。"""
        codes = [lot.ts_code for lot in self.state.lots]
        if len(set(codes)) != len(codes) or set(codes) != set(self.meta):
            raise AccountError(f"{day} 重复持股或持仓元信息不一致")
        if self.physical.cash < 0 or self.state.cash < 0 or self.state.credit != 0:
            raise AccountError(f"{day} 非杠杆现金约束失败：cash={self.physical.cash}")
        for batch in (0, 1):
            if sum(item["tranche_idx"] == batch for item in self.meta.values()) > 10:
                raise AccountError(f"{day} 批次 {batch} 超过十槽")

    def _close(self, day: str, value: float) -> None:
        """记录含应收的收盘账户；首日无持仓且 NAV=1，为收益计算的明确起点。"""
        trades = [row for row in self.rows["trades"] if row["trade_date"] == day]
        self.rows["daily"].append(
            {
                "trade_date": day,
                "nav": value / self.capital,
                "total_value": value,
                "cash": self.state.cash,
                "receivable": self.state.receivable,
                "market_value": value - self.state.cash - self.state.receivable,
                "fees": sum(row["fee"] for row in trades),
                "buy_amount": sum(row["amount"] for row in trades if row["kind"] == "market_buy"),
                "sell_amount": sum(row["amount"] for row in trades if row["kind"] == "market_sell"),
                "holding_count": len(self.state.lots),
            }
        )

    def _plan(self, day: str, nav: float) -> None:
        """收盘后先判断续持/卖出，再为排期批次冻结下一日新买预算。"""
        active = day in self.signal_days
        ranked = self.inputs.ranked[day] if active else []
        for lot in tuple(self.state.lots):
            code = lot.ts_code
            if code in self.pending_sells:
                continue
            meta = self.meta[code]
            held_days = self.index[day] - self.index[lot.buy_date]
            if not is_holding_period_exit_due(held_days, PERIOD * (meta["renewal_count"] + 1)):
                continue
            decision = due_decision(
                code,
                held_days,
                meta["renewal_count"],
                ranked,
                self.enabled,
                active_signal=active,
            )
            if not decision["due"]:
                raise AccountError(f"{day} 共享到期判定与 policy 不一致：{code}")
            self.rows["decisions"].append(
                {
                    "trade_date": day,
                    "kind": "expiry",
                    "ts_code": code,
                    "buy_date": lot.buy_date,
                    "tranche_idx": meta["tranche_idx"],
                    "holding_days": held_days,
                    "renewal_count_before": meta["renewal_count"],
                    "active_signal": active,
                    "model_version": (
                        self.inputs.params.model_version_of_day(day) if active else None
                    ),
                    **decision,
                }
            )
            if decision["renew"]:
                meta["renewal_count"] = int(decision["next_renewal_count"])
                due_idx = self.index[lot.buy_date] + PERIOD * (meta["renewal_count"] + 1)
                due_date = self.days[due_idx] if due_idx < len(self.days) else ""
                self.sleeves[SLEEVE] = replace(
                    self.state,
                    lots=tuple(
                        replace(item, due_date=due_date) if item.ts_code == code else item
                        for item in self.state.lots
                    ),
                )
            else:
                self.pending_sells[code] = {
                    "signal_date": day,
                    "due_date": lot.due_date,
                    "model_version": lot.origin_model_version,
                }
        if active and day in self.inputs.schedule:
            self._plan_batch(day, int(self.inputs.schedule[day]), ranked, nav)

    def _plan_batch(
        self, day: str, batch: int, ranked: Sequence[tuple[str, float]], nav: float
    ) -> None:
        """保留股占槽且占预算；待卖股仍在 held 中，禁止同日卖出又重买。"""
        self.pending_batch_limits[batch] = {
            "signal_date": day,
            "ceiling": 0.5 * nav,
            "holding_values": {
                lot.ts_code: lot.shares * self.prices.valuation(lot.ts_code, lot.acquired_price)[0]
                for lot in self.state.lots
                if self.meta[lot.ts_code]["tranche_idx"] == batch
            },
        }
        retained = [
            lot
            for lot in self.state.lots
            if self.meta[lot.ts_code]["tranche_idx"] == batch
            and lot.ts_code not in self.pending_sells
        ]
        retained_mv = sum(
            lot.shares * self.prices.valuation(lot.ts_code, lot.acquired_price)[0]
            for lot in retained
        )
        plan = plan_tranche(
            day,
            batch,
            ranked,
            set(self.meta),
            {lot.ts_code for lot in retained},
            retained_mv,
            nav,
            self.inputs.variance_book,
        )
        for row in plan["new_buys"]:
            order = {
                **row,
                "signal_date": day,
                "tranche_idx": batch,
                "model_version": self.inputs.params.model_version_of_day(day),
            }
            self.pending_buys.append(order)
            self.rows["decisions"].append({"trade_date": day, "kind": "buy_plan", **order})
        self.rows["decisions"].append(
            {
                "trade_date": day,
                "kind": "batch_plan",
                "tranche_idx": batch,
                "retained_count": len(retained),
                "retained_market_value": retained_mv,
                "nav_for_budget": nav,
                "new_budget": sum(row["budget"] for row in plan["new_buys"]),
                "retained_missing_score": list(plan["retained_missing_score"]),
            }
        )

    def _holdings(self, day: str) -> None:
        """日终持仓留原始 lot、买入日、续持次数与未决退出义务。"""
        for lot in self.state.lots:
            price, price_date, source = self.prices.valuation(lot.ts_code, lot.acquired_price)
            self.rows["holdings"].append(
                {
                    "trade_date": day,
                    **asdict(lot),
                    **self.meta[lot.ts_code],
                    "market_value": lot.shares * price,
                    "valuation_price": price,
                    "valuation_source": source,
                    "valuation_date": price_date,
                    "pending_exit": lot.ts_code in self.pending_sells,
                }
            )
