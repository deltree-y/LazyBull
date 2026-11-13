# -*- coding: utf-8 -*-
"""P7 T5 宿主：配额 → 授权 → 预净额 → 净卖 → 转账与净买 → 一次结算 → 借用服务。

日内顺序（**写死，不得交换**；任务书 §4.1）
========================================
1. 收集意图（T4 既有语义：买按执行日、卖按到期日 + 袖级顺延残差）；
2. **配额与授权**：``G``（账户级原始目标）→ ``D=max(0,G−MV)``、``R=max(0,ρG−MV)``
   → 逐袖缩放到 ``min(D, R)`` 并整手 → 费用后现金上界（``cash_open``）；
3. **预净额计划（零副作用）**：从授权后意图生成，只登记不落账；credit **不改写意图**，
   故「预净额 = 最终净额」，不存在第二次入账（证据登记 ``plan_identical``）；
4. **授信（VV-S）**：按「本日买入负债 − ``cash_open``」缺口申请，出借方按闲置比例分摊，
   借入方按缺口降序 + ID 升序决胜；``L`` **只本日转账一次**入账（``cash``/``credit`` 同步）；
   ``credit_outstanding`` **不参与**授权；``E_i <= 0`` 的袖禁新增买入与 credit；
   授信确定后**再授权一次**（``credit_granted`` 进入 ``R+L`` 与 ``cash_open+L`` 两条上界）；
5. **执行（复用 T4）**：内部抵消 → 净市场卖出 → 净市场买入 → 一次结算；
   出借方执行期现金保留 ``hold_i = max(0, −credit_i)``（``cash_i + credit_i >= 0`` 的直接推论），
   结算后加回（纯临时限制，不是转账，不改任何恒等式）；
6. **一次结算 + 双口径**：I1/I1b/I2/I3 含 credit 逐日闭合，``E_i = cash_i + MV_i − credit_i``；
7. **借用服务（结算后）**：召回 → FIFO 归还 → ``deadlocked`` 永久登记 → ``late_recovery``；
8. **同代提交**：账本本体 + 续跑态（价源缓存 / 禁买 / 序号 / **credit 台账**）一次校验。

硬契约
======
- ``credit=off & quota=off`` 时第 2/4/7 步全部为无操作 ⇒ 逐日账本摘要、费用台账、
  实际/独立 NAV、账本投影行、守恒锚与 T4 完成根**逐位一致**（T4 兼容门）；
- 原始卖出**不受配额约束**且**不得直接入市场**（仍走预净额 → 最终净额）；
- 本金转移**零 P&L**：借入/归还只改 ``cash``/``credit``，不改 ``MV`` 与 ``E``；
- 零 α 消耗：``alpha_consumed=false``，借期与 cap-binding 只作机械量。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

from scripts.v2_p7.t4 import fees as fee_mod
from scripts.v2_p7.t4.dual import run_daily_identities
from scripts.v2_p7.t4.host import (
    HistoricalVirtualSleeveHost,
    HostResult,
    _available,
    _dual_caliber,
    _fee_progress,
    _initial_physical,
    _initial_sleeves,
    _tracked_codes,
)
from scripts.v2_p7.t4.ledger import execute_plan
from scripts.v2_p7.t4.netting import Intent, plan_day
from scripts.v2_p7.t4.state import (
    SleeveState,
    _physical_payload,
    _sleeve_payload,
    stable_digest,
)
from scripts.v2_p7.t5.borrow import next_due_date
from scripts.v2_p7.t5.credit import CreditBook, allocate_credit
from scripts.v2_p7.t5.denials import (
    _append_denial_rows,
    _apply_denials_to_rows,
    _denied_split,
)
from scripts.v2_p7.t5.planutil import (
    _conservation_check,
    _cross_day,
    _hold_capped_notional,
    _per_sleeve_min,
    _plan_signature,
    plan_cash,
    plan_credit,
)
from scripts.v2_p7.t5.quota import (
    SleeveQuotaDecision,
    authorized_by_cash,
    decide_quota,
    holdings_market_value,
    quota_bounds,
)
from scripts.v2_p7.t5.recall import (
    deadlocked_snapshot,
    due_today,
    late_recovery_snapshot,
    service_credit_day,
)
from scripts.v2_p7.t5.runstate import (
    DayPlan,
    Settled,
    T5RunState,
)
from scripts.v2_p7.t5.state import (
    SleeveBalance,
    T5StateEnvelope,
    assert_nonnegative_gate,
    sleeve_balances,
    zero_pnl_check,
)

#: 配额拒绝原因（写入 ``voids``；与 T4 资金不足作废并列）


class T5HostError(RuntimeError):
    """T5 宿主硬门不满足（配额/credit/非负门），一律 fail-fast。"""


@dataclass(frozen=True)
class T5Mode:
    """机械模式：``quota`` / ``credit`` 两个独立开关（VV-H / VV-S / T4 互证）。"""

    quota: bool
    credit: bool
    label: str

    @property
    def is_reference(self) -> bool:
        """``credit=off & quota=off`` = T4 兼容模式。"""
        return not self.quota and not self.credit

    def payload(self) -> Dict[str, Any]:
        return {"label": self.label, "quota": bool(self.quota), "credit": bool(self.credit)}


MODE_T4_PARITY = T5Mode(quota=False, credit=False, label="T4-parity")
MODE_VVH = T5Mode(quota=True, credit=False, label="VV-H")
MODE_VVS = T5Mode(quota=True, credit=True, label="VV-S")


@dataclass
class T5HostResult(HostResult):
    """T5 结果（T4 结果 + 配额/借用/逐袖资产负债/非负门证据）。"""

    mode: Mapping[str, Any] = field(default_factory=dict)
    quota_rows: List[Dict[str, Any]] = field(default_factory=list)
    quota_daily: List[Dict[str, Any]] = field(default_factory=list)
    balance_rows: List[Dict[str, Any]] = field(default_factory=list)
    credit_payload: Dict[str, Any] = field(default_factory=dict)
    credit_daily: List[Dict[str, Any]] = field(default_factory=list)
    credit_series: List[Tuple[str, str]] = field(default_factory=list)
    gate_payload: Dict[str, Any] = field(default_factory=dict)
    credit_balance: Dict[str, Any] = field(default_factory=dict)
    zero_pnl: Dict[str, Any] = field(default_factory=dict)
    equity_blocks: List[Dict[str, Any]] = field(default_factory=list)
    credit_nav_rows: List[Dict[str, Any]] = field(default_factory=list)


class QuotaCreditHost(HistoricalVirtualSleeveHost):
    """逐日驱动「配额 + 借用」生命周期的历史宿主（VV-H / VV-S / T4 互证共用）。"""

    def __init__(
        self,
        *,
        mode: T5Mode = MODE_T4_PARITY,
        credit_book: Optional[CreditBook] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if mode.credit and not mode.quota:
            raise T5HostError(
                "非法模式组合：credit=on 必须同时 quota=on（无授权则授信无负债可背书）"
            )
        self.mode = mode
        self.credit = credit_book if credit_book is not None else CreditBook()
        #: 逐日授信/召回/死局/迟回收登记
        self.credit_daily: List[Dict[str, Any]] = []
        #: 逐日非负门行
        self.gate_rows: List[Dict[str, Any]] = []
        #: 最近一次授信分摊诊断（证据用）
        self._last_credit_diag: Dict[str, Any] = {}
        #: 逐日出借方执行期现金保留（``cash_i + credit_i >= 0`` 的执行期实现）
        self._hold_by_day: Dict[str, Dict[str, float]] = {}

    # ────────────────────────── 主流程 ──────────────────────────

    def run(self) -> T5HostResult:  # type: ignore[override]
        state = (
            self._restore_state(self._restore)
            if self._restore is not None
            else T5RunState(
                sleeves=_initial_sleeves(self.params),
                physical=_initial_physical(self.params),
                envelope=T5StateEnvelope(run_id=self.run_id),
            )
        )
        days = self._days
        for index, day in enumerate(days):
            self._run_day(day, index, len(days), state)
        return self._finalize(days, state)

    def _restore_state(self, payload: Mapping[str, Any]) -> T5RunState:
        """从序列化 T5 信封 + 费用快照恢复（credit 台账取自信封内 ``credit_state``）。"""
        envelope = T5StateEnvelope(run_id=self.run_id)
        envelope.restore_from(payload.get("envelope") or {})
        self.fee_book = fee_mod.FeeBook.restore(payload.get("fee_book") or {})
        self.credit = CreditBook.restore(envelope.credit_state())
        resume = envelope.resume_state
        recorded_mode = dict(resume.get("mode") or {})
        if recorded_mode and str(recorded_mode.get("label") or "") != self.mode.label:
            raise T5HostError(
                "信封模式与本宿主不一致（禁止跨模式续跑）："
                f"信封={recorded_mode.get('label')!r} 宿主={self.mode.label!r}"
            )
        self.seq = int(resume.get("seq", payload.get("seq") or 0))
        self._no_new_buy = {str(item) for item in (resume.get("no_new_buy") or ())}
        self.prices.restore_last_known(resume.get("price_last_known") or {})
        sleeves = envelope.sleeves or _initial_sleeves(self.params)
        physical = (
            envelope.physical if envelope.physical is not None else _initial_physical(self.params)
        )
        return T5RunState(sleeves=sleeves, physical=physical, envelope=envelope)

    def _run_day(self, day: str, index: int, total: int, state: T5RunState) -> None:
        """单日：配额 → 授信 → 预净额 → 执行 → 结算 → 借用服务 → 同代提交。"""
        intents = self._collect_intents(day, state.sleeves)
        plan = self._plan_stage(day, index, total, intents, state)
        outcome = self._execute_stage(day, state, plan)
        self._apply_quota_denials(day, plan.quota_post_credit, plan.decisions, outcome, state)
        day_costs = self._fee_stage(day, outcome, state)
        settled = self._settle_stage(day, outcome, state, plan)
        self._commit_stage(day, index, state, plan, settled, day_costs)
        self._record_stage(day, state, plan, settled, day_costs)

    # ────────────────────────── 阶段：计划 ──────────────────────────

    def _coverage(self, day: str, state: T5RunState, tracked: Sequence[str]) -> Dict[str, bool]:
        """续跑必须覆盖的状态（T4 三类 + T5 的 credit 三态）。"""
        held = {lot.ts_code for sleeve in state.sleeves.values() for lot in sleeve.lots}
        gaps = set(self.prices.active_gaps())
        return {
            "carry_residual": any(sleeve.residuals for sleeve in state.sleeves.values()),
            "suspension": bool(held & gaps) or bool(set(tracked) & gaps),
            "no_new_buy": bool(self._no_new_buy),
            "borrow_outstanding": bool(self.credit.open_lots()),
            "recall_due": bool(due_today(self.credit, day)),
            "deadlocked": any(lot.deadlocked and lot.open for lot in self.credit.lots),
        }

    def _plan_stage(
        self, day: str, index: int, total: int, intents: Sequence[Intent], state: T5RunState
    ) -> "DayPlan":
        """配额（额度口径）→ 授信（VV-S）→ 授权（含费用后现金）→ 预净额计划。"""
        tracked = _tracked_codes(state.sleeves, intents)
        self.prices.begin_day(day, tracked=tracked)
        coverage = self._coverage(day, state, tracked)
        is_checkpoint = state.wants_checkpoint(index, total, coverage)
        cash_open = {sid: float(item.cash) for sid, item in state.sleeves.items()}
        credit_open = {sid: float(item.credit) for sid, item in state.sleeves.items()}
        # C-T5-02：到期归还**优先于**新增买入（netting_freeze「归还 = 放弃自身新下单」）
        due_reserve = {
            sid: self._due_reserve(day, sid, float(cash_open[sid])) for sid in state.sleeves
        }
        cash_budget = {
            sid: max(0.0, float(cash_open[sid]) - float(due_reserve[sid])) for sid in state.sleeves
        }
        quota_only: Dict[str, SleeveQuotaDecision] = {}
        quota_post_credit: Dict[str, SleeveQuotaDecision] = {}
        decisions: Dict[str, SleeveQuotaDecision] = {}
        credit_events: List[Dict[str, Any]] = []
        cash_delta: Dict[str, float] = {}
        credit_delta: Dict[str, float] = {}
        pre_intents: List[Intent] = list(intents)
        if self.mode.is_reference:
            final_intents: List[Intent] = list(intents)
        else:
            quota_only, pre_intents = self._quota_stage(
                day, intents, state, credit_granted={}, cash_open=cash_open, apply_cash=False
            )
            grants = self._credit_stage(
                day=day,
                decisions=quota_only,
                cash_open=cash_open,
                cash_budget=cash_budget,
                due_reserve=due_reserve,
                credit_open=credit_open,
                state=state,
                intents=intents,
            )
            granted: Dict[str, float] = {}
            if self.mode.credit and grants:
                cash_delta, credit_delta, credit_events = self.credit.apply_grants(
                    day=day, grants=grants
                )
                # ``L_i`` **有符号**（出借方为负）：ΣL = 0 ⇒ ΣA ≤ ΣR
                granted = {sid: float(value) for sid, value in cash_delta.items()}
            decisions, final_intents = self._quota_stage(
                day,
                intents,
                state,
                credit_granted=granted,
                cash_open=cash_budget,
                apply_cash=True,
            )
            # R2-03：拒绝**原因分类**用「同一实际 L、未施加现金约束」的额度决策
            quota_post_credit, _ = self._quota_stage(
                day,
                intents,
                state,
                credit_granted=granted,
                cash_open=cash_budget,
                apply_cash=False,
            )
        available = _available(state.sleeves)
        # **预净额计划（零副作用）**：授权前意图集（额度口径）的净额计划，
        # 与授权后**最终净额**逐腿对照（契约顺序：预净额 → 授权 → 最终净额）
        pre_netting = plan_day(day, pre_intents, available)
        netting = plan_day(day, final_intents, available)
        pre_plan = _plan_signature(pre_netting)
        final_plan = _plan_signature(netting)
        # A-02：授信**转账腿**的零 P&L（借入/出借只改 cash 与 credit）
        grant_before = sleeve_balances(trade_date=day, sleeves=state.sleeves, prices=self.prices)
        grant_after_sleeves = {
            sid: replace(
                item,
                cash=float(plan_cash(cash_open, cash_delta, sid)),
                credit=float(plan_credit(credit_open, credit_delta, sid)),
            )
            for sid, item in state.sleeves.items()
        }
        grant_after = sleeve_balances(
            trade_date=day, sleeves=grant_after_sleeves, prices=self.prices
        )
        grant_zero_pnl = zero_pnl_check(before=grant_before, after=grant_after)
        return DayPlan(
            day=str(day),
            index=int(index),
            coverage=coverage,
            is_checkpoint=bool(is_checkpoint),
            pre=(state.sleeves, state.physical, self.seq),
            begin_envelope=state.envelope.to_payload() if is_checkpoint else {},
            begin_fees=self.fee_book.snapshot() if is_checkpoint else {},
            begin_credit=self.credit.snapshot() if is_checkpoint else {},
            cash_open=cash_open,
            cash_budget=cash_budget,
            due_reserve=due_reserve,
            credit_open=credit_open,
            quota_only=quota_only,
            quota_post_credit=quota_post_credit,
            decisions=decisions,
            final_intents=list(final_intents),
            netting=netting,
            plan_digest=stable_digest(
                [
                    {
                        "ts_code": group.ts_code,
                        "scenario": group.scenario,
                        "offset": group.offset_shares,
                        "market": group.market_shares,
                    }
                    for group in netting.groups
                ]
            ),
            pre_netting=pre_netting,
            pre_plan=pre_plan,
            final_plan=final_plan,
            pre_intents=list(pre_intents),
            grant_zero_pnl=grant_zero_pnl,
            cash_delta=cash_delta,
            credit_delta=credit_delta,
            credit_events=credit_events,
        )

    # ────────────────────────── 阶段：执行 ──────────────────────────

    def _execute_stage(self, day: str, state: T5RunState, plan: "DayPlan") -> Any:
        """执行（复用 T4）：出借方执行期现金保留，结算后原额加回（不是转账）。

        ``hold_i = max(0, −credit_i)`` 是 ``cash_i + credit_i ≥ 0`` 的执行期实现：
        出借方贷出 ``L`` 后仍须保有 ``L`` 现金，故执行期可用现金再减 ``L``。
        该保留**逐日登记**（``hold_by_sleeve``），其导致的额外截断另计
        ``hold_capped_notional``（不与授权期/市场阻断混报）。
        """
        credit_hold = {
            sid: max(0.0, -float(plan.credit_open.get(sid, 0.0) + plan.credit_delta.get(sid, 0.0)))
            for sid in state.sleeves
        }
        self._hold_by_day[day] = {sid: float(value) for sid, value in sorted(credit_hold.items())}
        # 到期归还预留同样在执行期冻结（否则新买可花掉应归还的现金）
        hold = {
            sid: float(credit_hold[sid]) + float(plan.due_reserve.get(sid, 0.0))
            for sid in state.sleeves
        }
        exec_sleeves = {
            sid: SleeveState(
                sleeve_id=sid,
                cash=float(plan.cash_budget[sid])
                + float(plan.cash_delta.get(sid, 0.0))
                - float(credit_hold[sid]),
                credit=float(plan.credit_open[sid]) + float(plan.credit_delta.get(sid, 0.0)),
                lots=state.sleeves[sid].lots,
                residuals=state.sleeves[sid].residuals,
                borrow_lots=state.sleeves[sid].borrow_lots,
                recalled_amount=float(state.sleeves[sid].recalled_amount),
                deadlocked_amount=float(state.sleeves[sid].deadlocked_amount),
            )
            for sid in state.sleeves
        }
        outcome = execute_plan(
            plan=plan.netting,
            sleeves=exec_sleeves,
            physical=state.physical,
            prices=self.prices,
            cost_model=self.cost_model,
            lot_size=int(self.params.lot_size),
            seq_base=self.seq,
        )
        self.seq += max(0, len(outcome.trades)) + 1
        restored = {
            sid: replace(item, cash=float(item.cash) + float(hold.get(sid, 0.0)))
            for sid, item in outcome.sleeves.items()
        }
        return replace(outcome, sleeves=restored)

    def _fee_stage(self, day: str, outcome: Any, state: T5RunState) -> List[Any]:
        """费用台账（Q3=A 复用 T4）与逐日累计进度。"""
        day_costs = self.fee_book.add_day(
            day,
            outcome.fee_entries,
            cost_of_by_side=self.cost_of_by_side,
            closures=outcome.closures,
            declared_fees=outcome.declared_fees,
        )
        state.fee_series.append((day, _fee_progress(self.fee_book)))
        return day_costs

    # ────────────────────────── 阶段：结算 ──────────────────────────

    def _settle_stage(
        self, day: str, outcome: Any, state: T5RunState, plan: "DayPlan"
    ) -> "Settled":
        """借用服务（结算后）→ 双口径 → 非负门 → 零 P&L 核对。"""
        balance_before = sleeve_balances(
            trade_date=day, sleeves=outcome.sleeves, prices=self.prices
        )
        # B-13：执行期 credit 必须等于「期初 + 本日转账一次」，不得由第三处改写
        for sid in sorted(outcome.sleeves):
            expected = float(plan.credit_open.get(sid, 0.0)) + float(
                plan.credit_delta.get(sid, 0.0)
            )
            observed = float(outcome.sleeves[sid].credit)
            if abs(observed - expected) > 1e-9:
                raise T5HostError(
                    f"{day} 袖 {sid} 执行期 credit 被改写：{observed!r} != 期初+转账 {expected!r}"
                )
        service: Dict[str, Any] = {}
        if self.mode.credit or self.credit.open_lots():
            service = service_credit_day(
                book=self.credit,
                day=day,
                cash={sid: float(item.cash) for sid, item in outcome.sleeves.items()},
            )
        final_sleeves = self._settle_credit_state(outcome.sleeves, service)
        outcome = replace(outcome, sleeves=final_sleeves)
        savings = self.fee_book.savings_by_sleeve()
        report = run_daily_identities(
            trade_date=day,
            sleeves=final_sleeves,
            physical=outcome.physical,
            prices=self.prices,
            physical_initial=float(self.params.physical_capital),
            sleeve_capital=dict(self.params.sleeve_capital),
            cumulative_savings=savings,
        )
        balance_after = sleeve_balances(trade_date=day, sleeves=final_sleeves, prices=self.prices)
        gate = assert_nonnegative_gate(balance_after)
        zero_pnl = zero_pnl_check(before=balance_before, after=balance_after)
        state.balance_rows.extend(row.payload() for row in balance_after)
        self.gate_rows.append({"trade_date": day, **gate})
        state.record_physical_cash(day, float(outcome.physical.cash))
        state.offset_capped_shares += int(getattr(outcome, "offset_capped_shares", 0))
        state.record_identity(day, report)
        if not report.ok:  # pragma: no cover - record_identity 已登记，双口径失败即停
            raise T5HostError(f"{day} 恒等式破灭（I1/I2/I3）")
        state.sleeves = final_sleeves
        state.physical = outcome.physical
        self._apply_equity_policy(report, state)
        state.record_cash_guard(day, final_sleeves)
        credit_balance = self.credit.assert_balanced(
            tuple(sorted(final_sleeves)), day=day, record=True
        )
        state.credit_series.append((day, self.credit.digest()))
        return Settled(
            outcome=outcome,
            report=report,
            balances=balance_after,
            gate=gate,
            zero_pnl=zero_pnl,
            service=service,
            savings=savings,
            credit_balance=credit_balance,
            credit_digest=self.credit.digest(),
        )

    # ────────────────────────── 阶段：提交与登记 ──────────────────────────

    def _commit_stage(
        self,
        day: str,
        index: int,
        state: T5RunState,
        plan: "DayPlan",
        settled: "Settled",
        day_costs: Sequence[Any],
    ) -> None:
        """同代提交（账本 + 续跑态 + credit 台账）与检查点快照。"""
        state.envelope.t5_view = self._t5_view(
            day, settled.credit_balance, settled.service, settled.balances
        )
        state.envelope.commit(
            key=f"{self.run_id}:{day}",
            generation=state.envelope.generation + 1,
            resume_state={
                "price_last_known": self.prices.snapshot_last_known(),
                "no_new_buy": sorted(self._no_new_buy),
                "seq": int(self.seq),
                "mode": self.mode.payload(),
                "credit_state": self.credit.snapshot(),
            },
            fingerprint=stable_digest(
                {
                    "day": day,
                    "scenarios": settled.outcome.scenario_counts,
                    "offset_shares": settled.outcome.offset_shares,
                    "market_filled": settled.outcome.market_filled,
                    "trades": len(settled.outcome.trades),
                    "day_costs": len(day_costs),
                }
            ),
            sleeves=state.sleeves,
            physical=state.physical,
            as_of=day,
        )
        state.digest_series.append((day, state.envelope.state_digest()))
        if not plan.is_checkpoint:
            return
        state.checkpoints.append(
            {
                "trade_date": day,
                "index": int(index),
                "sleeves": plan.pre[0],
                "physical": plan.pre[1],
                "seq": int(plan.pre[2]),
                "no_new_buy": sorted(self._no_new_buy),
                "state_payload": {
                    "as_of": day,
                    "sleeves": _sleeve_payload(plan.pre[0]),
                    "physical": _physical_payload(plan.pre[1]),
                },
                "envelope": dict(plan.begin_envelope),
                "fee_state": dict(plan.begin_fees),
                "credit_state": dict(plan.begin_credit),
                "end_state_payload": state.envelope.state_payload(),
                "end_credit_digest": settled.credit_digest,
                "price_last_known": self.prices.snapshot_last_known(),
                "valuation_probe": self._valuation_probe(plan.pre[0]),
                "coverage": {str(k): bool(v) for k, v in plan.coverage.items()},
            }
        )

    def _record_stage(
        self,
        day: str,
        state: T5RunState,
        plan: "DayPlan",
        settled: "Settled",
        day_costs: Sequence[Any],
    ) -> None:
        """逐日序列登记（NAV / 配额 / 借用 / 账本投影行）。"""
        state.record_nav(day, settled.report, int(self.params.split_of(day)))
        state.envelope.dual_caliber = _dual_caliber(
            self.params.arms, settled.report, settled.savings, self.fee_book
        )
        state.accumulate(day, settled.outcome, len(day_costs))
        state.quota_rows.extend(
            decision.payload() for _sid, decision in sorted(plan.decisions.items())
        )
        state.quota_daily.append(
            {
                "trade_date": day,
                "gross_target": float(self.params.target_of(day)),
                "arm_target": {
                    arm: float(self.params.arm_target_of(day, arm)) for arm in self.params.arms
                },
                "decisions": {
                    sid: decision.payload() for sid, decision in sorted(plan.decisions.items())
                },
                "plan_digest": plan.plan_digest,
                "quota_post_credit": {
                    sid: {
                        "authorized_shares": int(decision.authorized_shares),
                        "authorized_notional": float(decision.authorized_notional),
                        "denied_shares": int(decision.denied_shares),
                        "quota_bound": float(decision.quota_bound),
                    }
                    for sid, decision in sorted(plan.quota_post_credit.items())
                },
                "pre_plan": dict(plan.pre_plan),
                "final_plan": dict(plan.final_plan),
                "plan_identical": bool(plan.pre_plan["digest"] == plan.final_plan["digest"]),
                "authorization_delta": {
                    "offset_shares": int(plan.final_plan["offset_shares"])
                    - int(plan.pre_plan["offset_shares"]),
                    "market_shares": int(plan.final_plan["market_shares"])
                    - int(plan.pre_plan["market_shares"]),
                    "market_legs": int(plan.final_plan["market_legs"])
                    - int(plan.pre_plan["market_legs"]),
                },
                "quota_only": {
                    sid: {
                        "authorized_shares": int(decision.authorized_shares),
                        "authorized_notional": float(decision.authorized_notional),
                        "authorized_fee": float(decision.authorized_fee),
                        "denied_shares": int(decision.denied_shares),
                        "quota_bound": float(decision.quota_bound),
                    }
                    for sid, decision in sorted(plan.quota_only.items())
                },
                "cap_binding_sleeves": sorted(
                    sid for sid, decision in plan.decisions.items() if decision.cap_binding
                ),
                "quota_enabled": bool(self.mode.quota),
                "credit_enabled": bool(self.mode.credit),
            }
        )
        credit_day = {
            "trade_date": day,
            "grants": [dict(item) for item in plan.credit_events],
            "allocation": dict(self._last_credit_diag),
            "hold_by_sleeve": dict(self._hold_by_day.get(day, {})),
            "due_reserve_by_sleeve": {
                sid: float(value) for sid, value in sorted(plan.due_reserve.items())
            },
            "hold_capped_notional": _hold_capped_notional(
                plan, settled.outcome.trades, self._hold_by_day.get(day, {})
            ),
            "transfer_cash_capped": int(
                (settled.outcome.blocks or {}).get("transfer_cash_capped", 0)
            ),
            "grant_zero_pnl": {
                "ok": bool(plan.grant_zero_pnl["ok"]),
                "rows": list(plan.grant_zero_pnl["rows"]),
            },
            "recall_queue": list((settled.service or {}).get("recall_queue") or ()),
            "repayments": list((settled.service or {}).get("repayments") or ()),
            "deadlocks": list((settled.service or {}).get("deadlocks") or ()),
            "late_recoveries": list((settled.service or {}).get("late_recoveries") or ()),
            "right_censored": list((settled.service or {}).get("right_censored") or ()),
            "credit_by_sleeve": dict(settled.credit_balance["credit_by_sleeve"]),
            "nonnegative_gate": {
                "ok": bool(settled.gate["ok"]),
                "min": settled.gate["min_cash_plus_credit"],
            },
            "zero_pnl": {"ok": bool(settled.zero_pnl["ok"])},
        }
        state.record_credit_nav(
            day,
            settled.report,
            settled.credit_balance["credit_by_sleeve"],
            physical_capital=float(self.params.physical_capital),
            sleeve_capital=self.params.sleeve_capital,
            savings=settled.savings,
        )
        self.credit_daily.append(credit_day)
        state.credit_daily.append(credit_day)
        if len(state.day_rows) % 250 == 0:
            logger.info(
                f"{self.run_id}[{self.mode.label}] {day} 第 {len(state.day_rows)} 日；"
                f"物理现金={settled.outcome.physical.cash:.2f} "
                f"credit={settled.credit_balance['credit_by_sleeve']}"
            )

    # ────────────────────────── 配额与授信 ──────────────────────────

    def _quota_stage(
        self,
        day: str,
        intents: Sequence[Intent],
        state: T5RunState,
        credit_granted: Mapping[str, float],
        cash_open: Optional[Mapping[str, float]] = None,
        apply_cash: bool = True,
    ) -> Tuple[Dict[str, SleeveQuotaDecision], List[Intent]]:
        """逐袖配额与授权（``quota=off`` 时逐位返回入参意图）。

        ``cash_open`` 缺省取袖当前现金；**二次授权**时传入 ``cash + L``
        （``L`` 已按本日转账一次入账，不叠加存量 ``credit_outstanding``）。
        ``apply_cash=False`` 只套额度上界（借用申请额与 cap-binding 诊断口径）。
        """
        by_sleeve: Dict[str, List[Intent]] = {sid: [] for sid in state.sleeves}
        for intent in intents:
            by_sleeve.setdefault(intent.sleeve_id, []).append(intent)
        decisions: Dict[str, SleeveQuotaDecision] = {}
        final: List[Intent] = []
        gross = float(self.params.target_of(day))
        for sid in sorted(state.sleeves):
            sleeve = state.sleeves[sid]
            cash = float((cash_open or {}).get(sid, sleeve.cash))
            mv_hold = 0.0
            if self.mode.quota:
                mv_hold, _detail = holdings_market_value(sleeve, self.prices)
            bounds = quota_bounds(
                gross_target=gross, mv_hold=mv_hold, ratio=self.params.ratio_of(sid)
            )
            kept, decision = decide_quota(
                trade_date=day,
                sleeve=sleeve,
                intents=by_sleeve.get(sid, ()),
                bounds=bounds,
                cash_open=cash,
                credit_granted=float(credit_granted.get(sid, 0.0)),
                credit_outstanding=float(sleeve.credit),
                price_of=self._open_price,
                buy_cost=self.cost_model.calculate_buy_cost,
                lot_size=int(self.params.lot_size),
                enabled=bool(self.mode.quota),
                apply_cash=bool(apply_cash),
            )
            decisions[sid] = decision
            final.extend(kept)
        return decisions, final

    def _open_price(self, ts_code: str) -> Optional[float]:
        lookup = self.prices.market_open(ts_code)
        if not lookup.ok or lookup.price is None:
            return None
        return float(lookup.price)

    def _credit_stage(
        self,
        *,
        day: str,
        decisions: Mapping[str, SleeveQuotaDecision],
        cash_open: Mapping[str, float],
        cash_budget: Mapping[str, float],
        due_reserve: Mapping[str, float],
        credit_open: Mapping[str, float],
        state: T5RunState,
        intents: Optional[Sequence[Intent]] = None,
    ) -> List[Any]:
        """缺口申请 → 出借方按闲置比例分摊 → 授信清单（T+1 召回日）。

        * **原始需求**（未受 ``ρG`` 截断的申报买入负债）决定申请额与出借方保留（C-T5-03）：
          ``need = max(原始需求含费, 配额后负债含费)``，不再把配额后需求当原始需求；
        * **到期归还预留**先于新增买入扣减可用现金（C-T5-02）：
          ``归还 = 借入方放弃自身新下单``；
        * 是否值得立 lot 用「授信前后的整手可授权量是否变化」判定，取代原先
          「缺口 < 一手名义」的绝对门（C-T5-05；只缺手续费/已备大部分本金的情形必须能借）。
        """
        requests: Dict[str, float] = {}
        idle: Dict[str, float] = {}
        blocked_requests: List[Tuple[str, float, str]] = []
        raw_need = {sid: self._raw_buy_need(sid, intents or ()) for sid in sorted(state.sleeves)}
        for sid in sorted(state.sleeves):
            decision = decisions[sid]
            cash = float(cash_open[sid])
            budget = float(cash_budget[sid])
            quota_need = float(decision.authorized_notional) + float(decision.authorized_fee)
            need = max(quota_need, float(raw_need[sid]))
            gap = float(max(0.0, need - budget))
            reserve = max(0.0, -float(credit_open.get(sid, 0.0)))
            # E_i = cash + MV − credit（MV 取当日开盘口径，即授权时点已知量）
            equity = cash + float(decision.bounds.mv_hold) - float(credit_open.get(sid, 0.0))
            blocked = sid in self._no_new_buy or equity <= 0.0
            if blocked and gap > 0.0:
                blocked_requests.append(
                    (
                        sid,
                        float(gap),
                        "blocked_nonpositive_equity" if equity <= 0.0 else "blocked_no_new_buy",
                    )
                )
                continue
            if gap > 0.0:
                bound = float(decision.quota_bound)
                room = min(float(decision.bounds.demand), bound + gap)
                shares_now = self._authorized_shares(sid, intents or (), budget, bound)
                shares_with = self._authorized_shares(sid, intents or (), budget + gap, room)
                if shares_with <= shares_now:
                    # 授信不改变任何整手成交 => 不立 lot（不是「缺口太小」）
                    blocked_requests.append((sid, float(gap), "credit_cannot_change_fill"))
                    continue
                requests[sid] = float(gap)
            # 可贷闲置：``cash_i + credit_i >= 0`` 要求出借方贷出后仍保有不低于**累计贷出额**
            # 的现金；而贷出本身已扣减现金 ⇒ 代数上 ``L <= (cash_budget − 自有原始需求 − 既有出借)/2``。
            # 该系数是冻结硬门的直接推论（不是保守 haircut）；保留量取**原始需求**而非 ρG 后的 R。
            idle[sid] = float(max(0.0, (budget - reserve - need) / 2.0))
        for sid, gap, status in blocked_requests:
            self.credit.record_request(
                day=day,
                sleeve=sid,
                gap=gap,
                need=gap,
                cash_open=float(cash_open[sid]),
                granted=0.0,
                status=status,
            )
        if not self.mode.credit:
            for sid in sorted(requests):
                self.credit.record_request(
                    day=day,
                    sleeve=sid,
                    gap=float(requests[sid]),
                    need=float(requests[sid]),
                    cash_open=float(cash_open[sid]),
                    granted=0.0,
                    status="credit_disabled",
                )
            self._last_credit_diag = {
                "grant_count": 0,
                "granted_total": 0.0,
                "credit": "off",
                "due_reserve": {sid: float(value) for sid, value in sorted(due_reserve.items())},
            }
            return []
        # T+1 召回日按**全窗日历**解析（分段/续跑子窗口的末日仍有下一交易日）
        due = next_due_date(tuple(self.params.extended_days), day)
        grants, diag = allocate_credit(requests=requests, idle=idle, due_date=due)
        # R1-04：完整请求「可成交」不等于**实际部分授信**可成交；按真实 L 重验后再落账
        grants, dropped = self._filter_grants_by_fill_gain(
            day=day,
            decisions=decisions,
            cash_budget=cash_budget,
            cash_open=cash_open,
            requests=requests,
            intents=intents or (),
            grants=grants,
        )
        if dropped:
            diag["partial_no_gain"] = dropped
        granted_by_borrower: Dict[str, float] = {}
        for grant in grants:
            granted_by_borrower[grant.borrower] = granted_by_borrower.get(
                grant.borrower, 0.0
            ) + float(grant.amount)
        for sid in sorted(requests):
            self.credit.record_request(
                day=day,
                sleeve=sid,
                gap=float(requests[sid]),
                need=float(requests[sid]),
                cash_open=float(cash_open[sid]),
                granted=float(granted_by_borrower.get(sid, 0.0)),
                status="granted" if granted_by_borrower.get(sid, 0.0) > 0.0 else "no_idle",
            )
        diag["due_reserve"] = {sid: float(value) for sid, value in sorted(due_reserve.items())}
        diag["raw_need"] = {sid: float(value) for sid, value in sorted(raw_need.items())}
        self._last_credit_diag = diag
        return grants

    def _filter_grants_by_fill_gain(
        self,
        *,
        day: str,
        decisions: Mapping[str, SleeveQuotaDecision],
        cash_budget: Mapping[str, float],
        cash_open: Mapping[str, float],
        requests: Mapping[str, float],
        intents: Sequence[Intent],
        grants: Sequence[Any],
    ) -> Tuple[List[Any], int]:
        """按**实际**分配额重验「能否多成交一手」；不能则拒绝落账（R1-04）。"""
        kept: List[Any] = []
        dropped = 0
        for grant in grants:
            sid = str(grant.borrower)
            decision = decisions[sid]
            bound = float(decision.quota_bound)
            budget = float(cash_budget[sid])
            amount = float(grant.amount)
            room = min(float(decision.bounds.demand), bound + amount)
            before = self._authorized_shares(sid, intents, budget, bound)
            after = self._authorized_shares(sid, intents, budget + amount, room)
            if after <= before:
                dropped += 1
                self.credit.record_request(
                    day=day,
                    sleeve=sid,
                    gap=float(requests.get(sid, 0.0)),
                    need=float(requests.get(sid, 0.0)),
                    cash_open=float(cash_open[sid]),
                    granted=0.0,
                    status="credit_partial_no_gain",
                )
                continue
            kept.append(grant)
        return kept, dropped

    def _due_reserve(self, day: str, sleeve_id: str, cash: float) -> float:
        """当日到期（FIFO）本金的归还预留，按该袖现金上限截断。"""
        total = 0.0
        for lot in due_today(self.credit, day):
            if str(getattr(lot, "borrower", "")) != str(sleeve_id):
                continue
            if not getattr(lot, "open", False):
                continue
            total += float(getattr(lot, "outstanding", 0.0) or 0.0)
        return float(min(max(0.0, total), max(0.0, float(cash))))

    def _raw_buy_need(self, sleeve_id: str, intents: Sequence[Intent]) -> float:
        """该袖当日**原始申报买入负债**（未整手、未配额）= 名义 + 申报费用。

        费用按 ``ts_code`` 分组逐股票计算：不同证券不能合并享受一次最低佣金（R2-02）。
        """
        by_code: Dict[str, float] = {}
        for intent in intents or ():
            if intent.sleeve_id != sleeve_id or intent.side != "buy":
                continue
            price = self._open_price(intent.ts_code)
            if price is None or price <= 0.0:
                continue
            code = str(intent.ts_code)
            by_code[code] = by_code.get(code, 0.0) + float(intent.shares) * float(price)
        total = 0.0
        for code in sorted(by_code):
            amount = float(by_code[code])
            if amount <= 0.0:
                continue
            total += amount + float(self.cost_model.calculate_buy_cost(amount))
        return float(total)

    def _authorized_shares(
        self, sleeve_id: str, intents: Sequence[Intent], cash: float, bound: float
    ) -> int:
        """给定现金与额度上界时该袖可授权的整手股数（授信可成交性判定用）。"""
        priced: List[Tuple[Any, float]] = []
        for intent in intents or ():
            if intent.sleeve_id != sleeve_id or intent.side != "buy":
                continue
            price = self._open_price(intent.ts_code)
            if price is None or price <= 0.0:
                continue
            priced.append((intent, float(price)))
        if not priced:
            return 0
        result = authorized_by_cash(
            priced=priced,
            cash=float(cash),
            buy_cost=self.cost_model.calculate_buy_cost,
            lot_size=int(self.params.lot_size),
            bound=float(bound),
        )
        return int(result["shares"])

    def _min_lot_notional(self, intents: Sequence[Intent], sleeve_id: str) -> float:
        """该袖当日**最小一手名义**（缺口低于此值不可能改变任何成交）。"""
        prices = [
            float(price)
            for intent in intents
            if intent.sleeve_id == sleeve_id and intent.side == "buy"
            for price in [self._open_price(intent.ts_code)]
            if price is not None and float(price) > 0.0
        ]
        if not prices:
            return 0.0
        return float(min(prices)) * float(self.params.lot_size)

    def _settle_credit_state(
        self, sleeves: Mapping[str, SleeveState], service: Mapping[str, Any]
    ) -> Dict[str, SleeveState]:
        """把借用服务后的现金与派生 credit 余额写回袖快照（Σ credit 精确 0）。"""
        cash = dict((service or {}).get("cash") or {})
        payload = self.credit.credit_by_sleeve(tuple(sorted(sleeves)))
        balances = payload["balances"]
        out: Dict[str, SleeveState] = {}
        for sid in sorted(sleeves):
            state = sleeves[sid]
            out[sid] = replace(
                state,
                cash=float(cash.get(sid, state.cash)),
                credit=float(balances.get(sid, 0.0)),
            )
        return out

    def _apply_quota_denials(
        self,
        day: str,
        quota_only: Mapping[str, SleeveQuotaDecision],
        decisions: Mapping[str, SleeveQuotaDecision],
        outcome: Any,
        state: T5RunState,
    ) -> None:
        """配额/现金拒绝并入当日守恒行（同一意图一行：``original`` 复原为配额前股数）。

        两类拒绝**分字段计数**（不得混报）：``quota_capped_voided_eod``（被
        ``min(D, R+L)`` 额度截断）与 ``insufficient_cash_voided_eod``（额度内但费用后
        现金不足，T4 既有语义）。逐意图拆分以**额度口径**为准：额度拒绝取额度口径下
        的拒绝股数，其余计入现金拒绝（两类拒绝随缩放比例单调，故差值非负）。
        """
        del state
        quota_denied, cash_denied = _denied_split(quota_only, decisions)
        if not quota_denied and not cash_denied:
            return
        seen = _apply_denials_to_rows(
            outcome.conservation_rows, quota_denied, cash_denied, outcome.voids
        )
        _append_denial_rows(day, outcome, quota_denied, cash_denied, seen)

    def _apply_equity_policy(
        self,
        report: Any,
        state: Optional[T5RunState] = None,
        no_new_buy: Optional[set] = None,
    ) -> set:
        """``E_i = cash_i + MV_i − credit_i <= 0`` 的袖：禁新增买入与 **credit**。"""
        target = self._no_new_buy if no_new_buy is None else no_new_buy
        credit = {sid: float(item.credit) for sid, item in (state.sleeves.items() if state else ())}
        blocked: List[str] = []
        for arm in self.params.arms:
            total = float(report.total_value.get(arm, 1.0))
            equity = total - float(credit.get(arm, 0.0))
            if equity <= 0.0:
                blocked.append(arm)
        if state is not None:
            for arm in blocked:
                state.equity_blocks.append(
                    {
                        "trade_date": report.trade_date,
                        "sleeve": arm,
                        "equity": float(report.total_value.get(arm, 0.0))
                        - float(credit.get(arm, 0.0)),
                        "action": "no_new_buy_no_new_credit",
                    }
                )
        for arm in blocked:
            target.add(arm)
        return target

    # ────────────────────────── 证据 ──────────────────────────

    def _t5_view(
        self,
        day: str,
        credit_balance: Mapping[str, Any],
        service: Mapping[str, Any],
        balances: Sequence[SleeveBalance],
    ) -> Dict[str, Any]:
        """信封内的 credit 扩展视图（credit / 借款 lot / 召回 / 死局 / 迟回收）。"""
        return {
            "trade_date": str(day),
            "credit_by_sleeve": dict(credit_balance["credit_by_sleeve"]),
            "credit_residue_adjustment": float(credit_balance["residue_adjustment"]),
            "borrow_lots": [lot.payload() for lot in self.credit.open_lots()],
            "borrow_lot_count": len(self.credit.lots),
            "recall_queue": list((service or {}).get("recall_queue") or ()),
            "deadlocked": deadlocked_snapshot(self.credit),
            "late_recovery": late_recovery_snapshot(self.credit),
            "cash_plus_credit_min": min((row.cash_plus_credit for row in balances), default=None),
            "e_sum": float(sum(row.equity for row in balances)),
        }

    def _finalize(self, days: Tuple[str, ...], state: T5RunState) -> T5HostResult:
        """收尾：费用闭合 + 意图守恒 + 双口径 + 非负门 + 重放/重启自验。"""
        self.fee_book.assert_closed()
        conservation_check = _conservation_check(state.conservation)
        cross_day = _cross_day(state.conservation)
        if conservation_check["violations"]:
            raise T5HostError(
                f"意图守恒破缺：{conservation_check['violations']} 条不满足 "
                "original = offset + filled + voided + carry"
            )
        price_payload = self.prices.payload()
        if self._self_check:
            from scripts.v2_p7.t5.restart import audit_restarts

            restart = audit_restarts(self, state)
            replay = self._replay_check()
        else:
            restart = {"checked": False, "note": "续跑子宿主跳过重放/重启自验（避免递归）"}
            replay = {"checked": False, "note": "续跑子宿主跳过重放/重启自验（避免递归）"}
        credit_balance = self.credit.assert_balanced(tuple(sorted(state.sleeves)))
        gate_rows = [row for row in self.gate_rows]
        min_gate = min(
            (
                float(row["min_cash_plus_credit"])
                for row in gate_rows
                if row.get("min_cash_plus_credit") is not None
            ),
            default=None,
        )
        return T5HostResult(
            run_id=self.run_id,
            days=days,
            daily_rows=state.day_rows,
            trades=state.trades,
            conservation_rows=state.conservation,
            scenario_counts=dict(sorted(state.scenario_counts.items())),
            offset_legs=state.offset_legs,
            offset_shares=state.offset_shares,
            market_filled=dict(sorted(state.market_filled.items())),
            blocks=dict(sorted(state.blocks.items())),
            voids=dict(sorted(state.voids.items())),
            carries=dict(sorted(state.carries.items())),
            envelope=state.envelope,  # type: ignore[arg-type]
            fee_book=self.fee_book,
            price_payload=price_payload,
            identity_failures=state.identity_failures,
            nav_series=state.nav_series,
            invalid_days=state.invalid_days,
            conservation=conservation_check,
            cross_day=cross_day,
            replay=replay,
            restart=restart,
            fold_boundaries=self._fold_boundaries(),
            digest_series=state.digest_series,
            fee_series=state.fee_series,
            min_sleeve_cash=state.min_sleeve_cash,
            min_physical_cash=state.min_physical_cash,
            offset_capped_shares=int(state.offset_capped_shares),
            negative_physical_cash_days=int(state.physical_cash_days_negative),
            negative_physical_cash_examples=list(state.physical_cash_examples),
            negative_cash_days=int(state.negative_cash_days),
            negative_cash_examples=list(state.negative_cash_examples),
            equity_policy_events=list(state.equity_blocks),
            equity_blocks=list(state.equity_blocks),
            credit_nav_rows=state.credit_nav_rows,
            mode=self.mode.payload(),
            quota_rows=list(state.quota_rows),
            quota_daily=list(state.quota_daily),
            balance_rows=list(state.balance_rows),
            credit_payload=self.credit.payload(),
            credit_daily=list(state.credit_daily),
            credit_series=list(state.credit_series),
            gate_payload={
                "ok": not any(row["ok"] is False for row in gate_rows),
                "days": len(gate_rows),
                "violation_days": sum(1 for row in gate_rows if row["ok"] is False),
                "min_cash_plus_credit": min_gate,
                "per_sleeve_min": _per_sleeve_min(state.balance_rows),
                "tolerance": 1e-6,
                "rule": "cash_i + credit_i >= 0（逐日硬门；负现金必须由 credit 背书）",
            },
            credit_balance=dict(credit_balance),
            zero_pnl={
                "checked_days": len(state.credit_daily),
                "violations": [
                    item for item in state.credit_daily if item["zero_pnl"]["ok"] is False
                ],
                "rule": "本金转移零 P&L：无价变时 ΔE_i = 0（借入/归还只改 cash 与 credit）",
            },
        )

    def _replay_check(self) -> Dict[str, Any]:
        """同一配置独立重放（含配额与借用）：逐日状态摘要必须逐位一致。"""
        window = self._replay_window()
        if not window:
            return {"checked": False, "note": "日历过短，未抽样"}
        first = self._replay_window_once(window)
        second = self._replay_window_once(window)
        identical = [item[1] for item in first] == [item[1] for item in second]
        return {
            "checked": True,
            "window": [window[0], window[-1]],
            "days": len(window),
            "all_bitwise_identical": bool(identical),
            "price_book_isolated": self._price_factory is not None,
            "credit_included": True,
            "first_digest": first[0][1] if first else "",
            "last_digest": first[-1][1] if first else "",
        }

    def _replay_window_once(self, window: Sequence[str]) -> List[Tuple[str, str]]:
        """从零初资跑一个连续窗口，返回逐日 ``(日期, 状态+credit 摘要)``。"""
        child = QuotaCreditHost(
            params=self.params,
            run_id=f"{self.run_id}:replay",
            prices=self._fresh_prices(),
            cost_model=self.cost_model,
            buy_index=self.buy_index,
            sell_index=self.sell_index,
            days=window,
            price_factory=self._price_factory,
            mode=self.mode,
            self_check=False,
        )
        result = child.run()
        credit_by_day = {day: digest for day, digest in result.credit_series}
        return [
            (day, stable_digest({"state": digest, "credit": credit_by_day.get(day, "")}))
            for day, digest in result.digest_series
        ]


def serialize_envelope(result: T5HostResult) -> str:
    """信封 JSON 序列化（证据原文）。"""
    return json.dumps(result.envelope.to_payload(), ensure_ascii=False, sort_keys=True)
