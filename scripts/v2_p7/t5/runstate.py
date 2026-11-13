# -*- coding: utf-8 -*-
"""P7 T5 逐日运行累加器与阶段产物（计划 / 结算）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from scripts.v2_p7.t4.host import _finite_or_none
from scripts.v2_p7.t4.netting import Intent
from scripts.v2_p7.t4.state import PhysicalBook, SleeveState
from scripts.v2_p7.t5.quota import SleeveQuotaDecision
from scripts.v2_p7.t5.state import T5StateEnvelope


@dataclass
class T5RunState:
    """T5 全窗运行的可变累加器（T4 状态 + 配额/借用/资产负债序列）。"""

    sleeves: Dict[str, SleeveState]
    physical: PhysicalBook
    envelope: T5StateEnvelope
    day_rows: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    conservation: List[Dict[str, Any]] = field(default_factory=list)
    scenario_counts: Dict[str, int] = field(default_factory=dict)
    blocks: Dict[str, int] = field(default_factory=dict)
    voids: Dict[str, int] = field(default_factory=dict)
    carries: Dict[str, int] = field(default_factory=dict)
    identity_failures: List[Dict[str, Any]] = field(default_factory=list)
    nav_series: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    invalid_days: Dict[str, int] = field(default_factory=dict)
    digest_series: List[Tuple[str, str]] = field(default_factory=list)
    fee_series: List[Tuple[str, str]] = field(default_factory=list)
    quota_rows: List[Dict[str, Any]] = field(default_factory=list)
    quota_daily: List[Dict[str, Any]] = field(default_factory=list)
    balance_rows: List[Dict[str, Any]] = field(default_factory=list)
    credit_daily: List[Dict[str, Any]] = field(default_factory=list)
    credit_series: List[Tuple[str, str]] = field(default_factory=list)
    equity_blocks: List[Dict[str, Any]] = field(default_factory=list)
    coverage_seen: set = field(default_factory=set)
    offset_legs: int = 0
    offset_shares: int = 0
    market_filled: Dict[str, int] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    negative_cash_hits: List[Dict[str, Any]] = field(default_factory=list)
    negative_cash_examples: List[Dict[str, Any]] = field(default_factory=list)
    negative_cash_days: int = 0
    min_sleeve_cash: Optional[float] = None
    physical_cash_hits: List[Dict[str, Any]] = field(default_factory=list)
    physical_cash_examples: List[Dict[str, Any]] = field(default_factory=list)
    physical_cash_days_negative: int = 0
    min_physical_cash: Optional[float] = None
    offset_capped_shares: int = 0

    def __post_init__(self) -> None:
        arms = tuple(self.sleeves)
        self.nav_series = {arm: [] for arm in arms}
        #: credit 调整后的袖级权益序列（T6 用的权威口径；credit=off 时与 T4 逐位相同）
        self.credit_nav_rows: List[Dict[str, Any]] = []
        self.invalid_days = {arm: 0 for arm in arms}
        self.market_filled = {"buy": 0, "sell": 0}

    # ── T4 同名登记（逐位一致；同一实现语义） ──
    def record_identity(self, day: str, report: Any) -> None:
        if report.ok:
            return
        self.identity_failures.append(
            {
                "trade_date": day,
                "identities": [
                    {
                        "identity": row.identity,
                        "left": row.left,
                        "right": row.right,
                        "diff": row.diff,
                        "tolerance": row.tolerance,
                        "unit": row.unit,
                    }
                    for row in report.identities
                    if abs(row.diff) > row.tolerance
                ],
            }
        )

    def record_nav(self, day: str, report: Any, split_index: int) -> None:
        for arm in self.nav_series:
            self.nav_series[arm].append(
                {
                    "trade_date": day,
                    "nav": _finite_or_none(report.contribution_nav.get(arm)),
                    "independent_nav": _finite_or_none(report.independent_nav.get(arm)),
                    "total_value": report.total_value.get(arm),
                    "independent_equity": report.independent_equity.get(arm),
                    "split_index": int(split_index),
                }
            )
            if arm in report.invalid_sleeves:
                self.invalid_days[arm] = self.invalid_days.get(arm, 0) + 1

    def record_credit_nav(
        self,
        day: str,
        report: Any,
        credit_by_sleeve: Mapping[str, float],
        *,
        physical_capital: float,
        sleeve_capital: Mapping[str, float],
        savings: Mapping[str, float],
    ) -> None:
        """按**冻结输入**直接派生 credit 调整后的袖级序列（R2-01）。

        * 贡献 ``= E_i / 物理初资``（可加、**保持有符号**；物理初资非零时永不置空）；
        * 独立反事实权益 ``= E_i − 累计节省_i``；``<= 0`` 时独立指标无效并置空，
          **不**沿用旧指标是否已置空的状态；
        * 不再从可能为零/已置空的旧值反推（旧 ``total=0`` 或旧独立指标为空时仍正确）。
        """
        pc = float(physical_capital)
        for arm in self.nav_series:
            total = float(report.total_value.get(arm, 0.0))
            credit = float(credit_by_sleeve.get(arm, 0.0))
            equity = total - credit
            capital = float(sleeve_capital.get(arm, 0.0))
            saved = float(savings.get(arm, 0.0))
            counterfactual = equity - saved
            # 无效条件同时看**实际 E** 与**调整后独立反事实权益**（R3-01）：
            # 负节省可能把已经非正的 E 抬成正的反事实权益，仍属独立指标无效。
            invalid = bool(equity <= 0.0 or counterfactual <= 0.0)
            self.credit_nav_rows.append(
                {
                    "trade_date": str(day),
                    "sleeve": str(arm),
                    "total_value": total,
                    "credit": credit,
                    "equity_credit": equity,
                    "savings": saved,
                    "independent_equity_credit": counterfactual,
                    "contribution_credit_adjusted": (equity / pc) if pc != 0.0 else None,
                    "independent_nav_credit_adjusted": (
                        None if invalid or capital == 0.0 else counterfactual / capital
                    ),
                    "equity_nonpositive": bool(equity <= 0.0),
                    "invalid_nonpositive_equity": invalid,
                    "t4_contribution_nav": _finite_or_none(report.contribution_nav.get(arm)),
                    "t4_independent_nav": _finite_or_none(report.independent_nav.get(arm)),
                }
            )

    def record_cash_guard(self, day: str, sleeves: Mapping[str, SleeveState]) -> None:
        minimum: Optional[float] = None
        for sleeve_id in sorted(sleeves):
            cash = float(sleeves[sleeve_id].cash)
            minimum = cash if minimum is None else min(minimum, cash)
            if cash < 0.0:
                self.negative_cash_hits.append(
                    {"trade_date": day, "sleeve": sleeve_id, "cash": cash}
                )
                if len(self.negative_cash_examples) < 20:
                    self.negative_cash_examples.append(
                        {"trade_date": day, "sleeve": sleeve_id, "cash": cash}
                    )
        if minimum is None:  # pragma: no cover - 空袖集合不可达
            return
        self.min_sleeve_cash = (
            minimum if self.min_sleeve_cash is None else min(self.min_sleeve_cash, minimum)
        )
        if self.negative_cash_hits:
            self.negative_cash_days = len({item["trade_date"] for item in self.negative_cash_hits})

    def record_physical_cash(self, day: str, cash: float) -> None:
        value = float(cash)
        self.min_physical_cash = (
            value if self.min_physical_cash is None else min(self.min_physical_cash, value)
        )
        if value < 0.0:
            self.physical_cash_hits.append({"trade_date": day, "physical_cash": value})
            if len(self.physical_cash_examples) < 20:
                self.physical_cash_examples.append({"trade_date": day, "physical_cash": value})
            self.physical_cash_days_negative = len(
                {item["trade_date"] for item in self.physical_cash_hits}
            )

    def wants_checkpoint(
        self, index: int, total: int, coverage: Optional[Mapping[str, bool]] = None
    ) -> bool:
        marks = {total // 2, int(total * 0.9), total - 1}
        new_coverage = False
        for key in sorted(coverage or {}):
            if coverage[key] and key not in self.coverage_seen:
                self.coverage_seen.add(key)
                new_coverage = True
        return bool(new_coverage or index in marks)

    def accumulate(self, day: str, outcome: Any, fee_events: int) -> None:
        for key, value in outcome.scenario_counts.items():
            self.scenario_counts[key] = self.scenario_counts.get(key, 0) + int(value)
        for key, value in outcome.blocks.items():
            self.blocks[key] = self.blocks.get(key, 0) + int(value)
        for key, value in outcome.voids.items():
            self.voids[key] = self.voids.get(key, 0) + int(value)
        for key, value in outcome.carries.items():
            self.carries[key] = self.carries.get(key, 0) + int(value)
        for key, value in outcome.market_filled.items():
            self.market_filled[key] = self.market_filled.get(key, 0) + int(value)
        self.offset_legs += int(outcome.offset_legs)
        self.offset_shares += int(outcome.offset_shares)
        self.trades.extend(outcome.trades)
        self.conservation.extend(outcome.conservation_rows)
        self.day_rows.append(
            {
                "trade_date": day,
                "generation": self.envelope.generation,
                "scenarios": dict(sorted(outcome.scenario_counts.items())),
                "offset_legs": int(outcome.offset_legs),
                "offset_shares": int(outcome.offset_shares),
                "market_filled": dict(sorted(outcome.market_filled.items())),
                "blocks": dict(sorted(outcome.blocks.items())),
                "voids": dict(sorted(outcome.voids.items())),
                "carries": dict(sorted(outcome.carries.items())),
                "fee_events": int(fee_events),
                "physical_cash": float(outcome.physical.cash),
                "sleeves_cash": {
                    sid: float(st.cash) for sid, st in sorted(outcome.sleeves.items())
                },
                "residual_entries": {
                    sid: len(st.residuals) for sid, st in sorted(outcome.sleeves.items())
                },
                "identity_ok": True,
            }
        )


@dataclass
@dataclass
class DayPlan:
    """单日计划阶段产物（配额/授信/授权/预净额；**零副作用**）。"""

    day: str
    index: int
    coverage: Dict[str, bool]
    is_checkpoint: bool
    pre: Tuple[Any, Any, int]
    begin_envelope: Dict[str, Any]
    begin_fees: Dict[str, Any]
    begin_credit: Dict[str, Any]
    cash_open: Dict[str, float]
    credit_open: Dict[str, float]
    #: 授权可用现金（``cash_open − 到期归还预留``）：归还优先于新增买入
    cash_budget: Dict[str, float]
    #: 到期归还预留（FIFO 到期本金，按现金上限截断）
    due_reserve: Dict[str, float]
    quota_only: Dict[str, SleeveQuotaDecision]
    #: 同一实际 ``L`` 下、**未施加现金约束**的额度决策（拒绝原因分类依据，R2-03）
    quota_post_credit: Dict[str, SleeveQuotaDecision]
    decisions: Dict[str, SleeveQuotaDecision]
    final_intents: List[Intent]
    netting: Any
    plan_digest: str
    #: 授权前意图集（额度口径）与**预净额计划**（零副作用）
    pre_intents: List[Intent]
    pre_netting: Any
    pre_plan: Dict[str, Any]
    final_plan: Dict[str, Any]
    #: 授信转账腿的零 P&L 核对（借入/出借只改 cash 与 credit）
    grant_zero_pnl: Dict[str, Any]
    #: 授权前意图集（额度口径）与**预净额计划**（零副作用）
    pre_intents: List[Intent]
    pre_netting: Any
    pre_plan: Dict[str, Any]
    final_plan: Dict[str, Any]
    #: 授信转账腿的零 P&L 核对（借入/出借只改 cash 与 credit）
    grant_zero_pnl: Dict[str, Any]
    #: 授权前意图集（额度口径）与**预净额计划**（零副作用）
    pre_intents: List[Intent]
    pre_netting: Any
    pre_plan: Dict[str, Any]
    final_plan: Dict[str, Any]
    #: 授信转账腿的零 P&L 核对（借入/出借只改 cash 与 credit）
    grant_zero_pnl: Dict[str, Any]
    cash_delta: Dict[str, float]
    credit_delta: Dict[str, float]
    credit_events: List[Dict[str, Any]]


@dataclass
@dataclass
class Settled:
    """单日结算阶段产物（借用服务后 + 双口径 + 两道门）。"""

    outcome: Any
    report: Any
    balances: List[Any]
    gate: Dict[str, Any]
    zero_pnl: Dict[str, Any]
    service: Dict[str, Any]
    savings: Mapping[str, float]
    credit_balance: Dict[str, Any]
    credit_digest: str
