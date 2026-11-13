# -*- coding: utf-8 -*-
"""P7 T4 ``HistoricalVirtualSleeveHost``：跨折连续账户的逐日生命周期宿主。

职责
====
把 T2.5 已收口的相位目标（``plans_{v20,v10}.parquet``）当作**原始意图**输入，
在真实历史行情上逐日驱动「预净额 -> 内部抵消 -> 净市场卖出 -> 净市场买入 ->
一次结算 -> 双口径估值」全流程，并逐日提交 P7 专用状态信封。

硬契约
======
1. **跨折连续账户**：模型只在各折 ``test_start`` 切换；持仓/lot/残差/借款/召回/价源
   全部承接，账户**不清零**；
2. **成交价只认真实 T+1 开盘**；估值价按「当日价 -> 最后已知价 -> 买入价」共享解析；
   内部转账只认当日真实开盘价（缓存估值不完成 B1 转账）；
3. **双口径守恒**：I1/I2/I3 逐日闭合（``ledger.run_daily_identities``），NAV 只经
   ``calculate_nav``，链指标只经 ``calculate_chain_metrics``；
4. **零 α 消耗**：``alpha_consumed=false``，不产生任何 pass/fail 经济判定；
5. **幂等 + 同代提交**：每日一次 ``StateEnvelope.commit``；同键同代重复提交返回
   False 且状态不变，跨代提交拒绝；``restart_check`` 从检查点重放末日逐位一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

from scripts.v2_p7.t4 import fees as fee_mod
from scripts.v2_p7.t4.dual import DailyLedgerReport, assert_identities, run_daily_identities
from scripts.v2_p7.t4.ledger import execute_plan
from scripts.v2_p7.t4.netting import (
    Intent,
    assert_intent_conservation,
    carried_intent,
    cross_day_conservation,
    intent_from_plan_row,
    plan_day,
)
from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t4.state import (
    PhysicalBook,
    SleeveState,
    StateEnvelope,
    _physical_payload,
    _sleeve_payload,
    stable_digest,
)


class HostError(RuntimeError):
    """宿主前置条件不满足（身份/日历/守恒），一律 fail-fast。"""


@dataclass
class HostResult:
    """宿主运行结果（证据形态；不写盘）。"""

    run_id: str
    days: Tuple[str, ...]
    daily_rows: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    conservation_rows: List[Dict[str, Any]]
    scenario_counts: Dict[str, int]
    offset_legs: int
    offset_shares: int
    market_filled: Dict[str, int]
    blocks: Dict[str, int]
    voids: Dict[str, int]
    carries: Dict[str, int]
    envelope: StateEnvelope
    fee_book: fee_mod.FeeBook
    price_payload: Mapping[str, Any]
    identity_failures: List[Dict[str, Any]]
    nav_series: Mapping[str, List[Dict[str, Any]]]
    invalid_days: Mapping[str, int]
    conservation: Mapping[str, Any]
    replay: Mapping[str, Any]
    restart: Mapping[str, Any]
    fold_boundaries: List[Dict[str, Any]]
    digest_series: List[Tuple[str, str]]
    #: 逐日费用台账快照摘要（C4-04 续跑比对用）
    fee_series: List[Tuple[str, str]] = field(default_factory=list)
    #: 全窗逐袖最小虚拟现金（可为负：逐袖非负门归 T5）
    min_sleeve_cash: Optional[float] = None
    #: 按意图身份跨日聚合的守恒结果（次日重净额不重复抵消）
    cross_day: Dict[str, Any] = field(default_factory=dict)
    #: 逐袖现金为负的日数与样例（登记不拒判）
    negative_cash_days: int = 0
    negative_cash_examples: List[Dict[str, Any]] = field(default_factory=list)
    #: 因买方袖现金不足而未执行的内部抵消股数（C4-01 / 口径 a）
    offset_capped_shares: int = 0
    #: 全窗最小物理现金与负值日数（硬门：必须 >= 0，C4-01）
    min_physical_cash: Optional[float] = None
    negative_physical_cash_days: int = 0
    negative_physical_cash_examples: List[Dict[str, Any]] = field(default_factory=list)
    #: E<=0 禁新增买入的策略事件（F26）
    equity_policy_events: List[Dict[str, Any]] = field(default_factory=list)


def _load_plan_rows(root: Any, arms: Sequence[str]) -> List[Dict[str, Any]]:
    """读取 T2.5 计划帧（只读；不重算选择/截断/持仓排除）。"""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for arm in arms:
        path = root / "targets" / f"plans_{arm}.parquet"
        frame = pd.read_parquet(path)
        for row in frame.itertuples(index=False):
            shares = int(row.shares)
            if shares <= 0:
                continue
            rows.append(
                {
                    "sleeve_id": str(arm),
                    "ts_code": str(row.ts_code),
                    "shares": shares,
                    "signal_date": str(row.signal_date),
                    "execution_date": str(row.execution_date),
                    "due_date": str(row.due_date),
                    "model_version": int(row.model_version),
                    "split_index": int(row.split_index),
                }
            )
    return rows


def build_intent_index(
    root: Any, arms: Sequence[str]
) -> Tuple[Dict[str, List[Intent]], Dict[str, List[Intent]]]:
    """按执行日/到期日建立意图索引（买 = 执行日；卖 = 到期日，即 S7 到期结算）。"""
    buys: Dict[str, List[Intent]] = {}
    sells: Dict[str, List[Intent]] = {}
    for row in _load_plan_rows(root, arms):
        intent = intent_from_plan_row(
            sleeve_id=row["sleeve_id"],
            ts_code=row["ts_code"],
            side="buy",
            shares=row["shares"],
            signal_date=row["signal_date"],
            due_date=row["due_date"],
            model_version=row["model_version"],
        )
        buys.setdefault(str(row["execution_date"]), []).append(intent)
        sell = intent_from_plan_row(
            sleeve_id=row["sleeve_id"],
            ts_code=row["ts_code"],
            side="sell",
            shares=row["shares"],
            signal_date=row["signal_date"],
            due_date=row["due_date"],
            model_version=row["model_version"],
        )
        sells.setdefault(str(row["due_date"]), []).append(sell)
    return buys, sells


def _available(sleeves: Mapping[str, SleeveState]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for sleeve_id, state in sleeves.items():
        held: Dict[str, int] = {}
        for lot in state.lots:
            held[lot.ts_code] = held.get(lot.ts_code, 0) + lot.shares
        out[sleeve_id] = held
    return out


@dataclass
class _RunState:
    """全窗运行的可变累加器（逐日证据与跨日状态）。"""

    sleeves: Dict[str, SleeveState]
    physical: PhysicalBook
    envelope: StateEnvelope
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
    #: 逐日费用台账快照摘要（C4-04：续跑必须逐日复现同一累计态）
    fee_series: List[Tuple[str, str]] = field(default_factory=list)
    #: 已设置过“覆盖型检查点”的场景类别（停牌/顺延/禁买各至少一个）
    coverage_seen: set = field(default_factory=set)
    offset_legs: int = 0
    offset_shares: int = 0
    market_filled: Dict[str, int] = field(default_factory=dict)
    #: 检查点（~50% / ~90% / 末日）：末日开始状态 + 该日结束后的状态本体载荷
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    #: 逐袖现金为负的命中记录（T4 登记不拒判；逐袖非负门归 T5）
    negative_cash_hits: List[Dict[str, Any]] = field(default_factory=list)
    negative_cash_examples: List[Dict[str, Any]] = field(default_factory=list)
    negative_cash_days: int = 0
    min_sleeve_cash: Optional[float] = None
    #: 物理现金登记（硬门：必须 >= 0，C4-01）
    physical_cash_hits: List[Dict[str, Any]] = field(default_factory=list)
    physical_cash_examples: List[Dict[str, Any]] = field(default_factory=list)
    physical_cash_days_negative: int = 0
    min_physical_cash: Optional[float] = None
    #: 被买方现金限额而未执行的抵消股数（C4-01）
    offset_capped_shares: int = 0
    #: E<=0 禁止新增买入的策略事件（F26）
    equity_policy_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        arms = tuple(self.sleeves)
        self.nav_series = {arm: [] for arm in arms}
        self.invalid_days = {arm: 0 for arm in arms}
        self.market_filled = {"buy": 0, "sell": 0}

    # ---- 逐日登记 ----

    def record_identity(self, day: str, report: Any) -> None:
        """恒等式破缺逐日登记并立即停止（不是告警）。"""
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
        assert_identities(report)

    def record_nav(self, day: str, report: Any, split_index: int) -> None:
        """逐袖登记双口径 NAV（实际贡献 / 只读反事实）与无效日计数、折号。"""
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

    def record_cash_guard(self, day: str, sleeves: Mapping[str, SleeveState]) -> None:
        """逐袖现金符号登记（复审 A-03 / B-03）。

        口径（F26）：``Σ_i cash_i ≡ 物理现金`` 是 I1 的硬约束（已逐日校验）；
        **逐袖** ``cash_i >= 0`` 属 T5 的授权/借用门（T4 无配额、无 credit），
        故此处**登记不拒判**：``min_sleeve_cash`` / ``negative_cash_days`` / 逐条样例
        进证据，并在 ``_t4_boundaries`` 显式声明「T4 未实现逐袖现金非负，归 T5」。

        负现金成因：净额化的**内部转账**是无 credit 背书的虚拟付款（B1 抵消无条件完成），
        买方袖在自身现金不足时仍完成抵消 -> 逐袖现金短期为负。
        """
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
        """物理现金逐日登记（C4-01 硬门：非杠杆前提要求 ``cash >= 0``）。

        与 I1 的 ``Σ cash_i ≡ 物理现金`` 互补：那是**相等**关系，两侧一起变负仍成立；
        这里是**非负**关系，才是「账户付得起」的可执行性判据。
        """
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

    def record_equity_policy(self, report: Any, blocked: Sequence[str]) -> None:
        """登记当日 ``E_i <= 0`` 的袖（禁新增买入由宿主执行，见 ``_no_new_buy``）。"""
        for arm in blocked:
            self.equity_policy_events.append(
                {
                    "trade_date": report.trade_date,
                    "sleeve": arm,
                    "equity": float(report.total_value.get(arm, 0.0)),
                    "action": "no_new_buy",
                }
            )

    def wants_checkpoint(
        self, index: int, total: int, coverage: Optional[Mapping[str, bool]] = None
    ) -> bool:
        """该日是否要设检查点（在**当日开始时**判定，供调用方决定是否做整份快照）。

        固定三个位置（~50% / ~90% / 末日），另对「停牌中 / 顺延残差 / 禁买状态」
        各取**首次命中日**——这三类是 C4-04 要求必须覆盖的重启场景。
        """
        marks = {total // 2, int(total * 0.9), total - 1}
        new_coverage = False
        for key in sorted(coverage or {}):
            if coverage[key] and key not in self.coverage_seen:
                self.coverage_seen.add(key)
                new_coverage = True
        return bool(new_coverage or index in marks)

    def maybe_checkpoint(
        self,
        day: str,
        index: int,
        pre: Any,
        no_new_buy: Sequence[str] = (),
        price_last_known: Optional[Mapping[str, Any]] = None,
        valuation_probe: Optional[Mapping[str, Any]] = None,
        state_payload: Optional[Mapping[str, Any]] = None,
        begin_envelope: Optional[Mapping[str, Any]] = None,
        begin_fees: Optional[Mapping[str, Any]] = None,
        coverage: Optional[Mapping[str, bool]] = None,
    ) -> None:
        """记一个**当日开始时**的检查点（调用方须先用 ``wants_checkpoint`` 判定）。

        ``no_new_buy`` 是**该时点**的 E<=0 禁买集合快照；重启重放必须用它，
        否则会回落到宿主末日的全局集合（复审 N-4）。
        """
        self.checkpoints.append(
            {
                "trade_date": day,
                "index": int(index),
                "sleeves": pre[0],
                "physical": pre[1],
                "seq": int(pre[2]),
                "no_new_buy": list(no_new_buy),
                # **当日开始时**的状态载荷（不是信封里的上一日/当日结束态）：
                # 重启必须能从这里原样还原，再重算当日（C4-04）。
                "state_payload": dict(state_payload or {}),
                # 当日**开始时的完整版本化信封**（代际 + 幂等历史 + 双口径 + 续跑态）：
                # 续跑必须接着原代际推进，不能新开 1 代（C4-04）。
                "envelope": dict(begin_envelope or {}),
                # 当日**开始时的费用台账累计态**（双通道 + 逐键 + 浮点轨迹）
                "fee_state": dict(begin_fees or {}),
                # 当日**结束时**的状态载荷：重启重算后与之逐位比对
                "end_state_payload": self.envelope.state_payload(),
                # 续跑所需的**价源缓存**与估值探针（C4-04）：重启必须从检查点恢复，
                # 否则估值会从 last_known 退化为 buy_price，NAV 悄悄变化。
                "price_last_known": dict(price_last_known or {}),
                "valuation_probe": dict(valuation_probe or {}),
                # 该检查点覆盖的重启场景（停牌中 / 顺延残差 / 禁买状态）
                "coverage": {str(key): bool(value) for key, value in (coverage or {}).items()},
            }
        )

    def accumulate(self, day: str, outcome: Any, fee_events: int) -> None:
        """累加场景/阻断/作废/顺延/成交与逐日明细行。"""
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


def _fee_progress(book: Any) -> str:
    """费用台账**逐日累计进度**摘要（O(1)，不含逐键明细）。

    逐键明细由检查点整份快照 + 末个检查点的**完整 payload** 对照覆盖；
    逐日只需要能抓住累计口径（事件数/物理/申报/运行实际/逐袖双通道）漂移。
    """
    return stable_digest(
        {
            "events": int(book.events),
            "physical": float(book.physical_fee_total),
            "declared": float(book.declared_fee_total),
            "running": float(book.running_actual),
            "actual": dict(sorted(book.actual_by_sleeve.items())),
            "independent": dict(sorted(book.independent_by_sleeve.items())),
        }
    )


def _tracked_codes(
    sleeves: Mapping[str, SleeveState], intents: Sequence[Intent]
) -> Tuple[str, ...]:
    """停牌段跟踪集合 = 当日持仓代码 ∪ 当日意图代码。"""
    codes = {lot.ts_code for state in sleeves.values() for lot in state.lots}
    codes.update(intent.ts_code for intent in intents)
    return tuple(sorted(codes))


def _initial_sleeves(params: Any) -> Dict[str, SleeveState]:
    return {
        arm: SleeveState(sleeve_id=arm, cash=float(params.sleeve_capital[arm]), credit=0.0, lots=())
        for arm in params.arms
    }


def _initial_physical(params: Any) -> PhysicalBook:
    return PhysicalBook(cash=float(params.physical_capital))


class HistoricalVirtualSleeveHost:
    """逐日驱动净额化生命周期的历史宿主（跨折连续账户）。"""

    def __init__(
        self,
        *,
        params: Any,
        run_id: str,
        prices: PriceBook,
        cost_model: Any,
        buy_index: Mapping[str, Sequence[Intent]],
        sell_index: Mapping[str, Sequence[Intent]],
        days: Optional[Sequence[str]] = None,
        price_factory: Optional[Any] = None,
        restore: Optional[Mapping[str, Any]] = None,
        self_check: bool = True,
    ) -> None:
        self.params = params
        self.run_id = str(run_id)
        self.prices = prices
        #: 重放/重启自验用的**独立**价源工厂（缺省复用同一价源，证据中会标注未隔离）
        self._price_factory = price_factory
        self.cost_model = cost_model
        self.buy_index = buy_index
        self.sell_index = sell_index
        #: 运行日历（缺省 = 全窗 ``extended_days``；分段验证可显式指定子窗口）
        self._days = tuple(days) if days else tuple(params.extended_days)
        self.fee_book = fee_mod.FeeBook()
        self.cost_of_by_side = {
            "buy": cost_model.calculate_buy_cost,
            "sell": cost_model.calculate_sell_cost,
        }
        self.seq = 0
        #: E<=0 的袖：禁止新增买入（F26 政策；本 run 未触发）
        self._no_new_buy: set = set()
        #: C4-04：从序列化信封 + 费用快照续跑（None = 全新运行）
        self._restore = dict(restore) if restore else None
        #: 续跑子宿主不再递归做重放/重启自验（否则按窗口长度无限套娃）
        self._self_check = bool(self_check)

    # ────────────────────────── 主流程 ──────────────────────────

    def run(self) -> HostResult:
        """全窗逐日运行（一次通过；不写盘；``restore`` 时接着原代际推进）。"""
        state = (
            self._restore_state(self._restore)
            if self._restore is not None
            else _RunState(
                sleeves=_initial_sleeves(self.params),
                physical=_initial_physical(self.params),
                envelope=StateEnvelope(run_id=self.run_id),
            )
        )
        days = self._days
        for index, day in enumerate(days):
            self._run_day(day, index, len(days), state)
        return self._finalize(days, state)

    def _restore_state(self, payload: Mapping[str, Any]) -> _RunState:
        """从**序列化信封 + 费用台账快照**恢复原代际，并接着推进（C4-04）。

        还原项：完整袖/物理账本、``generation``、幂等历史、双口径累计、
        续跑态（价源缓存 / 禁买集合 / 序号）与费用台账双通道累计。

        续跑态以**信封内的 ``resume_state`` 为唯一来源**（``payload`` 同名键只作缺省兜底），
        避免「信封锁一份、旁路再传一份」的两套真相。
        """
        envelope = StateEnvelope(run_id=self.run_id)
        envelope.restore_from(payload.get("envelope") or {})
        self.fee_book = fee_mod.FeeBook.restore(payload.get("fee_book") or {})
        resume = envelope.resume_state
        self.seq = int(resume.get("seq", payload.get("seq") or 0))
        no_new_buy = resume.get("no_new_buy", payload.get("no_new_buy") or ())
        self._no_new_buy = {str(item) for item in no_new_buy or ()}
        cache = resume.get("price_last_known", payload.get("price_last_known") or {})
        self.prices.restore_last_known(cache or {})
        # 第 0 代检查点（首日开始时）信封里还没有账本本体：回落到参数初值
        sleeves = envelope.sleeves or _initial_sleeves(self.params)
        physical = (
            envelope.physical if envelope.physical is not None else _initial_physical(self.params)
        )
        return _RunState(sleeves=sleeves, physical=physical, envelope=envelope)

    def _run_day(self, day: str, index: int, total: int, state: _RunState) -> None:
        """单日：预净额 -> 落账 -> 费用台账 -> 双口径恒等式 -> 同代提交。"""
        intents = self._collect_intents(day, state.sleeves)
        tracked = _tracked_codes(state.sleeves, intents)
        self.prices.begin_day(day, tracked=tracked)
        # 停牌中重启：当日有**持仓**股票无真实有价日（估值只能走 last_known 缓存）
        held_codes = {lot.ts_code for sleeve in state.sleeves.values() for lot in sleeve.lots}
        suspended_held = bool(held_codes & set(self.prices.active_gaps()))
        coverage = {
            # 重启时这三类状态的还原最容易悄悄错（C4-04 指定覆盖）
            "carry_residual": any(sleeve.residuals for sleeve in state.sleeves.values()),
            "suspension": bool(suspended_held or (set(tracked) & set(self.prices.active_gaps()))),
            "no_new_buy": bool(self._no_new_buy),
        }
        # 只有真正设检查点的日子才做整份信封/费用快照（全窗逐日快照是 O(n^2)）
        is_checkpoint = state.wants_checkpoint(index, total, coverage)
        begin_envelope = state.envelope.to_payload() if is_checkpoint else {}
        begin_fees = self.fee_book.snapshot() if is_checkpoint else {}
        plan = plan_day(day, intents, _available(state.sleeves))
        pre = (state.sleeves, state.physical, self.seq)
        outcome = execute_plan(
            plan=plan,
            sleeves=state.sleeves,
            physical=state.physical,
            prices=self.prices,
            cost_model=self.cost_model,
            lot_size=int(self.params.lot_size),
            seq_base=self.seq,
        )
        self.seq += max(0, len(outcome.trades)) + 1
        day_costs = self.fee_book.add_day(
            day,
            outcome.fee_entries,
            cost_of_by_side=self.cost_of_by_side,
            closures=outcome.closures,
            declared_fees=outcome.declared_fees,
        )
        # C4-04：逐日费用台账累计进度摘要（续跑必须复现同一累计态，不止“当日闭合”）
        state.fee_series.append((day, _fee_progress(self.fee_book)))
        savings = self.fee_book.savings_by_sleeve()
        report = run_daily_identities(
            trade_date=day,
            sleeves=outcome.sleeves,
            physical=outcome.physical,
            prices=self.prices,
            physical_initial=float(self.params.physical_capital),
            sleeve_capital=dict(self.params.sleeve_capital),
            cumulative_savings=savings,
        )
        state.record_physical_cash(day, float(outcome.physical.cash))
        state.offset_capped_shares += int(getattr(outcome, "offset_capped_shares", 0))
        state.record_identity(day, report)
        self._apply_equity_policy(report, state)
        state.record_cash_guard(day, outcome.sleeves)
        state.sleeves = outcome.sleeves
        state.physical = outcome.physical
        state.envelope.commit(
            key=f"{self.run_id}:{day}",
            generation=state.envelope.generation + 1,
            # C4-04：续跑态随信封一起落盘（价源缓存 / 禁买集合 / 序号）
            resume_state={
                "price_last_known": self.prices.snapshot_last_known(),
                "no_new_buy": sorted(self._no_new_buy),
                "seq": int(self.seq),
            },
            fingerprint=stable_digest(
                {
                    "day": day,
                    "scenarios": outcome.scenario_counts,
                    "offset_shares": outcome.offset_shares,
                    "market_filled": outcome.market_filled,
                    "trades": len(outcome.trades),
                    "day_costs": len(day_costs),
                }
            ),
            sleeves=state.sleeves,
            physical=state.physical,
            as_of=day,
        )
        state.digest_series.append((day, state.envelope.state_digest()))
        if is_checkpoint:
            state.maybe_checkpoint(
                day,
                index,
                pre,
                sorted(self._no_new_buy),
                price_last_known=self.prices.snapshot_last_known(),
                valuation_probe=self._valuation_probe(pre[0]),
                state_payload={
                    "as_of": day,
                    "sleeves": _sleeve_payload(pre[0]),
                    "physical": _physical_payload(pre[1]),
                },
                begin_envelope=begin_envelope,
                begin_fees=begin_fees,
                coverage=coverage,
            )
        state.record_nav(day, report, int(self.params.split_of(day)))
        state.envelope.dual_caliber = _dual_caliber(
            self.params.arms, report, savings, self.fee_book
        )
        state.accumulate(day, outcome, len(day_costs))
        if len(state.day_rows) % 250 == 0:
            logger.info(
                f"{self.run_id} {day} 第 {len(state.day_rows)} 日；物理现金="
                f"{outcome.physical.cash:.2f}"
            )

    def _finalize(self, days: Tuple[str, ...], state: _RunState) -> HostResult:
        """收尾：费用闭合 + 意图守恒 + 价源证据 + 重放/重启自验。"""
        self.fee_book.assert_closed()
        conservation_check = assert_intent_conservation(state.conservation)
        cross_day = cross_day_conservation(state.conservation)
        if conservation_check["violations"]:
            raise HostError(
                f"意图守恒破缺：{conservation_check['violations']} 条不满足 "
                "original = offset + filled + voided + carry"
            )
        # 价源证据必须在重放/重启自验**之前**取：自验会重新装载帧并追加估值登记
        price_payload = self.prices.payload()
        if self._self_check:
            # 延迟导入：restart 模块需要构造宿主，放在模块顶部会形成循环导入
            from scripts.v2_p7.t4.restart import audit_restarts

            restart = audit_restarts(self, state)
        else:
            restart = {"checked": False, "note": "续跑子宿主跳过重放/重启自验（避免递归）"}
        replay = (
            self._replay_check()
            if self._self_check
            else {"checked": False, "note": "续跑子宿主跳过重放/重启自验（避免递归）"}
        )
        return HostResult(
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
            envelope=state.envelope,
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
            equity_policy_events=list(state.equity_policy_events),
        )

    def _apply_equity_policy(
        self,
        report: Any,
        state: Optional[_RunState] = None,
        no_new_buy: Optional[set] = None,
    ) -> set:
        """``E_i <= 0`` 的袖：此后**禁止新增买入**（F26 借用态恒等式政策）。

        ``E`` 取**实际**口径（``cash + MV - credit``；T4 ``credit = 0``，故等于
        ``report.total_value``），**不是**反事实（不含节省）口径——后者只用于独立
        绩效标注（``invalid_sleeves``）。被动卖出不开放；已有持仓照常持有与到期。
        """
        target = self._no_new_buy if no_new_buy is None else no_new_buy
        blocked = [
            arm for arm in self.params.arms if float(report.total_value.get(arm, 1.0)) <= 0.0
        ]
        if state is not None:
            state.record_equity_policy(report, blocked)
        for arm in blocked:
            target.add(arm)
        return target

    # ────────────────────────── 意图收集 ──────────────────────────

    def _collect_intents(
        self,
        day: str,
        sleeves: Mapping[str, SleeveState],
        no_new_buy: Optional[set] = None,
    ) -> List[Intent]:
        """当日意图 = 新原始意图（买按执行日 / 卖按到期日）+ 袖级顺延残差。"""
        blocked = self._no_new_buy if no_new_buy is None else no_new_buy
        intents: List[Intent] = [
            intent for intent in self.buy_index.get(day, ()) if intent.sleeve_id not in blocked
        ]
        intents.extend(self.sell_index.get(day, ()))
        for sleeve_id in sorted(sleeves):
            for entry in sleeves[sleeve_id].residuals:
                if entry.side == "buy" and sleeve_id in blocked:
                    # E<=0 的袖：连顺延的买入残差也不再重试（F26 政策）
                    continue
                intents.append(carried_intent(entry))
        return intents

    # ────────────────────────── 自验 ──────────────────────────

    def _replay_check(self) -> Dict[str, Any]:
        """同参数重放抽样窗口：逐日状态摘要必须逐位一致（同 generation 幂等）。

        两次重放各用**独立的 ``PriceBook``**（``self._price_factory``），
        避免共享「最后已知价」缓存造成的假一致（复审 B-06 / B-17）。
        """
        window = self._replay_window()
        if not window:
            return {"checked": False, "note": "日历过短，未抽样"}
        first = self._simulate_window(window)
        second = self._simulate_window(window)
        identical = [a[1] for a in first] == [b[1] for b in second]
        return {
            "checked": True,
            "window": [window[0], window[-1]],
            "days": len(window),
            "all_bitwise_identical": bool(identical),
            "price_book_isolated": self._price_factory is not None,  # type: ignore[dict-item]
            "first_digest": first[0][1] if first else "",
            "last_digest": first[-1][1] if first else "",
        }

    def _replay_window(self) -> Tuple[str, ...]:
        """重放抽样窗口：日历三分之一处的 40 个连续交易日。"""
        days = self._days
        if len(days) < 40:
            return ()
        start = len(days) // 3
        return tuple(days[start:])[:40]

    def _fresh_prices(self) -> Any:
        """取一份独立价源（有工厂则新建；否则复用并标注未隔离）。"""
        return self._price_factory() if self._price_factory is not None else self.prices

    def _simulate_window(self, window: Sequence[str]) -> List[Tuple[str, str]]:
        """从零初资重跑一个连续窗口，返回逐日 ``(日期, 状态摘要)``。"""
        prices = self._fresh_prices()
        sleeves = _initial_sleeves(self.params)
        physical = _initial_physical(self.params)
        envelope = StateEnvelope(run_id=f"{self.run_id}:replay")
        fee_book = fee_mod.FeeBook()
        #: 重放窗口用**局部**禁买集合（不复用宿主状态，否则与原 run 不同步，复审 N-4）
        no_new_buy: set = set()
        seq = 0
        digests: List[Tuple[str, str]] = []
        for day in window:
            intents = self._collect_intents(day, sleeves, no_new_buy)
            prices.begin_day(day, tracked=_tracked_codes(sleeves, intents))
            plan = plan_day(day, intents, _available(sleeves))
            outcome = execute_plan(
                plan=plan,
                sleeves=sleeves,
                physical=physical,
                prices=prices,
                cost_model=self.cost_model,
                lot_size=int(self.params.lot_size),
                seq_base=seq,
            )
            seq += max(0, len(outcome.trades)) + 1
            fee_book.add_day(
                day,
                outcome.fee_entries,
                cost_of_by_side=self.cost_of_by_side,
                closures=outcome.closures,
                declared_fees=outcome.declared_fees,
            )
            report = run_daily_identities(
                trade_date=day,
                sleeves=outcome.sleeves,
                physical=outcome.physical,
                prices=prices,
                physical_initial=float(self.params.physical_capital),
                sleeve_capital=dict(self.params.sleeve_capital),
                cumulative_savings=fee_book.savings_by_sleeve(),
            )
            self._apply_equity_policy(report, None, no_new_buy)
            sleeves = outcome.sleeves
            physical = outcome.physical
            envelope.commit(
                key=f"replay:{day}",
                generation=envelope.generation + 1,
                fingerprint=stable_digest(
                    {
                        "day": day,
                        "offset_shares": outcome.offset_shares,
                        "market_filled": outcome.market_filled,
                    }
                ),
                sleeves=sleeves,
                physical=physical,
                as_of=day,
            )
            digests.append((day, envelope.state_digest()))
        return digests

    def _valuation_probe(self, sleeves: Mapping[str, Any], prices: Any = None) -> Dict[str, Any]:
        """逐 (袖, 股票) 记录当前价源下的估值结果（价 + 日期 + 来源）。

        重启核对时用**恢复后的价源**重算同一探针，逐位比对 —— 这是「价源缓存
        也必须可还原」的可判据（C4-04 复现 B：last_known 与 buy_price 的 NAV 差）。
        """
        probe: Dict[str, Any] = {}
        for sleeve_id in sorted(sleeves):
            for lot in sleeves[sleeve_id].lots:
                price, day, source = (prices or self.prices).valuation(
                    lot.ts_code, float(lot.acquired_price)
                )
                probe[f"{sleeve_id}|{lot.ts_code}"] = [
                    None if price is None else float(price),
                    str(day),
                    str(source),
                ]
        return probe

    def _fold_boundaries(self) -> List[Dict[str, Any]]:
        """折界账户承接登记（模型切换点只在各折首日；持仓不清零）。"""
        folds = list(self.params.t25.folds)
        return [
            {
                "split_index": int(split),
                "model_version": int(model),
                "switch_at": str(start),
                "previous_model_version": (int(folds[pos - 1][3]) if pos > 0 else None),
            }
            for pos, (split, start, _end, model) in enumerate(folds)
        ]


def _dual_caliber(
    arms: Sequence[str],
    report: DailyLedgerReport,
    savings: Mapping[str, float],
    fee_book: fee_mod.FeeBook,
) -> Any:
    """把当日双口径（实际/独立费用、总值、贡献 NAV、反事实 NAV）写入信封。"""
    from scripts.v2_p7.t4.state import DualCaliber

    return DualCaliber(
        actual_fee={arm: float(fee_book.actual_by_sleeve.get(arm, 0.0)) for arm in arms},
        independent_fee={arm: float(fee_book.independent_by_sleeve.get(arm, 0.0)) for arm in arms},
        total_value={arm: float(report.total_value.get(arm, 0.0)) for arm in arms},
        contribution_nav={arm: float(report.contribution_nav.get(arm, 0.0)) for arm in arms},
        independent_nav={arm: report.independent_nav.get(arm) for arm in arms},
        invalid_sleeves=dict(report.invalid_sleeves),
    )


def _finite_or_none(value: Any) -> Optional[float]:
    import math

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def build_nav_frames(result: HostResult, params: Any) -> Dict[str, Any]:
    """按袖装配链式净值帧并计算链指标（``calculate_chain_metrics`` 唯一实现）。"""
    import pandas as pd

    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    out: Dict[str, Any] = {}
    for arm in params.arms:
        rows = result.nav_series.get(arm, [])
        frame = pd.DataFrame(rows)
        if frame.empty:
            out[arm] = {"actual": None, "independent": None}
            continue
        actual = frame[["trade_date", "nav", "split_index"]].rename(columns={"nav": "nav"})
        # C4-06：**第二个出口**必须与 `runner._sleeve_metrics` 用同一失效策略。
        # 出现无效权益日（独立 NAV 被标 None）即整体标 invalid，不得 dropna 后跨日拼接。
        invalid_rows = frame[frame["independent_nav"].isna()]
        invalid_dates = [str(value) for value in invalid_rows["trade_date"].tolist()]
        independent = (
            None
            if invalid_dates
            else calculate_chain_metrics(
                frame[["trade_date", "independent_nav", "split_index"]]
                .rename(columns={"independent_nav": "nav"})
                .reset_index(drop=True)
            )
        )
        out[arm] = {
            "actual": calculate_chain_metrics(actual.dropna(subset=["nav"]).reset_index(drop=True)),
            "independent": independent,
            "independent_status": ("ok" if not invalid_dates else "invalid_nonpositive_equity_day"),
            "independent_invalid_dates": invalid_dates,
            "days": int(len(frame)),
            "invalid_days": int(result.invalid_days.get(arm, 0)),
        }
    return out


def digest_payload(result: HostResult) -> str:
    """整体结果摘要（重放/证据身份用）。"""
    return stable_digest(
        {
            "run_id": result.run_id,
            "days": len(result.days),
            "scenario_counts": result.scenario_counts,
            "offset_shares": result.offset_shares,
            "market_filled": result.market_filled,
            "envelope": result.envelope.digest(),
        }
    )


def serialize_envelope(result: HostResult) -> str:
    """信封 JSON 序列化（证据原文，便于人工复核）。"""
    return json.dumps(result.envelope.to_payload(), ensure_ascii=False, sort_keys=True)
