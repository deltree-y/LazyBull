"""P6.5 T1/T3 合成宿主：逐日三恒等式挂载 + 订单级全链承载（不含 T4/T5）。

契约依据：设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md``
§4.4-3/§4.4-5 与 §7.2（Q21=A/Q23=A）、``docs/contracts/netting_freeze.md`` F25 §4、
``docs/contracts/protocols.md`` F13 §3.3/§3.4。

职责与边界：
- 继承现有 ``NativeMergeBacktestEngineML``，覆盖 ``_finalize_day``：在调用父类之前完整
  执行 I1 → I2 → I3；另覆盖终态闩锁入口 ``run`` / ``run_step``（T1 停止语义，终态后
  在任何父类可变操作前拒绝），其余全部为**新增**方法；父类与 ``run_loop.py`` 零改动
  （父类行为守卫测试锁定）；
- 无成交日同样执行；任一破缺抛 ``ReconciliationError``（RuntimeError），当日停止，
  **不写日终快照、不提交状态、不进入下一日**；
- T3：``execute_order_chain`` 承载真实 ``Order`` → 既有执行器 → ``Ledger.apply_fills``
  → ``InternalSettlementBatch`` → ``VirtualLedger.apply_settlement`` 全链；预演门在
  物理执行前完成，任何拒绝保持零副作用（实现见 ``hosts/order_chain.py``）；
- 双臂 runs 与自拆（T4）、状态信封恢复（T5）不在本模块。逐日校验证据
  （``identity_reports``）与链结果只存内存，**不进入** ``EngineState`` 快照键集；
- 本类**不进**生产默认路径（``hosts/backtest/runtime.py`` 与 ``PaperHost`` 不变），
  只由合成测试显式构造。
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.lazybull.v2.common.types import OrderReason, TradeDate
from src.lazybull.v2.core.accounting.ledger import LedgerError
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLotView
from src.lazybull.v2.core.accounting.reconciliation import (
    DailyIdentityReport,
    IdentityContext,
    ReconciliationError,
    run_daily_identities,
)
from src.lazybull.v2.core.accounting.settlement_kernel import (
    InternalSettlementBatch,
    SettlementResult,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedger,
    VirtualLedgerConservationError,
    VirtualLedgerError,
)
from src.lazybull.v2.core.decision.virtual_merger import NettingPlan, VirtualMergerError
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML
from src.lazybull.v2.hosts.order_chain import (
    ChainFill,
    OrderChainError,
    OrderChainInvariantError,
    OrderChainResult,
    assert_fills_booked,
    assert_no_physical_sells,
    assert_physical_base,
    assert_plan_integrity,
    assert_settleable,
    build_chain_batch,
    build_sleeve_orders,
    chain_request_fingerprint,
    execute_fills,
    plan_order_chain,
    predict_fills,
    predicted_positions,
    register_fills,
    validate_sleeve_targets,
)

#: 合成宿主标识（进入恒等式错误上下文）
HOST_ID = "virtual_native_p65"


class IdentityTerminatedError(RuntimeError):
    """恒等式破缺后的实例终态：禁止复用/重试，恢复须从破缺前信封新建实例。"""

    def __init__(self, *, trade_date: Any, original: ReconciliationError) -> None:
        self.trade_date = trade_date
        self.original = original
        super().__init__(
            f"[{original.context.stage}][TERMINATED] 本合成宿主实例已于 {trade_date} "
            f"因恒等式破缺终止（原始破缺编号={list(original.identities)}）；"
            "禁止复用、重试或清空证据解除终态，恢复必须按设计 §4.5 从破缺前完整信封新建实例"
        )


class ChainFailedTerminatedError(RuntimeError):
    """订单级全链执行后失败（物理可能已部分成交）：实例锁定为终态。"""

    def __init__(self, *, trade_date: Any, batch_id: str, original: BaseException) -> None:
        self.trade_date = trade_date
        self.batch_id = batch_id
        self.original = original
        super().__init__(
            f"[P6.5][TERMINATED] 本合成宿主实例已于 {trade_date} 因批次 {batch_id} 执行后失败"
            f"终止（{type(original).__name__}: {original}）；物理可能已部分成交，"
            "禁止复用、重试或清空证据解除终态，恢复必须新建实例并从破缺前完整信封重放"
        )


class VirtualNativeMergeBacktestEngineML(NativeMergeBacktestEngineML):
    """承载虚拟子袖逐日对账与订单级全链的合成回测宿主（P6.5 T1/T3）。

    构造期注入（全部为显式合成输入，不读配置、不读真实行情）：
    - ``virtual_ledger``：最小虚拟账本（纯运行时子袖）；
    - ``physical_lot_view``：物理来源 lot 伴随视图（I2 右值）；
    - ``price_source_provider``：``date -> PriceSource``；缺省时从引擎自身收盘成交价
      （``_get_trade_price``）构造同日同价源，缺价即拒绝（P6.5 无价 fail-fast）。
      显式 provider 必须与引擎物理估值**逐股同日同价**（组合总值相等不算同源）。

    终态语义（T1 停止语义）：任一日任一恒等式破缺 ⇒ 本实例锁定为终态
    （``IdentityTerminatedError``）；订单级全链在物理执行后失败（真实结算拒绝/
    中途异常）⇒ 同样锁定为终态（``ChainFailedTerminatedError``）。二者均使后续
    ``run`` / ``run_step`` / ``execute_order_chain`` / ``_finalize_day`` 在任何可变
    操作前即拒绝；清空证据不解除终态，恢复只能新建实例。

    挂载级缺价顺序（T1 复审 P3 显式登记，未新增独立编号）：``_build_price_source``
    先于三项校验执行，缺价/价源日期不符以 ``ReconciliationError(identity="I3")``
    上报——I3 是唯一消费价源的恒等式；该日不会产出 I1/I2 结果，属"价源缺失即整日
    拒绝、零提交"的显式顺序，不代表 I1/I2 已被验证或跳过。
    """

    def __init__(
        self,
        *args: Any,
        virtual_ledger: VirtualLedger,
        physical_lot_view: PhysicalLotView,
        price_source_provider: Optional[Callable[[pd.Timestamp], PriceSource]] = None,
        run_id: str = "",
        **kwargs: Any,
    ) -> None:
        if not isinstance(virtual_ledger, VirtualLedger):
            raise TypeError(f"合成宿主需要 VirtualLedger，实际 {type(virtual_ledger).__name__}")
        if not isinstance(physical_lot_view, PhysicalLotView):
            raise TypeError(
                f"合成宿主需要 PhysicalLotView，实际 {type(physical_lot_view).__name__}"
            )
        if price_source_provider is not None and not callable(price_source_provider):
            raise TypeError("price_source_provider 必须可调用（date -> PriceSource）")
        super().__init__(*args, **kwargs)
        self.virtual_ledger = virtual_ledger
        self.physical_lot_view = physical_lot_view
        self.price_source_provider = price_source_provider
        self.identity_run_id = str(run_id)
        # 逐日校验证据（仅内存；不入 EngineState 键集，P4 契约不变）
        self.identity_reports: List[DailyIdentityReport] = []
        self.identity_check_count = 0
        # 终态闩锁：首次破缺即锁定，任何后续日步进在可变操作前拒绝
        self._terminated: Optional[IdentityTerminatedError] = None
        # 订单级全链执行后失败（物理可能已部分成交）：同为不可恢复终态，
        # 与恒等式终态共用 run/run_step/execute_order_chain/_finalize_day 守卫
        self._chain_failed: Optional[ChainFailedTerminatedError] = None
        # 订单级全链结果（仅内存；同一 batch_id 重放返回 applied=False，不重复执行）
        self._chain_results: Dict[str, OrderChainResult] = {}
        # 执行期占用/失败批次（物理可能已部分成交）：禁止同键重试；同实例亦不可
        # 继续（物理基数与来源视图已分叉），恢复须新建实例并从破缺前信封重放
        self._chain_failures: Dict[str, str] = {}

    # ────────────────────────── 终态 ──────────────────────────

    @property
    def is_terminated(self) -> bool:
        """本实例是否已进入终态（恒等式破缺或订单级全链执行后失败）。"""
        return self._terminated is not None or self._chain_failed is not None

    def _assert_not_terminated(self) -> None:
        """终态守卫：破缺/执行后失败后禁止任何日步进（复用实例不得产生新证据）。"""
        if self._terminated is not None:
            raise IdentityTerminatedError(
                trade_date=self._terminated.trade_date,
                original=self._terminated.original,
            )
        if self._chain_failed is not None:
            raise ChainFailedTerminatedError(
                trade_date=self._chain_failed.trade_date,
                batch_id=self._chain_failed.batch_id,
                original=self._chain_failed.original,
            )

    def run(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
        *,
        trading_dates_full: Optional[List[pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """连续运行入口：终态实例在父类任何可变操作（日历/价源/排期/缓存）前拒绝。"""
        self._assert_not_terminated()
        return super().run(
            start_date,
            end_date,
            trading_dates,
            price_data,
            trading_dates_full=trading_dates_full,
        )

    def run_step(
        self,
        date: pd.Timestamp,
        trading_dates_full: List[pd.Timestamp],
        price_data: pd.DataFrame,
    ) -> None:
        """日步进入口：终态实例在任何可变操作前拒绝，其余委托父类。"""
        self._assert_not_terminated()
        super().run_step(date, trading_dates_full, price_data)

    # ────────────────────────── T3 订单级全链 ──────────────────────────

    def execute_order_chain(
        self,
        date: pd.Timestamp,
        sleeve_targets: Mapping[str, Mapping[str, int]],
        *,
        batch_id: str,
        price_source: Optional[PriceSource] = None,
        fill_capacity: Optional[Mapping[str, int]] = None,
        reason: OrderReason = OrderReason.REBALANCE,
    ) -> OrderChainResult:
        """执行一次订单级全链（真实 Order → 执行器 → 物理入账 → 结算 → 虚拟投影）。

        顺序与零副作用边界：① 计划与物理基数对账；② 预演（预测成交 + 副本账本/
        视图上跑一次真实结算，任何输入拒绝都在物理执行前完成）；③ 真实执行
        （``_buy_stock_direct`` → ``Ledger.apply_fills``，成交事实由受控注入点产出
        并与预演逐值比对）；④ 构造批次并真实结算 → 成功后替换物理 lot 伴随视图。
        同一 ``batch_id`` 重放返回 ``applied=False`` 且**不重复执行**（物理幂等键
        会随成交条数变化，链级身份必须独立守卫）。

        Raises:
            OrderChainError: 计划/预演拒绝（零副作用）。
            OrderChainInvariantError: 执行链观测与预演不一致（实现缺陷）。
            ReconciliationError: 结算后 I1/I2 破缺（终态，当日停止）。
        """
        self._assert_not_terminated()
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise OrderChainError("阶段=P6.5：结算批次身份 batch_id 不得为空（稳定幂等键）")
        request_fp = self._chain_request_fingerprint(
            date=date,
            sleeve_targets=sleeve_targets,
            batch_id=batch_id,
            price_source=price_source,
            fill_capacity=fill_capacity,
            reason=reason,
        )
        if batch_id in self._chain_results:
            recorded = self._chain_results[batch_id]
            if recorded.request_fingerprint != request_fp:
                raise OrderChainError(
                    f"阶段=P6.5 批次={batch_id}：该身份已承载另一套请求（日期/目标/上限/"
                    "原因/价源标识不同）；禁止一个批次键承载两套事实"
                )
            return replace(recorded, applied=False)
        restored_fingerprints = getattr(self, "restored_chain_fingerprints", None) or {}
        if batch_id in restored_fingerprints:
            # T5 恢复代：链结果（计划/成交/结算事实）不随信封重建，因此同键同内容
            # **也只能拒绝**；但必须与"同键异内容"可区分（幂等语义不得退化成一类拒绝）。
            recorded_fingerprint = restored_fingerprints[batch_id]
            if recorded_fingerprint == request_fp:
                raise OrderChainError(
                    f"阶段=P6.5 批次={batch_id}：该身份在恢复代已登记为**同一请求**"
                    "（同键同内容）；恢复不重建链结果，禁止重放（零副作用），"
                    "后续批次必须使用新的稳定批次键"
                )
            raise OrderChainError(
                f"阶段=P6.5 批次={batch_id}：该身份在恢复代已承载**另一套请求**"
                f"（登记指纹={recorded_fingerprint} 本次={request_fp}）；"
                "禁止同一批次键承载两套事实"
            )
        if batch_id in self.virtual_ledger.batch_ids():
            raise OrderChainError(
                f"阶段=P6.5 批次={batch_id}：该身份已被虚拟账结算占用；"
                "链级身份不得复用（必须使用新的稳定批次键）"
            )
        day = pd.Timestamp(date)
        source, plan, predicted, context = self._chain_preflight(
            day=day,
            sleeve_targets=sleeve_targets,
            batch_id=batch_id,
            price_source=price_source,
            fill_capacity=fill_capacity,
            reason=reason,
        )
        observed, batch, settlement = self._chain_execute_and_settle(
            day=day,
            batch_id=batch_id,
            plan=plan,
            predicted=predicted,
            source=source,
            context=context,
        )
        result = OrderChainResult(
            trade_date=TradeDate(day.date()),
            batch_id=batch_id,
            request_fingerprint=request_fp,
            plan=plan,
            batch=batch,
            settlement=settlement,
            predicted=predicted,
            observed=observed,
            physical_cash_before=float(batch.physical_cash_before),
            physical_cash_after=float(batch.physical_cash_after),
            applied=bool(settlement.applied),
        )
        self._chain_results[batch_id] = result
        return result

    def _chain_price_source(
        self,
        date: pd.Timestamp,
        states: Mapping[str, Any],
        provided: Optional[PriceSource],
        *,
        extra_codes: Sequence[str],
    ) -> PriceSource:
        """全链价源：缺省走引擎同日收盘价；显式价源必须与物理估值逐股同价。

        ``extra_codes`` = 本批目标股票（可能尚无持仓/虚拟批次的**新买入**），
        价源必须覆盖它们，否则下单前无法定价（P6.5 无价即拒绝，不建顺延义务）。
        """
        if provided is None:
            return self._build_price_source(date, states=states, extra_codes=extra_codes)
        if not isinstance(provided, PriceSource):
            raise TypeError(f"全链价源必须为 PriceSource，实际 {type(provided).__name__}")
        self._assert_source_matches_physical(
            date, states=states, source=provided, extra_codes=extra_codes
        )
        return provided

    def _chain_request_fingerprint(
        self,
        *,
        date: Any,
        sleeve_targets: Mapping[str, Mapping[str, int]],
        batch_id: str,
        price_source: Optional[PriceSource],
        fill_capacity: Optional[Mapping[str, int]],
        reason: OrderReason,
    ) -> str:
        """链入口校验（终态/失败登记/身份/目标/上限）+ 请求指纹（全部零副作用）。"""
        if batch_id in self._chain_failures:
            raise OrderChainError(
                f"阶段=P6.5 批次={batch_id}：该批次已登记为执行中/执行后失败"
                f"（状态={self._chain_failures[batch_id]}）；物理可能已部分成交，"
                "禁止同键重试；同实例亦不可继续（物理基数与来源视图已分叉），"
                "恢复须新建实例并从破缺前信封重放"
            )
        if self.virtual_ledger.poisoned_reason is not None:
            raise OrderChainError(
                f"阶段=P6.5：虚拟账本已因守恒/恒等式破缺进入终态"
                f"（原因={self.virtual_ledger.poisoned_reason}）；"
                "禁止提交新链，恢复须新建实例并重放"
            )
        if batch_id in self.virtual_ledger.settlement_ids():
            raise OrderChainError(
                f"阶段=P6.5 批次={batch_id}：该身份已被虚拟账内部转账/派生键占用；"
                "禁止同一结算键承载两套事实"
            )
        normalized = validate_sleeve_targets(targets=sleeve_targets, batch_id=batch_id, date=date)
        if fill_capacity is not None:
            if not isinstance(fill_capacity, Mapping):
                raise OrderChainError(
                    f"阶段=P6.5：合成成交上限必须为 Mapping，实际 {type(fill_capacity).__name__}"
                )
            for code, value in fill_capacity.items():
                if isinstance(value, bool) or not hasattr(value, "__index__"):
                    raise OrderChainError(
                        f"阶段=P6.5 股票={code}：合成成交上限必须为整数股，实际 "
                        f"{type(value).__name__}={value!r}"
                    )
        return chain_request_fingerprint(
            date=pd.Timestamp(date),
            targets=normalized,
            capacity=fill_capacity,
            reason=reason,
            price_source_id=(
                price_source.source_id if isinstance(price_source, PriceSource) else ""
            ),
            price_source_as_of=(
                str(price_source.as_of) if isinstance(price_source, PriceSource) else ""
            ),
            price_map=(
                dict(price_source.prices) if isinstance(price_source, PriceSource) else None
            ),
        )

    def _chain_preflight(
        self,
        *,
        day: pd.Timestamp,
        sleeve_targets: Mapping[str, Mapping[str, int]],
        batch_id: str,
        price_source: Optional[PriceSource],
        fill_capacity: Optional[Mapping[str, int]],
        reason: OrderReason,
    ) -> Tuple[PriceSource, NettingPlan, Tuple[ChainFill, ...], IdentityContext]:
        """计划 + 预演（全部在物理执行前完成；任何拒绝零副作用）。"""
        states = self.virtual_ledger.snapshot()
        target_codes = tuple(
            sorted({str(code) for sleeve in sleeve_targets.values() for code in sleeve})
        )
        source = self._chain_price_source(day, states, price_source, extra_codes=target_codes)
        try:
            orders = build_sleeve_orders(
                targets=sleeve_targets,
                ledger=self.virtual_ledger,
                date=day,
                batch_id=batch_id,
                reason=reason,
            )
            plan = plan_order_chain(
                merger=self.portfolio_merger,
                ledger=self.virtual_ledger,
                price_source=source,
                orders=orders,
                batch_id=batch_id,
                date=day,
            )
        except (VirtualMergerError, VirtualLedgerError) as error:
            raise OrderChainError(
                f"阶段=P6.5 批次={batch_id} 日期={day.date()}：{error}"
            ) from error
        assert_plan_integrity(plan, batch_id=batch_id, date=day, source_orders=orders)
        assert_physical_base(plan, self.positions, day, batch_id)
        assert_no_physical_sells(plan, day, batch_id)
        ratio = float(getattr(self, "min_buy_value_ratio", 0.0) or 0.0)
        if ratio > 0:
            raise OrderChainError(
                f"批次={batch_id} 日期={day.date()}：订单级全链要求 min_buy_value_ratio=0，"
                f"实际 {ratio}；执行期逐单重算的市值阈值会让预演与执行产生漂移"
                "（拒绝发生在物理执行前，零副作用）"
            )
        predicted = predict_fills(
            engine=self,
            plan=plan,
            price_source=source,
            date=day,
            batch_id=batch_id,
            capacity=fill_capacity,
        )
        context = self._identity_context(day, states, source.source_id)
        predicted_batch = build_chain_batch(
            plan=plan,
            fills=predicted,
            batch_id=batch_id,
            date=day,
            price_source=source,
            engine=self,
        )
        assert_settleable(
            ledger=self.virtual_ledger,
            batch=predicted_batch,
            price_source=source,
            view=register_fills(self.physical_lot_view, predicted),
            context=context,
            positions=predicted_positions(self, predicted),
            batch_id=batch_id,
        )
        return source, plan, predicted, context

    def _chain_execute_and_settle(
        self,
        *,
        day: pd.Timestamp,
        batch_id: str,
        plan: NettingPlan,
        predicted: Tuple[ChainFill, ...],
        source: PriceSource,
        context: IdentityContext,
    ) -> Tuple[Tuple[ChainFill, ...], InternalSettlementBatch, SettlementResult]:
        """真实执行 + 结算（执行期占用：物理执行前登记，成功结算后释放）。

        异常类型不可枚举，占用不依赖 except 分支是否命中：任何中途异常（含
        KeyboardInterrupt 等未转换异常）都会保留登记并**锁定实例终态**
        （``ChainFailedTerminatedError``），使同键重试、新批次与后续日步进全部被拒。
        """
        self._chain_failures[batch_id] = "执行中（未完成；物理可能已部分成交）"
        try:
            observed = execute_fills(
                engine=self, plan=plan, predictions=predicted, date=day, batch_id=batch_id
            )
            batch = build_chain_batch(
                plan=plan,
                fills=observed,
                batch_id=batch_id,
                date=day,
                price_source=source,
                engine=self,
            )
            assert_fills_booked(self, observed, batch_id=batch_id, date=day)
            view = register_fills(self.physical_lot_view, observed)
            settlement = self.virtual_ledger.apply_settlement(
                batch,
                price_source=source,
                physical_view=view,
                identity_context=context,
                physical_positions=self.positions,
            )
        except BaseException as error:  # 执行期失败一律锁定实例终态（类型不可枚举）
            self._chain_failed = ChainFailedTerminatedError(
                trade_date=TradeDate(day.date()), batch_id=batch_id, original=error
            )
            if isinstance(error, (OrderChainInvariantError, LedgerError)):
                raise
            if isinstance(
                error,
                (VirtualLedgerError, VirtualLedgerConservationError, ReconciliationError),
            ):
                raise OrderChainInvariantError(
                    f"阶段=P6.5 批次={batch_id} 日期={day.date()}：预演通过但真实结算拒绝"
                    f"（物理成交已发生，链路实现缺陷，禁止同键重试）——{error}"
                ) from error
            if isinstance(error, Exception):
                raise OrderChainInvariantError(
                    f"阶段=P6.5 批次={batch_id} 日期={day.date()}：执行链中途异常"
                    f"（{type(error).__name__}: {error}）；物理可能已部分成交，禁止同键重试"
                ) from error
            raise
        del self._chain_failures[batch_id]
        self.physical_lot_view = view
        return observed, batch, settlement

    def order_chain_result(self, batch_id: str) -> OrderChainResult:
        """已执行的链结果（只读；未执行返回 KeyError）。"""
        return self._chain_results[batch_id]

    # ────────────────────────── 挂载点 ──────────────────────────

    def _finalize_day(
        self,
        idx: int,
        date: pd.Timestamp,
        trade_start_idx: int,
        total_days: int,
        cycle_day: int,
        portfolio_value: float,
        trading_dates: List[pd.Timestamp],
        date_to_idx: Dict[pd.Timestamp, int],
    ) -> None:
        """先验 I1/I2/I3（无成交日也执行），通过后才调用父类日终快照。

        破缺时本方法在调用父类前抛出：``portfolio_values`` / ``holdings_snapshots``
        不追加、``identity_reports`` 不登记、不进入下一日，异常由 ``run_step``
        向外传播即当日停止。**边界（如实）**：当日已发生的成交写入与估值缓存
        （``last_known_price``）回写不在回滚范围；恢复按设计 §4.5 从破缺前信封重放。
        破缺后实例进入终态，后续日步进被 ``_assert_not_terminated`` 拒绝。
        """
        self._assert_not_terminated()
        self._run_identity_gate(date=date, portfolio_value=portfolio_value)
        super()._finalize_day(
            idx,
            date,
            trade_start_idx,
            total_days,
            cycle_day,
            portfolio_value,
            trading_dates,
            date_to_idx,
        )

    # ────────────────────────── 校验 ──────────────────────────

    def _run_identity_gate(
        self, *, date: pd.Timestamp, portfolio_value: float
    ) -> DailyIdentityReport:
        """执行当日三恒等式并登记证据（失败即抛并锁定终态，不登记证据、不写日终）。"""
        states = self.virtual_ledger.snapshot()
        try:
            price_source = self._build_price_source(date, states=states)
            context = self._identity_context(date, states, price_source.source_id)
            report = run_daily_identities(
                physical_cash=float(self.current_capital),
                physical_total_value=float(portfolio_value),
                physical_initial_capital=float(self.initial_capital),
                physical_positions=self.positions,
                physical_view=self.physical_lot_view,
                virtual_states=states,
                price_source=price_source,
                context=context,
            )
        except ReconciliationError as error:
            self._terminated = IdentityTerminatedError(
                trade_date=TradeDate(pd.Timestamp(date).date()),
                original=error,
            )
            raise
        self.identity_reports.append(report)
        self.identity_check_count += 1
        return report

    def _identity_context(
        self,
        date: pd.Timestamp,
        states: Mapping[str, Any],
        price_source_id: str,
    ) -> IdentityContext:
        """构造当日恒等式上下文（宿主/运行 ID/袖集合/价源标识/最近结算）。"""
        return IdentityContext(
            trade_date=TradeDate(pd.Timestamp(date).date()),
            host=HOST_ID,
            run_id=self.identity_run_id or "<未指定>",
            sleeve_ids=tuple(states),
            price_source_id=price_source_id,
            last_settlement_id=self.virtual_ledger.last_settlement_id or "",
            last_settlement_source_id=self.virtual_ledger.last_settlement_source_id or "",
        )

    def _price_codes(self, states: Mapping[str, Any], extra_codes: Sequence[str] = ()) -> set:
        """当日需要同源价格的股票集（物理持仓 ∪ 虚拟批次 ∪ 本批目标股票）。"""
        codes = {str(code) for code in self.positions}
        for state in states.values():
            codes.update(lot.ts_code for lot in state.lots)
        codes.update(str(code) for code in extra_codes)
        return codes

    def _build_price_source(
        self,
        date: pd.Timestamp,
        *,
        states: Mapping[str, Any],
        extra_codes: Sequence[str] = (),
    ) -> PriceSource:
        """构造当日价格源（显式注入优先；缺省走引擎收盘成交价，缺价即拒绝）。

        ``states`` 必须与本次被校验的虚拟快照是**同一份**，避免未来在取快照与
        构造价源之间插入结算导致价源股票集与校验对象漂移。
        """
        if self.price_source_provider is not None:
            source = self.price_source_provider(date)
            if not isinstance(source, PriceSource):
                raise TypeError(
                    "price_source_provider 必须返回 PriceSource，实际 " f"{type(source).__name__}"
                )
            self._assert_source_matches_physical(
                date, states=states, source=source, extra_codes=extra_codes
            )
            return source
        return self._build_engine_price_source(date, states=states, extra_codes=extra_codes)

    def _assert_source_matches_physical(
        self,
        date: pd.Timestamp,
        *,
        states: Mapping[str, Any],
        source: PriceSource,
        extra_codes: Sequence[str] = (),
    ) -> None:
        """显式价源必须与物理估值**逐股同日同价**（组合总值相等不能证明同源）。

        这是 I3 的同源前置：物理 ``portfolio_value`` 由引擎自己的收盘成交价索引算出，
        虚拟侧若用另一套价格，即使两套价格的总值恰好相等，袖级贡献 NAV 仍会被错分。
        """
        context = self._identity_context(date, states, source.source_id)
        for code in sorted(self._price_codes(states, extra_codes)):
            try:
                physical_price = self._get_trade_price(date, code)
            except AttributeError as error:
                raise ReconciliationError(
                    identity="I3",
                    context=context,
                    detail=(
                        f"引擎价格索引未就绪，无法证明显式价源与物理估值同源（{code}）；"
                        "拒绝在未绑定物理价源时估值"
                    ),
                ) from error
            if physical_price is None or pd.isna(physical_price):
                raise ReconciliationError(
                    identity="I3",
                    context=context,
                    detail=f"物理估值缺少 {code} 的当日收盘成交价，无法证明显式价源同源",
                )
            try:
                source_price = source.price_of(code)
            except VirtualLedgerError as error:
                raise ReconciliationError(
                    identity="I3",
                    context=context,
                    detail=f"显式价源缺少 {code} 的当日价格：{error}",
                ) from error
            if float(physical_price) != source_price:
                raise ReconciliationError(
                    identity="I3",
                    context=context,
                    detail=(
                        f"显式价源与物理估值不同源：股票 {code} 价源={source_price!r} "
                        f"引擎物理收盘价={float(physical_price)!r}；同日逐股必须同价，"
                        "组合总值相等不能证明同源"
                    ),
                )

    def _build_engine_price_source(
        self,
        date: pd.Timestamp,
        *,
        states: Mapping[str, Any],
        extra_codes: Sequence[str] = (),
    ) -> PriceSource:
        """以引擎收盘成交价（``_get_trade_price``）构造同日同价源。"""
        trade_date = TradeDate(pd.Timestamp(date).date())
        prices: Dict[str, float] = {}
        missing: List[str] = []
        for code in sorted(self._price_codes(states, extra_codes)):
            price = self._get_trade_price(date, code)
            if price is None or pd.isna(price):
                missing.append(code)
                continue
            prices[code] = float(price)
        if missing:
            context = self._identity_context(date, states, self._engine_source_id(trade_date))
            raise ReconciliationError(
                identity="I3",
                context=context,
                detail=(
                    f"价格源缺少 {missing} 的当日收盘成交价；P6.5 合成边界无价即拒绝，"
                    "不建立跨日顺延义务，也不得回退缓存价参与 I3"
                ),
            )
        return PriceSource(
            as_of=trade_date,
            source_id=self._engine_source_id(trade_date),
            prices=MappingProxyType(prices),
        )

    @staticmethod
    def _engine_source_id(trade_date: TradeDate) -> str:
        """引擎价格源标识（日期 + 口径，进入错误上下文与证据）。"""
        return f"engine_trade_price:{trade_date}"

    # ────────────────────────── 只读视图 ──────────────────────────

    def identity_report_snapshot(self) -> Sequence[DailyIdentityReport]:
        """已通过的逐日恒等式报告（只读投影，供验收观测）。

        证据为**实例级、单轮使用**；同一实例复跑多轮前必须显式
        ``reset_identity_evidence()``，避免上一轮通过项混入本轮。
        """
        return tuple(self.identity_reports)

    def reset_identity_evidence(self) -> None:
        """清空本实例的恒等式证据（同一实例多轮运行的显式边界）。"""
        self.identity_reports.clear()
        self.identity_check_count = 0

    def virtual_accounts(self, price_source: PriceSource) -> Mapping[str, Any]:
        """当前虚拟账的协议值对象投影（类型兼容用；I2 仍以 VirtualSleeveState 为准）。"""
        return self.virtual_ledger.as_virtual_accounts(price_source)
