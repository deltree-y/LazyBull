"""P6.5 T1 三恒等式校验器（I1 现金 / I2 持仓 / I3 贡献 NAV）。

契约依据：``docs/contracts/netting_freeze.md`` F25 §4（容差与口径）、
``docs/contracts/protocols.md`` F13 §3.3、设计
``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §4.3/§7。

口径（F25 冻结，本模块为唯一实现）：
- **I1**：``Σ_i 虚拟现金_i``（含借用占用对消）与物理现金绝对差 ≤ ``1e-6`` 元；
  P6.5 无借用子集另要求**逐袖** ``borrowed_credit == 0`` 且费用后自有虚拟现金非负。
  组合现金非负或 ``(+100,-100)`` 占用合计为 0 都**不能**替代逐袖守卫。
- **I2**：先按 ``(ts_code, 物理来源 lot_id)`` 汇总虚拟整数股数，再与物理伴随视图
  逐键精确比较（0 股差）；两侧来源键集相同，虚拟侧不得出现物理不存在的键；
  **不比较 lot 条目数**；虚拟取得批次按 ``(sleeve_id, virtual_lot_id)`` 唯一。
  右值是**合成伴随视图**，不得表述为已与生产物理账逐 lot 对账。
- **I3**：``NAV_i_contrib = total_value_i / 物理账户初始资金``，断言
  ``Σ NAV_i_contrib`` 与物理 NAV 绝对差 ≤ ``1e-6``（无量纲）；``Σ total_value_i``
  与物理总值另以同容差作辅证；所有袖共用同日同价源。各袖自身初始资金为分母的
  独立 NAV 不参与 I3（Q19）。I3 容差门**不是** G4 的逐位门。

任何破缺抛 ``ReconciliationError``（``RuntimeError`` 语义），禁止降级为日志告警。
I2 右值前置破缺（伴随视图与引擎聚合持仓不一致）在本层统一包裹为
``ReconciliationError(identity="I2")``，调用方只需捕获一种异常。

**失败模式（设计 §7.2-5）**：成功路径按 I1 → I2 → I3 固定顺序执行；失败路径在**同一
只读快照**上把三项全部评估完，汇总**全部破缺编号**与逐项上下文后一次抛出
``ReconciliationError``（``identities`` 给出全部编号），随即终止当日——不提交、
不写日终快照、不进入下一日。三项校验均无写副作用，因此"聚合诊断 + 立即停止"
与"任一破缺即报错停止"并不冲突。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.core.accounting.physical_lot_view import (
    PhysicalLotMismatchError,
    PhysicalLotView,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedgerError,
    VirtualSleeveState,
)

#: I1 货币绝对容差（元；F25 / Q5=B）
CASH_TOLERANCE = 1e-6
#: I2 整股容差（股；精确零差）
HOLDINGS_TOLERANCE_SHARES = 0
#: I3 直接贡献 NAV 绝对容差（无量纲；F25 / Q19）
NAV_TOLERANCE = 1e-6

#: 本模块唯一阶段标识（进入全部错误上下文）
STAGE = "P6.5"


@dataclass(frozen=True, slots=True)
class IdentityContext:
    """恒等式校验上下文（错误消息与证据的共同载体）。"""

    trade_date: TradeDate
    host: str
    run_id: str
    sleeve_ids: Tuple[str, ...]
    price_source_id: str
    last_settlement_id: str = ""
    stage: str = STAGE
    #: 最近一次内部结算所用价格源标识（T1 复审 P3：定价证据进入上下文）
    last_settlement_source_id: str = ""

    def describe(self) -> str:
        """单行上下文（阶段/日期/宿主/运行 ID/袖集合/价格源/最近结算）。"""
        settlement = self.last_settlement_id or "<无>"
        settlement_source = self.last_settlement_source_id or "<无>"
        return (
            f"阶段={self.stage} 日期={self.trade_date} 宿主={self.host} 运行ID={self.run_id} "
            f"袖集合={list(self.sleeve_ids)} 价格源={self.price_source_id} "
            f"最近结算={settlement} 结算价源={settlement_source}"
        )


@dataclass(frozen=True, slots=True)
class IdentityResult:
    """单条恒等式的成功证据（左值/右值/差/容差/单位）。

    I1/I3 为浮点（元/无量纲），I2 为**整数股**（``tolerance=0``，禁止浮点化）。
    """

    identity: str
    left: float | int
    right: float | int
    diff: float | int
    tolerance: float | int
    unit: str
    #: I2 逐来源键明细（``(ts_code, 物理来源 lot_id, 整数股)``，已排序；I1/I3 为空）
    sources: Tuple[Tuple[str, str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class DailyIdentityReport:
    """单日三恒等式报告（逐日挂载的成功证据）。"""

    trade_date: TradeDate
    host: str
    run_id: str
    price_source_id: str
    sleeve_ids: Tuple[str, ...]
    results: Tuple[IdentityResult, ...]
    last_settlement_id: str = ""
    last_settlement_source_id: str = ""

    @property
    def identities(self) -> Tuple[str, ...]:
        """本报告已执行的恒等式编号顺序。"""
        return tuple(result.identity for result in self.results)


class ReconciliationError(RuntimeError):
    """三恒等式破缺（终态；当日停止，不写快照、不继续下一日）。

    联动破缺时 ``related`` 携带其余破缺项，``identities`` 给出**全部**破缺编号
    （§7.2-5 的"必须含全部破缺编号"），消息内逐项保留上下文。
    """

    def __init__(
        self,
        *,
        identity: str,
        context: IdentityContext,
        detail: str,
        related: Tuple["ReconciliationError", ...] = (),
    ) -> None:
        self.identity = identity
        self.context = context
        self.detail = detail
        self.related = tuple(related)
        self.identities = (identity, *(error.identity for error in self.related))
        message = f"[{context.stage}][{identity}] {detail}｜{context.describe()}"
        if self.related:
            others = "；".join(f"[{error.identity}] {error.detail}" for error in self.related)
            message = f"{message}；联动破缺（全部编号={list(self.identities)}）：{others}"
        super().__init__(message)


def _raise_identity_error(identity: str, context: IdentityContext, detail: str) -> None:
    raise ReconciliationError(identity=identity, context=context, detail=detail)


def _run_checks(
    checks: Tuple[Any, ...], context: IdentityContext
) -> Tuple[IdentityResult, ...]:
    """只读检查批量执行：成功按序返回；失败汇总**全部**破缺编号后一次抛出。

    与设计 §7.2-5 一致：失败路径在同一只读快照上评估完全部检查，调用方因此
    总能拿到 ``ReconciliationError.identities`` 的完整编号，而不是首个破缺。
    """
    results: list[IdentityResult] = []
    errors: list[ReconciliationError] = []
    for check in checks:
        try:
            results.append(check())
        except ReconciliationError as error:
            errors.append(error)
    if errors:
        first = errors[0]
        if len(errors) == 1:
            raise first
        raise ReconciliationError(
            identity=first.identity,
            context=context,
            detail=first.detail,
            related=tuple(errors[1:]),
        )
    return tuple(results)


def _ordered_sleeves(virtual_states: Mapping[str, VirtualSleeveState]) -> Tuple[str, ...]:
    """确定性袖顺序（字典序，避免求和顺序随插入顺序漂移）。"""
    return tuple(sorted(virtual_states))


def _require_finite(value: Any, label: str, identity: str, context: IdentityContext) -> float:
    """有限性守卫：NaN/Inf 会让 ``abs(diff) > tol`` 恒为 False，必须先拒绝。"""
    number = float(value)
    if not math.isfinite(number):
        _raise_identity_error(identity, context, f"{label} 非有限（{value!r}），拒绝参与恒等式比较")
    return number


def sleeve_total_value(state: VirtualSleeveState, price_source: PriceSource) -> float:
    """单袖货币总值（现金 + 同日同价源市值；I3 的 ``total_value_i``）。

    I3 的唯一估值入口；**不是** T2 净额接口的公共 API（多袖上下文由 T2 单独立）。
    """
    return state.total_value(price_source)


# ────────────────────────── I1 现金 ──────────────────────────


def check_cash_identity(
    *,
    physical_cash: float,
    virtual_states: Mapping[str, VirtualSleeveState],
    context: IdentityContext,
) -> IdentityResult:
    """I1：``Σ虚拟现金（含占用对消）== 物理现金``（≤1e-6 元）+ 逐袖无借用守卫。"""
    right = _require_finite(physical_cash, "物理现金", "I1", context)
    cash_with_occupation = 0.0
    for sleeve_id in _ordered_sleeves(virtual_states):
        state = virtual_states[sleeve_id]
        cash = _require_finite(state.cash, f"袖 {sleeve_id} 虚拟现金", "I1", context)
        credit = _require_finite(
            state.borrowed_credit, f"袖 {sleeve_id} borrowed_credit", "I1", context
        )
        if credit != 0.0:
            _raise_identity_error(
                "I1",
                context,
                f"袖 {sleeve_id} 借用占用 borrowed_credit={credit!r} 必须为 0"
                "（P6.5 无借用模式；占用合计为 0 不能替代逐袖守卫）",
            )
        if cash < 0.0:
            _raise_identity_error(
                "I1",
                context,
                f"袖 {sleeve_id} 费用后自有虚拟现金为负：cash={cash!r}（元）；"
                "组合现金非负不能替代逐袖守卫",
            )
        cash_with_occupation += cash + credit
    left = _require_finite(cash_with_occupation, "Σ虚拟现金", "I1", context)
    diff = left - right
    if abs(diff) > CASH_TOLERANCE:
        _raise_identity_error(
            "I1",
            context,
            f"Σ虚拟现金(含占用对消)={left!r} 元 != 物理现金={right!r} 元；"
            f"差={diff!r} 元 容差={CASH_TOLERANCE} 单位=元 袖集合={list(context.sleeve_ids)}",
        )
    return IdentityResult("I1", left, right, diff, CASH_TOLERANCE, "元")


# ────────────────────────── I2 持仓 ──────────────────────────


def _aggregate_virtual_shares(
    virtual_states: Mapping[str, VirtualSleeveState], context: IdentityContext
) -> Tuple[Dict[Tuple[str, str], int], Tuple[Tuple[str, str], ...]]:
    """按 ``(ts_code, 物理来源 lot_id)`` 汇总虚拟股数并校验取得批次唯一性。"""
    virtual_shares: Dict[Tuple[str, str], int] = {}
    acquisitions: list[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for sleeve_id in _ordered_sleeves(virtual_states):
        for lot in virtual_states[sleeve_id].lots:
            key = (lot.ts_code, lot.source_lot_id)
            virtual_shares[key] = virtual_shares.get(key, 0) + lot.shares
            acquisition = (sleeve_id, lot.virtual_lot_id)
            if acquisition in seen:
                _raise_identity_error(
                    "I2",
                    context,
                    f"虚拟取得批次身份重复：袖={sleeve_id} 虚拟取得 lot_id={lot.virtual_lot_id} "
                    "（(sleeve_id, virtual_lot_id) 必须唯一）",
                )
            seen.add(acquisition)
            acquisitions.append(acquisition)
    return virtual_shares, tuple(acquisitions)


def _compare_source_keys(
    *,
    virtual_shares: Mapping[Tuple[str, str], int],
    physical_shares: Mapping[Tuple[str, str], int],
    context: IdentityContext,
) -> None:
    """来源键集与逐键股数精确比较（整数零差；不比条目数）。"""
    missing = sorted(set(physical_shares) - set(virtual_shares))
    extra = sorted(set(virtual_shares) - set(physical_shares))
    if missing or extra:
        _raise_identity_error(
            "I2",
            context,
            f"来源键集不一致：物理有而虚拟缺={missing}，虚拟有而物理无={extra}；"
            f"左(虚拟)键数={len(virtual_shares)} 右(物理伴随视图)键数={len(physical_shares)} "
            f"容差={HOLDINGS_TOLERANCE_SHARES} 单位=股",
        )
    for key in sorted(physical_shares):
        left = virtual_shares[key]
        right = physical_shares[key]
        if left != right:
            _raise_identity_error(
                "I2",
                context,
                f"来源批次股数不精确相等：股票={key[0]} 物理来源lot={key[1]}；"
                f"左(虚拟)={left} 股 右(物理伴随视图)={right} 股 差={left - right} 股 "
                f"容差={HOLDINGS_TOLERANCE_SHARES} 股",
            )


def check_holdings_identity(
    *,
    physical_view: PhysicalLotView,
    virtual_states: Mapping[str, VirtualSleeveState],
    context: IdentityContext,
    physical_positions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> IdentityResult:
    """I2：来源 lot 汇总股数逐键精确相等（右值 = 合成物理 lot 伴随视图）。"""
    if physical_positions is not None:
        try:
            physical_view.validate_against_positions(
                physical_positions, context=context.describe()
            )
        except PhysicalLotMismatchError as error:
            raise ReconciliationError(
                identity="I2",
                context=context,
                detail=f"右值前置破缺（伴随视图 vs 引擎聚合持仓）：{error}",
            ) from error
    virtual_shares, _acquisitions = _aggregate_virtual_shares(virtual_states, context)
    physical_shares = dict(physical_view.aggregate())
    _compare_source_keys(
        virtual_shares=virtual_shares,
        physical_shares=physical_shares,
        context=context,
    )
    left = sum(virtual_shares.values())
    right = sum(physical_shares.values())
    sources = tuple(
        (ts_code, source_lot_id, shares)
        for (ts_code, source_lot_id), shares in sorted(virtual_shares.items())
    )
    return IdentityResult(
        "I2",
        left,
        right,
        left - right,
        HOLDINGS_TOLERANCE_SHARES,
        "股",
        sources=sources,
    )


# ────────────────────────── I3 贡献 NAV ──────────────────────────


def check_nav_identity(
    *,
    virtual_states: Mapping[str, VirtualSleeveState],
    physical_total_value: float,
    physical_initial_capital: float,
    price_source: PriceSource,
    context: IdentityContext,
) -> IdentityResult:
    """I3：``Σ total_value_i / 物理初始资金 == 物理 NAV``（≤1e-6 无量纲）。"""
    initial = _require_finite(physical_initial_capital, "物理账户初始资金", "I3", context)
    if initial <= 0.0:
        _raise_identity_error(
            "I3", context, f"物理账户初始资金必须为正数，实际 {initial!r}（不得用各袖自身初始资金）"
        )
    if price_source.as_of != context.trade_date:
        _raise_identity_error(
            "I3",
            context,
            f"价格源日期 {price_source.as_of} != 校验日期 {context.trade_date}；"
            "所有袖必须共用同日同价源",
        )
    physical_value = _require_finite(physical_total_value, "物理组合总值", "I3", context)
    physical_nav = float(calculate_nav(physical_value, initial))
    contribution = 0.0
    total_value_sum = 0.0
    for sleeve_id in _ordered_sleeves(virtual_states):
        try:
            raw_value = sleeve_total_value(virtual_states[sleeve_id], price_source)
        except VirtualLedgerError as error:
            raise ReconciliationError(
                identity="I3",
                context=context,
                detail=f"袖 {sleeve_id} 同日同价源估值失败：{error}",
            ) from error
        value = _require_finite(raw_value, f"袖 {sleeve_id} 货币总值", "I3", context)
        total_value_sum += value
        # 贡献 NAV 与物理 NAV 共用 nav.py 唯一派生入口，禁止第二套除法实现
        contribution += float(calculate_nav(value, initial))
    left = _require_finite(contribution, "Σ贡献NAV", "I3", context)
    diff = left - physical_nav
    if abs(diff) > NAV_TOLERANCE:
        _raise_identity_error(
            "I3",
            context,
            f"Σ NAV_i_contrib={left!r} != 物理NAV={physical_nav!r}；差={diff!r} "
            f"容差={NAV_TOLERANCE} 单位=无量纲（分母=物理初始资金 {initial!r} 元）",
        )
    value_diff = total_value_sum - physical_value
    if abs(value_diff) > NAV_TOLERANCE:
        _raise_identity_error(
            "I3",
            context,
            f"辅证：Σ total_value_i={total_value_sum!r} 元 != 物理总值={physical_value!r} 元；"
            f"差={value_diff!r} 元 容差={NAV_TOLERANCE} 元",
        )
    return IdentityResult("I3", left, physical_nav, diff, NAV_TOLERANCE, "无量纲")


# ────────────────────────── 逐日编排 ──────────────────────────


def run_daily_identities(
    *,
    physical_cash: float,
    physical_total_value: float,
    physical_initial_capital: float,
    physical_positions: Mapping[str, Mapping[str, Any]],
    physical_view: PhysicalLotView,
    virtual_states: Mapping[str, VirtualSleeveState],
    price_source: PriceSource,
    context: IdentityContext,
) -> DailyIdentityReport:
    """按 I1 → I2 → I3 固定顺序执行三恒等式。

    成功路径按序返回三项结果；失败路径在同一份只读快照上把三项都评估完，
    汇总**全部**破缺编号（``ReconciliationError.identities``）后一次抛出；
    任一破缺都不提交、不写日终快照、不进入下一日。
    """
    if tuple(sorted(context.sleeve_ids)) != _ordered_sleeves(virtual_states):
        raise ReconciliationError(
            identity="I1",
            context=context,
            detail=(
                f"上下文袖集合={sorted(context.sleeve_ids)} 与虚拟账户映射键集="
                f"{list(_ordered_sleeves(virtual_states))} 不一致（证据不得标注未校验的袖）"
            ),
        )
    checks = (
        lambda: check_cash_identity(
            physical_cash=physical_cash,
            virtual_states=virtual_states,
            context=context,
        ),
        lambda: check_holdings_identity(
            physical_view=physical_view,
            virtual_states=virtual_states,
            context=context,
            physical_positions=physical_positions,
        ),
        lambda: check_nav_identity(
            virtual_states=virtual_states,
            physical_total_value=physical_total_value,
            physical_initial_capital=physical_initial_capital,
            price_source=price_source,
            context=context,
        ),
    )
    # 三项均为只读校验：失败路径在同一天只读快照上汇总全部破缺（§7.2-5），
    # 汇总后一次抛出并以 RuntimeError 语义终止当日（不提交、不进入下一日）。
    results = list(_run_checks(checks, context))
    return DailyIdentityReport(
        trade_date=context.trade_date,
        host=context.host,
        run_id=context.run_id,
        price_source_id=price_source.source_id,
        sleeve_ids=tuple(context.sleeve_ids),
        results=tuple(results),
        last_settlement_id=context.last_settlement_id,
        last_settlement_source_id=context.last_settlement_source_id,
    )


# ────────────────────────── 结算后回验（T2） ──────────────────────────


def check_settlement_identities(
    *,
    physical_cash: float,
    physical_view: PhysicalLotView,
    virtual_states: Mapping[str, VirtualSleeveState],
    context: IdentityContext,
    physical_positions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[IdentityResult, ...]:
    """内部结算批次应用后的 I1/I2 回验（T1 登记的 T2 义务）。

    只验 I1 与 I2：I3 依赖日终统一估价，归逐日挂载（``run_daily_identities``）。
    两项均为只读；任一破缺抛 ``ReconciliationError``（调用方必须保持**零提交**），
    联动破缺时 ``identities`` 给出全部编号。
    """
    checks = (
        lambda: check_cash_identity(
            physical_cash=physical_cash,
            virtual_states=virtual_states,
            context=context,
        ),
        lambda: check_holdings_identity(
            physical_view=physical_view,
            virtual_states=virtual_states,
            context=context,
            physical_positions=physical_positions,
        ),
    )
    return _run_checks(checks, context)
