"""P6.5 虚拟账本（T1 最小内部转账 + T2 结算批次写点 / FIFO / 守恒守卫）。

契约依据：
- ``docs/contracts/netting_freeze.md`` F25 §4（I1~I3 口径、来源 lot 归一）与
  §5（内部转账零费用、物理来源 lot 谱系 + 虚拟取得批次、FIFO、Q13 费用瀑布、
  Q18 意图上界与余数、全零成交短路）；
- ``docs/contracts/protocols.md`` F13 §3.3/§3.4（虚拟账为运行时概念、
  ``(sleeve_id, 虚拟取得 lot_id)`` 唯一、无借用时逐袖现金非负）；
- 设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md``
  §4.2/§4.3/§6.1/§7/§8.2/§8.3。

模块划分（T2）：
- 本模块 = 账本地基（价源、lot、内部转账、只读快照、守卫）与**唯一写点**；
- ``settlement_batch`` = 结算批次事实、Q18/Q13 两个确定性内核与批次应用编排；
  本模块在文件末尾对全部公开名做重导出，T1 调用方导入路径保持不变；
- 写点语义：结算编排只在候选副本上修改，校验通过后才经
  ``VirtualLedger._commit_settlement`` 一次提交；任何拒绝/破缺都**零提交**。

范围与边界：
- 子袖 ID 只存在于运行时账本，**不进入** ``SleeveSpec``、注册表或配置；
- 虚拟账不得写引擎的 ``current_capital`` / ``positions`` / ``trades``，也不修改
  生产 ``Ledger``（本模块不 import ``ledger``，不持有引擎引用）；
- 物理成交仍由生产 ``Ledger.apply_fills`` 作为唯一入账口；结算批次只消费
  **显式注入的物理事实**（成交/费用/现金），不产生任何市场指令；
- 本模块不实现跨袖借用调度（``FundScheduler``）、公司行为、S2/S3/S5/S7。

定价契约（F25 §5）：内部转账与物理成交都必须使用与物理净额指令**同一天同一
价源**定价——``price_source.as_of == trade_date`` 且成交价/转账价必须等于
``price_source.price_of(ts_code)``；缺价即拒绝、不顺延。

失败语义分层：
- **输入拒绝**（未开户/批次不足/现金不足/错价/跨日价源/日期倒序/身份冲突/
  超意图成交/零成交有费用/费用精确和不成立）⇒ ``VirtualLedgerError``，全部在
  写前完成，**零副作用**；
- **内部守恒破缺**（落账后自检：Σ 现金与物理现金不符、Σ 来源股数漂移、逐袖
  现金转负）⇒ ``VirtualLedgerConservationError``（RuntimeError 语义），并把本
  实例置为**终态**（禁止再次提交；恢复须新建实例）；
- **结算后恒等式破缺** ⇒ ``ReconciliationError``（RuntimeError 语义），同样
  保持零提交并置终态。

幂等身份：``InternalTransfer.settlement_id`` / ``InternalSettlementBatch.batch_id``
是稳定结算键。同键同内容重复提交返回 ``False`` / ``applied=False`` 且不改现金、
lot；同键不同内容直接拒绝（禁止一个身份承载两套事实）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.common.types import (
    Lot,
    Money,
    Shares,
    TradeDate,
    TSCode,
    VirtualAccount,
    VirtualPosition,
)
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLotView

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查（避免与 reconciliation 循环导入）
    from src.lazybull.v2.core.accounting.reconciliation import IdentityContext

#: 本模块唯一阶段标识（进入全部错误上下文）
_STAGE = "P6.5"
#: 落账自检相对容差（float64 累加误差量级；最终以 min(相对, CASH_TOLERANCE) 封顶）
_CONSERVATION_REL_TOLERANCE = 1e-9


class VirtualLedgerError(ValueError):
    """虚拟账本输入非法或结算身份冲突（**零副作用**拒绝）。"""


class VirtualLedgerConservationError(RuntimeError):
    """虚拟账落账后自检发现守恒破缺（实现缺陷/费用分摊未同步守卫）。"""


# ────────────────────────── 基础校验 ──────────────────────────


def _require_positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    """整数校验（bool 不算整数；错误含字段名与实值）。"""
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise VirtualLedgerError(f"{field} 必须为 int，实际 {type(value).__name__}={value!r}")
    number = int(value.__index__())
    if allow_zero:
        if number < 0:
            raise VirtualLedgerError(f"{field} 必须为非负整数，实际 {number}")
        return number
    if number <= 0:
        raise VirtualLedgerError(f"{field} 必须为正整数，实际 {number}")
    return number


def _require_finite_positive(value: Any, field: str) -> float:
    """有限正数校验（价格/费率边界）。"""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise VirtualLedgerError(f"{field} 必须为有限正数，实际 {value!r}")
    return number


def _require_finite_non_negative(value: Any, field: str) -> float:
    """有限非负数校验（费用/现金边界）。"""
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise VirtualLedgerError(f"{field} 必须为有限非负数，实际 {value!r}")
    return number


def _cash_tolerance(scale: float) -> float:
    """落账自检容差：相对容差按量级缩放，但**封顶**到 I1 货币容差。

    T1 复审 P3（T1-B-R2-04）：1e9 量级下相对容差会放宽到约 1.0 元，必须以
    ``min(相对, CASH_TOLERANCE)`` 收敛；容差常量从 reconciliation 单点取用
    （局部导入避免模块级循环依赖）。
    """
    from src.lazybull.v2.core.accounting.reconciliation import CASH_TOLERANCE

    return min(_CONSERVATION_REL_TOLERANCE * max(1.0, abs(scale)), CASH_TOLERANCE)


# ────────────────────────── 价源与 lot ──────────────────────────


@dataclass(frozen=True, slots=True)
class PriceSource:
    """同日同价源（P6.5 合成边界显式输入；所有袖与物理侧共用）。

    ``source_id`` 进入恒等式错误上下文与证据；``as_of`` 必须等于校验交易日
    （跨日错价由 ``reconciliation.check_nav_identity`` 拒绝）。
    """

    as_of: TradeDate
    source_id: str
    prices: Mapping[str, float]

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise VirtualLedgerError("价格源标识 source_id 不得为空")
        snapshot = {
            str(code): _require_finite_positive(price, f"价格源 {self.source_id} 的 {code}")
            for code, price in self.prices.items()
        }
        object.__setattr__(self, "prices", MappingProxyType(snapshot))

    def price_of(self, ts_code: str) -> float:
        """取单只股票价格；缺失即拒绝（P6.5 无价拒绝，不顺延）。"""
        price = self.prices.get(str(ts_code))
        if price is None:
            raise VirtualLedgerError(
                f"价格源 {self.source_id}（{self.as_of}）缺少 {ts_code} 价格；"
                "P6.5 合成边界无价即拒绝，不建立跨日顺延义务"
            )
        return price


@dataclass(frozen=True, slots=True)
class VirtualLot:
    """虚拟取得批次：物理来源谱系 + 虚拟取得信息（F25 / Q14=A）。

    - ``source_lot_id``：物理来源 lot 身份（I2 比较键的一半，本 lot **不再**发明来源）；
    - ``virtual_lot_id``：虚拟取得批次身份，``(sleeve_id, virtual_lot_id)`` 全局唯一；
    - ``acquired_date`` / ``acquired_price`` / ``acquired_fee``：本袖取得该批次的
      日期、价格（内部转账价 / 物理成交价）与该批次分摊到的最终费用。
    """

    virtual_lot_id: str
    ts_code: str
    source_lot_id: str
    shares: int
    acquired_date: TradeDate
    acquired_price: float
    acquired_fee: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.virtual_lot_id).strip():
            raise VirtualLedgerError("虚拟取得 lot_id 不得为空")
        if not str(self.ts_code).strip() or not str(self.source_lot_id).strip():
            raise VirtualLedgerError(
                f"虚拟 lot {self.virtual_lot_id} 的 ts_code/source_lot_id 不得为空"
            )
        _require_positive_int(self.shares, f"虚拟 lot {self.virtual_lot_id} shares")
        _require_finite_positive(
            self.acquired_price, f"虚拟 lot {self.virtual_lot_id} acquired_price"
        )
        fee = _require_finite_non_negative(
            self.acquired_fee, f"虚拟 lot {self.virtual_lot_id} acquired_fee"
        )
        object.__setattr__(self, "acquired_fee", fee)

    @property
    def acquired_cost(self) -> float:
        """虚拟取得经济成本（价 × 股数 + 最终费用；I2 不以成本判来源）。"""
        return self.shares * self.acquired_price + self.acquired_fee


@dataclass(frozen=True, slots=True)
class LotAllocation:
    """内部转账的卖方扣减指令（显式指定虚拟取得批次）。"""

    virtual_lot_id: str
    shares: int

    def __post_init__(self) -> None:
        if not str(self.virtual_lot_id).strip():
            raise VirtualLedgerError("内部转账分配 virtual_lot_id 不得为空")
        _require_positive_int(self.shares, f"内部转账分配 {self.virtual_lot_id} shares")


@dataclass(frozen=True, slots=True)
class InternalTransfer:
    """最小内部转账事实（S1 等量反向的虚拟侧；双方零费用）。

    ``allocations`` 只描述卖方扣减哪些虚拟取得批次、各多少股；跨袖股数即其和。
    批次顺序必须与 FIFO（最早取得优先）一致。
    """

    settlement_id: str
    trade_date: TradeDate
    ts_code: str
    from_sleeve: str
    to_sleeve: str
    price: float
    allocations: Tuple[LotAllocation, ...]

    def __post_init__(self) -> None:
        if not str(self.settlement_id).strip():
            raise VirtualLedgerError("内部结算 settlement_id 不得为空（幂等身份）")
        if not str(self.ts_code).strip():
            raise VirtualLedgerError(f"内部结算 {self.settlement_id} 的 ts_code 不得为空")
        if self.from_sleeve == self.to_sleeve:
            raise VirtualLedgerError(
                f"内部结算 {self.settlement_id} 的转出/转入袖相同（{self.from_sleeve}）"
            )
        _require_finite_positive(self.price, f"内部结算 {self.settlement_id} price")
        if not self.allocations:
            raise VirtualLedgerError(f"内部结算 {self.settlement_id} 缺少分配明细")
        lot_ids = [allocation.virtual_lot_id for allocation in self.allocations]
        if len(set(lot_ids)) != len(lot_ids):
            raise VirtualLedgerError(
                f"内部结算 {self.settlement_id} 的分配明细含重复虚拟取得 lot_id：{lot_ids}"
            )

    @property
    def total_shares(self) -> int:
        """跨袖股数（= Σ 分配股数）。"""
        return sum(allocation.shares for allocation in self.allocations)

    def fingerprint(self) -> Tuple[Any, ...]:
        """稳定内容指纹（幂等键冲突判定；浮点用 repr 保留位模式）。"""
        return (
            str(self.trade_date),
            self.ts_code,
            self.from_sleeve,
            self.to_sleeve,
            repr(self.price),
            tuple((a.virtual_lot_id, a.shares) for a in self.allocations),
        )


@dataclass(frozen=True, slots=True)
class VirtualSleeveState:
    """单袖虚拟账只读快照（I1/I2/I3 的输入形态）。"""

    sleeve_id: str
    cash: float
    borrowed_credit: float
    lots: Tuple[VirtualLot, ...]

    def __post_init__(self) -> None:
        """类型守卫：I2/I3 不允许浮点股数或非有限现金混入（防「恒等式被浮点化」）。"""
        if not str(self.sleeve_id).strip():
            raise VirtualLedgerError("虚拟袖快照 sleeve_id 不得为空")
        for field_name in ("cash", "borrowed_credit"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise VirtualLedgerError(
                    f"袖 {self.sleeve_id} 的 {field_name} 必须为有限数，实际 {value!r}"
                )
            # 构造期归一为 float（T1 复审 P3：校验不归一 → 同值异型）
            object.__setattr__(self, field_name, value)
        for lot in self.lots:
            if not isinstance(lot, VirtualLot):
                raise VirtualLedgerError(
                    f"袖 {self.sleeve_id} 的 lots 含非 VirtualLot 元素：{type(lot).__name__}"
                )

    def shares_of(self, ts_code: str) -> int:
        """该袖在某股票上的虚拟股数。"""
        return sum(lot.shares for lot in self.lots if lot.ts_code == str(ts_code))

    def shares_by_source(self) -> Dict[Tuple[str, str], int]:
        """按 ``(ts_code, 物理来源 lot_id)`` 汇总的虚拟整数股数（I2 左值）。"""
        aggregated: Dict[Tuple[str, str], int] = {}
        for lot in self.lots:
            key = (lot.ts_code, lot.source_lot_id)
            aggregated[key] = aggregated.get(key, 0) + lot.shares
        return aggregated

    def total_value(self, price_source: PriceSource) -> float:
        """货币总值 = 现金 + Σ 股数 × 同日同价源价格（I3 的 total_value_i）。"""
        market_value = 0.0
        for lot in self.lots:
            market_value += lot.shares * price_source.price_of(lot.ts_code)
        return self.cash + market_value


class _SleeveBook:
    """单袖可变账（内部结构；外部只经 ``VirtualSleeveState`` 读取）。"""

    __slots__ = ("cash", "borrowed_credit", "lots")

    def __init__(
        self, cash: float, lots: Sequence[VirtualLot], borrowed_credit: float = 0.0
    ) -> None:
        self.cash = float(cash)
        # 借用占用必须随账本复制（A-R1-03）：候选副本不得静默清零该字段，
        # 否则 reconciliation 的逐袖「borrowed_credit == 0」守卫在结算路径不可达。
        self.borrowed_credit = float(borrowed_credit)
        self.lots: List[VirtualLot] = list(lots)


class VirtualLedger:
    """虚拟账本（P6.5 T1 最小转账 + T2 内部结算批次；纯运行时，不落盘）。

    用法：``open_account`` 建期初账 → ``transfer`` 记内部转账（S1）或
    ``apply_settlement`` 记结算批次（S1/S4）→ ``snapshot`` 供 ``reconciliation``
    逐日校验。账本不接收引擎或 ``Ledger`` 引用，物理入账仍由生产
    ``Ledger.apply_fills`` 唯一负责。
    """

    def __init__(
        self,
        physical_lot_view: Optional[PhysicalLotView] = None,
        *,
        cost_model: Optional[Any] = None,
    ) -> None:
        self._books: Dict[str, _SleeveBook] = {}
        self._settlements: Dict[str, Tuple[Any, ...]] = {}
        self._batches: Dict[str, Tuple[Tuple[Any, ...], "SettlementResult"]] = {}
        self._last_settlement_id: Optional[str] = None
        self._last_settlement_source_id: Optional[str] = None
        # 可选：同一结算事件上同步校验物理 lot 伴随视图（S1 物理侧零变动，只校验）
        self._physical_lot_view = physical_lot_view
        #: 买入成本模型（``calculate_buy_cost``）；合成调用方必须显式注入合成费率
        self._cost_model = cost_model
        #: 守恒破缺后的实例终态（禁止再次提交；恢复须新建实例）
        self._poisoned: Optional[str] = None

    # ────────────────────────── 期初 ──────────────────────────

    def open_account(
        self,
        sleeve_id: str,
        cash: float,
        lots: Sequence[VirtualLot] = (),
    ) -> None:
        """建立纯运行时子袖期初账（重复开户、负现金、重复 lot_id 均拒绝）。"""
        self._assert_writable()
        if not str(sleeve_id).strip():
            raise VirtualLedgerError("袖 ID 不得为空")
        if sleeve_id in self._books:
            raise VirtualLedgerError(f"虚拟账本重复开户拒绝：袖 {sleeve_id!r} 已存在")
        cash_value = float(cash)
        if not math.isfinite(cash_value) or cash_value < 0:
            raise VirtualLedgerError(
                f"期初虚拟现金必须为有限非负数，实际 {cash!r}（袖 {sleeve_id}）"
            )
        seen: set = set()
        for lot in lots:
            if lot.virtual_lot_id in seen:
                raise VirtualLedgerError(
                    f"袖 {sleeve_id} 期初虚拟取得 lot_id 重复：{lot.virtual_lot_id}"
                )
            seen.add(lot.virtual_lot_id)
        self._books[sleeve_id] = _SleeveBook(cash=cash_value, lots=lots)

    # ────────────────────────── 内部转账（S1） ──────────────────────────

    def transfer(self, transfer: InternalTransfer, *, price_source: PriceSource) -> bool:
        """记一笔最小内部转账（跨袖股数 + 现金，双方零费用）。

        Args:
            transfer: 内部结算事实；``allocations`` 必须与 FIFO（最早取得优先）
                的实际消耗批次一致
            price_source: **当日**同一价源（必须满足 ``as_of == transfer.trade_date``，
                且其 ``price_of(ts_code)`` 与 ``transfer.price`` 完全相等）

        Returns:
            True = 本次首次应用；False = 同身份同内容重复提交（**状态不变**）。

        Raises:
            VirtualLedgerError: 错价/跨日价源/缺价/日期倒序/袖未开户/分配批次不存在或
                股数不足/FIFO 顺序不符/转入袖现金不足（P6.5 无借用）/同
                ``settlement_id`` 内容不同。上述全部在写前完成，**零副作用**。
            PhysicalLotViewError: 伴随视图校验失败（来源批次不存在/股数不足；
                本方法不改聚合）。
            VirtualLedgerConservationError: 落账后自检发现守恒破缺（实现缺陷语义；
                实例随即置终态）。
        """
        self._assert_writable()
        self._assert_same_day_price(transfer, price_source)
        fingerprint = transfer.fingerprint()
        recorded = self._settlements.get(transfer.settlement_id)
        if recorded is not None:
            if recorded == fingerprint:
                return False
            raise VirtualLedgerError(
                f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                "内部结算幂等键冲突（已登记但内容不同）；禁止同一身份承载不同结算事实"
            )
        if transfer.settlement_id in self._batches:
            raise VirtualLedgerError(
                f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                "该身份已被内部结算批次占用；同一结算键不得承载两套事实（转账/批次）"
            )
        seller = self._require_book(transfer.from_sleeve, "from_sleeve")
        buyer = self._require_book(transfer.to_sleeve, "to_sleeve")
        consumed = self._resolve_allocations(seller, transfer)
        self._assert_fifo_order(transfer, consumed)
        self._assert_lot_dates(transfer, consumed)
        amount = transfer.total_shares * transfer.price
        if buyer.cash - amount < 0.0:
            raise VirtualLedgerError(
                f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                f"虚拟买入现金不足（无借用）：袖 {transfer.to_sleeve} 现金 "
                f"{buyer.cash!r} < 应付 {amount!r}"
            )
        self._assert_acquisition_ids_free(buyer, transfer)
        if self._physical_lot_view is not None:
            self._physical_lot_view.validate_transfer(
                ts_code=transfer.ts_code,
                allocations=[(lot.source_lot_id, shares) for lot, shares in consumed],
            )
        cash_before = self._cash_total()
        shares_before = self._source_share_totals()
        self._apply_transfer(transfer, seller, buyer, consumed, amount)
        try:
            self._assert_cash_total(
                context=self._context_label(transfer),
                expected=cash_before,
                phrase="转账后 Σ 虚拟现金漂移",
            )
            self._assert_source_shares(
                context=self._context_label(transfer), expected=shares_before
            )
            self._assert_no_negative_cash(context=self._context_label(transfer))
        except VirtualLedgerConservationError as error:
            self._poison(str(error))
            raise
        self._settlements[transfer.settlement_id] = fingerprint
        self._last_settlement_id = transfer.settlement_id
        self._last_settlement_source_id = price_source.source_id
        return True

    @staticmethod
    def _assert_same_day_price(transfer: InternalTransfer, price_source: PriceSource) -> None:
        """F25 §5 定价绑定：同日同价源 + 转账价完全相等（缺价即拒绝，不顺延）。"""
        if price_source.as_of != transfer.trade_date:
            raise VirtualLedgerError(
                f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                f"价格源交易日 {price_source.as_of} 与结算日不一致（必须同日同价源）"
            )
        try:
            source_price = price_source.price_of(transfer.ts_code)
        except VirtualLedgerError as error:
            raise VirtualLedgerError(
                f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}：{error}"
            ) from error
        if transfer.price != source_price:
            raise VirtualLedgerError(
                f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                f"转账价 {transfer.price!r} != 价源 {price_source.source_id} 的 "
                f"{transfer.ts_code} 价格 {source_price!r}（F25 §5 定价契约）"
            )

    @staticmethod
    def _assert_lot_dates(
        transfer: InternalTransfer, consumed: Sequence[Tuple[VirtualLot, int]]
    ) -> None:
        """lot 谱系日期序：不得从晚于结算日的批次转出（否则 FIFO 顺序失真）。"""
        for lot, _shares in consumed:
            if lot.acquired_date > transfer.trade_date:
                raise VirtualLedgerError(
                    f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                    f"批次 {lot.virtual_lot_id} 的取得日 {lot.acquired_date} 晚于结算日"
                )

    def _resolve_allocations(
        self, seller: _SleeveBook, transfer: InternalTransfer
    ) -> List[Tuple[VirtualLot, int]]:
        """解析并校验卖方扣减批次（不修改状态）。"""
        by_id = {lot.virtual_lot_id: lot for lot in seller.lots}
        consumed: List[Tuple[VirtualLot, int]] = []
        for allocation in transfer.allocations:
            lot = by_id.get(allocation.virtual_lot_id)
            if lot is None:
                raise VirtualLedgerError(
                    f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                    f"转出袖 {transfer.from_sleeve} 不存在虚拟取得批次 "
                    f"{allocation.virtual_lot_id}"
                )
            if lot.ts_code != transfer.ts_code:
                raise VirtualLedgerError(
                    f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                    f"批次 {lot.virtual_lot_id} 属于 {lot.ts_code}，与结算股票 "
                    f"{transfer.ts_code} 不一致"
                )
            if allocation.shares > lot.shares:
                raise VirtualLedgerError(
                    f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                    f"超出批次持仓：{lot.virtual_lot_id} 持有 {lot.shares} 股，申请转出 "
                    f"{allocation.shares} 股（差={allocation.shares - lot.shares} 股）"
                )
            consumed.append((lot, allocation.shares))
        return consumed

    @staticmethod
    def _fifo_lots(book: _SleeveBook, ts_code: str) -> List[VirtualLot]:
        """单袖某股票的 FIFO 批次序：``(取得日升序, 登记序升序)``。"""
        indexed = [
            (index, lot) for index, lot in enumerate(book.lots) if lot.ts_code == str(ts_code)
        ]
        indexed.sort(key=lambda item: (item[1].acquired_date, item[0]))
        return [lot for _index, lot in indexed]

    def fifo_allocations(
        self, sleeve_id: str, ts_code: str, shares: int
    ) -> Tuple[LotAllocation, ...]:
        """按 FIFO 生成卖方扣减批次（S1 结算/测试共用；股数不足即拒绝）。"""
        book = self._require_book(sleeve_id, "sleeve_id")
        return self._plan_fifo_allocations(book, sleeve_id, ts_code, shares)

    def _plan_fifo_allocations(
        self,
        book: _SleeveBook,
        sleeve_id: str,
        ts_code: str,
        shares: int,
    ) -> Tuple[LotAllocation, ...]:
        """FIFO 扣减计划唯一实现（提交态与候选态共用，避免两处规则漂移）。"""
        wanted = _require_positive_int(shares, f"袖 {sleeve_id} FIFO 转出股数")
        allocations: List[LotAllocation] = []
        remaining = wanted
        for lot in self._fifo_lots(book, ts_code):
            if remaining == 0:
                break
            take = min(lot.shares, remaining)
            allocations.append(LotAllocation(lot.virtual_lot_id, take))
            remaining -= take
        if remaining != 0:
            held = sum(lot.shares for lot in book.lots if lot.ts_code == str(ts_code))
            raise VirtualLedgerError(
                f"阶段={_STAGE} 袖={sleeve_id} 股票={ts_code}：FIFO 批次不足——"
                f"申请转出 {wanted} 股，可用 {held} 股，仍缺 {remaining} 股；"
                f"批次={[lot.virtual_lot_id for lot in book.lots if lot.ts_code == str(ts_code)]}"
            )
        return tuple(allocations)

    def _assert_fifo_order(
        self,
        transfer: InternalTransfer,
        consumed: Sequence[Tuple[VirtualLot, int]],
    ) -> None:
        """逐袖断言实际消耗批次与 FIFO 计划一致（不能用组合总量守恒替代）。"""
        expected = {
            allocation.virtual_lot_id: allocation.shares
            for allocation in self.fifo_allocations(
                transfer.from_sleeve, transfer.ts_code, transfer.total_shares
            )
        }
        actual = {lot.virtual_lot_id: shares for lot, shares in consumed}
        if expected != actual:
            raise VirtualLedgerError(
                f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}："
                f"袖 {transfer.from_sleeve} 的消耗批次不符合 FIFO（按取得日/批次顺序）；"
                f"期望={sorted(expected.items())} 实际={sorted(actual.items())} "
                f"容差=0 股 单位=股"
            )

    @staticmethod
    def _assert_acquisition_ids_free(buyer: _SleeveBook, transfer: InternalTransfer) -> None:
        """虚拟取得批次身份写入前置：``(sleeve_id, virtual_lot_id)`` 不得覆盖既有条目。"""
        existing = {lot.virtual_lot_id for lot in buyer.lots}
        for index in range(len(transfer.allocations)):
            candidate = f"{transfer.settlement_id}#{index}"
            if candidate in existing:
                raise VirtualLedgerError(
                    f"虚拟取得批次身份冲突：袖 {transfer.to_sleeve} 已存在 {candidate}；"
                    "禁止用同一身份覆盖既有取得批次"
                )

    @staticmethod
    def _apply_transfer(
        transfer: InternalTransfer,
        seller: _SleeveBook,
        buyer: _SleeveBook,
        consumed: Sequence[Tuple[VirtualLot, int]],
        amount: float,
    ) -> None:
        """落账：卖方扣减批次并收款，买方按同一物理来源新建取得批次并付款。

        部分消耗时按股数比例拆分该批次的取得费用（C65-T2-03）：剩余批次只保留对应
        份额，禁止保留整笔原手续费；浮点尾差（≤1ulp）归被消耗部分，剩余成本严格为
        ``acquired_fee × 剩余股数 / 原股数``。内部转账本身不收费，买方新批次费用为 0。
        """
        deducted = {lot.virtual_lot_id: shares for lot, shares in consumed}
        remaining: List[VirtualLot] = []
        for lot in seller.lots:
            shares = deducted.get(lot.virtual_lot_id)
            if shares is None:
                remaining.append(lot)
            elif lot.shares > shares:
                kept = lot.shares - shares
                kept_fee = lot.acquired_fee * kept / lot.shares
                remaining.append(replace(lot, shares=kept, acquired_fee=kept_fee))
        seller.lots = remaining
        seller.cash += amount
        buyer.cash -= amount
        for index, (lot, shares) in enumerate(consumed):
            buyer.lots.append(
                VirtualLot(
                    virtual_lot_id=f"{transfer.settlement_id}#{index}",
                    ts_code=lot.ts_code,
                    source_lot_id=lot.source_lot_id,
                    shares=shares,
                    acquired_date=transfer.trade_date,
                    acquired_price=transfer.price,
                )
            )

    # ────────────────────────── 内部结算批次（T2） ──────────────────────────

    def apply_settlement(
        self,
        batch: "InternalSettlementBatch",
        *,
        price_source: PriceSource,
        physical_view: PhysicalLotView,
        identity_context: "IdentityContext",
        physical_positions: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> "SettlementResult":
        """应用内部结算批次（S1 内部转账 + S4 合并买入分摊），并回验 I1/I2。

        实现委托 ``settlement_batch.apply_settlement``（同包内部写点）；物理事实
        必须由验收器显式注入：本方法不产生市场指令、不写生产账本。任何拒绝或
        破缺都保持**零提交**，且守恒/恒等式破缺会把本实例置为终态。
        """
        from src.lazybull.v2.core.accounting.settlement_batch import (
            apply_settlement as _apply_settlement,
        )

        return _apply_settlement(
            self,
            batch,
            price_source=price_source,
            physical_view=physical_view,
            identity_context=identity_context,
            physical_positions=physical_positions,
        )

    def _commit_settlement(
        self,
        *,
        batch: "InternalSettlementBatch",
        fingerprint: Tuple[Any, ...],
        books: Dict[str, _SleeveBook],
        result: "SettlementResult",
        source_id: str,
    ) -> None:
        """结算提交点（唯一写点）：候选账、批次身份与最近结算标识一次落定。

        派生转账子键（``<batch>#T<i>``）进入同一幂等键空间：外部若以同键转账，
        因指纹不同必然命中"幂等键冲突"，不得用派生键冒充实盘转账身份（A-R3-02）。
        反向（外部先占派生键）由 ``settlement_batch._assert_derived_ids_free`` 在
        应用前拦截；此处仍做防御性复检。**复检必须先于任何写入**（T2-P3-1）：
        否则防御分支触发时会出现"拒绝但已换账/已登记批次"的半提交。
        """
        if batch.batch_id in self._settlements:
            raise VirtualLedgerError(
                f"阶段={_STAGE} 批次={batch.batch_id}：该身份已被内部转账占用；"
                "同一结算键不得承载两套事实（转账/批次）"
            )
        marker = ("__batch_transfer__", batch.batch_id)
        derived: List[str] = []
        for transfer in result.transfers:
            recorded = self._settlements.get(transfer.settlement_id)
            if recorded is not None and recorded != marker:
                raise VirtualLedgerError(
                    f"阶段={_STAGE} 批次={batch.batch_id}：派生转账子键 "
                    f"{transfer.settlement_id} 已被既有身份占用；禁止静默覆盖幂等指纹"
                )
            if transfer.settlement_id in derived:
                raise VirtualLedgerError(
                    f"阶段={_STAGE} 批次={batch.batch_id}：派生转账子键 "
                    f"{transfer.settlement_id} 在本次结算结果中重复；禁止同键承载两笔转账"
                )
            derived.append(transfer.settlement_id)
        self._books = books
        self._batches[batch.batch_id] = (fingerprint, result)
        for settlement_id in derived:
            self._settlements[settlement_id] = marker
        self._last_settlement_id = batch.batch_id
        self._last_settlement_source_id = source_id

    # ────────────────────────── 读取 ──────────────────────────

    def sleeve_ids(self) -> Tuple[str, ...]:
        """已开户袖 ID（开盘顺序，确定性）。"""
        return tuple(self._books)

    def state_of(self, sleeve_id: str) -> VirtualSleeveState:
        """单袖只读快照。"""
        book = self._require_book(sleeve_id, "sleeve_id")
        return self._state_from(sleeve_id, book)

    def snapshot(self) -> Mapping[str, VirtualSleeveState]:
        """全袖只读快照（插入顺序，供逐日恒等式校验）。"""
        return MappingProxyType({sid: self.state_of(sid) for sid in self._books})

    @property
    def last_settlement_id(self) -> Optional[str]:
        """最近一次内部结算身份（进入错误上下文与证据）。"""
        return self._last_settlement_id

    @property
    def last_settlement_source_id(self) -> Optional[str]:
        """最近一次内部结算所用价格源标识（定价契约证据）。"""
        return self._last_settlement_source_id

    @property
    def poisoned_reason(self) -> Optional[str]:
        """实例终态原因（None = 可继续写入）；终态实例只能新建恢复。"""
        return self._poisoned

    @property
    def cost_model(self) -> Optional[Any]:
        """买入成本模型（只读；T3 预演副本复制同一费率，禁止第二套成本口径）。"""
        return self._cost_model

    def batch_ids(self) -> Tuple[str, ...]:
        """已应用的结算批次身份序列（重放核对用）。"""
        return tuple(self._batches)

    def settlement_ids(self) -> Tuple[str, ...]:
        """已应用的内部转账身份序列。

        包含两类键：① 直接 ``transfer`` 的 ``settlement_id``；② 结算批次内派生转账键
        ``"<batch_id>#T<i>"``（B-R4-02：派生键同样占用幂等键空间，禁止外部复用）。
        """
        return tuple(self._settlements)

    def as_virtual_accounts(self, price_source: PriceSource) -> Mapping[str, VirtualAccount]:
        """协议值对象投影（``VirtualAccount``），供 T2 净额接口上下文兼容。

        注意：``VirtualAccount``/``Lot`` 不携带物理来源 lot_id，**不能**作为 I2 输入；
        I2 只以 ``VirtualSleeveState`` 为准（F25）。
        """
        accounts: Dict[str, VirtualAccount] = {}
        for sleeve_id in self._books:
            state = self.state_of(sleeve_id)
            grouped: Dict[str, List[Lot]] = {}
            for lot in sorted(
                state.lots, key=lambda item: (item.ts_code, item.acquired_date, item.virtual_lot_id)
            ):
                grouped.setdefault(lot.ts_code, []).append(
                    Lot(
                        shares=Shares(lot.shares),
                        buy_date=lot.acquired_date,
                        cost_basis=Money(lot.acquired_cost),
                    )
                )
            positions = {
                TSCode(code): VirtualPosition(ts_code=TSCode(code), lots=tuple(lots))
                for code, lots in grouped.items()
            }
            accounts[sleeve_id] = VirtualAccount(
                sleeve_id=sleeve_id,
                cash=Money(state.cash),
                positions=MappingProxyType(positions),
                total_value=Money(state.total_value(price_source)),
                borrowed_credit=Money(state.borrowed_credit),
            )
        return MappingProxyType(accounts)

    # ────────────────────────── 内部工具 ──────────────────────────

    def _require_book(self, sleeve_id: str, field: str) -> _SleeveBook:
        book = self._books.get(sleeve_id)
        if book is None:
            raise VirtualLedgerError(
                f"{field}={sleeve_id!r} 未在虚拟账本开户；已开户袖={sorted(self._books)}"
            )
        return book

    def _assert_writable(self) -> None:
        """终态守卫：守恒/恒等式破缺后禁止任何再次提交。"""
        if self._poisoned is not None:
            raise VirtualLedgerError(
                f"阶段={_STAGE}：虚拟账本实例已因守恒/恒等式破缺进入终态，禁止再次提交；"
                f"原始破缺={self._poisoned}；恢复必须新建实例并重放"
            )

    def _poison(self, reason: str) -> None:
        """登记实例终态（首次破缺即锁定；不覆盖首因）。"""
        if self._poisoned is None:
            self._poisoned = reason

    @staticmethod
    def _context_label(transfer: InternalTransfer) -> str:
        """内部转账错误上下文（阶段/结算/日期）。"""
        return f"阶段={_STAGE} 结算={transfer.settlement_id} 日期={transfer.trade_date}"

    def _copy_books(self) -> Dict[str, _SleeveBook]:
        """候选账副本（提交前一切修改只落在副本上；借用占用一并复制）。"""
        return {
            sleeve_id: _SleeveBook(book.cash, list(book.lots), book.borrowed_credit)
            for sleeve_id, book in self._books.items()
        }

    def preflight_clone(self) -> "VirtualLedger":
        """预演副本（T3）：账本内容 + **幂等身份 + 终态** 一并镜像。

        ``assert_settleable`` 用它跑一次真实结算，因此副本必须与真账本同样
        "记得"已占用的 ``batch_id``/``settlement_id``（含派生键）、最近结算标识与
        守恒破缺终态；否则预演会放行真账本必然拒绝的批次，导致"物理已成交、
        随后才拒绝"的零副作用破坏。本方法只读创建，不修改本实例。
        """
        clone = VirtualLedger(cost_model=self._cost_model)
        clone._books = self._copy_books()
        clone._settlements = dict(self._settlements)
        clone._batches = dict(self._batches)
        clone._last_settlement_id = self._last_settlement_id
        clone._last_settlement_source_id = self._last_settlement_source_id
        clone._poisoned = self._poisoned
        return clone

    def _cash_total(self, books: Optional[Mapping[str, _SleeveBook]] = None) -> float:
        """全袖现金合计（落账守恒自检用；按袖 ID 字典序确定性求和）。"""
        source = self._books if books is None else books
        total = 0.0
        for sleeve_id in sorted(source):
            total += source[sleeve_id].cash
        return total

    def _source_share_totals(
        self, books: Optional[Mapping[str, _SleeveBook]] = None
    ) -> Dict[Tuple[str, str], int]:
        """全账本按来源键的整数股合计（落账守恒自检用）。"""
        source = self._books if books is None else books
        totals: Dict[Tuple[str, str], int] = {}
        for sleeve_id in sorted(source):
            for lot in source[sleeve_id].lots:
                key = (lot.ts_code, lot.source_lot_id)
                totals[key] = totals.get(key, 0) + lot.shares
        return totals

    def _states_of(self, books: Mapping[str, _SleeveBook]) -> Dict[str, VirtualSleeveState]:
        """候选账 → 只读状态快照映射。"""
        return {sleeve_id: self._state_from(sleeve_id, books[sleeve_id]) for sleeve_id in books}

    @staticmethod
    def _state_from(sleeve_id: str, book: _SleeveBook) -> VirtualSleeveState:
        """单账 → 只读状态快照。"""
        return VirtualSleeveState(
            sleeve_id=sleeve_id,
            cash=book.cash,
            borrowed_credit=book.borrowed_credit,
            lots=tuple(book.lots),
        )

    def _assert_cash_total(
        self,
        *,
        context: str,
        expected: float,
        books: Optional[Mapping[str, _SleeveBook]] = None,
        phrase: str = "Σ 虚拟现金与期望不一致",
    ) -> None:
        """Σ 虚拟现金与期望值一致性守卫（费用存在时期望值 = 物理现金）。"""
        actual = self._cash_total(books)
        tolerance = _cash_tolerance(max(abs(actual), abs(expected)))
        if abs(actual - expected) > tolerance:
            raise VirtualLedgerConservationError(
                f"{context}：{phrase}——Σ 虚拟现金 {actual!r} 元 != 期望 {expected!r} 元"
                f"（差={actual - expected!r} 元 容差={tolerance} 元）；"
                "费用分摊必须让 Σ 虚拟现金等于物理现金，禁止游离费用"
            )

    def _assert_source_shares(
        self,
        *,
        context: str,
        expected: Mapping[Tuple[str, str], int],
        books: Optional[Mapping[str, _SleeveBook]] = None,
    ) -> None:
        """来源股数守恒守卫（整数零差；物理买入按同批注入的来源批次计入）。"""
        actual = self._source_share_totals(books)
        if actual != expected:
            changed = {
                key: (expected.get(key, 0), actual.get(key, 0))
                for key in set(expected) | set(actual)
                if expected.get(key, 0) != actual.get(key, 0)
            }
            raise VirtualLedgerConservationError(
                f"{context}：来源股数漂移 {changed}（左=期望 右=实际，单位=股，"
                "必须整数守恒）；容差=0 股"
            )

    def _assert_no_negative_cash(
        self, *, context: str, books: Optional[Mapping[str, _SleeveBook]] = None
    ) -> None:
        """逐袖现金非负守卫（无借用模式；组合现金非负不能替代）。

        口径登记（A-R4-05）：本守卫是**0 容差**的保守实现守卫，比 I1 的 1e-6 元门更严；
        贴边浮点噪声（如 ``-1.4e-14``）也会被判破缺并置终态。这是有意的 fail-closed
        选择——不得由实现擅自放宽到 I1 容差；若契约给出逐袖阈值，须先改契约定值。
        """
        source = self._books if books is None else books
        for sleeve_id in sorted(source):
            cash = source[sleeve_id].cash
            if cash < 0.0:
                raise VirtualLedgerConservationError(
                    f"{context}：落账后袖 {sleeve_id} 现金为负 {cash!r} 元（无借用模式）；"
                    "组合现金非负不能替代逐袖守卫"
                )

    def _buy_cost(self, amount: float) -> float:
        """买入成本模型的唯一入口（**必须显式注入**，fail-closed）。

        A-R2-06：此前缺省会静默读取生产费率（fail-open），与"合成边界不读真实配置"
        冲突；现改为缺模型即拒绝，生产/合成调用方都必须显式注入
        （生产可传 ``get_default_cost_model()``）。
        """
        if self._cost_model is None:
            raise VirtualLedgerError(
                f"阶段={_STAGE}：费用分摊需要显式注入买入成本模型（cost_model），"
                "禁止在合成边界静默读取生产费率配置"
            )
        return float(self._cost_model.calculate_buy_cost(amount))


# ────────────────────────── T2 公开名重导出（T1 导入路径不变） ──────────────────────────

from src.lazybull.v2.core.accounting.settlement_batch import (  # noqa: E402
    FeeWaterfall,
    InternalSettlementBatch,
    PhysicalFillFact,
    SettlementResult,
    ShareAllocation,
    SleeveIntent,
    TransferFact,
    allocate_fees,
    allocate_shares,
)

__all__ = [
    "FeeWaterfall",
    "InternalSettlementBatch",
    "InternalTransfer",
    "LotAllocation",
    "PhysicalFillFact",
    "PriceSource",
    "SettlementResult",
    "ShareAllocation",
    "SleeveIntent",
    "TransferFact",
    "VirtualLedger",
    "VirtualLedgerConservationError",
    "VirtualLedgerError",
    "VirtualLot",
    "VirtualSleeveState",
    "allocate_fees",
    "allocate_shares",
]
