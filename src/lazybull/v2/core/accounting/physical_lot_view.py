"""P6.5 物理 lot 伴随视图（I2 右值；**不是**生产物理账）。

契约依据：``docs/contracts/netting_freeze.md`` F25 §4（I2 按
``(ts_code, 物理来源 lot_id)`` 汇总整数股数、键集一致、不比条目数）与
``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §4.3/§8.2（Q3=C）。

范围与边界：
- 以**显式期初 lots** 建立，并与引擎现有聚合持仓字典（``positions``）先对账；
- 同一结算事件（内部转账）只做**来源批次存在性与股数校验**——S1 物理侧零变动，
  视图聚合不变；
- T2 起支持**物理合成买入**的同事件维护（``apply_buy_fill``）：S4 结算后把成交
  追加为来源批次。物理事实仍由验收器显式注入，本视图只登记；唯一成交入账口
  仍是 T3 的 ``Ledger.apply_fills``（本模块不 import 也不修改生产账本）；
- 视图是只读伴随件：不写生产 ``Ledger``、不回写引擎持仓、不构成第二套可成交账；
- I2 证据等级 = **合成伴随视图**，不得表述为“已与生产物理账逐 lot 对账”。

T5 扩展（**不改变 I2 口径**）：来源批次除股数外还携带**成本**（``cost``，买入
金额 + 费用），并保留**批次顺序**（登记序）、**事件水位**（``apply_buy_fill`` 的
``event_id`` 序列）与**初始冻结输入身份**（构造输入的规范化指纹）。这些字段只
供独立状态信封同代落盘/恢复使用；I2 仍只按 ``(ts_code, 来源 lot_id)`` 汇总**整数
股数**比较，成本与顺序不参与 I2，也不得从虚拟侧反推。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


class PhysicalLotViewError(ValueError):
    """伴随视图构造/结算校验输入非法（零副作用拒绝）。"""


class PhysicalLotMismatchError(RuntimeError):
    """伴随视图与物理聚合字典不一致（恒等式破缺语义：报错停止，不降级为告警）。"""


def _strict_integer_shares(raw: Any, stock: str, context: str) -> int:
    """物理聚合股数严格整数校验：拒绝 float/bool/NaN/Inf，禁止 ``int()`` 静默截断。"""
    if isinstance(raw, bool) or isinstance(raw, float) or not hasattr(raw, "__index__"):
        raise PhysicalLotMismatchError(
            f"I2 右值前置破缺：股票 {stock} shares={raw!r} 非整数股（类型 "
            f"{type(raw).__name__}）；差=<未比较> 容差=0 股 单位=股｜{context}"
        )
    try:
        value = int(raw.__index__())
    except (TypeError, ValueError, OverflowError) as error:
        raise PhysicalLotMismatchError(
            f"I2 右值前置破缺：股票 {stock} shares={raw!r} 无法转为整数（{error}）；"
            f"容差=0 股 单位=股｜{context}"
        ) from error
    if value < 0:
        raise PhysicalLotMismatchError(
            f"I2 右值前置破缺：股票 {stock} shares={value} 为负；"
            f"差=<未比较> 容差=0 股 单位=股｜{context}"
        )
    return value


def _require_finite_non_negative_cost(raw: Any, lot_id: str) -> float:
    """来源批次成本校验（有限非负；成本不参与 I2，但必须可逐位落盘）。"""
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        raise PhysicalLotViewError(f"物理 lot {lot_id} cost 必须为有限数，实际 {raw!r}")
    if value < 0.0:
        raise PhysicalLotViewError(f"物理 lot {lot_id} cost 必须非负，实际 {raw!r}")
    return value


@dataclass(frozen=True, slots=True)
class PhysicalLot:
    """合成物理来源 lot（显式期初输入）。

    ``cost`` 是该来源批次的实际买入成本（金额 + 费用，元）。缺省 0.0 保持既有
    调用方不变；I2 不以成本判来源（只比整数股数），成本仅供状态信封同代恢复。
    """

    ts_code: str
    source_lot_id: str
    shares: int
    cost: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.ts_code).strip() or not str(self.source_lot_id).strip():
            raise PhysicalLotViewError("物理 lot 的 ts_code/source_lot_id 不得为空")
        if isinstance(self.shares, bool) or not isinstance(self.shares, int):
            raise PhysicalLotViewError(
                f"物理 lot {self.source_lot_id} shares 必须为 int，实际 {self.shares!r}"
            )
        if self.shares <= 0:
            raise PhysicalLotViewError(
                f"物理 lot {self.source_lot_id} shares 必须为正整数，实际 {self.shares}"
            )
        object.__setattr__(
            self, "cost", _require_finite_non_negative_cost(self.cost, self.source_lot_id)
        )


def lot_order_fingerprint(order: Sequence[Tuple[str, str]]) -> str:
    """批次顺序指纹（``(ts_code, 来源 lot_id)`` 登记序的规范化 SHA-256）。"""
    payload = "\n".join(f"{ts_code}|{source_lot_id}" for ts_code, source_lot_id in order)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PhysicalLotView:
    """物理来源 lot 伴随视图（只读；I2 右值）。

    合成买入事件（S4）通过 ``apply_buy_fill`` 与结算同事件维护：同一 ``event_id``
    同内容重复注入不改聚合；同键异内容拒绝（禁止一个物理事件承载两套事实）。

    T5 附加状态（同代落盘/恢复用，不改变 I2 口径）：``_costs``（来源批次成本）、
    ``_order``（批次登记顺序）、``_events``（合成买入事件水位）、``_origin``
    （**初始冻结输入身份** = 构造输入的规范化指纹）。
    """

    def __init__(self, lots: Iterable[PhysicalLot] = ()) -> None:
        self._lots: Dict[Tuple[str, str], int] = {}
        self._costs: Dict[Tuple[str, str], float] = {}
        self._order: List[Tuple[str, str]] = []
        self._events: Dict[str, Tuple[Any, ...]] = {}
        for lot in lots:
            key = (lot.ts_code, lot.source_lot_id)
            if key in self._lots:
                raise PhysicalLotViewError(
                    f"物理 lot 伴随视图重复来源键：股票 {lot.ts_code} 来源 "
                    f"{lot.source_lot_id}"
                )
            self._lots[key] = lot.shares
            self._costs[key] = float(lot.cost)
            self._order.append(key)
        # 初始冻结输入身份：只由构造输入决定，之后的结算/成交不得改写
        self._origin = lot_order_fingerprint(self._order)

    def aggregate(self) -> Mapping[Tuple[str, str], int]:
        """``(ts_code, 物理来源 lot_id) → 整数股数``（I2 右值原样）。"""
        return MappingProxyType(dict(self._lots))

    def copy(self) -> "PhysicalLotView":
        """同态副本（T3 全链零副作用用法：先构造候选视图，结算成功后再替换）。

        结算是最后才提交的：物理执行前必须先证明"成交 + 结算"整体可通过。副本
        让调用方在同事件上预演（注册成交/校验），任何拒绝都不污染既有视图。
        成本/顺序/事件水位/初始身份一并复制（T5：副本必须与本体同代）。
        """
        clone = PhysicalLotView()
        clone._lots = dict(self._lots)
        clone._costs = dict(self._costs)
        clone._order = list(self._order)
        clone._events = dict(self._events)
        clone._origin = self._origin
        return clone

    # ────────────────────────── T5 状态访问器 ──────────────────────────

    def batch_order(self) -> Tuple[Tuple[str, str], ...]:
        """来源批次登记顺序（``(ts_code, 来源 lot_id)``；恢复后必须逐个相同）。"""
        return tuple(self._order)

    def cost_of(self, ts_code: str, source_lot_id: str) -> float:
        """单来源批次成本（不存在返回 0.0；不参与 I2）。"""
        return float(self._costs.get((str(ts_code), str(source_lot_id)), 0.0))

    def contents(self) -> Tuple[Tuple[str, str, int, float], ...]:
        """全部来源批次 ``(ts_code, 来源 lot_id, 股数, 成本)``（**批次顺序**）。"""
        return tuple(
            (ts_code, source_lot_id, int(self._lots[(ts_code, source_lot_id)]),
             float(self._costs.get((ts_code, source_lot_id), 0.0)))
            for ts_code, source_lot_id in self._order
        )

    def event_watermark(self) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
        """合成买入事件水位（``(event_id, 内容指纹)``，登记序）。"""
        return tuple((label, tuple(fingerprint)) for label, fingerprint in self._events.items())

    def origin_fingerprint(self) -> str:
        """初始冻结输入身份（构造期来源批次顺序的规范化 SHA-256）。"""
        return self._origin

    def shares_by_stock(self) -> Dict[str, int]:
        """按股票汇总的来源股数（与引擎聚合持仓对账用）。"""
        totals: Dict[str, int] = {}
        for (ts_code, _source_lot_id), shares in self._lots.items():
            totals[ts_code] = totals.get(ts_code, 0) + shares
        return totals

    def shares_of(self, ts_code: str, source_lot_id: str) -> int:
        """单来源键股数（不存在返回 0）。"""
        return self._lots.get((str(ts_code), str(source_lot_id)), 0)

    def apply_buy_fill(
        self,
        *,
        ts_code: str,
        source_lot_id: str,
        shares: int,
        event_id: str,
        cost: float = 0.0,
    ) -> bool:
        """同一结算事件登记一笔**物理合成买入**（S4；追加/合并来源批次）。

        物理事实由验收器显式注入（合成注入），本方法只登记，不产生任何成交语义。
        同一 ``event_id`` 同内容重复注入返回 ``False`` 且聚合不变；同键异内容拒绝。

        Args:
            ts_code: 股票
            source_lot_id: 物理来源批次身份（新买入产生新批次；同批次追加合并）
            shares: 正整数股数
            event_id: 稳定事件身份（结算批次内的成交身份）
            cost: 本笔成交成本（金额 + 费用，元；缺省 0.0 = 未登记成本）

        Returns:
            True = 首次登记；False = 同身份同内容重复注入（聚合不变）。

        Raises:
            PhysicalLotViewError: 空标识/非正整数股数/非有限负成本/同 ``event_id``
                内容不同。
        """
        code = str(ts_code)
        lot_id = str(source_lot_id)
        label = str(event_id)
        if not code.strip() or not lot_id.strip():
            raise PhysicalLotViewError("物理合成买入的 ts_code/source_lot_id 不得为空")
        if not label.strip():
            raise PhysicalLotViewError("物理合成买入的 event_id 不得为空（幂等身份）")
        if isinstance(shares, bool) or not hasattr(shares, "__index__"):
            raise PhysicalLotViewError(
                f"物理合成买入股数必须为整数，实际 {shares!r}（股票 {code}，事件 {label}）"
            )
        value = int(shares.__index__())
        if value <= 0:
            raise PhysicalLotViewError(
                f"物理合成买入股数必须为正整数，实际 {value}（股票 {code}，事件 {label}）"
            )
        money = _require_finite_non_negative_cost(cost, lot_id)
        fingerprint = (code, lot_id, value, repr(money))
        recorded = self._events.get(label)
        if recorded is not None:
            if recorded == fingerprint:
                return False
            raise PhysicalLotViewError(
                f"物理合成买入幂等键冲突：事件 {label} 已登记 {recorded}，本次 {fingerprint}；"
                "禁止同一物理事件承载不同事实"
            )
        key = (code, lot_id)
        if key not in self._lots:
            self._order.append(key)
            self._costs[key] = 0.0
        self._lots[key] = self._lots.get(key, 0) + value
        self._costs[key] = self._costs.get(key, 0.0) + money
        self._events[label] = fingerprint
        return True

    def applied_event_ids(self) -> Tuple[str, ...]:
        """已登记的物理合成事件身份（登记顺序，证据用）。"""
        return tuple(self._events)

    def validate_transfer(
        self,
        *,
        ts_code: str,
        allocations: Sequence[Tuple[str, int]],
    ) -> None:
        """校验同一结算事件引用的物理来源批次存在且股数足够（**不改聚合**）。"""
        for source_lot_id, shares in allocations:
            held = self._lots.get((str(ts_code), str(source_lot_id)))
            if held is None:
                raise PhysicalLotViewError(
                    f"内部结算引用物理不存在的来源批次：股票 {ts_code} 来源 "
                    f"{source_lot_id}（伴随视图当前键集 "
                    f"{sorted((code, lot) for code, lot in self._lots)}）"
                )
            if isinstance(shares, bool) or not isinstance(shares, int) or shares <= 0:
                raise PhysicalLotViewError(
                    f"内部结算分配股数必须为正整数，实际 {shares!r}（来源 {source_lot_id}）"
                )
            if shares > held:
                raise PhysicalLotViewError(
                    f"内部结算超出物理来源批次：股票 {ts_code} 来源 {source_lot_id} "
                    f"持有 {held} 股，结算引用 {shares} 股"
                )

    def validate_against_positions(
        self,
        positions: Mapping[str, Mapping[str, Any]],
        *,
        context: str = "",
    ) -> None:
        """与引擎现有聚合持仓字典对账（期初及逐日；不一致即终态报错）。

        Args:
            positions: 引擎 ``positions``（``{股票: {"shares": int, ...}}``）
            context: 附加错误上下文（阶段/日期/宿主/运行 ID 等）
        """
        expected: Dict[str, int] = {}
        for stock, info in positions.items():
            value = _strict_integer_shares(info.get("shares", 0), str(stock), context)
            if value:
                expected[str(stock)] = value
        actual = self.shares_by_stock()
        self._check_stock_sets(expected, actual, context)
        for stock in sorted(expected):
            if expected[stock] != actual[stock]:
                raise PhysicalLotMismatchError(
                    f"I2 右值前置破缺：股票 {stock} 伴随视图来源股数和={actual[stock]} 股 "
                    f"!= 物理聚合持仓={expected[stock]} 股；差={actual[stock] - expected[stock]} 股"
                    f"｜{context}"
                )

    @staticmethod
    def _check_stock_sets(
        expected: Mapping[str, int],
        actual: Mapping[str, int],
        context: str,
    ) -> None:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra:
            raise PhysicalLotMismatchError(
                f"I2 右值前置破缺：伴随视图与物理持仓股票集不一致；"
                f"物理有而视图缺={missing}，视图有而物理无={extra}；"
                f"左(物理)键数={len(expected)} 右(视图)键数={len(actual)} "
                f"容差=0 股 单位=股｜{context}"
            )
