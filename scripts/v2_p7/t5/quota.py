# -*- coding: utf-8 -*-
"""P7 T5 配额与授权：原始目标 ``G`` → 需求 ``D`` → 常规额度 ``R`` → 授权 ``A``。

契约依据（``docs/contracts/netting_freeze.md``「P7 机械复核适用注记」F26）
=====================================================================
- ``R_i = max(0, ρ_i · G_i − MV_hold_i)``，``(ρ_V20, ρ_V10) = (0.7, 0.3)``；
- ``D_i = max(0, G_i − MV_hold_i)``；
- ``A^H_i = max{a ≤ min(D_i, R_i) : a + Fee(a) ≤ cash_open_i}``；
- ``A^S_i(L) = max{a ≤ min(D_i, R_i + L) : a + Fee(a) ≤ cash_open_i + L}``；
- **``credit_outstanding`` 不参与授权**；``L`` 只按**本日转账一次**入账；禁止
  ``cash_open + credit_outstanding`` 或成交后 ``cash + credit`` 双重背书；
- **原始卖出不受配额约束**（配额只作用于买入负债），且原始卖出不得直接入市场。

原始目标 ``G`` 的唯一来源
========================
T2.5 已收口 ``targets/daily_{arm}.parquet`` 的 ``target_notional``：
**账户级** ``G = Σ_arm target_notional``（同一 A 信号的双相位副本合成账户原始目标），
``ρ_i`` 为各袖的**资本份额**。故 ``Σ_i ρ_i = 1``，``Σ_i R_i ≈ G − Σ_i MV_hold_i``。
本模块**不改写**逐袖归一化权重对象：配额只缩放**袖级预算**（整手），
权重向量、选股与排序一律复用 T2.5 冻结结果。

授权与落地的单位约定
====================
- 授权额度 ``a`` 是**货币量**，落地时必须整手：``shares = floor(a / price / lot) · lot``；
- 因此本模块以「逐意图股数 → 货币」的映射（``shares × 当日开盘价``）为准，
  用**单变量比例缩放 + 整手向下取整**实现 ``A ≤ min(D, R + L)`` 与
  ``A + Fee(A) ≤ cash_open + L`` 两条上界，逐项可复算；
- 无当日有效开盘价的意图**不计入**买卖负债（它当日必然被市场可达性阻断），
  但其股数原样保留在意图集里，由 T4 既有语义登记阻断/顺延。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.v2_p7.t4.netting import Intent, replace_intent

#: 现金约束的整手回退轮数上限（与 T4 同值，超过即拒绝，不静默放宽）
CASH_ROUNDS = 64
#: 比例缩放二分的迭代次数（单调函数；1e-12 精度已远超 1 股）
SCALE_ITERATIONS = 48
#: 配额拒绝原因标签（日终作废，不顺延；与 T4 资金不足作废同构）
QUOTA_VOID_REASON = "quota_capped_voided_eod"
#: 整手后无额度
QUOTA_NO_LOT_REASON = "quota_no_lot_after_rounding"


class QuotaError(RuntimeError):
    """配额/授权前置条件不满足（身份缺失、参数非法），一律 fail-fast。"""


def lot_round_down(shares: float, lot_size: int) -> int:
    """向下取整到整手（``lot_size <= 1`` 时按 1 股取整）。"""
    size = max(1, int(lot_size))
    value = int(math.floor(float(shares)))
    if value <= 0:
        return 0
    return (value // size) * size


@dataclass(frozen=True)
class QuotaBounds:
    """单袖单日的额度与需求（货币量）。"""

    gross_target: float
    mv_hold: float
    ratio: float
    demand: float
    quota: float

    def with_credit(self, credit_granted: float) -> float:
        """``R_i + L_i``（借用后额度上界；``L_i`` **有符号**，出借方为负）。

        有符号是关键：``Σ_i L_i = 0`` ⇒ ``Σ_i A_i ≤ Σ_i R_i ≈ G − Σ_i MV_i``，
        即**账户级目标不被借用突破**；出借方把额度让给借入方，自己不超额买入。
        """
        return max(0.0, float(self.quota) + float(credit_granted))

    def payload(self) -> Dict[str, Any]:
        return {
            "gross_target": float(self.gross_target),
            "mv_hold": float(self.mv_hold),
            "quota_ratio": float(self.ratio),
            "demand": float(self.demand),
            "quota": float(self.quota),
        }


def quota_bounds(*, gross_target: float, mv_hold: float, ratio: float) -> QuotaBounds:
    """``D = max(0, G − MV)``、``R = max(0, ρG − MV)``（``ρ ∈ (0, 1]``）。"""
    target = float(gross_target)
    held = float(mv_hold)
    rho = float(ratio)
    if not math.isfinite(target) or target < 0.0:
        raise QuotaError(f"原始目标 G 必须为非负有限数：{gross_target!r}")
    if not math.isfinite(held):
        raise QuotaError(f"持仓市值 MV_hold 必须为有限数：{mv_hold!r}")
    if not math.isfinite(rho) or not (0.0 < rho <= 1.0):
        raise QuotaError(f"配额比率 ρ 必须落在 (0, 1]：{ratio!r}")
    demand = max(0.0, target - held)
    quota = max(0.0, rho * target - held)
    # ρ <= 1 => R <= D 恒成立（``borrow only when D > R`` 由结构保证，非口径假设）
    if quota > demand + 1e-9:
        raise QuotaError(
            f"配额上界破缺：R={quota!r} > D={demand!r}（ρ={rho!r}, G={target!r}, MV={held!r}）"
        )
    return QuotaBounds(gross_target=target, mv_hold=held, ratio=rho, demand=demand, quota=quota)


def holdings_market_value(sleeve: Any, prices: Any) -> Tuple[float, Dict[str, Any]]:
    """袖持仓按**开盘时点已知量**计值（``R/D`` 的冻结输入）。

    解析顺序（**不得**使用当日收盘价：那是开盘决策点不可知的未来量）：

    1. 当日真实开盘价（``market_open``，成交价同源）；
    2. 最后已知**历史**收盘价（``last_known``，日期早于当日）；
    3. 批次取得价（``acquired_price``，兜底且逐股登记）。

    逐股 ``source/price_date`` 与 ``fallback_count`` 一并返回，供证据登记
    （审计 ``R = max(0, ρG − MV)`` 的输入是否含回退）。
    """
    total = 0.0
    fallback: List[Dict[str, Any]] = []
    for lot in sleeve.lots:
        lookup = prices.market_open(lot.ts_code)
        if lookup.ok and lookup.price is not None:
            price = float(lookup.price)
            source = "open"
            price_date = str(getattr(prices, "_day", "") or "")
        else:
            known = prices.last_known(lot.ts_code)
            if known is not None and float(known[0]) > 0.0:
                price = float(known[0])
                source = "last_known"
                price_date = str(known[1])
            else:
                price = float(lot.acquired_price)
                source = "acquired_price"
                price_date = str(lot.acquired_date)
            fallback.append(
                {
                    "ts_code": lot.ts_code,
                    "reason": lookup.reason,
                    "source": source,
                    "price_date": price_date,
                }
            )
        if not math.isfinite(price) or price <= 0.0:
            raise QuotaError(
                f"袖 {sleeve.sleeve_id} 的 {lot.ts_code} 开盘口径估值不可用"
                f"（source={source}, price={price!r}）"
            )
        total += float(lot.shares) * float(price)
    return float(total), {"fallback": fallback, "fallback_count": len(fallback)}


# ────────────────────────── 买入负债与缩放 ──────────────────────────


def _priced_buy_intents(
    intents: Sequence[Intent], price_of: Callable[[str], Optional[float]]
) -> List[Tuple[Intent, float]]:
    """有当日有效开盘价的**买入**意图（无价意图不构成当日买入负债）。"""
    priced: List[Tuple[Intent, float]] = []
    for intent in intents:
        if intent.side != "buy":
            continue
        price = price_of(intent.ts_code)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
            continue
        priced.append((intent, float(price)))
    return priced


def buy_notional(
    intents: Sequence[Intent], price_of: Callable[[str], Optional[float]]
) -> Tuple[int, float]:
    """当日买入负债 ``(股数, 货币量)``（只计有当日有效开盘价的意图）。"""
    shares = 0
    amount = 0.0
    for intent, price in _priced_buy_intents(intents, price_of):
        shares += int(intent.shares)
        amount += float(intent.shares) * price
    return shares, float(amount)


def _scale_intents(
    priced: Sequence[Tuple[Intent, float]], scale: float, lot_size: int
) -> Tuple[Dict[str, int], int, float]:
    """按比例缩放并整手向下取整：``(逐意图股数, 授权股数, 授权货币)``。"""
    factor = max(0.0, min(1.0, float(scale)))
    shares_by_key: Dict[str, int] = {}
    total_shares = 0
    total_amount = 0.0
    for intent, price in priced:
        if factor >= 1.0:
            shares = int(intent.shares)
        else:
            shares = lot_round_down(int(intent.shares) * factor, lot_size)
        shares_by_key[intent.intent_key] = int(shares)
        total_shares += int(shares)
        total_amount += float(shares) * float(price)
    return shares_by_key, int(total_shares), float(total_amount)


def quota_scaled_shares(
    *, priced: Sequence[Tuple[Intent, float]], bound: float, lot_size: int
) -> Dict[str, Any]:
    """**只按额度**缩放（不含现金上限）：``a ≤ min(D, R+L)`` 的整手实现。

    借用申请额必须由「额度口径的买入负债」与 ``cash_open`` 相减得到；若先套现金上限，
    缺口会恒为 0，借用永不发生（口径错误）。
    """
    wanted_shares = sum(int(intent.shares) for intent, _price in priced)
    full_amount = float(sum(float(intent.shares) * price for intent, price in priced))
    limit = max(0.0, float(bound))
    if wanted_shares <= 0:
        return {
            "shares_by_key": {},
            "shares": 0,
            "notional": 0.0,
            "fee": 0.0,
            "reason": "no_priced_intent",
            "quota_factor": 0.0,
        }
    factor = 1.0 if full_amount <= limit else (limit / full_amount if full_amount else 0.0)
    shares_by_key, shares, amount = _scale_intents(priced, factor, lot_size)
    return {
        "shares_by_key": shares_by_key,
        "shares": int(shares),
        "notional": float(amount),
        "fee": 0.0,
        "reason": "ok" if shares >= wanted_shares else "quota_binding",
        "quota_factor": float(factor),
    }


def authorized_by_cash(
    *,
    priced: Sequence[Tuple[Intent, float]],
    cash: float,
    buy_cost: Callable[[float], float],
    lot_size: int,
    bound: float,
) -> Dict[str, Any]:
    """``A = max{a ≤ bound : a + Fee(a) ≤ cash}`` 的整手实现（单变量比例二分）。

    Args:
        priced: 当日可定价买入意图与开盘价；
        cash: ``cash_open``（VV-H）或 ``cash_open + L``（VV-S）——**不含**存量 credit，
            也**不含**净卖/转账预计回款；
        buy_cost: 申报买入费用函数（与 T4 同一 ``CostModel.calculate_buy_cost``）；
        lot_size: 整手步长；
        bound: ``min(D, R (+L))`` 的货币上界。

    Returns:
        ``{"shares_by_key":..., "shares":..., "notional":..., "fee":..., "reason":...}``。
        ``reason`` ∈ ``{"ok", "no_priced_intent", "no_lot_after_rounding", "cash_binding"}``。
    """
    wanted_shares = sum(int(intent.shares) for intent, _price in priced)
    if wanted_shares <= 0:
        return {
            "shares_by_key": {},
            "shares": 0,
            "notional": 0.0,
            "fee": 0.0,
            "reason": "no_priced_intent",
            "bound": float(bound),
            "cash": float(cash),
        }
    available = max(0.0, float(cash))
    limit = max(0.0, float(bound))
    if available <= 0.0 or limit <= 0.0:
        return {
            "shares_by_key": {intent.intent_key: 0 for intent, _price in priced},
            "shares": 0,
            "notional": 0.0,
            "fee": 0.0,
            "reason": "cash_binding" if limit > 0.0 else "quota_binding",
            "bound": float(limit),
            "cash": float(available),
        }
    full_shares = sum(int(intent.shares) for intent, _price in priced)
    full_amount = float(sum(float(intent.shares) * price for intent, price in priced))
    quota_factor = 1.0 if full_amount <= limit else (limit / full_amount if full_amount else 0.0)
    shares_by_key, shares, amount = _scale_intents(priced, quota_factor, lot_size)
    # 零份额不收费（``buy_cost(0)`` 会返回 min_commission，不得据此派生幽灵缺口）
    fee = float(buy_cost(amount)) if amount > 0.0 else 0.0
    reason = "ok" if shares >= full_shares else "quota_binding"
    if amount + fee <= available:
        return {
            "shares_by_key": shares_by_key,
            "shares": shares,
            "notional": amount,
            "fee": fee,
            "reason": reason,
            "bound": float(limit),
            "cash": float(available),
            "quota_factor": float(quota_factor),
        }
    # 现金（含费用）不足：在 [0, quota_factor] 上二分比例，整手向下取整保证可行
    low = 0.0
    high = float(quota_factor)
    best: Dict[str, Any] = {
        "shares_by_key": {key: 0 for key in shares_by_key},
        "shares": 0,
        "notional": 0.0,
        "fee": 0.0,
    }
    for _step in range(SCALE_ITERATIONS):
        middle = (low + high) / 2.0
        candidate = _scale_intents(priced, middle, lot_size)
        candidate_fee = float(buy_cost(candidate[2])) if candidate[2] > 0.0 else 0.0
        if candidate[2] + candidate_fee <= available:
            best = {
                "shares_by_key": candidate[0],
                "shares": candidate[1],
                "notional": candidate[2],
                "fee": candidate_fee,
            }
            low = middle
        else:
            high = middle
        if high - low <= 1e-12:
            break
    # 整手回退：二分可能落在不可行侧，逐手回退确保严格可行（最多 CASH_ROUNDS 轮）
    step = max(1, int(lot_size))
    rounds = 0
    while best["shares"] > 0 and best["notional"] + best["fee"] > available + 1e-9:
        rounds += 1
        if rounds > CASH_ROUNDS:
            raise QuotaError(
                f"现金授权整手回退超过 {CASH_ROUNDS} 轮：notional={best['notional']!r} "
                f"fee={best['fee']!r} cash={available!r}（拒绝而非放宽）"
            )
        best = _reduce_one_lot(priced, best["shares_by_key"], step, buy_cost)
    return {
        **best,
        "reason": "cash_binding",
        "bound": float(limit),
        "cash": float(available),
        "quota_factor": float(quota_factor),
    }


def _reduce_one_lot(
    priced: Sequence[Tuple[Intent, float]],
    shares_by_key: Mapping[str, int],
    step: int,
    buy_cost: Callable[[float], float],
) -> Dict[str, Any]:
    """按「最后一只意图」逐手减少（确定性：字典序末位优先），并重算费用。"""
    keys = [intent.intent_key for intent, _price in priced]
    reduced = dict(shares_by_key)
    for key in reversed(keys):
        if int(reduced.get(key, 0)) >= step:
            reduced[key] = int(reduced[key]) - step
            break
    price_map = _price_map(priced)
    shares = sum(int(value) for value in reduced.values())
    amount = sum(float(int(reduced[key])) * float(price_map[key]) for key in reduced)
    return {
        "shares_by_key": reduced,
        "shares": int(shares),
        "notional": float(amount),
        "fee": float(buy_cost(amount)) if shares > 0 else 0.0,
    }


def _price_map(priced: Sequence[Tuple[Intent, float]]) -> Dict[str, float]:
    return {intent.intent_key: float(price) for intent, price in priced}


# ────────────────────────── 逐袖配额决策 ──────────────────────────


@dataclass
class SleeveQuotaDecision:
    """单袖单日的配额与授权证据（货币量 + 股数，逐项可复算）。"""

    trade_date: str
    sleeve: str
    bounds: QuotaBounds
    wanted_shares: int
    wanted_notional: float
    quota_bound: float
    authorized_shares: int
    authorized_notional: float
    authorized_fee: float
    denied_shares: int
    cap_binding: bool
    reason: str
    credit_granted: float = 0.0
    credit_outstanding: float = 0.0
    cash_open: float = 0.0
    #: 逐意图最终股数（含未被配额触碰的意图）
    final_shares: Dict[str, int] = field(default_factory=dict)
    #: 逐意图被配额拒绝的股数
    denied_by_intent: Dict[str, int] = field(default_factory=dict)

    @property
    def quota_with_credit(self) -> float:
        return self.bounds.with_credit(self.credit_granted)

    @property
    def not_binding(self) -> bool:
        return not self.cap_binding

    def payload(self) -> Dict[str, Any]:
        return {
            "trade_date": str(self.trade_date),
            "sleeve": str(self.sleeve),
            **self.bounds.payload(),
            "wanted_shares": int(self.wanted_shares),
            "wanted_notional": float(self.wanted_notional),
            "quota_bound": float(self.quota_bound),
            "quota_with_credit": float(self.quota_with_credit),
            "credit_granted": float(self.credit_granted),
            "credit_outstanding": float(self.credit_outstanding),
            "cash_open": float(self.cash_open),
            "authorized_shares": int(self.authorized_shares),
            "authorized_notional": float(self.authorized_notional),
            "authorized_fee": float(self.authorized_fee),
            "denied_shares": int(self.denied_shares),
            "cap_binding": bool(self.cap_binding),
            "not_binding": bool(self.not_binding),
            "reason": str(self.reason),
            "denied_by_intent": {
                key: int(value) for key, value in sorted(self.denied_by_intent.items())
            },
            "authorization_rule": (
                "A = max{a <= min(D, R + L) : a + Fee(a) <= cash_open + L}；"
                "credit_outstanding 不参与授权；L 只本日转账一次入账"
            ),
        }


def decide_quota(
    *,
    trade_date: str,
    sleeve: Any,
    intents: Sequence[Intent],
    bounds: QuotaBounds,
    cash_open: float,
    credit_granted: float,
    credit_outstanding: float,
    price_of: Callable[[str], Optional[float]],
    buy_cost: Callable[[float], float],
    lot_size: int,
    enabled: bool = True,
    apply_cash: bool = True,
) -> Tuple[List[Intent], SleeveQuotaDecision]:
    """配额 + 授权一次落定：返回**最终意图集**与决策证据。

    ``enabled=False``（T4 兼容门 / ``quota=off``）时逐位返回入参意图，
    决策证据标 ``reason="quota_disabled"``（``bound`` 记为 ``min(D, R+L)`` 仅供对照）。

    ``apply_cash=False`` 时只套额度上界 ``min(D, R+L)``（**不含**费用后现金上界）：
    借用申请额与 cap-binding 诊断必须用这一口径（否则缺口恒为 0，借用永不发生）。
    """
    priced = _priced_buy_intents(intents, price_of)
    wanted_shares, wanted_notional = buy_notional(intents, price_of)
    bound = min(bounds.demand, bounds.quota)
    if enabled:
        bound = min(bounds.demand, bounds.with_credit(credit_granted))
    if not enabled:
        decision = SleeveQuotaDecision(
            trade_date=str(trade_date),
            sleeve=str(sleeve.sleeve_id),
            bounds=bounds,
            wanted_shares=int(wanted_shares),
            wanted_notional=float(wanted_notional),
            quota_bound=float(bound),
            authorized_shares=int(wanted_shares),
            authorized_notional=float(wanted_notional),
            authorized_fee=0.0,
            denied_shares=0,
            cap_binding=False,
            reason="quota_disabled",
            credit_granted=float(credit_granted),
            credit_outstanding=float(credit_outstanding),
            cash_open=float(cash_open),
            final_shares={intent.intent_key: int(intent.shares) for intent in intents},
            denied_by_intent={},
        )
        return list(intents), decision
    if apply_cash:
        authorization = authorized_by_cash(
            priced=priced,
            cash=float(cash_open) + float(credit_granted),
            buy_cost=buy_cost,
            lot_size=int(lot_size),
            bound=float(bound),
        )
    else:
        authorization = quota_scaled_shares(
            priced=priced, bound=float(bound), lot_size=int(lot_size)
        )
        # 零份额不收费（``A=0, F=0``）：费用只按实际授权名义计算
        notional = float(authorization["notional"])
        authorization["fee"] = float(buy_cost(notional)) if notional > 0.0 else 0.0
    shares_by_key: Dict[str, int] = dict(authorization["shares_by_key"])
    final: List[Intent] = []
    denied_by_intent: Dict[str, int] = {}
    denied_shares = 0
    for intent in intents:
        if intent.side != "buy":
            final.append(intent)
            continue
        if intent.intent_key not in shares_by_key:
            # 无当日有效开盘价：不构成当日买入负债，原样保留（由 T4 登记阻断/顺延）
            final.append(intent)
            continue
        keep = int(shares_by_key[intent.intent_key])
        lost = int(intent.shares) - keep
        if lost > 0:
            denied_by_intent[intent.intent_key] = int(lost)
            denied_shares += int(lost)
        if keep > 0:
            final.append(replace_intent(intent, keep) if keep != intent.shares else intent)
    reason = str(authorization["reason"])
    if denied_shares > 0 and int(authorization["shares"]) == 0 and wanted_shares > 0:
        reason = QUOTA_NO_LOT_REASON if bound > 0.0 else "quota_binding"
    cap_binding = bool(denied_shares > 0)
    decision = SleeveQuotaDecision(
        trade_date=str(trade_date),
        sleeve=str(sleeve.sleeve_id),
        bounds=bounds,
        wanted_shares=int(wanted_shares),
        wanted_notional=float(wanted_notional),
        quota_bound=float(bound),
        authorized_shares=int(authorization["shares"]),
        authorized_notional=float(authorization["notional"]),
        authorized_fee=float(authorization["fee"]),
        denied_shares=int(denied_shares),
        cap_binding=cap_binding,
        reason=reason,
        credit_granted=float(credit_granted),
        credit_outstanding=float(credit_outstanding),
        cash_open=float(cash_open),
        final_shares={
            intent.intent_key: int(shares_by_key.get(intent.intent_key, int(intent.shares)))
            for intent in final
            if intent.side == "buy"
        },
        denied_by_intent=denied_by_intent,
    )
    if apply_cash:
        _assert_authorization_bounds(decision)
    return final, decision


def _assert_authorization_bounds(decision: SleeveQuotaDecision) -> None:
    """``A ≤ min(D, R+L)`` 与 ``A + Fee(A) ≤ cash_open + L`` 的即时硬校验。"""
    limit = min(decision.bounds.demand, decision.bounds.with_credit(decision.credit_granted))
    if decision.authorized_notional > limit + 1e-6:
        raise QuotaError(
            f"{decision.sleeve} {decision.trade_date} 授权越界："
            f"A={decision.authorized_notional!r} > min(D,R+L)={limit!r}"
        )
    available = float(decision.cash_open) + float(decision.credit_granted)
    total = decision.authorized_notional + decision.authorized_fee
    if total > available + 1e-6:
        raise QuotaError(
            f"{decision.sleeve} {decision.trade_date} 费用后现金越界："
            f"A+Fee={total!r} > cash_open+L={available!r}"
        )


def funding_gap(decision: SleeveQuotaDecision, *, cash_open: float) -> float:
    """本日资金缺口（借用申请额）：``max(0, 授权买入负债 − cash_open)``。

    只按 ``cash_open`` 计（净卖/转账预计回款**不得**用于当日买入背书）。
    """
    need = float(decision.authorized_notional) + float(decision.authorized_fee)
    return float(max(0.0, need - float(cash_open)))


def summarize_decisions(
    decisions: Iterable[SleeveQuotaDecision],
) -> Dict[str, Any]:
    """逐袖决策的汇总机械量（不产出任何经济结论）。"""
    rows = list(decisions)
    binding = [row for row in rows if row.cap_binding]
    return {
        "days": len({row.trade_date for row in rows}),
        "sleeve_days": len(rows),
        "cap_binding_sleeve_days": len(binding),
        "not_binding_sleeve_days": len(rows) - len(binding),
        "wanted_notional_total": float(sum(row.wanted_notional for row in rows)),
        "authorized_notional_total": float(sum(row.authorized_notional for row in rows)),
        "denied_shares_total": int(sum(row.denied_shares for row in rows)),
        "no_lot_days": sorted(
            {row.trade_date for row in rows if row.reason == QUOTA_NO_LOT_REASON}
        ),
        "rule": "cap_binding = 当日该袖买入负债被 min(D, R+L) 或费用后现金截断（含整手）",
    }
