# -*- coding: utf-8 -*-
"""P7 T6 黄金例（§4.6 口径锚）：三条，全部**可手算复现**。

1. **一致日**：首个授信日之前最后一个交易日，两臂逐位一致（NAV / 费用 / 成交 / 逐袖资产负债）；
2. **差异日**：每个授信日，借入袖的授权股数由 ``authorized_by_cash`` **重算**，并与
   「不给 L 的反事实授权」对照 ⇒ 多成交完全由 L 解释，额外费用由 ``calculate_buy_cost`` 复核；
3. **死局 / 迟回收**：用 T5 借款台账原语构造 2 袖合成例，把「本金转移零 P&L」与
   「死局不改 E、迟回收只搬 cash/credit」按手算数字逐项核对（历史窗口内该态未出现，
   故以合成例提供等价锚点；缺席事实已在 T5 证据中登记）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from loguru import logger

from scripts.v2_p7.t5.credit import CreditBook, CreditGrant
from scripts.v2_p7.t5.quota import authorized_by_cash
from scripts.v2_p7.t5.recall import service_credit_day
from scripts.v2_p7.t6.params import T6Params

GOLDEN_SCHEMA = "p7_t6_golden/v1"


class GoldenError(RuntimeError):
    """黄金例前置条件不满足（缺授信日/授权不一致），一律 fail-fast。"""


def build_golden(
    *,
    params: T6Params,
    frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    results: Mapping[str, Any],
    buy_index: Mapping[str, Any],
) -> Dict[str, Any]:
    """三条黄金例（任一不通过即拒绝发布）。"""
    cases = [
        _identical_day(params=params, frames=frames),
        _borrow_day_recompute(params=params, frames=frames, results=results, buy_index=buy_index),
        _deadlock_late_recovery_handcalc(params=params),
    ]
    failed = [case["name"] for case in cases if not case["ok"]]
    return {
        "schema": GOLDEN_SCHEMA,
        "ok": not failed,
        "failed": failed,
        "cases": cases,
        "rule": "三条全部通过才允许发布（G6）",
    }


# ────────────────────────── ① 一致日 ──────────────────────────


def _identical_day(
    *, params: T6Params, frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]
) -> Dict[str, Any]:
    vvs_daily = frames["vvs"]["daily"]
    event_days = [str(row["trade_date"]) for row in vvs_daily if float(row["granted_total"]) > 0.0]
    dates = [str(row["trade_date"]) for row in frames["vvh"]["daily"]]
    if not event_days:
        raise GoldenError("VV-S 全窗无授信日，无法构造「一致日」锚点")
    candidates = [day for day in dates if day < event_days[0]]
    if not candidates:
        raise GoldenError(f"首个授信日 {event_days[0]} 之前无可比对交易日")
    day = candidates[-1]
    control = _day_payload(frames["vvh"], day)
    treatment = _day_payload(frames["vvs"], day)
    checks = {
        "nav_equal": control["nav"] == treatment["nav"],
        "fee_equal": control["fee_day"] == treatment["fee_day"],
        "trades_equal": control["trades"] == treatment["trades"],
        "sleeve_equal": control["sleeve"] == treatment["sleeve"],
        "credit_all_zero": treatment["credit_sum"] == 0.0,
    }
    return {
        "name": "identical_day_before_first_grant",
        "kind": "constructed_on_history",
        "date": day,
        "first_grant_date": event_days[0],
        # 首笔授信后两臂持仓永久分叉（多出的整手不会消失），因此「成交非空且两臂一致」
        # 的日子在本窗口内**不存在**；本锚点只能证明「分叉前逐位一致」。
        "trivial_anchor": True,
        "documented_limit": (
            "两臂在首笔授信（{first}）后永久分叉，窗口内不存在「成交非空且两臂一致」的日子；"
            "本锚点只覆盖分叉前的一致性，不构成 credit 侧实现的独立验证（该验证由差异日逐笔复算承担）"
        ).format(first=event_days[0]),
        "checks": checks,
        "observed": {"control": control, "treatment": treatment},
        "hand_calc": (
            "授信发生前两臂参数逐位相同（仅 credit 开关不同且未触发）⇒ "
            "同日 NAV、费用、成交、逐袖资产负债必须**逐位相等**（平凡一致：差异尚未产生）"
        ),
        "ok": bool(all(checks.values())),
    }


def _day_payload(frame: Mapping[str, Sequence[Mapping[str, Any]]], day: str) -> Dict[str, Any]:
    daily = next(row for row in frame["daily"] if str(row["trade_date"]) == day)
    sleeve = [
        {
            "sleeve": row["sleeve"],
            "cash": float(row["cash"]),
            "credit": float(row["credit"]),
            "equity": float(row["equity"]),
        }
        for row in frame["sleeve"]
        if str(row["trade_date"]) == day
    ]
    trades = [
        {key: value for key, value in row.items() if key != "market_notional"}
        for row in frame["trades"]
        if str(row["trade_date"]) == day
    ]
    return {
        "date": day,
        "nav": float(daily["nav"]),
        "fee_day": float(daily["fee_day"]),
        "turnover": float(daily["turnover"]),
        "credit_sum": float(daily["credit_sum"]),
        "sleeve": sleeve,
        "trades": trades,
    }


# ────────────────────────── ② 差异日逐笔复算 ──────────────────────────


def _borrow_day_recompute(
    *,
    params: T6Params,
    frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    results: Mapping[str, Any],
    buy_index: Mapping[str, Any],
) -> Dict[str, Any]:
    model = _cost_model(params)
    prices = _price_book(params)
    events: List[Dict[str, Any]] = []
    quota_daily = {str(row["trade_date"]): row for row in results["vvs"].quota_daily}
    for row in frames["vvs"]["daily"]:
        day = str(row["trade_date"])
        if float(row["granted_total"]) <= 0.0:
            continue
        record = quota_daily.get(day)
        if record is None:
            raise GoldenError(f"VV-S {day} 有授信但缺配额决策记录")
        for sleeve, grant in (record.get("decisions") or {}).items():
            amount = float(grant.get("credit_granted") or 0.0)
            if amount <= 0.0:
                continue
            events.append(
                _recompute_sleeve_day(
                    day=day,
                    sleeve=str(sleeve),
                    grant=amount,
                    decision=grant,
                    intents=tuple(buy_index.get(day) or ()),
                    prices=prices,
                    model=model,
                    lot_size=int(params.lot_size),
                )
            )
    checks = {
        "events_present": bool(events),
        "all_recomputed": all(
            item["authorized_recomputed"] == item["authorized_recorded"] for item in events
        ),
        "all_extra_positive": all(item["extra_shares"] > 0 for item in events),
        "all_fee_recomputed": all(item["fee_consistent"] for item in events),
    }
    return {
        "name": "borrow_day_authorization_recompute",
        "kind": "constructed_on_history",
        "events": events,
        "checks": checks,
        "hand_calc": (
            "A^S = max{a ≤ min(D, R+L) : a + Fee(a) ≤ cash_open + L}；"
            "反事实 A^H(=不给 L) 用同一函数以 cash_open 与 min(D,R) 重算；"
            "二者之差必须 > 0（否则该授信不成立），额外费用 = calculate_buy_cost(额外名义)"
        ),
        "ok": bool(all(checks.values())),
    }


def _recompute_sleeve_day(
    *,
    day: str,
    sleeve: str,
    grant: float,
    decision: Mapping[str, Any],
    intents: Sequence[Any],
    prices: Any,
    model: Any,
    lot_size: int,
) -> Dict[str, Any]:
    priced: List[Any] = []
    prices.begin_day(day, tracked=tuple(sorted({str(item.ts_code) for item in intents})))
    for intent in intents:
        if str(intent.sleeve_id) != sleeve or intent.side != "buy":
            continue
        lookup = prices.market_open(str(intent.ts_code))
        if lookup.ok and lookup.price is not None and float(lookup.price) > 0.0:
            priced.append((intent, float(lookup.price)))
    cash_open = float(decision["cash_open"])
    quota_bound = float(decision["quota_bound"])
    demand = float(decision["demand"])
    quota = float(decision["quota"])
    with_credit = authorized_by_cash(
        priced=priced,
        cash=cash_open + grant,
        buy_cost=model.calculate_buy_cost,
        lot_size=lot_size,
        bound=quota_bound,
    )
    without = authorized_by_cash(
        priced=priced,
        cash=cash_open,
        buy_cost=model.calculate_buy_cost,
        lot_size=lot_size,
        bound=min(demand, quota),
    )
    extra = int(with_credit["shares"]) - int(without["shares"])
    extra_notional = float(with_credit["notional"]) - float(without["notional"])
    # 费用按**申报整体**计一次（最低佣金由整体满足），故增量费用取两条授权的费用差；
    # 「extra_notional 单独下单」口径同时登记作对照（会命中最低佣金，不可作增量口径）。
    marginal_fee = float(with_credit["fee"]) - float(without["fee"])
    standalone_fee = float(model.calculate_buy_cost(extra_notional)) if extra_notional > 0 else 0.0
    recorded_fee = float(decision["authorized_fee"]) - float(without["fee"])
    return {
        "trade_date": day,
        "sleeve": sleeve,
        "grant": grant,
        "authorized_recorded": int(decision["authorized_shares"]),
        "authorized_recomputed": int(with_credit["shares"]),
        "authorized_without_credit": int(without["shares"]),
        "extra_shares": extra,
        "extra_notional": extra_notional,
        "extra_fee_marginal": marginal_fee,
        "extra_fee_standalone_reference": standalone_fee,
        "extra_fee_recorded": recorded_fee,
        "fee_consistent": bool(abs(marginal_fee - recorded_fee) <= 1e-6),
        "cash_open": cash_open,
        "quota_bound": quota_bound,
        "quota_without_credit": float(min(demand, quota)),
        "authorization_rule": "A = max{a ≤ min(D,R+L) : a + Fee(a) ≤ cash_open + L}",
    }


# ────────────────────────── ③ 死局 / 迟回收手算 ──────────────────────────


def _deadlock_late_recovery_handcalc(*, params: T6Params) -> Dict[str, Any]:
    pc = float(params.physical_capital)
    book = CreditBook()
    # 开局（手算给定，MV 恒 0 ⇒ E = cash − credit）
    opening_cash = {"A": 5000.0, "B": 1000.0}
    expected = {
        "equity": {"A": 5000.0, "B": 1000.0},
        "credit_after_grant": {"A": -1000.0, "B": 1000.0},
        "credit_after_deadlock": {"A": -1000.0, "B": 1000.0},
        "credit_after_late": {"A": 0.0, "B": 0.0},
        "deadlocked_total": 1000.0,
        "late_recovery_total": 1000.0,
        "nav_effect": 0.0,
    }
    grant = CreditGrant(lender="A", borrower="B", amount=1000.0, due_date="20991231")
    cash_delta, credit_delta, _events = book.apply_grants(day="20991230", grants=[grant])
    balances_grant = book.credit_by_sleeve(("A", "B"), day="20991230", record=True)["balances"]
    cash_after_grant = {
        key: float(opening_cash[key]) + float(cash_delta.get(key, 0.0)) for key in opening_cash
    }
    equity_after_grant = _equity_from(cash_after_grant, balances_grant)
    deadlock_provided = {key: 0.0 for key in opening_cash}
    deadlock = service_credit_day(book=book, day="20991231", cash=dict(deadlock_provided))
    balances_deadlock = book.credit_by_sleeve(("A", "B"), day="20991231", record=True)["balances"]
    # 死局日不归还 ⇒ 现金变动为 0；权益按**死局日日终**的 cash/credit 独立计算
    cash_after_deadlock = {
        key: float(cash_after_grant[key])
        + (float(deadlock["cash"][key]) - float(deadlock_provided[key]))
        for key in opening_cash
    }
    equity_after_deadlock = _equity_from(cash_after_deadlock, balances_deadlock)
    provided_cash = {"A": 0.0, "B": 1200.0}
    late = service_credit_day(book=book, day="21000104", cash=dict(provided_cash))
    balances_late = book.credit_by_sleeve(("A", "B"), day="21000104", record=True)["balances"]
    # 服务返回的是**当时可动用现金**（不是袖累计现金）：按差额回加到上一日现金
    cash_after_late = {
        key: float(cash_after_deadlock[key])
        + (float(late["cash"][key]) - float(provided_cash[key]))
        for key in opening_cash
    }
    equity_after_late = _equity_from(cash_after_late, balances_late)
    nav_before = _equity_from(opening_cash, {"A": 0.0, "B": 0.0})
    nav_effects = {
        "grant": _nav_delta(nav_before, equity_after_grant, pc),
        "deadlock": _nav_delta(equity_after_grant, equity_after_deadlock, pc),
        "late_recovery": _nav_delta(equity_after_deadlock, equity_after_late, pc),
    }
    checks = {
        "grant_cash_sum_zero": float(sum(cash_delta.values())) == 0.0,
        "grant_credit_sum_zero": float(sum(credit_delta.values())) == 0.0,
        "grant_credit_matches_hand_calc": _close(balances_grant, expected["credit_after_grant"]),
        "grant_equity_matches_hand_calc": _close(equity_after_grant, expected["equity"]),
        "deadlock_registered": float(book.totals()["deadlocked_total"])
        == expected["deadlocked_total"],
        "deadlock_credit_unchanged": _close(balances_deadlock, expected["credit_after_deadlock"]),
        "deadlock_equity_matches_hand_calc": _close(equity_after_deadlock, expected["equity"]),
        "late_recovery_registered": float(book.totals()["late_recovery_total"])
        == expected["late_recovery_total"],
        "late_recovery_credit_zero": _close(balances_late, expected["credit_after_late"]),
        "late_recovery_equity_matches_hand_calc": _close(equity_after_late, expected["equity"]),
        "nav_effect_zero": all(abs(value) <= 1e-12 for value in nav_effects.values()),
    }
    return {
        "name": "deadlock_and_late_recovery_handcalc",
        "kind": "synthetic",
        "physical_capital": pc,
        "hand_calc": {
            "step1_grant": "B 借入 1,000：cash_B +1,000、credit_B +1,000；cash_A −1,000、credit_A −1,000",
            "step2_deadlock": "到期日 B 现金 0 ⇒ 不归还、登记 deadlocked（本金仍在，E 不变）",
            "step3_late": "末日 B 现金 1,200 ⇒ FIFO 归还 1,000；记 late_recovery 1,000；credit 归零",
            "nav_effect": (
                "本金转移与死局/迟回收只改 cash 与 credit（E = cash + MV − credit 不变）⇒ "
                f"三步账户 NAV 影响均为 0（0/PC = {0.0 / pc!r}）；历史窗口内 deadlock/"
                "late_recovery 未出现，故以**原语级合成例**提供等价锚点，"
                "不构成历史路径证据（该态在 1,745 日窗口内缺席）"
            ),
            "expected": expected,
        },
        "observed": {
            "cash_delta": {key: float(value) for key, value in sorted(cash_delta.items())},
            "credit_delta": {key: float(value) for key, value in sorted(credit_delta.items())},
            "credit_after_grant": {k: float(v) for k, v in sorted(balances_grant.items())},
            "credit_after_deadlock": {k: float(v) for k, v in sorted(balances_deadlock.items())},
            "credit_after_late": {k: float(v) for k, v in sorted(balances_late.items())},
            "equity_after_grant": equity_after_grant,
            "equity_after_deadlock": equity_after_deadlock,
            "equity_after_late": equity_after_late,
            "nav_effects": nav_effects,
            "recall_queue": deadlock.get("recall_queue"),
            "deadlocks": deadlock.get("deadlocks"),
            "late_recoveries": late.get("late_recoveries"),
            "totals": book.totals(),
        },
        "checks": checks,
        "ok": bool(all(checks.values())),
    }


def _equity_from(cash: Mapping[str, float], credit: Mapping[str, float]) -> Dict[str, float]:
    """``E_i = cash_i + MV_i − credit_i``（本合成例 MV ≡ 0）。"""
    return {key: float(cash[key]) - float(credit.get(key, 0.0)) for key in sorted(cash)}


def _close(observed: Mapping[str, float], expected: Mapping[str, float]) -> bool:
    if sorted(observed) != sorted(expected):
        return False
    return all(abs(float(observed[key]) - float(expected[key])) <= 1e-9 for key in sorted(expected))


def _nav_delta(
    before: Mapping[str, float], after: Mapping[str, float], physical_capital: float
) -> float:
    """账户 NAV 变化（ΣE / PC 之差；本金转移必须为 0）。"""
    return float(sum(after.values()) - sum(before.values())) / float(physical_capital)


# ────────────────────────── 复用件 ──────────────────────────


def _cost_model(params: T6Params) -> Any:
    from scripts.v2_p7.t6.arms import cost_model

    return cost_model(params)


def _price_book(params: T6Params) -> Any:
    from scripts.v2_p7.t4.price import PriceBook
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    logger.debug("T6 黄金例构造独立价源")
    return PriceBook(DataLoader(Storage(root_path=str(params.data_root))))
