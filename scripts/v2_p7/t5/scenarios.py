# -*- coding: utf-8 -*-
"""P7 T5 黄金例与负例：配额、授权、借用、召回、死局与 B4 逐步核对。

口径
====
- 每个用例都在**确定的合成或历史状态**上构造，逐项断言并按
  ``synthetic`` / ``constructed_on_history`` 分列（禁止混报）；
- 用例只产出**机械量**（额度/授权/现金/credit/lot 状态/指令顺序），
  不做任何经济结论（``alpha_consumed=false``）；
- 断言失败即整例失败（不降级为告警），编排器据此阻断发布。

覆盖清单（任务书 §4.6 / §6）
==========================
无借款、需求不足、双向同时需求、整手后无额度、部分归还、T+1 召回、死局、
T+1 未还永久 ``deadlocked``、自然卖出记 ``late_recovery``、cap-binding 非真空、
VV-H 与「同配额禁 credit」参照逐位一致、B4 黄金例（预净额/授权/最终净额/现金 credit/
内部转账/净市场指令/残差逐步核对），以及「既有未还借款 + 本日新增借款 +
出借方负 credit + 无新借款但 ``cash_open > R`` + 费用预留」组合例。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from scripts.v2_p7.t4.netting import Intent, intent_from_plan_row
from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t5.host import MODE_VVH, MODE_VVS, QuotaCreditHost, T5HostResult

#: 构造口径标签
SYNTHETIC = "synthetic"
ON_HISTORY = "constructed_on_history"

#: 构造例默认交易日（合成日历）
DEFAULT_DAYS: Tuple[str, ...] = tuple(f"202401{idx:02d}" for idx in range(2, 22))


class ScenarioError(RuntimeError):
    """构造例前置条件不满足（价源/参数/断言），一律 fail-fast。"""


@dataclass
class CaseResult:
    """单条构造例结果（``passed`` 为假即整批失败）。"""

    name: str
    kind: str
    passed: bool
    expected: Dict[str, Any] = field(default_factory=dict)
    observed: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "passed": bool(self.passed),
            "expected": dict(self.expected),
            "observed": dict(self.observed),
            "error": self.error,
        }


# ────────────────────────── 合成宿主 ──────────────────────────


class _MiniLoader:
    """按日给出固定行情（可选逐日覆盖/停牌/涨跌停）。"""

    def __init__(
        self,
        quotes: Mapping[str, Mapping[str, Any]],
        overrides: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
    ) -> None:
        self.quotes = {str(code): dict(values) for code, values in quotes.items()}
        self.overrides = {
            str(day): {str(code): dict(values) for code, values in rows.items()}
            for day, rows in (overrides or {}).items()
        }

    def load_clean_daily_by_date(self, day: str, auto_ensure: bool = False) -> Any:
        del auto_ensure
        override = self.overrides.get(str(day)) or {}
        rows = []
        for code in sorted(self.quotes):
            item = dict(self.quotes[code])
            item.update(override.get(code) or {})
            if item.get("absent"):
                continue
            rows.append(
                {
                    "ts_code": code,
                    "open": float(item.get("open") or 0.0),
                    "close": float(item.get("close") or item.get("open") or 0.0),
                    "is_suspended": int(item.get("suspended") or 0),
                    "is_limit_up": int(item.get("limit_up") or 0),
                    "is_limit_down": int(item.get("limit_down") or 0),
                }
            )
        return pd.DataFrame(rows)


class _MiniParams:
    """构造例参数（只实现宿主读取面；与真实参数同口径）。"""

    def __init__(
        self,
        *,
        days: Sequence[str],
        capital: Mapping[str, float],
        account_target: Mapping[str, float],
        ratios: Mapping[str, float],
        lot_size: int = 100,
    ) -> None:
        self.arms = tuple(sorted(capital))
        self.sleeve_capital = {str(k): float(v) for k, v in capital.items()}
        self.physical_capital = float(sum(capital.values()))
        self.lot_size = int(lot_size)
        self.extended_days = tuple(str(day) for day in days)
        self.quota_ratio = {str(k): float(v) for k, v in ratios.items()}
        self.account_target = {str(k): float(v) for k, v in account_target.items()}
        self.arm_target = {
            day: {arm: float(value) * float(self.quota_ratio[arm]) for arm in self.arms}
            for day, value in self.account_target.items()
        }

        class _T25:
            folds = ((0, self.extended_days[0], self.extended_days[-1], 24008),)

        self.t25 = _T25()

    def ratio_of(self, sleeve_id: str) -> float:
        return float(self.quota_ratio[str(sleeve_id)])

    def target_of(self, day: str) -> float:
        return float(self.account_target.get(str(day), 0.0))

    def arm_target_of(self, day: str, arm: str) -> float:
        return float(self.arm_target.get(str(day), {}).get(str(arm), 0.0))

    def split_of(self, day: str) -> int:
        del day
        return 0

    def model_of(self, day: str) -> int:
        del day
        return 24008


def _intents(
    plan: Sequence[Tuple[str, str, str, int, str]],
) -> Tuple[Dict[str, List[Intent]], Dict[str, List[Intent]]]:
    """``(buy_day, sleeve, ts_code, shares, due_day)`` -> 买卖意图索引。"""
    buys: Dict[str, List[Intent]] = {}
    sells: Dict[str, List[Intent]] = {}
    for day, sleeve, code, shares, due in plan:
        buys.setdefault(str(day), []).append(
            intent_from_plan_row(
                sleeve_id=sleeve,
                ts_code=code,
                side="buy",
                shares=int(shares),
                signal_date=str(day),
                due_date=str(due),
            )
        )
        sells.setdefault(str(due), []).append(
            intent_from_plan_row(
                sleeve_id=sleeve,
                ts_code=code,
                side="sell",
                shares=int(shares),
                signal_date=str(day),
                due_date=str(due),
            )
        )
    return buys, sells


def _cost_model() -> Any:
    from src.lazybull.v2.common.cost import CostModel

    return CostModel(
        commission_rate=0.0001954, min_commission=5.0, stamp_tax=0.0005, slippage=0.0005
    )


def _run(
    *,
    params: Any,
    loader: Any,
    plan: Sequence[Tuple[str, str, str, int, str]],
    mode: Any,
    cost_model: Any = None,
    price_factory: Optional[Callable[[], Any]] = None,
) -> T5HostResult:
    buys, sells = _intents(plan)
    model = cost_model or _cost_model()
    return QuotaCreditHost(
        params=params,
        run_id=f"case-{mode.label}",
        prices=PriceBook(loader),
        cost_model=model,
        buy_index=buys,
        sell_index=sells,
        price_factory=price_factory or (lambda: PriceBook(loader)),
        mode=mode,
    ).run()


def _case(
    name: str,
    kind: str,
    body: Callable[[], Dict[str, Any]],
    expected: Optional[Mapping[str, Any]] = None,
) -> CaseResult:
    """执行单例并逐项比对（异常即失败，不吞错）。"""
    want = {str(key): value for key, value in (expected or {}).items()}
    try:
        observed = body()
    except Exception as error:  # noqa: BLE001 - 构造例必须把失败落成证据
        return CaseResult(
            name=name,
            kind=kind,
            passed=False,
            expected=want,
            error=f"{type(error).__name__}: {error}",
        )
    mismatched = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in want.items()
        if not _equal(value, observed.get(key))
    }
    return CaseResult(
        name=name,
        kind=kind,
        passed=not mismatched,
        expected=want,
        observed={**observed, "mismatched": mismatched} if mismatched else observed,
        error="" if not mismatched else f"断言不符：{mismatched}",
    )


def _equal(expected: Any, observed: Any) -> bool:
    if isinstance(expected, float) or isinstance(observed, float):
        try:
            return abs(float(expected) - float(observed)) <= 1e-6
        except (TypeError, ValueError):
            return False
    return expected == observed


# ────────────────────────── 逐例实现 ──────────────────────────


def _case_no_borrow(cost_model: Any) -> Dict[str, Any]:
    """无借款：需求完全被自有现金覆盖（不产生 lot、不产生 request）。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 500_000.0, "v10": 500_000.0},
        account_target={day: 1_000_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    loader = _MiniLoader({"000001.SZ": {"open": 10.0}})
    result = _run(
        params=params,
        loader=loader,
        plan=[(days[0], "v20", "000001.SZ", 10_000, days[5])],
        mode=MODE_VVS,
        cost_model=cost_model,
    )
    return {
        "lots": result.credit_payload["totals"]["lots"],
        "requests": len(result.credit_payload["requests"]),
        "filled_buy": result.market_filled["buy"],
        "gate_ok": result.gate_payload["ok"],
    }


def _case_demand_shortfall(cost_model: Any) -> Dict[str, Any]:
    """需求不足：``D < R`` 无可达情形（票据化）——用极小需求验证 cap 不生效。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 500_000.0, "v10": 500_000.0},
        account_target={day: 10_000_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    loader = _MiniLoader({"000001.SZ": {"open": 10.0}})
    result = _run(
        params=params,
        loader=loader,
        plan=[(days[0], "v20", "000001.SZ", 1_000, days[5])],
        mode=MODE_VVH,
        cost_model=cost_model,
    )
    row = next(item for item in result.quota_rows if item["wanted_notional"] > 0)
    return {
        "cap_binding": row["cap_binding"],
        "authorized_notional": row["authorized_notional"],
        "wanted_notional": row["wanted_notional"],
        "demand_gt_quota": row["demand"] > row["quota"],
    }


def _case_simultaneous_demand(cost_model: Any) -> Dict[str, Any]:
    """双向同时需求：两袖同日缺口都 > 0 ⇒ 无闲置出借方，缺口未被满足。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 500_000.0, "v10": 500_000.0},
        account_target={day: 4_000_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    loader = _MiniLoader({"000001.SZ": {"open": 10.0}})
    result = _run(
        params=params,
        loader=loader,
        plan=[
            (days[0], "v20", "000001.SZ", 48_000, days[5]),
            (days[0], "v10", "000001.SZ", 48_000, days[5]),
            (days[1], "v20", "000001.SZ", 48_000, days[6]),
            (days[1], "v10", "000001.SZ", 48_000, days[6]),
        ],
        mode=MODE_VVS,
        cost_model=cost_model,
    )
    requests = [item for item in result.credit_payload["requests"] if item["trade_date"] == days[1]]
    return {
        "both_requested": sorted(item["sleeve"] for item in requests),
        "granted_total_day1": float(sum(item.get("granted") or 0.0 for item in requests)),
        "lots_created": result.credit_payload["totals"]["lots"],
        "gate_ok": result.gate_payload["ok"],
    }


def _case_no_lot_after_rounding(cost_model: Any) -> Dict[str, Any]:
    """整手后无额度：``R < 1 手`` ⇒ 授权 0 股、记 ``quota_no_lot_after_rounding``。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 500_000.0, "v10": 500_000.0},
        # ρ·G = 0.7 × 1,000 = 700 元 < 1 手（10 元 × 100 股）
        account_target={day: 1_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    loader = _MiniLoader({"000001.SZ": {"open": 10.0}})
    result = _run(
        params=params,
        loader=loader,
        plan=[(days[0], "v20", "000001.SZ", 1_000, days[5])],
        mode=MODE_VVH,
        cost_model=cost_model,
    )
    row = next(item for item in result.quota_rows if item["wanted_notional"] > 0)
    return {
        "authorized_shares": row["authorized_shares"],
        "reason": row["reason"],
        "voids": dict(result.voids),
        "filled_buy": result.market_filled["buy"],
    }


def _restart_substitute(result: Any, state: str) -> Dict[str, Any]:
    """构造例中命中 ``state`` 的检查点续跑证据（R1-06：不得只报 coverage 布尔）。"""
    rows = [
        dict(row)
        for row in ((result.restart or {}).get("checkpoints") or ())
        if bool((row.get("coverage") or {}).get(state))
    ]
    failures = [
        {
            "trade_date": row.get("trade_date"),
            "days_continued": row.get("days_continued"),
            "all_bitwise_identical": row.get("all_bitwise_identical"),
            "credit_identical": row.get("credit_identical"),
            "credit_nav_identical": row.get("credit_nav_identical"),
            "fee_identical": row.get("fee_identical"),
            "ledger_identical": row.get("ledger_identical"),
            "daily_row_identical": row.get("daily_row_identical"),
        }
        for row in rows
        if not (
            bool(row.get("all_bitwise_identical"))
            and bool(row.get("credit_identical"))
            and bool(row.get("fee_identical"))
            and bool(row.get("ledger_identical"))
            and bool(row.get("daily_row_identical"))
            and int(row.get("days_continued") or 0) >= 3
        )
    ]
    return {
        "state": str(state),
        "checkpoint_dates": [str(row.get("trade_date")) for row in rows],
        "days_continued": max((int(row.get("days_continued") or 0) for row in rows), default=0),
        "credit_nav_identical": (
            all(bool(row.get("credit_nav_identical", True)) for row in rows) if rows else False
        ),
        "ok": bool(rows) and not failures,
        "failures": failures,
    }


def _partial_return_and_recall(cost_model: Any) -> Dict[str, Any]:
    """部分归还 + T+1 召回 + 永久 deadlocked + 自然卖出 late_recovery。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 500_000.0, "v10": 500_000.0},
        account_target={day: 1_000_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    loader = _MiniLoader({"000001.SZ": {"open": 10.0}})
    result = _run(
        params=params,
        loader=loader,
        plan=[
            (days[0], "v20", "000001.SZ", 45_000, days[6]),
            (days[1], "v20", "000001.SZ", 45_000, days[7]),
        ],
        mode=MODE_VVS,
        cost_model=cost_model,
    )
    events = [str(item["kind"]) for item in result.credit_payload["events"]]
    lots = result.credit_payload["lots"]
    return {
        "has_grant": "grant" in events,
        "has_recall": "recall_due" in events,
        "has_deadlock": "deadlock" in events,
        "has_late_recovery": "repay_late_recovery" in events,
        "deadlocked_total": float(result.credit_payload["totals"]["deadlocked_total"]),
        "late_recovery_total": float(result.credit_payload["totals"]["late_recovery_total"]),
        "all_lots_closed": all(float(lot["outstanding"]) == 0.0 for lot in lots),
        "coverage": dict((result.restart or {}).get("coverage") or {}).get("deadlocked"),
        "deadlock_restart_substitute": _restart_substitute(result, "deadlocked"),
    }


def _case_partial_return_only(cost_model: Any) -> Dict[str, Any]:
    """部分归还：到期日现金不足以全额归还 ⇒ 部分归还 + 余额 deadlocked。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 300_000.0, "v10": 700_000.0},
        account_target={day: 4_000_000.0 for day in days},
        ratios={"v20": 0.3, "v10": 0.7},
    )
    loader = _MiniLoader(
        {"000001.SZ": {"open": 10.0}, "000002.SZ": {"open": 20.0}, "000003.SZ": {"open": 10.0}}
    )
    # day0：v20 建仓 A/B（B 次日到期）并用尽自有现金后借入；
    # day1（借款到期日）：B 自然卖出产生 4 万元现金 => **部分归还**，余额转 deadlocked；
    # day5：A 自然卖出 => 余额 `late_recovery`。
    result = _run(
        params=params,
        loader=loader,
        plan=[
            (days[0], "v20", "000001.SZ", 20_000, days[5]),
            (days[0], "v20", "000002.SZ", 2_000, days[1]),
            (days[0], "v20", "000003.SZ", 30_000, days[9]),
        ],
        mode=MODE_VVS,
        cost_model=cost_model,
    )
    partials = [item for item in result.credit_payload["events"] if item["kind"] == "repay_partial"]
    late = [
        item for item in result.credit_payload["events"] if item["kind"] == "repay_late_recovery"
    ]
    return {
        "has_partial": bool(partials),
        "has_deadlock": bool(
            [item for item in result.credit_payload["events"] if item["kind"] == "deadlock"]
        ),
        "has_late_recovery": bool(late),
        "lots": result.credit_payload["totals"]["lots"],
        "deadlocked_total": float(result.credit_payload["totals"]["deadlocked_total"]),
        "gate_ok": result.gate_payload["ok"],
    }


def _case_existing_and_new_credit(cost_model: Any) -> Dict[str, Any]:
    """既有未还借款 + 本日新增借款 + 出借方负 credit + 无新借款但 cash_open > R。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 500_000.0, "v10": 500_000.0},
        # 目标放大到配额不紧绑定：缺口只来自「费用后现金不足」（借用条件）
        account_target={day: 4_000_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    loader = _MiniLoader({"000001.SZ": {"open": 10.0}})
    result = _run(
        params=params,
        loader=loader,
        plan=[
            (days[0], "v20", "000001.SZ", 45_000, days[12]),
            (days[1], "v20", "000001.SZ", 20_000, days[13]),
            (days[2], "v20", "000001.SZ", 10_000, days[14]),
        ],
        mode=MODE_VVS,
        cost_model=cost_model,
    )
    non_zero = [row for row in result.balance_rows if abs(float(row["credit"])) > 1e-9]
    negative_credit = [row for row in non_zero if float(row["credit"]) < 0.0]
    positive_credit = [row for row in non_zero if float(row["credit"]) > 0.0]
    lots = [item for item in result.credit_payload["events"] if item["kind"] == "grant"]
    return {
        "has_two_grants": len(lots) >= 2,
        "has_lender_negative_credit": bool(negative_credit),
        "has_borrower_credit": bool(positive_credit),
        "has_deadlock": bool(
            [item for item in result.credit_payload["events"] if item["kind"] == "deadlock"]
        ),
        "sigma_credit": float(sum(float(row["credit"]) for row in result.balance_rows[-2:])),
        "gate_ok": result.gate_payload["ok"],
        "zero_pnl_ok": not result.zero_pnl["violations"],
    }


def _case_cap_binding_nonvacuous(cost_model: Any) -> Dict[str, Any]:
    """cap 必然生效：合成例中 R 严格小于需求 ⇒ VV-H 配额非真空。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 500_000.0, "v10": 500_000.0},
        account_target={day: 1_000_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    loader = _MiniLoader({"000001.SZ": {"open": 10.0}})
    result = _run(
        params=params,
        loader=loader,
        plan=[(days[0], "v20", "000001.SZ", 90_000, days[8])],
        mode=MODE_VVH,
        cost_model=cost_model,
    )
    row = next(item for item in result.quota_rows if item["wanted_notional"] > 0)
    binding_days = sorted(
        {item["trade_date"] for item in result.quota_daily if item["cap_binding_sleeves"]}
    )
    return {
        "quota": row["quota"],
        "demand": row["demand"],
        "cap_binding": row["cap_binding"],
        "authorized_shares": row["authorized_shares"],
        "wanted_shares": row["wanted_shares"],
        "quota_voided": int(result.voids.get("quota_capped_voided_eod", 0)),
        "cash_voided": int(result.voids.get("insufficient_cash_voided_eod", 0)),
        "binding_days": binding_days,
        "quota_strictly_below_demand": row["quota"] < row["demand"],
    }


def _case_vvh_matches_reference(cost_model: Any) -> Dict[str, Any]:
    """VV-H 与「同配额、禁 credit」参照逐位一致（独立价源、独立宿主）。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 500_000.0, "v10": 500_000.0},
        account_target={day: 1_000_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    loader = _MiniLoader({"000001.SZ": {"open": 10.0}})
    plan = [
        (days[0], "v20", "000001.SZ", 90_000, days[8]),
        (days[0], "v10", "000001.SZ", 20_000, days[8]),
        (days[1], "v10", "000001.SZ", 5_000, days[9]),
    ]
    first = _run(params=params, loader=loader, plan=plan, mode=MODE_VVH, cost_model=cost_model)
    second = _run(
        params=params,
        loader=_MiniLoader({"000001.SZ": {"open": 10.0}}),
        plan=plan,
        mode=MODE_VVH,
        cost_model=cost_model,
    )
    return {
        "digest_equal": [a[1] for a in first.digest_series] == [b[1] for b in second.digest_series],
        "nav_equal": first.nav_series == second.nav_series,
        "trades_equal": first.trades == second.trades,
        "fee_equal": [a[1] for a in first.fee_series] == [b[1] for b in second.fee_series],
        "credit_all_zero": all(abs(float(row["credit"])) == 0.0 for row in first.balance_rows),
        "lots": first.credit_payload["totals"]["lots"],
        "requests_only": all(
            item["status"] in ("credit_disabled", "blocked_no_new_buy")
            or item.get("granted") == 0.0
            for item in first.credit_payload["requests"]
        ),
    }


def _case_b4_golden(cost_model: Any) -> Dict[str, Any]:
    """B4 黄金例：预净额 → 授权 → 最终净额 → 现金/credit → 内部转账 → 净市场指令 → 残差。"""
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 500_000.0, "v10": 500_000.0},
        # 目标放大到配额不紧绑定：本必须只核对「预净额 → 授权 → 最终净额」的机械顺序
        account_target={day: 4_000_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    loader = _MiniLoader({"000001.SZ": {"open": 10.0}, "000002.SZ": {"open": 20.0}})
    result = _run(
        params=params,
        loader=loader,
        plan=[
            (days[0], "v20", "000002.SZ", 20_000, days[1]),
            (days[1], "v20", "000001.SZ", 20_000, days[6]),
            (days[1], "v10", "000002.SZ", 20_000, days[6]),
        ],
        mode=MODE_VVS,
        cost_model=cost_model,
    )
    day = days[1]
    trades = [item for item in result.trades if item["trade_date"] == day]
    kinds = [item["kind"] for item in trades]
    offsets = [item for item in trades if item["kind"] == "internal_offset"]
    markets = [item for item in trades if item["kind"].startswith("market_")]
    row = next(
        item for item in result.quota_rows if item["trade_date"] == day and item["sleeve"] == "v20"
    )
    balances = {item["sleeve"]: item for item in result.balance_rows if item["trade_date"] == day}
    day_rows = [item for item in result.daily_rows if item["trade_date"] == day]
    market_positions = [pos for pos, kind in enumerate(kinds) if kind.startswith("market_")]
    return {
        "offset_first": bool(offsets)
        and bool(market_positions)
        and kinds.index("internal_offset") < min(market_positions),
        "offset_shares": int(sum(float(item["shares"]) for item in offsets)),
        "market_buy_shares": int(
            sum(float(item["shares"]) for item in markets if item["kind"] == "market_buy")
        ),
        "offset_price_is_open": all(float(item["price"]) == 20.0 for item in offsets),
        "authorized_within_quota": row["authorized_notional"]
        <= row["quota"] + row["credit_granted"] + 1e-6,
        "auction_fee_reserved": row["authorized_notional"] + row["authorized_fee"]
        <= row["cash_open"] + row["credit_granted"] + 1e-6,
        "residuals_zero": all(
            int(value) == 0 for item in day_rows for value in item["residual_entries"].values()
        ),
        "conservation_violations": int(result.conservation["violations"]),
        "cross_day_violations": int(result.cross_day["violations"]),
        "cash_credit_identity": bool(
            abs(
                sum(float(item["cash"]) for item in balances.values())
                - float(day_rows[0]["physical_cash"])
            )
            <= 1e-6
        ),
    }


def _case_nonpositive_equity_blocks(cost_model: Any) -> Dict[str, Any]:
    """``E_i <= 0``：禁新增买入与 credit（不被动卖出、不删日）。

    构造：袖 v20 仅 1,000 元本金，借入约 1,010 元买入 200 股（10 元）；次日该股价格
    崩到 1 元 ⇒ ``E = 0 + 200 − 1,010 < 0`` ⇒ 政策触发；**触发后的新买入必须为 0**，
    并与「同案不崩价」基线对照（基线确有成交），避免用例空转。
    """
    days = DEFAULT_DAYS
    params = _MiniParams(
        days=days,
        capital={"v20": 1_000.0, "v10": 1_000_000.0},
        account_target={day: 4_000_000.0 for day in days},
        ratios={"v20": 0.7, "v10": 0.3},
    )
    plan = [
        (days[0], "v20", "000001.SZ", 200, days[6]),
        (days[1], "v20", "000001.SZ", 200, days[7]),
        (days[2], "v20", "000001.SZ", 200, days[8]),
    ]
    crashed = _MiniLoader(
        {"000001.SZ": {"open": 10.0}, "000002.SZ": {"open": 20.0}},
        overrides={day: {"000001.SZ": {"open": 1.0, "close": 1.0}} for day in days[1:]},
    )
    baseline_loader = _MiniLoader({"000001.SZ": {"open": 10.0}, "000002.SZ": {"open": 20.0}})
    crashed_result = _run(
        params=params, loader=crashed, plan=plan, mode=MODE_VVS, cost_model=cost_model
    )
    baseline = _run(
        params=params, loader=baseline_loader, plan=plan, mode=MODE_VVS, cost_model=cost_model
    )

    def _buy_trades_after(result: Any, day: str) -> int:
        return sum(
            1
            for trade in result.trades
            if trade["kind"] == "market_buy"
            and str(trade["trade_date"]) > str(day)
            and "v20" in (trade.get("sleeves") or {})
        )

    trigger_days = sorted({str(item["trade_date"]) for item in crashed_result.equity_blocks})
    trigger = trigger_days[0] if trigger_days else ""
    return {
        "has_blocks": bool(crashed_result.equity_blocks),
        "equity_blocks": len(crashed_result.equity_blocks),
        "trigger_day": trigger,
        "crashed_buys_after_trigger": (
            _buy_trades_after(crashed_result, trigger) if trigger else None
        ),
        "baseline_buys_after_trigger": (_buy_trades_after(baseline, trigger) if trigger else None),
        "credit_nav_invalid_days": sum(
            1 for row in crashed_result.credit_nav_rows if bool(row["invalid_nonpositive_equity"])
        ),
        "credit_nav_clean": all(
            (row["independent_nav_credit_adjusted"] is None)
            == bool(row["invalid_nonpositive_equity"])
            for row in crashed_result.credit_nav_rows
        ),
        "gate_ok": crashed_result.gate_payload["ok"],
        "days": len(crashed_result.daily_rows),
    }


#: 全部构造例（顺序即证据顺序）
CASES: Tuple[Tuple[str, str, Callable[[Any], Dict[str, Any]], Mapping[str, Any]], ...] = (
    (
        "no_borrow",
        SYNTHETIC,
        _case_no_borrow,
        {"lots": 0, "requests": 0, "filled_buy": 10_000, "gate_ok": True},
    ),
    (
        "demand_shortfall",
        SYNTHETIC,
        _case_demand_shortfall,
        {"cap_binding": False, "demand_gt_quota": True},
    ),
    (
        "simultaneous_both_sides_demand",
        SYNTHETIC,
        _case_simultaneous_demand,
        {"both_requested": ["v10", "v20"], "granted_total_day1": 0.0, "gate_ok": True},
    ),
    (
        "no_lot_after_rounding",
        SYNTHETIC,
        _case_no_lot_after_rounding,
        {
            "authorized_shares": 0,
            "reason": "quota_no_lot_after_rounding",
            "filled_buy": 0,
        },
    ),
    (
        "partial_return",
        SYNTHETIC,
        _case_partial_return_only,
        {"has_partial": True, "gate_ok": True},
    ),
    (
        "t1_recall_deadlock_late_recovery",
        SYNTHETIC,
        _partial_return_and_recall,
        {
            "has_grant": True,
            "has_recall": True,
            "has_deadlock": True,
            "has_late_recovery": True,
            "all_lots_closed": True,
            "coverage": True,
        },
    ),
    (
        "existing_and_new_credit",
        SYNTHETIC,
        _case_existing_and_new_credit,
        {
            "has_two_grants": True,
            "has_lender_negative_credit": True,
            "has_borrower_credit": True,
            "has_deadlock": True,
            "sigma_credit": 0.0,
            "gate_ok": True,
            "zero_pnl_ok": True,
        },
    ),
    (
        "cap_binding_nonvacuous",
        SYNTHETIC,
        _case_cap_binding_nonvacuous,
        {
            "cap_binding": True,
            "quota_strictly_below_demand": True,
            "wanted_shares": 90_000,
            "authorized_shares": 49_900,
            "quota_voided": 20_000,
            "cash_voided": 20_100,
        },
    ),
    (
        "vvh_matches_no_credit_reference",
        SYNTHETIC,
        _case_vvh_matches_reference,
        {
            "digest_equal": True,
            "nav_equal": True,
            "trades_equal": True,
            "fee_equal": True,
            "credit_all_zero": True,
            "lots": 0,
            "requests_only": True,
        },
    ),
    (
        "b4_golden_stepwise",
        SYNTHETIC,
        _case_b4_golden,
        {
            "offset_first": True,
            "offset_shares": 20_000,
            "offset_price_is_open": True,
            "authorized_within_quota": True,
            "auction_fee_reserved": True,
            "residuals_zero": True,
            "conservation_violations": 0,
            "cross_day_violations": 0,
            "cash_credit_identity": True,
        },
    ),
    (
        "nonpositive_equity_blocks_buy_and_credit",
        SYNTHETIC,
        _case_nonpositive_equity_blocks,
        {
            "has_blocks": True,
            "equity_blocks": 19,
            "credit_nav_invalid_days": 19,
            "credit_nav_clean": True,
            "trigger_day": DEFAULT_DAYS[1],
            "crashed_buys_after_trigger": 0,
            "baseline_buys_after_trigger": 1,
            "gate_ok": True,
            "days": len(DEFAULT_DAYS),
        },
    ),
)


def run_constructed_cases(*, cost_model: Any = None) -> List[CaseResult]:
    """执行全部构造例（本代全部 ``synthetic``；历史口径分支未启用，不留死参数）。"""
    model = cost_model or _cost_model()
    results: List[CaseResult] = []
    for name, kind, body, expected in CASES:
        results.append(_case(name, kind, lambda body=body: body(model), expected))
    return results


def summarize_cases(cases: Sequence[CaseResult]) -> Dict[str, Any]:
    """构造例汇总（失败清单 + 口径分列；不合并计数）。"""
    rows = list(cases)
    failed = [item.name for item in rows if not item.passed]
    synthetic = [item.name for item in rows if item.kind == SYNTHETIC]
    on_history = [item.name for item in rows if item.kind == ON_HISTORY]
    return {
        "total": len(rows),
        "passed": len(rows) - len(failed),
        "failed": failed,
        "synthetic_all": bool(synthetic) and not on_history,
        "synthetic": synthetic,
        "constructed_on_history": on_history,
        "cases": [item.payload() for item in rows],
    }
