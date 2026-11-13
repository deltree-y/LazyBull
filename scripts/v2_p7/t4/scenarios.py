# -*- coding: utf-8 -*-
"""P7 T4 场景提取（自然实例）与构造正负例。

两类证据**分列、禁止混报**（任务书 §4.2 / §4.5）
=============================================
1. **自然实例**：从 T2.5 证据根的计划帧**逐笔复算** S1~S5 计数与股数，并与
   ``intent_precheck.json`` 及冻结计数逐项对账；S1 自然计数为 0，登记为已登记缺口，
   **不得声称 S1 已被自然验证**；
2. **构造例**：在真实历史价源上构造正负例（``synthetic`` + ``constructed_on_history``），
   覆盖 S1 严格等量黄金例、``F > Σ独立费用``、单袖零份额、尾单、部分成交 + 跨日残差 +
   恢复、内部抵消双方独立费用与印花税、有符号节省为负、``A=0,F=0`` / ``A=0,F≠0``、
   缓存估值不得完成 B1 转账、连续停牌估值与恢复逐位一致。

纪律
====
- 自然实例的价格无关字段（事件数/股数/抵消/净额/逐日分布）必须与 T2.5 逐项一致，
  不一致即 fail-fast；
- 构造例只证明机制可被触发并正确记账，**不冒充自然观测**，也不产生任何经济判定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from scripts.v2_p7.t4 import fees as fee_mod
from scripts.v2_p7.t4.netting import Intent, intent_from_plan_row, plan_day
from scripts.v2_p7.t4.price import SOURCE_DAY_CLOSE, ExecutionLookup
from scripts.v2_p7.t4.state import (
    P7Lot,
    PhysicalBook,
    PhysicalLot,
    SleeveState,
)

#: 自然实例必须逐笔复算一致的场景（S1 为已登记缺口，单独处理）
NATURAL_SCENARIOS: Tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")


class ScenarioError(RuntimeError):
    """自然实例复算不一致或构造例前置条件不满足。"""


# ────────────────────────── 自然实例 ──────────────────────────


def load_plans(root: Any, arms: Sequence[str]) -> pd.DataFrame:
    """读取两袖 T2.5 计划帧并合并（只读；不重算选择/截断/持仓排除）。"""
    frames = [pd.read_parquet(root / "targets" / f"plans_{arm}.parquet") for arm in arms]
    return pd.concat(frames, ignore_index=True)


def recompute_natural(
    *,
    root: Any,
    arms: Sequence[str],
    precheck: Mapping[str, Any],
    frozen_counts: Mapping[str, int],
) -> Dict[str, Any]:
    """逐笔复算自然实例（复用 T2.5 同一分类实现，禁止第二套口径）。

    Returns:
        ``{"matches_t25": bool, "matches_frozen": bool, "scenarios": {...},
        "instances": {...}, "natural_status": str}``
    """
    from scripts.v2_p7.t25.intent_precheck import (
        build_intents,
        classify_intents,
        summarize_scenarios,
    )

    plans = load_plans(root, arms)
    buys, sells = build_intents(plans, lambda _day: {})
    records = classify_intents(buys, sells, arms=tuple(arms))
    summary = summarize_scenarios(records)
    scenarios = {name: dict(values) for name, values in summary["scenarios"].items()}

    t25_scenarios = (precheck.get("scenarios") or {}).get("scenarios") or {}
    compares = []
    for name in NATURAL_SCENARIOS + ("S6",):
        mine = scenarios.get(name) or {}
        theirs = t25_scenarios.get(name) or {}
        compares.append(
            {
                "scenario": name,
                "events": [int(mine.get("events", -1)), int(theirs.get("events", -1))],
                "shares": [int(mine.get("shares", -1)), int(theirs.get("shares", -1))],
                "offset_shares": [
                    int(mine.get("offset_shares", -1)),
                    int(theirs.get("offset_shares", -1)),
                ],
                "net_shares": [int(mine.get("net_shares", -1)), int(theirs.get("net_shares", -1))],
                "match": bool(
                    int(mine.get("events", -1)) == int(theirs.get("events", -1))
                    and int(mine.get("shares", -1)) == int(theirs.get("shares", -1))
                    and int(mine.get("offset_shares", -1)) == int(theirs.get("offset_shares", -1))
                    and int(mine.get("net_shares", -1)) == int(theirs.get("net_shares", -1))
                ),
            }
        )
    theirs_days = (precheck.get("scenarios") or {}).get("day_scenario_counts") or {}
    day_scenario_match = bool(dict(summary["day_scenario_counts"]) == dict(theirs_days))
    frozen_ok = all(
        int(scenarios.get(name, {}).get("events", -1)) == int(expected)
        for name, expected in frozen_counts.items()
    )
    instances = _natural_instances(records, arms)
    natural = {
        "S1": int(scenarios.get("S1", {}).get("events", 0)),
        "S2": int(scenarios.get("S2", {}).get("events", 0)),
        "S3": int(scenarios.get("S3", {}).get("events", 0)),
        "S4": int(scenarios.get("S4", {}).get("events", 0)),
        "S5": int(scenarios.get("S5", {}).get("events", 0)),
        "S6": int(scenarios.get("S6", {}).get("events", 0)),
    }
    missing = [name for name in NATURAL_SCENARIOS if natural[name] == 0]
    return {
        "schema": "p7_t4_natural_instances/v1",
        "source": "T2.5 plans 逐笔复算（同一 build_intents/classify_intents 实现）",
        "classifier": "scripts.v2_p7.t25.intent_precheck.classify_intents",
        "price_dependency": (
            "本证据只含**价格无关字段**（事件数/股数/抵消/净额/逐日分布）；"
            "名义额与 T2.5 的估计额同源困难（本模块不给价格表），故**不输出**名义额，"
            "实际成交金额见 trades.parquet。"
        ),
        "notional_omitted": True,
        "scenarios": {
            name: {
                "events": int(values.get("events", 0)),
                "shares": int(values.get("shares", 0)),
                "offset_shares": int(values.get("offset_shares", 0)),
                "net_shares": int(values.get("net_shares", 0)),
            }
            for name, values in scenarios.items()
        },
        "natural_counts": natural,
        "frozen_counts": dict(frozen_counts),
        "matches_frozen_counts": bool(frozen_ok),
        "matches_t25_precheck": bool(
            all(item["match"] for item in compares) and day_scenario_match
        ),
        "compare_rows": compares,
        "day_scenario_counts_match": day_scenario_match,
        "missing_scenarios": missing,
        "s1_status": "observed_with_registered_gap" if natural["S1"] == 0 else "observed",
        "instances": instances,
        "note": (
            "S1（严格等量反向）自然计数为 0，按用户 2026-11-12 裁决登记为已登记缺口，"
            "由构造黄金例覆盖并标 synthetic；不得合并计数、不得声称自然验证。"
        ),
    }


def _natural_instances(
    records: Sequence[Mapping[str, Any]], arms: Sequence[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """按场景输出自然实例清单（逐笔可复算：日期/股票/分袖股数/抵消/净额）。"""
    out: Dict[str, List[Dict[str, Any]]] = {name: [] for name in NATURAL_SCENARIOS}
    for item in records:
        scenario = str(item["scenario"])
        if scenario not in out:
            continue
        out[scenario].append(
            {
                "trade_date": str(item["trade_date"]),
                "ts_code": str(item["ts_code"]),
                "sleeves": {
                    str(arm): {
                        "buy": int(item.get(f"{arm}_buy_shares", 0)),
                        "sell": int(item.get(f"{arm}_sell_shares", 0)),
                    }
                    for arm in arms
                },
                "reverse_sell_shares": int(item.get("reverse_sell_shares", 0)),
                "reverse_buy_shares": int(item.get("reverse_buy_shares", 0)),
                "offset_shares": int(item.get("offset_shares", 0)),
                "net_shares": int(item.get("net_shares", 0)),
            }
        )
    return {name: rows for name, rows in out.items()}


# ────────────────────────── 构造例 ──────────────────────────


class CasePrices:
    """构造例用价源（真实历史某一日的开盘/收盘 + 可选的人为状态覆盖）。"""

    def __init__(
        self,
        day: str,
        quotes: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.day = str(day)
        self.quotes = {str(code): dict(values) for code, values in quotes.items()}
        self.blocks: Dict[str, int] = {}

    def begin_day(self, day: str, tracked: Optional[Sequence[str]] = None) -> None:
        del tracked
        self.day = str(day)

    def market_open(self, ts_code: str) -> ExecutionLookup:
        quote = self.quotes.get(str(ts_code))
        if quote is None:
            return ExecutionLookup(None, "row_absent")
        if quote.get("suspended"):
            return ExecutionLookup(None, "suspended")
        price = float(quote.get("open") or 0.0)
        if price <= 0.0:
            return ExecutionLookup(None, "no_valid_open")
        return ExecutionLookup(price, "ok")

    def execution_price(self, ts_code: str, side: str) -> ExecutionLookup:
        available = self.market_open(ts_code)
        if not available.ok:
            return available
        quote = self.quotes[str(ts_code)]
        if side == "buy" and quote.get("limit_up"):
            return ExecutionLookup(None, "limit_up_blocked")
        if side == "sell" and quote.get("limit_down"):
            return ExecutionLookup(None, "limit_down_blocked")
        return available

    def record_block(self, reason: str) -> None:
        self.blocks[reason] = self.blocks.get(reason, 0) + 1

    def valuation(self, ts_code: str, buy_price: float) -> Tuple[float, str, str]:
        quote = self.quotes.get(str(ts_code))
        if quote is None:
            return float(buy_price), "", "buy_price"
        close = float(quote.get("close") or 0.0)
        if close > 0.0:
            return close, self.day, SOURCE_DAY_CLOSE
        return float(buy_price), "", "buy_price"

    def note_valuation(self, ts_code: str, price: float, source: str, price_date: str = "") -> None:
        del ts_code, price, source, price_date


def real_quote_sample(params: Any, day: str, codes: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """从真实 ``clean/daily`` 取指定日指定股票的行情（构造例的历史价源）。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    loader = DataLoader(Storage(root_path=str(params.data_root)))
    frame = loader.load_clean_daily_by_date(day, auto_ensure=False)
    if frame is None or len(frame) == 0:
        raise ScenarioError(f"构造例价源缺件：{day}")
    wanted = set(str(code) for code in codes)
    out: Dict[str, Dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        if str(row.ts_code) not in wanted:
            continue
        out[str(row.ts_code)] = {
            "open": float(row.open),
            "close": float(row.close),
            "limit_up": int(getattr(row, "is_limit_up", 0) or 0),
            "limit_down": int(getattr(row, "is_limit_down", 0) or 0),
            "suspended": int(getattr(row, "is_suspended", 0) or 0),
        }
    return out


def _make_sleeve(
    sleeve_id: str,
    cash: float,
    lots: Sequence[P7Lot] = (),
) -> SleeveState:
    return SleeveState(sleeve_id=sleeve_id, cash=float(cash), credit=0.0, lots=tuple(lots))


def _lot(
    virtual_lot_id: str,
    ts_code: str,
    source_lot_id: str,
    shares: int,
    price: float,
    day: str,
) -> P7Lot:
    return P7Lot(
        virtual_lot_id=virtual_lot_id,
        ts_code=ts_code,
        source_lot_id=source_lot_id,
        shares=int(shares),
        acquired_date=day,
        acquired_price=float(price),
        acquired_fee=0.0,
        buy_date=day,
        due_date="",
        signal_date=day,
        origin_model_version=0,
    )


@dataclass
class CaseResult:
    """单条构造例结果。"""

    name: str
    kind: str
    synthetic: bool
    constructed_on_history: bool
    passed: bool
    observed: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "synthetic": bool(self.synthetic),
            "constructed_on_history": bool(self.constructed_on_history),
            "passed": bool(self.passed),
            "expected": self.expected,
            "observed": self.observed,
            "note": self.note,
        }


def run_constructed_cases(
    *,
    params: Any,
    cost_model: Any,
    sample_day: str,
) -> List[CaseResult]:
    """跑全部构造例（真实历史价源 + 显式人为状态）。"""
    cases: List[CaseResult] = []
    codes = _pick_codes(params, sample_day, count=6)
    quotes = real_quote_sample(params, sample_day, codes)
    if len(quotes) < 3:
        raise ScenarioError(f"构造例可用股票不足：{sorted(quotes)}")
    ordered = sorted(quotes)
    cases.extend(_s1_strict_equal(ordered, quotes, sample_day, cost_model))
    cases.extend(_stamp_tax_and_both_sides(ordered, quotes, sample_day, cost_model))
    cases.append(_s5_proportional_split())
    cases.append(_s4_f_gt_sum_costs())
    cases.append(_audited_waterfall_rejects_1ulp())
    cases.append(_single_sleeve_zero_share())
    cases.append(_tail_order())
    cases.append(_a_zero_cases())
    cases.append(_negative_saving())
    cases.extend(_partial_fill_carry_recovery(ordered, quotes, sample_day, cost_model))
    cases.extend(_cached_valuation_cannot_transfer(ordered, quotes, sample_day, cost_model))
    return cases


def _pick_codes(params: Any, day: str, count: int) -> List[str]:
    """从真实行情里挑若干正常交易（非停牌、开盘为正）的股票。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    loader = DataLoader(Storage(root_path=str(params.data_root)))
    frame = loader.load_clean_daily_by_date(day, auto_ensure=False)
    if frame is None or len(frame) == 0:
        raise ScenarioError(f"构造例取样日无行情：{day}")
    good = frame[
        (frame["open"] > 0)
        & (frame["close"] > 0)
        & (frame.get("is_suspended", 0) == 0)
        & (frame.get("is_limit_up", 0) == 0)
        & (frame.get("is_limit_down", 0) == 0)
    ]
    return [str(code) for code in good["ts_code"].tolist()[:count]]


def _s1_strict_equal(
    codes: Sequence[str], quotes: Mapping[str, Mapping[str, Any]], day: str, cost_model: Any
) -> List[CaseResult]:
    """S1 严格等量黄金例（synthetic）：无市场指令、双方零费用、I1/I2 保持。"""
    from scripts.v2_p7.t4.dual import run_daily_identities
    from scripts.v2_p7.t4.ledger import execute_plan

    code = codes[0]
    price = float(quotes[code]["open"])
    shares = 1000
    sleeves = {
        "v20": _make_sleeve("v20", 100_000.0, [_lot("L1", code, "P1", shares, price, day)]),
        "v10": _make_sleeve("v10", 100_000.0),
    }
    physical = PhysicalBook(
        cash=200_000.0,
        lots={"P1": PhysicalLot("P1", code, shares, day, price, 0.0)},
    )
    prices = CasePrices(day, quotes)
    intents = [
        intent_from_plan_row(
            sleeve_id="v20",
            ts_code=code,
            side="sell",
            shares=shares,
            signal_date=day,
            due_date=day,
        ),
        intent_from_plan_row(
            sleeve_id="v10",
            ts_code=code,
            side="buy",
            shares=shares,
            signal_date=day,
            due_date=day,
        ),
    ]
    plan = plan_day(day, intents, {"v20": {code: shares}, "v10": {}})
    outcome = execute_plan(
        plan=plan,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=cost_model,
        lot_size=100,
    )
    report = run_daily_identities(
        trade_date=day,
        sleeves=outcome.sleeves,
        physical=outcome.physical,
        prices=prices,
        physical_initial=200_000.0,
        sleeve_capital={"v20": 100_000.0, "v10": 100_000.0},
        cumulative_savings={"v20": 0.0, "v10": 0.0},
    )
    fee_book = fee_mod.FeeBook()
    fee_book.add_day(
        day,
        outcome.fee_entries,
        cost_of_by_side={
            "buy": cost_model.calculate_buy_cost,
            "sell": cost_model.calculate_sell_cost,
        },
    )
    independent = fee_book.total_independent_fee
    expected = {
        "scenario": "S1",
        "market_shares": 0,
        "physical_fee": 0.0,
        "internal_offset_shares": shares,
        "independent_fee_positive": True,
        "identities_ok": True,
        "cash_moved_between_sleeves": True,
    }
    observed = {
        "scenario": _only_scenario(outcome.scenario_counts),
        "market_shares": int(sum(outcome.market_filled.values())),
        "physical_fee": float(sum(event.physical_fee for event, _ in outcome.fee_entries)),
        "internal_offset_shares": int(outcome.offset_shares),
        "independent_fee_positive": bool(independent > 0.0),
        "identities_ok": bool(report.ok),
        "cash_moved_between_sleeves": bool(
            outcome.sleeves["v20"].cash > sleeves["v20"].cash
            and outcome.sleeves["v10"].cash < sleeves["v10"].cash
        ),
        "independent_fee": independent,
        "savings": fee_book.savings_by_sleeve(),
    }
    return [
        CaseResult(
            name="s1_strict_equal_synthetic",
            kind="golden",
            synthetic=True,
            constructed_on_history=True,
            passed=_compare(expected, observed),
            observed=observed,
            expected=expected,
            note="S1 严格等量抵消无自然实例（已登记缺口）；本例在真实历史开盘价上构造。",
        )
    ]


def _stamp_tax_and_both_sides(
    codes: Sequence[str], quotes: Mapping[str, Mapping[str, Any]], day: str, cost_model: Any
) -> List[CaseResult]:
    """内部抵消双方独立费用与印花税：卖出侧独立费用必须高于同额买入侧。"""
    code = codes[1] if len(codes) > 1 else codes[0]
    price = float(quotes[code]["open"])
    shares = 100
    amount = shares * price
    buy_cost = float(cost_model.calculate_buy_cost(amount))
    sell_cost = float(cost_model.calculate_sell_cost(amount))
    stamp = float(cost_model.calculate_stamp_tax(amount))
    expected = {
        "sell_cost_gt_buy_cost": True,
        "stamp_tax_positive": True,
        "difference_equals_stamp_tax": True,
    }
    observed = {
        "buy_cost": buy_cost,
        "sell_cost": sell_cost,
        "stamp_tax": stamp,
        "sell_cost_gt_buy_cost": bool(sell_cost > buy_cost),
        "stamp_tax_positive": bool(stamp > 0.0),
        "difference_equals_stamp_tax": bool(abs((sell_cost - buy_cost) - stamp) < 1e-12),
    }
    return [
        CaseResult(
            name="offset_both_sides_independent_fee_and_stamp_tax",
            kind="golden",
            synthetic=True,
            constructed_on_history=True,
            passed=_compare(expected, observed),
            observed=observed,
            expected=expected,
            note="抵消双方各自按视同独立成交留存储蓄；卖出侧独立费用含印花税。",
        )
    ]


def _s5_proportional_split() -> CaseResult:
    """S5 比例分摊（Q3=A）本身：``f_i = F·a_i/A``，与 ``F>Σc`` 无关。"""
    split = fee_mod.allocate_s5_fees(
        intent_shares={"v20": 100, "v10": 900},
        allocated_shares={"v20": 100, "v10": 900},
        price=1.0,
        physical_fee=5.0,
    )
    expected = {"sum_equals_f": True, "v20_fee": 0.5, "v10_fee": 4.5}
    observed = {
        "sum_equals_f": bool(split.total_fee == 5.0),
        "v20_fee": split.fee_of("v20"),
        "v10_fee": split.fee_of("v10"),
        "terminal_sleeve": split.terminal_sleeve,
    }
    return CaseResult(
        name="s5_proportional_split",
        kind="golden",
        synthetic=True,
        constructed_on_history=False,
        passed=_compare(expected, observed),
        observed=observed,
        expected=expected,
        note="Q3=A 是**比例分摊**，不是 F>Σc 分支；S5 无独立费用基准。",
    )


def _s4_f_gt_sum_costs() -> CaseResult:
    """S4 比例分支（``F > Σc``）：先按 ``F·a_i/Σa`` 分摊再终端闭合。"""
    constant = lambda _amount: 1.0  # noqa: E731 - 构造用固定成本函数
    waterfall = fee_mod.allocate_s4_fees(
        intent_amounts={"v20": 100.0, "v10": 900.0},
        allocated_amounts={"v20": 100.0, "v10": 900.0},
        physical_fee=5.0,
        buy_cost=constant,
    )
    expected = {
        "proportional_branch": True,
        "sum_equals_f": True,
        "v20_fee": 0.5,
        "v10_fee": 4.5,
    }
    observed = {
        "proportional_branch": bool(waterfall.proportional_branch),
        "sum_equals_f": bool(waterfall.total_fee == 5.0),
        "v20_fee": waterfall.fee_of("v20"),
        "v10_fee": waterfall.fee_of("v10"),
    }
    return CaseResult(
        name="s4_f_gt_sum_costs",
        kind="negative",
        synthetic=True,
        constructed_on_history=False,
        passed=_compare(expected, observed),
        observed=observed,
        expected=expected,
        note="Σ独立费用 < F 时按比例分摊并照常闭合，不因费用高于独立基准而拒绝。",
    )


def find_audited_waterfall_rejection(limit: int = 4000) -> Dict[str, Any]:
    """确定性搜索一个让**受审** ``allocate_fees`` 因 1 ulp 尾差拒绝的样例。

    搜索空间固定（袖份额 × 价格 × 费用），返回首个失败样例与其异常文本；
    未命中则返回 ``{"found": False}``（不伪造）。
    """
    from src.lazybull.v2.common.cost import CostModel
    from src.lazybull.v2.core.accounting.settlement_kernel import allocate_fees

    cost = CostModel(
        commission_rate=0.0001954, min_commission=5.0, stamp_tax=0.0005, slippage=0.0005
    )
    tried = 0
    for shares_a in (100, 300, 700, 1300, 2100):
        for shares_b in (100, 300, 700, 1300, 2100):
            for price in (3.31, 7.77, 12.34, 27.53, 61.19):
                if tried >= limit:
                    return {"found": False, "tried": tried}
                tried += 1
                intent_amounts = {
                    "v20": shares_a * price,
                    "v10": shares_b * price,
                }
                allocated_amounts = dict(intent_amounts)
                physical_fee = float(cost.calculate_buy_cost((shares_a + shares_b) * price))
                try:
                    allocate_fees(
                        intent_amounts=intent_amounts,
                        allocated_amounts=allocated_amounts,
                        physical_fee=physical_fee,
                        buy_cost=cost.calculate_buy_cost,
                    )
                except Exception as error:  # noqa: BLE001 - 只记录失败样例
                    return {
                        "found": True,
                        "tried": tried,
                        "intent_amounts": intent_amounts,
                        "allocated_amounts": allocated_amounts,
                        "physical_fee": physical_fee,
                        "error_type": type(error).__name__,
                        "error_text": str(error)[:400],
                    }
    return {"found": False, "tried": tried}


def _audited_waterfall_rejects_1ulp() -> CaseResult:
    """受审 ``allocate_fees`` 在真实数据上因 1 ulp 尾差拒绝 —— 把失败样例落盘（A-02）。"""
    from src.lazybull.v2.common.cost import CostModel
    from src.lazybull.v2.core.accounting.settlement_kernel import allocate_fees

    sample = find_audited_waterfall_rejection()
    if not sample.get("found"):
        return CaseResult(
            name="audited_waterfall_rejects_1ulp",
            kind="evidence",
            synthetic=True,
            constructed_on_history=False,
            passed=True,
            observed={"searched": sample.get("tried", 0), "found": False},
            expected={"found": "informational"},
            note=(
                "本次搜索网格内未复现受审件的 1 ulp 拒绝；P7 侧同公式实现仍在每次闭合时"
                "登记轨迹（fees.json.float_closure），不依赖该复现结论。"
            ),
        )
    cost = CostModel(
        commission_rate=0.0001954, min_commission=5.0, stamp_tax=0.0005, slippage=0.0005
    )
    mine = fee_mod.allocate_s4_fees(
        intent_amounts=sample["intent_amounts"],
        allocated_amounts=sample["allocated_amounts"],
        physical_fee=sample["physical_fee"],
        buy_cost=cost.calculate_buy_cost,
    )
    rejected = False
    try:
        allocate_fees(
            intent_amounts=sample["intent_amounts"],
            allocated_amounts=sample["allocated_amounts"],
            physical_fee=sample["physical_fee"],
            buy_cost=cost.calculate_buy_cost,
        )
    except Exception:  # noqa: BLE001 - 等价性判据：受审件必须在此样例上拒绝
        rejected = True
    expected = {
        "found": True,
        "audited_rejects": True,
        "p7_closes_exactly": True,
        "fee_gap_within_1ulp": True,
    }
    observed = {
        "found": True,
        "audited_rejects": bool(rejected),
        "p7_closes_exactly": bool(mine.total_fee == mine.physical_fee),
        "fee_gap_within_1ulp": bool(
            abs(mine.declared_fee - mine.physical_fee) <= max(abs(mine.declared_fee), 1.0) * 2.3e-16
        ),
        "sample": {
            key: sample[key]
            for key in (
                "intent_amounts",
                "allocated_amounts",
                "physical_fee",
                "error_type",
                "error_text",
            )
        },
        "p7_declared_fee": mine.declared_fee,
        "p7_applied_fee": mine.physical_fee,
        "p7_closure_path": None if mine.closure is None else mine.closure.path,
    }
    return CaseResult(
        name="audited_waterfall_rejects_1ulp",
        kind="evidence",
        synthetic=True,
        constructed_on_history=False,
        passed=_compare(expected, observed),
        observed=observed,
        expected=expected,
        note=(
            "记录受审 allocate_fees 因浮点尾差拒绝的最小样例与异常文本；"
            "P7 侧同公式实现在此样例上仍精确闭合（偏差仅体现为可登记的 <=1 ulp 物理费用）。"
        ),
    )


def _single_sleeve_zero_share() -> CaseResult:
    """单袖零份额不收费（尾单/零成交短路）。"""
    split = fee_mod.allocate_s5_fees(
        intent_shares={"v20": 100, "v10": 300},
        allocated_shares={"v20": 0, "v10": 300},
        price=10.0,
        physical_fee=7.0,
    )
    expected = {"zero_sleeve_fee": 0.0, "other_fee": 7.0, "sum_equals_f": True}
    observed = {
        "zero_sleeve_fee": split.fee_of("v20"),
        "other_fee": split.fee_of("v10"),
        "sum_equals_f": bool(split.total_fee == 7.0),
        "terminal_sleeve": split.terminal_sleeve,
    }
    return CaseResult(
        name="single_sleeve_zero_share",
        kind="negative",
        synthetic=True,
        constructed_on_history=False,
        passed=_compare(expected, observed),
        observed=observed,
        expected=expected,
        note="分得 0 股的袖不得被收费（禁止对零成交额触发最低佣金）。",
    )


def _tail_order() -> CaseResult:
    """尾单：167/167/167 -> 物理成交 500 股按 Q18 分得 167/167/166（复用受审纯函数）。"""
    from src.lazybull.v2.core.accounting.settlement_kernel import allocate_shares

    result = allocate_shares({"a": 167, "b": 167, "c": 167}, 500)
    observed = {item.sleeve_id: int(item.allocated_shares) for item in result}
    expected = {"a": 167, "b": 167, "c": 166}
    return CaseResult(
        name="tail_order_q18",
        kind="golden",
        synthetic=True,
        constructed_on_history=False,
        passed=bool(observed == expected and sum(observed.values()) == 500),
        observed=observed,
        expected=expected,
        note="Q18 股数分摊（1 股单位）复用受审 allocate_shares，不重写。",
    )


def _a_zero_cases() -> CaseResult:
    """``A=0,F=0`` 全零短路；``A=0,F≠0`` 拒绝。"""
    zero = fee_mod.allocate_s5_fees(
        intent_shares={"v20": 100, "v10": 100},
        allocated_shares={"v20": 0, "v10": 0},
        price=10.0,
        physical_fee=0.0,
    )
    rejected = False
    try:
        fee_mod.allocate_s5_fees(
            intent_shares={"v20": 100},
            allocated_shares={"v20": 0},
            price=10.0,
            physical_fee=1.0,
        )
    except fee_mod.FeeError:
        rejected = True
    expected = {"zero_fill_all_zero": True, "nonzero_fee_rejected": True}
    observed = {
        "zero_fill_all_zero": bool(
            zero.zero_fill and zero.total_fee == 0.0 and all(v == 0.0 for _s, v in zero.final_fees)
        ),
        "nonzero_fee_rejected": bool(rejected),
    }
    return CaseResult(
        name="a_zero_f_zero_and_a_zero_f_nonzero",
        kind="negative",
        synthetic=True,
        constructed_on_history=False,
        passed=_compare(expected, observed),
        observed=observed,
        expected=expected,
        note="全零成交且 F=0 返回全零（不调用成本模型）；全零成交且 F≠0 必须拒绝。",
    )


def _negative_saving() -> CaseResult:
    """有符号节省允许为负（``c_i < f_i``），方向正确。"""
    zero_cost = lambda _amount: 0.0  # noqa: E731 - 构造用零成本函数
    split = fee_mod.allocate_s5_fees(
        intent_shares={"v20": 100, "v10": 100},
        allocated_shares={"v20": 100, "v10": 100},
        price=10.0,
        physical_fee=4.0,
    )
    savings = {
        "v20": zero_cost(1000.0) - split.fee_of("v20"),
        "v10": zero_cost(1000.0) - split.fee_of("v10"),
    }
    expected = {"both_savings_negative": True, "sum_equals_negative_f": True}
    observed = {
        "savings": savings,
        "both_savings_negative": bool(all(value < 0.0 for value in savings.values())),
        "sum_equals_negative_f": bool(abs(sum(savings.values()) + 4.0) < 1e-12),
    }
    return CaseResult(
        name="negative_signed_saving",
        kind="negative",
        synthetic=True,
        constructed_on_history=False,
        passed=_compare(expected, observed),
        observed=observed,
        expected=expected,
        note="s_i = c_i - f_i 允许为负；不得截断为 0，也不得因此拒绝。",
    )


def _partial_fill_carry_recovery(
    codes: Sequence[str], quotes: Mapping[str, Mapping[str, Any]], day: str, cost_model: Any
) -> List[CaseResult]:
    """部分成交 + 跨日残差 + 恢复（涨停阻断 -> 次日顺延成交，不重复抵消）。"""
    from scripts.v2_p7.t4.ledger import execute_plan

    code = codes[2] if len(codes) > 2 else codes[0]
    price = float(quotes[code]["open"])
    blocked_quotes = {code: dict(quotes[code], limit_up=True)}
    prices = CasePrices(day, blocked_quotes)
    sleeves = {
        "v20": _make_sleeve("v20", 100_000.0),
        "v10": _make_sleeve("v10", 100_000.0),
    }
    physical = PhysicalBook(cash=200_000.0)
    buy = intent_from_plan_row(
        sleeve_id="v20",
        ts_code=code,
        side="buy",
        shares=1000,
        signal_date=day,
        due_date=day,
    )
    plan1 = plan_day(day, [buy], {"v20": {}, "v10": {}})
    out1 = execute_plan(
        plan=plan1,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=cost_model,
        lot_size=100,
    )
    carried = out1.sleeves["v20"].residuals
    # 次日恢复：真实价源放行
    prices2 = CasePrices(day, quotes)
    plan2 = plan_day(
        day,
        [
            Intent(
                intent_key=item.intent_key,
                sleeve_id=item.sleeve_id,
                ts_code=item.ts_code,
                side=item.side,
                shares=item.shares,
                origin="carry",
                signal_date=item.signal_date,
                due_date=item.due_date,
            )
            for item in carried
        ],
        {"v20": {}, "v10": {}},
    )
    out2 = execute_plan(
        plan=plan2,
        sleeves=out1.sleeves,
        physical=out1.physical,
        prices=prices2,
        cost_model=cost_model,
        lot_size=100,
    )
    filled = sum(row["filled"] for row in out2.conservation_rows if row["sleeve"] == "v20")
    expected = {
        "day1_blocked": True,
        "day1_carry_shares": 1000,
        "day1_filled": 0,
        "day2_filled": 1000,
        "day2_no_residual": True,
        "conservation_ok": True,
    }
    observed = {
        "day1_blocked": bool(out1.blocks.get("limit_up_blocked", 0) >= 1),
        "day1_carry_shares": int(sum(item.shares for item in carried)),
        "day1_filled": int(out1.market_filled.get("buy", 0)),
        "day2_filled": int(filled),
        "day2_no_residual": bool(not out2.sleeves["v20"].residuals),
        "conservation_ok": bool(
            all(
                row["original"] == row["offset"] + row["filled"] + row["voided"] + row["carry"]
                for row in out1.conservation_rows + out2.conservation_rows
            )
        ),
        "day1_reason": [item.reason for item in carried],
        "day2_price": price,
    }
    return [
        CaseResult(
            name="partial_fill_carry_recovery",
            kind="golden",
            synthetic=True,
            constructed_on_history=True,
            passed=_compare(expected, observed),
            observed=observed,
            expected=expected,
            note="涨停阻断不成交 -> 残差顺延 -> 次日真实开盘价成交；残差只承载未成交部分。",
        )
    ]


def _cached_valuation_cannot_transfer(
    codes: Sequence[str], quotes: Mapping[str, Mapping[str, Any]], day: str, cost_model: Any
) -> List[CaseResult]:
    """缓存估值不得完成 B1 转账：无真实开盘价时抵消不落账且残差顺延。"""
    from scripts.v2_p7.t4.ledger import execute_plan

    code = codes[3] if len(codes) > 3 else codes[0]
    price = float(quotes[code]["open"])
    suspended = {code: dict(quotes[code], suspended=True, open=0.0)}
    prices = CasePrices(day, suspended)
    sleeves = {
        "v20": _make_sleeve("v20", 100_000.0, [_lot("L1", code, "P1", 1000, price, day)]),
        "v10": _make_sleeve("v10", 100_000.0),
    }
    physical = PhysicalBook(
        cash=200_000.0, lots={"P1": PhysicalLot("P1", code, 1000, day, price, 0.0)}
    )
    intents = [
        intent_from_plan_row(
            sleeve_id="v20",
            ts_code=code,
            side="sell",
            shares=500,
            signal_date=day,
            due_date=day,
        ),
        intent_from_plan_row(
            sleeve_id="v10",
            ts_code=code,
            side="buy",
            shares=500,
            signal_date=day,
            due_date=day,
        ),
    ]
    plan = plan_day(day, intents, {"v20": {code: 1000}, "v10": {}})
    outcome = execute_plan(
        plan=plan,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=cost_model,
        lot_size=100,
    )
    cached_price, _cached_day, cached_source = prices.valuation(code, price)
    expected = {
        "offset_shares_applied": 0,
        "residual_carried": 1000,
        "sleeve_cash_unchanged": True,
        "valuation_fallback_available": True,
    }
    observed = {
        "offset_shares_applied": int(outcome.offset_shares),
        "residual_carried": int(
            sum(item.shares for state in outcome.sleeves.values() for item in state.residuals)
        ),
        "sleeve_cash_unchanged": bool(
            outcome.sleeves["v20"].cash == sleeves["v20"].cash
            and outcome.sleeves["v10"].cash == sleeves["v10"].cash
        ),
        "valuation_fallback_available": bool(cached_price > 0.0),
        "valuation_fallback_source": cached_source,
        "scenario": _only_scenario(outcome.scenario_counts),
    }
    return [
        CaseResult(
            name="cached_valuation_cannot_complete_transfer",
            kind="negative",
            synthetic=True,
            constructed_on_history=True,
            passed=_compare(expected, observed),
            observed=observed,
            expected=expected,
            note="估值链可用（买价/最后已知价）但转账仍需当日真实开盘价；无价即顺延。",
        )
    ]


def _only_scenario(counts: Mapping[str, int]) -> str:
    names = [name for name, value in counts.items() if int(value) > 0]
    return names[0] if len(names) == 1 else "|".join(sorted(names))


def _compare(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    """逐字段比对（浮点用 1e-12 相对容差，其余精确）。"""
    for key, want in expected.items():
        if key not in observed:
            return False
        got = observed[key]
        if isinstance(want, float) or isinstance(got, float):
            if abs(float(got) - float(want)) > 1e-12 * max(1.0, abs(float(want))):
                return False
        elif isinstance(want, list):
            if list(got) != list(want):
                return False
        elif got != want:
            return False
    return True


def summarize_cases(cases: Sequence[CaseResult]) -> Dict[str, Any]:
    """构造例汇总（``scenarios/constructed.json``）。"""
    rows = [case.payload() for case in cases]
    return {
        "schema": "p7_t4_constructed_cases/v1",
        "total": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "failed": [row["name"] for row in rows if not row["passed"]],
        "synthetic_all": all(row["synthetic"] for row in rows),
        "constructed_on_history": sum(1 for row in rows if row["constructed_on_history"]),
        "cases": rows,
        "note": (
            "构造例只证明机制可被触发并正确记账（synthetic / constructed_on_history），"
            "不冒充自然观测，也不产生任何经济判定。"
        ),
    }
