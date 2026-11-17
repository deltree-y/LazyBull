"""资金可行续持研究宿主的隔离合成验证，不读取生产配置/价格/模型。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from scripts.v2_p7 import buffer_path_account as account_module
from scripts.v2_p7.buffer_path_account import AccountError, run_account

DIV_COLUMNS = [
    "ts_code",
    "end_date",
    "ann_date",
    "div_proc",
    "stk_div",
    "cash_div",
    "cash_div_tax",
    "record_date",
    "ex_date",
    "pay_date",
    "imp_ann_date",
]


class SyntheticLoader:
    """严格只读合成日截面，并记录自动补齐请求。"""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def load_clean_daily_by_date(self, day: str, auto_ensure: bool = False) -> pd.DataFrame:
        assert not auto_ensure
        return self.frames[day].copy()


def _inputs(tmp_path: Path, n: int = 90, active: int = 70) -> Any:
    """创建 40 股、固定报价、完整合法排名与可独立重建的空公司行为簿。"""
    days = tuple(pd.bdate_range("20200101", periods=n).strftime("%Y%m%d"))
    codes = [f"{i:06d}.SZ" for i in range(40)]
    frames = {}
    for day in days:
        frames[day] = pd.DataFrame(
            {
                "trade_date": day,
                "ts_code": codes,
                "open": 10.0,
                "close": 10.0,
                "close_adj": 10.0,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
        )
    dividends = tmp_path / "raw" / "dividend"
    dividends.mkdir(parents=True)
    pd.DataFrame(columns=DIV_COLUMNS).to_parquet(dividends / "2020-12-31.parquet", index=False)
    ranked = {day: [(code, float(40 - i)) for i, code in enumerate(codes)] for day in days[:active]}
    return SimpleNamespace(
        calendar=days,
        signal_days=days[:active],
        schedule={days[i]: (i // 10) % 2 for i in range(0, active, 10)},
        ranked=ranked,
        variance_book=None,
        loader=SyntheticLoader(frames),
        params=SimpleNamespace(data_root=tmp_path, model_version_of_day=lambda _day: 24008),
    )


def _write_event(inputs: Any, *, ex: int, pay: int, ratio: float, cash: float) -> None:
    """为首只股票登记税后现金与送转，事件在前期已公告。"""
    days = inputs.calendar
    event = dict(zip(DIV_COLUMNS, [None] * len(DIV_COLUMNS)))
    event.update(
        {
            "ts_code": "000000.SZ",
            "end_date": "20191231",
            "ann_date": days[0],
            "div_proc": "实施",
            "stk_div": ratio,
            "cash_div": cash,
            "cash_div_tax": cash,
            "record_date": days[ex - 1],
            "ex_date": days[ex],
            "pay_date": days[pay],
            "imp_ann_date": days[0],
        }
    )
    pd.DataFrame([event]).to_parquet(
        inputs.params.data_root / "raw" / "dividend" / "2020-12-31.parquet",
        index=False,
    )
    for day in days[ex:]:
        frame = inputs.loader.frames[day]
        frame.loc[frame.ts_code.eq("000000.SZ"), ["open", "close", "close_adj"]] = (10.0 - cash) / (
            1.0 + ratio
        )


def test_repeated_renewal_keeps_buy_date_and_runoff_exits(tmp_path: Path) -> None:
    """同股至少续持三次，原买入时钟保持，runoff 到期后执行且没有新增续持。"""
    inputs = _inputs(tmp_path)
    result = run_account(inputs, 1_000_000, True, "candidate")
    decisions = [
        row
        for row in result["decisions"]
        if row["kind"] == "expiry" and row["ts_code"] == "000000.SZ"
    ]
    assert [row["holding_days"] for row in decisions] == [19, 39, 59, 79]
    assert [row["renew"] for row in decisions] == [True, True, True, False]
    assert {row["buy_date"] for row in decisions} == {inputs.calendar[1]}
    sells = [
        row
        for row in result["trades"]
        if row["kind"] == "market_sell" and row["ts_code"] == "000000.SZ"
    ]
    assert len(sells) == 1 and sells[0]["trade_date"] == inputs.calendar[81]
    assert result["validation"]["initial_nav_is_one"]
    assert result["validation"]["terminal_holdings"] == []
    assert result["validation"]["maximum_holding_count"] == 20
    assert result["validation"]["minimum_cash"] >= 0


def test_baseline_aa_and_buy_budget_do_not_read_execution_close(tmp_path: Path) -> None:
    """独立价格簿重复运行逐位一致；T+1 close 巨变不改变此前冻结订单成交。"""
    inputs = _inputs(tmp_path, n=8, active=7)
    first = run_account(inputs, 1_000_000, False, "base")
    second = run_account(inputs, 1_000_000, False, "aa")
    for key in ("daily", "trades", "holdings", "decisions", "blocked", "corporate"):
        assert first[key] == second[key]
    inputs.loader.frames[inputs.calendar[1]]["close"] = 1000.0
    changed = run_account(inputs, 1_000_000, False, "changed_close")
    assert first["trades"] == changed["trades"]
    assert first["daily"][0]["nav"] == changed["daily"][0]["nav"] == 1.0
    assert first["daily"][1]["nav"] != changed["daily"][1]["nav"]


def test_buy_block_cancels_without_chasing_and_cashguard_includes_fees(tmp_path: Path) -> None:
    """涨停买单只记取消；小账户被实际含费现金缩量且永不借款。"""
    inputs = _inputs(tmp_path, n=8, active=7)
    inputs.loader.frames[inputs.calendar[1]].loc[0, "is_limit_up"] = True
    result = run_account(inputs, 20_000, False, "small")
    assert not any(row["ts_code"] == "000000.SZ" for row in result["trades"])
    blocks = [row for row in result["blocked"] if row["ts_code"] == "000000.SZ"]
    assert len(blocks) == 1 and blocks[0]["handling"] == "cancel"
    assert blocks[0]["reason"] == "limit_up_blocked"
    assert result["validation"]["minimum_cash"] >= 0
    assert all(row["shares"] % 100 == 0 for row in result["trades"])
    buy_cash = sum(row["amount"] + row["fee"] for row in result["trades"])
    assert result["daily"][-1]["cash"] == pytest.approx(20_000 - buy_cash)


def test_pending_sell_cannot_be_withdrawn_by_rank_recovery(tmp_path: Path) -> None:
    """到期出 Top20 后虽次日回升，跌停受阻义务仍每日重试，保持批次槽位上限。"""
    inputs = _inputs(tmp_path, n=35, active=30)
    ranks = inputs.ranked[inputs.calendar[20]]
    inputs.ranked[inputs.calendar[20]] = ranks[1:] + ranks[:1]
    for index in (21, 22):
        inputs.loader.frames[inputs.calendar[index]].loc[0, "is_limit_down"] = True
    result = run_account(inputs, 1_000_000, True, "candidate")
    sells = [
        row
        for row in result["trades"]
        if row["kind"] == "market_sell" and row["ts_code"] == "000000.SZ"
    ]
    assert len(sells) == 1 and sells[0]["trade_date"] == inputs.calendar[23]
    blocks = [row for row in result["blocked"] if row["side"] == "sell"]
    assert [row["trade_date"] for row in blocks] == [inputs.calendar[21], inputs.calendar[22]]
    assert any(row["reason"] == "tranche_slots_occupied" for row in result["blocked"])
    grouped: dict[tuple[str, int], int] = {}
    for row in result["holdings"]:
        key = (row["trade_date"], row["tranche_idx"])
        grouped[key] = grouped.get(key, 0) + 1
    assert max(grouped.values()) <= 10


def test_ex_day_sale_uses_bonus_shares_and_dividend_stays_receivable(tmp_path: Path) -> None:
    """公司行为与到期卖同日，出售全部送转后股份；现金分红支付前只计应收。"""
    inputs = _inputs(tmp_path, n=28, active=25)
    _write_event(inputs, ex=21, pay=24, ratio=0.4, cash=1.0)
    result = run_account(inputs, 1_000_000, False, "base")
    buy = next(
        row
        for row in result["trades"]
        if row["kind"] == "market_buy" and row["ts_code"] == "000000.SZ"
    )
    sell = next(
        row
        for row in result["trades"]
        if row["kind"] == "market_sell" and row["ts_code"] == "000000.SZ"
    )
    assert sell["trade_date"] == inputs.calendar[21]
    assert sell["shares"] == int(buy["shares"] * 1.4)
    assert result["daily"][21]["receivable"] == buy["shares"]
    assert result["daily"][23]["receivable"] == buy["shares"]
    assert result["daily"][24]["receivable"] == 0
    assert result["daily"][24]["cash"] - result["daily"][23]["cash"] == buy["shares"]
    assert result["validation"]["identity_violations"] == 0


def test_missing_rank_rejected_and_terminal_blocked_obligation_preserved(tmp_path: Path) -> None:
    """预期排名缺日硬失败；末端停牌卖不掉的义务不伪造平仓。"""
    inputs = _inputs(tmp_path, n=24, active=21)
    missing_day = inputs.calendar[5]
    saved = inputs.ranked.pop(missing_day)
    with pytest.raises(AccountError, match="排名缺日"):
        run_account(inputs, 1_000_000, False, "bad")
    inputs.ranked[missing_day] = saved
    for day in inputs.calendar[21:]:
        inputs.loader.frames[day].loc[0, "is_suspended"] = True
    result = run_account(inputs, 1_000_000, False, "base")
    pending = result["validation"]["terminal_pending_sells"]
    assert "000000.SZ" in pending
    assert pending["000000.SZ"]["signal_date"] == inputs.calendar[20]
    assert any(
        row["ts_code"] == "000000.SZ" and row["pending_exit"]
        for row in result["validation"]["terminal_holdings"]
    )


def test_cashguard_reduces_nominal_order_to_fee_affordable_lots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """隔离现金守卫：人工提高合成费用，成交仍必须为含费可负担整手。"""
    inputs = _inputs(tmp_path, n=3, active=2)

    def oversized_plan(*_args: Any) -> dict[str, Any]:
        return {
            "new_buys": [{"ts_code": "000000.SZ", "score": 1.0, "weight": 1.0, "budget": 5_000.0}],
            "retained_missing_score": [],
        }

    monkeypatch.setattr(account_module, "plan_tranche", oversized_plan)
    cost_factory = account_module.CostModel
    monkeypatch.setattr(
        account_module,
        "CostModel",
        lambda **_kwargs: cost_factory(
            commission_rate=0,
            min_commission=6000,
            stamp_tax=0,
            slippage=0,
        ),
    )
    result = run_account(inputs, 10_000, False, "cash_guard")
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["shares"] == 400
    assert trade["amount"] + trade["fee"] <= 10_000
    assert (trade["shares"] + 100) * trade["price"] + trade["fee"] > 10_000
    assert result["daily"][1]["cash"] == 10_000 - trade["amount"] - trade["fee"]
    assert result["blocked"][0]["reason"] == "insufficient_cash"
    assert result["blocked"][0]["unfilled_shares"] == 100
    assert result["blocked"][0]["handling"] == "cancel"


def test_due_ranking_uses_current_fold_and_empty_legal_day_does_not_renew(tmp_path: Path) -> None:
    """续持决策记当期模型，不沿用建仓模型；合法空排名日按到期退出。"""
    inputs = _inputs(tmp_path, n=25, active=24)
    boundary = inputs.calendar[20]
    inputs.params.model_version_of_day = lambda day: 24008 if day < boundary else 24009
    inputs.ranked[boundary] = []
    result = run_account(inputs, 1_000_000, True, "candidate")
    decision = next(
        row
        for row in result["decisions"]
        if row["kind"] == "expiry" and row["ts_code"] == "000000.SZ"
    )
    assert decision["model_version"] == 24009
    assert not decision["renew"] and decision["reason"] == "missing_ranking"
    holding = next(
        row
        for row in result["holdings"]
        if row["trade_date"] == boundary and row["ts_code"] == "000000.SZ"
    )
    assert holding["origin_model_version"] == 24008
    assert holding["buy_date"] == inputs.calendar[1]
    sell = next(
        row
        for row in result["trades"]
        if row["kind"] == "market_sell" and row["ts_code"] == "000000.SZ"
    )
    assert sell["trade_date"] == inputs.calendar[21]


def test_blocked_exit_keeps_t0_value_in_tranche_budget(tmp_path: Path) -> None:
    """到期大仓卖出受阻后，其他新买不能吃掉该仓尚未释放的批次额度。"""
    inputs = _inputs(tmp_path, n=23, active=21)
    for day in inputs.calendar[20:]:
        inputs.loader.frames[day].loc[0, "close"] = 40.0
    inputs.loader.frames[inputs.calendar[21]].loc[0, "is_limit_down"] = True
    result = run_account(inputs, 1_000_000, False, "blocked_budget")
    old_value = next(
        row["market_value"]
        for row in result["holdings"]
        if row["trade_date"] == inputs.calendar[20] and row["ts_code"] == "000000.SZ"
    )
    bought = sum(
        row["amount"]
        for row in result["trades"]
        if row["trade_date"] == inputs.calendar[21]
        and row["kind"] == "market_buy"
        and row["tranche_idx"] == 0
    )
    ceiling = 0.5 * result["daily"][20]["total_value"]
    assert old_value + bought <= ceiling + 1e-8
    assert any(row["reason"] == "tranche_budget_occupied" for row in result["blocked"])
    # 额度只能用 T0 快照，受阻当日收盘即使改变百倍也不能改变开盘订单。
    inputs.loader.frames[inputs.calendar[21]].loc[0, "close"] = 4000.0
    changed = run_account(inputs, 1_000_000, False, "different_t1_close")
    assert result["trades"] == changed["trades"]
