# -*- coding: utf-8 -*-
"""P7 T4/T5 公司行为入账回归（现金分红 / 送转 / 同时发生 / 停牌跨事件 / 复牌）。

外部检视实例（值取自真实分红分区，测试用同值小样本重建，不读生产数据）
====================================================================
- ``000937.SZ`` 2022-06-29：每股现金 1.0（税后），登记日持 15,900 股 ⇒ 应收 15,900 元；
- ``002020.SZ`` 2021-05-28：每股送转 0.2 + 现金 0.35，持 2,500 股 ⇒ 3,000 股 + 875 元；
- ``603358.SH`` 2021-06-09：每股送转 0.4 + 现金 0.43，持 900 股 ⇒ 1,260 股 + 387 元；
- ``000539.SZ`` 2015-06-26：每股送转 0.2 + 现金 0.18（税后），停牌期间除权。

覆盖点
======
1. 除息日确认应收、到账日转现金（不是除息日直接进现金）；
2. 送转新股并入**原 lot**（买入日/到期日不重置），物理与虚拟同步，I2 零差；
3. 现金 + 送转同时发生；
4. 停牌跨登记/除权/支付：缺价 carry 按理论除权价调整，权益不机械跳升；复牌回到真实收盘；
5. 状态信封序列化/还原（应收逐笔 + 新股 lot）与「按日一次」幂等。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pytest

from scripts.v2_p7.t4.corporate import (
    CorporateActionBook,
    CorporateError,
    assert_corporate_state,
)
from scripts.v2_p7.t4.dual import run_daily_identities
from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t4.state import (
    P7Lot,
    PhysicalBook,
    PhysicalLot,
    SleeveState,
    physical_from_payload,
    sleeves_from_payload,
    _physical_payload,
    _sleeve_payload,
)

#: 冻结列（与 ``data/raw/dividend`` 一致）
DIV_COLUMNS = [
    "ts_code",
    "end_date",
    "ann_date",
    "div_proc",
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div",
    "cash_div_tax",
    "record_date",
    "ex_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
    "base_date",
    "base_share",
]

#: 外部检视实例（行值与真实分区一致；测试自建，不读生产数据）
EXTERNAL_EVENTS: Tuple[Dict[str, Any], ...] = (
    {
        "ts_code": "000937.SZ",
        "end_date": "20211231",
        "ann_date": "20220617",
        "stk_div": 0.0,
        "cash_div": 1.0,
        "cash_div_tax": 1.0,
        "record_date": "20220628",
        "ex_date": "20220629",
        "pay_date": "20220629",
    },
    {
        "ts_code": "002020.SZ",
        "end_date": "20201231",
        "ann_date": "20210421",
        "stk_div": 0.2,
        "cash_div": 0.35,
        "cash_div_tax": 0.35,
        "record_date": "20210527",
        "ex_date": "20210528",
        "pay_date": "20210528",
    },
    {
        "ts_code": "603358.SH",
        "end_date": "20201231",
        "ann_date": "20210522",
        "stk_div": 0.4,
        "cash_div": 0.43,
        "cash_div_tax": 0.43,
        "record_date": "20210608",
        "ex_date": "20210609",
        "pay_date": "20210609",
    },
    {
        "ts_code": "000539.SZ",
        "end_date": "20141231",
        "ann_date": "20150418",
        "stk_div": 0.2,
        "cash_div": 0.18,
        "cash_div_tax": 0.2,
        "record_date": "20150625",
        "ex_date": "20150626",
        "pay_date": "20150626",
    },
)

#: 运行日历（覆盖四条事件；含跨停牌的 000539 段）
CALENDAR: Tuple[str, ...] = (
    "20150601",
    "20150602",
    "20150605",
    "20150608",
    "20150609",
    "20150610",
    "20150625",
    "20150626",
    "20150629",
    "20150630",
    "20210526",
    "20210527",
    "20210528",
    "20210531",
    "20210607",
    "20210608",
    "20210609",
    "20210610",
    "20220627",
    "20220628",
    "20220629",
    "20220630",
)


def _write_dividends(directory: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "div_proc": "实施",
                "stk_bo_rate": None,
                "stk_co_rate": None,
                "div_listdate": None,
                "imp_ann_date": row["ann_date"],
                "base_date": row["record_date"],
                "base_share": 0.0,
                **row,
            }
            for row in EXTERNAL_EVENTS
        ],
        columns=DIV_COLUMNS,
    )
    path = directory / "2015-12-31.parquet"
    frame.to_parquet(path, index=False)
    return path


class StubLoader:
    """价源装载桩件（按日给出 ``clean/daily`` 帧；停牌日无行）。"""

    def __init__(self, frames: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self.frames = {str(day): list(rows) for day, rows in frames.items()}
        self.loads: Dict[str, int] = {}

    def load_clean_daily_by_date(self, day: str, auto_ensure: bool = False) -> Optional[Any]:
        del auto_ensure
        self.loads[str(day)] = self.loads.get(str(day), 0) + 1
        rows = self.frames.get(str(day))
        if not rows:
            return None
        return pd.DataFrame(
            [
                {
                    "ts_code": row["ts_code"],
                    "open": float(row.get("open", row["close"])),
                    "close": float(row["close"]),
                    "is_suspended": int(row.get("is_suspended", 0)),
                    "is_limit_up": int(row.get("is_limit_up", 0)),
                    "is_limit_down": int(row.get("is_limit_down", 0)),
                }
                for row in rows
            ]
        )


def _book(directory: Path) -> CorporateActionBook:
    return CorporateActionBook(calendar=CALENDAR, dividend_dir=directory, cutoff=CALENDAR[-1])


def _sleeve(cash: float, lots: Sequence[P7Lot], sleeve_id: str = "V20") -> SleeveState:
    return SleeveState(sleeve_id=sleeve_id, cash=cash, credit=0.0, lots=tuple(lots))


def _lot(
    code: str,
    shares: int,
    price: float,
    *,
    buy_date: str,
    source_lot_id: str,
    virtual_lot_id: str = "L1",
    due_date: str = "20991231",
) -> P7Lot:
    return P7Lot(
        virtual_lot_id=virtual_lot_id,
        ts_code=code,
        source_lot_id=source_lot_id,
        shares=int(shares),
        acquired_date=buy_date,
        acquired_price=float(price),
        acquired_fee=0.0,
        buy_date=buy_date,
        due_date=due_date,
        signal_date=buy_date,
        origin_model_version=1,
    )


def _physical(lots: Sequence[PhysicalLot], cash: float = 1_000_000.0) -> PhysicalBook:
    book = PhysicalBook(cash=float(cash))
    for lot in lots:
        book.add_lot(lot)
    return book


def _phys_lot(code: str, shares: int, price: float, *, buy_date: str, lot_id: str) -> PhysicalLot:
    return PhysicalLot(
        source_lot_id=lot_id,
        ts_code=code,
        shares=int(shares),
        buy_date=buy_date,
        buy_price=float(price),
        buy_fee=0.0,
    )


# ---------------------------------------------------------------- 现金分红
def test_cash_dividend_receivable_then_pay(tmp_path: Path) -> None:
    """000937：除息日登记 15,900 元应收，到账日转现金；权益不因除息改变。"""
    _write_dividends(tmp_path)
    book = _book(tmp_path)
    lot = _lot("000937.SZ", 15_900, 10.0, buy_date="20220616", source_lot_id="P1")
    sleeves = {"V20": _sleeve(100_000.0, [lot])}
    physical = _physical([_phys_lot("000937.SZ", 15_900, 10.0, buy_date="20220616", lot_id="P1")])

    rows = book.apply_ex_day(day="20220629", sleeves=sleeves, physical=physical, prices=_NoPrices())
    assert len(rows) == 1
    assert rows[0]["cash_receivable"] == pytest.approx(15_900.0)
    assert sleeves["V20"].receivable == pytest.approx(15_900.0)
    assert sleeves["V20"].cash == pytest.approx(100_000.0)  # 除息日不进现金
    assert physical.receivable == pytest.approx(15_900.0)
    assert_corporate_state(sleeves, physical)

    paid = book.apply_pay_day(day="20220629", sleeves=sleeves, physical=physical)
    assert len(paid) == 1
    assert paid[0]["cash_paid"] == pytest.approx(15_900.0)
    assert sleeves["V20"].cash == pytest.approx(115_900.0)
    assert sleeves["V20"].receivable == 0.0
    assert physical.cash == pytest.approx(1_015_900.0)
    assert physical.receivable == 0.0
    assert_corporate_state(sleeves, physical)


# ---------------------------------------------------------------- 送转
def test_bonus_shares_merge_into_original_lot(tmp_path: Path) -> None:
    """002020：2,500 → 3,000 股（并入原 lot，买入日不变）+ 875 元应收。"""
    _write_dividends(tmp_path)
    book = _book(tmp_path)
    lot = _lot(
        "002020.SZ", 2_500, 20.0, buy_date="20210506", source_lot_id="P2", due_date="20210604"
    )
    sleeves = {"V20": _sleeve(50_000.0, [lot])}
    physical = _physical(
        [_phys_lot("002020.SZ", 2_500, 20.0, buy_date="20210506", lot_id="P2")]
    )

    rows = book.apply_ex_day(day="20210528", sleeves=sleeves, physical=physical, prices=_NoPrices())
    assert len(rows) == 1
    assert rows[0]["bonus_shares"] == 500
    assert rows[0]["shares_after"] == 3_000
    assert sleeves["V20"].shares_of("002020.SZ") == 3_000
    assert physical.shares_of("002020.SZ") == 3_000
    updated = sleeves["V20"].lots[0]
    assert updated.buy_date == "20210506"
    assert updated.due_date == "20210604"
    assert updated.acquired_price == pytest.approx(20.0 / 1.2)
    # 成本总额守恒（floor 无残差时精确）
    assert updated.acquired_cost == pytest.approx(2_500 * 20.0)
    assert sleeves["V20"].receivable == pytest.approx(875.0)
    assert_corporate_state(sleeves, physical)


def test_cash_and_bonus_same_event(tmp_path: Path) -> None:
    """603358：900 → 1,260 股 + 387 元（现金与送转同时发生）。"""
    _write_dividends(tmp_path)
    book = _book(tmp_path)
    lot = _lot("603358.SH", 900, 23.10, buy_date="20210601", source_lot_id="P3")
    sleeves = {"V20": _sleeve(10_000.0, [lot])}
    physical = _physical([_phys_lot("603358.SH", 900, 23.10, buy_date="20210601", lot_id="P3")])

    rows = book.apply_ex_day(day="20210609", sleeves=sleeves, physical=physical, prices=_NoPrices())
    assert rows[0]["bonus_shares"] == 360
    assert rows[0]["cash_receivable"] == pytest.approx(387.0)
    assert sleeves["V20"].shares_of("603358.SH") == 1_260
    assert physical.shares_of("603358.SH") == 1_260
    assert physical.lots["P3"].buy_price == pytest.approx(23.10 / 1.4)
    assert sleeves["V20"].lots[0].acquired_price == pytest.approx(23.10 / 1.4)


def test_bonus_allocation_across_two_sleeves_and_sources(tmp_path: Path) -> None:
    """两袖共享同一物理来源 + 两个来源批次：Σ虚拟新股 == 物理新股，I2 键集不变。"""
    _write_dividends(tmp_path)
    book = _book(tmp_path)
    sleeves = {
        "V20": _sleeve(
            0.0,
            [
                _lot(
                    "002020.SZ", 1_000, 20.0, buy_date="20210506",
                    source_lot_id="A", virtual_lot_id="a1",
                ),
                _lot(
                    "002020.SZ", 500, 21.0, buy_date="20210510",
                    source_lot_id="B", virtual_lot_id="b1",
                ),
            ],
            sleeve_id="V20",
        ),
        "V10": _sleeve(
            0.0,
            [
                _lot(
                    "002020.SZ", 500, 22.0, buy_date="20210512",
                    source_lot_id="A", virtual_lot_id="a2",
                ),
            ],
            sleeve_id="V10",
        ),
    }
    physical = _physical(
        [
            _phys_lot("002020.SZ", 1_500, 20.0, buy_date="20210506", lot_id="A"),
            _phys_lot("002020.SZ", 500, 21.0, buy_date="20210510", lot_id="B"),
        ]
    )
    book.apply_ex_day(day="20210528", sleeves=sleeves, physical=physical, prices=_NoPrices())
    assert physical.shares_of("002020.SZ") == 2_400
    virtual_total = sum(state.shares_of("002020.SZ") for state in sleeves.values())
    assert virtual_total == 2_400
    assert physical.shares_by_source() == {
        ("002020.SZ", "A"): 1_800,
        ("002020.SZ", "B"): 600,
    }
    merged: Dict[Tuple[str, str], int] = {}
    for state in sleeves.values():
        for key, value in state.shares_by_source().items():
            merged[key] = merged.get(key, 0) + value
    assert merged == physical.shares_by_source()


# ---------------------------------------------------------------- 停牌跨事件
class _NoPrices:
    """无行情价源的桩件（只验证除权调整被调用，不做估值）。"""

    def __init__(self) -> None:
        self.adjustments: list = []

    def apply_ex_adjustment(
        self, ts_code: str, *, cash_per_share: float, share_ratio: float, ex_date: str
    ) -> None:
        self.adjustments.append(
            {
                "ts_code": ts_code,
                "cash": cash_per_share,
                "ratio": share_ratio,
                "ex_date": ex_date,
            }
        )


def test_suspended_carry_is_ex_adjusted_and_resume_uses_real_close(tmp_path: Path) -> None:
    """000539：停牌跨除权/到账，缺价 carry 按理论除权价调整；复牌回到真实收盘。"""
    _write_dividends(tmp_path)
    book = _book(tmp_path)
    loader = StubLoader(
        {
            "20150625": [{"ts_code": "000539.SZ", "close": 14.53}],
            # 6/26 起停牌：无行；7/21 复牌真实收盘 11.02（未复权）
            "20150629": [],
            "20150630": [],
            "20220629": [],
        }
    )
    prices = PriceBook(loader)
    lot = _lot("000539.SZ", 4_800, 12.0, buy_date="20150410", source_lot_id="S1")
    sleeves = {"V20": _sleeve(10_000.0, [lot])}
    physical = _physical([_phys_lot("000539.SZ", 4_800, 12.0, buy_date="20150410", lot_id="S1")])

    prices.begin_day("20150625", tracked=("000539.SZ",))
    before = _value(sleeves, prices)
    book.apply_ex_day(day="20150626", sleeves=sleeves, physical=physical, prices=prices)
    prices.begin_day("20150626", tracked=("000539.SZ",))
    after = _value(sleeves, prices)
    # 4,800 × 14.53 = 69,744 → 5,760 × ((14.53 − 0.18) / 1.2) + 864 应收
    assert sleeves["V20"].shares_of("000539.SZ") == 5_760
    assert sleeves["V20"].receivable == pytest.approx(864.0)
    assert after == pytest.approx(before, abs=1e-6)

    book.apply_pay_day(day="20150626", sleeves=sleeves, physical=physical)
    assert sleeves["V20"].receivable == 0.0
    assert sleeves["V20"].cash == pytest.approx(10_864.0)

    # 停牌期估值仍用调整后的最后已知价（不产生机械增值）
    prices.begin_day("20150629", tracked=("000539.SZ",))
    price, day, source = prices.valuation("000539.SZ", 12.0)
    assert source == "last_known"
    assert day == "20150625"
    assert price == pytest.approx((14.53 - 0.18) / 1.2)
    prices.note_valuation("000539.SZ", price, source, day)
    payload = prices.payload()
    assert payload["suspension_valuation"]["drift"] == 0

    # 复牌：回到真实收盘
    loader.frames["20150630"] = [{"ts_code": "000539.SZ", "close": 11.02}]
    prices.begin_day("20150630", tracked=("000539.SZ",))
    price, day, source = prices.valuation("000539.SZ", 12.0)
    assert (source, day) == ("day_close", "20150630")
    assert price == pytest.approx(11.02)
    assert prices.payload()["resume_consistency"]["mismatch"] == []


def _value(sleeves: Mapping[str, SleeveState], prices: PriceBook) -> float:
    total = 0.0
    for state in sleeves.values():
        total += float(state.cash) + float(state.receivable)
        for lot in state.lots:
            price, _day, _source = prices.valuation(lot.ts_code, lot.acquired_price)
            total += float(lot.shares) * float(price)
    return total


def test_receivable_survives_until_later_pay_date(tmp_path: Path) -> None:
    """除息日确认、支付日晚于除息日：应收跨日存在，到账日才转现金。"""
    frame = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "end_date": "20201231",
                "ann_date": "20210520",
                "div_proc": "实施",
                "stk_div": 0.0,
                "stk_bo_rate": None,
                "stk_co_rate": None,
                "cash_div": 0.5,
                "cash_div_tax": 0.5,
                "record_date": "20210527",
                "ex_date": "20210528",
                "pay_date": "20210607",
                "div_listdate": None,
                "imp_ann_date": "20210520",
                "base_date": "20210527",
                "base_share": 0.0,
            }
        ],
        columns=DIV_COLUMNS,
    )
    frame.to_parquet(tmp_path / "2015-12-31.parquet", index=False)
    book = _book(tmp_path)
    sleeves = {
        "V20": _sleeve(
            0.0, [_lot("600000.SH", 2_000, 10.0, buy_date="20210506", source_lot_id="P1")]
        )
    }
    physical = _physical([_phys_lot("600000.SH", 2_000, 10.0, buy_date="20210506", lot_id="P1")])
    book.apply_ex_day(day="20210528", sleeves=sleeves, physical=physical, prices=_NoPrices())
    assert sleeves["V20"].receivable == pytest.approx(1_000.0)
    assert sleeves["V20"].cash == 0.0
    # 非支付日：不动
    assert book.apply_pay_day(day="20210531", sleeves=sleeves, physical=physical) == []
    assert sleeves["V20"].receivable == pytest.approx(1_000.0)
    paid = book.apply_pay_day(day="20210607", sleeves=sleeves, physical=physical)
    assert [row["cash_paid"] for row in paid] == [pytest.approx(1_000.0)]
    assert sleeves["V20"].cash == pytest.approx(1_000.0)
    assert sleeves["V20"].receivable == 0.0
    assert physical.cash == pytest.approx(1_001_000.0)
    assert physical.receivable == 0.0


def test_commit_copies_physical_book(tmp_path: Path) -> None:
    """信封提交必须复制物理账本：后续原地应收变更不得改写已提交代际。"""
    from scripts.v2_p7.t4.state import StateEnvelope

    _write_dividends(tmp_path)
    book = _book(tmp_path)
    lot = _lot("000937.SZ", 1_000, 10.0, buy_date="20220616", source_lot_id="P1")
    sleeves = {"V20": _sleeve(10_000.0, [lot])}
    physical = _physical(
        [_phys_lot("000937.SZ", 1_000, 10.0, buy_date="20220616", lot_id="P1")]
    )
    envelope = StateEnvelope(run_id="t")
    assert envelope.commit(
        key="k0",
        generation=1,
        fingerprint="f",
        sleeves=sleeves,
        physical=physical,
        as_of="20220628",
    )
    committed = envelope.to_payload()["physical"]
    book.apply_ex_day(day="20220629", sleeves=sleeves, physical=physical, prices=_NoPrices())
    assert physical.receivable == pytest.approx(1_000.0)
    assert envelope.to_payload()["physical"] == committed
    assert envelope.to_payload()["physical"]["receivable"] == 0.0


# ---------------------------------------------------------------- 恒等式
def test_identities_close_with_receivable(tmp_path: Path) -> None:
    """I1（现金）/ I1r（应收）/ I2（股数）/ I3（NAV）逐日闭合。"""
    _write_dividends(tmp_path)
    book = _book(tmp_path)
    loader = StubLoader({"20210528": [{"ts_code": "002020.SZ", "close": 16.5}]})
    prices = PriceBook(loader)
    sleeves = {
        "V20": _sleeve(
            30_000.0,
            [_lot("002020.SZ", 1_200, 20.0, buy_date="20210506", source_lot_id="P1")],
            sleeve_id="V20",
        ),
        "V10": _sleeve(
            20_000.0,
            [_lot("002020.SZ", 800, 20.0, buy_date="20210506", source_lot_id="P2")],
            sleeve_id="V10",
        ),
    }
    physical = _physical(
        [
            _phys_lot("002020.SZ", 1_200, 20.0, buy_date="20210506", lot_id="P1"),
            _phys_lot("002020.SZ", 800, 20.0, buy_date="20210506", lot_id="P2"),
        ],
        cash=50_000.0,
    )
    book.apply_ex_day(day="20210528", sleeves=sleeves, physical=physical, prices=prices)
    prices.begin_day("20210528", tracked=("002020.SZ",))
    report = run_daily_identities(
        trade_date="20210528",
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        physical_initial=50_000.0,
        sleeve_capital={"V20": 30_000.0, "V10": 20_000.0},
        cumulative_savings={},
    )
    assert report.ok
    kinds = {row.identity for row in report.identities}
    assert {"I1", "I1b", "I1r", "I2", "I3", "I3b"} <= kinds
    for row in report.identities:
        assert abs(row.diff) <= row.tolerance
    assert sleeves["V20"].receivable == pytest.approx(1_200 * 0.35)
    assert sleeves["V10"].receivable == pytest.approx(800 * 0.35)


# ---------------------------------------------------------------- 序列化 / 幂等
def test_state_payload_round_trip_keeps_receivables_and_bonus(tmp_path: Path) -> None:
    """信封序列化 → 还原：应收逐笔与新股 lot 都可续跑，且再跑同日不重复入账。"""
    _write_dividends(tmp_path)
    book = _book(tmp_path)
    sleeves = {
        "V20": _sleeve(
            1_000.0,
            [_lot("603358.SH", 900, 23.10, buy_date="20210601", source_lot_id="P3")],
        )
    }
    physical = _physical([_phys_lot("603358.SH", 900, 23.10, buy_date="20210601", lot_id="P3")])
    book.apply_ex_day(day="20210609", sleeves=sleeves, physical=physical, prices=_NoPrices())

    restored_sleeves = sleeves_from_payload(_sleeve_payload(sleeves))
    restored_physical = physical_from_payload(_physical_payload(physical))
    assert restored_sleeves["V20"].receivable == pytest.approx(387.0)
    assert restored_sleeves["V20"].receivables[0].receivable_id.startswith("DIV-603358.SH")
    assert restored_sleeves["V20"].shares_of("603358.SH") == 1_260
    assert restored_physical.shares_of("603358.SH") == 1_260
    assert restored_physical.receivable == pytest.approx(387.0)
    assert _sleeve_payload(restored_sleeves) == _sleeve_payload(sleeves)
    assert _physical_payload(restored_physical) == _physical_payload(physical)

    # 按日一次：除息只在 20210609 触发
    assert (
        book.apply_ex_day(
            day="20210610",
            sleeves=restored_sleeves,
            physical=restored_physical,
            prices=_NoPrices(),
        )
        == []
    )
    assert restored_sleeves["V20"].shares_of("603358.SH") == 1_260


def test_record_date_after_ex_is_registered(tmp_path: Path) -> None:
    """记录日晚于除权生效日属数据异常：入账照常但必须逐条登记。"""
    frame = pd.DataFrame(
        [
            {
                "ts_code": "603358.SH",
                "end_date": "20201231",
                "ann_date": "20210522",
                "div_proc": "实施",
                "stk_div": 0.4,
                "stk_bo_rate": None,
                "stk_co_rate": None,
                "cash_div": 0.43,
                "cash_div_tax": 0.43,
                "record_date": "20210610",
                "ex_date": "20210609",
                "pay_date": "20210609",
                "div_listdate": None,
                "imp_ann_date": "20210522",
                "base_date": "20210610",
                "base_share": 0.0,
            }
        ],
        columns=DIV_COLUMNS,
    )
    frame.to_parquet(tmp_path / "2015-12-31.parquet", index=False)
    book = _book(tmp_path)
    sleeves = {
        "V20": _sleeve(
            0.0, [_lot("603358.SH", 100, 10.0, buy_date="20210601", source_lot_id="P1")]
        )
    }
    physical = _physical([_phys_lot("603358.SH", 100, 10.0, buy_date="20210601", lot_id="P1")])
    book.apply_ex_day(day="20210609", sleeves=sleeves, physical=physical, prices=_NoPrices())
    payload = book.payload()
    assert payload["record_date_gap_count"] == 1
    assert payload["record_date_gaps"][0]["ts_code"] == "603358.SH"


def test_missing_physical_lot_raises(tmp_path: Path) -> None:
    """虚拟侧持有但物理来源缺失：fail-fast（不得静默跳过送转）。"""
    _write_dividends(tmp_path)
    book = _book(tmp_path)
    sleeves = {
        "V20": _sleeve(
            0.0, [_lot("002020.SZ", 100, 10.0, buy_date="20210506", source_lot_id="X")]
        )
    }
    physical = _physical([])
    with pytest.raises(CorporateError):
        book.apply_ex_day(day="20210528", sleeves=sleeves, physical=physical, prices=_NoPrices())


# ---------------------------------------------------------------- 真实数据交叉核对
@pytest.mark.parametrize(
    "year,code,ex_date,cash,ratio",
    [
        (2022, "000937.SZ", "20220629", 1.0, 0.0),
        (2021, "002020.SZ", "20210528", 0.35, 0.2),
        (2021, "603358.SH", "20210609", 0.43, 0.4),
    ],
)
def test_real_dividend_partition_matches_external_examples(
    year: int, code: str, ex_date: str, cash: float, ratio: float
) -> None:
    """真实分区交叉核对（缺件即跳过）：外部检视实例的现金与送转比例一致。"""
    path = Path("data/raw/dividend") / f"{year}-12-31.parquet"
    if not path.exists():  # pragma: no cover - 生产数据缺失时跳过
        pytest.skip(f"分红分区缺件：{path}")
    frame = pd.read_parquet(
        path, columns=["ts_code", "div_proc", "stk_div", "cash_div", "cash_div_tax", "ex_date"]
    )
    picked = frame.loc[
        frame["ts_code"].eq(code)
        & frame["div_proc"].astype(str).eq("实施")
        & frame["ex_date"].astype(str).str.replace("-", "", regex=False).eq(ex_date)
    ]
    assert len(picked) == 1
    row = picked.iloc[0]
    assert float(row["cash_div"]) == pytest.approx(cash)
    assert float(row["stk_div"]) == pytest.approx(ratio)
