# -*- coding: utf-8 -*-
"""P7 T4 宿主级回归（桩件驱动，不碰真实数据/证据根）。

覆盖点（对应 T4 复审意见）
=========================
- 跨折连续 + generation 逐日 +1，折号随日历推进（B-09）；
- 按意图身份跨日守恒：``first_original = Σoffset + Σfilled + Σvoided + 末次 carry``（B-04）；
- 检查点多点重启逐位一致、重放用**独立价源**（B-06 / B-17）；
- 逐袖现金非负守卫接线（A-03 / B-03）；
- 同键同代异状态拒绝（B-05）；
- 价源证据含逐股日期与恢复一致性字段（B-01 / B-02）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import pytest

from scripts.v2_p7.t4 import fees as fee_mod
from scripts.v2_p7.t4.host import HistoricalVirtualSleeveHost, build_intent_index
from scripts.v2_p7.t4.netting import (
    Intent,
    NettingError,
    assert_single_direction_precondition,
    cross_day_conservation,
    intent_from_plan_row,
)
from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t4.state import SleeveState, StateEnvelope, StateError, stable_digest
from src.lazybull.v2.common.cost import CostModel

COST = CostModel(commission_rate=0.0001954, min_commission=5.0, stamp_tax=0.0005, slippage=0.0005)
DAYS: Tuple[str, ...] = tuple(f"202401{idx:02d}" for idx in range(2, 32)) + tuple(
    f"202402{idx:02d}" for idx in range(1, 21)
)
CODES = ("000001.SZ", "000002.SZ", "000003.SZ")


class _FrameLoader:
    """按日生成合成 ``clean/daily`` 帧（价格随日线性上涨，无停牌）。"""

    def __init__(self, prices: Mapping[str, float], days: Sequence[str]) -> None:
        self._base = dict(prices)
        self._days = {day: pos for pos, day in enumerate(days)}

    def load_clean_daily_by_date(self, day: str, auto_ensure: bool = False) -> Any:
        del auto_ensure
        pos = self._days.get(str(day))
        if pos is None:
            return None
        rows = []
        for code in CODES:
            base = self._base[code]
            rows.append(
                {
                    "ts_code": code,
                    "open": base,
                    "close": base + 0.01 * pos,
                    "is_suspended": 0,
                    "is_limit_up": 0,
                    "is_limit_down": 0,
                }
            )
        return pd.DataFrame(rows)


class _Fold:
    def __init__(self, split_index: int, signal_days: Sequence[str], model_version: int) -> None:
        self.split_index = split_index
        self.signal_days = tuple(signal_days)
        self.model_version = model_version


class _T25Stub:
    def __init__(self, folds: Sequence[_Fold]) -> None:
        self.folds = tuple(
            (f.split_index, f.signal_days[0], f.signal_days[-1], f.model_version) for f in folds
        )


class _ParamsStub:
    """宿主只读取这些属性（与 ``T4Params`` 的读取面一致）。"""

    def __init__(self, days: Sequence[str], folds: Sequence[_Fold]) -> None:
        self.arms = ("v20", "v10")
        self.sleeve_capital = {"v20": 500_000.0, "v10": 500_000.0}
        self.physical_capital = 1_000_000.0
        self.lot_size = 100
        self.extended_days = tuple(days)
        self.t25 = _T25Stub(folds)
        self._folds = list(folds)

    def split_of(self, day: str) -> int:
        for fold in self._folds:
            if fold.signal_days[0] <= str(day) <= fold.signal_days[-1]:
                return int(fold.split_index)
        return int(self._folds[-1].split_index)

    def model_of(self, day: str) -> int:
        for fold in self._folds:
            if fold.signal_days[0] <= str(day) <= fold.signal_days[-1]:
                return int(fold.model_version)
        return int(self._folds[-1].model_version)


def _build_setup() -> Tuple[_ParamsStub, Dict[str, List[Intent]], Dict[str, List[Intent]], Any]:
    half = len(DAYS) // 2
    folds = [
        _Fold(0, DAYS[:half], 24008),
        _Fold(1, DAYS[half:], 24009),
    ]
    params = _ParamsStub(DAYS, folds)
    buys: Dict[str, List[Intent]] = {}
    sells: Dict[str, List[Intent]] = {}
    # 每 5 个交易日一轮：v20 买 code0、v10 买 code0（S4 同向合并），
    # 第 8 个交易日后 v20 卖 code0（与 v10 买同股 -> S1/S2/S3 抵消）
    for pos, day in enumerate(DAYS[:-10]):
        code = CODES[pos % len(CODES)]
        if pos % 5 == 0:
            for arm in ("v20", "v10"):
                buy = intent_from_plan_row(
                    sleeve_id=arm,
                    ts_code=code,
                    side="buy",
                    shares=1000,
                    signal_date=day,
                    due_date=DAYS[pos + 4],
                    model_version=params.model_of(day),
                )
                buys.setdefault(day, []).append(buy)
                sell = intent_from_plan_row(
                    sleeve_id=arm,
                    ts_code=code,
                    side="sell",
                    shares=1000,
                    signal_date=day,
                    due_date=DAYS[pos + 4],
                    model_version=params.model_of(day),
                )
                sells.setdefault(DAYS[pos + 4], []).append(sell)
    loader = _FrameLoader({"000001.SZ": 10.0, "000002.SZ": 20.0, "000003.SZ": 30.0}, DAYS)
    return params, buys, sells, loader


def _run_host() -> Any:
    params, buys, sells, loader = _build_setup()
    host = HistoricalVirtualSleeveHost(
        params=params,
        run_id="hosttest",
        prices=PriceBook(loader),
        cost_model=COST,
        buy_index=buys,
        sell_index=sells,
        price_factory=lambda: PriceBook(loader),
    )
    return host.run()


def test_host_generation_and_split_progress() -> None:
    """generation 逐日 +1、折号随日历推进、逐日恒等式全过（B-09）。"""
    result = _run_host()
    assert result.envelope.generation == len(DAYS)
    assert len(result.daily_rows) == len(DAYS)
    assert [row["generation"] for row in result.daily_rows] == list(range(1, len(DAYS) + 1))
    splits = {row["split_index"] for row in result.nav_series["v20"]}
    assert splits == {0, 1}
    assert not result.identity_failures
    # 逐袖现金符号是**登记**项（非负门归 T5）；合成数据上应无负值
    assert result.negative_cash_days == 0
    assert result.min_sleeve_cash is not None and result.min_sleeve_cash >= 0.0


def test_host_cross_day_conservation_has_no_repeat_offset() -> None:
    """跨日按意图身份聚合：抵消总额不超过首次原始意图（B-04）。"""
    result = _run_host()
    assert result.conservation["violations"] == 0
    assert result.cross_day["violations"] == 0, result.cross_day["examples"]
    assert result.cross_day["intents"] > 0


def test_host_replay_and_restart_are_bitwise_identical() -> None:
    """重放与多点重启逐位一致，且各自使用独立价源（B-06）。"""
    result = _run_host()
    assert result.replay["checked"] is True
    assert result.replay["all_bitwise_identical"] is True
    assert result.replay["price_book_isolated"] is True
    assert result.restart["checked"] is True
    assert result.restart["checkpoint_count"] >= 2
    assert result.restart["all_bitwise_identical"] is True
    assert result.restart["price_book_isolated"] is True


def test_host_fee_book_closes_and_tracks_float_paths() -> None:
    """费用台账双通道闭合 + 浮点闭合轨迹登记（A-01 / A-06）。"""
    result = _run_host()
    result.fee_book.assert_closed()
    payload = result.fee_book.payload()
    assert payload["closed"] is True
    assert payload["closure_channels"]["running_vs_physical_exact"] is True
    assert abs(payload["closure_channels"]["by_sleeve_vs_physical"]) <= 1e-6
    assert payload["float_closure"]["paths"], "浮点闭合轨迹必须逐笔登记"


def test_host_price_payload_records_dates_and_resume_fields() -> None:
    """价源证据含逐股日期与恢复一致性字段（B-01 / B-02）。"""
    result = _run_host()
    payload = result.price_payload
    assert payload["valuation_dates_missing"] == 0
    assert all(row["last_price_date"] for row in payload["per_stock"])
    assert "bitwise_equal" in payload["resume_consistency"]
    assert payload["resume_consistency"]["mismatch"] == []


# ---------------------------------------------------------------- 纯函数
def test_cross_day_conservation_detects_repeat_offset() -> None:
    """同一意图次日重复抵消（carry 未随抵消递减）必须被抓出。"""
    legit = [
        {
            "trade_date": "20240102",
            "intent_key": "v20|sell|20240102|000001.SZ",
            "original": 1000,
            "offset": 600,
            "filled": 0,
            "voided": 0,
            "carry": 400,
        },
        {
            "trade_date": "20240103",
            "intent_key": "v20|sell|20240102|000001.SZ",
            "original": 400,
            "offset": 400,
            "filled": 0,
            "voided": 0,
            "carry": 0,
        },
    ]
    assert cross_day_conservation(legit)["violations"] == 0
    repeated = [
        legit[0],
        dict(legit[1], carry=400),  # 已抵消却仍把 400 挂账 -> 跨日总量超出首次原意图
    ]
    report = cross_day_conservation(repeated)
    assert report["violations"] == 1
    assert report["examples"][0]["closed_total"] == 1400


def test_single_direction_precondition_rejects_same_sleeve_both_sides() -> None:
    """同袖同股同日双向必须显式拒绝（B-15）。"""
    day = "20240102"
    day_intents = [
        intent_from_plan_row(
            sleeve_id="v20",
            ts_code="000001.SZ",
            side="buy",
            shares=100,
            signal_date=day,
            due_date=day,
        ),
        intent_from_plan_row(
            sleeve_id="v20",
            ts_code="000001.SZ",
            side="sell",
            shares=100,
            signal_date=day,
            due_date=day,
        ),
    ]
    with pytest.raises(NettingError):
        assert_single_direction_precondition(day_intents)


def test_envelope_rejects_replay_with_different_state() -> None:
    """同键同代同指纹但状态本体不同必须拒绝（B-05）。"""
    envelope = StateEnvelope(run_id="x")
    sleeves = {"v20": SleeveState(sleeve_id="v20", cash=100.0, credit=0.0)}
    physical = __import__("scripts.v2_p7.t4.state", fromlist=["PhysicalBook"]).PhysicalBook(
        cash=100.0
    )
    envelope.commit(
        key="k", generation=1, fingerprint="f", sleeves=sleeves, physical=physical, as_of="d"
    )
    other = {"v20": SleeveState(sleeve_id="v20", cash=200.0, credit=0.0)}
    with pytest.raises(StateError):
        envelope.commit(
            key="k", generation=1, fingerprint="f", sleeves=other, physical=physical, as_of="d"
        )


def test_equity_policy_blocks_new_buys_for_nonpositive_sleeve() -> None:
    """``E_i <= 0`` 的袖必须真的被挡住新增买入（复审 A-03 / N-4 / N-5）。"""

    class _Report:
        trade_date = "20240105"
        invalid_sleeves = {"v10": "invalid_nonpositive_equity"}
        total_value = {"v20": 100_000.0, "v10": -1.0}

    params, buys, sells, loader = _build_setup()
    host = HistoricalVirtualSleeveHost(
        params=params,
        run_id="policytest",
        prices=PriceBook(loader),
        cost_model=COST,
        buy_index=buys,
        sell_index=sells,
        price_factory=lambda: PriceBook(loader),
    )
    host._apply_equity_policy(_Report())  # noqa: SLF001 - 定向策略单测
    assert "v10" in host._no_new_buy  # noqa: SLF001 - 同上
    day = sorted(buys)[0]
    intents = host._collect_intents(day, {"v20": _sleeve_stub("v20"), "v10": _sleeve_stub("v10")})
    assert {intent.sleeve_id for intent in intents} == {"v20"}


def _sleeve_stub(sleeve_id: str) -> Any:
    return SleeveState(sleeve_id=sleeve_id, cash=1.0, credit=0.0)


def test_intent_index_requires_unique_keys() -> None:
    """``intent_key`` 含袖身份，跨袖同股同向不冲突。"""
    day = "20240102"
    left = intent_from_plan_row(
        sleeve_id="v20",
        ts_code="000001.SZ",
        side="buy",
        shares=100,
        signal_date=day,
        due_date=day,
    )
    right = intent_from_plan_row(
        sleeve_id="v10",
        ts_code="000001.SZ",
        side="buy",
        shares=100,
        signal_date=day,
        due_date=day,
    )
    assert left.intent_key != right.intent_key
    assert stable_digest({"k": left.intent_key}) != stable_digest({"k": right.intent_key})


def test_intent_index_builder_is_importable() -> None:
    """``build_intent_index`` 供 runner 使用（签名稳定）。"""
    assert callable(build_intent_index)


# ---------------------------------------------------------------- C4-04 续跑
class _ScenarioLoader(_FrameLoader):
    """在基础帧上制造停牌（整行缺失）与涨停（买单阻断）。"""

    def __init__(
        self,
        prices: Mapping[str, float],
        days: Sequence[str],
        *,
        suspended_days: Sequence[str] = (),
        limit_up_days: Sequence[str] = (),
        suspended_code: str = "000002.SZ",
    ) -> None:
        super().__init__(prices, days)
        self._suspended = set(suspended_days)
        self._limit_up = set(limit_up_days)
        self._suspended_code = str(suspended_code)

    def load_clean_daily_by_date(self, day: str, auto_ensure: bool = False) -> Any:
        frame = super().load_clean_daily_by_date(day, auto_ensure)
        if frame is None or len(frame) == 0:
            return frame
        if day in self._suspended:
            frame = frame[frame["ts_code"] != self._suspended_code].reset_index(drop=True)
        if day in self._limit_up:
            frame = frame.copy()
            frame.loc[frame["ts_code"] == "000001.SZ", "is_limit_up"] = 1
        return frame


def _coverage_setup() -> Tuple[Any, Dict[str, List[Intent]], Dict[str, List[Intent]], Any]:
    """跨停牌持仓 + 涨停阻断顺延的最小场景（供重启覆盖用例）。"""
    half = len(DAYS) // 2
    folds = [_Fold(0, DAYS[:half], 24008), _Fold(1, DAYS[half:], 24009)]
    params = _ParamsStub(DAYS, folds)
    buys: Dict[str, List[Intent]] = {}
    sells: Dict[str, List[Intent]] = {}

    def _add(day: str, arm: str, code: str, shares: int, due: str) -> None:
        buys.setdefault(day, []).append(
            intent_from_plan_row(
                sleeve_id=arm,
                ts_code=code,
                side="buy",
                shares=shares,
                signal_date=day,
                due_date=due,
                model_version=params.model_of(day),
            )
        )
        sells.setdefault(due, []).append(
            intent_from_plan_row(
                sleeve_id=arm,
                ts_code=code,
                side="sell",
                shares=shares,
                signal_date=day,
                due_date=due,
                model_version=params.model_of(day),
            )
        )

    # 持仓跨停牌段：第 2 日买入、第 36 日到期（停牌段在 DAYS[20:25]）
    _add(DAYS[1], "v20", "000002.SZ", 1000, DAYS[35])
    # 顺延残差：第 11 日买入当日涨停 -> 阻断顺延到次日
    _add(DAYS[10], "v10", "000001.SZ", 500, DAYS[30])
    loader = _ScenarioLoader(
        {"000001.SZ": 10.0, "000002.SZ": 20.0, "000003.SZ": 30.0},
        DAYS,
        suspended_days=DAYS[20:25],
        limit_up_days=(DAYS[10],),
    )
    return params, buys, sells, loader


def test_restart_continues_from_serialized_envelope_for_multiple_days() -> None:
    """C4-04：新宿主从序列化信封续跑多日，逐日复现账本/费用/NAV/账本投影行。"""
    result = _run_host()
    restart = result.restart
    assert restart["checked"] is True
    assert max(restart["continuation_days"]) >= 2
    assert restart["generation_all_continued"] is True
    assert restart["ledger_all_identical"] is True
    assert restart["fee_all_identical"] is True
    assert restart["nav_all_identical"] is True
    assert restart["daily_row_all_identical"] is True
    for row in restart["checkpoints"]:
        # 原代际与幂等历史必须还原（不是新开 generation=1）
        assert row["restored_generation"] == row["index"]
        assert row["restored_idempotency_keys"] == row["index"]
        assert row["all_bitwise_identical"] is True
        assert row["first_mismatch"] is None
    # 续跑到全窗末日 => 完整费用 payload（含逐键明细）逐位相等
    last = max(restart["checkpoints"], key=lambda row: row["index"])
    assert last["full_fee_payload_identical"] is True
    # C4-04「禁买状态」：同一信封注入禁买集合后，续跑日该袖市场买入必须为 0
    probe = restart["no_new_buy_probe"]
    assert probe["covered"] is True
    rows_with_probe = [row for row in probe["rows"] if row.get("blocked") is not None]
    assert rows_with_probe
    for row in rows_with_probe:
        assert row["baseline_buy_shares"] > 0
        assert row["blocked_buy_shares"] == 0
    # 幂等探针在续跑信封上仍成立
    assert restart["idempotency"]["same_key_same_generation_is_noop"] is True
    assert restart["idempotency"]["cross_generation_rejected"] is True


def test_restart_covers_suspension_and_carry_residual() -> None:
    """C4-04：停牌中重启与顺延残差各自首次命中日必须被检查点覆盖并逐位一致。"""
    params, buys, sells, loader = _coverage_setup()
    host = HistoricalVirtualSleeveHost(
        params=params,
        run_id="coveragerestart",
        prices=PriceBook(loader),
        cost_model=COST,
        buy_index=buys,
        sell_index=sells,
        price_factory=lambda: PriceBook(loader),
    )
    result = host.run()
    restart = result.restart
    assert restart["all_bitwise_identical"] is True
    assert restart["coverage"]["suspension"] is True, "停牌中重启必须被覆盖"
    assert restart["coverage"]["carry_residual"] is True, "顺延残差重启必须被覆盖"
    covered = [row for row in restart["checkpoints"] if any(row["coverage"].values())]
    assert covered, restart["checkpoints"]
    assert all(row["all_bitwise_identical"] for row in covered)
    # 停牌期间的持仓估值必须来自恢复后的 last_known（否则 NAV 会变）
    assert any(row["price_cache_restored"] for row in covered)


def test_restored_host_keeps_no_new_buy_state_and_blocks_new_buys() -> None:
    """C4-04：禁买状态随信封还原，并在续跑日**真的**挡住新增买入。"""
    params, buys, sells, loader = _build_setup()
    warmup = HistoricalVirtualSleeveHost(
        params=params,
        run_id="policyrestart",
        prices=PriceBook(loader),
        cost_model=COST,
        buy_index={},
        sell_index={},
        days=DAYS[:10],
    )
    # 等价于此前某日 E<=0 触发禁买（策略态必须在信封里随代际落盘）
    warmup._no_new_buy.add("v10")  # noqa: SLF001 - 定向策略单测
    warm = warmup.run()
    payload = json.loads(json.dumps(warm.envelope.to_payload(), ensure_ascii=False))
    assert payload["generation"] == len(DAYS[:10])
    assert payload["resume_state"]["no_new_buy"] == ["v10"]

    def _resume(no_new_buy: Sequence[str]) -> Any:
        migrated = json.loads(json.dumps(payload, ensure_ascii=False))
        migrated["resume_state"]["no_new_buy"] = list(no_new_buy)
        host = HistoricalVirtualSleeveHost(
            params=params,
            run_id="policyrestart",
            prices=PriceBook(loader),
            cost_model=COST,
            buy_index=buys,
            sell_index=sells,
            days=DAYS[10:12],
            restore={"envelope": migrated, "fee_book": fee_mod.FeeBook().snapshot()},
            self_check=False,
        )
        return host, host.run()

    blocked_host, blocked = _resume(["v10"])
    open_host, baseline = _resume([])
    assert "v10" in blocked_host._no_new_buy  # noqa: SLF001 - 状态还原断言
    assert "v10" not in open_host._no_new_buy  # noqa: SLF001 - 同上

    def _buy_shares(result: Any) -> int:
        return sum(
            int(split)
            for trade in result.trades
            if trade["kind"] == "market_buy"
            for sleeve, split in trade["sleeves"].items()
            if sleeve == "v10"
        )

    assert _buy_shares(baseline) > 0, "对照组必须有 v10 买入（否则用例无效）"
    assert _buy_shares(blocked) == 0, "禁买状态必须真的挡住续跑日的新增买入"
    # 续跑仍接着原代际推进
    assert baseline.envelope.generation == len(DAYS[:10]) + 2
    assert blocked.envelope.generation == len(DAYS[:10]) + 2
