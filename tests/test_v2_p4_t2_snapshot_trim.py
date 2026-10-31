# -*- coding: utf-8 -*-
"""v2 P4 T2 硬项⑤ 快照裁剪黄金测试（tests/ 全量豁免复杂度）。

黄金命题：**裁剪恢复 vs 完整恢复 ⇒ 次日行为逐位一致**。

``state_store.trim_engine_state_blob``（⑤定稿，2026-10-28 评审处置修订）在快照落盘前
把 blob ``fields["trades"]`` 与 ``fields["holdings_snapshots"]`` 中**日期早于权威
「最后处理日」**的历史条目裁掉（纯日志/旁路历史，恢复不需要；**当日条目保留**——当日
成交是半提交恢复重建当日 trades/nav 投影的持久原料；``portfolio_values`` 保留）。
本测试在合成小区间（22 个交易日、
freq=5 覆盖 4 个完整调仓周期、A/B ↔ C/D 轮换含买卖双侧、holdings 快照开关开）
驱动三臂：

- 连续参照臂：``run_live_continuous``（单引擎跨日存活、无快照往返）——T1 步进
  等价性语义基线；
- 形态 A（完整）：逐日「全新引擎 + export→dumps→loads→restore（完整 blob）」；
- 形态 B（裁剪）：逐日「全新引擎 + export→dumps→**trim_engine_state_blob**
  →loads→restore」——即纸面宿主 commit_day 落盘形态的真实复刻。

判据（逐日比对，逐位一致，实测取精确相等）：
- 成交逐笔：逐日成交内容三臂一致（``_deep_equal``，不经计数代理）；
- 持仓 / 现金 / 组合估值（NAV）：逐日一致，max|Δnav| ≤ 1e-6（理想 = 0）；
- 旁路：execution_attribution_records / completion_stats 终态逐值一致；
  trades/holdings_snapshots 在裁剪臂按 ⑤ 修订口径只留存「落盘日当日」条目——末日
  引擎终态 = 前一日落盘 blob 恢复携带的前日幸存段 + 末日当日新增（= 完整臂末两日段）；
  断言幸存内容与完整臂同段逐位一致（幸存内容不变），完整臂 vs 连续参照全量一致；
- 每日恢复后重导出字节一致（export→restore→export，生产 dumps_blob 口径）。

反例/边界（⑤ 修订后 ``make_fill_key`` 的 seq = ``len(engine.trades)`` 重计基数 =
恢复携带的前日落盘日幸存条目数，而非旧口径的日内 0 基）：
- 跨日不冲突：幂等键形如 ``yyyymmdd|action|stock|seq``——键内含成交日，裁剪臂
  全程所有入账键全局唯一且与成交逐日对齐（``len(applied_keys) == 成交总笔数``）；
  与完整臂的键前缀多重集（``date|action|stock``）一致，仅 seq 段按裁剪后口径
  重计（键身份 = 入账去重，不承载数值语义，登记为合法分叉）；
- 同日崩溃重放：恢复裁剪快照（含末日当日成交条目）后，以**既有键**重放成交 ⇒
  ``ledger.applied_keys`` 拒绝（LedgerError 重复入账）；以同日同股新 seq 键重放 ⇒
  持仓前置校验兜底拒绝（建仓入账前置非法）——两道防线在裁剪后均有效，
  且拒绝对保留的当日成交日志零副作用。

测试隔离硬契约：全合成数据（t05b 基建），零真实 data/ 读取，零落盘。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytest

from scripts.v2_p4.t05b_spike_runner import (
    _DeterministicSignal,
    _make_engine_factory,
    _make_price_data,
)
from scripts.v2_p4.t05b_step_driver import (
    SteppedRunResult,
    _deep_equal,
    _probe,
    compare_runs,
    run_live_continuous,
)
from src.lazybull.v2.core.accounting.engine_state import (
    dumps_blob,
    export_state,
    loads_blob,
    restore_state,
    state_from_blob,
    state_to_blob,
)
from src.lazybull.v2.core.accounting.ledger import BookingEntry, LedgerError
from src.lazybull.v2.core.execution.live_schedule import (
    apply_live_policy,
    init_live_anchor,
    settle_live_day,
)
from src.lazybull.v2.hosts.paper.state_store import trim_engine_state_blob

# ---------------------------------------------------------------------------
# 合成场景（22 个交易日 + 3 个未来日历日；freq=5 覆盖 4 个完整调仓周期）
# ---------------------------------------------------------------------------

_DAYS: List[pd.Timestamp] = list(pd.bdate_range("2026-01-05", periods=22))
_FULL_CALENDAR: List[pd.Timestamp] = _DAYS + list(
    pd.bdate_range(_DAYS[-1] + pd.Timedelta(days=1), periods=3)
)
_STOCKS = ["A", "B", "C", "D"]


def _signal_plan() -> Dict[str, List[Tuple[str, float]]]:
    """每 5 日轮换 AB ↔ CD（驱动调仓日的卖出+买入双侧成交）。"""
    plan: Dict[str, List[Tuple[str, float]]] = {}
    for i, day in enumerate(_DAYS):
        key = day.strftime("%Y%m%d")
        plan[key] = [("A", 0.9), ("B", 0.8)] if (i // 5) % 2 == 0 else [("C", 0.95), ("D", 0.85)]
    return plan


def _make_factory() -> Callable:
    """引擎工厂：freq=5 / K=1 / holdings 快照开关开（裁剪字段确有内容）。"""
    plan = _signal_plan()
    signal = _DeterministicSignal(2, plan, plan[_DAYS[0].strftime("%Y%m%d")])
    return _make_engine_factory(_STOCKS, signal, post_set={"record_holdings_snapshot": True})


def _price_data() -> pd.DataFrame:
    """确定性温和上行价格（行情严格截止于窗口末日；未来日历日零行情）。"""
    n = len(_DAYS)
    closes = {
        s: [10.0 * (1 + 0.1 * j) * (1 + 0.001 * i) for i in range(n)] for j, s in enumerate(_STOCKS)
    }
    return _make_price_data(_DAYS, closes)


class _SteppedBytesRun:
    """逐日字节级快照往返驱动产物（纸面宿主逐日驱动形态的直接复刻）。"""

    def __init__(self) -> None:
        self.engine = None  # 末日引擎（终态比对载体；裁剪臂 trades/holdings = 前日落盘日幸存段 + 末日当日新增）
        self.probes: List = []  # 逐日探针（引擎活体采集，先于落盘裁剪）
        self.day_fills: List[List[Dict]] = []  # 逐日成交内容（裁剪臂的逐笔比对载体）
        self.restore_byte_ok = True  # 每日恢复后重导出逐位一致
        self.trim_evidence: List[Tuple[str, int, int]] = (
            []
        )  # (日期, 裁剪前 trades 数, 裁剪前 holdings 数)
        self.carried_blobs: List[bytes] = []  # 逐日落盘携带 blob（已变换）


def _run_stepped_bytes(
    make_engine: Callable,
    days: List[pd.Timestamp],
    calendar: List[pd.Timestamp],
    price_data: pd.DataFrame,
    anchor_date: pd.Timestamp,
    transform: Callable[[bytes], bytes],
    evidence: Optional[_SteppedBytesRun] = None,
) -> _SteppedBytesRun:
    """逐日「全新引擎 + 字节级快照往返」驱动（live 锚定形态，对齐纸面宿主日驱动）。

    每日：全新引擎 → 接线（wiring 每日重建）→（首日 init_live_anchor / 否则
    loads→restore 上一日落盘 blob）→ 恢复后重导出字节一致断言 →
    apply_live_policy → run_step → settle_live_day → 探针/逐日成交采集 →
    export→dumps→``transform``（形态 A = 恒等；形态 B = trim_engine_state_blob）。
    """
    result = evidence or _SteppedBytesRun()
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    blob_bytes: Optional[bytes] = None
    for date in days:
        engine = make_engine()
        # live 形态复刻旧链「仅空仓提前调仓」语义（同纸面宿主 _new_engine 双保险）
        engine.enable_trailing_early_rebalance = False
        engine._trade_date_index = date_to_idx
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data
        engine._run_total_days = len(days)
        if blob_bytes is None:
            init_live_anchor(engine, anchor_date, calendar)
        else:
            restore_state(engine, state_from_blob(loads_blob(blob_bytes)))
            # 恢复后原样重导出 ⇒ 必须与落盘 blob 逐位一致（含裁剪形态）
            if dumps_blob(state_to_blob(export_state(engine))) != blob_bytes:
                result.restore_byte_ok = False
        day = apply_live_policy(engine, date, calendar)
        engine._signal_dates = day.working
        trades_before = len(engine.trades)
        engine.run_step(date, calendar, price_data)
        settle_live_day(engine, date, day)
        result.probes.append(_probe(engine, date))
        result.day_fills.append([dict(t) for t in engine.trades[trades_before:]])
        full_bytes = dumps_blob(state_to_blob(export_state(engine)))
        if transform is not _identity:
            pre = loads_blob(full_bytes)["fields"]
            result.trim_evidence.append(
                (date.strftime("%Y%m%d"), len(pre["trades"]), len(pre["holdings_snapshots"]))
            )
        blob_bytes = transform(full_bytes)
        result.carried_blobs.append(blob_bytes)
        result.engine = engine
    return result


def _identity(data: bytes) -> bytes:
    """形态 A 变换：恒等（完整 blob 字节直接携带）。"""
    return data


def _prefix_multiset(keys: List[str]) -> List[str]:
    """幂等键前缀多重集（``yyyymmdd|action|stock``，去掉 seq 段）。"""
    return sorted(key.rsplit("|", 1)[0] for key in keys)


# ---------------------------------------------------------------------------
# 模块级三臂驱动（秒级；各测试方法共享只读产物）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def arms() -> Dict:
    """三臂驱动一次，供全部断言共享（连续参照 / 形态 A 完整 / 形态 B 裁剪）。"""
    price_data = _price_data()
    cont_engine, cont_probes, _anchor, _trace = run_live_continuous(
        _make_factory(), _DAYS, _FULL_CALENDAR, price_data, anchor_date=_DAYS[0]
    )
    form_a = _run_stepped_bytes(
        _make_factory(), _DAYS, _FULL_CALENDAR, price_data, _DAYS[0], _identity
    )
    form_b = _run_stepped_bytes(
        _make_factory(), _DAYS, _FULL_CALENDAR, price_data, _DAYS[0], trim_engine_state_blob
    )
    return {
        "cont_engine": cont_engine,
        "cont_probes": cont_probes,
        "form_a": form_a,
        "form_b": form_b,
    }


# ---------------------------------------------------------------------------
# 黄金断言
# ---------------------------------------------------------------------------


class TestTrimmedRestoreGolden:
    """裁剪恢复 vs 完整恢复 vs 连续参照：次日行为逐位一致（T2 硬项⑤）。"""

    def test_full_form_matches_continuous_reference(self, arms):
        """形态 A（完整 blob 往返）≡ 连续参照：compare_runs 全口径（含旁路终态）。"""
        form_a = arms["form_a"]
        stepped_a = SteppedRunResult(
            engine_state=state_to_blob(export_state(form_a.engine)), probes=form_a.probes
        )
        verdict = compare_runs(arms["cont_engine"], arms["cont_probes"], stepped_a)
        assert verdict.passed, f"完整形态 vs 连续参照不一致: {verdict.probe_mismatches}"
        assert verdict.trades_identical
        assert verdict.nav_max_abs_delta <= 1e-6
        assert form_a.restore_byte_ok, "完整形态恢复后重导出字节不一致"

    def test_trimmed_form_day_fills_bit_identical(self, arms):
        """成交逐笔：裁剪臂逐日成交内容与完整臂、连续参照逐位一致。"""
        cont_trades = arms["cont_engine"].trades
        # 连续参照的逐日成交切片（按 trade date 分组，顺序 = 入账序）
        cont_by_day: Dict[str, List[Dict]] = {}
        for trade in cont_trades:
            cont_by_day.setdefault(trade["date"].strftime("%Y%m%d"), []).append(trade)
        fills_a = arms["form_a"].day_fills
        fills_b = arms["form_b"].day_fills
        assert len(fills_a) == len(_DAYS) and len(fills_b) == len(_DAYS)
        total_fills = 0
        for i, date in enumerate(_DAYS):
            key = date.strftime("%Y%m%d")
            expected = cont_by_day.get(key, [])
            total_fills += len(expected)
            for arm_name, arm_fills in (("完整臂", fills_a[i]), ("裁剪臂", fills_b[i])):
                assert len(arm_fills) == len(
                    expected
                ), f"{key} {arm_name}成交笔数 {len(arm_fills)} != 连续参照 {len(expected)}"
                for j, (got, ref) in enumerate(zip(arm_fills, expected)):
                    diff = _deep_equal(got, ref, path=f"{key}.fills[{j}]")
                    assert diff is None, f"{arm_name}成交不一致: {diff}"
        assert total_fills > 0, "场景必须实际产生成交（轮换买卖双侧）"

    def test_trimmed_form_daily_positions_and_nav_identical(self, arms):
        """持仓/现金/组合估值逐日一致；NAV max|Δ| ≤ 1e-6（实测精确相等）。"""
        probes_a = arms["form_a"].probes
        probes_b = arms["form_b"].probes
        probes_c = arms["cont_probes"]
        assert len(probes_a) == len(probes_b) == len(probes_c) == len(_DAYS)
        nav_max_delta = 0.0
        for ref, full, trimmed in zip(probes_c, probes_a, probes_b):
            assert ref.date == full.date == trimmed.date
            for name, left, right in (
                ("完整臂", full, ref),
                ("裁剪臂", trimmed, ref),
            ):
                diff = _deep_equal(left.positions, right.positions, path="positions")
                assert diff is None, f"{ref.date} {name}持仓不一致: {diff}"
                diff = _deep_equal(left.capital, right.capital, path="capital")
                assert diff is None, f"{ref.date} {name}现金不一致: {diff}"
                diff = _deep_equal(
                    left.pending_signal_dates, right.pending_signal_dates, path="pending_signals"
                )
                assert diff is None, f"{ref.date} {name}信号队列不一致: {diff}"
                diff = _deep_equal(
                    left.last_ranked_candidates, right.last_ranked_candidates, path="candidates"
                )
                assert diff is None, f"{ref.date} {name}候选不一致: {diff}"
                assert left.last_signal_date == right.last_signal_date
                assert left.last_rebalance_nav == right.last_rebalance_nav
            nav_max_delta = max(
                nav_max_delta,
                abs(trimmed.portfolio_value - ref.portfolio_value),
                abs(full.portfolio_value - ref.portfolio_value),
            )
        assert nav_max_delta <= 1e-6, f"NAV max|Δ|={nav_max_delta} 超容差"

    def test_trimmed_form_sidecar_terminal_state(self, arms):
        """旁路产物终态：归因/补齐统计三臂逐值一致；trades/holdings 幸存内容一致。"""
        cont = arms["cont_engine"]
        eng_a = arms["form_a"].engine
        eng_b = arms["form_b"].engine
        for attr in ("execution_attribution_records", "completion_stats"):
            for name, eng in (("完整臂", eng_a), ("裁剪臂", eng_b)):
                diff = _deep_equal(getattr(eng, attr), getattr(cont, attr), path=attr)
                assert diff is None, f"{name}旁路产物 {diff}"
        # holdings_snapshots：完整臂 ≡ 连续参照（全量）；裁剪臂按 ⑤ 修订口径只留
        # 「落盘日当日」条目——末日引擎 = 前一日 blob 恢复（含前一日幸存段）+ 末日新增，
        # 终态 = 完整臂末两日段，幸存内容必须与完整臂同段逐位一致
        diff = _deep_equal(eng_a.holdings_snapshots, cont.holdings_snapshots, path="holdings")
        assert diff is None, f"完整臂 holdings_snapshots 与连续参照不一致: {diff}"
        tail_from = _DAYS[-2]
        expected_holdings_tail = [h for h in eng_a.holdings_snapshots if h["date"] >= tail_from]
        assert any(
            h["date"] == _DAYS[-1] for h in expected_holdings_tail
        ), "末日有持仓 ⇒ 完整臂末日必有持仓快照"
        diff = _deep_equal(
            eng_b.holdings_snapshots, expected_holdings_tail, path="holdings[幸存段]"
        )
        assert diff is None, f"裁剪臂幸存 holdings 快照内容不一致: {diff}"
        # trades 同口径：裁剪臂终态 = 完整臂末两日成交段（前日幸存 + 末日当日）
        expected_trades_tail = [t for t in eng_a.trades if t["date"] >= tail_from]
        assert any(t["date"] == _DAYS[-1] for t in expected_trades_tail), "末日为执行日必有成交"
        diff = _deep_equal(eng_b.trades, expected_trades_tail, path="trades[幸存段]")
        assert diff is None, f"裁剪臂幸存成交日志内容不一致: {diff}"

    def test_trim_engaged_and_restore_byte_identity(self, arms):
        """裁剪确实施行（裁掉过真实历史内容）+ 裁剪臂每日恢复重导出逐位一致。"""
        form_b = arms["form_b"]
        assert form_b.restore_byte_ok, "裁剪形态恢复后重导出字节不一致"
        evidence = form_b.trim_evidence
        assert len(evidence) == len(_DAYS)
        days_with_trades = [d for d, n_trades, _h in evidence if n_trades > 0]
        days_with_holdings = [d for d, _t, n_hold in evidence if n_hold > 0]
        assert (
            len(days_with_trades) >= 3
        ), f"裁剪必须裁掉过真实成交日志（成交日 {len(days_with_trades)} < 3）"
        # holdings 快照开关开 ⇒ 有持仓日均有内容（空仓日不记录属既有语义：
        # holdings_snapshot.py:57 空仓早退；本场景仅首日无持仓）
        assert (
            len(days_with_holdings) >= len(_DAYS) - 1
        ), f"holdings 快照几乎逐日有内容（实测 {len(days_with_holdings)}/{len(_DAYS)}）"
        # 落盘携带 blob 逐日均为 ⑤ 修订裁剪形态：trades/holdings 仅含当日条目
        # （历史裁掉、当日保留作半提交恢复原料），portfolio_values 保留
        day_strs = [d.strftime("%Y%m%d") for d in _DAYS]
        carried_trade_counts = {}
        for date, blob_bytes in zip(day_strs, form_b.carried_blobs):
            fields = loads_blob(blob_bytes)["fields"]
            for name in ("trades", "holdings_snapshots"):
                for entry in fields[name]:
                    entry_date = pd.Timestamp(entry["date"]).strftime("%Y%m%d")
                    assert (
                        entry_date == date
                    ), f"{date} 落盘 blob {name} 含历史条目（{entry_date}），裁剪未按 ⑤ 修订口径施行"
            carried_trade_counts[date] = len(fields["trades"])
            assert fields["portfolio_values"], f"{date} portfolio_values 必须保留（最后处理日口径）"
        # 历史确实被裁：恢复携带的前日幸存段（前日为成交日时非空）在次日落盘时被裁掉
        cut_days = [date for date, n_pre, _h in evidence if carried_trade_counts[date] < n_pre]
        assert cut_days, "必须有交易日的历史成交条目被实际裁掉（裁前含恢复携带的前日段）"

    def test_fill_key_seq_reset_no_cross_day_collision(self, arms):
        """seq 重计的跨日安全性：键含成交日 ⇒ 全程唯一且与成交逐日对齐。"""
        eng_a = arms["form_a"].engine
        eng_b = arms["form_b"].engine
        keys_a = list(eng_a.ledger.export_state()["applied_keys"])
        keys_b = list(eng_b.ledger.export_state()["applied_keys"])
        day_fills_b = arms["form_b"].day_fills
        n_fills = sum(len(fills) for fills in day_fills_b)
        # 全局唯一且与成交一一对应（dict 结构上重复键会静默覆盖 ⇒ 数量守恒即无冲突证据）
        assert len(keys_b) == len(set(keys_b)) == n_fills
        assert len(keys_a) == n_fills
        # 键前缀多重集（date|action|stock）两臂一致；seq 段合法分叉（登记）
        assert _prefix_multiset(keys_b) == _prefix_multiset(keys_a)
        if keys_a != keys_b:  # 有跨日历史成交后，裁剪臂 seq 重计必与完整臂分叉
            assert any(
                ka.rsplit("|", 1)[0] == kb.rsplit("|", 1)[0] and ka != kb
                for ka, kb in zip(sorted(keys_a), sorted(keys_b))
            )
        # 逐日对齐：裁剪臂每日新入账键的日期段 == 当日成交日
        ledger_keys_ordered = keys_b  # applied_keys 保序（入账序）
        cursor = 0
        for date, fills in zip(_DAYS, day_fills_b):
            day_key = date.strftime("%Y%m%d")
            for fill in fills:
                key = ledger_keys_ordered[cursor]
                cursor += 1
                assert key.startswith(
                    f"{day_key}|{fill['action']}|{fill['stock']}|"
                ), f"幂等键与成交日/股票错位: {key} vs {fill}"
        assert cursor == len(ledger_keys_ordered)


class TestSameDayReplayRejection:
    """反例/边界：同日崩溃重放的幂等防线（⑤ 修订后快照保留末日当日成交，防线仍有效）。"""

    def _restored_engine(self, arms):
        """自裁剪臂末日落盘 blob 恢复的全新引擎（含全部 applied_keys 与末日当日成交
        ——⑤ 修订口径当日条目保留，作半提交恢复原料）。"""
        form_b = arms["form_b"]
        engine = _make_factory()()
        engine.enable_trailing_early_rebalance = False
        restore_state(engine, state_from_blob(loads_blob(form_b.carried_blobs[-1])))
        return engine, form_b

    def test_existing_key_replay_rejected(self, arms):
        """以既有幂等键重放 ⇒ applied_keys 拒绝（LedgerError 重复入账）。"""
        engine, form_b = self._restored_engine(arms)
        applied = list(engine.ledger.export_state()["applied_keys"])
        assert applied, "前提：末日快照已含入账键"
        # ⑤ 修订前置：末日当日成交保留在快照内（= 裁剪臂末日逐日成交段）
        assert len(engine.trades) == len(form_b.day_fills[-1]) > 0
        trades_before = [dict(t) for t in engine.trades]
        last_key = applied[-1]
        day_text, action, stock, _seq = last_key.split("|")
        date = pd.Timestamp(day_text)
        entry = BookingEntry(
            fill_key=last_key,
            date=date,
            stock=stock,
            action=action,
            cash_delta=-1.0,
            trade_record={"date": date, "stock": stock, "action": action},
            new_position={"shares": 100, "buy_date": date},
        )
        with pytest.raises(LedgerError, match="重复入账"):
            engine.ledger.apply_fills(engine, [entry])
        # 拒绝零副作用（对保留的当日成交日志同样零副作用）
        assert [dict(t) for t in engine.trades] == trades_before
        assert engine.current_capital == form_b.engine.current_capital

    def test_fresh_seq_key_blocked_by_position_precondition(self, arms):
        """同日同股新 seq 键重放（键级绕过）⇒ 持仓前置校验兜底拒绝（建仓已有持仓）。"""
        engine, form_b = self._restored_engine(arms)
        held = sorted(engine.positions)
        assert held, "前提：末日有持仓（末日 = 轮换执行日，A/B 在仓）"
        assert len(engine.trades) == len(form_b.day_fills[-1]) > 0, "前提：末日当日成交保留"
        stock = held[0]
        date = _DAYS[-1]
        forged_key = f"{date.strftime('%Y%m%d')}|buy|{stock}|99999"  # 既有键之外的 seq
        assert forged_key not in engine.ledger.export_state()["applied_keys"]
        capital_before = engine.current_capital
        positions_before = {k: dict(v) for k, v in engine.positions.items()}
        trades_before = [dict(t) for t in engine.trades]
        entry = BookingEntry(
            fill_key=forged_key,
            date=date,
            stock=stock,
            action="buy",
            cash_delta=-1000.0,
            trade_record={"date": date, "stock": stock, "action": "buy"},
            new_position={"shares": 100, "buy_date": date},
        )
        with pytest.raises(LedgerError, match="建仓入账前置非法"):
            engine.ledger.apply_fills(engine, [entry])
        # 兜底拒绝同样零副作用（现金/持仓/当日成交日志均不变）
        assert engine.current_capital == capital_before
        assert engine.positions == positions_before
        assert [dict(t) for t in engine.trades] == trades_before
