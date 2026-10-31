# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主状态库（PaperStateStore）契约测试（tests/ 全量豁免复杂度）。

覆盖（对应任务卡测试矩阵）：布局创建 / commit_day 全字段往返 / 原子提交点写序
（权威先行）/ 半提交滞后识别与受控重投影修复（只应用一次）/ 冲突形态 fail-fast /
写锁竞争与 stale 抢占 / SMB 远端只读（写静默跳过 + 4 白名单读 + 其余 miss）/
指令投影六语义 / nav 跳空语义 / 快照裁剪 / 双状态一致性三形态。

测试隔离硬契约：全部用 tmp_path 隔离根 + 合成数据，禁止读真实 data/paper。
"""

import json
import os
import pickle
import time
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.v2.core.accounting.engine_state import ENGINE_STATE_SCHEMA_VERSION
from src.lazybull.v2.core.execution.live_schedule import LiveAnchorState
from src.lazybull.v2.core.execution.pending_order import PendingOrder
from src.lazybull.v2.hosts.paper.state_store import (
    INSTRUCTION_COLUMNS,
    NAV_COLUMNS,
    TRADES_COLUMNS,
    PaperLockError,
    PaperStateError,
    PaperStateStore,
    build_account_projection,
    build_instruction_frame,
    build_ranked_candidates_projection,
    build_rebalance_state_projection,
    build_stop_loss_projection,
    build_strategy_state_projection,
    trim_engine_state_blob,
)

D0 = "20260602"  # T0 信号日
D1 = "20260603"  # T+1 执行日 / 下一处理日


# ---------------------------------------------------------------------------
# 合成数据构造
# ---------------------------------------------------------------------------


def _make_positions() -> dict:
    return {
        "000001.SZ": {
            "shares": 300,
            "buy_date": pd.Timestamp("2026-06-01"),
            "signal_date": pd.Timestamp("2026-05-29"),
            "buy_trade_price": 10.0,
            "buy_pnl_price": 10.5,
            "buy_cost_cash": 3002.085,  # 300×10.0 本金 + 2.085 费用
        }
    }


def _make_blob(
    *,
    cash: float = 496997.915,
    positions: dict = None,
    pv_dates=(D0,),
    trades=None,
    holdings=None,
    ranked=None,
    last_signal_date=None,
    last_rebalance_nav=619000.0,
    live_anchor=None,
    stop_loss_monitor=None,
) -> bytes:
    """构造合成引擎快照字节（全 21 字段；trades/holdings 默认非空以证裁剪）。"""
    fields = {
        "current_capital": cash,
        "positions": _make_positions() if positions is None else positions,
        "pending_signals": {},
        "pending_stop_loss_sells": {},
        "pending_condition_sells": {},
        "_cycle_anchor_idx": 0,
        "portfolio_values": [
            {
                "date": pd.Timestamp(d),
                "portfolio_value": cash + 3000.0,
                "capital": cash,
                "market_value": 3000.0,
            }
            for d in pv_dates
        ],
        "trades": (
            [{"date": pd.Timestamp(D0), "stock": "000001.SZ", "action": "buy"}]
            if trades is None
            else trades
        ),
        "execution_attribution_records": [],
        "holdings_snapshots": (
            [{"date": pd.Timestamp(D0), "snap": 1}] if holdings is None else holdings
        ),
        "unfilled_slots": {},
        "completion_stats": {
            "total_unfilled": 0,
            "total_completed": 0,
            "completion_attempts": 0,
            "total_abandoned": 0,
        },
        "_normalize_log_count": 0,
        "_last_ranked_candidates": [] if ranked is None else ranked,
        "_last_signal_date": last_signal_date,
        "_last_rebalance_nav": last_rebalance_nav,
        "_signal_dates": {},
        "pending_order_manager": None,
        "stop_loss_monitor": stop_loss_monitor,
        "ledger": {"applied_keys": [], "last_fill_date": None},
        "live_anchor_state": live_anchor,
    }
    return pickle.dumps(
        {"schema_version": ENGINE_STATE_SCHEMA_VERSION, "fields": fields}, protocol=5
    )


def _commit_kwargs(date: str = D0, **overrides) -> dict:
    """commit_day 全字段合成入参（account 投影与 blob 同源，保证一致性校验 ok）。"""
    positions = _make_positions()
    cash = 496997.915
    kwargs = dict(
        date=date,
        engine_state_blob=_make_blob(cash=cash, positions=positions, pv_dates=[date]),
        account=build_account_projection(cash, positions, date),
        strategy_state=build_strategy_state_projection(619000.0),
        ranked_candidates=build_ranked_candidates_projection(
            date, [("000001.SZ", 0.9), ("000002.SZ", 0.8)]
        ),
        stop_loss_state=build_stop_loss_projection({"000003.SZ": 2}),
        rebalance_state=build_rebalance_state_projection(
            LiveAnchorState(
                anchor_date=pd.Timestamp(date),
                rebalance_freq=20,
                stagger_tranches=1,
                last_scheduled_rebalance_date=pd.Timestamp(date),
                last_rebalance_date=pd.Timestamp(date),
            )
        ),
        instructions=None,
        instructions_date=None,
        pending_buys=[
            {
                "ts_code": "000004.SZ",
                "target_weight": 0.05,
                "reason": "补位-涨停",
                "create_date": date,
                "attempts": 1,
                "last_attempt_date": date,
                "original_signal_date": date,
            }
        ],
        pending_sells=[],
        nav_row={
            "trade_date": date,
            "cash": cash,
            "position_value": 3000.0,
            "total_value": cash + 3000.0,
            "nav": 1.0,
        },
        trade_rows=[
            {
                "trade_date": date,
                "ts_code": "000001.SZ",
                "action": "buy",
                "shares": 300,
                "price": 10.0,
                "amount": 3000.0,
                "commission": 0.585,
                "stamp_tax": 0.0,
                "slippage": 1.5,
                "total_cost": 2.085,
                "reason": "信号生成",
            }
        ],
        run_kind="t0",
        run_record={"trade_date": date, "instructions_count": 0},
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture()
def store(tmp_path):
    return PaperStateStore(root_path=str(tmp_path / "paper"))


# ---------------------------------------------------------------------------
# 1. 布局创建
# ---------------------------------------------------------------------------


class TestLayout:
    def test_seven_subdirs_created(self, store, tmp_path):
        root = tmp_path / "paper"
        for name in (
            "state",
            "trades",
            "nav",
            "runs",
            "pending_buys",
            "pending_sells",
            "instructions",
        ):
            assert (root / name).is_dir(), f"缺子目录 {name}"


# ---------------------------------------------------------------------------
# 2/3. commit_day 全字段往返 + 原子提交点
# ---------------------------------------------------------------------------


class TestCommitDay:
    def test_full_field_roundtrip(self, store, tmp_path):
        kwargs = _commit_kwargs()
        store.commit_day(**kwargs)
        root = tmp_path / "paper"

        # account.json 逐字段一致
        account = json.loads((root / "state/account.json").read_text(encoding="utf-8"))
        assert account == kwargs["account"]
        pos = account["positions"]["000001.SZ"]
        assert pos["buy_cost"] == pytest.approx(2.085)  # 费用额口径（buy_cost_cash − 本金）
        assert pos["buy_price"] == 10.0
        assert pos["buy_date"] == "20260601"
        assert pos["buy_atr_pct"] == 0.0
        assert pos["status"] == "持有"
        assert pos["notes"] == ""

        assert json.loads((root / "state/strategy_state.json").read_text(encoding="utf-8")) == {
            "last_rebalance_nav": 619000.0
        }
        ranked = json.loads((root / "state/ranked_candidates.json").read_text(encoding="utf-8"))
        assert ranked == {
            "signal_date": D0,
            "candidates": [["000001.SZ", 0.9], ["000002.SZ", 0.8]],
        }
        stop_loss = json.loads((root / "state/stop_loss_state.json").read_text(encoding="utf-8"))
        assert stop_loss == {"consecutive_limit_down_days": {"000003.SZ": 2}}
        last_date = json.loads((root / "state/last_trade_date.json").read_text(encoding="utf-8"))
        assert last_date == {"last_trade_date": D0}
        rebalance = json.loads((root / "runs/rebalance_state.json").read_text(encoding="utf-8"))
        assert rebalance == kwargs["rebalance_state"]
        pending_buys = json.loads(
            (root / "pending_buys/pending_buys.json").read_text(encoding="utf-8")
        )
        assert pending_buys == kwargs["pending_buys"]
        assert (
            json.loads((root / "pending_sells/pending_sells.json").read_text(encoding="utf-8"))
            == []
        )

        nav = pd.read_parquet(root / "nav/nav.parquet")
        assert list(nav.columns) == list(NAV_COLUMNS)
        assert len(nav) == 1 and nav.iloc[0]["trade_date"] == D0
        trades = pd.read_parquet(root / "trades/trades.parquet")
        assert list(trades.columns) == list(TRADES_COLUMNS)
        assert len(trades) == 1 and trades.iloc[0]["shares"] == 300

        run_record = json.loads((root / f"runs/t0_{D0}.json").read_text(encoding="utf-8"))
        assert run_record == kwargs["run_record"]
        assert store.check_run_exists("t0", D0)
        assert store.load_run_record("t0", D0) == kwargs["run_record"]
        # 权威快照可读回且 schema 校验通过
        state = store.load_engine_state()
        assert state.current_capital == kwargs["account"]["cash"]
        assert store.get_authority_last_date() == D0

    def test_blob_written_first_as_commit_point(self, store, tmp_path, monkeypatch):
        """原子提交点写序：任一投影写失败 ⇒ 权威 blob 已在盘，投影与幂等记录未写。"""
        original = store._atomic_write_json

        def fail_on_account(path, payload, *, indent=2):
            if Path(path).name == "account.json":
                raise PaperStateError("模拟投影写入崩溃")
            return original(path, payload, indent=indent)

        monkeypatch.setattr(store, "_atomic_write_json", fail_on_account)
        with pytest.raises(PaperStateError, match="模拟投影写入崩溃"):
            store.commit_day(**_commit_kwargs())
        root = tmp_path / "paper"
        assert (root / "state/engine_state.bin").exists(), "权威 blob 必须先于投影落盘"
        assert not (root / "state/account.json").exists()
        assert not (root / f"runs/t0_{D0}.json").exists(), "崩溃在⑤前 ⇒ 幂等记录缺失"
        assert not (root / "state/.write.lock").exists(), "异常路径必须释放写锁"

    def test_optional_projections_skipped(self, store, tmp_path):
        """None 跳过项：ranked/stop_loss/instructions 不落盘；pending 队列恒写。"""
        store.commit_day(
            **_commit_kwargs(
                ranked_candidates=None,
                stop_loss_state=None,
                instructions=None,
                instructions_date=None,
            )
        )
        root = tmp_path / "paper"
        assert not (root / "state/ranked_candidates.json").exists()
        assert not (root / "state/stop_loss_state.json").exists()
        assert list((root / "instructions").glob("*.parquet")) == []
        assert (root / "pending_buys/pending_buys.json").exists()

    def test_empty_instruction_frame_skipped(self, store, tmp_path):
        empty = pd.DataFrame(columns=list(INSTRUCTION_COLUMNS))
        store.commit_day(**_commit_kwargs(instructions=empty, instructions_date=D1))
        assert list((tmp_path / "paper/instructions").glob("*.parquet")) == []

    def test_instructions_missing_date_rejected(self, store):
        with pytest.raises(PaperStateError, match="instructions_date"):
            store.commit_day(
                **_commit_kwargs(
                    instructions=pd.DataFrame(
                        [{"ts_code": "000001.SZ", "action": "buy", "shares": 100}]
                    ),
                    instructions_date=None,
                )
            )


# ---------------------------------------------------------------------------
# 9. nav 跳空语义 / 追加形态
# ---------------------------------------------------------------------------


class TestAppendSemantics:
    def test_nav_gap_semantics(self, store, tmp_path):
        """无成交日 nav_row=None ⇒ 不追加（跳空语义保持）；有成交日才追加。"""
        store.commit_day(**_commit_kwargs(nav_row=None, trade_rows=[]))
        root = tmp_path / "paper"
        assert not (root / "nav/nav.parquet").exists(), "无成交日不得产生 nav 行"
        assert not (root / "trades/trades.parquet").exists()
        store.commit_day(**_commit_kwargs(date=D1))
        nav = store.load_all_nav()
        assert len(nav) == 1 and nav.iloc[0]["trade_date"] == D1, "跳空后仅追加有成交日"
        trades = store.load_all_trades()
        assert len(trades) == 1

    def test_append_public_entry(self, store):
        store.append_nav([])  # 空行静默跳过
        store.append_trades([])
        store.append_nav(
            [{"trade_date": D0, "cash": 1.0, "position_value": 2.0, "total_value": 3.0, "nav": 1.0}]
        )
        store.append_trades(
            [
                {
                    "trade_date": D0,
                    "ts_code": "000001.SZ",
                    "action": "sell",
                    "shares": 100,
                    "price": 10.0,
                    "amount": 1000.0,
                    "commission": 0.2,
                    "stamp_tax": 1.0,
                    "slippage": 0.5,
                    "total_cost": 1.7,
                    "reason": "调仓卖出",
                }
            ]
        )
        store.append_nav(
            [{"trade_date": D1, "cash": 2.0, "position_value": 2.0, "total_value": 4.0, "nav": 1.1}]
        )
        nav = store.load_all_nav()
        assert list(nav["trade_date"]) == [D0, D1], "整文件重写式追加保序"
        assert len(store.load_all_trades()) == 1


# ---------------------------------------------------------------------------
# 10. 快照裁剪
# ---------------------------------------------------------------------------


class TestBlobTrimming:
    def test_trades_and_holdings_trimmed_portfolio_values_kept(self, store, tmp_path):
        """⑤ 修订口径（2026-10-28）：仅裁日期早于权威末日的历史条目，当日条目保留
        （当日成交 = 半提交恢复重建当日 trades/nav 投影的持久原料）。"""
        history_trade = {"date": pd.Timestamp("20260601"), "stock": "000002.SZ", "action": "sell"}
        sameday_trade = {"date": pd.Timestamp(D0), "stock": "000001.SZ", "action": "buy"}
        history_snap = {"date": pd.Timestamp("20260601"), "snap": 0}
        sameday_snap = {"date": pd.Timestamp(D0), "snap": 1}
        store.commit_day(
            **_commit_kwargs(
                engine_state_blob=_make_blob(
                    pv_dates=[D0],
                    trades=[history_trade, sameday_trade],
                    holdings=[history_snap, sameday_snap],
                )
            )
        )
        raw = (tmp_path / "paper/state/engine_state.bin").read_bytes()
        blob = pickle.loads(raw)
        assert blob["fields"]["trades"] == [sameday_trade], "历史成交日志裁掉、当日条目保留"
        assert blob["fields"]["holdings_snapshots"] == [sameday_snap], "历史持仓快照裁掉、当日保留"
        assert len(blob["fields"]["portfolio_values"]) == 1, "portfolio_values 必须保留"
        # 裁剪后 blob 仍通过 schema 校验
        state = store.load_engine_state()
        assert state.positions["000001.SZ"]["shares"] == 300

    def test_trim_function_directly(self):
        """纯函数级：历史裁掉 / 当日与无日期条目保留 / portfolio_values 为空不裁剪。"""
        history = {"date": pd.Timestamp(D0), "stock": "000001.SZ", "action": "buy"}
        sameday = {"date": pd.Timestamp(D1), "stock": "000002.SZ", "action": "sell"}
        trimmed = trim_engine_state_blob(
            _make_blob(pv_dates=[D0, D1], trades=[history, sameday], holdings=[history, sameday])
        )
        blob = pickle.loads(trimmed)
        assert blob["fields"]["trades"] == [sameday], "早于权威末日（D1）的条目裁掉"
        assert blob["fields"]["holdings_snapshots"] == [sameday]
        # 无日期条目防御保留（无法判定 ⇒ 不裁）
        no_date = {"stock": "000003.SZ", "action": "buy"}
        trimmed_no_date = trim_engine_state_blob(_make_blob(pv_dates=[D1], trades=[no_date]))
        assert pickle.loads(trimmed_no_date)["fields"]["trades"] == [no_date]
        # portfolio_values 为空 ⇒ 无法判定边界 ⇒ 不裁剪整段保留（恢复原料优先于体积）
        kept = trim_engine_state_blob(_make_blob(pv_dates=[], trades=[history], holdings=[history]))
        blob2 = pickle.loads(kept)
        assert blob2["fields"]["trades"] == [history]
        assert blob2["fields"]["holdings_snapshots"] == [history]

    def test_malformed_blob_rejected(self, store):
        with pytest.raises(PaperStateError, match="非 dict"):
            trim_engine_state_blob(pickle.dumps([1, 2, 3]))
        with pytest.raises(PaperStateError, match="fields"):
            trim_engine_state_blob(pickle.dumps({"schema_version": 1}))
        with pytest.raises(PaperStateError):
            store.commit_day(**_commit_kwargs(engine_state_blob=pickle.dumps([1])))


# ---------------------------------------------------------------------------
# 4/5/11. 双状态一致性三形态 + 半提交恢复
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_fresh_and_ok(self, store):
        assert store.check_consistency() == "fresh"
        store.commit_day(**_commit_kwargs())
        assert store.check_consistency() == "ok"

    def test_legacy_projection_without_authority(self, store, tmp_path):
        """旧链遗产布局（有 account.json 无 blob）⇒ 告警跳过，不校验不修复。"""
        path = tmp_path / "paper/state/account.json"
        path.write_text(
            json.dumps({"cash": 1.0, "last_update": D0, "positions": {}}), encoding="utf-8"
        )
        assert store.check_consistency() == "legacy_projection"
        assert json.loads(path.read_text(encoding="utf-8"))["cash"] == 1.0, "遗产投影不得被改写"

    def test_crash_after_commit_point_lag_repaired_once(self, tmp_path, monkeypatch):
        """半提交：权威新代 + 投影滞后 ⇒ 重启识别 ⇒ 受控重投影修复且只应用一次。"""
        store = PaperStateStore(root_path=str(tmp_path / "paper"))
        store.commit_day(**_commit_kwargs(date=D0))  # 第 1 代完整提交

        # 第 2 代：blob 落盘后、account.json 投影写入前崩溃
        original = store._atomic_write_json

        def fail_on_account(path, payload, *, indent=2):
            if Path(path).name == "account.json":
                raise PaperStateError("模拟崩溃")
            return original(path, payload, indent=indent)

        positions2 = _make_positions()
        positions2["000002.SZ"] = {
            "shares": 200,
            "buy_date": pd.Timestamp(D1),
            "signal_date": pd.Timestamp(D0),
            "buy_trade_price": 20.0,
            "buy_pnl_price": 21.0,
            "buy_cost_cash": 4004.0,
        }
        cash2 = 492993.915
        monkeypatch.setattr(store, "_atomic_write_json", fail_on_account)
        with pytest.raises(PaperStateError, match="模拟崩溃"):
            store.commit_day(
                **_commit_kwargs(
                    date=D1,
                    engine_state_blob=_make_blob(
                        cash=cash2, positions=positions2, pv_dates=[D0, D1]
                    ),
                    account=build_account_projection(cash2, positions2, D1),
                )
            )

        # 重启（新实例）：滞后识别 ⇒ 修复；account.json 以权威为准重建
        store2 = PaperStateStore(root_path=str(tmp_path / "paper"))
        assert store2.check_consistency() == "repaired"
        repaired = json.loads((tmp_path / "paper/state/account.json").read_text(encoding="utf-8"))
        assert repaired == build_account_projection(cash2, positions2, D1)
        assert "000002.SZ" in repaired["positions"]
        # 只应用一次：二次校验 ⇒ ok（不再重写）
        assert store2.check_consistency() == "ok"

    def test_corrupt_projection_treated_as_lag(self, store, tmp_path):
        store.commit_day(**_commit_kwargs())
        (tmp_path / "paper/state/account.json").write_text("{broken json", encoding="utf-8")
        assert store.check_consistency() == "repaired"
        assert store.check_consistency() == "ok"

    def test_lag_detected_without_auto_repair(self, tmp_path):
        store = PaperStateStore(root_path=str(tmp_path / "paper"))
        store.commit_day(**_commit_kwargs(date=D0))
        store.commit_day(**_commit_kwargs(date=D1, engine_state_blob=_make_blob(pv_dates=[D0, D1])))
        # 手工把投影回滚到上一代（模拟滞后）
        account = build_account_projection(496997.915, _make_positions(), D0)
        (tmp_path / "paper/state/account.json").write_text(
            json.dumps(account, ensure_ascii=False), encoding="utf-8"
        )
        assert store.check_consistency(auto_repair=False) == "lag_detected"
        assert store.check_consistency() == "repaired"

    def test_conflict_same_date_failfast(self, store, tmp_path):
        """冲突形态：投影日期 = 权威日期但现金不一致 ⇒ fail-fast。"""
        store.commit_day(**_commit_kwargs())
        path = tmp_path / "paper/state/account.json"
        account = json.loads(path.read_text(encoding="utf-8"))
        account["cash"] = account["cash"] + 1.0
        path.write_text(json.dumps(account, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(PaperStateError, match="冲突"):
            store.check_consistency()

    def test_conflict_projection_newer_than_authority_failfast(self, store, tmp_path):
        store.commit_day(**_commit_kwargs())
        path = tmp_path / "paper/state/account.json"
        account = json.loads(path.read_text(encoding="utf-8"))
        account["last_update"] = "20991231"
        path.write_text(json.dumps(account, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(PaperStateError, match="冲突"):
            store.check_consistency()

    def test_repair_projections_explicit_full_rewrite(self, store, tmp_path):
        """显式修复入口：全量重建可派生投影（含锚定/候选/止损/last_trade_date）。"""
        anchor = LiveAnchorState(
            anchor_date=pd.Timestamp(D0),
            rebalance_freq=20,
            stagger_tranches=2,
            last_scheduled_rebalance_date=pd.Timestamp(D0),
            last_rebalance_date=pd.Timestamp(D0),
        )
        store.commit_day(
            **_commit_kwargs(
                engine_state_blob=_make_blob(
                    ranked=[("000002.SZ", 0.7), ("000001.SZ", 0.9)],
                    last_signal_date=pd.Timestamp(D0),
                    live_anchor=anchor.to_dict(),
                    stop_loss_monitor={"consecutive_limit_down_days": {"000009.SZ": 1}},
                )
            )
        )
        # 破坏/清空全部投影后显式修复
        root = tmp_path / "paper"
        for rel in (
            "state/account.json",
            "state/strategy_state.json",
            "state/ranked_candidates.json",
            "state/last_trade_date.json",
            "runs/rebalance_state.json",
        ):
            (root / rel).unlink()
        repaired = store.repair_projections()
        assert "state/account.json" in repaired
        assert "state/last_trade_date.json" in repaired
        assert "state/strategy_state.json" in repaired
        assert "state/ranked_candidates.json" in repaired
        assert "runs/rebalance_state.json" in repaired
        # 候选重投影后稳定降序
        ranked = json.loads((root / "state/ranked_candidates.json").read_text(encoding="utf-8"))
        assert ranked["candidates"] == [["000001.SZ", 0.9], ["000002.SZ", 0.7]]
        rebalance = json.loads((root / "runs/rebalance_state.json").read_text(encoding="utf-8"))
        assert rebalance == {
            "last_rebalance_date": D0,
            "last_scheduled_rebalance_date": D0,
            "tranche_anchor_date": D0,
            "rebalance_freq": 20,
            "stagger_tranches": 2,
        }
        assert store.check_consistency() == "ok"

    def test_repair_without_authority_rejected(self, store):
        with pytest.raises(PaperStateError, match="无权威快照"):
            store.repair_projections()


# ---------------------------------------------------------------------------
# 6. 写锁
# ---------------------------------------------------------------------------


class TestWriteLock:
    def test_contention_raises_with_holder(self, store, tmp_path):
        store.acquire_write_lock()
        try:
            other = PaperStateStore(root_path=str(tmp_path / "paper"))
            with pytest.raises(PaperLockError, match="holder"):
                other.acquire_write_lock()
        finally:
            store.release_write_lock()
        # 释放后他人可获取
        other = PaperStateStore(root_path=str(tmp_path / "paper"))
        other.acquire_write_lock()
        other.release_write_lock()

    def test_stale_lock_preempted(self, tmp_path):
        store = PaperStateStore(root_path=str(tmp_path / "paper"), lock_timeout_seconds=300.0)
        lock_path = tmp_path / "paper/state/.write.lock"
        lock_path.write_text(
            json.dumps({"pid": 99999, "host": "ghost", "acquired_at": "2026-01-01T00:00:00"}),
            encoding="utf-8",
        )
        old = time.time() - 1000.0  # 锁龄 1000s > 超时 300s ⇒ stale
        os.utime(lock_path, (old, old))
        store.acquire_write_lock()  # 抢占成功
        assert lock_path.exists()
        holder = json.loads(lock_path.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid(), "抢占后 holder 必须换成本进程"
        store.release_write_lock()
        assert not lock_path.exists()

    def test_fresh_lock_not_preempted(self, tmp_path):
        store = PaperStateStore(root_path=str(tmp_path / "paper"), lock_timeout_seconds=300.0)
        lock_path = tmp_path / "paper/state/.write.lock"
        lock_path.write_text(
            json.dumps({"pid": 99999, "host": "ghost", "acquired_at": "2026-01-01T00:00:00"}),
            encoding="utf-8",
        )  # mtime = 现在 ⇒ 未超时
        with pytest.raises(PaperLockError, match="99999"):
            store.acquire_write_lock()

    def test_context_manager_and_reentry(self, store, tmp_path):
        with store.write_lock():
            with store.write_lock():  # 同实例同线程可重入
                assert (tmp_path / "paper/state/.write.lock").exists()
        assert not (tmp_path / "paper/state/.write.lock").exists()

    def test_active_lock_heartbeat_prevents_stale_preemption(self, tmp_path):
        store = PaperStateStore(root_path=str(tmp_path / "paper"), lock_timeout_seconds=0.2)
        lock_path = tmp_path / "paper/state/.write.lock"
        with store.write_lock():
            initial_mtime = lock_path.stat().st_mtime
            time.sleep(0.55)  # 超过固定超时，但持锁者仍活跃，心跳必须持续续期
            assert lock_path.stat().st_mtime > initial_mtime
            other = PaperStateStore(root_path=str(tmp_path / "paper"), lock_timeout_seconds=0.2)
            with pytest.raises(PaperLockError, match="写锁已被持有"):
                other.acquire_write_lock()
        assert not lock_path.exists()


# ---------------------------------------------------------------------------
# 7. SMB 远端只读模式
# ---------------------------------------------------------------------------


class _MockSMBReader:
    def __init__(self):
        self.calls = []

    def read_json(self, path):
        self.calls.append(("read_json", path))
        if path == "state/account.json":
            return {"cash": 1.0, "last_update": D0, "positions": {}}
        if path == "runs/rebalance_state.json":
            return {"last_rebalance_date": D0}
        return {}

    def read_yaml(self, path):
        self.calls.append(("read_yaml", path))
        return {"portfolio": {"top_n": 20}}

    def read_parquet(self, path):
        self.calls.append(("read_parquet", path))
        return pd.DataFrame(
            [
                {
                    "trade_date": D0,
                    "cash": 1.0,
                    "position_value": 0.0,
                    "total_value": 1.0,
                    "nav": 1.0,
                }
            ]
        )


class TestRemoteReadOnly:
    @pytest.fixture()
    def remote(self, tmp_path):
        return PaperStateStore(root_path=str(tmp_path / "paper"), smb_reader=_MockSMBReader())

    def test_no_local_dirs_and_writes_silently_skipped(self, remote, tmp_path):
        root = tmp_path / "paper"
        assert not root.exists(), "远端只读模式禁止创建本地目录"
        remote.commit_day(**_commit_kwargs())
        remote.append_nav([{"trade_date": D0}])
        remote.append_trades([{"trade_date": D0}])
        assert remote.repair_projections() == []
        assert not root.exists(), "远端只读模式一切写操作必须静默跳过"

    def test_four_whitelist_reads_via_smb(self, remote):
        assert remote.load_account() == {"cash": 1.0, "last_update": D0, "positions": {}}
        assert remote.load_config() == {"portfolio": {"top_n": 20}}
        nav = remote.load_all_nav()
        assert len(nav) == 1 and nav.iloc[0]["trade_date"] == D0
        assert remote.load_rebalance_state() == {"last_rebalance_date": D0}
        assert ("read_json", "state/account.json") in remote._smb_reader.calls
        assert ("read_yaml", "config.yaml") in remote._smb_reader.calls
        assert ("read_parquet", "nav/nav.parquet") in remote._smb_reader.calls
        assert ("read_json", "runs/rebalance_state.json") in remote._smb_reader.calls

    def test_other_reads_silent_miss(self, remote):
        assert remote.load_engine_state_blob() is None
        assert remote.load_engine_state() is None
        assert remote.get_authority_last_date() is None
        assert remote.load_all_trades() is None
        assert remote.load_strategy_state() == {}
        assert remote.load_ranked_candidates() is None
        assert remote.load_stop_loss_state() is None
        assert remote.load_last_trade_date() is None
        assert remote.load_pending_buys() == []
        assert remote.load_pending_sells() == []
        assert remote.load_instructions_frame(D1) is None
        assert remote.load_run_record("t0", D0) is None
        assert remote.check_run_exists("t0", D0) is False
        assert remote.check_consistency() == "remote_skipped"


# ---------------------------------------------------------------------------
# 8. 指令投影六语义
# ---------------------------------------------------------------------------


def _sell_positions() -> dict:
    return {
        "000003.SZ": {"shares": 250},  # 整手 ⇒ 200
        "000004.SZ": {"shares": 100},
        "000008.SZ": {"shares": 200},
        "000009.SZ": {"shares": 50},  # 不足一手 ⇒ 跳过
    }


def _build_frame(existing=None) -> pd.DataFrame:
    delayed = [
        # 补位上限：retry_count=6 > 5 ⇒ 不生成
        PendingOrder(
            stock="000006.SZ",
            action="buy",
            target_value=30000.0,
            signal_date=pd.Timestamp(D0),
            create_date=pd.Timestamp(D0),
            retry_count=6,
            last_reason="涨停",
        ),
        # 正常补位买入：retry_attempt=2
        PendingOrder(
            stock="000007.SZ",
            action="buy",
            target_value=30000.0,
            signal_date=pd.Timestamp(D0),
            create_date=pd.Timestamp(D0),
            retry_count=2,
            last_reason="停牌",
        ),
        # 延迟卖出：与条件卖出同股 ⇒ 去重后既有（条件卖出）优先
        PendingOrder(
            stock="000003.SZ",
            action="sell",
            target_value=None,
            signal_date=pd.Timestamp(D0),
            create_date=pd.Timestamp(D0),
            retry_count=1,
            last_reason="跌停",
        ),
        # 延迟卖出：独立股
        PendingOrder(
            stock="000008.SZ",
            action="sell",
            target_value=None,
            signal_date=pd.Timestamp(D0),
            create_date=pd.Timestamp(D0),
            retry_count=1,
            last_reason="跌停",
        ),
    ]
    return build_instruction_frame(
        signal_date=D0,
        buy_plan={"signals": {"000001.SZ": 0.05, "000002.SZ": 0.05}, "desired_position_count": 20},
        condition_sells={
            "000003.SZ": {"trigger_date": pd.Timestamp(D0), "sell_type": "rebalance"},
            "000009.SZ": {"trigger_date": pd.Timestamp(D0), "sell_type": "holding_period"},
        },
        stop_loss_sells={"000004.SZ": {"reason": "连续跌停", "trigger_type": "x"}},
        delayed_orders=delayed,
        positions=_sell_positions(),
        prices={"000001.SZ": 10.0, "000002.SZ": 10.0, "000007.SZ": 10.0},
        total_capital=600000.0,
        buy_price_type="close",
        sell_price_type="open",
        existing=existing,
    )


class TestInstructionProjection:
    def test_sells_before_buys_and_shapes(self):
        frame = _build_frame()
        assert list(frame.columns) == list(INSTRUCTION_COLUMNS)
        actions = list(frame["action"])
        first_buy = actions.index("buy")
        assert all(action == "sell" for action in actions[:first_buy]), "先卖后买"
        # 条件卖出整手（250 ⇒ 200）且 reason 映射 sell_type
        row3 = frame[frame["ts_code"] == "000003.SZ"].iloc[0]
        assert row3["shares"] == 200 and row3["reason"] == "调仓卖出"
        assert row3["price_type"] == "open" and row3["source_date"] == D0
        # 止损卖出 reason 透传
        row4 = frame[frame["ts_code"] == "000004.SZ"].iloc[0]
        assert row4["reason"] == "连续跌停"
        # 不足一手不生成
        assert "000009.SZ" not in set(frame["ts_code"])

    def test_dedup_key_action_ts_code(self):
        frame = _build_frame()
        keys = list(zip(frame["action"], frame["ts_code"]))
        assert len(keys) == len(set(keys)), "去重键 (action, ts_code)"
        # 000003.SZ 的条件卖出与延迟卖出命中同键 ⇒ 条件卖出（先行）优先
        rows = frame[frame["ts_code"] == "000003.SZ"]
        assert len(rows) == 1 and rows.iloc[0]["reason"] == "调仓卖出"

    def test_replacement_cap_and_retry_attempt(self):
        frame = _build_frame()
        assert "000006.SZ" not in set(frame["ts_code"]), "补位上限：retry_count>5 不再生成"
        row7 = frame[frame["ts_code"] == "000007.SZ"].iloc[0]
        assert row7["action"] == "buy" and row7["retry_attempt"] == 2
        assert row7["shares"] == 3000  # 30000/10 整手
        assert row7["original_signal_date"] == D0
        assert row7["replacement_slot_code"] == ""  # v2 延迟订单无槽位概念（登记差异）

    def test_keep_buy_date_false_and_buy_sizing(self):
        frame = _build_frame()
        assert not frame["keep_buy_date"].any(), "keep_buy_date 恒 False（暴露政策默认关）"
        row1 = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
        assert row1["action"] == "buy" and row1["shares"] == 3000  # 600000×0.05/10
        assert row1["target_weight"] == pytest.approx(0.05)
        assert row1["desired_position_count"] == 20
        assert row1["price_type"] == "close"

    def test_merge_with_existing_prefers_existing(self):
        existing = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "action": "buy",
                    "shares": 111,
                    "price_type": "open",
                    "reason": "既有指令",
                    "source_date": D0,
                    "target_weight": 0.01,
                    "original_signal_date": D0,
                    "desired_position_count": 20,
                    "retry_attempt": 0,
                    # 旧 10 列形态：缺 replacement_slot_code / keep_buy_date
                }
            ]
        )
        frame = _build_frame(existing=existing)
        row1 = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
        assert row1["shares"] == 111 and row1["reason"] == "既有指令", "合并既有优先"
        assert row1["replacement_slot_code"] == "" and not bool(row1["keep_buy_date"])
        # 既有行排在最前
        assert frame.iloc[0]["ts_code"] == "000001.SZ"

    def test_commit_names_file_by_execution_date(self, store, tmp_path):
        frame = _build_frame()
        store.commit_day(**_commit_kwargs(instructions=frame, instructions_date=D1))
        path = tmp_path / "paper/instructions" / f"{D1}.parquet"
        assert path.exists(), "指令文件名 = T+1 执行日"
        loaded = store.load_instructions_frame(D1)
        assert list(loaded.columns) == list(INSTRUCTION_COLUMNS)
        assert set(loaded["source_date"]) == {D0}, "source_date 列 = T0 信号日"
        assert len(loaded) == len(frame)

    def test_read_compat_legacy_10_columns(self, store, tmp_path):
        """读侧向后兼容旧 10 列形态（缺 replacement_slot_code/keep_buy_date）。"""
        legacy = pd.DataFrame(
            [
                {
                    "ts_code": "000010.SZ",
                    "action": "sell",
                    "shares": 100,
                    "price_type": "open",
                    "reason": "调仓卖出",
                    "source_date": D0,
                    "target_weight": 0.0,
                    "original_signal_date": "",
                    "desired_position_count": 0,
                    "retry_attempt": 0,
                }
            ]
        )
        path = tmp_path / "paper/instructions" / f"{D1}.parquet"
        legacy.to_parquet(path, index=False)
        loaded = store.load_instructions_frame(D1)
        assert list(loaded.columns) == list(INSTRUCTION_COLUMNS)
        row = loaded.iloc[0]
        assert row["replacement_slot_code"] == "" and not bool(row["keep_buy_date"])
        assert row["shares"] == 100
