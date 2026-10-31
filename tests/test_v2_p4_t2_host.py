# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主核心（PaperHost）测试（tests/ 全量豁免复杂度）。

覆盖（对应任务卡测试矩阵 host 行）：
- 单日 dry-run 形态就绪（跑通一日、产出 PaperDayResult、指令/快照/幂等记录落盘）；
- 同日重复 T0 幂等（同实例与新实例两形态均不重复执行；记录缺失 ⇒ 补写不重跑）；
- 日期回退守卫拒绝（corrected < 权威最后处理日）；
- 未就绪 ⇒ 不生成信号、不落盘、缺口登记（runs/gap_<date>.json）；
- 指令文件层零未来行情断言 fail-fast（§16.6 T2 增项）；
- 恢复启动不重锚（锚随快照恢复；跨实例排期推进正确）；
- 双状态滞后修复 / 冲突拒绝经宿主路径；
- fills 核对（None ⇒ 引擎行情记账权威 / 驱动日提交前核对：一致接受、不一致零副作用
  拒绝可重试 / 同日已提交后核对不一致 ⇒ 报错且已提交状态不变）；
- run_t1 退役政策空操作相位顺序断言（§7「run_t1 顺序」行）；
- 集成：run_t0 产出 Order 序列 + 指令文件落盘 + 次日 run_t1 驱动成交入账；
- 旧链遗产投影（无权威有 account.json）⇒ 拒绝启动（防接回，Q9）。

测试隔离硬契约：全部 tmp_path 隔离根 + 合成数据（合成 panel store / 临时
Storage / 合成信号与股票池），禁止读真实 data/；合成基建复用
``scripts/v2_p4/t05b_spike_runner.py`` 与 ``tests/test_v2_p4_t2_daily_assembly.py``
的构造方式。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.v2_p4.t05b_spike_runner import (
    _COST_SETTINGS,
    _DeterministicSignal,
    _StaticUniverse,
)
from src.lazybull.data import DataLoader, Storage
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.types import (
    Fill,
    Money,
    Order,
    OrderReason,
    OrderSide,
    Price,
    Shares,
    TradeDate,
    TSCode,
)
from src.lazybull.v2.hosts.paper.config_adapter import (
    PaperHostError,
    to_engine_kwargs,
    to_trading_config,
)
from src.lazybull.v2.hosts.paper.factory import create_paper_host
from src.lazybull.v2.hosts.paper.host import PaperDayResult, PaperHost
from src.lazybull.v2.hosts.paper.state_store import PaperStateError, PaperStateStore
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import bootstrap_manifest

# ---------------------------------------------------------------------------
# 合成数据基建（镜像 test_v2_p4_t2_daily_assembly.py 的构造方式）
# ---------------------------------------------------------------------------

#: 12 只合成股票（行数带需要 >5 行）；前 4 只为交易域（A/B ↔ C/D 轮换）
_CODES = [f"0000{i:02d}.SZ" for i in range(12)]
_TRADED = _CODES[:4]
#: 30 个合成交易日（2026-07-06 起工作日；与 daily_assembly 测试同段）
_CALENDAR_STRS = pd.bdate_range("2026-07-06", periods=30).strftime("%Y%m%d").tolist()
_CALENDAR = [pd.Timestamp(day) for day in _CALENDAR_STRS]
_MODEL_COLUMNS = ["ret_1", "ret_5", "ret_20", "bp", "turnover_rate"]

#: 纸面扁平配置（合成；top_n=2 / freq=5 / K=1 / 等权 / close 买 open 卖）
_FLAT_CONFIG = {
    "model_version": 7,
    "top_n": 2,
    "rebalance_freq": 5,
    "stagger_tranches": 1,
    "exclude_st": True,
    "min_list_days": 365,
    "position_sizing": "equal",
    "buy_price": "close",
    "sell_price": "open",
    "initial_capital": 1_000_000.0,
    "min_buy_value_ratio": 0.0,
    "horizon": 20,
    "universe": "all",
    "engine": "v2",
}


def _signal_plan() -> dict:
    """每 5 日轮换 AB ↔ CD（驱动调仓日的卖出+买入双侧）。"""
    plan = {}
    for i, day in enumerate(_CALENDAR_STRS):
        plan[day] = (
            [(_TRADED[0], 0.9), (_TRADED[1], 0.8)]
            if (i // 5) % 2 == 0
            else [(_TRADED[2], 0.95), (_TRADED[3], 0.85)]
        )
    return plan


def _panel_day_frame(day: str, n_rows: int = 6) -> pd.DataFrame:
    """单日 core 族 panel 帧（模型列 + 选股过滤列，全非空）。"""
    n = n_rows
    return pd.DataFrame(
        {
            "trade_date": [day] * n,
            "ts_code": _CODES[:n],
            "ret_1": np.linspace(0.10, 0.14, n),
            "ret_5": np.linspace(0.20, 0.24, n),
            "ret_20": np.linspace(-0.10, -0.06, n),
            "bp": np.linspace(0.01, 0.05, n),
            "turnover_rate": np.linspace(1.0, 1.4, n),
            "amount_ma20": np.full(n, 60000.0),
            "total_mv": np.full(n, 800000.0),
            "sw_l1_code": ["801780"] * n,
        }
    )


def _make_panel_store(root, skip_days=frozenset()) -> PanelDataStore:
    """指定根真实 PanelDataStore：bootstrap manifest + 逐日写 core 分区。"""
    store = PanelDataStore(root)
    bootstrap_manifest(store)
    for day in _CALENDAR_STRS:
        if day in skip_days:
            continue
        store.append_features(TradeDate.from_str(day), "core", _panel_day_frame(day))
    return store


def _make_data_storage(root, skip_days=frozenset()) -> Storage:
    """指定根 Storage：合成 clean/daily 全 replay 口径列（价格温和上行）。"""
    storage = Storage(root_path=str(root))
    daily_dir = storage.clean_path / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    for day in _CALENDAR_STRS:
        if day in skip_days:
            continue
        n = len(_CODES)
        # 逐日恒定价（per-code 价差）：T0 指令投影（T0 价）与 T+1 引擎撮合（T+1 价）
        # 在恒定价下逐位一致，供「指令股数 == 成交股数」强断言接缝使用
        closes = [10.0 * (1 + 0.1 * j) for j in range(n)]
        frame = pd.DataFrame(
            {
                "trade_date": [day] * n,
                "ts_code": _CODES,
                "close": closes,
                "close_adj": closes,
                "open": [round(c * 0.999, 4) for c in closes],
                "open_adj": [round(c * 0.999, 4) for c in closes],
                "is_suspended": [0] * n,
                "is_limit_up": [0] * n,
                "is_limit_down": [0] * n,
                "vol": np.full(n, 1000.0),
                "pct_chg": np.full(n, 0.5),
                "is_st": [0] * n,
                "list_days": [1000] * n,
                "tradable": [1] * n,
            }
        )
        frame.to_parquet(daily_dir / f"{day[:4]}-{day[4:6]}-{day[6:8]}.parquet", index=False)
    return storage


def _make_host(
    tmp_path,
    *,
    panel_skip_days=frozenset(),
    daily_skip_days=frozenset(),
    sub: str = "",
) -> PaperHost:
    """直接构造宿主（注入全部依赖；config→TradingConfig→engine_kwargs 走生产适配）。

    ``sub``：数据面子根后缀（paper 状态根始终共享 tmp_path/"paper"——状态连续性
    是跨实例测试的前提；panel/io 数据面按 sub 隔离，供就绪门缺口场景使用）。
    """
    store = PaperStateStore(root_path=str(tmp_path / "paper"))
    trading_config = to_trading_config(_FLAT_CONFIG)
    engine_kwargs = to_engine_kwargs(_FLAT_CONFIG, trading_config)
    return PaperHost(
        store=store,
        trading_config=trading_config,
        universe=_StaticUniverse(_TRADED),
        signal=_DeterministicSignal(2, _signal_plan(), _signal_plan()[_CALENDAR_STRS[0]]),
        cost_model=CostModel(**_COST_SETTINGS),
        stock_basic=None,
        data_storage=_make_data_storage(tmp_path / f"io{sub}", skip_days=daily_skip_days),
        panel_store=_make_panel_store(tmp_path / f"store{sub}", skip_days=panel_skip_days),
        trading_calendar=list(_CALENDAR),
        engine_kwargs=engine_kwargs,
        feature_columns=list(_MODEL_COLUMNS),
        train_params={},
        engine_mode="v2",
        verbose=False,
    )


def _drive(host: PaperHost, day: str) -> PaperDayResult:
    """经 run_t0 驱动当日并返回 PaperDayResult。"""
    host.run_t0(day)
    return host.last_day_result


def _authority_last_date(host: PaperHost):
    return host.store.get_authority_last_date()


# ---------------------------------------------------------------------------
# 单日 dry-run 形态
# ---------------------------------------------------------------------------


class TestSingleDayDrive:
    def test_first_day_drive_commits_full_layout(self, tmp_path):
        host = _make_host(tmp_path)
        day0 = _CALENDAR_STRS[0]
        orders = host.run_t0(day0)
        result = host.last_day_result

        assert result.driven is True
        assert result.t0_status == "success"  # 首次启动：当日放行 tranche0
        assert result.corrected_date == day0
        assert result.nav == pytest.approx(1_000_000.0)
        assert result.fills == []  # T0 只收单，次日才成交
        assert len(orders) == 2
        assert host.portfolio_merger.net_call_count == 1
        assert all(order.source_sleeve == "value" for order in orders)
        assert all(order.side is OrderSide.BUY for order in orders)
        assert {order.ts_code.value for order in orders} == set(_TRADED[:2])
        assert all(order.target_shares.value > 0 for order in orders)
        assert all(order.reason is OrderReason.REBALANCE for order in orders)
        assert "下一交易日指令" in result.push_text

        root = tmp_path / "paper"
        assert (root / "state/engine_state.bin").exists(), "权威快照必须落盘"
        assert (root / "state/account.json").exists()
        assert (root / "state/last_trade_date.json").exists()
        assert (root / "runs/rebalance_state.json").exists()
        assert (root / f"runs/t0_{day0}.json").exists()
        assert _authority_last_date(host) == day0
        # 指令文件（T+1 执行日命名）落盘且为买入行
        day1 = _CALENDAR_STRS[1]
        frame = host.store.load_instructions_frame(day1)
        assert frame is not None and len(frame) == 2
        assert set(frame["action"]) == {"buy"}
        assert set(frame["price_type"]) == {"close"}
        assert set(frame["ts_code"]) == set(_TRADED[:2])
        # 无成交日：nav/trades 跳空（不落盘）
        assert host.store.load_all_nav() is None
        assert host.store.load_all_trades() is None

    def test_s6_mutation_fails_before_commit_and_cannot_leave_published_orders(
        self, tmp_path, monkeypatch
    ):
        host = _make_host(tmp_path)
        day0, day1 = _CALENDAR_STRS[:2]
        monkeypatch.setattr(host.portfolio_merger, "net_across_sleeves", lambda orders: [])
        with pytest.raises(PaperHostError, match="当日尚未提交"):
            host.run_t0(day0)
        assert _authority_last_date(host) is None
        assert host.store.load_instructions_frame(day1) is None
        assert not (tmp_path / "paper" / f"runs/t0_{day0}.json").exists()

    def test_non_trading_day_rolls_forward(self, tmp_path):
        host = _make_host(tmp_path)
        # 2026-07-11 为周六（日历 07-06 周一起）：滚到其后首个开市日 07-13（周一）
        orders = host.run_t0("20260711")
        result = host.last_day_result
        assert result.corrected_date == "20260713"
        assert _authority_last_date(host) == "20260713"
        assert len(orders) == 2

    def test_date_beyond_calendar_fails_fast(self, tmp_path):
        host = _make_host(tmp_path)
        with pytest.raises(PaperHostError, match="交易日历"):
            host.run_t0("20261231")


# ---------------------------------------------------------------------------
# 同日重复 T0 幂等守卫（硬项⑦）
# ---------------------------------------------------------------------------


class TestSameDayIdempotency:
    def test_repeat_t0_same_instance_returns_projection_without_rerun(self, tmp_path):
        host = _make_host(tmp_path)
        day0 = _CALENDAR_STRS[0]
        orders_first = host.run_t0(day0)
        orders_second = host.run_t0(day0)
        result = host.last_day_result
        assert result.driven is False
        assert result.t0_status == "already_run"
        assert orders_second == orders_first, "幂等路径必须读回同一指令投影"
        # 不重复执行：无成交、nav 跳空保持
        assert host.store.load_all_trades() is None
        assert host.store.load_all_nav() is None

    def test_repeat_t0_fresh_instance_idempotent(self, tmp_path):
        day0 = _CALENDAR_STRS[0]
        orders_first = _make_host(tmp_path).run_t0(day0)
        host_b = _make_host(tmp_path)  # 新实例（跨进程形态）
        orders_second = host_b.run_t0(day0)
        assert host_b.last_day_result.t0_status == "already_run"
        assert orders_second == orders_first
        assert host_b.store.load_all_trades() is None

    def test_missing_run_record_backfilled_without_rerun(self, tmp_path):
        host = _make_host(tmp_path)
        day0 = _CALENDAR_STRS[0]
        host.run_t0(day0)
        (tmp_path / "paper" / "runs" / f"t0_{day0}.json").unlink()  # 崩溃在记录写入前形态
        host_b = _make_host(tmp_path)
        orders = host_b.run_t0(day0)
        assert host_b.last_day_result.t0_status == "already_run"
        assert len(orders) == 2
        record = host_b.store.load_run_record("t0", day0)
        assert record is not None and "幂等补写" in record.get("note", "")
        # 未重跑：账户与投影仍一致（一致性校验 ok）
        assert host_b.store.check_consistency(auto_repair=False) == "ok"
        # 补写后次日可正常驱动
        result = _drive(host_b, _CALENDAR_STRS[1])
        assert result.driven is True


# ---------------------------------------------------------------------------
# 日期回退守卫（§7 测试矩阵行）
# ---------------------------------------------------------------------------


class TestDateRegressionGuard:
    def test_rewind_rejected(self, tmp_path):
        host = _make_host(tmp_path)
        _drive(host, _CALENDAR_STRS[0])
        _drive(host, _CALENDAR_STRS[1])
        with pytest.raises(PaperHostError, match="日期回退"):
            host.run_t0(_CALENDAR_STRS[0])
        # run_t1 同样拒绝
        with pytest.raises(PaperHostError, match="日期回退"):
            host.run_t1(_CALENDAR_STRS[0])

    def test_fresh_instance_rewind_rejected(self, tmp_path):
        _drive(_make_host(tmp_path), _CALENDAR_STRS[2])
        host_b = _make_host(tmp_path)
        with pytest.raises(PaperHostError, match="日期回退"):
            host_b.run_t0(_CALENDAR_STRS[1])


# ---------------------------------------------------------------------------
# 就绪门（未就绪 ⇒ 不生成信号、不落盘、缺口登记）
# ---------------------------------------------------------------------------


class TestReadinessGate:
    def test_not_ready_day_defers_without_commit(self, tmp_path):
        day0, day1 = _CALENDAR_STRS[0], _CALENDAR_STRS[1]
        _drive(_make_host(tmp_path), day0)
        # 第二宿主：隔离数据面子根且 panel 缺 day1 分区 ⇒ 就绪门第 1 条不过
        host = _make_host(tmp_path, panel_skip_days={day1}, sub="_degraded")
        orders = host.run_t0(day1)
        result = host.last_day_result
        assert orders == []
        assert result.driven is False
        assert result.t0_status == "not_ready"
        assert result.readiness is not None and result.readiness.ok is False
        # 不落盘：权威最后处理日停在 day0，无 day1 幂等记录、无次日指令
        assert _authority_last_date(host) == day0
        assert not host.store.check_run_exists("t0", day1)
        assert host.store.load_instructions_frame(_CALENDAR_STRS[2]) is None
        # 缺口登记
        gap_path = tmp_path / "paper" / "runs" / f"gap_{day1}.json"
        assert gap_path.exists()
        gap = json.loads(gap_path.read_text(encoding="utf-8"))
        assert gap["kind"] == "readiness_gap" and gap["date"] == day1
        assert any("panel" in item for item in gap["gaps"])
        # 推迟后可恢复：数据补齐（第三宿主全量数据）当日正常驱动
        result = _drive(_make_host(tmp_path), day1)
        assert result.driven is True


# ---------------------------------------------------------------------------
# 指令文件层零未来行情断言（§16.6 T2 增项）
# ---------------------------------------------------------------------------


class TestNoFuturePriceAssertion:
    def test_future_rows_rejected(self, tmp_path):
        host = _make_host(tmp_path)
        day0 = _CALENDAR_STRS[0]
        frame = pd.DataFrame(
            {"ts_code": [_TRADED[0]], "trade_date": [_CALENDAR_STRS[1]], "close": [10.0]}
        )
        with pytest.raises(PaperHostError, match="零未来行情|未来交易日"):
            host._assert_no_future_price(frame, day0)

    def test_clean_window_passes(self, tmp_path):
        host = _make_host(tmp_path)
        day1 = _CALENDAR_STRS[1]
        frame = pd.DataFrame(
            {
                "ts_code": [_TRADED[0], _TRADED[0]],
                "trade_date": [_CALENDAR_STRS[0], day1],
                "close": [10.0, 10.1],
            }
        )
        host._assert_no_future_price(frame, day1)  # 不抛
        host._assert_no_future_price(pd.DataFrame(), day1)  # 空帧不抛


# ---------------------------------------------------------------------------
# 恢复启动不重锚 + 跨实例排期推进
# ---------------------------------------------------------------------------


class TestRestoreNoReanchor:
    def test_anchor_survives_restart_and_schedule_advances(self, tmp_path):
        day0 = _CALENDAR_STRS[0]
        host = _make_host(tmp_path)
        _drive(host, day0)
        anchor_day0 = host.store.load_rebalance_state()
        assert anchor_day0["tranche_anchor_date"] == day0
        assert anchor_day0["last_rebalance_date"] == day0

        # 次日新实例（恢复启动）：锚不重建（tranche_anchor_date 仍 = day0）
        host_b = _make_host(tmp_path)
        _drive(host_b, _CALENDAR_STRS[1])
        state = host_b.store.load_rebalance_state()
        assert state["tranche_anchor_date"] == day0, "恢复启动禁止重锚"
        blob_state = host_b.store.load_engine_state()
        assert pd.Timestamp(day0) == blob_state.live_anchor_state["anchor_date"]

    def test_k1_schedule_by_actual_success_day(self, tmp_path):
        """K=1：freq=5，day0 成功 ⇒ idx1~4 非调仓日，idx5 再次调仓（跨实例逐日驱动）。"""
        statuses = {}
        for i in range(0, 6):
            host = _make_host(tmp_path)  # 每日新实例 = 跨进程语义
            statuses[i] = _drive(host, _CALENDAR_STRS[i]).t0_status
        assert statuses[0] == "success"
        assert all(statuses[i] == "not_rebalance_day" for i in range(1, 5))
        assert statuses[5] == "success", "距实际成功日满 5 个交易日在 idx5 到期"


# ---------------------------------------------------------------------------
# 双状态一致性经宿主路径（滞后修复 / 冲突拒绝）
# ---------------------------------------------------------------------------


class TestConsistencyViaHost:
    def test_lag_projection_auto_repaired_on_drive(self, tmp_path):
        day0, day1 = _CALENDAR_STRS[0], _CALENDAR_STRS[1]
        _drive(_make_host(tmp_path), day0)
        # 构造滞后形态：account.json last_update 改到权威之前
        account_path = tmp_path / "paper" / "state" / "account.json"
        account = json.loads(account_path.read_text(encoding="utf-8"))
        account["last_update"] = "20200101"
        account_path.write_text(json.dumps(account, ensure_ascii=False), encoding="utf-8")
        host = _make_host(tmp_path)
        assert host.store.check_consistency(auto_repair=False) == "lag_detected"
        result = _drive(host, day1)  # 宿主路径内 auto_repair=True 修复后驱动
        assert result.driven is True
        assert host.store.check_consistency(auto_repair=False) == "ok"
        account = json.loads(account_path.read_text(encoding="utf-8"))
        assert account["last_update"] == day1

    def test_conflict_projection_rejected(self, tmp_path):
        day0, day1 = _CALENDAR_STRS[0], _CALENDAR_STRS[1]
        _drive(_make_host(tmp_path), day0)
        # 构造冲突形态：同日但现金值与权威不一致
        account_path = tmp_path / "paper" / "state" / "account.json"
        account = json.loads(account_path.read_text(encoding="utf-8"))
        account["cash"] = account["cash"] + 1.0
        account_path.write_text(json.dumps(account, ensure_ascii=False), encoding="utf-8")
        host = _make_host(tmp_path)
        with pytest.raises(PaperStateError, match="不一致"):
            host.run_t0(day1)
        # 冲突拒绝 ⇒ 当日未驱动
        assert _authority_last_date(host) == day0


# ---------------------------------------------------------------------------
# fills 三形态（硬项③）
# ---------------------------------------------------------------------------


def _fills_as_dicts(result: PaperDayResult) -> list:
    return [
        {
            "ts_code": row["ts_code"],
            "action": row["action"],
            "shares": row["shares"],
            "price": row["price"],
        }
        for row in result.fills
    ]


def _fills_as_v2_fills(result: PaperDayResult) -> list:
    fills = []
    for row in result.fills:
        order = Order(
            ts_code=TSCode(str(row["ts_code"])),
            side=OrderSide.BUY if row["action"] == "buy" else OrderSide.SELL,
            target_shares=Shares(int(row["shares"])),
            order_type=None,  # 比对只读 ts_code/side/shares/price
            reason=OrderReason.REBALANCE,
            source_sleeve="value",
        )
        fills.append(
            Fill(
                order=order,
                fill_price=Price(float(row["price"])),
                fill_shares=Shares(int(row["shares"])),
                fill_date=TradeDate.from_str(str(row["trade_date"])),
                commission=Money(float(row["commission"])),
            )
        )
    return fills


class TestFillsReconciliation:
    def _drive_two_days(self, tmp_path) -> PaperDayResult:
        _drive(_make_host(tmp_path), _CALENDAR_STRS[0])
        host = _make_host(tmp_path)
        host.run_t1(_CALENDAR_STRS[1])  # fills=None：引擎行情记账为权威
        return host.last_day_result

    def test_fills_none_books_by_engine_market(self, tmp_path):
        result = self._drive_two_days(tmp_path)
        assert result.driven is True
        assert len(result.fills) == 2, "次日 T+1 执行 T0 计划买入 2 笔"
        assert {row["action"] for row in result.fills} == {"buy"}

    def test_fills_consistent_idempotent_accept(self, tmp_path):
        result = self._drive_two_days(tmp_path)
        trades_before = self._trades_count(tmp_path)
        # 幂等读路径 + dict 形态 fills：一致 ⇒ 接受，不重复入账
        host = _make_host(tmp_path)
        host.run_t1(_CALENDAR_STRS[1], fills=_fills_as_dicts(result))
        assert host.last_day_result.t0_status == "already_run"
        assert self._trades_count(tmp_path) == trades_before
        # v2 Fill 对象形态同样接受
        host.run_t1(_CALENDAR_STRS[1], fills=_fills_as_v2_fills(result))
        assert self._trades_count(tmp_path) == trades_before

    def test_fills_consistent_on_driving_call(self, tmp_path):
        """驱动路径上的 fills 核对：另一隔离根先跑出权威 fills，本根驱动时供给一致 fills。"""
        day0, day1 = _CALENDAR_STRS[0], _CALENDAR_STRS[1]
        reference = self._drive_two_days(tmp_path / "ref")
        _drive(_make_host(tmp_path), day0)
        host = _make_host(tmp_path)
        host.run_t1(day1, fills=_fills_as_dicts(reference))  # 一致 ⇒ 幂等接受
        assert host.last_day_result.driven is True
        assert self._trades_count(tmp_path) == 2

    def test_fills_mismatch_rejected(self, tmp_path):
        result = self._drive_two_days(tmp_path)
        host = _make_host(tmp_path)
        bad = _fills_as_dicts(result)[:-1]  # 少一笔 ⇒ 键集合不一致
        with pytest.raises(PaperHostError, match="不一致"):
            host.run_t1(_CALENDAR_STRS[1], fills=bad)

    def test_fills_mismatch_on_driving_call_zero_side_effect_retryable(self, tmp_path):
        """驱动路径 fills 提交前核对不一致 ⇒ 零副作用拒绝（当日未落盘），修正后可重试。"""
        day0, day1 = _CALENDAR_STRS[0], _CALENDAR_STRS[1]
        _drive(_make_host(tmp_path), day0)
        blob_before = (tmp_path / "paper/state/engine_state.bin").read_bytes()
        host = _make_host(tmp_path)
        with pytest.raises(PaperHostError, match="不一致"):
            host.run_t1(day1, fills=[])  # 外部声称零成交 vs 引擎 2 笔
        # 零副作用拒绝：当日未落盘——权威末日停在 day0、快照字节不变、无成交/净值/幂等记录
        assert _authority_last_date(host) == day0
        assert (tmp_path / "paper/state/engine_state.bin").read_bytes() == blob_before
        assert self._trades_count(tmp_path) == 0
        assert host.store.load_all_nav() is None
        assert not host.store.check_run_exists("t1", day1)
        # 修正后重试（fills=None ⇒ 引擎行情记账为权威）⇒ 正常提交且成交恰一次
        host.run_t1(day1)
        assert host.last_day_result.driven is True
        assert self._trades_count(tmp_path) == 2
        assert _authority_last_date(host) == day1

    def _trades_count(self, tmp_path) -> int:
        trades = PaperStateStore(root_path=str(tmp_path / "paper")).load_all_trades()
        return 0 if trades is None else len(trades)


# ---------------------------------------------------------------------------
# run_t1 退役政策空操作相位顺序断言（§7「run_t1 顺序」行）
# ---------------------------------------------------------------------------


class TestRunT1PhaseOrder:
    def test_policy_phase_precedes_day_drive(self, tmp_path):
        host = _make_host(tmp_path)
        host.run_t0(_CALENDAR_STRS[0])
        host.run_t1(_CALENDAR_STRS[1])
        day1 = _CALENDAR_STRS[1]
        trace = host._phase_trace
        policy_idx = trace.index((day1, "policy_eval"))
        drive_idx = trace.index((day1, "day_drive"))
        assert policy_idx < drive_idx, "政策评估相位必须先于 T1 执行（退役空操作，顺序保留）"

    def test_policy_phase_precedes_idempotent_read(self, tmp_path):
        host = _make_host(tmp_path)
        host.run_t0(_CALENDAR_STRS[0])
        host.run_t1(_CALENDAR_STRS[0])  # 同日：幂等读产物路径
        day0 = _CALENDAR_STRS[0]
        trace = host._phase_trace
        assert trace.index((day0, "policy_eval")) > trace.index((day0, "day_drive"))
        assert trace.index((day0, "policy_eval")) < trace.index((day0, "day_read"))


# ---------------------------------------------------------------------------
# 集成：T0 指令 → 次日 T1 成交入账 → 多日轮换
# ---------------------------------------------------------------------------


class TestIntegrationT0T1:
    def test_t0_orders_then_t1_fills_booked_by_engine(self, tmp_path):
        day0, day1 = _CALENDAR_STRS[0], _CALENDAR_STRS[1]
        host = _make_host(tmp_path)
        orders = host.run_t0(day0)
        assert len(orders) == 2, "run_t0 返回次日计划 Order 序列"
        # 指令文件落盘（经 store.commit_day）
        frame = host.store.load_instructions_frame(day1)
        assert frame is not None and len(frame) == 2
        instruction_shares = dict(zip(frame["ts_code"], frame["shares"]))

        host_t1 = _make_host(tmp_path)
        host_t1.run_t1(day1)
        result = host_t1.last_day_result
        assert result.driven is True and len(result.fills) == 2
        # 指令投影（T0 静态预算）vs 引擎撮合（T+1 顺序成交现金约束）：首笔逐位一致；
        # 满仓权重下末笔受首笔成本约束缩量（buy_execution 资金不足分支），
        # 缩量方向 = 成交 ≤ 指令，幅度 ≦ 2 手（合成成本率 ~1.3‰ 量级）
        for row in result.fills:
            assert row["action"] == "buy" and row["price"] > 0
            assert row["shares"] > 0 and row["shares"] % 100 == 0
            assert row["shares"] <= instruction_shares[row["ts_code"]]
            assert instruction_shares[row["ts_code"]] - row["shares"] <= 200
            assert row["total_cost"] == pytest.approx(
                row["commission"] + row["stamp_tax"] + row["slippage"]
            )
            assert row["stamp_tax"] == 0.0  # 买入免印花税
        fills_by_code = {row["ts_code"]: row for row in result.fills}
        # 首笔（买入序列第一只，无现金约束）指令股数与成交逐位一致
        assert fills_by_code[_TRADED[0]]["shares"] == instruction_shares[_TRADED[0]]
        # 账户投影：持仓 2 只、现金扣减
        account = host_t1.store.load_account()
        assert set(account["positions"]) == set(_TRADED[:2])
        assert account["cash"] < 1_000_000.0
        # 有成交日：nav/trades 落盘（跳空语义保持）
        nav = host_t1.store.load_all_nav()
        assert len(nav) == 1 and nav.iloc[0]["trade_date"] == day1
        trades = host_t1.store.load_all_trades()
        assert len(trades) == 2
        assert _authority_last_date(host_t1) == day1

    def test_multi_day_rotation_produces_sell_and_buy_trades(self, tmp_path):
        """跨调仓周期：idx5 调仓 ⇒ idx6 执行卖出（open）+ 买入（close）双侧成交。"""
        results = {}
        for i in range(0, 7):
            host = _make_host(tmp_path)
            results[i] = _drive(host, _CALENDAR_STRS[i])
        # idx6 = idx5 调仓计划的执行日：卖 A/B + 买 C/D
        fills = results[6].fills
        sells = [row for row in fills if row["action"] == "sell"]
        buys = [row for row in fills if row["action"] == "buy"]
        assert {row["ts_code"] for row in sells} == set(_TRADED[:2])
        assert {row["ts_code"] for row in buys} == set(_TRADED[2:])
        assert all(row["stamp_tax"] > 0 for row in sells), "卖出印花税投影"
        assert all(row["reason"] == "调仓卖出" for row in sells)
        # 持仓轮换完成
        account = _make_host(tmp_path).store.load_account()
        assert set(account["positions"]) == set(_TRADED[2:])
        trades = PaperStateStore(root_path=str(tmp_path / "paper")).load_all_trades()
        assert len(trades) == 2 + 4  # day1 两笔买入 + day6 两卖两买
        # 指令投影含先卖后买序（当日 idx5 产出）
        frame = PaperStateStore(root_path=str(tmp_path / "paper")).load_instructions_frame(
            _CALENDAR_STRS[6]
        )
        assert frame is not None
        actions = list(frame["action"])
        assert actions == sorted(actions, key=lambda a: a != "sell"), "卖出段必须先于买入段"
        # 双状态一致性收官校验
        assert PaperStateStore(root_path=str(tmp_path / "paper")).check_consistency() == "ok"


# ---------------------------------------------------------------------------
# 旧链遗产投影拒绝启动（Q9 防接回）
# ---------------------------------------------------------------------------


class TestLegacyProjectionRejected:
    def test_account_without_authority_blob_rejected(self, tmp_path):
        store = PaperStateStore(root_path=str(tmp_path / "paper"))
        store.state_path.mkdir(parents=True, exist_ok=True)
        (store.state_path / "account.json").write_text(
            json.dumps({"cash": 1.0, "last_update": "20200101", "positions": {}}),
            encoding="utf-8",
        )
        host = _make_host(tmp_path)
        with pytest.raises(PaperHostError, match="T5 迁移"):
            host.run_t0(_CALENDAR_STRS[0])
        assert not (tmp_path / "paper" / "state" / "engine_state.bin").exists()


# ---------------------------------------------------------------------------
# 工厂（create_paper_host）
# ---------------------------------------------------------------------------


def _write_factory_storage(tmp_path) -> Storage:
    """工厂测试用 Storage：clean/daily + raw stock_basic/trade_cal（含未来日）。"""
    storage = _make_data_storage(tmp_path / "io")
    stock_basic = pd.DataFrame(
        {
            "ts_code": _CODES[:3] + ["300001.SZ"],
            "name": ["甲", "乙", "丙", "创业板丁"],
            "market": ["主板", "主板", "主板", "创业板"],
            "list_date": ["20200101"] * 4,
        }
    )
    stock_basic.to_parquet(storage.raw_path / "stock_basic.parquet", index=False)
    cal = pd.DataFrame(
        {
            "cal_date": _CALENDAR_STRS,
            "is_open": [1] * len(_CALENDAR_STRS),
        }
    )
    cal.to_parquet(storage.raw_path / "trade_cal.parquet", index=False)
    return storage


def _write_factory_config(root, **overrides) -> None:
    config = {
        "model": {"model_version": 7, "model_version_b": None},
        "portfolio": {"top_n": 2, "rebalance_freq": 5, "stagger_tranches": 1},
        "paper_trade": {"buy_price": "close", "sell_price": "open", "initial_capital": 1e6},
        "extra": {"engine": "v2"},
    }
    for key, value in overrides.items():
        config.setdefault("extra", {})[key] = value
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


class TestCreatePaperHost:
    def _factory_kwargs(self, tmp_path) -> dict:
        storage = _write_factory_storage(tmp_path)
        return dict(
            store=str(tmp_path / "paper"),
            data_root=str(tmp_path),
            signal=_DeterministicSignal(2, _signal_plan(), _signal_plan()[_CALENDAR_STRS[0]]),
            loader=DataLoader(storage=storage),
            data_storage=storage,
            panel_store=_make_panel_store(tmp_path / "store"),
            feature_columns=list(_MODEL_COLUMNS),
            train_params={},
        )

    def test_factory_assembles_and_drives(self, tmp_path):
        paper_root = tmp_path / "paper"
        _write_factory_config(paper_root)
        host = create_paper_host(**self._factory_kwargs(tmp_path))
        assert host.engine_mode == "v2"
        assert host.trading_config.top_n == 2
        assert host.engine_kwargs["enable_trailing_early_rebalance"] is False
        assert host.engine_kwargs["sell_timing"] == "open"
        # 日历自 trade_cal 派生（含未来日形态 = 全长 30 日）
        assert [d.strftime("%Y%m%d") for d in host.trading_calendar] == _CALENDAR_STRS
        # 工厂产物可真实驱动一日
        orders = host.run_t0(_CALENDAR_STRS[0])
        assert len(orders) == 2
        assert (paper_root / "state" / "engine_state.bin").exists()

    def test_factory_mainboard_universe_filter(self, tmp_path):
        paper_root = tmp_path / "paper"
        _write_factory_config(paper_root)
        kwargs = self._factory_kwargs(tmp_path)
        host = create_paper_host(**kwargs)
        # universe=all（_FLAT 默认）；改用 mainboard 配置验证过滤
        assert host.universe is not None
        _write_factory_config(paper_root, universe="mainboard")
        flat_config_override = dict(_FLAT_CONFIG, universe="mainboard")
        host_mb = create_paper_host(config_flat=flat_config_override, **kwargs)
        assert len(host_mb.universe.stock_basic) == 3  # 创业板被剔除

    def test_factory_invalid_engine_rejected(self, tmp_path):
        _write_factory_config(tmp_path / "paper", engine="v9")
        with pytest.raises(PaperHostError, match="engine 切换键非法"):
            create_paper_host(**self._factory_kwargs(tmp_path))

    def test_factory_buy_price_open_rejected(self, tmp_path):
        _write_factory_config(tmp_path / "paper", buy_price="open")
        with pytest.raises(PaperHostError, match="buy_price"):
            create_paper_host(**self._factory_kwargs(tmp_path))

    def test_factory_missing_config_rejected(self, tmp_path):
        kwargs = self._factory_kwargs(tmp_path)  # 不写 config.yaml
        with pytest.raises(PaperHostError, match="未找到配置文件"):
            create_paper_host(**kwargs)

    def test_factory_blank_model_version_without_registry_rejected(self, tmp_path):
        _write_factory_config(tmp_path / "paper", model_version=None)
        with pytest.raises(PaperHostError, match="无可用主模型版本"):
            create_paper_host(**self._factory_kwargs(tmp_path))

    def test_factory_config_flat_override_skips_file(self, tmp_path):
        # config_flat 注入时不读 config.yaml（影子/测试接缝）
        host = create_paper_host(config_flat=dict(_FLAT_CONFIG), **self._factory_kwargs(tmp_path))
        assert host.engine_mode == "v2"
