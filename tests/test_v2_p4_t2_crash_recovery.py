# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主崩溃恢复与防御边界测试（tests/ 全量豁免复杂度）。

评审处置轮硬性要求逐条落实（合成数据 + tmp_path 隔离根，零真实 data/ 依赖）：

1. 崩溃边界矩阵（11 个注入点参数化）：day1（含 2 笔买入的成交日）commit 各写入边界
   注入 OSError ⇒ 恢复后同日重入不抛错（blob 已写边界 = ``already_run`` 幂等读产物；
   blob 未写边界 = 正常重驱动）、全部产物与**无中断参照根**逐行/逐键相等（runs 记录
   剔除 timestamp 比对）、再次同日调用成交/NAV 不重复、跨日驱动后 day1 行仍恰一份。
   夹具 freq=1（每日调仓 A/B ↔ C/D 轮换）使 day1 同时具备「2 笔买入成交」与「非空
   T+1 指令帧」——freq=5 形态下成交日不是信号日、指令帧恒空，指令注入点将无写盘可断。
   「account.json 已写、后续文件未写」边界 = 注入点 strategy_state.json（评审明确要求
   覆盖）；写序内位于 account.json 之后的状态投影在该类半提交下保持上一代（stale，
   account 族受控修复仅由 account.json 滞后/缺失触发），次日正常 commit 全量重写收敛
   ——本测试 same_day 阶段断言其停留在上一代、cross_day 阶段断言收敛相等。
2. 单写入者整日写锁：另一 PaperStateStore 实例持锁 ⇒ 宿主调用 PaperLockError；释放
   后重试正常驱动。
3. commit_day 比较交换守卫：expected_base_date 过期 ⇒ PaperStateError 拒绝；等价 ⇒
   正常提交；缺省跳过校验（工具直调形态）。
4. fills 三形态（驱动日）：不一致 ⇒ PaperHostError 零副作用拒绝（当日未落盘，修正后
   可重试）/ 一致 ⇒ 正常提交且成交恰一次 / 同日已提交后不一致 ⇒ PaperHostError 且
   已提交状态不变。
5. 排期兜底边界（§17.7 硬项⑥）：分支 1「尚无成功」构表异常**上冒**（宿主不吞、
   零落盘）；分支 3/4（已有成功锚定）构表异常 ⇒ **fail-open** 不抛错（当日按
   tranche0 放行）。
6. 远端只读 fail-fast：带 smb_reader 的 PaperStateStore ⇒ PaperHost 构造抛
   PaperHostError。
7. ``_fill_identity`` Mapping 分支防御：缺 ts_code/shares/price/action 键或形态非法
   ⇒ PaperHostError（含上下文），不再裸 KeyError。
8. 持有期到期 reason 带交易日天数：``build_instruction_frame(..., trade_dates=...)``
   ⇒ ``f"持有期到期: 持有{N}天"``（N = ``calc_holding_trade_days`` 口径）；不传
   trade_dates ⇒ 回退静态 ``"持有期到期"``。

合成基建复用 ``tests/test_v2_p4_t2_host.py``（panel/行情/桩信号夹具）与
``tests/test_v2_p4_t2_state_store.py``（commit_day 合成入参）的构造方式。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

import src.lazybull.v2.core.execution.live_schedule as live_schedule
from scripts.v2_p4.t05b_spike_runner import (
    _COST_SETTINGS,
    _DeterministicSignal,
    _StaticUniverse,
)
from src.lazybull.common.date_utils import calc_holding_trade_days
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.hosts.paper.config_adapter import (
    PaperHostError,
    to_engine_kwargs,
    to_trading_config,
)
from src.lazybull.v2.hosts.paper.host import PaperHost
from src.lazybull.v2.hosts.paper.state_store import (
    PaperLockError,
    PaperStateError,
    PaperStateStore,
    build_instruction_frame,
)
from tests.test_v2_p4_t2_host import (
    _CALENDAR,
    _CALENDAR_STRS,
    _FLAT_CONFIG,
    _MODEL_COLUMNS,
    _TRADED,
    _fills_as_dicts,
    _make_data_storage,
    _make_panel_store,
    _signal_plan,
)
from tests.test_v2_p4_t2_state_store import D0 as SS_D0
from tests.test_v2_p4_t2_state_store import D1 as SS_D1
from tests.test_v2_p4_t2_state_store import _commit_kwargs

_DAY0 = _CALENDAR_STRS[0]  # 首次启动日（无成交）
_DAY1 = _CALENDAR_STRS[1]  # 崩溃注入日（含 2 笔买入的成交日 + 非空 T+1 指令帧）
_DAY2 = _CALENDAR_STRS[2]
_DAY3 = _CALENDAR_STRS[3]  # 持有期到期卖出日（2 笔卖出）

#: 崩溃注入点（commit 写序全覆盖）：按目标文件名在对应原子写方法处注入 OSError
_CRASH_TARGETS: List[str] = [
    "engine_state.bin",  # ② 原子提交点（权威 blob）
    "account.json",  # ③ 投影族首写（崩溃 ⇒ account 缺失 ⇒ 滞后修复路径）
    "strategy_state.json",  # ③「account.json 已写、后续文件未写」边界（评审点名）
    "last_trade_date.json",  # ③ 中段
    "rebalance_state.json",  # ③ 末段
    f"{_DAY2}.parquet",  # ③ T+1 指令帧（<T+1>.parquet）
    "pending_buys.json",  # ③ pending 买入队
    "pending_sells.json",  # ③ pending 卖出队
    "nav.parquet",  # ④ nav 追加
    "trades.parquet",  # ④ trades 追加
    f"t0_{_DAY1}.json",  # ⑤ 幂等记录
]

#: （历史注记）③ 中段崩溃（account 已写、后续状态投影未写）曾一度允许 same_day 阶段
#: stale（次日 commit 收敛）；2026-10-28 处置轮新增 store.sync_state_projections
#: （状态投影族确定性比对补齐）后，same_day 阶段即完全收敛——本矩阵全量断言与
#: 无中断参照根逐位一致，不留 stale 宽限。


# ---------------------------------------------------------------------------
# 合成基建（数据面模块级共享：宿主对其只读消费；状态根每用例独立）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_data(tmp_path_factory) -> Dict[str, Any]:
    """模块级共享只读数据面（合成 clean/daily + panel 分区，构建一次全程只读）。"""
    base = tmp_path_factory.mktemp("shared_data")
    return {
        "storage": _make_data_storage(base / "io"),
        "panel": _make_panel_store(base / "store"),
    }


def _signal_plan_daily() -> Dict[str, List]:
    """freq=1 场景信号计划：逐日轮换 A/B ↔ C/D（每个驱动日都产生非空调仓计划）。"""
    plan: Dict[str, List] = {}
    for i, day in enumerate(_CALENDAR_STRS):
        plan[day] = (
            [(_TRADED[0], 0.9), (_TRADED[1], 0.8)]
            if i % 2 == 0
            else [(_TRADED[2], 0.95), (_TRADED[3], 0.85)]
        )
    return plan


def _make_host(
    state_base: Path,
    shared_data: Dict[str, Any],
    *,
    freq: int = 5,
    store: Optional[PaperStateStore] = None,
) -> PaperHost:
    """构造宿主（状态根 = state_base/"paper"；数据面共享只读；freq 控制调仓频率）。

    freq=5 ⇒ 复用宿主测试的 5 日轮换计划；freq=1 ⇒ 逐日轮换计划（崩溃矩阵专用，
    使 day1 同时具备 2 笔买入成交与非空 T+1 指令帧）。
    """
    config = dict(_FLAT_CONFIG, rebalance_freq=freq)
    trading_config = to_trading_config(config)
    engine_kwargs = to_engine_kwargs(config, trading_config)
    plan = _signal_plan() if freq == 5 else _signal_plan_daily()
    return PaperHost(
        store=store or PaperStateStore(root_path=str(state_base / "paper")),
        trading_config=trading_config,
        universe=_StaticUniverse(_TRADED),
        signal=_DeterministicSignal(2, plan, plan[_CALENDAR_STRS[0]]),
        cost_model=CostModel(**_COST_SETTINGS),
        stock_basic=None,
        data_storage=shared_data["storage"],
        panel_store=shared_data["panel"],
        trading_calendar=list(_CALENDAR),
        engine_kwargs=engine_kwargs,
        feature_columns=list(_MODEL_COLUMNS),
        train_params={},
        engine_mode="v2",
        verbose=False,
    )


# ---------------------------------------------------------------------------
# 产物快照与比对（读侧口径：经 PaperStateStore 加载，缺文件 = 读侧缺省值）
# ---------------------------------------------------------------------------


def _snapshot_root(state_base: Path) -> Dict[str, Any]:
    """快照一个状态根的全部持久产物（读侧口径：缺文件 ⇒ None/[]，与生产读侧一致）。

    runs/ 下仅收 ``{kind}_{date}.json`` 幂等记录（rebalance_state.json 归投影键单列；
    gap_*.json 缺口登记非提交产物，不在比对面）。
    """
    store = PaperStateStore(root_path=str(state_base / "paper"))
    trades = store.load_all_trades()
    nav = store.load_all_nav()
    instructions = {
        path.name: pd.read_parquet(path).to_dict("records")
        for path in sorted(store.instructions_path.glob("*.parquet"))
    }
    runs = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(store.runs_path.glob("*.json"))
        if path.name != "rebalance_state.json" and not path.name.startswith("gap_")
    }
    return {
        "account": store.load_account(),
        "strategy": store.load_strategy_state(),
        "ranked": store.load_ranked_candidates(),
        "stop_loss": store.load_stop_loss_state(),
        "last_trade_date": store.load_last_trade_date(),
        "rebalance": store.load_rebalance_state(),
        "pending_buys": store.load_pending_buys(),
        "pending_sells": store.load_pending_sells(),
        "trades": None if trades is None else trades.to_dict("records"),
        "nav": None if nav is None else nav.to_dict("records"),
        "instructions": instructions,
        "runs": runs,
    }


def _day_rows(state_base: Path, kind: str, day: str) -> List[Dict[str, Any]]:
    """trades/nav 当日行（读侧口径；无文件 ⇒ 空列表）。"""
    store = PaperStateStore(root_path=str(state_base / "paper"))
    frame = store.load_all_trades() if kind == "trades" else store.load_all_nav()
    if frame is None:
        return []
    day_mask = frame["trade_date"].astype(str).str.replace("-", "") == day
    return frame.loc[day_mask].to_dict("records")


def _assert_runs_equal(
    crash_runs: Dict[str, Any], reference_runs: Dict[str, Any], crash_target: str
) -> None:
    """runs 幂等记录比对：剔除 timestamp 后逐键相等。

    blob 已写边界的半提交恢复形态下，day1 记录 = 幂等补写 note 记录（不重跑；
    与参照根的全量记录形态差异属契约形态，由 test_v2_p4_t2_host 幂等补写专测锁定）。
    """
    assert set(crash_runs) == set(
        reference_runs
    ), f"runs 记录文件集不一致: {sorted(crash_runs)} vs {sorted(reference_runs)}"
    for name, reference_record in reference_runs.items():
        crash_record = crash_runs[name]
        if name == f"t0_{_DAY1}.json" and crash_target != "engine_state.bin":
            assert "幂等补写" in str(
                crash_record.get("note", "")
            ), f"{name} 应为幂等补写记录: {crash_record}"
            continue
        crash_keys = {k: v for k, v in crash_record.items() if k != "timestamp"}
        reference_keys = {k: v for k, v in reference_record.items() if k != "timestamp"}
        assert crash_keys == reference_keys, f"runs 记录 {name} 剔除 timestamp 后不一致"


def _assert_products_equal(
    crash: Dict[str, Any],
    reference: Dict[str, Any],
    crash_target: str,
    *,
    stage: str,
) -> None:
    """全部产物逐行/逐键比对（两阶段均要求与无中断参照根完全收敛——无 stale 宽限）。"""
    for key in (
        "account",
        "strategy",
        "ranked",
        "stop_loss",
        "last_trade_date",
        "rebalance",
        "pending_buys",
        "pending_sells",
        "trades",
        "nav",
        "instructions",
    ):
        assert crash[key] == reference[key], f"{stage} 阶段 {key} 与无中断参照根不一致"
    _assert_runs_equal(crash["runs"], reference["runs"], crash_target)


def _inject_write_crash(
    store: PaperStateStore, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """按目标文件名在对应原子写方法注入 OSError（实例级补丁，仅作用于该 store）。"""
    marker = f"模拟写盘崩溃: {target}"
    if target.endswith(".bin"):
        original_bytes = store._atomic_write_bytes

        def crash_bytes(path: Path, data: bytes) -> None:
            if Path(path).name == target:
                raise OSError(marker)
            return original_bytes(path, data)

        monkeypatch.setattr(store, "_atomic_write_bytes", crash_bytes)
    elif target.endswith(".parquet"):
        original_parquet = store._atomic_write_parquet

        def crash_parquet(path: Path, frame: pd.DataFrame) -> None:
            if Path(path).name == target:
                raise OSError(marker)
            return original_parquet(path, frame)

        monkeypatch.setattr(store, "_atomic_write_parquet", crash_parquet)
    else:
        original_json = store._atomic_write_json

        def crash_json(path: Path, payload: Any, *, indent: Optional[int] = 2) -> None:
            if Path(path).name == target:
                raise OSError(marker)
            return original_json(path, payload, indent=indent)

        monkeypatch.setattr(store, "_atomic_write_json", crash_json)


# ---------------------------------------------------------------------------
# 1. 崩溃边界矩阵（评审硬项 1）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_products(tmp_path_factory, shared_data) -> Dict[str, Any]:
    """无中断参照根：逐日驱动 day0→day3（每日新实例 = 跨进程形态），分两段快照。"""
    base = tmp_path_factory.mktemp("reference")
    for day in (_DAY0, _DAY1):
        _make_host(base, shared_data, freq=1).run_t0(day)
    after_day1 = _snapshot_root(base)
    for day in (_DAY2, _DAY3):
        _make_host(base, shared_data, freq=1).run_t0(day)
    return {"after_day1": after_day1, "after_day3": _snapshot_root(base)}


class TestCrashBoundaryMatrix:
    """day1 commit 各写入边界崩溃 ⇒ 恢复后产物与无中断参照根逐位一致。

    前提形态（freq=1 合成小区间实测锁定）：day1 = 2 笔买入成交 + T+1 指令帧（买 C/D），
    day2 无成交（跳空语义），day3 = 2 笔持有期到期卖出。
    """

    @pytest.mark.parametrize("target", _CRASH_TARGETS)
    def test_commit_crash_boundary_recovers_to_reference(
        self,
        tmp_path,
        monkeypatch,
        reference_products,
        shared_data,
        target,
    ):
        base = tmp_path
        _make_host(base, shared_data, freq=1).run_t0(_DAY0)  # 第 0 代完整提交
        host = _make_host(base, shared_data, freq=1)
        _inject_write_crash(host.store, target, monkeypatch)
        with pytest.raises(OSError, match="模拟写盘崩溃"):
            host.run_t0(_DAY1)

        # (a) 新建宿主实例同日重入：不抛错
        host_b = _make_host(base, shared_data, freq=1)
        host_b.run_t0(_DAY1)
        result = host_b.last_day_result
        assert result is not None
        if target == "engine_state.bin":
            assert (
                result.driven is True and result.t0_status == "success"
            ), "blob 未写入边界 ⇒ 当日未提交，同日重入 = 正常重驱动"
        else:
            assert (
                result.driven is False and result.t0_status == "already_run"
            ), "blob 已写边界 ⇒ 完整代核对重建后幂等读产物（不重复执行）"
        # (b) 全部产物与无中断参照根逐行/逐键相等（runs 记录剔除 timestamp）
        _assert_products_equal(
            _snapshot_root(base),
            reference_products["after_day1"],
            target,
            stage="same_day",
        )
        # (c) 再次同日调用：幂等读产物，成交/NAV 不重复
        trades_before = _snapshot_root(base)["trades"]
        nav_before = _snapshot_root(base)["nav"]
        host_b.run_t0(_DAY1)
        assert host_b.last_day_result is not None
        assert host_b.last_day_result.t0_status == "already_run"
        assert _snapshot_root(base)["trades"] == trades_before
        assert _snapshot_root(base)["nav"] == nav_before
        assert len(_day_rows(base, "trades", _DAY1)) == 2
        assert len(_day_rows(base, "nav", _DAY1)) == 1
        # (d) 跨日：继续驱动 day2/day3 ⇒ day1 行仍恰一份，全部产物与参照收官一致
        for day in (_DAY2, _DAY3):
            host_x = _make_host(base, shared_data, freq=1)
            host_x.run_t0(day)
            assert host_x.last_day_result is not None
            assert host_x.last_day_result.driven is True
        assert len(_day_rows(base, "trades", _DAY1)) == 2
        assert len(_day_rows(base, "nav", _DAY1)) == 1
        assert len(_day_rows(base, "trades", _DAY3)) == 2, "day3 到期卖出 2 笔（跨日推进证据）"
        _assert_products_equal(
            _snapshot_root(base), reference_products["after_day3"], target, stage="cross_day"
        )


class TestCrossDayRecovery:
    @pytest.mark.parametrize("target", _CRASH_TARGETS)
    def test_next_day_first_restart_publishes_previous_generation(
        self, tmp_path, monkeypatch, shared_data, target
    ):
        """11 个中断点均覆盖「首次重启直接次日」，旧代不得永久缺发布标记。"""
        _make_host(tmp_path, shared_data, freq=1).run_t0(_DAY0)
        host = _make_host(tmp_path, shared_data, freq=1)
        _inject_write_crash(host.store, target, monkeypatch)
        with pytest.raises(OSError, match="模拟写盘崩溃"):
            host.run_t0(_DAY1)

        restarted = _make_host(tmp_path, shared_data, freq=1)
        restarted.run_t0(_DAY2)
        store = restarted.store
        if target == "engine_state.bin":
            assert not store.check_run_exists("t0", _DAY1), "权威未提交则无旧代可发布"
        else:
            assert store.check_run_exists("t0", _DAY1), "次日推进前必须补写原 t0 发布标记"
            assert not store.check_run_exists("t1", _DAY1), "不得把原 t0 代猜成其他相位"
            assert len(_day_rows(tmp_path, "trades", _DAY1)) == 2
            assert len(_day_rows(tmp_path, "nav", _DAY1)) == 1
        assert store.check_run_exists("t0", _DAY2)
        trades_before = _snapshot_root(tmp_path)["trades"]
        nav_before = _snapshot_root(tmp_path)["nav"]
        restarted.run_t0(_DAY2)
        assert _snapshot_root(tmp_path)["trades"] == trades_before
        assert _snapshot_root(tmp_path)["nav"] == nav_before


class TestRecoveryConfigurationDrift:
    def test_cost_change_does_not_block_or_rewrite_published_trades(self, tmp_path, shared_data):
        _make_host(tmp_path, shared_data).run_t0(_DAY0)
        _make_host(tmp_path, shared_data).run_t0(_DAY1)
        trades_before = _day_rows(tmp_path, "trades", _DAY1)
        nav_before = _day_rows(tmp_path, "nav", _DAY1)
        assert len(trades_before) == 2 and len(nav_before) == 1

        restarted = _make_host(tmp_path, shared_data)
        restarted.cost_model = CostModel(**dict(_COST_SETTINGS, commission_rate=0.01))
        restarted.run_t0(_DAY2)  # 启动恢复不得因历史成本分解按新配置不同而永久阻断
        assert _day_rows(tmp_path, "trades", _DAY1) == trades_before
        assert _day_rows(tmp_path, "nav", _DAY1) == nav_before

    def test_initial_capital_change_does_not_rewrite_published_nav(self, tmp_path, shared_data):
        _make_host(tmp_path, shared_data).run_t0(_DAY0)
        _make_host(tmp_path, shared_data).run_t0(_DAY1)
        nav_before = _day_rows(tmp_path, "nav", _DAY1)
        assert len(nav_before) == 1

        restarted = _make_host(tmp_path, shared_data)
        restarted.engine_kwargs["initial_capital"] = 500_000.0
        restarted.run_t0(_DAY1)
        assert restarted.last_day_result is not None
        assert restarted.last_day_result.t0_status == "already_run"
        assert _day_rows(tmp_path, "nav", _DAY1) == nav_before


# ---------------------------------------------------------------------------
# 2. 单写入者整日写锁（评审硬项 2）
# ---------------------------------------------------------------------------


class TestWholeDayWriteLock:
    def test_second_store_holding_lock_blocks_host_until_release(self, tmp_path, shared_data):
        _make_host(tmp_path, shared_data).run_t0(_DAY0)
        blocker = PaperStateStore(root_path=str(tmp_path / "paper"))
        blocker.acquire_write_lock()
        try:
            host = _make_host(tmp_path, shared_data)
            with pytest.raises(PaperLockError):
                host.run_t0(_DAY1)
            # 锁冲突 fail-fast：当日未推进、零副作用
            assert host.store.get_authority_last_date() == _DAY0
            assert not host.store.check_run_exists("t0", _DAY1)
        finally:
            blocker.release_write_lock()
        # 释放后重试 ⇒ 正常驱动
        host_retry = _make_host(tmp_path, shared_data)
        host_retry.run_t0(_DAY1)
        assert host_retry.last_day_result is not None
        assert host_retry.last_day_result.driven is True
        assert len(host_retry.last_day_result.fills) == 2


# ---------------------------------------------------------------------------
# 3. commit_day 比较交换守卫（评审硬项 3）
# ---------------------------------------------------------------------------


class TestCommitCompareAndSwap:
    def test_stale_expected_base_rejected_equal_accepted(self, tmp_path):
        store = PaperStateStore(root_path=str(tmp_path / "paper"))
        store.commit_day(**_commit_kwargs(date=SS_D0))
        assert store.get_authority_last_date() == SS_D0
        # 过期基线 ⇒ 拒绝（锁内核对，零写入：权威末日不变）
        with pytest.raises(PaperStateError, match="比较交换"):
            store.commit_day(**_commit_kwargs(date=SS_D1, expected_base_date="20260101"))
        assert store.get_authority_last_date() == SS_D0
        # 等价基线 ⇒ 正常提交
        store.commit_day(**_commit_kwargs(date=SS_D1, expected_base_date=SS_D0))
        assert store.get_authority_last_date() == SS_D1
        # 缺省（不传）⇒ 跳过校验（仅供测试/工具直调形态；宿主驱动路径恒显式传入）
        store.commit_day(**_commit_kwargs(date="20260604"))
        assert store.get_authority_last_date() == "20260604"

    def test_same_store_other_thread_rejected_after_cas_barrier(self, tmp_path, monkeypatch):
        """同一 store 异线程不得绕过整日锁，在 CAS 后交错覆盖新代。"""
        store = PaperStateStore(root_path=str(tmp_path / "paper"))
        store.commit_day(**_commit_kwargs(date=SS_D0))
        cas_passed = threading.Event()
        release_first = threading.Event()
        original_write = store._atomic_write_bytes
        worker_errors: List[BaseException] = []

        def pause_first_blob_write(path: Path, data: bytes) -> None:
            if path.name == "engine_state.bin" and threading.current_thread().name == "commit-A":
                cas_passed.set()
                if not release_first.wait(timeout=5.0):
                    raise TimeoutError("测试屏障等待超时")
            original_write(path, data)

        def commit_first() -> None:
            try:
                store.commit_day(**_commit_kwargs(date=SS_D1, expected_base_date=SS_D0))
            except BaseException as exc:  # 测试须汇集线程异常，避免 pytest 仅告警
                worker_errors.append(exc)

        monkeypatch.setattr(store, "_atomic_write_bytes", pause_first_blob_write)
        worker = threading.Thread(target=commit_first, name="commit-A")
        worker.start()
        assert cas_passed.wait(timeout=5.0), "线程 A 必须在 CAS 后、权威写入前停下"
        with pytest.raises(PaperLockError, match="其他线程"):
            store.commit_day(**_commit_kwargs(date="20260604", expected_base_date=SS_D0))
        assert store.get_authority_last_date() == SS_D0
        assert not store.check_run_exists("t0", "20260604")
        release_first.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert worker_errors == []
        assert store.get_authority_last_date() == SS_D1, "权威日期不得回退或被第二提交覆盖"
        assert store.check_run_exists("t0", SS_D1)
        assert not store.check_run_exists("t0", "20260604")


# ---------------------------------------------------------------------------
# 4. fills 三形态（评审硬项 4，驱动日语义）
# ---------------------------------------------------------------------------


class TestFillsThreeForms:
    @staticmethod
    def _reference_fills(base: Path, shared_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """隔离参照根跑出 day1 权威 fills（dict 投影形态）。"""
        _make_host(base, shared_data).run_t0(_DAY0)
        host = _make_host(base, shared_data)
        host.run_t1(_DAY1)
        assert host.last_day_result is not None
        fills = _fills_as_dicts(host.last_day_result)
        assert len(fills) == 2, "前提：day1 引擎撮合 2 笔买入"
        return fills

    def test_mismatch_precommit_zero_side_effect_then_retry(self, tmp_path, shared_data):
        _make_host(tmp_path, shared_data).run_t0(_DAY0)
        host = _make_host(tmp_path, shared_data)
        blob_before = (tmp_path / "paper/state/engine_state.bin").read_bytes()
        with pytest.raises(PaperHostError, match="提交前核对拒绝"):
            host.run_t1(_DAY1, fills=[])  # 外部声称零成交 vs 引擎 2 笔
        # 零落盘：权威停在 day0、快照字节不变、无当日 trades/nav/runs 记录
        store = host.store
        assert store.get_authority_last_date() == _DAY0
        assert (tmp_path / "paper/state/engine_state.bin").read_bytes() == blob_before
        assert store.load_all_trades() is None
        assert store.load_all_nav() is None
        assert not store.check_run_exists("t1", _DAY1)
        # 修正后重试（fills=None ⇒ 引擎行情记账为权威）⇒ 正常提交、成交恰一次
        host.run_t1(_DAY1)
        assert host.last_day_result is not None and host.last_day_result.driven is True
        trades = store.load_all_trades()
        assert trades is not None and len(trades) == 2
        assert store.check_run_exists("t1", _DAY1)
        assert store.get_authority_last_date() == _DAY1

    def test_consistent_fills_commit_exactly_once(self, tmp_path, shared_data):
        reference_fills = self._reference_fills(tmp_path / "ref", shared_data)
        _make_host(tmp_path, shared_data).run_t0(_DAY0)
        host = _make_host(tmp_path, shared_data)
        host.run_t1(_DAY1, fills=reference_fills)  # 提交前核对一致 ⇒ 正常提交
        assert host.last_day_result is not None and host.last_day_result.driven is True
        trades = host.store.load_all_trades()
        assert trades is not None and len(trades) == 2, "成交恰一次入账（引擎已入账，不重复）"
        # 同日再次以一致 fills 调用 ⇒ 幂等读路径接受，仍恰一次
        host.run_t1(_DAY1, fills=reference_fills)
        assert host.last_day_result is not None
        assert host.last_day_result.t0_status == "already_run"
        trades = host.store.load_all_trades()
        assert trades is not None and len(trades) == 2

    def test_mismatch_readback_after_commit_keeps_committed_state(self, tmp_path, shared_data):
        _make_host(tmp_path, shared_data).run_t0(_DAY0)
        host = _make_host(tmp_path, shared_data)
        host.run_t1(_DAY1)  # fills=None ⇒ 引擎行情记账提交
        assert host.last_day_result is not None
        committed_fills = _fills_as_dicts(host.last_day_result)
        trades_before = host.store.load_all_trades()
        account_before = host.store.load_account()
        blob_before = (tmp_path / "paper/state/engine_state.bin").read_bytes()
        bad = committed_fills[:-1]  # 少一笔 ⇒ 键集合不一致
        with pytest.raises(PaperHostError, match="无法回滚"):
            host.run_t1(_DAY1, fills=bad)
        # 幂等读路径报错不 rollback：已提交状态不变
        trades_after = host.store.load_all_trades()
        assert trades_before is not None and trades_after is not None
        assert trades_after.to_dict("records") == trades_before.to_dict("records")
        assert host.store.load_account() == account_before
        assert (tmp_path / "paper/state/engine_state.bin").read_bytes() == blob_before
        assert host.store.get_authority_last_date() == _DAY1


# ---------------------------------------------------------------------------
# 5. live 排期兜底边界（评审硬项 5；§17.7 硬项⑥）
# ---------------------------------------------------------------------------


class TestLiveScheduleFallback:
    def test_first_start_schedule_error_propagates(self, tmp_path, monkeypatch, shared_data):
        """分支 1「尚无成功」构表异常上冒（宿主不吞）：init 放行、apply_live_policy 注入。"""
        original = live_schedule.build_tranche_schedule_from_anchor
        calls = {"count": 0}

        def boom_after_init(*args: Any, **kwargs: Any) -> Any:
            calls["count"] += 1
            if calls["count"] >= 2:
                raise RuntimeError("模拟构表崩溃")
            return original(*args, **kwargs)

        monkeypatch.setattr(live_schedule, "build_tranche_schedule_from_anchor", boom_after_init)
        host = _make_host(tmp_path, shared_data)
        with pytest.raises(RuntimeError, match="模拟构表崩溃"):
            host.run_t0(_DAY0)
        # 上冒即当日失败：零落盘（无快照、无幂等记录）
        assert not (tmp_path / "paper/state/engine_state.bin").exists()
        assert not host.store.check_run_exists("t0", _DAY0)

    def test_anchored_schedule_error_fail_open(self, tmp_path, monkeypatch, shared_data):
        """分支 3/4（已有成功锚定）构表异常 ⇒ fail-open 不抛错，当日按 tranche0 放行。"""
        _make_host(tmp_path, shared_data).run_t0(_DAY0)  # 成功锚定（last_rebalance_date=day0）

        def always_boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("模拟构表崩溃")

        monkeypatch.setattr(live_schedule, "build_tranche_schedule_from_anchor", always_boom)
        host = _make_host(tmp_path, shared_data)
        host.run_t0(_DAY1)  # 恢复启动 ⇒ 分支 3/4 整段异常被 fail-open 吞掉
        result = host.last_day_result
        assert result is not None and result.driven is True
        assert result.t0_status == "success", "fail-open 当日放行 tranche0 ⇒ 按调仓日处理"
        assert len(result.fills) == 2, "day0 计划在 day1 照常成交"


# ---------------------------------------------------------------------------
# 6. 远端只读 fail-fast（评审硬项 6）+ create_layout=False 只读形态（处置项 10）
# ---------------------------------------------------------------------------


class _StubSMBReader:
    """最小 SMB 读取桩（宿主构造期 fail-fast 只消费 store.is_remote，不会真读）。"""

    def read_json(self, path: str) -> Dict[str, Any]:
        return {}

    def read_yaml(self, path: str) -> Dict[str, Any]:
        return {}

    def read_parquet(self, path: str) -> None:
        return None


class TestRemoteStoreRejected:
    def test_host_construction_rejects_remote_store(self, tmp_path, shared_data):
        remote_store = PaperStateStore(
            root_path=str(tmp_path / "paper"), smb_reader=_StubSMBReader()
        )
        assert remote_store.is_remote
        assert not (tmp_path / "paper").exists(), "远端只读模式不建本地目录"
        with pytest.raises(PaperHostError, match="远端只读"):
            _make_host(tmp_path, shared_data, store=remote_store)


class TestCreateLayoutFalse:
    def test_no_layout_created_and_write_fail_fast(self, tmp_path):
        store = PaperStateStore(root_path=str(tmp_path / "paper"), create_layout=False)
        assert not (tmp_path / "paper").exists(), "只读配置形态不建七子目录"
        with pytest.raises(PaperStateError, match="状态目录不存在"):
            store.acquire_write_lock()
        with pytest.raises(PaperStateError, match="状态目录不存在"):
            store.commit_day(**_commit_kwargs())


# ---------------------------------------------------------------------------
# 7. _fill_identity Mapping 分支防御（评审硬项 7）
# ---------------------------------------------------------------------------


class TestFillIdentityDefense:
    def test_missing_ts_code_shares_price_keys(self):
        with pytest.raises(PaperHostError, match="缺必需键.*shares.*price"):
            PaperHost._fill_identity({"ts_code": "000001.SZ", "action": "buy"})
        with pytest.raises(PaperHostError, match="缺必需键.*ts_code"):
            PaperHost._fill_identity({"action": "buy", "shares": 100, "price": 10.0})

    def test_missing_action_rejected_with_context(self):
        with pytest.raises(PaperHostError, match="缺 action/side 键.*000001.SZ"):
            PaperHost._fill_identity({"ts_code": "000001.SZ", "shares": 100, "price": 10.0})

    def test_bad_numeric_shape_rejected(self):
        with pytest.raises(PaperHostError, match="形态非法"):
            PaperHost._fill_identity(
                {"ts_code": "000001.SZ", "action": "buy", "shares": "一百股", "price": 10.0}
            )

    def test_non_mapping_non_fill_rejected(self):
        with pytest.raises(PaperHostError, match="形态非法"):
            PaperHost._fill_identity(42)

    def test_side_alias_accepted(self):
        identity = PaperHost._fill_identity(
            {"ts_code": "000001.SZ", "side": "buy", "shares": 100, "price": 10.0}
        )
        assert identity == ("000001.SZ", "buy", 100, 10.0)


# ---------------------------------------------------------------------------
# 8. 持有期到期 reason 带交易日天数（评审硬项 8）
# ---------------------------------------------------------------------------


class TestHoldingPeriodReason:
    def test_reason_includes_trade_day_count(self):
        trade_dates = list(_CALENDAR_STRS)
        buy_date, trigger_date = trade_dates[0], trade_dates[9]
        frame = build_instruction_frame(
            signal_date=trigger_date,
            condition_sells={
                "000001.SZ": {
                    "trigger_date": pd.Timestamp(trigger_date),
                    "sell_type": "holding_period",
                }
            },
            positions={"000001.SZ": {"shares": 1000, "buy_date": pd.Timestamp(buy_date)}},
            trade_dates=trade_dates,
        )
        expected_days = calc_holding_trade_days(buy_date, trigger_date, trade_dates)
        assert expected_days == 9, "前提：持有天数 = 交易日历索引差（不含买入当日）"
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["action"] == "sell" and row["shares"] == 1000
        assert row["reason"] == f"持有期到期: 持有{expected_days}天"

    def test_reason_static_fallback_without_trade_dates(self):
        frame = build_instruction_frame(
            signal_date=_CALENDAR_STRS[9],
            condition_sells={
                "000001.SZ": {
                    "trigger_date": pd.Timestamp(_CALENDAR_STRS[9]),
                    "sell_type": "holding_period",
                }
            },
            positions={"000001.SZ": {"shares": 1000, "buy_date": pd.Timestamp(_CALENDAR_STRS[0])}},
        )
        assert frame.iloc[0]["reason"] == "持有期到期", "不传 trade_dates ⇒ 回退静态文本"
