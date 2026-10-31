# -*- coding: utf-8 -*-
"""P4 T1 步进等价性驱动（生产实现版；阶段专用临时开发文件）。

复杂度豁免：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。

T1 验收增项①（评审处置 P2-6）：**生产 ``run_step`` 已替换 spike 原型副本
``run_one_day``**——本驱动不再持有日级循环体副本，逐日驱动直接调用
``src/lazybull/v2/core/execution/run_loop.py::BacktestRunLoopMixin.run_step``；
快照导出/恢复 = 生产 ``v2/core/accounting/engine_state.py``（**fail-fast**：
缺字段/schema 版本不符即拒绝，spike 的宽松恢复形态已退役——变异自证机制
相应改为「字段剔除 ⇒ 恢复拒绝」检出）；live 排期 = 生产
``v2/core/execution/live_schedule.py``（提案-提交制，K=1 按实际成功日）。

保留件（宿主模拟线束，非引擎逻辑副本）：逐日「全新引擎 + 落盘恢复」驱动
（``run_stepped``）、live 连续参照（``run_live_continuous``）、连续 run 参照
（``run_continuous_with_probes``）、逐日探针与双侧比对（``compare_runs``）。

判据（同 spike 冻结版）：成交逐笔一致 + 持仓逐日一致 + 旁路产物终态逐值一致 +
NAV max|Δ| ≤ 1e-6 + export→restore→export 字节一致（生产 dumps_blob）。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.lazybull.v2.core.accounting.engine_state import (
    SNAPSHOT_FIELDS,
    dumps_blob,
    export_state,
    loads_blob,
    restore_state,
    state_from_blob,
    state_to_blob,
)
from src.lazybull.v2.core.execution.live_schedule import (
    LiveAnchorState,
    apply_live_policy,
    init_live_anchor,
    settle_live_day,
)

#: 快照字段清单（生产权威 = engine_state.SNAPSHOT_FIELDS；本别名供套件与测试引用）
RUNTIME_STATE_FIELDS: Tuple[str, ...] = SNAPSHOT_FIELDS


def _deep_equal(a, b, path: str = "") -> Optional[str]:
    """值语义无共享深度相等；不等时返回首个差异路径（供归因），相等返回 None。

    与 pickle 字节比对不同：不受跨字段对象共享（memo）与插入序噪声影响；
    float 精确相等（同序计算应逐位一致），NaN==NaN 视为相等；
    ndarray 走 np.array_equal（评审 P3-6：``a == b`` 对数组是歧义真值）。
    """
    if a is None or b is None:
        return None if a is b else f"{path}: {a!r} != {b!r}"
    import numpy as np

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return None if np.array_equal(a, b, equal_nan=True) else f"{path}: ndarray 不等"
        except Exception:  # 非标量数组退化到 repr 比对
            return None if repr(a) == repr(b) else f"{path}: ndarray repr 不等"
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return f"{path}: 键集 {sorted(set(a.keys()) ^ set(b.keys()))} 不对称"
        for key in a:
            diff = _deep_equal(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: 长度 {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            diff = _deep_equal(x, y, f"{path}[{i}]")
            if diff:
                return diff
        return None
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:  # NaN 双側
            return None
        return None if a == b else f"{path}: {a!r} != {b!r}"
    return None if a == b else f"{path}: {a!r} != {b!r}"


@dataclass
class DayProbe:
    """逐日探针记录（比对两侧日终状态用）。"""

    date: str
    positions: Dict
    pending_signal_dates: List[str]
    last_signal_date: Optional[str]
    last_rebalance_nav: Optional[float]
    last_ranked_candidates: List
    signal_dates_remaining: List[str]
    portfolio_value: float
    capital: float
    n_trades: int


@dataclass
class SteppedRunResult:
    """步进驱动产物。"""

    engine_state: Dict  # 末日导出状态 blob（state_to_blob 投影，含 schema_version）
    probes: List[DayProbe] = field(default_factory=list)
    snapshot_roundtrip_ok: bool = True
    # live 形态：末日 T0 计划（最后行情日生成的次日计划；无则为 None）
    last_day_pending_plan: Optional[Dict] = None
    # live 形态：末日锚定状态（随快照落盘的生产 LiveAnchorState）
    live_anchor_state: Optional[LiveAnchorState] = None
    # live 形态：逐日信号日与批次（驱动决策留痕，供锚定场景断言）
    signal_day_trace: List[Tuple[str, int]] = field(default_factory=list)


def export_blob(engine, omit_fields: Sequence[str] = ()) -> Dict:
    """生产 export_state → blob 投影；``omit_fields`` 为变异自证用的字段剔除。

    剔除字段后的 blob 在下次恢复时被生产 fail-fast 拒绝（EngineStateError）——
    即「快照字段遗漏」的检出形态（spike 宽松恢复发散形态的生产替代）。
    """
    blob = state_to_blob(export_state(engine))
    for field_name in omit_fields:
        blob["fields"].pop(field_name, None)
    return blob


def restore_blob(engine, blob: Dict) -> None:
    """生产 fail-fast 恢复（blob 校验 + restore_state）。"""
    restore_state(engine, state_from_blob(blob))


def _roundtrip_bytes_ok(make_engine: Callable[[], object], blob: Dict) -> bool:
    """export→restore→export 字节一致断言（生产 dumps_blob，整 blob 单次往返）。"""
    data1 = dumps_blob(blob)
    probe_engine = make_engine()
    restore_blob(probe_engine, loads_blob(data1))
    data2 = dumps_blob(state_to_blob(export_state(probe_engine)))
    return data2 == data1


def _wire_day_inputs(engine, date_to_idx: Dict, price_data: pd.DataFrame, total_days: int) -> None:
    """每日输入接线（输入接线字段每日重建，不入快照——字段归类注册表 wiring 类）。"""
    engine._trade_date_index = date_to_idx
    engine._prepare_price_index(price_data)
    engine.price_data_cache = price_data
    engine._run_total_days = total_days


def _probe(engine, date: pd.Timestamp) -> DayProbe:
    signal_dates = engine._signal_dates
    portfolio_value = (
        engine.portfolio_values[-1]["portfolio_value"] if engine.portfolio_values else float("nan")
    )
    return DayProbe(
        date=date.strftime("%Y%m%d"),
        positions=pickle.loads(pickle.dumps(engine.positions)),
        pending_signal_dates=sorted(d.strftime("%Y%m%d") for d in engine.pending_signals),
        last_signal_date=(
            engine._last_signal_date.strftime("%Y%m%d")
            if engine._last_signal_date is not None
            else None
        ),
        last_rebalance_nav=engine._last_rebalance_nav,
        last_ranked_candidates=pickle.loads(pickle.dumps(engine._last_ranked_candidates)),
        signal_dates_remaining=sorted(d.strftime("%Y%m%d") for d in signal_dates),
        portfolio_value=portfolio_value,
        capital=engine.current_capital,
        n_trades=len(engine.trades),
    )


def run_stepped(
    make_engine: Callable[[], object],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    trading_dates_full: List[pd.Timestamp],
    price_data: pd.DataFrame,
    live_full_calendar: bool = False,
    anchor_date: Optional[pd.Timestamp] = None,
    omit_fields: Sequence[str] = (),
    resume_state: Optional[Dict] = None,
) -> SteppedRunResult:
    """逐日步进驱动：每日全新引擎实例 + 生产快照落盘/恢复（D1 纸面宿主形态）。

    Args:
        make_engine: 引擎工厂（每次调用返回同配置全新实例）。
        start_date/end_date: 驱动窗口（含端点）。
        trading_dates_full: 完整交易日历（live 形态含窗口外未来交易日）。
        price_data: 行情数据（live 形态可截止于窗口末日，未来日无行情）。
        live_full_calendar: True 时按 live 形态——日历注入完整交易日历，
            排期以持久化锚经生产 ``apply_live_policy``/``settle_live_day``
            逐日判定；False 时按回放形态——窗口锚定（``_get_rebalance_dates``）。
        anchor_date: live 形态**首次启动**的批次0锚定日；None 时取窗口首日。
            恢复启动（``resume_state`` 传入）时该参数被忽略——锚定状态随快照
            同代落盘/恢复（blob ``live_anchor_state`` 键），**禁止驱动首日重锚**。
        omit_fields: 变异自证用——从快照 blob 剔除的字段名（生产 fail-fast
            下剔除 ⇒ 次日恢复拒绝）。
        resume_state: 前一阶段导出的引擎快照 blob（含 schema_version/fields）。
    """
    window = [d for d in trading_dates_full if start_date <= d <= end_date]
    if not window:
        raise ValueError(f"驱动窗口为空: {start_date} ~ {end_date}")
    if live_full_calendar:
        calendar = list(trading_dates_full)
        anchor = anchor_date if anchor_date is not None else window[0]
    else:
        calendar = list(window)
        anchor = window[0]
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    day_list = [(date_to_idx[d], d) for d in window]

    blob: Optional[Dict] = resume_state
    probes: List[DayProbe] = []
    roundtrip_ok = True
    total_days = len(window)
    signal_day_trace: List[Tuple[str, int]] = []

    for idx, date in day_list:
        engine = make_engine()
        _wire_day_inputs(engine, date_to_idx, price_data, total_days)
        if live_full_calendar:
            # live 形态复刻旧纸面「仅空仓提前调仓」语义（P2-5：拖尾分支关闭）
            engine.enable_trailing_early_rebalance = False
        if blob is None:
            if live_full_calendar:
                init_live_anchor(engine, anchor, calendar)
            else:
                engine._signal_dates = engine._get_rebalance_dates(calendar)
                engine._cycle_anchor_idx = 0
        else:
            # 生产 fail-fast 恢复（缺字段/版本不符 ⇒ EngineStateError 上冒）
            restore_blob(engine, blob)

        if live_full_calendar:
            # live 排期政策（生产实现）：纯提案 → 当日工作副本入引擎 →
            # run_step → 日终提交 + 排期记录写回（提案-提交制；每日自锚重建）
            day = apply_live_policy(engine, date, calendar)
            engine._signal_dates = day.working
            engine.run_step(date, calendar, price_data)
            settle_live_day(engine, date, day)
            if date in engine.pending_signals and "decision_trace" in engine.pending_signals[date]:
                signal_day_trace.append(
                    (date.strftime("%Y%m%d"), engine.pending_signals[date].get("tranche_idx"))
                )
        else:
            engine.run_step(date, calendar, price_data)
        probes.append(_probe(engine, date))

        exported = export_blob(engine, omit_fields=omit_fields)
        # export→restore→export 字节一致（omit 变异时跳过该断言）
        if not omit_fields:
            if not _roundtrip_bytes_ok(make_engine, exported):
                roundtrip_ok = False
        blob = exported

    result = SteppedRunResult(
        engine_state=blob,
        probes=probes,
        snapshot_roundtrip_ok=roundtrip_ok,
        live_anchor_state=None,
        signal_day_trace=signal_day_trace,
    )
    if live_full_calendar and blob is not None:
        # 末日锚定状态（blob 投影重建，与落盘形态一致）
        anchor_field = blob["fields"].get("live_anchor_state")
        if anchor_field is not None:
            result.live_anchor_state = LiveAnchorState.from_dict(anchor_field)
        # 末日 T0 计划：窗口最后一日（= 最后已知行情日）生成的次日计划
        last_window_day = window[-1]
        plan = blob["fields"]["pending_signals"].get(last_window_day)
        if plan is not None:
            result.last_day_pending_plan = {
                "signal_date": last_window_day.strftime("%Y%m%d"),
                "n_signals": len(plan.get("signals", {})),
                "tranche_idx": plan.get("tranche_idx"),
            }
    return result


def run_live_continuous(
    make_engine: Callable[[], object],
    day_dates: List[pd.Timestamp],
    trading_dates_full: List[pd.Timestamp],
    price_data: pd.DataFrame,
    anchor_date: pd.Timestamp,
    config_events: Optional[Dict[str, Dict]] = None,
) -> Tuple[object, List[DayProbe], LiveAnchorState, List[Tuple[str, int]]]:
    """live 连续参照侧：**单一引擎实例**跨日存活 + 同一生产 live 排期政策
    逐日判定 + **无每日落盘/恢复**——与步进侧（每日全新实例 + 快照往返）构成
    双侧比较，证明「快照/恢复接缝」在 live 语义（含配置变更/漏批追赶事件序列）
    下行为零变化。

    Args:
        day_dates: 实际驱动的交易日序列（含停机空洞——空洞日不驱动，等价于不开市）；
        trading_dates_full: 完整日历（供 T+1 解析与排期索引）；
        anchor_date: 批次0锚定日（起跑锚，与步进侧注入一致）；
        config_events: ``{YYYYMMDD: {"rebalance_freq":..,"stagger_tranches":..}}``，
            当日起作用于引擎实例（模拟配置变更）。
    """
    config_events = config_events or {}
    engine = make_engine()
    engine.enable_trailing_early_rebalance = False  # live 形态复刻旧链仅空仓语义（P2-5）
    date_to_idx = {d: i for i, d in enumerate(trading_dates_full)}
    _wire_day_inputs(engine, date_to_idx, price_data, len(day_dates))
    init_live_anchor(engine, anchor_date, trading_dates_full)
    probes: List[DayProbe] = []
    trace: List[Tuple[str, int]] = []
    for date in day_dates:
        event = config_events.get(date.strftime("%Y%m%d"))
        if event:
            for attr, value in event.items():
                setattr(engine, attr, value)
        day = apply_live_policy(engine, date, trading_dates_full)
        engine._signal_dates = day.working
        engine.run_step(date, trading_dates_full, price_data)
        settle_live_day(engine, date, day)
        if date in engine.pending_signals and "decision_trace" in engine.pending_signals[date]:
            trace.append((date.strftime("%Y%m%d"), engine.pending_signals[date].get("tranche_idx")))
        probes.append(_probe(engine, date))
    return engine, probes, engine.live_anchor_state, trace


def run_continuous_with_probes(make_engine: Callable[[], object], start_date, end_date, trading_dates, price_data):
    """连续 run 参照侧：包 ``_finalize_day`` 逐日捕获探针（不改类行为）。"""
    engine = make_engine()
    probes: List[DayProbe] = []
    # signal_dates 是引擎态 self._signal_dates（_accept_early_rebalance 会删条目）：
    # 日终读取即为当日全部改写后的终态。_try_early_rebalance 每日必调
    #（run_loop.run_step），是稳定的抓取点；此处只需确认引擎态对象就位。
    orig_finalize = engine._finalize_day

    def _spy_finalize(idx, date, *args):
        orig_finalize(idx, date, *args)
        probes.append(_probe(engine, date))

    engine._finalize_day = _spy_finalize
    nav_df = engine.run(start_date, end_date, trading_dates, price_data)
    return engine, nav_df, probes


@dataclass
class EquivalenceVerdict:
    """等价性判定结果。"""

    passed: bool
    trades_identical: bool
    daily_positions_identical: bool
    sidecar_identical: bool
    nav_max_abs_delta: float
    probe_mismatches: List[str]
    detail: str


def compare_runs(
    cont_engine,
    cont_probes: List[DayProbe],
    stepped: SteppedRunResult,
    probe_fields: Optional[Sequence[str]] = None,
    signal_dates_max: Optional[str] = None,
) -> EquivalenceVerdict:
    """连续 vs 步进逐日比对：成交逐笔 + 持仓逐日 + NAV + 探针字段 + 旁路产物。

    ``probe_fields`` 可裁剪参与比对的探针字段（live 形态末日合法差异由场景层
    单独断言时使用，**只允许收窄到末日前已证一致的字段**——R2-05 处置：
    禁止对整个窗口豁免关键字段）。
    ``signal_dates_max`` 对排期表剩余键做范围归一化（live 侧排期含窗口外
    未来日属合法差异，两側均裁剪到 <= 该日后比对）。
    旁路产物（execution_attribution_records / holdings_snapshots /
    completion_stats，D2 投影清单在册）参与终态逐值比对（P3-3 处置）。
    """
    mismatches: List[str] = []
    trades_identical = True
    daily_positions_identical = True

    # 成交逐笔（精确相等）
    stepped_trades = stepped.engine_state["fields"]["trades"]
    if len(cont_engine.trades) != len(stepped_trades):
        trades_identical = False
        mismatches.append(f"trades 计数: 连续 {len(cont_engine.trades)} != 步进 {len(stepped_trades)}")
    else:
        for i, (t_ref, t_step) in enumerate(zip(cont_engine.trades, stepped_trades)):
            diff = _deep_equal(t_ref, t_step, path=f"trades[{i}]")
            if diff:
                trades_identical = False
                mismatches.append(f"成交不一致: {diff}")
                break

    # 逐日探针（持仓/候选/排期/估值）
    if len(cont_probes) != len(stepped.probes):
        daily_positions_identical = False
        mismatches.append(f"探针日数: 连续 {len(cont_probes)} != 步进 {len(stepped.probes)}")
    for ref, step in zip(cont_probes, stepped.probes):
        if ref.date != step.date:
            mismatches.append(f"探针日期错位: {ref.date} != {step.date}")
            continue
        pos_diff = _deep_equal(ref.positions, step.positions, path="positions")
        if pos_diff:
            daily_positions_identical = False
            mismatches.append(f"{ref.date} 持仓不一致: {pos_diff}")
        for attr in (
            "pending_signal_dates",
            "last_signal_date",
            "last_rebalance_nav",
            "last_ranked_candidates",
            "signal_dates_remaining",
            "portfolio_value",
            "capital",
            "n_trades",
        ):
            if probe_fields is not None and attr not in probe_fields:
                continue
            v_ref, v_step = getattr(ref, attr), getattr(step, attr)
            if attr == "signal_dates_remaining" and signal_dates_max is not None:
                v_ref = [d for d in v_ref if d <= signal_dates_max]
                v_step = [d for d in v_step if d <= signal_dates_max]
            diff = _deep_equal(v_ref, v_step, path=attr)
            if diff:
                mismatches.append(f"{ref.date} 探针字段 {diff}")

    # 旁路产物终态逐值比对（D2 投影清单在册项；P3-3 处置）
    sidecar_identical = True
    for attr in ("execution_attribution_records", "holdings_snapshots", "completion_stats"):
        diff = _deep_equal(
            getattr(cont_engine, attr), stepped.engine_state["fields"][attr], path=attr
        )
        if diff:
            sidecar_identical = False
            mismatches.append(f"旁路产物不一致: {diff}")

    # NAV max|Δ|
    nav_deltas = [
        abs(ref.portfolio_value - step.portfolio_value)
        for ref, step in zip(cont_probes, stepped.probes)
    ]
    nav_max = max(nav_deltas) if nav_deltas else 0.0

    passed = (
        trades_identical
        and daily_positions_identical
        and sidecar_identical
        and nav_max <= 1e-6
        and not mismatches
    )
    return EquivalenceVerdict(
        passed=passed,
        trades_identical=trades_identical,
        daily_positions_identical=daily_positions_identical,
        sidecar_identical=sidecar_identical,
        nav_max_abs_delta=nav_max,
        probe_mismatches=mismatches,
        detail=f"trades={len(cont_engine.trades)} 笔, 探针 {len(cont_probes)} 日, NAV max|Δ|={nav_max:.3e}",
    )
