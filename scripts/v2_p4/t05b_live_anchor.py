# -*- coding: utf-8 -*-
"""T0.5b spike：live 持久化锚定四场景 + 隔离根（R14）实证（阶段专用临时开发文件）。

覆盖 P4 规划 §5-T0.5b 清单的「live 持久化锚定场景」（F25/D1 排期来源二分）：

- 场景A 首次运行 tranche0：无 rebalance_state ⇒ (True, 0)，计划日落当日；
- 场景B 分批命中：锚定日 + freq/K 一致 ⇒ 命中排期返回对应 tranche_idx；
- 场景C 配置变更重建 anchor：存储 freq/K 与当前不一致 ⇒ (True, 0) 且
  计划日 = 当前交易日（当日成为新周期锚）；
- 场景D 漏批逐日追赶：last_scheduled 落后多个批次 ⇒ 返回最早未完成批次
  （计划日 < 当日，逐日消化）；
- 场景E 锚定等价：anchor == 窗口首元素时 ``build_tranche_schedule_from_anchor``
  与 ``compute_tranche_schedule`` 逐键一致（门 2 live 侧构造前提）。

同时实证 R14：``PaperStorage(root_path=<隔离 tmp 根>)`` 可注入任意根，
调仓状态文件读写全部落在隔离根内（不触生产 ``data/paper/``）。

驱动对象为旧纸面链路的**生产实现** ``paper/runner/rebalance.py``
（PaperRebalanceMixin._check_rebalance_day），以最小 harness 装配
（paper_storage 注入隔离根、日历用合成桩），不改被测代码。

复杂度豁免依据：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.lazybull.paper.runner.rebalance import PaperRebalanceMixin
from src.lazybull.paper.storage import PaperStorage
from src.lazybull.v2.common.rules.stagger import (
    build_tranche_schedule_from_anchor,
    compute_tranche_schedule,
)


class _AnchorHarness(PaperRebalanceMixin):
    """最小装配：隔离根 PaperStorage + 合成交易日历桩（仅供给被测方法）。"""

    def __init__(self, paper_root: Path, trade_dates: List[str]):
        self.paper_storage = PaperStorage(root_path=str(paper_root), verbose=False)
        self._trade_dates = list(trade_dates)
        self._resolved_rebalance_plan_date: Optional[str] = None
        # 不分批路径（_check_single_rebalance_day）经 self.loader 读交易日历
        self.loader = SimpleNamespace(
            load_clean_trade_cal=lambda: pd.DataFrame(
                {"cal_date": list(self._trade_dates), "is_open": [1] * len(self._trade_dates)}
            )
        )

    def _get_open_trade_dates(self) -> List[str]:
        return list(self._trade_dates)


@dataclass
class AnchorScenarioResult:
    """单场景结果。"""

    name: str
    passed: bool
    detail: str


def _check(harness: _AnchorHarness, trade_date: str, freq: int, k: int) -> Tuple[bool, int, str]:
    is_due, tranche_idx = harness._check_rebalance_day(trade_date, freq, k)
    return is_due, tranche_idx, harness._resolved_rebalance_plan_date


def run_anchor_scenarios(paper_root: Path, trade_dates: List[str], freq: int = 20, k: int = 3) -> List[AnchorScenarioResult]:
    """依次跑 A~E 五场景；``paper_root`` 为隔离临时根（每场景独立子目录）。"""
    results: List[AnchorScenarioResult] = []

    # ── A. 首次运行 tranche0（无 rebalance_state）──
    h = _AnchorHarness(paper_root / "case_a", trade_dates)
    is_due, tranche, plan_date = _check(h, trade_dates[0], freq, k)
    ok = (is_due, tranche, plan_date) == (True, 0, trade_dates[0])
    results.append(
        AnchorScenarioResult(
            "A_first_run_tranche0",
            ok,
            f"is_due={is_due}, tranche={tranche}, plan_date={plan_date}（期望 True/0/{trade_dates[0]}）",
        )
    )

    # ── B. 分批命中（锚定 + 配置一致）──
    anchor = trade_dates[0]
    h = _AnchorHarness(paper_root / "case_b", trade_dates)
    h.paper_storage.save_rebalance_state(
        {
            "last_rebalance_date": anchor,
            "tranche_anchor_date": anchor,
            "last_scheduled_rebalance_date": anchor,
            "rebalance_freq": freq,
            "stagger_tranches": k,
        }
    )
    schedule = build_tranche_schedule_from_anchor(anchor, trade_dates, freq, k)
    future_hits = sorted(d for d in schedule if d > anchor)
    if not future_hits:
        results.append(AnchorScenarioResult("B_stagger_hit", False, "排期表无未来命中日（日历过短）"))
    else:
        hit_date = future_hits[0]
        expected_tranche = schedule[hit_date]
        is_due, tranche, plan_date = _check(h, hit_date, freq, k)
        ok = (is_due, tranche, plan_date) == (True, expected_tranche, hit_date)
        results.append(
            AnchorScenarioResult(
                "B_stagger_hit",
                ok,
                f"命中日 {hit_date}: is_due={is_due}, tranche={tranche}（期望 {expected_tranche}）, "
                f"plan_date={plan_date}",
            )
        )

    # ── C. 配置变更重建 anchor（存储 freq 与当前不一致）──
    h = _AnchorHarness(paper_root / "case_c", trade_dates)
    h.paper_storage.save_rebalance_state(
        {
            "last_rebalance_date": anchor,
            "tranche_anchor_date": anchor,
            "last_scheduled_rebalance_date": anchor,
            "rebalance_freq": freq,
            "stagger_tranches": k,
        }
    )
    changed_day = trade_dates[min(3, len(trade_dates) - 1)]
    is_due, tranche, plan_date = _check(h, changed_day, freq + 1, k)
    ok = (is_due, tranche, plan_date) == (True, 0, changed_day)
    results.append(
        AnchorScenarioResult(
            "C_config_change_rebuild_anchor",
            ok,
            f"freq {freq}->{freq + 1}: is_due={is_due}, tranche={tranche}, plan_date={plan_date}"
            f"（期望 True/0/{changed_day} = 当日重建锚）",
        )
    )

    # ── D. 漏批逐日追赶（last_scheduled 落后多批 ⇒ 取最早未完成批次）──
    h = _AnchorHarness(paper_root / "case_d", trade_dates)
    full_schedule = build_tranche_schedule_from_anchor(anchor, trade_dates, freq, k)
    scheduled_sorted = sorted(full_schedule)
    if len(scheduled_sorted) < 3:
        results.append(AnchorScenarioResult("D_missed_batch_catchup", False, "排期日不足 3 个"))
    else:
        # 已完成批次 = 第 1 个计划日；当前日 = 第 3 个计划日 ⇒ 到期 = 第 2、3 个，取最早
        done_date, catchup_from, today = (
            scheduled_sorted[0],
            scheduled_sorted[1],
            scheduled_sorted[2],
        )
        h.paper_storage.save_rebalance_state(
            {
                "last_rebalance_date": done_date,
                "tranche_anchor_date": anchor,
                "last_scheduled_rebalance_date": done_date,
                "rebalance_freq": freq,
                "stagger_tranches": k,
            }
        )
        is_due, tranche, plan_date = _check(h, today, freq, k)
        expected_tranche = full_schedule[catchup_from]
        ok = (is_due, tranche, plan_date) == (True, expected_tranche, catchup_from)
        results.append(
            AnchorScenarioResult(
                "D_missed_batch_catchup",
                ok,
                f"漏批追赶: 当日 {today}，执行最早未完成计划日 {plan_date}（期望 {catchup_from}"
                f" < 当日；tranche={tranche} 期望 {expected_tranche}）",
            )
        )

    # ── E. 锚定 == 窗口首元素时两排期函数逐键一致（门 2 live 侧构造前提）──
    sched_anchor = build_tranche_schedule_from_anchor(anchor, trade_dates, freq, k)
    sched_window = compute_tranche_schedule(trade_dates, freq, k)
    ok = sched_anchor == sched_window
    results.append(
        AnchorScenarioResult(
            "E_anchor_equals_window_start",
            ok,
            f"anchor={anchor}: from_anchor {len(sched_anchor)} 键 vs window {len(sched_window)} 键, "
            f"逐键相等={ok}",
        )
    )

    return results


def drive_paper_oracle(
    paper_root: Path,
    trade_dates: List[str],
    rebalance_freq: int,
    stagger_tranches: int,
    blocked: frozenset = frozenset(),
    config_events: Optional[Dict[str, Tuple[int, int]]] = None,
    drive_days: Optional[List[str]] = None,
    force_days: frozenset = frozenset(),
) -> List[Tuple[str, int, str]]:
    """以旧链**生产函数**逐日驱动调仓语义，产出调仓日轨迹（语义参照侧）。

    - 判定 = 生产 ``PaperRebalanceMixin._check_rebalance_day``（隔离根 PaperStorage）；
    - 成功 T0 后的状态保存复刻 ``paper/runner/execution.py:267-290``：
      last_rebalance=当日（实际成功日）、last_scheduled=计划日、freq/K 落盘、
      K>1 且（配置变更/force/tranche0）⇒ tranche_anchor=计划日，否则沿用旧值；
    - ``blocked``：T0 无产出日（信号为空）——不保存状态（配置变更同样不提交，
      次日仍按旧配置检出变更后从当日重建）；
    - ``config_events``：``{YYYYMMDD: (new_freq, new_k)}`` 当日起生效；
    - ``drive_days``：实际驱动日序列（缺省=全日历；空洞日不驱动 = 停机/不开市）；
    - ``force_days``：强制 T0 日（空仓提前调仓等；复刻 ``execution.py:101-114``：
      跳过调仓日校验、计划日=当日、tranche_idx=0）。

    Returns:
        [(trade_date, tranche_idx, plan_date)] 成功调仓日轨迹。
    """
    harness = _AnchorHarness(paper_root, trade_dates)
    cur_freq, cur_k = rebalance_freq, stagger_tranches
    days = drive_days if drive_days is not None else trade_dates
    trace: List[Tuple[str, int, str]] = []
    for day in days:
        if config_events and day in config_events:
            cur_freq, cur_k = config_events[day]
        if day in force_days:
            # 强制 T0（旧链 force_rebalance 路径：execution.py:101-114）
            is_due, tranche_idx, plan_date = True, 0, day
        else:
            try:
                is_due, tranche_idx = harness._check_rebalance_day(day, cur_freq, cur_k)
            except RuntimeError:
                continue  # 非调仓日
            if not is_due or day in blocked:
                continue  # 非调仓日或 T0 无产出（不保存状态）
            plan_date = harness._resolved_rebalance_plan_date
        prev = harness.paper_storage.load_rebalance_state() or {}
        config_changed = (
            prev.get("rebalance_freq", cur_freq) != cur_freq
            or prev.get("stagger_tranches", 1) != cur_k
        )
        state = {
            "last_rebalance_date": day,
            "last_scheduled_rebalance_date": plan_date,
            "rebalance_freq": cur_freq,
            "stagger_tranches": cur_k,
        }
        if cur_k > 1 and (config_changed or tranche_idx == 0):
            state["tranche_anchor_date"] = plan_date
        elif cur_k > 1 and prev.get("tranche_anchor_date"):
            state["tranche_anchor_date"] = prev["tranche_anchor_date"]
        harness.paper_storage.save_rebalance_state(state)
        trace.append((day, tranche_idx, plan_date))
    return trace


def snapshot_protected_root(root: Path) -> Dict[str, str]:
    """受保护根文件集快照：``{相对路径: sha256}``（根不存在 ⇒ 空集，照常可比）。"""
    import hashlib

    root = Path(root)
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def verify_isolated_root_containment(
    isolated_root: Path, protected_root: Path, protected_before: Dict[str, str]
) -> Tuple[bool, str]:
    """R14 实证（前后快照口径，评审 R2-06/r1-P2-2/r3-P2-2 处置）。

    双条件：① 受保护根（生产 ``data/paper/``）在场景运行前后**零变更**
    （文件集 + 逐文件 sha256 一致——根外写入必然落在受保护根外，而唯一合法的
    旧链写入口就是受保护根本身，故「受保护根零变更」⟺「零越界写入」的可证伪形式）；
    ② 隔离根内预期状态文件（rebalance_state.json）在册且全部位于隔离根内。
    """
    isolated_root = Path(isolated_root)
    if not isolated_root.exists():
        return False, f"隔离根不存在: {isolated_root}"
    files = [p for p in isolated_root.rglob("*") if p.is_file()]
    rebalance_files = [p for p in files if p.name == "rebalance_state.json"]
    if not rebalance_files:
        return False, "隔离根内未找到 rebalance_state.json（场景未实际写入）"

    after = snapshot_protected_root(protected_root)
    if after != protected_before:
        added = sorted(set(after) - set(protected_before))
        removed = sorted(set(protected_before) - set(after))
        changed = sorted(k for k in after.keys() & protected_before if after[k] != protected_before[k])
        return (
            False,
            f"受保护根发生变更: 新增 {added[:3]}, 删除 {removed[:3]}, 内容变化 {changed[:3]}",
        )
    return (
        True,
        f"受保护根（{protected_root}）前后快照零变更（{len(after)} 个文件）；"
        f"隔离根内文件 {len(files)} 个（含 rebalance_state {len(rebalance_files)} 个）",
    )
