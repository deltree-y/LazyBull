"""live 持久化锚定排期（P4-T1；D1 排期来源二分之 live 侧，F25 语义复刻）。

排期来源二分：
- 回放/replay = 窗口锚定（``run()`` 内 ``_get_rebalance_dates`` 以窗口首元素为锚，
  与回测同源，门 2b/门 2/门 3 证明）；
- live/纸面 = **持久化锚定**（本模块）：锚定状态（``LiveAnchorState``）随
  ``EngineState`` 快照同代落盘/恢复；**排期表每日自锚经
  ``build_tranche_schedule_from_anchor`` 以当前完整日历重建**——与旧链
  ``paper/runner/rebalance.py::_check_staggered_rebalance_day`` 每日重建同一形态
  （日历水位增长自动覆盖；无「持久排期表删除回流」——提前调仓的改道效果由
  成功 T0 后的**锚前移**承载，与旧链 ``execution.py:267-290`` 一致）。

判定语义（复刻 ``rebalance.py::_check_rebalance_day`` 的判定顺序）：
1. **尚无成功 T0**（``last_rebalance_date`` 为 None）⇒ 当日放行 tranche0
   （``_resolved_rebalance_plan_date = 当日``；旧链 :32-41）——配置变更检出在
   首次成功前不单独生效（旧链判定顺序：无成功记录先于配置变化）；
2. **配置变更**（freq/K 与锚定指纹不一致）⇒ 当日拟议调仓日（tranche 0）+
   新排期提案；**提案—成功 T0 后提交**（T0 失败零提交，次日仍按旧配置检出）；
3. **K=1**：按**实际成功日**（``last_rebalance_date``）计交易日判到期
   （``idx(today) - idx(last_rebalance_date) >= freq``；排期表不构成成员资格，
   工作副本纯到期注入）；日历/日期异常 ⇒ **fail-open 放行**（旧链 :80-117
   ``except`` 兜底语义复刻，记 error 日志）；
4. **K>1**：``due = (last_scheduled, today] ∩ 排期表``；当日执行最早到期批次
   （漏批逐日追赶）；当日自身批次（若在排期）留待后续逐日消化；
   **非到期日当日不得因表成员资格自行触发**（due 为空 ⇒ 从工作副本剔除当日，
   复刻旧链 due_dates 为空 ⇒ 非调仓日）；异常 ⇒ fail-open 放行（旧链 :184-186）。

提交语义（``settle_live_day``，复刻 ``execution.py:267-290``）：
- 成功 = ``pending_signals`` 含当日**调仓**计划（带 decision_trace；补位/顺延
  计划不算）；T0 失败零提交；
- 提交：``last_rebalance_date`` = 当日（实际成功日）；
  ``last_scheduled_rebalance_date`` = due[0]（计划日；无 due 的强制/早调仓 = 当日）；
  配置指纹（freq/K）恒同步为引擎当前值；K>1 且（提案提交或 tranche0 成功）
  ⇒ 锚 = 计划日（提前调仓 ⇒ 当日）；
- K=1 不写锚定日（旧链 K=1 不落 tranche_anchor_date）。

**前提写死（D1）**：live 排期不得由「驱动首日」重锚——锚定状态随
``EngineState`` 快照同代落盘/恢复（blob ``live_anchor_state`` 键）。

本模块为无状态纯函数 + 两个值对象；引擎侧状态载体 = ``engine._signal_dates``
（live 形态 = 当日排期记录，非权威排期表）与 ``engine.live_anchor_state``。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.v2.common.rules.stagger import build_tranche_schedule_from_anchor

#: live 锚定状态字段（to_dict/from_dict 的封闭键集，fail-fast 校验用）
LIVE_ANCHOR_KEYS: Tuple[str, ...] = (
    "anchor_date",
    "rebalance_freq",
    "stagger_tranches",
    "last_scheduled_rebalance_date",
    "last_rebalance_date",
)


class LiveScheduleError(RuntimeError):
    """live 排期政策错误（锚定状态缺字段/形态非法）。"""


def _check_anchor_date_field(name: str, value: Any) -> None:
    """锚定日期字段类型校验（pd.Timestamp 或 None）。"""
    if value is not None and not isinstance(value, pd.Timestamp):
        raise LiveScheduleError(f"live_anchor_state.{name} 类型非法: 期望 Timestamp/None，实际 {type(value).__name__}")


@dataclass
class LiveAnchorState:
    """live 持久化锚定状态（F25 三字段 + 配置指纹；随引擎快照同代落盘）。

    - ``anchor_date``：批次0锚定日（对应旧纸面 ``tranche_anchor_date``；
      K>1 时分批排期的锚）；
    - ``last_rebalance_date``：最近**实际成功 T0** 的交易日（K=1 判到期唯一来源，
      对应旧链 ``_check_single_rebalance_day`` 只读该字段的语义）；
    - ``last_scheduled_rebalance_date``：最近成功 T0 的**计划日**（K>1 时 due
      窗口左端；漏批追赶 = ``(last_scheduled, today] ∩ 排期表`` 逐日消化最早批次）；
    - ``rebalance_freq`` / ``stagger_tranches``：锚定提交时点的配置指纹
      （配置变更检出 = 与引擎当前配置比较）。
    """

    anchor_date: Optional[pd.Timestamp]
    rebalance_freq: int
    stagger_tranches: int
    last_scheduled_rebalance_date: Optional[pd.Timestamp] = None
    last_rebalance_date: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为快照 blob 投影（封闭键集）。"""
        return {
            "anchor_date": self.anchor_date,
            "rebalance_freq": self.rebalance_freq,
            "stagger_tranches": self.stagger_tranches,
            "last_scheduled_rebalance_date": self.last_scheduled_rebalance_date,
            "last_rebalance_date": self.last_rebalance_date,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiveAnchorState":
        """自快照 blob 恢复（**fail-fast**：缺键/多键/类型非法即拒绝，D2 生产恢复契约）。"""
        keys = set(data.keys())
        expected = set(LIVE_ANCHOR_KEYS)
        if keys != expected:
            raise LiveScheduleError(
                f"live_anchor_state 键集不符: 期望 {sorted(expected)}，实际 {sorted(keys)}"
            )
        anchor = cls(
            anchor_date=data["anchor_date"],
            rebalance_freq=data["rebalance_freq"],
            stagger_tranches=data["stagger_tranches"],
            last_scheduled_rebalance_date=data["last_scheduled_rebalance_date"],
            last_rebalance_date=data["last_rebalance_date"],
        )
        for name in ("anchor_date", "last_scheduled_rebalance_date", "last_rebalance_date"):
            _check_anchor_date_field(name, getattr(anchor, name))
        for name in ("rebalance_freq", "stagger_tranches"):
            value = getattr(anchor, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise LiveScheduleError(
                    f"live_anchor_state.{name} 非法: 期望正整数，实际 {value!r}"
                )
        return anchor


@dataclass
class LiveDaySchedule:
    """live 当日排期判定结果（提案—提交制的暂存上下文）。

    - ``working``：当日工作副本（引擎当日消费对象；到期注入/漏批覆写/早调仓
      删除均为当日有效，不回流——改道效果由成功 T0 后的锚前移承载）；
    - ``due``：到期批次计划日清单（首个为当日执行的批次计划日）；
    - ``proposal``：配置变更提案（无变更时 None）；
    - ``rebuilt``：提案对应的新排期表（无提案时 None）；
    - ``effective``：当日自锚重建的排期表（提案时 = rebuilt）；
      settle 时写回 ``engine._signal_dates`` 作为当日排期记录。
    """

    working: Dict[pd.Timestamp, int]
    due: List[pd.Timestamp]
    proposal: Optional[Dict[str, Any]]
    rebuilt: Optional[Dict[pd.Timestamp, int]]
    effective: Dict[pd.Timestamp, int]


def init_live_anchor(
    engine: Any,
    anchor_date: pd.Timestamp,
    calendar: List[pd.Timestamp],
) -> None:
    """live 首次启动的锚定初始化（**仅限首次**；恢复启动禁止调用——锚随快照恢复）。

    写入：``engine._signal_dates``（自锚推算的初始排期记录）、
    ``engine._cycle_anchor_idx``（锚在完整日历中的位置）、
    ``engine.live_anchor_state``（配置指纹 = 引擎当前配置，两个 last_* 字段
    None = 从未成功 T0）。
    """
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    if anchor_date not in date_to_idx:
        raise LiveScheduleError(f"锚定日 {anchor_date} 不在完整日历内，无法初始化 live 锚定")
    engine._signal_dates = build_tranche_schedule_from_anchor(
        anchor_date, calendar, engine.rebalance_freq, engine.stagger_tranches
    )
    engine._cycle_anchor_idx = date_to_idx[anchor_date]
    engine.live_anchor_state = LiveAnchorState(
        anchor_date=anchor_date,
        rebalance_freq=engine.rebalance_freq,
        stagger_tranches=engine.stagger_tranches,
    )


def _fail_open_day(date: pd.Timestamp, reason: str) -> "LiveDaySchedule":
    """fail-open 形态（旧链 except 兜底复刻）：记 error 日志 + 当日放行 tranche0。"""
    logger.error(f"live 排期判定异常（{reason}），按 fail-open 放行当日 tranche0（旧链兜底语义）")
    working = {date: 0}
    return LiveDaySchedule(working, [date], None, None, working)


def apply_live_policy(
    engine: Any,
    date: pd.Timestamp,
    calendar: List[pd.Timestamp],
) -> LiveDaySchedule:
    """live 排期每日判定（**纯提案**：不改锚定状态）。

    判定顺序复刻旧链 ``_check_rebalance_day``：尚无成功 T0 ⇒ 当日放行 tranche0
    → 配置变更 ⇒ 提案 → K=1 实际成功日计数 / K>1 计划日窗口 + 漏批追赶。
    排期表每日自锚以当前日历重建（日历水位增长自动覆盖）。

    调用方义务：返回的 ``working`` 须赋给 ``engine._signal_dates`` 后再调
    ``run_step``；日终调 :func:`settle_live_day` 提交/写回排期记录。
    """
    anchor = engine.live_anchor_state
    if anchor is None:
        raise LiveScheduleError("live 排期判定需要 engine.live_anchor_state（锚定状态缺失）")

    # 1. 尚无成功 T0（旧链 :32-41：无状态/无 last_rebalance_date ⇒ 当日放行 tranche0）
    if anchor.last_rebalance_date is None:
        working = {date: 0}
        # 排期记录 = 自锚重建表（诊断/探针用；对 K=1 不构成成员资格）
        effective = build_tranche_schedule_from_anchor(
            anchor.anchor_date, calendar, engine.rebalance_freq, engine.stagger_tranches
        )
        return LiveDaySchedule(working, [date], None, None, effective)

    # 2. 配置变更检出 ⇒ 提案（当日拟议 tranche0 + 新排期表自当日起重建）
    if (
        anchor.rebalance_freq != engine.rebalance_freq
        or anchor.stagger_tranches != engine.stagger_tranches
    ):
        proposal = {
            "anchor_date": date,
            "rebalance_freq": engine.rebalance_freq,
            "stagger_tranches": engine.stagger_tranches,
        }
        rebuilt = build_tranche_schedule_from_anchor(
            date, calendar, engine.rebalance_freq, engine.stagger_tranches
        )
        working = dict(rebuilt)
        working[date] = rebuilt.get(date, 0)
        return LiveDaySchedule(working, [date], proposal, rebuilt, rebuilt)

    # 3/4. 排期表自锚每日重建（旧链 :148-150 同形态）
    try:
        effective = build_tranche_schedule_from_anchor(
            anchor.anchor_date, calendar, engine.rebalance_freq, engine.stagger_tranches
        )
        if engine.stagger_tranches <= 1:
            # K=1：按实际成功日计数（旧链 _check_single_rebalance_day 语义；
            # 排期表不构成成员资格——工作副本纯到期注入）
            last_idx = calendar.index(anchor.last_rebalance_date)
            cur_idx = calendar.index(date)
            due = [date] if cur_idx - last_idx >= engine.rebalance_freq else []
            working = {date: 0} if due else {}
            return LiveDaySchedule(working, due, None, None, effective)
        # K>1：due = (last_scheduled, today] ∩ 排期表，当日执行最早到期批次
        last = anchor.last_scheduled_rebalance_date
        due = sorted(d for d in effective if (last is None or d > last) and d <= date)
        working = dict(effective)
        if due:
            working[date] = effective[due[0]]
        else:
            # 非到期日不得因表成员资格自行触发（已履行/未到期；复刻旧链 due 空 ⇒ 非调仓日）
            working.pop(date, None)
        return LiveDaySchedule(working, due, None, None, effective)
    except Exception as exc:
        return _fail_open_day(date, str(exc))


def settle_live_day(
    engine: Any,
    date: pd.Timestamp,
    day: LiveDaySchedule,
) -> None:
    """live 日终结算：成功 T0 后提交锚定状态 + 写回当日排期记录。

    提交语义（旧链 ``execution.py:267-290`` 复刻：成功 T0 后才落盘
    rebalance_state）：
    - 成功 = ``pending_signals`` 含当日**调仓**计划（带 decision_trace；
      补位/顺延计划不算）；T0 失败零提交（次日仍按旧配置检出变更）；
    - 提交：``last_rebalance_date`` = 当日（实际成功日）；
      ``last_scheduled_rebalance_date`` = due[0]（计划日；无 due 的强制/早调仓
      = 当日）；配置指纹（freq/K）恒同步为引擎当前值；K>1 且（提案提交或
      tranche0 成功）⇒ 锚 = 计划日；K=1 不写锚定日；
    - ``engine._signal_dates`` 写回当日自锚重建的排期记录（live 形态下其为
      记录而非权威表——权威语义由锚定状态承载，次日自锚重建）。
    """
    anchor = engine.live_anchor_state
    if anchor is None:
        raise LiveScheduleError("live 日终结算需要 engine.live_anchor_state（锚定状态缺失）")

    if date in engine.pending_signals:
        entry = engine.pending_signals[date]
        if "decision_trace" in entry:
            anchor.last_rebalance_date = date
            anchor.last_scheduled_rebalance_date = day.due[0] if day.due else date
            anchor.rebalance_freq = engine.rebalance_freq
            anchor.stagger_tranches = engine.stagger_tranches
            if day.proposal is not None:
                # 配置变更提案提交：锚 = 当日（提案日），排期自当日重建（次日 apply 生效）
                anchor.anchor_date = day.proposal["anchor_date"]
            elif engine.stagger_tranches > 1 and entry.get("tranche_idx") == 0:
                # tranche 0 成功 ⇒ 锚前移到计划日（含提前调仓 ⇒ 当日）
                anchor.anchor_date = day.due[0] if day.due else date

    engine._signal_dates = day.rebuilt if day.rebuilt is not None else day.effective
