# -*- coding: utf-8 -*-
"""P5b T5 崩盘窗选择（设计 v4 §5.4.3 唯一规则 + §5.5「回撤窗口」行）。

冻结规则（不得实现后调整）
========================
1. 只在 **B0 最后 500 个收益样本**（:data:`mdd.WINDOW_SAMPLES`）上识别全部峰到谷 drawdown
   episode；全链 1714 日只作历史描述，**不参与选窗**；
2. 按深度降序取**最多 3 个互不重叠** episode；深度相同取 **peak 序位较早**者；
3. 窗口 = ``peak_date+1 .. trough_date``（序位语义：收益索引 ``[peak, trough)``）；
4. B0 无回撤或不足一个 episode ⇒ ``not_constructible``；
5. 窗口集合在训练、holdout 与全部噪声实现间**固定**，运行期必须与 manifest 登记的窗口
   逐字段一致（:func:`assert_windows_fixed`），禁止按实现重选（选择内生）。

序位语义
========
``peak``/``trough`` 是**累计曲线网格**上的位置（``0..N``，``0`` = 窗口起点，收益 ``t`` 落在
``cum[t]``）；因此窗口覆盖的收益索引是 ``[peak, trough)``。``ordinal`` 给出窗口内最早/最深
收益的 0 基序位，供报告与 manifest 登记（源 CSV 无自然日，只有 fold-local ``date``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    DrawdownReadingError,
    cumulative_log_path,
    log_to_percent,
    max_drawdown_log,
)

__all__ = [
    "MAX_CRASH_WINDOWS",
    "CrashWindow",
    "CrashWindowError",
    "assert_windows_fixed",
    "crash_window_mask",
    "find_drawdown_episodes",
    "select_crash_windows",
    "window_mdd_percent",
    "window_section",
]

#: 最多保留的崩盘窗个数（§5.4.3）
MAX_CRASH_WINDOWS = 3


class CrashWindowError(TestbedError):
    """崩盘窗不可构造/与冻结值不符（一律 fail-closed）。"""

    __test__ = False


@dataclass(frozen=True)
class CrashWindow:
    """一个峰到谷 episode（累计曲线网格位置 + 深度 + 选择名次）。"""

    peak: int
    trough: int
    depth_log: float
    rank: int = 0

    @property
    def start(self) -> int:
        """窗口在收益数组上的 0 基起点（= ``peak``，对应 ``peak_date+1``）。"""
        return int(self.peak)

    @property
    def stop(self) -> int:
        """窗口在收益数组上的 0 基终点（不含；= ``trough``）。"""
        return int(self.trough)

    @property
    def length(self) -> int:
        """窗口覆盖的收益样本数（``trough - peak``）。"""
        return int(self.trough - self.peak)

    @property
    def depth_pp(self) -> float:
        """深度（pp，负值；越负越深）。"""
        return float(log_to_percent(np.asarray([self.depth_log], dtype=float))[0])

    def overlaps(self, other: "CrashWindow") -> bool:
        """闭区间 ``[peak, trough]`` 是否相交（含端点，防嵌套 episode 被同时选中）。"""
        return not (self.trough < other.peak or other.trough < self.peak)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "peak": int(self.peak),
            "trough": int(self.trough),
            "depth_pp": self.depth_pp,
            "window_start_ordinal": self.start,
            "window_stop_ordinal": self.stop,
            "window_samples": self.length,
        }


def find_drawdown_episodes(log_returns: np.ndarray) -> Tuple[CrashWindow, ...]:
    """识别全部峰到谷 episode（peak 取当前 running max，trough 取该段最深点）。

    段定义：从 running max 处（``dd == 0``）出发，直到回撤重新回到 0（或序列结束）为止的
    最深点。未回到 0 的尾部段仍作为 episode 记录（trough = 序列末最深点），否则会低估风险。

    Args:
        log_returns: ``(N,)`` 逐日 log 收益。

    Returns:
        按出现顺序排列的 episode 元组（``rank`` 全为 0，由 :func:`select_crash_windows` 赋）。

    Raises:
        DrawdownReadingError: 输入非有限/维度非法（由 :func:`cumulative_log_path` 抛出）。
    """
    values = np.asarray(log_returns, dtype=float)
    if values.ndim != 1:
        raise CrashWindowError(f"崩盘窗识别只接受 1D 收益序列，实际 shape={values.shape}")
    cumulative = cumulative_log_path(values)
    peak_curve = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak_curve
    episodes = []
    index = 0
    total = int(cumulative.size)
    while index < total:
        if drawdown[index] < 0:
            index += 1
            continue
        cursor = index + 1
        deepest = index
        best = 0.0
        while cursor < total and drawdown[cursor] < 0:
            if drawdown[cursor] < best:
                best = float(drawdown[cursor])
                deepest = cursor
            cursor += 1
        if deepest > index and best < 0.0:
            episodes.append(CrashWindow(peak=int(index), trough=int(deepest), depth_log=best))
        index = cursor
    return tuple(episodes)


def select_crash_windows(
    episodes: Iterable[CrashWindow], *, max_windows: int = MAX_CRASH_WINDOWS
) -> Tuple[CrashWindow, ...]:
    """按冻结规则取最多 ``max_windows`` 个不重叠窗口（深度降序、同深取较早 peak）。

    Returns:
        按**时间顺序**（peak 升序）排列的窗口元组，``rank`` = 深度名次（0 最深）。

    Raises:
        CrashWindowError: 没有任何 episode（B0 无回撤 ⇒ ``not_constructible``）。
    """
    items = list(episodes)
    if not items:
        raise CrashWindowError(
            "B0 在冻结窗口内没有峰到谷回撤 episode ⇒ 回撤注入 not_constructible"
            "（设计 §5.4.3：无回撤不得构造崩盘窗）"
        )
    limit = int(max_windows)
    if limit <= 0:
        raise CrashWindowError(f"max_windows 必须为正，实际 {max_windows}")
    ordered = sorted(items, key=lambda item: (item.depth_log, item.peak))
    chosen = []
    for candidate in ordered:
        if any(candidate.overlaps(existing) for existing in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= limit:
            break
    ranked = [
        CrashWindow(peak=item.peak, trough=item.trough, depth_log=item.depth_log, rank=position)
        for position, item in enumerate(chosen)
    ]
    return tuple(sorted(ranked, key=lambda item: item.peak))


def crash_window_mask(samples: int, windows: Sequence[CrashWindow]) -> np.ndarray:
    """窗口并集的布尔掩码（``(samples,)``；窗口外恒 False）。

    Raises:
        CrashWindowError: 窗口越界或 ``samples`` 非法（禁止静默裁剪）。
    """
    total = int(samples)
    if total <= 0:
        raise CrashWindowError(f"samples 必须为正，实际 {samples}")
    mask = np.zeros(total, dtype=bool)
    for window in windows:
        if window.start < 0 or window.stop > total or window.length <= 0:
            raise CrashWindowError(
                f"崩盘窗越界：peak={window.peak} trough={window.trough} 不在 0..{total}"
            )
        start = window.start
        stop = window.stop
        mask[start:stop] = True
    return mask


def window_section(windows: Sequence[CrashWindow], *, samples: int) -> Dict[str, Any]:
    """manifest/report 的崩盘窗段（冻结清单 + 规则文字，全部可逐字比对）。"""
    return {
        "window_samples": int(samples),
        "max_windows": int(MAX_CRASH_WINDOWS),
        "selection_rule": (
            "在 B0 最后 window_samples 个收益样本上识别全部峰到谷 episode，按深度降序取最多 "
            "max_windows 个互不重叠者"
        ),
        "tie_break": "深度相同取 peak 序位较早者",
        "boundary_rule": "窗口 = peak_date+1..trough_date（收益索引 [peak, trough)）",
        "fixed_for_all_realizations": True,
        "not_constructible_when": "B0 无回撤或不足一个 episode",
        "frozen_list": [window.as_dict() for window in windows],
        "selection_uses_only_b0": True,
        "full_chain_used_for_selection": False,
    }


def assert_windows_fixed(
    frozen: Sequence[Mapping[str, Any]], recomputed: Sequence[CrashWindow], *, samples: int
) -> None:
    """运行期校验窗口集合与 manifest 冻结值逐字段一致（禁止按实现重选）。

    Raises:
        CrashWindowError: 字段缺失/数量不符/peak-trough-深度不一致。
    """
    fixed = list(frozen or [])
    if len(fixed) != len(recomputed):
        raise CrashWindowError(
            f"崩盘窗个数与 manifest 冻结值不符：冻结 {len(fixed)} 个，实际 {len(recomputed)} 个"
            "（禁止按实现重选窗口）"
        )
    if not fixed:
        raise CrashWindowError("manifest 未冻结任何崩盘窗（回撤仪器不可运行）")
    for stored, window in zip(fixed, recomputed):
        for key, actual in (
            ("peak", int(window.peak)),
            ("trough", int(window.trough)),
            ("window_samples", int(window.length)),
        ):
            if int(stored.get(key, -(10**9))) != actual:
                raise CrashWindowError(
                    f"崩盘窗 {key} 与 manifest 冻结值不符：冻结 {stored.get(key)!r} 实际 {actual}"
                )
        if abs(float(stored.get("depth_pp", 0.0)) - window.depth_pp) > 1e-9:
            raise CrashWindowError(
                "崩盘窗深度与 manifest 冻结值不符："
                f"冻结 {stored.get('depth_pp')!r} 实际 {window.depth_pp!r}"
            )
    mask_check = crash_window_mask(samples, recomputed)
    if not mask_check.any():
        raise CrashWindowError("崩盘窗并集为空（回撤注入不可构造）")


def window_mdd_percent(log_returns: np.ndarray) -> float:
    """窗口（或全链）基准 MDD（pp，便于与 T0.5b 预检数值对照）。"""
    values = np.asarray(log_returns, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise DrawdownReadingError(f"MDD 输入必须是非空 1D，实际 shape={values.shape}")
    raw = np.asarray([max_drawdown_log(cumulative_log_path(values))], dtype=float)
    return float(log_to_percent(raw)[0])
