# -*- coding: utf-8 -*-
"""P5b T5 回撤仪器的实现生成：B0 冻结窗口 + 配对噪声实现（设计 v4 §5.4.3/§5.5）。

B0 冻结窗口
===========
T0.5b 已批口径（`docs/reports/p5b_t05b_feasibility_20261101.md` §5/§7）：回撤窗口统一取
**B0 最后 500 个收益样本**、``T_anchor=500``；全链 1714 日只作历史描述，**不参与调参或
混合读数**。本模块把该口径写死：

- 读 B0 链式净值 CSV（列 ``date``/``nav``/``split_index``）；
- 按 ``(split_index, date)`` 排序、**折内**相邻日 pct_change ⇒ 每个折 ``n-1`` 个收益；
- 取最后 :data:`mdd.WINDOW_SAMPLES` 个收益并按 ``ln(1+r)`` 转 log 尺度；
- 同时登记全链 MDD 与全链前三个 episode 深度，**显式标注 usable_for_calibration=false**
  （禁止把 1714 与 500 锚混用）。

配对噪声实现
===========
每个实现是 ``x = B0_window + n``：``n`` 由收益裁判冻结的四类场景（``ma20``/``ar1_06``/
``regime_hetero``/``real_seed_block``）按同一 ``DgpParams``（``path_days=500``、``block_length=H``、
circular 环回、按池自身均值去中心、不截断）生成。A/A 需要**同分布独立路径**：一次抽
``2M`` 条噪声后对半切分（``arm`` / ``base``），**禁止**用同一路径零注入冒充 A/A。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.crash_window import (
    window_mdd_percent,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import WINDOW_SAMPLES
from src.lazybull.v2.hosts.testbed.referee.scenes import DgpParams, SceneRegistry

__all__ = [
    "RealizationError",
    "SourceWindow",
    "draw_arm_paths",
    "draw_noise_paths",
    "load_source_window",
    "window_identity",
]

#: 全链历史描述里登记的 episode 个数（仅描述，不参与任何校准）
_HISTORY_EPISODES = 3


class RealizationError(TestbedError):
    """实现生成非法（源文件缺失/哈希不符/窗口过短/注册表角色不符）。"""

    __test__ = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceWindow:
    """B0 的冻结窗口（最后 500 个收益样本）+ 全链历史描述（不参与校准）。"""

    log_returns: np.ndarray
    keys: Tuple[Tuple[int, int], ...]
    source_path: str
    source_sha256: str
    total_returns: int
    fold_count: int
    mdd_percent: float
    history: Mapping[str, Any]

    @property
    def samples(self) -> int:
        return int(self.log_returns.size)

    def key_of(self, ordinal: int) -> Tuple[int, int]:
        """窗口内 0 基序位 → ``(split_index, date)``（源 CSV 无自然日）。"""
        return tuple(self.keys[int(ordinal)])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "samples": self.samples,
            "total_returns": int(self.total_returns),
            "fold_count": int(self.fold_count),
            "mdd_percent": float(self.mdd_percent),
            "first_key": list(self.key_of(0)),
            "last_key": list(self.key_of(self.samples - 1)),
            "date_semantics": "fold_local_ordinal",
            "usable_for_calibration": True,
            "full_chain": dict(self.history),
        }


def load_source_window(
    base_path: Path,
    *,
    expected_sha256: Optional[str] = None,
    window: int = WINDOW_SAMPLES,
    expected_mdd_percent: Optional[float] = None,
    mdd_tolerance: float = 1e-4,
) -> SourceWindow:
    """读取 B0 链式净值并取最后 ``window`` 个收益样本（冻结口径）。

    Args:
        base_path: B0 ``chain_nav`` CSV。
        expected_sha256: 冻结哈希（不符即拒绝换源）。
        window: 窗口收益样本数（冻结 500；不得改成全链 1714）。
        expected_mdd_percent: 可选的基准 MDD 核对值（T0.5b 已批读数），用于证明
            「窗口未被重选」。
        mdd_tolerance: 上述核对的绝对容差（pp）。

    Returns:
        :class:`SourceWindow`。

    Raises:
        RealizationError: 文件缺失/哈希不符/列缺失/窗口不足/回撤未复现。
    """
    source = Path(base_path)
    if not source.is_file():
        raise RealizationError(f"B0 链式净值缺失（回撤仪器不可运行）: {source}")
    actual_sha = _sha256(source)
    if expected_sha256 is not None and actual_sha != str(expected_sha256):
        raise RealizationError(
            f"B0 链式净值哈希与冻结值不符（禁止换源）: {source} 实际={actual_sha}"
        )
    frame = pd.read_csv(source)
    for column in ("split_index", "date", "nav"):
        if column not in frame.columns:
            raise RealizationError(f"B0 链式净值缺列 {column!r}（实际列 {list(frame.columns)}）")
    ordered = frame.sort_values(["split_index", "date"]).reset_index(drop=True)
    # 每个折的首行没有相邻前值 ⇒ 逐折丢一行（不是只丢全局第一行）
    changes = ordered.groupby("split_index")["nav"].pct_change()
    valid = changes.notna()
    returns = changes[valid].to_numpy(dtype=float)
    if returns.size < int(window):
        raise RealizationError(f"B0 收益样本 {returns.size} 少于冻结窗口 {window}（窗口不可构造）")
    # 显式转 Python int：manifest 封存用 canonical JSON，numpy 标量不可序列化
    keys = [
        (int(fold), int(day))
        for fold, day in zip(
            ordered.loc[valid, "split_index"].to_numpy(dtype=int),
            ordered.loc[valid, "date"].to_numpy(dtype=int),
        )
    ]
    if len(keys) != returns.size:
        raise RealizationError(f"收益键与收益条数不一致: keys={len(keys)} returns={returns.size}")
    size = int(window)
    window_returns = returns[-size:]
    window_keys = keys[-size:]
    if not np.isfinite(window_returns).all() or (window_returns <= -1.0).any():
        raise RealizationError("B0 窗口收益含非有限值或 <= -100%（无法取对数）")
    log_returns = np.log1p(window_returns)
    mdd = window_mdd_percent(log_returns)
    if expected_mdd_percent is not None and abs(mdd - float(expected_mdd_percent)) > float(
        mdd_tolerance
    ):
        raise RealizationError(
            "B0 冻结窗口的基准 MDD 与 T0.5b 已批读数不符（窗口被重选或源被替换）: "
            f"实际={mdd!r} 期望={expected_mdd_percent!r} 容差={mdd_tolerance}"
        )
    history = _history_description(returns)
    return SourceWindow(
        log_returns=log_returns,
        keys=tuple(window_keys),
        source_path=source.as_posix(),
        source_sha256=actual_sha,
        total_returns=int(returns.size),
        fold_count=int(ordered["split_index"].nunique()),
        mdd_percent=float(mdd),
        history=history,
    )


def _history_description(returns: np.ndarray) -> Dict[str, Any]:
    """全链历史描述（**不参与调参或混合读数**；仅供与 T0.5b 对照）。"""
    from src.lazybull.v2.hosts.testbed.referee.drawdown.crash_window import (
        find_drawdown_episodes,
        select_crash_windows,
    )

    log_full = np.log1p(np.asarray(returns, dtype=float))
    try:
        episodes = find_drawdown_episodes(log_full)
        chosen = select_crash_windows(episodes, max_windows=_HISTORY_EPISODES)
        depths = [
            {"rank": int(item.rank), "depth_pp": float(item.depth_pp)}
            for item in sorted(chosen, key=lambda item: item.depth_pp)
        ]
    except TestbedError:
        depths = []
    return {
        "window_samples": int(log_full.size),
        "mdd_percent": float(window_mdd_percent(log_full)),
        "top_episode_depths_pp": depths,
        "usable_for_calibration": False,
        "note": (
            "全链 1714 日（1700 收益）只作历史描述；T0.5b 已批口径冻结为最后 500 个收益样本，"
            "两者禁止混用或混合读数"
        ),
    }


def draw_noise_paths(
    registry: SceneRegistry, scene: str, rng: np.random.Generator, count: int
) -> np.ndarray:
    """按场景抽 ``count`` 条噪声路径（``(count, path_days)``）。

    Raises:
        RealizationError: 路径数非正或注册表场景不符。
    """
    total = int(count)
    if total <= 0:
        raise RealizationError(f"路径数必须为正，实际 {count!r}")
    return np.asarray(registry.build(scene, rng, total), dtype=float)


def draw_arm_paths(
    window: SourceWindow,
    registry: SceneRegistry,
    scene: str,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    """抽 ``count`` 条**独立**实现 ``x = B0_window + n``（``(count, samples)``）。

    A/A 用：这些实现是独立抽取的噪声路径，基准是冻结的 B0 窗口本身（``base = B0``），
    因此 ``D_k`` 里保留噪声、且**不是**同一路径零注入（后者会让 ``D_k ≡ 0``）。

    Raises:
        RealizationError: 噪声天数与窗口长度不符（禁止错位相加）。
    """
    noise = draw_noise_paths(registry, scene, rng, int(count))
    if noise.shape[1] != window.samples:
        raise RealizationError(
            f"噪声路径天数 {noise.shape[1]} 与 B0 冻结窗口 {window.samples} 不符"
            "（DGP path_days 必须等于窗口样本数）"
        )
    return window.log_returns[None, :] + noise


def window_identity(window: SourceWindow, dgp: DgpParams) -> Dict[str, Any]:
    """manifest ``source_window`` 段的身份摘要（含窗口 MDD 与全链描述）。"""
    return {
        "window": window.as_dict(),
        "dgp": dgp.as_dict(),
        "noise_model": "x = B0_window + n；n 由收益裁判冻结的四类场景按同一 DgpParams 生成",
        "pairing": (
            "A/A 的 D_k 由两条独立同分布路径逐实现配对（arm=B0+n_a 对 base=B0+n_b）；"
            "效应单元的 D_k 以冻结基准臂 B0 为减数（两侧机制不同，各自在对应段落登记）"
            "（B0+n 或 B0+n+inj），禁止同一路径零注入"
        ),
        "baseline": "B0 冻结窗口（各 look 的 MDD_base 只算一次，全实现共用）",
        "anchor_horizon": int(WINDOW_SAMPLES),
    }
