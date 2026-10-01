# -*- coding: utf-8 -*-
"""信号层尺子固化（P5a-1 交付物④；方案 §3.5/§8-P5a-1）。

把「列级 / 信号级改动先过 ±10bps 尺子（相对门槛 ≥9%）才允许进净值裁决」
的现有实践固化为证据机器的强制流程入口。薄封装：复用生产尺子
`scripts/compare/signal_metrics.py`（v0.122.0 落位）的配对差分 + 分块自举实现，
不复制公式；输入改为 runs schema 的 `topk_detail.parquet`（经 runs_loader 读入）。

口径边界（必须随结论报告，沿 v0.122.0 契约）：信号级代理，不含成本 / 调仓 /
Kelly / 路径，**只作筛选**；最终裁决以链式净值判据（ΔMaxDD 为主）为准。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.lazybull.risk.terminal_loss.block_stats import (
    BootstrapConfig,
    paired_day_mean_ci,
)

#: 尺子噪声带（v0.122.0 首轮标定）：Δ平均持有期收益 ±10 bps（≈±9% 相对）、Δ命中率 ±0.6pp
NOISE_BAND_RETURN_BPS = 10.0
NOISE_BAND_HIT_RATE_PP = 0.6
#: 筛选相对门槛：加列默认带稀释成本，相对降幅 ≥9% 才值得跑净值 A/B
RELATIVE_GATE = 0.09


@dataclass
class SignalRulerVerdict:
    """信号层尺子裁决（筛选口径）。"""

    delta_return_bps: float  # Δ 平均持有期收益（bps）
    delta_return_ci95: tuple  # 95% 区间（bps）
    delta_hit_rate_pp: float  # Δ 命中率（pp）
    delta_hit_rate_ci95: tuple  # 95% 区间（pp）
    relative_drop: float  # 相对基线降幅（正 = 变差）
    passes: bool  # 是否通过筛选（值得跑净值 A/B）
    note: str
    n_signal_days: int
    block_days: int


def _daily_topk_return(topk_df: pd.DataFrame, topk: int) -> pd.Series:
    """逐信号日 Top-K 平均持有期收益（评估单位：一个信号日一个数）。"""
    sub = topk_df[topk_df["topk"] == topk]
    return sub.groupby("trade_date")["true_return"].mean()


def _daily_topk_hit_rate(topk_df: pd.DataFrame, topk: int) -> pd.Series:
    """逐信号日 Top-K 命中率（true_return > 0 占比）。"""
    sub = topk_df[topk_df["topk"] == topk]
    return sub.groupby("trade_date")["true_return"].apply(lambda s: (s > 0).mean())


def ruler_verdict(
    base_topk: pd.DataFrame,
    arm_topk: pd.DataFrame,
    topk: int = 20,
    block_days: int = 20,
    seed: int = 42,
) -> SignalRulerVerdict:
    """信号层尺子裁决：两臂逐日 Top-K 配对差分 + 交易日分块自举区间。

    Args:
        base_topk / arm_topk: 两臂的 topk_detail（runs schema：trade_date/topk/rank/ts_code/true_return）
        topk: Top-K 口径（默认 20）
        block_days: 交易日分块自举块长（默认 20；相邻信号日前瞻窗口重叠，必须分块）
        seed: 随机种子

    Returns:
        SignalRulerVerdict（Δ 收益 bps / Δ 命中率 pp / 95% 区间 / 相对降幅 / 通过判定）
    """
    base_ret = _daily_topk_return(base_topk, topk)
    arm_ret = _daily_topk_return(arm_topk, topk)
    base_hit = _daily_topk_hit_rate(base_topk, topk)
    arm_hit = _daily_topk_hit_rate(arm_topk, topk)

    # 同日配对（信号日相同、前瞻窗口相同）
    common_dates = base_ret.index.intersection(arm_ret.index)
    if len(common_dates) == 0:
        raise ValueError("两臂无共同信号日——无法配对")
    base_ret = base_ret.loc[common_dates]
    arm_ret = arm_ret.loc[common_dates]
    base_hit = base_hit.loc[common_dates]
    arm_hit = arm_hit.loc[common_dates]

    d_ret = (arm_ret - base_ret).to_numpy()  # 日收益差
    d_hit = (arm_hit - base_hit).to_numpy()
    days = [str(d) for d in common_dates]  # 分块自举需交易日序列

    cfg = BootstrapConfig(block_days=block_days, n_resamples=1000, seed=seed)
    ret_ci = paired_day_mean_ci(d_ret, days=days, config=cfg, units="ret")  # 区间（日收益口径）
    hit_ci = paired_day_mean_ci(d_hit, days=days, config=cfg, units="hit")

    # paired_day_mean_ci 返回 dict（生产实现键名：delta_point / ci_low / ci_high）
    delta_ret_bps = float(ret_ci["delta_point"]) * 1e4
    delta_hit_pp = float(hit_ci["delta_point"]) * 100
    ret_ci_bps = (float(ret_ci["ci_low"]) * 1e4, float(ret_ci["ci_high"]) * 1e4)
    hit_ci_pp = (float(hit_ci["ci_low"]) * 100, float(hit_ci["ci_high"]) * 100)

    # 相对降幅（基线均值的相对变化；加列默认带稀释成本 ⇒ 相对降幅 ≥9% 触发阻断）
    base_mean = float(base_ret.mean())
    relative_drop = -delta_ret_bps / 1e4 / abs(base_mean) if base_mean else 0.0

    # 通过判定：Δ 收益 95% 区间下界 > -噪声带 且 相对降幅 < 9%
    passes = bool(ret_ci_bps[0] > -NOISE_BAND_RETURN_BPS and relative_drop < RELATIVE_GATE)
    note = (
        f"Δ收益 {delta_ret_bps:+.1f}bps（95% [{ret_ci_bps[0]:+.1f},{ret_ci_bps[1]:+.1f}]）"
        f"，相对基线 {'+' if relative_drop<0 else ''}{-relative_drop:.1%}；"
        f"{'过尺（值得跑净值 A/B）' if passes else '不过尺（相对降幅≥9% 或区间破噪声带）'}"
    )
    return SignalRulerVerdict(
        delta_return_bps=delta_ret_bps,
        delta_return_ci95=ret_ci_bps,
        delta_hit_rate_pp=delta_hit_pp,
        delta_hit_rate_ci95=hit_ci_pp,
        relative_drop=relative_drop,
        passes=passes,
        note=note,
        n_signal_days=len(common_dates),
        block_days=block_days,
    )


def ruler_from_runs(base_dir: Path, arm_dir: Path, topk: int = 20) -> SignalRulerVerdict:
    """从 runs 产物（契约 schema）直接裁决（经 runs_loader 读入 topk_detail）。"""
    from src.lazybull.v2.evidence.runs_loader import load_runs_batch

    base = load_runs_batch(base_dir)
    arm = load_runs_batch(arm_dir)
    base_topk = pd.concat(
        [f.topk_detail for f in base.folds.values() if f.topk_detail is not None],
        ignore_index=True,
    )
    arm_topk = pd.concat(
        [f.topk_detail for f in arm.folds.values() if f.topk_detail is not None],
        ignore_index=True,
    )
    if base_topk.empty or arm_topk.empty:
        raise ValueError("topk_detail 缺失（契约三态=可缺+不可重建 ⇒ 该批不参与尺子）")
    return ruler_verdict(base_topk, arm_topk, topk=topk)
