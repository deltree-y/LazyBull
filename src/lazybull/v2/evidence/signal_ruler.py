# -*- coding: utf-8 -*-
"""信号层尺子固化（P5a-1 交付物④；方案 §3.5/§8-P5a-1）。

把「列级 / 信号级改动先过 ±10bps 尺子（相对门槛 ≥9%）才允许进净值裁决」
的现有实践固化为证据机器的强制流程入口；输入为 runs schema 的
`topk_detail.parquet`（经 runs_loader 读入）。

实现口径（如实登记）：复用 `risk/terminal_loss/block_stats.paired_day_mean_ci`
分块自举核心；面板构建 / 配对 / 判据按生产尺子 `scripts/compare/signal_metrics.py`
（v0.122.0 落位）的语义在本模块重组装（输入从 WF 目录改为 runs schema）——
**配对硬校验与生产对齐**：(split_index, trade_date) 双键 outer 配对，
未完全对齐必须报错（仅 `allow_day_mismatch` 显式降级为告警）；
`ruler_from_runs` 另校验两臂折集合与数据态 ID 一致（沿 `validate_alignment` 语义）。

判据口径（与历史四轮裁决一致：holdertrade A1/A2、repurchase、top10fh、top_inst）：
- 阻断看**点估计**：Δ平均持有期收益 ≤ −10bps 或 相对基线降幅 ≥9% ⇒ 不过尺；
  95% 区间按块长敏感性（20/40/60 交易日）三档报告，作可读性参考不进判定；
- Δ命中率噪声带 ±0.6pp 只作带内/超带标注（历史四轮从未以命中率阻断），不进判定。

口径边界（必须随结论报告，沿 v0.122.0 契约）：信号级代理，不含成本 / 调仓 /
Kelly / 路径，**只作筛选**；最终裁决以链式净值判据（ΔMaxDD 为主）为准。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.risk.terminal_loss.block_stats import (
    BLOCK_DAYS_SENSITIVITY,
    PRIMARY_BLOCK_DAYS,
    BootstrapConfig,
    paired_day_mean_ci,
)

#: 尺子噪声带（v0.122.0 首轮标定）：Δ平均持有期收益 ±10 bps（≈±9% 相对）、Δ命中率 ±0.6pp
NOISE_BAND_RETURN_BPS = 10.0
NOISE_BAND_HIT_RATE_PP = 0.6
#: 筛选相对门槛：相对基线降幅 ≥9% ⇒ 阻断（不过尺，不值得跑净值 A/B）
RELATIVE_GATE = 0.09

#: topk_detail 必需列（缺列报错，不裸 KeyError）
_TOPK_REQUIRED_COLS = {"split_index", "trade_date", "topk", "rank", "true_return"}


@dataclass
class SignalRulerVerdict:
    """信号层尺子裁决（筛选口径）。"""

    delta_return_bps: float  # Δ 平均持有期收益点估计（bps）
    delta_return_ci95: tuple  # 主块长 95% 区间（bps）
    delta_hit_rate_pp: float  # Δ 命中率点估计（pp）
    delta_hit_rate_ci95: tuple  # 主块长 95% 区间（pp）
    relative_drop: float  # 相对基线降幅（正 = 变差）
    passes: bool  # 是否通过筛选（值得跑净值 A/B）
    note: str
    n_signal_days: int
    block_days: int  # 主块长
    delta_return_ci95_by_block: Dict[int, tuple] = field(default_factory=dict)  # 块长敏感性
    delta_hit_rate_ci95_by_block: Dict[int, tuple] = field(default_factory=dict)


def _check_topk_columns(df: pd.DataFrame, name: str) -> None:
    missing = sorted(_TOPK_REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"{name} 缺必需列 {missing}（runs 契约 §7.1）")


def _daily_topk_return(topk_df: pd.DataFrame, topk: int) -> pd.Series:
    """逐（折, 信号日）Top-K 平均持有期收益（评估单位：一个信号日一个数）。"""
    sub = topk_df[(topk_df["topk"] == topk) & (topk_df["rank"] <= topk)]
    return sub.groupby(["split_index", "trade_date"])["true_return"].mean()


def _daily_topk_hit_rate(topk_df: pd.DataFrame, topk: int) -> pd.Series:
    """逐（折, 信号日）Top-K 命中率（true_return > 0 占比）。"""
    sub = topk_df[(topk_df["topk"] == topk) & (topk_df["rank"] <= topk)]
    return sub.groupby(["split_index", "trade_date"])["true_return"].apply(
        lambda s: (s > 0).mean()
    )


def _pair_panels(
    base_ret: pd.Series, arm_ret: pd.Series, allow_day_mismatch: bool
) -> tuple:
    """(split_index, trade_date) 双键 outer 配对；未完全对齐报错（生产 align_panels 语义）。"""
    merged = pd.concat(
        [base_ret.rename("base"), arm_ret.rename("arm")], axis=1, join="outer"
    )
    both = merged.dropna()
    if len(both) != len(base_ret) or len(both) != len(arm_ret):
        only_base = len(base_ret) - len(both)
        only_arm = len(arm_ret) - len(both)
        message = (
            f"两臂逐日面板未完全对齐（仅基线有 {only_base} 行，仅臂有 {only_arm} 行）——"
            "折集合/窗口/股票池不同则配对设计失效"
        )
        if allow_day_mismatch:
            logger.warning(message)
        else:
            raise ValueError(message)
    return both["base"], both["arm"]


def ruler_verdict(
    base_topk: pd.DataFrame,
    arm_topk: pd.DataFrame,
    topk: int = 20,
    block_days_list: Sequence[int] = BLOCK_DAYS_SENSITIVITY,
    seed: int = 42,
    allow_day_mismatch: bool = False,
) -> SignalRulerVerdict:
    """信号层尺子裁决：两臂逐（折, 信号日）Top-K 配对差分 + 交易日分块自举区间。

    Args:
        base_topk / arm_topk: 两臂的 topk_detail（runs schema：
            split_index/trade_date/topk/rank/ts_code/true_return）
        topk: Top-K 口径（默认 20；多档由调用方逐档调用）
        block_days_list: 交易日分块自举块长敏感性（默认 20/40/60，沿生产口径；
            相邻信号日前瞻窗口重叠，必须分块）
        seed: 随机种子
        allow_day_mismatch: 面板未完全对齐时仅告警（默认 False 报错，配对设计失效
            不得静默继续）

    Returns:
        SignalRulerVerdict（Δ 收益 bps / Δ 命中率 pp / 三档块长 95% 区间 /
        相对降幅 / 通过判定）
    """
    _check_topk_columns(base_topk, "base_topk")
    _check_topk_columns(arm_topk, "arm_topk")
    base_ret = _daily_topk_return(base_topk, topk)
    arm_ret = _daily_topk_return(arm_topk, topk)
    base_hit = _daily_topk_hit_rate(base_topk, topk)
    arm_hit = _daily_topk_hit_rate(arm_topk, topk)
    if base_ret.empty or arm_ret.empty:
        raise ValueError(f"Top-K={topk} 无有效明细行（检查 topk/rank 列）")

    # 同（折, 信号日）配对（信号日相同、前瞻窗口相同）；未完全对齐必须报错
    base_ret, arm_ret = _pair_panels(base_ret, arm_ret, allow_day_mismatch)
    base_hit, arm_hit = _pair_panels(base_hit, arm_hit, allow_day_mismatch)

    d_ret = (arm_ret - base_ret).to_numpy()  # 日收益差
    d_hit = (arm_hit - base_hit).to_numpy()
    days = [str(d) for d in arm_ret.index.get_level_values("trade_date")]  # 分块自举需交易日序列

    # 块长敏感性：逐块长算区间（判据用点估计，区间作可读性参考）
    ret_ci_by_block: Dict[int, tuple] = {}
    hit_ci_by_block: Dict[int, tuple] = {}
    delta_ret_bps = 0.0
    delta_hit_pp = 0.0
    for block_days in block_days_list:
        cfg = BootstrapConfig(block_days=int(block_days), n_resamples=1000, seed=seed)
        ret_ci = paired_day_mean_ci(d_ret, days=days, config=cfg, units="ret")
        hit_ci = paired_day_mean_ci(d_hit, days=days, config=cfg, units="hit")
        delta_ret_bps = float(ret_ci["delta_point"]) * 1e4
        delta_hit_pp = float(hit_ci["delta_point"]) * 100
        ret_ci_by_block[int(block_days)] = (
            float(ret_ci["ci_low"]) * 1e4, float(ret_ci["ci_high"]) * 1e4)
        hit_ci_by_block[int(block_days)] = (
            float(hit_ci["ci_low"]) * 100, float(hit_ci["ci_high"]) * 100)
    primary = int(PRIMARY_BLOCK_DAYS) if PRIMARY_BLOCK_DAYS in ret_ci_by_block else int(
        list(block_days_list)[0])

    # 相对降幅（基线均值的相对变化）
    base_mean = float(base_ret.mean())
    relative_drop = -delta_ret_bps / 1e4 / abs(base_mean) if base_mean else 0.0

    # 通过判定（历史四轮点估计口径）：Δ 收益点估计 > -噪声带 且 相对降幅 < 9%
    passes = bool(delta_ret_bps > -NOISE_BAND_RETURN_BPS and relative_drop < RELATIVE_GATE)
    hit_band = "带内" if abs(delta_hit_pp) <= NOISE_BAND_HIT_RATE_PP else "超带"
    ci_txt = " / ".join(
        f"{bd}d [{ret_ci_by_block[bd][0]:+.1f},{ret_ci_by_block[bd][1]:+.1f}]"
        for bd in sorted(ret_ci_by_block)
    )
    note = (
        f"Δ收益 {delta_ret_bps:+.1f}bps（95% {ci_txt}），"
        f"Δ命中率 {delta_hit_pp:+.2f}pp（噪声带 ±{NOISE_BAND_HIT_RATE_PP}pp，{hit_band}），"
        f"相对基线 {'+' if relative_drop < 0 else ''}{-relative_drop:.1%}；"
        f"{'过尺（值得跑净值 A/B）' if passes else '不过尺（点估计破噪声带或相对降幅≥9%）'}"
    )
    return SignalRulerVerdict(
        delta_return_bps=delta_ret_bps,
        delta_return_ci95=ret_ci_by_block[primary],
        delta_hit_rate_pp=delta_hit_pp,
        delta_hit_rate_ci95=hit_ci_by_block[primary],
        relative_drop=relative_drop,
        passes=passes,
        note=note,
        n_signal_days=len(base_ret),
        block_days=primary,
        delta_return_ci95_by_block=ret_ci_by_block,
        delta_hit_rate_ci95_by_block=hit_ci_by_block,
    )


def ruler_from_runs(
    base_dir: Path,
    arm_dir: Path,
    topk: int = 20,
    block_days_list: Sequence[int] = BLOCK_DAYS_SENSITIVITY,
    allow_state_mismatch: bool = False,
    allow_day_mismatch: bool = False,
) -> SignalRulerVerdict:
    """从 runs 产物（契约 schema）直接裁决（经 runs_loader 读入 topk_detail）。

    两臂校验（沿 `fold_subset.validate_alignment` 语义）：
    - 参与尺度的折集合（有 topk_detail 的折）必须一致；
    - 数据态 ID（data_state_id）必须一致（`allow_state_mismatch` 显式降级为告警，
      用于「仅 git 标记不同、数据水位一致」的例外）。
    """
    from src.lazybull.v2.evidence.runs_loader import load_runs_batch

    base = load_runs_batch(base_dir)
    arm = load_runs_batch(arm_dir)

    base_ds = (base.batch_meta.get("data_state") or {}).get("data_state_id")
    arm_ds = (arm.batch_meta.get("data_state") or {}).get("data_state_id")
    if base_ds != arm_ds:
        message = f"两臂数据态 ID 不一致: {base_ds} vs {arm_ds}——跨数据态结论需登记"
        if allow_state_mismatch:
            logger.warning(message)
        else:
            raise ValueError(message)

    base_folds = [f for f in base.folds.values() if f.topk_detail is not None]
    arm_folds = [f for f in arm.folds.values() if f.topk_detail is not None]
    if {f.split_index for f in base_folds} != {f.split_index for f in arm_folds}:
        raise ValueError(
            f"两臂参与尺度的折集合不一致: 基线 {sorted(f.split_index for f in base_folds)} "
            f"vs 臂 {sorted(f.split_index for f in arm_folds)}——配对设计失效"
        )
    if not base_folds or not arm_folds:
        raise ValueError("topk_detail 缺失（契约三态=可缺+不可重建 ⇒ 该批不参与尺子）")

    base_topk = pd.concat([f.topk_detail for f in base_folds], ignore_index=True)
    arm_topk = pd.concat([f.topk_detail for f in arm_folds], ignore_index=True)
    return ruler_verdict(
        base_topk, arm_topk, topk=topk, block_days_list=block_days_list,
        allow_day_mismatch=allow_day_mismatch,
    )
