# -*- coding: utf-8 -*-
"""日频信号级配对对比：把 WF 单臂对比从「14 个折净值读数」提升到「逐日配对读数」。

背景（2026-09-16 实测）：链式净值口径的噪声带极宽——同配置**仅换随机种子**即
ΔCAGR −3.62pp，列集增删更到 −5.8pp 量级，而列级改动预期效应只有 0~1pp
⇒ 在净值口径上永远判不出列级差异。而 `walk_forward_topk_details_*.csv` 保留了
**逐日 Top-K 明细**（`pred_score` + `true_return`），可构造逐日配对差值：

- 评估单位：**一个信号日一个数**（该日所选 Top-K 股票的平均持有期收益 / 命中率），
  14 折共约 1700 个信号日（折窗口互不重叠，串成一条连续时间轴）；
- 配对：两个臂在**同一信号日**上相减。注意"同日"配对指**信号日相同、前瞻窗口相同**
  （`t+1 … t+21` 是同一段日历窗口），不是"收益在同一天实现"——因此该窗口内的市场/行业
  共同冲击在相减时抵消，剩下的是**选股差异**；相邻信号日的前瞻窗口高度重叠（t 与 t+1 共享
  19 日），故区间必须用交易日分块自举，不能按天当独立样本；
- 区间：交易日分块自举（复用 `risk/terminal_loss/block_stats.py::paired_day_mean_ci`），
  块长默认取组合级多年 OOS 口径（20/40/60 日敏感性）。

**口径边界（必须随结论一起报告）**：本口径是**信号级代理**，不含交易成本、调仓节奏、
Kelly 仓位与路径效应，因此只能用于**筛选**（"这个改动是否真的动了信号"）；
最终裁决仍以链式净值判据（ΔMaxDD 为主判据）为准，且只对通过筛选的改动花 2 小时跑全量。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from scripts.compare.fold_subset import load_run, resolve_raw_dir, validate_alignment
from src.lazybull.risk.terminal_loss.block_stats import (
    BLOCK_DAYS_SENSITIVITY,
    BootstrapConfig,
    paired_day_mean_ci,
)

#: 逐日明细文件名模式（`ml/walk_forward/reporting.py::build_daily_topk_detail_df` 产物）
TOPK_DETAIL_GLOB = "walk_forward_topk_details_*_split*.csv"

#: 默认对比的 Top-K 档位（明细文件含 rank≤K 的逐股行）
DEFAULT_TOPK_LIST: Sequence[int] = (20, 30)

#: 默认块长（交易日）——组合级多年 OOS 口径，见 block_stats.BLOCK_DAYS_SENSITIVITY
DEFAULT_BLOCK_DAYS: Sequence[int] = BLOCK_DAYS_SENSITIVITY


@dataclass(frozen=True)
class DayPanel:
    """单个臂的逐日信号面板（一天一行）。"""

    label: str
    topk: int
    frame: pd.DataFrame  # 列：split_index / trade_date / 股票数 / 日均收益 / 命中率


def _read_topk_details(target: Path) -> pd.DataFrame:
    """读取一个 WF 批次下全部逐日 Top-K 明细（拼接后返回）。"""
    raw_dir = resolve_raw_dir(Path(target))
    files = sorted(raw_dir.glob(TOPK_DETAIL_GLOB))
    if not files:
        raise FileNotFoundError(f"未找到逐日 Top-K 明细 {TOPK_DETAIL_GLOB}: {raw_dir}")
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in files]
    detail = pd.concat(frames, ignore_index=True)
    required = {"split_index", "trade_date", "topk", "rank", "true_return"}
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"逐日 Top-K 明细缺少必需列: {missing}（文件数 {len(files)}）")
    return detail


def build_day_panels(
    directory: Path, topk_list: Sequence[int] = DEFAULT_TOPK_LIST, label: str = ""
) -> Dict[int, DayPanel]:
    """从 WF 批次产物目录构造逐日信号面板（每个 Top-K 档位一个面板）。"""
    detail = _read_topk_details(Path(directory))
    panels: Dict[int, DayPanel] = {}
    for topk in topk_list:
        subset = detail[(detail["topk"] == int(topk)) & (detail["rank"] <= int(topk))].copy()
        if subset.empty:
            raise ValueError(f"Top-K={topk} 无有效明细行（检查 topk/rank 列）")
        subset["ret"] = pd.to_numeric(subset["true_return"], errors="coerce")
        grouped = subset.groupby(["split_index", "trade_date"], sort=True)
        panel = grouped.agg(
            股票数=("ret", "size"),
            日均收益=("ret", "mean"),
            命中率=("ret", lambda s: float((s > 0).mean())),
            有效收益数=("ret", "count"),
        ).reset_index()
        duplicates = panel.duplicated(subset=["split_index", "trade_date"]).any()
        if duplicates:
            raise ValueError(f"Top-K={topk} 面板存在重复的 (折, 交易日) 行")
        panels[int(topk)] = DayPanel(label=label, topk=int(topk), frame=panel)
    logger.info(
        f"{label or directory}: 逐日面板构建完成 "
        f"（Top-K={sorted(panels.keys())}，合计 {sum(len(p.frame) for p in panels.values())} 行）"
    )
    return panels


def align_panels(
    panel_a: DayPanel, panel_b: DayPanel, allow_day_mismatch: bool = False
) -> pd.DataFrame:
    """按（折, 信号日）对齐两个臂的逐日面板，返回含配对差值的宽表。

    配对键是**信号日**：两臂在同一个信号日上各选出自己的一套 Top-K 组合，
    其平均持有期收益相减。两臂的前瞻窗口是同一段日历窗口（`t+1 … t+21`），
    窗口内的共同冲击因而在差值里抵消。
    """
    if panel_a.topk != panel_b.topk:
        raise ValueError(f"Top-K 不一致: A={panel_a.topk}, B={panel_b.topk}")
    merged = panel_a.frame.merge(
        panel_b.frame,
        on=["split_index", "trade_date"],
        how="outer",
        suffixes=("_A", "_B"),
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"]
    if len(both) != len(panel_a.frame) or len(both) != len(panel_b.frame):
        only_a = int((merged["_merge"] == "left_only").sum())
        only_b = int((merged["_merge"] == "right_only").sum())
        message = (
            f"两臂逐日面板未完全对齐（仅 A 有 {only_a} 行，仅 B 有 {only_b} 行）——"
            "折集合/窗口/股票池不同则配对设计失效"
        )
        if allow_day_mismatch:
            logger.warning(message)
        else:
            raise ValueError(message)
    aligned = both.copy()
    aligned["Δ日均收益"] = aligned["日均收益_A"] - aligned["日均收益_B"]
    aligned["Δ命中率"] = aligned["命中率_A"] - aligned["命中率_B"]
    return aligned.sort_values(["split_index", "trade_date"], kind="stable").reset_index(drop=True)


def paired_metric_table(
    aligned: pd.DataFrame,
    block_days_list: Sequence[int] = DEFAULT_BLOCK_DAYS,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> pd.DataFrame:
    """对每个配对指标 × 每个块长做分块自举，输出区间表（一指标 × 一块长一行）。

    口径说明：`true_return` 即评估用标签（默认 `neu_y_ret_20`，20 个交易日持有期的
    中性化前瞻收益），因此面板里的「日均收益」是**每个观测日一个横截面的平均持有期
    收益**（窗口重叠），不是单日收益；区间用交易日分块自举覆盖采样不确定性。
    """
    metric_specs = [
        ("Δ平均持有期收益(bps)", "Δ日均收益", 1e4, "bps", "日均收益_B", 1e4),
        ("Δ命中率(pp)", "Δ命中率", 100.0, "pp", "命中率_B", 100.0),
    ]
    rows: List[Dict[str, object]] = []
    for name, column, scale, units, level_column, level_scale in metric_specs:
        deltas = aligned[column].to_numpy(dtype=float) * scale
        days = aligned["trade_date"].astype(str).to_numpy()
        level = float(aligned[level_column].to_numpy(dtype=float).mean() * level_scale)
        fold_means = (
            pd.DataFrame({"split_index": aligned["split_index"].to_numpy(), "delta": deltas})
            .groupby("split_index")["delta"]
            .mean()
        )
        for block_days in block_days_list:
            config = BootstrapConfig(
                block_days=int(block_days),
                n_resamples=int(n_resamples),
                seed=int(seed),
                ci=float(ci),
            )
            stats = paired_day_mean_ci(deltas, days, config=config, units=units)
            point = float(stats["delta_point"])
            rows.append(
                {
                    "指标": name,
                    "块长(交易日)": int(block_days),
                    "基线水平": round(level, 4),
                    "点估计": round(point, 4),
                    "相对基线水平(%)": round(100.0 * point / level, 4) if level else np.nan,
                    "区间下限": round(float(stats["ci_low"]), 4),
                    "区间上限": round(float(stats["ci_high"]), 4),
                    "区间水平": stats["ci_level"],
                    "正差值概率": round(float(stats["prob_positive"]), 4),
                    "逐日同向占比": round(float(stats["day_share_positive"]), 4),
                    "折内同向折数": int((fold_means > 0).sum()),
                    "折内Δ中位数": (
                        round(float(fold_means.median()), 4) if len(fold_means) else np.nan
                    ),
                    "折数": int(aligned["split_index"].nunique()),
                    "交易日数": int(stats["n_days"]),
                    "缺失日数": int(stats["n_days_dropped"]),
                    "块数": int(np.ceil(stats["n_days"] / int(block_days))),
                    "重采样次数": int(stats["n_resamples"]),
                    "种子": int(stats["seed"]),
                    "单位": units,
                }
            )
    return pd.DataFrame(rows)


def fold_metric_table(aligned: pd.DataFrame) -> pd.DataFrame:
    """逐折信号指标（点估计，不含区间）——用于看"差异集中在哪几折"。"""
    grouped = aligned.groupby("split_index", sort=True)
    table = grouped.agg(
        交易日数=("trade_date", "size"),
        基线平均持有期收益bps=("日均收益_B", lambda s: float(s.mean() * 1e4)),
        臂平均持有期收益bps=("日均收益_A", lambda s: float(s.mean() * 1e4)),
        Δ平均持有期收益bps=("Δ日均收益", lambda s: float(s.mean() * 1e4)),
        基线命中率pp=("命中率_B", lambda s: float(s.mean() * 100.0)),
        臂命中率pp=("命中率_A", lambda s: float(s.mean() * 100.0)),
        Δ命中率pp=("Δ命中率", lambda s: float(s.mean() * 100.0)),
    ).reset_index()
    numeric_columns = [c for c in table.columns if c not in ("split_index", "交易日数")]
    for column in numeric_columns:
        table[column] = table[column].round(4)
    return table


def compare_signal_metrics(
    baseline: Path,
    arms: Sequence[Path],
    arm_labels: Sequence[str] = (),
    topk_list: Sequence[int] = DEFAULT_TOPK_LIST,
    block_days_list: Sequence[int] = DEFAULT_BLOCK_DAYS,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
    allow_state_mismatch: bool = False,
    allow_day_mismatch: bool = False,
) -> Dict[str, object]:
    """主入口：基线 vs 多个实验臂的逐日信号级对比。"""
    baseline_run = load_run(Path(baseline))
    arm_runs = [
        load_run(Path(arm), label=(arm_labels[index] if arm_labels else None))
        for index, arm in enumerate(arms)
    ]
    if arm_labels and len(arm_labels) != len(arm_runs):
        raise ValueError(f"--arm-label 数量({len(arm_labels)})须与 --arm 数量({len(arm_runs)})一致")
    validate_alignment([baseline_run, *arm_runs], allow_state_mismatch=allow_state_mismatch)

    baseline_panels = build_day_panels(
        baseline_run.directory, topk_list=topk_list, label=baseline_run.label
    )
    metric_tables: List[pd.DataFrame] = []
    fold_tables: List[pd.DataFrame] = []
    for run in arm_runs:
        arm_panels = build_day_panels(run.directory, topk_list=topk_list, label=run.label)
        for topk in topk_list:
            aligned = align_panels(
                arm_panels[int(topk)],
                baseline_panels[int(topk)],
                allow_day_mismatch=allow_day_mismatch,
            )
            table = paired_metric_table(
                aligned,
                block_days_list=block_days_list,
                n_resamples=n_resamples,
                seed=seed,
                ci=ci,
            )
            table.insert(0, "基线", baseline_run.label)
            table.insert(1, "臂", run.label)
            table.insert(2, "TopK", int(topk))
            metric_tables.append(table)

            folds = fold_metric_table(aligned)
            folds.insert(0, "基线", baseline_run.label)
            folds.insert(1, "臂", run.label)
            folds.insert(2, "TopK", int(topk))
            fold_tables.append(folds)
    return {
        "基线": baseline_run.label,
        "基线目录": str(baseline_run.directory),
        "臂": [run.label for run in arm_runs],
        "指标表": pd.concat(metric_tables, ignore_index=True),
        "逐折表": pd.concat(fold_tables, ignore_index=True),
    }
