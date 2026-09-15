# -*- coding: utf-8 -*-
"""折子集链式对比：只读既有 walk-forward 产物，按 split 子集重算链式指标。

用途：消融实验若只跑部分折（``walk_forward.py --selected-split-indices``）以省时，
需要一个与基线**在同一折子集上**可比的对照口径。本模块只读
``chain_nav_*.csv`` 与 ``walk_forward_summary_*.csv``（以及数据态快照），
不重训、不写回任何产物。

口径：
    - 折子集指标 = 过滤出目标 ``split_index`` 的行、把净值归一化到起点 1.0 后，
      复用 :func:`calculate_chain_metrics` 重新计算（跨折边界的收益不计入交易日数）；
    - 逐折指标在**未归一化**的原始链上按折内起止净值计算，保证与全周期口径一致；
    - 对齐校验：折集合、逐折窗口（``test_start``/``test_end``）与数据态 ID
      三者不一致时直接报错，禁止把不同折定义或不同数据态的产物并表比较。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.ml.walk_forward.chain_metrics import calculate_chain_metrics

CHAIN_GLOB = "chain_nav_*.csv"
SUMMARY_GLOB = "walk_forward_summary_*.csv"
DATA_STATE_GLOB = "data_state_*.json"


@dataclass
class RunArtifacts:
    """一次 walk-forward 运行的可比对产物。"""

    label: str
    directory: Path
    chain: pd.DataFrame  # 列: date / nav / split_index
    windows: pd.DataFrame  # 列: split_index / test_start / test_end
    data_state_id: Optional[str] = None
    chain_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)


def parse_split_spec(raw: str) -> List[int]:
    """解析折规格：支持 ``8-13``、``8,9,10``、``v8-13`` 等写法。"""
    splits = set()
    for part in str(raw).split(","):
        part = part.strip().lower().lstrip("v")
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start > end:
                raise ValueError(f"折区间起点大于终点: {part}")
            splits.update(range(start, end + 1))
        else:
            splits.add(int(part))
    if not splits:
        raise ValueError("未提供有效折号")
    return sorted(splits)


def resolve_raw_dir(target: Path) -> Path:
    """接受 batch 目录、raw 目录或 chain_nav 文件路径，返回含产物的目录。"""
    target = Path(target)
    if target.is_file():
        return target.parent
    raw_dir = target / "raw"
    return raw_dir if raw_dir.is_dir() else target


def _pick_latest(directory: Path, pattern: str, warnings: List[str]) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{directory} 下未找到 {pattern}")
    if len(files) > 1:
        message = f"{directory} 下存在 {len(files)} 个 {pattern}，取最新: {files[-1].name}"
        warnings.append(message)
        logger.warning(message)
    return files[-1]


def _read_data_state_id(directory: Path, warnings: List[str]) -> Optional[str]:
    files = sorted(directory.glob(DATA_STATE_GLOB))
    if not files:
        warnings.append(f"{directory} 下无数据态快照，无法校验数据态一致性")
        return None
    try:
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"数据态快照解析失败: {files[-1].name} — {exc}")
        return None
    value = payload.get("data_state_id")
    return str(value) if value is not None else None


def load_run(target: Path, label: Optional[str] = None) -> RunArtifacts:
    """加载一次 walk-forward 运行的 chain_nav 与逐折窗口。"""
    directory = resolve_raw_dir(target)
    warnings: List[str] = []
    chain_path = _pick_latest(directory, CHAIN_GLOB, warnings)
    chain = pd.read_csv(chain_path, encoding="utf-8-sig")
    missing = {"nav", "split_index"} - set(chain.columns)
    if missing:
        raise ValueError(f"{chain_path} 缺少必需列: {sorted(missing)}")
    chain["split_index"] = pd.to_numeric(chain["split_index"], errors="raise").astype(int)
    chain["nav"] = pd.to_numeric(chain["nav"], errors="coerce")

    summary_path = _pick_latest(directory, SUMMARY_GLOB, warnings)
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    needed = {"split_index", "test_start", "test_end"}
    missing_summary = needed - set(summary.columns)
    if missing_summary:
        raise ValueError(f"{summary_path} 缺少必需列: {sorted(missing_summary)}")
    windows = summary[["split_index", "test_start", "test_end"]].copy()
    windows["split_index"] = pd.to_numeric(windows["split_index"], errors="raise").astype(int)
    windows = windows.drop_duplicates(subset=["split_index"]).sort_values("split_index")

    return RunArtifacts(
        label=label or Path(target).name,
        directory=directory,
        chain=chain,
        windows=windows.reset_index(drop=True),
        data_state_id=_read_data_state_id(directory, warnings),
        chain_path=chain_path,
        warnings=warnings,
    )


def validate_alignment(runs: Sequence[RunArtifacts]) -> None:
    """校验折集合、逐折窗口与数据态 ID 一致（不一致直接报错）。"""
    if not runs:
        raise ValueError("未提供任何运行")
    basis = runs[0]
    basis_splits = set(basis.windows["split_index"].tolist())
    for run in runs[1:]:
        splits = set(run.windows["split_index"].tolist())
        if splits != basis_splits:
            raise ValueError(
                f"折集合不一致（禁止并表比较）: {basis.label} 独有 {sorted(basis_splits - splits)}，"
                f"{run.label} 独有 {sorted(splits - basis_splits)}"
            )
    merged = basis.windows.rename(columns={"test_start": "b_start", "test_end": "b_end"})
    for run in runs[1:]:
        compare = merged.merge(
            run.windows.rename(columns={"test_start": "a_start", "test_end": "a_end"}),
            on="split_index",
            how="inner",
        )
        mismatch = compare[
            (compare["b_start"].astype(str) != compare["a_start"].astype(str))
            | (compare["b_end"].astype(str) != compare["a_end"].astype(str))
        ]
        if not mismatch.empty:
            detail = ", ".join(
                f"折{int(row.split_index)}({row.b_start}~{row.b_end} vs {row.a_start}~{row.a_end})"
                for row in mismatch.itertuples()
            )
            raise ValueError(
                f"逐折窗口不一致（禁止并表比较）: {basis.label} vs {run.label} — {detail}"
            )

    states = {run.label: run.data_state_id for run in runs}
    distinct = {value for value in states.values() if value is not None}
    if len(distinct) > 1:
        raise ValueError(f"数据态不一致（需冻结数据态复跑）: {states}")
    if any(value is None for value in states.values()):
        logger.warning(f"部分运行缺少数据态 ID，无法完整校验: {states}")


def split_set(run: RunArtifacts) -> List[int]:
    """该运行可用的折号（升序）。"""
    return sorted(run.chain["split_index"].unique().tolist())


def fold_table(run: RunArtifacts, splits: Sequence[int]) -> pd.DataFrame:
    """逐折起止净值、折收益与折内最大回撤（在原始链上计算，不归一化）。"""
    records: List[Dict[str, object]] = []
    for split in splits:
        segment = run.chain[run.chain["split_index"] == split].reset_index(drop=True)
        nav = segment["nav"].dropna()
        if nav.empty or nav.iloc[0] == 0:
            records.append(
                {
                    "运行": run.label,
                    "折序号": int(split),
                    "折起始净值": np.nan,
                    "折结束净值": np.nan,
                    "折收益": np.nan,
                    "折最大回撤": np.nan,
                    "折交易日数": 0,
                }
            )
            continue
        cumulative_max = nav.cummax()
        drawdown = ((nav - cumulative_max) / cumulative_max).min()
        records.append(
            {
                "运行": run.label,
                "折序号": int(split),
                "折起始净值": float(nav.iloc[0]),
                "折结束净值": float(nav.iloc[-1]),
                "折收益": float(nav.iloc[-1] / nav.iloc[0] - 1),
                "折最大回撤": float(drawdown),
                "折交易日数": int(len(nav)),
            }
        )
    return pd.DataFrame(records)


def subset_metrics(run: RunArtifacts, splits: Sequence[int]) -> Dict[str, object]:
    """折子集链式指标（净值归一化到子集起点后复用公共 chain_metrics）。"""
    segment = run.chain[run.chain["split_index"].isin(list(splits))].reset_index(drop=True)
    if segment.empty:
        raise ValueError(f"{run.label} 在折子集 {list(splits)} 上无数据")
    nav = segment["nav"].astype(float)
    base = nav.dropna()
    if base.empty or base.iloc[0] == 0:
        raise ValueError(f"{run.label} 折子集起点净值缺失")
    normalized = segment.copy()
    normalized["nav"] = nav / base.iloc[0]
    metrics = calculate_chain_metrics(normalized)
    return {
        "运行": run.label,
        "折数": len(splits),
        "子集总收益": metrics["total_return"],
        "子集CAGR": metrics["cagr"],
        "子集最大回撤": metrics["max_drawdown"],
        "子集夏普": metrics["sharpe"],
        "子集波动": metrics["volatility"],
        "子集交易日数": metrics["trading_days"],
    }


def full_metrics(run: RunArtifacts) -> Dict[str, object]:
    """全周期链式指标（与既有对比工具口径一致，未归一化）。"""
    metrics = calculate_chain_metrics(run.chain)
    return {
        "运行": run.label,
        "全周期总收益": metrics["total_return"],
        "全周期CAGR": metrics["cagr"],
        "全周期最大回撤": metrics["max_drawdown"],
        "全周期夏普": metrics["sharpe"],
        "全周期交易日数": metrics["trading_days"],
    }


def compare_runs(
    baseline: RunArtifacts,
    arms: Sequence[RunArtifacts],
    splits: Optional[Sequence[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """产出折子集对比表与逐折明细表（中文表头）。"""
    runs = [baseline] + list(arms)
    validate_alignment(runs)

    available = split_set(baseline)
    target_splits = sorted(splits) if splits else available
    unknown = [split for split in target_splits if split not in available]
    if unknown:
        raise ValueError(f"折子集 {unknown} 不在可用折 {available} 内")

    summary_rows = []
    for run in runs:
        row = {}
        row.update(full_metrics(run))
        row.update(subset_metrics(run, target_splits))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    folds = pd.concat([fold_table(run, target_splits) for run in runs], ignore_index=True)
    wide = folds.pivot(index="折序号", columns="运行", values="折收益")
    wide_dd = folds.pivot(index="折序号", columns="运行", values="折最大回撤")

    baseline_label = baseline.label
    summary["ΔCAGR(vs基线)"] = (
        summary["子集CAGR"] - summary.loc[summary["运行"] == baseline_label, "子集CAGR"].iloc[0]
    )
    summary["Δ最大回撤(vs基线)"] = (
        summary["子集最大回撤"]
        - summary.loc[summary["运行"] == baseline_label, "子集最大回撤"].iloc[0]
    )
    summary["Δ夏普(vs基线)"] = (
        summary["子集夏普"] - summary.loc[summary["运行"] == baseline_label, "子集夏普"].iloc[0]
    )

    fold_valid = wide.dropna(subset=[baseline_label]) if baseline_label in wide.columns else wide
    positive_counts = {}
    aligned_counts = {}
    for label in wide.columns:
        series = fold_valid[label]
        positive_counts[label] = int((series > 0).sum())
        if label == baseline_label:
            aligned_counts[label] = int(len(series))
        else:
            aligned_counts[label] = int(
                (np.sign(series) == np.sign(fold_valid[baseline_label])).sum()
            )
    summary["折收益为正(子集)"] = summary["运行"].map(positive_counts).fillna(0).astype(int)
    summary["折收益同向(vs基线)"] = summary["运行"].map(aligned_counts).fillna(0).astype(int)

    ordered = [
        "运行",
        "折数",
        "子集总收益",
        "子集CAGR",
        "子集最大回撤",
        "子集夏普",
        "子集波动",
        "子集交易日数",
        "全周期CAGR",
        "全周期最大回撤",
        "全周期夏普",
        "ΔCAGR(vs基线)",
        "Δ最大回撤(vs基线)",
        "Δ夏普(vs基线)",
        "折收益为正(子集)",
        "折收益同向(vs基线)",
    ]
    summary = summary.reindex(columns=ordered)

    detailed = folds.copy()
    baseline_fold = folds[folds["运行"] == baseline_label][["折序号", "折收益"]].rename(
        columns={"折收益": "基线折收益"}
    )
    detailed = detailed.merge(baseline_fold, on="折序号", how="left")
    detailed["相对基线折收益差"] = detailed["折收益"] - detailed["基线折收益"]
    detailed = detailed.sort_values(["折序号", "运行"]).reset_index(drop=True)
    return {
        "summary": summary,
        "folds": detailed,
        "fold_returns": wide,
        "fold_drawdowns": wide_dd,
    }
