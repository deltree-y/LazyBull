"""OOS 全域预测、确定排序与 AD_SIMPLE 固定对照排名（协议 §9）。

- 分数 = 模型原始预测（预测 20 日行业中性收益）；不设正负门、不做仓位择时；
- 排序固定为 `score DESC, ts_code ASC`；
- 非有限预测属于模型/输入错误，必须显式报错，不得静默剔除；
- AD_SIMPLE 只用两个扩散列，NaN 排最后，不替换主候选。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import spec


class PredictionError(RuntimeError):
    """非有限预测或域不完整（fail-closed）。"""


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def predict_matrix(
    model: Any, frame: pd.DataFrame, features: Sequence[str] = spec.FEATURES
) -> np.ndarray:
    """按冻结列序预测；非有限即抛错。"""
    matrix = frame[list(features)].to_numpy(dtype=np.float32)
    values = np.asarray(model.predict(matrix), dtype=float)
    nonfinite = int((~np.isfinite(values)).sum())
    if nonfinite:
        raise PredictionError(f"非有限预测 {nonfinite} 行（模型/输入错误，不得静默剔除）")
    return values


def rank_frame(
    frame: pd.DataFrame, score: np.ndarray, *, tie_break: str = "ts_code"
) -> pd.DataFrame:
    """score 降序、ts_code 升序的确定排名。"""
    work = pd.DataFrame(
        {
            "trade_date": frame["trade_date"].astype(str).to_numpy(),
            "ts_code": frame["ts_code"].astype(str).to_numpy(),
            "score": np.asarray(score, dtype=float),
        }
    )
    work = work.sort_values(
        ["trade_date", "score", tie_break], ascending=[True, False, True], kind="mergesort"
    )
    work["rank"] = work.groupby("trade_date", sort=False).cumcount() + 1
    return work.reset_index(drop=True)


def ad_simple_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """AD_SIMPLE 固定对照：净扩散降序 → 中位修正降序 → 代码升序（NaN 最后）。"""
    work = frame[["trade_date", "ts_code"]].copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work["ts_code"] = work["ts_code"].astype(str)
    work["net"] = pd.to_numeric(frame["ad_eps_net_diffusion_90"], errors="coerce").to_numpy()
    work["median"] = pd.to_numeric(frame["ad_eps_revision_median_90"], errors="coerce").to_numpy()
    work["net_missing"] = work["net"].isna()
    work["median_missing"] = work["median"].isna()
    work = work.sort_values(
        ["trade_date", "net_missing", "net", "median_missing", "median", "ts_code"],
        ascending=[True, True, False, True, False, True],
        kind="mergesort",
    )
    work["rank"] = work.groupby("trade_date", sort=False).cumcount() + 1
    return work[["trade_date", "ts_code", "net", "median", "rank"]].reset_index(drop=True)


def predict_fold(
    frame: pd.DataFrame,
    model: Any,
    fold: dict[str, object],
    version: int,
    spec_sha: str,
    simple: pd.DataFrame | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """对单折 OOS 全域打分并输出排名（含 AD_SIMPLE 名次映射）。"""
    emit = _emit(log)
    subset = frame[frame["trade_date"].astype(str).isin(set(fold["oos_days"]))].copy()
    if subset.empty:
        raise PredictionError(f"折 {fold['fold_id']} 的 OOS 为空")
    subset = subset.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    scores = predict_matrix(model, subset)
    ranked = rank_frame(subset, scores)
    ranked["fold_id"] = str(fold["fold_id"])
    ranked["model_version"] = int(version)
    ranked["spec_sha"] = spec_sha
    merge_keys = ["trade_date", "ts_code"]
    if simple is not None:
        ranked = ranked.merge(
            simple.rename(columns={"rank": "simple_rank"})[merge_keys + ["simple_rank"]],
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )
    emit(f"折 {fold['fold_id']}: {len(ranked)} 行 / {ranked['trade_date'].nunique()} 日")
    return {"predictions": ranked, "rows": int(len(ranked))}


def write_fold_predictions(directory: Path, fold_id: str, frame: pd.DataFrame) -> str:
    """逐折落盘（CSV 之外的 parquet，便于合并核对）。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"predictions_{fold_id}.parquet"
    frame.to_parquet(path, index=False)
    return str(path)


def daily_counts(ranked: pd.DataFrame, min_scored: int) -> pd.DataFrame:
    """逐日评分数量与 100 只门状态。"""
    grouped = ranked.groupby("trade_date", sort=True)
    frame = grouped.size().rename("scored_count").reset_index()
    frame["min100_status"] = np.where(
        frame["scored_count"] >= int(min_scored), "ok", "universe_below_minimum"
    )
    return frame


def ranking_lookup(ranked: pd.DataFrame, days: Sequence[str]) -> dict[str, list[tuple[str, float]]]:
    """排名表 -> 引擎输入 `{日期: [(code, score)]}`（按 rank 升序）。

    当日评分数量不足 100 只时**不产生新买排名**（协议 §4.2）。
    """
    counts = ranked.groupby("trade_date").size()
    min_scored = int(spec.UNIVERSE["min_scored_per_day"])
    work = ranked[ranked["trade_date"].astype(str).isin(set(str(d) for d in days))].copy()
    out: dict[str, list[tuple[str, float]]] = {}
    for day, part in work.groupby("trade_date", sort=True):
        if int(counts.get(day, 0)) < min_scored:
            out[str(day)] = []
            continue
        ordered = part.sort_values("rank", kind="mergesort")
        out[str(day)] = [
            (str(code), float(score)) for code, score in zip(ordered["ts_code"], ordered["score"])
        ]
    return out


def rank_table_lookup(
    table: pd.DataFrame, days: Sequence[str]
) -> dict[str, list[tuple[str, float]]]:
    """只含 `rank` 的排名表 -> 引擎输入（score 用 -rank 保持降序语义）。

    评分数量不足 100 只的日期返回空列表（当日不产生新买排名，协议 §4.2）。
    """
    min_scored = int(spec.UNIVERSE["min_scored_per_day"])
    work = table.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    counts = work.groupby("trade_date").size()
    wanted = {str(day) for day in days}
    work = work[work["trade_date"].isin(wanted)]
    # 未出现在表中的评分日也要显式返回空列表（当日不产生新买排名）
    out: dict[str, list[tuple[str, float]]] = {str(day): [] for day in sorted(wanted)}
    for day, part in work.groupby("trade_date", sort=True):
        if int(counts.get(day, 0)) < min_scored:
            out[str(day)] = []
            continue
        ordered = part.sort_values("rank", kind="mergesort")
        out[str(day)] = [
            (str(code), -float(rank)) for code, rank in zip(ordered["ts_code"], ordered["rank"])
        ]
    return out


def evaluation_metrics(frame: pd.DataFrame, ranked: pd.DataFrame) -> dict[str, Any]:
    """内部验证/成熟 OOS 的 MAE 与按日期平均 RankIC（只报告，不用于调参）。"""
    merged = ranked.merge(
        frame[["trade_date", "ts_code", spec.LABEL["column"], "label_status"]],
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    mature = merged[(merged["label_status"] == "valid") & merged[spec.LABEL["column"]].notna()]
    if mature.empty:
        return {"rows": 0, "mae": None, "rank_ic": None, "rank_ic_days": 0}
    mae = float(np.mean(np.abs(mature["score"] - mature[spec.LABEL["column"]])))
    ics = []
    for _, part in mature.groupby("trade_date"):
        if len(part) < 10:
            continue
        ic = part["score"].corr(part[spec.LABEL["column"]], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    return {
        "rows": int(len(mature)),
        "mae": mae,
        "rank_ic": float(np.mean(ics)) if ics else None,
        "rank_ic_days": len(ics),
    }
