"""OOS 全域分数、分位交叉固定修正与截面排名（协议 §7.3）。

规则：
- 逐行把三个原始分位升序排列得到 fixed 值（不因收益好坏决定是否修正）；
- `score = 0.5*q10_fixed + 0.5*q50_fixed`；
- 排序 = score 降序，再按 ts_code 升序；**不使用未来收益决胜**；
- 非有限预测一律报错，不得静默从候选池剔除。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.defensive_ml import spec


def predict_matrix(
    model: Any, frame: pd.DataFrame, features: Sequence[str] = spec.FEATURES
) -> np.ndarray:
    """模型预测并校验有限性。"""
    values = model.predict(frame[list(features)].to_numpy(dtype=np.float32))
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"出现 {bad} 个非有限预测，禁止静默剔除")
    return values


def fix_crossing(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐行升序重排三个分位；返回 (fixed, 是否发生过交叉)。"""
    ordered = np.sort(raw, axis=1)
    crossed = (raw[:, 0] > raw[:, 1]) | (raw[:, 1] > raw[:, 2])
    return ordered, crossed


def score_frame(
    frame: pd.DataFrame,
    raw: np.ndarray,
    fold_id: str,
    versions: dict[str, int],
    spec_sha: str,
) -> pd.DataFrame:
    """组装逐日全域预测表（含 fixed 分位、分数、排名、交叉标记）。"""
    fixed, crossed = fix_crossing(raw)
    out = pd.DataFrame(
        {
            "trade_date": frame["trade_date"].astype(str).to_numpy(),
            "ts_code": frame["ts_code"].astype(str).to_numpy(),
            "fold_id": fold_id,
            "q10_version": int(versions["0.1"]),
            "q50_version": int(versions["0.5"]),
            "q90_version": int(versions["0.9"]),
            "raw_q10": raw[:, 0],
            "raw_q50": raw[:, 1],
            "raw_q90": raw[:, 2],
            "q10_fixed": fixed[:, 0],
            "q50_fixed": fixed[:, 1],
            "q90_fixed": fixed[:, 2],
            "score": 0.5 * fixed[:, 0] + 0.5 * fixed[:, 1],
            "crossed": crossed,
            "spec_sha": spec_sha,
        }
    )
    out = out.sort_values(["trade_date", "score", "ts_code"], ascending=[True, False, True])
    out["rank"] = out.groupby("trade_date").cumcount() + 1
    return out.reset_index(drop=True)


def predict_fold(
    panel: pd.DataFrame,
    fold: dict[str, object],
    models: dict[str, Any],
    spec_sha: str,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """对单折 OOS 全域打分（不得只在持仓或 Top 名单内排名）。"""
    emit = log or (lambda message: None)
    frame = panel[panel["trade_date"].isin(set(fold["oos_days"]))]
    if frame.empty:
        emit(f"fold {fold['fold_id']} 无 OOS 行")
        return pd.DataFrame()
    matrix = frame[list(spec.FEATURES)].to_numpy(dtype=np.float32)
    raw = np.column_stack([models[key].predict(matrix) for key in ("0.1", "0.5", "0.9")])
    if not np.isfinite(raw).all():
        raise ValueError(f"折 {fold['fold_id']} 出现非有限预测")
    versions = {key: int(fold["versions"][key]) for key in ("0.1", "0.5", "0.9")}
    out = score_frame(frame, raw, str(fold["fold_id"]), versions, spec_sha)
    emit(f"fold {fold['fold_id']}: {len(out)} rows, {out['trade_date'].nunique()} days")
    return out


def write_fold_predictions(directory: Path, fold_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    """落盘单折预测。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"predict_{fold_id}.parquet"
    frame.to_parquet(path, index=False)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "days": int(frame["trade_date"].nunique()),
        "crossed_rate": float(frame["crossed"].mean()) if len(frame) else 0.0,
    }


def pinball_loss(y: np.ndarray, pred: np.ndarray, quantile: float) -> float:
    """分位损失（诊断用，不参与选模）。"""
    diff = y - pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def quantile_diagnostics(
    actual: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, Any]:
    """验证/成熟 OOS 诊断：pinball、覆盖、交叉率。"""
    merged = predictions.merge(
        actual[["trade_date", "ts_code", "y_def_ret60"]],
        on=["trade_date", "ts_code"],
        how="inner",
    )
    y = merged["y_def_ret60"].to_numpy(dtype=float)
    result: dict[str, Any] = {"rows": int(len(merged))}
    if not len(merged):
        return result
    for key, quantile in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)):
        pred = merged[f"{key}_fixed"].to_numpy(dtype=float)
        result[f"pinball_{key}"] = pinball_loss(y, pred, quantile)
        result[f"below_{key}"] = float(np.mean(y <= pred))
    result["crossed_rate"] = float(merged["crossed"].mean())
    result["coverage_width_median"] = float(
        np.median(merged["q90_fixed"] - merged["q10_fixed"])
    )
    return result
