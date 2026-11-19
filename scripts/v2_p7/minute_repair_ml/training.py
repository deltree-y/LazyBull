"""固定 XGBoost 回归器：64 次冻结拟合（诊断/部署 × 主/日线对照）与隔离注册（协议 §8）。

固定 300 轮、无早停、无调参、无特征筛选；每个训练日权重总和相等（1/n_d 后归一）。
主模型用 24 列，日线对照用同一行域的前 6 列；两者样本 SHA 必须一致。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import spec, splits

PANEL_GLOB = "panel_*.parquet"
MODEL_TYPE = "xgboost"
LABEL_COLUMN = str(spec.LABEL["column"])


class TrainingError(RuntimeError):
    """训练层错误（样本域不一致、缺列、折条件不足）。"""


def load_panel(dataset_dir: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """读入分年缓存（只取需要的列以控制内存）。"""
    files = sorted(Path(dataset_dir).glob(PANEL_GLOB))
    if not files:
        raise FileNotFoundError(f"{dataset_dir} 下没有 {PANEL_GLOB}")
    frames = [pd.read_parquet(path, columns=list(columns) if columns else None) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    for name in ("trade_date", "label_end_date", "entry_date", "asof_date"):
        if name in frame.columns:
            frame[name] = frame[name].astype(str)
    return frame


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """先按日内样本数取 1/n_d，再整体归一为均值 1。"""
    counts = frame.groupby("trade_date")["trade_date"].transform("size").to_numpy()
    weights = 1.0 / counts.astype(float)
    mean = float(weights.mean())
    return weights / mean if mean > 0 else weights


def model_params(n_jobs: int) -> dict[str, Any]:
    """冻结超参（唯一来源 spec.MODEL）。"""
    return {
        "objective": str(spec.MODEL["objective"]),
        "tree_method": str(spec.MODEL["tree_method"]),
        "device": str(spec.MODEL["device"]),
        "max_depth": int(spec.MODEL["max_depth"]),
        "n_estimators": int(spec.MODEL["n_estimators"]),
        "learning_rate": float(spec.MODEL["learning_rate"]),
        "min_child_weight": float(spec.MODEL["min_child_weight"]),
        "reg_lambda": float(spec.MODEL["reg_lambda"]),
        "reg_alpha": float(spec.MODEL["reg_alpha"]),
        "gamma": float(spec.MODEL["gamma"]),
        "subsample": float(spec.MODEL["subsample"]),
        "colsample_bytree": float(spec.MODEL["colsample_bytree"]),
        "max_bin": int(spec.MODEL["max_bin"]),
        "random_state": int(spec.MODEL["random_state"]),
        "n_jobs": int(n_jobs),
    }


def fit_model(train: pd.DataFrame, n_jobs: int, features: Sequence[str]) -> Any:
    """训练单个回归模型（固定参数，无早停）。"""
    from xgboost import XGBRegressor

    missing = [name for name in features if name not in train.columns]
    if missing:
        raise TrainingError(f"训练矩阵缺列（禁止自动降列）：{missing}")
    model = XGBRegressor(**model_params(n_jobs))
    model.fit(
        train[list(features)].to_numpy(dtype=np.float32),
        train[LABEL_COLUMN].to_numpy(dtype=np.float32),
        sample_weight=sample_weights(train),
        verbose=False,
    )
    return model


def fold_slices(panel: pd.DataFrame, fold: dict[str, object]) -> dict[str, pd.DataFrame]:
    """诊断/部署四段样本（主与对照共用同一行域）。"""
    train_mask, val_mask = splits.split_masks(panel, fold)
    deploy_mask = splits.deploy_mask(panel, fold)
    return {
        "diag_train": panel.loc[train_mask],
        "diag_val": panel.loc[val_mask],
        "deploy_train": panel.loc[deploy_mask],
    }


def assert_fold_conditions(fold: dict[str, object], slices: dict[str, pd.DataFrame]) -> None:
    """按 purge 后真实样本校验折条件（不是只看理论日期窗）。"""
    train = slices["diag_train"]
    val = slices["diag_val"]
    train_days = int(train["trade_date"].nunique())
    val_days = int(val["trade_date"].nunique())
    if train_days < int(spec.SPLITS["min_sampled_train_days"]):
        raise TrainingError(f"折 {fold['fold_id']} 训练采样日 {train_days} < 100")
    if val_days < int(spec.SPLITS["min_sampled_val_days"]):
        raise TrainingError(f"折 {fold['fold_id']} 验证采样日 {val_days} < 10")
    span = (
        pd.Timestamp(str(train["trade_date"].max())) - pd.Timestamp(str(train["trade_date"].min()))
    ).days / 30.44
    if span < float(spec.SPLITS["min_train_span_months"]):
        raise TrainingError(f"折 {fold['fold_id']} 训练跨度 {span:.1f} 月 < 24")


def rank_ic(frame: pd.DataFrame, score_column: str = "score") -> dict[str, Any]:
    """逐日 RankIC（Spearman）与误差；验证段只报告，不用于选模型。"""
    if frame.empty:
        return {"days": 0, "rank_ic_mean": None, "rank_ic_std": None, "mae": None}
    values = []
    for _, part in frame.groupby("trade_date"):
        if len(part) < 5:
            continue
        ic = part[score_column].corr(part[LABEL_COLUMN], method="spearman")
        if pd.notna(ic):
            values.append(float(ic))
    errors = (frame[score_column] - frame[LABEL_COLUMN]).abs()
    return {
        "days": len(values),
        "rank_ic_mean": float(np.mean(values)) if values else None,
        "rank_ic_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "mae": float(errors.mean()) if len(errors) else None,
    }


def _feature_sha(train: pd.DataFrame, features: Sequence[str]) -> str:
    """训练矩阵内容 SHA（列序 + 值）。"""
    import hashlib

    data = train[list(features)].to_numpy(dtype=np.float32).tobytes()
    digest = hashlib.sha256()
    digest.update("|".join(features).encode("utf-8"))
    digest.update(data)
    return digest.hexdigest()


def run_fold(
    panel: pd.DataFrame,
    fold: dict[str, object],
    registry: Any,
    n_jobs: int,
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """单折 4 次拟合（诊断/部署 × 主/对照），返回注册清单与验证读数。"""
    emit = log or (lambda message: None)
    started = time.perf_counter()
    slices = fold_slices(panel, fold)
    assert_fold_conditions(fold, slices)

    results: dict[str, Any] = {"fold_id": fold["fold_id"], "models": {}, "diagnostics": {}}
    variants = (
        ("main", tuple(spec.FEATURES)),
        ("daily", tuple(spec.DAILY_FEATURES)),
    )
    for role, key in (("diagnostic", "diag_train"), ("deploy", "deploy_train")):
        train = slices[key]
        if train.empty:
            raise TrainingError(f"折 {fold['fold_id']} {role} 训练集为空")
        for variant, features in variants:
            model = fit_model(train, n_jobs, features)
            version = registry.register_model(
                model=model,
                model_type=MODEL_TYPE,
                train_start_date=str(train["trade_date"].min()),
                train_end_date=str(train["trade_date"].max()),
                feature_columns=list(features),
                label_column=LABEL_COLUMN,
                n_samples=int(len(train)),
                train_params={
                    **model_params(n_jobs),
                    "candidate_id": spec.CANDIDATE_ID,
                    "spec_sha256": spec.spec_sha256(),
                    "fold_id": fold["fold_id"],
                    "role": role,
                    "variant": variant,
                    "feature_count": len(features),
                    "label_version": f"{spec.LABEL['formula']}",
                    "weight_rule": "per_date_1_over_n_then_mean_1",
                    "purge_rule": "label_end_date<V(diag train)/<C(deploy)",
                },
                performance_metrics={
                    "train_rows": int(len(train)),
                    "train_dates": int(train["trade_date"].nunique()),
                    "train_label_mean": float(train[LABEL_COLUMN].mean()),
                    "feature_sha": _feature_sha(train, features),
                },
            )
            loaded, _ = registry.load_model(version=version)
            probe = train[list(features)].to_numpy(dtype=np.float32)
            identical = bool(np.array_equal(model.predict(probe), loaded.predict(probe)))
            results["models"][f"{role}_{variant}"] = {
                "version": int(version),
                "reload_identical": identical,
                "train_rows": int(len(train)),
                "train_dates": int(train["trade_date"].nunique()),
                "feature_sha": _feature_sha(train, features),
            }
            if not identical:
                raise TrainingError(f"折 {fold['fold_id']} {role}/{variant} 重载预测不一致")
            if role == "diagnostic":
                val = slices["diag_val"].copy()
                if len(val):
                    val["score"] = loaded.predict(val[list(features)].to_numpy(dtype=np.float32))
                    results["diagnostics"][variant] = rank_ic(val)
                else:
                    results["diagnostics"][variant] = {"days": 0, "rank_ic_mean": None}
    results["seconds"] = round(time.perf_counter() - started, 2)
    emit(
        f"{fold['fold_id']}: diag_train={len(slices['diag_train'])} "
        f"deploy_train={len(slices['deploy_train'])} val={len(slices['diag_val'])} "
        f"({results['seconds']}s)"
    )
    return results


def fold_manifest(path: Path, payload: dict[str, Any]) -> None:
    """折模型清单（fold_models.json）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
