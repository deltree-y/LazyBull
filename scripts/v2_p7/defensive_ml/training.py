"""三个分位 XGBoost 的训练、注册与重载核验（协议 §7.1 / §8）。

固定 300 轮、无早停、无调参、无特征筛选；每个训练日期权重相同（1/n_d 后归一）。
现有 `v2/train/core/xgb.py` 不暴露分位损失参数，故这里直接走官方接口，
模型仍经 `v2/common/model_registry.py` 做隔离根版本化注册。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.defensive_ml import spec

PANEL_GLOB = "panel_*.parquet"
MODEL_TYPE = "xgboost"
LABEL_COLUMN = "y_def_ret60"


def load_panel(dataset_dir: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """读入分年缓存（只取需要的列以控制内存）。"""
    files = sorted(Path(dataset_dir).glob(PANEL_GLOB))
    if not files:
        raise FileNotFoundError(f"{dataset_dir} 下没有 {PANEL_GLOB}")
    frames = [pd.read_parquet(path, columns=list(columns) if columns else None) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    for name in ("trade_date", "label_end_date"):
        if name in frame.columns:
            frame[name] = frame[name].astype(str)
    return frame


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """先按日内样本数取 1/n_d，再整体归一为均值 1。"""
    counts = frame.groupby("trade_date")["trade_date"].transform("size").to_numpy()
    weights = 1.0 / counts.astype(float)
    mean = float(weights.mean())
    return weights / mean if mean > 0 else weights


def training_slice(
    panel: pd.DataFrame, fold: dict[str, object], require_valid: bool = True
) -> pd.DataFrame:
    """训练行：日期窗 + 标签在 V 前成熟 + 仅 valid。"""
    dates = panel["trade_date"]
    mask = dates.isin(set(fold["train_days"])) & (panel["label_end_date"] < str(fold["V"]))
    if require_valid:
        mask &= panel["label_status"] == "valid"
    return panel.loc[mask]


def validation_slice(panel: pd.DataFrame, fold: dict[str, object]) -> pd.DataFrame:
    """验证行：V<=T<C 且标签在 C 前成熟。"""
    dates = panel["trade_date"]
    mask = (
        dates.isin(set(fold["val_days"]))
        & (panel["label_end_date"] < str(fold["C"]))
        & (panel["label_status"] == "valid")
    )
    return panel.loc[mask]


class FoldConditionError(RuntimeError):
    """折的训练/验证条件不满足（协议 §7.2 的硬门，不得静默跳过）。"""


def assert_fold_conditions(fold: dict[str, object], train: pd.DataFrame, val: pd.DataFrame) -> None:
    """按 **purge 后的真实样本** 校验折条件（C3：不是只看理论日期窗）。

    要求：训练采样日 ≥100、验证采样日 ≥10、训练跨度 ≥36 个自然月。
    """
    train_days = sorted({str(day) for day in train["trade_date"].unique()})
    val_days = sorted({str(day) for day in val["trade_date"].unique()})
    need_train = int(spec.SPLITS["min_sampled_train_days"])
    need_val = int(spec.SPLITS["min_sampled_val_days"])
    if len(train_days) < need_train:
        raise FoldConditionError(
            f"折 {fold.get('fold_id')} 实际训练采样日 {len(train_days)} < {need_train}"
        )
    if len(val_days) < need_val:
        raise FoldConditionError(
            f"折 {fold.get('fold_id')} 实际验证采样日 {len(val_days)} < {need_val}"
        )
    span_months = _month_span(train_days[0], train_days[-1])
    need_span = int(spec.SPLITS["min_train_span_months"])
    if span_months < need_span:
        raise FoldConditionError(
            f"折 {fold.get('fold_id')} 训练跨度 {span_months} 月 < {need_span}"
        )


def _month_span(first: str, last: str) -> int:
    """两个 YYYYMMDD 之间的自然月数（向下取整）。"""
    y0, m0 = int(first[:4]), int(first[4:6])
    y1, m1 = int(last[:4]), int(last[4:6])
    return (y1 - y0) * 12 + (m1 - m0)


def _model_params(quantile: float, n_jobs: int) -> dict[str, Any]:
    return {
        "objective": str(spec.MODEL["objective"]),
        "quantile_alpha": float(quantile),
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


def fit_quantile(
    train: pd.DataFrame,
    quantile: float,
    n_jobs: int,
    features: Sequence[str] = spec.FEATURES,
) -> Any:
    """训练单个分位模型（固定参数，无早停）。"""
    from xgboost import XGBRegressor

    model = XGBRegressor(**_model_params(quantile, n_jobs))
    model.fit(
        train[list(features)].to_numpy(dtype=np.float32),
        train[LABEL_COLUMN].to_numpy(dtype=np.float32),
        sample_weight=sample_weights(train),
        verbose=False,
    )
    return model


def register_fold_models(
    registry: Any,
    fold: dict[str, object],
    train: pd.DataFrame,
    models: dict[float, Any],
    extra: dict[str, Any],
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """三个分位分别注册，返回版本与重载一致性。"""
    emit = log or (lambda message: None)
    versions: dict[str, int] = {}
    reload_ok: dict[str, bool] = {}
    sample = train.sample(n=min(512, len(train)), random_state=spec.MODEL["random_state"])
    matrix = sample[list(spec.FEATURES)].to_numpy(dtype=np.float32)
    for quantile, model in models.items():
        params = _model_params(quantile, int(model.get_params().get("n_jobs", 1)))
        params.update(
            {
                "candidate_id": spec.CANDIDATE_ID,
                "spec_sha256": spec.spec_sha256(),
                "fold_id": fold["fold_id"],
                "quantile": quantile,
                "weight_rule": "per_date_1_over_n_then_mean_1",
                "sample_rule": "every_5_trading_days_anchor_20100104",
                "purge_rule": "label_end_date<V",
                "protocol_sha256": extra.get("protocol_sha256"),
                "early_stopping": False,
            }
        )
        before = model.predict(matrix)
        version = registry.register_model(
            model=model,
            model_type=MODEL_TYPE,
            train_start_date=str(fold["L"]),
            train_end_date=str(fold["V"]),
            feature_columns=list(spec.FEATURES),
            label_column=LABEL_COLUMN,
            n_samples=int(len(train)),
            train_params=params,
            performance_metrics={
                "quantile": quantile,
                "n_dates": int(train["trade_date"].nunique()),
            },
        )
        loaded, _ = registry.load_model(version=version)
        after = loaded.predict(matrix)
        reload_ok[str(quantile)] = bool(np.array_equal(before, after))
        versions[str(quantile)] = int(version)
        emit(f"  q={quantile} version={version} reload_identical={reload_ok[str(quantile)]}")
    return {"versions": versions, "reload_identical": reload_ok}


def fold_model_manifest(path: Path, payload: dict[str, Any]) -> None:
    """折模型清单（fold_models.json）。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_fold(
    panel: pd.DataFrame,
    fold: dict[str, object],
    n_jobs: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """首折性能探针：只报规模/耗时/内存/机械正确性，不看收益。"""
    emit = log or (lambda message: None)
    started = time.perf_counter()
    train = training_slice(panel, fold)
    val = validation_slice(panel, fold)
    shape = {
        "fold_id": fold["fold_id"],
        "train_rows": int(len(train)),
        "train_dates": int(train["trade_date"].nunique()),
        "val_rows": int(len(val)),
        "val_dates": int(val["trade_date"].nunique()),
        "features": len(spec.FEATURES),
    }
    emit(f"probe fold {fold['fold_id']}: {shape}")
    if train.empty:
        raise ValueError(f"折 {fold['fold_id']} 训练集为空")
    models = {
        quantile: fit_quantile(train, quantile, n_jobs)
        for quantile in spec.MODEL["quantiles"]
    }
    elapsed = time.perf_counter() - started
    finite = all(
        np.isfinite(model.predict(train[list(spec.FEATURES)].to_numpy(dtype=np.float32))).all()
        for model in models.values()
    )
    return {
        **shape,
        "seconds": round(elapsed, 2),
        "finite_predictions": bool(finite),
        "rows_per_second": round(len(train) / max(elapsed, 1e-9), 1),
        "n_jobs": n_jobs,
    }
