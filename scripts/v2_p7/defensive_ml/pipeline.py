"""R2 编排：首折性能探针、全折训练/注册/重载核验与 OOS 全域预测。

不读取任何收益用于选模；探针只报规模、耗时、内存与机械正确性。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from scripts.v2_p7.defensive_ml import (
    artifacts,
    panel,
    prediction,
    spec,
    splits,
    training,
)

PANEL_COLUMNS = (
    ["trade_date", "ts_code"]
    + list(spec.FEATURES)
    + ["y_def_ret60", "label_status", "label_end_date"]
)


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def load_working_panel(dataset_dir: Path, log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """装载 R1 缓存（只取训练/预测所需列）。"""
    emit = _emit(log)
    started = time.perf_counter()
    frame = training.load_panel(dataset_dir, columns=PANEL_COLUMNS)
    emit(f"panel {len(frame)} rows, {frame['trade_date'].nunique()} days, "
         f"{time.perf_counter() - started:.1f}s")
    return frame


def build_fold_plan(calendar: list[str], log: Callable[[str], None] | None = None) -> list[dict]:
    """构造并校验 48 模型所需的全部折。"""
    emit = _emit(log)
    folds = splits.build_folds(calendar, str(spec.WINDOW["end"]))
    problems: list[str] = []
    for fold in folds:
        for issue in splits.check_fold_requirements(fold):
            problems.append(f"{fold['fold_id']}:{issue}")
    emit(f"folds={len(folds)} models={len(folds) * 3} problems={len(problems)}")
    if problems:
        emit("fold problems: " + ", ".join(problems))
    return folds


def run_model_probe(
    repo_root: Path, campaign_dir: Path, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """首折性能探针（不写模型注册表）。"""
    emit = _emit(log)
    calendar = panel.load_calendar(repo_root)
    folds = build_fold_plan(calendar, emit)
    if not folds:
        raise ValueError("没有可用折")
    work = load_working_panel(campaign_dir / "dataset", emit)
    n_jobs = spec.n_jobs(os.cpu_count() or 1)
    result = training.probe_fold(work, folds[0], n_jobs, emit)
    result["n_folds_planned"] = len(folds)
    result["estimated_train_seconds"] = round(result["seconds"] * len(folds), 1)
    return result


def run_train_predict(
    repo_root: Path,
    campaign_dir: Path,
    protocol_sha: str,
    log: Callable[[str], None] | None = None,
    folds_limit: int | None = None,
) -> dict[str, Any]:
    """全折训练、注册、重载核验与 OOS 全域预测。"""
    from src.lazybull.v2.common.model_registry import ModelRegistry

    emit = _emit(log)
    started = time.perf_counter()
    calendar = panel.load_calendar(repo_root)
    folds = build_fold_plan(calendar, emit)
    if folds_limit:
        folds = folds[:folds_limit]
    work = load_working_panel(campaign_dir / "dataset", emit)
    n_jobs = spec.n_jobs(os.cpu_count() or 1)
    spec_sha = spec.spec_sha256()
    registry = ModelRegistry(str(campaign_dir / "models"))
    pred_dir = campaign_dir / "predictions"

    records: list[dict[str, Any]] = []
    oos_frames: list[pd.DataFrame] = []
    for fold in folds:
        fold_started = time.perf_counter()
        train = training.training_slice(work, fold)
        val = training.validation_slice(work, fold)
        # C3：按 purge 后的真实样本执行硬门，不足即失败（不得打印后继续）
        training.assert_fold_conditions(fold, train, val)
        models = {
            quantile: training.fit_quantile(train, quantile, n_jobs)
            for quantile in spec.MODEL["quantiles"]
        }
        registered = training.register_fold_models(
            registry, fold, train, models, {"protocol_sha256": protocol_sha}, emit
        )
        fold["versions"] = registered["versions"]
        loaded = {
            key: registry.load_model(version=registered["versions"][key])[0]
            for key in ("0.1", "0.5", "0.9")
        }
        frame = prediction.predict_fold(work, fold, loaded, spec_sha, emit)
        info = prediction.write_fold_predictions(pred_dir, str(fold["fold_id"]), frame)
        oos_frames.append(frame)
        records.append(
            {
                "fold_id": fold["fold_id"],
                "C": fold["C"],
                "V": fold["V"],
                "L": fold["L"],
                "train_rows": int(len(train)),
                "train_dates": int(train["trade_date"].nunique()),
                "versions": registered["versions"],
                "reload_identical": registered["reload_identical"],
                "oos_rows": info["rows"],
                "oos_days": info["days"],
                "crossed_rate": info["crossed_rate"],
                "seconds": round(time.perf_counter() - fold_started, 1),
            }
        )
    combined = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    if len(combined):
        combined.to_parquet(pred_dir / "predictions_all.parquet", index=False)
    summary = {
        "folds_run": len(records),
        "models_registered": sum(len(r.get("versions", {})) for r in records),
        "rows": int(len(combined)),
        "days": int(combined["trade_date"].nunique()) if len(combined) else 0,
        "crossed_rate": float(combined["crossed"].mean()) if len(combined) else 0.0,
        "reload_all_identical": all(
            all(r.get("reload_identical", {}).values()) for r in records if r.get("versions")
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "records": records,
    }
    return summary


def write_stage_json(
    run_dir: Path, name: str, payload: dict[str, Any]
) -> dict[str, str]:
    """阶段结果落盘（拒绝覆盖不同内容）。"""
    return artifacts.write_json_checked(run_dir / name, payload)


def fold_manifest(campaign_dir: Path, payload: dict[str, Any]) -> Path:
    """fold_models.json（三模型版本与身份）。"""
    path = campaign_dir / "models" / "fold_models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
