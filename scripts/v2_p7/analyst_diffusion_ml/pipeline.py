"""阶段编排：数据集构建、折计划、训练与预测（R1/R2）。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import (
    dataset,
    panel,
    prediction,
    report_events,
    spec,
    splits,
    training,
)


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def fold_plan(root: Path) -> list[dict[str, object]]:
    """16 折刷新计划（按完整日历，断言数量与协议一致）。"""
    calendar = panel.load_calendar(root)
    folds = splits.build_folds(calendar, end=str(spec.WINDOW["end"]))
    expected = int(spec.SPLITS["expected_folds"])
    if len(folds) != expected:
        raise ValueError(f"折数 {len(folds)} 与协议冻结的 {expected} 不一致")
    problems = {fold["fold_id"]: splits.check_fold_requirements(fold) for fold in folds}
    bad = {key: value for key, value in problems.items() if value}
    if bad:
        raise ValueError(f"折条件不足：{bad}")
    return folds


def run_dataset(
    root: Path,
    campaign_dir: Path,
    *,
    source_manifest_sha: str,
    work_dir: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """R1：构建分年面板、事件表与台账，写入 `work_dir`。"""
    emit = _emit(log)
    started = time.perf_counter()
    folds = fold_plan(root)
    feature_start = str(folds[0]["L"])
    emit(f"折计划：{len(folds)} 折，首折 L={feature_start}")

    out_dir = Path(work_dir) / "dataset"
    events_dir = Path(work_dir) / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    result = dataset.build_dataset(
        root,
        feature_start=feature_start,
        end=str(spec.WINDOW["end"]),
        out_dir=out_dir,
        source_manifest_sha=source_manifest_sha,
        log=emit,
    )
    events = result["events"]
    for name in ("eps_events", "target_events", "rating_events", "primary", "activity"):
        frame = events[name]
        frame.to_parquet(events_dir / f"{name}.parquet", index=False)
    (events_dir / "event_audit.json").write_text(
        json.dumps(events["audit"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fingerprints = report_events.event_fingerprint(events)
    (events_dir / "event_fingerprint.json").write_text(
        json.dumps(fingerprints, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fold_rows = []
    for fold in folds:
        fold_rows.append({k: v for k, v in fold.items() if not isinstance(v, list)})
    pd.DataFrame(fold_rows).to_parquet(Path(work_dir) / "dataset" / "folds.parquet", index=False)

    summary = {
        **result["summary"],
        "folds": len(folds),
        "feature_start": feature_start,
        "event_fingerprint": fingerprints,
        "event_audit": events["audit"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (Path(work_dir) / "dataset" / "stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit(f"R1 完成：{summary['rows']} 行 / {summary['days']} 日，{summary['elapsed_seconds']}s")
    return {"summary": summary, "folds": folds}


def load_panel(dataset_dir: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """读入分年缓存。"""
    files = sorted(Path(dataset_dir).glob("panel_*.parquet"))
    if not files:
        raise FileNotFoundError(f"{dataset_dir} 下没有 panel_*.parquet")
    frames = [pd.read_parquet(path, columns=columns) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    for name in ("trade_date", "label_end_date", "entry_date"):
        if name in frame.columns:
            frame[name] = frame[name].astype(str)
    return frame


def dataset_domain(dataset_dir: Path) -> pd.DataFrame:
    """读入逐日域台账。"""
    return pd.read_parquet(Path(dataset_dir) / "daily_domain.parquet")


def write_stage_json(path: Path, payload: dict[str, Any]) -> str:
    """写阶段 JSON（显式 UTF-8）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=float)
    path.write_text(text, encoding="utf-8")
    return str(path)


def fold_manifest(folds: list[dict[str, object]]) -> dict[str, Any]:
    """折清单摘要（供报告与复跑核对）。"""
    return {
        "count": len(folds),
        "folds": [
            {
                key: fold[key]
                for key in ("fold_id", "L", "V", "C", "train_first", "train_last",
                            "val_first", "val_last", "n_train_days", "n_val_days",
                            "oos_first", "oos_last", "n_oos_days")
            }
            for fold in folds
        ],
        "expected": int(spec.SPLITS["expected_folds"]),
        "sampling": f"every {spec.SPLITS['sample_every_trading_days']} trading days "
        f"anchor {spec.SPLITS['sample_anchor']}",
        "purge": "label_end_date < V (train) / < C (val)",
    }


def count_nonfinite(values: np.ndarray) -> int:
    """非有限预测计数（模型/输入错误的显式门）。"""
    return int((~np.isfinite(values)).sum())


#: 训练/预测需要的列（FEATURES 已含两个扩散列，不得重复，否则列名重复）
PANEL_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "ts_code",
    *spec.FEATURES,
    spec.LABEL["column"],
    spec.LABEL["raw_column"],
    "label_status",
    "label_end_date",
    "industry_bucket",
)


def run_model_probe(
    root: Path,
    campaign_dir: Path,
    dataset_dir: Path,
    *,
    work_dir: Path,
    n_jobs: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """首折性能探针：只看规模/耗时/有限性，不读取该折 OOS 收益。"""
    emit = _emit(log)
    folds = fold_plan(root)
    frame = pipeline_load(dataset_dir)
    probe = training.probe_fold(frame, folds[0], n_jobs, emit)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    (Path(work_dir) / "model_probe.json").write_text(
        json.dumps(probe, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    emit(f"探针：{probe}")
    return probe


def pipeline_load(dataset_dir: Path) -> pd.DataFrame:
    """装载训练/预测需要的列（控制内存）。"""
    frame = training.load_panel(Path(dataset_dir), columns=list(PANEL_COLUMNS))
    frame["label_status"] = frame["label_status"].astype(str)
    return frame


def run_train_predict(
    root: Path,
    campaign_dir: Path,
    dataset_dir: Path,
    *,
    work_dir: Path,
    n_jobs: int,
    protocol_sha: str,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """R2：16 折固定模型训练、隔离注册、全域 OOS 预测与 AD_SIMPLE 排名。"""
    from src.lazybull.v2.common.model_registry import ModelRegistry

    emit = _emit(log)
    started = time.perf_counter()
    folds = fold_plan(root)
    frame = pipeline_load(dataset_dir)
    registry = ModelRegistry(str(Path(campaign_dir) / "models"))
    predictions_dir = Path(work_dir) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    oos_days = sorted({str(day) for fold in folds for day in fold["oos_days"]})
    simple_source = frame[frame["trade_date"].astype(str).isin(set(oos_days))]
    simple = prediction.ad_simple_rank(simple_source)

    records: list[dict[str, Any]] = []
    fold_predictions: list[pd.DataFrame] = []
    for fold in folds:
        train = training.training_slice(frame, fold)
        val = training.validation_slice(frame, fold)
        training.assert_fold_conditions(fold, train, val)
        model = training.fit_model(train, n_jobs)
        registration = training.register_fold_model(
            registry, fold, train, val, model, {"protocol_sha256": protocol_sha}, emit
        )
        validation_ranked = prediction.rank_frame(val, prediction.predict_matrix(model, val))
        val_metrics = prediction.evaluation_metrics(val, validation_ranked)
        result = prediction.predict_fold(
            frame,
            model,
            fold,
            registration["version"],
            spec.spec_sha256(),
            simple=simple,
            log=emit,
        )
        fold_predictions.append(result["predictions"])
        records.append(
            {
                "fold_id": fold["fold_id"],
                "L": fold["L"],
                "V": fold["V"],
                "C": fold["C"],
                "model_version": registration["version"],
                "reload_identical": registration["reload_identical"],
                "train_rows": int(len(train)),
                "train_dates": int(train["trade_date"].nunique()),
                "val_rows": int(len(val)),
                "val_dates": int(val["trade_date"].nunique()),
                "val_metrics": val_metrics,
                "oos_rows": result["rows"],
            }
        )

    predictions = pd.concat(fold_predictions, ignore_index=True)
    predictions = predictions.sort_values(["trade_date", "rank"], kind="mergesort")
    counts = prediction.daily_counts(predictions, int(spec.UNIVERSE["min_scored_per_day"]))
    predictions.to_parquet(predictions_dir / "predictions_all.parquet", index=False)
    simple.to_parquet(predictions_dir / "predictions_simple.parquet", index=False)
    counts.to_parquet(predictions_dir / "daily_counts.parquet", index=False)
    oos_metrics = prediction.evaluation_metrics(
        frame[frame["trade_date"].astype(str).isin(set(oos_days))], predictions
    )
    summary = {
        "folds": len(folds),
        "models_registered": len(records),
        "reload_all_identical": bool(all(item["reload_identical"] for item in records)),
        "oos_rows": int(len(predictions)),
        "oos_days": int(predictions["trade_date"].nunique()),
        "min_scored": int(counts["scored_count"].min()),
        "days_below_100": int((counts["min100_status"] != "ok").sum()),
        "oos_metrics": oos_metrics,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    training.fold_model_manifest(
        Path(campaign_dir) / "models" / "fold_models.json",
        {
            "candidate_id": spec.CANDIDATE_ID,
            "spec_sha256": spec.spec_sha256(),
            "protocol_sha256": protocol_sha,
            "feature_columns": list(spec.FEATURES),
            "label_column": spec.LABEL["column"],
            "folds": records,
        },
    )
    (Path(work_dir) / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    emit(f"R2 完成：{summary['models_registered']} 模型 / {summary['elapsed_seconds']}s")
    return {"summary": summary, "records": records}
