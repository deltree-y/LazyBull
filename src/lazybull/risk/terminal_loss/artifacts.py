"""terminal_loss 模型产物落盘：flat / 版本化双模式公共层

- flat 模式（脚本 --fixed-name）：固定名五件套（terminal_loss_model.joblib +
  sidecar + report + 2 CSV），同名覆盖，供 WF 折目录研究与
  summarize_terminal_risk_wf.py 消费；行为与 0.104.x 完全一致。
- 版本化模式（默认）：复用 ModelRegistry 版本递增注册（永不覆盖），
  产出 v{N}_model.joblib / v{N}_features.json / v{N}_metadata.json 及
  v{N}_report.json / v{N}_calibration_by_h_sigma.csv / v{N}_label_coverage.csv，
  与目录级 model_registry.json / latest_model_version.txt。

契约说明：
- TERMINAL_LOSS_MODEL_TYPE 禁止包含 "classifier" 子串——ModelRegistry.
  get_latest_version 会跳过 model_type 含 "classifier" 的模型，本目录
  registry 内全部为 terminal_loss 模型，含该子串将导致 load_model(None)
  永远取不到 latest（目录内无主模型可回退）。
- 版本化产物 v{N}_model.joblib 内是 TerminalLossModel 实例（register_model
  走 joblib 分支：外层包装无 save_model 属性），加载走
  registry.load_model(version) 直接返回可用实例，不经过 TerminalLossModel.
  load() 的 artifact_version 校验；版本字段仍在实例 config.artifact_version
  与 v{N}_metadata.json 中。flat 模式产物仍是 payload dict（走 load()）。
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from ...ml import ModelRegistry
from .model import TerminalLossModel, _json_safe

TERMINAL_LOSS_MODEL_TYPE = "xgboost_terminal_loss"

# 冻结标签列名（register_model 元数据用；实际矩阵列为 loss_label）
TERMINAL_LOSS_LABEL_COLUMN = "loss_label"


def build_performance_metrics(
    es_report: Dict[str, Any], train_report: Dict[str, Any]
) -> Dict[str, Any]:
    """从概率质量报告提取注册用指标（口径与 summarize_terminal_risk_wf 对齐）。

    lift = ES PR-AUC / ES 事件率（相对 null 基线的排序信息量）；
    pred_bias = ES mean_pred - ES 事件率（校准漂移，正值=概率高估）。
    """
    event_rate = es_report.get("event_rate")
    mean_pred = es_report.get("mean_pred")
    return _json_safe(
        {
            "es_logloss": es_report.get("logloss"),
            "es_brier": es_report.get("brier"),
            "es_pr_auc": es_report.get("pr_auc"),
            "es_event_rate": event_rate,
            "es_mean_pred": mean_pred,
            "lift": (
                es_report["pr_auc"] / event_rate
                if event_rate is not None and event_rate > 0
                else None
            ),
            "pred_bias": (
                mean_pred - event_rate if mean_pred is not None and event_rate is not None else None
            ),
            "train_logloss": train_report.get("logloss"),
            "train_event_rate": train_report.get("event_rate"),
        }
    )


def save_flat_artifacts(
    out_dir: Path,
    model: TerminalLossModel,
    report_payload: Dict[str, Any],
    calibration_df: Optional[pd.DataFrame],
    coverage_df: pd.DataFrame,
) -> None:
    """flat 固定名落盘（WF 折目录研究用，同名覆盖）。"""
    model.save(str(out_dir / "terminal_loss_model.joblib"))
    with open(out_dir / "terminal_loss_report.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    if calibration_df is not None:
        calibration_df.to_csv(
            out_dir / "calibration_by_h_sigma.csv", index=False, encoding="utf-8-sig"
        )
    coverage_df.to_csv(out_dir / "label_coverage.csv", index=False, encoding="utf-8-sig")
    logger.info(f"terminal_loss 产物已保存（flat 固定名）: {out_dir}")


def save_versioned_artifacts(
    out_dir: Path,
    model: TerminalLossModel,
    *,
    train_start_date: str,
    train_end_date: str,
    n_samples: int,
    train_params: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    report_payload: Dict[str, Any],
    calibration_df: Optional[pd.DataFrame],
    coverage_df: pd.DataFrame,
) -> int:
    """版本化落盘：注册 ModelRegistry 新版本（永不覆盖），返回版本号。

    Args:
        train_params: 训练与标签配置快照（model_config.metadata，JSON 安全）
        performance_metrics: build_performance_metrics 产出
        report_payload: 概率质量报告（与 flat 模式 terminal_loss_report.json 同构）
    """
    registry = ModelRegistry(models_dir=str(out_dir))
    version = registry.register_model(
        model=model,
        model_type=TERMINAL_LOSS_MODEL_TYPE,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        feature_columns=model.feature_names,
        label_column=TERMINAL_LOSS_LABEL_COLUMN,
        n_samples=n_samples,
        train_params=_json_safe(train_params),
        performance_metrics=performance_metrics,
    )
    version_str = f"v{version}"
    with open(out_dir / f"{version_str}_report.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    if calibration_df is not None:
        calibration_df.to_csv(
            out_dir / f"{version_str}_calibration_by_h_sigma.csv",
            index=False,
            encoding="utf-8-sig",
        )
    coverage_df.to_csv(
        out_dir / f"{version_str}_label_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    logger.info(f"terminal_loss 产物已保存（版本化 {version_str}）: {out_dir}")
    return version
