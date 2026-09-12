"""terminal_loss 模型产物落盘：注册制版本化 + 固定名别名

落盘语义（v0.108.0 起）：

- **始终走 ModelRegistry 版本化注册**（``save_versioned_artifacts``）：每次
  训练新增 ``v{N}``，永不覆盖历史。研究折目录（WF）与生产目录同规则——
  "只保存一个固定名文件、下次训练即被覆盖"的旧行为已取消。
- ``--fixed-name`` 不再是"只写固定名"的模式，而是**额外**写一套固定名别名
  （joblib/json/report/csv），供 ``summarize_terminal_risk_wf.py`` 等按固定
  文件名读取的既有工具零改动使用；别名可被覆盖，因此不能作为唯一副本。
- ES 评估行（``*_es_predictions.parquet``）随模型落盘：门禁区间重判
  （``block_stats``）与逐折区间需要逐行预测，只有聚合指标无法重采样。
- 版本化产物 ``v{N}_model.joblib`` 内是 TerminalLossModel 实例（register_model
  走 joblib 分支：外层包装无 save_model 属性），加载走
  ``registry.load_model(version)`` 直接返回可用实例，不经过
  ``TerminalLossModel.load()`` 的 artifact_version 校验；版本字段仍在实例
  ``config.artifact_version`` 与 ``v{N}_metadata.json`` 中。固定名别名的
  joblib 仍是 payload dict（走 ``load()``）。
- ``TERMINAL_LOSS_MODEL_TYPE`` 禁止包含 "classifier" 子串——ModelRegistry.
  ``get_latest_version`` 会跳过含该子串的 model_type，目录内全部为
  terminal_loss 模型，含该子串将导致 ``load_model(None)`` 取不到 latest。
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from ...ml import ModelRegistry
from .model import TerminalLossModel, _json_safe

TERMINAL_LOSS_MODEL_TYPE = "xgboost_terminal_loss"

#: 冻结标签列名（register_model 元数据用；实际矩阵列为 loss_label）
TERMINAL_LOSS_LABEL_COLUMN = "loss_label"

#: ES 评估行必需列（逐行预测，供分块重采样）
ES_PREDICTION_COLUMNS = ["trade_date", "ts_code", "h", "loss_label", "p_loss"]


def _write_es_predictions(path: Path, es_predictions: Optional[pd.DataFrame]) -> Optional[str]:
    """落盘 ES 逐行预测（紧凑 dtype：int8 标签 + float32 概率）。"""
    if es_predictions is None or es_predictions.empty:
        return None
    missing = [c for c in ES_PREDICTION_COLUMNS if c not in es_predictions.columns]
    if missing:
        raise ValueError(f"ES 评估行缺少列: {missing}，无法落盘供门禁重采样")
    frame = es_predictions[ES_PREDICTION_COLUMNS].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["h"] = frame["h"].astype("int16")
    frame["loss_label"] = frame["loss_label"].astype("int8")
    frame["p_loss"] = frame["p_loss"].astype("float32")
    frame.to_parquet(path, index=False)
    logger.info(f"ES 评估行已落盘（{len(frame)} 行）: {path}")
    return str(path)


def build_performance_metrics(
    es_report: Dict[str, Any],
    train_report: Dict[str, Any],
    val_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """从概率质量报告提取注册用指标（口径与 summarize_terminal_risk_wf 对齐）。

    lift = ES PR-AUC / ES 事件率（相对 null 基线的排序信息量）；
    pred_bias = ES mean_pred - ES 事件率（校准漂移，正值=概率高估）。

    ES 是**评估段**（不参与早停，方案 5.2）；``val_report`` 为早停段报告，
    仅旁路登记 val_* 审计列（判断早停段与评估段的差距），不参与门禁口径。
    """
    event_rate = es_report.get("event_rate")
    mean_pred = es_report.get("mean_pred")
    metrics = {
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
    if val_report is not None:
        metrics.update(
            {
                "val_logloss": val_report.get("logloss"),
                "val_brier": val_report.get("brier"),
                "val_pr_auc": val_report.get("pr_auc"),
                "val_event_rate": val_report.get("event_rate"),
            }
        )
    return _json_safe(metrics)


def save_flat_artifacts(
    out_dir: Path,
    model: TerminalLossModel,
    report_payload: Dict[str, Any],
    calibration_df: Optional[pd.DataFrame],
    coverage_df: pd.DataFrame,
    es_predictions: Optional[pd.DataFrame] = None,
) -> None:
    """写固定名别名（供既有工具按固定文件名读取；可被覆盖，非唯一副本）。

    调用方（``save_terminal_loss_artifacts``）已先完成注册制版本化落盘；
    本函数单独使用时不提供版本保护，仅供测试与别名场景。
    """
    model.save(str(out_dir / "terminal_loss_model.joblib"))
    with open(out_dir / "terminal_loss_report.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    if calibration_df is not None:
        calibration_df.to_csv(
            out_dir / "calibration_by_h_sigma.csv", index=False, encoding="utf-8-sig"
        )
    coverage_df.to_csv(out_dir / "label_coverage.csv", index=False, encoding="utf-8-sig")
    _write_es_predictions(out_dir / "terminal_loss_es_predictions.parquet", es_predictions)
    logger.info(f"terminal_loss 固定名别名已保存: {out_dir}")


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
    es_predictions: Optional[pd.DataFrame] = None,
) -> int:
    """版本化落盘：注册 ModelRegistry 新版本（永不覆盖），返回版本号。

    Args:
        train_params: 训练与标签配置快照（model_config.metadata，JSON 安全）
        performance_metrics: build_performance_metrics 产出
        report_payload: 概率质量报告（与固定名模式 terminal_loss_report.json 同构）
        es_predictions: ES 逐行预测（供门禁区间重采样）
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
    _write_es_predictions(out_dir / f"{version_str}_es_predictions.parquet", es_predictions)
    logger.info(f"terminal_loss 产物已保存（版本化 {version_str}）: {out_dir}")
    return version


def save_terminal_loss_artifacts(
    out_dir: Path,
    model: TerminalLossModel,
    *,
    fixed_name: bool,
    train_start_date: str,
    train_end_date: str,
    n_samples: int,
    train_params: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    report_payload: Dict[str, Any],
    calibration_df: Optional[pd.DataFrame],
    coverage_df: pd.DataFrame,
    es_predictions: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """统一落盘入口：总是注册制版本化，``fixed_name`` 时追加固定名别名。

    背景：旧实现下 ``--fixed-name``（WF 折目录）只写一套固定名文件，下一次
    同配置训练即整目录覆盖，模型只存在一个副本、无法追溯。现在任何一次
    训练都会在输出目录的 ModelRegistry 里新增版本（``v{N}``，含 metadata /
    features / report / 校准表 / 覆盖表 / ES 评估行），``--fixed-name`` 仅
    额外写别名供既有工具按固定文件名读取。

    Returns:
        dict：version / version_str / fixed_name / es_predictions_path
    """
    version = save_versioned_artifacts(
        out_dir,
        model,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        n_samples=n_samples,
        train_params=train_params,
        performance_metrics=performance_metrics,
        report_payload=report_payload,
        calibration_df=calibration_df,
        coverage_df=coverage_df,
        es_predictions=es_predictions,
    )
    if fixed_name:
        save_flat_artifacts(
            out_dir,
            model,
            report_payload,
            calibration_df,
            coverage_df,
            es_predictions=es_predictions,
        )
    return {
        "version": version,
        "version_str": f"v{version}",
        "fixed_name": bool(fixed_name),
        "es_predictions_path": (
            str(out_dir / f"v{version}_es_predictions.parquet")
            if es_predictions is not None
            else None
        ),
    }
