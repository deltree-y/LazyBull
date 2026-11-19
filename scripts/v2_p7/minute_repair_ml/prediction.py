"""OOS 每日全域评分与冻结排名（协议 §8.3）。

产物覆盖**完整基础合格域**：可评分行给有限分数，不可评分行给 NaN 与明确原因；
传入引擎的排名只含当日有效域；不足 100 只的日期不生成新买信号（由回放层执行）。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.minute_repair_ml import spec, splits

VALID = "scored"
NOT_INFORMED = "minute_not_informed"
NO_MODEL = "model_missing"
EMPTY = "empty_domain"


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def score_fold(
    panel: pd.DataFrame,
    fold: dict[str, object],
    registry: Any,
    models: dict[str, Any],
    variant: str,
    *,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """单折 OOS 评分（部署模型），返回全域长表。"""
    emit = _emit(log)
    key = f"deploy_{variant}"
    entry = models.get(key)
    if entry is None:
        raise KeyError(f"折 {fold['fold_id']} 缺部署模型 {key}")
    model, _meta = registry.load_model(version=int(entry["version"]))
    features = list(spec.FEATURES) if variant == "main" else list(spec.DAILY_FEATURES)
    rows = panel.loc[splits.oos_mask(panel, fold)].copy()
    rows["fold_id"] = str(fold["fold_id"])
    rows["model_version"] = int(entry["version"])
    rows["model_sha"] = str(entry.get("feature_sha", ""))[:16]
    rows["variant"] = variant
    informed = rows["minute_informed"].astype(bool).to_numpy()
    rows["domain_status"] = np.where(informed, VALID, NOT_INFORMED)
    rows["score"] = np.nan
    if informed.any():
        block = rows.loc[informed]
        rows.loc[informed, "score"] = model.predict(
            block[features].to_numpy(dtype=np.float32)
        ).astype(float)
    rows["score_rank"] = np.nan
    scored = rows["domain_status"] == VALID
    if scored.any():
        order = rows.loc[scored].sort_values(
            ["trade_date", "score", "ts_code"], ascending=[True, False, True]
        )
        ranks = order.groupby("trade_date").cumcount() + 1
        rows.loc[order.index, "score_rank"] = ranks.astype(float)
    emit(
        f"{fold['fold_id']}/{variant}: rows={len(rows)} scored={int(scored.sum())} "
        f"days={rows['trade_date'].nunique()}"
    )
    return rows


def freeze_predictions(frames: Sequence[pd.DataFrame], dataset_dir: Path) -> pd.DataFrame:
    """合并全部折并落盘（每折每行一条；同股同日只应出现一次）。"""
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    duplicated = frame.duplicated(subset=["trade_date", "ts_code", "variant"]).sum()
    if duplicated:
        raise ValueError(f"OOS 评分存在同日同股重复：{duplicated}")
    for year, part in frame.groupby(frame["trade_date"].str[:4]):
        part.reset_index(drop=True).to_parquet(
            Path(dataset_dir) / f"predictions_{year}.parquet", index=False
        )
    return frame


def ranked_from_predictions(frame: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """把评分长表转成引擎需要的 `{trade_date: [(code, score)…]}`（已按冻结序）。"""
    ranked: dict[str, list[tuple[str, float]]] = {}
    usable = frame[(frame["domain_status"] == VALID) & np.isfinite(frame["score"])]
    for day, part in usable.groupby("trade_date"):
        ordered = part.sort_values(["score", "ts_code"], ascending=[False, True])
        ranked[str(day)] = [
            (str(code), float(score)) for code, score in zip(ordered["ts_code"], ordered["score"])
        ]
    return ranked


def daily_score_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """逐日可评分数与门槛状态。"""
    threshold = int(spec.UNIVERSE["min_scored_per_day"])
    rows = []
    for day, part in frame.groupby("trade_date"):
        scored = int((part["domain_status"] == VALID).sum())
        rows.append(
            {
                "trade_date": str(day),
                "domain_rows": int(len(part)),
                "scored": scored,
                "status": "ok" if scored >= threshold else "below_minimum",
            }
        )
    return pd.DataFrame(rows)


def skip_days(frame: pd.DataFrame) -> set[str]:
    """不足 100 只可评分 -> 当日跳过新买信号（既有持仓与卖出义务照常）。"""
    counts = daily_score_counts(frame)
    return set(counts.loc[counts["status"] != "ok", "trade_date"].astype(str))
