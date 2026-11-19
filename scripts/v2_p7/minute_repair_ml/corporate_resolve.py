"""分红原件冲突隔离（研究根内，不改生产 raw）。

触发条件：同一 `(ts_code, ex_date)` 在同一可知时点 `known_date = max(ann_date, imp_ann_date)`
下存在多于一个**经济字段组合**。经济字段只取真正影响账户的现金/送转，且按数值比较
（缺失与 0 等价，不得因 `""` 与 `"0.0"` 的表示差异判成冲突）。

处置（协议 §9：冲突必须隔离并登记未决，**不得任取一条**）：
- 冲突组的全部行一并剔除，账户因此拿不到该次现金/送转（保守、偏向否决）；
- 逐组写入 `unresolved` 审计（含全部变体原值），报告必须原样引用；
- 未命中冲突的行一律不动；生产 `data/raw/dividend` 不被修改。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

#: 真正进入账户的经济字段（现金与送转）
NUMERIC_FIELDS: tuple[str, ...] = ("cash_div", "cash_div_tax", "stk_div")

#: 仅登记、不参与冲突判定的说明性字段
AUDIT_FIELDS: tuple[str, ...] = ("end_date", "record_date", "pay_date", "div_listdate")

RULE_ID = "conflict_isolated_drop_group"


def _text(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)
    return frame[name].astype(str).replace({"nan": "", "None": "", "NaT": ""}).fillna("")


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    """经济字段数值化：缺失/不可解析一律记 0（与 0.0 等价）。"""
    if name not in frame.columns:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    values = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    return values.round(6)


def resolve_frames(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """对一批分区做冲突隔离，返回（保留行的分区, 未决记录）。"""
    total = pd.concat(
        [frame.assign(_src=name) for name, frame in frames.items()], ignore_index=True
    )
    if total.empty:
        return frames, []
    total = total.copy()
    for name in NUMERIC_FIELDS:
        total[name] = _numeric(total, name)
    for name in AUDIT_FIELDS:
        total[name] = _text(total, name)
    total["_known"] = pd.concat(
        [_text(total, "ann_date"), _text(total, "imp_ann_date")], axis=1
    ).max(axis=1)
    total["_ex"] = _text(total, "ex_date")
    keys = ["ts_code", "_ex", "_known"]
    keep_mask = pd.Series(True, index=total.index)
    records: list[dict[str, Any]] = []
    for key, group in total.groupby(keys, sort=True):
        if len(group) < 2:
            continue
        signature = group[list(NUMERIC_FIELDS)].drop_duplicates()
        if len(signature) == 1:
            continue
        # fail-closed：整组剔除，不做任意挑选
        keep_mask.loc[group.index] = False
        records.append(
            {
                "rule": RULE_ID,
                "ts_code": str(key[0]),
                "ex_date": str(key[1]),
                "known_date": str(key[2]),
                "variants": int(len(signature)),
                "rows_dropped": int(len(group)),
                "combos": signature.to_dict("records"),
                "audit_fields": group[list(AUDIT_FIELDS)].drop_duplicates().to_dict("records"),
            }
        )
    resolved = total.loc[keep_mask]
    output: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        kept = resolved[resolved["_src"] == name].drop(
            columns=["_src", "_known", "_ex"], errors="ignore"
        )
        output[name] = kept.reset_index(drop=True)
    return output, records


def resolve_in_place(directory: Path, audit_path: Path | None = None) -> dict[str, Any]:
    """就地隔离分区目录中的冲突并落盘审计（只作用于研究根副本）。"""
    directory = Path(directory)
    frames = {path.name: pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))}
    resolved, records = resolve_frames(frames)
    for name, frame in resolved.items():
        frame.to_parquet(directory / name, index=False)
    audit = {
        "rule": RULE_ID,
        "scope": "研究根内分红副本；生产 data/raw/dividend 未修改",
        "numeric_fields": list(NUMERIC_FIELDS),
        "audit_fields": list(AUDIT_FIELDS),
        "partitions": len(frames),
        "rows_before": int(sum(len(frame) for frame in frames.values())),
        "rows_after": int(sum(len(frame) for frame in resolved.values())),
        "unresolved": records,
        "n_unresolved": len(records),
        "rows_dropped": int(sum(item["rows_dropped"] for item in records)),
    }
    target = Path(audit_path) if audit_path is not None else directory.parent / "resolutions.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return audit
