"""分红冲突实施版本的窄口径消解（用户裁决 B，2026-11-18）。

触发条件（与 `buffer_native_corporate` 的冲突判定一致）：
同一 `(ts_code, ex_date)` 在**同一可知时点** `known_date = max(ann_date, imp_ann_date)`
下存在多于一个不同的经济字段组合。

消解规则（只处理这一种冲突，不外推）：
1. 保留 `end_date`（报告期）最晚者；
2. 仍并列时保留 `pay_date` 最晚者；
3. 仍并列时保留 `div_listdate` 最晚者；
4. 仍并列时按经济字段元组的字符串序取最大者（确定性，不依赖行序）。

每次消解都写入 `resolutions.json`（append-only 覆盖写，内容为全量记录），
报告必须原样引用；未命中冲突的行一律不动。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ECONOMIC_FIELDS = (
    "end_date",
    "record_date",
    "pay_date",
    "stk_div",
    "cash_div",
    "cash_div_tax",
    "div_listdate",
)

RULE_ID = "conflict_latest_report_period_then_pay_then_list"


def _text(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)
    return frame[name].astype(str).replace({"nan": "", "None": "", "NaT": ""}).fillna("")


def resolve_frames(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """对一批分区做冲突消解，返回（保留行的分区, 消解记录）。"""
    total = pd.concat(
        [frame.assign(_src=name) for name, frame in frames.items()], ignore_index=True
    )
    if total.empty:
        return frames, []
    total = total.copy()
    for name in ECONOMIC_FIELDS:
        if name not in total.columns:
            total[name] = ""
        total[name] = _text(total, name)
    known = pd.concat(
        [_text(total, "ann_date"), _text(total, "imp_ann_date")], axis=1
    ).max(axis=1)
    total["_known"] = known
    total["_ex"] = _text(total, "ex_date")
    keys = ["ts_code", "_ex", "_known"]
    keep_mask = pd.Series(True, index=total.index)
    records: list[dict[str, Any]] = []
    for key, group in total.groupby(keys, sort=True):
        if len(group) < 2:
            continue
        signature = group[list(ECONOMIC_FIELDS)].drop_duplicates()
        if len(signature) == 1:
            continue
        ordered = group.sort_values(
            by=["end_date", "pay_date", "div_listdate"], ascending=True, kind="mergesort"
        )
        tuples = ordered[list(ECONOMIC_FIELDS)].agg("|".join, axis=1)
        chosen_index = tuples.idxmax()
        dropped = [idx for idx in group.index if idx != chosen_index]
        keep_mask.loc[dropped] = False
        records.append(
            {
                "rule": RULE_ID,
                "ts_code": str(key[0]),
                "ex_date": str(key[1]),
                "known_date": str(key[2]),
                "variants": int(len(signature)),
                "kept": ordered.loc[chosen_index, list(ECONOMIC_FIELDS)].to_dict(),
                "dropped": ordered.loc[dropped, list(ECONOMIC_FIELDS)].to_dict("records"),
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


def resolve_in_place(directory: Path) -> dict[str, Any]:
    """就地消解分区目录并落盘审计。"""
    directory = Path(directory)
    frames = {
        path.name: pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))
    }
    resolved, records = resolve_frames(frames)
    for name, frame in resolved.items():
        frame.to_parquet(directory / name, index=False)
    audit = {
        "rule": RULE_ID,
        "authorization": "specification/authorization_conflict_rule_20261118.json",
        "economic_fields": list(ECONOMIC_FIELDS),
        "partitions": len(frames),
        "resolutions": records,
        "n_resolutions": len(records),
    }
    (directory.parent / "resolutions.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return audit
