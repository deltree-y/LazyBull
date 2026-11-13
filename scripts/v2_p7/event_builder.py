# -*- coding: utf-8 -*-
"""P7 事件构建器（T1/T2 共用的只读纯事件构建器）。

契约（`docs/plans/v2/p7_design_20261111.md` §6.1）
=================================================
- 输入：`data/raw/{forecast,express,fina_indicator}` 分区，只读；
- 输出：内存中的七元组事件流，不写任何文件、不读运行期配置、不联网；
- 同一输入逐字节可复现（排序键与去重键全部显式，不依赖文件内行序）；
- T1 与 T2 必须绑定同一实现 SHA（:func:`event_builder_identity`）；T2 只把本模块的
  输出物化为事件资产，**不得另写 `ann_date`→`knowledge_date` 映射**。

七元组
======
``(event_source, ts_code, event_date, knowledge_date, revision_key,
predicate_value, fingerprint)``

- `event_source`：``forecast`` / ``express`` / ``fina_report``；
- `event_date`：报告期（`end_date`，fina 族沿用冻结映射）；
- `knowledge_date`：公告日（`ann_date`，PIT 唯一可得性字段）；
- `revision_key`：源内版本键，同键至多一个事件，修订版本全留；
- `predicate_value`：冻结谓词的取值（forecast 分值 / express 惊喜 / fina 净利同比）；
- `fingerprint`：七元组身份摘要（sha256，含源与上述字段）。

版本治理
========
- 只做**源内**版本治理：同源同 ``(ts_code, end_date, knowledge_date)`` 至多一个事件；
- 同日多版本按冻结规则择优：fina ``update_flag=1`` 优先，其余按全字段字典序取最后一行；
- 跨源不做报告期抢占（不同源可各自独立存在同一 `(ts_code, end_date)` 事件）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BUILDER_VERSION = "p7_event_builder/v1"

#: 事件源（顺序即 B0 `event_source` 升序决胜顺序）
EVENT_SOURCES: Tuple[str, ...] = ("forecast", "express", "fina_report")

#: forecast 类型分值（冻结口径；正值即正面事件）
FORECAST_TYPE_SCORE: Dict[str, float] = {
    "预增": 1.0,
    "略增": 0.5,
    "续盈": 0.3,
    "扭亏": 0.8,
    "不确定": 0.0,
    "略减": -0.5,
    "预减": -1.0,
    "首亏": -1.0,
    "续亏": -0.8,
}

EVENT_COLUMNS: Tuple[str, ...] = (
    "event_source",
    "ts_code",
    "event_date",
    "knowledge_date",
    "revision_key",
    "predicate_value",
    "fingerprint",
)

_FINGERPRINT_LEN = 64


@dataclass(frozen=True)
class BuilderIdentity:
    """事件构建器身份（路径 + 字节 SHA + 版本）。"""

    path: str
    sha256: str
    version: str

    def as_dict(self) -> Dict[str, str]:
        return {"path": self.path, "sha256": self.sha256, "version": self.version}


def event_builder_identity(module_path: Optional[Path] = None) -> BuilderIdentity:
    """返回本模块的字节身份；T1/T2 都必须绑定同一个 SHA。"""
    path = Path(module_path or __file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return BuilderIdentity(path=path.as_posix(), sha256=digest, version=BUILDER_VERSION)


def _normalize_date(values: pd.Series) -> pd.Series:
    """日期字段归一为 8 位字符串（非法/空值 → NaN）。"""
    text = values.astype(str).str.slice(0, 8)
    return text.where(text.str.fullmatch(r"\d{8}"), np.nan)


def _read_partitions(data_root: Path, name: str) -> pd.DataFrame:
    """按文件名升序读取某个 raw 数据集的全部分区。"""
    files = sorted((Path(data_root) / name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 raw 分区：{Path(data_root) / name}/*.parquet")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def _content_key(frame: pd.DataFrame) -> pd.Series:
    """全字段字典序键（按列顺序拼接），用于同键冲突的确定性决胜。"""
    columns = [column for column in frame.columns if not str(column).startswith("_")]
    if not columns:
        return pd.Series([""] * len(frame), index=frame.index)
    key = frame[columns[0]].astype(str)
    for column in columns[1:]:
        key = key + "\u241f" + frame[column].astype(str)
    return key


def _stable_sort(frame: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """按给定键排序，并以**全字段字典序**作最终决胜（与输入行序无关）。"""
    keyed = frame.assign(_content=_content_key(frame))
    ordered = keyed.sort_values(
        list(keys) + ["_content"], kind="mergesort", na_position="first"
    ).reset_index(drop=True)
    return ordered.drop(columns=["_content"])


def _dedup_same_day(frame: pd.DataFrame, prefer_updated: bool) -> pd.DataFrame:
    """同源同日去重：同 ``(ts_code, end_date, knowledge_date)`` 只留一行。

    决胜次序：身份键 → （fina）``update_flag=1`` 优先 → 全字段字典序。不依赖分区或行序。
    """
    keys = ["ts_code", "end_date", "knowledge_date"]
    ordered = frame
    if prefer_updated:
        ordered = ordered.assign(_updated=ordered["update_flag"].astype(str).str.strip().eq("1"))
        ordered = _stable_sort(ordered, keys + ["_updated"])
    else:
        ordered = _stable_sort(ordered, keys)
    ordered = ordered.drop_duplicates(subset=keys, keep="last")
    return ordered.drop(columns=[col for col in ("_updated",) if col in ordered.columns])


def _attach_identity(frame: pd.DataFrame, source: str, value_column: str) -> pd.DataFrame:
    """补齐七元组：revision key 与 fingerprint。"""
    code = frame["ts_code"].astype(str)
    end = frame["end_date"].astype(str)
    ann = frame["knowledge_date"].astype(str)
    if source == "fina_report":
        revision = code + "|" + end + "|" + ann + "|" + frame["update_flag"].astype(str).str.strip()
    else:
        revision = code + "|" + end + "|" + ann
    payload = pd.DataFrame(
        {
            "event_source": source,
            "ts_code": code,
            "event_date": end,
            "knowledge_date": ann,
            "revision_key": revision,
            "predicate_value": frame[value_column].astype(float),
        }
    )
    fingerprints = [
        hashlib.sha256(
            "|".join(
                [
                    row.event_source,
                    row.ts_code,
                    row.event_date,
                    row.knowledge_date,
                    row.revision_key,
                    f"{row.predicate_value:.10g}",
                ]
            ).encode("utf-8")
        ).hexdigest()
        for row in payload.itertuples(index=False)
    ]
    payload["fingerprint"] = fingerprints
    return payload.loc[:, list(EVENT_COLUMNS)]


def _forecast_frame(data_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = _read_partitions(data_root, "forecast")
    stats: Dict[str, Any] = {"rows_raw": int(len(raw))}
    frame = raw.loc[:, ["ts_code", "ann_date", "end_date", "type"]].copy()
    frame["knowledge_date"] = _normalize_date(frame["ann_date"])
    frame["end_date"] = _normalize_date(frame["end_date"])
    invalid = int(frame["knowledge_date"].isna().sum() + frame["end_date"].isna().sum())
    frame = frame.dropna(subset=["knowledge_date", "end_date"])
    frame["score"] = frame["type"].map(FORECAST_TYPE_SCORE)
    unmapped = int(frame["score"].isna().sum())
    stats.update({"invalid_date_rows": invalid, "unmapped_type_rows": unmapped})
    frame = frame.dropna(subset=["score"])
    frame = _dedup_same_day(frame, prefer_updated=False)
    events = frame[frame["score"] > 0].copy()
    stats.update({"rows_after_dedup": int(len(frame)), "events": int(len(events))})
    return _attach_identity(events, "forecast", "score"), stats


def _express_frame(data_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    express = _read_partitions(data_root, "express")
    forecast = _read_partitions(data_root, "forecast")
    stats: Dict[str, Any] = {"rows_raw": int(len(express))}

    express = express.loc[:, ["ts_code", "ann_date", "end_date", "yoy_net_profit"]].copy()
    express["knowledge_date"] = _normalize_date(express["ann_date"])
    express["end_date"] = _normalize_date(express["end_date"])
    invalid = int(express["knowledge_date"].isna().sum() + express["end_date"].isna().sum())
    stats["invalid_date_rows"] = invalid
    express = express.dropna(subset=["knowledge_date", "end_date"])
    express["yoy_net_profit"] = pd.to_numeric(express["yoy_net_profit"], errors="coerce")
    express = _dedup_same_day(express, prefer_updated=False)

    forecast = forecast.loc[:, ["ts_code", "ann_date", "end_date", "p_change_min", "p_change_max"]]
    forecast = forecast.copy()
    forecast["knowledge_date"] = _normalize_date(forecast["ann_date"])
    forecast["end_date"] = _normalize_date(forecast["end_date"])
    forecast = forecast.dropna(subset=["knowledge_date", "end_date"])
    for column in ("p_change_min", "p_change_max"):
        forecast[column] = pd.to_numeric(forecast[column], errors="coerce")
    forecast["chg_mid"] = (forecast["p_change_min"] + forecast["p_change_max"]) / 2
    forecast = forecast.dropna(subset=["chg_mid"])
    forecast = _dedup_same_day(forecast, prefer_updated=False)

    lookup: Dict[Tuple[str, str], Tuple[List[str], List[float]]] = {}
    for (code, end_date), group in forecast.groupby(["ts_code", "end_date"], sort=False):
        ordered = _stable_sort(group, ["knowledge_date"])
        lookup[(str(code), str(end_date))] = (
            ordered["knowledge_date"].astype(str).tolist(),
            ordered["chg_mid"].astype(float).tolist(),
        )

    surprises: List[float] = []
    for code, end_date, ann, profit_yoy in zip(
        express["ts_code"].astype(str),
        express["end_date"].astype(str),
        express["knowledge_date"].astype(str),
        express["yoy_net_profit"],
    ):
        hist = lookup.get((code, end_date))
        if hist is None or pd.isna(profit_yoy):
            surprises.append(float("nan"))
            continue
        ann_dates, values = hist
        position = int(np.searchsorted(np.asarray(ann_dates), ann, side="right")) - 1
        if position < 0:
            surprises.append(float("nan"))
            continue
        surprises.append(float(profit_yoy) - float(values[position]))
    express["express_surprise"] = surprises
    stats["missing_prior_forecast"] = int(np.isnan(np.asarray(surprises, dtype=float)).sum())
    events = express[express["express_surprise"] > 0].copy()
    stats.update({"row_after_dedup": int(len(express)), "events": int(len(events))})
    return _attach_identity(events, "express", "express_surprise"), stats


def _fina_frame(data_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = _read_partitions(data_root, "fina_indicator")
    stats: Dict[str, Any] = {"rows_raw": int(len(raw))}
    columns = ["ts_code", "ann_date", "end_date", "or_yoy", "netprofit_yoy", "update_flag"]
    frame = raw.loc[:, [c for c in columns if c in raw.columns]].copy()
    if "update_flag" not in frame.columns:
        frame["update_flag"] = ""
    frame["knowledge_date"] = _normalize_date(frame["ann_date"])
    frame["end_date"] = _normalize_date(frame["end_date"])
    invalid = int(frame["knowledge_date"].isna().sum() + frame["end_date"].isna().sum())
    frame = frame.dropna(subset=["knowledge_date", "end_date"])
    for column in ("or_yoy", "netprofit_yoy"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = _dedup_same_day(frame, prefer_updated=True)
    events = frame[(frame["or_yoy"] > 0) & (frame["netprofit_yoy"] > 0)].copy()
    stats.update({"invalid_date_rows": invalid, "rows_after_dedup": int(len(frame))})
    stats["events"] = int(len(events))
    return _attach_identity(events, "fina_report", "netprofit_yoy"), stats


def build_events(
    data_root: Path,
    cutoff: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """构建三源事件流（只读纯函数）。

    Args:
        data_root: `data/raw` 根目录（其下为 `forecast/` `express/` `fina_indicator/`）。
        cutoff: 可选共同数据截止（`knowledge_date <= cutoff` 才返回）。

    Returns:
        ``(events, stats)``：每个源一个七元组 DataFrame（按 knowledge_date/ts_code/
        fingerprint 升序），以及逐源行数与无效行统计。
    """
    root = Path(data_root)
    events: Dict[str, pd.DataFrame] = {}
    stats: Dict[str, Any] = {"builder_version": BUILDER_VERSION, "sources": {}}
    for source, builder in (
        ("forecast", _forecast_frame),
        ("express", _express_frame),
        ("fina_report", _fina_frame),
    ):
        frame, source_stats = builder(root)
        if cutoff is not None:
            before = len(frame)
            frame = frame[frame["knowledge_date"] <= str(cutoff)]
            source_stats["dropped_after_cutoff"] = int(before - len(frame))
        frame = frame.sort_values(
            ["knowledge_date", "ts_code", "fingerprint"], kind="mergesort"
        ).reset_index(drop=True)
        events[source] = frame.loc[:, list(EVENT_COLUMNS)]
        source_stats["events_in_domain"] = int(len(frame))
        stats["sources"][source] = source_stats
    stats["total_events_in_domain"] = int(sum(len(frame) for frame in events.values()))
    return events, stats


def build_announcement_index(
    data_root: Path,
    cutoff: Optional[str] = None,
) -> pd.DataFrame:
    """全部公告记录（不限谓词方向）的 `(event_source, ts_code, knowledge_date)`。

    对照池的冻结定义是「当日没有任何 forecast/express/fina 公告记录」，因此这里必须
    保留**全部**合法行（含负面/中性/未映射类型），不能用事件谓词再过滤一次。
    """
    root = Path(data_root)
    rows: List[pd.DataFrame] = []
    for source, name in (
        ("forecast", "forecast"),
        ("express", "express"),
        ("fina_report", "fina_indicator"),
    ):
        raw = _read_partitions(root, name)
        frame = pd.DataFrame(
            {
                "event_source": source,
                "ts_code": raw["ts_code"].astype(str),
                "knowledge_date": _normalize_date(raw["ann_date"]),
            }
        )
        rows.append(frame.dropna(subset=["knowledge_date"]))
    merged = pd.concat(rows, ignore_index=True)
    if cutoff is not None:
        merged = merged[merged["knowledge_date"] <= str(cutoff)]
    merged = merged.drop_duplicates(subset=["ts_code", "knowledge_date"]).reset_index(drop=True)
    return merged.sort_values(["knowledge_date", "ts_code"], kind="mergesort").reset_index(
        drop=True
    )


def scan_fina_watermark(data_root: Path) -> Dict[str, Any]:
    """fina 水位推导：全部 raw 分区只读扫描（无 `_sync_watermark.json`）。

    记录逐文件行数/SHA/max(ann_date)、无效日期行数与全族最大 `ann_date`。
    """
    root = Path(data_root)
    files = sorted((root / "fina_indicator").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 fina 分区：{root / 'fina_indicator'}")
    partitions: List[Dict[str, Any]] = []
    total_rows = 0
    invalid_rows = 0
    max_ann = ""
    min_ann = ""
    max_end = ""
    for path in files:
        frame = pd.read_parquet(path, columns=["ts_code", "ann_date", "end_date"])
        total_rows += len(frame)
        ann = _normalize_date(frame["ann_date"])
        end = _normalize_date(frame["end_date"])
        invalid_rows += int(ann.isna().sum())
        valid = ann.dropna()
        partition_max = str(valid.max()) if not valid.empty else ""
        if partition_max:
            max_ann = max(max_ann, partition_max)
            min_ann = str(valid.min()) if not min_ann else min(min_ann, str(valid.min()))
        valid_end = end.dropna()
        if not valid_end.empty:
            max_end = max(max_end, str(valid_end.max()))
        partitions.append(
            {
                "file": path.name,
                "rows": int(len(frame)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "max_ann_date": partition_max or None,
            }
        )
    return {
        "api": "fina_indicator",
        "partition_count": len(files),
        "rows": int(total_rows),
        "invalid_ann_date_rows": int(invalid_rows),
        "min_ann_date": min_ann or None,
        "max_ann_date": max_ann or None,
        "max_end_date": max_end or None,
        "watermark_rule": "对全部 raw 分区只读扫描有效 ann_date 取最大值",
        "partitions": partitions,
    }


def events_frame_to_json_bytes(frame: pd.DataFrame) -> bytes:
    """事件流规范化 JSON（按行导出，供证据根逐源留档）。"""
    payload = frame.to_dict(orient="records")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
