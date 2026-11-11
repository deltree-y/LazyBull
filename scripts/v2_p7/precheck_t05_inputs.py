# -*- coding: utf-8 -*-
"""P7 T0.5 输入完整性预检（只读、只计数、不判 alpha、不消耗 α）。

用途
====
对应 `docs/plans/v2/p7_implementation_plan_20261111.md` §3 产出 8：

1. A 锚 14 折 `test_start/test_end` 逐值核对，并另存 `bt_start/bt_end` 差异表与 SHA；
2. 锚模型（24008~24021）与特征清单的身份（SHA/列数）复核；
3. 逐折 test 信号域的面板覆盖与相位日历可构造性；
4. 三源（forecast/express/fina）事件日分布、guard 后可构造
   `effective_history_block` 数与"块—年"换算（Q5=A 的 fina 核心格预检）。

约束
====
- 只读输入，不写 raw/clean/features/models/runs，不运行训练或推理，不生成经济 verdict；
- 只输出到调用方指定的预检目录（默认落 P7 测试床）；
- 事件谓词与 B0 候选定义一致（`docs/plans/v2/p7_design_20261111.md` §5.2）：
  forecast `type` 分值为正、express `express_surprise>0`、fina `or_yoy>0 且 netprofit_yoy>0`；
- 计数口径只证明"可构造性"，不声称已存在可解析匹配组（那是 T1 的结论）。

用法
====
``python scripts/v2_p7/precheck_t05_inputs.py``
``python scripts/v2_p7/precheck_t05_inputs.py --out-dir <governance_dir>/precheck``
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    WriteScopeError,
    assert_targets_within_write_roots,
    resolve_write_roots,
    sha256_file as _sha256_file,
    write_json as _write_json,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = _REPO_ROOT / "data"
ANCHOR_RUN_REL = "data/runs/p2a_b0replay_20261008_144307_d79f3f"
MANIFEST_REL = "data/frozen_reference/p2a_b0_input_manifest_20261008_withconfig.json"
EVIDENCE_ID = "p7_t05_20261111_r6"
DEFAULT_OUT_REL = f"data/paper_shadow/p7_testbed/{EVIDENCE_ID}/precheck"

# 与 docs/plans/v2/p7_implementation_plan_20261111.md §2.1 冻结表逐值核对
FROZEN_FOLDS: List[Tuple[int, int, str, str, str, str]] = [
    (0, 24008, "20121123", "20181123", "20181126", "20190524"),
    (1, 24009, "20130524", "20190524", "20190527", "20191125"),
    (2, 24010, "20131125", "20191125", "20191126", "20200526"),
    (3, 24011, "20140526", "20200526", "20200527", "20201127"),
    (4, 24012, "20141127", "20201127", "20201130", "20210528"),
    (5, 24013, "20150528", "20210528", "20210531", "20211130"),
    (6, 24014, "20151130", "20211130", "20211201", "20220601"),
    (7, 24015, "20160601", "20220601", "20220602", "20221202"),
    (8, 24016, "20161202", "20221202", "20221205", "20230605"),
    (9, 24017, "20170605", "20230605", "20230606", "20231206"),
    (10, 24018, "20171206", "20231206", "20231207", "20240607"),
    (11, 24019, "20180607", "20240607", "20240611", "20241210"),
    (12, 24020, "20181210", "20241210", "20241211", "20250611"),
    (13, 24021, "20190611", "20250611", "20250612", "20260105"),
]

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

# 评价块与 guard（Q5=A 冻结口径）
BLOCK_DAYS = 20
GUARD_DAYS = 10
CORE_HORIZON = 10
SENS_HORIZON = 5
ENTRY_MAX_DAYS = 10
EXIT_MAX_DAYS = 10
MIN_EFFECTIVE_BLOCKS = 60
TRADING_DAYS_PER_YEAR = 250.0


def truncate_calendar(calendar: Sequence[str], cutoff: str) -> Tuple[List[str], Dict[str, Any]]:
    """把交易日历截到共同数据截止（含），并登记被截掉的尾部。

    可构造性必须以**数据可得性**为界，日历存在性不能替代行情观察（R1-T05-01）。
    """
    bounded = [day for day in calendar if day <= cutoff]
    if not bounded:
        raise ValueError(f"共同截止 {cutoff} 之前没有交易日")
    info = {
        "calendar_last_date": calendar[-1],
        "calendar_trading_days": len(calendar),
        "bounded_trading_days": len(bounded),
        "dropped_trading_days": len(calendar) - len(bounded),
        "bounded_last_date": bounded[-1],
        "truncated_at_cutoff": True,
    }
    return bounded, info


def _load_trading_days() -> List[str]:
    """全市场交易日（SSE 开市日，YYYYMMDD 升序）。"""
    cal = pd.read_parquet(DATA / "raw" / "trade_cal.parquet")
    cal = cal[cal["exchange"] == "SSE"]
    cal = cal[cal["is_open"] == 1]
    days = sorted(str(int(v)) for v in cal["cal_date"])
    return days


def _read_partitions(pattern: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """按字典序读取一组 parquet 分区并纵向拼接。"""
    files = sorted(DATA.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到分区：{pattern}")
    frames = [pd.read_parquet(path, columns=list(columns) if columns else None) for path in files]
    return pd.concat(frames, ignore_index=True)


def check_folds() -> Dict[str, Any]:
    """14 折 test 域逐值核对，并输出 test/bt 差异表。"""
    summary_path = _REPO_ROOT / ANCHOR_RUN_REL / "summary.parquet"
    frame = pd.read_parquet(summary_path).sort_values("split_index").reset_index(drop=True)
    if len(frame) != len(FROZEN_FOLDS):
        raise ValueError(f"锚批次折数 {len(frame)} 与冻结表 {len(FROZEN_FOLDS)} 不一致")

    rows: List[Dict[str, Any]] = []
    mismatch: List[str] = []
    for (split, model, train_start, train_end, test_start, test_end), row in zip(
        FROZEN_FOLDS, frame.itertuples(index=False)
    ):
        actual = {
            "split_index": int(row.split_index),
            "model_version": int(row.model_version),
            "train_start": str(int(row.train_start)),
            "train_end": str(int(row.train_end)),
            "test_start": str(int(row.test_start)),
            "test_end": str(int(row.test_end)),
            "bt_start": str(int(row.bt_start)),
            "bt_end": str(int(row.bt_end)),
            "bt_trading_days": int(row.bt_trading_days),
            "data_state_id": str(row.data_state_id),
            "wf_run_id": str(row.wf_run_id),
        }
        frozen = {
            "split_index": split,
            "model_version": model,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        }
        for key, expected in frozen.items():
            if actual[key] != expected:
                mismatch.append(f"split={split} {key}: actual={actual[key]} frozen={expected}")
        rows.append(
            {
                **actual,
                "frozen_test_start": test_start,
                "frozen_test_end": test_end,
                "test_bt_start_diff_days": _cal_gap(actual["test_start"], actual["bt_start"]),
                "test_bt_end_diff_days": _cal_gap(actual["test_end"], actual["bt_end"]),
                "bt_extends_test_start": actual["bt_start"] < actual["test_start"],
                "bt_extends_test_end": actual["bt_end"] > actual["test_end"],
            }
        )

    payload = {
        "anchor_run": ANCHOR_RUN_REL,
        "summary_parquet_sha256": _sha256_file(summary_path),
        "summary_rows": int(len(frame)),
        "frozen_folds_match": not mismatch,
        "mismatch": mismatch,
        "fold_count": len(rows),
        "data_state_ids": sorted({row["data_state_id"] for row in rows}),
        "wf_run_ids": sorted({row["wf_run_id"] for row in rows}),
        "model_versions": [row["model_version"] for row in rows],
        "test_domain_contiguous": _test_domain_contiguous(rows),
        "rows": rows,
    }
    return payload


def _cal_gap(start: str, end: str) -> int:
    """两个 YYYYMMDD 的自然日差（end - start）。"""
    delta = pd.Timestamp(end) - pd.Timestamp(start)
    return int(delta.days)


def _test_domain_contiguous(rows: Sequence[Dict[str, Any]]) -> bool:
    """相邻折 test 域是否无缺口（后折起点紧跟前折终点）。"""
    for prev, cur in zip(rows, rows[1:]):
        if cur["test_start"] <= prev["test_end"]:
            return False
    return True


def check_models() -> Dict[str, Any]:
    """锚模型与特征清单身份：逐文件 SHA 对冻结清单比对。"""
    manifest = json.loads((_REPO_ROOT / MANIFEST_REL).read_text(encoding="utf-8"))
    entries: Dict[str, Any] = manifest["files"]
    rows: List[Dict[str, Any]] = []
    mismatch: List[str] = []
    feature_columns: Dict[str, int] = {}
    for _, model, *_ in FROZEN_FOLDS:
        for suffix in ("model.joblib", "metadata.json", "features.json"):
            key = f"models/stock_selection/v{model}_{suffix}"
            expected = entries.get(key)
            path = DATA / key
            if expected is None:
                mismatch.append(f"清单缺少 {key}")
                continue
            if not path.exists():
                mismatch.append(f"文件缺失 {key}")
                continue
            actual_sha = _sha256_file(path)
            actual_size = path.stat().st_size
            if actual_sha != expected["sha256"] or actual_size != int(expected["size"]):
                mismatch.append(f"SHA/大小不符 {key}")
            if suffix == "features.json":
                feature_columns[str(model)] = len(json.loads(path.read_text(encoding="utf-8")))
            rows.append(
                {
                    "model_version": model,
                    "file": key,
                    "sha256": actual_sha,
                    "size": actual_size,
                    "matches_manifest": actual_sha == expected["sha256"],
                }
            )

    payload = {
        "manifest": MANIFEST_REL,
        "manifest_sha256": _sha256_file(_REPO_ROOT / MANIFEST_REL),
        "manifest_created_at": manifest.get("created_at"),
        "manifest_window": manifest.get("window"),
        "manifest_model_versions": manifest.get("model_versions"),
        "files_checked": len(rows),
        "identity_match": not mismatch,
        "mismatch": mismatch,
        "feature_columns_by_model": feature_columns,
        "feature_columns_note": "逐折特征清单随 WF 折变化，属既有设计；此处只登记每折列数，不作一致性断言",
        "rows": rows,
    }
    return payload


def check_panel(trading_days: Sequence[str]) -> Dict[str, Any]:
    """逐折 test 信号域的特征面板覆盖与相位日历可构造性。"""
    index = {day: i for i, day in enumerate(trading_days)}
    train_files = sorted(path.stem for path in (DATA / "features" / "cs_train").glob("*.parquet"))
    infer_files = sorted(path.stem for path in (DATA / "features" / "cs_infer").glob("*.parquet"))
    available = set(train_files) | set(infer_files)

    rows: List[Dict[str, Any]] = []
    total_missing = 0
    for split, model, _, _, test_start, test_end in FROZEN_FOLDS:
        start = index[test_start]
        end = index[test_end]
        stop = end + 1
        expected = trading_days[start:stop]
        missing = [day for day in expected if day not in available]
        total_missing += len(missing)
        rows.append(
            {
                "split_index": split,
                "model_version": model,
                "test_start": test_start,
                "test_end": test_end,
                "expected_signal_days": len(expected),
                "missing_signal_days": len(missing),
                "missing_examples": missing[:5],
            }
        )

    payload = {
        "cs_train_partitions": len(train_files),
        "cs_train_range": [train_files[0], train_files[-1]] if train_files else None,
        "cs_infer_partitions": len(infer_files),
        "cs_infer_range": [infer_files[0], infer_files[-1]] if infer_files else None,
        "signal_days_missing_total": total_missing,
        "signal_days_complete": total_missing == 0,
        "rows": rows,
    }
    return payload


def check_phase_calendar(trading_days: Sequence[str]) -> Dict[str, Any]:
    """V20/V10 相位日历与末折 runoff 可构造性（R1-T05-06 修正）。

    冻结语义（设计 §7.1）：V10 = 同一信号、同一 20 日持有与 sizing，仅把**两档**建仓/
    再平衡日历整体错后 10 个交易日；因此
    V20 档1 = 0,20,40…、档2 = 10,30,50…；V10 档1 = 10,30,50…、档2 = 20,40,60…
    任何「绕回起点」的构造都使相位身份错误，必须由独立的平移集合比对检出。
    """
    index = {day: i for i, day in enumerate(trading_days)}
    union_start = index[FROZEN_FOLDS[0][4]]
    union_end = index[FROZEN_FOLDS[-1][5]]
    union_stop = union_end + 1
    horizon_days = trading_days[union_start:union_stop]
    phases = derive_phase_tranches(horizon_days)
    offset = phases["offset"]
    v20_t1, v20_t2 = phases["v20_t1"], phases["v20_t2"]
    v10_t1, v10_t2 = phases["v10_t1"], phases["v10_t2"]
    v20_union, v10_union = phases["v20_union"], phases["v10_union"]
    v10_positions = phases["v10_positions"]
    runoff_days, runoff_complete = runoff_window(trading_days, union_end)

    def _tranche(days: Sequence[str]) -> Dict[str, Any]:
        return {
            "count": len(days),
            "first": days[0] if days else None,
            "last": days[-1] if days else None,
        }

    payload = {
        "signal_domain": [FROZEN_FOLDS[0][4], FROZEN_FOLDS[-1][5]],
        "signal_trading_days": len(horizon_days),
        "holding_days": BLOCK_DAYS,
        "stagger_tranches": 2,
        "phase_offset_trading_days": offset,
        "tranches": {
            "v20_t1": _tranche(v20_t1),
            "v20_t2": _tranche(v20_t2),
            "v10_t1": _tranche(v10_t1),
            "v10_t2": _tranche(v10_t2),
        },
        "v20_rebalance_dates": len(v20_union),
        "v20_first_last": [v20_union[0], v20_union[-1]],
        "v10_rebalance_dates": len(v10_union),
        "v10_first_last": [v10_union[0], v10_union[-1]],
        "v10_earliest_position": v10_positions[0] if v10_positions else None,
        "v10_shifted_matches_v20": phases["shifted_matches"],
        "v10_no_wraparound": phases["no_wraparound"],
        "phase_offset_matches": phases["shifted_matches"] and phases["no_wraparound"],
        "runoff_start": runoff_days[0] if runoff_days else None,
        "runoff_end": runoff_days[-1] if runoff_days else None,
        "runoff_trading_days": len(runoff_days),
        "runoff_required_trading_days": BLOCK_DAYS,
        "runoff_complete": runoff_complete,
        "runoff_within_calendar": runoff_complete,
    }
    return payload


def check_watermarks() -> Dict[str, Any]:
    """三源水位与 clean 水位：forecast/express 读水位文件，fina 只读扫描推导。"""
    sources: Dict[str, Any] = {}
    for name in ("forecast", "express"):
        path = DATA / "raw" / name / "_sync_watermark.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources[name] = {
            "watermark_file": str(path.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "watermark_file_sha256": _sha256_file(path),
            "synced_to": str(payload.get("synced_to")),
        }

    fina_files = sorted((DATA / "raw" / "fina_indicator").glob("*.parquet"))
    fina_rows = 0
    fina_invalid = 0
    fina_max_ann = ""
    fina_min_ann = ""
    fina_period_max = ""
    file_shas: List[Dict[str, Any]] = []
    ann_dates: List[str] = []
    for path in fina_files:
        frame = pd.read_parquet(path, columns=["ts_code", "ann_date", "end_date"])
        fina_rows += len(frame)
        ann = frame["ann_date"].astype(str).str.slice(0, 8)
        valid = ann.str.fullmatch(r"\d{8}") & ann.str.startswith("20")
        fina_invalid += int((~valid).sum())
        clean_ann = ann[valid]
        if not clean_ann.empty:
            ann_dates.append(clean_ann.max())
        period = frame["end_date"].astype(str).str.slice(0, 8)
        periods = period.tolist()
        fina_period_max = max([fina_period_max, *periods]) if periods else fina_period_max
        file_shas.append(
            {
                "file": path.name,
                "rows": int(len(frame)),
                "sha256": _sha256_file(path),
                "max_ann_date": clean_ann.max() if not clean_ann.empty else None,
            }
        )
    if ann_dates:
        fina_max_ann = max(ann_dates)
        fina_min_ann = min(ann_dates)

    sources["fina_indicator"] = {
        "partition_count": len(fina_files),
        "rows": fina_rows,
        "invalid_ann_date_rows": fina_invalid,
        "min_ann_date": fina_min_ann,
        "max_ann_date": fina_max_ann,
        "max_end_date": fina_period_max,
        "watermark_source": "raw 分区只读扫描（无 _sync_watermark.json）",
        "partitions": file_shas,
    }

    clean_files = sorted((DATA / "clean" / "daily").glob("*.parquet"))
    basic_files = sorted((DATA / "clean" / "daily_basic").glob("*.parquet"))
    sources["clean_daily"] = {
        "partition_count": len(clean_files),
        "range": [clean_files[0].stem.replace("-", ""), clean_files[-1].stem.replace("-", "")],
        "daily_basic_range": (
            [
                basic_files[0].stem.replace("-", ""),
                basic_files[-1].stem.replace("-", ""),
            ]
            if basic_files
            else None
        ),
    }

    cutoffs = {
        "forecast": sources["forecast"]["synced_to"],
        "express": sources["express"]["synced_to"],
        "fina_indicator": sources["fina_indicator"]["max_ann_date"],
        "clean_daily": sources["clean_daily"]["range"][1],
    }
    common_cutoff = min(value for value in cutoffs.values() if value)

    payload = {
        "sources": sources,
        "cutoffs": cutoffs,
        "common_data_cutoff": common_cutoff,
        "fina_watermark_derivable": bool(sources["fina_indicator"]["max_ann_date"]),
    }
    return payload


def _forecast_events() -> pd.DataFrame:
    """forecast 正面事件：type 分值 > 0，主键 (ts_code,end_date,ann_date)。"""
    frame = _read_partitions(
        "raw/forecast/*.parquet", columns=["ts_code", "ann_date", "end_date", "type"]
    )
    frame["ann_date"] = frame["ann_date"].astype(str).str.slice(0, 8)
    frame["end_date"] = frame["end_date"].astype(str).str.slice(0, 8)
    frame = frame.dropna(subset=["ann_date"])
    frame["score"] = frame["type"].map(FORECAST_TYPE_SCORE)
    frame = frame[frame["score"] > 0]
    frame = frame.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")
    return frame[["ts_code", "end_date", "ann_date"]]


def _express_events() -> pd.DataFrame:
    """express 正面事件：express_surprise > 0（对同一报告期最近一次预告中值）。

    与 `src/lazybull/factors/express.py` 的口径一致：PIT 只用 ann_date <= 快报公告日。
    """
    express = _read_partitions(
        "raw/express/*.parquet",
        columns=["ts_code", "ann_date", "end_date", "yoy_net_profit"],
    )
    forecast = _read_partitions(
        "raw/forecast/*.parquet",
        columns=["ts_code", "ann_date", "end_date", "p_change_min", "p_change_max"],
    )
    for frame in (express, forecast):
        frame["ann_date"] = frame["ann_date"].astype(str).str.slice(0, 8)
        frame["end_date"] = frame["end_date"].astype(str).str.slice(0, 8)
    express = express.dropna(subset=["ann_date"])
    express["yoy_net_profit"] = pd.to_numeric(express["yoy_net_profit"], errors="coerce")
    express = express.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")

    forecast = forecast.dropna(subset=["ann_date"])
    for col in ("p_change_min", "p_change_max"):
        forecast[col] = pd.to_numeric(forecast[col], errors="coerce")
    forecast["chg_mid"] = (forecast["p_change_min"] + forecast["p_change_max"]) / 2
    forecast = forecast.dropna(subset=["chg_mid"])
    forecast = forecast.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")

    lookup: Dict[Tuple[str, str], Tuple[List[str], List[float]]] = {}
    for (ts_code, end_date), group in forecast.groupby(["ts_code", "end_date"], sort=False):
        ordered = group.sort_values("ann_date")
        lookup[(ts_code, end_date)] = (
            ordered["ann_date"].tolist(),
            ordered["chg_mid"].astype(float).tolist(),
        )

    surprises: List[float] = []
    for ts_code, end_date, ann_date, profit_yoy in zip(
        express["ts_code"], express["end_date"], express["ann_date"], express["yoy_net_profit"]
    ):
        hist = lookup.get((ts_code, end_date))
        if hist is None or pd.isna(profit_yoy):
            surprises.append(np.nan)
            continue
        ann_dates, values = hist
        pos = bisect.bisect_right(ann_dates, ann_date) - 1
        if pos < 0:
            surprises.append(np.nan)
            continue
        surprises.append(float(profit_yoy) - float(values[pos]))
    express["express_surprise"] = surprises
    events = express[express["express_surprise"] > 0]
    return events[["ts_code", "end_date", "ann_date"]]


def _fina_events() -> pd.DataFrame:
    """fina_report 正面事件：or_yoy>0 且 netprofit_yoy>0，同日修正版优先。"""
    frame = _read_partitions(
        "raw/fina_indicator/*.parquet",
        columns=["ts_code", "ann_date", "end_date", "or_yoy", "netprofit_yoy", "update_flag"],
    )
    frame["ann_date"] = frame["ann_date"].astype(str).str.slice(0, 8)
    frame["end_date"] = frame["end_date"].astype(str).str.slice(0, 8)
    frame = frame.dropna(subset=["ann_date"])
    for col in ("or_yoy", "netprofit_yoy"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["_updated"] = frame["update_flag"].astype(str).str.strip().eq("1")
    frame = frame.sort_values(["ts_code", "end_date", "ann_date", "_updated"], kind="mergesort")
    frame = frame.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")
    events = frame[(frame["or_yoy"] > 0) & (frame["netprofit_yoy"] > 0)]
    return events[["ts_code", "end_date", "ann_date"]]


def _block_table(
    signal_positions: np.ndarray,
    n_days: int,
    horizon: int,
    block_days: int = BLOCK_DAYS,
    guard_days: int = GUARD_DAYS,
) -> Dict[str, Any]:
    """按 20 日评价块 + 10 日 guard 统计可构造有效块（T0.5 只计数）。

    冻结口径（与 T1 实现一致，只是 T0.5 不含匹配组可解析性判断）：
    - 归块键 = **计划入场日**（信号日 T 的下一交易日 T+1），而非信号日；这与 B0 "T 日信号、
      T+1 入场" 的执行口径一致，也令归块与真实成交日同键；
    - 落在 guard 的入场日不归属任何块（仍计入事件总量），因此相邻块的持有窗口不重叠：
      guard 宽度 = 持有期 h（核心 10 日、敏感性 5 日），入场日最晚在本块最后一日，
      到期日落在本块+guard 内，绝不跨入下一块；
    - `effective` 块 = 至少一个归块入场且其到期日（入场 + h）在数据截止内；
    - 右删失（到期日超过截止）只单列，不算有效块，不改写为 0 收益。
    """
    if guard_days < horizon:
        raise ValueError(f"guard({guard_days}) 必须不小于持期({horizon})")
    block_rows: List[Dict[str, Any]] = []
    assigned = 0
    cursor = 0
    block_index = 0
    while cursor + block_days <= n_days:
        block_end = cursor + block_days - 1
        guard_end = min(block_end + guard_days, n_days - 1)
        inside = signal_positions[(signal_positions >= cursor) & (signal_positions <= block_end)]
        mature = 0
        censored = 0
        for position in inside:
            if int(position) + horizon <= n_days - 1:
                mature += 1
            else:
                censored += 1
        assigned += len(inside)
        block_rows.append(
            {
                "block_index": block_index,
                "start_position": cursor,
                "end_position": block_end,
                "guard_positions": [block_end + 1, guard_end],
                "events": int(len(inside)),
                "unique_entry_days": int(len(np.unique(inside))),
                "events_matured": mature,
                "events_right_censored": censored,
                "effective": mature > 0,
            }
        )
        cursor = block_end + 1 + guard_days
        block_index += 1

    effective = sum(1 for row in block_rows if row["effective"])
    years = n_days / TRADING_DAYS_PER_YEAR
    blocks_per_year = TRADING_DAYS_PER_YEAR / (block_days + guard_days)
    in_guard = int(len(signal_positions) - assigned)
    # 右删失按**全部有入场日的事件**统计（含 guard 与末块之外），
    # 与块内计数分开：块内 events_right_censored 只描述该块，不重复计入总量。
    right_censored = int(np.sum(signal_positions + horizon > n_days - 1))
    return {
        "horizon": horizon,
        "block_days": block_days,
        "guard_days": guard_days,
        "blocks_total": len(block_rows),
        "effective_blocks": effective,
        "block_years": round(effective / blocks_per_year, 3),
        "timeline_trading_days": n_days,
        "timeline_years": round(years, 3),
        "blocks_per_year": round(blocks_per_year, 4),
        "min_effective_blocks": MIN_EFFECTIVE_BLOCKS,
        "gate_pass": effective >= MIN_EFFECTIVE_BLOCKS,
        "events_assigned_to_blocks": int(assigned),
        "events_matured_in_blocks": int(sum(row["events_matured"] for row in block_rows)),
        "events_in_guard_or_outside": in_guard,
        "events_right_censored": right_censored,
        "rows": block_rows,
    }


def entry_positions_within(signal_positions: Sequence[int], n_days: int) -> Tuple[List[int], int]:
    """计划入场位置（T+1）；超出已观察区间的信号单独计数（右删失入口）。"""
    entries = [int(position) + 1 for position in signal_positions]
    valid = [entry for entry in entries if entry <= n_days - 1]
    return valid, len(entries) - len(valid)


def derive_phase_tranches(
    horizon_days: Sequence[str], block_days: int = BLOCK_DAYS
) -> Dict[str, Any]:
    """V20/V10 两档相位日历（V10 = V20 两档整体后移 block_days/2 个交易日）。"""
    offset = block_days // 2
    v20_t1 = list(horizon_days[0::block_days])
    v20_t2 = list(horizon_days[offset::block_days])
    v10_t1 = list(horizon_days[offset::block_days])
    v10_t2 = list(horizon_days[block_days::block_days])
    v20_union = sorted(set(v20_t1) | set(v20_t2))
    v10_union = sorted(set(v10_t1) | set(v10_t2))
    position_of = {day: position for position, day in enumerate(horizon_days)}
    expected_v10 = sorted(
        {
            horizon_days[position_of[day] + offset]
            for day in v20_union
            if position_of[day] + offset < len(horizon_days)
        }
    )
    v10_positions = sorted({position_of[day] for day in v10_union})
    return {
        "offset": offset,
        "v20_t1": v20_t1,
        "v20_t2": v20_t2,
        "v10_t1": v10_t1,
        "v10_t2": v10_t2,
        "v20_union": v20_union,
        "v10_union": v10_union,
        "v10_positions": v10_positions,
        "shifted_matches": v10_union == expected_v10,
        "no_wraparound": bool(v10_positions) and v10_positions[0] >= offset,
    }


def runoff_window(trading_days: Sequence[str], union_end: int, block_days: int = BLOCK_DAYS):
    """末折 runoff：必须恰好覆盖 block_days 个交易日，否则记不完整（不得由 min 保证通过）。"""
    start = union_end + 1
    stop = start + block_days
    days = list(trading_days[start:stop])
    return days, len(days) == block_days


def analyze_events(trading_days: Sequence[str], cutoff: str) -> Dict[str, Any]:
    """三源事件分布与有效块计数（核心 10 日 / 敏感性 5 日）。

    ``trading_days`` 必须已截断到共同数据截止：可构造性以数据可得性为界，
    入场与到期不得落在尚未观察的日历尾部（R1-T05-01）。
    """
    if trading_days[-1] > cutoff:
        raise ValueError(f"交易日历未截断到截止 {cutoff}（末日 {trading_days[-1]}）")
    index = {day: i for i, day in enumerate(trading_days)}
    clean_files = sorted((DATA / "clean" / "daily").glob("*.parquet"))
    clean_range_start = clean_files[0].stem.replace("-", "")
    start_position = index[clean_range_start] + 20  # 20 日协变量回看
    timeline = trading_days[start_position:]
    n_days = len(timeline)
    cutoff_position = len(trading_days) - 1

    builders = {
        "forecast": _forecast_events,
        "express": _express_events,
        "fina_report": _fina_events,
    }
    cells: Dict[str, Any] = {}
    for name, builder in builders.items():
        events = builder()
        events["ann_date"] = events["ann_date"].astype(str)
        after_cutoff = int((events["ann_date"] > cutoff).sum())
        events = events[(events["ann_date"] >= timeline[0]) & (events["ann_date"] <= cutoff)]
        signal_positions: List[int] = []
        unmapped = 0
        for ann_date in events["ann_date"]:
            position = index.get(ann_date)
            if position is None:
                position = int(np.searchsorted(np.asarray(trading_days), ann_date))
                if position >= len(trading_days):
                    unmapped += 1
                    continue
            if position > cutoff_position:
                unmapped += 1
                continue
            signal_positions.append(position)
        positions = np.asarray(sorted(signal_positions), dtype=np.int64)
        entry_valid, entry_missing = entry_positions_within(positions.tolist(), len(trading_days))
        tradable_entry = np.asarray(entry_valid, dtype=np.int64)
        relative_entry = tradable_entry - start_position
        relative_entry = relative_entry[relative_entry >= 0]

        by_year: Dict[str, int] = {}
        for position in positions:
            by_year[trading_days[position][:4]] = by_year.get(trading_days[position][:4], 0) + 1

        core_table = _block_table(relative_entry, n_days, CORE_HORIZON)
        cells[name] = {
            "events_total_in_window": int(len(events)),
            "events_after_cutoff": after_cutoff,
            "events_with_signal_day": int(len(positions)),
            "events_with_entry_day": int(len(tradable_entry)),
            "events_without_entry_day": int(entry_missing),
            "events_right_censored_maturity": int(core_table["events_right_censored"]),
            "events_right_censored_total": int(core_table["events_right_censored"])
            + int(entry_missing),
            "unmapped_after_cutoff": int(unmapped),
            "distinct_stocks": int(events["ts_code"].nunique()),
            "distinct_signal_days": int(len(np.unique(positions))),
            "events_by_year": dict(sorted(by_year.items())),
            "core_10d": core_table,
            "sensitivity_5d": _block_table(relative_entry, n_days, SENS_HORIZON),
        }

    payload = {
        "evaluation_start": timeline[0],
        "evaluation_end": trading_days[min(cutoff_position, len(trading_days) - 1)],
        "common_data_cutoff": cutoff,
        "covariate_lookback_days": 20,
        "timeline_trading_days": n_days,
        "calendar_bounded_by_cutoff": True,
        "cells": cells,
    }
    return payload


EXIT_OK = 0
EXIT_INPUT_GATE_FAIL = 2
EXIT_CORE_GATE_FAIL = 3
EXIT_WRITE_SCOPE = 4
EXIT_HARD_ERROR = 5


def build_input_gate(
    folds: Dict[str, Any],
    models: Dict[str, Any],
    panel: Dict[str, Any],
    phase: Dict[str, Any],
    watermarks: Dict[str, Any],
    calendar_info: Dict[str, Any],
    events: Dict[str, Any],
) -> Dict[str, Any]:
    """完整输入验收门：全部必需检查共同决定能否继续生成治理工件（R1-T05-02）。"""
    frozen_models = [model for _, model, *_ in FROZEN_FOLDS]
    checks = {
        "folds_match_frozen_table": bool(folds.get("frozen_folds_match")),
        "fold_count_is_14": int(folds.get("fold_count", -1)) == len(FROZEN_FOLDS),
        "single_data_state": len(folds.get("data_state_ids") or []) == 1,
        "model_versions_match_frozen": list(folds.get("model_versions") or []) == frozen_models,
        "test_domain_contiguous": bool(folds.get("test_domain_contiguous")),
        "models_identity_match": bool(models.get("identity_match")),
        "signal_days_complete": bool(panel.get("signal_days_complete")),
        "phase_offset_matches": bool(phase.get("phase_offset_matches")),
        "runoff_complete": bool(phase.get("runoff_complete")),
        "fina_watermark_derivable": bool(watermarks.get("fina_watermark_derivable")),
        "calendar_truncated_at_cutoff": bool(calendar_info.get("truncated_at_cutoff")),
        "calendar_covers_evaluation": bool(
            calendar_info.get("bounded_last_date")
            and events.get("evaluation_end")
            and str(events["evaluation_end"]) <= str(calendar_info["bounded_last_date"])
        ),
    }
    return {
        "checks": checks,
        "input_gate_pass": all(checks.values()),
        "failed_checks": sorted(name for name, value in checks.items() if not value),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P7 T0.5 输入完整性预检（只读）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_REL, help="预检输出目录")
    parser.add_argument(
        "--allow-root", action="append", default=[], help="追加允许写根（测试用，生产不得追加）"
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else _REPO_ROOT / args.out_dir
    output_names = (
        "fold_check.json",
        "model_identity.json",
        "panel_coverage.json",
        "phase_calendar.json",
        "source_watermarks.json",
        "event_blocks.json",
        "precheck_summary.json",
    )
    try:
        resolved = assert_targets_within_write_roots(
            [out_dir / name for name in output_names], resolve_write_roots(args.allow_root)
        )
    except WriteScopeError as error:
        print(f"写域拒绝：{error}", file=sys.stderr)
        return EXIT_WRITE_SCOPE
    out_dir = resolved[0].parent

    try:
        watermarks = check_watermarks()
        cutoff = str(watermarks["common_data_cutoff"])
        calendar = _load_trading_days()
        trading_days, calendar_info = truncate_calendar(calendar, cutoff)
        folds = check_folds()
        models = check_models()
        panel = check_panel(trading_days)
        phase = check_phase_calendar(trading_days)
        events = analyze_events(trading_days, cutoff)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"输入不可用，未写出任何文件：{error}", file=sys.stderr)
        return EXIT_HARD_ERROR

    input_gate = build_input_gate(folds, models, panel, phase, watermarks, calendar_info, events)
    core_gates = {
        name: {
            "effective_blocks": cell["core_10d"]["effective_blocks"],
            "block_years": cell["core_10d"]["block_years"],
            "gate_pass": cell["core_10d"]["gate_pass"],
            "sensitivity_5d_effective_blocks": cell["sensitivity_5d"]["effective_blocks"],
        }
        for name, cell in events["cells"].items()
    }
    core_gate_all_pass = all(item["gate_pass"] for item in core_gates.values())

    shas = {
        "fold_check.json": _write_json(out_dir / "fold_check.json", folds),
        "model_identity.json": _write_json(out_dir / "model_identity.json", models),
        "panel_coverage.json": _write_json(out_dir / "panel_coverage.json", panel),
        "phase_calendar.json": _write_json(out_dir / "phase_calendar.json", phase),
        "source_watermarks.json": _write_json(out_dir / "source_watermarks.json", watermarks),
        "event_blocks.json": _write_json(out_dir / "event_blocks.json", events),
    }

    summary = {
        "schema": "p7_t05_precheck/v2",
        "evidence_id": out_dir.parent.name if out_dir.name == "precheck" else out_dir.name,
        "generated_by": "scripts/v2_p7/precheck_t05_inputs.py",
        "readonly": True,
        "alpha_consumed": False,
        "folds_match_frozen_table": input_gate["checks"]["folds_match_frozen_table"],
        "test_domain_contiguous": input_gate["checks"]["test_domain_contiguous"],
        "models_identity_match": input_gate["checks"]["models_identity_match"],
        "signal_days_complete": input_gate["checks"]["signal_days_complete"],
        "phase_offset_matches": input_gate["checks"]["phase_offset_matches"],
        "fina_watermark_derivable": input_gate["checks"]["fina_watermark_derivable"],
        "input_gate": input_gate["checks"],
        "input_gate_pass": input_gate["input_gate_pass"],
        "input_gate_failed_checks": input_gate["failed_checks"],
        "common_data_cutoff": cutoff,
        "calendar_boundary": calendar_info,
        "evaluation_window": [events["evaluation_start"], events["evaluation_end"]],
        "core_cells": core_gates,
        "core_gate_all_pass": core_gate_all_pass,
        "min_effective_blocks": MIN_EFFECTIVE_BLOCKS,
        "block_rule": {
            "block_days": BLOCK_DAYS,
            "guard_days": GUARD_DAYS,
            "assignment_key": "planned_entry_day=T+1",
            "guard_entries_unassigned": True,
            "window": "entry+horizon",
            "bounded_by": "common_data_cutoff",
        },
        "outputs_sha256": shas,
    }
    summary["record_sha256"] = _write_json(out_dir / "precheck_summary.json", summary)

    print(f"precheck_dir={out_dir}")
    print(f"common_data_cutoff={cutoff} calendar_last={calendar_info['calendar_last_date']}")
    print(f"input_gate_pass={input_gate['input_gate_pass']} failed={input_gate['failed_checks']}")
    for name, gate in core_gates.items():
        print(
            f"{name}: core10_effective_blocks={gate['effective_blocks']}"
            f" block_years={gate['block_years']} pass={gate['gate_pass']}"
            f" sens5={gate['sensitivity_5d_effective_blocks']}"
        )
    print(f"summary={out_dir / 'precheck_summary.json'}")
    if not input_gate["input_gate_pass"]:
        print("输入验收门未通过：治理工件不得生成", file=sys.stderr)
        return EXIT_INPUT_GATE_FAIL
    if not core_gate_all_pass:
        print("核心格有效块不足：按 Q5=A 停止并回用户重新裁决", file=sys.stderr)
        return EXIT_CORE_GATE_FAIL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
