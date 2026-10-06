# -*- coding: utf-8 -*-
"""v2 P1 单元 3：重放对账器（冻结参照分区 vs cs_train 逐日逐值比对 + 三段式聚合判定）。

三段式（沿母截面先例）：
1. 日级统计：逐日逐列比对（行集 ts_code 差集、列交集、数值 atol=1e-6 且 NaN==NaN
   视为一致、非数值列精确相等）；
2. 合并：日级计数合并为全窗口列级口径（超容差行占比在合并分母上重算，禁止逐日平均；
   分母 = 全窗口该列实际可比日的可比行数合计，含零差异日——零差异日的分母由日级
   ``col_rows`` 透传，R3-04 整改前分母只累计差异日，会高估占比误报超门）；
3. 判定：列级 超容差行占比 > 1e-4 或 max|Δ| > 0.05 ⇒ ``over_gate=True``；
   每个差异列带 ``attribution``（命中冻结 §8 D-04 清单的填对应条目；zscore/neu 子列
   随被归因母列标记派生连锁；未命中填 "待调查"）。

并行：joblib loky 逐日并行（沿 `features/pipeline.py` 先例）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

__all__ = [
    "COMPARE_ATOL",
    "GATE_MAX_ABS_DIFF",
    "GATE_OVER_SHARE",
    "compare_partitions",
    "attribute_column",
]

COMPARE_ATOL = 1e-6
GATE_OVER_SHARE = 1e-4
GATE_MAX_ABS_DIFF = 0.05
_SAMPLE_CAP = 5
_KEY_COLUMNS = ("trade_date", "ts_code")

#: D-04①：share_float/block_trade/top_list 水位晚于基线数据态（直接列）
_D04_1_EXACT = {"days_to_unlock", "unlock_ratio", "unlock_risk_flag"}
_D04_1_PREFIXES = ("block_discount_", "lhb_")
#: D-04②：raw daily 2024-01-16/17 丢停牌股行（等权市场序列微移）
_D04_2_EXACT = {"downside_corr_20"}
#: D-04③：季频财务/基金持仓 raw 修订（NaN 型差异，单元 2 标定实证集）
_D04_3_EXACT = {"dividend_payout_ratio", "cf_nm", "ocf_to_profit"}
_D04_3_PREFIXES = ("fund_",)
#: D-04④：margin_detail raw 分区修订（丢行；单元 3 探测实证 2026-04-17 分区缺行）
_D04_4_EXACT = {"rzye_chg_5", "rzye_chg_20", "short_balance_change_5", "short_sell_vol_change_5"}
#: D-05：pandas 3.0 rolling.skew/kurt 回归（环境漂移，已修复；残留差异 = 异常待查）
_D05_EXACT = {"skewness_20", "kurtosis_20"}

_D04_1_TEXT = "D-04① share_float/block_trade/top_list 水位晚于基线数据态"
_D04_2_TEXT = "D-04② raw daily 2024-01-16/17 分区丢行（等权市场序列微移）"
_D04_3_TEXT = "D-04③ 季频财务/基金持仓 raw 修订"
_D04_4_TEXT = "D-04④ margin_detail raw 分区修订（丢行）"
_D05_TEXT = "D-05 pandas 3.0 rolling.skew/kurt 回归（已修复，残留=异常）"


def _attribute_base(col: str) -> Optional[str]:
    """基础列归因（未命中返回 None）。"""
    if col in _D04_1_EXACT or col.startswith(_D04_1_PREFIXES):
        return _D04_1_TEXT
    if col in _D04_2_EXACT:
        return _D04_2_TEXT
    if col in _D04_3_EXACT or col.startswith(_D04_3_PREFIXES):
        return _D04_3_TEXT
    if col in _D04_4_EXACT:
        return _D04_4_TEXT
    if col in _D05_EXACT:
        return _D05_TEXT
    return None


def attribute_column(col: str) -> str:
    """列归因：基础列查 D-04 清单；zscore_X / zscore_X_sz / neu_X 子列随母列归因。"""
    base = _attribute_base(col)
    if base is not None:
        return base
    for prefix in ("zscore_", "neu_"):
        if col.startswith(prefix):
            plen = len(prefix)  # 切片界先赋值（black/flake8 E203 兼容）
            parent = col[plen:]
            if parent.endswith("_sz"):
                parent = parent[: -len("_sz")]
            parent_attr = _attribute_base(parent)
            if parent_attr is not None:
                return f"{parent_attr}（zscore/neu 派生连锁）"
    return "待调查"


def _resolve_cs_dir(root: Path | str) -> Path:
    """cs_train 分区目录解析（兼容 scratch 根与直接分区目录两种传法）。"""
    path = Path(root)
    nested = path / "features" / "cs_train"
    return nested if nested.exists() else path


def _compare_numeric_column(
    left: pd.Series, right: pd.Series, atol: float
) -> tuple[int, float, pd.Series]:
    """数值列比对：返回 (超容差行数, max|Δ|, 超容差行布尔掩码)。NaN==NaN 一致。

    max|Δ| 只取数值差：over 行全为 NaN 型差异（两侧 isna 不一致）时 diff[over]
    全 NaN、.max() 得 nan——此处显式归零，不靠 pandas/Python 默认行为侥幸。
    """
    lval = pd.to_numeric(left, errors="coerce")
    rval = pd.to_numeric(right, errors="coerce")
    both_nan = lval.isna() & rval.isna()
    diff = (lval - rval).abs()
    over = ((diff > atol) | (lval.isna() != rval.isna())) & ~both_nan
    if bool(over.any()):
        peak = diff[over].max()
        max_abs = 0.0 if pd.isna(peak) else float(peak)
    else:
        max_abs = 0.0
    return int(over.sum()), max_abs, over


def _compare_nonnumeric_column(left: pd.Series, right: pd.Series) -> tuple[int, pd.Series]:
    """非数值列比对：NaN==NaN 一致，其余精确相等；返回 (不一致行数, 掩码)。"""
    both_nan = left.isna() & right.isna()
    ne = (left != right) & ~both_nan
    return int(ne.sum()), ne


def _count_nan_mismatch(left: pd.Series, right: pd.Series) -> int:
    """NaN 型差异行数（两侧 isna 不一致；D-04③/④ 纯 NaN 型判定的日级统计口径）。"""
    return int((left.isna() != right.isna()).sum())


def _to_json_value(value: Any) -> Any:
    """numpy/pandas 标量转 JSON 原生类型（float32/int64 等不可直接序列化）。"""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _column_samples(mask: pd.Series, left: pd.Series, right: pd.Series) -> list[dict[str, Any]]:
    """离群样例（每列每日上限 5 条；数值已转 JSON 原生类型）。"""
    idx = mask[mask].index[:_SAMPLE_CAP]
    samples = []
    for code in idx:
        lv, rv = left.loc[code], right.loc[code]
        try:
            diff: Any = abs(float(lv) - float(rv))
        except (TypeError, ValueError):
            diff = None
        samples.append(
            {
                "ts_code": str(code),
                "replay": _to_json_value(lv),
                "cs_train": _to_json_value(rv),
                "abs_diff": diff,
            }
        )
    return samples


def _compare_one_day(date: str, replay_dir: Path, cs_dir: Path) -> dict[str, Any]:
    """单日比对：行集、列交集、逐列逐值（atol + NaN 语义）。"""
    replay_path = replay_dir / f"{date}.parquet"
    cs_path = cs_dir / f"{date}.parquet"
    if not replay_path.exists() or not cs_path.exists():
        return {
            "date": date,
            "status": "missing",
            "note": f"replay={replay_path.exists()}, cs_train={cs_path.exists()}",
        }
    replay = pd.read_parquet(replay_path)
    cs = pd.read_parquet(cs_path)
    replay_codes = set(replay["ts_code"].astype(str))
    cs_codes = set(cs["ts_code"].astype(str))
    common_codes = sorted(replay_codes & cs_codes)
    common_cols = sorted((set(replay.columns) & set(cs.columns)) - set(_KEY_COLUMNS))
    left = replay.set_index("ts_code").loc[common_codes]
    right = cs.set_index("ts_code").loc[common_codes]
    col_stats: dict[str, Any] = {}
    for col in common_cols:
        lcol, rcol = left[col], right[col]
        if pd.api.types.is_numeric_dtype(lcol) and pd.api.types.is_numeric_dtype(rcol):
            over, max_abs, mask = _compare_numeric_column(lcol, rcol, COMPARE_ATOL)
        else:
            over, mask = _compare_nonnumeric_column(lcol, rcol)
            max_abs = 0.0
        if over == 0:
            continue
        col_stats[col] = {
            "rows": len(common_codes),
            "over_rows": over,
            "over_nan_rows": _count_nan_mismatch(lcol, rcol),
            "max_abs_diff": max_abs,
            "samples": _column_samples(mask, lcol, rcol),
        }
    return {
        "date": date,
        "status": "ok",
        "rows_replay": len(replay),
        "rows_cs_train": len(cs),
        "codes_only_replay": sorted(replay_codes - cs_codes)[:_SAMPLE_CAP],
        "codes_only_cs_train": sorted(cs_codes - replay_codes)[:_SAMPLE_CAP],
        "n_codes_only_replay": len(replay_codes - cs_codes),
        "n_codes_only_cs_train": len(cs_codes - replay_codes),
        # 键列噪声在源头剔除（列集差异只登记真实特征/标签列）
        "cols_only_replay": sorted(set(replay.columns) - set(cs.columns) - set(_KEY_COLUMNS)),
        "cols_only_cs_train": sorted(set(cs.columns) - set(replay.columns) - set(_KEY_COLUMNS)),
        "common_cols": common_cols,
        # 当日每个 common 列的可比行数（含零差异列；R3-04 全窗口分母的数据来源）
        "col_rows": {col: len(common_codes) for col in common_cols},
        "columns": col_stats,
    }


def _merge_day_results(day_results: list[dict[str, Any]]) -> dict[str, Any]:
    """三段式后两段：日级统计合并为全窗口列级口径 + 判定与归因。"""
    merged: dict[str, Any] = {}
    col_total_rows: dict[str, int] = {}
    row_diff_days = []
    missing_days = []
    cols_only_replay: set[str] = set()
    cols_only_cs: set[str] = set()
    compared_cols: set[str] = set()
    for day in sorted(day_results, key=lambda d: d["date"]):
        if day["status"] != "ok":
            missing_days.append({"date": day["date"], "note": day["note"]})
            continue
        compared_cols.update(day.get("common_cols", []))
        if day["n_codes_only_replay"] or day["n_codes_only_cs_train"]:
            row_diff_days.append(
                {
                    "date": day["date"],
                    "n_only_replay": day["n_codes_only_replay"],
                    "n_only_cs_train": day["n_codes_only_cs_train"],
                    "sample_only_replay": day["codes_only_replay"],
                    "sample_only_cs_train": day["codes_only_cs_train"],
                }
            )
        cols_only_replay.update(day["cols_only_replay"])
        cols_only_cs.update(day["cols_only_cs_train"])
        # 分母全窗口累计（含零差异日；列在某日不可比则当日不计入——R3-04）
        for col, n_rows in day.get("col_rows", {}).items():
            col_total_rows[col] = col_total_rows.get(col, 0) + n_rows
        for col, stat in day["columns"].items():
            agg = merged.setdefault(
                col, {"total_rows": 0, "over_rows": 0, "max_abs_diff": 0.0, "samples": []}
            )
            agg["over_rows"] += stat["over_rows"]
            agg["max_abs_diff"] = max(agg["max_abs_diff"], stat["max_abs_diff"])
            if len(agg["samples"]) < _SAMPLE_CAP:
                for sample in stat["samples"]:
                    sample["date"] = day["date"]
                agg["samples"].extend(stat["samples"])
                agg["samples"] = agg["samples"][:_SAMPLE_CAP]
    for col, agg in merged.items():
        # over_share 分母 = 全窗口该列实际可比日的可比行数合计（不是有差异日的行数）
        agg["total_rows"] = col_total_rows.get(col, 0)
        agg["over_share"] = agg["over_rows"] / agg["total_rows"] if agg["total_rows"] else 0.0
        agg["over_gate"] = bool(
            agg["over_share"] > GATE_OVER_SHARE or agg["max_abs_diff"] > GATE_MAX_ABS_DIFF
        )
        agg["attribution"] = attribute_column(col)
    return {
        "columns": merged,
        "missing_days": missing_days,
        "row_set_diff_days": row_diff_days,
        "cols_only_replay": sorted(cols_only_replay),
        "cols_only_cs_train": sorted(cols_only_cs),
        "n_compared_columns": len(compared_cols),
    }


def compare_partitions(
    replay_root: Path | str,
    cs_train_root: Path | str,
    dates: Iterable[str],
    n_jobs: int = -1,
) -> dict[str, Any]:
    """重放对账主入口：逐日并行比对 → 三段式聚合 → 判定 + 归因。

    Args:
        replay_root: 重放 scratch 根（或其下 features/cs_train 目录直传）
        cs_train_root: cs_train 分区根（生产只读）
        dates: 比对交易日清单（YYYYMMDD）
        n_jobs: joblib loky worker 数（-1 = 全核；1 = 串行）

    Returns:
        对账报告 dict（days / columns / summary 三段）。
    """
    from joblib import Parallel, delayed

    replay_dir = _resolve_cs_dir(replay_root)
    cs_dir = _resolve_cs_dir(cs_train_root)
    date_list = [str(d) for d in dates]
    logger.info(f"重放对账启动: {len(date_list)} 天，replay={replay_dir}，cs_train={cs_dir}")
    if n_jobs == 1:
        day_results = [_compare_one_day(d, replay_dir, cs_dir) for d in date_list]
    else:
        day_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compare_one_day)(d, replay_dir, cs_dir) for d in date_list
        )
    merged = _merge_day_results(day_results)
    columns = merged["columns"]
    diff_cols = {c: a for c, a in columns.items() if a["over_rows"] > 0}
    by_attribution: dict[str, int] = {}
    for agg in diff_cols.values():
        by_attribution[agg["attribution"]] = by_attribution.get(agg["attribution"], 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "replay_root": str(replay_root),
        "cs_train_root": str(cs_train_root),
        "window": {
            "start": min(date_list) if date_list else None,
            "end": max(date_list) if date_list else None,
        },
        "days": {
            "requested": len(date_list),
            "compared": len(date_list) - len(merged["missing_days"]),
            "missing": merged["missing_days"],
            "row_set_diff_days": merged["row_set_diff_days"],
        },
        "cols_only_replay": merged["cols_only_replay"],
        "cols_only_cs_train": merged["cols_only_cs_train"],
        "columns": columns,
        "summary": {
            "total_columns_compared": merged["n_compared_columns"],
            "columns_with_diff": len(diff_cols),
            "over_gate_columns": sorted(c for c, a in diff_cols.items() if a["over_gate"]),
            "by_attribution": by_attribution,
            "worst": sorted(
                (
                    {"column": c, "over_share": a["over_share"], "max_abs_diff": a["max_abs_diff"]}
                    for c, a in diff_cols.items()
                ),
                key=lambda r: -r["max_abs_diff"],
            )[:10],
        },
    }
