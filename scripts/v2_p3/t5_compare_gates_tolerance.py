# -*- coding: utf-8 -*-
"""P3 T5 比较边·净值类容差面子模块（自 ``t5_compare_gates`` 拆出，R2-T5A-R1 拆分消化）。

承载 G-code ② / 回归边共用的容差与有限性件（判据冻结，本模块只实现判定）：

- **容差面列清单（R2-T5A-R1-02 收紧）**：链式净值 ``chain_nav.nav`` + 逐折 daily
  ``nav/total_value`` + summary **绩效统计列**——允许清单 = 权威统计输出列
  ``SUMMARY_STATS_COLUMNS``（来源 = ``v2/evidence/fingerprint_keys.py::EXCLUDE_EXACT``
  中 9 个 ``bt_*`` 统计列，契约附录 A 唯一权威源）∩ 两侧浮点 dtype 交集。
  **允许清单外 ``bt_*`` 浮点列 = 配置列**（``bt_max_weight_per_stock`` /
  ``bt_initial_capital`` 等），由 ``compare_summary_config_exact`` 在任何容差投影
  **之前**逐位精确核验（含 <1e-6 微差 ⇒ FAIL），禁止配置差异被容差投影抹平。
- **有限性先验（R2-T5A-07）**：chain_nav/daily/summary 净值类列任一 NaN/Inf ⇒
  FAIL（非「容差内」）；chain_nav.nav 与 daily.nav 复用 P2b
  ``compare_merger_batches.validate_nav_finite``（前置条件 = ``load_runs_batch``
  契约校验已过），本模块补 daily.total_value 与 summary ``bt_*`` 浮点列两缺口。
- **折集交叉核验（R2-T5A-03-①）**：``_check_expected_fold_set`` —— runs 折集必须
  恰为 ``EXPECTED_SPLITS``（14 折），缺一/多一即指出折号。

机制（P2-2 选项 (a) 冻结口径）：容差核验先行 ⇒ 三面（chain_nav/summary/daily）
皆对齐且容差内 ⇒ 对有逐位差的面把臂②投影到臂①取值（``_project_*_within_tolerance``，
各面 max|Δ| 由核验函数登记）；任一面超容差 ⇒ 不投影。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p2b.compare_merger_batches import validate_nav_finite  # noqa: E402
from src.lazybull.v2.evidence.fingerprint_keys import EXCLUDE_EXACT  # noqa: E402
from src.lazybull.v2.evidence.runs_loader import RunsBatch, RunsFold  # noqa: E402

#: 链式净值容差（runs 契约 §9.1 明文 1e-6；单一来源，主文件 re-export）
NAV_TOLERANCE = 1e-6
#: G-code ② 容差面·逐折 daily 净值类列（§7.1-② 冻结判据收口，选项 (a)）
DAILY_TOLERANCE_COLS: Tuple[str, ...] = ("nav", "total_value")
#: summary 绩效统计列允许清单（R2-T5A-R1-02）：来源 = fingerprint_keys.EXCLUDE_EXACT
#: 中 9 个 bt_* 统计列（契约附录 A 唯一权威源）；其中 bt_trading_days（整数）与
#: bt_start/bt_end（日期串）非浮点，实际容差列 = 本清单 ∩ 两侧浮点 dtype 交集。
SUMMARY_STATS_COLUMNS: Tuple[str, ...] = tuple(
    sorted(column for column in EXCLUDE_EXACT if column.startswith("bt_"))
)


def _fold_name(split: int) -> str:
    """折名（函数内延迟 import 主文件唯一实现，避免模块循环依赖）。"""
    from scripts.v2_p3.t5_compare_gates import _fold_name as fold_name

    return fold_name(split)


def _expected_splits() -> Tuple[int, ...]:
    """预期折集（函数内延迟 import 主文件唯一来源，避免模块循环依赖）。"""
    from scripts.v2_p3.t5_compare_gates import EXPECTED_SPLITS

    return EXPECTED_SPLITS


# ---------------------------------------------------------------------------
# 容差核验（三面：chain_nav / summary 绩效统计列 / 逐折 daily）
# ---------------------------------------------------------------------------


def compare_chain_nav_tolerance(
    chain_a: pd.DataFrame, chain_b: pd.DataFrame, tol: float = NAV_TOLERANCE
) -> Dict[str, object]:
    """链式净值 1e-6 容差比较：逐折按位次对齐（date 为折内序号口径），max|Δnav| ≤ tol。

    返回 aligned（折集/长度/date 序列一致）与 within_tolerance（容差内）两档；皆真
    调用方方可做容差投影。有限性先验（R2-T5A-07）：任一 nav 为 NaN/Inf ⇒
    within_tolerance=False（非「容差内」），禁止 NaN 绕过容差门。
    """
    result: Dict[str, object] = {
        "tolerance": tol,
        "aligned": True,
        "within_tolerance": True,
        "max_abs_diff": 0.0,
        "fold_max_abs_diff": {},
        "errors": [],
    }
    splits_a = sorted(int(s) for s in chain_a["split_index"].unique())
    splits_b = sorted(int(s) for s in chain_b["split_index"].unique())
    if splits_a != splits_b:
        result["aligned"] = result["within_tolerance"] = False
        result["errors"].append(f"折集不一致: {splits_a} vs {splits_b}")  # type: ignore[union-attr]
        return result
    max_diff = 0.0
    fold_diffs: Dict[str, float] = {}
    for split in splits_a:
        seg_a = chain_a[chain_a["split_index"] == split].sort_values("date")
        seg_b = chain_b[chain_b["split_index"] == split].sort_values("date")
        if len(seg_a) != len(seg_b) or seg_a["date"].tolist() != seg_b["date"].tolist():
            result["aligned"] = result["within_tolerance"] = False
            result["errors"].append(  # type: ignore[union-attr]
                f"split{split:02d} 链式净值行数/date 序列不一致: " f"{len(seg_a)} vs {len(seg_b)}"
            )
            continue
        values_a = pd.to_numeric(seg_a["nav"], errors="coerce").to_numpy(dtype=float)
        values_b = pd.to_numeric(seg_b["nav"], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(values_a).all() and np.isfinite(values_b).all()):
            result["within_tolerance"] = False
            result["errors"].append(  # type: ignore[union-attr]
                f"split{split:02d} 链式净值含非有限值（NaN/Inf ⇒ FAIL，非「容差内」）"
            )
            continue
        diff = float(np.abs(values_a - values_b).max(initial=0.0))
        fold_diffs[_fold_name(split)] = diff
        max_diff = max(max_diff, diff)
        if diff > tol:
            result["within_tolerance"] = False
            result["errors"].append(  # type: ignore[union-attr]
                f"split{split:02d} 链式净值超容差 max|Δ|={diff:.3e}（容差 {tol}）"
            )
    result["max_abs_diff"] = max_diff
    result["fold_max_abs_diff"] = fold_diffs
    return result


def _summary_float_metric_columns(summary: pd.DataFrame) -> List[str]:
    """summary 全部 ``bt_*`` 浮点列（有限性面用全量；配置/统计不区分）。

    ``bt_start/bt_end``（日期串）与 ``bt_trading_days/bt_top_n``（整数）等非浮点列
    不在其内。
    """
    return sorted(
        column
        for column in summary.columns
        if column.startswith("bt_") and pd.api.types.is_float_dtype(summary[column])
    )


def _summary_tolerance_columns(summary_a: pd.DataFrame, summary_b: pd.DataFrame) -> List[str]:
    """summary 容差列 = 绩效统计允许清单 ∩ 两侧 ``bt_*`` 浮点列交集（R2-T5A-R1-02）。

    允许清单外 ``bt_*`` 浮点列（``bt_max_weight_per_stock`` 等配置列）一律不进容差面，
    由 ``compare_summary_config_exact`` 逐位精确核验。
    """
    shared = set(_summary_float_metric_columns(summary_a)) & set(
        _summary_float_metric_columns(summary_b)
    )
    return sorted(shared & set(SUMMARY_STATS_COLUMNS))


def _summary_config_float_columns(summary_a: pd.DataFrame, summary_b: pd.DataFrame) -> List[str]:
    """summary 配置浮点列 = 两侧 ``bt_*`` 浮点列交集 − 绩效统计允许清单（逐位精确面）。"""
    shared = set(_summary_float_metric_columns(summary_a)) & set(
        _summary_float_metric_columns(summary_b)
    )
    return sorted(shared - set(SUMMARY_STATS_COLUMNS))


def _ordered_by_split(summary: pd.DataFrame, label: str) -> pd.DataFrame:
    """summary 按 split_index 升序对齐副本（split_index 重复 ⇒ 输入契约违反抛错）。"""
    if summary["split_index"].duplicated().any():
        raise ValueError(f"{label} summary 存在重复 split_index 行（容差对齐前提违反）")
    return summary.sort_values("split_index").reset_index(drop=True)


def compare_summary_tolerance(
    summary_a: pd.DataFrame, summary_b: pd.DataFrame, tol: float = NAV_TOLERANCE
) -> Dict[str, object]:
    """summary 绩效统计列（允许清单 ∩ ``bt_*`` 浮点列）1e-6 容差比较（P2-2 + R1-02）。

    行对齐 = 行数 + split_index 序列（升序）一致；容差列任一 NaN/Inf ⇒
    within_tolerance=False（非「容差内」）。逐列登记 max|Δ|。
    """
    columns = _summary_tolerance_columns(summary_a, summary_b)
    result: Dict[str, object] = {
        "tolerance": tol,
        "columns": columns,
        "aligned": True,
        "within_tolerance": True,
        "max_abs_diff": 0.0,
        "column_max_abs_diff": {},
        "errors": [],
    }
    errors: List[str] = result["errors"]  # type: ignore[assignment]
    ordered_a = _ordered_by_split(summary_a, "臂①")
    ordered_b = _ordered_by_split(summary_b, "臂②")
    splits_a = ordered_a["split_index"].astype(int).tolist()
    splits_b = ordered_b["split_index"].astype(int).tolist()
    if len(ordered_a) != len(ordered_b) or splits_a != splits_b:
        result["aligned"] = result["within_tolerance"] = False
        errors.append(f"summary 行数/折序列不一致: {splits_a} vs {splits_b}")
        return result
    max_diff = 0.0
    column_diffs: Dict[str, float] = {}
    for column in columns:
        values_a = pd.to_numeric(ordered_a[column], errors="coerce").to_numpy(dtype=float)
        values_b = pd.to_numeric(ordered_b[column], errors="coerce").to_numpy(dtype=float)
        bad_a = np.flatnonzero(~np.isfinite(values_a))
        bad_b = np.flatnonzero(~np.isfinite(values_b))
        if len(bad_a) or len(bad_b):
            result["within_tolerance"] = False
            errors.append(
                f"summary.{column} 含非有限值（NaN/Inf ⇒ FAIL，非「容差内」）: "
                f"臂①行 {bad_a[:5].tolist()} / 臂②行 {bad_b[:5].tolist()}"
            )
            continue
        diff = float(np.abs(values_a - values_b).max(initial=0.0))
        column_diffs[column] = diff
        max_diff = max(max_diff, diff)
        if diff > tol:
            result["within_tolerance"] = False
            errors.append(f"summary.{column} 超容差 max|Δ|={diff:.3e}（容差 {tol}）")
    result["max_abs_diff"] = max_diff
    result["column_max_abs_diff"] = column_diffs
    return result


def compare_summary_config_exact(
    summary_a: pd.DataFrame, summary_b: pd.DataFrame
) -> Dict[str, object]:
    """summary 配置浮点列（允许清单外 ``bt_*``）**逐位精确核验**（R2-T5A-R1-02）。

    在任何容差投影之前执行：任一配置列任一折存在差异（含 <1e-6 微差）⇒ FAIL，
    禁止配置差异被容差投影抹平。NaN==NaN 视等（与 ``_frame_equal_exact`` 同口径）；
    NaN 单侧出现亦属差异。verdict 登记核验列清单与逐列差异明细。
    """
    columns = _summary_config_float_columns(summary_a, summary_b)
    result: Dict[str, object] = {
        "columns": columns,
        "aligned": True,
        "match": True,
        "column_diffs": {},
        "errors": [],
    }
    errors: List[str] = result["errors"]  # type: ignore[assignment]
    ordered_a = _ordered_by_split(summary_a, "臂①")
    ordered_b = _ordered_by_split(summary_b, "臂②")
    splits_a = ordered_a["split_index"].astype(int).tolist()
    splits_b = ordered_b["split_index"].astype(int).tolist()
    if len(ordered_a) != len(ordered_b) or splits_a != splits_b:
        result["aligned"] = result["match"] = False
        errors.append(f"summary 行数/折序列不一致: {splits_a} vs {splits_b}")
        return result
    column_diffs: Dict[str, List[str]] = {}
    for column in columns:
        values_a = pd.to_numeric(ordered_a[column], errors="coerce").to_numpy(dtype=float)
        values_b = pd.to_numeric(ordered_b[column], errors="coerce").to_numpy(dtype=float)
        nan_mismatch = np.isnan(values_a) != np.isnan(values_b)
        value_mismatch = (values_a != values_b) & ~np.isnan(values_a) & ~np.isnan(values_b)
        bad_positions = np.flatnonzero(nan_mismatch | value_mismatch)
        if len(bad_positions):
            result["match"] = False
            examples = [
                f"split{int(ordered_a['split_index'].iloc[p]):02d} "
                f"臂①={ordered_a[column].iloc[p]!r} 臂②={ordered_b[column].iloc[p]!r}"
                for p in bad_positions[:3]
            ]
            column_diffs[column] = examples
            errors.append(
                f"summary 配置列 {column} 逐位精确核验失败（{len(bad_positions)} 折差异，"
                f"配置差异不得容差抹平）: {examples}"
            )
    result["column_diffs"] = column_diffs
    return result


def compare_daily_tolerance(
    folds_a: Dict[int, RunsFold],
    folds_b: Dict[int, RunsFold],
    tol: float = NAV_TOLERANCE,
) -> Dict[str, object]:
    """逐折 daily 净值类列（nav/total_value）1e-6 容差比较（P2-2 选项 (a) 收口）。

    行对齐 = 行数 + trade_date 序列（升序、唯一）一致；双侧均无 daily ⇒ 该折跳过；
    单侧缺失 ⇒ 不对齐；容差列任一 NaN/Inf ⇒ within_tolerance=False（非「容差内」）。
    逐折登记 max|Δ|。
    """
    result: Dict[str, object] = {
        "tolerance": tol,
        "columns": list(DAILY_TOLERANCE_COLS),
        "aligned": True,
        "within_tolerance": True,
        "max_abs_diff": 0.0,
        "fold_max_abs_diff": {},
        "errors": [],
    }
    errors: List[str] = result["errors"]  # type: ignore[assignment]
    max_diff = 0.0
    fold_diffs: Dict[str, float] = {}
    for split in sorted(set(folds_a) | set(folds_b)):
        name = _fold_name(split)
        daily_a = folds_a[split].daily if split in folds_a else None
        daily_b = folds_b[split].daily if split in folds_b else None
        if daily_a is None and daily_b is None:
            continue
        if daily_a is None or daily_b is None:
            result["aligned"] = result["within_tolerance"] = False
            side_a = "有" if daily_a is not None else "无"
            side_b = "有" if daily_b is not None else "无"
            errors.append(f"{name} daily 单侧缺失（臂①={side_a} / 臂②={side_b}）")
            continue
        dates_a = sorted(str(d) for d in daily_a["trade_date"])
        dates_b = sorted(str(d) for d in daily_b["trade_date"])
        if len(daily_a) != len(daily_b) or dates_a != dates_b or len(set(dates_a)) != len(dates_a):
            result["aligned"] = result["within_tolerance"] = False
            errors.append(f"{name} daily 行数/trade_date 序列不一致或含重复日期")
            continue
        sorted_a = daily_a.assign(_dkey=dates_a).sort_values("_dkey")
        sorted_b = daily_b.assign(_dkey=dates_b).sort_values("_dkey")
        fold_diff = 0.0
        for column in DAILY_TOLERANCE_COLS:
            values_a = pd.to_numeric(sorted_a[column], errors="coerce").to_numpy(dtype=float)
            values_b = pd.to_numeric(sorted_b[column], errors="coerce").to_numpy(dtype=float)
            if not (np.isfinite(values_a).all() and np.isfinite(values_b).all()):
                result["within_tolerance"] = False
                errors.append(f"{name} daily.{column} 含非有限值（NaN/Inf ⇒ FAIL，非「容差内」）")
                continue
            diff = float(np.abs(values_a - values_b).max(initial=0.0))
            fold_diff = max(fold_diff, diff)
            if diff > tol:
                result["within_tolerance"] = False
                errors.append(f"{name} daily.{column} 超容差 max|Δ|={diff:.3e}（容差 {tol}）")
        fold_diffs[name] = fold_diff
        max_diff = max(max_diff, fold_diff)
    result["max_abs_diff"] = max_diff
    result["fold_max_abs_diff"] = fold_diffs
    return result


# ---------------------------------------------------------------------------
# 折集交叉核验 + 净值类面有限性先验
# ---------------------------------------------------------------------------


def _check_expected_fold_set(batch: RunsBatch, label: str) -> List[str]:
    """runs 折集交叉核验（R2-T5A-03-①）：必须恰为 EXPECTED_SPLITS（14 折），缺一/多一
    即指出折号；与指纹门共用同一 EXPECTED_SPLITS ⇒ 两门折集结构性相等。"""
    expected = list(_expected_splits())
    splits = sorted(int(split) for split in batch.folds)
    if splits == expected:
        return []
    parts = []
    missing = [_fold_name(split) for split in expected if split not in batch.folds]
    extra = [_fold_name(split) for split in splits if split not in expected]
    if missing:
        parts.append(f"缺折 {missing}")
    if extra:
        parts.append(f"多折 {extra}")
    return [
        f"{label} runs 折集必须恰为 {len(expected)} 折（{expected[0]}..{expected[-1]}）："
        f"实际 {len(splits)} 折，{'，'.join(parts)}"
    ]


def _finite_surface_errors(batch: RunsBatch, label: str) -> List[str]:
    """净值类面有限性先验（R2-T5A-07）：任一 NaN/Inf ⇒ FAIL（非「容差内」）。

    chain_nav.nav 与逐折 daily.nav 复用 P2b ``validate_nav_finite``（前置条件 =
    ``load_runs_batch`` 契约校验已过），本函数补 daily.total_value 与 summary
    ``bt_*`` 浮点列两缺口。
    """
    errors = [f"{label} {message}" for message in validate_nav_finite(batch)]
    for split, fold in sorted(batch.folds.items()):
        daily = fold.daily
        if daily is None:
            continue
        values = pd.to_numeric(daily["total_value"], errors="coerce").to_numpy(dtype=float)
        for position in np.flatnonzero(~np.isfinite(values))[:5]:
            errors.append(
                f"{label} 批次 {batch.batch_id} 折 {_fold_name(split)} daily 行={position} "
                f"日期={daily['trade_date'].iloc[position]!r} "
                f"列=total_value 非有限值={daily['total_value'].iloc[position]!r}"
            )
    for column in _summary_float_metric_columns(batch.summary):
        values = pd.to_numeric(batch.summary[column], errors="coerce").to_numpy(dtype=float)
        for position in np.flatnonzero(~np.isfinite(values))[:5]:
            row = batch.summary.iloc[position]
            errors.append(
                f"{label} 批次 {batch.batch_id} summary 行={position} "
                f"折={row['split_index']!r} 列={column} 非有限值={row[column]!r}"
            )
    return errors


# ---------------------------------------------------------------------------
# 容差内投影（仅容差列；配置列不在其内，永不投影）
# ---------------------------------------------------------------------------


def _project_summary_within_tolerance(
    batch_a: RunsBatch, batch_b: RunsBatch, columns: Sequence[str]
) -> RunsBatch:
    """容差内投影：batch_b summary 的容差列按 split_index 对齐取臂①值（其余列不动）。"""
    indexed_a = batch_a.summary.set_index("split_index")
    projected = batch_b.summary.copy()
    split_keys = [int(s) for s in projected["split_index"]]
    for column in columns:
        projected[column] = [indexed_a.loc[split, column] for split in split_keys]
    return replace(batch_b, summary=projected)


def _project_daily_within_tolerance(batch_a: RunsBatch, batch_b: RunsBatch) -> RunsBatch:
    """容差内投影：batch_b 逐折 daily 的 nav/total_value 按 trade_date 对齐取臂①值。"""
    folds: Dict[int, RunsFold] = {}
    for split, fold_b in batch_b.folds.items():
        fold_a = batch_a.folds.get(split)
        daily_a = fold_a.daily if fold_a is not None else None
        if fold_b.daily is None or daily_a is None:
            folds[split] = fold_b
            continue
        keyed_a = {str(row["trade_date"]): row for _, row in daily_a.iterrows()}
        projected = fold_b.daily.copy()
        for column in DAILY_TOLERANCE_COLS:
            projected[column] = [keyed_a[str(day)][column] for day in fold_b.daily["trade_date"]]
        folds[split] = replace(fold_b, daily=projected)
    return replace(batch_b, folds=folds)
