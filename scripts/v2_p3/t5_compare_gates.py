# -*- coding: utf-8 -*-
"""P3 T5 三道比较边判定工具（验收专用，不入生产链路；§7.1 / §19.4 实现落点）。

三道边（判据全部已在契约/规划中冻结，本工具只实现判定、**不修订判据本身**）：

- **G-code**（臂① legacy_cs_train vs 臂② v2_cs_train_verify，Q2-A 代码迁移等价门）：
  ① 主门 = 逐折预测指纹逐位一致（14/14 折）+ 全批汇总指纹逐位一致（§13.3.3 S1/M1）；
  指纹值先验合法 64 位 sha256 hex 再比较（缺字段/None/非法 ⇒ FAIL，空摘要不可过）；
  ② runs 批次**成交逐笔精确 + 净值类列 1e-6 容差**（§7.1-② 冻结判据如实实现）：
  容差面 = chain_nav.nav + summary 绩效统计允许清单列 + 逐折 daily nav/total_value
  （容差核验先行 + 容差内投影 + 其余产物逐位精确，各面 max|Δ| 登记）；**允许清单外
  ``bt_*`` 配置列在任何投影前逐位精确核验**（R2-T5A-R1-02，含 <1e-6 微差 ⇒ FAIL）；
  两臂 runs 折集交叉核验必须恰为 14 折（缺一/多一即 FAIL 指出折号）；净值类面
  有限性先验（NaN/Inf ⇒ FAIL，非「容差内」）；复用 P2b ``compare_runs_exact``；
  ③ 辅证 = 评估路径 topk 明细 CSV 逐行对比（**双臂全折同缺 ⇒ FAIL**，R2-T5A-R1-01）
  + 信号路径 top-k 对比（两臂 ``folds/splitNN/signal_topk.parquet`` 真实信号过滤通路
  产物，fingerprint.json 增键 + sha256 + rows 登记核验 + 完整性门，缺/漂移 ⇒ FAIL；
  零辅证不得过总门）+ 并列分数互换计数（分组身份配对口径，>入选数 5% 登记，
  信息性，§7.1-③）。
- **G-data 判定表**（臂③ v2_panel chain_metrics vs B0' 基线 chain_metrics，Q9-A/Q11-A/Q12-A）：
  ΔCAGR = arm3 − b0p，接受带 ±8pp，超带 = ΔCAGR < −8pp（严格大于、等号带内，C02）；
  改善 >+8pp 登记不精读；ΔSharpe 同向记录不设带；ΔMaxDD 更负 = 劣化，零容忍——
  任何劣化须归因闭合标记才放行，否则 FAIL。放行/拒绝表达式：
  超带（严格大于）+ 可归因 ⇒ 放行并登记；超带 + 不可归因 ⇒ FAIL。
- **回归边**（R2-R1-04）：新复跑批 vs 冻结批 ``data/runs/p2b_t5_closed_20261017_203508``
  成对精确比较（``compare_runs_exact`` 薄包装 + 同一配对投影归一化 + 净值类面
  有限性先验：chain_nav/daily/summary 任一 NaN/Inf ⇒ FAIL）。

模块拆分（R2-T5A-R1 拆分消化，主文件 <1200 行硬上限）：容差/有限性件在
``t5_compare_gates_tolerance.py``，topk 辅证件（评估路径 + 信号路径 + 分组身份
配对核心）在 ``t5_compare_gates_signal.py``；子模块对主文件共享件（
``EXPECTED_SPLITS``/``_fold_name``/``_load_json``/``_project_frame`` 等）一律经
函数内延迟 import 消费（避免模块循环依赖，同 t5_attribution_io 先例）。

配对投影（§7.2 / §13.3.3，**验收投影**——禁止动 ``v2/evidence/fingerprint_keys.py``
通用排除表；正式产物保留原始值）：比较前剔除 ``model_version / registered_at /
data_state_id / batch_id / wf_run_id``。链式指标一律消费既有 ``chain_metrics.json``
（由 ``v2/common/chain_metrics.py`` 唯一实现算出），本工具不自建指标公式。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p2b.compare_merger_batches import (  # noqa: E402
    MergerCompareReport,
    compare_runs_exact,
)
from scripts.v2_p3.t5_compare_gates_signal import (  # noqa: E402
    compare_signal_topk,
    compare_topk_details,
)
from scripts.v2_p3.t5_compare_gates_tolerance import (  # noqa: E402
    NAV_TOLERANCE,
    _check_expected_fold_set,
    _finite_surface_errors,
    _project_daily_within_tolerance,
    _project_summary_within_tolerance,
    compare_chain_nav_tolerance,
    compare_daily_tolerance,
    compare_summary_config_exact,
    compare_summary_tolerance,
)
from src.lazybull.v2.evidence.runs_loader import (  # noqa: E402
    RunsBatch,
    RunsFold,
    load_runs_batch,
)

#: 配对投影归一化键（§7.2/§13.3.3 冻结；验收投影，不动通用指纹排除表）
PAIR_PROJECTION_KEYS: Tuple[str, ...] = (
    "model_version",
    "registered_at",
    "data_state_id",
    "batch_id",
    "wf_run_id",
)
#: 配对投影占位值（两侧归一为同一占位 ⇒ 内容键之外的运行标识不参与比较）
PROJECTION_PLACEHOLDER = "<pair-projection>"
#: G-code 主门预期折集（WF 14 折现役定义，§13.3.3；子模块经函数内延迟 import 消费）
EXPECTED_SPLITS: Tuple[int, ...] = tuple(range(14))
#: G-data ΔCAGR 接受带（Q11-A 冻结 ±8pp；超带判定 = 严格大于、等号带内，C02）
GDATA_CAGR_BAND = Decimal("0.08")
#: 回归边冻结批（R2-R1-04 写死）
REGRESSION_FROZEN_BATCH = Path("data/runs/p2b_t5_closed_20261017_203508")


# ---------------------------------------------------------------------------
# 通用小件
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """证据时间戳（秒级，同 t05b 证据骨架口径）。"""
    return datetime.now().isoformat(timespec="seconds")


def _fold_name(split: int) -> str:
    return f"split{split:02d}"


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    """读取 JSON object；文件缺失返回 None（缺文件属判定输入，由调用方定成败）。"""
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return doc


def _resolve_runs_batch_dir(path: Path) -> Path:
    """解析 runs 批次目录：直指批次目录（含 batch_meta.json）或臂目录（含唯一 runs/<批次>）。"""
    path = Path(path)
    if (path / "batch_meta.json").exists():
        return path
    runs_root = path / "runs"
    if runs_root.is_dir():
        candidates = sorted(d for d in runs_root.iterdir() if d.is_dir())
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(f"{runs_root} 下 runs 批次须唯一，实际 {[d.name for d in candidates]}")
    raise FileNotFoundError(f"无法解析 runs 批次目录: {path}")


def load_chain_metrics(run_dir: Path) -> Dict[str, object]:
    """读取 ``<run_dir>/chain_metrics.json``（由 chain_metrics 唯一实现算出，本工具只消费）。"""
    path = Path(run_dir) / "chain_metrics.json"
    doc = _load_json(path)
    if doc is None:
        raise FileNotFoundError(f"缺 chain_metrics.json: {path}")
    return doc


def write_verdict_json(verdict: Dict[str, object], out: Path) -> None:
    """判定结果 JSON 落盘（--out 指定路径，含各门明细与判据口径说明）。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# G-code ① 主门：逐折预测指纹 + 全批汇总指纹逐位一致
# ---------------------------------------------------------------------------


def _load_fold_fingerprint(arm_dir: Path, fold: str) -> Optional[Dict[str, object]]:
    """读取 ``folds/<fold>/fingerprint.json``；缺文件返回 None，结构非法抛错。"""
    doc = _load_json(Path(arm_dir) / "folds" / fold / "fingerprint.json")
    if doc is not None and "fingerprint" not in doc:
        raise ValueError(f"指纹文件缺 fingerprint 键: {arm_dir}/folds/{fold}/fingerprint.json")
    return doc


def _is_sha256_hex(value: object) -> bool:
    """合法 sha256 十六进制摘要形态（64 位小写 hex；t05b ``sha256_text`` 全摘要口径）。"""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _check_batch_fold_consistency(
    batch_doc: Dict[str, object], fold_fps: Dict[str, str], label: str
) -> List[str]:
    """批次汇总 folds 映射与逐折指纹文件的内部自洽核验（输入身份校验，非判据修订）。"""
    folds_map = batch_doc.get("folds")
    if not isinstance(folds_map, dict):
        return [f"{label} batch_fingerprint.json 缺 folds 映射"]
    errors = []
    for name, fingerprint in sorted(fold_fps.items()):
        registered = folds_map.get(name)
        if registered != fingerprint:
            errors.append(
                f"{label} 批次汇总 folds.{name} 与逐折指纹文件不自洽: "
                f"汇总={registered} vs 逐折={fingerprint}"
            )
    return errors


def compare_excluded_days(
    arm_a: Path,
    arm_b: Path,
    *,
    label_a: str = "臂①",
    label_b: str = "臂②",
) -> Dict[str, object]:
    """闭市排除日臂间子门（CLAUDE §5 验收契约的机器证据）。

    逐折读取两侧 ``fingerprint.json`` 的 ``excluded_days`` 与
    ``excluded_day_reasons``；集合不等/某侧缺文件/字段非法 ⇒ FAIL 并点名折号；
    两侧完全一致 ⇒ PASS，逐折值写入 verdict 供审计。禁止只依赖覆盖门间接保障
    （R1-P2-1）。
    """
    errors: List[str] = []
    folds: List[Dict[str, object]] = []
    for split in EXPECTED_SPLITS:
        name = _fold_name(split)
        doc_a = _load_fold_fingerprint(Path(arm_a), name)
        doc_b = _load_fold_fingerprint(Path(arm_b), name)
        if doc_a is None or doc_b is None:
            errors.append(
                f"缺折指纹 {name}（{label_a}={'有' if doc_a else '无'} / "
                f"{label_b}={'有' if doc_b else '无'}）"
            )
            continue
        days_a = doc_a.get("excluded_days") or []
        days_b = doc_b.get("excluded_days") or []
        if not isinstance(days_a, list) or not isinstance(days_b, list):
            errors.append(f"折 {name} excluded_days 字段非法（须 list）")
            continue
        sorted_a = sorted({str(day) for day in days_a})
        sorted_b = sorted({str(day) for day in days_b})
        reasons_a = doc_a.get("excluded_day_reasons") or {}
        reasons_b = doc_b.get("excluded_day_reasons") or {}
        reasons_match = (
            isinstance(reasons_a, dict) and isinstance(reasons_b, dict) and (reasons_a == reasons_b)
        )
        match = sorted_a == sorted_b and reasons_match
        folds.append(
            {
                "fold": name,
                "match": match,
                "excluded_days_a": sorted_a,
                "excluded_days_b": sorted_b,
                "reasons_match": reasons_match,
            }
        )
        if sorted_a != sorted_b:
            errors.append(
                f"折 {name} 闭市排除日集合不一致: {label_a}={sorted_a} vs {label_b}={sorted_b}"
            )
        elif not reasons_match:
            errors.append(f"折 {name} 闭市排除日原因映射不一致")
    return {
        "verdict": "FAIL" if errors else "PASS",
        "criterion": "逐折 excluded_days 集合与 excluded_day_reasons 完全一致（闭市窄口径）",
        "folds": folds,
        "errors": errors,
    }


def _compare_batch_fingerprints(
    arm1_dir: Path,
    arm2_dir: Path,
    fps_a: Dict[str, str],
    fps_b: Dict[str, str],
) -> Tuple[Dict[str, object], List[str]]:
    """全批汇总指纹逐位一致 + 两臂 folds 映射内部自洽。空摘要不可过（R2-T5A-03-③）：
    ``batch_fingerprint`` 缺字段 / None / 非法 64 位 sha256 hex ⇒ FAIL（双侧同缺同非法
    亦 FAIL，禁止双 None 误判 PASS）。"""
    doc_a = _load_json(Path(arm1_dir) / "batch_fingerprint.json")
    doc_b = _load_json(Path(arm2_dir) / "batch_fingerprint.json")
    result: Dict[str, object] = {
        "match": None,
        "batch_fingerprint_a": None,
        "batch_fingerprint_b": None,
    }
    if doc_a is None or doc_b is None:
        return result, [
            f"缺批次汇总指纹文件（臂①={'有' if doc_a else '无'} / 臂②={'有' if doc_b else '无'}）"
        ]
    fp_a, fp_b = doc_a.get("batch_fingerprint"), doc_b.get("batch_fingerprint")
    valid_a, valid_b = _is_sha256_hex(fp_a), _is_sha256_hex(fp_b)
    result.update(
        match=bool(valid_a and valid_b and fp_a == fp_b),
        batch_fingerprint_a=fp_a,
        batch_fingerprint_b=fp_b,
    )
    errors = []
    if not valid_a:
        errors.append(f"臂① batch_fingerprint.json 缺 batch_fingerprint 字段或值非法: {fp_a!r}")
    if not valid_b:
        errors.append(f"臂② batch_fingerprint.json 缺 batch_fingerprint 字段或值非法: {fp_b!r}")
    if valid_a and valid_b and fp_a != fp_b:
        errors.append(f"全批汇总指纹不一致: 臂①={fp_a} vs 臂②={fp_b}")
    errors += _check_batch_fold_consistency(doc_a, fps_a, "臂①")
    errors += _check_batch_fold_consistency(doc_b, fps_b, "臂②")
    return result, errors


def compare_gcode_fingerprints(arm1_dir: Path, arm2_dir: Path) -> Dict[str, object]:
    """G-code ① 主门：14 折预测指纹逐位一致 + 全批汇总指纹逐位一致（§13.3.3）。

    折不一致 / 缺文件 ⇒ FAIL 指出折号；指纹值先验合法 64 位 sha256 hex 再比较
    （非法 ⇒ FAIL，空摘要不可过）；汇总不一致或汇总/逐折不自洽 ⇒ FAIL。
    """
    arm1_dir, arm2_dir = Path(arm1_dir), Path(arm2_dir)
    errors: List[str] = []
    fold_results: List[Dict[str, object]] = []
    fps_a: Dict[str, str] = {}
    fps_b: Dict[str, str] = {}
    for split in EXPECTED_SPLITS:
        name = _fold_name(split)
        doc_a = _load_fold_fingerprint(arm1_dir, name)
        doc_b = _load_fold_fingerprint(arm2_dir, name)
        if doc_a is None or doc_b is None:
            errors.append(
                f"缺折指纹 {name}（臂①={'有' if doc_a else '无'} / 臂②={'有' if doc_b else '无'}）"
            )
            continue
        value_a, value_b = doc_a["fingerprint"], doc_b["fingerprint"]
        if not _is_sha256_hex(value_a) or not _is_sha256_hex(value_b):
            errors.append(
                f"折 {name} 指纹值非法（须 64 位 sha256 hex）: 臂①={value_a!r} / 臂②={value_b!r}"
            )
            continue
        fps_a[name] = str(value_a)
        fps_b[name] = str(value_b)
        match = fps_a[name] == fps_b[name]
        fold_results.append(
            {
                "fold": name,
                "match": match,
                "fingerprint_a": fps_a[name],
                "fingerprint_b": fps_b[name],
                "n_days_a": doc_a.get("n_days"),
                "n_days_b": doc_b.get("n_days"),
                "n_rows_a": doc_a.get("n_rows"),
                "n_rows_b": doc_b.get("n_rows"),
            }
        )
        if not match:
            errors.append(
                f"折 {name} 预测指纹不一致: 臂①={fps_a[name][:16]}… vs 臂②={fps_b[name][:16]}…"
            )
    batch_result, batch_errors = _compare_batch_fingerprints(arm1_dir, arm2_dir, fps_a, fps_b)
    errors += batch_errors
    return {
        "verdict": "FAIL" if errors else "PASS",
        "expected_folds": len(EXPECTED_SPLITS),
        "fold_results": fold_results,
        "batch_fingerprint": batch_result,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# 配对投影（验收投影：仅作用于比较副本，正式产物保留原始值）
# ---------------------------------------------------------------------------


def _project_frame(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """帧副本上把配对投影键整列归一为占位值（键不存在则跳过）。"""
    projected = frame.copy()
    for key in keys:
        if key in projected.columns:
            projected[key] = PROJECTION_PLACEHOLDER
    return projected


def apply_pair_projection(
    batch: RunsBatch, keys: Sequence[str] = PAIR_PROJECTION_KEYS
) -> RunsBatch:
    """验收配对投影（§7.2/§13.3.3）：summary 列、逐折帧列、逐折 _meta 键三处归一。

    返回新 ``RunsBatch``（原对象不变）；``batch_id/wf_run_id`` 另由
    ``compare_runs_exact`` 内建归一化兜底（逐名身份规范化先例），此处一并投影。
    """
    folds: Dict[int, RunsFold] = {}
    for split, fold in batch.folds.items():
        meta = dict(fold.meta)
        for key in keys:
            if key in meta:
                meta[key] = PROJECTION_PLACEHOLDER
        frames = {
            name: (
                _project_frame(getattr(fold, name), keys)
                if getattr(fold, name) is not None
                else None
            )
            for name in ("trades", "daily", "attribution", "holdings_snapshot", "topk_detail")
        }
        folds[split] = RunsFold(split_index=fold.split_index, meta=meta, **frames)  # type: ignore
    return replace(batch, summary=_project_frame(batch.summary, keys), folds=folds)


# ---------------------------------------------------------------------------
# runs 批次成对比较（G-code ② + 回归边共用薄包装；容差/有限性件在
# t5_compare_gates_tolerance，topk 辅证件在 t5_compare_gates_signal）
# ---------------------------------------------------------------------------


def _compare_runs_pair(
    batch_a: RunsBatch, batch_b: RunsBatch, edge: str
) -> Tuple[List[str], List[str]]:
    """``compare_runs_exact`` 薄包装：配对投影已先行，模式键/子集豁免一律不用。"""
    report = MergerCompareReport()
    compare_runs_exact(
        batch_a,
        batch_b,
        report,
        edge,
        migration_only_values={},
        shared_expected_values={},
        normalize_data_state=False,
        allow_subset=False,
    )
    return list(report.errors), list(report.notes)


def compare_gcode_runs(arm1_dir: Path, arm2_dir: Path) -> Dict[str, object]:
    """G-code ②：**成交逐笔精确 + 净值类 1e-6 容差**（§7.1-② 冻结判据的如实实现）。

    流程（容差核验先行 + 容差内投影 + 其余产物精确）：折集交叉核验（R2-T5A-03-①：
    两臂恰 14 折，缺一/多一指出折号）⇒ 有限性先验（R2-T5A-07：NaN/Inf ⇒ FAIL，
    非「容差内」）⇒ **summary 配置列逐位精确核验**（R2-T5A-R1-02：允许清单外
    ``bt_*`` 浮点配置列，任何投影之前执行，含 <1e-6 微差 ⇒ FAIL）⇒ 三面容差核验
    （chain_nav.nav / summary 绩效统计允许清单列 / 逐折 daily nav、total_value）⇒
    三面皆对齐且容差内 ⇒ 对有逐位差的面把臂②投影到臂①取值（各面 max|Δ| 登记）；
    任一面超容差 ⇒ 不投影、FAIL（仍跑精确比较收集完整差异清单）；最后复用
    ``compare_runs_exact`` 精确比较全部产物（其余产物逐位精确）。
    """
    dir_a = _resolve_runs_batch_dir(arm1_dir)
    dir_b = _resolve_runs_batch_dir(arm2_dir)
    batch_a = apply_pair_projection(load_runs_batch(dir_a))
    batch_b = apply_pair_projection(load_runs_batch(dir_b))
    errors: List[str] = []
    notes: List[str] = []
    errors += _check_expected_fold_set(batch_a, "臂①")
    errors += _check_expected_fold_set(batch_b, "臂②")
    finite_errors = _finite_surface_errors(batch_a, "臂①") + _finite_surface_errors(batch_b, "臂②")
    errors += finite_errors
    config_check = compare_summary_config_exact(batch_a.summary, batch_b.summary)
    errors += config_check["errors"]  # type: ignore[operator]
    nav_check = compare_chain_nav_tolerance(batch_a.chain_nav, batch_b.chain_nav)
    summary_check = compare_summary_tolerance(batch_a.summary, batch_b.summary)
    daily_check = compare_daily_tolerance(batch_a.folds, batch_b.folds)
    tolerance_surfaces = {
        "chain_nav": nav_check,
        "summary": summary_check,
        "daily": daily_check,
    }
    tolerance_ok = not finite_errors and all(
        check["aligned"] and check["within_tolerance"] for check in tolerance_surfaces.values()
    )
    if tolerance_ok:
        if nav_check["max_abs_diff"] > 0.0:
            batch_b = replace(batch_b, chain_nav=batch_a.chain_nav)
            notes.append(
                f"链式净值逐位差 max|Δ|={nav_check['max_abs_diff']:.3e} ≤ {NAV_TOLERANCE}，"
                "按 G-code ② 容差口径投影对齐（判据 = 1e-6 容差，非逐位）"
            )
        if summary_check["max_abs_diff"] > 0.0:
            batch_b = _project_summary_within_tolerance(
                batch_a, batch_b, summary_check["columns"]  # type: ignore[arg-type]
            )
            notes.append(
                f"summary 绩效统计列逐位差 max|Δ|={summary_check['max_abs_diff']:.3e} ≤ "
                f"{NAV_TOLERANCE}，按同一容差口径投影对齐（列={summary_check['columns']}）"
            )
        if daily_check["max_abs_diff"] > 0.0:
            batch_b = _project_daily_within_tolerance(batch_a, batch_b)
            notes.append(
                f"逐折 daily 净值列逐位差 max|Δ|={daily_check['max_abs_diff']:.3e} ≤ "
                f"{NAV_TOLERANCE}，按同一容差口径投影对齐（nav/total_value）"
            )
    cmp_errors, cmp_notes = _compare_runs_pair(batch_a, batch_b, edge="gcode_runs")
    errors += cmp_errors
    notes += cmp_notes
    for surface, check in tolerance_surfaces.items():
        if not check["within_tolerance"]:
            errors.append(f"{surface} 1e-6 容差比较失败: {check['errors']}")
    return {
        "verdict": "FAIL" if errors else "PASS",
        "batch_a": dir_a.name,
        "batch_b": dir_b.name,
        "criterion": "成交逐笔精确 + 净值类 1e-6 容差（§7.1-② 冻结判据）：容差面 = "
        "chain_nav.nav + summary 绩效统计允许清单列（fingerprint_keys.EXCLUDE_EXACT 的 "
        "bt_* 统计列 ∩ 浮点）+ 逐折 daily nav/total_value；允许清单外 bt_* 配置列"
        "投影前逐位精确（R2-T5A-R1-02）；trades/attribution/holdings_snapshot 等其余"
        "产物逐位精确；NaN/Inf ⇒ FAIL",
        "nav_tolerance_check": nav_check,
        "summary_tolerance_check": summary_check,
        "daily_tolerance_check": daily_check,
        "summary_config_exact_check": config_check,
        "errors": errors,
        "notes": notes,
    }


def compare_regression_edge(
    new_runs_dir: Path,
    frozen_runs_dir: Path = REGRESSION_FROZEN_BATCH,
) -> Dict[str, object]:
    """回归边（R2-R1-04）：新复跑批 vs 冻结批成对精确比较（薄包装 + 配对投影归一化）。

    逐折 trades/attribution/holdings_snapshot/daily 全列行序精确 + summary + chain_nav。
    有限性先验（R2-T5A-07）：净值类列任一 NaN/Inf ⇒ FAIL（NaN==NaN 视等的精确
    口径不得被 NaN 绕过）。
    """
    new_dir = _resolve_runs_batch_dir(new_runs_dir)
    frozen_dir = _resolve_runs_batch_dir(frozen_runs_dir)
    batch_new = apply_pair_projection(load_runs_batch(new_dir))
    batch_frozen = apply_pair_projection(load_runs_batch(frozen_dir))
    errors, notes = _compare_runs_pair(batch_new, batch_frozen, edge="regression_edge")
    finite_errors = _finite_surface_errors(batch_new, "新复跑批") + _finite_surface_errors(
        batch_frozen, "冻结批"
    )
    errors = finite_errors + errors
    return {
        "gate": "regression_edge",
        "verdict": "FAIL" if errors else "PASS",
        "generated_at": _now_iso(),
        "new_batch": str(new_dir),
        "frozen_batch": str(frozen_dir),
        "projection_keys": list(PAIR_PROJECTION_KEYS),
        "criterion": "两 runs 批次成对精确比较（R2-R1-04）：配对投影归一化后逐折 "
        "trades/attribution/holdings_snapshot/daily 全列行序精确 + summary + chain_nav；"
        "净值类面有限性先验（chain_nav/daily/summary 任一 NaN/Inf ⇒ FAIL）",
        "errors": errors,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# G-data 判定表（Q9-A/Q11-A/Q12-A 冻结口径）
# ---------------------------------------------------------------------------


def _require_metric(chain: Dict[str, object], key: str, label: str) -> float:
    """判定表输入校验：指标必须存在、非空且有限（缺/非有限 ⇒ 输入契约违反）。"""
    value = chain.get(key)
    if value is None:
        raise ValueError(f"{label} chain_metrics 缺指标 {key}（判定表输入契约）")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"{label} chain_metrics.{key} 非有限值: {value!r}")
    return result


def _judge_cagr(delta: Decimal, attribution_closed: bool) -> Dict[str, object]:
    """ΔCAGR 分支（Q11-A/C02）：带内 ⇒ 放行；>+8pp 改善 ⇒ 登记不精读；
    <−8pp 超带 ⇒ 归因闭合放行并登记、否则拒绝。等号 = 带内（严格大于才超带）。"""
    if delta < -GDATA_CAGR_BAND:
        branch = "overshoot_attributed" if attribution_closed else "overshoot_unattributed"
        return {"branch": branch, "passed": attribution_closed, "overshoot": True}
    if delta > GDATA_CAGR_BAND:
        return {"branch": "improvement", "passed": True, "overshoot": False}
    return {"branch": "within_band", "passed": True, "overshoot": False}


def _judge_maxdd(delta: Decimal, attribution_closed: bool) -> Dict[str, object]:
    """ΔMaxDD 分支（零容忍 + 归因出口）：更负即劣化，须归因闭合才放行，否则拒绝。"""
    if delta < 0:
        branch = "deterioration_attributed" if attribution_closed else "deterioration_unattributed"
        return {"branch": branch, "passed": attribution_closed, "deterioration": True}
    return {"branch": "no_deterioration", "passed": True, "deterioration": False}


def _gdata_registrations(
    cagr_judgment: Dict[str, object],
    maxdd_judgment: Dict[str, object],
    delta_cagr: Decimal,
    delta_maxdd: Decimal,
) -> List[str]:
    """判定表登记项（放行并登记 / 改善登记不精读，均为信息性记录）。"""
    registrations: List[str] = []
    if cagr_judgment["branch"] == "improvement":
        registrations.append(f"ΔCAGR = {float(delta_cagr) * 100:+.2f}pp > +8pp：改善登记不精读")
    elif cagr_judgment["branch"] == "overshoot_attributed":
        registrations.append(
            f"ΔCAGR = {float(delta_cagr) * 100:+.2f}pp 超带 + 归因闭合 ⇒ 放行并登记"
        )
    if maxdd_judgment["branch"] == "deterioration_attributed":
        registrations.append(
            f"ΔMaxDD = {float(delta_maxdd) * 100:+.2f}pp 劣化 + 归因闭合 ⇒ 放行并登记"
        )
    return registrations


def judge_gdata_table(
    arm3_chain: Dict[str, object],
    b0p_chain: Dict[str, object],
    *,
    maxdd_attribution_closed: bool = False,
    cagr_attribution_closed: bool = False,
) -> Dict[str, object]:
    """G-data 判定表（§13.3.6，Q9-A/Q11-A/Q12-A 冻结口径；判据本身禁止修订）。

    ΔCAGR = arm3.cagr − b0p.cagr，接受带 ±8pp；超带 = ΔCAGR < −8pp（严格大于，
    等号带内，C02）；改善 >+8pp 登记不精读。ΔSharpe 同向记录不设带。
    ΔMaxDD（更负 = 劣化）：零容忍——任何劣化须归因闭合标记才放行，否则 FAIL。
    放行/拒绝表达式：超带（严格大于）+ 可归因 ⇒ 放行并登记；超带 + 不可归因 ⇒ FAIL。
    boundary_semantics 注记：边界比较经 ``Decimal(str(...))`` 精确化（恰好 −8pp 带内）。
    """
    cagr_a = _require_metric(arm3_chain, "cagr", "臂③")
    cagr_b = _require_metric(b0p_chain, "cagr", "B0'")
    mdd_a = _require_metric(arm3_chain, "max_drawdown", "臂③")
    mdd_b = _require_metric(b0p_chain, "max_drawdown", "B0'")
    sharpe_a = _require_metric(arm3_chain, "sharpe", "臂③")
    sharpe_b = _require_metric(b0p_chain, "sharpe", "B0'")

    delta_cagr = Decimal(str(cagr_a)) - Decimal(str(cagr_b))
    delta_maxdd = Decimal(str(mdd_a)) - Decimal(str(mdd_b))
    cagr_judgment = _judge_cagr(delta_cagr, cagr_attribution_closed)
    maxdd_judgment = _judge_maxdd(delta_maxdd, maxdd_attribution_closed)
    passed = bool(cagr_judgment["passed"] and maxdd_judgment["passed"])
    return {
        "verdict": "PASS" if passed else "FAIL",
        "inputs": {
            "arm3": {"cagr": cagr_a, "max_drawdown": mdd_a, "sharpe": sharpe_a},
            "b0p": {"cagr": cagr_b, "max_drawdown": mdd_b, "sharpe": sharpe_b},
            "maxdd_attribution_closed": maxdd_attribution_closed,
            "cagr_attribution_closed": cagr_attribution_closed,
        },
        "deltas": {
            "cagr": float(delta_cagr),
            "max_drawdown": float(delta_maxdd),
            "sharpe": sharpe_a - sharpe_b,
        },
        "cagr_judgment": {"band_pp": float(GDATA_CAGR_BAND * 100), **cagr_judgment},
        "maxdd_judgment": maxdd_judgment,
        "sharpe_note": "ΔSharpe 同向记录、不设带（Q11-A）",
        "registrations": _gdata_registrations(
            cagr_judgment, maxdd_judgment, delta_cagr, delta_maxdd
        ),
        "boundary_semantics": "恰好 ΔCAGR = −8pp 为带内（等号带内，C02）；"
        "ΔCAGR < −8pp（严格大于）为超带；ΔMaxDD 更负即劣化，零容忍 + 归因出口。"
        "cagr_attribution_closed 仅在 CAGR 超带分支参与放行判定；within_band 分支"
        "该标记不改变 passed（R1-P3-2 登记口径）",
    }


# ---------------------------------------------------------------------------
# 门级 verdict 组装 + CLI
# ---------------------------------------------------------------------------


def build_gcode_verdict(arm1_dir: Path, arm2_dir: Path) -> Dict[str, object]:
    """G-code 总 verdict：① 指纹主门 + ② runs 比较 + ③ 双路 topk 辅证，全 PASS 才 PASS。"""
    arm1_dir, arm2_dir = Path(arm1_dir), Path(arm2_dir)
    sub_gates = {
        "fingerprint": compare_gcode_fingerprints(arm1_dir, arm2_dir),
        "runs": compare_gcode_runs(arm1_dir, arm2_dir),
        "excluded_days": compare_excluded_days(arm1_dir, arm2_dir, label_a="臂①", label_b="臂②"),
        "topk_auxiliary": compare_topk_details(arm1_dir, arm2_dir),
        "signal_topk_auxiliary": compare_signal_topk(arm1_dir, arm2_dir),
    }
    passed = all(gate["verdict"] == "PASS" for gate in sub_gates.values())
    return {
        "gate": "gcode",
        "verdict": "PASS" if passed else "FAIL",
        "generated_at": _now_iso(),
        "arm_a": str(arm1_dir),
        "arm_b": str(arm2_dir),
        "projection_keys": list(PAIR_PROJECTION_KEYS),
        "criterion": "G-code（Q2-A）：① 逐折指纹 14/14 + 全批汇总逐位一致（指纹值先验 "
        "合法 sha256，空摘要不可过）；② runs 成交逐笔精确 + 净值类 1e-6 容差"
        "（chain_nav.nav / summary 绩效统计允许清单列 / daily nav/total_value；允许清单外"
        " bt_* 配置列投影前逐位精确；折集恰 14 折交叉核验；NaN/Inf ⇒ FAIL）；"
        "③ 辅证 = 评估路径 topk（双臂全折同缺 = FAIL）+ 信号路径 top-k 对比"
        "（signal_topk.parquet 真实信号通路产物，增键/sha256/rows 核验 + 完整性门，"
        "零辅证 = FAIL），并列互换 >5% 登记",
        "sub_gates": sub_gates,
    }


def build_gdata_verdict(
    arm3_dir: Path,
    b0p_dir: Path,
    *,
    maxdd_attribution_closed: bool = False,
    cagr_attribution_closed: bool = False,
) -> Dict[str, object]:
    """G-data 总 verdict：读取两侧 chain_metrics.json 后跑判定表（指标由他人算，本工具只消费）。"""
    arm3_dir, b0p_dir = Path(arm3_dir), Path(b0p_dir)
    table = judge_gdata_table(
        load_chain_metrics(arm3_dir),
        load_chain_metrics(b0p_dir),
        maxdd_attribution_closed=maxdd_attribution_closed,
        cagr_attribution_closed=cagr_attribution_closed,
    )
    excluded = compare_excluded_days(arm3_dir, b0p_dir, label_a="臂③", label_b="B0'")
    passed = table["verdict"] == "PASS" and excluded["verdict"] == "PASS"
    return {
        "gate": "gdata",
        "generated_at": _now_iso(),
        "arm3": str(arm3_dir),
        "b0p": str(b0p_dir),
        "criterion": "G-data 判定表（Q9-A/Q11-A/Q12-A）：臂③ vs B0'（臂①冻结）；"
        "ΔCAGR ±8pp（严格大于、等号带内）；ΔSharpe 记录不设带；ΔMaxDD 零容忍 + 归因出口；"
        "闭市排除日逐折集合须完全一致（R1-P2-1）",
        "verdict": "PASS" if passed else "FAIL",
        "excluded_days": excluded,
        "errors": list(excluded["errors"]),
        **{key: value for key, value in table.items() if key != "verdict"},
    }


def _build_parser() -> argparse.ArgumentParser:
    """CLI 参数：gcode / gdata / regression-edge 三子命令 + --run-dir-a/--run-dir-b/--out。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gcode = sub.add_parser("gcode", help="G-code 等价门：臂① legacy vs 臂② v2_cs_train_verify")
    p_gcode.add_argument("--run-dir-a", required=True, help="臂① 运行目录")
    p_gcode.add_argument("--run-dir-b", required=True, help="臂② 运行目录")
    p_gcode.add_argument("--out", required=True, help="verdict JSON 输出路径")

    p_gdata = sub.add_parser("gdata", help="G-data 判定表：臂③ v2_panel vs B0' 基线")
    p_gdata.add_argument("--run-dir-a", required=True, help="臂③ 运行目录（含 chain_metrics.json）")
    p_gdata.add_argument("--run-dir-b", required=True, help="B0' 基线目录（含 chain_metrics.json）")
    p_gdata.add_argument(
        "--maxdd-attribution-closed", action="store_true", help="ΔMaxDD 劣化归因已闭合"
    )
    p_gdata.add_argument(
        "--cagr-attribution-closed", action="store_true", help="ΔCAGR 超带归因已闭合"
    )
    p_gdata.add_argument("--out", required=True, help="verdict JSON 输出路径")

    p_reg = sub.add_parser("regression-edge", help="回归边：新复跑批 vs 冻结批成对精确比较")
    p_reg.add_argument("--run-dir-a", required=True, help="新复跑 runs 批次目录（或运行目录）")
    p_reg.add_argument(
        "--run-dir-b",
        default=str(REGRESSION_FROZEN_BATCH),
        help=f"冻结批目录（默认 {REGRESSION_FROZEN_BATCH}）",
    )
    p_reg.add_argument("--out", required=True, help="verdict JSON 输出路径")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口：verdict != PASS ⇒ exit 1；判定结果 JSON 一律落盘 --out。"""
    args = _build_parser().parse_args(argv)
    if args.command == "gcode":
        verdict = build_gcode_verdict(Path(args.run_dir_a), Path(args.run_dir_b))
    elif args.command == "gdata":
        verdict = build_gdata_verdict(
            Path(args.run_dir_a),
            Path(args.run_dir_b),
            maxdd_attribution_closed=args.maxdd_attribution_closed,
            cagr_attribution_closed=args.cagr_attribution_closed,
        )
    else:
        verdict = compare_regression_edge(Path(args.run_dir_a), Path(args.run_dir_b))
    write_verdict_json(verdict, Path(args.out))
    logger.info(f"{args.command} 判定 verdict={verdict['verdict']}，结果已落盘: {args.out}")
    if verdict["verdict"] != "PASS":
        for message in verdict.get("errors") or []:  # type: ignore[union-attr]
            logger.error(f"  {message}")
        sub_gates = verdict.get("sub_gates")
        if isinstance(sub_gates, dict):
            for name, gate in sub_gates.items():
                for message in gate.get("errors") or []:
                    logger.error(f"  [{name}] {message}")
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
