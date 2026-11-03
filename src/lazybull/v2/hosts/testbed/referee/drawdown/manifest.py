# -*- coding: utf-8 -*-
"""P5b T5 回撤仪器 manifest：§5.5 回撤冻结清单、字段校验与**三环 + 上游互链**。

链结构（缺一环即拒绝运行）
========================
::

    drawdown_manifest.json                # 先冻结，之后不可改（独占创建；lineage.sequence=1）
      ├─ files.calibrator.sha256          # 回撤校准器代码身份（模块字节集 canonical 哈希）
      └─ files.parent_manifest.sha256     # 上游：收益仪器 manifest（545d2be4… 口径）
         files.parent_thresholds.sha256   # 上游：收益临界值产物（af37b217… 口径）
    drawdown_thresholds.json              # 训练产物：记录 manifest/calibrator 哈希
    drawdown_holdout.json / drawdown_power.json   # 记录 manifest/calibrator/thresholds 三个哈希

- 校准器启动即复算自身身份，与 manifest 登记值不符 ⇒ 拒绝运行；
- 上游收益 manifest 与收益临界值必须**逐字节**匹配登记值，且收益 manifest 的回撤段必须仍是
  ``not_implemented_in_t4`` 占位（证明 T5 没有就地改写收益仪器）；
- 任一环被改写（封存不符 / 文件 sha 变化 / 交叉引用不符）⇒ 链校验失败并拒绝运行。

**禁止**在本模块之外改写收益 manifest 或收益产物；本仪器的冻结件另立文件
``docs/plans/v2/p5b_t5_drawdown_manifest_<日期>.json``。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    ANCHOR_HORIZON,
    DIRECTIONS,
    HOLD_H,
    LOOK_DAYS,
    WINDOW_SAMPLES,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.sampling import (
    IMPLEMENTATION_ID,
    NAMESPACE_CODE,
    implementation_ids,
)
from src.lazybull.v2.hosts.testbed.storage import read_json, seal_record, verify_seal

__all__ = [
    "CALIBRATOR_EXCLUSIONS",
    "CALIBRATOR_MODULES",
    "H6_STATUSES",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_SEAL_FIELD",
    "ManifestError",
    "PARENT_DRAWDOWN_PLACEHOLDER",
    "SCALE_FLOOR",
    "calibrator_identity",
    "file_sha256",
    "freeze_manifest",
    "inspect_parent",
    "load_manifest",
    "manifest_sha256",
    "scale_floor_errors",
    "validate_manifest",
    "verify_calibration_chain",
    "verify_upstream",
]

#: manifest schema
MANIFEST_SCHEMA = "p5b_testbed_drawdown_manifest/v1"
#: campaign ``reports/`` 下的文件名
MANIFEST_FILENAME = "drawdown_manifest.json"
#: 封存字段（复用试车场记录封存口径）
MANIFEST_SEAL_FIELD = "record_sha256"
#: 回撤校准器身份覆盖的模块（决定数值与冻结文字的全部模块）
CALIBRATOR_MODULES: Tuple[str, ...] = (
    "mdd.py",
    "crash_window.py",
    "injection.py",
    "sampling.py",
    "realizations.py",
    "thresholds.py",
    "holdout.py",
    "power.py",
    "artifacts.py",
    "manifest.py",
    "sections.py",
    "calibration.py",
)
#: 未纳入回撤校准器身份的模块与理由（显式登记，不得默认沉默）
CALIBRATOR_EXCLUSIONS: Mapping[str, str] = {
    "report.py": "呈现层与哈希链链头；其 sha 由 CHANGELOG 与 T5 闭合报告人工登记",
    "__init__.py": "仅导出面，无统计或冻结逻辑",
    "referee/statistics.py": "上游共享层（§5.1 look 映射），由收益校准器身份负责",
    "referee/scenes.py": "上游共享层（四类场景 DGP 与 circular block），由收益校准器身份负责",
    "referee/holdout.py": "上游共享层（Wilson 区间实现），由收益校准器身份负责",
    "testbed/storage.py": "上游共享层（封存/规范化序列化），由 T1/T2 冻结链负责",
    "testbed/paths.py": "上游共享层（写域守卫），由 T1/T2 冻结链负责",
}
#: 实现数下限（§5.3 表 + 任务书：不得低于 T4 已冻结口径，只能上调）
SCALE_FLOOR: Mapping[str, int] = {
    "train_per_unit": 2000,
    "holdout_per_unit": 10000,
    "effect_per_scene_per_level": 2000,
}
#: 收益仪器 manifest 中回撤段的占位状态（T5 不得就地改写）
PARENT_DRAWDOWN_PLACEHOLDER = "not_implemented_in_t4"
#: H6 占位状态白名单（T5 不运行 H6）
H6_STATUSES = ("placeholder_not_run", "not_run_in_t5")
#: 偏差登记条目的必填字段
DEVIATION_FIELDS: Tuple[str, ...] = ("item", "design_value", "frozen_value", "reason")
#: 必登记的偏差项（缺一即校验失败，防"静默替换"）
REQUIRED_DEVIATIONS: Tuple[str, ...] = (
    "命名种子",
    "回撤仪器独立冻结",
    "崩盘窗窗口口径",
    "A/A 独立路径",
    "效应档实现数",
    "纯注入效应目标",
    "含噪观察效应",
)
#: 必须存在的字段路径（§5.5 回撤行 + §7.2 报告身份）
_REQUIRED_PATHS: Tuple[str, ...] = (
    "schema",
    "run_id",
    "status",
    "scope",
    "evidence.evidence_class",
    "evidence.virtual_clock",
    "evidence.not_forward_evidence",
    "evidence.not_sla_evidence",
    "evidence.no_ledger_entry",
    "evidence.row_domain",
    "evidence.schedule_mode",
    "drawdown_referee.mdd.definition",
    "drawdown_referee.mdd.log_space",
    "drawdown_referee.mdd.start_point_included",
    "drawdown_referee.mdd.percent_conversion",
    "drawdown_referee.reading.formula",
    "drawdown_referee.reading.base_arm",
    "drawdown_referee.reading.sign_convention",
    "drawdown_referee.reading.horizons",
    "drawdown_referee.reading.H",
    "drawdown_referee.reading.K",
    "drawdown_referee.reading.t_anchor",
    "drawdown_referee.reading.independent_horizon_per_look",
    "drawdown_referee.reading.improve_negation_forbidden",
    "drawdown_referee.reading.full_chain_used",
    "drawdown_referee.windows.window_samples",
    "drawdown_referee.windows.max_windows",
    "drawdown_referee.windows.selection_rule",
    "drawdown_referee.windows.tie_break",
    "drawdown_referee.windows.boundary_rule",
    "drawdown_referee.windows.fixed_for_all_realizations",
    "drawdown_referee.windows.selection_uses_only_b0",
    "drawdown_referee.windows.frozen_list",
    "drawdown_referee.injection.dd_improve",
    "drawdown_referee.injection.dd_worsen",
    "drawdown_referee.injection.positive_days_unchanged",
    "drawdown_referee.injection.outside_window_unchanged",
    "drawdown_referee.injection.reconstruction",
    "drawdown_referee.counterfactual.noise",
    "drawdown_referee.counterfactual.arm",
    "drawdown_referee.counterfactual.target",
    "drawdown_referee.counterfactual.observed",
    "drawdown_referee.counterfactual.shared_injection_across_looks",
    "drawdown_referee.counterfactual.independent_parameter_solve",
    "drawdown_referee.counterfactual.window_reselection_forbidden",
    "drawdown_referee.solver.scan_points",
    "drawdown_referee.solver.max_refine_steps",
    "drawdown_referee.solver.tolerance_pp",
    "drawdown_referee.solver.feasible_rule",
    "drawdown_referee.solver.multi_solution_rule",
    "drawdown_referee.solver.not_constructible_semantics",
    "drawdown_referee.threshold.worsen_rule",
    "drawdown_referee.threshold.improve_rule",
    "drawdown_referee.threshold.alpha_train",
    "drawdown_referee.threshold.alpha_per_look",
    "drawdown_referee.threshold.quantile_interpolation",
    "drawdown_referee.threshold.worsen_event",
    "drawdown_referee.threshold.improve_event",
    "drawdown_referee.threshold.improve_is_d_k",
    "drawdown_referee.threshold.no_cross_offsetting",
    "drawdown_referee.aa.independent_paths",
    "drawdown_referee.aa.same_path_zero_injection_forbidden",
    "drawdown_referee.aa.arm",
    "drawdown_referee.aa.baseline",
    "drawdown_referee.aa.unit_definition",
    "drawdown_referee.aa.shared_across_directions",
    "drawdown_referee.holdout.unit_definition",
    "drawdown_referee.holdout.wilson_confidence",
    "drawdown_referee.holdout.wilson_upper_max",
    "drawdown_referee.holdout.pooled_or_weighted_average_forbidden",
    "drawdown_referee.holdout.per_direction_naming",
    "drawdown_referee.effect.grid_pp",
    "drawdown_referee.effect.directions",
    "drawdown_referee.effect.target_power",
    "drawdown_referee.effect.mde_names",
    "drawdown_referee.effect.mde_definition",
    "drawdown_referee.effect.main_denominator",
    "drawdown_referee.effect.constructibility_rate_field",
    "drawdown_referee.effect.units",
    "drawdown_referee.effect.display_conversion",
    "drawdown_referee.effect.annualization",
    "scenes",
    "calibration.dgp.sd_daily",
    "calibration.dgp.path_days",
    "calibration.dgp.ar_rho",
    "calibration.dgp.regime_block_length",
    "calibration.dgp.regime_multipliers",
    "calibration.dgp.block_length",
    "block_resample.method",
    "block_resample.block_length",
    "block_resample.center",
    "block_resample.truncation_forbidden",
    "block_resample.cross_pool_sampling_forbidden",
    "counts.train_per_unit",
    "counts.holdout_per_unit",
    "counts.effect_per_scene_per_level",
    "counts.seed",
    "counts.realization_prefix",
    "substreams",
    "source_pair.source_run_id",
    "source_pair.base_chain_nav",
    "source_pair.noise_chain_nav",
    "source_pair.base_sha256",
    "source_pair.noise_sha256",
    "source_pair.noise_arm_note",
    "source_pair.train_folds",
    "source_pair.holdout_folds",
    "source_pair.date_blocks",
    "source_pair.pool_isolation",
    "source_window.window.samples",
    "source_window.window.mdd_percent",
    "source_window.window.usable_for_calibration",
    "source_window.window.full_chain",
    "files.calibrator.implementation_id",
    "files.calibrator.sha256",
    "files.calibrator.modules",
    "files.calibrator.exclusions",
    "files.parent_manifest.path",
    "files.parent_manifest.sha256",
    "files.parent_manifest.run_id",
    "files.parent_manifest.lineage_sequence",
    "files.parent_thresholds.path",
    "files.parent_thresholds.sha256",
    "h6_numeric.status",
    "deviation_registry",
    "limitations",
    "lineage.sequence",
)


class ManifestError(TestbedError):
    """manifest 缺失/字段不全/口径不符/哈希链断裂（一律 fail-closed）。"""

    __test__ = False


def file_sha256(path: Path) -> str:
    """文件字节 sha256（哈希链唯一口径）。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_sha256(path: Path) -> str:
    """manifest 文件字节 sha256（语义化别名）。"""
    return file_sha256(path)


def calibrator_identity(
    module_dir: Optional[Path] = None,
    *,
    implementation_id: str = IMPLEMENTATION_ID,
    modules: Sequence[str] = CALIBRATOR_MODULES,
) -> Dict[str, Any]:
    """回撤校准器代码身份 = ``{implementation_id, modules, sha256}``。"""
    base = Path(module_dir) if module_dir is not None else Path(__file__).resolve().parent
    digests: Dict[str, str] = {}
    for name in modules:
        path = base / name
        if not path.is_file():
            raise ManifestError(f"回撤裁判模块缺失（无法计算校准器身份）: {path}")
        digests[name] = file_sha256(path)
    payload = {"implementation_id": str(implementation_id), "modules": digests}
    combined = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"implementation_id": str(implementation_id), "modules": digests, "sha256": combined}


def _get(doc: Mapping[str, Any], path: str) -> Any:
    node: Any = doc
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _missing_fields(doc: Mapping[str, Any]) -> List[str]:
    missing: List[str] = []
    for path in _REQUIRED_PATHS:
        try:
            _get(doc, path)
        except KeyError:
            missing.append(path)
    return missing


def _check_evidence(doc: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    evidence = doc["evidence"]
    expected = {
        "evidence_class": "calibration_only",
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "no_ledger_entry": True,
        "row_domain": "structural",
        "schedule_mode": "replay",
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            errors.append(f"evidence.{key} 必须是 {value!r}，实际 {evidence.get(key)!r}")
    if str(doc.get("scope")) != "historical_testbed":
        errors.append(f"scope 必须是 historical_testbed，实际 {doc.get('scope')!r}")
    if str(doc.get("status")) != "frozen_before_evidence":
        errors.append(f"status 必须是 frozen_before_evidence，实际 {doc.get('status')!r}")
    return errors


def _check_reading(reading: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if int(reading.get("H", -1)) != HOLD_H:
        errors.append(f"reading.H 必须是 {HOLD_H}，实际 {reading.get('H')!r}")
    horizons = [int(day) for day in reading.get("horizons") or []]
    if horizons != list(LOOK_DAYS):
        errors.append(f"reading.horizons 必须是 {list(LOOK_DAYS)}，实际 {horizons}")
    if int(reading.get("K", -1)) != len(LOOK_DAYS):
        errors.append(f"reading.K 必须是 {len(LOOK_DAYS)}，实际 {reading.get('K')!r}")
    if int(reading.get("t_anchor", -1)) != ANCHOR_HORIZON:
        errors.append(f"reading.t_anchor 必须是 {ANCHOR_HORIZON}，实际 {reading.get('t_anchor')!r}")
    anchor_from_blocks = HOLD_H * (ANCHOR_HORIZON // HOLD_H)
    if int(reading.get("t_anchor", -1)) != anchor_from_blocks:
        errors.append(f"reading.t_anchor 必须等于 H*J_max={anchor_from_blocks}（唯一锚口径）")
    for key in (
        "independent_horizon_per_look",
        "improve_negation_forbidden",
    ):
        if reading.get(key) is not True:
            errors.append(f"reading.{key} 必须为 true")
    if reading.get("full_chain_used") is not False:
        errors.append("reading.full_chain_used 必须为 false（1714 全链不参与校准）")
    return errors


def _check_windows(windows: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if int(windows.get("window_samples", -1)) != WINDOW_SAMPLES:
        errors.append(
            f"windows.window_samples 必须是 {WINDOW_SAMPLES}，实际 {windows.get('window_samples')!r}"
        )
    if int(windows.get("max_windows", -1)) != 3:
        errors.append(f"windows.max_windows 必须是 3，实际 {windows.get('max_windows')!r}")
    if windows.get("fixed_for_all_realizations") is not True:
        errors.append("windows.fixed_for_all_realizations 必须为 true")
    if windows.get("selection_uses_only_b0") is not True:
        errors.append("windows.selection_uses_only_b0 必须为 true")
    if windows.get("full_chain_used_for_selection") is not False:
        errors.append("windows.full_chain_used_for_selection 必须为 false")
    frozen = list(windows.get("frozen_list") or [])
    if not frozen:
        errors.append("windows.frozen_list 不能为空（B0 无回撤 ⇒ 回撤仪器 not_constructible）")
        return errors
    if len(frozen) > 3:
        errors.append(f"windows.frozen_list 最多 3 个，实际 {len(frozen)}")
    errors.extend(_check_frozen_list(frozen))
    return errors


def _check_frozen_list(frozen: Sequence[Mapping[str, Any]]) -> List[str]:
    """逐窗口字段/边界/互不重叠/升序（拆出以控制单函数复杂度）。"""
    errors: List[str] = []
    spans: List[Tuple[int, int]] = []
    for index, item in enumerate(frozen):
        try:
            peak = int(item["peak"])
            trough = int(item["trough"])
            depth = float(item["depth_pp"])
            samples = int(item["window_samples"])
            start = int(item["window_start_ordinal"])
            stop = int(item["window_stop_ordinal"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"windows.frozen_list[{index}] 字段非法: {exc!r}")
            continue
        if not (0 <= peak < trough <= WINDOW_SAMPLES):
            errors.append(f"windows.frozen_list[{index}] 峰谷越界: peak={peak} trough={trough}")
        if start != peak or stop != trough:
            errors.append(
                f"windows.frozen_list[{index}] 窗口必须是 peak..trough（实际 {start}..{stop}）"
            )
        if samples != trough - peak:
            errors.append(f"windows.frozen_list[{index}] window_samples 与峰谷不符")
        if not (depth < 0.0):
            errors.append(f"windows.frozen_list[{index}].depth_pp 必须为负，实际 {depth}")
        if any(not (trough < left or right < peak) for left, right in spans):
            errors.append(f"windows.frozen_list[{index}] 与已登记窗口重叠（必须互不重叠）")
        spans.append((peak, trough))
    peaks = [int(item.get("peak", -1)) for item in frozen]
    if peaks != sorted(peaks):
        errors.append("windows.frozen_list 必须按 peak 升序登记")
    return errors


def _check_injection(injection: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    improve = dict(injection.get("dd_improve") or {})
    worsen = dict(injection.get("dd_worsen") or {})
    for name, entry, low, high in (
        ("dd_improve", improve, 0.0, 1.0),
        ("dd_worsen", worsen, 0.0, 2.0),
    ):
        bounds = [float(item) for item in entry.get("range") or []]
        if bounds != [low, high]:
            errors.append(f"injection.{name}.range 必须是 {[low, high]}，实际 {bounds}")
        if not str(entry.get("formula") or ""):
            errors.append(f"injection.{name}.formula 不能为空")
        if not str(entry.get("parameter") or ""):
            errors.append(f"injection.{name}.parameter 不能为空")
    if str(improve.get("parameter")) != "lambda":
        errors.append("injection.dd_improve.parameter 必须是 'lambda'（λ）")
    if str(worsen.get("parameter")) != "zeta":
        errors.append("injection.dd_worsen.parameter 必须是 'zeta'（ζ）")
    if improve.get("target_sign") != "positive":
        errors.append("injection.dd_improve.target_sign 必须是 'positive'")
    if worsen.get("target_sign") != "negative":
        errors.append("injection.dd_worsen.target_sign 必须是 'negative'")
    for key in ("positive_days_unchanged", "outside_window_unchanged"):
        if injection.get(key) is not True:
            errors.append(f"injection.{key} 必须为 true")
    return errors


def _check_solver(solver: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if int(solver.get("scan_points", -1)) != 101:
        errors.append(f"solver.scan_points 必须是 101，实际 {solver.get('scan_points')!r}")
    if int(solver.get("max_refine_steps", -1)) != 80:
        errors.append(f"solver.max_refine_steps 必须是 80，实际 {solver.get('max_refine_steps')!r}")
    if not math.isclose(float(solver.get("tolerance_pp", -1.0)), 0.05, rel_tol=0.0, abs_tol=1e-12):
        errors.append(f"solver.tolerance_pp 必须是 0.05，实际 {solver.get('tolerance_pp')!r}")
    for key in ("truncation_forbidden", "report_target_as_achieved_forbidden"):
        if solver.get(key) is not True:
            errors.append(f"solver.{key} 必须为 true")
    if "最小" not in str(solver.get("multi_solution_rule") or ""):
        errors.append("solver.multi_solution_rule 必须写明取最小干预量")
    if "not_constructible" not in str(solver.get("not_constructible_semantics") or ""):
        errors.append("solver.not_constructible_semantics 必须写明 not_constructible 语义")
    return errors


def _check_threshold(threshold: Mapping[str, Any], k: int) -> List[str]:
    errors: List[str] = []
    alpha = float(threshold.get("alpha_train", 0.0))
    per_look = float(threshold.get("alpha_per_look", 0.0))
    if not (0.0 < alpha < 1.0):
        errors.append(f"threshold.alpha_train 必须在 (0,1)，实际 {alpha}")
    if k <= 0 or not math.isclose(per_look, alpha / k, rel_tol=1e-12, abs_tol=0.0):
        errors.append(f"threshold.alpha_per_look({per_look}) 必须等于 alpha_train/K({alpha}/{k})")
    if str(threshold.get("quantile_interpolation")) != "linear":
        errors.append("threshold.quantile_interpolation 必须是 linear（分位口径冻结）")
    if threshold.get("improve_is_d_k") is not True:
        errors.append("threshold.improve_is_d_k 必须为 true（I_k = D_k，不得再取负）")
    if threshold.get("no_cross_offsetting") is not True:
        errors.append("threshold.no_cross_offsetting 必须为 true")
    if not str(threshold.get("worsen_event") or "") or not str(
        threshold.get("improve_event") or ""
    ):
        errors.append("threshold.worsen_event/improve_event 不能为空")
    if str(threshold.get("worsen_event")) == str(threshold.get("improve_event")):
        errors.append("两个裁判的事件名不得相同（必须分别命名）")
    if "min" not in str(threshold.get("worsen_conservative") or ""):
        errors.append("threshold.worsen_conservative 必须写明 min over scene")
    if "max" not in str(threshold.get("improve_conservative") or ""):
        errors.append("threshold.improve_conservative 必须写明 max over scene")
    return errors


def _check_gates(doc: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    aa = dict(doc["drawdown_referee"].get("aa") or {})
    if aa.get("independent_paths") is not True:
        errors.append("aa.independent_paths 必须为 true（A/A 用独立路径）")
    if aa.get("same_path_zero_injection_forbidden") is not True:
        errors.append("aa.same_path_zero_injection_forbidden 必须为 true")
    for key in ("arm", "baseline"):
        if not str(aa.get(key) or "").strip():
            errors.append(f"aa.{key} 必须写明生成方式（A/A 与效应单元的减数结构要对齐）")
    if "B0" not in str(aa.get("baseline") or ""):
        errors.append("aa.baseline 必须写明以 B0 冻结基准臂为减数")
    holdout = dict(doc["drawdown_referee"].get("holdout") or {})
    if float(holdout.get("wilson_upper_max", 1.0)) > 0.06:
        errors.append("holdout.wilson_upper_max 不得放宽到 0.06 以上")
    if not math.isclose(float(holdout.get("wilson_confidence", 0.0)), 0.95, abs_tol=1e-12):
        errors.append("holdout.wilson_confidence 必须为 0.95")
    if holdout.get("pooled_or_weighted_average_forbidden") is not True:
        errors.append("holdout.pooled_or_weighted_average_forbidden 必须为 true")
    naming = dict(holdout.get("per_direction_naming") or {})
    if set(naming.values()) != set(DIRECTIONS):
        errors.append(f"holdout.per_direction_naming 必须恰为 {list(DIRECTIONS)}")
    return errors


def _check_effect(effect: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    grid = [float(item) for item in effect.get("grid_pp") or []]
    expected = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    if grid != expected:
        errors.append(f"effect.grid_pp 必须是 {expected}，实际 {grid}")
    if list(effect.get("directions") or []) != list(DIRECTIONS):
        errors.append(f"effect.directions 必须恰为 {list(DIRECTIONS)}")
    if not math.isclose(float(effect.get("target_power", 0.0)), 0.80, abs_tol=1e-12):
        errors.append("effect.target_power 必须为 0.80")
    names = dict(effect.get("mde_names") or {})
    if names.get("worsen") != "MDE_drawdown_worsen":
        errors.append("effect.mde_names.worsen 必须是 MDE_drawdown_worsen")
    if names.get("improve") != "MDE_drawdown_improve":
        errors.append("effect.mde_names.improve 必须是 MDE_drawdown_improve")
    if "M" not in str(effect.get("main_denominator") or ""):
        errors.append("effect.main_denominator 必须写明分母为全部实现 M")
    if "C/M" not in str(effect.get("constructibility_rate_field") or ""):
        errors.append("effect.constructibility_rate_field 必须写明 C/M")
    annual = dict(effect.get("annualization") or {})
    if annual.get("applicable") is not False:
        errors.append("effect.annualization.applicable 必须为 false（pp 不折算收益/δ）")
    return errors


def _dgp_range(doc: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    dgp = dict(doc["calibration"]["dgp"])
    if float(dgp["sd_daily"]) <= 0:
        errors.append("calibration.dgp.sd_daily 必须为正")
    if int(dgp["block_length"]) != HOLD_H:
        errors.append(f"calibration.dgp.block_length 必须等于 H={HOLD_H}")
    if int(dgp["path_days"]) != WINDOW_SAMPLES:
        errors.append(
            f"calibration.dgp.path_days 必须等于窗口样本数 {WINDOW_SAMPLES}"
            "（噪声路径与 B0 窗口逐日对齐）"
        )
    if not (0.0 <= float(dgp["ar_rho"]) < 1.0):
        errors.append("calibration.dgp.ar_rho 必须在 [0,1)")
    if len(dgp["regime_multipliers"]) != 2:
        errors.append("calibration.dgp.regime_multipliers 必须是两个倍数")
    block = dict(doc["block_resample"])
    if str(block.get("method")) != "circular":
        errors.append(f"block_resample.method 必须是 circular，实际 {block.get('method')!r}")
    if int(block.get("block_length", -1)) != HOLD_H:
        errors.append(f"block_resample.block_length 必须等于 H={HOLD_H}")
    for key in ("truncation_forbidden", "cross_pool_sampling_forbidden"):
        if block.get(key) is not True:
            errors.append(f"block_resample.{key} 必须为 true")
    if "mean" not in str(block.get("center") or ""):
        errors.append("block_resample.center 必须声明按源序列自身均值去中心")
    return errors


def _counts_errors(counts: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in SCALE_FLOOR:
        try:
            value = int(counts[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"counts.{key} 缺失或非整数")
            continue
        if value <= 0:
            errors.append(f"counts.{key} 必须为正，实际 {value}")
    if not isinstance(counts.get("seed"), int):
        errors.append("counts.seed 必须是整数（命名种子根）")
    if not str(counts.get("realization_prefix") or ""):
        errors.append("counts.realization_prefix 不能为空")
    return errors


def _check_substreams(substreams: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    declared = dict(substreams or {})
    expected = implementation_ids()
    if declared.get("implementation_id") != expected["implementation_id"]:
        errors.append(
            f"substreams.implementation_id 必须是 {expected['implementation_id']!r}，"
            f"实际 {declared.get('implementation_id')!r}"
        )
    if int(declared.get("namespace_code", -1)) != NAMESPACE_CODE:
        errors.append(f"substreams.namespace_code 必须是 {NAMESPACE_CODE}")
    if dict(declared.get("direction_codes") or {}) != expected["direction_codes"]:
        errors.append("substreams.direction_codes 必须与实现一致")
    if dict(declared.get("stages") or {}) != expected["stages"]:
        errors.append("substreams.stages 必须与实现一致")
    independence = dict(declared.get("independence") or {})
    for key in (
        "train_holdout_disjoint",
        "independent_of_income_referee",
        "aa_uses_independent_paths_not_zero_injection",
    ):
        if independence.get(key) is not True:
            errors.append(f"substreams.independence.{key} 必须为 true")
    return errors


def _check_source_pair(pair: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    train = {int(x) for x in pair.get("train_folds") or []}
    holdout = {int(x) for x in pair.get("holdout_folds") or []}
    if not train or not holdout:
        errors.append("source_pair.train_folds/holdout_folds 都不能为空")
    elif train & holdout:
        errors.append(f"训练/holdout 源折重叠（必须互斥）: {sorted(train & holdout)}")
    blocks = pair.get("date_blocks") or {}
    if list(blocks.get("key") or []) != ["split_index", "date"]:
        errors.append("source_pair.date_blocks.key 必须是 ['split_index','date']")
    if blocks.get("date_semantics") != "fold_local_ordinal":
        errors.append("source_pair.date_blocks.date_semantics 必须是 fold_local_ordinal")
    if not str(pair.get("source_run_id") or ""):
        errors.append("source_pair.source_run_id 不能为空")
    for key in ("base_sha256", "noise_sha256"):
        if len(str(pair.get(key) or "")) != 64:
            errors.append(f"source_pair.{key} 必须是 64 位 sha256")
    note = str(pair.get("noise_arm_note") or "")
    if "P3" not in note or "500" not in note:
        errors.append("source_pair.noise_arm_note 必须明确 B0' 不是 P3 的 500 树重训臂")
    isolation = dict(pair.get("pool_isolation") or {})
    if isolation.get("cross_pool_sampling_forbidden") is not True:
        errors.append("source_pair.pool_isolation.cross_pool_sampling_forbidden 必须为 true")
    return errors


def _check_source_window(window: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    inner = dict(window.get("window") or {})
    if int(inner.get("samples", -1)) != WINDOW_SAMPLES:
        errors.append(f"source_window.window.samples 必须是 {WINDOW_SAMPLES}")
    try:
        mdd = float(inner["mdd_percent"])
    except (KeyError, TypeError, ValueError):
        errors.append("source_window.window.mdd_percent 缺失或非数值")
    else:
        if not math.isfinite(mdd) or mdd >= 0.0:
            errors.append(f"source_window.window.mdd_percent 必须是有限负值，实际 {mdd}")
    if inner.get("usable_for_calibration") is not True:
        errors.append("source_window.window.usable_for_calibration 必须为 true")
    history = dict(inner.get("full_chain") or {})
    if history.get("usable_for_calibration") is not False:
        errors.append(
            "source_window.window.full_chain.usable_for_calibration 必须为 false"
            "（全链 1714 日不参与校准）"
        )
    if int(history.get("window_samples", 0)) <= WINDOW_SAMPLES:
        errors.append("source_window.window.full_chain.window_samples 必须长于冻结窗口")
    return errors


def _check_files(files: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    calibrator = dict(files.get("calibrator") or {})
    if calibrator.get("implementation_id") != IMPLEMENTATION_ID:
        errors.append(f"files.calibrator.implementation_id 必须是 {IMPLEMENTATION_ID!r}")
    if len(str(calibrator.get("sha256") or "")) != 64:
        errors.append("files.calibrator.sha256 必须是 64 位 sha256")
    if set(calibrator.get("modules") or {}) != set(CALIBRATOR_MODULES):
        errors.append(f"files.calibrator.modules 必须恰为 {list(CALIBRATOR_MODULES)}")
    exclusions = dict(calibrator.get("exclusions") or {})
    if set(exclusions) != set(CALIBRATOR_EXCLUSIONS):
        errors.append(
            "files.calibrator.exclusions 必须逐项登记未纳入身份的模块与理由"
            f"（期望 {list(CALIBRATOR_EXCLUSIONS)}）"
        )
    if "report.py" not in str(exclusions):
        errors.append("files.calibrator.exclusions 必须登记 report.py 的排除理由")
    for name in ("parent_manifest", "parent_thresholds"):
        entry = dict(files.get(name) or {})
        if len(str(entry.get("sha256") or "")) != 64:
            errors.append(f"files.{name}.sha256 必须是 64 位 sha256")
        if not str(entry.get("path") or ""):
            errors.append(f"files.{name}.path 不能为空")
    parent = dict(files.get("parent_manifest") or {})
    if not str(parent.get("run_id") or ""):
        errors.append("files.parent_manifest.run_id 不能为空")
    if int(parent.get("lineage_sequence", 0)) < 1:
        errors.append("files.parent_manifest.lineage_sequence 必须 >= 1")
    return errors


def _check_deviations(registry: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(registry, list) or not registry:
        return ["deviation_registry 必须是非空列表（登记与设计/冻结值的已知差异）"]
    items = set()
    for index, entry in enumerate(registry):
        if not isinstance(entry, Mapping):
            errors.append(f"deviation_registry[{index}] 必须是对象")
            continue
        for field in DEVIATION_FIELDS:
            if not str(entry.get(field) or "").strip():
                errors.append(f"deviation_registry[{index}].{field} 缺失或为空")
        if str(entry.get("item") or "").strip():
            items.add(str(entry["item"]).strip())
    missing = [name for name in REQUIRED_DEVIATIONS if name not in items]
    if missing:
        errors.append(f"deviation_registry 缺少必登记项: {missing}")
    return errors


def _check_lineage(lineage: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(lineage, Mapping):
        return ["lineage 必须是对象（登记冻结序号与废止前身）"]
    try:
        sequence = int(lineage.get("sequence"))
    except (TypeError, ValueError):
        return ["lineage.sequence 必须是正整数"]
    if sequence < 1:
        return [f"lineage.sequence 必须 >= 1，实际 {sequence}"]
    superseded = lineage.get("superseded_manifest_sha256")
    superseded_calibrator = lineage.get("superseded_calibrator_sha256")
    reason = str(lineage.get("supersede_reason") or "").strip()
    if sequence == 1:
        for key, value in (
            ("superseded_manifest_sha256", superseded),
            ("superseded_calibrator_sha256", superseded_calibrator),
            ("supersede_reason", reason or None),
        ):
            if value not in (None, ""):
                errors.append(f"首次冻结（sequence=1）不得登记 {key}")
        return errors
    for key, value in (
        ("superseded_manifest_sha256", superseded),
        ("superseded_calibrator_sha256", superseded_calibrator),
    ):
        if len(str(value or "")) != 64:
            errors.append(f"重冻必须登记 {key}（64 位 sha256）")
    if len(reason) < 8:
        errors.append("重冻必须登记 supersede_reason（>=8 字，说明为何重冻）")
    return errors


def validate_manifest(doc: Mapping[str, Any]) -> List[str]:
    """结构与口径一致性校验（不含实现数下限）。返回错误列表（``[]`` = 可执行）。"""
    errors: List[str] = []
    missing = _missing_fields(doc)
    if missing:
        return [f"缺字段 {path}" for path in missing]
    if doc.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"schema 非法: {doc.get('schema')!r}（期望 {MANIFEST_SCHEMA}）")
    referee = doc["drawdown_referee"]
    reading = referee["reading"]
    errors.extend(_check_evidence(doc))
    errors.extend(_check_reading(reading))
    errors.extend(_check_windows(referee["windows"]))
    errors.extend(_check_injection(referee["injection"]))
    errors.extend(_check_solver(referee["solver"]))
    errors.extend(_check_threshold(referee["threshold"], int(reading.get("K", 0))))
    errors.extend(_check_gates(doc))
    errors.extend(_check_effect(referee["effect"]))
    if set(doc.get("scenes") or {}) != {"ma20", "ar1_06", "regime_hetero", "real_seed_block"}:
        errors.append("scenes 必须恰为四类依赖场景（禁止删除困难场景）")
    errors.extend(_dgp_range(doc))
    errors.extend(_counts_errors(doc["counts"]))
    errors.extend(_check_substreams(doc["substreams"]))
    errors.extend(_check_source_pair(doc["source_pair"]))
    errors.extend(_check_source_window(doc["source_window"]))
    errors.extend(_check_files(doc["files"]))
    if str(doc["h6_numeric"].get("status")) not in H6_STATUSES:
        errors.append("h6_numeric.status 必须是占位状态（H6 单条演练不在 T5 范围）")
    errors.extend(_check_deviations(doc.get("deviation_registry")))
    if not isinstance(doc.get("limitations"), list) or len(doc["limitations"]) < 3:
        errors.append("limitations 至少 3 条（§5.6 限制声明）")
    errors.extend(_check_lineage(doc.get("lineage")))
    return errors


def scale_floor_errors(counts: Mapping[str, Any]) -> List[str]:
    """实现数下限校验（只能上调不得下调）。"""
    errors: List[str] = []
    for key, floor in SCALE_FLOOR.items():
        try:
            value = int(counts[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"counts.{key} 缺失或非整数，无法核对下限 {floor}")
            continue
        if value < floor:
            errors.append(f"counts.{key}={value} 低于冻结下限 {floor}（禁止下调）")
    return errors


def inspect_parent(
    parent_manifest_path: Path,
    parent_thresholds_path: Path,
    *,
    expected_manifest_sha256: Optional[str] = None,
    expected_thresholds_sha256: Optional[str] = None,
    require_scale_floor: bool = True,
    verify_calibrator: bool = True,
) -> Dict[str, Any]:
    """读取并核验上游收益仪器（manifest + 临界值），返回其身份与源对（冻结阶段用）。

    Raises:
        ManifestError: 文件缺失/哈希不符/回撤段被改写/收益仪器链断裂。
    """
    from src.lazybull.v2.hosts.testbed.referee.manifest import load_manifest as load_income

    parent_path = Path(parent_manifest_path)
    thresholds_path = Path(parent_thresholds_path)
    if not parent_path.is_file():
        raise ManifestError(f"上游收益仪器 manifest 缺失（拒绝运行）: {parent_path}")
    parent_sha = file_sha256(parent_path)
    if expected_manifest_sha256 is not None and parent_sha != str(expected_manifest_sha256):
        raise ManifestError(
            f"上游收益 manifest 哈希与期望不符（收益证据已变，拒绝运行）: 实际={parent_sha} "
            f"期望={expected_manifest_sha256}"
        )
    income_doc = load_income(
        parent_path,
        expected_sha256=parent_sha,
        require_scale_floor=require_scale_floor,
        verify_calibrator=verify_calibrator,
    )
    placeholder = str((income_doc.get("drawdown_referee") or {}).get("status"))
    if placeholder != PARENT_DRAWDOWN_PLACEHOLDER:
        raise ManifestError(
            "上游收益 manifest 的回撤段已被改写（T5 不得就地改写收益仪器）: "
            f"status={placeholder!r}"
        )
    if not thresholds_path.is_file():
        raise ManifestError(f"上游收益临界值产物缺失（拒绝运行）: {thresholds_path}")
    thresholds_sha = file_sha256(thresholds_path)
    if expected_thresholds_sha256 is not None and thresholds_sha != str(expected_thresholds_sha256):
        raise ManifestError(
            f"上游收益临界值哈希与期望不符（拒绝运行）: 实际={thresholds_sha} "
            f"期望={expected_thresholds_sha256}"
        )
    thresholds_doc = read_json(thresholds_path, what="上游收益临界值产物")
    if str(thresholds_doc.get("manifest_sha256")) != parent_sha:
        raise ManifestError("上游收益临界值产物与收益 manifest 不互链（manifest_sha256 不符）")
    return {
        "parent_manifest": {"path": parent_path.as_posix(), "sha256": parent_sha},
        "parent_thresholds": {"path": thresholds_path.as_posix(), "sha256": thresholds_sha},
        "parent_run_id": str(income_doc.get("run_id")),
        "parent_lineage_sequence": int((income_doc.get("lineage") or {}).get("sequence", 0)),
        "parent_drawdown_status": placeholder,
        "parent_source_pair": dict(income_doc["source_pair"]),
        "parent_calibration": dict(income_doc["calibration"]),
        "parent_counts": dict(income_doc["counts"]),
    }


def verify_upstream(
    doc: Mapping[str, Any],
    *,
    repo_root: Path,
    require_scale_floor: bool = True,
    verify_calibrator: bool = True,
) -> Dict[str, Any]:
    """核验上游收益仪器、哈希登记与源对一致性（任一不符即拒绝运行）。

    额外语义检查：收益 manifest 的回撤段必须仍是 T4 占位
    （``status=not_implemented_in_t4``）——若被就地改写则说明收益仪器已被污染，拒绝运行。

    Raises:
        ManifestError: 上游文件缺失/哈希不符/回撤段被改写/源对不一致/收益仪器链断裂。
    """
    root = Path(repo_root)
    files = dict(doc["files"])
    parent = dict(files["parent_manifest"])
    thresholds = dict(files["parent_thresholds"])
    info = inspect_parent(
        root / str(parent["path"]),
        root / str(thresholds["path"]),
        expected_manifest_sha256=str(parent["sha256"]),
        expected_thresholds_sha256=str(thresholds["sha256"]),
        require_scale_floor=require_scale_floor,
        verify_calibrator=verify_calibrator,
    )
    ours = dict(doc["source_pair"])
    theirs = dict(info["parent_source_pair"])
    for key in ("base_sha256", "noise_sha256", "base_chain_nav", "noise_chain_nav"):
        if str(ours.get(key)) != str(theirs.get(key)):
            raise ManifestError(
                f"回撤仪器与收益仪器的源对不一致（{key}）: {ours.get(key)!r} vs {theirs.get(key)!r}"
            )
    payload = {
        "parent_manifest": dict(info["parent_manifest"]),
        "parent_thresholds": dict(info["parent_thresholds"]),
        "parent_run_id": info["parent_run_id"],
        "parent_lineage_sequence": info["parent_lineage_sequence"],
        "parent_drawdown_status": info["parent_drawdown_status"],
        "source_pair_match": True,
    }
    return payload


def freeze_manifest(doc: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    """冻结 manifest（校验 + 封存 + **独占创建**，已存在即拒绝）。"""
    errors = validate_manifest(doc)
    if errors:
        raise ManifestError("manifest 不可冻结（校验失败）: " + "; ".join(errors))
    target = Path(path)
    if target.exists():
        raise ManifestError(f"manifest 已存在（冻结件不可重写）: {target}")
    sealed = seal_record(
        {key: value for key, value in dict(doc).items() if key != MANIFEST_SEAL_FIELD}
    )
    payload = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
    except FileExistsError as exc:
        raise ManifestError(f"manifest 已存在（冻结件不可重写）: {target}") from exc
    return {
        "path": target.as_posix(),
        "sha256": manifest_sha256(target),
        "seal": sealed[MANIFEST_SEAL_FIELD],
    }


def load_manifest(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    require_scale_floor: bool = True,
    verify_calibrator: bool = True,
    module_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """读取并核验回撤 manifest（封存 + 字段 + 口径 + 实现数 + 校准器身份）。"""
    source = Path(path)
    if not source.is_file():
        raise ManifestError(f"回撤 manifest 缺失（未冻结不得取证）: {source}")
    actual = manifest_sha256(source)
    if expected_sha256 is not None and actual != str(expected_sha256):
        raise ManifestError(
            f"manifest 文件哈希与期望不符（被改写或指向错文件）: {source} "
            f"实际={actual} 期望={expected_sha256}"
        )
    doc = read_json(source, what="回撤 manifest")
    try:
        verify_seal(doc, what=f"回撤 manifest（{source}）")
    except TestbedError as exc:
        raise ManifestError(f"回撤 manifest 封存校验失败: {exc}") from exc
    errors = validate_manifest(doc)
    if errors:
        raise ManifestError(f"回撤 manifest 校验失败（{source}）: " + "; ".join(errors))
    if require_scale_floor:
        floor_errors = scale_floor_errors(doc["counts"])
        if floor_errors:
            raise ManifestError(
                "回撤 manifest 实现数低于冻结下限（禁止下调）: " + "; ".join(floor_errors)
            )
    if verify_calibrator:
        expected = calibrator_identity(module_dir)
        recorded = dict(doc["files"]["calibrator"])
        if recorded.get("sha256") != expected["sha256"]:
            raise ManifestError(
                "回撤裁判代码身份与 manifest 登记不符（代码已变，必须重新冻结）: "
                f"实际={expected['sha256']} 登记={recorded.get('sha256')}"
            )
        if dict(recorded.get("modules") or {}) != expected["modules"]:
            raise ManifestError("files.calibrator.modules 与当前回撤模块字节不符（代码已变）")
    return doc


def verify_calibration_chain(
    *,
    manifest_doc: Mapping[str, Any],
    manifest_file_sha256: str,
    calibrator: Optional[Mapping[str, Any]] = None,
    thresholds: Optional[Mapping[str, Any]] = None,
    thresholds_sha256: Optional[str] = None,
    holdout: Optional[Mapping[str, Any]] = None,
    power: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """回撤三环（manifest ↔ 校准器 ↔ 临界值）与 holdout/power 的哈希互链校验。"""
    identity = dict(calibrator) if calibrator is not None else calibrator_identity()
    checks: List[Dict[str, Any]] = []

    def record(name: str, actual: Any, expected: Any) -> None:
        checks.append(
            {"link": name, "actual": actual, "expected": expected, "ok": actual == expected}
        )

    record("manifest:calibrator", manifest_doc["files"]["calibrator"]["sha256"], identity["sha256"])
    for label, artifact in (("thresholds", thresholds), ("holdout", holdout), ("power", power)):
        if artifact is None:
            continue
        record(f"{label}:manifest", artifact.get("manifest_sha256"), manifest_file_sha256)
        record(f"{label}:calibrator", artifact.get("calibrator_sha256"), identity["sha256"])
    if thresholds is not None:
        if thresholds_sha256 is None:
            raise ManifestError("校验 holdout/power 与临界值的互链必须提供 thresholds_sha256")
        for label, artifact in (("holdout", holdout), ("power", power)):
            if artifact is None:
                continue
            record(f"{label}:thresholds", artifact.get("thresholds_sha256"), thresholds_sha256)
    failed = [check for check in checks if not check["ok"]]
    if failed:
        raise ManifestError(
            "回撤哈希链断裂（拒绝运行）: "
            + "; ".join(
                f"{item['link']} 实际={item['actual']} 期望={item['expected']}" for item in failed
            )
        )
    return {"checks": checks, "ok": True}
