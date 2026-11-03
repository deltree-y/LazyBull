# -*- coding: utf-8 -*-
"""P5b T4 校准 manifest：§5.5 全表冻结、字段校验与**哈希互链**。

三环链（缺一环即拒绝运行）
==========================
::

    calibration_manifest.json           # 先冻结，之后不可改（独占创建）
      └─ files.calibrator.sha256        # = 裁判代码身份（模块字节集 canonical 哈希）
    calibration_thresholds.json         # 训练产物：记录 manifest_sha256 + calibrator_sha256
      └─ thresholds_sha256
    income_holdout.json / income_power.json   # 记录 manifest/calibrator/thresholds 三个哈希

- 校准器启动即复算自身身份，与 manifest 登记值不符 ⇒ 拒绝运行（代码改动必须重新冻结）；
- 训练产物只能写一次（``artifacts.write_json_once``），holdout 只读训练产物、不得回写临界值；
- 任一环被改写（封存不符 / 文件 sha 变化 / 交叉引用不符）⇒ 链校验失败并拒绝运行。

manifest 路径约定：冻结时同时写两份**同字节**文件——
仓库冻结件 ``docs/plans/v2/p5b_t4_calibration_manifest_20261102.json``（按 ``AGENTS.md`` 属本地
``plans/`` 目录、**不入 git**，供本机评审复读）与 campaign 副本
``<campaign>/reports/calibration_manifest.json``（正式证据，随 campaign 归档）。冻结时两份必须
逐字节一致，否则冻结失败；运行期以 campaign 副本为准，仓库件只作本地对照。若给出
``--repo-manifest``，运行期会额外核验两份文件字节一致（不一致即拒绝运行）。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.sampling import IMPLEMENTATION_ID, implementation_ids
from src.lazybull.v2.hosts.testbed.referee.statistics import (
    ANNUAL_TRADING_DAYS,
    DELTA_ANNUAL_TARGET,
    DELTA_LOG_DAILY,
    MIN_BLOCKS_PER_LOOK,
    MIN_LOOKS,
    REGISTERED_STATISTICS,
    delta_annualized_ok,
    make_look_plan,
)
from src.lazybull.v2.hosts.testbed.storage import read_json, seal_record, verify_seal

__all__ = [
    "CALIBRATOR_MODULES",
    "DRAWDOWN_PLACEHOLDER_STATUS",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_SEAL_FIELD",
    "ManifestError",
    "SCALE_FLOOR",
    "calibrator_identity",
    "file_sha256",
    "freeze_manifest",
    "load_manifest",
    "manifest_sha256",
    "scale_floor_errors",
    "validate_manifest",
    "verify_calibration_chain",
    "verify_source_files",
]

#: manifest 文件名（campaign ``reports/`` 下与仓库冻结件同名）
MANIFEST_FILENAME = "calibration_manifest.json"
#: manifest schema
MANIFEST_SCHEMA = "p5b_testbed_calibration_manifest/v1"
#: 封存字段（复用试车场记录封存口径）
MANIFEST_SEAL_FIELD = "record_sha256"
#: 裁判代码身份覆盖的模块（**收益裁判**实现；T5 回撤裁判另列，互不影响）
#:
#: 覆盖范围 = 决定**注册读数与冻结文字**的全部模块：
#: - 统计/抽样/场景/训练/holdout/功效/账本（数值面）；
#: - ``calibration.py``（counts/seed/prefix 取用、``require_scale_floor`` 传递、产物编排）；
#: - ``manifest.py`` + ``manifest_sections.py``（§5.5 冻结清单、偏差白名单与全部冻结文字）；
#: - ``artifacts.py``（确定性序列化口径，``finite_safe`` 落盘形态）。
#:
#: **不覆盖** ``report.py``：它是呈现层（T5 必须在其上扩展回撤段），且是哈希链的**链头**——
#: 其 sha 由 CHANGELOG 与 T4 闭合报告人工登记（机器链只到 ``verdict_sha256``），改写会使
#: 已存证据无法逐字节复现，故不需要用模块身份再拦一次。
#:
#: **上游共享层不在本集合**：``testbed/storage.py``（``seal_record``/``canonical_json_sha256``）
#: 与 ``testbed/paths.py``（写域守卫）决定产物字节，但属 T1/T2 共享层，由 T1/T2 的冻结链负责；
#: 本 manifest 只登记收益裁判自身的代码身份。
CALIBRATOR_MODULES: Tuple[str, ...] = (
    "statistics.py",
    "sampling.py",
    "scenes.py",
    "training.py",
    "holdout.py",
    "power.py",
    "ledger_source.py",
    "calibration.py",
    "manifest.py",
    "manifest_sections.py",
    "artifacts.py",
)
#: 未纳入校准器身份的模块与理由（必须显式登记，不得默认沉默）
CALIBRATOR_EXCLUSIONS: Mapping[str, str] = {
    "report.py": "呈现层与哈希链链头；T5 需扩展回撤段；其 sha 由 CHANGELOG/闭合报告人工登记",
    "__init__.py": "仅导出面，无统计或冻结逻辑",
    "testbed/storage.py": "上游共享层（封存/规范化序列化），由 T1/T2 冻结链负责",
    "testbed/paths.py": "上游共享层（写域守卫），由 T1/T2 冻结链负责",
}
#: 设计 §5.3 表：实现数下限（只能上调，不得下调）
SCALE_FLOOR: Mapping[str, int] = {
    "train_per_unit": 2000,
    "holdout_per_unit": 10000,
    "effect_per_scene_per_level": 2000,
}
#: 回撤裁判占位状态（T4 不实现，但必须登记）
DRAWDOWN_PLACEHOLDER_STATUS = "not_implemented_in_t4"
#: 偏差登记条目的必填字段
DEVIATION_FIELDS: Tuple[str, ...] = ("item", "design_value", "frozen_value", "reason")
#: 必登记的偏差项（缺一即校验失败，防"静默替换"）
REQUIRED_DEVIATIONS: Tuple[str, ...] = (
    "命名种子",
    "holdout realization 身份",
    "统计量",
    "配对账本样本",
    "冷启动与超 H 依赖场景",
    "B0' 臂口径",
)
#: 必须存在的 manifest 字段路径（§5.5 冻结清单 + §7.2 报告身份）
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
    "income_referee.statistic.name",
    "income_referee.statistic.formula",
    "income_referee.statistic.variance",
    "income_referee.statistic.gamma_denominator",
    "income_referee.statistic.delta_log_daily",
    "income_referee.statistic.delta_annualization",
    "income_referee.statistic.boundary_centering",
    "income_referee.statistic.epsilon_or_clamp_forbidden",
    "income_referee.statistic.zero_variance_policy",
    "income_referee.sample.n_max",
    "income_referee.sample.H",
    "income_referee.sample.look_blocks",
    "income_referee.sample.look_days",
    "income_referee.sample.K",
    "income_referee.sample.min_blocks_per_look",
    "income_referee.threshold.alpha_train",
    "income_referee.threshold.alpha_per_look",
    "income_referee.threshold.rule_steps",
    "income_referee.threshold.veto_event",
    "income_referee.threshold.invalid_rate_per_look_max",
    "income_referee.threshold.min_valid_quantile_sample",
    "income_referee.threshold.holdout_wilson_upper_max",
    "income_referee.effect.grid_delta",
    "income_referee.effect.detection_event",
    "income_referee.effect.target_power",
    "income_referee.effect.mde_definition",
    "income_referee.holdout.unit_definition",
    "income_referee.holdout.wilson_confidence",
    "income_referee.holdout.pooled_or_weighted_average_forbidden",
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
    "files.prereg.path",
    "files.prereg.sha256",
    "files.psd_manifest.path",
    "files.psd_manifest.sha256",
    "files.calibrator.implementation_id",
    "files.calibrator.sha256",
    "files.calibrator.modules",
    "drawdown_referee.status",
    "h6_numeric.status",
    "deviation_registry",
    "limitations",
    "lineage.sequence",
)


class ManifestError(TestbedError):
    """manifest 缺失/字段不全/不一致/哈希链断裂（一律 fail-closed）。"""

    __test__ = False


def file_sha256(path: Path) -> str:
    """文件字节 sha256（哈希链的唯一口径）。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_sha256(path: Path) -> str:
    """manifest 文件字节 sha256（与 :func:`file_sha256` 同口径，语义化别名）。"""
    return file_sha256(path)


def calibrator_identity(
    module_dir: Optional[Path] = None,
    *,
    implementation_id: str = IMPLEMENTATION_ID,
    modules: Sequence[str] = CALIBRATOR_MODULES,
) -> Dict[str, Any]:
    """裁判代码身份 = ``{implementation_id, modules: {name: sha256}, sha256}``。

    任一被覆盖模块的字节变化都会改变 ``sha256``，从而使已冻结 manifest 失配
    （这是「代码改动必须重新冻结」的强制机制，不是缓存抖动）。
    """
    base = Path(module_dir) if module_dir is not None else Path(__file__).resolve().parent
    digests: Dict[str, str] = {}
    for name in modules:
        path = base / name
        if not path.is_file():
            raise ManifestError(f"裁判模块缺失（无法计算校准器身份）: {path}")
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


def validate_manifest(doc: Mapping[str, Any]) -> List[str]:
    """结构与口径一致性校验（不含实现数下限）。返回错误列表（``[]`` = 可执行）。"""
    errors: List[str] = []
    missing = _missing_fields(doc)
    if missing:
        errors.extend(f"缺字段 {path}" for path in missing)
        return errors
    if doc.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"schema 非法: {doc.get('schema')!r}（期望 {MANIFEST_SCHEMA}）")

    evidence = doc["evidence"]
    expected_evidence = {
        "evidence_class": "calibration_only",
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "no_ledger_entry": True,
        "row_domain": "structural",
        "schedule_mode": "replay",
    }
    for key, expected in expected_evidence.items():
        if evidence.get(key) != expected:
            errors.append(f"evidence.{key} 必须是 {expected!r}，实际 {evidence.get(key)!r}")
    if str(doc.get("scope")) != "historical_testbed":
        errors.append(f"scope 必须是 historical_testbed，实际 {doc.get('scope')!r}")

    referee = doc["income_referee"]
    errors.extend(_validate_statistic(referee["statistic"]))
    errors.extend(_validate_sample(referee["sample"]))
    errors.extend(_validate_threshold(referee["threshold"], referee["sample"]))
    errors.extend(_validate_effect(referee["effect"]))
    errors.extend(_validate_holdout(referee["holdout"]))
    errors.extend(_validate_scenes(doc["scenes"]))
    errors.extend(_validate_dgp(doc))
    errors.extend(_validate_block(doc["block_resample"], referee["sample"]))
    errors.extend(_validate_counts(doc["counts"]))
    errors.extend(_validate_source_pair(doc["source_pair"]))
    errors.extend(_validate_substreams(doc["substreams"]))
    errors.extend(_validate_files(doc["files"]))
    if (doc["drawdown_referee"] or {}).get("status") != DRAWDOWN_PLACEHOLDER_STATUS:
        errors.append(
            "drawdown_referee.status 必须是 "
            f"{DRAWDOWN_PLACEHOLDER_STATUS!r}（T4 不实现回撤裁判，只登记占位）"
        )
    if (doc["h6_numeric"] or {}).get("status") not in {"placeholder_not_run", "not_run_in_t4"}:
        errors.append("h6_numeric.status 必须是占位状态（H6 单条演练不在 T4 范围）")
    errors.extend(_validate_deviation_registry(doc.get("deviation_registry")))
    if not isinstance(doc.get("limitations"), list) or len(doc["limitations"]) < 3:
        errors.append("limitations 至少 3 条（§5.6 限制声明）")
    errors.extend(_validate_lineage(doc.get("lineage")))
    return errors


def _validate_lineage(lineage: Any) -> List[str]:
    """冻结谱系：首次冻结必须无前身；重冻必须留废止记录（否则无法判断"先看读数再改"）。"""
    errors: List[str] = []
    if not isinstance(lineage, Mapping):
        return ["lineage 必须是对象（登记冻结序号与废止前身）"]
    try:
        sequence = int(lineage.get("sequence"))
    except (TypeError, ValueError):
        errors.append("lineage.sequence 必须是正整数")
        return errors
    if sequence < 1:
        errors.append(f"lineage.sequence 必须 >= 1，实际 {sequence}")
        return errors
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


def _validate_deviation_registry(registry: Any) -> List[str]:
    """偏差登记表：条目 schema + 必登记项白名单（防止"静默替换"）。

    必登记项 = 本次实现与设计/冻结值确有差异、且会影响读数解释的 6 类：种子、holdout
    realization 身份、统计量、配对账本样本、冷启动/超 H 依赖场景缺口、B0' 口径。
    """
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


def _validate_statistic(statistic: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    name = str(statistic.get("name"))
    if name not in REGISTERED_STATISTICS:
        errors.append(f"statistic.name 不在可注册白名单 {list(REGISTERED_STATISTICS)}: {name!r}")
    if str(statistic.get("gamma_denominator")) != "J-1":
        errors.append(
            f"statistic.gamma_denominator 必须是 'J-1'，实际 {statistic.get('gamma_denominator')!r}"
        )
    delta = float(statistic.get("delta_log_daily"))
    if not delta_annualized_ok(delta):
        errors.append(
            f"δ={delta} 与 ln(1+{DELTA_ANNUAL_TARGET})/{ANNUAL_TRADING_DAYS} 不一致（年化换算冻结口径）"
        )
    if not math.isclose(delta, DELTA_LOG_DAILY, rel_tol=0.0, abs_tol=1e-18):
        errors.append(
            f"δ 必须等于冻结字面值 {DELTA_LOG_DAILY}（保证与 T0.5b r3 可复现），实际 {delta}"
        )
    annual = dict(statistic.get("delta_annualization") or {})
    if float(annual.get("trading_days", 0)) != float(ANNUAL_TRADING_DAYS):
        errors.append("statistic.delta_annualization.trading_days 必须为 252")
    if annual.get("scale") != "log_linear":
        errors.append(
            "statistic.delta_annualization.scale 必须是 'log_linear'（不是账户自然日复合年化）"
        )
    if statistic.get("epsilon_or_clamp_forbidden") is not True:
        errors.append("statistic.epsilon_or_clamp_forbidden 必须为 true（禁止 max(V,eps) 截断）")
    if statistic.get("boundary_centering") is not True:
        errors.append("statistic.boundary_centering 必须为 true（PSD 预登记口径）")
    return errors


def _validate_sample(sample: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    try:
        plan = make_look_plan(int(sample["H"]), int(sample["n_max"]))
    except (KeyError, TypeError, ValueError, TestbedError) as exc:
        return [f"sample 无法构造 look 映射: {exc}"]
    if [int(j) for j in sample.get("look_blocks") or []] != list(plan.j_looks):
        errors.append(
            f"look_blocks 必须等于 §5.1 映射 {list(plan.j_looks)}，实际 {sample.get('look_blocks')}"
        )
    if [int(day) for day in sample.get("look_days") or []] != list(plan.look_days):
        errors.append(
            f"look_days 必须等于 H*J 映射 {list(plan.look_days)}，实际 {sample.get('look_days')}"
        )
    if int(sample.get("K", -1)) != plan.k:
        errors.append(f"K 必须等于 len(look_blocks)={plan.k}，实际 {sample.get('K')}")
    if int(sample.get("min_blocks_per_look", 0)) < MIN_BLOCKS_PER_LOOK:
        errors.append(f"min_blocks_per_look 必须 >= {MIN_BLOCKS_PER_LOOK}")
    if plan.k < MIN_LOOKS:
        errors.append(f"K={plan.k} < {MIN_LOOKS}（该统计配置必须拒绝）")
    return errors


def _validate_threshold(threshold: Mapping[str, Any], sample: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    alpha = float(threshold.get("alpha_train"))
    alpha_look = float(threshold.get("alpha_per_look"))
    k = int(sample.get("K", 0))
    if k <= 0 or not math.isclose(alpha_look, alpha / k, rel_tol=1e-12, abs_tol=0.0):
        errors.append(f"alpha_per_look({alpha_look}) 必须等于 alpha_train/K({alpha}/{k})")
    if not (0.0 < alpha < 1.0) or not (0.0 < alpha_look < 1.0):
        errors.append(f"alpha 必须在 (0,1): alpha_train={alpha} alpha_per_look={alpha_look}")
    if float(threshold.get("invalid_rate_per_look_max")) > 0.10:
        errors.append("invalid_rate_per_look_max 不得放宽到 0.10 以上")
    if float(threshold.get("holdout_wilson_upper_max")) > 0.06:
        errors.append("holdout_wilson_upper_max 不得放宽到 0.06 以上")
    if int(threshold.get("min_valid_quantile_sample", 0)) < 100:
        errors.append("min_valid_quantile_sample 不得低于 100（分位不可估计即配置失败）")
    steps = list(threshold.get("rule_steps") or [])
    if len(steps) < 5:
        errors.append(
            "threshold.rule_steps 至少 5 步（分组→逐 look 分位→逐单元 c→max→不依赖 holdout）"
        )
    if not str(threshold.get("veto_event") or ""):
        errors.append("threshold.veto_event 不能为空")
    return errors


def _validate_effect(effect: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    grid = [float(x) for x in effect.get("grid_delta") or []]
    expected = [0.5, 1.0, 1.5, 3.0, 5.0]
    if grid != expected:
        errors.append(f"effect.grid_delta 必须是 {expected}，实际 {grid}")
    if not math.isclose(float(effect.get("target_power")), 0.80, rel_tol=0, abs_tol=1e-12):
        errors.append("effect.target_power 必须为 0.80（点估计口径）")
    if not str(effect.get("mde_definition") or ""):
        errors.append("effect.mde_definition 不能为空")
    if not str(effect.get("detection_event") or ""):
        errors.append("effect.detection_event 不能为空")
    return errors


def _validate_holdout(holdout: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if holdout.get("pooled_or_weighted_average_forbidden") is not True:
        errors.append(
            "holdout.pooled_or_weighted_average_forbidden 必须为 true（禁止跨场景加权平均）"
        )
    if float(holdout.get("wilson_confidence")) != 0.95:
        errors.append("holdout.wilson_confidence 必须为 0.95")
    if not str(holdout.get("unit_definition") or ""):
        errors.append("holdout.unit_definition 不能为空")
    return errors


def _validate_scenes(scenes: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    expected = {"ma20", "ar1_06", "regime_hetero", "real_seed_block"}
    if set(scenes) != expected:
        errors.append(
            f"scenes 必须恰为 {sorted(expected)}（禁止删除困难场景），实际 {sorted(scenes)}"
        )
    return errors


def _dgp_numbers(dgp: Mapping[str, Any]) -> Dict[str, Any]:
    """取 DGP 数值（缺失/类型错误即抛 ``ValueError``）。"""
    return {
        "sd_daily": float(dgp["sd_daily"]),
        "path_days": int(dgp["path_days"]),
        "rho": float(dgp["ar_rho"]),
        "regime_block": int(dgp["regime_block_length"]),
        "multipliers": [float(x) for x in dgp["regime_multipliers"]],
        "block_length": int(dgp["block_length"]),
    }


def _validate_dgp_range(number: Mapping[str, Any], sample: Mapping[str, Any]) -> List[str]:
    """DGP 数值的取值域与样本上限一致性。"""
    errors: List[str] = []
    if number["sd_daily"] <= 0:
        errors.append(f"calibration.dgp.sd_daily 必须为正，实际 {number['sd_daily']}")
    if not (0.0 <= number["rho"] < 1.0):
        errors.append(f"calibration.dgp.ar_rho 必须在 [0,1)，实际 {number['rho']}")
    if number["block_length"] != int(sample.get("H", -1)):
        errors.append(
            f"calibration.dgp.block_length({number['block_length']}) "
            f"必须等于 H({sample.get('H')})"
        )
    look_days = list(sample.get("look_days") or [0])
    if number["path_days"] < int(look_days[-1]):
        errors.append(
            f"calibration.dgp.path_days({number['path_days']}) 短于最大 look 终点 "
            f"{look_days[-1]}（block 不可构造）"
        )
    return errors


def _validate_scene_numbers(scenes: Mapping[str, Any], number: Mapping[str, Any]) -> List[str]:
    """``scenes`` 描述块与 DGP 数值块的一致性（防两处数字漂移）。"""
    try:
        checks = [
            (
                math.isclose(
                    float(scenes["ma20"]["sd_daily"]),
                    number["sd_daily"],
                    rel_tol=0.0,
                    abs_tol=0.0,
                ),
                "scenes.ma20.sd_daily 与 calibration.dgp.sd_daily 不一致",
            ),
            (
                math.isclose(
                    float(scenes["ar1_06"]["rho"]), number["rho"], rel_tol=0.0, abs_tol=0.0
                ),
                "scenes.ar1_06.rho 与 calibration.dgp.ar_rho 不一致",
            ),
            (
                int(scenes["regime_hetero"]["block"]) == number["regime_block"],
                "scenes.regime_hetero.block 与 calibration.dgp.regime_block_length 不一致",
            ),
            (
                [float(x) for x in scenes["regime_hetero"]["multipliers"]] == number["multipliers"],
                "scenes.regime_hetero.multipliers 与 calibration.dgp.regime_multipliers 不一致",
            ),
            (
                int(scenes["real_seed_block"]["block_length"]) == number["block_length"],
                "scenes.real_seed_block.block_length 与 calibration.dgp.block_length 不一致",
            ),
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return [f"scenes 描述块缺少可核对字段: {exc!r}"]
    return [message for ok, message in checks if not ok]


def _validate_dgp(doc: Mapping[str, Any]) -> List[str]:
    """DGP 数值块 + 与描述块一致性（拆三段以控制单函数复杂度）。"""
    dgp = dict((doc.get("calibration") or {}).get("dgp") or {})
    sample = dict((doc.get("income_referee") or {}).get("sample") or {})
    scenes = dict(doc.get("scenes") or {})
    try:
        number = _dgp_numbers(dgp)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"calibration.dgp 数值非法: {exc!r}"]
    return _validate_dgp_range(number, sample) + _validate_scene_numbers(scenes, number)


def _validate_block(block: Mapping[str, Any], sample: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(block.get("method")) != "circular":
        errors.append(f"block_resample.method 必须是 circular，实际 {block.get('method')!r}")
    if int(block.get("block_length", -1)) != int(sample.get("H", -2)):
        errors.append(f"block_length({block.get('block_length')}) 必须等于 H({sample.get('H')})")
    if block.get("truncation_forbidden") is not True:
        errors.append("block_resample.truncation_forbidden 必须为 true")
    if block.get("cross_pool_sampling_forbidden") is not True:
        errors.append("block_resample.cross_pool_sampling_forbidden 必须为 true")
    if "mean" not in str(block.get("center") or ""):
        errors.append("block_resample.center 必须声明按源序列自身均值去中心")
    return errors


def _validate_counts(counts: Mapping[str, Any]) -> List[str]:
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


def _validate_source_pair(pair: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    train = {int(x) for x in pair.get("train_folds") or []}
    holdout = {int(x) for x in pair.get("holdout_folds") or []}
    if not train or not holdout:
        errors.append("source_pair.train_folds/holdout_folds 都不能为空")
    elif train & holdout:
        errors.append(f"训练/ holdout 源折重叠（必须互斥）: {sorted(train & holdout)}")
    blocks = pair.get("date_blocks") or {}
    if list(blocks.get("key") or []) != ["split_index", "date"]:
        errors.append("source_pair.date_blocks.key 必须是 ['split_index','date']")
    if blocks.get("date_semantics") != "fold_local_ordinal":
        errors.append(
            "source_pair.date_blocks.date_semantics 必须是 fold_local_ordinal（源文件无自然日）"
        )
    if not str(pair.get("source_run_id") or ""):
        errors.append("source_pair.source_run_id 不能为空（共享同一对真实源）")
    for key in ("base_sha256", "noise_sha256"):
        if len(str(pair.get(key) or "")) != 64:
            errors.append(f"source_pair.{key} 必须是 64 位 sha256")
    note = str(pair.get("noise_arm_note") or "")
    if "P3" not in note or "500" not in note:
        errors.append(
            "source_pair.noise_arm_note 必须明确 B0' 不是 P3 的 500 树重训臂" f"（实际 {note!r}）"
        )
    return errors


def _validate_substreams(substreams: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    declared = dict(substreams or {})
    expected = implementation_ids()
    if declared.get("implementation_id") != expected["implementation_id"]:
        errors.append(
            f"substreams.implementation_id 必须是 {expected['implementation_id']!r}，"
            f"实际 {declared.get('implementation_id')!r}"
        )
    if dict(declared.get("stages") or {}) != expected["stages"]:
        errors.append(
            f"substreams.stages 必须是 {expected['stages']}，实际 {declared.get('stages')}"
        )
    if int(declared.get("level_code_scale", 0)) != expected["level_code_scale"]:
        errors.append("substreams.level_code_scale 必须与实现一致")
    independence = dict(declared.get("independence") or {})
    if independence.get("train_holdout_disjoint") is not True:
        errors.append("substreams.independence.train_holdout_disjoint 必须为 true")
    return errors


def _validate_files(files: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for name in ("prereg", "psd_manifest", "calibrator"):
        entry = files.get(name) or {}
        if not entry:
            errors.append(f"files.{name} 缺失")
    for name in ("prereg", "psd_manifest"):
        entry = files.get(name) or {}
        if len(str(entry.get("sha256") or "")) != 64:
            errors.append(f"files.{name}.sha256 必须是 64 位 sha256")
        if not str(entry.get("path") or ""):
            errors.append(f"files.{name}.path 不能为空")
    calibrator = files.get("calibrator") or {}
    if calibrator.get("implementation_id") != IMPLEMENTATION_ID:
        errors.append(
            f"files.calibrator.implementation_id 必须是 {IMPLEMENTATION_ID!r}，"
            f"实际 {calibrator.get('implementation_id')!r}"
        )
    if len(str(calibrator.get("sha256") or "")) != 64:
        errors.append("files.calibrator.sha256 必须是 64 位 sha256")
    modules = calibrator.get("modules") or {}
    if set(modules) != set(CALIBRATOR_MODULES):
        errors.append(f"files.calibrator.modules 必须恰为 {list(CALIBRATOR_MODULES)}")
    return errors


def scale_floor_errors(counts: Mapping[str, Any]) -> List[str]:
    """实现数下限校验（§5.3 表：只能上调不得下调）。"""
    errors: List[str] = []
    for key, floor in SCALE_FLOOR.items():
        try:
            value = int(counts[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"counts.{key} 缺失或非整数，无法核对下限 {floor}")
            continue
        if value < floor:
            errors.append(f"counts.{key}={value} 低于 §5.3 冻结下限 {floor}（禁止下调）")
    return errors


def load_manifest(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    require_scale_floor: bool = True,
    verify_calibrator: bool = True,
    module_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """读取并核验 manifest（封存 + 字段 + 一致性 + 校准器身份）。

    Args:
        path: manifest 路径。
        expected_sha256: 期望的文件字节 sha256（例如仓库冻结件与 campaign 副本互校）。
        require_scale_floor: 是否强制 §5.3 实现数下限（生产路径必须 True）。
        verify_calibrator: 是否复算裁判代码身份并与登记值比对。
        module_dir: 仅测试用（指定裁判模块目录）。

    Raises:
        ManifestError: 任一校验失败（fail-closed）。
    """
    source = Path(path)
    if not source.is_file():
        raise ManifestError(f"校准 manifest 缺失（未冻结不得取证）: {source}")
    actual_sha = manifest_sha256(source)
    if expected_sha256 is not None and actual_sha != str(expected_sha256):
        raise ManifestError(
            f"manifest 文件哈希与期望不符（被改写或指向错文件）: {source} "
            f"实际={actual_sha} 期望={expected_sha256}"
        )
    doc = read_json(source, what="校准 manifest")
    try:
        verify_seal(doc, what=f"校准 manifest（{source}）")
    except TestbedError as exc:
        raise ManifestError(f"校准 manifest 封存校验失败: {exc}") from exc
    errors = validate_manifest(doc)
    if errors:
        raise ManifestError(f"校准 manifest 校验失败（{source}）: " + "; ".join(errors))
    if require_scale_floor:
        floor_errors = scale_floor_errors(doc["counts"])
        if floor_errors:
            raise ManifestError(
                f"校准 manifest 实现数低于冻结下限（{source}）: " + "; ".join(floor_errors)
            )
    if verify_calibrator:
        expected = calibrator_identity(module_dir)
        actual = dict(doc["files"]["calibrator"])
        if actual.get("sha256") != expected["sha256"]:
            raise ManifestError(
                "裁判代码身份与 manifest 登记不符（代码已变，必须重新冻结 manifest）: "
                f"实际={expected['sha256']} 登记={actual.get('sha256')}"
            )
        if dict(actual.get("modules") or {}) != expected["modules"]:
            raise ManifestError("files.calibrator.modules 与当前裁判模块字节不符（代码已变）")
    return doc


def verify_source_files(doc: Mapping[str, Any], *, root: Path) -> Dict[str, Any]:
    """核验 manifest 引用的源对/预登记/PSD manifest 文件哈希（缺一即失败）。"""
    base = Path(root)
    checks: Dict[str, Any] = {}
    pair = doc["source_pair"]
    for key in ("base_chain_nav", "noise_chain_nav"):
        path = base / str(pair[key])
        expected = str(pair["base_sha256" if key == "base_chain_nav" else "noise_sha256"])
        if not path.is_file():
            raise ManifestError(f"源对文件缺失: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ManifestError(
                f"源对文件哈希不符（禁止换源）: {path} 实际={actual} 期望={expected}"
            )
        checks[key] = {"path": path.as_posix(), "sha256": actual}
    for name in ("prereg", "psd_manifest"):
        entry = doc["files"][name]
        path = base / str(entry["path"])
        if not path.is_file():
            raise ManifestError(f"manifest 引用文件缺失（{name}）: {path}")
        actual = file_sha256(path)
        if actual != str(entry["sha256"]):
            raise ManifestError(f"manifest 引用文件哈希不符（{name}）: {path} 实际={actual}")
        checks[name] = {"path": path.as_posix(), "sha256": actual}
    return checks


def freeze_manifest(doc: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    """冻结 manifest（校验 + 封存 + **独占创建**，已存在即拒绝）。

    Returns:
        ``{path, sha256, seal}``。

    Raises:
        ManifestError: 校验失败或目标已存在（禁止重写已冻结件）。
    """
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
    """校验 manifest ↔ 校准器 ↔ 临界值（↔ holdout/power）的哈希互链。

    Args:
        manifest_doc: 已加载 manifest。
        manifest_file_sha256: manifest 文件字节 sha256。
        calibrator: 当前校准器身份（缺省 = 复算）。
        thresholds/holdout/power: 各阶段产物（缺省 = 跳过该环）。
        thresholds_sha256: 临界值产物**文件** sha256（产物内不含自身哈希，需显式传入）。

    Returns:
        ``{"checks": [...], "ok": bool}``。

    Raises:
        ManifestError: 任一环不符（fail-closed）。
    """
    identity = dict(calibrator) if calibrator is not None else calibrator_identity()
    checks: List[Dict[str, Any]] = []

    def record(name: str, actual: Any, expected: Any) -> None:
        ok = actual == expected
        checks.append({"link": name, "actual": actual, "expected": expected, "ok": ok})

    record("manifest:calibrator", manifest_doc["files"]["calibrator"]["sha256"], identity["sha256"])
    for label, artifact in (("thresholds", thresholds), ("holdout", holdout), ("power", power)):
        if artifact is None:
            continue
        record(f"{label}:manifest", artifact.get("manifest_sha256"), manifest_file_sha256)
        record(f"{label}:calibrator", artifact.get("calibrator_sha256"), identity["sha256"])
    if thresholds is not None:
        if thresholds_sha256 is None:
            raise ManifestError("校验 holdout/power 与临界值的互链必须提供 thresholds_sha256")
        # thresholds:manifest / thresholds:calibrator 已由上方通用循环登记，此处不再重复追加
        for label, artifact in (("holdout", holdout), ("power", power)):
            if artifact is None:
                continue
            record(f"{label}:thresholds", artifact.get("thresholds_sha256"), thresholds_sha256)
    failed = [check for check in checks if not check["ok"]]
    if failed:
        raise ManifestError(
            "校准哈希链断裂（拒绝运行）: "
            + "; ".join(
                f"{item['link']} 实际={item['actual']} 期望={item['expected']}" for item in failed
            )
        )
    return {"checks": checks, "ok": True}
