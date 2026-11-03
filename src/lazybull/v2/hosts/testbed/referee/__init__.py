# -*- coding: utf-8 -*-
"""P5b T4 收益裁判（校准器库层）：look 映射、block 统计量、场景 DGP、命名子流、
临界值训练、holdout 验收、功效/MDE、配对账本样本与 §5.5 冻结 manifest 哈希链。

边界（设计 v4 §5 / 架构 F22）
=============================
- 只做**历史**校准：不运行真实前向影子、不等待真实日历、不注册真实候选；
- 产物只落 campaign ``reports/``（``data/paper_shadow/p5b_testbed/<campaign_id>/reports/``），
  不写 ``data/paper`` / ``data/runs`` / ``data/models``，不写 ``LedgerEntry``；
- 证据一律 ``evidence_class=calibration_only``；H4 闸门由用户在合并验收时确认；
- 回撤裁判（DD-W/DD-I）不在本子包内（P5b T5）。

模块划分
========
- ``statistics``：§5.1 look 映射 + §5.2 block 统计量（PSD 为注册件，原式为复核件）；
- ``sampling``：命名 seed 子流与 realization/implementation 身份；
- ``scenes``：§5.3 四类场景、circular block 重采样、B0/B0' 源对读取；
- ``training``：§5.2 五步临界值生成（只读训练池）；
- ``holdout``：§5.3 逐单元 Wilson 门（禁止跨场景聚合）；
- ``power``：§5.3 效应网格、检出率与 MDE；
- ``ledger_source``：从试车场配对账本构造 ``d_t``（只读，走 ``query_shards`` 门）；
- ``manifest``：§5.5 全表冻结、字段校验与 manifest↔校准器↔临界值哈希互链；
- ``manifest_sections``：manifest 分节构造器（源对/统计量/阈值/效应/占位/偏差/限制/谱系）；
- ``artifacts``：确定性独占落盘（逐字节可复现）；
- ``report``：判定表与 §7.2 收益部分报告；
- ``calibration``：编排（冻结 → 训练 → holdout → 功效 → 判定表/报告）。
"""

from src.lazybull.v2.hosts.testbed.referee.artifacts import (
    ARTIFACT_HOLDOUT,
    ARTIFACT_MANIFEST,
    ARTIFACT_MACHINE_TIME,
    ARTIFACT_NAMES,
    ARTIFACT_POWER,
    ARTIFACT_REPORT,
    ARTIFACT_REPORT_MD,
    ARTIFACT_THRESHOLDS,
    ARTIFACT_VERDICT,
    ArtifactError,
    artifact_path,
    load_artifact,
    reports_dir,
    sha256_file,
    write_json_once,
    write_text_once,
)
from src.lazybull.v2.hosts.testbed.referee.calibration import (
    DEFAULT_CHUNK,
    build_manifest_document,
    freeze_calibration_manifest,
    load_calibration_manifest,
    run_holdout,
    source_pair_for_manifest,
    train_calibration,
    write_machine_time,
)
from src.lazybull.v2.hosts.testbed.referee.holdout import (
    HOLDOUT_SCHEMA,
    WILSON_CONFIDENCE,
    WILSON_Z,
    HoldoutError,
    evaluate_holdout,
    holdout_unit_count,
    wilson_interval,
)
from src.lazybull.v2.hosts.testbed.referee.ledger_source import (
    LedgerSampleError,
    ledger_sample_identity,
    nav_series_from_ledger,
    paired_returns_from_ledger,
)
from src.lazybull.v2.hosts.testbed.referee.manifest import (
    CALIBRATOR_EXCLUSIONS,
    CALIBRATOR_MODULES,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    ManifestError,
    SCALE_FLOOR,
    calibrator_identity,
    freeze_manifest,
    load_manifest,
    manifest_sha256,
    scale_floor_errors,
    validate_manifest,
    verify_calibration_chain,
    verify_source_files,
)
from src.lazybull.v2.hosts.testbed.referee.power import (
    EFFECT_GRID_DELTA,
    MDE_NOT_REACHED,
    MDE_NOT_REGISTERED,
    POWER_SCHEMA,
    TARGET_POWER,
    PowerError,
    evaluate_power,
    register_mde,
)
from src.lazybull.v2.hosts.testbed.referee.report import (
    REPORT_SCHEMA,
    VERDICT_SCHEMA,
    build_income_report,
    build_referee_verdict,
    render_markdown,
)
from src.lazybull.v2.hosts.testbed.referee.sampling import (
    IMPLEMENTATION_ID,
    STAGE_CODES,
    Substream,
    SubstreamError,
    assert_distinct_streams,
    implementation_ids,
    stream_codes,
    substream,
)
from src.lazybull.v2.hosts.testbed.referee.scenes import (
    SCENE_NAMES,
    SCENE_REQUIRED,
    DgpParams,
    SceneError,
    SceneRegistry,
    SourcePair,
    build_scene_registry,
    circular_block_resample,
    load_source_pair,
    paired_log_diff,
    pool_for_folds,
    scene_parameters,
)
from src.lazybull.v2.hosts.testbed.referee.statistics import (
    DELTA_LOG_DAILY,
    LOOK_N_MAX,
    MIN_BLOCKS_PER_LOOK,
    MIN_LOOKS,
    REGISTERED_STATISTICS,
    STATISTIC_DESIGN_V4,
    STATISTIC_PSD,
    LookPlan,
    LookPlanError,
    ScoreBlock,
    StatisticError,
    block_means,
    make_look_plan,
    score_paths,
    veto_flags,
    variance_gamma0_plus_2gamma1,
    variance_psd_lag1,
)
from src.lazybull.v2.hosts.testbed.referee.training import (
    MIN_VALID_QUANTILE_SAMPLE,
    ThresholdConfig,
    ThresholdError,
    TrainingResult,
    generate_scores,
    train_income_thresholds,
)

__all__ = [
    # statistics
    "DELTA_LOG_DAILY",
    "LOOK_N_MAX",
    "MIN_BLOCKS_PER_LOOK",
    "MIN_LOOKS",
    "REGISTERED_STATISTICS",
    "STATISTIC_DESIGN_V4",
    "STATISTIC_PSD",
    "LookPlan",
    "LookPlanError",
    "ScoreBlock",
    "StatisticError",
    "block_means",
    "make_look_plan",
    "score_paths",
    "veto_flags",
    "variance_gamma0_plus_2gamma1",
    "variance_psd_lag1",
    # sampling
    "IMPLEMENTATION_ID",
    "STAGE_CODES",
    "Substream",
    "SubstreamError",
    "assert_distinct_streams",
    "implementation_ids",
    "stream_codes",
    "substream",
    # scenes
    "SCENE_NAMES",
    "SCENE_REQUIRED",
    "DgpParams",
    "SceneError",
    "SceneRegistry",
    "SourcePair",
    "build_scene_registry",
    "circular_block_resample",
    "load_source_pair",
    "paired_log_diff",
    "pool_for_folds",
    "scene_parameters",
    # training
    "MIN_VALID_QUANTILE_SAMPLE",
    "ThresholdConfig",
    "ThresholdError",
    "TrainingResult",
    "generate_scores",
    "train_income_thresholds",
    # holdout / power
    "HOLDOUT_SCHEMA",
    "WILSON_CONFIDENCE",
    "WILSON_Z",
    "HoldoutError",
    "evaluate_holdout",
    "holdout_unit_count",
    "wilson_interval",
    "EFFECT_GRID_DELTA",
    "MDE_NOT_REACHED",
    "MDE_NOT_REGISTERED",
    "POWER_SCHEMA",
    "TARGET_POWER",
    "PowerError",
    "evaluate_power",
    "register_mde",
    # ledger sample
    "LedgerSampleError",
    "ledger_sample_identity",
    "nav_series_from_ledger",
    "paired_returns_from_ledger",
    # manifest / artifacts
    "CALIBRATOR_EXCLUSIONS",
    "CALIBRATOR_MODULES",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "ManifestError",
    "SCALE_FLOOR",
    "calibrator_identity",
    "freeze_manifest",
    "load_manifest",
    "manifest_sha256",
    "scale_floor_errors",
    "validate_manifest",
    "verify_calibration_chain",
    "verify_source_files",
    "ARTIFACT_HOLDOUT",
    "ARTIFACT_MANIFEST",
    "ARTIFACT_MACHINE_TIME",
    "ARTIFACT_NAMES",
    "ARTIFACT_POWER",
    "ARTIFACT_REPORT",
    "ARTIFACT_REPORT_MD",
    "ARTIFACT_THRESHOLDS",
    "ARTIFACT_VERDICT",
    "ArtifactError",
    "artifact_path",
    "load_artifact",
    "reports_dir",
    "sha256_file",
    "write_json_once",
    "write_text_once",
    # report / calibration
    "REPORT_SCHEMA",
    "VERDICT_SCHEMA",
    "build_income_report",
    "build_referee_verdict",
    "render_markdown",
    "DEFAULT_CHUNK",
    "build_manifest_document",
    "freeze_calibration_manifest",
    "load_calibration_manifest",
    "run_holdout",
    "source_pair_for_manifest",
    "train_calibration",
    "write_machine_time",
]
