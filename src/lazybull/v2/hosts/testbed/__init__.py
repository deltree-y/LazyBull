# -*- coding: utf-8 -*-
"""P5b 历史试车场（P5b-T1/T2/T3）：模拟臂注册、D0 fresh、身份共用件、虚拟时钟多臂运行、
族级事务、长期分片账本与故障注入矩阵。

边界（设计 v4 / 架构 F22）：
- 只做历史虚拟时钟试车场，不运行真实前向影子、不注册真实候选；
- 产物只落 ``data/paper_shadow/p5b_testbed/<campaign_id>/`` 或 ``temp/``，
  **禁止**写 ``data/paper/``、``data/runs/``、``data/models/``；
- 校准证据一律 ``evidence_class=calibration_only``，不写正式 ``LedgerEntry``。

模块划分：
- ``paths``：产物根解析与写域守卫；
- ``clock``：虚拟时钟（D0 估值日 / D1 首个相邻收益 / 逐日连续 + 日历身份）；
- ``identity``：input / config / model / code / window-state 五段身份；
- ``storage``：注册记录独占创建、封存哈希与回滚；
- ``registration``：campaign 与 arm 注册台账（重复拒绝、未注册拒绝）；
- ``d0``：各臂 D0 fresh 空现金起点与一致性核验（盘上观测 + 记录封存）；
- ``layout``（T2）：产物布局、根内容身份与目录搬运原语；
- ``journal``（T2）：族级事务 journal、读侧 index、逐日分片与 manifest；
- ``arm_runtime``（T2）：臂规格、宿主工厂、权威估值读取与虚拟日成功资格；
- ``family_txn``（T2）：族级事务状态机（暂存/提升/唯一发布点/回滚/恢复）；
- ``runner``（T2）：虚拟时钟多臂逐日推进与运行报告；
- ``shard_chain``（T3）：分片链纯函数（逐片 sha → 链 digest，缺片哨兵参与推进）；
- ``ledger_audit``（T3）：三方一致性审计、缺口分类与修复、INVALIDATED 终态、只读查询；
- ``fault_matrix``（T3）：§4.3 故障注入矩阵表、containment 分级、五不变量校验。
"""

from src.lazybull.v2.hosts.testbed.clock import (
    VirtualClock,
    build_virtual_clock,
    calendar_identity,
    normalize_day,
)
from src.lazybull.v2.hosts.testbed.d0 import (
    D0FreshError,
    EmptyStateObservation,
    empty_account_digest,
    initialize_d0_fresh,
    load_d0_record,
    observe_empty_account,
    verify_d0_fresh,
)
from src.lazybull.v2.hosts.testbed.identity import (
    collect_code_identity,
    collect_config_identity,
    collect_data_state_identity,
    collect_input_identity,
    collect_model_identity,
    identity_summary,
)
from src.lazybull.v2.hosts.testbed.paths import (
    ARM_ROLES,
    EVIDENCE_CLASS,
    SCOPE,
    TestbedError,
    TestbedScopeError,
    assert_write_scope,
    resolve_campaign_root,
    testbed_root,
)
from src.lazybull.v2.hosts.testbed.registration import (
    DuplicateRegistrationError,
    UnregisteredArmError,
    list_arms,
    load_arm,
    load_campaign,
    register_arm,
    register_campaign,
)

# T2（虚拟时钟多臂运行 + 族级事务）：只新增导出，不改动 T1 已闭合的注册/D0 语义
from src.lazybull.v2.hosts.testbed.arm_runtime import (
    ArmSpec,
    DrivenQualificationError,
    VirtualDayAlreadyRun,
    VirtualDayNotReady,
    drive_virtual_day,
    load_arm_specs,
    make_default_host_factory,
    read_authoritative_valuation,
)
from src.lazybull.v2.hosts.testbed.family_txn import (
    FamilyTransaction,
    InjectedFault,
    abort_attempt,
    open_family_transaction,
    recover_campaign,
    top_up_published,
)
from src.lazybull.v2.hosts.testbed.journal import (
    published_days,
    read_index,
    rebuild_index,
    rebuild_manifest,
)
from src.lazybull.v2.hosts.testbed.layout import (
    CampaignInvalidError,
    CampaignLayout,
    FamilyTxnError,
    gen_id,
    layout_for,
    root_identity,
)
from src.lazybull.v2.hosts.testbed.runner import (
    CampaignBusyError,
    CampaignRunLock,
    assert_window_drift_free,
    campaign_fingerprint,
    run_virtual_campaign,
)

# T3（长期分片账本 + 故障注入矩阵 + containment）：只新增导出，不改 T1/T2 语义
from src.lazybull.v2.hosts.testbed.ledger_common import (
    GAP_CLASSES,
    LedgerGap,
    LedgerInvalidatedError,
    LedgerPaths,
    invalidated_days,
    ledger_paths,
    load_invalidations,
    record_invalidations,
)
from src.lazybull.v2.hosts.testbed.ledger_audit import (
    audit_ledger,
    build_chain_document,
    shard_sha_map,
    write_ledger_artifacts,
)
from src.lazybull.v2.hosts.testbed.ledger_repair import (
    LedgerRepairRequiredError,
    assert_ledger_advanceable,
    build_ledger_report,
    prune_staging,
    query_shards,
    repair_ledger,
)
from src.lazybull.v2.hosts.testbed.shard_chain import (
    CHAIN_LINK_MISSING,
    CHAIN_SCHEMA,
    build_chain,
    campaign_chain_digest,
    chain_seed,
    verify_chain,
)
from src.lazybull.v2.hosts.testbed.fault_matrix import (
    CONTAINMENT,
    INVARIANTS,
    MATRIX,
    POSTCONDITIONS,
    ContainmentLevel,
    FaultCase,
    build_verdict_table,
    check_invariants,
    derive_postconditions,
    evaluate_case,
    first_attempt_id,
    matrix_fingerprint,
    observed_terminal,
    snapshot_campaign,
    table_markdown,
)

# T4（收益裁判训练/holdout/功效）：只新增导出，不改 T1/T2/T3 语义
from src.lazybull.v2.hosts.testbed.referee import (
    ARTIFACT_MANIFEST,
    ARTIFACT_THRESHOLDS,
    DELTA_LOG_DAILY,
    EFFECT_GRID_DELTA,
    IMPLEMENTATION_ID,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    SCALE_FLOOR,
    STATISTIC_DESIGN_V4,
    STATISTIC_PSD,
    LedgerSampleError,
    LookPlan,
    LookPlanError,
    ManifestError,
    SceneError,
    ThresholdConfig,
    ThresholdError,
    build_manifest_document,
    calibrator_identity,
    evaluate_holdout,
    evaluate_power,
    freeze_calibration_manifest,
    freeze_manifest,
    load_calibration_manifest,
    load_manifest,
    make_look_plan,
    paired_returns_from_ledger,
    run_holdout,
    score_paths,
    train_calibration,
    train_income_thresholds,
    validate_manifest,
    verify_calibration_chain,
    veto_flags,
    wilson_interval,
)

__all__ = [
    "ARM_ROLES",
    "D0FreshError",
    "DuplicateRegistrationError",
    "EVIDENCE_CLASS",
    "EmptyStateObservation",
    "SCOPE",
    "TestbedError",
    "TestbedScopeError",
    "UnregisteredArmError",
    "VirtualClock",
    "assert_write_scope",
    "build_virtual_clock",
    "calendar_identity",
    "collect_code_identity",
    "collect_config_identity",
    "collect_data_state_identity",
    "collect_input_identity",
    "collect_model_identity",
    "empty_account_digest",
    "identity_summary",
    "initialize_d0_fresh",
    "list_arms",
    "load_arm",
    "load_campaign",
    "load_d0_record",
    "normalize_day",
    "observe_empty_account",
    "register_arm",
    "register_campaign",
    "resolve_campaign_root",
    "testbed_root",
    "verify_d0_fresh",
    # ---- T2：虚拟时钟多臂运行与族级事务 ----
    "ArmSpec",
    "CampaignBusyError",
    "CampaignInvalidError",
    "CampaignLayout",
    "CampaignRunLock",
    "DrivenQualificationError",
    "FamilyTransaction",
    "FamilyTxnError",
    "InjectedFault",
    "VirtualDayAlreadyRun",
    "VirtualDayNotReady",
    "abort_attempt",
    "assert_window_drift_free",
    "campaign_fingerprint",
    "drive_virtual_day",
    "gen_id",
    "layout_for",
    "load_arm_specs",
    "make_default_host_factory",
    "open_family_transaction",
    "published_days",
    "read_authoritative_valuation",
    "read_index",
    "rebuild_index",
    "rebuild_manifest",
    "recover_campaign",
    "root_identity",
    "run_virtual_campaign",
    "top_up_published",
    # ---- T3：长期分片账本 + 故障注入矩阵 + containment ----
    "CHAIN_LINK_MISSING",
    "CHAIN_SCHEMA",
    "CONTAINMENT",
    "ContainmentLevel",
    "FaultCase",
    "GAP_CLASSES",
    "INVARIANTS",
    "LedgerGap",
    "LedgerInvalidatedError",
    "LedgerPaths",
    "LedgerRepairRequiredError",
    "MATRIX",
    "POSTCONDITIONS",
    "assert_ledger_advanceable",
    "audit_ledger",
    "build_chain",
    "build_chain_document",
    "build_ledger_report",
    "build_verdict_table",
    "campaign_chain_digest",
    "chain_seed",
    "check_invariants",
    "derive_postconditions",
    "evaluate_case",
    "first_attempt_id",
    "invalidated_days",
    "matrix_fingerprint",
    "ledger_paths",
    "load_invalidations",
    "observed_terminal",
    "prune_staging",
    "query_shards",
    "record_invalidations",
    "repair_ledger",
    "shard_sha_map",
    "snapshot_campaign",
    "table_markdown",
    "verify_chain",
    "write_ledger_artifacts",
    # ---- T4：收益裁判（训练 / holdout / 功效 / manifest 哈希链）----
    "ARTIFACT_MANIFEST",
    "ARTIFACT_THRESHOLDS",
    "DELTA_LOG_DAILY",
    "EFFECT_GRID_DELTA",
    "IMPLEMENTATION_ID",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "SCALE_FLOOR",
    "STATISTIC_DESIGN_V4",
    "STATISTIC_PSD",
    "LedgerSampleError",
    "LookPlan",
    "LookPlanError",
    "ManifestError",
    "SceneError",
    "ThresholdConfig",
    "ThresholdError",
    "build_manifest_document",
    "calibrator_identity",
    "evaluate_holdout",
    "evaluate_power",
    "freeze_calibration_manifest",
    "freeze_manifest",
    "load_calibration_manifest",
    "load_manifest",
    "make_look_plan",
    "paired_returns_from_ledger",
    "run_holdout",
    "score_paths",
    "train_calibration",
    "train_income_thresholds",
    "validate_manifest",
    "verify_calibration_chain",
    "veto_flags",
    "wilson_interval",
]
