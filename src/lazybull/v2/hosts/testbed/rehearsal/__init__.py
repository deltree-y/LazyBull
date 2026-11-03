# -*- coding: utf-8 -*-
"""P5b 试车场 T6 历史端到端演练子包（运行器 → 配对账本 → 裁判 → 报告）。

模块分工
========
- :mod:`.manifest` / :mod:`.sections`：T6 manifest 冻结、字段校验、上游链接与三环互链；
- :mod:`.sources`：structural 窗口、交易日历、上游哈希、冻结崩盘窗、λ/ζ（盲态求解）；
- :mod:`.injection`：效应注入的**纯函数**（只改 log 收益/NAV，不改成交/现金/持仓/日期）；
- :mod:`.hosts`：宿主层注入（``host_factory`` 包装默认宿主，读侧覆写当日估值）；
- :mod:`.aa`：机械 A/A 逐值一致（从分片复算，不作统计 A/A）；
- :mod:`.readings`：账本 → 配对收益 → 裁判读数 → H6 三条判定；
- :mod:`.verdict` / :mod:`.report`：判定表（固定 ``evaluated_not_declared_pass``）与 §7.2 报告；
- :mod:`.assembly`：事实汇集、报告落盘、版本与运行手册一致性；
- :mod:`.runner`：冻结 → 注册/D0 → 运行（含 20 分钟授权门）。

边界：历史虚拟时钟、``evidence_class=calibration_only``；不写 ``data/paper``/``data/runs``/
``data/models``，不写正式 runs 或 LedgerEntry，不注册真实候选。闸门判定归用户。
"""

from __future__ import annotations

from src.lazybull.v2.hosts.testbed.rehearsal.assembly import (
    build_readings,
    collect_facts,
    verify_runbook,
    version_sync,
    write_outputs,
)
from src.lazybull.v2.hosts.testbed.rehearsal.injection import (
    ANCHOR_TARGET_PP,
    EFFECT_INCOME,
    EFFECT_NONE,
    EFFECT_WINDOW,
    INCOME_DELTA_MULTIPLE,
    InjectionError,
    RehearsalEffect,
    apply_single_return,
    arm_effect_plan,
    build_arm_effect,
    income_daily_shift,
    log_returns_from_nav,
    nav_from_log_returns,
    solve_anchor_parameter,
    transform_log_returns,
    window_mask_from_frozen,
)
from src.lazybull.v2.hosts.testbed.rehearsal.manifest import (
    CALIBRATOR_MODULES,
    IMPLEMENTATION_ID,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    ManifestError,
    build_manifest_document,
    calibrator_identity,
    freeze_manifest,
    load_manifest,
    verify_chain,
    verify_upstream,
)
from src.lazybull.v2.hosts.testbed.rehearsal.readings import (
    drawdown_look_reading,
    frozen_thresholds,
    h6_drawdown_item,
    h6_income_item,
    income_look_reading,
    paired_paths,
)
from src.lazybull.v2.hosts.testbed.rehearsal.runner import (
    LongTaskAuthorizationRequired,
    RehearsalError,
    campaign_root_for,
    freeze_rehearsal,
    load_authorization,
    project_batch_seconds,
    register_rehearsal,
    rehearsal_layout,
    run_rehearsal,
    verify_rehearsal_window,
)

__all__ = [
    "ANCHOR_TARGET_PP",
    "CALIBRATOR_MODULES",
    "EFFECT_INCOME",
    "EFFECT_NONE",
    "EFFECT_WINDOW",
    "IMPLEMENTATION_ID",
    "INCOME_DELTA_MULTIPLE",
    "InjectionError",
    "LongTaskAuthorizationRequired",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "ManifestError",
    "RehearsalEffect",
    "RehearsalError",
    "apply_single_return",
    "arm_effect_plan",
    "build_arm_effect",
    "build_manifest_document",
    "build_readings",
    "calibrator_identity",
    "campaign_root_for",
    "collect_facts",
    "drawdown_look_reading",
    "freeze_manifest",
    "freeze_rehearsal",
    "frozen_thresholds",
    "h6_drawdown_item",
    "h6_income_item",
    "income_daily_shift",
    "income_look_reading",
    "load_authorization",
    "load_manifest",
    "log_returns_from_nav",
    "nav_from_log_returns",
    "paired_paths",
    "project_batch_seconds",
    "register_rehearsal",
    "rehearsal_layout",
    "run_rehearsal",
    "solve_anchor_parameter",
    "transform_log_returns",
    "verify_chain",
    "verify_rehearsal_window",
    "verify_runbook",
    "verify_upstream",
    "version_sync",
    "window_mask_from_frozen",
    "write_outputs",
]
