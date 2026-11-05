# -*- coding: utf-8 -*-
"""P6 报告系统 T0.5 冻结契约（机器可读唯一来源）。

契约来源：
- 设计 ``docs/plans/v2/p6_report_design_20261104.md`` v6 §2.5 / §5.3.1 / §7.3 / §8.1 / §10；
- 方案 ``docs/contracts/v2_architecture_plan.md`` F23 §4.1 / §8-P6。

本模块只承载 T0.5 必须冻结的**结构性常量**（compare 映射、canonical 字段、精确写白名单、
validation 根、依赖上下限、验收精确版本、语料规模），不含渲染、读取或指标计算实现。
冻结快照落 ``docs/contracts/p6_t05_contract_freeze_20261105.json``（由
``scripts/v2_p6/freeze_t05_contracts.py`` 生成），测试断言两者一致，禁止文档与代码静默漂移。
"""

from __future__ import annotations

from typing import Any

from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "ACCEPTANCE_VERSIONS",
    "CANONICAL_RULES",
    "COMPARE_MAPPING",
    "CONTRACT_DATE",
    "CONTRACT_VERSION",
    "CORPUS_PROFILES",
    "CORPUS_SCHEMA",
    "DEPENDENCY_BOUNDS",
    "EVIDENCE_CLASSES",
    "FREEZE_SCHEMA",
    "MANIFEST_REQUIRED_FIELDS",
    "REPORT_KINDS",
    "REPORT_ID_PATTERN",
    "REPORTS_ROOT",
    "THRESHOLDS",
    "VALIDATION_ROOT",
    "WRITE_WHITELIST",
    "as_frozen_payload",
    "frozen_contract_sha256",
]

#: 冻结记录 schema（``docs/contracts/p6_t05_contract_freeze_20261105.json`` 的 ``schema``）
FREEZE_SCHEMA = "p6_t05_contract_freeze/v1"
#: 本冻结件对应的阶段版本
CONTRACT_VERSION = "T0.5"
#: 冻结日期
CONTRACT_DATE = "2026-11-05"
#: 生效方案引用（实现只能引用该 F 版本 + commit）
ARCHITECTURE_REFERENCE = "F23@b476ef506c550f86d4268fc537bf040dbb6dd3b3"

#: 三类证据域，物理与视觉均隔离，不提供跨类总览
EVIDENCE_CLASSES: tuple[str, ...] = ("production", "research", "calibration_only")
#: 三类报告面
REPORT_KINDS: tuple[str, ...] = ("dashboard", "experiment", "trend")
#: 报告产物根（相对仓库根）
REPORTS_ROOT = "data/reports/p6"
#: validation 根模式（不存在的 validation_id 才能创建）
VALIDATION_ROOT = "data/reports/p6/validations/<evidence_class>/<validation_id>"
#: 正式 report_id 形态：``p6_<kind>_<UTC YYYYMMDDTHHMMSSZ>_<input_manifest_sha256 前 12 位>``
REPORT_ID_PATTERN = r"^p6_(dashboard|experiment|trend)_\d{8}T\d{6}Z_[0-9a-f]{12}$"

#: 精确副作用白名单（按命令分别断言；``/**`` 表示该目录内全部新产物）
WRITE_WHITELIST: dict[str, tuple[str, ...]] = {
    "generate": (
        "data/reports/p6/.staging/<attempt>/**",
        "data/reports/p6/<evidence_class>/<kind>/<report_id>/**",
    ),
    "verify": (),
    "reproduce": ("data/reports/p6/reproductions/<new_id>/**",),
    "measure": ("data/reports/p6/validations/<evidence_class>/<validation_id>/**",),
    "real_golden": ("data/reports/p6/validations/<evidence_class>/<validation_id>/**",),
}

#: manifest 必需字段（设计 §8.1）；顺序即 canonical 输出的稳定顺序参考
MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "report_id",
    "report_kind",
    "evidence_class",
    "input_evidence_classes",
    "generated_at",
    "data_window",
    "source_watermarks",
    "root_map",
    "inputs",
    "projection_sha256",
    "input_manifest_sha256",
    "generator",
    "outputs",
    "csv_schemas",
    "degradations",
    "record_sha256",
)

#: canonical 输出规则（设计 §8.2/§8.3；只登记可机械核对的口径）
CANONICAL_RULES: tuple[str, ...] = (
    "文件编码 UTF-8 无 BOM、换行 LF；HTML 不引入运行时 JavaScript",
    "来源文本先做 Unicode NFC，再转义 < > & U+2028 U+2029；DOM 只写文本节点",
    "路径一律 root-relative POSIX；禁止绝对路径、目录枚举序与 mtime 参与输出",
    "canonical JSON = sort_keys + compact separators + UTF-8（复用 canonical_json_sha256）",
    "generated_at 只取 UTC YYYYMMDDTHHMMSSZ 且作为显式输入；复算读回原值",
    "数值先验有限，JSON 序列化 allow_nan=False；集合按稳定键排序",
    "性能时长、环境信息、浏览器 profile 只进 validations，不进入 canonical 报告",
)

#: 唯一实现输出 → P6 列映射（设计 §5.3.1；禁止按英文同名键猜测）
COMPARE_MAPPING: tuple[dict[str, str], ...] = (
    {"source": "compare_runs summary 运行", "p6_column": "run_label", "rule": "原值"},
    {"source": "compare_runs 子集CAGR", "p6_column": "subset_cagr", "rule": "原值"},
    {"source": "compare_runs 子集最大回撤", "p6_column": "subset_max_drawdown", "rule": "原值"},
    {"source": "compare_runs 子集夏普", "p6_column": "subset_sharpe", "rule": "原值"},
    {"source": "compare_runs ΔCAGR(vs基线)", "p6_column": "delta_cagr", "rule": "不重算"},
    {
        "source": "compare_runs Δ最大回撤(vs基线)",
        "p6_column": "delta_max_drawdown",
        "rule": "原值；正值改善",
    },
    {"source": "compare_runs Δ夏普(vs基线)", "p6_column": "delta_sharpe", "rule": "不重算"},
    {
        "source": "compare_runs 折收益同向(vs基线)",
        "p6_column": "fold_return_aligned_count",
        "rule": "计数原值",
    },
    {
        "source": "compare_runs fold_table 折序号/折收益/折最大回撤",
        "p6_column": "split_index/fold_return/fold_max_drawdown",
        "rule": "原值",
    },
    {
        "source": "旧 WF summary KEY_Top20_hit_rate / KEY_Top30_hit_rate",
        "p6_column": "topk20_positive_day_ratio / topk30_positive_day_ratio",
        "rule": "仅字段改名",
    },
    {
        "source": "aggregate_run key_top20_hit_rate_mean / key_top30_hit_rate_mean",
        "p6_column": "同名趋势列",
        "rule": "既有跨折不加权均值",
    },
)

#: 验收阈值（设计 §7.1；判定逻辑在 ``perf_gate``，数值唯一来源在此）
THRESHOLDS: dict[str, int] = {
    "max_report_bytes": 3_000_000,
    "max_open_ms": 2_000,
    "max_generation_ms": 10_000,
    "cold_repeats": 5,
    "hot_repeats": 5,
    "generation_repeats": 5,
}

#: 依赖兼容上下限；Playwright/Chromium 只允许验收侧使用
DEPENDENCY_BOUNDS: dict[str, dict[str, str]] = {
    "jinja2": {"minimum": "3.1", "maximum_exclusive": "4.0", "scope": "runtime"},
    "playwright": {"minimum": "1.47", "maximum_exclusive": "2.0", "scope": "verification_only"},
}

#: 验收环境固定精确版本（T0.5 实测安装值；变更须重新标定并出新冻结件）
ACCEPTANCE_VERSIONS: dict[str, str] = {
    "jinja2": "3.1.6",
    "playwright": "1.64.1",
    "chromium": "156.0.8078.4",
}

#: 合成语料 schema（规模语料，非 P6 输入语义契约；语义夹具归 T1/T3）
CORPUS_SCHEMA = "p6_t05_corpus/v1"
#: §7.3 标准/压力语料规模
CORPUS_PROFILES: dict[str, dict[str, dict[str, int]]] = {
    "standard": {
        "dashboard": {
            "positions": 20,
            "instructions": 35,
            "nav_rows": 250,
            "quality_abnormal_rows": 100,
        },
        "experiment": {
            "arms": 2,
            "folds": 14,
            "chain_rows_per_arm": 1_700,
            "daily_rows_per_arm": 1_700,
            "trade_rows_per_arm": 2_000,
            "attribution_rows_per_arm": 2_000,
            "topk_detail_rows_per_arm": 0,
        },
        "trend": {
            "previous_runs": 499,
            "new_runs": 1,
            "visible_rows": 500,
            "full_rows": 500,
        },
    },
    "stress": {
        "dashboard": {
            "positions": 200,
            "instructions": 500,
            "nav_rows": 5_000,
            "quality_abnormal_rows": 1_000,
        },
        "experiment": {
            "arms": 2,
            "folds": 14,
            "chain_rows_per_arm": 5_000,
            "daily_rows_per_arm": 5_000,
            "trade_rows_per_arm": 10_000,
            "attribution_rows_per_arm": 10_000,
            "topk_detail_rows_per_arm": 20_000,
        },
        "trend": {
            "previous_runs": 1_999,
            "new_runs": 1,
            "visible_rows": 500,
            "full_rows": 2_000,
        },
    },
}


def as_frozen_payload() -> dict[str, Any]:
    """返回可 JSON 序列化的冻结载荷；顺序与列表形态固定，供快照逐字节比对。"""
    return {
        "schema": FREEZE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_date": CONTRACT_DATE,
        "architecture_reference": ARCHITECTURE_REFERENCE,
        "evidence_classes": list(EVIDENCE_CLASSES),
        "report_kinds": list(REPORT_KINDS),
        "reports_root": REPORTS_ROOT,
        "validation_root": VALIDATION_ROOT,
        "report_id_pattern": REPORT_ID_PATTERN,
        "write_whitelist": {name: list(patterns) for name, patterns in WRITE_WHITELIST.items()},
        "compare_mapping": [dict(row) for row in COMPARE_MAPPING],
        "manifest_required_fields": list(MANIFEST_REQUIRED_FIELDS),
        "canonical_rules": list(CANONICAL_RULES),
        "dependency_bounds": {name: dict(bounds) for name, bounds in DEPENDENCY_BOUNDS.items()},
        "acceptance_versions": dict(ACCEPTANCE_VERSIONS),
        "thresholds": dict(THRESHOLDS),
        "corpus_schema": CORPUS_SCHEMA,
        "corpus_profiles": {
            profile: {section: dict(counts) for section, counts in sections.items()}
            for profile, sections in CORPUS_PROFILES.items()
        },
    }


def frozen_contract_sha256() -> str:
    """冻结载荷的 canonical SHA-256；快照文件与代码同源校验使用。"""
    return canonical_json_sha256(as_frozen_payload())
