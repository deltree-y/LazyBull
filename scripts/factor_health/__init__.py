# -*- coding: utf-8 -*-
"""因子体检子包：覆盖率 / 跨期 RankIC 稳定性 / 相关性聚类 / 模型使用度。

入口脚本：``scripts/analyze_factor_health.py``。
"""

from scripts.factor_health.analysis import (
    add_year_profiles,
    assemble_register,
    attach_clusters,
    attach_families,
    attach_twin_info,
    candidate_tables,
    cluster_features,
    compute_model_usage,
    family_of,
    flag_candidates,
    list_available_model_versions,
    map_booster_scores,
    parse_version_spec,
    resolve_latest_feature_file,
)
from scripts.factor_health.constants import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MARKET_VOL_COLUMN,
    HealthThresholds,
)
from scripts.factor_health.report import (
    build_exclude_lists,
    build_report_markdown,
    summarize,
    write_health_outputs,
)
from scripts.factor_health.scan import (
    ScanResult,
    pick_corr_dates,
    pick_partition_files,
    scan_features,
)

__all__ = [
    "DEFAULT_LABEL_COLUMN",
    "DEFAULT_MARKET_VOL_COLUMN",
    "HealthThresholds",
    "ScanResult",
    "add_year_profiles",
    "assemble_register",
    "attach_clusters",
    "attach_families",
    "attach_twin_info",
    "build_exclude_lists",
    "build_report_markdown",
    "candidate_tables",
    "cluster_features",
    "compute_model_usage",
    "family_of",
    "flag_candidates",
    "list_available_model_versions",
    "map_booster_scores",
    "parse_version_spec",
    "pick_corr_dates",
    "pick_partition_files",
    "resolve_latest_feature_file",
    "scan_features",
    "summarize",
    "write_health_outputs",
]
