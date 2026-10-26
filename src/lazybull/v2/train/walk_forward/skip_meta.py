# -*- coding: utf-8 -*-
"""skip-training 复用旧模型的元数据一致性校验(v2 侧)。

P3-T3 搬运改写自旧 runner.py:109-204 ``_load_skip_training_metadata``
(mccabe 15 在册;orchestrator.py 文件行硬上限拆分,§17.8):schema 版本常量与
read_* 纯函数改指 ``v2.common.model_schema``(R2-02 唯一来源,train 不 import
factors);**v2 注册表读路径 sidecar 缺失即报错**(S3,差异登记 §16.6-1)。
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from src.lazybull.v2.common.model_registry import ModelRegistry

from .training_core import TrainFoldParams


def _load_skip_training_metadata(
    registry: ModelRegistry, model_version: int, params: TrainFoldParams
) -> Optional[Dict[str, Any]]:
    """skip-training 复用旧模型时,核验一致预期修正/现金流质量开关与 schema 版本一致性。

    P3-T3 搬运改写(旧 runner.py:109-204,mccabe 15 在册拆分):schema 版本常量与
    read_* 纯函数改指 ``v2.common.model_schema``(R2-02 唯一来源,train 不 import
    factors);**v2 注册表读路径 sidecar 缺失即报错**(S3——旧 = 静默回填,差异已登记
    §16.6-1;skip 复用的验收模型均有 sidecar)。
    开关不一致或旧模型未记录 schema 版本时仅告警(模型仍可用)。
    """
    metadata = _read_skip_metadata(registry, model_version)
    if metadata is None:
        return None
    train_params = metadata.get("train_params") or {}
    _warn_cons_revision_mismatch(train_params, params, model_version)
    _warn_cashflow_mismatch(train_params, params, model_version)
    _warn_dividend_mismatch(train_params, params, model_version)
    metadata["feature_columns"] = _read_skip_feature_columns(registry, model_version, metadata)
    return metadata


def _read_skip_metadata(registry: ModelRegistry, model_version: int) -> Optional[Dict[str, Any]]:
    """读旧模型 metadata(失败 None);v2 注册表 sidecar 缺失即报错(S3)。"""
    metadata = registry._load_metadata(model_version)
    if not metadata:
        logger.warning(f"[skip-training] 无法读取 v{model_version} 元数据,跳过开关一致性校验")
        return None
    return metadata


def _warn_cons_revision_mismatch(
    train_params: Dict[str, Any], params: TrainFoldParams, model_version: int
) -> None:
    """一致预期修正开关与 schema 版本一致性校验(逐字)。"""
    from src.lazybull.v2.common.model_schema import (
        CONSENSUS_REVISION_SCHEMA_VERSION,
        read_cons_revision_schema_version,
    )

    requested = bool(params.enable_consensus_revision_features)
    recorded = bool(train_params.get("enable_consensus_revision_features", False))
    if requested != recorded:
        logger.warning(
            f"[skip-training] v{model_version} 训练时 enable_consensus_revision_features="
            f"{recorded},当前配置为 {requested},开关与模型实际特征不一致,"
            "消融归因请以模型 metadata 的实际 feature_columns 为准"
        )
    if recorded:
        schema_version = read_cons_revision_schema_version(train_params)
        if schema_version != CONSENSUS_REVISION_SCHEMA_VERSION:
            logger.warning(
                f"[skip-training] v{model_version} 未记录 v2 修正 schema 版本"
                f"(记录值: {train_params.get('cons_revision_schema_version')}),"
                "若复用当前特征分区将静默读取 v2 语义列,存在 train/serve 语义偏差,"
                "建议停用或重训"
            )


def _warn_cashflow_mismatch(
    train_params: Dict[str, Any], params: TrainFoldParams, model_version: int
) -> None:
    """现金流质量开关与 schema 版本一致性校验(逐字)。"""
    from src.lazybull.v2.common.model_schema import (
        CASHFLOW_QUALITY_SCHEMA_VERSION,
        read_cashflow_quality_schema_version,
    )

    requested_cfq = bool(params.enable_cashflow_quality_features)
    recorded_cfq = bool(train_params.get("enable_cashflow_quality_features", False))
    if requested_cfq != recorded_cfq:
        logger.warning(
            f"[skip-training] v{model_version} 训练时 enable_cashflow_quality_features="
            f"{recorded_cfq},当前配置为 {requested_cfq},开关与模型实际特征不一致,"
            "消融归因请以模型 metadata 的实际 feature_columns 为准"
        )
    if recorded_cfq:
        recorded_cfq_schema = read_cashflow_quality_schema_version(train_params)
        if recorded_cfq_schema != CASHFLOW_QUALITY_SCHEMA_VERSION:
            logger.warning(
                f"[skip-training] v{model_version} 记录的现金流质量 schema 版本为 "
                f"{train_params.get('cashflow_quality_schema_version')},当前构建管线为 "
                f"v{CASHFLOW_QUALITY_SCHEMA_VERSION}(依赖修订事件驱动 TTM + 确定性"
                "版本去重),复用该模型存在 train/serve 语义偏差,建议停用或重训"
            )


def _warn_dividend_mismatch(
    train_params: Dict[str, Any], params: TrainFoldParams, model_version: int
) -> None:
    """分红政策开关与 schema 版本一致性校验(逐字)。"""
    from src.lazybull.v2.common.model_schema import (
        DIVIDEND_POLICY_SCHEMA_VERSION,
        read_dividend_policy_schema_version,
    )

    requested_div = bool(params.enable_dividend_policy_features)
    recorded_div = bool(train_params.get("enable_dividend_policy_features", False))
    if requested_div != recorded_div:
        logger.warning(
            f"[skip-training] v{model_version} 训练时 enable_dividend_policy_features="
            f"{recorded_div},当前配置为 {requested_div},开关与模型实际特征不一致,"
            "消融归因请以模型 metadata 的实际 feature_columns 为准"
        )
    if recorded_div:
        recorded_div_schema = read_dividend_policy_schema_version(train_params)
        if recorded_div_schema != DIVIDEND_POLICY_SCHEMA_VERSION:
            logger.warning(
                f"[skip-training] v{model_version} 记录的分红政策 schema 版本为 "
                f"{train_params.get('dividend_policy_schema_version')},当前构建管线为 "
                f"v{DIVIDEND_POLICY_SCHEMA_VERSION},复用该模型存在 train/serve 语义偏差,"
                "建议停用或重训"
            )


def _read_skip_feature_columns(
    registry: ModelRegistry, model_version: int, metadata: Dict[str, Any]
) -> List[str]:
    """特征列保存在独立 features 文件(metadata 本身不含),补齐后供汇总透传(逐字)。"""
    features_file_name = metadata.get("features_file") or f"v{model_version}_features.json"
    features_file = registry.models_dir / features_file_name
    feature_columns: List[str] = []
    if features_file.exists():
        try:
            import json

            with open(features_file, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, list):
                feature_columns = [str(c) for c in loaded]
        except (OSError, ValueError) as exc:
            logger.warning(f"[skip-training] 读取 v{model_version} 特征列表失败: {exc}")
    else:
        logger.warning(f"[skip-training] v{model_version} 特征列表文件缺失: {features_file.name}")
    return feature_columns
