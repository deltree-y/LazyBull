#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于已训练模型分析因子使用稳定性。

本脚本只读取模型和特征文件，不修改模型注册表或因子排除清单。对于
EnsembleModel，会展开内部子模型后分别统计，避免顶层对象没有
feature_importances_ 时丢失信息。

使用示例：
    python scripts/ana/analyze_factor_stability.py --versions 22626-22639
    python scripts/ana/analyze_factor_stability.py \
        --versions 22626-22639,22654 --output data/reports/factor_stability.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_data_root
from src.lazybull.common.logger import setup_logger


def parse_versions(raw_versions: str) -> List[int]:
    """解析逗号分隔的版本号和闭区间，返回去重后的升序版本列表。"""
    versions = set()
    for raw_part in str(raw_versions).split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw.lstrip("v"))
            end = int(end_raw.lstrip("v"))
            if start > end:
                raise ValueError(f"版本区间起点大于终点: {raw_part}")
            versions.update(range(start, end + 1))
        else:
            versions.add(int(part.lstrip("v")))

    if not versions:
        raise ValueError("未提供有效模型版本")
    return sorted(versions)


def _iter_leaf_models(model: Any) -> Iterable[Any]:
    """递归展开集成模型，产出具有 feature_importances_ 的叶子模型。"""
    if hasattr(model, "feature_importances_"):
        yield model
        return

    child_models = getattr(model, "models", None)
    if child_models is None:
        raise TypeError(f"模型 {type(model).__name__} 不含 feature_importances_ 或 models")
    for child_model in child_models:
        yield from _iter_leaf_models(child_model)


def _resolve_feature_names(model: Any, fallback_names: Sequence[str]) -> List[str]:
    """优先使用子模型记录的特征名，缺失时回退到版本 features 文件。"""
    model_names = getattr(model, "feature_names_in_", None)
    if model_names is not None:
        return [str(name) for name in model_names]
    return [str(name) for name in fallback_names]


def load_importance_records(models_dir: Path, versions: Sequence[int]) -> pd.DataFrame:
    """加载指定顶层模型版本，并展开为逐子模型、逐因子的 importance 记录。"""
    records: List[Dict[str, Any]] = []

    for version in versions:
        version_name = f"v{version}"
        model_path = models_dir / f"{version_name}_model.joblib"
        features_path = models_dir / f"{version_name}_features.json"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {features_path}")

        model = joblib.load(model_path)
        with open(features_path, "r", encoding="utf-8") as file:
            fallback_names = json.load(file)
        if not isinstance(fallback_names, list):
            raise ValueError(f"特征文件格式错误，应为列表: {features_path}")

        leaf_models = list(_iter_leaf_models(model))
        if not leaf_models:
            raise ValueError(f"模型未包含可分析的叶子模型: {version_name}")

        for component_index, leaf_model in enumerate(leaf_models):
            importance = np.asarray(leaf_model.feature_importances_, dtype=float)
            feature_names = _resolve_feature_names(leaf_model, fallback_names)
            if len(importance) != len(feature_names):
                raise ValueError(
                    f"{version_name} 子模型{component_index} importance/特征长度不一致: "
                    f"{len(importance)} != {len(feature_names)}"
                )

            total_importance = float(np.nansum(importance))
            normalized = (
                importance / total_importance
                if total_importance > 0
                else np.zeros(len(importance), dtype=float)
            )
            component_key = f"{version_name}#{component_index:02d}"
            component_rank = pd.Series(normalized).rank(ascending=False, method="average")
            top_half_limit = max(1, int(np.ceil(len(feature_names) / 2)))

            for feature, raw_value, normalized_value, rank in zip(
                feature_names, importance, normalized, component_rank
            ):
                records.append(
                    {
                        "version": version,
                        "component": component_key,
                        "feature": feature,
                        "importance": float(raw_value),
                        "normalized_importance": float(normalized_value),
                        "rank": float(rank),
                        "is_zero": bool(raw_value <= 0),
                        "is_top_half": bool(rank <= top_half_limit),
                    }
                )

        logger.info(f"{version_name}: 展开 {len(leaf_models)} 个子模型")

    return pd.DataFrame(records)


def compute_stability_stats(
    records: pd.DataFrame,
    bottom_ratio: float = 0.20,
    min_zero_ratio: float = 0.50,
    max_top_half_ratio: float = 0.20,
) -> pd.DataFrame:
    """聚合因子稳定性，并标记长期低使用候选。"""
    if records.empty:
        raise ValueError("importance 记录为空")
    if not 0 < bottom_ratio < 1:
        raise ValueError("bottom_ratio 必须位于 (0, 1)")

    component_count = records["component"].nunique()
    stats = (
        records.groupby("feature")
        .agg(
            mean_normalized_importance=("normalized_importance", "mean"),
            median_normalized_importance=("normalized_importance", "median"),
            std_normalized_importance=("normalized_importance", "std"),
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            zero_ratio=("is_zero", "mean"),
            top_half_ratio=("is_top_half", "mean"),
            appear_count=("component", "nunique"),
        )
        .reset_index()
    )
    stats["appear_ratio"] = stats["appear_count"] / component_count

    low_importance_threshold = stats["mean_normalized_importance"].quantile(bottom_ratio)
    stats["low_importance"] = stats["mean_normalized_importance"] <= low_importance_threshold
    stats["high_zero_ratio"] = stats["zero_ratio"] >= min_zero_ratio
    stats["low_top_half_ratio"] = stats["top_half_ratio"] <= max_top_half_ratio
    stats["importance_candidate"] = (
        stats["low_importance"] & stats["high_zero_ratio"] & stats["low_top_half_ratio"]
    )
    stats["review_candidate"] = stats["low_importance"] & stats["low_top_half_ratio"]
    stats["candidate_reason"] = np.where(
        stats["importance_candidate"],
        "低归一化重要性 + 高零值率 + 低Top50%出现率",
        np.where(
            stats["review_candidate"],
            "低归一化重要性 + 低Top50%出现率（待IC复核）",
            "",
        ),
    )

    return stats.sort_values(
        [
            "importance_candidate",
            "review_candidate",
            "mean_normalized_importance",
            "mean_rank",
        ],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def print_report(stats: pd.DataFrame, versions: Sequence[int], component_count: int) -> None:
    """打印紧凑的分析摘要。"""
    candidates = stats[stats["importance_candidate"]]
    review_candidates = stats[stats["review_candidate"]]
    print("\n" + "=" * 88)
    print("因子使用稳定性分析")
    print("=" * 88)
    print(f"顶层模型版本: v{versions[0]} ~ v{versions[-1]}（共 {len(versions)} 个）")
    print(f"子模型数: {component_count}")
    print(f"因子数: {len(stats)}")
    print(f"importance 低使用候选数: {len(candidates)}")
    print(f"待IC复核观察数: {len(review_candidates)}")
    print("注意: importance 没有方向信息，候选仍需结合单因子IC和消融实验确认。")

    if len(review_candidates) > 0:
        display_columns = [
            "feature",
            "mean_normalized_importance",
            "mean_rank",
            "zero_ratio",
            "top_half_ratio",
            "appear_ratio",
        ]
        print("\n待IC复核观察名单:")
        print(
            review_candidates[display_columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="分析已训练模型的因子使用稳定性")
    parser.add_argument(
        "--versions",
        required=True,
        help="模型版本，支持逗号和闭区间，例如 22626-22639,22654",
    )
    parser.add_argument("--data-root", default=None, help="数据根目录，默认读取项目配置")
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV，默认 data/reports/factor_stability_<首版本>_<末版本>.csv",
    )
    parser.add_argument("--bottom-ratio", type=float, default=0.20)
    parser.add_argument("--min-zero-ratio", type=float, default=0.50)
    parser.add_argument("--max-top-half-ratio", type=float, default=0.20)
    args = parser.parse_args()

    setup_logger("INFO")
    versions = parse_versions(args.versions)
    data_root = Path(args.data_root or get_data_root())
    models_dir = data_root / "models"
    output_path = (
        Path(args.output)
        if args.output
        else (data_root / "reports" / f"factor_stability_{versions[0]}_{versions[-1]}.csv")
    )

    records = load_importance_records(models_dir, versions)
    stats = compute_stability_stats(
        records,
        bottom_ratio=args.bottom_ratio,
        min_zero_ratio=args.min_zero_ratio,
        max_top_half_ratio=args.max_top_half_ratio,
    )
    print_report(stats, versions, records["component"].nunique())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"稳定性报告已保存: {output_path}")


if __name__ == "__main__":
    main()
