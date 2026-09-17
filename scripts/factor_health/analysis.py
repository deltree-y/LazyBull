# -*- coding: utf-8 -*-
"""因子体检分析：台账组装、跨期稳定性、相关性聚类与候选标记。"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from scripts.factor_health.constants import (
    BASE_GROUPS,
    FAMILY_CONSTANT_NAMES,
    HealthThresholds,
)

_VALUE_COLUMNS = [
    "coverage",
    "coverage_min_year",
    "ic_mean",
    "ic_t",
    "ic_ir",
    "gain_share_mean",
    "split_use_frac",
    "years_opposite",
    "years_flips",
    "year_ic_str",
    "cluster_id",
    "cluster_rep",
]


@lru_cache(maxsize=1)
def _family_membership() -> Dict[str, str]:
    """构建 特征名 -> 家族 的静态映射（含基础分组与开关常量家族）。"""
    from src.lazybull.ml.train_core import constants as train_constants

    mapping: Dict[str, str] = {}
    for constant_name, family in FAMILY_CONSTANT_NAMES.items():
        members = getattr(train_constants, constant_name, None)
        if not isinstance(members, (list, tuple, set)):
            continue
        for name in members:
            mapping[str(name)] = family
    for group, members in BASE_GROUPS.items():
        for name in members:
            mapping.setdefault(name, f"base-{group}")
    return mapping


def family_of(name: str) -> str:
    """解析特征所属家族；``_sz`` 变体继承其基础特征家族。"""
    membership = _family_membership()
    if name in membership:
        return membership[name]
    if name.endswith("_sz") and name[:-3] in membership:
        return membership[name[:-3]]
    if name.endswith("_freshness_days"):
        return "meta-freshness"
    if str(name).startswith("has_"):
        # 运行时可用性标记（factors/availability.py 定义，不写回特征分区）
        return "availability_marker"
    return "base-其他"


def attach_families(register: pd.DataFrame) -> pd.DataFrame:
    """为台账追加 ``family2`` 列（家族归属，``_sz`` 继承基础特征家族）。"""
    register["family2"] = [family_of(name) for name in register.index]
    return register


def parse_version_spec(raw: str) -> List[int]:
    """解析逗号分隔的版本规格（支持 ``v10-12,11`` 形式的开闭区间）。"""
    versions = set()
    for part in str(raw).split(","):
        part = part.strip().lower().lstrip("v")
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start > end:
                raise ValueError(f"版本区间起点大于终点: {part}")
            versions.update(range(start, end + 1))
        else:
            versions.add(int(part))
    if not versions:
        raise ValueError("未提供有效模型版本")
    return sorted(versions)


def map_booster_scores(scores: Dict[str, float], feature_names: Sequence[str]) -> Dict[str, float]:
    """将 XGBoost booster 的 ``f{i}`` 特征名映射回原始列名。"""
    mapped: Dict[str, float] = {}
    for key, value in scores.items():
        match = re.fullmatch(r"f(\d+)", str(key))
        if match and feature_names and int(match.group(1)) < len(feature_names):
            mapped[feature_names[int(match.group(1))]] = value
        else:
            mapped[str(key)] = value
    return mapped


def list_available_model_versions(model_dir: Path) -> List[int]:
    """列出目录内具备 ``v{N}_model.joblib`` 的版本号（升序）。"""
    versions: List[int] = []
    for path in Path(model_dir).glob("v*_model.joblib"):
        match = re.fullmatch(r"v(\d+)_model", path.stem)
        if match:
            versions.append(int(match.group(1)))
    return sorted(versions)


def resolve_latest_feature_file(model_dir: Path) -> Path:
    """解析最新模型的特征清单文件路径。"""
    model_dir = Path(model_dir)
    latest_file = model_dir / "latest_model_version.txt"
    version: Optional[int] = None
    if latest_file.exists():
        raw = latest_file.read_text(encoding="utf-8").strip()
        match = re.search(r"(\d+)", raw)
        if match:
            version = int(match.group(1))
    if version is None:
        versions = list_available_model_versions(model_dir)
        if not versions:
            raise FileNotFoundError(f"{model_dir} 下未找到任何模型文件，需显式指定 --feature-file")
        version = versions[-1]
    feature_file = model_dir / f"v{version}_features.json"
    if not feature_file.exists():
        raise FileNotFoundError(f"特征清单文件不存在: {feature_file}")
    return feature_file


def _iter_leaf_models(model: Any) -> List[Any]:
    """展开集成模型为叶子模型列表。"""
    if hasattr(model, "get_booster"):
        return [model]
    children = getattr(model, "models", None)
    if children is None:
        return []
    leaves: List[Any] = []
    for child in children:
        leaves.extend(_iter_leaf_models(child))
    return leaves


def compute_model_usage(
    model_dir: Path,
    versions: Sequence[int],
    feature_names: Sequence[str],
    loader: Optional[Callable[[Path], Any]] = None,
) -> pd.DataFrame:
    """统计因子在指定模型版本中的使用度（gain 份额与分裂使用率）。

    兼容 ``EnsembleModel``（展开子模型后逐子模型累计）。
    """
    if loader is None:
        import joblib  # 延迟导入，避免测试环境依赖

        loader = joblib.load

    feature_names = list(feature_names)
    gain_frames: List[pd.Series] = []
    split_frames: List[pd.Series] = []
    loaded_versions: List[int] = []

    for version in versions:
        model_path = Path(model_dir) / f"v{version}_model.joblib"
        if not model_path.exists():
            logger.warning(f"模型文件不存在，跳过: {model_path}")
            continue
        try:
            model = loader(model_path)
        except Exception as exc:  # noqa: BLE001 - 单个模型加载失败不阻断体检
            logger.warning(f"模型 v{version} 加载失败，跳过: {exc}")
            continue
        loaded_versions.append(version)
        for leaf in _iter_leaf_models(model):
            names = [str(name) for name in getattr(leaf, "feature_names_in_", feature_names)]
            booster = leaf.get_booster()
            gain = map_booster_scores(booster.get_score(importance_type="gain"), names)
            weight = map_booster_scores(booster.get_score(importance_type="weight"), names)
            total_gain = sum(gain.values()) or 1.0
            gain_frames.append(pd.Series({k: v / total_gain for k, v in gain.items()}))
            split_frames.append(pd.Series({k: 1.0 for k in weight}))

    if not gain_frames:
        empty = pd.DataFrame(columns=["gain_share_mean", "split_use_frac"])
        empty.attrs["loaded_versions"] = []
        empty.attrs["sub_model_count"] = 0
        return empty

    gain_table = pd.concat(gain_frames, axis=1).reindex(feature_names)
    split_table = pd.concat(split_frames, axis=1).reindex(feature_names)
    usage = pd.DataFrame(
        {
            "gain_share_mean": gain_table.mean(axis=1),
            "split_use_frac": split_table.notna().mean(axis=1),
        }
    )
    usage.attrs["loaded_versions"] = loaded_versions
    usage.attrs["sub_model_count"] = len(gain_frames)
    logger.info(f"模型使用度统计完成：版本 {loaded_versions}，子模型 {len(gain_frames)} 个")
    return usage


def cluster_features(
    corr_avg: pd.DataFrame, represent_score: pd.Series, abs_threshold: float = 0.85
) -> pd.DataFrame:
    """按平均截面相关矩阵对因子聚类，簇内选取代表因子。

    Args:
        corr_avg: 因子间平均相关系数矩阵。
        represent_score: 代表因子选择依据（一般取 |ic_ir|），值大者优先。
        abs_threshold: |rho| 达到该阈值视为同簇。

    Returns:
        DataFrame（cluster_id/size/members/max_abs_corr/representative/rep_score）。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    matrix = np.nan_to_num(corr_avg.to_numpy(dtype=float), nan=0.0)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    distance = 1.0 - np.abs(matrix)
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2, 0.0, None)
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    cluster_ids = fcluster(linkage_matrix, t=1.0 - abs_threshold, criterion="distance")
    cluster_series = pd.Series(cluster_ids, index=corr_avg.index, name="cluster_id")

    scores = represent_score.reindex(corr_avg.index).abs().fillna(0.0)
    records: List[Dict[str, object]] = []
    for cluster_id, members in cluster_series.groupby(cluster_series).groups.items():
        members = list(members)
        representative = str(scores.reindex(members).idxmax())
        if len(members) == 1:
            max_corr = np.nan
        else:
            sub = np.abs(
                matrix[
                    np.ix_(
                        [corr_avg.index.get_loc(m) for m in members],
                        [corr_avg.index.get_loc(m) for m in members],
                    )
                ].copy()
            )
            np.fill_diagonal(sub, 0.0)
            max_corr = float(np.max(sub))
        records.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(members),
                "members": "|".join(members),
                "max_abs_corr": max_corr,
                "representative": representative,
                "rep_score": float(scores[representative]),
            }
        )
    return pd.DataFrame(records).sort_values(["size", "cluster_id"], ascending=[False, True])


def assemble_register(
    coverage_by_date: pd.DataFrame,
    daily_ic: pd.DataFrame,
    market_vol_by_date: pd.Series,
) -> pd.DataFrame:
    """组装逐因子台账（覆盖、IC、分年、regime、聚类与使用度在外部合并）。"""
    features = list(coverage_by_date.columns)
    coverage = coverage_by_date.mean()
    years = sorted({str(date)[:4] for date in coverage_by_date.index})
    register = pd.DataFrame({"coverage": coverage.reindex(features)})
    for year in years:
        year_rows = [str(d)[:4] == year for d in coverage_by_date.index]
        register[f"cov_{year}"] = coverage_by_date.loc[year_rows].mean().reindex(features)
    register["coverage_min_year"] = register[[f"cov_{y}" for y in years]].min(axis=1)

    daily = daily_ic.copy()
    daily["year"] = daily["date"].astype(str).str[:4]
    grouped = daily.groupby("feature")["ic"]
    ic_stats = grouped.agg(["mean", "std", "count"]).rename(
        columns={"mean": "ic_mean", "std": "ic_std", "count": "ic_days"}
    )
    ic_stats["ic_ir"] = ic_stats["ic_mean"] / ic_stats["ic_std"]
    ic_stats["ic_t"] = ic_stats["ic_ir"] * np.sqrt(ic_stats["ic_days"])
    register = register.join(ic_stats)

    year_pivot = daily.pivot_table(index="feature", columns="year", values="ic", aggfunc="mean")
    year_counts = daily.pivot_table(index="feature", columns="year", values="ic", aggfunc="count")
    for year in year_pivot.columns:
        register[f"ic_{year}"] = year_pivot[year]
        register[f"ndays_{year}"] = year_counts[year]

    market_vol = market_vol_by_date.dropna()
    if len(market_vol) > 10:
        median = market_vol.median()
        high_vol_dates = set(market_vol[market_vol >= median].index)
        daily["regime"] = np.where(daily["date"].isin(high_vol_dates), "high_vol", "low_vol")
        regime_pivot = daily.pivot_table(
            index="feature", columns="regime", values="ic", aggfunc="mean"
        )
        register = register.join(regime_pivot)

    register.index.name = "feature"
    return register


def add_year_profiles(register: pd.DataFrame, min_abs_year_ic: float) -> pd.DataFrame:
    """追加跨年符号统计列：异号年数、相邻翻转次数、逐年 IC 文本。"""
    year_columns = [c for c in register.columns if re.fullmatch(r"ic_\d{4}", str(c))]

    def profile(row: pd.Series) -> Tuple[float, float, str]:
        yearly: List[Tuple[str, float]] = []
        for column in year_columns:
            year = str(column)[3:]
            count = row.get(f"ndays_{year}", 0)
            value = row[column]
            if pd.notna(count) and count >= 30 and pd.notna(value):
                yearly.append((year, float(value)))
        if len(yearly) < 2:
            return np.nan, np.nan, ""
        overall = (
            np.sign(row["ic_mean"])
            if pd.notna(row["ic_mean"])
            else np.sign(np.mean([v for _, v in yearly]))
        )
        strong = [(y, v) for y, v in yearly if abs(v) >= min_abs_year_ic and np.sign(v) != 0]
        opposite = int(sum(1 for _, v in strong if np.sign(v) != overall))
        flips = sum(
            1
            for previous, current in zip(strong, strong[1:])
            if np.sign(current[1]) != np.sign(previous[1])
        )
        text = "|".join(f"{y}:{v:+.3f}" for y, v in yearly)
        return opposite, flips, text

    profiles = register.apply(profile, axis=1, result_type="expand")
    register["years_opposite"] = profiles[0]
    register["years_flips"] = profiles[1]
    register["year_ic_str"] = profiles[2]
    return register


def flag_candidates(register: pd.DataFrame, thresholds: HealthThresholds) -> pd.DataFrame:
    """标记市场级常数、低覆盖、翻号、弱信息、未使用与同簇冗余。"""
    register["market_level"] = register["ic_mean"].isna() & (register["coverage"] > 0.9)
    register["flag_low_coverage"] = (register["coverage"] < thresholds.low_coverage) | (
        register["coverage_min_year"] < thresholds.low_year_coverage
    )
    register["flag_flip"] = (~register["market_level"]) & (
        (register["years_opposite"] >= thresholds.flip_years_opposite)
        | (register["years_flips"] >= thresholds.flip_year_count)
    )
    register["flag_weak"] = (
        (~register["market_level"])
        & (register["ic_t"].abs() < thresholds.weak_abs_t)
        & (register["gain_share_mean"].fillna(0.0) < thresholds.weak_gain_share)
    )
    register["flag_unused"] = (~register["market_level"]) & (
        (register["split_use_frac"].fillna(0.0) < thresholds.unused_split_use)
        | (register["gain_share_mean"].fillna(0.0) < thresholds.unused_gain_share)
    )
    register["flag_dup"] = (~register["market_level"]) & (
        register["cluster_rep"].notna() & (register.index != register["cluster_rep"])
    )
    return register


def attach_twin_info(register: pd.DataFrame) -> pd.DataFrame:
    """标注孪生关系（``X`` 与 ``X_sz`` 互为孪生）。"""
    names = set(register.index)
    register["is_sz"] = [name.endswith("_sz") for name in register.index]
    register["twin_base"] = [
        name[:-3] if name.endswith("_sz") else (name + "_sz" if name + "_sz" in names else "")
        for name in register.index
    ]
    return register


def attach_clusters(register: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """将聚类结果（cluster_id 与代表因子）挂载到台账。"""
    if clusters.empty:
        register["cluster_id"] = np.nan
        register["cluster_size"] = np.nan
        register["cluster_rep"] = register.index
        return register

    name_to_cluster: Dict[str, int] = {}
    cluster_sizes: Dict[int, int] = {}
    representatives: Dict[int, str] = {}
    for _, row in clusters.iterrows():
        cluster_id = int(row["cluster_id"])
        cluster_sizes[cluster_id] = int(row["size"])
        representatives[cluster_id] = str(row["representative"])
        for member in str(row["members"]).split("|"):
            name_to_cluster[member] = cluster_id

    register["cluster_id"] = register.index.map(name_to_cluster)
    register["cluster_size"] = register["cluster_id"].map(cluster_sizes)
    register["cluster_rep"] = register["cluster_id"].map(representatives)
    single_member = register["cluster_size"].fillna(0) <= 1
    register.loc[single_member, "cluster_rep"] = register.index[single_member]
    return register


def prefer_plain_representatives(
    clusters: pd.DataFrame, register: pd.DataFrame, score_column: str = "ic_ir"
) -> pd.DataFrame:
    """按「孪生对优先保留非 ``_sz`` 口径」重选簇代表（簇划分保持不变）。

    用于把「去重效应」与「size（市值中性化）暴露变化」分开归因：默认代表按 ``|ic_ir|``
    选出，而实测 37 对孪生中有 31 对是 ``_sz`` 口径更高，因此直接用默认去重清单会隐式削减
    size 暴露，必须配合「口径交换」清单做单变量对照。

    Args:
        clusters: :func:`cluster_features` 产物（含 members/representative）。
        register: 因子台账（提供打分列，索引 = 特征名）。
        score_column: 代表选择依据的列名（取绝对值，值大者优先）。

    Returns:
        与输入同形的 DataFrame，``representative`` 列已替换，并新增 ``swapped`` 列
        标记该簇是否发生了口径交换。
    """
    if clusters.empty:
        return clusters.copy()
    scores = register[score_column].abs().fillna(0.0)
    rows: List[Dict[str, object]] = []
    for _, row in clusters.iterrows():
        members = [name for name in str(row["members"]).split("|") if name]
        plain_members = [name for name in members if not name.endswith("_sz")]
        original = str(row["representative"])
        if plain_members:
            representative = str(scores.reindex(plain_members).idxmax())
        else:
            representative = original
        record = row.to_dict()
        record["representative"] = representative
        record["swapped"] = representative != original
        rows.append(record)
    return pd.DataFrame(rows)


def candidate_tables(register: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """产出五类候选表（列结构统一，供 CSV 落盘与报告引用）。"""
    display = [c for c in _VALUE_COLUMNS if c in register.columns]
    dedup = register[register["flag_dup"]]
    if "cluster_id" in dedup.columns:
        dedup = dedup.sort_values("cluster_id")
    tables = {
        "low_coverage": register[register["flag_low_coverage"]].sort_values("coverage"),
        "sign_flip": register[register["flag_flip"]].sort_values(
            ["years_opposite", "years_flips"], ascending=False
        ),
        "weak": register[register["flag_weak"]].sort_values("ic_t", key=lambda s: s.abs()),
        "unused": register[register["flag_unused"]].sort_values(
            ["split_use_frac", "gain_share_mean"]
        ),
        "dedup": dedup,
    }
    return {name: table[display] for name, table in tables.items()}
