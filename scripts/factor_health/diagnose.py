# -*- coding: utf-8 -*-
"""因子诊断 v2：增量信息（偏 IC）、覆盖显著性、使用度稳定性。

与体检（v1）的分工：
    - 体检输出“列级描述”（覆盖率、IC、聚类、使用度）与候选清单；
    - 本模块进一步回答“**这条候选值不值得改数据**”：
      ① 偏 IC —— 控制簇代表后还剩多少截面信息（真正的“冗余”定义）；
      ② 覆盖-标签差 —— 有值/缺失两类子样本的标签与规模差（缺失是否携带信息/是否只是规模代理）；
      ③ 使用度稳定性 —— 跨模型版本的 gain 份额离散度（稳定出力 vs 偶发出力）。

只读 cs_train 分区与模型文件，不写回任何数据；结论落 CSV/markdown，供人工筛选后再走 WF A/B。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from scripts.factor_health.analysis import (
    _iter_leaf_models,
    map_booster_scores,
)


@dataclass
class DiagnoseThresholds:
    """诊断阈值（全部可经 CLI 覆盖）。"""

    min_pairs: int = 200  # 单日截面最少配对样本
    coverage_max: float = 0.95  # 覆盖率低于该值的列参与覆盖-标签诊断
    min_group_rows: int = 30  # 有值/缺失任一组的最少行数
    weak_partial_t: float = 1.5  # |偏 IC t| 低于该值视为无增量信息
    strong_ic_t: float = 5.0  # |IC t| 高于该值视为有信号
    unused_split_use: float = 0.6  # 使用率低于该值视为未被模型使用
    unstable_cv: float = 1.0  # gain 份额的变异系数高于该值视为使用不稳定
    collapse_ratio: float = 0.1  # 近年截面 std 中位 / 历史中位 低于该值视为口径退化
    missing_signal_diff: float = 0.005  # 有值/缺失标签中位差（绝对）高于该值视为缺失携带信息
    missing_size_diff: float = 0.3  # 规模分位差（绝对）低于该值视为“不是规模代理”
    extra: Dict[str, float] = field(default_factory=dict)


def _dup_free(columns: Sequence[str]) -> List[str]:
    """按原顺序去重列名（避免 read_parquet 返回重复列）。"""
    return list(dict.fromkeys(str(name) for name in columns))


def partial_ic_table(
    files: Sequence[Path],
    pairs: Dict[str, str],
    label_column: str,
    min_pairs: int = 200,
    progress_every: int = 0,
) -> pd.DataFrame:
    """控制簇代表后的偏 RankIC（逐日：秩残差 vs 标签秩）。

    Args:
        files: 采样分区文件。
        pairs: ``{待检验列: 控制列}``。
        label_column: 标签列。

    Returns:
        DataFrame（索引 = 待检验列，列含 partial_ic_mean/std/t/days/control）。
    """
    if not pairs:
        return pd.DataFrame()
    needed = _dup_free(list(pairs.keys()) + list(pairs.values()) + [label_column])
    records: Dict[str, List[float]] = {name: [] for name in pairs}
    for index, path in enumerate(files):
        frame = pd.read_parquet(path, columns=needed)
        label_rank = frame[label_column].rank(numeric_only=True)
        for feature, control in pairs.items():
            x = frame[feature].rank(numeric_only=True)
            c = frame[control].rank(numeric_only=True)
            mask = x.notna() & c.notna() & label_rank.notna()
            if int(mask.sum()) < min_pairs:
                continue
            xv = x[mask].to_numpy(dtype=float)
            cv = c[mask].to_numpy(dtype=float)
            yv = label_rank[mask].to_numpy(dtype=float)
            variance = cv.var()
            if variance == 0 or yv.std() == 0:
                continue
            beta = np.cov(xv, cv, ddof=0)[0, 1] / variance
            residual = xv - (xv.mean() - beta * cv.mean()) - beta * cv
            residual_rank = pd.Series(residual).rank().to_numpy(dtype=float)
            if residual_rank.std() == 0:
                # 残差无方差 = 该列完全被控制列解释（如单调复制）→ 增量信息恰为 0
                records[feature].append(0.0)
                continue
            records[feature].append(float(np.corrcoef(residual_rank, yv)[0, 1]))
        if progress_every and (index + 1) % progress_every == 0:
            logger.info(f"  偏 IC 扫描进度 {index + 1}/{len(files)}")

    rows = []
    for feature, values in records.items():
        series = np.array([v for v in values if np.isfinite(v)], dtype=float)
        if series.size < 30:
            rows.append(
                {
                    "feature": feature,
                    "control": pairs[feature],
                    "partial_ic_mean": np.nan,
                    "partial_ic_std": np.nan,
                    "partial_ic_t": np.nan,
                    "partial_days": int(series.size),
                }
            )
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=1))
        rows.append(
            {
                "feature": feature,
                "control": pairs[feature],
                "partial_ic_mean": mean,
                "partial_ic_std": std,
                "partial_ic_t": float(mean / std * np.sqrt(series.size)) if std > 0 else 0.0,
                "partial_days": int(series.size),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def column_year_profile(
    picks: Sequence[Tuple[str, Path]],
    features: Sequence[str],
) -> pd.DataFrame:
    """逐列逐年剖面：覆盖率（非空占比）与截面幅度（标准差）。

    Args:
        picks: ``[(年份, 该年代表分区路径), ...]``（每年一个分区即可）。
        features: 待统计列（不在某些年份 schema 中的列记为缺失）。

    Returns:
        长表（feature/year/rows/coverage/std/in_schema）。
    """
    features = [str(name) for name in _dup_free(features)]
    rows: List[Dict[str, object]] = []
    for year, path in picks:
        schema = set(pq.read_schema(path).names)
        available = [name for name in features if name in schema]
        frame = pd.read_parquet(path, columns=available) if available else pd.DataFrame()
        n_rows = int(len(frame))
        for name in features:
            if name not in schema:
                rows.append(
                    {
                        "feature": name,
                        "year": str(year),
                        "rows": n_rows,
                        "coverage": 0.0,
                        "std": np.nan,
                        "in_schema": False,
                    }
                )
                continue
            series = pd.to_numeric(frame[name], errors="coerce")
            valid = int(series.notna().sum())
            rows.append(
                {
                    "feature": name,
                    "year": str(year),
                    "rows": n_rows,
                    "coverage": float(valid / n_rows) if n_rows else 0.0,
                    "std": float(series.std()) if valid > 1 else np.nan,
                    "in_schema": True,
                }
            )
    return pd.DataFrame(rows)


def summarize_year_profile(
    profile: pd.DataFrame,
    min_effective_coverage: float = 0.2,
    recent_years: int = 3,
    collapse_ratio: float = 0.1,
    expansion_delta: float = 0.15,
) -> pd.DataFrame:
    """逐列根因判定：源头起点 / 资格扩张 / 覆盖收缩 / 口径退化（幅度塌缩）。

    启发式判据（均为“证据提示”，不替代人工核对 raw 层起点）：
        - **源头起点**：首个有效年份晚于数据起点 ≥3 年，且起点年覆盖率 ≤5%（如 cyq 2018、north 2014-11）；
        - **资格扩张**：末年少与首年少覆盖率提升 ≥``expansion_delta``（如两融标的名单扩容）；
        - **覆盖收缩**：覆盖率显著下降（披露率下降 / 上市数量增长稀释）；
        - **口径退化**：近 ``recent_years`` 年截面 std 中位 < 历史中位 × ``collapse_ratio``（如转融券暂停后的
          `rqye_rzye_ratio`）。
    """
    if profile.empty:
        return pd.DataFrame()
    profile = profile.copy()
    profile["coverage"] = pd.to_numeric(profile["coverage"], errors="coerce").fillna(0.0)
    profile["std"] = pd.to_numeric(profile["std"], errors="coerce")
    years = sorted(profile["year"].unique().tolist())
    first_year = int(years[0])

    records: List[Dict[str, object]] = []
    for feature, group in profile.groupby("feature"):
        group = group.sort_values("year")
        effective = group[(group["coverage"] >= min_effective_coverage) & (group["std"] > 0)]
        if effective.empty:
            records.append(
                {
                    "feature": feature,
                    "first_effective_year": None,
                    "years_effective": 0,
                    "coverage_first": np.nan,
                    "coverage_last": float(group["coverage"].iloc[-1]),
                    "hist_std_median": np.nan,
                    "recent_std_median": np.nan,
                    "collapse_ratio": np.nan,
                    "根因": "全程无截面幅度（常数或 schema 缺失）",
                }
            )
            continue
        first_effective_year = int(effective["year"].iloc[0])
        coverage_first = float(effective["coverage"].iloc[0])
        coverage_last = float(group["coverage"].iloc[-1])
        hist_median = float(effective["std"].median())
        span = (int(str(effective["year"].iloc[-1])) - first_effective_year) + 1
        recent_mask = (
            effective["year"].astype(int) > int(str(effective["year"].iloc[-1])) - recent_years
        )
        recent = effective[recent_mask]
        base = effective[~recent_mask]
        recent_median = float(recent["std"].median()) if not recent.empty else np.nan
        # 退化参考取“历史工作峰值”（排除近窗）：近 3 年幅度不足其历史峰值 10% 则判为幅度塔缩
        hist_peak = float(base["std"].max()) if not base.empty else np.nan
        ratio = recent_median / hist_peak if hist_peak and hist_peak > 0 else np.nan
        if span < 3:
            ratio = np.nan  # 有效跨度太短无法判退化

        labels: List[str] = []
        if first_effective_year - first_year >= 3:
            labels.append(f"源头起点（首个有效年 {first_effective_year}）")
        if coverage_first <= 0.35 and coverage_last - coverage_first >= expansion_delta:
            labels.append("资格/覆盖扩张")
        elif coverage_first - coverage_last >= expansion_delta:
            labels.append("覆盖收缩")
        if pd.notna(ratio) and ratio < collapse_ratio:
            labels.append(f"口径退化（幅度塌缩 {ratio:.3f}×）")
        records.append(
            {
                "feature": feature,
                "first_effective_year": first_effective_year,
                "years_effective": int(len(effective)),
                "coverage_first": coverage_first,
                "coverage_last": coverage_last,
                "hist_std_median": hist_median,
                "hist_std_max": hist_peak,
                "recent_std_median": recent_median,
                "collapse_ratio": ratio,
                "根因": "；".join(labels) if labels else "正常",
            }
        )
    return pd.DataFrame(records).set_index("feature")


def coverage_return_table(
    files: Sequence[Path],
    features: Sequence[str],
    label_column: str,
    size_column: Optional[str] = None,
    min_group_rows: int = 30,
    progress_every: int = 0,
) -> pd.DataFrame:
    """有值 / 缺失子样本的标签与规模差（逐日取中位数后跨日平均）。

    Returns:
        DataFrame（索引 = 特征；列含 has_median_mean/missing_median_mean/diff_mean/diff_pos_ratio、
        size_diff_mean（规模分位差，仅当提供 size_column）、days）。
    """
    features = [name for name in _dup_free(features)]
    if not features:
        return pd.DataFrame()
    needed = _dup_free(features + [label_column] + ([size_column] if size_column else []))
    records: Dict[str, Dict[str, List[float]]] = {
        name: {"diff": [], "size_diff": [], "has": [], "missing": []} for name in features
    }
    for index, path in enumerate(files):
        frame = pd.read_parquet(path, columns=needed)
        label = frame[label_column]
        size = frame[size_column] if size_column and size_column in frame.columns else None
        for feature in features:
            values = frame[feature]
            has = values.notna()
            n_has = int(has.sum())
            n_missing = int((~has).sum())
            if n_has < min_group_rows or n_missing < min_group_rows:
                continue
            has_label = label[has]
            missing_label = label[~has]
            if (
                has_label.notna().sum() < min_group_rows
                or missing_label.notna().sum() < min_group_rows
            ):
                continue
            has_median = float(has_label.median())
            missing_median = float(missing_label.median())
            record = records[feature]
            record["has"].append(has_median)
            record["missing"].append(missing_median)
            record["diff"].append(has_median - missing_median)
            if size is not None:
                record["size_diff"].append(float(size[has].median() - size[~has].median()))
        if progress_every and (index + 1) % progress_every == 0:
            logger.info(f"  覆盖-标签诊断进度 {index + 1}/{len(files)}")

    rows = []
    for feature, record in records.items():
        diffs = np.array(record["diff"], dtype=float)
        if diffs.size == 0:
            continue
        rows.append(
            {
                "feature": feature,
                "days": int(diffs.size),
                "has_median_mean": float(np.mean(record["has"])),
                "missing_median_mean": float(np.mean(record["missing"])),
                "diff_mean": float(diffs.mean()),
                "diff_pos_ratio": float((diffs > 0).mean()),
                "size_diff_mean": (
                    float(np.mean(record["size_diff"])) if record["size_diff"] else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def usage_stability_table(
    model_dir: Path,
    versions: Sequence[int],
    feature_names: Sequence[str],
    loader=None,
) -> pd.DataFrame:
    """跨模型版本的 gain 份额统计（均值/标准差/变异系数/最大最小）。"""
    if loader is None:
        import joblib

        loader = joblib.load

    feature_names = [str(name) for name in feature_names]
    frames: List[pd.Series] = []
    loaded: List[int] = []
    for version in versions:
        model_path = Path(model_dir) / f"v{version}_model.joblib"
        if not model_path.exists():
            continue
        try:
            model = loader(model_path)
        except Exception as exc:  # noqa: BLE001 - 单个模型失败不阻断诊断
            logger.warning(f"模型 v{version} 加载失败，跳过: {exc}")
            continue
        loaded.append(version)
        for leaf in _iter_leaf_models(model):
            names = [str(name) for name in getattr(leaf, "feature_names_in_", feature_names)]
            booster = leaf.get_booster()
            gain = map_booster_scores(booster.get_score(importance_type="gain"), names)
            total = sum(gain.values()) or 1.0
            frames.append(pd.Series({k: v / total for k, v in gain.items()}))
    if not frames:
        return pd.DataFrame()
    matrix = pd.concat(frames, axis=1).reindex(feature_names)
    table = pd.DataFrame(
        {
            "versions": len(frames),
            "gain_mean": matrix.mean(axis=1),
            "gain_std": matrix.std(axis=1),
            "gain_min": matrix.min(axis=1),
            "gain_max": matrix.max(axis=1),
            "gain_present_ratio": matrix.notna().mean(axis=1),
        }
    )
    table["gain_cv"] = table["gain_std"] / table["gain_mean"].replace(0.0, np.nan)
    for version in loaded:
        table.attrs.setdefault("loaded_versions", []).append(version)
    logger.info(f"使用度稳定性：加载模型 {len(loaded)} 个版本，子模型 {len(frames)} 个")
    return table


def build_candidate_list(
    register: pd.DataFrame,
    partial_ic: Optional[pd.DataFrame] = None,
    coverage_return: Optional[pd.DataFrame] = None,
    usage_stability: Optional[pd.DataFrame] = None,
    thresholds: Optional[DiagnoseThresholds] = None,
    year_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """汇总“数据改良候选清单”（含预期效应量级与可检出性标注）。"""
    thresholds = thresholds or DiagnoseThresholds()
    rows: List[Dict[str, object]] = []

    def add(
        kind: str, feature: str, evidence: str, scale: str, detectability: str, action: str
    ) -> None:
        rows.append(
            {
                "候选类型": kind,
                "因子": feature,
                "证据": evidence,
                "预期效应量级": scale,
                "可检出性(14折/MDE≈5pp)": detectability,
                "建议动作": action,
            }
        )

    for feature, row in register.iterrows():
        coverage = (
            float(row.get("coverage", np.nan)) if pd.notna(row.get("coverage", np.nan)) else np.nan
        )
        ic_t = row.get("ic_t", np.nan)

        # A 覆盖缺口：先判根因，不默认“补数据”
        if row.get("flag_low_coverage", False):
            detail = ""
            if coverage_return is not None and feature in coverage_return.index:
                info = coverage_return.loc[feature]
                detail = (
                    f"；有值/缺失标签中位差 {info['diff_mean']:+.4f}"
                    f"（正差占比 {info['diff_pos_ratio']:.0%}），规模分位差 {info['size_diff_mean']:+.3f}"
                )
            root_cause = ""
            if year_summary is not None and feature in year_summary.index:
                info = year_summary.loc[feature]
                first_year = info["first_effective_year"]
                first_label = int(first_year) if pd.notna(first_year) else "-"
                root_cause = (
                    f"；逐年覆盖 {info['coverage_first']:.2f}({first_label}) → "
                    f"{info['coverage_last']:.2f}；根因判定：{info['根因']}"
                )
            add(
                "A 覆盖缺口",
                feature,
                f"覆盖率 {coverage:.3f}{detail}{root_cause}",
                "结构性（+2~5pp 量级）",
                "可检出",
                "先判根因（源头起点/资格名单/口径切换），勿默认补数据；资格类考虑加可用性标记",
            )

        # F 口径退化：截面幅度塌缩（如转融券暂停后的 rqye 比率）；市场级常数不参与
        if (
            year_summary is not None
            and feature in year_summary.index
            and not bool(row.get("market_level", False))
        ):
            info = year_summary.loc[feature]
            ratio = info["collapse_ratio"]
            if pd.notna(ratio) and ratio < thresholds.collapse_ratio:
                add(
                    "F 口径退化",
                    feature,
                    f"近年截面 std 中位 {info['recent_std_median']:.4g} / 历史峰值 "
                    f"{info['hist_std_max']:.4g} = {ratio:.3f}×",
                    "族级（±2~5pp）",
                    "可检出",
                    "按 regime 处理或登记裁剪；勿当作稳定因子依赖",
                )

        # B 无增量信息：控制簇代表后仍无残余 IC（真正的冗余）
        if partial_ic is not None and feature in partial_ic.index:
            info = partial_ic.loc[feature]
            partial_t = info["partial_ic_t"]
            if pd.notna(partial_t) and abs(partial_t) < thresholds.weak_partial_t:
                add(
                    "B 无增量信息",
                    feature,
                    f"控制 {info['control']} 后偏 IC t={partial_t:.2f}"
                    f"（日数 {int(info['partial_days'])}）",
                    "列级（0~1pp）",
                    "不可检出",
                    "仅作冗余登记；要删需换“只删一侧”语义后重测",
                )

        # C 有信号但未被使用
        if pd.notna(ic_t) and abs(ic_t) >= thresholds.strong_ic_t:
            use = row.get("split_use_frac", np.nan)
            if pd.notna(use) and use < thresholds.unused_split_use:
                add(
                    "C 有信号未被使用",
                    feature,
                    f"|t|={abs(ic_t):.1f} 但分裂使用率仅 {use:.2f}",
                    "族级（±2~5pp）",
                    "可检出",
                    "查可交易性/执行约束，或做该族开关消融",
                )

        # D 使用不稳定
        if usage_stability is not None and feature in usage_stability.index:
            info = usage_stability.loc[feature]
            cv = info["gain_cv"]
            if pd.notna(cv) and cv >= thresholds.unstable_cv and info["gain_mean"] > 0:
                add(
                    "D 使用不稳定",
                    feature,
                    f"gain 份额 均值 {info['gain_mean']:.5f} / CV {cv:.2f}"
                    f"（版本 {int(info['versions'])} 个）",
                    "列级（0~1pp）",
                    "不可检出",
                    "训练不稳定登记；换种子复跑确认",
                )

        # E 缺失携带信息（不是规模代理时）
        if coverage_return is not None and feature in coverage_return.index:
            info = coverage_return.loc[feature]
            diff = info["diff_mean"]
            size_diff = info["size_diff_mean"]
            if (
                pd.notna(diff)
                and abs(diff) >= thresholds.missing_signal_diff
                and (pd.isna(size_diff) or abs(size_diff) <= thresholds.missing_size_diff)
            ):
                add(
                    "E 缺失携带信息",
                    feature,
                    f"有值/缺失标签中位差 {diff:+.4f}，规模分位差 {size_diff:+.3f}",
                    "族级（±2~3pp）",
                    "可检出",
                    "补 missing 标记因子或修正缺失语义",
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "候选类型",
                "因子",
                "证据",
                "预期效应量级",
                "可检出性(14折/MDE≈5pp)",
                "建议动作",
            ]
        )
    table = pd.DataFrame(rows)
    order = [
        "A 覆盖缺口",
        "B 无增量信息",
        "C 有信号未被使用",
        "D 使用不稳定",
        "E 缺失携带信息",
        "F 口径退化",
    ]
    table["__order"] = table["候选类型"].map({name: index for index, name in enumerate(order)})
    return table.sort_values(["__order", "因子"]).drop(columns="__order").reset_index(drop=True)
