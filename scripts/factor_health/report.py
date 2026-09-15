# -*- coding: utf-8 -*-
"""因子体检报告生成与产物落盘。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.factor_health.constants import HealthThresholds

_MAX_ROWS = 40


def _markdown_table(table: pd.DataFrame, max_rows: int = _MAX_ROWS) -> str:
    """渲染 DataFrame 为 markdown 表（为空时返回占位文本）。"""
    if table is None or table.empty:
        return "（无）"
    return table.head(max_rows).round(4).to_markdown()


def summarize(register: pd.DataFrame, clusters: pd.DataFrame) -> Dict[str, Any]:
    """汇总关键计数，供日志与报告头使用。"""
    return {
        "feature_count": int(len(register)),
        "market_level": int(register.get("market_level", pd.Series(dtype=bool)).sum()),
        "low_coverage": int(register.get("flag_low_coverage", pd.Series(dtype=bool)).sum()),
        "sign_flip": int(register.get("flag_flip", pd.Series(dtype=bool)).sum()),
        "weak": int(register.get("flag_weak", pd.Series(dtype=bool)).sum()),
        "unused": int(register.get("flag_unused", pd.Series(dtype=bool)).sum()),
        "dedup": int(register.get("flag_dup", pd.Series(dtype=bool)).sum()),
        "multi_member_clusters": int((clusters["size"] > 1).sum()) if not clusters.empty else 0,
        "twin_pairs": int(
            register.get("twin_base", pd.Series(dtype=str)).astype(str).str.len().gt(0).sum()
        )
        // 2,
    }


def build_exclude_lists(
    register: pd.DataFrame, clusters: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, Any]]:
    """构建实验排除清单（弱信息+未用、去重、并集，以及可选的口径交换）。

    Args:
        register: 因子台账。
        clusters: 聚类表；提供时会额外产出「口径交换」清单（同一簇划分下改为
            优先保留非 ``_sz`` 口径），用于把去重效应与 size 暴露变化分开归因。
    """
    weak = register[register["flag_weak"]].index.tolist()
    unused = register[register["flag_unused"]].index.tolist()
    dedup = register[register["flag_dup"]].index.tolist()
    weak_unused = sorted(set(weak) | set(unused))
    payload: Dict[str, Dict[str, Any]] = {
        "exclude_weak_v1": {
            "description": "因子体检：弱信息(|t|<阈值 且 gain<阈值)+几乎未使用（使用率/gain 低于阈值）",
            "factors": weak_unused,
        },
        "exclude_dedup_v1": {
            "description": "因子体检：同簇去重非代表成员（保留簇内 |ic_ir| 最高的代表）",
            "factors": sorted(dedup),
        },
        "exclude_weak_dedup_v1": {
            "description": "因子体检：弱信息+未用 与 同簇去重 的并集",
            "factors": sorted(set(weak_unused) | set(dedup)),
        },
    }
    if clusters is not None and not clusters.empty:
        payload["exclude_dedup_plain_v1"] = {
            "description": (
                "因子体检：口径交换去重（同一簇划分，代表改为优先保留非 _sz 口径）——"
                "用于把「去重效应」与「size（市值中性化）暴露变化」分开归因"
            ),
            "factors": plain_preference_dedup(register, clusters),
        }
    return payload


def plain_preference_dedup(register: pd.DataFrame, clusters: pd.DataFrame) -> List[str]:
    """口径交换清单：同一簇划分下，代表改为优先保留非 ``_sz`` 口径后应剔除的成员。"""
    from scripts.factor_health.analysis import prefer_plain_representatives

    swapped = prefer_plain_representatives(clusters, register)
    market_level = (
        register["market_level"]
        if "market_level" in register.columns
        else pd.Series(False, index=register.index)
    )
    dropped: List[str] = []
    for _, row in swapped.iterrows():
        members = [name for name in str(row["members"]).split("|") if name]
        if len(members) <= 1:
            continue
        representative = str(row["representative"])
        for name in members:
            if name == representative or name not in register.index:
                continue
            if bool(market_level.get(name, False)):
                continue
            dropped.append(name)
    return sorted(set(dropped))


def build_report_markdown(
    register: pd.DataFrame,
    clusters: pd.DataFrame,
    thresholds: HealthThresholds,
    context: Dict[str, Any],
    candidates: Dict[str, pd.DataFrame],
) -> str:
    """生成体检 markdown 报告。"""
    summary = summarize(register, clusters)
    cross_section = register[~register["market_level"]]
    family_table = cross_section.groupby("family2").agg(
        因子数=("ic_mean", "size"),
        IC均值=("ic_mean", "mean"),
        absIC中位=("ic_mean", lambda s: s.abs().median()),
        gain份额合计=("gain_share_mean", "sum"),
        使用率=("split_use_frac", "mean"),
        覆盖=("coverage", "mean"),
    )
    family_table = family_table.sort_values("因子数", ascending=False)

    lines: List[str] = []
    lines.append("# 选股因子体检报告")
    lines.append("")
    lines.append(
        f"> 特征清单：`{context.get('feature_file', 'N/A')}`（{summary['feature_count']} 个特征）；"
        f"区间 {context.get('start')} ~ {context.get('end')}，采样间隔 {context.get('every')} 个交易日"
        f"（{context.get('file_count')} 个分区）；标签 `{context.get('label')}`"
    )
    lines.append(
        f"> 股票域：{context.get('code_count', 'N/A')} 只"
        f"（{'主板过滤' if context.get('mainboard_filter') else '不做过滤'}）；"
        f"模型使用度：版本 {context.get('model_versions', [])}"
        f"（子模型 {context.get('sub_model_count', 0)} 个）"
    )
    lines.append(f"> 产物目录：`{context.get('out_dir', 'N/A')}`")
    lines.append("")
    lines.append("## 一、总体结论")
    lines.append("")
    lines.append(
        f"1. **市场级截面常数 {summary['market_level']} 个**：截面 RankIC 无定义，只能按 regime/择时评估，"
        "不能进截面 IC 筛选。"
    )
    lines.append(
        f"2. **孪生结构 {summary['twin_pairs']} 对 `X/X_sz`**：成对出现的同因子不同中性化口径，"
        "去重实验应整簇对照。"
    )
    lines.append(
        f"3. **低覆盖候选 {summary['low_coverage']} 个**"
        f"（全期 <{thresholds.low_coverage} 或单年 <{thresholds.low_year_coverage}）。"
    )
    lines.append(
        f"4. **跨期翻号候选 {summary['sign_flip']} 个**（年 |IC|≥{thresholds.flip_min_abs_year_ic} 才计符号；"
        f"异号年 ≥{thresholds.flip_years_opposite} 或相邻翻转 ≥{thresholds.flip_year_count}）。"
    )
    lines.append(
        f"5. **弱信息 {summary['weak']} 个**"
        f"（|t|<{thresholds.weak_abs_t} 且 gain<{thresholds.weak_gain_share}）；"
        f"**几乎未用 {summary['unused']} 个**"
        f"（使用率<{thresholds.unused_split_use} 或 gain<{thresholds.unused_gain_share}）。"
    )
    lines.append(
        f"6. **同簇冗余 {summary['dedup']} 个非代表成员**"
        f"（{summary['multi_member_clusters']} 个多成员簇，|rho|≥{thresholds.cluster_abs_corr}）。"
    )
    lines.append("")
    lines.append("## 二、家族画像（截面 IC 与使用度，剔除市场级常数）")
    lines.append("")
    lines.append(_markdown_table(family_table))
    lines.append("")
    lines.append("## 三、候选清单（供 WF 消融实验，非结论）")
    lines.append("")
    sections = [
        (
            "L1 低覆盖",
            "low_coverage",
            ["coverage", "coverage_min_year", "ic_mean", "ic_t", "split_use_frac"],
        ),
        (
            "L2 跨期翻号",
            "sign_flip",
            ["ic_mean", "ic_t", "years_opposite", "years_flips", "year_ic_str"],
        ),
        ("L3 弱信息", "weak", ["ic_mean", "ic_t", "gain_share_mean", "split_use_frac"]),
        ("L4 几乎未被使用", "unused", ["ic_mean", "ic_t", "gain_share_mean", "split_use_frac"]),
        ("L5 去重候选（同簇非代表）", "dedup", ["ic_t", "cluster_id", "cluster_rep"]),
    ]
    for title, key, columns in sections:
        table = candidates.get(key)
        lines.append(f"### {title}")
        lines.append("")
        if table is None or table.empty:
            lines.append("（无）")
        else:
            available = [c for c in columns if c in table.columns]
            lines.append(_markdown_table(table[available]))
        lines.append("")
    lines.append("## 四、方法论与注意事项")
    lines.append("")
    lines.append(
        "- IC 为按固定步长采样的逐日 RankIC 均值口径，用于估计均值/稳定性方向，**不构成采纳依据**；"
        "采纳必须走 WF A/B（预登记判据 + 噪声带）。"
    )
    lines.append(
        "- **负 IC ≠ 坏因子**（模型自行学习方向）；本报告只标记「不稳定」「低信息」「未被使用」。"
    )
    lines.append("- 市场级截面常数（如北向/市场环境）不能按截面 IC 判定，消融实验中应单独成组。")
    lines.append(
        "- 孪生对（`X/X_sz`）代表不同中性化口径，去重实验建议整簇对照（留 X / 留 X_sz / 都不留）。"
    )
    lines.append(
        "- 建议实验：弱信息+未用 并集与同簇去重**分开做两轮**温和裁剪（`--factor-prune --factor-exclude-file`），"
        "单变量对照；本目录已生成可直接使用的排除清单 JSON。"
    )
    lines.append(
        "- **口径交换清单**（`exclude_dedup_plain_v1.json`）：同一簇划分下改为优先保留非 `_sz`（未做市值中性化）口径，"
        "用于把「去重效应」与「size 暴露变化」分开归因；孪生对里多数 `_sz` 口径 |IC-IR| 更高，"
        "直接用默认去重清单会同时削减 size 暴露。"
    )
    lines.append("")
    return "\n".join(lines)


def write_health_outputs(
    out_dir: Path,
    register: pd.DataFrame,
    daily_ic: pd.DataFrame,
    corr_avg: Optional[pd.DataFrame],
    clusters: pd.DataFrame,
    candidates: Dict[str, pd.DataFrame],
    report_markdown: str,
    exclude_lists: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """落盘体检全部产物。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    register.to_csv(out_dir / "factor_register.csv", encoding="utf-8-sig")
    daily_ic.to_csv(out_dir / "daily_ic.csv.gz", index=False, compression="gzip")
    if corr_avg is not None:
        corr_avg.to_csv(out_dir / "corr_matrix.csv", encoding="utf-8-sig")
    if clusters is not None and not clusters.empty:
        clusters.to_csv(out_dir / "clusters.csv", encoding="utf-8-sig", index=False)
    for name, table in candidates.items():
        table.to_csv(out_dir / f"candidates_{name}.csv", encoding="utf-8-sig")
    (out_dir / "factor_health_report.md").write_text(report_markdown, encoding="utf-8")

    if exclude_lists:
        for name, payload in exclude_lists.items():
            data = {
                "description": payload["description"],
                "exclude_count": len(payload["factors"]),
                "exclude_factors": payload["factors"],
            }
            (out_dir / f"{name}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
