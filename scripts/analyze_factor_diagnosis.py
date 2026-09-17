#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""因子诊断 v2 入口（薄入口）：增量信息 / 覆盖显著性 / 使用度稳定性。

在体检（`scripts/analyze_factor_health.py`）产物基础上回答“**哪条候选值不值得改数据**”：
    - 偏 IC：控制簇代表后的残余截面信息（真正的“冗余”定义）；
    - 覆盖-标签差：有值 / 缺失子样本的标签与规模差（缺失是否携带信息、是否只是规模代理）；
    - 使用度稳定性：跨模型版本的 gain 份额离散度（稳定出力 vs 偶发出力）；
    - 汇总为《数据改良候选清单》，每行标注“预期效应量级”与“在 14 折设计下能否检出（MDE≈5pp）”。
使用示例：
    python scripts/analyze_factor_diagnosis.py
    python scripts/analyze_factor_diagnosis.py --health-dir <体检目录>
    python scripts/analyze_factor_diagnosis.py --start 20200101 --every 5 --usage-model-count 40

本文件为薄入口：CLI 解析 + 编排 scripts/factor_health/diagnose.py。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from scripts.factor_health import DEFAULT_LABEL_COLUMN, pick_partition_files
from scripts.factor_health.diagnose import (
    DiagnoseThresholds,
    build_candidate_list,
    column_year_profile,
    coverage_return_table,
    partial_ic_table,
    summarize_year_profile,
    usage_stability_table,
)
from src.lazybull.common.config import (
    get_data_root,
    get_logs_dir,
    get_stock_selection_models_root,
)
from src.lazybull.common.logger import setup_logger

DEFAULT_SIZE_COLUMN = "zscore_size"


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="因子诊断 v2（增量 IC / 覆盖显著性 / 使用度稳定性）"
    )
    parser.add_argument(
        "--health-dir", default="", help="体检产物目录，默认取 data/reports/factor_health 最新"
    )
    parser.add_argument("--start", default="20200101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="99999999", help="结束日期 YYYYMMDD")
    parser.add_argument("--every", type=int, default=3, help="采样间隔（交易日）")
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN, help="标签列")
    parser.add_argument(
        "--size-column", default=DEFAULT_SIZE_COLUMN, help="规模代理列（缺失则跳过规模差）"
    )
    parser.add_argument(
        "--usage-model-count", type=int, default=40, help="统计使用度稳定性的最近模型版本数"
    )
    parser.add_argument(
        "--out", default="", help="输出目录，默认 data/reports/factor_diagnosis/<时间戳>"
    )
    parser.add_argument("--min-pairs", type=int, default=200, help="单日截面最少配对样本")
    parser.add_argument("--min-group-rows", type=int, default=30, help="有值/缺失任一组最少行数")
    parser.add_argument("--coverage-max", type=float, default=0.95, help="参与覆盖诊断的覆盖率上限")
    parser.add_argument(
        "--weak-partial-t", type=float, default=1.5, help="|偏 IC t| 低于该值视为无增量信息"
    )
    parser.add_argument("--strong-ic-t", type=float, default=5.0, help="|IC t| 高于该值视为有信号")
    parser.add_argument(
        "--unused-split-use", type=float, default=0.6, help="使用率低于该值视为未被使用"
    )
    parser.add_argument(
        "--unstable-cv", type=float, default=1.0, help="gain 份额变异系数高于该值视为不稳定"
    )
    parser.add_argument(
        "--missing-signal-diff", type=float, default=0.005, help="缺失携带信息的最小标签中位差"
    )
    parser.add_argument(
        "--missing-size-diff", type=float, default=0.3, help="规模代理判定的最大规模分位差"
    )
    parser.add_argument("--progress-every", type=int, default=0, help="扫描进度日志间隔（文件数）")
    parser.add_argument(
        "--year-profile",
        action="store_true",
        help="额外产出逐年覆盖/幅度剖面与根因判定（默认开启，可用 --no-year-profile 关闭）",
    )
    parser.add_argument("--no-year-profile", dest="year_profile", action="store_false")
    parser.set_defaults(year_profile=True)
    parser.add_argument(
        "--collapse-ratio", type=float, default=0.1, help="口径退化：近年 std/早期 std 阈值"
    )
    parser.add_argument(
        "--profile-start",
        default="20120101",
        help="逐年剖面的起始日期（默认全历史，用于识别源头起点）",
    )
    return parser


def _pick_yearly_partitions(cs_train_dir: Path, start: str, end: str) -> list:
    """每年挑一个代表分区（优先取该年 6 月之后的首个分区）。"""
    files = sorted(path for path in cs_train_dir.glob("*.parquet") if start <= path.stem <= end)
    picks = {}
    for path in files:
        year = path.stem[:4]
        if year not in picks and path.stem[4:6] >= "06":
            picks[year] = path
    return sorted(picks.items())


def _resolve_health_dir(raw: str, data_root: Path) -> Path:
    """解析体检产物目录（默认取最新）。"""
    if raw:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"体检产物目录不存在: {path}")
        return path
    root = data_root / "reports" / "factor_health"
    candidates = sorted([p for p in root.glob("*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"{root} 下没有体检产物，请先运行 scripts/analyze_factor_health.py")
    return candidates[-1]


def main() -> None:
    """编排诊断流程。"""
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(
        log_level="INFO",
        log_file=str(Path(get_logs_dir()) / f"factor_diagnosis_{timestamp}.log"),
    )

    data_root = Path(get_data_root())
    health_dir = _resolve_health_dir(args.health_dir, data_root)
    register = pd.read_csv(health_dir / "factor_register.csv", index_col="feature")
    logger.info(f"体检台账: {health_dir}（{len(register)} 列）")

    cs_train_dir = data_root / "features" / "cs_train"
    files = pick_partition_files(cs_train_dir, args.start, args.end, args.every)
    logger.info(f"采样分区 {len(files)} 个（{files[0].stem} ~ {files[-1].stem}）")

    thresholds = DiagnoseThresholds(
        min_pairs=args.min_pairs,
        coverage_max=args.coverage_max,
        min_group_rows=args.min_group_rows,
        weak_partial_t=args.weak_partial_t,
        strong_ic_t=args.strong_ic_t,
        unused_split_use=args.unused_split_use,
        unstable_cv=args.unstable_cv,
        missing_signal_diff=args.missing_signal_diff,
        missing_size_diff=args.missing_size_diff,
        collapse_ratio=args.collapse_ratio,
    )

    # ── 逐年剖面：覆盖率与截面幅度 → 根因判定（源头起点/资格扩张/覆盖收缩/口径退化）
    year_profile = pd.DataFrame()
    year_summary = pd.DataFrame()
    if args.year_profile:
        picks = _pick_yearly_partitions(cs_train_dir, args.profile_start, args.end)
        if picks:
            logger.info(f"逐年剖面：{len(picks)} 个年份代表分区（{picks[0][0]} ~ {picks[-1][0]}）")
            year_profile = column_year_profile(picks, register.index.tolist())
            year_summary = summarize_year_profile(year_profile, collapse_ratio=args.collapse_ratio)
            abnormal = year_summary[year_summary["根因"] != "正常"]
            logger.info(f"逐年剖面：异常根因 {len(abnormal)} 列")

    # ── 偏 IC：仅对“存在簇代表”的列计算（代表 = 同簇 |ic_ir| 最大者）
    pairs: dict = {}
    if "cluster_rep" in register.columns:
        for feature, row in register.iterrows():
            rep = row.get("cluster_rep")
            if pd.isna(rep) or str(rep) == str(feature):
                continue
            if bool(row.get("market_level", False)):
                continue
            pairs[str(feature)] = str(rep)
    logger.info(f"偏 IC 目标列 {len(pairs)} 个（控制簇代表）")
    partial = partial_ic_table(
        files,
        pairs,
        args.label_column,
        min_pairs=args.min_pairs,
        progress_every=args.progress_every,
    )

    # ── 覆盖-标签诊断：仅对覆盖率低于阈值的列计算
    low_cov = register[register["coverage"] < thresholds.coverage_max].index.tolist()
    logger.info(f"覆盖-标签诊断目标列 {len(low_cov)} 个（覆盖率 < {thresholds.coverage_max}）")
    coverage = coverage_return_table(
        files,
        low_cov,
        args.label_column,
        size_column=args.size_column,
        min_group_rows=args.min_group_rows,
        progress_every=args.progress_every,
    )

    # ── 使用度稳定性
    model_dir = Path(get_stock_selection_models_root(str(data_root)))
    from scripts.factor_health.analysis import list_available_model_versions

    available = list_available_model_versions(model_dir)
    versions = available[-max(args.usage_model_count, 1) :]
    stability = usage_stability_table(model_dir, versions, register.index.tolist())

    candidates = build_candidate_list(
        register, partial, coverage, stability, thresholds, year_summary
    )
    logger.info(f"数据改良候选 {len(candidates)} 条")

    out_dir = Path(args.out) if args.out else data_root / "reports" / "factor_diagnosis" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    if not partial.empty:
        partial.to_csv(out_dir / "partial_ic.csv", encoding="utf-8-sig")
    if not coverage.empty:
        coverage.to_csv(out_dir / "coverage_return.csv", encoding="utf-8-sig")
    if not stability.empty:
        stability.to_csv(out_dir / "usage_stability.csv", encoding="utf-8-sig")
    if not year_profile.empty:
        year_profile.to_csv(out_dir / "column_year_profile.csv", index=False, encoding="utf-8-sig")
    if not year_summary.empty:
        year_summary.to_csv(out_dir / "column_year_summary.csv", encoding="utf-8-sig")
    candidates.to_csv(out_dir / "数据改良候选清单.csv", index=False, encoding="utf-8-sig")

    counts = candidates["候选类型"].value_counts().to_dict() if not candidates.empty else {}
    lines = [
        "# 因子诊断 v2（增量信息 / 覆盖显著性 / 使用度稳定性）",
        "",
        f"- 体检台账：`{health_dir}`（{len(register)} 列）",
        f"- 采样区间：{files[0].stem} ~ {files[-1].stem}"
        f"（间隔 {args.every}，{len(files)} 个分区）；标签 `{args.label_column}`",
        f"- 使用度稳定性：模型版本 {versions[0]}~{versions[-1]}（{len(versions)} 个）",
        f"- 候选统计：{counts}",
        "",
        "## 一、数据改良候选清单（按类型）",
        "",
        candidates.to_markdown(index=False) if not candidates.empty else "（无）",
        "",
        "## 二、覆盖-标签诊断 Top20（标签中位差绝对值）",
        "",
    ]
    if not coverage.empty:
        top = coverage.reindex(coverage["diff_mean"].abs().sort_values(ascending=False).index).head(
            20
        )
        lines.append(top.round(5).to_markdown())
    else:
        lines.append("（无）")
    lines += ["", "## 三、偏 IC 最弱 20 列（控制簇代表后）", ""]
    if not partial.empty:
        weak = partial.sort_values("partial_ic_t", key=lambda s: s.abs()).head(20)
        lines.append(weak.round(5).to_markdown())
    else:
        lines.append("（无）")
    lines += ["", "## 四、使用度最不稳定 20 列（gain 份额 CV）", ""]
    if not stability.empty:
        unstable = stability.sort_values("gain_cv", ascending=False).head(20)
        lines.append(unstable.round(6).to_markdown())
    else:
        lines.append("（无）")
    lines += ["", "## 五、逐年覆盖/幅度剖面：根因异常列", ""]
    if not year_summary.empty:
        abnormal = year_summary.reset_index()
        abnormal = abnormal[abnormal["根因"] != "正常"]
        actionable = abnormal[
            abnormal["根因"].str.contains("源头起点|口径退化|覆盖收缩", regex=True)
        ].reset_index(drop=True)
        lines.append(
            f"根因异常 {len(abnormal)} 列，其中需关注（源头起点/口径退化/覆盖收缩）**{len(actionable)} 列**；"
            "完整表见 `column_year_summary.csv`。"
        )
        lines.append("")
        lines.append(
            actionable.round(4).to_markdown(index=False) if not actionable.empty else "（无）"
        )
    else:
        lines.append("（未启用逐年剖面）")
    lines += [
        "",
        "## 六、方法论与可检出性",
        "",
        "- 偏 IC = 逐日 `rank(特征) ~ rank(簇代表)` 的秩残差与 `rank(标签)` 的相关（跨日均值 / t 值）；",
        "  |t| < 阈值 才可称“无增量信息”，仅凭高相关（|ρ|≥0.85）不足以判定冗余。",
        "- 覆盖-标签差 = 逐日“有值组标签中位 − 缺失组标签中位”的跨日均值；规模分位差用于排除“只是规模代理”的解释。",
        "- 使用度稳定性 = 跨模型版本（含集成子模型）gain 份额的均值/标准差/变异系数；CV 高说明该列偶发出力。",
        "- **逐年剖面** = 每年取一个代表分区，统计每列覆盖率与截面 std；据此判根因：",
        "  源头起点（如 cyq 2018+ / north 2014-11+）、资格扩张（如两融标的扩容）、覆盖收缩、口径退化（幅度塌缩）。",
        "  ⚠️ 覆盖缺口**不得默认“补数据”**：本地 raw 已含 2005+ 数据，缺口多数来自 TuShare 起点、标的资格或披露口径。",
        "- **可检出性**：以 2026-09-16 A-B/B0 实测（逐折差值 SD=3.5~8.1pp，n=14）估算，当前 14 折设计可检出下限约 **5pp**",
        "  （3pp 需约 28 折、2pp 需约 63 折）；列级候选（0~1pp）在此设计下**不可判定**，只能作登记项；",
        "  结构性候选（≥3pp）方可进入 WF A/B。",
    ]
    (out_dir / "因子诊断报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(f"产物目录: {out_dir}")
    for name, count in counts.items():
        logger.info(f"  {name}: {count} 条")


if __name__ == "__main__":
    main()
