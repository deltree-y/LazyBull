# -*- coding: utf-8 -*-
r"""terminal_loss WF 门禁区间重判（方案第 6 节规则 4/5、8.1）

背景：``summarize_terminal_risk_wf.py`` 的门禁是"组内 lift 最小值 ≥ 阈值"
的点估计判定。当余量很薄时（如 lift_min=1.151 vs 阈值 1.1），"通过"很可能
只是超参选择的结果，而不是稳定的排序信息。本脚本用分块重采样给出区间口径：

1. **折级重判**（不需要重跑训练）：以 8 折为复制单位对 lift 自举，输出
   均值 / 最小值的区间与 ``P(最小 lift ≥ 阈值)``——直接回答"门禁是否
   只是选型伪影"。
2. **逐折区间**（需要 ES 逐行预测）：折目录有
   ``terminal_loss_es_predictions.parquet``（或 ``v{N}_es_predictions.parquet``）
   时，按连续交易日分块做 moving-block bootstrap，输出单折 lift 区间。
   块长默认取 ES 段口径（5/10/20 日）——方案第 6 节的 40 日块是组合级
   多年 OOS 口径，ES 段只有几十个交易日，用 40 日会使块数不足。

产物：``gate_ci.csv``（折级）与 ``gate_ci_block.csv``（逐折分块区间），
均写入 ``--wf-root``。

分组选择（关键：``--wf-root`` 下常有多个历史实验组，选错就会重判到别的组）：

- ``--select latest``（默认）：取**折目录写入时间最新**的组，即“刚跑完的那批”；
- ``--select best``：取 ``tuning_scores.csv`` 排名第一（调参分最高）的组；
- ``--signature`` ：显式指定超参签名，优先级最高；
- ``--list-groups``：列出全部分组（折目录、最新写入时间、调参分/排名）后退出。

用法：
py .\scripts\analyze_terminal_risk_gate.py --wf-root data\walk_forward\terminal_risk_wf
py .\scripts\analyze_terminal_risk_gate.py --wf-root data\walk_forward\terminal_risk_wf --list-groups
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lazybull.risk.terminal_loss import (  # noqa: E402
    BootstrapConfig,
    fold_level_gate,
    moving_block_metric_sensitivity,
)

#: 折级台账列
GATE_COLUMNS = [
    "param_signature",
    "n_folds",
    "point_min",
    "point_median",
    "point_mean",
    "point_max",
    "fold_std",
    "point_gate_pass",
    "fold_min_pass_share",
    "ci_mean_low",
    "ci_mean_high",
    "prob_mean_pass",
    "threshold",
    "n_resamples",
    "seed",
]

#: 逐折分块区间列
BLOCK_COLUMNS = [
    "fold",
    "metric",
    "block_days",
    "point",
    "ci_low",
    "ci_high",
    "std",
    "n_days",
    "n_resamples",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="terminal_loss WF 门禁区间重判")
    parser.add_argument(
        "--wf-root",
        default="data/walk_forward/terminal_risk_wf",
        help="WF 根目录（含 summary.csv / tuning_scores.csv / 各折目录）",
    )
    parser.add_argument("--threshold", type=float, default=1.1, help="lift 最小值门禁阈值")
    parser.add_argument("--n-resamples", type=int, default=2000, help="折级自举次数")
    parser.add_argument("--block-resamples", type=int, default=200, help="逐折分块自举次数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--signature",
        default=None,
        help="只重判该超参签名（优先级最高，默认见 --select）",
    )
    parser.add_argument(
        "--select",
        choices=("latest", "best"),
        default="latest",
        help="未指定 --signature 时的分组选择：latest=折目录最新（默认，刚跑完的那批）；"
        "best=tuning_scores.csv 排名第一",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="列出全部分组（折目录/最新写入时间/调参分）后退出，不做重判",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="跳过逐折分块区间（无 ES 逐行预测时可用）",
    )
    return parser.parse_args()


def _fold_mtime(fold_dir: Path) -> Optional[float]:
    """折目录（或产物文件）内文件的最新写入时间戳；不存在返回 None。"""
    if not fold_dir.exists():
        return None
    if not fold_dir.is_dir():
        return fold_dir.stat().st_mtime
    times = [p.stat().st_mtime for p in fold_dir.iterdir() if p.is_file()]
    return max(times) if times else fold_dir.stat().st_mtime


def load_group_table(wf_root: Path) -> pd.DataFrame:
    """汇总每个超参签名所属的折目录与最新写入时间（分组选择依据）。

    Args:
        wf_root: WF 根目录（含 summary.csv / tuning_scores.csv / 各折目录）

    Returns:
        DataFrame(rank/suffix/param_signature/n_folds/fold_dirs/latest_mtime/
        tuning_score/lift_min)，按 latest_mtime 升序
    """
    summary_path = wf_root / "summary.csv"
    summary = pd.read_csv(summary_path)
    if "param_signature" not in summary.columns:
        raise ValueError("summary.csv 缺少 param_signature 列，请先升级 summarize 工具")
    rows = []
    for signature, grp in summary.groupby("param_signature", sort=False):
        folds = [str(f) for f in grp["fold"].tolist()]
        times = [t for t in (_fold_mtime(wf_root / f) for f in folds) if t is not None]
        rows.append(
            {
                "param_signature": str(signature),
                "n_folds": len(folds),
                "fold_dirs": ",".join(folds),
                "latest_mtime": max(times) if times else None,
            }
        )
    table = pd.DataFrame(rows)
    scores_path = wf_root / "tuning_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        keep = [
            c
            for c in ("param_signature", "rank", "suffix", "tuning_score", "lift_min")
            if c in scores.columns
        ]
        if "param_signature" in keep and len(keep) > 1:
            table = table.merge(scores[keep], on="param_signature", how="left")
    return table.sort_values("latest_mtime", na_position="first").reset_index(drop=True)


def pick_signature(
    groups: pd.DataFrame,
    explicit: Optional[str],
    select: str = "latest",
) -> Optional[str]:
    """选定待重判的超参签名。

    Args:
        groups: ``load_group_table`` 的结果
        explicit: 显式指定的签名（优先级最高）
        select: latest=折目录最新（默认）；best=tuning_scores 排名第一

    Returns:
        超参签名字符串；无可用分组时返回 None
    """
    if explicit:
        return explicit
    if groups.empty:
        return None
    has_rank = "rank" in groups.columns and groups["rank"].notna().any()
    if select == "best" and has_rank:
        ranked = groups.sort_values("rank", na_position="last")
        return str(ranked.iloc[0]["param_signature"])
    dated = groups[groups["latest_mtime"].notna()]
    pool = dated if not dated.empty else groups
    return str(pool.iloc[-1]["param_signature"])


def format_groups(groups: pd.DataFrame) -> str:
    """分组表的人读文本（用于 --list-groups 与选定说明）。"""
    lines = []
    for _, row in groups.iterrows():
        mtime = row.get("latest_mtime")
        stamp = (
            pd.Timestamp(float(mtime), unit="s").strftime("%Y-%m-%d %H:%M")
            if pd.notna(mtime)
            else "无折目录"
        )
        rank = row.get("rank")
        rank_txt = f"rank={int(rank)}" if pd.notna(rank) else "rank=-"
        score = row.get("tuning_score")
        score_txt = f"调参分={float(score):.4f}" if pd.notna(score) else "调参分=-"
        suffix = row.get("suffix") if pd.notna(row.get("suffix")) else "-"
        lines.append(
            f"  [{rank_txt}] suffix={suffix} {score_txt} 折{int(row['n_folds'])} "
            f"最新写入={stamp}\n      目录: {row['fold_dirs']}\n      签名: {row['param_signature']}"
        )
    return "\n".join(lines)


def collect_fold_lifts(summary: pd.DataFrame, signature: Optional[str]) -> pd.DataFrame:
    """按签名筛出该组的折 lift（签名缺失时使用全部折并告警）。"""
    if "param_signature" not in summary.columns:
        raise ValueError("summary.csv 缺少 param_signature 列，请先升级 summarize 工具")
    if signature is None:
        logger.warning("未找到超参签名（tuning_scores.csv 缺失），对全部折统一重判")
        return summary
    subset = summary[summary["param_signature"] == signature]
    if subset.empty:
        raise ValueError(f"summary.csv 中不存在签名 {signature}，请检查 --signature")
    return subset


def find_es_predictions(wf_root: Path, fold: str) -> Optional[Path]:
    """定位折的 ES 逐行预测（固定名优先，其次最新版本化文件）。"""
    fold_dir = wf_root / fold
    if not fold_dir.is_dir():
        return None
    fixed = fold_dir / "terminal_loss_es_predictions.parquet"
    if fixed.exists():
        return fixed
    versioned = sorted(
        fold_dir.glob("v*_es_predictions.parquet"),
        key=lambda p: int(p.name.split("_")[0].lstrip("v")),
    )
    return versioned[-1] if versioned else None


def main() -> int:
    args = parse_args()
    wf_root = Path(args.wf_root)
    summary_path = wf_root / "summary.csv"
    if not summary_path.exists():
        logger.error(f"缺少 {summary_path}，请先运行 summarize_terminal_risk_wf.py")
        return 1
    summary = pd.read_csv(summary_path)
    groups = load_group_table(wf_root)

    if args.list_groups:
        print("")
        print(f"  ── terminal_loss WF 分组（{wf_root}，共 {len(groups)} 组）──")
        print(format_groups(groups))
        print("")
        print("  提示: 默认 --select latest 取折目录最新的一组；--select best 取排名第一；")
        print("        --signature <签名> 可显式指定。")
        return 0

    signature = pick_signature(groups, args.signature, args.select)
    subset = collect_fold_lifts(summary, signature)
    if subset["lift"].isna().all():
        logger.error("该组全部折 lift 缺失，无法重判")
        return 1

    # 分组来源回显：避免把历史实验组当成刚跑完的那批
    selected = groups[groups["param_signature"] == signature]
    best_signature = pick_signature(groups, None, "best")
    if not args.signature:
        source = "折目录最新" if args.select == "latest" else "tuning_scores 排名第一"
        logger.info(f"分组选择: {source}（--select {args.select}）→ {signature}")
    if not selected.empty:
        logger.info(f"该组折目录: {selected.iloc[0]['fold_dirs']}")
    if best_signature and best_signature != signature:
        logger.info(
            f"注意: 所得组不是 tuning_scores 排名第一的组（{best_signature}）；"
            f"如需重判该组请加 --select best 或 --signature <签名>"
        )

    gate = fold_level_gate(
        subset["lift"].tolist(),
        lift_min_threshold=args.threshold,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    row = {
        "param_signature": signature,
        "n_folds": gate["n_folds"],
        "point_min": gate["point_min"],
        "point_median": gate["point_median"],
        "point_mean": gate["point_mean"],
        "point_max": gate["point_max"],
        "fold_std": gate["fold_std"],
        "point_gate_pass": gate["point_gate_pass"],
        "fold_min_pass_share": gate["fold_min_pass_share"],
        "ci_mean_low": gate["ci_mean"][0],
        "ci_mean_high": gate["ci_mean"][1],
        "prob_mean_pass": gate["prob_mean_pass"],
        "threshold": gate["threshold"],
        "n_resamples": gate["n_resamples"],
        "seed": gate["seed"],
    }
    gate_csv = wf_root / "gate_ci.csv"
    pd.DataFrame([row])[GATE_COLUMNS].to_csv(gate_csv, index=False, encoding="utf-8-sig")
    logger.info(f"折级门禁区间已写入 {gate_csv}")

    print("")
    print("  ── terminal_loss 折级门禁重判（折为单位，8 折 = 8 个独立制度）──")
    print(f"  超参签名      : {signature}")
    if not selected.empty:
        row_sel = selected.iloc[0]
        mtime = row_sel.get("latest_mtime")
        stamp = (
            pd.Timestamp(float(mtime), unit="s").strftime("%Y-%m-%d %H:%M")
            if pd.notna(mtime)
            else "无折目录"
        )
        print(
            f"  分组来源      : {'--signature 显式指定' if args.signature else args.select}"
            f"（折目录: {row_sel['fold_dirs']}，最新写入 {stamp}）"
        )
        if best_signature and best_signature != signature:
            print(
                "  提示          : tuning_scores 排名第一为另一组；"
                "如需重判请用 --select best 或 --signature"
            )
    print(
        f"  折间分布      : min={gate['point_min']:.3f}  median={gate['point_median']:.3f}  "
        f"mean={gate['point_mean']:.3f}  max={gate['point_max']:.3f}  "
        f"std={gate['fold_std']:.3f}"
    )
    print(
        f"  原口径        : min ≥ {gate['threshold']} "
        f"{'通过' if gate['point_gate_pass'] else '不通过'}；"
        f"达标折 {gate['fold_min_pass_share'] * 100:.0f}%（{gate['n_folds']} 折）"
    )
    print(
        f"  均值 lift 区间: [{gate['ci_mean'][0]:.3f}, {gate['ci_mean'][1]:.3f}]"
        f"（{int(gate['ci_level'] * 100)}%，P(均值 ≥ 阈值) = {gate['prob_mean_pass']:.2f}）"
    )
    print(
        "  口径边界      : 折级自举只能反映「换一批制度」，重采样抽不到比观测最小值\n"
        "                  更差的折，因此它无法判定最差折是否真正高于阈值；\n"
        "                  该问题须看下面的逐折分块区间（需 ES 逐行预测）。"
    )
    print("")

    if args.no_block:
        return 0

    block_rows: List[dict] = []
    missing: List[str] = []
    for fold in subset["fold"].tolist():
        path = find_es_predictions(wf_root, fold)
        if path is None:
            missing.append(fold)
            continue
        frame = pd.read_parquet(path)
        cfg = BootstrapConfig(n_resamples=args.block_resamples, seed=args.seed)
        try:
            results = moving_block_metric_sensitivity(frame, metric="lift", config=cfg)
        except ValueError as exc:
            logger.warning(f"{fold}: 分块区间不可行（{exc}）")
            missing.append(fold)
            continue
        for result in results:
            block_rows.append({"fold": fold, **result})
        logger.info(
            f"{fold}: 日块 {results[0]['n_days']} 天，主块长 {
                results[min(1, len(results) - 1)]['block_days']
            } 日 lift = {results[0]['point']:.3f} "
            f"[{results[min(1, len(results) - 1)]['ci_low']:.3f}, "
            f"{results[min(1, len(results) - 1)]['ci_high']:.3f}]"
        )
    if missing:
        logger.warning(
            f"{len(missing)} 个折缺 ES 逐行预测（{missing[:3]}...）："
            f"逐折区间需重跑训练（去掉 --no-es-predictions）；"
            f"若这些折不属于刚跑的那批，请用 --list-groups 核对分组后用 --select/--signature 指定"
        )
    if block_rows:
        block_csv = wf_root / "gate_ci_block.csv"
        pd.DataFrame(block_rows)[BLOCK_COLUMNS].to_csv(block_csv, index=False, encoding="utf-8-sig")
        logger.info(f"逐折分块区间已写入 {block_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
