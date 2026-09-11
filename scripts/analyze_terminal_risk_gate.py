# -*- coding: utf-8 -*-
"""terminal_loss WF 门禁区间重判（方案第 6 节规则 4/5、8.1）

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

用法：
py .\\scripts\\analyze_terminal_risk_gate.py --wf-root data\\walk_forward\\terminal_risk_wf
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
        help="只重判该超参签名（默认取 tuning_scores.csv 之首行，即当次调参分最高组）",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="跳过逐折分块区间（无 ES 逐行预测时可用）",
    )
    return parser.parse_args()


def pick_signature(wf_root: Path, explicit: Optional[str]) -> Optional[str]:
    """选定待重判的超参签名（默认当次调参分最高的组）。"""
    if explicit:
        return explicit
    scores_path = wf_root / "tuning_scores.csv"
    if not scores_path.exists():
        return None
    scores = pd.read_csv(scores_path)
    if scores.empty:
        return None
    return str(scores.iloc[0]["param_signature"])


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
    signature = pick_signature(wf_root, args.signature)
    subset = collect_fold_lifts(summary, signature)
    if subset["lift"].isna().all():
        logger.error("该组全部折 lift 缺失，无法重判")
        return 1

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
            f"逐折区间需重跑训练（去掉 --no-es-predictions）"
        )
    if block_rows:
        block_csv = wf_root / "gate_ci_block.csv"
        pd.DataFrame(block_rows)[BLOCK_COLUMNS].to_csv(block_csv, index=False, encoding="utf-8-sig")
        logger.info(f"逐折分块区间已写入 {block_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
