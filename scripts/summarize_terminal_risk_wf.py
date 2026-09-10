# -*- coding: utf-8 -*-
r"""期末异常亏损风险模型滚动 WF 汇总工具

扫描 batch_terminal_risk_wf.ps1 产出的各折目录（terminal_loss_report.json），
拼接 summary CSV 并打印跨折门禁结论。

核心指标（研究与选型层，详见方案 8.1）：
- lift = ES PR-AUC / ES 事件率：相对 null 基线的排序信息量（风险模型版的 RankIC）
- pred_bias = ES mean_pred - ES 事件率：校准漂移（C 段校准前的已知偏差方向）
- 跨折门禁：lift 最小值 >= 阈值（默认 1.1）且跨折稳定，才建议进入第二阶段

用法：
py .\scripts\summarize_terminal_risk_wf.py --wf-root data\walk_forward\terminal_risk_wf
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SUMMARY_COLUMNS = [
    "fold",
    "train_start",
    "train_end",
    "es_start",
    "es_end",
    "n_train",
    "n_es",
    "train_event_rate",
    "es_event_rate",
    "best_iteration",
    "es_logloss",
    "es_brier",
    "es_pr_auc",
    "lift",
    "pred_bias",
]


def collect_fold_rows(wf_root: Path) -> pd.DataFrame:
    """读取各折目录的 report 与模型元数据，拼 summary 行。"""
    rows = []
    for fold_dir in sorted(p for p in wf_root.iterdir() if p.is_dir()):
        report_path = fold_dir / "terminal_loss_report.json"
        meta_path = fold_dir / "terminal_loss_model.json"
        if not report_path.exists():
            logger.warning(f"跳过 {fold_dir.name}: 缺 terminal_loss_report.json")
            continue
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        es = report["es"]
        stage_dates = {}
        best_iteration = None
        n_train = es_n = None
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            stage_dates = meta.get("metadata", {}).get("stage_dates", {})
            best_iteration = meta.get("metadata", {}).get("best_iteration")
            n_train = meta.get("metadata", {}).get("n_train")
            es_n = meta.get("metadata", {}).get("n_es")
        event_rate = es["event_rate"]
        rows.append(
            {
                "fold": fold_dir.name,
                "train_start": stage_dates.get("train", [None, None])[0],
                "train_end": stage_dates.get("train", [None, None])[1],
                "es_start": stage_dates.get("es", [None, None])[0],
                "es_end": stage_dates.get("es", [None, None])[1],
                "n_train": n_train,
                "n_es": es_n if es_n is not None else es.get("n"),
                "train_event_rate": report.get("train", {}).get("event_rate"),
                "es_event_rate": event_rate,
                "best_iteration": best_iteration,
                "es_logloss": es["logloss"],
                "es_brier": es["brier"],
                "es_pr_auc": es["pr_auc"],
                "lift": es["pr_auc"] / event_rate if event_rate > 0 else None,
                "pred_bias": es.get("mean_pred", float("nan")) - event_rate,
            }
        )
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description="terminal_loss 滚动 WF 汇总")
    parser.add_argument("--wf-root", default="data/walk_forward/terminal_risk_wf",
                        help="各折输出目录的父目录")
    parser.add_argument("--lift-min-threshold", type=float, default=1.1,
                        help="跨折 lift 最小值门禁（默认 1.1）")
    args = parser.parse_args()

    wf_root = Path(args.wf_root)
    if not wf_root.exists():
        logger.error(f"目录不存在: {wf_root}")
        return 1
    summary = collect_fold_rows(wf_root)
    if summary.empty:
        logger.error("未找到任何折的 report")
        return 1

    out_csv = wf_root / "summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"summary 已写入 {out_csv}（{len(summary)} 折）")

    lifts = summary["lift"].dropna()
    if len(lifts):
        gate_pass = float(lifts.min()) >= args.lift_min_threshold
        if len(lifts) > 1:
            lift_std = float(lifts.std())
            logger.info(
                f"跨折 lift: mean={lifts.mean():.3f}, min={lifts.min():.3f}, "
                f"max={lifts.max():.3f}, std={lift_std:.3f}"
            )
        else:
            logger.info(f"跨折 lift: 仅 {len(lifts)} 折，无法评估稳定性")
        logger.info(
            f"门禁（lift_min >= {args.lift_min_threshold}）: "
            f"{'通过 → 建议进入第二阶段（C 段校准 + V 段阈值）' if gate_pass else '未通过 → 建议先做超参/特征消融再重跑'}"
        )
    bias = summary["pred_bias"].dropna()
    if len(bias):
        logger.info(
            f"pred_bias 跨折: mean={bias.mean():+.4f}（正值=概率高估，"
            f"C 段校准的输入信号）"
        )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
