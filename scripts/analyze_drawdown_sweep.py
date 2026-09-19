"""回撤侧总扫描分析入口（terminal_loss 政策层 P2-4 前置条件）

薄入口：CLI + 编排 `scripts/compare/drawdown_sweep.py`（口径与实现见模块 docstring）。
预登记：`docs/plans/drawdown_side_sweep_prereg.md`。

用法示例：

    python scripts/analyze_drawdown_sweep.py \
        --arms "A0=wf_batch_20260914_134741,A1=wf_batch_20260914_142415,B1=wf_batch_20260919_xxx" \
        --baseline A0 \
        --tables "A1=temp/exposure_e2_rolling250.csv,B1=temp/exposure_lam03.csv" \
        --window-start 20240102 --window-end 20251204 \
        --out-dir temp/ddsweep_20260919
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from compare.drawdown_sweep import (
    fold_pass_summary,
    format_metrics_table,
    monthly_attribution,
    summarize_arms,
)
from loguru import logger


def _parse_pairs(text: str) -> dict:
    """解析 `K=V,K=V` 形式的参数。"""
    pairs = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"参数需为 K=V 形式: {item}")
        key, value = item.split("=", 1)
        pairs[key.strip()] = value.strip()
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="回撤侧总扫描分析（暴露门控/回补/止损）")
    parser.add_argument(
        "--arms",
        required=True,
        help="臂清单 `K=<批次目录名>,K=<批次目录名>`（相对 --batches-root）",
    )
    parser.add_argument("--baseline", required=True, help="基线臂代号（如 A0）")
    parser.add_argument(
        "--tables",
        default="",
        help="可选：臂→暴露系数表路径 `K=<csv 路径>`（缺省臂按全 λ=1 归类）",
    )
    parser.add_argument("--batches-root", default="data/walk_forward/batches")
    parser.add_argument("--data-root", default="data", help="交易日历所在数据根")
    parser.add_argument("--window-start", required=True, help="窗口起点（YYYYMMDD）")
    parser.add_argument("--window-end", required=True, help="窗口终点（YYYYMMDD）")
    parser.add_argument("--out-dir", default="", help="产物目录（空 = 只打印）")
    args = parser.parse_args()

    arms = {key: Path(args.batches_root) / name for key, name in _parse_pairs(args.arms).items()}
    tables = {key: Path(value) for key, value in _parse_pairs(args.tables).items()}
    if args.baseline not in arms:
        raise ValueError(f"基线臂 {args.baseline} 不在 --arms 内")

    metrics, folds, attribution, trades = summarize_arms(
        arms=arms,
        baseline_key=args.baseline,
        data_root=Path(args.data_root),
        window_start=args.window_start,
        window_end=args.window_end,
        tables=tables,
    )
    passes = fold_pass_summary(folds, args.baseline)
    monthly = monthly_attribution(attribution)

    logger.info("=== 窗口指标（主判据：ΔMaxDD 优先）===\n" + format_metrics_table(metrics))
    logger.info("=== 逐折判定（M2：改善 ≥ 3/4）===\n" + passes.to_string(index=False))
    logger.info(
        "=== 逐折明细 ===\n" + folds.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    logger.info(
        "=== 按月归因（pp）===\n"
        + monthly.to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "逐臂指标.csv", index=False, encoding="utf-8-sig")
        folds.to_csv(out / "逐折指标.csv", index=False, encoding="utf-8-sig")
        passes.to_csv(out / "逐折判定.csv", index=False, encoding="utf-8-sig")
        attribution.to_csv(out / "日差归因_逐日.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(out / "成交明细_窗口.csv", index=False, encoding="utf-8-sig")
        attribution.groupby(["臂", "组别"])["日差"].agg(
            日数="count", 日差合计_pp=lambda s: s.sum() * 100.0
        ).reset_index().to_csv(out / "日差归因_分组.csv", index=False, encoding="utf-8-sig")
        monthly.to_csv(out / "日差归因_按月.csv", index=False, encoding="utf-8-sig")
        logger.info(f"产物已写入: {out}")

    pd.set_option("display.width", 220)


if __name__ == "__main__":
    main()
