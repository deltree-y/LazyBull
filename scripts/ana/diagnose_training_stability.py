#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练稳定性诊断脚本

目的：在与 batch_walk_forward.ps1 完全一致的训练配置下，对【最近期 split】
用多个随机种子 × 两种早停指标重复训练同一组数据，量化训练的随机不稳定性，
并评估“多种子预测平均(bagging)”能降低多少方差。

回答三个问题：
- A1 best_iteration 方差：单一 holdout 早停导致的“树数量(模型复杂度)”在不同种子间的离散程度。
- A2 全池 RankIC vs 按日截面 RankIC：早停所用的全池 spearmanr 与真实选股口径(按日截面)的差距。
- #1 bagging 收益：10 个种子预测平均后，按日截面 RankIC 的种子间方差下降幅度。

设计要点：
- 数据只准备一次，所有种子复用，确保唯一变量是 random_state(与 early_stopping_metric)。
- 评估按日截面 RankIC 时直接用“保留 NaN”的验证集自行 predict，绕开
  evaluate_validation_daily 的 fillna(0)，避免口径污染(对应待修 bugfix B1)。
- 不修改任何训练主链路；本脚本为只读分析工具。

用法示例：
    # 全量(10 种子 × auto/rank_ic, 对齐 batch 0101 组最近期 split)
    python scripts/ana/diagnose_training_stability.py --split-count 13 --final-date 20251231

    # 快速冒烟(少种子 + 小树数, 仅验证流程跑通)
    python scripts/ana/diagnose_training_stability.py --split-count 13 --final-date 20251231 \
        --seeds 42,1 --metrics auto --n-estimators 100
"""

import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 注入项目根目录到 sys.path（scripts/ana/ 的上三级）
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_data_root, get_reports_root
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml.eval_utils import (
    evaluate_predictions_by_date,
    summarize_daily_metrics,
)
from src.lazybull.ml.train_core import (
    build_rank_sample_weights,
    load_features_data,
    prepare_training_data,
    train_xgboost_model,
    transform_labels_cs_zscore,
)
from src.lazybull.ml.walk_forward_utils import (
    WalkForwardSplit,
    generate_walk_forward_splits_by_count,
)

# ── 与 batch_walk_forward.ps1 当前配置对齐的固定超参 ──────────────
# 若 batch 配置变更，请同步以下常量，保证诊断的“方差基线”代表真实训练。
LABEL_COLUMN = "neu_y_ret_20"
WINSORIZE_P = 0.01
VAL_RATIO = 0.2

# XGBoost 训练超参（n_estimators 由 CLI 提供以便冒烟）
TRAIN_HP: Dict[str, float] = dict(
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.3,
    min_child_weight=175,
    reg_alpha=0.05,
    reg_lambda=5.0,
    gamma=0.5,
    early_stopping_rounds=500,
)

# rank-weight 样本增强（Top/Bottom K）
RANK_WEIGHT = dict(topk=50, top_weight=3.0)

# 因子开关（对齐 batch 当前 ON/OFF 状态）
FACTOR_FLAGS: Dict[str, bool] = dict(
    enable_fundamental_features=True,
    enable_alt_features=True,
    enable_margin_features=False,
    enable_cyq_features=True,
    enable_fund_features=False,
    enable_express_features=True,
    enable_enhanced_features=True,
    enable_north_features=False,
    enable_lhb_features=True,
    enable_consensus_features=True,
    enable_consensus_revision_features=False,
    enable_cashflow_quality_features=False,
    feature_stability_filter=False,
    factor_prune=False,
)

# 评估按日截面 RankIC 时一并统计的 TopK（10 对齐 bt_top_n）
TOPK_VALUES = [10]


def resolve_split(
    loader: DataLoader,
    split_count: int,
    final_date: str,
    split_index: Optional[int],
    train_window_years: int,
    test_window_months: int,
    rebalance_freq: int,
) -> WalkForwardSplit:
    """生成 walk-forward 切分并选取目标 split（默认最近期 = 列表最后一个）。"""
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()

    splits = generate_walk_forward_splits_by_count(
        trade_cal=trade_cal,
        split_count=split_count,
        final_date=final_date,
        train_window_years=train_window_years,
        test_window_months=test_window_months,
        rebalance_freq=rebalance_freq,
    )
    if len(splits) == 0:
        raise ValueError("未生成任何切分，请检查 split-count / final-date 参数")

    if split_index is None:
        # 最近期 = 最大 split_index（列表已按 split_index 升序，取最后一个）
        selected = splits[-1]
    else:
        matched = [s for s in splits if s.split_index == split_index]
        if not matched:
            raise ValueError(
                f"未找到 split_index={split_index}，可用范围 0~{splits[-1].split_index}"
            )
        selected = matched[0]
    return selected


def prepare_once(
    storage: Storage,
    loader: DataLoader,
    train_start: str,
    train_end: str,
) -> Dict:
    """加载训练数据并按 batch 配置准备一次，所有种子复用。"""
    df_train, _ = load_features_data(storage, loader, train_start, train_end)

    label_transform_fn = lambda d: transform_labels_cs_zscore(
        d, label_column=LABEL_COLUMN, winsorize_p=WINSORIZE_P
    )
    (
        X_train,
        y_train,
        X_val,
        y_val,
        feature_columns,
        df_train_split,
        df_val_split,
        data_stats,
        df_val_split_original,
    ) = prepare_training_data(
        df_train,
        LABEL_COLUMN,
        val_ratio=VAL_RATIO,
        label_transform_fn=label_transform_fn,
        **FACTOR_FLAGS,
    )
    del df_train
    gc.collect()

    sample_weight = build_rank_sample_weights(
        df_train=df_train_split,
        label_column=LABEL_COLUMN,
        topk=RANK_WEIGHT["topk"],
        top_weight=RANK_WEIGHT["top_weight"],
    )

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_columns=feature_columns,
        df_val_original=df_val_split_original,
        sample_weight=sample_weight,
    )


def compute_cs_rankic(model, df_val_original: pd.DataFrame, feature_columns: List[str]):
    """用保留 NaN 的验证集自行 predict，计算按日截面 RankIC（绕开 fillna(0)）。

    Returns:
        (preds, summary) — preds 为逐样本预测向量（与 df_val_original 行对齐，用于 bagging），
        summary 为 summarize_daily_metrics 的输出。
    """
    X_eval = df_val_original[feature_columns].astype(np.float32)  # 保留 NaN
    preds = model.predict(X_eval)

    df_eval = df_val_original[["trade_date", "ts_code", LABEL_COLUMN]].copy()
    df_eval["pred_score"] = preds

    daily = evaluate_predictions_by_date(
        df=df_eval,
        date_col="trade_date",
        prediction_col="pred_score",
        return_col=LABEL_COLUMN,
        topk_values=TOPK_VALUES,
    )
    summary = summarize_daily_metrics(daily)
    return preds, summary


def _summary_to_record(metric: str, seed, summary: Dict, **extra) -> Dict:
    """从 summarize_daily_metrics 输出构造一行结果记录。"""
    top10_key = "Top10平均收益_均值"
    return dict(
        metric=metric,
        seed=seed,
        cs_rankic_mean=summary.get("RankIC_均值", np.nan),
        cs_rankic_std=summary.get("RankIC_标准差", np.nan),
        cs_rankic_ir=summary.get("RankIC_IR", np.nan),
        top10_ret_mean=summary.get(top10_key, np.nan),
        **extra,
    )


def run_diagnosis(
    data: Dict,
    seeds: List[int],
    metrics: List[str],
    n_estimators: int,
) -> pd.DataFrame:
    """对每个 (metric, seed) 训练一次并评估；auto 组缓存预测用于 bagging。"""
    records: List[Dict] = []
    auto_pred_vectors: List[np.ndarray] = []

    # 训练循环期间静音训练/评估模块内部日志，保持诊断进度清晰
    logger.disable("src.lazybull.ml.train_core")
    logger.disable("src.lazybull.ml.eval_utils")
    try:
        for metric in metrics:
            for seed in seeds:
                t0 = time.time()
                model, train_params, _, val_metrics = train_xgboost_model(
                    data["X_train"],
                    data["y_train"],
                    data["X_val"],
                    data["y_val"],
                    task="regression",
                    skip_label_winsorize=True,  # cs_zscore 已 winsorize
                    sample_weight=data["sample_weight"],
                    n_estimators=n_estimators,
                    random_state=seed,
                    early_stopping_metric=metric,
                    **TRAIN_HP,
                )
                preds, summary = compute_cs_rankic(
                    model, data["df_val_original"], data["feature_columns"]
                )
                rec = _summary_to_record(
                    metric=metric,
                    seed=seed,
                    summary=summary,
                    best_iteration=train_params.get("best_iteration", np.nan),
                    pool_rank_ic=val_metrics.get("rank_ic", np.nan),
                    elapsed_s=round(time.time() - t0, 1),
                )
                records.append(rec)
                if metric == "auto":
                    auto_pred_vectors.append(preds)

                del model
                gc.collect()

                logger.info(
                    f">>> [{metric} seed={seed}] best_iter={rec['best_iteration']} "
                    f"pool_ic={rec['pool_rank_ic']:.4f} cs_ic={rec['cs_rankic_mean']:.4f} "
                    f"top10={rec['top10_ret_mean']:.4f} ({rec['elapsed_s']}s)"
                )
    finally:
        logger.enable("src.lazybull.ml.train_core")
        logger.enable("src.lazybull.ml.eval_utils")

    df = pd.DataFrame(records)

    # bagging：auto 组多种子预测平均后再算按日截面 RankIC
    if len(auto_pred_vectors) >= 2:
        mean_pred = np.mean(np.vstack(auto_pred_vectors), axis=0)
        df_eval = data["df_val_original"][["trade_date", "ts_code", LABEL_COLUMN]].copy()
        df_eval["pred_score"] = mean_pred
        daily = evaluate_predictions_by_date(
            df=df_eval,
            date_col="trade_date",
            prediction_col="pred_score",
            return_col=LABEL_COLUMN,
            topk_values=TOPK_VALUES,
        )
        bag_summary = summarize_daily_metrics(daily)
        bag_rec = _summary_to_record(
            metric="auto_bagging",
            seed=f"ALL({len(auto_pred_vectors)})",
            summary=bag_summary,
            best_iteration=np.nan,
            pool_rank_ic=np.nan,
            elapsed_s=np.nan,
        )
        df = pd.concat([df, pd.DataFrame([bag_rec])], ignore_index=True)

    return df


def print_report(df: pd.DataFrame, split: WalkForwardSplit) -> None:
    """终端打印诊断汇总报告。"""
    print("\n" + "=" * 88)
    print("  训练稳定性诊断报告")
    print("=" * 88)
    print(
        f"  目标 split: index={split.split_index}  "
        f"train=[{split.train_start}~{split.train_end}]  "
        f"test=[{split.test_start}~{split.test_end}]"
    )

    # 明细表（按 metric 分组）
    seed_rows = df[df["metric"] != "auto_bagging"]
    cols = [
        "metric",
        "seed",
        "best_iteration",
        "pool_rank_ic",
        "cs_rankic_mean",
        "cs_rankic_std",
        "cs_rankic_ir",
        "top10_ret_mean",
    ]
    for metric in seed_rows["metric"].unique():
        g = seed_rows[seed_rows["metric"] == metric]
        print(f"\n── 明细（metric={metric}）" + "─" * 56)
        print(g[cols].to_string(index=False))

    # 汇总：A1 / A2 逐 metric
    print("\n" + "=" * 88)
    print("  汇总分析")
    print("=" * 88)
    for metric in seed_rows["metric"].unique():
        g = seed_rows[seed_rows["metric"] == metric]
        bi = g["best_iteration"].astype(float)
        bi_mean, bi_std = bi.mean(), bi.std()
        bi_cv = (bi_std / bi_mean) if bi_mean else np.nan
        pool_mean = g["pool_rank_ic"].mean()
        cs_mean = g["cs_rankic_mean"].mean()
        cs_seed_std = g["cs_rankic_mean"].std()  # 种子间离散
        print(f"\n  [metric={metric}]  (n_seeds={len(g)})")
        print(
            f"    A1 best_iteration: 均值={bi_mean:.1f}  标准差={bi_std:.1f}  "
            f"变异系数CV={bi_cv:.1%}   <- CV 越大=早停越不稳"
        )
        print(
            f"    A2 全池RankIC均值={pool_mean:.4f}  vs  按日截面RankIC均值={cs_mean:.4f}  "
            f"(差={pool_mean - cs_mean:+.4f})"
        )
        print(
            f"       单种子按日截面RankIC: 种子间均值={cs_mean:.4f}  种子间标准差={cs_seed_std:.4f}"
        )

    # A1 对比：auto vs rank_ic 的 best_iteration CV
    if {"auto", "rank_ic"}.issubset(set(seed_rows["metric"].unique())):
        def _cv(m: str) -> float:
            bi = seed_rows[seed_rows["metric"] == m]["best_iteration"].astype(float)
            return (bi.std() / bi.mean()) if bi.mean() else np.nan

        print("\n  [A1 对比] best_iteration 变异系数：")
        print(
            f"    auto(MAE)={_cv('auto'):.1%}   rank_ic(全池)={_cv('rank_ic'):.1%}   "
            f"<- rank_ic 更小则说明换早停指标能降方差"
        )

    # #1 对比：bagging vs 单种子（auto 组）
    bag = df[df["metric"] == "auto_bagging"]
    auto_g = seed_rows[seed_rows["metric"] == "auto"]
    if len(bag) == 1 and len(auto_g) >= 2:
        single_mean = auto_g["cs_rankic_mean"].mean()
        single_std = auto_g["cs_rankic_mean"].std()
        bag_val = bag.iloc[0]["cs_rankic_mean"]
        print("\n  [#1 多种子 bagging 收益]（auto 组）：")
        print(
            f"    单种子按日截面RankIC: 均值={single_mean:.4f}  种子间标准差={single_std:.4f}"
        )
        print(
            f"    bagging(预测平均)按日截面RankIC = {bag_val:.4f}  "
            f"(相对单种子均值 {bag_val - single_mean:+.4f})"
        )
        print(
            "    说明：bagging 把种子间标准差直接收敛为 0（确定性结果），"
            "若 RankIC 不降反升即坐实 bagging 值得实装。"
        )
    print("=" * 88 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练稳定性诊断（多种子 × 早停指标）")
    parser.add_argument("--split-count", type=int, default=14, help="切分数量，默认 13（batch 0101 组）")
    parser.add_argument("--final-date", type=str, default="20260324", help="最终日期 YYYYMMDD")
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help="目标 split 索引；默认 None=最近期(最大索引)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,1,2,3,4,5,6,7,8,9",
        help="逗号分隔的随机种子列表，默认 10 个",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="auto,rank_ic",
        help="逗号分隔的早停指标，默认 auto,rank_ic",
    )
    parser.add_argument("--train-window-years", type=int, default=6, help="训练窗口年数，默认 6")
    parser.add_argument("--test-window-months", type=int, default=6, help="测试窗口月数，默认 6")
    parser.add_argument("--rebalance-freq", type=int, default=20, help="调仓频率（影响 split 边界），默认 20")
    parser.add_argument("--n-estimators", type=int, default=3000, help="树数量上限，默认 3000")
    parser.add_argument("--data-root", type=str, default=None, help="数据根目录；默认用 base.yaml")
    args = parser.parse_args()

    setup_logger()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not seeds or not metrics:
        logger.error("seeds / metrics 解析为空，请检查参数")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("训练稳定性诊断")
    logger.info(f"  种子: {seeds}")
    logger.info(f"  早停指标: {metrics}")
    logger.info(f"  n_estimators: {args.n_estimators}")
    logger.info(f"  总训练次数: {len(seeds) * len(metrics)}")
    logger.info("=" * 60)

    storage = Storage(root_path=args.data_root)
    loader = DataLoader(storage)

    # 1. 定位目标 split（默认最近期）并打印供核对
    split = resolve_split(
        loader=loader,
        split_count=args.split_count,
        final_date=args.final_date,
        split_index=args.split_index,
        train_window_years=args.train_window_years,
        test_window_months=args.test_window_months,
        rebalance_freq=args.rebalance_freq,
    )
    logger.info(
        f"选中 split: index={split.split_index} "
        f"train=[{split.train_start}~{split.train_end}] "
        f"test=[{split.test_start}~{split.test_end}]  "
        f"(请核对 test_end≈final_date 即为最近期)"
    )

    # 2. 数据准备（只做一次）
    data = prepare_once(storage, loader, split.train_start, split.train_end)

    # 3. 多种子 × 早停指标训练 + bagging
    t_start = time.time()
    df = run_diagnosis(data, seeds, metrics, args.n_estimators)
    logger.info(f"全部训练完成，总耗时 {time.time() - t_start:.0f}s")

    # 4. 报告 + CSV
    df.insert(0, "split_index", split.split_index)
    print_report(df, split)

    reports_dir = Path(get_reports_root(str(Path(get_data_root()) / "reports")))
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = reports_dir / f"diagnose_training_stability_split{split.split_index}_{ts}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
