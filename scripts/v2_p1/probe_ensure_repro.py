# -*- coding: utf-8 -*-
"""P1 先行探针：ensure 链路可复现性探测（v2 P1 阶段）。

**注意：口径 A 已被单元 3 口径 B（replay_compare）取代，本脚本仅历史诊断留存，
其『一致』结论不作验收证据。**

目的（方案 §8-P1 闸门）：判定「同一 raw 快照 → 同一特征产物」是否成立，
即 ensure 增量链路（cs_infer）与 build_clean_features 批量链路（cs_train）
在同一 raw 上是否逐值一致（1e-6 容差，对齐母截面先例）。

**实测约束（2026-10-01 探测发现）**：cs_infer 按设计**只滚动保留最新交易日**
（单日快照，纸面链路每次重建），不存在多月历史分区 ⇒ 「抽 3 个月分区比对」
在产物层不可行。本探针因此改为两档口径：
- **口径 A（可立即做，本脚本实现）**：cs_infer 最新日 vs cs_train 同日逐值比对
  ——直接回答「两条路径在同一日的同股同列是否一致」（ensure 可复现性的核心证据）；
- **口径 B（登记为 P1 后续任务）**：历史区间重放比对——在临时数据根上对 3 个月区间
  跑「ensure 链路重放 vs cs_train 冻结快照」，属冻结参照实现的前置，机器时间另申请。

设计原则：只读、不触发任何下载或重建；报告落 temp/（临时产物）。

用法：
    python scripts/v2_p1/probe_ensure_repro.py [--tolerance 1e-6]
        [--out temp/p1_ensure_repro_probe.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

# 项目根
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.data import Storage  # noqa: E402


def _load_partition(
    storage: Storage, trade_date: str, subdir: str
) -> Optional[pd.DataFrame]:
    """加载指定日期的特征分区（cs_train 或 cs_infer）。"""
    try:
        if not storage.is_feature_exists(trade_date, subdir=subdir):
            return None
        # Storage 真实方法名：load_cs_train_day(trade_date, subdir=...)
        return storage.load_cs_train_day(trade_date, subdir=subdir)
    except Exception as exc:  # noqa: BLE001 - 探测器须吞单点失败继续
        logger.warning(f"加载 {subdir}/{trade_date} 失败: {exc}")
        return None


def _list_partition_dates(storage: Storage, subdir: str) -> List[str]:
    """列出某子目录下全部特征分区日期（YYYYMMDD，按文件名）。"""
    target = storage.features_path / subdir
    if not target.exists():
        return []
    return sorted(p.stem for p in target.glob("*.parquet"))


# 截面口径差异列（P1 探测实测，2026-10-01）：cs_train 批量构建用「当日全市场截面」做
# zscore/百分位，cs_infer 单日重建用「当日可得截面」——两者截面边界不同 ⇒ 逐值必差。
# 这是 R1-4 已登记的「构建窗口口径漂移」的预期实例，**不是实现漂移**，比对时排除。
SECTION_CALIBER_COLS = frozenset(
    {"vol_regime_percentile"}  # 滚动百分位，窗口口径敏感
)
SECTION_CALIBER_PREFIXES = ("zscore_",)  # 截面 zscore 全族（含 _sz 后缀）


def _is_section_caliber_col(col: str) -> bool:
    """判定是否为截面口径差异列（从「真实漂移」判据中排除，单列登记）。"""
    return col in SECTION_CALIBER_COLS or col.startswith(SECTION_CALIBER_PREFIXES)


def _compare_pair(
    df_train: pd.DataFrame, df_infer: pd.DataFrame, tolerance: float
) -> Dict[str, object]:
    """比对同一日的 cs_train 与 cs_infer 分区（按 ts_code 对齐、交集列逐值）。

    返回: {交集列数, 交集行数, 逐值一致率（非截面口径列）, 最大绝对差, 结论,
           截面口径列不一致数（预期内，单列登记）}
    结论 ∈ {一致, 数值漂移, 列集不一致, 行集不一致}
    """
    if df_train is None or df_infer is None:
        return {"结论": "缺分区"}

    # 按 ts_code 对齐
    key = "ts_code"
    if key not in df_train.columns or key not in df_infer.columns:
        return {"结论": "缺 ts_code 键"}

    common_cols = sorted(
        c for c in df_train.columns if c in df_infer.columns and c != key
    )
    only_train = sorted(set(df_train.columns) - set(df_infer.columns) - {key})
    only_infer = sorted(set(df_infer.columns) - set(df_train.columns) - {key})

    merged = df_train[[key] + common_cols].merge(
        df_infer[[key] + common_cols],
        on=key,
        how="inner",
        suffixes=("_tr", "_inf"),
    )
    n_rows = len(merged)
    if n_rows == 0:
        return {"结论": "行集不一致", "交集列数": len(common_cols), "交集行数": 0}

    # 数值列逐值比对（截面口径列单列登记、不进漂移判据）
    max_abs_diff = 0.0
    n_total = 0
    n_match = 0
    section_caliber_mismatch_cols = 0
    for col in common_cols:
        a = merged[f"{col}_tr"]
        b = merged[f"{col}_inf"]
        # 只比数值列
        if not (
            pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        ):
            continue
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        both_valid = a_num.notna() & b_num.notna()
        diff = (a_num - b_num).abs()
        col_mism = int((diff[both_valid] > tolerance).sum())
        if _is_section_caliber_col(col):
            # 截面口径差异列：登记不一致列数，不进逐值一致率判据
            if col_mism > 0:
                section_caliber_mismatch_cols += 1
            continue
        n_total += int(both_valid.sum())
        n_match += int((diff[both_valid] <= tolerance).sum())
        if len(diff[both_valid]) > 0:
            max_abs_diff = max(max_abs_diff, float(diff[both_valid].max()))

    match_ratio = (n_match / n_total) if n_total else 1.0
    if only_train or only_infer:
        conclusion = "列集不一致"
    elif match_ratio >= 1.0 - 1e-9:
        conclusion = "一致"
    else:
        conclusion = "数值漂移"

    return {
        "交集列数": len(common_cols),
        "仅cs_train列数": len(only_train),
        "仅cs_infer列数": len(only_infer),
        "交集行数": n_rows,
        "逐值一致率_非截面口径": round(match_ratio, 6),
        "最大绝对差_非截面口径": max_abs_diff,
        "截面口径不一致列数_预期内": section_caliber_mismatch_cols,
        "结论": conclusion,
    }


def probe_overlap(storage: Storage, tolerance: float) -> List[Dict[str, object]]:
    """口径 A：cs_infer 最新日 vs cs_train 同日逐值比对。

    cs_infer 只滚动保留最新日（实测单日快照）⇒ 取 cs_infer 全部分区
    （正常为 1 个日期），逐日与 cs_train 同日期比对。
    """
    rows: List[Dict[str, object]] = []
    infer_dates = _list_partition_dates(storage, "cs_infer")
    logger.info(f"cs_infer 现有分区: {infer_dates}（滚动保留最新日）")
    for trade_date in infer_dates:
        df_train = _load_partition(storage, trade_date, "cs_train")
        df_infer = _load_partition(storage, trade_date, "cs_infer")
        if df_infer is None:
            continue
        if df_train is None:
            logger.warning(f"{trade_date}: cs_infer 存在但 cs_train 缺失（无法比对）")
            rows.append({"日期": trade_date, "结论": "cs_train 缺分区"})
            continue
        result = _compare_pair(df_train, df_infer, tolerance)
        result["日期"] = trade_date
        rows.append(result)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    # 输出编码防御：管道 / 重定向到文件时强制 UTF-8（含符号字符）
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty():
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
    parser = argparse.ArgumentParser(description="P1 先行：ensure 可复现性探测（口径 A）")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="逐值容差")
    parser.add_argument(
        "--out",
        default="temp/p1_ensure_repro_probe.csv",
        help="探测报告输出路径（临时产物，落 temp/）",
    )
    args = parser.parse_args(argv)

    # Storage 无参构造：自动走 data.root 配置（真实分区）
    storage = Storage()
    logger.info(f"口径 A：cs_infer 最新日 vs cs_train 同日；容差: {args.tolerance}")

    all_rows = probe_overlap(storage, args.tolerance)
    if not all_rows:
        logger.error("cs_infer 无分区（或 cs_train 同日缺失）——无法探测")
        print("\n提示：cs_infer 是纸面链路的滚动快照，需先跑 paper_trade.py run 生成；")
        print("历史区间的可复现性需走口径 B（临时数据根重放比对，机器时间另申请）。")
        return 1

    df = pd.DataFrame(all_rows)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 摘要
    total = len(df)
    consistent = int((df["结论"] == "一致").sum())
    drift = int((df["结论"] == "数值漂移").sum())
    col_mismatch = int((df["结论"] == "列集不一致").sum())
    row_mismatch = int((df["结论"] == "行集不一致").sum())

    print("\n=== P1 可复现性探测摘要（口径 A：cs_infer 最新日 vs cs_train 同日）===")
    print(f"比对分区数: {total}")
    print(f"  一致: {consistent} ({consistent/total*100:.1f}%)")
    print(f"  数值漂移: {drift} ({drift/total*100:.1f}%)")
    print(f"  列集不一致: {col_mismatch} ({col_mismatch/total*100:.1f}%)")
    print(f"  行集不一致: {row_mismatch} ({row_mismatch/total*100:.1f}%)")
    diff_col = "最大绝对差_非截面口径"
    if diff_col in df.columns and df[diff_col].notna().any():
        print(f"  全局最大绝对差（非截面口径列）: {df[diff_col].max():.2e}")
    sec_col = "截面口径不一致列数_预期内"
    if sec_col in df.columns and df[sec_col].notna().any():
        print(f"  截面口径差异列数（zscore_*/vol_regime，预期内，不进判据）: {int(df[sec_col].iloc[0])}")
    if "仅cs_infer列数" in df.columns and df["仅cs_infer列数"].notna().any():
        extra = df["仅cs_infer列数"].iloc[0]
        if extra:
            print(f"  cs_infer 多出列数（运行时派生族，预期内）: {extra}")
    print(f"\n明细报告: {out_path}")
    print("\n判定（方案 §8-P1 闸门对齐母截面先例）：")
    if drift == 0 and row_mismatch == 0:
        print("  [OK] 非截面口径列逐值一致；截面口径差异与运行时派生列属预期内")
        print("  下一步：口径 B（历史区间重放比对）登记为 P1 任务，机器时间另申请")
        return 0
    else:
        print("  [WARN] 存在非截面口径数值漂移/行集不一致——需先归因")
        return 1


if __name__ == "__main__":
    sys.exit(main())
