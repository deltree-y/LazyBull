#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子相关性分析脚本

功能：
- 计算因子间截面 Spearman 相关性矩阵（逐日平均）
- 识别高度相关的冗余因子组
- 结合 ICIR 推荐每组保留的代表因子

输出：
- 终端报告：冗余组列表
- CSV：相关性矩阵 + 冗余分析结果

使用示例：
    # 全量因子相关性分析
    python scripts/ana/analyze_factor_correlation.py --start 20200101 --end 20251231

    # 指定因子子集
    python scripts/ana/analyze_factor_correlation.py --start 20200101 --end 20251231 \
        --factors neu_ret_20,zscore_bp,roe_waa --ic-file data/reports/factor_ic_compare.csv
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_data_root, get_reports_root
from src.lazybull.common.logger import setup_logger
from src.lazybull.factors.factor_evaluation import (
    auto_detect_factor_columns,
    compute_factor_correlation_matrix,
    find_redundant_factors,
)


def load_features_data(data_root, start_date, end_date, columns=None):
    features_dir = Path(data_root) / "features" / "cs_train"
    if not features_dir.exists():
        logger.error(f"特征目录不存在: {features_dir}")
        sys.exit(1)

    date_files = sorted(f for f in features_dir.iterdir() if f.suffix == ".parquet")
    filtered = [f for f in date_files if start_date <= f.stem <= end_date]

    if not filtered:
        logger.error(f"在 {start_date}~{end_date} 范围内未找到特征文件")
        sys.exit(1)

    logger.info(f"加载特征数据: {start_date}~{end_date}, 共 {len(filtered)} 个交易日")
    dfs = []
    for f in filtered:
        try:
            df = pd.read_parquet(f, columns=columns) if columns else pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"加载 {f.name} 失败: {e}")

    data = pd.concat(dfs, ignore_index=True)
    logger.info(f"加载完成: {len(data)} 行, {len(data.columns)} 列")
    return data


def load_icir_series(ic_file):
    """从 IC 分析结果构建 ICIR Series（取 ind_size 模式下 y_ret_20）"""
    df = pd.read_csv(ic_file)
    sub = df[(df["neutralize"] == "ind_size") & (df["label"] == "y_ret_20")]
    return sub.set_index("factor")["ICIR"]


def main():
    parser = argparse.ArgumentParser(description="因子相关性分析")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--factors", type=str, default=None, help="逗号分隔的因子列表")
    parser.add_argument("--threshold", type=float, default=0.75, help="冗余阈值 (默认 0.75)")
    parser.add_argument("--ic-file", type=str, default=None, help="IC分析CSV，用于选代表因子")
    parser.add_argument("--method", type=str, default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--columns-only", type=str, default=None, help="仅加载指定列")
    args = parser.parse_args()

    setup_logger("INFO")
    data_root = args.data_root or get_data_root()

    # 解析因子
    factor_list = None
    if args.factors:
        factor_list = [f.strip() for f in args.factors.split(",")]

    column_filter = None
    if args.columns_only:
        column_filter = [c.strip() for c in args.columns_only.split(",")]

    # 加载数据
    data = load_features_data(data_root, args.start, args.end, columns=column_filter)
    if factor_list is None:
        factor_list = auto_detect_factor_columns(data)
    else:
        factor_list = [f for f in factor_list if f in data.columns]

    logger.info(f"分析 {len(factor_list)} 个因子")

    # ── 预过滤：剔除 NaN 率 > 50% 的稀疏因子（加速计算）──
    nan_rates = data[factor_list].isna().mean()
    sparse = nan_rates[nan_rates > 0.5].index.tolist()
    if sparse:
        factor_list = [f for f in factor_list if f not in sparse]
        logger.info(f"剔除 {len(sparse)} 个稀疏因子 (NaN率>50%): {sparse[:5]}...")
        logger.info(f"剩余 {len(factor_list)} 个因子")

    # 计算相关性矩阵
    corr_matrix = compute_factor_correlation_matrix(
        data, factor_list, method=args.method, verbose=True
    )

    # 保存矩阵
    out_dir = Path(get_reports_root(str(Path(data_root) / "reports")))
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(out_dir / "factor_correlation_matrix.csv", encoding="utf-8-sig")
    logger.info(f"相关性矩阵已保存: factor_correlation_matrix.csv")

    # 加载 ICIR 并识别冗余
    icir_series = None
    ic_file = args.ic_file or str(Path(data_root) / "reports" / "factor_ic_compare.csv")
    ic_path = Path(ic_file)
    if ic_path.exists():
        icir_series = load_icir_series(ic_path)
        logger.info(f"加载 ICIR 数据: {len(icir_series)} 个因子")
    else:
        logger.warning(f"IC 文件不存在: {ic_file}，将随机保留代表因子")

    redundant = find_redundant_factors(corr_matrix, icir_series, threshold=args.threshold)
    redundant.to_csv(out_dir / "factor_redundancy.csv", encoding="utf-8-sig", index=False)
    logger.info(f"冗余分析已保存: factor_redundancy.csv")

    # 报告
    keepers = redundant[redundant["keep"]]
    clusters = redundant["cluster_id"].nunique()
    n_drop = len(redundant) - len(keepers)

    print(f"\n{'='*80}")
    print(f"  因子相关性分析报告")
    print(f"{'='*80}")
    print(f"  因子总数: {len(factor_list)}")
    print(f"  独立组数: {clusters} (阈值 |corr| > {args.threshold})")
    print(f"  可精简: {n_drop} 个冗余因子")

    # 展示多因子冗余组
    large_clusters = redundant.groupby("cluster_id").filter(lambda g: len(g) > 1)
    if len(large_clusters) > 0:
        print(f"\n── 冗余因子组（每组保留 ICIR 最高的）──")
        for cid in sorted(large_clusters["cluster_id"].unique()):
            group = large_clusters[large_clusters["cluster_id"] == cid]
            print(f"\n  组 {cid}:")
            for _, r in group.iterrows():
                mark = "★" if r["keep"] else " "
                icir_str = f"IR={r['ICIR']:.3f}" if pd.notna(r["ICIR"]) else ""
                print(f"    {mark} {r['factor']:35s} corr={r['max_corr']:.3f}  {icir_str}")

    if n_drop > 0:
        print(f"\n💡 精简建议：在 factor_exclude_list.json 中添加以下 {n_drop} 个冗余因子即可自动排除：")
        drops = redundant[~redundant["keep"]]["factor"].tolist()
        for d in sorted(drops):
            print(f"    - {d}")


if __name__ == "__main__":
    main()
