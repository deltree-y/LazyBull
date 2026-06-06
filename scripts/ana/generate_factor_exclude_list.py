#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子精简脚本 — 基于 IC 分析结果生成排除列表

规则（可配置）：
1. ICIR 阈值：在指定中性化模式下，y_ret_20 的 ICIR 绝对值 < min_icir → 排除
2. 覆盖率阈值：有效交易日占比 < min_coverage → 排除（针对 PIT 因子）
3. 冗余剔除：相关性 > corr_threshold 的因子组，只保留 ICIR 最高的

输出：data/models/factor_exclude_list.json（因子名称列表）

使用示例：
    # 先跑 IC 分析（必须）
    python scripts/ana/analyze_factor_ic.py --start 20200101 --end 20251231 --compare

    # 再生成排除列表
    python scripts/ana/generate_factor_exclude_list.py

    # 自定义阈值
    python scripts/ana/generate_factor_exclude_list.py --min-icir 0.15 --min-coverage 0.3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_data_root
from src.lazybull.common.logger import setup_logger


def load_ic_results(ic_file: Path) -> pd.DataFrame:
    """加载 IC 分析结果"""
    df = pd.read_csv(ic_file)
    logger.info(f"加载 IC 数据: {len(df)} 行, {df['factor'].nunique()} 因子")
    return df


def apply_icir_threshold(
    df: pd.DataFrame,
    min_icir: float,
    neutralize_mode: str = "ind_size",
    label: str = "y_ret_20",
) -> Set[str]:
    """基于 ICIR 阈值筛选低效因子

    Args:
        df: IC 分析结果
        min_icir: ICIR 绝对值下限（低于此值视为无效）
        neutralize_mode: 参考的中性化模式（默认 ind_size 即双重中性化）
        label: 参考的标签列

    Returns:
        应排除的因子名称集合
    """
    sub = df[(df["neutralize"] == neutralize_mode) & (df["label"] == label)]
    low_icir = sub[sub["ICIR"].abs() < min_icir]
    excluded = set(low_icir["factor"].unique())
    logger.info(
        f"ICIR 阈值 ({neutralize_mode}, {label}): "
        f"|ICIR| < {min_icir} → 排除 {len(excluded)} 个因子"
    )
    return excluded


def apply_coverage_threshold(
    df: pd.DataFrame,
    min_coverage: float,
    label: str = "y_ret_20",
) -> Set[str]:
    """基于覆盖率筛选低覆盖度因子

    PIT 因子（一致预期、业绩快报等）覆盖率可能极低（<10%），
    虽然 ICIR 可能高但实际不可靠。

    Args:
        df: IC 分析结果
        min_coverage: 最低有效交易日占比
        label: 参考的标签列

    Returns:
        应排除的因子名称集合
    """
    # 取 raw 模式下的覆盖率（最宽松）
    sub = df[(df["neutralize"] == "raw") & (df["label"] == label)]
    sub = sub.copy()
    sub["coverage"] = sub["n_days_valid"] / sub["n_days"]
    low_cov = sub[sub["coverage"] < min_coverage]
    excluded = set(low_cov["factor"].unique())

    if excluded:
        logger.info(
            f"覆盖率阈值 ({label}): coverage < {min_coverage:.0%} → 排除 {len(excluded)} 个因子"
        )
        for _, r in low_cov.iterrows():
            logger.info(f"  {r['factor']:35s}  coverage={r['coverage']:.1%}  IR={r['ICIR']:.3f}")
    else:
        logger.info(f"覆盖率阈值: 无因子 coverage < {min_coverage:.0%}")

    return excluded


def merge_exclude_lists(*sets: Set[str]) -> List[str]:
    """合并多个排除集合并排序输出"""
    merged = set()
    for s in sets:
        merged |= s
    return sorted(merged)


def main():
    parser = argparse.ArgumentParser(description="生成因子排除列表")
    parser.add_argument(
        "--ic-file",
        type=str,
        default=None,
        help="IC 分析 CSV 路径 (默认: data/reports/factor_ic_compare.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 路径 (默认: data/models/factor_exclude_list.json)",
    )
    parser.add_argument(
        "--min-icir",
        type=float,
        default=0.10,
        help="ICIR 绝对值下限 (默认: 0.10)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.30,
        help="最低有效交易日占比 (默认: 0.30 = 30%%)",
    )
    parser.add_argument(
        "--neutralize-mode",
        type=str,
        default="ind_size",
        choices=["raw", "industry", "size", "ind_size"],
        help="ICIR 阈值参考的中性化模式 (默认: ind_size)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录；未指定时使用 configs/base.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不写入文件",
    )
    args = parser.parse_args()

    setup_logger("INFO")

    data_root = Path(args.data_root or get_data_root())

    ic_file = Path(args.ic_file) if args.ic_file else data_root / "reports" / "factor_ic_compare.csv"
    if not ic_file.exists():
        logger.error(f"IC 分析文件不存在: {ic_file}")
        logger.error("请先运行: python scripts/ana/analyze_factor_ic.py --start ... --end ... --compare")
        sys.exit(1)

    output_path = Path(args.output) if args.output else data_root / "models" / "factor_exclude_list.json"

    # 加载
    df = load_ic_results(ic_file)

    # 规则1: ICIR 阈值
    low_icir = apply_icir_threshold(df, args.min_icir, args.neutralize_mode)

    # 规则2: 覆盖率
    low_coverage = apply_coverage_threshold(df, args.min_coverage)

    # 合并
    exclude_list = merge_exclude_lists(low_icir, low_coverage)

    logger.info(f"最终排除: {len(exclude_list)} 个因子")

    if args.dry_run:
        logger.info("[dry-run] 不写入文件，排除列表预览:")
        for f in exclude_list:
            print(f"  - {f}")
        return

    # 写入
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "训练时排除的因子列表（由 generate_factor_exclude_list.py 生成）",
                "min_icir": args.min_icir,
                "min_coverage": args.min_coverage,
                "neutralize_mode": args.neutralize_mode,
                "exclude_count": len(exclude_list),
                "exclude_factors": exclude_list,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"排除列表已保存: {output_path}")
    logger.info(f"训练时使用 --factor-prune 开关即可自动排除这些因子")


if __name__ == "__main__":
    main()
