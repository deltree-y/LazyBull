#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子重要性分析脚本

功能：
- 从已训练的 XGBoost 模型中提取 feature_importances_
- 跨多个模型聚合，计算每个因子的平均/中位数重要性、排名、贡献占比
- 按因子类别分组统计
- 输出排序报告（终端 + CSV）

使用示例：
    python scripts/analyze_factor_importance.py
    python scripts/analyze_factor_importance.py --last-n 100
    python scripts/analyze_factor_importance.py --output data/reports/factor_importance.csv
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.common.logger import setup_logger

# ── 因子分类映射 ──────────────────────────────────────────────────
# 将每个特征列映射到所属类别，用于分组统计
FACTOR_CATEGORY = {
    # 动量与收益
    "neu_ret_1": "动量",
    "neu_ret_5": "动量",
    "neu_ret_20": "动量",
    "alpha_industry_5": "行业动量",
    "alpha_industry_20": "行业动量",
    "ind_ret_avg": "行业动量",
    "ind_momentum_rank": "行业动量",
    "zscore_acceleration": "动量",
    # 技术指标
    "zscore_ma_deviation_20": "技术指标",
    "zscore_macd_hist": "技术指标",
    "bb_pct": "技术指标",
    "rsi_14": "技术指标",
    "kdj_j": "技术指标",
    # 流动性与资金流
    "zscore_turnover_rate": "流动性",
    "vol_ratio_20": "流动性",
    "vol_burst_20": "流动性",
    "zscore_amount_ma20": "流动性",
    "zscore_net_mf_amount": "资金流",
    "zscore_elg_net_amount_sum_20": "资金流",
    "lg_net_amount_sum_5": "资金流",
    # 波动与形态
    "zscore_volatility_20": "波动率",
    "zscore_volatility_5": "波动率",
    "amplitude": "K线形态",
    "zscore_bb_width": "波动率",
    "upper_shadow": "K线形态",
    "lower_shadow": "K线形态",
    "spec_score": "投机度",
    # 估值与质量
    "zscore_size": "估值",
    "zscore_bp": "估值",
    "zscore_dv_ttm": "估值",
    "zscore_pe_ttm": "估值",
    "is_loss": "质量",
    "list_days": "质量",
    # 市场环境
    "mkt_adv_dec_ratio": "市场环境",
    "mkt_ret_avg_20": "市场环境",
    "mkt_turnover_std": "市场环境",
    "mkt_vol_20": "市场环境",
    # 基本面
    "zscore_roe_waa": "基本面",
    "zscore_or_yoy": "基本面",
    "zscore_netprofit_yoy": "基本面",
    "zscore_debt_to_assets": "基本面",
    "zscore_q_gr_yoy": "基本面",
    # 另类数据 - 融资融券
    "rzye_chg_5": "融资融券",
    "rzye_chg_20": "融资融券",
    "rqye_rzye_ratio": "融资融券",
    "margin_net_buy_ratio": "融资融券",
    # 另类数据 - 股东人数
    "holder_num_chg": "股东人数",
    "holder_num_chg_2q": "股东人数",
    # 另类数据 - 业绩预告
    "forecast_type_score": "业绩预告",
    "forecast_chg_mid": "业绩预告",
    # 筹码胜率 (5000积分)
    "winner_rate": "筹码胜率",
    "weight_avg_bias": "筹码胜率",
    "cost_concentration": "筹码胜率",
    "winner_rate_chg_5": "筹码胜率",
    "winner_rate_chg_20": "筹码胜率",
    # 业绩快报 (5000积分)
    "express_revenue_yoy": "业绩快报",
    "express_profit_yoy": "业绩快报",
    "express_roe": "业绩快报",
    "express_surprise": "业绩快报",
    # 基金持仓 (5000积分)
    "fund_hold_ratio": "基金持仓",
    "fund_hold_ratio_chg": "基金持仓",
    "fund_count": "基金持仓",
    "fund_count_chg": "基金持仓",
}


def _get_category(feature_name: str) -> str:
    """获取因子所属类别，未知因子归为'其他'"""
    return FACTOR_CATEGORY.get(feature_name, "其他")


def load_models_and_importance(
    models_dir: Path, registry: dict, last_n: int = 0
) -> pd.DataFrame:
    """从已保存的模型中提取 feature importance

    Args:
        models_dir: 模型存储目录
        registry: 模型注册表
        last_n: 只分析最近 N 个模型，0 表示全部

    Returns:
        DataFrame，每行为一个 (模型版本, 因子) 的 importance 记录
    """
    models = registry.get("models", [])
    if last_n > 0:
        models = models[-last_n:]

    logger.info(f"将分析 {len(models)} 个模型的因子重要性")

    records = []
    skipped = 0

    for meta in models:
        version_str = meta["version_str"]
        model_file = models_dir / meta["model_file"]
        features_file = models_dir / meta["features_file"]

        if not model_file.exists() or not features_file.exists():
            skipped += 1
            continue

        try:
            model = joblib.load(model_file)
            with open(features_file, "r", encoding="utf-8") as f:
                feature_names = json.load(f)

            importance = model.feature_importances_
            if len(importance) != len(feature_names):
                logger.warning(f"{version_str}: importance 长度不匹配，跳过")
                skipped += 1
                continue

            for feat, imp in zip(feature_names, importance):
                records.append(
                    {
                        "version": meta["version"],
                        "feature": feat,
                        "importance": float(imp),
                    }
                )
        except Exception as e:
            logger.warning(f"{version_str}: 加载失败 - {e}")
            skipped += 1

    if skipped > 0:
        logger.info(f"跳过 {skipped} 个模型（文件缺失或加载失败）")

    return pd.DataFrame(records)


def compute_importance_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个因子的聚合重要性统计

    Args:
        df: load_models_and_importance 的输出

    Returns:
        按 mean_importance 降序排列的统计表
    """
    n_models = df["version"].nunique()

    # 在每个模型内计算排名（1=最重要）
    df["rank"] = df.groupby("version")["importance"].rank(ascending=False, method="min")

    # 在每个模型内计算贡献占比
    total_per_model = df.groupby("version")["importance"].transform("sum")
    df["contribution_pct"] = df["importance"] / total_per_model.where(total_per_model > 0, 1)

    stats = (
        df.groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            median_importance=("importance", "median"),
            std_importance=("importance", "std"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            appear_count=("version", "nunique"),
            zero_count=("importance", lambda x: (x == 0).sum()),
            mean_contribution_pct=("contribution_pct", "mean"),
        )
        .reset_index()
    )

    stats["zero_ratio"] = stats["zero_count"] / stats["appear_count"]
    stats["appear_ratio"] = stats["appear_count"] / n_models
    stats["category"] = stats["feature"].map(_get_category)

    # 按平均重要性降序排列
    stats = stats.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    stats.index = stats.index + 1  # 排名从1开始
    stats.index.name = "排名"

    return stats


def compute_category_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """按因子类别聚合统计"""
    cat_stats = (
        stats.groupby("category")
        .agg(
            factor_count=("feature", "count"),
            total_contribution=("mean_contribution_pct", "sum"),
            avg_contribution=("mean_contribution_pct", "mean"),
            best_rank=("mean_rank", "min"),
            worst_rank=("mean_rank", "max"),
        )
        .sort_values("total_contribution", ascending=False)
    )
    return cat_stats


def find_low_value_factors(stats: pd.DataFrame, top_pct: float = 0.8) -> pd.DataFrame:
    """识别低价值因子

    规则：
    1. 累计贡献占比排在后 20% 的因子
    2. 或 zero_ratio > 50%（超过一半模型中 importance=0）
    3. 或 mean_rank 排在最后 20%

    Args:
        stats: compute_importance_stats 的输出
        top_pct: 累计贡献阈值（默认 0.8 表示后 20%）

    Returns:
        低价值因子列表
    """
    total_features = len(stats)
    rank_threshold = total_features * top_pct

    # 累计贡献
    sorted_by_contrib = stats.sort_values("mean_contribution_pct", ascending=False)
    cumsum = sorted_by_contrib["mean_contribution_pct"].cumsum()
    total_contrib = sorted_by_contrib["mean_contribution_pct"].sum()
    low_contrib_features = sorted_by_contrib[cumsum > total_contrib * top_pct]["feature"].tolist()

    # zero_ratio 高
    high_zero = stats[stats["zero_ratio"] > 0.5]["feature"].tolist()

    # 排名靠后
    low_rank = stats[stats["mean_rank"] > rank_threshold]["feature"].tolist()

    # 取交集：至少满足2个条件
    candidates = set()
    all_flags = defaultdict(list)
    for feat in low_contrib_features:
        all_flags[feat].append("低贡献")
    for feat in high_zero:
        all_flags[feat].append("高零值率")
    for feat in low_rank:
        all_flags[feat].append("排名靠后")

    for feat, flags in all_flags.items():
        if len(flags) >= 2:
            candidates.add(feat)

    result = stats[stats["feature"].isin(candidates)].copy()
    result["标记原因"] = result["feature"].map(lambda x: " + ".join(all_flags.get(x, [])))
    return result.sort_values("mean_rank", ascending=False)


def print_report(stats: pd.DataFrame, cat_stats: pd.DataFrame, low_value: pd.DataFrame) -> None:
    """打印分析报告到终端"""
    print("\n" + "=" * 90)
    print("                        因子重要性分析报告")
    print("=" * 90)

    # 1. 总览
    print(f"\n分析因子数: {len(stats)}")
    print(f"分析模型数: {stats['appear_count'].max()}")

    # 2. Top 15 因子
    print("\n── Top 15 最重要因子 " + "─" * 60)
    top15 = stats.head(15)[
        ["feature", "category", "mean_importance", "mean_rank", "mean_contribution_pct"]
    ].copy()
    top15["mean_contribution_pct"] = top15["mean_contribution_pct"].map(lambda x: f"{x:.2%}")
    top15["mean_importance"] = top15["mean_importance"].map(lambda x: f"{x:.4f}")
    top15["mean_rank"] = top15["mean_rank"].map(lambda x: f"{x:.1f}")
    print(top15.to_string())

    # 3. Bottom 15 因子
    print("\n── Bottom 15 最不重要因子 " + "─" * 55)
    bottom15 = stats.tail(15)[
        [
            "feature",
            "category",
            "mean_importance",
            "mean_rank",
            "zero_ratio",
            "mean_contribution_pct",
        ]
    ].copy()
    bottom15["mean_contribution_pct"] = bottom15["mean_contribution_pct"].map(
        lambda x: f"{x:.2%}"
    )
    bottom15["mean_importance"] = bottom15["mean_importance"].map(lambda x: f"{x:.4f}")
    bottom15["mean_rank"] = bottom15["mean_rank"].map(lambda x: f"{x:.1f}")
    bottom15["zero_ratio"] = bottom15["zero_ratio"].map(lambda x: f"{x:.1%}")
    print(bottom15.to_string())

    # 4. 分类统计
    print("\n── 按因子类别汇总 " + "─" * 62)
    cat_display = cat_stats.copy()
    cat_display["total_contribution"] = cat_display["total_contribution"].map(
        lambda x: f"{x:.2%}"
    )
    cat_display["avg_contribution"] = cat_display["avg_contribution"].map(lambda x: f"{x:.2%}")
    cat_display["best_rank"] = cat_display["best_rank"].map(lambda x: f"{x:.1f}")
    cat_display["worst_rank"] = cat_display["worst_rank"].map(lambda x: f"{x:.1f}")
    print(cat_display.to_string())

    # 5. 建议删除的因子
    if len(low_value) > 0:
        print("\n── 建议关注的低价值因子（满足≥2项：低贡献 / 高零值率 / 排名靠后）" + "─" * 15)
        low_display = low_value[
            [
                "feature",
                "category",
                "mean_rank",
                "zero_ratio",
                "mean_contribution_pct",
                "标记原因",
            ]
        ].copy()
        low_display["mean_contribution_pct"] = low_display["mean_contribution_pct"].map(
            lambda x: f"{x:.2%}"
        )
        low_display["mean_rank"] = low_display["mean_rank"].map(lambda x: f"{x:.1f}")
        low_display["zero_ratio"] = low_display["zero_ratio"].map(lambda x: f"{x:.1%}")
        print(low_display.to_string())
    else:
        print("\n未发现明显的低价值因子。")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="因子重要性分析")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="数据根目录 (默认: ./data)",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=0,
        help="只分析最近 N 个模型 (默认: 全部)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV 输出路径 (默认: data/reports/factor_importance.csv)",
    )
    args = parser.parse_args()

    setup_logger("INFO")

    data_root = Path(args.data_root)
    models_dir = data_root / "models"
    registry_file = models_dir / "model_registry.json"

    if not registry_file.exists():
        logger.error(f"注册表文件不存在: {registry_file}")
        sys.exit(1)

    with open(registry_file, "r", encoding="utf-8") as f:
        registry = json.load(f)

    total_models = len(registry.get("models", []))
    logger.info(f"注册表中共有 {total_models} 个模型")

    # 1. 提取 importance
    raw_df = load_models_and_importance(models_dir, registry, last_n=args.last_n)
    if raw_df.empty:
        logger.error("未能从任何模型中提取 importance，请检查模型文件")
        sys.exit(1)

    logger.info(
        f"成功提取 {raw_df['version'].nunique()} 个模型、"
        f"{raw_df['feature'].nunique()} 个因子的 importance 数据"
    )

    # 2. 计算统计
    stats = compute_importance_stats(raw_df)
    cat_stats = compute_category_stats(stats)
    low_value = find_low_value_factors(stats)

    # 3. 打印报告
    print_report(stats, cat_stats, low_value)

    # 4. 保存 CSV
    output_path = args.output or str(data_root / "reports" / "factor_importance.csv")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 输出列重命名为中文
    export_cols = {
        "feature": "因子名称",
        "category": "类别",
        "mean_importance": "平均重要性",
        "median_importance": "中位数重要性",
        "std_importance": "重要性标准差",
        "mean_rank": "平均排名",
        "median_rank": "中位数排名",
        "appear_count": "出现模型数",
        "appear_ratio": "出现比例",
        "zero_count": "零值次数",
        "zero_ratio": "零值比例",
        "mean_contribution_pct": "平均贡献占比",
    }
    export_df = stats.rename(columns=export_cols)
    export_df.to_csv(output_path, encoding="utf-8-sig")
    logger.info(f"详细报告已保存: {output_path}")

    # 5. 保存分类统计
    cat_output = output_path.parent / "factor_importance_by_category.csv"
    cat_export = cat_stats.rename(
        columns={
            "factor_count": "因子数",
            "total_contribution": "总贡献占比",
            "avg_contribution": "平均贡献占比",
            "best_rank": "最佳排名",
            "worst_rank": "最差排名",
        }
    )
    cat_export.to_csv(cat_output, encoding="utf-8-sig")
    logger.info(f"分类统计已保存: {cat_output}")


if __name__ == "__main__":
    main()
