#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建clean和features数据脚本（薄入口）

功能：
- 假设raw数据已存在，若缺失则报错
- 只负责计算clean和feature并保存（partitioned存储）
- 不进行raw数据下载
- 支持force参数强制重新构建已存在的数据

核心逻辑已下沉：
- clean 层批量构建: src/lazybull/data/build_clean.py (build_clean_data)
- features 层批量流水线: src/lazybull/features/pipeline.py (build_features_data)
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataCleaner, DataLoader, Storage
from src.lazybull.data.build_clean import build_clean_data
from src.lazybull.features import FeatureBuilder
from src.lazybull.features.pipeline import build_features_data

OPTIONAL_FEATURE_FLAG_ATTRS = (
    "enable_fundamental_features",
    "enable_alt_features",
    "enable_margin_features",
    "enable_cyq_features",
    "enable_fund_features",
    "enable_express_features",
    "enable_north_features",
    "enable_lhb_features",
    "enable_consensus_features",
    "enable_cashflow_quality_features",
    "enable_consensus_revision_features",
    "enable_dividend_policy_features",
    "enable_announcement_risk_features",
)


def apply_build_all_feature_flags(args: argparse.Namespace) -> argparse.Namespace:
    """当 --build-all 启用时，统一打开全部可选因子开关。"""
    if not getattr(args, "build_all", False):
        return args

    for attr in OPTIONAL_FEATURE_FLAG_ATTRS:
        setattr(args, attr, True)
    return args


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建clean和features数据（假设raw已存在）")
    parser.add_argument(
        "--start-date", default="20200101", help="开始日期，格式YYYYMMDD（默认：20200101）"
    )
    parser.add_argument(
        "--end-date", default="20251231", help="结束日期，格式YYYYMMDD（默认：20251231）"
    )
    parser.add_argument("--only-clean", action="store_true", help="仅构建clean层，不构建features")
    parser.add_argument(
        "--only-features", action="store_true", help="仅构建features层，不构建clean"
    )
    parser.add_argument("--force", action="store_true", help="强制重新构建，即使文件已存在")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用多进程并行构建（默认串行），利用所有CPU核心加速",
    )
    parser.add_argument(
        "--parallel-jobs", type=int, default=-1, help="并行 worker 数（默认 -1=全部核心）"
    )
    parser.add_argument(
        "--enable-industry-neutralization", action="store_true", help="启用行业中性特征构建"
    )
    parser.add_argument(
        "--enable-size-neutralization",
        action="store_true",
        help="启用市值中性化（在行业中性化基础上按市值分位再做Z-Score，生成zscore_*_sz列）",
    )
    parser.add_argument(
        "--min-list-days", type=int, default=365, help="最小上市自然日天数（默认：365，约12个月）"
    )
    horizon_group = parser.add_mutually_exclusive_group(required=True)
    horizon_group.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="单 horizon 模式：按此主标签 y_ret_N 非空过滤样本（如 --horizon 20）。"
        "仍生成 y_ret_5/10/20 三列标签，仅过滤时只看主 horizon",
    )
    horizon_group.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="多 horizon 模式：按 AND 过滤，要求给定 horizons 对应的所有标签同时非空"
        "（如 --horizons 5 10 20）",
    )
    parser.add_argument(
        "--build-all",
        action="store_true",
        help="启用全部可选因子（基本面、另类数据、融资融券、筹码胜率、基金持仓、业绩快报、北向资金、龙虎榜、一致预期；不含行业中性化）",
    )
    parser.add_argument(
        "--enable-fundamental-features",
        action="store_true",
        help="启用基本面因子（ROE、营收增速等），需先下载 fina_indicator 数据",
    )
    parser.add_argument(
        "--enable-alt-features",
        action="store_true",
        help="启用另类数据因子（股东人数、业绩预告等）",
    )
    parser.add_argument(
        "--enable-margin-features",
        action="store_true",
        help="启用融资融券因子（融资余额变动、融券/融资比、净买入比等）",
    )
    parser.add_argument(
        "--enable-cyq-features",
        action="store_true",
        help="启用筹码胜率因子（winner_rate、成本偏离等）",
    )
    parser.add_argument(
        "--enable-fund-features",
        action="store_true",
        help="启用基金持仓因子（持股比例、基金数量等）",
    )
    parser.add_argument(
        "--enable-express-features",
        action="store_true",
        help="启用业绩快报因子（实际营收/净利润增速等）",
    )
    parser.add_argument(
        "--enable-north-features",
        action="store_true",
        help="启用北向资金因子（moneyflow_hsgt 市场级广播）",
    )
    parser.add_argument(
        "--enable-lhb-features", action="store_true", help="启用龙虎榜因子（top_list 个股级）"
    )
    parser.add_argument(
        "--enable-consensus-features",
        action="store_true",
        help="启用一致预期因子（report_rc 研报滚动聚合）",
    )
    parser.add_argument(
        "--enable-cashflow-quality-features",
        action="store_true",
        help="启用现金流质量因子（需 cashflow 接口，2000 积分，需先下载 cashflow 数据）",
    )
    parser.add_argument(
        "--enable-consensus-revision-features",
        action="store_true",
        help="启用一致预期修正因子（基于已有 report_rc 构建时序修正信号，无需额外下载）",
    )
    parser.add_argument(
        "--enable-dividend-policy-features",
        action="store_true",
        help="启用分红政策质量因子（分红稳定性/增长率/支付率/双日期事件，需先下载 dividend 数据）",
    )
    parser.add_argument(
        "--enable-announcement-risk-features",
        action="store_true",
        help="启用风控公告类因子（质押/解禁/大宗，PIT 前向填充；需先下载 pledge_stat/share_float/block_trade）",
    )

    args = parser.parse_args()
    args = apply_build_all_feature_flags(args)

    # 初始化日志
    setup_logger(log_level="INFO")

    logger.info("=" * 60)
    logger.info("开始构建clean和features数据")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} - {args.end_date}")
    logger.info(f"仅构建clean: {'是' if args.only_clean else '否'}")
    logger.info(f"仅构建features: {'是' if args.only_features else '否'}")
    logger.info(f"强制重新构建: {'是' if args.force else '否'}")
    logger.info(f"全部可选因子: {'启用' if args.build_all else '禁用'}")
    logger.info(f"基本面因子: {'启用' if args.enable_fundamental_features else '禁用'}")
    logger.info(f"另类数据因子: {'启用' if args.enable_alt_features else '禁用'}")
    logger.info(f"融资融券因子: {'启用' if args.enable_margin_features else '禁用'}")
    logger.info(f"筹码胜率因子: {'启用' if args.enable_cyq_features else '禁用'}")
    logger.info(f"基金持仓因子: {'启用' if args.enable_fund_features else '禁用'}")
    logger.info(f"业绩快报因子: {'启用' if args.enable_express_features else '禁用'}")
    logger.info(f"北向资金因子: {'启用' if args.enable_north_features else '禁用'}")
    logger.info(f"龙虎榜因子: {'启用' if args.enable_lhb_features else '禁用'}")
    logger.info(f"一致预期因子: {'启用' if args.enable_consensus_features else '禁用'}")
    logger.info(f"现金流质量因子: {'启用' if args.enable_cashflow_quality_features else '禁用'}")
    logger.info(
        f"一致预期修正因子: {'启用' if args.enable_consensus_revision_features else '禁用'}"
    )
    logger.info(f"分红政策因子: {'启用' if args.enable_dividend_policy_features else '禁用'}")
    logger.info(f"风控公告类因子: {'启用' if args.enable_announcement_risk_features else '禁用'}")
    if args.horizon is not None:
        logger.info(f"标签过滤模式: single (主 horizon={args.horizon})")
    else:
        logger.info(f"标签过滤模式: all (horizons={args.horizons})")
    logger.info("=" * 60)

    try:
        # 初始化组件
        storage = Storage()
        loader = DataLoader(storage)
        shenwan_industry = loader.load_shenwan_industry()
        cleaner = DataCleaner()
        if args.horizon is not None:
            # 单值模式：生成全部标准标签列，仅按主 horizon 过滤
            builder = FeatureBuilder(
                min_list_days=args.min_list_days,
                horizon=args.horizon,
                horizons=[5, 10, 20],
                require_label=True,
                label_filter_mode="single",
            )
        else:
            # 多值模式：生成用户指定的标签列，AND 过滤
            builder = FeatureBuilder(
                min_list_days=args.min_list_days,
                horizons=args.horizons,
                require_label=True,
                label_filter_mode="all",
            )

        # 构建clean数据
        if not args.only_features:
            build_clean_data(
                storage,
                loader,
                cleaner,
                args.start_date,
                args.end_date,
                force=args.force,
                min_list_days=args.min_list_days,
            )

        # 构建features数据
        if not args.only_clean:
            build_features_data(
                storage,
                loader,
                builder,
                args.start_date,
                args.end_date,
                force=args.force,
                shenwan_industry=shenwan_industry if args.enable_industry_neutralization else None,
                apply_industry_neutralization=args.enable_industry_neutralization,
                apply_size_neutralization=args.enable_size_neutralization,
                enable_fundamental=args.enable_fundamental_features,
                enable_alt=args.enable_alt_features,
                enable_margin=args.enable_margin_features,
                enable_cyq=args.enable_cyq_features,
                enable_fund=args.enable_fund_features,
                enable_express=args.enable_express_features,
                enable_north=args.enable_north_features,
                enable_lhb=args.enable_lhb_features,
                enable_consensus=args.enable_consensus_features,
                enable_cashflow_quality=args.enable_cashflow_quality_features,
                enable_consensus_revision=args.enable_consensus_revision_features,
                enable_dividend_policy=args.enable_dividend_policy_features,
                enable_announcement_risk=args.enable_announcement_risk_features,
                use_parallel=args.parallel,
                parallel_jobs=args.parallel_jobs,
            )

        logger.info("=" * 60)
        logger.info("数据构建完成！")
        logger.info(f"clean数据位置: {storage.clean_path}")
        logger.info(f"features数据位置: {storage.features_path}")
        logger.info("=" * 60)

    except ValueError as e:
        logger.error("=" * 60)
        logger.error("数据构建失败")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("请先下载raw数据:")
        logger.error("  python scripts/download_raw.py")
        logger.error("=" * 60)
        sys.exit(1)

    except Exception as e:
        logger.exception(f"构建过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
