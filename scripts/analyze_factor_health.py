#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""选股因子体检脚本（薄入口）。

对生产模型入模的特征清单做一次体检：
  1. 采样 cs_train 特征分区，逐日截面 RankIC 与覆盖率；
  2. 跨年符号稳定性（异号年数 / 相邻翻转）与高/低波动分层；
  3. 平均截面相关矩阵聚类（识别孪生与冗余簇）；
  4. 既有模型（含 EnsembleModel）的因子使用度（gain 份额 / 分裂使用率）；
  5. 输出台账、候选清单与 markdown 报告，并生成可直接用于
     `--factor-prune --factor-exclude-file` 的实验排除清单。

使用示例：
    python scripts/analyze_factor_health.py
    python scripts/analyze_factor_health.py --start 20200101 --end 20260702 --every 3
    python scripts/analyze_factor_health.py --feature-file <特征清单.json>
    python scripts/analyze_factor_health.py --skip-models --out temp/factor_health

本文件为薄入口：CLI 参数解析 + 编排 scripts/factor_health/ 子包。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from scripts.factor_health import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MARKET_VOL_COLUMN,
    HealthThresholds,
    add_year_profiles,
    assemble_register,
    attach_clusters,
    attach_families,
    attach_twin_info,
    build_exclude_lists,
    build_report_markdown,
    candidate_tables,
    cluster_features,
    compute_model_usage,
    flag_candidates,
    list_available_model_versions,
    parse_version_spec,
    pick_corr_dates,
    pick_partition_files,
    resolve_latest_feature_file,
    scan_features,
    summarize,
    write_health_outputs,
)
from src.lazybull.common.config import (
    get_data_root,
    get_logs_dir,
    get_stock_selection_models_root,
)
from src.lazybull.common.logger import setup_logger


def _load_features(feature_file: Path) -> list:
    """读取特征清单 JSON（支持 list 或含 features 键的 dict）。"""
    payload = json.loads(Path(feature_file).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("features") or list(payload.keys())
    features = [str(name) for name in payload]
    if not features:
        raise ValueError(f"特征清单为空: {feature_file}")
    return features


def _resolve_mainboard_codes() -> set:
    """加载主板股票代码集合（与生产训练域一致）。"""
    from src.lazybull.data import DataLoader

    loader = DataLoader()
    stock_basic = loader.load_clean_stock_basic()
    if stock_basic is None:
        stock_basic = loader.load_stock_basic()
    if stock_basic is None:
        raise FileNotFoundError("无法加载 stock_basic，无法构建主板股票池")
    mask = stock_basic["market"].astype(str).str.contains("主板")
    return set(stock_basic.loc[mask, "ts_code"].astype(str))


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description="选股因子体检（覆盖率/IC 稳定性/聚类/使用度）")
    parser.add_argument("--start", default="20200101", help="起始日期 YYYYMMDD，默认 20200101")
    parser.add_argument("--end", default="99999999", help="结束日期 YYYYMMDD，默认取全部分区")
    parser.add_argument("--every", type=int, default=3, help="采样间隔（交易日），默认 3")
    parser.add_argument(
        "--out", default="", help="输出目录，默认 data/reports/factor_health/<时间戳>"
    )
    parser.add_argument("--feature-file", default="", help="特征清单 JSON，默认取最新注册模型")
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN, help="标签列")
    parser.add_argument("--market-vol-column", default=DEFAULT_MARKET_VOL_COLUMN, help="市场波动列")
    parser.add_argument("--skip-models", action="store_true", help="跳过模型使用度统计")
    parser.add_argument(
        "--model-versions", default="", help="模型版本规格（如 24008-24021,24052），默认取最新 N 个"
    )
    parser.add_argument(
        "--usage-model-count", type=int, default=15, help="默认统计的最近模型版本数"
    )
    parser.add_argument("--no-mainboard-filter", action="store_true", help="关闭主板股票池过滤")
    parser.add_argument("--progress-every", type=int, default=40, help="扫描进度日志间隔（文件数）")
    parser.add_argument("--min-pairs", type=int, default=200, help="单日截面计算 IC 的最少配对样本")
    parser.add_argument("--corr-min-periods", type=int, default=100, help="相关矩阵最少配对样本")
    parser.add_argument("--cluster-abs-corr", type=float, default=0.85, help="聚类合并阈值 |rho|")
    parser.add_argument("--low-coverage", type=float, default=0.6, help="低覆盖阈值（全期）")
    parser.add_argument("--low-year-coverage", type=float, default=0.4, help="低覆盖阈值（单年）")
    parser.add_argument("--weak-abs-t", type=float, default=1.5, help="弱信息 |t| 阈值")
    parser.add_argument("--weak-gain-share", type=float, default=0.005, help="弱信息 gain 份额阈值")
    parser.add_argument(
        "--unused-split-use", type=float, default=0.5, help="未使用判定：分裂使用率阈值"
    )
    parser.add_argument(
        "--unused-gain-share", type=float, default=0.002, help="未使用判定：gain 份额阈值"
    )
    parser.add_argument(
        "--flip-abs-year-ic", type=float, default=0.01, help="翻号统计的最少年 |IC|"
    )
    parser.add_argument("--flip-years-opposite", type=int, default=3, help="翻号候选：异号年数阈值")
    parser.add_argument("--flip-year-count", type=int, default=2, help="翻号候选：相邻翻转次数阈值")
    return parser


def main() -> None:
    """编排体检全流程。"""
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(get_logs_dir()) / f"factor_health_{timestamp}.log"
    setup_logger(log_level="INFO", log_file=str(log_file))

    data_root = Path(get_data_root())
    cs_train_dir = data_root / "features" / "cs_train"
    model_dir = Path(get_stock_selection_models_root(str(data_root)))

    feature_file = (
        Path(args.feature_file) if args.feature_file else resolve_latest_feature_file(model_dir)
    )
    features = _load_features(feature_file)
    logger.info(f"特征清单: {feature_file}（{len(features)} 个特征）")

    out_dir = Path(args.out) if args.out else data_root / "reports" / "factor_health" / timestamp

    files = pick_partition_files(cs_train_dir, args.start, args.end, args.every)
    years = sorted({path.stem[:4] for path in files})
    corr_dates = pick_corr_dates(files, [int(year) for year in years])
    logger.info(
        f"分区 {len(files)} 个（{files[0].stem} ~ {files[-1].stem}），相关性采样日 {len(corr_dates)} 个"
    )

    code_filter = None
    if not args.no_mainboard_filter:
        code_filter = _resolve_mainboard_codes()
        logger.info(f"主板股票池: {len(code_filter)} 只")

    scan_result = scan_features(
        files=files,
        feature_names=features,
        label_column=args.label_column,
        market_vol_column=args.market_vol_column,
        corr_dates=corr_dates,
        min_pairs=args.min_pairs,
        corr_min_periods=args.corr_min_periods,
        code_filter=code_filter,
        progress_every=args.progress_every,
    )
    logger.info(f"逐日 IC 记录 {len(scan_result.daily_ic)} 行")

    thresholds = HealthThresholds(
        low_coverage=args.low_coverage,
        low_year_coverage=args.low_year_coverage,
        weak_abs_t=args.weak_abs_t,
        weak_gain_share=args.weak_gain_share,
        unused_split_use=args.unused_split_use,
        unused_gain_share=args.unused_gain_share,
        flip_min_abs_year_ic=args.flip_abs_year_ic,
        flip_years_opposite=args.flip_years_opposite,
        flip_year_count=args.flip_year_count,
        cluster_abs_corr=args.cluster_abs_corr,
        min_pairs=args.min_pairs,
        corr_min_periods=args.corr_min_periods,
    )

    register = assemble_register(
        scan_result.coverage_by_date, scan_result.daily_ic, scan_result.market_vol_by_date
    )
    register = add_year_profiles(register, thresholds.flip_min_abs_year_ic)
    register = attach_twin_info(register)
    register = attach_families(register)

    clusters = None
    if scan_result.corr_avg is not None:
        clusters = cluster_features(
            scan_result.corr_avg, register["ic_ir"], thresholds.cluster_abs_corr
        )
        logger.info(
            f"相关性聚类: {len(clusters)} 簇，多成员簇 {int((clusters['size'] > 1).sum())} 个"
        )
    if clusters is None:
        clusters = pd.DataFrame()
    register = attach_clusters(register, clusters)

    model_versions = []
    sub_model_count = 0
    if not args.skip_models:
        if args.model_versions:
            versions = parse_version_spec(args.model_versions)
        else:
            available = list_available_model_versions(model_dir)
            versions = available[-max(args.usage_model_count, 1) :]
        usage = compute_model_usage(model_dir, versions, features)
        sub_model_count = int(usage.attrs.get("sub_model_count", 0))
        model_versions = list(usage.attrs.get("loaded_versions", []))
        register = register.join(usage)
    for column in ("gain_share_mean", "split_use_frac"):
        if column not in register.columns:
            register[column] = float("nan")

    register = flag_candidates(register, thresholds)
    candidates = candidate_tables(register)
    summary = summarize(register, clusters)

    report_context = {
        "feature_file": str(feature_file),
        "start": files[0].stem,
        "end": files[-1].stem,
        "every": args.every,
        "file_count": len(files),
        "label": args.label_column,
        "code_count": len(code_filter) if code_filter else "全部",
        "mainboard_filter": code_filter is not None,
        "model_versions": model_versions,
        "sub_model_count": sub_model_count,
        "out_dir": str(out_dir),
    }
    report_markdown = build_report_markdown(
        register, clusters, thresholds, report_context, candidates
    )
    exclude_lists = build_exclude_lists(register)
    write_health_outputs(
        out_dir,
        register,
        scan_result.daily_ic,
        scan_result.corr_avg,
        clusters,
        candidates,
        report_markdown,
        exclude_lists,
    )

    logger.info(
        f"体检完成: 特征 {summary['feature_count']}，市场级 {summary['market_level']}，"
        f"低覆盖 {summary['low_coverage']}，翻号 {summary['sign_flip']}，弱 {summary['weak']}，"
        f"未用 {summary['unused']}，去重 {summary['dedup']}"
    )
    logger.info(f"产物目录: {out_dir}")
    for name, table in candidates.items():
        logger.info(f"  候选 {name}: {len(table)} 个")
    for name, payload in exclude_lists.items():
        logger.info(f"  排除清单 {name}.json: {len(payload['factors'])} 个因子")


if __name__ == "__main__":
    main()
