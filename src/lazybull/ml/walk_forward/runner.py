#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward 运行编排与执行。"""

import re
import sys
import traceback
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

from dateutil.relativedelta import relativedelta
from loguru import logger

from src.lazybull.backtest.exposure_override import load_exposure_table
from src.lazybull.common.config import get_data_root, get_stock_selection_models_root
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml import ModelRegistry
from src.lazybull.risk.terminal_loss.exposure_online import (
    DailyExposureProvider,
    OnlinePolicyConfig,
    log_exported_table_provenance,
)

from .backtest import run_oos_backtest
from .deploy_training import execute_deploy_training
from .reporting import (
    chain_nav_splits,
    write_walk_forward_holdings_snapshot,
    write_walk_forward_topk_details,
    write_walk_forward_trade_details,
)
from .split_training import execute_skip_training_evaluation, execute_split_training
from .summary import write_walk_forward_summary
from .training_core import _build_main_board_codes
from src.lazybull.universe.domains import domain_market_whitelist, resolve_stock_domain
from .utils import (
    WalkForwardSplit,
    generate_walk_forward_splits_by_count,
    print_splits_summary,
    resolve_deploy_train_window,
)


def write_policy_lambda_series(provider, summary_csv_path, wf_run_id: str) -> None:
    """落盘在线政策的逐日 λ 与统计（`policy_lambda_<run_id>.csv` / `.json`）。

    CSV 是**证据归档**（可回放、可对比），不参与决胜：引擎不读它。
    除 `date`/`multiplier` 外并入日级面板列（折号/市场波动/平均风险分等）供审计；
    未被判定的日期（无持仓/窗口不足）无面板行，只保留顺延后的 λ。
    """
    import json

    out_dir = Path(summary_csv_path).parent
    series = provider.lambda_series()
    if series.empty:
        logger.warning("在线政策：无判定日，跳过逐日 λ 落盘")
        return
    panel = provider.daily_series()
    if not panel.empty and "date" in panel.columns:
        series = series.merge(panel, on="date", how="left")
    csv_path = out_dir / f"policy_lambda_{wf_run_id}.csv"
    series.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = provider.stats_summary()
    summary["lambda_days"] = int(len(series))
    summary["lambda_trigger_days"] = int((series["multiplier"] < 1.0).sum())
    json_path = out_dir / f"policy_lambda_{wf_run_id}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"在线政策逐日 λ 已落盘: {csv_path.name}（{summary['lambda_days']} 日，"
        f"降暴露 {summary['lambda_trigger_days']} 日）"
    )


def build_online_policy_provider(args, data_root: str) -> DailyExposureProvider:
    """构建在线政策 provider（P2-5：λ_t 随持仓现算；不读任何预导出系数表）。"""
    policy_root = getattr(args, "policy_model_root", None)
    arm_suffix = getattr(args, "policy_arm_suffix", None)
    if not policy_root:
        raise ValueError("--exposure-policy 需要同时提供 --policy-model-root（折模型来源）")
    if not arm_suffix:
        raise ValueError(
            "--exposure-policy 需要同时提供 --policy-arm-suffix（折目录后缀，如 _d5_v6m_fscore）"
        )
    config = OnlinePolicyConfig.parse(args.exposure_policy)
    return DailyExposureProvider(
        config,
        risk_root=policy_root,
        arm_suffix=arm_suffix,
        data_root=str(data_root),
        coverage_start=getattr(args, "policy_coverage_start", None),
        book="pre_exec",
        verbose=True,
    )


warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")
# test 期延伸到数据末尾时，标签列（如 y_ret_20）在最近 N 个交易日全为 NaN，concat 时触发此警告
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*DataFrame concatenation with empty or all-NA entries.*",
)


def _load_skip_training_metadata(registry, model_version: int, args):
    """skip-training 复用旧模型时，核验一致预期修正/现金流质量开关与 schema 版本一致性。

    返回旧模型 metadata（失败时为 None），并从独立 features 文件补齐实际
    feature_columns，供 result 透传给汇总。开关不一致或旧模型未记录 schema
    版本时仅告警（模型仍可用），提示消融归因可能把开关语义与模型实际列混为一谈。
    """
    from src.lazybull.factors.cashflow_quality import CASHFLOW_QUALITY_SCHEMA_VERSION
    from src.lazybull.factors.consensus_revision import CONSENSUS_REVISION_SCHEMA_VERSION
    from src.lazybull.ml.train_core.constants import (
        read_cashflow_quality_schema_version,
        read_cons_revision_schema_version,
    )

    metadata = registry._load_metadata(model_version)
    if not metadata:
        logger.warning(f"[skip-training] 无法读取 v{model_version} 元数据，跳过开关一致性校验")
        return None
    train_params = metadata.get("train_params") or {}
    requested = bool(getattr(args, "enable_consensus_revision_features", False))
    recorded = bool(train_params.get("enable_consensus_revision_features", False))
    if requested != recorded:
        logger.warning(
            f"[skip-training] v{model_version} 训练时 enable_consensus_revision_features="
            f"{recorded}，当前 CLI 为 {requested}，开关与模型实际特征不一致，"
            "消融归因请以模型 metadata 的实际 feature_columns 为准"
        )
    if recorded:
        schema_version = read_cons_revision_schema_version(train_params)
        if schema_version != CONSENSUS_REVISION_SCHEMA_VERSION:
            logger.warning(
                f"[skip-training] v{model_version} 未记录 v2 修正 schema 版本"
                f"（记录值: {train_params.get('cons_revision_schema_version')}），"
                "若复用当前特征分区将静默读取 v2 语义列，存在 train/serve 语义偏差，"
                "建议停用或重训"
            )

    # 现金流质量开关与 schema 版本一致性校验（与一致预期修正同级）
    requested_cfq = bool(getattr(args, "enable_cashflow_quality_features", False))
    recorded_cfq = bool(train_params.get("enable_cashflow_quality_features", False))
    if requested_cfq != recorded_cfq:
        logger.warning(
            f"[skip-training] v{model_version} 训练时 enable_cashflow_quality_features="
            f"{recorded_cfq}，当前 CLI 为 {requested_cfq}，开关与模型实际特征不一致，"
            "消融归因请以模型 metadata 的实际 feature_columns 为准"
        )
    if recorded_cfq:
        recorded_cfq_schema = read_cashflow_quality_schema_version(train_params)
        if recorded_cfq_schema != CASHFLOW_QUALITY_SCHEMA_VERSION:
            logger.warning(
                f"[skip-training] v{model_version} 记录的现金流质量 schema 版本为 "
                f"{train_params.get('cashflow_quality_schema_version')}，当前构建管线为 "
                f"v{CASHFLOW_QUALITY_SCHEMA_VERSION}（依赖修订事件驱动 TTM + 确定性"
                "版本去重），复用该模型存在 train/serve 语义偏差，建议停用或重训"
            )

    # 分红政策开关与 schema 版本一致性校验（与现金流质量同级）
    requested_div = bool(getattr(args, "enable_dividend_policy_features", False))
    recorded_div = bool(train_params.get("enable_dividend_policy_features", False))
    if requested_div != recorded_div:
        logger.warning(
            f"[skip-training] v{model_version} 训练时 enable_dividend_policy_features="
            f"{recorded_div}，当前 CLI 为 {requested_div}，开关与模型实际特征不一致，"
            "消融归因请以模型 metadata 的实际 feature_columns 为准"
        )
    if recorded_div:
        from src.lazybull.factors.dividend import DIVIDEND_POLICY_SCHEMA_VERSION
        from src.lazybull.ml.train_core.constants import read_dividend_policy_schema_version

        recorded_div_schema = read_dividend_policy_schema_version(train_params)
        if recorded_div_schema != DIVIDEND_POLICY_SCHEMA_VERSION:
            logger.warning(
                f"[skip-training] v{model_version} 记录的分红政策 schema 版本为 "
                f"{train_params.get('dividend_policy_schema_version')}，当前构建管线为 "
                f"v{DIVIDEND_POLICY_SCHEMA_VERSION}，复用该模型存在 train/serve 语义偏差，"
                "建议停用或重训"
            )

    # 特征列保存在独立 features 文件（metadata 本身不含），补齐后供汇总透传
    features_file_name = metadata.get("features_file") or f"v{model_version}_features.json"
    features_file = registry.models_dir / features_file_name
    feature_columns: List[str] = []
    if features_file.exists():
        try:
            import json

            with open(features_file, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, list):
                feature_columns = [str(c) for c in loaded]
        except (OSError, ValueError) as exc:
            logger.warning(f"[skip-training] 读取 v{model_version} 特征列表失败: {exc}")
    else:
        logger.warning(f"[skip-training] v{model_version} 特征列表文件缺失: {features_file.name}")
    metadata["feature_columns"] = feature_columns
    return metadata


def _filter_splits_by_selected_indices(
    splits: List[WalkForwardSplit],
    selected_split_indices: List[int],
) -> List[WalkForwardSplit]:
    """按指定 split 下标过滤切分；为空时返回原列表。"""
    if not selected_split_indices:
        return splits

    existing_indices = {split.split_index for split in splits}
    missing_indices = [index for index in selected_split_indices if index not in existing_indices]
    if missing_indices:
        raise ValueError(f"selected_split_indices 包含不存在的 split 下标: {missing_indices}")

    selected_index_set = set(selected_split_indices)
    return [split for split in splits if split.split_index in selected_index_set]


def run_walk_forward(args) -> None:
    """执行 walk-forward 主流程。"""
    # 设置日志
    setup_logger()

    logger.info("=" * 80)
    logger.info("Walk-forward 滚动训练")
    logger.info("=" * 80)
    logger.info(f"切分数量: {args.split_count}")
    logger.info(f"最终日期: {args.final_date}")
    logger.info(f"训练窗口: {args.train_window_years} 年")
    logger.info(f"测试窗口: {args.test_window_months} 个月")
    logger.info(
        "指定 split: %s" % (args.selected_split_indices if args.selected_split_indices else "全部")
    )
    logger.info(f"标签列: {args.label_column}")
    logger.info(f"行业中性标签混合权重: {args.neutral_label_blend_weight:.2f}")
    logger.info(f"任务类型: {args.task}")
    logger.info(
        f"早停: rounds={args.early_stopping_rounds if args.early_stopping_rounds else '禁用'}, "
        f"metric={args.early_stopping_metric}, "
        f"min_best_iteration={getattr(args, 'min_best_iteration', 0) or '禁用'}"
    )
    logger.info(
        f"多种子筛选: top_ratio={args.ensemble_seed_keep_top_ratio:.0%}, "
        f"min_models={args.ensemble_seed_keep_min_models}"
    )
    if args.enable_enhanced_features:
        logger.info("因子增强: 启用（开盘强度、日内波动结构、委托不平衡）")
    # oos_backtest_months=0 表示自动对齐 test_window_months
    if args.oos_backtest_months <= 0:
        args.oos_backtest_months = args.test_window_months

    logger.info(f"OOS 回测: {'启用' if args.oos_backtest else '禁用'}")
    if args.oos_backtest:
        logger.info(f"  回测时长: {args.oos_backtest_months} 个月")
        logger.info(f"  持仓 Top N: {args.bt_top_n}")
        logger.info(f"  卖出时机: {args.bt_sell_timing}")
        logger.info(f"  调仓频率: {args.bt_rebalance_freq or '自动推断'}")
        logger.info(f"  排除 ST: {'是' if args.bt_exclude_st else '否'}")
        logger.info(f"  最少上市天数: {args.bt_min_list_days}")
        if args.bt_max_weight_per_stock is not None:
            logger.info(f"  单股最大权重: {args.bt_max_weight_per_stock:.2%}")
        if args.bt_max_per_industry is not None:
            logger.info(f"  单行业最大持仓数: {args.bt_max_per_industry}")
        logger.info(f"  止损: {'启用' if args.bt_stop_loss_enabled else '关闭'}")
        if args.bt_stop_loss_enabled:
            logger.info(
                f"    drawdown={args.bt_stop_loss_drawdown_pct}%, "
                f"consecutive_limit_down={args.bt_stop_loss_consecutive_limit_down}"
            )
        if args.stagger_tranches > 1:
            logger.info(f"  分批调仓: {args.stagger_tranches} 批")
    effective_data_root = args.data_root or get_data_root()
    logger.info(f"数据目录: {effective_data_root}")

    exposure_table = None
    if getattr(args, "exposure_table", None):
        exposure_table = load_exposure_table(Path(args.exposure_table))
        reduced = sum(1 for value in exposure_table.values() if value < 1.0)
        logger.info(
            f"  暴露覆盖（P2-3 shadow）: {args.exposure_table} → {len(exposure_table)} 日"
            f"（降暴露 {reduced} 日，系数集合={sorted(set(exposure_table.values()))}）"
        )
        # 冻结产物溯源：打印 meta，缺失时告警（表绑定持仓路径，改配置必须重导）
        log_exported_table_provenance(Path(args.exposure_table))
    if getattr(args, "exposure_policy", None) and exposure_table is not None:
        raise ValueError("--exposure-policy 与 --exposure-table 互斥（单变量直读，禁止叠加）")
    exposure_policy = None
    if getattr(args, "exposure_policy", None):
        exposure_policy = build_online_policy_provider(args, effective_data_root)
        logger.info(
            f"  暴露政策（P2-5 在线现算）: {exposure_policy.config.describe()}"
            f"（指纹 {exposure_policy.fingerprint}，模型根 {exposure_policy.risk_root}"
            f"{exposure_policy.arm_suffix}）"
        )

    try:
        # 初始化组件
        storage = Storage(root_path=args.data_root)
        loader = DataLoader(storage)
        registry = ModelRegistry(models_dir=get_stock_selection_models_root(args.data_root))

        # 加载股票基本信息（OOS 回测需要）
        stock_basic = None
        if args.oos_backtest:
            stock_basic = loader.load_clean_stock_basic()
            if stock_basic is None:
                stock_basic = loader.load_stock_basic()
            if stock_basic is None:
                logger.warning("无法加载股票基本信息，OOS 回测将被禁用")
                args.oos_backtest = False

        # 训练/评估统一使用域股票池（默认主板 = 生产现状），保证与交易口径一致
        if stock_basic is None:
            stock_basic = loader.load_clean_stock_basic()
        if stock_basic is None:
            stock_basic = loader.load_stock_basic()
        if stock_basic is None:
            raise ValueError("无法加载股票基本信息，无法执行主板过滤训练")
        stock_domain = str(getattr(args, "stock_domain", "main") or "main")
        domain_markets = domain_market_whitelist(stock_domain)
        main_board_codes = _build_main_board_codes(stock_basic, markets=tuple(domain_markets))
        logger.info(
            f"域股票池加载完成（--stock-domain {stock_domain}，markets={domain_markets}）: "
            f"{len(main_board_codes)} 只"
        )

        # 生成 walk-forward ID
        wf_run_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Walk-forward 运行ID: {wf_run_id}")

        # 1. 生成 walk-forward 切分
        trade_cal = loader.load_clean_trade_cal()
        if trade_cal is None:
            trade_cal = loader.load_trade_cal()

        # 推断调仓频率（与 run_oos_backtest 内逻辑保持一致）
        if args.bt_rebalance_freq is not None:
            _rebalance_freq = args.bt_rebalance_freq
        else:
            _match = re.search(r"(\d+)", args.label_column)
            _rebalance_freq = int(_match.group(1)) if _match else 20

        splits = generate_walk_forward_splits_by_count(
            trade_cal=trade_cal,
            split_count=args.split_count,
            final_date=args.final_date,
            train_window_years=args.train_window_years,
            test_window_months=args.test_window_months,
            rebalance_freq=_rebalance_freq,
        )
        generated_split_count = len(splits)

        splits = _filter_splits_by_selected_indices(
            splits=splits,
            selected_split_indices=args.selected_split_indices,
        )

        if len(splits) == 0:
            logger.error("未生成任何切分，请检查参数设置")
            sys.exit(1)

        if args.selected_split_indices:
            logger.info(
                f"按下标筛选 split: {args.selected_split_indices}，"
                f"保留 {len(splits)} / {generated_split_count} 个"
            )
        else:
            logger.info(f"未指定 split 下标，默认训练全部 {len(splits)} 个 split")

        # 兼容汇总与对比脚本：写入推导出的 WF 覆盖区间
        args.wf_start_date = splits[0].train_start
        args.wf_end_date = splits[-1].test_end
        logger.info(
            f"推导区间: {args.wf_start_date} 至 {args.wf_end_date} "
            f"（由 split_count={args.split_count}, final_date={args.final_date} 反推）"
        )

        # skip-training 模式参数校验
        skip_training = getattr(args, "skip_training", False)
        start_model_version = getattr(args, "start_model_version", None)
        if skip_training and start_model_version is None:
            logger.error("--skip-training 模式必须指定 --start-model-version")
            sys.exit(1)

        # 股东增减持因子：**运行时派生**（不写回 cs_train）→ 按 WF 覆盖区间构建查询表；
        # 折目录只做切片，全部折共用同一张表（查询表按自然日窗口，与请求区间无关）。
        args.holdertrade_lookup = None
        if getattr(args, "enable_holdertrade_features", False):
            from src.lazybull.factors.holdertrade import build_holdertrade_runtime_lookup

            args.holdertrade_lookup = build_holdertrade_runtime_lookup(
                loader, args.wf_start_date, args.wf_end_date
            )
            logger.info(
                f"股东增减持运行时查询表: {len(args.holdertrade_lookup)} 个交易日"
                f"（派生区间 {args.wf_start_date}~{args.wf_end_date}）"
            )

        # 股票回购因子：同一先例（运行时派生、全折共用一张查询表）
        args.repurchase_lookup = None
        if getattr(args, "enable_repurchase_features", False):
            from src.lazybull.factors.repurchase import build_repurchase_runtime_lookup

            args.repurchase_lookup = build_repurchase_runtime_lookup(
                loader, args.wf_start_date, args.wf_end_date
            )
            logger.info(
                f"股票回购运行时查询表: {len(args.repurchase_lookup)} 个交易日"
                f"（派生区间 {args.wf_start_date}~{args.wf_end_date}）"
            )

        # 十大流通股东因子：同一先例（运行时派生、全折共用同一份报告期面板）
        args.top10fh_panel = None
        if getattr(args, "enable_top10fh_features", False):
            from src.lazybull.factors.top10_floatholders import load_top10fh_panel

            args.top10fh_panel = load_top10fh_panel(loader)
            if args.top10fh_panel is None or len(args.top10fh_panel) == 0:
                raise ValueError(
                    "enable_top10fh_features=True 但 raw/top10_floatholders 为空。请先运行: "
                    "python scripts/download_raw.py --download top10_floatholders"
                )
            logger.info(
                f"十大流通股东报告期面板: {len(args.top10fh_panel):,} 行 / "
                f"{args.top10fh_panel['ts_code'].nunique():,} 只股票"
            )

        # 龙虎榜机构席位因子：同一先例（运行时派生、全折共用一张查询表）
        args.top_inst_lookup = None
        if getattr(args, "enable_top_inst_features", False):
            from src.lazybull.factors.top_inst import build_top_inst_runtime_lookup

            args.top_inst_lookup = build_top_inst_runtime_lookup(
                loader, args.wf_start_date, args.wf_end_date
            )
            if not args.top_inst_lookup:
                raise ValueError(
                    "enable_top_inst_features=True 但 raw/top_inst 为空。请确认 "
                    "data/raw/top_inst/ 年分区已下载（探索期下载器见 "
                    "data/reports/top_inst_20260923/）"
                )
            logger.info(
                f"龙虎榜机构席位运行时查询表: {len(args.top_inst_lookup)} 个交易日"
                f"（派生区间 {args.wf_start_date}~{args.wf_end_date}）"
            )

        deploy_train_start = None
        deploy_train_end_for_run = None
        if not args.no_deploy_train and not skip_training:
            deploy_train_start, deploy_train_end_for_run = resolve_deploy_train_window(
                trade_cal=trade_cal,
                deploy_train_end=args.final_date,
                train_window_years=args.train_window_years,
            )

            if deploy_train_start is None or deploy_train_end_for_run is None:
                logger.warning(
                    f"部署训练区间解析失败，无法在切分汇总中展示（目标train_end={args.final_date}）"
                )
                deploy_train_start = None
                deploy_train_end_for_run = None
            else:
                last_split = splits[-1]
                if (
                    deploy_train_start == last_split.train_start
                    and deploy_train_end_for_run == last_split.train_end
                ):
                    logger.error(
                        "部署训练区间与最后一个 split 的训练区间完全重叠，"
                        "请调整 split_count 或 final_date 后重试"
                    )
                    sys.exit(1)

                if deploy_train_end_for_run <= last_split.train_end:
                    logger.error(
                        f"部署训练结束日({deploy_train_end_for_run}) 不晚于"
                        f"最后一个 split 训练结束日({last_split.train_end})，"
                        "会造成训练区间冲突，请调整 final_date"
                    )
                    sys.exit(1)

        print_splits_summary(
            splits,
            deploy_train_start=deploy_train_start,
            deploy_train_end=deploy_train_end_for_run,
        )

        # 2. 执行每个 split 的训练
        results = []
        topk_values = sorted({20, 30, 100, 300})

        # 创建跨 split 持久化 signal（仅 OOS 回测时使用）
        # 作用：门控历史缓冲区在 split 间累积，百分位归一化/自校准阈值能够完成预热
        persistent_signal = None
        if args.oos_backtest:
            from src.lazybull.signals import MLSignal

            # 推理侧市值上下限来自域定义（universe/domains.py 单一来源；
            # main_small 放开下限至 25 亿，main_gem 维持 50 亿——市场白名单在 universe 侧放开）
            bt_domain = resolve_stock_domain(
                str(getattr(args, "stock_domain", "main") or "main")
            )
            persistent_signal = MLSignal(
                top_n=args.bt_top_n,
                model_version=None,  # 首次 split 时通过 update_model_version 设置
                models_dir=get_stock_selection_models_root(args.data_root),
                min_total_mv=float(bt_domain["min_total_mv_wan"]),  # type: ignore[arg-type]
                max_total_mv=float(bt_domain["max_total_mv_wan"]),  # type: ignore[arg-type]
                verbose=False,
                downside_penalty=float(getattr(args, "downside_penalty", 0.0) or 0.0),
                downside_penalty_column=str(
                    getattr(args, "downside_penalty_column", "downside_vol_20")
                ),
            )
            logger.info(
                f"持久化 MLSignal 已创建（域 {getattr(args, 'stock_domain', 'main')}，"
                f"市值区间 [{bt_domain['min_total_mv_wan']:.0f}, "
                f"{bt_domain['max_total_mv_wan']:.0f}] 万元），"
                f"将跨 {len(splits)} 个 split 复用"
            )

        for split in splits:
            try:
                if skip_training:
                    # 跳过训练，直接用预设版本号构造 result
                    model_version = start_model_version + split.split_index
                    skip_metadata = _load_skip_training_metadata(registry, model_version, args)
                    logger.info(
                        f"[跳过训练] Split {split.split_index}: "
                        f"使用已有模型 v{model_version}，"
                        f"测试区间 {split.test_start} ~ {split.test_end}"
                    )
                    if getattr(args, "skip_training_eval", False):
                        # A5/政策类实验：复用同一批折模型时仍需标准逐日明细与测试指标
                        result = execute_skip_training_evaluation(
                            split=split,
                            wf_run_id=wf_run_id,
                            storage=storage,
                            loader=loader,
                            registry=registry,
                            args=args,
                            main_board_codes=main_board_codes,
                            topk_values=topk_values,
                            model_version=model_version,
                        )
                    else:
                        result = {
                            "split_index": split.split_index,
                            "train_start": split.train_start,
                            "train_end": split.train_end,
                            "test_start": split.test_start,
                            "test_end": split.test_end,
                            "model_version": model_version,
                            "feature_columns": (skip_metadata or {}).get("feature_columns") or [],
                            "bt_metrics": {},
                        }
                else:
                    result = execute_split_training(
                        split=split,
                        wf_run_id=wf_run_id,
                        storage=storage,
                        loader=loader,
                        registry=registry,
                        args=args,
                        main_board_codes=main_board_codes,
                        topk_values=topk_values,
                        trade_cal=trade_cal,
                    )

                # OOS 回测（每个 split 训练后运行真实回测）
                if args.oos_backtest and result.get("model_version"):
                    try:
                        bt_start = split.test_start
                        bt_end_dt = datetime.strptime(bt_start, "%Y%m%d") + relativedelta(
                            months=args.oos_backtest_months
                        )
                        bt_end = bt_end_dt.strftime("%Y%m%d")
                        bt_metrics = run_oos_backtest(
                            model_version=result["model_version"],
                            bt_start=bt_start,
                            bt_end=bt_end,
                            storage=storage,
                            loader=loader,
                            trade_cal=trade_cal,
                            stock_basic=stock_basic,
                            label_column=args.label_column,
                            bt_top_n=args.bt_top_n,
                            bt_rebalance_freq=args.bt_rebalance_freq,
                            data_root=args.data_root,
                            persistent_signal=persistent_signal,
                            bt_exclude_st=args.bt_exclude_st,
                            bt_min_list_days=args.bt_min_list_days,
                            bt_sell_timing=args.bt_sell_timing,
                            bt_max_weight_per_stock=args.bt_max_weight_per_stock,
                            bt_max_per_industry=args.bt_max_per_industry,
                            bt_stop_loss_enabled=args.bt_stop_loss_enabled,
                            bt_stop_loss_drawdown_pct=args.bt_stop_loss_drawdown_pct,
                            bt_stop_loss_consecutive_limit_down=args.bt_stop_loss_consecutive_limit_down,
                            position_sizing=getattr(args, "position_sizing", "equal"),
                            kelly_vol_window=getattr(args, "kelly_vol_window", 60),
                            kelly_max_leverage=getattr(args, "kelly_max_leverage", 0.25),
                            stagger_tranches=args.stagger_tranches,
                            enable_early_rebalance_on_empty=args.enable_early_rebalance_on_empty,
                            initial_capital=args.bt_initial_capital,
                            split_num=split.split_index,
                            exposure_table=exposure_table,
                            exposure_policy=exposure_policy,
                            exposure_replenish=bool(getattr(args, "exposure_replenish", False)),
                            exposure_trim_tolerance=getattr(args, "exposure_trim_tolerance", None),
                            exposure_budget_discount_replenish=bool(
                                getattr(args, "exposure_budget_discount_replenish", False)
                            ),
                            stock_domain=stock_domain,
                            holdertrade_lookup=getattr(args, "holdertrade_lookup", None),
                            repurchase_lookup=getattr(args, "repurchase_lookup", None),
                            top10fh_panel=getattr(args, "top10fh_panel", None),
                            top_inst_lookup=getattr(args, "top_inst_lookup", None),
                        )
                        # 提取 nav_curve 用于串联，不写入 CSV
                        nav_curve = bt_metrics.pop("_nav_curve", None)
                        if nav_curve is not None:
                            result["_nav_curve"] = nav_curve
                        trades = bt_metrics.pop("_trades", None)
                        if trades is not None:
                            result["_trades"] = trades
                        execution_attribution = bt_metrics.pop("_execution_attribution", None)
                        if execution_attribution is not None:
                            result["_execution_attribution"] = execution_attribution
                        holdings_snapshot = bt_metrics.pop("_holdings_snapshot", None)
                        if holdings_snapshot is not None:
                            result["_holdings_snapshot"] = holdings_snapshot
                        result["bt_metrics"] = bt_metrics
                    except Exception as e:
                        logger.error(f"Split {split.split_index} OOS回测失败: {e}")
                        logger.error(traceback.format_exc())
                        result["bt_metrics"] = {}

                results.append(result)
            except Exception as e:
                logger.error(f"Split {split.split_index} 训练失败: {e}")
                logger.error(traceback.format_exc())
                logger.warning("继续执行下一个 split...")
                continue

        # 3. 部署模型训练（使用最新可用数据）
        if not args.no_deploy_train and not skip_training and len(results) > 0:
            if deploy_train_end_for_run is None:
                logger.error("部署训练区间未成功解析，跳过部署模型训练")
                deploy_train_end = None
            else:
                deploy_train_end = deploy_train_end_for_run
            logger.info("=" * 80)
            logger.info("开始部署模型训练（使用最新可用数据）")
            logger.info(f"  部署模型 train_end: {deploy_train_end}（由 final_date 对齐）")
            logger.info("=" * 80)
            if deploy_train_end is not None:
                try:
                    deploy_result = execute_deploy_training(
                        deploy_train_end=deploy_train_end,
                        wf_run_id=wf_run_id,
                        storage=storage,
                        loader=loader,
                        registry=registry,
                        args=args,
                        main_board_codes=main_board_codes,
                        topk_values=topk_values,
                        trade_cal=trade_cal,
                    )
                    if deploy_result:
                        logger.info(f"部署模型已注册: v{deploy_result['model_version']}")
                except Exception as e:
                    logger.error(f"部署模型训练失败: {e}")
                    logger.error(traceback.format_exc())

        # 4. 生成 walk-forward 汇总文件（统一输出到 raw/ 子目录）
        if len(results) > 0:
            if args.wf_summary_csv:
                summary_csv_path = args.wf_summary_csv
            else:
                summary_csv_path = str(
                    Path(args.data_root or get_data_root())
                    / "walk_forward"
                    / "raw"
                    / f"walk_forward_summary_{wf_run_id}.csv"
                )

            write_walk_forward_summary(results, summary_csv_path, args, wf_run_id)

            if getattr(args, "export_topk_details", True):
                write_walk_forward_topk_details(results, summary_csv_path, wf_run_id)

            write_walk_forward_trade_details(results, summary_csv_path, wf_run_id)

            # 政策旁路 P2-1：逐日持仓快照（中文表头，供 terminal_loss 政策层离线打分）
            write_walk_forward_holdings_snapshot(results, summary_csv_path, wf_run_id)

            # ── 串联各 split 的 OOS 回测净值曲线 ──────────────────
            chain_nav_splits(results, summary_csv_path, wf_run_id)

            # 在线政策（P2-5）：逐日 λ 与统计落盘（审计与口径比对）
            if exposure_policy is not None:
                write_policy_lambda_series(exposure_policy, summary_csv_path, wf_run_id)
        else:
            logger.warning("没有成功完成的训练，跳过生成汇总文件")

        logger.info("=" * 80)
        logger.info("Walk-forward 滚动训练完成！")
        logger.info(f"  成功完成: {len(results)} / {len(splits)} 个切分")
        logger.info(f"  运行ID: {wf_run_id}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Walk-forward 训练失败: {e}")
        traceback.print_exc()
        sys.exit(1)
