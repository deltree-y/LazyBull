# -*- coding: utf-8 -*-
"""features 层批量流水线：从 scripts/build_clean_features.py 下沉。

提供 build_features_data（含可选并行路径 _build_features_parallel），
负责加载 clean 基础数据与可选因子查询表，逐日构建并保存 cs_train 特征分区。
"""

import time
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from ..data import DataLoader, Storage
from . import FeatureBuilder
from .ensure import _check_features_schema


def _build_features_parallel(
    pending_dates,
    builder,
    storage,
    trade_cal,
    daily_clean,
    adj_factor,
    stock_basic,
    daily_basic_clean,
    moneyflow_clean,
    shenwan_industry,
    apply_industry_neutralization,
    apply_size_neutralization,
    fundamental_lookup,
    margin_lookup,
    holder_lookup,
    earnings_lookup,
    cyq_perf_lookup,
    fund_portfolio_lookup,
    express_lookup,
    north_flow_lookup,
    lhb_lookup,
    consensus_lookup,
    cashflow_lookup,
    consensus_revision_lookup,
    pledge_lookup,
    share_float_lookup,
    block_trade_lookup,
    n_jobs,
):
    """并行构建特征：预构建 FeatureContext 列表后调用 parallel 模块。"""
    from joblib import Parallel, delayed

    from src.lazybull.features.context import FeatureContext
    from src.lazybull.features.factor_handlers import create_factor_registry
    from src.lazybull.features.parallel import build_features_for_day_static

    # 共享只读缓存（通过 loky copy-on-write 共享，不深拷贝）
    daily_adj_dict = builder._daily_adj_dict
    tech_cache_dict = builder._tech_factor_cache_dict
    market_state_cache = builder._market_state_cache
    trading_dates_list = builder._trading_dates_cache or []
    trading_date_index = builder._trading_date_index or {}
    daily_adj_precomputed = builder._daily_adj_precomputed
    factor_registry = create_factor_registry()

    # 风控因子批量预计算（主进程一次性完成，worker 仅查表）
    risk_cache_dict = None
    risk_names = None
    if daily_adj_precomputed is not None:
        risk_cache_dict = builder._get_risk_factor_cache(daily_adj_precomputed)
        risk_names = builder._risk_factor_names if risk_cache_dict is not None else None

    # 预构建所有 FeatureContext（在主进程中完成，避免 pickle 复杂对象）
    ctx_list = []
    for trade_date in pending_dates:
        funda_today = fundamental_lookup.get(trade_date) if fundamental_lookup else None
        margin_today = margin_lookup.get(trade_date) if margin_lookup else None
        holder_today = holder_lookup.get(trade_date) if holder_lookup else None
        earnings_today = earnings_lookup.get(trade_date) if earnings_lookup else None
        cyq_perf_today = cyq_perf_lookup.get(trade_date) if cyq_perf_lookup else None
        fund_portfolio_today = (
            fund_portfolio_lookup.get(trade_date) if fund_portfolio_lookup else None
        )
        express_today = express_lookup.get(trade_date) if express_lookup else None
        if north_flow_lookup is not None:
            north_flow_today = north_flow_lookup.get(trade_date) or {}
        else:
            north_flow_today = None
        if lhb_lookup is not None:
            lhb_today = lhb_lookup.get(trade_date)
            if lhb_today is None:
                lhb_today = pd.DataFrame()
        else:
            lhb_today = None
        if consensus_lookup is not None:
            consensus_today = consensus_lookup.get(trade_date)
            if consensus_today is None:
                consensus_today = pd.DataFrame()
        else:
            consensus_today = None
        if cashflow_lookup is not None:
            cashflow_today = cashflow_lookup.get(trade_date)
            if cashflow_today is None:
                cashflow_today = pd.DataFrame()
        else:
            cashflow_today = None
        if consensus_revision_lookup is not None:
            consensus_revision_today = consensus_revision_lookup.get(trade_date)
            if consensus_revision_today is None:
                consensus_revision_today = pd.DataFrame()
        else:
            consensus_revision_today = None
        if pledge_lookup is not None:
            pledge_today = pledge_lookup.get(trade_date)
            if pledge_today is None:
                pledge_today = pd.DataFrame()
        else:
            pledge_today = None
        if share_float_lookup is not None:
            share_float_today = share_float_lookup.get(trade_date)
            if share_float_today is None:
                share_float_today = pd.DataFrame()
        else:
            share_float_today = None
        if block_trade_lookup is not None:
            block_trade_today = block_trade_lookup.get(trade_date)
            if block_trade_today is None:
                block_trade_today = pd.DataFrame()
        else:
            block_trade_today = None

        ctx = FeatureContext(
            trade_date=trade_date,
            trade_cal=trade_cal,
            daily_data=daily_clean,
            adj_factor=adj_factor,
            stock_basic=stock_basic,
            daily_basic_data=daily_basic_clean,
            moneyflow_data=moneyflow_clean,
            shenwan_industry=shenwan_industry,
            apply_industry_neutralization=apply_industry_neutralization,
            apply_size_neutralization=apply_size_neutralization,
            fundamental_data=funda_today,
            margin_data=margin_today,
            holder_data=holder_today,
            earnings_data=earnings_today,
            cyq_perf_data=cyq_perf_today,
            express_data=express_today,
            fund_portfolio_data=fund_portfolio_today,
            north_flow_data=north_flow_today,
            lhb_data=lhb_today,
            consensus_data=consensus_today,
            cashflow_data=cashflow_today,
            consensus_revision_data=consensus_revision_today,
            pledge_data=pledge_today,
            share_float_data=share_float_today,
            block_trade_data=block_trade_today,
            horizons=builder.horizons,
            horizon=builder.horizon,
            lookback_windows=builder.lookback_windows,
            require_label=builder.require_label,
            label_filter_mode=builder.label_filter_mode,
            min_list_days=builder.min_list_days,
            shenwan_level=builder.shenwan_level,
            verbose=False,
        )
        ctx_list.append(ctx)

    total = len(ctx_list)
    logger.info(f"启动并行特征构建: {total} 天, workers={n_jobs}")

    def _process_one(ctx):
        try:
            df = build_features_for_day_static(
                trade_date=ctx.trade_date,
                ctx=ctx,
                daily_adj_dict=daily_adj_dict,
                tech_factor_cache_dict=tech_cache_dict,
                market_state_cache=market_state_cache,
                trading_dates_list=trading_dates_list,
                trading_date_index=trading_date_index,
                daily_adj_precomputed=daily_adj_precomputed,
                factor_registry=factor_registry,
                risk_factor_cache_dict=risk_cache_dict,
                risk_factor_names=risk_names,
            )
            return ctx.trade_date, df, None
        except Exception as e:
            return ctx.trade_date, None, str(e)

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_process_one)(ctx) for ctx in ctx_list
    )

    success = 0
    errors = 0
    for trade_date, df, err in results:
        if err is not None:
            logger.error(f"  {trade_date} 并行构建失败: {err}")
            errors += 1
            continue
        if df is not None and len(df) > 0:
            storage.save_cs_train_day(df, trade_date)
            success += 1

    logger.info("=" * 60)
    logger.info("features层数据构建完成（并行）")
    logger.info("=" * 60)
    logger.info(f"成功: {success} 个交易日")
    logger.info(f"失败: {errors} 个交易日")


def build_features_data(
    storage: Storage,
    loader: DataLoader,
    builder: FeatureBuilder,
    start_date: str,
    end_date: str,
    force: bool = False,
    shenwan_industry: pd.DataFrame = None,
    apply_industry_neutralization: bool = False,
    apply_size_neutralization: bool = False,
    enable_fundamental: bool = False,
    enable_alt: bool = False,
    enable_margin: bool = False,
    enable_cyq: bool = False,
    enable_fund: bool = False,
    enable_express: bool = False,
    enable_north: bool = False,
    enable_lhb: bool = False,
    enable_consensus: bool = False,
    enable_cashflow_quality: bool = False,
    enable_consensus_revision: bool = False,
    enable_announcement_risk: bool = False,
    use_parallel: bool = False,
    parallel_jobs: int = -1,
) -> None:
    """构建features层数据

    Args:
        storage: Storage实例
        loader: DataLoader实例
        builder: FeatureBuilder实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新构建
        shenwan_industry: 申万行业数据（可选，启用中性化时必需）
        apply_industry_neutralization: 是否应用行业中性化
        enable_fundamental: 是否启用基本面因子
        enable_alt: 是否启用另类数据因子
        enable_margin: 是否启用融资融券因子
        enable_cyq: 是否启用筹码胜率因子
        enable_fund: 是否启用基金持仓因子
        enable_express: 是否启用业绩快报因子
        enable_north: 是否启用北向资金因子
        enable_lhb: 是否启用龙虎榜因子
        enable_consensus: 是否启用一致预期因子
        enable_cashflow_quality: 是否启用现金流质量因子
        enable_consensus_revision: 是否启用一致预期修正因子
        enable_announcement_risk: 是否启用风控公告类因子（质押/解禁/大宗）
    """
    logger.info("=" * 60)
    logger.info("开始构建features层数据")
    logger.info("=" * 60)

    # 加载基础数据（从clean层）
    logger.info("加载基础数据...")
    trade_cal = loader.load_clean_trade_cal()
    stock_basic = loader.load_clean_stock_basic()

    if trade_cal is None:
        raise ValueError("缺少clean层trade_cal数据")
    if stock_basic is None:
        raise ValueError("缺少clean层stock_basic数据")

    # 转换日期格式
    if "cal_date" in trade_cal.columns:
        if not pd.api.types.is_datetime64_any_dtype(trade_cal["cal_date"]):
            trade_cal["cal_date"] = pd.to_datetime(trade_cal["cal_date"], format="%Y%m%d")

    # 获取交易日列表
    trading_dates = loader.get_trading_dates(
        start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:8],
        end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:8],
    )

    if len(trading_dates) == 0:
        raise ValueError(f"指定日期范围内没有交易日: {start_date} - {end_date}")

    # 转换为YYYYMMDD格式
    trading_dates_str = [
        d.strftime("%Y%m%d") if isinstance(d, pd.Timestamp) else d for d in trading_dates
    ]

    logger.info(f"共 {len(trading_dates_str)} 个交易日需要构建特征")

    # 加载clean层日线数据（扩展范围以包含足够的 warmup 历史，覆盖 120 个交易日 ≈ 7 个月）
    start_dt = pd.to_datetime(start_date, format="%Y%m%d") - pd.DateOffset(months=7)
    end_dt = pd.to_datetime(end_date, format="%Y%m%d") + pd.DateOffset(months=1)

    daily_clean = loader.load_clean_daily(start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))

    if daily_clean is None:
        raise ValueError("缺少clean层daily数据")

    logger.info(f"clean日线数据: {len(daily_clean)} 条记录")

    daily_close_lookup = {
        trade_date: grp[["ts_code", "close_adj"]].copy()
        for trade_date, grp in daily_clean.groupby("trade_date", sort=False)
        if "close_adj" in grp.columns
    }

    # 加载 daily_basic 数据
    daily_basic_clean = loader.load_clean_daily_basic(
        start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
    )
    if daily_basic_clean is not None:
        logger.info(f"clean daily_basic 数据: {len(daily_basic_clean)} 条记录")
    else:
        logger.warning("未找到 daily_basic 数据，价值红利特征将为空")

    # 加载 moneyflow 数据
    moneyflow_clean = loader.load_clean_moneyflow(
        start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
    )
    if moneyflow_clean is not None:
        logger.info(f"clean moneyflow 数据: {len(moneyflow_clean)} 条记录")
    else:
        logger.warning("未找到 moneyflow 数据（强制依赖项），资金流特征将为空")

    # 加载基本面数据（可选）
    fundamental_lookup = None
    if enable_fundamental:
        from src.lazybull.factors.fundamental import build_fundamental_lookup_by_date

        fina_indicator = loader.load_fina_indicator(start_date, end_date)
        if fina_indicator is not None:
            logger.info("构建基本面日频查询表...")
            fundamental_lookup = build_fundamental_lookup_by_date(fina_indicator, trading_dates_str)
        else:
            logger.warning(
                "未找到财务指标数据，基本面特征将被跳过。"
                "请先运行: python scripts/download_raw.py --download fina_indicator"
            )

    # 加载融资融券数据（可选，独立开关）
    margin_lookup = None
    if enable_margin:
        from src.lazybull.factors.margin import build_margin_lookup_by_date

        margin_detail = loader.load_margin_detail(
            start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
        )
        if margin_detail is not None:
            logger.info(f"融资融券数据: {len(margin_detail)} 条")
            margin_lookup = build_margin_lookup_by_date(margin_detail, trading_dates_str)
        else:
            logger.warning("未找到融资融券数据，相关特征将为空")

    # 加载另类数据（可选）
    holder_lookup = None
    earnings_lookup = None
    if enable_alt:
        from src.lazybull.factors.earnings import build_earnings_lookup_by_date
        from src.lazybull.factors.holder import build_holder_lookup_by_date

        # 股东人数
        stk_holdernumber = loader.load_stk_holdernumber()
        if stk_holdernumber is not None:
            logger.info(f"股东人数数据: {len(stk_holdernumber)} 条")
            holder_lookup = build_holder_lookup_by_date(stk_holdernumber, trading_dates_str)
        else:
            logger.warning("未找到股东人数数据，相关特征将为空")

        # 业绩预告
        forecast_df = loader.load_forecast()
        if forecast_df is not None:
            logger.info(f"业绩预告: {len(forecast_df)} 条")
            earnings_lookup = build_earnings_lookup_by_date(forecast_df, trading_dates_str)
        else:
            logger.warning("未找到业绩预告数据，相关特征将为空")

    # 加载筹码胜率数据（可选，按日分区存储，回溯加载以确保 diff(5)/diff(20) 有足够历史）
    cyq_perf_lookup = None
    if enable_cyq:
        from src.lazybull.factors.cyq_perf import build_cyq_perf_lookup_by_date

        cyq_perf_df = loader.load_cyq_perf(start_dt.strftime("%Y%m%d"), end_date)
        if cyq_perf_df is not None:
            logger.info(f"筹码胜率数据: {len(cyq_perf_df)} 条")
            cyq_perf_lookup = build_cyq_perf_lookup_by_date(cyq_perf_df, trading_dates_str)
        else:
            logger.warning(
                "未找到筹码胜率数据，相关特征将为空。"
                "请先运行: python scripts/download_raw.py --download cyq_perf"
            )

    # 加载基金持仓数据（可选，按季度分区存储，回溯加载以确保 point-in-time 查询覆盖前序季度）
    fund_portfolio_lookup = None
    if enable_fund:
        from src.lazybull.factors.fund_portfolio import build_fund_portfolio_lookup_by_date

        fund_portfolio_df = loader.load_fund_portfolio(start_dt.strftime("%Y%m%d"), end_date)
        if fund_portfolio_df is not None:
            logger.info(f"基金持仓数据: {len(fund_portfolio_df)} 条")
            fund_portfolio_lookup = build_fund_portfolio_lookup_by_date(
                fund_portfolio_df, trading_dates_str
            )
        else:
            logger.warning(
                "未找到基金持仓数据，相关特征将为空。"
                "请先运行: python scripts/download_raw.py --download fund_portfolio"
            )

    # 加载业绩快报数据（可选）
    express_lookup = None
    if enable_express:
        from src.lazybull.factors.express import build_express_lookup_by_date

        express_df = loader.load_express()
        if express_df is not None:
            logger.info(f"业绩快报数据: {len(express_df)} 条")
            # 加载业绩预告数据用于计算业绩惊喜
            forecast_df = loader.load_forecast()
            express_lookup = build_express_lookup_by_date(
                express_df, trading_dates_str, forecast_df=forecast_df
            )
        else:
            logger.warning(
                "未找到业绩快报数据，相关特征将为空。"
                "请先运行: python scripts/download_raw.py --download express"
            )

    # 加载北向资金数据（可选，市场级广播）
    north_flow_lookup = None
    if enable_north:
        from src.lazybull.factors.north_flow import build_north_flow_lookup_by_date

        hsgt_df = loader.load_moneyflow_hsgt(start_dt.strftime("%Y%m%d"), end_date)
        if hsgt_df is not None:
            logger.info(f"北向资金数据: {len(hsgt_df)} 条")
            north_flow_lookup = build_north_flow_lookup_by_date(hsgt_df, trading_dates_str)
        else:
            logger.warning("未找到北向资金数据，相关特征将为空")

    # 加载龙虎榜数据（可选）
    lhb_lookup = None
    if enable_lhb:
        from src.lazybull.factors.lhb import build_lhb_lookup_by_date

        top_list_df = loader.load_top_list(start_dt.strftime("%Y%m%d"), end_date)
        if top_list_df is not None:
            logger.info(f"龙虎榜数据: {len(top_list_df)} 条")
            # 滚动窗口需要包含预热期的完整交易日历, 否则区间前 19 个交易日
            # 的历史累计被 reindex 丢弃, 造成批量构建与单日推理不一致
            warm_cal = loader.get_trading_dates(
                start_dt.strftime("%Y-%m-%d"),
                end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:8],
            )
            calendar_str = [
                d.strftime("%Y%m%d") if isinstance(d, pd.Timestamp) else d for d in warm_cal
            ]
            lhb_lookup = build_lhb_lookup_by_date(
                top_list_df, trading_dates_str, calendar_dates=calendar_str
            )
        else:
            logger.warning("未找到龙虎榜数据，相关特征将为空")

    # 加载一致预期数据（可选）
    consensus_lookup = None
    report_rc_df = None
    if enable_consensus:
        from src.lazybull.factors.consensus import build_consensus_lookup_by_date

        report_rc_df = loader.load_report_rc()
        if report_rc_df is not None:
            logger.info(f"一致预期研报数据: {len(report_rc_df)} 条")
            consensus_lookup = build_consensus_lookup_by_date(report_rc_df, trading_dates_str)
        else:
            logger.warning("未找到一致预期研报数据，相关特征将为空")

    # 加载现金流质量因子（可选）
    cashflow_lookup = None
    if enable_cashflow_quality:
        from src.lazybull.factors.cashflow_quality import build_cashflow_quality_lookup_by_date

        cashflow_df = loader.load_cashflow(start_date, end_date)
        if cashflow_df is not None:
            logger.info(f"现金流量表数据: {len(cashflow_df)} 条")
            cashflow_lookup = build_cashflow_quality_lookup_by_date(
                cashflow_df,
                trading_dates_str,
            )
        else:
            logger.warning(
                "未找到现金流量表数据，相关特征将为空。"
                "请先运行: python scripts/download_raw.py --download cashflow"
            )

    # 加载一致预期修正因子（可选）
    consensus_revision_lookup = None
    if enable_consensus_revision:
        from src.lazybull.factors.consensus_revision import (
            build_consensus_revision_lookup_by_date,
        )

        if report_rc_df is None:
            report_rc_df = loader.load_report_rc()
        if report_rc_df is not None:
            logger.info(f"一致预期修正输入数据(report_rc): {len(report_rc_df)} 条")
            consensus_revision_lookup = build_consensus_revision_lookup_by_date(
                report_rc_df,
                trading_dates_str,
                daily_data_lookup=daily_close_lookup,
            )
        else:
            logger.warning(
                "未找到一致预期研报数据，修正因子将为空。"
                "请先运行: python scripts/download_raw.py --download report_rc"
            )

    # 加载风控公告类数据（可选：质押/解禁/大宗，PIT 日频查询表）
    pledge_lookup = None
    share_float_lookup = None
    block_trade_lookup = None
    if enable_announcement_risk:
        from src.lazybull.factors.risk.announcement_lookup import (
            build_block_trade_lookup_by_date,
            build_pledge_lookup_by_date,
            build_share_float_lookup_by_date,
        )

        # 质押（季分区，回溯 start_dt 覆盖 PIT 前向填充所需历史期）
        pledge_df = loader.load_pledge_stat(start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
        if pledge_df is not None and len(pledge_df) > 0:
            logger.info(f"质押数据: {len(pledge_df)} 条")
            pledge_lookup = build_pledge_lookup_by_date(pledge_df, trading_dates_str)
        else:
            logger.warning(
                "未找到质押数据，相关特征将为空。"
                "请先运行: python scripts/download_raw.py --download pledge_stat"
            )
        pledge_df = None

        # 限售解禁（年分区，PIT 按公告日，回溯 start_dt）
        share_float_df = loader.load_share_float(
            start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
        )
        if share_float_df is not None and len(share_float_df) > 0:
            logger.info(f"限售解禁数据: {len(share_float_df)} 条")
            share_float_lookup = build_share_float_lookup_by_date(share_float_df, trading_dates_str)
        else:
            logger.warning(
                "未找到限售解禁数据，相关特征将为空。"
                "请先运行: python scripts/download_raw.py --download share_float"
            )
        share_float_df = None

        # 大宗交易（日分区，近 10 交易日折价聚合，需未复权收盘价）
        block_trade_df = loader.load_block_trade(
            start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
        )
        if block_trade_df is not None and len(block_trade_df) > 0:
            logger.info(f"大宗交易数据: {len(block_trade_df)} 条")
            close_lookup = {
                trade_date: (
                    grp[["ts_code", "close"]].dropna().set_index("ts_code")["close"].to_dict()
                )
                for trade_date, grp in daily_clean.groupby("trade_date", sort=False)
                if "close" in grp.columns
            }
            block_trade_lookup = build_block_trade_lookup_by_date(
                block_trade_df, trading_dates_str, close_lookup=close_lookup
            )
        else:
            logger.warning(
                "未找到大宗交易数据，相关特征将为空。"
                "请先运行: python scripts/download_raw.py --download block_trade"
            )
        block_trade_df = None

    # clean数据已包含复权价格，使用空DataFrame
    adj_factor = pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])

    # 优化1/4：循环外一次性预计算 daily_adj（含 pre_close_adj）及日期索引字典
    builder.precompute_daily_adj(daily_clean, adj_factor)

    # 预筛：收集待处理日期（跳过已存在且schema完整的）
    total_dates = len(trading_dates_str)
    pending_dates = []
    skip_count = 0
    for trade_date in trading_dates_str:
        if not force and storage.is_feature_exists(trade_date):
            if _check_features_schema(storage, trade_date):
                skip_count += 1
                continue
            logger.warning(f"  {trade_date} 特征缓存缺少必要列，将重新构建")
        pending_dates.append(trade_date)

    logger.info(
        f"共 {total_dates} 个交易日: 跳过 {skip_count} (已存在), " f"待构建 {len(pending_dates)}"
    )

    if not pending_dates:
        logger.info("所有日期特征已存在，无需构建")
        return

    # ── 判断是否使用并行 ──
    if use_parallel and len(pending_dates) > 4:
        _build_features_parallel(
            pending_dates=pending_dates,
            builder=builder,
            storage=storage,
            trade_cal=trade_cal,
            daily_clean=daily_clean,
            adj_factor=adj_factor,
            stock_basic=stock_basic,
            daily_basic_clean=daily_basic_clean,
            moneyflow_clean=moneyflow_clean,
            shenwan_industry=shenwan_industry if apply_industry_neutralization else None,
            apply_industry_neutralization=apply_industry_neutralization,
            apply_size_neutralization=apply_size_neutralization,
            fundamental_lookup=fundamental_lookup,
            margin_lookup=margin_lookup,
            holder_lookup=holder_lookup,
            earnings_lookup=earnings_lookup,
            cyq_perf_lookup=cyq_perf_lookup,
            fund_portfolio_lookup=fund_portfolio_lookup,
            express_lookup=express_lookup,
            north_flow_lookup=north_flow_lookup,
            lhb_lookup=lhb_lookup,
            consensus_lookup=consensus_lookup,
            cashflow_lookup=cashflow_lookup,
            consensus_revision_lookup=consensus_revision_lookup,
            pledge_lookup=pledge_lookup,
            share_float_lookup=share_float_lookup,
            block_trade_lookup=block_trade_lookup,
            n_jobs=parallel_jobs,
        )
        return

    # ── 串行路径（保留作为回退）──
    success_count = 0
    error_count = 0
    loop_start_ts = time.time()

    for i, trade_date in enumerate(pending_dates, 1):
        try:
            # 获取当日因子数据
            funda_today = fundamental_lookup.get(trade_date) if fundamental_lookup else None
            margin_today = margin_lookup.get(trade_date) if margin_lookup else None
            holder_today = holder_lookup.get(trade_date) if holder_lookup else None
            earnings_today = earnings_lookup.get(trade_date) if earnings_lookup else None
            cyq_perf_today = cyq_perf_lookup.get(trade_date) if cyq_perf_lookup else None
            fund_portfolio_today = (
                fund_portfolio_lookup.get(trade_date) if fund_portfolio_lookup else None
            )
            express_today = express_lookup.get(trade_date) if express_lookup else None
            if north_flow_lookup is not None:
                north_flow_today = north_flow_lookup.get(trade_date) or {}
            else:
                north_flow_today = None
            if lhb_lookup is not None:
                lhb_today = lhb_lookup.get(trade_date)
                if lhb_today is None:
                    lhb_today = pd.DataFrame()
            else:
                lhb_today = None
            if consensus_lookup is not None:
                consensus_today = consensus_lookup.get(trade_date)
                if consensus_today is None:
                    consensus_today = pd.DataFrame()
            else:
                consensus_today = None
            if cashflow_lookup is not None:
                cashflow_today = cashflow_lookup.get(trade_date)
                if cashflow_today is None:
                    cashflow_today = pd.DataFrame()
            else:
                cashflow_today = None
            if consensus_revision_lookup is not None:
                consensus_revision_today = consensus_revision_lookup.get(trade_date)
                if consensus_revision_today is None:
                    consensus_revision_today = pd.DataFrame()
            else:
                consensus_revision_today = None
            if pledge_lookup is not None:
                pledge_today = pledge_lookup.get(trade_date)
                if pledge_today is None:
                    pledge_today = pd.DataFrame()
            else:
                pledge_today = None
            if share_float_lookup is not None:
                share_float_today = share_float_lookup.get(trade_date)
                if share_float_today is None:
                    share_float_today = pd.DataFrame()
            else:
                share_float_today = None
            if block_trade_lookup is not None:
                block_trade_today = block_trade_lookup.get(trade_date)
                if block_trade_today is None:
                    block_trade_today = pd.DataFrame()
            else:
                block_trade_today = None

            # 构建特征
            features_df = builder.build_features_for_day(
                trade_date=trade_date,
                trade_cal=trade_cal,
                daily_data=daily_clean,
                adj_factor=adj_factor,
                stock_basic=stock_basic,
                daily_basic_data=daily_basic_clean,
                moneyflow_data=moneyflow_clean,
                suspend_info=None,
                limit_info=None,
                shenwan_industry=shenwan_industry if apply_industry_neutralization else None,
                apply_industry_neutralization=apply_industry_neutralization,
                apply_size_neutralization=apply_size_neutralization,
                fundamental_data=funda_today,
                margin_data=margin_today,
                holder_data=holder_today,
                earnings_data=earnings_today,
                cyq_perf_data=cyq_perf_today,
                express_data=express_today,
                fund_portfolio_data=fund_portfolio_today,
                north_flow_data=north_flow_today,
                lhb_data=lhb_today,
                consensus_data=consensus_today,
                cashflow_data=cashflow_today,
                consensus_revision_data=consensus_revision_today,
                pledge_data=pledge_today,
                share_float_data=share_float_today,
                block_trade_data=block_trade_today,
            )

            # 保存结果
            if len(features_df) > 0:
                storage.save_cs_train_day(features_df, trade_date)
                success_count += 1
                elapsed = time.time() - loop_start_ts
                if i > 1 and elapsed > 0:
                    avg_per_date = elapsed / i
                    remaining = avg_per_date * (len(pending_dates) - i)
                    eta_str = (datetime.now() + timedelta(seconds=remaining)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                else:
                    eta_str = "计算中"
                _summary = getattr(
                    builder, "_last_summary", f"{trade_date} ✓ {len(features_df)}样本"
                )
                logger.info(
                    f"[{i}/{len(pending_dates)}] ({i/len(pending_dates):.1%}) {_summary} "
                    f"| 预计完成: {eta_str}"
                )
            else:
                logger.warning(f"  {trade_date} 没有有效样本，跳过保存")
                skip_count += 1

        except Exception as e:
            logger.error(f"  {trade_date} 构建失败: {str(e)}")
            error_count += 1
            continue

    logger.info("=" * 60)
    logger.info("features层数据构建完成")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count} 个交易日")
    logger.info(f"跳过: {skip_count} 个交易日（已存在或无效样本）")
    logger.info(f"失败: {error_count} 个交易日（共处理 {len(pending_dates)} 天）")
