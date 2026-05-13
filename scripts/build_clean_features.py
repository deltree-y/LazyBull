#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建clean和features数据脚本

功能：
- 假设raw数据已存在，若缺失则报错
- 只负责计算clean和feature并保存（partitioned存储）
- 不进行raw数据下载
- 支持force参数强制重新构建已存在的数据
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataCleaner, DataLoader, Storage
from src.lazybull.features import FeatureBuilder
from src.lazybull.features.ensure import _check_features_schema


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
)


def apply_build_all_feature_flags(args: argparse.Namespace) -> argparse.Namespace:
    """当 --build-all 启用时，统一打开全部可选因子开关。"""
    if not getattr(args, "build_all", False):
        return args

    for attr in OPTIONAL_FEATURE_FLAG_ATTRS:
        setattr(args, attr, True)
    return args


def build_clean_data(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    start_date: str,
    end_date: str,
    force: bool = False,
    min_list_days: int = 365,
) -> None:
    """构建clean层数据

    Args:
        storage: Storage实例
        loader: DataLoader实例
        cleaner: DataCleaner实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新构建
        min_list_days: 最小上市天数
    """
    logger.info("=" * 60)
    logger.info("开始构建clean层数据")
    logger.info("=" * 60)
    
    # 1. 检查并处理trade_cal
    logger.info("处理交易日历...")
    trade_cal_raw = storage.load_raw("trade_cal")
    if trade_cal_raw is None:
        raise ValueError("缺少raw层trade_cal数据，请先运行: python scripts/download_raw.py --only-basic")
    
    trade_cal_clean = cleaner.clean_trade_cal(trade_cal_raw)
    storage.save_clean(trade_cal_clean, "trade_cal", is_force=True)
    logger.info(f"交易日历清洗完成: {len(trade_cal_clean)} 条记录")
    
    # 2. 检查并处理stock_basic
    logger.info("处理股票基本信息...")
    stock_basic_raw = storage.load_raw("stock_basic")
    if stock_basic_raw is None:
        raise ValueError("缺少raw层stock_basic数据，请先运行: python scripts/download_raw.py --only-basic")
    
    stock_basic_clean = cleaner.clean_stock_basic(stock_basic_raw)
    storage.save_clean(stock_basic_clean, "stock_basic", is_force=True)
    logger.info(f"股票基本信息清洗完成: {len(stock_basic_clean)} 条记录")
    
    # 3. 按日期分区处理日线数据
    logger.info("使用分区模式处理日线数据...")
    
    # 获取交易日列表
    trading_dates = trade_cal_clean[
        (trade_cal_clean['cal_date'] >= start_date) &
        (trade_cal_clean['cal_date'] <= end_date) &
        (trade_cal_clean['is_open'] == 1)
    ]['cal_date'].tolist()
    
    logger.info(f"共 {len(trading_dates)} 个交易日需要处理")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, trade_date in enumerate(trading_dates, 1):
        logger.info(f"[{i}/{len(trading_dates)}] ({i/len(trading_dates):.1%}) 处理 {trade_date}...")
        
        try:
            # 检查clean数据是否已存在
            if not force and storage.is_data_exists("clean", "daily", trade_date):
                logger.info(f"  clean daily已存在，跳过")
                skip_count += 1
                continue
            
            # 加载raw数据
            daily_raw = storage.load_raw_by_date("daily", trade_date)
            if daily_raw is None or len(daily_raw) == 0:
                logger.warning(f"  未找到raw层daily数据，跳过")
                error_count += 1
                continue
            
            adj_factor_raw = storage.load_raw_by_date("adj_factor", trade_date)
            if adj_factor_raw is None or len(adj_factor_raw) == 0:
                logger.warning(f"  未找到复权因子，使用默认值1.0")
                adj_factor_raw = daily_raw[['ts_code', 'trade_date']].copy()
                adj_factor_raw['adj_factor'] = 1.0
            
            # 清洗日线数据
            daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)
            
            # 添加可交易标记
            suspend_raw = storage.load_raw_by_date("suspend", trade_date)
            limit_raw = storage.load_raw_by_date("stk_limit", trade_date)
            
            suspend_clean = None
            limit_clean = None
            
            if suspend_raw is not None and len(suspend_raw) > 0:
                suspend_clean = cleaner.clean_suspend_info(suspend_raw)
            
            if limit_raw is not None and len(limit_raw) > 0:
                limit_clean = cleaner.clean_limit_info(limit_raw)
            
            daily_clean = cleaner.add_tradable_universe_flag(
                daily_clean,
                stock_basic_clean,
                suspend_info_df=suspend_clean,
                limit_info_df=limit_clean,
                min_list_days=min_list_days
            )

            # 保存clean数据
            storage.save_clean_by_date(daily_clean, "daily", trade_date)
            success_count += 1
            logger.info(f"  已保存 {len(daily_clean)} 条clean记录")
            
            # 处理daily_basic
            daily_basic_raw = storage.load_raw_by_date("daily_basic", trade_date)
            if daily_basic_raw is not None and len(daily_basic_raw) > 0:
                if force or not storage.is_data_exists("clean", "daily_basic", trade_date):
                    daily_basic_clean = cleaner.clean_daily_basic(daily_basic_raw)
                    storage.save_clean_by_date(daily_basic_clean, "daily_basic", trade_date)
            
            # 处理moneyflow
            moneyflow_raw = storage.load_raw_by_date("moneyflow", trade_date)
            if moneyflow_raw is not None and len(moneyflow_raw) > 0:
                if force or not storage.is_data_exists("clean", "moneyflow", trade_date):
                    moneyflow_clean = cleaner.clean_moneyflow(moneyflow_raw)
                    storage.save_clean_by_date(moneyflow_clean, "moneyflow", trade_date)
            else:
                logger.warning(f"  未找到资金流向数据（moneyflow 为强制依赖项）")
            
        except Exception as e:
            logger.error(f"处理 {trade_date} 失败: {str(e)}")
            error_count += 1
            continue
    
    logger.info("=" * 60)
    logger.info("clean层数据构建完成")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count} 个交易日")
    logger.info(f"跳过: {skip_count} 个交易日（已存在）")
    logger.info(f"失败: {error_count} 个交易日")


def build_features_data(
    storage: Storage,
    loader: DataLoader,
    builder: FeatureBuilder,
    start_date: str,
    end_date: str,
    force: bool = False,
    shenwan_industry: pd.DataFrame = None,
    apply_industry_neutralization: bool = False,
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
    if 'cal_date' in trade_cal.columns:
        if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
            trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
    
    # 获取交易日列表
    trading_dates = loader.get_trading_dates(
        start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:8],
        end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:8]
    )
    
    if len(trading_dates) == 0:
        raise ValueError(f"指定日期范围内没有交易日: {start_date} - {end_date}")
    
    # 转换为YYYYMMDD格式
    trading_dates_str = [
        d.strftime('%Y%m%d') if isinstance(d, pd.Timestamp) else d
        for d in trading_dates
    ]
    
    logger.info(f"共 {len(trading_dates_str)} 个交易日需要构建特征")
    
    # 加载clean层日线数据（扩展范围以包含足够的 warmup 历史，覆盖 120 个交易日 ≈ 7 个月）
    start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=7)
    end_dt = pd.to_datetime(end_date, format='%Y%m%d') + pd.DateOffset(months=1)
    
    daily_clean = loader.load_clean_daily(
        start_dt.strftime('%Y%m%d'),
        end_dt.strftime('%Y%m%d')
    )
    
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
        start_dt.strftime('%Y%m%d'),
        end_dt.strftime('%Y%m%d')
    )
    if daily_basic_clean is not None:
        logger.info(f"clean daily_basic 数据: {len(daily_basic_clean)} 条记录")
    else:
        logger.warning("未找到 daily_basic 数据，价值红利特征将为空")
    
    # 加载 moneyflow 数据
    moneyflow_clean = loader.load_clean_moneyflow(
        start_dt.strftime('%Y%m%d'),
        end_dt.strftime('%Y%m%d')
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
            fundamental_lookup = build_fundamental_lookup_by_date(
                fina_indicator, trading_dates_str
            )
        else:
            logger.warning("未找到财务指标数据，基本面特征将被跳过。"
                         "请先运行: python scripts/download_raw.py --download fina_indicator")

    # 加载融资融券数据（可选，独立开关）
    margin_lookup = None
    if enable_margin:
        from src.lazybull.factors.margin import build_margin_lookup_by_date

        margin_detail = loader.load_margin_detail(
            start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d')
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
        from src.lazybull.factors.holder import build_holder_lookup_by_date
        from src.lazybull.factors.earnings import build_earnings_lookup_by_date

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
            earnings_lookup = build_earnings_lookup_by_date(
                forecast_df, trading_dates_str
            )
        else:
            logger.warning("未找到业绩预告数据，相关特征将为空")

    # 加载筹码胜率数据（可选，按日分区存储，回溯加载以确保 diff(5)/diff(20) 有足够历史）
    cyq_perf_lookup = None
    if enable_cyq:
        from src.lazybull.factors.cyq_perf import build_cyq_perf_lookup_by_date
        cyq_perf_df = loader.load_cyq_perf(start_dt.strftime('%Y%m%d'), end_date)
        if cyq_perf_df is not None:
            logger.info(f"筹码胜率数据: {len(cyq_perf_df)} 条")
            cyq_perf_lookup = build_cyq_perf_lookup_by_date(cyq_perf_df, trading_dates_str)
        else:
            logger.warning("未找到筹码胜率数据，相关特征将为空。"
                         "请先运行: python scripts/download_raw.py --download cyq_perf")

    # 加载基金持仓数据（可选，按季度分区存储，回溯加载以确保 point-in-time 查询覆盖前序季度）
    fund_portfolio_lookup = None
    if enable_fund:
        from src.lazybull.factors.fund_portfolio import build_fund_portfolio_lookup_by_date
        fund_portfolio_df = loader.load_fund_portfolio(start_dt.strftime('%Y%m%d'), end_date)
        if fund_portfolio_df is not None:
            logger.info(f"基金持仓数据: {len(fund_portfolio_df)} 条")
            fund_portfolio_lookup = build_fund_portfolio_lookup_by_date(
                fund_portfolio_df, trading_dates_str
            )
        else:
            logger.warning("未找到基金持仓数据，相关特征将为空。"
                         "请先运行: python scripts/download_raw.py --download fund_portfolio")

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
            logger.warning("未找到业绩快报数据，相关特征将为空。"
                         "请先运行: python scripts/download_raw.py --download express")

    # 加载北向资金数据（可选，市场级广播）
    north_flow_lookup = None
    if enable_north:
        from src.lazybull.factors.north_flow import build_north_flow_lookup_by_date
        hsgt_df = loader.load_moneyflow_hsgt(start_dt.strftime('%Y%m%d'), end_date)
        if hsgt_df is not None:
            logger.info(f"北向资金数据: {len(hsgt_df)} 条")
            north_flow_lookup = build_north_flow_lookup_by_date(hsgt_df, trading_dates_str)
        else:
            logger.warning("未找到北向资金数据，相关特征将为空")

    # 加载龙虎榜数据（可选）
    lhb_lookup = None
    if enable_lhb:
        from src.lazybull.factors.lhb import build_lhb_lookup_by_date
        top_list_df = loader.load_top_list(start_dt.strftime('%Y%m%d'), end_date)
        if top_list_df is not None:
            logger.info(f"龙虎榜数据: {len(top_list_df)} 条")
            lhb_lookup = build_lhb_lookup_by_date(top_list_df, trading_dates_str)
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
            logger.warning("未找到现金流量表数据，相关特征将为空。"
                         "请先运行: python scripts/download_raw.py --download cashflow")

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
            logger.warning("未找到一致预期研报数据，修正因子将为空。"
                         "请先运行: python scripts/download_raw.py --download report_rc")

    # clean数据已包含复权价格，使用空DataFrame
    adj_factor = pd.DataFrame(columns=['ts_code', 'trade_date', 'adj_factor'])

    # 优化1/4：循环外一次性预计算 daily_adj（含 pre_close_adj）及日期索引字典
    # 避免在 2000 次循环内重复执行全量 copy / sort_values / groupby.shift
    builder.precompute_daily_adj(daily_clean, adj_factor)

    # 构建特征
    success_count = 0
    skip_count = 0
    error_count = 0

    total_dates = len(trading_dates_str)
    loop_start_ts = time.time()

    for i, trade_date in enumerate(trading_dates_str, 1):
        # 预估完成时间：基于已处理的日期平均耗时线性外推
        elapsed = time.time() - loop_start_ts
        if i > 1 and elapsed > 0:
            avg_per_date = elapsed / (i - 1)
            remaining = avg_per_date * (total_dates - i + 1)
            eta_str = (datetime.now() + timedelta(seconds=remaining)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            eta_str = "计算中"
        logger.info(
            f"\n===== [{i}/{total_dates}] ({i/total_dates:.1%}) 构建 {trade_date} 特征 "
            f"| 预计完成: {eta_str} ====="
        )
        
        try:
            # 检查特征是否已存在
            if not force and storage.is_feature_exists(trade_date):
                if _check_features_schema(storage, trade_date):
                    logger.info(f"  特征已存在，跳过")
                    skip_count += 1
                    continue
                logger.warning("  特征缓存缺少必要列，将重新构建")
            
            # 获取当日基本面数据
            funda_today = None
            if fundamental_lookup is not None:
                funda_today = fundamental_lookup.get(trade_date)

            # 获取当日另类数据
            margin_today = margin_lookup.get(trade_date) if margin_lookup else None
            holder_today = holder_lookup.get(trade_date) if holder_lookup else None
            earnings_today = earnings_lookup.get(trade_date) if earnings_lookup else None
            # 获取当日高积分因子数据
            cyq_perf_today = cyq_perf_lookup.get(trade_date) if cyq_perf_lookup else None
            fund_portfolio_today = (
                fund_portfolio_lookup.get(trade_date) if fund_portfolio_lookup else None
            )
            express_today = express_lookup.get(trade_date) if express_lookup else None
            # 获取当日 C1/C2/C3 数据
            # 语义: 启用 (lookup 非 None) 时, 即使当日缺数据也传空容器, 保证特征 schema 一致
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
            )
            
            # 保存结果
            if len(features_df) > 0:
                storage.save_cs_train_day(features_df, trade_date)
                success_count += 1
                logger.info(f"  已保存 {len(features_df)} 条特征记录")
            else:
                logger.warning(f"  没有有效样本，跳过保存")
                skip_count += 1
                
        except Exception as e:
            logger.error(f"  构建失败: {str(e)}")
            error_count += 1
            continue
    
    logger.info("=" * 60)
    logger.info("features层数据构建完成")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count} 个交易日")
    logger.info(f"跳过: {skip_count} 个交易日（已存在或无效样本）")
    logger.info(f"失败: {error_count} 个交易日")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="构建clean和features数据（假设raw已存在）"
    )
    parser.add_argument(
        "--start-date",
        default="20200101",
        help="开始日期，格式YYYYMMDD（默认：20200101）"
    )
    parser.add_argument(
        "--end-date",
        default="20251231",
        help="结束日期，格式YYYYMMDD（默认：20251231）"
    )
    parser.add_argument(
        "--only-clean",
        action="store_true",
        help="仅构建clean层，不构建features"
    )
    parser.add_argument(
        "--only-features",
        action="store_true",
        help="仅构建features层，不构建clean"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新构建，即使文件已存在"
    )
    parser.add_argument(
        "--enable-industry-neutralization",
        action="store_true",
        help="启用中性特征构建"
    )
    parser.add_argument(
        "--min-list-days",
        type=int,
        default=365,
        help="最小上市自然日天数（默认：365，约12个月）"
    )
    horizon_group = parser.add_mutually_exclusive_group(required=True)
    horizon_group.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="单 horizon 模式：按此主标签 y_ret_N 非空过滤样本（如 --horizon 20）。"
             "仍生成 y_ret_5/10/20 三列标签，仅过滤时只看主 horizon"
    )
    horizon_group.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="多 horizon 模式：按 AND 过滤，要求给定 horizons 对应的所有标签同时非空"
             "（如 --horizons 5 10 20）"
    )
    parser.add_argument(
        "--build-all",
        action="store_true",
        help="启用全部可选因子（基本面、另类数据、融资融券、筹码胜率、基金持仓、业绩快报、北向资金、龙虎榜、一致预期；不含行业中性化）"
    )
    parser.add_argument(
        "--enable-fundamental-features",
        action="store_true",
        help="启用基本面因子（ROE、营收增速等），需先下载 fina_indicator 数据"
    )
    parser.add_argument(
        "--enable-alt-features",
        action="store_true",
        help="启用另类数据因子（股东人数、业绩预告等）"
    )
    parser.add_argument(
        "--enable-margin-features",
        action="store_true",
        help="启用融资融券因子（融资余额变动、融券/融资比、净买入比等）"
    )
    parser.add_argument(
        "--enable-cyq-features",
        action="store_true",
        help="启用筹码胜率因子（winner_rate、成本偏离等）"
    )
    parser.add_argument(
        "--enable-fund-features",
        action="store_true",
        help="启用基金持仓因子（持股比例、基金数量等）"
    )
    parser.add_argument(
        "--enable-express-features",
        action="store_true",
        help="启用业绩快报因子（实际营收/净利润增速等）"
    )
    parser.add_argument(
        "--enable-north-features",
        action="store_true",
        help="启用北向资金因子（moneyflow_hsgt 市场级广播）"
    )
    parser.add_argument(
        "--enable-lhb-features",
        action="store_true",
        help="启用龙虎榜因子（top_list 个股级）"
    )
    parser.add_argument(
        "--enable-consensus-features",
        action="store_true",
        help="启用一致预期因子（report_rc 研报滚动聚合）"
    )
    parser.add_argument(
        "--enable-cashflow-quality-features",
        action="store_true",
        help="启用现金流质量因子（需 cashflow 接口，2000 积分，需先下载 cashflow 数据）"
    )
    parser.add_argument(
        "--enable-consensus-revision-features",
        action="store_true",
        help="启用一致预期修正因子（基于已有 report_rc 构建时序修正信号，无需额外下载）"
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
    logger.info(f"一致预期修正因子: {'启用' if args.enable_consensus_revision_features else '禁用'}")
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
                storage, loader, cleaner,
                args.start_date, args.end_date,
                force=args.force,
                min_list_days=args.min_list_days,
            )
        
        # 构建features数据
        if not args.only_clean:
            build_features_data(
                storage, loader, builder,
                args.start_date, args.end_date,
                force=args.force,
                shenwan_industry=shenwan_industry if args.enable_industry_neutralization else None,
                apply_industry_neutralization=args.enable_industry_neutralization,
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
