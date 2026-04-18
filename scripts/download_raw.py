#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载原始数据脚本（仅下载raw层）

功能：
- 仅负责从TuShare/AKShare拉取原始数据并保存到raw层
- 不触发clean或feature的构建
- 支持force参数强制重新下载已存在的数据
- 支持--download参数选择下载特定数据集

数据集：
  基础数据（默认）：trade_cal, stock_basic
  日线数据（默认）：daily, daily_basic, adj_factor, suspend, stk_limit, moneyflow
  另类数据（需指定）：
    fina_indicator   - 财务指标（Tushare fina_indicator_vip，按季度全市场）
    margin_detail    - 融资融券明细（Tushare，按日分区）
    stk_holdernumber - 股东人数（Tushare，按月全市场）
    forecast         - 业绩预告（Tushare forecast_vip，按季度全市场）
    cyq_perf         - 筹码胜率（Tushare，按日分区）
    express          - 业绩快报（Tushare express_vip，按季度全市场）
    fund_portfolio   - 基金持仓（Tushare fund_portfolio，按季度全市场）
    moneyflow_hsgt   - 北向资金（Tushare moneyflow_hsgt，按日分区，市场级广播）
    top_list         - 龙虎榜（Tushare top_list，按日分区，无数据存空占位）
    report_rc        - 一致预期研报（Tushare report_rc，按年分页增量）

使用示例：
    # 默认：下载基础数据 + 日线数据
    python scripts/download_raw.py

    # 一键全量：日线 + 全部另类数据（推荐首次初始化使用）
    python scripts/download_raw.py --all

    # 下载特定另类数据
    python scripts/download_raw.py --download fina_indicator margin_detail

    # 下载全部另类数据（不含日线）
    python scripts/download_raw.py --download all_alt
"""

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient

if TYPE_CHECKING:
    import pandas as pd

# 所有另类数据集名称
ALT_DATASETS = [
    "fina_indicator", "margin_detail", "stk_holdernumber",
    "forecast", "cyq_perf", "express", "fund_portfolio",
    "moneyflow_hsgt", "top_list", "report_rc",
]


def download_basic_data(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False
) -> "pd.DataFrame":
    """下载基础数据（trade_cal和stock_basic）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新下载
        
    Returns:
        交易日历DataFrame
    """
    # 1. 下载交易日历
    logger.info("检查交易日历...")
    if not force and storage.check_basic_data_freshness("trade_cal", end_date):
        logger.info("交易日历数据已是最新，跳过下载")
        trade_cal = storage.load_raw("trade_cal")
    else:
        logger.info(f"下载交易日历（{start_date}-{end_date}）...")
        trade_cal = client.get_trade_cal(
            start_date=start_date,
            end_date=end_date,
            exchange="SSE"
        )
        storage.save_raw(trade_cal, "trade_cal", is_force=True)
        logger.info(f"交易日历下载完成: {len(trade_cal)} 条记录")
    
    # 2. 下载股票基本信息
    logger.info("检查股票基本信息...")
    if not force and storage.check_basic_data_freshness("stock_basic", end_date):
        logger.info("股票基本信息已存在，跳过下载")
    else:
        logger.info("下载股票基本信息...")
        stock_basic = client.get_stock_basic(list_status="L")
        storage.save_raw(stock_basic, "stock_basic", is_force=True)
        logger.info(f"股票基本信息下载完成: {len(stock_basic)} 条记录")
    
    return trade_cal


def download_daily_data(
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False
) -> None:
    """下载日线数据（按日期分区）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        trade_cal: 交易日历DataFrame
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新下载
    """
    import pandas as pd
    
    logger.info(f"下载日线数据（{start_date}-{end_date}）...")
    logger.info("使用按日分区存储模式")
    
    # 获取交易日列表
    trading_dates = trade_cal[
        (trade_cal['cal_date'] >= start_date) &
        (trade_cal['cal_date'] <= end_date) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    logger.info(f"共 {len(trading_dates)} 个交易日需要下载")
    
    total_daily = 0
    total_basic = 0
    skip_count = 0
    
    for i, trade_date in enumerate(trading_dates, 1):
        logger.info(f"[{i}/{len(trading_dates)}] ({i/len(trading_dates):.1%}) 处理 {trade_date}...")
        
        try:
            # 下载日线行情
            if not force and storage.is_data_exists("raw", "daily", trade_date):
                logger.info(f"  日线: 文件已存在，跳过下载")
                skip_count += 1
            else:
                daily_data = client.get_daily(trade_date=trade_date)
                if len(daily_data) > 0:
                    storage.save_raw_by_date(daily_data, "daily", trade_date)
                    total_daily += len(daily_data)
                    logger.info(f"  日线: 已保存 {len(daily_data)} 条记录")
            
            # 下载每日指标
            if not force and storage.is_data_exists("raw", "daily_basic", trade_date):
                logger.info(f"  指标: 文件已存在，跳过下载")
            else:
                daily_basic = client.get_daily_basic(trade_date=trade_date)
                if len(daily_basic) > 0:
                    storage.save_raw_by_date(daily_basic, "daily_basic", trade_date)
                    total_basic += len(daily_basic)
                    logger.info(f"  指标: 已保存 {len(daily_basic)} 条记录")

            # 下载复权因子
            if not force and storage.is_data_exists("raw", "adj_factor", trade_date):
                logger.info(f"  复权因子: 文件已存在，跳过下载")
            else:
                adj_factor = client.get_adj_factor(trade_date=trade_date)
                if len(adj_factor) > 0:
                    storage.save_raw_by_date(adj_factor, "adj_factor", trade_date)
                    logger.info(f"  复权因子: 已保存 {len(adj_factor)} 条记录")
                    
            # 下载停复牌信息
            if not force and storage.is_data_exists("raw", "suspend", trade_date):
                logger.info(f"  停复牌: 文件已存在，跳过下载")
            else:
                suspend = client.get_suspend_d(trade_date=trade_date)
                if len(suspend) > 0:
                    storage.save_raw_by_date(suspend, "suspend", trade_date)
                    logger.info(f"  停复牌: 已保存 {len(suspend)} 条记录")
                    
            # 下载涨跌停信息
            if not force and storage.is_data_exists("raw", "stk_limit", trade_date):
                logger.info(f"  涨跌停: 文件已存在，跳过下载")
            else:
                limit_up_down = client.get_stk_limit(trade_date=trade_date)
                if len(limit_up_down) > 0:
                    storage.save_raw_by_date(limit_up_down, "stk_limit", trade_date)
                    logger.info(f"  涨跌停: 已保存 {len(limit_up_down)} 条记录")
            
            # 下载资金流向
            if not force and storage.is_data_exists("raw", "moneyflow", trade_date):
                logger.info(f"  资金流向: 文件已存在，跳过下载")
            else:
                moneyflow = client.get_moneyflow(trade_date=trade_date)
                if len(moneyflow) > 0:
                    storage.save_raw_by_date(moneyflow, "moneyflow", trade_date)
                    logger.info(f"  资金流向: 已保存 {len(moneyflow)} 条记录")
                else:
                    logger.error(f"  资金流向数据缺失（moneyflow 为强制依赖项）")
                    
        except Exception as e:
            logger.error(f"下载 {trade_date} 数据失败: {str(e)}")
            continue
    
    logger.info("=" * 60)
    logger.info("日线数据下载完成")
    logger.info("=" * 60)
    logger.info(f"新下载日线行情: {total_daily} 条记录")
    logger.info(f"新下载每日指标: {total_basic} 条记录")
    logger.info(f"跳过已存在: {skip_count} 个交易日")


def _save_merged(storage, dataset_name, new_dfs, existing_df, dedup_cols):
    """合并新旧数据并保存"""
    result = pd.concat(new_dfs, ignore_index=True)
    if existing_df is not None and len(existing_df) > 0:
        result = pd.concat([existing_df, result], ignore_index=True)
    if dedup_cols:
        result = result.drop_duplicates(subset=dedup_cols, keep="last")
    storage.save_raw(result, dataset_name, is_force=True)
    logger.info(f"[{dataset_name}] 已保存: {len(result)} 条记录")


def download_margin_detail(
    client: TushareClient,
    storage: Storage,
    trade_cal: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载融资融券明细（按日分区，与 daily 同模式）"""
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    logger.info(f"[margin_detail] {len(trading_dates)} 个交易日")
    success = skip = errors = 0

    for i, td in enumerate(trading_dates, 1):
        try:
            if not force and storage.is_data_exists("raw", "margin_detail", td):
                skip += 1
                continue
            df = client.query("margin_detail", trade_date=td)
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, "margin_detail", td)
                success += 1
            # 融资融券覆盖面较窄，部分日期无数据属正常
        except Exception as e:
            errors += 1
            logger.warning(f"[margin_detail] {td} 失败: {e}")

        if i % 100 == 0 or i == len(trading_dates):
            logger.info(
                f"[margin_detail] [{i}/{len(trading_dates)}] "
                f"新下载={success} 跳过={skip} 失败={errors}"
            )

    logger.info(f"[margin_detail] 完成: 新下载={success} 跳过={skip} 失败={errors}")


def download_cyq_perf(
    client: TushareClient,
    storage: Storage,
    trade_cal: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载筹码胜率数据（按日分区，与 margin_detail 同模式）

    使用 trade_date 参数一次获取全市场当日数据，无需逐股下载。
    """
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    logger.info(f"[cyq_perf] {len(trading_dates)} 个交易日")
    success = skip = errors = 0

    for i, td in enumerate(trading_dates, 1):
        try:
            if not force and storage.is_data_exists("raw", "cyq_perf", td):
                skip += 1
                continue
            df = client.get_cyq_perf(trade_date=td)
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, "cyq_perf", td)
                success += 1
        except Exception as e:
            errors += 1
            logger.warning(f"[cyq_perf] {td} 失败: {e}")

        if i % 100 == 0 or i == len(trading_dates):
            logger.info(
                f"[cyq_perf] [{i}/{len(trading_dates)}] "
                f"新下载={success} 跳过={skip} 失败={errors}"
            )

    logger.info(f"[cyq_perf] 完成: 新下载={success} 跳过={skip} 失败={errors}")


def _generate_quarter_periods(start_date: str, end_date: str) -> List[str]:
    """生成日期范围内覆盖的所有季度末日期（YYYYMMDD 格式）

    例如 20230301~20240601 → [20230331, 20230630, 20230930, 20231231, 20240331, 20240630]
    """
    quarter_ends = ["0331", "0630", "0930", "1231"]
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    periods = []
    for year in range(start_year, end_year + 1):
        for qe in quarter_ends:
            p = f"{year}{qe}"
            if p >= start_date and p <= end_date:
                periods.append(p)
    return periods


def _generate_month_periods(start_date: str, end_date: str) -> List[str]:
    """生成日期范围内的月末日期列表，用于按月批量下载

    返回 [(start, end), ...] 每个元素为一个月的起止日期。
    """
    from datetime import datetime
    import calendar

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    periods = []
    current = start.replace(day=1)
    while current <= end:
        month_start = current.strftime("%Y%m%d")
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end_dt = current.replace(day=last_day)
        if month_end_dt > end:
            month_end_dt = end
        month_end = month_end_dt.strftime("%Y%m%d")
        periods.append((month_start, month_end))
        # 下个月
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
    return periods


def _query_with_pagination(
    client: TushareClient,
    api_name: str,
    page_limit: int = 50000,
    fields: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """带分页的 API 调用，自动检测并翻页获取全量数据

    当单次返回恰好 page_limit 条时，说明可能还有更多数据，
    自动通过 offset 翻页直到获取完毕。

    Args:
        client: TushareClient 实例
        api_name: TuShare API 名称
        page_limit: 单页上限（TuShare 默认 5000，部分接口 8000）
        fields: 返回字段
        **kwargs: 传递给 API 的查询参数（如 period, ann_date 等）

    Returns:
        合并后的完整 DataFrame
    """
    all_pages: List[pd.DataFrame] = []
    offset = 0

    while True:
        df = client.pro.query(
            api_name, fields=fields or "",
            limit=page_limit, offset=offset, **kwargs,
        )
        if df is None or len(df) == 0:
            break
        all_pages.append(df)
        if len(df) < page_limit:
            break  # 未满页，说明已获取全部
        offset += page_limit
        logger.debug(f"  [{api_name}] 分页: offset={offset}, 已获取 {sum(len(d) for d in all_pages)} 条")

    if not all_pages:
        return pd.DataFrame()
    return pd.concat(all_pages, ignore_index=True)


def download_by_period(
    client: TushareClient,
    storage: Storage,
    dataset_name: str,
    api_name: str,
    start_date: str,
    end_date: str,
    dedup_cols: List[str],
    fields: Optional[str] = None,
    force: bool = False,
    page_limit: int = 50000,
    partition_by_period: bool = False,
) -> None:
    """按报告期(period)批量下载数据（每季度1次API调用，自动分页）

    适用于：fina_indicator_vip, forecast_vip, express_vip, fund_portfolio
    这些 API 均支持 period 参数，一次获取一个季度全市场数据。
    当单季度数据超过单次上限时，自动通过 offset 分页获取全量。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        dataset_name: 数据集名称
        api_name: TuShare API 名称
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        dedup_cols: 去重列
        fields: 返回字段（仅部分 API 需要）
        force: 是否强制重下
        page_limit: 单次查询上限（默认50000，TuShare多数接口支持）
        partition_by_period: 是否按季度分区保存（适用于大数据量如 fund_portfolio）
    """
    periods = _generate_quarter_periods(start_date, end_date)
    if not periods:
        logger.warning(f"[{dataset_name}] 日期范围内无有效季度")
        return

    # 断点续传：检查哪些季度已有
    existing_df = None
    existing_periods: Set[str] = set()
    if not force:
        if partition_by_period:
            # 分区模式：检查每个季度的分区文件
            for p in periods:
                if storage.is_data_exists("raw", dataset_name, p):
                    existing_periods.add(p)
            if existing_periods:
                logger.info(
                    f"[{dataset_name}] 已有 {len(existing_periods)} 个季度分区（断点续传）"
                )
        else:
            # 单文件模式：从已有数据中提取季度
            existing_df = storage.load_raw(dataset_name)
            if existing_df is not None and len(existing_df) > 0:
                if "end_date" in existing_df.columns:
                    existing_periods = set(
                        existing_df["end_date"].astype(str)
                        .str.replace("-", "").str[:8].unique()
                    )
                logger.info(
                    f"[{dataset_name}] 已有 {len(existing_periods)} 个季度数据（断点续传）"
                )

    periods_to_download = [p for p in periods if p not in existing_periods]
    if not periods_to_download:
        logger.info(f"[{dataset_name}] 所有季度数据已存在，跳过。如需重下请加 --force")
        return

    logger.info(
        f"[{dataset_name}] 按季度批量下载: {len(periods_to_download)} 个季度 "
        f"({periods_to_download[0]}~{periods_to_download[-1]})"
    )

    all_dfs: List[pd.DataFrame] = []
    success = empty = errors = 0
    t0 = time.time()

    for period in periods_to_download:
        try:
            df = _query_with_pagination(
                client, api_name, page_limit=page_limit,
                fields=fields, period=period,
            )
            if df is not None and len(df) > 0:
                if partition_by_period:
                    storage.save_raw_by_date(df, dataset_name, period)
                else:
                    all_dfs.append(df)
                success += 1
                logger.info(f"  [{dataset_name}] {period}: {len(df)} 条")
            else:
                empty += 1
                logger.debug(f"  [{dataset_name}] {period}: 无数据")
        except Exception as e:
            errors += 1
            logger.warning(f"[{dataset_name}] {period} 失败: {e}")

    # 单文件模式：合并保存
    if not partition_by_period and all_dfs:
        _save_merged(storage, dataset_name, all_dfs, existing_df, dedup_cols)

    elapsed_total = time.time() - t0
    logger.info(
        f"[{dataset_name}] 完成: 成功={success} 空={empty} 失败={errors} "
        f"({len(periods_to_download)} 个季度, 耗时={elapsed_total:.0f}秒)"
    )


def download_stk_holdernumber(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    dedup_cols: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """按月批量下载股东人数数据（每月1次API调用，单次限3000条）

    stk_holdernumber 支持 start_date/end_date 参数获取全市场数据，
    但单次最多返回3000条，按月切分确保不超限。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        dedup_cols: 去重列
        force: 是否强制重下
    """
    if dedup_cols is None:
        dedup_cols = ["ts_code", "end_date"]

    month_ranges = _generate_month_periods(start_date, end_date)
    if not month_ranges:
        logger.warning("[stk_holdernumber] 日期范围无效")
        return

    existing_df = None
    if not force:
        existing_df = storage.load_raw("stk_holdernumber")
        if existing_df is not None and len(existing_df) > 0:
            logger.info(f"[stk_holdernumber] 已有 {len(existing_df)} 条数据")

    logger.info(
        f"[stk_holdernumber] 按月批量下载: {len(month_ranges)} 个月 "
        f"({month_ranges[0][0]}~{month_ranges[-1][1]})"
    )

    all_dfs: List[pd.DataFrame] = []
    success = empty = errors = 0
    t0 = time.time()

    for i, (m_start, m_end) in enumerate(month_ranges, 1):
        try:
            df = client.get_stk_holdernumber(start_date=m_start, end_date=m_end)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success += 1
                logger.debug(f"  [stk_holdernumber] {m_start}~{m_end}: {len(df)} 条")
            else:
                empty += 1
        except Exception as e:
            errors += 1
            logger.warning(f"[stk_holdernumber] {m_start}~{m_end} 失败: {e}")

        if i % 12 == 0 or i == len(month_ranges):
            logger.info(
                f"[stk_holdernumber] [{i}/{len(month_ranges)}] "
                f"成功={success} 空={empty} 失败={errors}"
            )

    # 合并保存
    if all_dfs:
        _save_merged(storage, "stk_holdernumber", all_dfs, existing_df, dedup_cols)

    elapsed_total = time.time() - t0
    logger.info(
        f"[stk_holdernumber] 完成: 成功={success} 空={empty} 失败={errors} "
        f"({len(month_ranges)} 个月, 耗时={elapsed_total:.0f}秒)"
    )


def download_moneyflow_hsgt(
    client: TushareClient,
    storage: Storage,
    trade_cal: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载北向资金（按日分区, 市场级单条记录）"""
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    logger.info(f"[moneyflow_hsgt] {len(trading_dates)} 个交易日, 分段批量下载 (接口单次上限 300 条)")

    # 接口单次上限约 300 条 (~14 个月交易日), 按半年切段拉取
    all_dfs: List[pd.DataFrame] = []
    seg_starts = _generate_month_periods(start_date, end_date)
    # 合并成半年段: 每 6 个月一段
    segments: List[tuple] = []
    i = 0
    while i < len(seg_starts):
        seg_start = seg_starts[i][0]
        j = min(i + 5, len(seg_starts) - 1)
        seg_end = seg_starts[j][1]
        segments.append((seg_start, seg_end))
        i = j + 1

    for s, e in segments:
        try:
            df = client.get_moneyflow_hsgt(start_date=s, end_date=e)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                logger.info(f"  [moneyflow_hsgt] {s}~{e}: {len(df)} 条")
            else:
                logger.debug(f"  [moneyflow_hsgt] {s}~{e}: 无数据")
        except Exception as ex:
            logger.warning(f"[moneyflow_hsgt] {s}~{e} 失败: {ex}")

    if not all_dfs:
        logger.warning("[moneyflow_hsgt] 全部分段返回空数据")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    merged["trade_date"] = merged["trade_date"].astype(str)
    merged = merged.drop_duplicates(subset=["trade_date"], keep="last")

    success = skip = 0
    for td in trading_dates:
        if not force and storage.is_data_exists("raw", "moneyflow_hsgt", td):
            skip += 1
            continue
        sub = merged[merged["trade_date"] == td]
        if len(sub) > 0:
            storage.save_raw_by_date(sub, "moneyflow_hsgt", td)
            success += 1

    logger.info(f"[moneyflow_hsgt] 完成: 新下载={success} 跳过={skip}")


def download_top_list(
    client: TushareClient,
    storage: Storage,
    trade_cal: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载龙虎榜（按日分区, 无数据存空占位避免重复下载）"""
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    logger.info(f"[top_list] {len(trading_dates)} 个交易日")
    success = skip = empty = errors = 0

    for i, td in enumerate(trading_dates, 1):
        try:
            if not force and storage.is_data_exists("raw", "top_list", td):
                skip += 1
                continue
            df = client.get_top_list(trade_date=td)
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, "top_list", td)
                success += 1
            else:
                # 无数据存空占位
                storage.save_raw_by_date(
                    pd.DataFrame(columns=[
                        "trade_date", "ts_code", "net_amount",
                        "net_rate", "amount_rate", "reason",
                    ]),
                    "top_list", td,
                )
                empty += 1
        except Exception as e:
            errors += 1
            logger.warning(f"[top_list] {td} 失败: {e}")

        if i % 100 == 0 or i == len(trading_dates):
            logger.info(
                f"[top_list] [{i}/{len(trading_dates)}] "
                f"新下载={success} 空占位={empty} 跳过={skip} 失败={errors}"
            )

    logger.info(
        f"[top_list] 完成: 新下载={success} 空占位={empty} 跳过={skip} 失败={errors}"
    )


def download_report_rc(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载一致预期研报（按年分页增量, 合并单文件）"""
    existing_df = None
    existing_years: Set[str] = set()
    if not force:
        existing_df = storage.load_raw("report_rc")
        if existing_df is not None and len(existing_df) > 0 and "report_date" in existing_df.columns:
            existing_years = set(
                existing_df["report_date"].astype(str).str[:4].unique()
            )
            logger.info(f"[report_rc] 已有 {len(existing_df)} 条数据, 覆盖 {len(existing_years)} 年")

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    years_to_download = [str(y) for y in range(start_year, end_year + 1) if str(y) not in existing_years]

    if not years_to_download:
        logger.info("[report_rc] 所有年份已存在, 跳过。如需重下请加 --force")
        return

    logger.info(f"[report_rc] 按年批量下载: {len(years_to_download)} 年 ({years_to_download[0]}~{years_to_download[-1]})")

    all_dfs: List[pd.DataFrame] = []
    success = empty = errors = 0
    for y in years_to_download:
        y_start = max(f"{y}0101", start_date)
        y_end = min(f"{y}1231", end_date)
        try:
            df = client.get_report_rc(start_date=y_start, end_date=y_end)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success += 1
                logger.info(f"  [report_rc] {y}: {len(df)} 条")
            else:
                empty += 1
        except Exception as e:
            errors += 1
            logger.warning(f"[report_rc] {y} 失败: {e}")

    if all_dfs:
        _save_merged(
            storage, "report_rc", all_dfs, existing_df,
            dedup_cols=["ts_code", "report_date", "org_name", "author_name"],
        )

    logger.info(f"[report_rc] 完成: 成功={success} 空={empty} 失败={errors}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载原始数据（仅raw层，不触发clean/feature构建）"
    )
    parser.add_argument(
        "--start-date",
        default="20120702",
        help="开始日期，格式YYYYMMDD（默认：20120702）"
    )
    parser.add_argument(
        "--end-date",
        default="20260313",
        help="结束日期，格式YYYYMMDD（默认：20260313）"
    )
    parser.add_argument(
        "--only-basic",
        action="store_true",
        help="仅下载基础数据（trade_cal和stock_basic）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载，即使文件已存在"
    )
    parser.add_argument(
        "--download",
        nargs="*",
        default=None,
        help="指定下载的另类数据集，可多选。"
             "可选值: fina_indicator, margin_detail, stk_holdernumber, "
             "forecast, cyq_perf, express, fund_portfolio, "
             "moneyflow_hsgt, top_list, report_rc, all_alt。"
             "不指定时仅下载基础+日线数据"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="下载日线数据 + 全部另类数据（除 basic 外的所有数据集，等效于 --download all_alt 并同时下载日线）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="（兼容旧参数）等效于不加 --force 的默认断点续传行为"
    )

    args = parser.parse_args()
    
    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()  # 确保配置已加载
    
    logger.info("=" * 60)
    logger.info("开始下载原始数据（raw层）")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} - {args.end_date}")
    logger.info(f"仅下载基础数据: {'是' if args.only_basic else '否'}")
    logger.info(f"强制重新下载: {'是' if args.force else '否'}")
    logger.info(f"另类数据集: {args.download or '无'}")
    logger.info(f"全量下载: {'是' if args.all else '否'}")
    logger.info("=" * 60)
    
    try:
        # 初始化客户端和存储
        client = TushareClient()
        storage = Storage()
        
        # 下载基础数据
        trade_cal = download_basic_data(
            client, storage,
            args.start_date, args.end_date,
            force=args.force
        )
        
        if args.only_basic:
            logger.info("=" * 60)
            logger.info("仅下载基础数据，操作完成！")
            logger.info(f"数据保存位置: {storage.root_path}/raw")
            logger.info("=" * 60)
            sys.exit(0)
        
        # ── 另类数据下载 ──────────────────────────────────────────
        download_set = set(args.download) if args.download else set()

        # --all：下载日线 + 全部另类数据集
        if args.all:
            download_set = set(ALT_DATASETS)
            download_daily_data(
                client, storage, trade_cal,
                args.start_date, args.end_date,
                force=args.force
            )
        elif not download_set:
            # 未指定 --download 时默认下载日线数据；指定了则只下载另类数据
            download_daily_data(
                client, storage, trade_cal,
                args.start_date, args.end_date,
                force=args.force
            )
        if "all_alt" in download_set:
            download_set = set(ALT_DATASETS)

        if download_set:
            # 加载股票列表
            stock_basic = storage.load_raw("stock_basic")
            if stock_basic is None:
                logger.error("未找到 stock_basic 数据，请先运行默认下载")
                sys.exit(1)
            stock_codes = sorted(stock_basic["ts_code"].unique().tolist())

            # 财务指标（按季度批量，fina_indicator_vip）
            if "fina_indicator" in download_set:
                download_by_period(
                    client, storage,
                    dataset_name="fina_indicator",
                    api_name="fina_indicator_vip",
                    start_date=args.start_date,
                    end_date=args.end_date,
                    dedup_cols=["ts_code", "end_date", "ann_date"],
                    fields="ts_code,ann_date,end_date,roe_waa,or_yoy,netprofit_yoy,debt_to_assets,q_gr_yoy",
                    force=args.force,
                )

            # 融资融券明细（按日分区）
            if "margin_detail" in download_set:
                download_margin_detail(
                    client, storage, trade_cal,
                    args.start_date, args.end_date,
                    force=args.force,
                )

            # 股东人数（按月批量，单次限3000条）
            if "stk_holdernumber" in download_set:
                download_stk_holdernumber(
                    client, storage,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    dedup_cols=["ts_code", "end_date"],
                    force=args.force,
                )

            # 业绩预告（按季度批量，forecast_vip）
            if "forecast" in download_set:
                download_by_period(
                    client, storage,
                    dataset_name="forecast",
                    api_name="forecast_vip",
                    start_date=args.start_date,
                    end_date=args.end_date,
                    dedup_cols=["ts_code", "end_date", "ann_date"],
                    force=args.force,
                )

            # 筹码胜率（按日分区，5000 积分）
            if "cyq_perf" in download_set:
                download_cyq_perf(
                    client, storage, trade_cal,
                    args.start_date, args.end_date,
                    force=args.force,
                )

            # 业绩快报（按季度批量，express_vip）
            if "express" in download_set:
                download_by_period(
                    client, storage,
                    dataset_name="express",
                    api_name="express_vip",
                    start_date=args.start_date,
                    end_date=args.end_date,
                    dedup_cols=["ts_code", "end_date", "ann_date"],
                    force=args.force,
                )

            # 基金持仓（按季度分区，fund_portfolio）
            if "fund_portfolio" in download_set:
                download_by_period(
                    client, storage,
                    dataset_name="fund_portfolio",
                    api_name="fund_portfolio",
                    start_date=args.start_date,
                    end_date=args.end_date,
                    dedup_cols=["ts_code", "symbol", "end_date"],
                    force=args.force,
                    page_limit=8000,  # fund_portfolio API 单次上限8000条
                    partition_by_period=True,
                )

            # 北向资金（按日分区, 市场级广播）
            if "moneyflow_hsgt" in download_set:
                download_moneyflow_hsgt(
                    client, storage, trade_cal,
                    args.start_date, args.end_date,
                    force=args.force,
                )

            # 龙虎榜（按日分区）
            if "top_list" in download_set:
                download_top_list(
                    client, storage, trade_cal,
                    args.start_date, args.end_date,
                    force=args.force,
                )

            # 一致预期研报（按年分页增量）
            if "report_rc" in download_set:
                download_report_rc(
                    client, storage,
                    args.start_date, args.end_date,
                    force=args.force,
                )

        logger.info("=" * 60)
        logger.info("原始数据下载完成！")
        logger.info(f"数据保存位置: {storage.root_path}/raw")
        logger.info("=" * 60)
        
    except (ValueError, ConnectionError, TimeoutError) as e:
        logger.error("=" * 60)
        logger.error("数据下载失败")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("请按以下步骤配置TuShare token:")
        logger.error("1. 访问 https://tushare.pro/register 注册账号")
        logger.error("2. 获取token")
        logger.error("3. 创建 .env 文件（参考 .env.example）")
        logger.error("4. 在 .env 文件中设置: TS_TOKEN=your_token_here")
        logger.error("=" * 60)
        sys.exit(1)
        
    except Exception as e:
        logger.exception(f"数据下载过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
