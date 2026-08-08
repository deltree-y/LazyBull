# -*- coding: utf-8 -*-
"""ensure 子包：按日/季度分区资产类因子的历史数据补齐。"""

import gc
from typing import List, Optional

import pandas as pd
from loguru import logger

from ...common.date_utils import is_recent_date_str
from ...data import Storage, TushareClient
from .bulk import _generate_quarter_periods, _query_with_pagination


def _try_ensure_historical_cyq_perf(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内缺失的筹码胜率历史数据

    cyq_perf 按日分区存储，需要 20+ 天历史数据才能计算胜率变化率。
    使用 trade_date 参数一次获取全市场当日数据。

    注意: 调用方必须确保 trading_dates_str 只包含 <= trade_date 的历史日期，
    避免下载未来数据导致前视偏差。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD，仅历史日期）

    Returns:
        合并后的 cyq_perf DataFrame，或 None
    """
    downloaded = 0
    for dt in trading_dates_str:
        if storage.is_data_exists("raw", "cyq_perf", dt):
            continue
        try:
            df = client.get_cyq_perf(trade_date=dt)
            if df is not None and not df.empty:
                storage.save_raw_by_date(df, "cyq_perf", dt)
                downloaded += 1
        except Exception as e:
            logger.debug(f"cyq_perf {dt} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"筹码胜率历史补齐: 新增 {downloaded} 个交易日")

    # 重新加载完整范围
    if trading_dates_str:
        from ...data.loader import DataLoader

        loader = DataLoader(storage)
        return loader.load_cyq_perf(trading_dates_str[0], trading_dates_str[-1])
    return None


def _try_ensure_historical_fund_portfolio(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内所需的基金持仓季度分区数据

    fund_portfolio 按季度（end_date=季度末）分区存储。
    根据 trading_dates 覆盖的时间范围，向前回溯 2 年（因子需要历史持仓），
    检查每个季度分区是否存在，缺失则通过 API 下载。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD）

    Returns:
        合并后的 fund_portfolio DataFrame，或 None
    """
    if not trading_dates_str:
        return None

    # 根据交易日范围确定需要的季度
    import datetime as _dt

    min_date = min(trading_dates_str)
    max_date = max(trading_dates_str)
    # 回溯 1 年获取历史持仓（point-in-time 只需最近季报，缩短以降低内存占用）
    start_year = int(min_date[:4]) - 1
    end_year = int(max_date[:4])
    periods = _generate_quarter_periods(start_year, end_year)

    # 只保留 <= max_date 的季度（未来季度无数据）
    periods = [p for p in periods if p <= max_date]

    downloaded = 0
    for period in periods:
        if storage.is_data_exists("raw", "fund_portfolio", period):
            continue
        try:
            df = _query_with_pagination(
                client, "fund_portfolio", page_limit=8000, period=period,
            )
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, "fund_portfolio", period)
                downloaded += 1
        except Exception as e:
            logger.debug(f"fund_portfolio {period} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"基金持仓历史补齐: 新增 {downloaded} 个季度")

    # 逐分区加载+聚合，避免一次性加载全量原始数据（可达百万行级）
    from ...factors.fund_portfolio import FUND_PORTFOLIO_RAW_COLS, _aggregate_fund_portfolio

    agg_dfs = []
    agg_dataset_name = "fund_portfolio_agg"
    for period in periods:
        agg = None
        if storage.is_data_exists("raw", agg_dataset_name, period):
            agg = storage.load_raw_by_date(agg_dataset_name, period)
        elif not storage.is_data_exists("raw", "fund_portfolio", period):
            continue
        else:
            raw_df = storage.load_raw_by_date(
                "fund_portfolio",
                period,
                columns=FUND_PORTFOLIO_RAW_COLS,
            )
            if raw_df is not None and len(raw_df) > 0:
                agg = _aggregate_fund_portfolio(raw_df)
                if agg is not None and len(agg) > 0:
                    storage.save_raw_by_date(agg, agg_dataset_name, period)
            raw_df = None
            gc.collect()

        if agg is not None and len(agg) > 0:
            agg_dfs.append(agg)
            agg = None
            gc.collect()

    if not agg_dfs:
        return None
    result = pd.concat(agg_dfs, ignore_index=True)
    logger.info(f"基金持仓: 逐分区聚合完成，{len(periods)} 个季度 → {len(result)} 条个股记录")
    return result


def _try_ensure_historical_margin(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内缺失的融资融券历史数据

    margin_detail 按日分区存储，需要 20+ 天历史数据才能计算滚动变化率。
    遍历每个交易日，若分区不存在则单独下载并保存。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD）

    Returns:
        合并后的 margin_detail DataFrame，或 None
    """
    downloaded = 0
    for dt in trading_dates_str:
        if storage.is_data_exists("raw", "margin_detail", dt):
            continue
        try:
            df = client.query("margin_detail", trade_date=dt)
            if df is not None and not df.empty:
                storage.save_raw_by_date(df, "margin_detail", dt)
                downloaded += 1
        except Exception as e:
            logger.debug(f"margin_detail {dt} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"融资融券历史补齐: 新增 {downloaded} 个交易日")

    # 重新加载完整范围
    if trading_dates_str:
        from ...data.loader import DataLoader

        loader = DataLoader(storage)
        return loader.load_margin_detail(trading_dates_str[0], trading_dates_str[-1])
    return None


def _try_ensure_historical_moneyflow_hsgt(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内缺失的北向资金历史数据

    moneyflow_hsgt 按日分区存储, 需要 20+ 天历史才能计算 z-score 与 streak。
    支持单次按 start_date/end_date 批量拉取（优于逐日循环）。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD, 仅历史日期）

    Returns:
        合并后的 moneyflow_hsgt DataFrame, 或 None
    """
    if not trading_dates_str:
        return None

    missing_dates = [
        dt for dt in trading_dates_str
        if not storage.is_data_exists("raw", "moneyflow_hsgt", dt)
    ]
    if missing_dates:
        # moneyflow_hsgt 单次返回上限 300 条 (约 14 个月), 需按半年分段拉取以覆盖长历史
        missing_set = set(missing_dates)
        seg_start = missing_dates[0]
        seg_end = missing_dates[-1]
        # 生成半年段 (从 seg_start 往后, 每 6 个日历月一段)
        from datetime import datetime, timedelta
        segments: List[tuple] = []
        cursor = datetime.strptime(seg_start, "%Y%m%d")
        end_dt = datetime.strptime(seg_end, "%Y%m%d")
        while cursor <= end_dt:
            nxt = cursor + timedelta(days=180)
            if nxt > end_dt:
                nxt = end_dt
            segments.append((cursor.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")))
            cursor = nxt + timedelta(days=1)

        saved = 0
        for s, e in segments:
            try:
                df = client.get_moneyflow_hsgt(start_date=s, end_date=e)
                if df is None or df.empty:
                    continue
                df["trade_date"] = (
                    df["trade_date"].astype(str).str.replace("-", "").str[:8]
                )
                for dt, grp in df.groupby("trade_date"):
                    if dt in missing_set:
                        storage.save_raw_by_date(grp, "moneyflow_hsgt", dt)
                        saved += 1
            except Exception as e:
                logger.warning(f"北向资金 {s}~{e} 分段下载失败: {e}")
        if saved > 0:
            logger.info(f"北向资金历史补齐: 新增 {saved} 个交易日")

    from ...data.loader import DataLoader
    loader = DataLoader(storage)
    return loader.load_moneyflow_hsgt(
        trading_dates_str[0], trading_dates_str[-1]
    )


# 龙虎榜"近期空占位"重新查询窗口（自然日）: 已被过早下载成 0 行占位的近期日期
# 在窗口内每次运行都重新查询一次, 修复已落盘的假空分区
_TOP_LIST_REDOWNLOAD_DAYS = 10


def _is_top_list_empty_placeholder(storage, date_str: str) -> bool:
    """判断某日 top_list 分区是否为 0 行空占位（假空候选, 需重新查询）。"""
    df = storage.load_raw_by_date("top_list", date_str)
    return df is not None and len(df) == 0


def _try_ensure_historical_top_list(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内缺失的龙虎榜历史数据

    top_list 按日分区, 稀疏数据（大多数日期只有个位数到几十条）。
    逐日循环下载, 下载失败或空记录都保存空标记以避免重复尝试。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD, 仅历史日期）

    Returns:
        合并后的 top_list DataFrame, 或 None
    """
    if not trading_dates_str:
        return None

    downloaded = 0
    skipped_recent = 0
    redownloaded = 0
    for dt in trading_dates_str:
        if storage.is_data_exists("raw", "top_list", dt):
            if (
                is_recent_date_str(dt, days=_TOP_LIST_REDOWNLOAD_DAYS)
                and _is_top_list_empty_placeholder(storage, dt)
            ):
                # 近期空占位重新查询（修复已落盘的假空分区）
                redownloaded += 1
            else:
                continue
        try:
            df = client.get_top_list(trade_date=dt)
            if df is not None and not df.empty:
                storage.save_raw_by_date(df, "top_list", dt)
                downloaded += 1
            elif is_recent_date_str(dt):
                # 近期日期数据可能尚未发布, 不落盘空占位, 下次运行重试（防假空）
                skipped_recent += 1
            else:
                # 历史日期确认无上榜: 保存 0 行空占位（与下载脚本 schema 一致,
                # 加载时会被过滤）, 避免重复下载
                storage.save_raw_by_date(
                    pd.DataFrame(
                        columns=[
                            "trade_date",
                            "ts_code",
                            "net_amount",
                            "net_rate",
                            "amount_rate",
                            "reason",
                        ]
                    ),
                    "top_list",
                    dt,
                )
        except Exception as e:
            logger.debug(f"top_list {dt} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"龙虎榜历史补齐: 新增 {downloaded} 个交易日")
    if skipped_recent > 0:
        logger.info(f"龙虎榜历史补齐: 近期 {skipped_recent} 个交易日空响应, 暂不落盘待重试")
    if redownloaded > 0:
        logger.info(f"龙虎榜历史补齐: 近期空占位 {redownloaded} 个交易日重新查询")

    from ...data.loader import DataLoader
    loader = DataLoader(storage)
    return loader.load_top_list(trading_dates_str[0], trading_dates_str[-1])
