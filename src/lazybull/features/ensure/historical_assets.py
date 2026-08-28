# -*- coding: utf-8 -*-
"""ensure 子包：按日/季度分区资产类因子的历史数据补齐。"""

import gc
from typing import List, Optional

import pandas as pd
from loguru import logger

from ...common.date_utils import is_recent_date_str
from ...data import Storage, TushareClient
from .bulk import _generate_quarter_periods, _query_with_pagination
from .concat_utils import _concat_no_warning
from .incremental import _get_latest_date


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
            # 下载失败保持警告可见：否则当日分区缺失时 cyq 因子静默缺列，
            # 造成推理侧与训练侧口径不一致（后续 schema 校验仅部分自愈）。
            logger.warning(f"cyq_perf {dt} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"筹码胜率历史补齐: 新增 {downloaded} 个交易日")

    # 重新加载完整范围
    if trading_dates_str:
        from ...data.loader import DataLoader

        loader = DataLoader(storage)
        return loader.load_cyq_perf(trading_dates_str[0], trading_dates_str[-1])
    return None


_FUND_PORTFOLIO_DEDUP_COLS = ["ts_code", "symbol", "end_date"]
# 距报告期末不足该月数视为仍在披露窗口内（年报披露最迟约 4 个月）
_FUND_PORTFOLIO_DISCLOSURE_MONTHS = 4


def _is_fund_portfolio_in_disclosure_window(period: str, max_date: str) -> bool:
    """判断报告期分区是否仍处于披露窗口内（报告期末后 4 个月内）。

    基金季报/半年报/年报在报告期结束后 1~4 个月内陆续披露，窗口内的分区
    可能只是部分快照（仅先披露的基金），需要按覆盖水位持续刷新。
    """
    period_dt = pd.to_datetime(period, format="%Y%m%d")
    max_date_dt = pd.to_datetime(max_date, format="%Y%m%d")
    return period_dt + pd.DateOffset(months=_FUND_PORTFOLIO_DISCLOSURE_MONTHS) >= max_date_dt


def _dedup_fund_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """按 (ts_code, symbol, end_date) 去重，保留 ann_date 最晚的记录。

    同一报告期存在"季报前十大"与"半年报/年报全量"两批公告（同 end_date 不同
    ann_date），与离线 cli.py 的 dedup_cols 口径一致，避免聚合 sum 双重计数。
    """
    cols_present = [c for c in _FUND_PORTFOLIO_DEDUP_COLS if c in df.columns]
    if not cols_present or len(df) == 0:
        return df
    if "ann_date" in df.columns:
        df = df.sort_values("ann_date", kind="stable")
    return df.drop_duplicates(subset=cols_present, keep="last").reset_index(drop=True)


def _try_ensure_historical_fund_portfolio(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内所需的基金持仓季度分区数据

    fund_portfolio 按季度（end_date=季度末）分区存储。
    根据 trading_dates 覆盖的时间范围，向前回溯 1 年（point-in-time 只需最近季报），
    检查每个季度分区是否存在，缺失则通过 API 下载。

    披露季内刷新：基金季报/半年报/年报在报告期结束后数月内陆续披露，分区若在
    披露季中期首次下载会冻结为部分快照。因此对"距报告期末不足 4 个月"的分区，
    只要分区内最新公告日未覆盖到 max_date 就重新下载并覆盖重写，
    并强制重算 fund_portfolio_agg 缓存，避免 serve 侧长期持有残缺数据。

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
    min_date = min(trading_dates_str)
    max_date = max(trading_dates_str)
    # 回溯 1 年获取历史持仓（point-in-time 只需最近季报，缩短以降低内存占用）
    start_year = int(min_date[:4]) - 1
    end_year = int(max_date[:4])
    periods = _generate_quarter_periods(start_year, end_year)

    # 只保留 <= max_date 的季度（未来季度无数据）
    periods = [p for p in periods if p <= max_date]

    # 披露季刷新门控：分区存在但距报告期末不足 4 个月时，按最新公告日覆盖水位
    # 判断是否需要重下；否则分区一旦存在便永久冻结（与训练侧事后全量下载口径分裂）。
    refreshed_periods = set()
    downloaded = 0
    for period in periods:
        exists = storage.is_data_exists("raw", "fund_portfolio", period)
        if exists and not _is_fund_portfolio_in_disclosure_window(period, max_date):
            continue
        if exists:
            existing = storage.load_raw_by_date("fund_portfolio", period, columns=["ann_date"])
            latest_ann = _get_latest_date(existing, "ann_date")
            if latest_ann is not None and latest_ann >= max_date:
                continue  # 披露窗口内但已覆盖到最新公告日，无需重下
        try:
            df = _query_with_pagination(
                client,
                "fund_portfolio",
                page_limit=8000,
                period=period,
            )
            if df is not None and len(df) > 0:
                # 覆盖式重写（不合并旧分区）并去重，刷新分区交给聚合阶段强制重算
                storage.save_raw_by_date(_dedup_fund_portfolio(df), "fund_portfolio", period)
                refreshed_periods.add(period)
                downloaded += 1
        except Exception as e:
            # 下载失败保持警告可见：否则披露季内分区停留在部分快照，
            # fund_hold_ratio 长期低估且 freshness 失真（与 cyq_perf 同类口径风险）。
            logger.warning(f"fund_portfolio {period} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"基金持仓历史补齐: 新增/刷新 {downloaded} 个季度")

    # 逐分区加载+聚合，避免一次性加载全量原始数据（可达百万行级）
    from ...factors.fund_portfolio import FUND_PORTFOLIO_RAW_COLS, _aggregate_fund_portfolio

    agg_dfs = []
    agg_dataset_name = "fund_portfolio_agg"
    for period in periods:
        agg = None
        force_recompute = period in refreshed_periods
        if force_recompute or (
            not storage.is_data_exists("raw", agg_dataset_name, period)
            and storage.is_data_exists("raw", "fund_portfolio", period)
        ):
            raw_df = storage.load_raw_by_date(
                "fund_portfolio",
                period,
                columns=FUND_PORTFOLIO_RAW_COLS,
            )
            if raw_df is not None and len(raw_df) > 0:
                agg = _aggregate_fund_portfolio(raw_df)
                if agg is not None and len(agg) > 0:
                    # 覆盖旧 agg 缓存，确保与新 raw 分区口径一致
                    storage.save_raw_by_date(agg, agg_dataset_name, period)
            raw_df = None
            gc.collect()
        elif storage.is_data_exists("raw", agg_dataset_name, period):
            agg = storage.load_raw_by_date(agg_dataset_name, period)
        else:
            continue

        if agg is not None and len(agg) > 0:
            agg_dfs.append(agg)
            agg = None
            gc.collect()

    if not agg_dfs:
        return None
    result = _concat_no_warning(agg_dfs)
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
        dt for dt in trading_dates_str if not storage.is_data_exists("raw", "moneyflow_hsgt", dt)
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
                # 空响应不落空占位: 北向市场级数据每个交易日必存在（沪港通开通后），
                # 空响应只可能是接口临时故障/停更，落 0 行占位会导致永久丢数。
                # 重复查询范围由调用方裁剪（factor_load 只传近 40 个交易日）控制。
                if df is None or df.empty:
                    continue
                df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]
                for dt, grp in df.groupby("trade_date"):
                    if dt in missing_set:
                        storage.save_raw_by_date(grp, "moneyflow_hsgt", dt)
                        saved += 1
            except Exception as exc:
                logger.warning(f"北向资金 {s}~{e} 分段下载失败: {exc}")
        if saved > 0:
            logger.info(f"北向资金历史补齐: 新增 {saved} 个交易日")

    from ...data.loader import DataLoader

    loader = DataLoader(storage)
    return loader.load_moneyflow_hsgt(trading_dates_str[0], trading_dates_str[-1])


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
            if is_recent_date_str(
                dt, days=_TOP_LIST_REDOWNLOAD_DAYS
            ) and _is_top_list_empty_placeholder(storage, dt):
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
