# -*- coding: utf-8 -*-
"""raw_download 子包：另类数据下载 (股东人数/北向/龙虎榜/一致预期/现金流)。"""

import threading
from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.data import Storage, TushareClient

from .core import ERROR_COLLECTOR, ProgressTracker, _run_concurrent
from .periodic import (
    _concat_no_warning,
    _generate_month_periods,
    _query_with_pagination,
    _save_merged,
    _to_int_date,
    download_by_period,
)

# report_rc 单请求服务端响应慢 (~5s); 并发过高 (16) 在长期运行下会让 TuShare
# 拒绝请求 (返回"查询数据失败"), 使用保守并发 + 接口级限频 (core.py) 保持稳定
_REPORT_RC_CONCURRENCY = 16


def download_stk_holdernumber(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    dedup_cols: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """按月批量下载股东人数数据。"""
    if dedup_cols is None:
        dedup_cols = ["ts_code", "end_date"]

    month_ranges = _generate_month_periods(start_date, end_date)
    if not month_ranges:
        logger.warning("[stk_holdernumber] 日期范围无效")
        return

    existing_df = None
    latest_ann = None
    if not force:
        existing_df = storage.load_raw("stk_holdernumber")
        if existing_df is not None and len(existing_df) > 0:
            logger.info(f"[stk_holdernumber] 已有 {len(existing_df)} 条数据")
            if "ann_date" in existing_df.columns:
                ann_dates = existing_df["ann_date"].astype(str).str.replace("-", "").str[:8]
                ann_dates = ann_dates[ann_dates.str.match(r"^\d{8}$", na=False)]
                if len(ann_dates) > 0:
                    latest_ann = ann_dates.max()

    if latest_ann is not None:
        # 断点续传：只下 ann_date 大于已有最大公告日的月份段，避免每次全量重下
        month_ranges = [mr for mr in month_ranges if mr[1] > latest_ann]
        if not month_ranges:
            logger.info(
                f"[stk_holdernumber] 已有数据覆盖至 {latest_ann}，无需增量。如需重下加 --force"
            )
            return
        logger.info(
            f"[stk_holdernumber] 断点续传：已有最新公告日 {latest_ann}，"
            f"待下 {len(month_ranges)} 个月"
        )

    logger.info(
        f"[stk_holdernumber] 按月下载: {len(month_ranges)} 月 "
        f"({month_ranges[0][0]}~{month_ranges[-1][1]})"
    )

    tracker = ProgressTracker(len(month_ranges), label="stk_holdernumber", log_every=12)
    all_dfs: List["pd.DataFrame"] = []
    success = empty = 0

    for m_start, m_end in month_ranges:
        try:
            # 单月分页拉取，规避 stk_holdernumber 单次 3000 条上限截断
            df = _query_with_pagination(
                client,
                "stk_holdernumber",
                page_limit=3000,
                start_date=m_start,
                end_date=m_end,
            )
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success += 1
            else:
                empty += 1
        except Exception as e:
            ERROR_COLLECTOR.add("stk_holdernumber", f"{m_start}~{m_end}", str(e))
        tracker.tick(extra_info=f"ok={success} empty={empty}")

    if all_dfs:
        _save_merged(
            storage,
            "stk_holdernumber",
            all_dfs,
            existing_df,
            dedup_cols,
            sort_cols=["ann_date", "end_date"],
        )

    logger.info(f"[stk_holdernumber] 完成: 成功={success} 空={empty}")


def download_moneyflow_hsgt(
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载北向资金 (按日分区)。

    修复 #3: 先检查哪些交易日尚未落盘, 仅针对缺失日期计算需要拉取的半年分段,
    避免"分段已全下、逐日写入才发现都跳过"的浪费。
    """
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    if not trading_dates:
        logger.warning("[moneyflow_hsgt] 区间无交易日, 跳过")
        return

    # 先筛待下载日期
    if force:
        pending_dates = list(trading_dates)
    else:
        pending_dates = [
            td for td in trading_dates if not storage.is_data_exists("raw", "moneyflow_hsgt", td)
        ]

    skip = len(trading_dates) - len(pending_dates)
    logger.info(
        f"[moneyflow_hsgt] 共 {len(trading_dates)} 个交易日, 跳过 {skip}, 待下 {len(pending_dates)}"
    )
    if not pending_dates:
        return

    # 基于 pending_dates 的首末日决定拉取窗口, 按半年切分
    pd_start, pd_end = pending_dates[0], pending_dates[-1]
    months = _generate_month_periods(pd_start, pd_end)
    segments: List[Tuple[str, str]] = []
    i = 0
    while i < len(months):
        seg_start = months[i][0]
        j = min(i + 5, len(months) - 1)
        seg_end = months[j][1]
        segments.append((seg_start, seg_end))
        i = j + 1

    logger.info(f"[moneyflow_hsgt] 将拉取 {len(segments)} 个半年分段覆盖待下载日期")

    all_dfs: List["pd.DataFrame"] = []
    for s, e in segments:
        try:
            df = client.get_moneyflow_hsgt(start_date=s, end_date=e)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                logger.info(f"  [moneyflow_hsgt] {s}~{e}: {len(df)} 条")
        except Exception as ex:
            ERROR_COLLECTOR.add("moneyflow_hsgt", f"{s}~{e}", str(ex))

    if not all_dfs:
        logger.warning("[moneyflow_hsgt] 全部分段返回空")
        return

    # 原样合并, 仅屏蔽 pandas 的 empty/all-NA concat 告警
    merged = _concat_no_warning(all_dfs)
    merged["trade_date"] = merged["trade_date"].astype(str)
    merged = merged.drop_duplicates(subset=["trade_date"], keep="last")

    tracker = ProgressTracker(len(pending_dates), label="moneyflow_hsgt_write", log_every=200)
    success = 0
    for td in pending_dates:
        sub = merged[merged["trade_date"] == td]
        if len(sub) > 0:
            storage.save_raw_by_date(sub, "moneyflow_hsgt", td)
            success += 1
        tracker.tick(extra_info=f"ok={success}")

    logger.info(f"[moneyflow_hsgt] 完成: 新下载={success} 跳过={skip}")


def download_top_list(
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载龙虎榜 (按日分区, 无数据存空占位避免重复下载)。"""
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    if not trading_dates:
        logger.warning("[top_list] 区间无交易日, 跳过")
        return

    pending = [
        td for td in trading_dates if force or not storage.is_data_exists("raw", "top_list", td)
    ]
    skip = len(trading_dates) - len(pending)

    logger.info(f"[top_list] 共 {len(trading_dates)} 天, 跳过 {skip}, 待下 {len(pending)}")
    if not pending:
        return

    tracker = ProgressTracker(len(pending), label="top_list", log_every=100)
    counters = {"success": 0, "empty": 0}
    counter_lock = threading.Lock()

    def _worker(td: str) -> None:
        try:
            df = client.get_top_list(trade_date=td)
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, "top_list", td)
                with counter_lock:
                    counters["success"] += 1
            else:
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
                    td,
                )
                with counter_lock:
                    counters["empty"] += 1
        except Exception as e:
            ERROR_COLLECTOR.add("top_list", td, str(e))
        tracker.tick(extra_info=f"ok={counters['success']} empty={counters['empty']}")

    _run_concurrent(pending, _worker, label="top_list")

    logger.info(f"[top_list] 完成: 新下载={counters['success']} 空占位={counters['empty']}")


def _mid_date_str(start_date: str, end_date: str) -> str:
    """返回 [start_date, end_date] 的中点日期 (YYYYMMDD)。"""
    s = datetime.strptime(start_date, "%Y%m%d")
    e = datetime.strptime(end_date, "%Y%m%d")
    return (s + (e - s) / 2).strftime("%Y%m%d")


def _next_date_str(date_str: str) -> str:
    """返回 date_str 的次日 (YYYYMMDD)。"""
    return (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")


def _is_report_rc_overlimit_error(err_msg: str) -> bool:
    """判断 report_rc 错误是否为"单次查询超限"。

    超限 (offset > 100000) 时 TuShare 返回"查询数据失败，请确认参数！"；
    网络超时/代理错误 (Read timed out) 等其它错误不是超限, 不应触发二分
    (二分对全局性问题无意义, 反而浪费时间递归)。
    """
    return "查询数据失败" in err_msg or "请确认参数" in err_msg


def _query_report_rc_adaptive(
    client: TushareClient,
    start_date: str,
    end_date: str,
    page_limit: int = 2000,
    depth: int = 0,
    max_depth: int = 6,
) -> pd.DataFrame:
    """查询 report_rc 日期范围；单次查询超限时自动二分重试。

    report_rc 接口对"一次查询 (start_date/end_date + offset 翻页)"的总行数上限
    为 100000 条 (offset 上限 100000)。整段查询超过该上限时, 继续翻页会返回
    "查询数据失败, 请确认参数！" (实测 2009 年在 offset=102000 失败, 2020/2023
    等年份约 20~30 万条同样会触发)。本函数在整段查询失败时把日期范围二分递归,
    保证任意规模数据都能取全, 已下载小年份 (整年 < 100000 条) 零额外开销。

    Args:
        client: TuShare 客户端
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        page_limit: 单页行数 (report_rc 单次上限 2000)
        depth: 当前递归深度 (内部使用)
        max_depth: 最大递归深度, 超过后抛出 (防无限递归, 2^6=64 段足够覆盖任何规模)

    Returns:
        区间内的 report_rc DataFrame (可能为空)
    """
    if start_date > end_date:
        return pd.DataFrame()
    try:
        return _query_with_pagination(
            client,
            "report_rc",
            page_limit=page_limit,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        if not _is_report_rc_overlimit_error(str(e)):
            # 非超限错误 (如代理/网络 Read timed out): 二分无意义, 直接上抛,
            # 由 download_report_rc 记录该年份失败, 重跑时断点续传
            raise
        if depth >= max_depth:
            raise RuntimeError(
                f"report_rc {start_date}~{end_date} 二分 {max_depth} 层后仍失败: {e}"
            ) from e
        # 大年份单次查询超限 -> 自动二分是预期正常流程, 用 debug 而非 warning
        logger.debug(
            f"[report_rc] {start_date}~{end_date} 整段查询失败, 自动二分重试 "
            f"(depth={depth + 1}): {e}"
        )
        mid = _mid_date_str(start_date, end_date)
        left = _query_report_rc_adaptive(client, start_date, mid, page_limit, depth + 1, max_depth)
        right = _query_report_rc_adaptive(
            client, _next_date_str(mid), end_date, page_limit, depth + 1, max_depth
        )
        parts = [d for d in (left, right) if d is not None and len(d) > 0]
        if not parts:
            return pd.DataFrame()
        # 二分结果合并同样需屏蔽 pandas 的 empty/all-NA concat 告警
        return _concat_no_warning(parts)


def _existing_report_rc_years(storage: Storage) -> Set[str]:
    """从 raw/report_rc 年分区中提取已有年份集合 (分区文件名 YYYY-MM-DD, 取 YYYY)。"""
    years: Set[str] = set()
    for partition in storage.list_partitions("raw", "report_rc"):
        if len(partition) == 10 and partition[4] == "-":
            years.add(partition[:4])
    return years


def download_report_rc(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载卖方研报一致预期 (按年分页增量, 每年独立分区落盘)。

    report_rc 按 report_date 年分区存储: 目录结构 data/raw/report_rc/{YYYY}-12-31.parquet,
    每年一份, 与"按年下载/断点续传"的既有节奏一致; 并发下各年份写独立文件天然线程安全。

    修复 #8: --force 模式下丢弃已有年份, 语义与其它函数一致 (强制重下不保留旧数据)。
    """
    existing_years: Set[str] = set()
    if not force:
        existing_years = _existing_report_rc_years(storage)
        if existing_years:
            logger.info(f"[report_rc] 已有 {len(existing_years)} 个年份分区")

    start_year = _to_int_date(start_date) // 10000
    end_year = _to_int_date(end_date) // 10000
    years_to_download = [
        str(y) for y in range(start_year, end_year + 1) if force or str(y) not in existing_years
    ]

    if not years_to_download:
        logger.info("[report_rc] 全部年份已存在, 跳过。如需重下加 --force")
        return

    logger.info(
        f"[report_rc] 按年下载 {len(years_to_download)} 年 "
        f"({years_to_download[0]}~{years_to_download[-1]})"
    )

    tracker = ProgressTracker(len(years_to_download), label="report_rc", log_every=1)
    success = empty = 0
    # 并发下 success/empty 计数需要线程保护; tracker.tick 内部自带锁, 可安全并发
    stats_lock = threading.Lock()

    def _worker(year: str) -> None:
        """下载单个年份 report_rc (含超限自动二分) 并独立落盘。"""
        nonlocal success, empty
        y_start = max(f"{year}0101", start_date)
        y_end = min(f"{year}1231", end_date)
        try:
            # 按年分页拉取, 规避 report_rc 单次 2000 条上限截断;
            # 单次查询总行数上限 100000 条, 超限年份 (如 2009 起多数年份)
            # 由 _query_report_rc_adaptive 自动二分分片下载
            df = _query_report_rc_adaptive(client, y_start, y_end)
            if df is not None and len(df) > 0:
                with stats_lock:
                    success += 1
                logger.info(f"  [report_rc] {year}: {len(df)} 条")
                # 每年独立分区落盘 (线程安全, 不同年份写不同文件)
                storage.save_raw_by_date(df, "report_rc", f"{year}-12-31")
            else:
                with stats_lock:
                    empty += 1
        except Exception as e:
            ERROR_COLLECTOR.add("report_rc", f"year={year}", str(e))
        finally:
            with stats_lock:
                info = f"ok={success} empty={empty}"
            tracker.tick(extra_info=info)

    # 按年并发下载: 不同年份的网络等待并行化 (总 QPS 仍受令牌桶限频约束);
    # 各年份写独立分区文件, 无需收集合并。
    #
    # report_rc 单请求服务端响应 ~5s, 全局限频 48 并发会让本地代理
    # (如 192.168.1.21:18081) 或 TuShare 服务端出现 Read timed out /
    # "查询数据失败" 全局性失败。这里使用保守并发, 优先保证稳定不失败。
    _run_concurrent(
        years_to_download,
        _worker,
        label="report_rc",
        max_workers=_REPORT_RC_CONCURRENCY,
    )

    logger.info(f"[report_rc] 完成: 成功={success} 空={empty}")


def download_cashflow(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载现金流量表数据 (cashflow_vip, 5000积分)。按报告期批量下载全市场数据。"""
    download_by_period(
        client,
        storage,
        dataset_name="cashflow",
        api_name="cashflow_vip",
        start_date=start_date,
        end_date=end_date,
        dedup_cols=["ts_code", "end_date", "ann_date"],
        fields=None,
        force=force,
        partition_by_period=True,
        sort_cols=["end_date", "ann_date"],
    )
