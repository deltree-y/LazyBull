# -*- coding: utf-8 -*-
"""raw_download 子包：风控公告类数据下载器（pledge_stat / share_float / block_trade）。

三类数据为风控模型专用公告因子（质押/解禁/大宗）的原始数据来源，按各自 API 特性：
  - block_trade : 大宗交易，按 trade_date 逐日查询 → 日分区（对齐 margin_detail/top_list）
  - pledge_stat : 股权质押统计，按 end_date(季末) 查询 → 季分区（对齐 fina_indicator/cashflow）
  - share_float : 限售解禁，按 float_date(解禁日) 区间查询 → 按 ann_date 年分区
                  （PIT 按公告日，对齐 report_rc 年分区）

全部复用现有模板（_download_by_trade_date / _query_with_pagination / _generate_quarter_periods），
本文件仅做数据集的参数映射与分区路由，不重复实现并发/限频/断点续传。
"""

import threading
from typing import List

import pandas as pd
from loguru import logger

from src.lazybull.data import Storage, TushareClient

from .core import ERROR_COLLECTOR, ProgressTracker, _run_concurrent
from .daily_partition import _download_by_trade_date
from .periodic import _concat_no_warning, _generate_quarter_periods

# 积分要求（用户 8000 积分均满足）：block_trade=2000, pledge_stat=5000, share_float=5000
# 接口级限频（次/分钟），走 client.query 令牌桶；值低于 TuShare 官方限频留余量
_API_RATE_LIMITS_RISK = {
    "block_trade": 200,
    "pledge_stat": 120,
    "share_float": 120,
}


def _ensure_rate_limit(api_name: str) -> None:
    """把本模块接口级限频写入 TushareClient 接口限频表（幂等，重复调用无害）。"""
    from src.lazybull.data.tushare_client.core import _API_RATE_LIMITS_DEFAULT

    limit = _API_RATE_LIMITS_RISK.get(api_name)
    if limit is not None:
        # 仅当表内尚无该接口或现有值高于本模块建议值时写入（不覆盖用户更严格的配置）
        existing = _API_RATE_LIMITS_DEFAULT.get(api_name)
        if existing is None or existing > limit:
            _API_RATE_LIMITS_DEFAULT[api_name] = limit


# ═══════════════════════════════════════════════════════════════
# block_trade：按交易日日分区
# ═══════════════════════════════════════════════════════════════


def download_block_trade(
    client: TushareClient,
    storage: Storage,
    trade_cal: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载大宗交易数据（按 trade_date 日分区）。

    复用 `_download_by_trade_date` 模板（与 margin_detail 同模式）：
    逐交易日查询 `block_trade`，落盘 `raw/block_trade/{YYYY-MM-DD}.parquet`。
    """
    _ensure_rate_limit("block_trade")
    _download_by_trade_date(
        "block_trade",
        lambda c, d: c.query("block_trade", trade_date=d),
        client,
        storage,
        trade_cal,
        start_date,
        end_date,
        force,
    )


# ═══════════════════════════════════════════════════════════════
# pledge_stat：按季末 end_date 季分区
# ═══════════════════════════════════════════════════════════════


def download_pledge_stat(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载股权质押统计（按 end_date 季末查询 → 季分区）。

    TuShare `pledge_stat` 参数为 `end_date`（单值，季末），每期返回全市场质押统计
    （约 5000 只 × 13 列）。复用 `_generate_quarter_periods` 生成季末序列，
    每期查询后落盘 `raw/pledge_stat/{YYYY-MM-DD}.parquet`。
    """
    _ensure_rate_limit("pledge_stat")
    periods = _generate_quarter_periods(start_date, end_date)
    if not periods:
        logger.warning("[pledge_stat] 区间内无有效季度")
        return

    pending = [p for p in periods if force or not storage.is_data_exists("raw", "pledge_stat", p)]
    skip = len(periods) - len(pending)
    logger.info(f"[pledge_stat] 共 {len(periods)} 个季度, 跳过 {skip}, 待下 {len(pending)}")
    if not pending:
        return

    tracker = ProgressTracker(len(pending), label="pledge_stat", log_every=4)
    counters = {"success": 0, "empty": 0}
    counter_lock = threading.Lock()

    def _worker(period: str) -> None:
        try:
            df = client.query("pledge_stat", end_date=period)
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, "pledge_stat", period)
                with counter_lock:
                    counters["success"] += 1
                logger.info(f"  [pledge_stat] {period}: {len(df)} 条")
            else:
                with counter_lock:
                    counters["empty"] += 1
        except Exception as e:
            ERROR_COLLECTOR.add("pledge_stat", f"end_date={period}", str(e))
        tracker.tick(extra_info=f"ok={counters['success']} empty={counters['empty']}")

    _run_concurrent(pending, _worker, label="pledge_stat")
    logger.info(f"[pledge_stat] 完成: 成功={counters['success']} 空={counters['empty']}")


# ═══════════════════════════════════════════════════════════════
# share_float：按 float_date 区间查询 → 按 ann_date 年分区（PIT）
# ═══════════════════════════════════════════════════════════════


def _generate_year_ranges(start_date: str, end_date: str) -> List[str]:
    """生成 [start_date, end_date] 覆盖的所有年份 YYYY 列表。"""
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    return [str(y) for y in range(start_year, end_year + 1)]


# share_float 下载保守并发：按 ann_date 逐交易日查询（单日单次请求，无分页），
# 令牌桶限频（120/分钟）自动节流，并发 8 在限频下安全
_SHARE_FLOAT_CONCURRENCY = 8


def _query_share_float_by_ann_date(client: TushareClient, date: str) -> pd.DataFrame:
    """查询 ann_date（公告日）为 [date] 的解禁记录。

    TuShare `share_float` 的 `ann_date` 为单值日期参数（YYYYMMDD），每天公告
    仅约 10-20 条，单次查询即取全（无翻页）。按 ann_date 逐日查询直接命中
    PIT 分区目标（分区=公告年），且数据量为公告级（每天 10 条），远小于按
    float_date 查询的解禁明细（每月 4 万+ 条），无需 offset 深翻页。
    """
    return client.query("share_float", ann_date=date)


def download_share_float(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载限售解禁数据（按 ann_date 年分区，PIT 按公告日）。

    PIT 契约：解禁事件在 `ann_date`（公告日）当天起才可见，故按 `ann_date` 分区
    （`raw/share_float/{YYYY}-12-31.parquet`），因子侧以 `ann_date <= T` 过滤。

    查询方式：TuShare `share_float` 的 `ann_date` 为单值参数（每天公告约 10-20 条，
    单次查询即取全），且 start_date/end_date 语义是 **float_date（解禁日）**——
    按 float_date 查询返回解禁明细（每月 4 万+ 条）且需 offset 深翻页（上限 100000，
    会报\"查询数据失败\"）。故改为**按 ann_date 逐交易日查询**：用交易日历生成目标
    公告年内的交易日，逐日 `ann_date=YYYYMMDD` 单次查询，直接命中 PIT 分区目标，
    无翻页、数据量为公告级。返回记录按 ann_date 年份分组写入对应分区。
    """
    _ensure_rate_limit("share_float")
    target_years = _generate_year_ranges(start_date, end_date)
    if not target_years:
        logger.warning("[share_float] 区间内无有效年份")
        return

    pending = [
        y
        for y in target_years
        if force or not storage.is_data_exists("raw", "share_float", f"{y}-12-31")
    ]
    skip = len(target_years) - len(pending)
    logger.info(
        f"[share_float] 目标公告年 {len(target_years)} 个, 跳过 {skip}, 待下 {len(pending)}"
    )
    if not pending:
        return
    target_set = set(pending)

    # 生成目标公告年内的交易日（ann_date 单值，逐交易日查询该日公告）
    trade_cal = client.get_trade_cal(start_date=start_date, end_date=end_date)
    if trade_cal is None or len(trade_cal) == 0:
        logger.warning("[share_float] 未获取到交易日历，无法逐日查询")
        return
    cal = trade_cal.copy()
    if "is_open" in cal.columns:
        cal = cal[cal["is_open"] == 1]
    cal["cal_date"] = cal["cal_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    query_dates = sorted(d for d in cal["cal_date"] if d.isdigit() and len(d) == 8)
    if not query_dates:
        logger.warning("[share_float] 区间内无交易日")
        return
    logger.info(
        f"[share_float] 按 ann_date 逐日查询: {len(query_dates)} 个交易日 "
        f"({query_dates[0]} ~ {query_dates[-1]})"
    )

    tracker = ProgressTracker(len(query_dates), label="share_float", log_every=200)
    counters = {"success": 0, "empty": 0}
    counter_lock = threading.Lock()

    def _worker(date: str):
        try:
            df = _query_share_float_by_ann_date(client, date)
            if df is not None and len(df) > 0:
                with counter_lock:
                    counters["success"] += 1
                return df
            with counter_lock:
                counters["empty"] += 1
            return None
        except Exception as e:
            ERROR_COLLECTOR.add("share_float", f"ann_date={date}", str(e))
            return None
        finally:
            tracker.tick(extra_info=f"ok={counters['success']} empty={counters['empty']}")

    parts = _run_concurrent(
        query_dates,
        _worker,
        label="share_float",
        collect=True,
        max_workers=_SHARE_FLOAT_CONCURRENCY,
    )
    parts = [p for p in parts if p is not None and len(p) > 0]
    if not parts:
        logger.warning("[share_float] 查询无数据，未落盘任何年份分区")
        return

    all_df = _concat_no_warning(parts)
    if "ann_date" not in all_df.columns:
        logger.warning("[share_float] 返回数据缺少 ann_date 列，无法按公告年分区")
        return
    # 按 ann_date 年份分组落盘（分区 = 公告年，PIT 契约）
    all_df = all_df.copy()
    all_df["_ann_year"] = all_df["ann_date"].astype(str).str.replace("-", "", regex=False).str[:4]
    saved = 0
    for year, grp in all_df.groupby("_ann_year", sort=False):
        if year not in target_set:
            continue
        storage.save_raw_by_date(grp.drop(columns="_ann_year"), "share_float", f"{year}-12-31")
        logger.info(f"  [share_float] {year} 公告: {len(grp)} 条")
        saved += 1
    logger.info(
        f"[share_float] 完成: 查询成功={counters['success']} 空={counters['empty']} "
        f"落盘公告年 {saved} 个 (共 {len(all_df)} 条原始记录)"
    )
