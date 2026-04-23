#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载原始数据脚本（仅下载raw层）

功能：
- 仅负责从TuShare拉取原始数据并保存到raw层
- 不触发clean或feature的构建
- 支持force参数强制重新下载已存在的数据
- 支持--download参数选择下载特定数据集
- 全程错误汇总：单条失败不中断，跑完在总结页统一打印所有错误
- 进度打印：基于已下载速率估算 ETA，方便无人值守

数据集：
  基础数据（默认）：trade_cal, stock_basic
  日线数据（默认）：daily, daily_basic, adj_factor, suspend, stk_limit, moneyflow
  另类数据（需指定）：见 ALT_DATASETS

使用示例：
    python scripts/download_raw.py                            # 基础 + 日线
    python scripts/download_raw.py --all                      # 日线 + 全部另类
    python scripts/download_raw.py --download fina_indicator  # 只下指定另类
    python scripts/download_raw.py --download all_alt         # 全部另类(不含日线)
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_config, get_tushare_settings
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

# 日线组需要保持原子性的子数据集（任一失败即视为当日失败，不写盘任一）
DAILY_SUBSETS = ["daily", "daily_basic", "adj_factor", "suspend", "stk_limit", "moneyflow"]

# 全局并发数, 在 main() 里从 base.yaml 读取后设置; 1=串行(退化行为)
# 说明: 并发仅使"网络等待"并行化, 真实 QPS 仍被 TushareClient 全局锁 + rate_limit 严格限制
_DOWNLOAD_CONCURRENCY = 1


# ────────────────────────────────────────────────────────────────────
# 错误汇总：全局收集器，跑完后统一在总结页打印
# ────────────────────────────────────────────────────────────────────

class ErrorCollector:
    """全局错误收集器: 记录每个数据集下每条失败记录，最后统一输出。

    结构: {dataset_name: [(key, error_msg), ...]}
    key 可以是 trade_date / period / year / "{start}~{end}" 等定位键。
    """

    def __init__(self) -> None:
        self._errors: Dict[str, List[Tuple[str, str]]] = {}
        self._lock = threading.Lock()

    def add(self, dataset: str, key: str, msg: str) -> None:
        with self._lock:
            self._errors.setdefault(dataset, []).append((key, msg))
        # 即时打印一条 warning 方便实时观察，最终仍会在总结页汇总
        logger.warning(f"[{dataset}] {key} 失败: {msg}")

    def has_errors(self) -> bool:
        with self._lock:
            return any(self._errors.values())

    def total_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._errors.values())

    def print_summary(self) -> None:
        """在脚本结束时统一打印全部错误，供离线查看日志。"""
        if not self.has_errors():
            logger.info("✓ 全程无任何下载错误")
            return
        logger.error("=" * 70)
        logger.error(f"⚠ 下载过程累计发生错误 {self.total_count()} 条，按数据集列出如下：")
        logger.error("=" * 70)
        for dataset, errs in self._errors.items():
            logger.error(f"[{dataset}] 共 {len(errs)} 条错误:")
            for key, msg in errs:
                logger.error(f"  - {key}: {msg}")
        logger.error("=" * 70)
        logger.error("请检查上述错误；失败项目不会阻断其他数据，可重新运行脚本按断点续传")
        logger.error("=" * 70)


ERROR_COLLECTOR = ErrorCollector()


# ────────────────────────────────────────────────────────────────────
# 进度与 ETA
# ────────────────────────────────────────────────────────────────────

class ProgressTracker:
    """进度追踪器：基于已完成项的平均耗时估算剩余时间 (ETA)。"""

    def __init__(self, total: int, label: str, log_every: int = 50) -> None:
        self.total = max(total, 1)
        self.label = label
        self.log_every = log_every
        self.start_ts = time.time()
        self.done = 0
        self._lock = threading.Lock()

    def tick(self, extra_info: str = "") -> None:
        """完成一项(无论成功/失败/跳过)后调用，按 log_every 间隔打印进度。线程安全。"""
        with self._lock:
            self.done += 1
            should_log = self.done % self.log_every == 0 or self.done == self.total
            done = self.done
        if should_log:
            elapsed = time.time() - self.start_ts
            rate = done / elapsed if elapsed > 0 else 0.0
            remain = self.total - done
            eta_sec = remain / rate if rate > 0 else 0.0
            eta_dt = datetime.now() + timedelta(seconds=eta_sec)
            logger.info(
                f"[{self.label}] [{done}/{self.total}] "
                f"({done / self.total:.1%}) "
                f"elapsed={_fmt_duration(elapsed)} "
                f"rate={rate:.2f}/s "
                f"ETA={_fmt_duration(eta_sec)} "
                f"(预计完成 {eta_dt.strftime('%Y-%m-%d %H:%M:%S')}) "
                f"{extra_info}"
            )


def _fmt_duration(sec: float) -> str:
    """把秒数格式化为 HH:MM:SS。"""
    if sec < 0 or sec > 86400 * 30:
        return "--:--:--"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ────────────────────────────────────────────────────────────────────
# 基础下载：trade_cal / stock_basic
# ────────────────────────────────────────────────────────────────────

def download_basic_data(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> "pd.DataFrame":
    """下载 trade_cal 和 stock_basic。

    注意修复 #2: trade_cal 需要扩展日历窗口时, 必须合并旧数据后再保存，
    否则短窗口调用会截断历史。stock_basic 同时拉取 L/D/P 修复 #12 生存者偏差。
    """
    # 1. 交易日历: 按需合并新旧数据
    logger.info("检查交易日历...")
    existing_cal: Optional[pd.DataFrame] = None
    need_download = force
    if not force:
        existing_cal = storage.load_raw("trade_cal")
        if existing_cal is None or "cal_date" not in existing_cal.columns:
            need_download = True
        else:
            latest = str(existing_cal["cal_date"].astype(str).max()).replace("-", "")[:8]
            earliest = str(existing_cal["cal_date"].astype(str).min()).replace("-", "")[:8]
            # 任一端不覆盖则需要扩展
            if latest < end_date or earliest > start_date:
                need_download = True
                logger.info(
                    f"交易日历需要扩展: 现有 {earliest}~{latest}, "
                    f"目标 {start_date}~{end_date}"
                )
            else:
                logger.info(f"交易日历已覆盖 {earliest}~{latest}, 跳过")

    if need_download:
        # 为了安全, 拉取并集窗口 (min(现有起点, 目标起点) ~ max(现有终点, 目标终点))
        query_start = start_date
        query_end = end_date
        if existing_cal is not None and "cal_date" in existing_cal.columns:
            ex_min = str(existing_cal["cal_date"].astype(str).min()).replace("-", "")[:8]
            ex_max = str(existing_cal["cal_date"].astype(str).max()).replace("-", "")[:8]
            query_start = min(ex_min, start_date)
            query_end = max(ex_max, end_date)
        logger.info(f"下载交易日历 ({query_start}~{query_end})...")
        new_cal = client.get_trade_cal(
            start_date=query_start, end_date=query_end, exchange="SSE"
        )
        if existing_cal is not None and len(existing_cal) > 0:
            new_cal = pd.concat([existing_cal, new_cal], ignore_index=True)
            new_cal = new_cal.drop_duplicates(subset=["cal_date"], keep="last")
            new_cal = new_cal.sort_values("cal_date").reset_index(drop=True)
        storage.save_raw(new_cal, "trade_cal", is_force=True)
        logger.info(f"交易日历已保存: {len(new_cal)} 条")
        trade_cal = new_cal
    else:
        trade_cal = existing_cal

    # 2. 股票基本信息: 同时拉 L/D/P 消除生存者偏差 (#12)
    logger.info("检查股票基本信息...")
    if not force and storage.check_basic_data_freshness("stock_basic", end_date):
        logger.info("股票基本信息已存在, 跳过")
    else:
        logger.info("下载股票基本信息 (L+D+P)...")
        dfs = []
        for status in ("L", "D", "P"):
            try:
                df = client.get_stock_basic(list_status=status)
                if df is not None and len(df) > 0:
                    dfs.append(df)
                    logger.info(f"  list_status={status}: {len(df)} 条")
            except Exception as e:
                ERROR_COLLECTOR.add("stock_basic", f"list_status={status}", str(e))
        if not dfs:
            raise RuntimeError("stock_basic 三种状态全部下载失败, 无法继续")
        stock_basic = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["ts_code"])
        storage.save_raw(stock_basic, "stock_basic", is_force=True)
        logger.info(f"股票基本信息已保存: {len(stock_basic)} 条 (含退市/暂停)")

    return trade_cal


# ────────────────────────────────────────────────────────────────────
# 并发执行器: 把 work_items 按 _DOWNLOAD_CONCURRENCY 并发分发给 worker
# ────────────────────────────────────────────────────────────────────

def _run_concurrent(work_items, worker: Callable, label: str) -> None:
    """并发执行 worker(item) 遍历 work_items。

    - _DOWNLOAD_CONCURRENCY == 1 时走同步路径, 与串行等价, 便于排障/降级
    - 真实并发度仍被 TushareClient 内置的令牌桶限频收紧, 不会触发 QPS 超标
    - worker 内的所有异常由 worker 自己捕获并记录到 ERROR_COLLECTOR,
      此处仅兜底一次避免单个线程崩溃吞掉其余任务
    """
    if _DOWNLOAD_CONCURRENCY <= 1 or len(work_items) <= 1:
        for item in work_items:
            try:
                worker(item)
            except Exception as e:
                ERROR_COLLECTOR.add(label, f"item={item!r}", f"worker 未捕获异常: {e}")
        return

    with ThreadPoolExecutor(
        max_workers=_DOWNLOAD_CONCURRENCY,
        thread_name_prefix=f"dl-{label}",
    ) as pool:
        futures = {pool.submit(worker, item): item for item in work_items}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                fut.result()
            except Exception as e:
                ERROR_COLLECTOR.add(label, f"item={item!r}", f"worker 未捕获异常: {e}")


# ────────────────────────────────────────────────────────────────────
# 日线组: 6 个子接口按日聚合, 任一失败视为当日失败(原子性, 修复 #5)
# ────────────────────────────────────────────────────────────────────

# 各子接口对应的 client 方法 (修复 #4: moneyflow 确实缺失必须报错)
_DAILY_FETCHERS: Dict[str, Callable] = {
    "daily": lambda c, d: c.get_daily(trade_date=d),
    "daily_basic": lambda c, d: c.get_daily_basic(trade_date=d),
    "adj_factor": lambda c, d: c.get_adj_factor(trade_date=d),
    "suspend": lambda c, d: c.get_suspend_d(trade_date=d),
    "stk_limit": lambda c, d: c.get_stk_limit(trade_date=d),
    "moneyflow": lambda c, d: c.get_moneyflow(trade_date=d),
}

# 允许当日无数据的接口 (停牌/涨跌停/资金流在极早期无数据是正常的)
_DAILY_ALLOW_EMPTY = {"suspend", "stk_limit", "moneyflow", "adj_factor"}


def _pending_daily_subsets(storage: Storage, trade_date: str, force: bool) -> List[str]:
    """返回当日还需要下载的子数据集名称。"""
    if force:
        return list(DAILY_SUBSETS)
    return [s for s in DAILY_SUBSETS if not storage.is_data_exists("raw", s, trade_date)]


def download_daily_data(
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载日线数据 (按日期分区, 原子性 + ETA 进度)。

    修复 #5: 单日 6 个接口原子性 —— 只要任一接口抛异常, 整天标记失败;
    已成功拉取的 DataFrame 不落盘, 下次重跑可重新尝试, 避免"半个日子"永久缺失。
    修复 #4: moneyflow 返回空时不再是 error 日志, 而是 raise 被记录到错误汇总。
    修复 #13: len(trading_dates)==0 时直接返回, 防止除零。
    """
    logger.info(f"下载日线数据 ({start_date}~{end_date})...")

    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    if not trading_dates:
        logger.warning(f"日期区间 {start_date}~{end_date} 内无交易日, 跳过日线下载")
        return

    logger.info(f"共 {len(trading_dates)} 个交易日需要检查")

    # 预筛: 已全部落盘的日期直接跳过
    pending: List[Tuple[str, List[str]]] = []
    skip_count = 0
    for td in trading_dates:
        subs = _pending_daily_subsets(storage, td, force)
        if not subs:
            skip_count += 1
        else:
            pending.append((td, subs))

    logger.info(f"跳过已存在: {skip_count} 天, 需要下载: {len(pending)} 天")
    if not pending:
        return

    tracker = ProgressTracker(len(pending), label="daily", log_every=20)
    total_rows: Dict[str, int] = {s: 0 for s in DAILY_SUBSETS}
    fail_days = 0
    # 并发下 total_rows / fail_days / Storage 写盘都需要线程保护
    stats_lock = threading.Lock()

    def _fetch_one_day(trade_date: str, subs: List[str]) -> None:
        """下载单个交易日的全部子接口 (原子性) 并落盘。"""
        nonlocal fail_days
        day_data: Dict[str, "pd.DataFrame"] = {}
        day_failed_reason: Optional[str] = None
        for sub in subs:
            try:
                df = _DAILY_FETCHERS[sub](client, trade_date)
                if df is None or len(df) == 0:
                    if sub in _DAILY_ALLOW_EMPTY:
                        day_data[sub] = pd.DataFrame()  # 占位, 不写盘
                    else:
                        day_failed_reason = f"{sub} 返回空 (强制依赖)"
                        break
                else:
                    day_data[sub] = df
            except Exception as e:
                day_failed_reason = f"{sub} 异常: {e}"
                break

        if day_failed_reason is not None:
            ERROR_COLLECTOR.add("daily", trade_date, day_failed_reason)
            with stats_lock:
                fail_days += 1
            tracker.tick(extra_info=f"fail={fail_days}")
            return

        # 全部接口都成功(或允许空) —— 统一落盘, 保证当日原子性
        # Storage.save_raw_by_date 对不同 (sub, trade_date) 路径写不同文件, 可并发
        for sub, df in day_data.items():
            if len(df) > 0:
                storage.save_raw_by_date(df, sub, trade_date)
                with stats_lock:
                    total_rows[sub] += len(df)

        tracker.tick(extra_info=f"fail={fail_days}")

    _run_concurrent(
        work_items=pending,
        worker=lambda item: _fetch_one_day(item[0], item[1]),
        label="daily",
    )

    logger.info("=" * 60)
    logger.info("日线数据下载完成")
    for sub in DAILY_SUBSETS:
        logger.info(f"  {sub:12s}: 新增 {total_rows[sub]} 条记录")
    logger.info(f"失败天数: {fail_days} (详见最终错误汇总)")
    logger.info("=" * 60)


# ────────────────────────────────────────────────────────────────────
# 按日分区类接口的通用下载 (margin_detail / cyq_perf)
# ────────────────────────────────────────────────────────────────────

def _download_by_trade_date(
    dataset_name: str,
    fetcher: Callable[[TushareClient, str], "pd.DataFrame"],
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """按交易日分区下载模板 (margin_detail, cyq_perf 等共用)。"""
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    if not trading_dates:
        logger.warning(f"[{dataset_name}] 区间无交易日, 跳过")
        return

    # 预筛
    pending = [td for td in trading_dates if force or not storage.is_data_exists("raw", dataset_name, td)]
    skip = len(trading_dates) - len(pending)

    logger.info(f"[{dataset_name}] 共 {len(trading_dates)} 天, 跳过 {skip}, 待下 {len(pending)}")
    if not pending:
        return

    tracker = ProgressTracker(len(pending), label=dataset_name, log_every=100)
    counters = {"success": 0, "empty": 0}
    counter_lock = threading.Lock()

    def _worker(td: str) -> None:
        try:
            df = fetcher(client, td)
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, dataset_name, td)
                with counter_lock:
                    counters["success"] += 1
            else:
                with counter_lock:
                    counters["empty"] += 1
        except Exception as e:
            ERROR_COLLECTOR.add(dataset_name, td, str(e))
        tracker.tick(extra_info=f"ok={counters['success']} empty={counters['empty']}")

    _run_concurrent(pending, _worker, label=dataset_name)

    logger.info(f"[{dataset_name}] 完成: 成功={counters['success']} 空={counters['empty']}")


def download_margin_detail(client, storage, trade_cal, start_date, end_date, force=False):
    _download_by_trade_date(
        "margin_detail",
        lambda c, d: c.query("margin_detail", trade_date=d),
        client, storage, trade_cal, start_date, end_date, force,
    )


def download_cyq_perf(client, storage, trade_cal, start_date, end_date, force=False):
    _download_by_trade_date(
        "cyq_perf",
        lambda c, d: c.get_cyq_perf(trade_date=d),
        client, storage, trade_cal, start_date, end_date, force,
    )


# ────────────────────────────────────────────────────────────────────
# 工具: 日期范围生成
# ────────────────────────────────────────────────────────────────────

def _to_int_date(s: str) -> int:
    """修复 #6: 强制把 YYYYMMDD 字符串转 int 比较, 杜绝字典序陷阱。"""
    s = str(s).strip().replace("-", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"非法日期格式 (需 YYYYMMDD): {s!r}")
    return int(s)


def _generate_quarter_periods(start_date: str, end_date: str) -> List[str]:
    """生成日期范围内覆盖的所有季度末 YYYYMMDD 列表 (数值比较, 修复 #6)。"""
    quarter_ends = ["0331", "0630", "0930", "1231"]
    start_int = _to_int_date(start_date)
    end_int = _to_int_date(end_date)
    start_year = start_int // 10000
    end_year = end_int // 10000
    periods = []
    for year in range(start_year, end_year + 1):
        for qe in quarter_ends:
            p = f"{year}{qe}"
            if start_int <= int(p) <= end_int:
                periods.append(p)
    return periods


def _generate_month_periods(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """生成按月切分的 (start, end) 列表。"""
    import calendar
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    periods: List[Tuple[str, str]] = []
    current = start.replace(day=1)
    while current <= end:
        month_start = current.strftime("%Y%m%d")
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end_dt = current.replace(day=last_day)
        if month_end_dt > end:
            month_end_dt = end
        periods.append((month_start, month_end_dt.strftime("%Y%m%d")))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
    return periods


# ────────────────────────────────────────────────────────────────────
# 合并保存（供按季度/年下载使用）
# ────────────────────────────────────────────────────────────────────

def _save_merged(
    storage: Storage,
    dataset_name: str,
    new_dfs: List["pd.DataFrame"],
    existing_df: Optional["pd.DataFrame"],
    dedup_cols: List[str],
    sort_cols: Optional[List[str]] = None,
) -> None:
    """合并新旧数据并保存。

    修复 #9: 合并前按 sort_cols 排序, dedup 的 keep="last" 才有明确语义。
    """
    new_dfs = [d for d in new_dfs if d is not None and len(d) > 0]
    if not new_dfs:
        result = existing_df if existing_df is not None else pd.DataFrame()
    else:
        result = pd.concat(new_dfs, ignore_index=True)
        if existing_df is not None and len(existing_df) > 0:
            result = pd.concat([existing_df, result], ignore_index=True)

    if dedup_cols and len(result) > 0:
        # 先按 sort_cols (如 ann_date) 升序, 然后 keep="last" 保留最新
        if sort_cols:
            cols_present = [c for c in sort_cols if c in result.columns]
            if cols_present:
                result = result.sort_values(cols_present, kind="stable")
        result = result.drop_duplicates(subset=dedup_cols, keep="last")
        result = result.reset_index(drop=True)

    storage.save_raw(result, dataset_name, is_force=True)
    logger.info(f"[{dataset_name}] 已保存: {len(result)} 条记录")


# ────────────────────────────────────────────────────────────────────
# 带分页的 API 调用 (修复 #7: 恰好整页不再多一次空调用)
# ────────────────────────────────────────────────────────────────────

def _query_with_pagination(
    client: TushareClient,
    api_name: str,
    page_limit: int = 50000,
    fields: Optional[str] = None,
    **kwargs,
) -> "pd.DataFrame":
    """翻页获取全量数据。

    修复 #7: 返回恰好 page_limit 时不再额外探查一次; 通过返回 DataFrame 的实际列
    数量判断是否还可能有下一页: 若 len(df)==page_limit 且第一页就触顶则继续,
    但直接用 len<page_limit 作为终止条件本身是正确的(TuShare 分页行为),
    只是上一版在"恰好整页"时会多一次空请求。这里改为先尝试额外只取 1 条
    探测是否还有下一页, 避免浪费 limit 调用。
    """
    all_pages: List["pd.DataFrame"] = []
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
            break
        # 恰好整页: 用 limit=1 探测是否还有下一页, 避免浪费一次 page_limit 查询
        probe = client.pro.query(
            api_name, fields=fields or "",
            limit=1, offset=offset + page_limit, **kwargs,
        )
        if probe is None or len(probe) == 0:
            break
        offset += page_limit
        logger.debug(f"  [{api_name}] 分页: offset={offset}")

    if not all_pages:
        return pd.DataFrame()
    return pd.concat(all_pages, ignore_index=True)


# ────────────────────────────────────────────────────────────────────
# 按季度批量下载 (fina_indicator / forecast / express / fund_portfolio)
# ────────────────────────────────────────────────────────────────────

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
    sort_cols: Optional[List[str]] = None,
) -> None:
    """按报告期(period)批量下载数据。"""
    periods = _generate_quarter_periods(start_date, end_date)
    if not periods:
        logger.warning(f"[{dataset_name}] 区间内无有效季度")
        return

    existing_df = None
    existing_periods: Set[str] = set()
    if not force:
        if partition_by_period:
            for p in periods:
                if storage.is_data_exists("raw", dataset_name, p):
                    existing_periods.add(p)
            if existing_periods:
                logger.info(f"[{dataset_name}] 已有 {len(existing_periods)} 个季度分区")
        else:
            existing_df = storage.load_raw(dataset_name)
            if existing_df is not None and len(existing_df) > 0:
                if "end_date" in existing_df.columns:
                    existing_periods = set(
                        existing_df["end_date"].astype(str)
                        .str.replace("-", "").str[:8].unique()
                    )
                logger.info(f"[{dataset_name}] 已有 {len(existing_periods)} 个季度数据")

    periods_to_download = [p for p in periods if p not in existing_periods]
    if not periods_to_download:
        logger.info(f"[{dataset_name}] 全部季度已存在, 跳过。如需重下加 --force")
        return

    logger.info(
        f"[{dataset_name}] 按季度下载 {len(periods_to_download)} 个 "
        f"({periods_to_download[0]}~{periods_to_download[-1]})"
    )

    tracker = ProgressTracker(len(periods_to_download), label=dataset_name, log_every=4)
    all_dfs: List["pd.DataFrame"] = []
    success = empty = 0

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
        except Exception as e:
            ERROR_COLLECTOR.add(dataset_name, f"period={period}", str(e))
        tracker.tick(extra_info=f"ok={success} empty={empty}")

    if not partition_by_period and all_dfs:
        _save_merged(
            storage, dataset_name, all_dfs, existing_df,
            dedup_cols, sort_cols=sort_cols or ["ann_date", "end_date"],
        )

    logger.info(f"[{dataset_name}] 完成: 成功={success} 空={empty}")


# ────────────────────────────────────────────────────────────────────
# 股东人数 (按月)
# ────────────────────────────────────────────────────────────────────

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
    if not force:
        existing_df = storage.load_raw("stk_holdernumber")
        if existing_df is not None and len(existing_df) > 0:
            logger.info(f"[stk_holdernumber] 已有 {len(existing_df)} 条数据")

    logger.info(
        f"[stk_holdernumber] 按月下载: {len(month_ranges)} 月 "
        f"({month_ranges[0][0]}~{month_ranges[-1][1]})"
    )

    tracker = ProgressTracker(len(month_ranges), label="stk_holdernumber", log_every=12)
    all_dfs: List["pd.DataFrame"] = []
    success = empty = 0

    for m_start, m_end in month_ranges:
        try:
            df = client.get_stk_holdernumber(start_date=m_start, end_date=m_end)
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
            storage, "stk_holdernumber", all_dfs, existing_df,
            dedup_cols, sort_cols=["ann_date", "end_date"],
        )

    logger.info(f"[stk_holdernumber] 完成: 成功={success} 空={empty}")


# ────────────────────────────────────────────────────────────────────
# 北向资金 (修复 #3: 先筛出真正需要下载的日期, 再决定拉哪些分段)
# ────────────────────────────────────────────────────────────────────

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
        pending_dates = [td for td in trading_dates if not storage.is_data_exists("raw", "moneyflow_hsgt", td)]

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

    merged = pd.concat(all_dfs, ignore_index=True)
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


# ────────────────────────────────────────────────────────────────────
# 龙虎榜 (按日分区)
# ────────────────────────────────────────────────────────────────────

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

    pending = [td for td in trading_dates if force or not storage.is_data_exists("raw", "top_list", td)]
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
                    pd.DataFrame(columns=[
                        "trade_date", "ts_code", "net_amount",
                        "net_rate", "amount_rate", "reason",
                    ]),
                    "top_list", td,
                )
                with counter_lock:
                    counters["empty"] += 1
        except Exception as e:
            ERROR_COLLECTOR.add("top_list", td, str(e))
        tracker.tick(extra_info=f"ok={counters['success']} empty={counters['empty']}")

    _run_concurrent(pending, _worker, label="top_list")

    logger.info(f"[top_list] 完成: 新下载={counters['success']} 空占位={counters['empty']}")


# ────────────────────────────────────────────────────────────────────
# 一致预期研报 (按年分页增量)
# ────────────────────────────────────────────────────────────────────

def download_report_rc(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载卖方研报一致预期 (按年分页增量)。

    修复 #8: --force 模式下丢弃 existing_df, 语义与其它函数一致 (强制重下不保留旧数据)。
    """
    existing_df = None
    existing_years: Set[str] = set()
    if not force:
        existing_df = storage.load_raw("report_rc")
        if existing_df is not None and len(existing_df) > 0 and "report_date" in existing_df.columns:
            existing_years = set(
                existing_df["report_date"].astype(str).str[:4].unique()
            )
            logger.info(f"[report_rc] 已有 {len(existing_df)} 条, 覆盖 {len(existing_years)} 年")

    start_year = _to_int_date(start_date) // 10000
    end_year = _to_int_date(end_date) // 10000
    years_to_download = [
        str(y) for y in range(start_year, end_year + 1)
        if force or str(y) not in existing_years
    ]

    if not years_to_download:
        logger.info("[report_rc] 全部年份已存在, 跳过。如需重下加 --force")
        return

    logger.info(f"[report_rc] 按年下载 {len(years_to_download)} 年 ({years_to_download[0]}~{years_to_download[-1]})")

    tracker = ProgressTracker(len(years_to_download), label="report_rc", log_every=1)
    all_dfs: List["pd.DataFrame"] = []
    success = empty = 0
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
            ERROR_COLLECTOR.add("report_rc", f"year={y}", str(e))
        tracker.tick(extra_info=f"ok={success} empty={empty}")

    if all_dfs:
        _save_merged(
            storage, "report_rc", all_dfs,
            # 修复 #8: force 时 existing_df 传 None
            existing_df if not force else None,
            dedup_cols=["ts_code", "report_date", "org_name", "author_name"],
            sort_cols=["report_date"],
        )

    logger.info(f"[report_rc] 完成: 成功={success} 空={empty}")


# ────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────

def main():
    """主函数"""
    # 修复 #1: 默认 end-date 用"今天" 而非硬编码未来日期
    today_str = datetime.now().strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description="下载原始数据 (仅 raw 层, 不触发 clean/feature 构建)"
    )
    parser.add_argument("--start-date", default="20120702",
                        help="开始日期 YYYYMMDD (默认 20120702)")
    parser.add_argument("--end-date", default=today_str,
                        help=f"结束日期 YYYYMMDD (默认当日 {today_str})")
    parser.add_argument("--only-basic", action="store_true",
                        help="仅下载基础数据 (trade_cal, stock_basic)")
    parser.add_argument("--force", action="store_true",
                        help="强制重新下载, 即使文件已存在")
    parser.add_argument(
        "--download", nargs="*", default=None,
        help="指定另类数据集, 可多选。可选: fina_indicator, margin_detail, "
             "stk_holdernumber, forecast, cyq_perf, express, fund_portfolio, "
             "moneyflow_hsgt, top_list, report_rc, all_alt。不指定时仅下基础+日线"
    )
    parser.add_argument("--all", action="store_true", default=False,
                        help="下载日线 + 全部另类数据")
    # 修复 #10: --resume 此前未使用, 改为从 help 中说明它等价于默认行为
    parser.add_argument("--resume", action="store_true",
                        help="(保留兼容参数, 等价于默认的断点续传行为, 无需单独指定)")
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="并发线程数, 覆盖 base.yaml 中 tushare.download_concurrency。"
             "1=串行(若触发限流可降级); 默认读取配置"
    )

    args = parser.parse_args()

    # 参数校验
    try:
        _to_int_date(args.start_date)
        _to_int_date(args.end_date)
    except ValueError as e:
        logger.error(f"日期参数错误: {e}")
        sys.exit(2)
    if _to_int_date(args.start_date) > _to_int_date(args.end_date):
        logger.error(f"start_date({args.start_date}) > end_date({args.end_date})")
        sys.exit(2)

    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()

    # 从配置 / 命令行读取并发数, 注入全局变量供 _run_concurrent 使用
    global _DOWNLOAD_CONCURRENCY
    ts_settings = get_tushare_settings()
    if args.concurrency is not None:
        _DOWNLOAD_CONCURRENCY = max(1, args.concurrency)
    else:
        _DOWNLOAD_CONCURRENCY = max(1, ts_settings["download_concurrency"])

    logger.info("=" * 70)
    logger.info("开始下载原始数据 (raw 层)")
    logger.info("=" * 70)
    logger.info(f"日期范围    : {args.start_date} ~ {args.end_date}")
    logger.info(f"仅基础数据  : {'是' if args.only_basic else '否'}")
    logger.info(f"强制重下    : {'是' if args.force else '否'}")
    logger.info(f"另类数据集  : {args.download or '无'}")
    logger.info(f"全量下载    : {'是' if args.all else '否'}")
    logger.info(f"并发线程数  : {_DOWNLOAD_CONCURRENCY} (1=串行降级)")
    logger.info(f"限频        : {ts_settings['rate_limit']}次/分钟, "
                f"限流重试等待={ts_settings['retry_rate_limit_sleep']}s")
    logger.info("=" * 70)

    script_start_ts = time.time()
    exit_code = 0

    try:
        client = TushareClient()
        storage = Storage()

        # 1. 基础数据
        trade_cal = download_basic_data(
            client, storage,
            args.start_date, args.end_date,
            force=args.force,
        )

        if args.only_basic:
            logger.info("仅下载基础数据, 完成")
        else:
            download_set: Set[str] = set(args.download) if args.download else set()

            # --all : 日线 + 全部另类
            if args.all:
                download_set = set(ALT_DATASETS)
                download_daily_data(
                    client, storage, trade_cal,
                    args.start_date, args.end_date, force=args.force,
                )
            elif not download_set:
                # 未指定 --download -> 默认下载日线
                download_daily_data(
                    client, storage, trade_cal,
                    args.start_date, args.end_date, force=args.force,
                )
            if "all_alt" in download_set:
                download_set = set(ALT_DATASETS)

            if download_set:
                # 另类数据需要 stock_basic 存在
                stock_basic = storage.load_raw("stock_basic")
                if stock_basic is None:
                    raise RuntimeError("未找到 stock_basic, 请先运行默认下载")

                if "fina_indicator" in download_set:
                    download_by_period(
                        client, storage,
                        dataset_name="fina_indicator",
                        api_name="fina_indicator_vip",
                        start_date=args.start_date, end_date=args.end_date,
                        dedup_cols=["ts_code", "end_date", "ann_date"],
                        fields="ts_code,ann_date,end_date,roe_waa,or_yoy,netprofit_yoy,debt_to_assets,q_gr_yoy",
                        force=args.force,
                        sort_cols=["ann_date", "end_date"],
                    )

                if "margin_detail" in download_set:
                    download_margin_detail(
                        client, storage, trade_cal,
                        args.start_date, args.end_date, force=args.force,
                    )

                if "stk_holdernumber" in download_set:
                    download_stk_holdernumber(
                        client, storage,
                        start_date=args.start_date, end_date=args.end_date,
                        dedup_cols=["ts_code", "end_date"],
                        force=args.force,
                    )

                if "forecast" in download_set:
                    download_by_period(
                        client, storage,
                        dataset_name="forecast",
                        api_name="forecast_vip",
                        start_date=args.start_date, end_date=args.end_date,
                        dedup_cols=["ts_code", "end_date", "ann_date"],
                        force=args.force,
                        sort_cols=["ann_date", "end_date"],
                    )

                if "cyq_perf" in download_set:
                    download_cyq_perf(
                        client, storage, trade_cal,
                        args.start_date, args.end_date, force=args.force,
                    )

                if "express" in download_set:
                    download_by_period(
                        client, storage,
                        dataset_name="express",
                        api_name="express_vip",
                        start_date=args.start_date, end_date=args.end_date,
                        dedup_cols=["ts_code", "end_date", "ann_date"],
                        force=args.force,
                        sort_cols=["ann_date", "end_date"],
                    )

                if "fund_portfolio" in download_set:
                    download_by_period(
                        client, storage,
                        dataset_name="fund_portfolio",
                        api_name="fund_portfolio",
                        start_date=args.start_date, end_date=args.end_date,
                        dedup_cols=["ts_code", "symbol", "end_date"],
                        force=args.force,
                        page_limit=8000,
                        partition_by_period=True,
                    )

                if "moneyflow_hsgt" in download_set:
                    download_moneyflow_hsgt(
                        client, storage, trade_cal,
                        args.start_date, args.end_date, force=args.force,
                    )

                if "top_list" in download_set:
                    download_top_list(
                        client, storage, trade_cal,
                        args.start_date, args.end_date, force=args.force,
                    )

                if "report_rc" in download_set:
                    download_report_rc(
                        client, storage,
                        args.start_date, args.end_date, force=args.force,
                    )

    except KeyboardInterrupt:
        # 修复 #11: 单独捕获 Ctrl+C, 优雅退出
        logger.warning("用户中断 (Ctrl+C), 正在打印已有错误汇总...")
        exit_code = 130

    except (ValueError, ConnectionError, TimeoutError) as e:
        logger.error(f"数据下载失败: {e}")
        logger.error("请检查 .env 中 TS_TOKEN 是否配置; 注册: https://tushare.pro/register")
        exit_code = 1

    except Exception as e:
        logger.exception(f"数据下载未预期异常: {e}")
        exit_code = 1

    finally:
        # 修复 #4/#5 收尾: 无论脚本成功/失败/中断, 都打印错误汇总与总耗时
        elapsed = time.time() - script_start_ts
        logger.info("=" * 70)
        logger.info(f"总耗时: {_fmt_duration(elapsed)}")
        logger.info("=" * 70)
        ERROR_COLLECTOR.print_summary()

        # 若有错误, 以非零退出码通知外层调度
        if ERROR_COLLECTOR.has_errors() and exit_code == 0:
            exit_code = 3

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
