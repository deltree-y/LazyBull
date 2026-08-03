# -*- coding: utf-8 -*-
"""raw_download 子包核心：常量、错误汇总、进度追踪、并发执行器。"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


ALT_DATASETS = [
    "fina_indicator",
    "margin_detail",
    "stk_holdernumber",
    "forecast",
    "cyq_perf",
    "express",
    "fund_portfolio",
    "moneyflow_hsgt",
    "top_list",
    "report_rc",
    "cashflow",
]


DAILY_SUBSETS = [
    "daily",
    "daily_basic",
    "adj_factor",
    "suspend",
    "stk_limit",
    "moneyflow",
    "stock_st",
]


_DOWNLOAD_CONCURRENCY = 1


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


def _run_concurrent(
    work_items,
    worker: Callable,
    label: str,
    collect: bool = False,
) -> Optional[List[Any]]:
    """并发执行 worker(item) 遍历 work_items。

    - _DOWNLOAD_CONCURRENCY == 1 时走同步路径, 与串行等价, 便于排障/降级
    - 真实并发度仍被 TushareClient 内置的令牌桶限频收紧, 不会触发 QPS 超标
    - worker 内的所有异常由 worker 自己捕获并记录到 ERROR_COLLECTOR,
      此处仅兜底一次避免单个线程崩溃吞掉其余任务
    - collect=True 时按 work_items 顺序返回 worker 返回值列表; 否则返回 None
      (既有调用方不传 collect, 行为完全不变)

    Returns:
        collect=True 时返回 List[Any] (长度与 work_items 一致), 否则 None
    """
    results: List[Any] = [None] * len(work_items)

    if _DOWNLOAD_CONCURRENCY <= 1 or len(work_items) <= 1:
        for i, item in enumerate(work_items):
            try:
                r = worker(item)
                if collect:
                    results[i] = r
            except Exception as e:
                ERROR_COLLECTOR.add(label, f"item={item!r}", f"worker 未捕获异常: {e}")
        return results if collect else None

    with ThreadPoolExecutor(
        max_workers=_DOWNLOAD_CONCURRENCY,
        thread_name_prefix=f"dl-{label}",
    ) as pool:
        futures = {pool.submit(worker, item): idx for idx, item in enumerate(work_items)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                r = fut.result()
                if collect:
                    results[idx] = r
            except Exception as e:
                item = work_items[idx]
                ERROR_COLLECTOR.add(label, f"item={item!r}", f"worker 未捕获异常: {e}")
    return results if collect else None
