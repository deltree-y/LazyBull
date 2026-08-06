# -*- coding: utf-8 -*-
"""raw_download 子包：CLI 主入口。"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Set, Tuple

from loguru import logger

from src.lazybull.common.config import get_config, get_tushare_settings
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient
from src.lazybull.data.tushare_client import FINA_INDICATOR_DEFAULT_FIELDS

from .alt import (
    download_cashflow,
    download_moneyflow_hsgt,
    download_report_rc,
    download_stk_holdernumber,
    download_top_list,
)
from .announcement_risk import (
    download_block_trade,
    download_pledge_stat,
    download_share_float,
)
from . import core as raw_core
from .basic import download_basic_data
from .core import ALT_DATASETS, ERROR_COLLECTOR, _fmt_duration
from .daily import download_daily_data
from .daily_partition import download_cyq_perf, download_margin_detail, download_stock_st
from .periodic import _to_int_date, download_by_period

# 终端/系统常通过环境变量注入 HTTP(S) 代理 (如 PowerShell profile 加载的
# http://192.168.1.21:18081), 导致 TuShare 请求走内网代理并出现 Read timed out。
# 下载原始数据默认在进程内绕过代理直连, 仅影响当前进程, 不修改终端设置。
# 开关: LAZYBULL_DOWNLOAD_BYPASS_PROXY=0 可关闭 (单开关、单默认值)。
_DOWNLOAD_PROXY_ENV_KEYS: Tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _should_bypass_proxy_for_download() -> bool:
    """判断下载原始数据时是否临时禁用代理 (默认启用)。"""
    raw = str(os.getenv("LAZYBULL_DOWNLOAD_BYPASS_PROXY", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _bypass_proxy_for_download() -> None:
    """清除当前进程内的代理环境变量, 使 TuShare/requests 直连。

    仅影响当前 Python 进程, 不修改终端/系统设置; 未注入代理时为空操作。
    """
    if not _should_bypass_proxy_for_download():
        return
    removed = [k for k in _DOWNLOAD_PROXY_ENV_KEYS if os.environ.pop(k, None) is not None]
    if removed:
        logger.info(
            f"已临时清除代理环境变量(直连): {', '.join(sorted(removed))} "
            "(如需走代理可设置 LAZYBULL_DOWNLOAD_BYPASS_PROXY=0)"
        )


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
        "--only-is-st", action="store_true",
        help="仅下载 ST 状态数据(stock_st)，不下载其它日线/另类数据"
    )
    parser.add_argument(
        "--download", nargs="*", default=None,
        help="指定另类数据集, 可多选。可选: fina_indicator, margin_detail, "
             "stk_holdernumber, forecast, cyq_perf, express, fund_portfolio, "
             "moneyflow_hsgt, top_list, report_rc, cashflow, pledge_stat, "
             "share_float, block_trade, all_alt。不指定时仅下基础+日线"
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
    if args.only_basic and args.only_is_st:
        logger.error("参数冲突: --only-basic 与 --only-is-st 不能同时使用")
        sys.exit(2)
    if args.only_is_st and (args.all or (args.download is not None and len(args.download) > 0)):
        logger.error("参数冲突: --only-is-st 不能与 --all 或 --download 同时使用")
        sys.exit(2)

    # 初始化日志
    setup_logger(log_level="INFO")
    # 默认绕过终端/系统注入的代理, 避免 TuShare 请求走代理 Read timed out
    _bypass_proxy_for_download()
    get_config()

    # 从配置 / 命令行读取并发数, 直接写入 core 模块供 _run_concurrent 读取
    # (拆分后不能再用 global, 否则只改 cli 模块变量, 不影响 core 的 _run_concurrent)
    ts_settings = get_tushare_settings()
    if args.concurrency is not None:
        raw_core._DOWNLOAD_CONCURRENCY = max(1, args.concurrency)
    else:
        raw_core._DOWNLOAD_CONCURRENCY = max(1, ts_settings["download_concurrency"])

    logger.info("=" * 70)
    logger.info("开始下载原始数据 (raw 层)")
    logger.info("=" * 70)
    logger.info(f"日期范围    : {args.start_date} ~ {args.end_date}")
    logger.info(f"仅基础数据  : {'是' if args.only_basic else '否'}")
    logger.info(f"仅ST数据    : {'是' if args.only_is_st else '否'}")
    logger.info(f"强制重下    : {'是' if args.force else '否'}")
    logger.info(f"另类数据集  : {args.download or '无'}")
    logger.info(f"全量下载    : {'是' if args.all else '否'}")
    logger.info(f"并发线程数  : {raw_core._DOWNLOAD_CONCURRENCY} (1=串行降级)")
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
        elif args.only_is_st:
            download_stock_st(
                client, storage, trade_cal,
                args.start_date, args.end_date, force=args.force,
            )
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
                        fields=FINA_INDICATOR_DEFAULT_FIELDS,
                        force=args.force,
                        partition_by_period=True,
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
                        partition_by_period=True,
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

                if "cashflow" in download_set:
                    download_cashflow(
                        client, storage,
                        args.start_date, args.end_date, force=args.force,
                    )

                # ── 风控公告类（质押/解禁/大宗）──
                if "pledge_stat" in download_set:
                    download_pledge_stat(
                        client, storage,
                        args.start_date, args.end_date, force=args.force,
                    )

                if "share_float" in download_set:
                    download_share_float(
                        client, storage,
                        args.start_date, args.end_date, force=args.force,
                    )

                if "block_trade" in download_set:
                    download_block_trade(
                        client, storage, trade_cal,
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