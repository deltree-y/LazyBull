# -*- coding: utf-8 -*-
"""raw_download 子包门面（re-export 全部公共符号）。

由 scripts/download_raw.py 薄入口委托运行；也支持 `from scripts import raw_download`。
"""

from .alt import (
    download_cashflow,
    download_moneyflow_hsgt,
    download_report_rc,
    download_stk_holdernumber,
    download_top_list,
)
from .basic import download_basic_data
from .cli import _bypass_proxy_for_download, _should_bypass_proxy_for_download, main
from .core import (
    _DOWNLOAD_CONCURRENCY,
    ALT_DATASETS,
    DAILY_SUBSETS,
    ERROR_COLLECTOR,
    ErrorCollector,
    ProgressTracker,
    _fmt_duration,
    _run_concurrent,
)
from .daily import (
    _DAILY_ALLOW_EMPTY,
    _DAILY_FETCHERS,
    _pending_daily_subsets,
    download_daily_data,
)
from .daily_partition import (
    _download_by_trade_date,
    download_cyq_perf,
    download_margin_detail,
    download_stock_st,
)
from .income import download_income
from .periodic import (
    _generate_month_periods,
    _generate_quarter_periods,
    _query_with_pagination,
    _save_merged,
    _to_int_date,
    download_by_period,
)

__all__ = [
    "ALT_DATASETS",
    "DAILY_SUBSETS",
    "ERROR_COLLECTOR",
    "ErrorCollector",
    "ProgressTracker",
    "_DAILY_ALLOW_EMPTY",
    "_DAILY_FETCHERS",
    "_DOWNLOAD_CONCURRENCY",
    "_bypass_proxy_for_download",
    "_download_by_trade_date",
    "_fmt_duration",
    "_generate_month_periods",
    "_generate_quarter_periods",
    "_pending_daily_subsets",
    "_query_with_pagination",
    "_run_concurrent",
    "_save_merged",
    "_should_bypass_proxy_for_download",
    "_to_int_date",
    "download_basic_data",
    "download_by_period",
    "download_cashflow",
    "download_income",
    "download_cyq_perf",
    "download_daily_data",
    "download_margin_detail",
    "download_moneyflow_hsgt",
    "download_report_rc",
    "download_stk_holdernumber",
    "download_stock_st",
    "download_top_list",
    "main",
]
