# -*- coding: utf-8 -*-
"""特征确保模块（ensure 子包门面，re-export 全部公共符号）。"""

from .bulk import (
    _bulk_download_by_period,
    _bulk_download_stk_holdernumber,
    _generate_quarter_periods,
    _query_with_pagination,
    _save_merged_bulk,
)
from .constants import (
    FEATURE_DATA_FUTURE_MONTHS,
    FEATURE_DATA_HISTORY_MONTHS,
    HISTORICAL_DATA_MONTHS,
    MAX_HISTORICAL_DAYS,
    _FINA_REQUIRED_RAW_COLS,
    _MIN_CASHFLOW_RECORDS,
    _MIN_EXPRESS_RECORDS,
    _MIN_FINA_RECORDS,
    _MIN_FORECAST_RECORDS,
    _MIN_HOLDER_RECORDS,
    _MIN_REPORT_RC_RECORDS,
)
from .downloads import (
    _try_download_cashflow,
    _try_download_express,
    _try_download_fina_indicator,
    _try_download_forecast,
    _try_download_report_rc,
    _try_download_stk_holdernumber,
)
from .entry import ensure_features_for_date
from .factor_load import _load_factor_data
from .historical import (
    _ensure_historical_clean_data,
    _merge_refreshed_rows,
    _refresh_existing_period_rows,
)
from .historical_assets import (
    _try_ensure_historical_cyq_perf,
    _try_ensure_historical_fund_portfolio,
    _try_ensure_historical_margin,
    _try_ensure_historical_moneyflow_hsgt,
    _try_ensure_historical_top_list,
)
from .incremental import (
    _append_and_save_raw,
    _get_latest_date,
    _incremental_catchup_by_calendar_date,
    _iter_calendar_dates,
    _normalize_date_str,
)
from .industry import _ensure_shenwan_industry
from .schema import _REQUIRED_FACTOR_COLS, _check_features_schema

__all__ = [
    "FEATURE_DATA_FUTURE_MONTHS",
    "FEATURE_DATA_HISTORY_MONTHS",
    "HISTORICAL_DATA_MONTHS",
    "MAX_HISTORICAL_DAYS",
    "_FINA_REQUIRED_RAW_COLS",
    "_MIN_CASHFLOW_RECORDS",
    "_MIN_EXPRESS_RECORDS",
    "_MIN_FINA_RECORDS",
    "_MIN_FORECAST_RECORDS",
    "_MIN_HOLDER_RECORDS",
    "_MIN_REPORT_RC_RECORDS",
    "_REQUIRED_FACTOR_COLS",
    "_append_and_save_raw",
    "_bulk_download_by_period",
    "_bulk_download_stk_holdernumber",
    "_check_features_schema",
    "_ensure_historical_clean_data",
    "_ensure_shenwan_industry",
    "_generate_quarter_periods",
    "_get_latest_date",
    "_incremental_catchup_by_calendar_date",
    "_iter_calendar_dates",
    "_load_factor_data",
    "_merge_refreshed_rows",
    "_normalize_date_str",
    "_query_with_pagination",
    "_refresh_existing_period_rows",
    "_save_merged_bulk",
    "_try_download_cashflow",
    "_try_download_express",
    "_try_download_fina_indicator",
    "_try_download_forecast",
    "_try_download_report_rc",
    "_try_download_stk_holdernumber",
    "_try_ensure_historical_cyq_perf",
    "_try_ensure_historical_fund_portfolio",
    "_try_ensure_historical_margin",
    "_try_ensure_historical_moneyflow_hsgt",
    "_try_ensure_historical_top_list",
    "ensure_features_for_date",
]
