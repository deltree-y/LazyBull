"""report_rc 研报身份、去重与完整查询契约。"""

import warnings
from datetime import datetime, timedelta
from typing import Callable, List, Sequence

import pandas as pd
from loguru import logger

REPORT_RC_REPORT_KEY_COLUMNS: Sequence[str] = (
    "ts_code",
    "report_date",
    "org_name",
    "author_name",
    "report_title",
)
REPORT_RC_ANALYST_KEY_COLUMNS: Sequence[str] = ("org_name", "author_name")
REPORT_RC_ROW_KEY_COLUMNS: Sequence[str] = REPORT_RC_REPORT_KEY_COLUMNS + ("quarter",)

ReportRcRangeQuery = Callable[[str, str], pd.DataFrame]
_REPORT_RC_OVERLIMIT_ERROR_MARKERS: Sequence[str] = ("查询数据失败", "请确认参数")


def report_rc_key_columns(df: pd.DataFrame, include_quarter: bool = True) -> List[str]:
    """返回当前 schema 中可用的研报键列。"""
    candidates = REPORT_RC_ROW_KEY_COLUMNS if include_quarter else REPORT_RC_REPORT_KEY_COLUMNS
    return [column for column in candidates if column in df.columns]


def validate_report_rc_identity_schema(
    df: pd.DataFrame,
    include_quarter: bool = True,
) -> None:
    """校验 report_rc 身份列完整，禁止静默降级为弱去重键。"""
    if df is None or len(df) == 0:
        return
    required = REPORT_RC_ROW_KEY_COLUMNS if include_quarter else REPORT_RC_REPORT_KEY_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "report_rc 身份 schema 不完整，"
            f"缺少列: {missing}。请使用 --download report_rc --force 重新下载目标区间"
        )


def _normalized_key_frame(df: pd.DataFrame, key_columns: Sequence[str]) -> pd.DataFrame:
    """生成仅用于身份比较的规范化字符串键，不改写 raw 数据。"""
    normalized = pd.DataFrame(index=df.index)
    for column in key_columns:
        values = df[column].astype("string").fillna("").str.strip()
        if column == "report_date":
            values = values.str.replace("-", "", regex=False).str[:8]
        normalized[column] = values
    return normalized


def deduplicate_report_rc(
    df: pd.DataFrame,
    include_quarter: bool = True,
    require_full_identity: bool = True,
) -> pd.DataFrame:
    """按统一研报身份去重，默认拒绝缺少身份列的旧 schema。"""
    if df is None or len(df) == 0:
        return df
    if require_full_identity:
        validate_report_rc_identity_schema(df, include_quarter=include_quarter)
    key_columns = report_rc_key_columns(df, include_quarter=include_quarter)
    if "ts_code" not in key_columns or "report_date" not in key_columns:
        return df.copy().reset_index(drop=True)
    normalized = _normalized_key_frame(df, key_columns)
    keep_mask = ~normalized.duplicated(subset=key_columns, keep="last")
    return df.loc[keep_mask].reset_index(drop=True)


def is_report_rc_overlimit_error(message: str) -> bool:
    """仅识别 TuShare report_rc 累计查询超限的完整稳定错误文案。"""
    return all(marker in message for marker in _REPORT_RC_OVERLIMIT_ERROR_MARKERS)


def _concat_report_rc_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """合并二分结果并屏蔽 pandas 全空列兼容告警。"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r"The behavior of DataFrame concatenation with empty or all-NA entries",
        )
        return pd.concat(parts, ignore_index=True)


def query_report_rc_adaptive(
    query_range: ReportRcRangeQuery,
    start_date: str,
    end_date: str,
    depth: int = 0,
    max_depth: int = 6,
) -> pd.DataFrame:
    """查询 report_rc 日期区间，累计行数超限时递归二分。"""
    if start_date > end_date:
        return pd.DataFrame()

    try:
        return query_range(start_date, end_date)
    except Exception as error:
        message = str(error)
        if not is_report_rc_overlimit_error(message):
            raise
        if depth >= max_depth or start_date == end_date:
            raise RuntimeError(
                f"report_rc {start_date}~{end_date} 二分 {max_depth} 层后仍失败: {error}"
            ) from error

        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        mid_dt = start_dt + (end_dt - start_dt) / 2
        mid_date = mid_dt.strftime("%Y%m%d")
        next_date = (mid_dt + timedelta(days=1)).strftime("%Y%m%d")
        logger.debug(
            "[report_rc] {}~{} 累计查询超限，按日期二分（depth={}）",
            start_date,
            end_date,
            depth + 1,
        )
        left = query_report_rc_adaptive(
            query_range,
            start_date,
            mid_date,
            depth=depth + 1,
            max_depth=max_depth,
        )
        right = query_report_rc_adaptive(
            query_range,
            next_date,
            end_date,
            depth=depth + 1,
            max_depth=max_depth,
        )
        parts = [part for part in (left, right) if part is not None and len(part) > 0]
        if not parts:
            return pd.DataFrame()
        return _concat_report_rc_parts(parts)
