"""纸面交易与只读展示共用的账户绩效口径。"""

import math
from typing import TYPE_CHECKING, Optional

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from .storage import PaperStorage


def load_account_start_date(
    storage: "PaperStorage", config: Optional[dict] = None
) -> Optional[str]:
    """优先使用配置起始日，否则取净值记录的最早有效日期。"""
    config = storage.load_config() if config is None else config
    start_date = str((config or {}).get("account_start_date") or "").strip()
    if start_date:
        return start_date
    try:
        nav = storage.load_all_nav()
        if nav is None or nav.empty or "trade_date" not in nav:
            return None
        dates = pd.to_datetime(nav["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        first_date = dates.min()
        return first_date.strftime("%Y%m%d") if pd.notna(first_date) else None
    except (OSError, ValueError, ConnectionError) as exc:
        logger.warning(f"读取账户年化起始日期失败: {exc}")
        return None


def calculate_annualized_return(
    initial_capital: float,
    current_value: float,
    current_date: Optional[str],
    account_start_date: Optional[str],
) -> Optional[float]:
    """按实际自然日计算复合年化收益率，返回百分数；不可用时返回 None。"""
    if not math.isfinite(initial_capital) or not math.isfinite(current_value):
        return None
    if current_value <= 0 or initial_capital <= 0:
        return 0.0
    if not account_start_date or not current_date:
        return None
    try:
        start = pd.to_datetime(str(account_start_date), format="%Y%m%d")
        end = pd.to_datetime(str(current_date), format="%Y%m%d")
        days = (end - start).days
        if days < 1:
            return 0.0
        result = ((current_value / initial_capital) ** (365.0 / days) - 1.0) * 100
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError) as exc:
        logger.warning(
            f"计算账户年化收益率失败: start={account_start_date}, end={current_date}, {exc}"
        )
        return None


def build_account_value_series(
    nav: Optional[pd.DataFrame], start_date: str, end_date: str
) -> pd.Series:
    """提取真实日度总资产，同日取最后记录，不填补缺失日期。"""
    if nav is None or nav.empty or not {"trade_date", "total_value"}.issubset(nav.columns):
        return pd.Series(dtype=float)
    records = nav[["trade_date", "total_value"]].copy()
    dates = pd.to_datetime(records["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    records["trade_date"] = dates.dt.strftime("%Y%m%d")
    records = records.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")
    values = pd.to_numeric(records.set_index("trade_date")["total_value"], errors="coerce")
    valid = values.notna() & (values >= 0) & (values < float("inf"))
    return values.loc[valid & (values.index >= start_date) & (values.index <= end_date)].sort_index()