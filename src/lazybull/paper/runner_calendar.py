# -*- coding: utf-8 -*-
"""PaperCalendarMixin：src/lazybull/paper/runner.py 拆分出的 _get_open_trade_dates, _load_open_trade_dates_from_storage, _extract_open_trade_dates, _ensure_trade_calendar_coverage, _correct_trade_date, _resolve_next_requested_trade_date, _get_next_trade_date, _get_prev_trade_date。"""

from ..data import ensure_basic_data
from datetime import datetime
from loguru import logger
from typing import List
from typing import Optional
import pandas as pd

class PaperCalendarMixin:
    def _get_open_trade_dates(self) -> List[str]:
        """返回开市交易日列表（带简单缓存）。"""
        if self._trade_dates_cache is None:
            self._trade_dates_cache = self._load_open_trade_dates_from_storage()
        return self._trade_dates_cache

    def _load_open_trade_dates_from_storage(self) -> List[str]:
        """从 clean/raw 交易日历中提取覆盖范围更完整的开市日列表。"""
        clean_dates = self._extract_open_trade_dates(self.loader.load_clean_trade_cal())
        raw_dates = self._extract_open_trade_dates(self.loader.load_trade_cal())

        if raw_dates and (not clean_dates or raw_dates[-1] > clean_dates[-1]):
            return raw_dates
        return clean_dates or raw_dates

    @staticmethod
    def _extract_open_trade_dates(trade_cal: Optional[pd.DataFrame]) -> List[str]:
        """从交易日历 DataFrame 中提取开市日列表。"""
        if trade_cal is None or trade_cal.empty:
            return []
        if "cal_date" not in trade_cal.columns or "is_open" not in trade_cal.columns:
            return []

        cal_dates = trade_cal["cal_date"]
        if pd.api.types.is_datetime64_any_dtype(cal_dates):
            cal_dates = cal_dates.dt.strftime("%Y%m%d")
        else:
            cal_dates = cal_dates.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)

        is_open = trade_cal["is_open"].astype(str)
        return cal_dates[is_open == "1"].dropna().tolist()

    def _ensure_trade_calendar_coverage(self, target_date: Optional[str]) -> List[str]:
        """确保交易日历至少覆盖到目标日期。"""
        normalized_target = str(target_date).strip() if target_date is not None else ""
        if normalized_target and len(normalized_target) == 10 and normalized_target[4] == "-":
            normalized_target = normalized_target.replace("-", "")
        if normalized_target and not normalized_target.isdigit():
            normalized_target = ""

        trade_dates = self._get_open_trade_dates()
        if trade_dates and (not normalized_target or trade_dates[-1] >= normalized_target):
            return trade_dates

        if normalized_target and ensure_basic_data(self.client, self.storage, normalized_target, force=False):
            self._trade_dates_cache = self._load_open_trade_dates_from_storage()
            return self._trade_dates_cache

        return trade_dates

    def _correct_trade_date(self, input_date: str) -> str:
        """校正交易日期：非交易日自动滚动到下一交易日
        
        Args:
            input_date: 输入日期 YYYYMMDD
            
        Returns:
            校正后的交易日期 YYYYMMDD
        """
        try:
            normalized_input = str(input_date).strip()
            if normalized_input.lower() == "next":
                normalized_input = self._resolve_next_requested_trade_date()
            if len(normalized_input) == 10 and normalized_input[4] == "-" and normalized_input[7] == "-":
                normalized_input = normalized_input.replace("-", "")

            trade_dates = self._ensure_trade_calendar_coverage(normalized_input)
            if not trade_dates:
                logger.error("无法加载交易日历")
                return normalized_input
            
            # 检查输入日期是否为交易日
            if normalized_input in trade_dates:
                return normalized_input
            
            # 找到输入日期后的第一个交易日
            for date in trade_dates:
                if date > normalized_input:
                    logger.warning(
                        f"输入日期 {normalized_input} 不是交易日，"
                        f"已自动校正到下一交易日: {date}"
                    )
                    return date
            
            # 如果没有找到后续交易日，返回原日期（可能是未来日期）
            logger.warning(f"未找到 {normalized_input} 之后的交易日，使用原日期")
            return normalized_input
            
        except Exception as e:
            logger.error(f"校正交易日期失败: {e}")
            return str(input_date).strip()

    def _resolve_next_requested_trade_date(self) -> str:
        """将 next 解析为最近执行日之后的下一个交易日。"""
        today = datetime.now().strftime("%Y%m%d")
        trade_dates = self._ensure_trade_calendar_coverage(today)
        if not trade_dates:
            logger.warning("交易日历为空，next 回退为原始输入")
            return "next"

        last_trade_date = self.paper_storage.load_last_trade_date() or ""
        if not last_trade_date:
            account_state = self.paper_storage.load_account_state()
            if account_state and account_state.last_update:
                last_trade_date = str(account_state.last_update)

        if last_trade_date:
            future_dates = [date for date in trade_dates if date > last_trade_date]
            if future_dates:
                resolved_date = future_dates[0]
                logger.info(f"trade_date=next 解析为上次执行日 {last_trade_date} 之后的 {resolved_date}")
                return resolved_date

        today = pd.Timestamp.today().strftime("%Y%m%d")
        future_dates = [date for date in trade_dates if date >= today]
        if future_dates:
            resolved_date = future_dates[0]
            logger.info(f"trade_date=next 解析为从今日起的最近交易日 {resolved_date}")
            return resolved_date

        logger.warning("未找到可用交易日，next 回退为原始输入")
        return "next"

    def _get_next_trade_date(self, trade_date: str) -> Optional[str]:
        """获取下一个交易日
        
        Args:
            trade_date: 当前交易日 YYYYMMDD
            
        Returns:
            下一个交易日 YYYYMMDD，不存在返回None
        """
        try:
            trade_dates = self._ensure_trade_calendar_coverage(trade_date)
            if not trade_dates:
                logger.error("无法加载交易日历")
                return None

            # 找到当前日期之后的第一个交易日；若当天本身是交易日，则返回其后一个交易日
            for i, date in enumerate(trade_dates):
                if date > trade_date:
                    return date
                if date == trade_date and i + 1 < len(trade_dates):
                    return trade_dates[i + 1]
            
            logger.debug(f"未找到 {trade_date} 的下一个交易日")
            return None
        except Exception as e:
            logger.error(f"获取下一个交易日失败: {e}")
            return None

    def _get_prev_trade_date(self, trade_date: str) -> Optional[str]:
        """获取上一个交易日。

        Args:
            trade_date: 当前交易日 YYYYMMDD

        Returns:
            上一个交易日 YYYYMMDD，不存在返回None
        """
        try:
            trade_dates = self._ensure_trade_calendar_coverage(trade_date)
            if not trade_dates:
                logger.error("无法加载交易日历")
                return None

            previous_date = None
            for date in trade_dates:
                if date >= trade_date:
                    break
                previous_date = date

            if previous_date is not None:
                return previous_date

            logger.warning(f"未找到 {trade_date} 的上一个交易日")
            return None
        except Exception as e:
            logger.error(f"获取上一个交易日失败: {e}")
            return None
