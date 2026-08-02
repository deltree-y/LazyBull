# -*- coding: utf-8 -*-
"""PaperTradabilityMixin：src/lazybull/paper/broker.py 拆分出的 _get_suspend_calendar, _get_open_trade_dates, _calc_holding_trade_days, _load_tradability_info, _check_can_buy, _check_can_sell。"""

from ...common.trade_status import evaluate_trade_status
from loguru import logger
from typing import Dict
from typing import List

class PaperTradabilityMixin:
    def _get_suspend_calendar(self):
        """获取停牌日历实例（延迟创建，共用 common 构建函数）"""
        if self._suspend_calendar is None:
            from ...common.suspend_calendar import get_suspend_calendar

            self._suspend_calendar, self.data_storage = get_suspend_calendar(self.data_storage)

        return self._suspend_calendar

    def _get_open_trade_dates(self) -> List[str]:
        """获取开市交易日列表（带缓存）。"""
        if self._open_trade_dates is not None:
            return self._open_trade_dates

        try:
            from ...data import DataLoader, Storage

            data_storage = self.data_storage if self.data_storage is not None else Storage()
            loader = DataLoader(data_storage, verbose=False)
            trade_cal = loader.load_clean_trade_cal()
            if trade_cal is None or trade_cal.empty:
                self._open_trade_dates = []
            else:
                self._open_trade_dates = (
                    trade_cal.loc[trade_cal["is_open"] == 1, "cal_date"].astype(str).tolist()
                )
        except Exception:
            self._open_trade_dates = []

        return self._open_trade_dates

    @staticmethod
    def _calc_holding_trade_days(buy_date: str, current_date: str, trade_dates_list: List[str]) -> int:
        """按交易日口径计算持有天数（共用 common.date_utils）。"""
        from ...common.date_utils import calc_holding_trade_days

        return calc_holding_trade_days(buy_date, current_date, trade_dates_list)

    def _load_tradability_info(self, trade_date: str) -> Dict[str, Dict]:
        """加载可交易性信息
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            
        Returns:
            {ts_code: {is_suspended, is_limit_up, is_limit_down, tradable}}
        """
        from ...data import DataLoader, Storage
        
        storage = Storage()
        loader = DataLoader(storage)
        
        daily_data = loader.load_clean_daily_by_date(trade_date)
        
        tradability = {}
        if daily_data is not None and not daily_data.empty:
            for _, row in daily_data.iterrows():
                ts_code = row['ts_code']
                tradability[ts_code] = {
                    'is_suspended': row.get('is_suspended', 0),
                    'is_limit_up': row.get('is_limit_up', 0),
                    'is_limit_down': row.get('is_limit_down', 0),
                    'tradable': row.get('tradable', 1)
                }
        
        return tradability

    def _check_can_buy(self, ts_code: str, tradability: Dict) -> tuple[bool, str]:
        """检查是否可以买入
        
        Args:
            ts_code: 股票代码
            tradability: 可交易性信息字典
            
        Returns:
            (can_buy, reason) 是否可买入及原因
        """
        if ts_code not in tradability:
            return True, "无可交易性数据"

        can_buy, reason = evaluate_trade_status(
            tradability[ts_code], "buy", require_tradable=True
        )
        return can_buy, reason or "可买入"

    def _check_can_sell(self, ts_code: str, tradability: Dict, trade_date: str = None) -> tuple[bool, str]:
        """检查是否可以卖出
        
        Args:
            ts_code: 股票代码
            tradability: 可交易性信息字典
            trade_date: 交易日期（可选），如提供则使用 SuspendCalendar 检查停牌
            
        Returns:
            (can_sell, reason) 是否可卖出及原因
        """
        info = tradability.get(ts_code)

        # 如果提供了 trade_date，优先使用 SuspendCalendar 检查停牌
        if trade_date:
            try:
                suspend_calendar = self._get_suspend_calendar()
                is_suspended = suspend_calendar.is_suspended(ts_code, trade_date)
                if is_suspended:
                    return False, "停牌"
                if info is not None:
                    info = {**info, "is_suspended": 0}
            except Exception as e:
                # 停牌数据加载失败，回退到使用 tradability
                logger.warning(f"停牌数据加载失败（{e}），使用 tradability 判断")

        if info is None:
            return True, "无可交易性数据"

        can_sell, reason = evaluate_trade_status(info, "sell")
        return can_sell, reason or "可卖出"
