# -*- coding: utf-8 -*-
"""PaperPricingMixin：src/lazybull/paper/runner.py 拆分出的 _load_prices, _record_nav, _load_kelly_window_data, _estimate_stock_variance, _kelly_weights, _calc_holding_days。"""

from ..common.date_utils import calc_holding_trade_days
from ..trading.sizing import compute_kelly_weights
from ..trading.sizing import estimate_variance_from_prices
from .models import NAVRecord
from loguru import logger
from typing import Dict
from typing import Optional
import pandas as pd

class PaperPricingMixin:
    def _load_prices(
        self,
        trade_date: str,
        buy_price_type: str,
        sell_price_type: str
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """加载价格数据（分开盘/收盘）
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            buy_price_type: 买入价格类型 open/close
            sell_price_type: 卖出价格类型 open/close
            
        Returns:
            (buy_prices, sell_prices) 价格字典元组
            buy_prices: {ts_code: price} 买入价格字典
            sell_prices: {ts_code: price} 卖出价格字典
        """
        daily_data = self.loader.load_clean_daily_by_date(trade_date)
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {trade_date} 的日线数据")
            return {}, {}
        
        buy_prices = {}
        sell_prices = {}
        
        # 处理买入价格
        buy_col = buy_price_type  # 'open' 或 'close'
        if buy_col not in daily_data.columns:
            logger.warning(f"买入价格列 {buy_col} 不存在，降级到 close")
            buy_col = 'close'
        
        # 处理卖出价格
        sell_col = sell_price_type  # 'open' 或 'close'
        if sell_col not in daily_data.columns:
            logger.warning(f"卖出价格列 {sell_col} 不存在，降级到 close")
            sell_col = 'close'
        
        # 填充价格字典
        for _, row in daily_data.iterrows():
            ts_code = row['ts_code']
            
            # 买入价格（如果缺失，尝试降级）
            buy_price = row.get(buy_col)
            if pd.isna(buy_price) or buy_price <= 0:
                # open缺失，降级到close
                if buy_col == 'open' and 'close' in row:
                    buy_price = row['close']
                    if not pd.isna(buy_price) and buy_price > 0:
                        logger.debug(f"{ts_code} open价格缺失，使用close={buy_price}")
            
            if not pd.isna(buy_price) and buy_price > 0:
                buy_prices[ts_code] = buy_price
            
            # 卖出价格（如果缺失，尝试降级）
            sell_price = row.get(sell_col)
            if pd.isna(sell_price) or sell_price <= 0:
                # open缺失，降级到close
                if sell_col == 'open' and 'close' in row:
                    sell_price = row['close']
                    if not pd.isna(sell_price) and sell_price > 0:
                        logger.debug(f"{ts_code} open价格缺失，使用close={sell_price}")
            
            if not pd.isna(sell_price) and sell_price > 0:
                sell_prices[ts_code] = sell_price
        
        logger.info(f"加载价格数据: 买入({buy_price_type})={len(buy_prices)}只, "
                   f"卖出({sell_price_type})={len(sell_prices)}只")
        
        return buy_prices, sell_prices

    def _record_nav(self, trade_date: str, prices: Dict[str, float]) -> None:
        """记录净值
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            prices: {ts_code: price} 价格字典
        """
        cash = self.account.get_cash()
        position_value = self.account.get_position_value(prices)
        total_value = cash + position_value
        nav = total_value / self.account.initial_capital
        
        nav_record = NAVRecord(
            trade_date=trade_date,
            cash=cash,
            position_value=position_value,
            total_value=total_value,
            nav=nav
        )
        
        self.paper_storage.append_nav(nav_record)
        logger.info(f"净值记录: 现金={cash:,.2f}, 持仓={position_value:,.2f}, "
                   f"总值={total_value:,.2f}, NAV={nav:.4f}")

    def _load_kelly_window_data(self, trade_date: str) -> Optional[pd.DataFrame]:
        """加载 Kelly 波动率估计所需的近窗价格数据。"""
        if self._kelly_cache_date == trade_date and self._kelly_cache_df is not None:
            return self._kelly_cache_df

        trade_dates = self._get_open_trade_dates()
        if trade_date not in trade_dates:
            return None

        current_idx = trade_dates.index(trade_date)
        start_idx = max(0, current_idx - max(self.horizon, 20, 2 * 60))
        start_date = trade_dates[start_idx]
        daily_df = self.loader.load_clean_daily(start_date=start_date, end_date=trade_date)
        self._kelly_cache_date = trade_date
        self._kelly_cache_df = daily_df
        return daily_df

    def _estimate_stock_variance(self, stock: str, trade_date: str) -> Optional[float]:
        """估计股票近期收益率方差，供 Kelly 仓位计算使用。"""
        daily_df = self._load_kelly_window_data(trade_date)
        if daily_df is None or daily_df.empty:
            return None

        stock_df = daily_df[daily_df["ts_code"] == stock].sort_values("trade_date")
        if len(stock_df) < 20:
            return None

        stock_df = stock_df.tail(max(20, min(self.horizon * 3, 120), 60))
        price_col = "close_adj" if "close_adj" in stock_df.columns else "close"
        # 方差口径统一由 trading.sizing 计算
        return estimate_variance_from_prices(stock_df[price_col].astype(float).to_numpy())

    def _kelly_weights(
        self,
        signals: Dict[str, float],
        trade_date: str,
        half: bool = False,
    ) -> Dict[str, float]:
        """计算 Kelly / 半 Kelly 仓位权重（委托 trading.sizing 共享实现）。"""
        result, _ = compute_kelly_weights(
            signals,
            variance_fn=lambda stock: self._estimate_stock_variance(stock, trade_date),
            half=half,
            max_leverage=float(getattr(self, "kelly_max_leverage", 1.0)),
        )
        return result

    def _calc_holding_days(
        self,
        buy_date: str,
        current_date: str,
        trade_dates_list: list,
    ) -> int:
        """计算两个日期之间的交易日数（共用 common.date_utils）。"""
        from ..common.date_utils import calc_holding_trade_days

        return calc_holding_trade_days(buy_date, current_date, trade_dates_list)
