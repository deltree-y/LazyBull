# -*- coding: utf-8 -*-
"""PaperPositionsMixin：src/lazybull/paper/broker.py 拆分出的 _estimate_total_assets_with_price_map, _get_min_buy_value_threshold, get_positions_detail, print_positions_summary, calculate_round_pnl_metrics, _calculate_annualized_return。"""

from ...common.print_table import format_row
from ...trading.sizing import compute_min_buy_value_threshold
from loguru import logger
from typing import Dict
from typing import Optional
import pandas as pd

class PaperPositionsMixin:
    def _estimate_total_assets_with_price_map(self, price_map: Dict[str, float]) -> float:
        """估算当前总资产（价格缺失时回退买入价）。"""
        total_assets = float(self.account.get_cash())
        for ts_code, pos in self.account.get_positions().items():
            ref_price = float(price_map.get(ts_code, 0.0))
            if ref_price <= 0:
                ref_price = float(pos.buy_price)
            total_assets += float(pos.shares) * ref_price
        return total_assets

    def _get_min_buy_value_threshold(self, price_map: Dict[str, float]) -> float:
        """计算最小买入后持仓市值阈值（与回测共用 trading.sizing 口径）。"""
        config = self.storage.load_config() or {}
        return compute_min_buy_value_threshold(
            total_assets=self._estimate_total_assets_with_price_map(price_map),
            target_count=int(config.get("top_n", 30) or 0),
            ratio=float(config.get("min_buy_value_ratio", 0.2) or 0.0),
        )

    def get_positions_detail(self, current_prices: Dict[str, float], current_date: Optional[str] = None, stock_names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """获取持仓明细（含收益信息）
        
        Args:
            current_prices: {ts_code: price} 当前价格字典
            current_date: 当前日期 YYYYMMDD（可选，用于计算持有天数）
            stock_names: {ts_code: name} 股票名称字典（可选）
            
        Returns:
            持仓明细DataFrame
        """
        positions = self.account.get_positions()
        
        if not positions:
            logger.info("当前无持仓")
            return pd.DataFrame()
        
        details = []
        trade_dates_list = self._get_open_trade_dates() if current_date else []
        config = self.storage.load_config() or {}
        rebalance_freq = int(config.get('rebalance_freq', 20))
        max_holding_days = rebalance_freq
        for ts_code, pos in positions.items():
            raw_price = current_prices.get(ts_code)
            try:
                current_price = float(raw_price)
            except (TypeError, ValueError):
                current_price = 0.0
            if pd.isna(current_price) or current_price <= 0:
                # 停牌或行情缺失时，估值回退到买入价，避免将持仓错误计为0。
                current_price = float(pos.buy_price)
            current_value = pos.shares * current_price
            cost_value = pos.shares * pos.buy_price + pos.buy_cost
            profit = current_value - cost_value
            profit_rate = (profit / cost_value * 100) if cost_value > 0 else 0.0
            
            # 计算持有交易日（缺失交易日历时回退自然日）
            holding_days = 0
            if current_date:
                if trade_dates_list:
                    holding_days = self._calc_holding_trade_days(
                        pos.buy_date,
                        current_date,
                        trade_dates_list,
                    )
                else:
                    holding_days = pos.get_holding_days(current_date)
            # 剩余天数按“可持有上限”计算：基础持有期 +（可选）盈利延续天数
            holding_remaining = max(0, max_holding_days - holding_days)
            
            # 构建股票代码显示（包含名称）
            stock_name = stock_names.get(ts_code, 'na') if stock_names else 'na'
            stock_display = f"{ts_code}({stock_name})"
            
            details.append({
                '股票代码': stock_display,
                '持仓股数': pos.shares,
                '当前价格': current_price,
                '买入均价': pos.buy_price,
                '买入成本': pos.buy_cost,  # 内部仍保留用于计算
                '买入日期': pos.buy_date,
                '持有天数': holding_days,
                '持有剩余': holding_remaining,
                '当前市值': current_value,
                '浮动盈亏': profit,
                '收益率(%)': profit_rate,
                '状态': pos.status,
                '备注': pos.notes
            })
        
        df = pd.DataFrame(details)
        return df

    def print_positions_summary(self, current_prices: Dict[str, float], current_date: Optional[str] = None, stock_names: Optional[Dict[str, str]] = None) -> None:
        """打印持仓汇总信息
        
        Args:
            current_prices: {ts_code: price} 当前价格字典
            current_date: 当前日期 YYYYMMDD（可选，用于计算持有天数）
            stock_names: {ts_code: name} 股票名称字典（可选）
        """
        df = self.get_positions_detail(current_prices, current_date, stock_names)
        
        if df.empty:
            logger.info("=" * 80)
            logger.info("当前无持仓")
            logger.info("=" * 80)
            return
        
        # 打印表头（新列顺序：股票代码、股数、当前价格、买入均价、买入日期、持有交易日、持有剩余、当前市值、浮盈、收益率(%)、状态）
        header = ["股票代码", "股数", "当前价格", "买入均价", "买入日期", "持有D", "剩余D", "当前市值", "浮盈", "收益率(%)", "状态"]
        logger.info(format_row(header, self.positions_table_widths, ['left'] * len(self.positions_table_widths)))

        logger.info("-" * 140)

        # 1. 先按“收益率(%)”进行降序排列 (ascending=False 表示降序)
        df_sorted = df.sort_values(by='收益率(%)', ascending=False)        
        # 2. 打印每行: 遍历排序后的 df_sorted
        for _, row in df_sorted.iterrows():
            row_data = [
                row['股票代码'], row['持仓股数'], 
                f"{row['当前价格']:.2f}", f"{row['买入均价']:.2f}",
                row['买入日期'], row['持有天数'], row['持有剩余'],
                f"{row['当前市值']:.2f}", 
                f"{row['浮动盈亏']:.2f}", f"{row['收益率(%)']:.2f}",
                row['状态']
            ]
            logger.info(format_row(row_data, self.positions_table_widths, self.positions_table_aligns))

        
        # 打印汇总（使用内部买入成本计算总成本，但不在表中显示）
        total_cost = df['买入成本'].sum() + (df['持仓股数'] * df['买入均价']).sum()
        total_value = df['当前市值'].sum()
        total_profit = df['浮动盈亏'].sum()
        total_profit_rate = (total_profit / total_cost * 100) if total_cost > 0 else 0.0
        
        logger.info("-" * 140)
        logger.info(f"{'合计(' + str(len(df)) + ')':<18} {df['持仓股数'].sum():<8} {'':<10} {'':<10} "
                   f"{'':<12} {'':<10} {'':<10} {total_value:<12.2f} {total_profit:<12.2f} {total_profit_rate:<12.2f}")
        logger.info("=" * 140)
        logger.info(f"账户现金: {self.account.get_cash():,.2f}")
        logger.info(f"持仓市值: {total_value:,.2f}")
        
        total_assets = self.account.get_cash() + total_value
        logger.info(f"总资产: {total_assets:,.2f}")

        # 从配置读取 initial_capital
        config = self.storage.load_config()
        if config and 'initial_capital' in config:
            initial_capital = config['initial_capital']
        else:
            # 如果配置不存在，使用账户的 initial_capital
            initial_capital = self.account.initial_capital
        # 本轮盈亏（优先使用“上次调仓总资产 -> 当前总资产”口径，缺失时回退旧口径）
        round_profit, round_start_value, round_current_value = self.calculate_round_pnl_metrics(
            total_assets=total_assets,
            total_cost=total_cost,
            total_profit=total_profit,
        )
        logger.info(
            f"本轮盈亏: {round_profit:,.2f}% ({round_start_value:,.2f} -> {round_current_value:,.2f})"
        )

        # 新增：总盈亏百分比
        total_profit_pct = ((total_assets - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0
        logger.info(f"  总盈亏: {total_profit_pct:.2f}% ({initial_capital:,.2f} -> {total_assets:,.2f})")
        
        # 新增：年化收益率
        # 计算年化收益率
        annualized_return = self._calculate_annualized_return(
            initial_capital, 
            total_assets,
            current_date
        )
        
        if annualized_return is not None:
            logger.info(f"年化收益率: {annualized_return:.2f}%")
        else:
            logger.info(f"年化收益率: 无法计算（缺少账户起始日期）")
        
        logger.info("=" * 140)

    def calculate_round_pnl_metrics(
        self,
        total_assets: float,
        total_cost: float,
        total_profit: float,
    ) -> tuple[float, float, float]:
        """计算本轮盈亏口径。

        优先口径：
            使用策略状态中的 last_rebalance_nav 作为本轮起点，
            本轮盈亏 = (当前总资产 - 本轮起点总资产) / 本轮起点总资产。
        回退口径（兼容旧数据）：
            若缺少 last_rebalance_nav，则沿用“当前持仓浮盈/持仓成本”口径。

        Args:
            total_assets: 当前总资产（现金 + 持仓市值）
            total_cost: 当前持仓成本（买入金额 + 买入手续费）
            total_profit: 当前持仓浮动盈亏

        Returns:
            (本轮盈亏百分比, 起点值, 当前值)
        """
        strategy_state = self.storage.load_strategy_state()
        if isinstance(strategy_state, dict):
            last_rebalance_nav = strategy_state.get("last_rebalance_nav")
            if last_rebalance_nav is not None:
                try:
                    start_value = float(last_rebalance_nav)
                except (TypeError, ValueError):
                    start_value = 0.0
                if start_value > 0:
                    round_profit = (total_assets - start_value) / start_value * 100
                    return round_profit, start_value, total_assets

        # 兼容旧口径：仅按当前持仓浮盈计算
        current_value = total_cost + total_profit
        round_profit = (total_profit / total_cost * 100) if total_cost > 0 else 0.0
        return round_profit, total_cost, current_value

    def _calculate_annualized_return(
        self, 
        initial_capital: float, 
        current_value: float, 
        current_date: Optional[str]
    ) -> Optional[float]:
        """计算年化收益率
        
        Args:
            initial_capital: 初始资金
            current_value: 当前总资产
            current_date: 当前日期 YYYYMMDD
            
        Returns:
            年化收益率（百分比），如果无法计算则返回 None
        """
        # 空仓时年化收益率为 0
        if current_value <= 0 or initial_capital <= 0:
            return 0.0
        
        # 尝试从配置获取账户起始日期
        config = self.storage.load_config()
        account_start_date = None
        
        if config and 'account_start_date' in config:
            account_start_date = config['account_start_date']
        
        # 如果没有起始日期，尝试从 NAV 记录获取最早日期
        if not account_start_date:
            nav_df = self.storage.load_all_nav()
            if nav_df is not None and len(nav_df) > 0:
                # 获取最早的交易日期
                account_start_date = str(nav_df['trade_date'].iloc[0])
        
        # 如果仍然没有起始日期，返回 None
        if not account_start_date or not current_date:
            return None
        
        # 计算持有天数
        try:
            start_dt = pd.to_datetime(account_start_date, format='%Y%m%d')
            current_dt = pd.to_datetime(current_date, format='%Y%m%d')
            days = (current_dt - start_dt).days
            
            # 如果天数太少（例如小于1天），返回 0
            if days < 1:
                return 0.0

            # 复合年化收益率 (CAGR)
            # 年化收益率 = ((当前总资产 / 初始资金) ^ (365 / 持有天数) - 1) * 100
            annualized = ((current_value / initial_capital) ** (365.0 / days) - 1.0) * 100
            return annualized
            
        except Exception as e:
            logger.warning(f"计算年化收益率失败: {e}")
            return None
