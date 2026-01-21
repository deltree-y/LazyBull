"""纸面交易经纪模块"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ..common.cost import CostModel
from .account import PaperAccount
from .models import Fill, Order, TargetWeight
from .storage import PaperStorage


class PaperBroker:
    """纸面交易经纪
    
    负责生成订单、计算成本、打印明细、记录成交
    """
    
    def __init__(
        self,
        account: PaperAccount,
        cost_model: Optional[CostModel] = None,
        storage: Optional[PaperStorage] = None
    ):
        """初始化经纪
        
        Args:
            account: 账户实例
            cost_model: 成本模型
            storage: 存储实例
        """
        self.account = account
        self.cost_model = cost_model or CostModel()
        self.storage = storage or PaperStorage()
    
    def generate_orders(
        self,
        targets: List[TargetWeight],
        buy_prices: Dict[str, float],
        sell_prices: Dict[str, float],
        trade_date: str
    ) -> List[Order]:
        """生成订单（含可交易性检查）
        
        从当前持仓和目标权重生成买卖订单，并检查涨跌停、停牌等可交易性
        
        Args:
            targets: 目标权重列表
            buy_prices: {ts_code: price} 买入价格字典
            sell_prices: {ts_code: price} 卖出价格字典
            trade_date: 交易日期 YYYYMMDD
            
        Returns:
            订单列表
        """
        orders = []
        
        # 加载当日可交易性信息
        tradability = self._load_tradability_info(trade_date)
        
        # 使用卖出价格计算总资产（因为卖出在前）
        all_prices = {**sell_prices, **buy_prices}  # 卖出价格优先
        total_value = self.account.get_total_value(all_prices)
        if total_value <= 0:
            logger.warning("总资产为0，无法生成订单")
            return orders
        
        # 目标权重字典
        target_weights = {t.ts_code: (t.target_weight, t.reason) for t in targets}
        
        # 当前持仓股票
        current_stocks = set(self.account.get_positions().keys())
        
        # 目标持仓股票
        target_stocks = set(target_weights.keys())
        
        # 1. 卖出订单：当前持有但不在目标中，或目标权重降低
        for ts_code in current_stocks:
            current_weight = self.account.get_position_weight(ts_code, all_prices)
            target_weight, reason = target_weights.get(ts_code, (0.0, "退出持仓"))
            
            if target_weight < current_weight:
                # 需要卖出
                pos = self.account.get_position(ts_code)
                if ts_code not in sell_prices:
                    logger.warning(f"股票 {ts_code} 无卖出价格数据，跳过卖出")
                    continue
                
                # 检查可交易性
                can_sell, sell_reason = self._check_can_sell(ts_code, tradability)
                if not can_sell:
                    logger.warning(f"股票 {ts_code} 不可卖出: {sell_reason}，订单延迟")
                    # 跌停或停牌，订单延迟
                    continue
                
                # 计算需要卖出的股数
                target_value = total_value * target_weight
                current_value = pos.shares * sell_prices[ts_code]
                sell_value = current_value - target_value
                sell_shares = int(sell_value / sell_prices[ts_code])
                
                # 确保不超过持仓
                sell_shares = min(sell_shares, pos.shares)
                
                if sell_shares > 0:
                    orders.append(Order(
                        ts_code=ts_code,
                        action='sell',
                        shares=sell_shares,
                        price=sell_prices[ts_code],
                        target_weight=target_weight,
                        current_weight=current_weight,
                        reason=reason if target_weight == 0 else "减仓"
                    ))
        
        # 2. 买入订单：目标持有但当前没有，或目标权重增加
        for ts_code in target_stocks:
            target_weight, reason = target_weights[ts_code]
            current_weight = self.account.get_position_weight(ts_code, all_prices)
            
            if target_weight > current_weight:
                # 需要买入
                if ts_code not in buy_prices:
                    logger.warning(f"股票 {ts_code} 无买入价格数据，跳过买入")
                    continue
                
                # 检查可交易性
                can_buy, buy_reason = self._check_can_buy(ts_code, tradability)
                if not can_buy:
                    logger.warning(f"股票 {ts_code} 不可买入: {buy_reason}，跳过订单")
                    # 涨停、停牌或其他不可交易，跳过
                    continue
                
                # 计算需要买入的金额
                target_value = total_value * target_weight
                current_value = total_value * current_weight
                buy_value = target_value - current_value
                
                # 预估成本
                estimated_cost = self.cost_model.calculate_buy_cost(buy_value)
                available_cash = self.account.get_cash()
                
                # 确保有足够现金（考虑成本）
                if buy_value + estimated_cost > available_cash:
                    buy_value = available_cash - estimated_cost
                    if buy_value <= 0:
                        logger.warning(f"现金不足，跳过买入 {ts_code}")
                        continue
                
                # 计算股数（向下取整到100的倍数）
                buy_shares = int(buy_value / buy_prices[ts_code] / 100) * 100
                
                if buy_shares > 0:
                    orders.append(Order(
                        ts_code=ts_code,
                        action='buy',
                        shares=buy_shares,
                        price=buy_prices[ts_code],
                        target_weight=target_weight,
                        current_weight=current_weight,
                        reason=reason if current_weight == 0 else "加仓"
                    ))
        
        logger.info(f"生成订单: {len([o for o in orders if o.action == 'buy'])} 买，"
                   f"{len([o for o in orders if o.action == 'sell'])} 卖")
        
        return orders
    
    def _load_tradability_info(self, trade_date: str) -> Dict[str, Dict]:
        """加载可交易性信息
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            
        Returns:
            {ts_code: {is_suspended, is_limit_up, is_limit_down, tradable}}
        """
        from ..data import DataLoader, Storage
        
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
        
        info = tradability[ts_code]
        
        # 停牌检查
        if info.get('is_suspended', 0) == 1:
            return False, "停牌"
        
        # 涨停检查（涨停不可买入）
        if info.get('is_limit_up', 0) == 1:
            return False, "涨停"
        
        # 基本可交易性检查
        if info.get('tradable', 1) == 0:
            return False, "不可交易（ST/上市不足等）"
        
        return True, "可买入"
    
    def _check_can_sell(self, ts_code: str, tradability: Dict) -> tuple[bool, str]:
        """检查是否可以卖出
        
        Args:
            ts_code: 股票代码
            tradability: 可交易性信息字典
            
        Returns:
            (can_sell, reason) 是否可卖出及原因
        """
        if ts_code not in tradability:
            return True, "无可交易性数据"
        
        info = tradability[ts_code]
        
        # 停牌检查
        if info.get('is_suspended', 0) == 1:
            return False, "停牌"
        
        # 跌停检查（跌停不可卖出）
        if info.get('is_limit_down', 0) == 1:
            return False, "跌停"
        
        return True, "可卖出"
    
    def execute_orders(
        self,
        orders: List[Order],
        trade_date: str,
        buy_price_type: str = 'close',
        sell_price_type: str = 'close'
    ) -> List[Fill]:
        """执行订单并打印明细
        
        Args:
            orders: 订单列表
            trade_date: 交易日期 YYYYMMDD
            buy_price_type: 买入价格类型 open/close
            sell_price_type: 卖出价格类型 open/close
            
        Returns:
            成交记录列表
        """
        fills = []
        
        # 打印标题
        logger.info("=" * 120)
        logger.info(f"纸面交易执行明细 - {trade_date}")
        logger.info("=" * 120)
        logger.info(f"{'股票代码':<12} {'方向':<6} {'目标权重':<10} {'当前权重':<10} "
                   f"{'股数':<8} {'价格类型':<8} {'参考价格':<10} {'成交金额':<12} "
                   f"{'佣金':<10} {'印花税':<10} {'滑点':<10} {'总成本':<10} {'原因':<15}")
        logger.info("-" * 120)
        
        # 先执行卖出订单
        sell_orders = [o for o in orders if o.action == 'sell']
        for order in sell_orders:
            fill = self._execute_single_order(order, trade_date, sell_price_type)
            if fill:
                fills.append(fill)
                self._print_order_detail(order, fill, sell_price_type)
        
        # 再执行买入订单
        buy_orders = [o for o in orders if o.action == 'buy']
        for order in buy_orders:
            fill = self._execute_single_order(order, trade_date, buy_price_type)
            if fill:
                fills.append(fill)
                self._print_order_detail(order, fill, buy_price_type)
        
        logger.info("=" * 120)
        logger.info(f"执行完成: {len([f for f in fills if f.action == 'buy'])} 买，"
                   f"{len([f for f in fills if f.action == 'sell'])} 卖")
        logger.info(f"账户现金: {self.account.get_cash():,.2f}")
        logger.info(f"持仓数量: {len(self.account.get_positions())}")
        logger.info("=" * 120)
        
        return fills
    
    def _execute_single_order(
        self,
        order: Order,
        trade_date: str,
        price_type: str
    ) -> Optional[Fill]:
        """执行单个订单
        
        Args:
            order: 订单
            trade_date: 交易日期
            price_type: 价格类型 open/close
            
        Returns:
            成交记录，失败返回None
        """
        # 使用订单中的参考价格（已根据价格类型设置）
        price = order.price
        amount = order.shares * price
        
        if order.action == 'buy':
            # 计算买入成本
            commission = self.cost_model.calculate_commission(amount)
            slippage = self.cost_model.calculate_slippage(amount)
            total_cost = commission + slippage
            
            # 检查现金是否足够
            total_required = amount + total_cost
            if total_required > self.account.get_cash():
                logger.warning(f"现金不足，取消买入 {order.ts_code}")
                return None
            
            # 更新账户
            self.account.update_cash(-total_required)
            self.account.add_position(
                ts_code=order.ts_code,
                shares=order.shares,
                buy_price=price,
                buy_cost=total_cost,
                buy_date=trade_date
            )
            
            # 创建成交记录
            fill = Fill(
                trade_date=trade_date,
                ts_code=order.ts_code,
                action='buy',
                shares=order.shares,
                price=price,
                amount=amount,
                commission=commission,
                stamp_tax=0.0,
                slippage=slippage,
                total_cost=total_cost,
                reason=order.reason
            )
            
        else:  # sell
            # 检查持仓
            pos = self.account.get_position(order.ts_code)
            if not pos or pos.shares < order.shares:
                logger.warning(f"持仓不足，取消卖出 {order.ts_code}")
                return None
            
            # 计算卖出成本
            commission = self.cost_model.calculate_commission(amount)
            stamp_tax = self.cost_model.calculate_stamp_tax(amount)
            slippage = self.cost_model.calculate_slippage(amount)
            total_cost = commission + stamp_tax + slippage
            
            # 更新账户
            cash_received = amount - total_cost
            self.account.update_cash(cash_received)
            self.account.reduce_position(order.ts_code, order.shares)
            
            # 创建成交记录
            fill = Fill(
                trade_date=trade_date,
                ts_code=order.ts_code,
                action='sell',
                shares=order.shares,
                price=price,
                amount=amount,
                commission=commission,
                stamp_tax=stamp_tax,
                slippage=slippage,
                total_cost=total_cost,
                reason=order.reason
            )
        
        # 记录成交
        self.storage.append_trade(fill)
        
        return fill
    
    def _print_order_detail(self, order: Order, fill: Fill, price_type: str) -> None:
        """打印订单明细
        
        Args:
            order: 订单
            fill: 成交记录
            price_type: 价格类型
        """
        logger.info(
            f"{order.ts_code:<12} {order.action:<6} {order.target_weight:<10.4f} {order.current_weight:<10.4f} "
            f"{order.shares:<8} {price_type:<8} {fill.price:<10.2f} {fill.amount:<12.2f} "
            f"{fill.commission:<10.2f} {fill.stamp_tax:<10.2f} {fill.slippage:<10.2f} "
            f"{fill.total_cost:<10.2f} {fill.reason:<15}"
        )
    
    def get_positions_detail(self, current_prices: Dict[str, float], current_date: Optional[str] = None) -> pd.DataFrame:
        """获取持仓明细（含收益信息）
        
        Args:
            current_prices: {ts_code: price} 当前价格字典
            current_date: 当前日期 YYYYMMDD（可选，用于计算持有天数）
            
        Returns:
            持仓明细DataFrame
        """
        positions = self.account.get_positions()
        
        if not positions:
            logger.info("当前无持仓")
            return pd.DataFrame()
        
        details = []
        for ts_code, pos in positions.items():
            current_price = current_prices.get(ts_code, 0.0)
            current_value = pos.shares * current_price
            cost_value = pos.shares * pos.buy_price + pos.buy_cost
            profit = current_value - cost_value
            profit_rate = (profit / cost_value * 100) if cost_value > 0 else 0.0
            
            # 计算持有天数
            holding_days = 0
            if current_date:
                holding_days = pos.get_holding_days(current_date)
            
            details.append({
                '股票代码': ts_code,
                '持仓股数': pos.shares,
                '买入均价': pos.buy_price,
                '买入成本': pos.buy_cost,
                '买入日期': pos.buy_date,
                '持有天数': holding_days,
                '当前价格': current_price,
                '当前市值': current_value,
                '浮动盈亏': profit,
                '收益率(%)': profit_rate,
                '状态': pos.status,
                '备注': pos.notes
            })
        
        df = pd.DataFrame(details)
        return df
    
    def print_positions_summary(self, current_prices: Dict[str, float], current_date: Optional[str] = None) -> None:
        """打印持仓汇总信息
        
        Args:
            current_prices: {ts_code: price} 当前价格字典
            current_date: 当前日期 YYYYMMDD（可选，用于计算持有天数）
        """
        df = self.get_positions_detail(current_prices, current_date)
        
        if df.empty:
            logger.info("=" * 80)
            logger.info("当前无持仓")
            logger.info("=" * 80)
            return
        
        logger.info("=" * 140)
        logger.info("持仓明细")
        logger.info("=" * 140)
        
        # 打印表头
        logger.info(f"{'股票代码':<12} {'股数':<8} {'买入均价':<10} {'买入成本':<10} "
                   f"{'买入日期':<10} {'持有天数':<8} {'当前价格':<10} {'当前市值':<12} "
                   f"{'浮盈':<12} {'收益率(%)':<10} {'状态':<8}")
        logger.info("-" * 140)
        
        # 打印每行
        for _, row in df.iterrows():
            logger.info(
                f"{row['股票代码']:<12} {row['持仓股数']:<8} {row['买入均价']:<10.2f} {row['买入成本']:<10.2f} "
                f"{row['买入日期']:<10} {row['持有天数']:<8} {row['当前价格']:<10.2f} {row['当前市值']:<12.2f} "
                f"{row['浮动盈亏']:<12.2f} {row['收益率(%)']:<10.2f} {row['状态']:<8}"
            )
        
        # 打印汇总
        total_cost = df['买入成本'].sum() + (df['持仓股数'] * df['买入均价']).sum()
        total_value = df['当前市值'].sum()
        total_profit = df['浮动盈亏'].sum()
        total_profit_rate = (total_profit / total_cost * 100) if total_cost > 0 else 0.0
        
        logger.info("-" * 140)
        logger.info(f"{'合计':<12} {df['持仓股数'].sum():<8} {'':<10} {df['买入成本'].sum():<10.2f} "
                   f"{'':<10} {'':<8} {'':<10} {total_value:<12.2f} {total_profit:<12.2f} {total_profit_rate:<10.2f}")
        logger.info("=" * 140)
        logger.info(f"账户现金: {self.account.get_cash():,.2f}")
        logger.info(f"持仓市值: {total_value:,.2f}")
        logger.info(f"总资产: {self.account.get_cash() + total_value:,.2f}")
        logger.info("=" * 140)
