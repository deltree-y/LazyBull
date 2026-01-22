"""纸面交易经纪模块"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ..common.print_table import format_row

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
        storage: Optional[PaperStorage] = None,
        verbose: bool = True,
    ):
        """初始化经纪
        
        Args:
            account: 账户实例
            cost_model: 成本模型
            storage: 存储实例
            verbose: 是否输出详细日志
        """
        self.account = account
        self.cost_model = cost_model or CostModel()
        self.storage = storage or PaperStorage()
        self.order_table_widths = [12, 6, 10, 10, 8, 8, 10, 12, 10, 10, 10, 10, 15]
        self.order_table_aligns = ['left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left']
        self.positions_table_widths = [12, 8, 10, 10, 12, 8, 10, 12, 12, 12, 8]
        self.positions_table_aligns = ['left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left']
        self.verbose = verbose
        # 加载延迟卖出队列
        self.pending_sells = self.storage.load_pending_sells()

    
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
        all_prices = {**buy_prices, **sell_prices}  # 卖出价格优先（后者覆盖前者）
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
                    # 跌停或停牌，加入延迟卖出队列
                    from .models import PendingSell
                    pending_sell = PendingSell(
                        ts_code=ts_code,
                        shares=pos.shares,  # 记录待卖出股数
                        target_weight=target_weight,
                        reason=reason if target_weight == 0 else "减仓",
                        create_date=trade_date,
                        attempts=0
                    )
                    self.pending_sells.append(pending_sell)
                    continue
                
                # 计算需要卖出的股数
                target_value = total_value * target_weight
                current_value = pos.shares * sell_prices[ts_code]
                sell_value = current_value - target_value
                sell_shares_raw = int(sell_value / sell_prices[ts_code])
                
                # 判断是否为清仓
                is_full_liquidation = (target_weight == 0)
                
                if is_full_liquidation:
                    # 清仓：必须卖出全部股数，不允许零股
                    sell_shares = pos.shares
                    
                    # 检查是否有零股（不是100的倍数）
                    if sell_shares % 100 != 0:
                        # 零股出现：详细日志并 raise 异常
                        current_price = sell_prices.get(ts_code, 0.0)
                        current_market_value = pos.shares * current_price if current_price > 0 else 0.0
                        
                        error_msg = (
                            f"清仓时检测到零股，必须中止执行！\n"
                            f"  股票代码: {ts_code}\n"
                            f"  持仓股数: {pos.shares} 股（非100倍数）\n"
                            f"  交易日期: {trade_date}\n"
                            f"  目标权重: {target_weight}\n"
                            f"  原因: {reason}\n"
                            f"  当前价格: {current_price:.2f}\n"
                            f"  当前市值: {current_market_value:.2f}\n"
                            f"  买入日期: {pos.buy_date}\n"
                            f"  买入价格: {pos.buy_price:.2f}\n"
                            f"  持仓备注: {pos.notes}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                else:
                    # 减仓：按100股向下取整
                    sell_shares = (sell_shares_raw // 100) * 100
                
                # 确保不超过持仓且不卖超
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
        
        # 保存延迟卖出队列
        self.storage.save_pending_sells(self.pending_sells)
        
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
        header = ["股票代码", "方向", "目标权重", "当前权重", "股数", "价格类型", "参考价格", "成交金额", "佣金", "印花税", "滑点", "总成本", "原因"]
        logger.info("=" * 120)
        logger.info(f"纸面交易执行明细 - {trade_date}")
        logger.info("=" * 120)
        #logger.info(f"{'股票代码':<12} {'方向':<6} {'目标权重':<10} {'当前权重':<10} "
        #           f"{'股数':<8} {'价格类型':<8} {'参考价格':<10} {'成交金额':<12} "
        #           f"{'佣金':<10} {'印花税':<10} {'滑点':<10} {'总成本':<10} {'原因':<15}")
        logger.info(format_row(header, self.order_table_widths, ['left'] * len(self.order_table_widths)))

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
        #logger.info(
        #    f"{order.ts_code:<12} {order.action:<6} {order.target_weight:<10.4f} {order.current_weight:<10.4f} "
        #    f"{order.shares:<8} {price_type:<8} {fill.price:<10.2f} {fill.amount:<12.2f} "
        #    f"{fill.commission:<10.2f} {fill.stamp_tax:<10.2f} {fill.slippage:<10.2f} "
        #    f"{fill.total_cost:<10.2f} {fill.reason:<15}"
        #)
        row = [
            order.ts_code,
            order.action,
            f"{order.target_weight:.4f}",
            f"{order.current_weight:.4f}",
            str(order.shares),
            price_type,
            f"{fill.price:.2f}",
            f"{fill.amount:.2f}",
            f"{fill.commission:.2f}",
            f"{fill.stamp_tax:.2f}",
            f"{fill.slippage:.2f}",
            f"{fill.total_cost:.2f}",
            fill.reason
        ]
        logger.info(format_row(row, self.order_table_widths, self.order_table_aligns))

    
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
        
        # 打印表头
        header = ["股票代码", "股数", "买入均价", "买入成本", "买入日期", "持有天数", "当前价格", "当前市值", "浮盈", "收益率(%)", "状态"]
        logger.info(format_row(header, self.positions_table_widths, ['left'] * len(self.positions_table_widths)))

        logger.info("-" * 140)
        
        # 打印每行
        for _, row in df.iterrows():
            row = [
                row['股票代码'], row['持仓股数'], 
                f"{row['买入均价']:.2f}", f"{row['买入成本']:.2f}", 
                row['买入日期'], row['持有天数'],
                f"{row['当前价格']:.2f}", f"{row['当前市值']:.2f}", 
                f"{row['浮动盈亏']:.2f}", f"{row['收益率(%)']:.2f}",
                row['状态']
            ]
            logger.info(format_row(row, self.positions_table_widths, self.positions_table_aligns))

        
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
    
    def retry_pending_sells(
        self, 
        trade_date: str, 
        sell_price_type: str = 'close'
    ) -> List[Fill]:
        """重试延迟卖出订单
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            sell_price_type: 卖出价格类型 open/close
            
        Returns:
            成交记录列表
        """
        if not self.pending_sells:
            logger.info("当前无延迟卖出订单")
            return []
        
        logger.info("=" * 80)
        logger.info(f"重试延迟卖出订单 - {trade_date}")
        logger.info(f"待处理订单数: {len(self.pending_sells)}")
        logger.info("=" * 80)
        
        # 加载当日可交易性
        tradability = self._load_tradability_info(trade_date)
        
        # 加载价格
        from ..data import DataLoader, Storage
        storage = Storage()
        loader = DataLoader(storage)
        daily_data = loader.load_clean_daily_by_date(trade_date)
        
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {trade_date} 的价格数据")
            return []
        
        # 构建价格字典
        sell_prices = {}
        price_col = sell_price_type  # 'open' 或 'close'
        if price_col not in daily_data.columns:
            logger.warning(f"价格列 {price_col} 不存在，降级到 close")
            price_col = 'close'
        
        for _, row in daily_data.iterrows():
            ts_code = row['ts_code']
            price = row.get(price_col)
            if not pd.isna(price) and price > 0:
                sell_prices[ts_code] = price
        
        # 重试每个订单
        fills = []
        remaining_sells = []
        
        for ps in self.pending_sells:
            # 检查是否同日重复执行：若 last_attempt_date == trade_date，则不增加 attempts
            if ps.last_attempt_date == trade_date:
                logger.info(
                    f"股票 {ps.ts_code} 今日已重试过（last_attempt_date={ps.last_attempt_date}），"
                    f"不重复推进 attempts（当前 attempts={ps.attempts}）"
                )
            else:
                # 不同日期，推进 attempts 并更新 last_attempt_date
                ps.attempts += 1
                ps.last_attempt_date = trade_date
                logger.debug(f"股票 {ps.ts_code} 尝试次数增加到 {ps.attempts}，更新 last_attempt_date={trade_date}")
            
            # 检查持仓是否还存在
            pos = self.account.get_position(ps.ts_code)
            if not pos or pos.shares == 0:
                logger.info(f"股票 {ps.ts_code} 已无持仓，移除延迟卖出订单")
                continue
            
            # 检查价格数据
            if ps.ts_code not in sell_prices:
                logger.warning(f"股票 {ps.ts_code} 无价格数据，保留订单")
                remaining_sells.append(ps)
                continue
            
            # 检查可交易性
            can_sell, reason = self._check_can_sell(ps.ts_code, tradability)
            if not can_sell:
                logger.warning(f"股票 {ps.ts_code} 仍不可卖出: {reason}，保留订单（尝试次数: {ps.attempts}）")
                remaining_sells.append(ps)
                continue
            
            # 可以卖出，生成订单
            # 计算实际可卖股数（取当前持仓和pending记录的最小值）
            sell_shares = min(ps.shares, pos.shares)
            # 按100股向下取整
            sell_shares = (sell_shares // 100) * 100
            
            if sell_shares == 0:
                logger.warning(f"股票 {ps.ts_code} 持仓不足100股，无法卖出，保留订单")
                remaining_sells.append(ps)
                continue
            
            # 构建订单
            order = Order(
                ts_code=ps.ts_code,
                action='sell',
                shares=sell_shares,
                price=sell_prices[ps.ts_code],
                target_weight=ps.target_weight,
                current_weight=0.0,  # 不重要
                reason=f"{ps.reason}(延迟)"
            )
            
            # 执行订单
            fill = self._execute_single_order(order, trade_date, sell_price_type)
            if fill:
                fills.append(fill)
                logger.info(f"成功卖出 {ps.ts_code} {sell_shares} 股")
            else:
                # 执行失败，保留订单
                logger.warning(f"股票 {ps.ts_code} 执行失败，保留订单")
                remaining_sells.append(ps)
        
        # 更新延迟卖出队列
        self.pending_sells = remaining_sells
        self.storage.save_pending_sells(self.pending_sells)
        
        logger.info("=" * 80)
        logger.info(f"重试完成: 成功卖出 {len(fills)} 笔，剩余 {len(remaining_sells)} 笔延迟订单")
        logger.info("=" * 80)
        
        return fills
