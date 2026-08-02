# -*- coding: utf-8 -*-
"""PaperRetryMixin：src/lazybull/paper/broker.py 拆分出的 retry_pending_sells, retry_pending_buys, get_failed_buy_targets, clear_failed_buy_targets。"""

from ...trading.sizing import compute_lot_shares
from ..models import Fill
from ..models import Order
from ..models import PendingBuy
from ..models import TargetWeight
from loguru import logger
from typing import List
import pandas as pd

class PaperRetryMixin:
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
        from ...data import DataLoader, Storage
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
            if ps.create_date >= trade_date:
                logger.info(
                    f"股票 {ps.ts_code} 延迟卖出于 {ps.create_date} 创建，"
                    f"按 T+1 规则等待后续交易日执行"
                )
                remaining_sells.append(ps)
                continue

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

            # 检查是否为 pending_sell 创建后新建的仓位（避免误卖）
            # 场景：T0 触发亏损换出 pending_sell，T1 同日先执行持有期卖出再买入新仓，
            # 下一日重试时 buy_date >= create_date 说明是新仓，不应被旧卖单触发
            if pos.buy_date >= ps.create_date:
                logger.info(
                    f"股票 {ps.ts_code} 当前持仓(买入日={pos.buy_date})晚于延迟卖出"
                    f"创建日({ps.create_date})，跳过旧卖单，视为已处理"
                )
                continue
            
            # 检查价格数据
            if ps.ts_code not in sell_prices:
                logger.warning(f"股票 {ps.ts_code} 无价格数据，保留订单")
                remaining_sells.append(ps)
                continue
            
            # 检查可交易性
            can_sell, reason = self._check_can_sell(ps.ts_code, tradability, trade_date)
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

    def retry_pending_buys(
        self,
        trade_date: str,
        buy_price_type: str = 'close',
        max_attempts: int = 5
    ) -> tuple[List[Fill], List[PendingBuy]]:
        """重试延迟买入订单（补位计划）
        
        对于延迟买入队列中的订单，检查是否可以买入。
        如果可以买入，执行买入；如果不可买入，继续保留并增加尝试次数。
        超过最大尝试次数的订单会被移除。
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            buy_price_type: 买入价格类型 open/close
            max_attempts: 最大尝试次数（默认5次）
            
        Returns:
            (成交记录列表, 仍失败的订单列表) 元组
        """
        if not self.pending_buys:
            logger.info("当前无延迟买入订单")
            return [], []
        
        logger.info("=" * 80)
        logger.info(f"重试延迟买入订单（补位计划） - {trade_date}")
        logger.info(f"待处理订单数: {len(self.pending_buys)}")
        logger.info("=" * 80)
        
        # 加载当日可交易性
        tradability = self._load_tradability_info(trade_date)
        
        # 加载价格
        from ...data import DataLoader, Storage
        storage = Storage()
        loader = DataLoader(storage)
        daily_data = loader.load_clean_daily_by_date(trade_date)
        
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {trade_date} 的价格数据")
            return [], self.pending_buys
        
        # 构建价格字典
        buy_prices = {}
        price_col = buy_price_type  # 'open' 或 'close'
        if price_col not in daily_data.columns:
            logger.warning(f"价格列 {price_col} 不存在，降级到 close")
            price_col = 'close'
        
        for _, row in daily_data.iterrows():
            ts_code = row['ts_code']
            price = row.get(price_col)
            if not pd.isna(price) and price > 0:
                buy_prices[ts_code] = price
        
        # 重试每个订单
        fills = []
        remaining_buys = []
        expired_buys = []
        min_buy_value_threshold = self._get_min_buy_value_threshold(buy_prices)
        
        for pb in self.pending_buys:
            # 检查是否同日重复执行：若 last_attempt_date == trade_date，则不增加 attempts
            if pb.last_attempt_date == trade_date:
                logger.info(
                    f"股票 {pb.ts_code} 今日已重试过（last_attempt_date={pb.last_attempt_date}），"
                    f"不重复推进 attempts（当前 attempts={pb.attempts}）"
                )
            else:
                # 不同日期，推进 attempts 并更新 last_attempt_date
                pb.attempts += 1
                pb.last_attempt_date = trade_date
                logger.debug(f"股票 {pb.ts_code} 尝试次数增加到 {pb.attempts}，更新 last_attempt_date={trade_date}")
            
            # 检查是否超过最大尝试次数
            if pb.attempts > max_attempts:
                logger.warning(f"股票 {pb.ts_code} 已达到最大尝试次数 {max_attempts}，移除补位订单")
                expired_buys.append(pb)
                continue
            
            # 检查价格数据
            if pb.ts_code not in buy_prices:
                logger.warning(f"股票 {pb.ts_code} 无价格数据，保留订单")
                remaining_buys.append(pb)
                continue
            
            # 检查可交易性
            can_buy, reason = self._check_can_buy(pb.ts_code, tradability)
            if not can_buy:
                logger.warning(f"股票 {pb.ts_code} 仍不可买入: {reason}，保留订单（尝试次数: {pb.attempts}/{max_attempts}）")
                remaining_buys.append(pb)
                continue
            
            # 可以买入，计算买入金额和股数
            # 使用当前剩余现金按权重分配
            available_cash = self.account.get_cash()
            price = buy_prices[pb.ts_code]
            
            # 按目标权重计算应买金额
            # 注意：这里使用剩余现金 * 权重，而不是总资产 * 权重
            # 因为补位时总资产可能已经变化
            buy_value = available_cash * pb.target_weight
            
            # 预估成本
            estimated_cost = self.cost_model.calculate_buy_cost(buy_value)
            
            # 确保有足够现金
            if buy_value + estimated_cost > available_cash:
                buy_value = available_cash - estimated_cost
                if buy_value <= 0:
                    logger.warning(f"股票 {pb.ts_code} 现金不足，保留订单")
                    remaining_buys.append(pb)
                    continue
            
            # 计算股数（向下取整到100的倍数）
            buy_shares = compute_lot_shares(buy_value, price)
            
            if buy_shares < 100:
                logger.warning(f"股票 {pb.ts_code} 不足一手（可买{buy_shares}股），保留订单")
                remaining_buys.append(pb)
                continue

            actual_buy_value = buy_shares * price
            if min_buy_value_threshold > 0 and actual_buy_value < min_buy_value_threshold:
                logger.warning(
                    f"股票 {pb.ts_code} 买入后市值 {actual_buy_value:.2f} 低于阈值 "
                    f"{min_buy_value_threshold:.2f}，保留订单"
                )
                remaining_buys.append(pb)
                continue
            
            # 构建订单
            order = Order(
                ts_code=pb.ts_code,
                action='buy',
                shares=buy_shares,
                price=price,
                target_weight=pb.target_weight,
                current_weight=0.0,
                reason=f"{pb.reason}(补位)"
            )
            
            # 执行订单
            fill = self._execute_single_order(order, trade_date, buy_price_type)
            if fill:
                fills.append(fill)
                logger.info(f"成功买入 {pb.ts_code} {buy_shares} 股（补位）")
            else:
                # 执行失败，保留订单
                logger.warning(f"股票 {pb.ts_code} 执行失败，保留订单")
                remaining_buys.append(pb)
        
        # 更新延迟买入队列
        self.pending_buys = remaining_buys
        self.storage.save_pending_buys(self.pending_buys)
        
        logger.info("=" * 80)
        logger.info(f"补位重试完成: 成功买入 {len(fills)} 笔，剩余 {len(remaining_buys)} 笔延迟订单，过期 {len(expired_buys)} 笔")
        logger.info("=" * 80)
        
        return fills, remaining_buys

    def get_failed_buy_targets(self) -> List[TargetWeight]:
        """获取最近一次买入失败的目标列表
        
        Returns:
            失败目标列表
        """
        return self._failed_buy_targets

    def clear_failed_buy_targets(self) -> None:
        """清空失败买入目标列表"""
        self._failed_buy_targets = []
