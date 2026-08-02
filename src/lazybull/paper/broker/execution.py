# -*- coding: utf-8 -*-
"""PaperExecutionMixin：src/lazybull/paper/broker.py 拆分出的 execute_instructions, _execute_single_order, _resolve_buy_pnl_price, _calculate_execution_stats。"""

from ...common.print_table import format_row
from ..models import Fill, Order, PendingBuy, PendingSell, TargetWeight
from ..models import normalize_trade_reason
from loguru import logger
from typing import Dict
from typing import List
from typing import Optional
import pandas as pd

class PaperExecutionMixin:
    def execute_instructions(
        self,
        instructions: List,
        buy_prices: Dict[str, float],
        sell_prices: Dict[str, float],
        trade_date: str
    ) -> List[Fill]:
        """执行交易指令（新模式）
        
        按指令明确执行，不再基于权重重新计算
        
        Args:
            instructions: 交易指令列表
            buy_prices: {ts_code: price} 买入价格字典
            sell_prices: {ts_code: price} 卖出价格字典
            trade_date: 交易日期 YYYYMMDD
            
        Returns:
            成交记录列表
        """
        from ..models import PendingSell, PendingBuy, Order, TargetWeight
        
        fills = []
        self._failed_buy_targets = []
        
        # 加载可交易性信息（如果加载失败，假设所有股票可交易）
        try:
            tradability = self._load_tradability_info(trade_date)
        except Exception as e:
            logger.warning(f"加载可交易性信息失败: {e}，假设所有股票可交易")
            tradability = {}
        
        # 记录执行前的持仓快照（用于统计）
        positions_before = {}
        for ts_code, pos in self.account.get_positions().items():
            positions_before[ts_code] = pos.shares
        
        # 打印标题
        header = ["股票代码", "方向", "指令股数", "价格类型", "参考价格", "实际股数", "成交金额", "佣金", "印花税", "滑点", "总成本", "原因"]
        logger.info("=" * 120)
        logger.info(f"纸面交易执行明细（指令模式）- {trade_date}")
        logger.info("=" * 120)
        logger.info(format_row(header, [12, 6, 10, 10, 10, 10, 12, 10, 10, 10, 10, 20], ['left'] * 12))
        logger.info("-" * 120)
        
        # 1. 先执行卖出指令
        sell_instructions = [i for i in instructions if i.action == 'sell']
        for inst in sell_instructions:
            ts_code = inst.ts_code
            target_shares = inst.shares
            price_type = inst.price_type
            reason = inst.reason
            
            # 检查价格数据
            if ts_code not in sell_prices:
                # 无卖出价格数据，使用 SuspendCalendar 判断原因（停牌优先，否则无价格数据）
                reason_suffix = ""
                try:
                    suspend_calendar = self._get_suspend_calendar()
                    is_suspended = suspend_calendar.is_suspended(ts_code, trade_date)
                    if is_suspended:
                        reason_suffix = "（停牌）"
                        logger.warning(f"股票 {ts_code} 停牌，无法卖出，加入延迟卖出队列")
                    else:
                        reason_suffix = "（无价格数据）"
                        logger.warning(f"股票 {ts_code} 无卖出价格数据，加入延迟卖出队列")
                except Exception as e:
                    # 停牌数据加载失败，使用通用描述
                    reason_suffix = "（无价格数据）"
                    logger.warning(f"股票 {ts_code} 无卖出价格数据，且停牌数据加载失败（{e}），加入延迟卖出队列")
                
                # 加入延迟卖出队列
                # 注意：使用当前持仓股数而非指令股数，因为：
                # 1) 无价格时无法验证指令股数是否合理
                # 2) retry 时会重新计算可卖股数（取 min(pending.shares, pos.shares)）
                # 3) 确保能够完成卖出意图
                pos = self.account.get_position(ts_code)
                if pos:
                    pending_sell = PendingSell(
                        ts_code=ts_code,
                        shares=pos.shares,  # 记录当前持仓股数，重试时重新计算
                        target_weight=inst.target_weight,
                        reason=f"{reason}{reason_suffix}",
                        create_date=trade_date,
                        attempts=0
                    )
                    self.pending_sells.append(pending_sell)
                continue
            
            # 检查可交易性
            can_sell, check_reason = self._check_can_sell(ts_code, tradability, trade_date)
            if not can_sell:
                logger.warning(f"股票 {ts_code} 不可卖出: {check_reason}，加入延迟队列")
                # 加入延迟卖出队列
                pos = self.account.get_position(ts_code)
                if pos:
                    pending_sell = PendingSell(
                        ts_code=ts_code,
                        shares=pos.shares,
                        target_weight=inst.target_weight,
                        reason=f"{reason}（{check_reason}）",
                        create_date=trade_date,
                        attempts=0
                    )
                    self.pending_sells.append(pending_sell)
                continue
            
            # 检查持仓
            pos = self.account.get_position(ts_code)
            if not pos:
                logger.warning(f"股票 {ts_code} 无持仓，跳过卖出")
                continue
            
            # 实际卖出股数不超过持仓
            actual_shares = min(target_shares, pos.shares)
            
            # 创建订单并执行
            order = Order(
                ts_code=ts_code,
                action='sell',
                shares=actual_shares,
                price=sell_prices[ts_code],
                target_weight=inst.target_weight,
                current_weight=0.0,  # 指令模式不需要权重
                reason=reason
            )
            
            fill = self._execute_single_order(order, trade_date, price_type)
            if fill:
                fills.append(fill)
                # 打印执行详情
                row = [
                    ts_code,
                    'sell',
                    str(target_shares),
                    price_type,
                    f"{sell_prices[ts_code]:.2f}",
                    str(actual_shares),
                    f"{fill.amount:.2f}",
                    f"{fill.commission:.2f}",
                    f"{fill.stamp_tax:.2f}",
                    f"{fill.slippage:.2f}",
                    f"{fill.total_cost:.2f}",
                    reason
                ]
                logger.info(format_row(row, [12, 6, 10, 10, 10, 10, 12, 10, 10, 10, 10, 20], ['left'] * 12))
        
        # 2. 再执行买入指令
        buy_instructions = [i for i in instructions if i.action == 'buy']
        failed_buy_targets = []  # 记录买入失败的目标
        desired_position_count = max(
            [int(getattr(inst, 'desired_position_count', 0) or 0) for inst in buy_instructions],
            default=0,
        )
        price_map_for_threshold = {**sell_prices, **buy_prices}
        min_buy_value_threshold = self._get_min_buy_value_threshold(price_map_for_threshold)

        # 预加载 ATR 数据（如果可用）
        atr_map = {}
        try:
            from ...data import Storage as DataStorage
            ds = self.data_storage or DataStorage()
            features_df = ds.load_features_by_date(trade_date, subdir="cs_infer")
            if features_df is not None and "atr_pct_14" in features_df.columns:
                for _, row in features_df.iterrows():
                    atr_map[row["ts_code"]] = float(row["atr_pct_14"])
        except Exception:
            pass  # ATR 加载失败不影响正常买入

        for inst in buy_instructions:
            ts_code = inst.ts_code
            target_shares = inst.shares
            price_type = inst.price_type
            reason = inst.reason
            original_signal_date = getattr(inst, 'original_signal_date', '') or inst.source_date

            current_position = self.account.get_position(ts_code)
            if (
                desired_position_count > 0
                and current_position is None
                and len(self.account.get_positions()) >= desired_position_count
            ):
                logger.warning(
                    f"股票 {ts_code} 无可用空槽（目标持仓 {desired_position_count}，"
                    f"当前持仓 {len(self.account.get_positions())}），加入补位计划"
                )
                failed_buy_targets.append(
                    TargetWeight(
                        ts_code=ts_code,
                        target_weight=inst.target_weight,
                        reason=normalize_trade_reason(reason, append_suffix="（无可用空槽）"),
                        original_signal_date=original_signal_date,
                    )
                )
                continue
            
            # 检查价格数据
            if ts_code not in buy_prices:
                logger.warning(f"股票 {ts_code} 无买入价格数据，加入补位计划")
                failed_buy_targets.append(TargetWeight(
                    ts_code=ts_code,
                    target_weight=inst.target_weight,
                    reason=normalize_trade_reason(reason, append_suffix="（无价格数据）"),
                    original_signal_date=original_signal_date,
                ))
                continue
            
            # 检查可交易性
            can_buy, check_reason = self._check_can_buy(ts_code, tradability)
            if not can_buy:
                logger.warning(f"股票 {ts_code} 不可买入: {check_reason}，加入补位计划")
                failed_buy_targets.append(TargetWeight(
                    ts_code=ts_code,
                    target_weight=inst.target_weight,
                    reason=f"{reason}（{check_reason}）",
                    original_signal_date=original_signal_date,
                ))
                continue
            
            # 计算实际可买入股数（考虑现金约束）
            price = buy_prices[ts_code]
            available_cash = self.account.get_cash()
            
            # 预估成本
            target_amount = target_shares * price
            estimated_cost = self.cost_model.calculate_buy_cost(target_amount)
            
            # 检查现金是否足够
            if target_amount + estimated_cost > available_cash:
                # 现金不足，缩比买入
                max_amount = available_cash - estimated_cost
                if max_amount > 0:
                    actual_shares = int(max_amount / price / 100) * 100
                else:
                    actual_shares = 0
                
                if actual_shares <= 0:
                    logger.warning(f"股票 {ts_code} 现金不足，不足1手，加入补位计划")
                    failed_buy_targets.append(TargetWeight(
                        ts_code=ts_code,
                        target_weight=inst.target_weight,
                        reason=f"{reason}（现金不足）",
                        original_signal_date=original_signal_date,
                    ))
                    continue
                else:
                    logger.warning(f"股票 {ts_code} 现金不足，缩比买入 {actual_shares} 股（指令 {target_shares} 股）")
            else:
                actual_shares = target_shares
            
            # 创建订单并执行
            actual_buy_value = actual_shares * price
            if min_buy_value_threshold > 0 and actual_buy_value < min_buy_value_threshold:
                logger.warning(
                    f"股票 {ts_code} 买入后市值 {actual_buy_value:.2f} 低于阈值 "
                    f"{min_buy_value_threshold:.2f}，加入补位计划"
                )
                failed_buy_targets.append(
                    TargetWeight(
                        ts_code=ts_code,
                        target_weight=inst.target_weight,
                        reason=f"{reason}（买入后市值过小）",
                        original_signal_date=original_signal_date,
                    )
                )
                continue

            order = Order(
                ts_code=ts_code,
                action='buy',
                shares=actual_shares,
                price=price,
                target_weight=inst.target_weight,
                current_weight=0.0,  # 指令模式不需要权重
                reason=reason
            )

            fill = self._execute_single_order(
                order, trade_date, price_type,
                buy_atr_pct=atr_map.get(ts_code, 0.0),
            )
            if fill:
                fills.append(fill)
                # 打印执行详情
                row = [
                    ts_code,
                    'buy',
                    str(target_shares),
                    price_type,
                    f"{price:.2f}",
                    str(actual_shares),
                    f"{fill.amount:.2f}",
                    f"{fill.commission:.2f}",
                    f"{fill.stamp_tax:.2f}",
                    f"{fill.slippage:.2f}",
                    f"{fill.total_cost:.2f}",
                    reason
                ]
                logger.info(format_row(row, [12, 6, 10, 10, 10, 10, 12, 10, 10, 10, 10, 20], ['left'] * 12))
        
        # 保存延迟卖出队列
        if self.pending_sells:
            self.storage.save_pending_sells(self.pending_sells)
        
        # 记录买入失败目标（用于后续补位）
        self._failed_buy_targets = failed_buy_targets
        
        # 统计交易类型
        stats = self._calculate_execution_stats(fills, positions_before)
        
        logger.info("=" * 120)
        logger.info(f"执行完成: {len([f for f in fills if f.action == 'buy'])} 买，"
                   f"{len([f for f in fills if f.action == 'sell'])} 卖")
        logger.info(f"  - 买入: 新建持仓 {stats['new_position']} 笔，加仓 {stats['add_position']} 笔")
        logger.info(f"  - 卖出: 清仓 {stats['liquidate']} 笔，减仓 {stats['reduce_position']} 笔")
        logger.info(f"账户现金: {self.account.get_cash():,.2f}")
        logger.info(f"持仓数量: {len(self.account.get_positions())}")
        logger.info("=" * 120)
        
        return fills

    def _execute_single_order(
        self,
        order: Order,
        trade_date: str,
        price_type: str,
        buy_atr_pct: float = 0.0,
    ) -> Optional[Fill]:
        """执行单个订单

        Args:
            order: 订单
            trade_date: 交易日期
            price_type: 价格类型 open/close
            buy_atr_pct: 买入时 ATR 百分比（仅买入时使用）

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
            buy_pnl_price = self._resolve_buy_pnl_price(
                ts_code=order.ts_code,
                trade_date=trade_date,
                price_type=price_type,
                fallback_price=price,
            )
            self.account.update_cash(-total_required)
            self.account.add_position(
                ts_code=order.ts_code,
                shares=order.shares,
                buy_price=price,
                buy_cost=total_cost,
                buy_date=trade_date,
                buy_pnl_price=buy_pnl_price,
                buy_atr_pct=buy_atr_pct,
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

    def _resolve_buy_pnl_price(
        self,
        ts_code: str,
        trade_date: str,
        price_type: str,
        fallback_price: float,
    ) -> float:
        """解析买入绩效价格（后复权），失败时回退成交价。"""
        if fallback_price <= 0:
            return 0.0

        try:
            from ...data import DataLoader

            if self.data_storage is None:
                return float(fallback_price)

            loader = DataLoader(self.data_storage, verbose=False)
            daily_df = loader.load_clean_daily_by_date(trade_date)
            if daily_df is None or daily_df.empty:
                return float(fallback_price)

            stock_df = daily_df[daily_df["ts_code"] == ts_code]
            if stock_df.empty:
                return float(fallback_price)

            row = stock_df.iloc[0]
            if str(price_type) == "open":
                candidates = ["open_adj", "open", "close_adj", "close"]
            else:
                candidates = ["close_adj", "close", "open_adj", "open"]

            for col in candidates:
                value = row.get(col)
                if value is not None and not pd.isna(value) and float(value) > 0:
                    return float(value)
        except Exception:
            return float(fallback_price)

        return float(fallback_price)

    def _calculate_execution_stats(
        self,
        fills: List[Fill],
        positions_before: Dict[str, int]
    ) -> Dict[str, int]:
        """计算执行统计
        
        根据成交记录和执行前的持仓快照，统计各类交易操作的笔数
        
        Args:
            fills: 成交记录列表
            positions_before: 执行前的持仓快照 {ts_code: shares}
            
        Returns:
            统计字典：{
                'new_position': 新建持仓笔数,
                'add_position': 加仓笔数,
                'liquidate': 清仓笔数,
                'reduce_position': 减仓笔数
            }
        """
        stats = {
            'new_position': 0,
            'add_position': 0,
            'liquidate': 0,
            'reduce_position': 0
        }
        
        for fill in fills:
            if fill.action == 'buy':
                # 买入操作
                if fill.ts_code not in positions_before or positions_before[fill.ts_code] == 0:
                    # 原本没有持仓 -> 新建持仓
                    stats['new_position'] += 1
                else:
                    # 原本有持仓 -> 加仓
                    stats['add_position'] += 1
            elif fill.action == 'sell':
                # 卖出操作
                original_shares = positions_before.get(fill.ts_code, 0)
                if original_shares > 0 and fill.shares == original_shares:
                    # 卖出股数 == 原持仓股数 -> 清仓
                    stats['liquidate'] += 1
                else:
                    # 卖出股数 < 原持仓股数 -> 减仓
                    stats['reduce_position'] += 1
        
        return stats
