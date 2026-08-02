# -*- coding: utf-8 -*-
"""PaperInstructionMixin：src/lazybull/paper/runner.py 拆分出的 _generate_instructions, evaluate_holding_period_actions, _build_rebalance_sell_instructions。"""

from ..common.constants import SHARE_LOT_SIZE
from ..trading.sell_rules import min_holding_days_for_rebalance_sell
from ..trading.sell_rules import select_rebalance_sell_candidates
from ..trading.sizing import compute_lot_shares
from ..trading.stagger import get_tranche_capital_fraction as _shared_tranche_capital_fraction
from .models import TargetWeight
from .models import TradeInstruction
from loguru import logger
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

class PaperInstructionMixin:
    def _generate_instructions(
        self,
        targets: List[TargetWeight],
        buy_price_type: str,
        sell_price_type: str,
        current_prices: Dict[str, float],
        source_date: str,
        protected_stocks: Optional[set] = None,
        desired_position_count: Optional[int] = None,
        tranche_idx: int = 0,
        overall_top_n: Optional[int] = None,
        stagger_tranches: int = 1,
    ) -> List[TradeInstruction]:
        """从目标权重生成明确的交易指令

        说明：与回测对齐后，纸面交易卖出主路径由"持有期/条件驱动"负责。
        本方法仅负责按目标权重生成买入/加仓指令，不再基于目标权重生成减仓/清仓卖出。

        Args:
            targets: 目标权重列表
            buy_price_type: 买入价格类型 open/close
            sell_price_type: 卖出价格类型 open/close（保留参数，兼容接口）
            current_prices: 当前价格字典
            source_date: 源日期（T0日期）
            protected_stocks: 盈利延续保护的股票集合，跳过卖出指令生成
            tranche_idx: 分批调仓批次索引（0-based）
            overall_top_n: 组合最终总持仓数（分批时使用）
            stagger_tranches: 分批调仓批次数（1=不分批）

        Returns:
            交易指令列表
        """
        instructions = []
        # 注：sell_price_type / protected_stocks 为兼容旧调用保留，本方法不再参与卖出逻辑

        # 目标权重字典（供快速查找）
        target_weights = {t.ts_code: (t.target_weight, t.reason) for t in targets}
        # 保持与信号输出一致的顺序，避免 set 无序遍历导致现金受限时结果不稳定
        ordered_target_codes = [t.ts_code for t in targets]

        # 当前持仓
        current_positions = self.account.get_positions()

        # 使用账户总资金计算
        capital_retention_ratio = self._get_cost_setting("capital_retention_ratio", 0.0)

        #total_capital = self.account.initial_capital #???应使用当前总资产,可以乘一个系数
        total_capital = self.account.get_total_value(current_prices) * (1 - capital_retention_ratio)  # 乘以系数以留出现金空间，避免过度买入

        # 分批调仓：按本批槽位占总 top_n 的比例分配组合价值（与回测对齐）
        if stagger_tranches > 1 and overall_top_n and overall_top_n > 0:
            capital_fraction = _shared_tranche_capital_fraction(
                tranche_idx, overall_top_n, stagger_tranches
            )
            total_capital *= capital_fraction
            logger.info(
                f"  分批资金分配: 批次 {tranche_idx + 1}/{stagger_tranches}, "
                f"预算比例={capital_fraction:.2%}, 可用资金={total_capital:,.0f}"
            )

        # 仅处理目标股票买入/加仓（按目标顺序）
        desired_position_count = int(desired_position_count or len(ordered_target_codes))
        for ts_code in ordered_target_codes:
            target_weight, reason = target_weights.get(ts_code, (0.0, "退出持仓"))
            pos = current_positions.get(ts_code)
            current_shares = pos.shares if pos else 0

            # 获取价格
            price = current_prices.get(ts_code, 0.0)
            if price <= 0:
                logger.warning(f"股票 {ts_code} 无价格数据，跳过生成指令")
                continue

            # 计算目标股数
            target_value = total_capital * target_weight
            target_shares = compute_lot_shares(target_value, price, SHARE_LOT_SIZE)

            # 判断操作类型
            if target_shares > current_shares:
                # 买入或加仓
                shares = (target_shares - current_shares) // SHARE_LOT_SIZE * SHARE_LOT_SIZE
                if shares > 0:
                    instructions.append(TradeInstruction(
                        ts_code=ts_code,
                        action='buy',
                        shares=shares,
                        price_type=buy_price_type,
                        reason=reason,
                        source_date=source_date,
                        target_weight=target_weight,
                        original_signal_date=source_date,
                        desired_position_count=desired_position_count,
                    ))
            # target_shares <= current_shares 时不生成卖出指令：
            # 卖出统一由持有期到期/条件触发路径处理。

        logger.info(f"生成 {len(instructions)} 条交易指令")
        return instructions

    def evaluate_holding_period_actions(
        self,
        trade_date: str,
        config: dict,
        exclude_stocks: Optional[set] = None,
    ) -> Tuple[set, list]:
        """按交易日评估持有期到期卖出（对齐回测口径）。

        盈亏动态持仓功能已移除，仅保留持有期到期硬卖逻辑：
        阈值 = max(1, rebalance_freq - 1)，确保 T+1 执行日恰好在持有期满当天卖出。

        Args:
            trade_date: 当前交易日期 YYYYMMDD
            config: 配置字典
            exclude_stocks: 需跳过评估的股票集合（如已有卖出指令）

        Returns:
            (protected_stocks, sell_actions)，保护集恒为空集
        """
        positions = self.account.get_positions()
        if not positions:
            return set(), []

        trade_cal = self.loader.load_clean_trade_cal()
        trade_dates_list = []
        if trade_cal is not None:
            trade_dates_list = trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()

        rebalance_freq = int(config.get("rebalance_freq", 20))
        min_holding_for_sell = min_holding_days_for_rebalance_sell(rebalance_freq, floor=1)
        exclude_stocks = exclude_stocks or set()

        sell_actions = []
        for ts_code, pos in positions.items():
            if ts_code in exclude_stocks:
                continue
            holding_days = self._calc_holding_days(pos.buy_date, trade_date, trade_dates_list)
            if holding_days < min_holding_for_sell:
                continue
            sell_shares = (pos.shares // SHARE_LOT_SIZE) * SHARE_LOT_SIZE
            if sell_shares <= 0:
                continue
            sell_actions.append(
                {
                    "ts_code": ts_code,
                    "shares": sell_shares,
                    "reason": f"持有期到期: 持有{holding_days}天",
                    "can_execute": True,
                }
            )
            if self.verbose:
                logger.info(f"  持有期到期卖出: {ts_code} 持有{holding_days}天 -> {sell_shares}股")
        return set(), sell_actions

    def _build_rebalance_sell_instructions(
        self,
        trade_date: str,
        sell_price_type: str,
        protected_stocks: set,
        target_codes: set,
        rebalance_freq: int = 20,
        trade_dates_list: Optional[list] = None,
    ) -> List[TradeInstruction]:
        """在调仓日生成当前持仓的清仓卖出指令。

        仅针对不在新目标列表且不受盈利延续保护的持仓生成卖出指令，
        使卖出与买入对齐到同一 T+1 执行日。

        提前调仓（holding tail）场景下，仅卖出已达到持有期阈值的持仓，
        避免误卖尚未到期的年轻持仓。

        Args:
            trade_date: 调仓日（T0 日期）
            sell_price_type: 卖出价格类型 open/close
            protected_stocks: 盈利延续保护的股票集合
            target_codes: 新信号目标股票集合（保留，不卖出）
            rebalance_freq: 调仓频率（交易日数），用于持有天数阈值判定
            trade_dates_list: 交易日列表，用于计算持有天数

        Returns:
            卖出指令列表
        """
        instructions: List[TradeInstruction] = []
        positions = self.account.get_positions()
        if not positions:
            return instructions

        protected = protected_stocks or set()

        # 构建持有天数映射（无交易日列表时为 None，不做年轻持仓过滤）
        holding_days_map: Dict[str, Optional[int]] = {}
        for ts_code, pos in positions.items():
            if trade_dates_list:
                holding_days_map[ts_code] = self._calc_holding_days(
                    pos.buy_date, trade_date, trade_dates_list
                )
            else:
                holding_days_map[ts_code] = None

        # 共享调仓卖出筛选（纸面阈值下限 floor=1：保护当日买入的持仓）
        decision = select_rebalance_sell_candidates(
            holding_days_map,
            min_holding_days=min_holding_days_for_rebalance_sell(rebalance_freq, floor=1),
            target_codes=target_codes,
            protected_codes=protected,
        )

        sell_count = 0
        for ts_code in decision.sells:
            pos = positions[ts_code]
            sell_shares = (pos.shares // SHARE_LOT_SIZE) * SHARE_LOT_SIZE
            if sell_shares <= 0:
                continue

            instructions.append(
                TradeInstruction(
                    ts_code=ts_code,
                    action="sell",
                    shares=sell_shares,
                    price_type=sell_price_type,
                    reason="调仓卖出",
                    source_date=trade_date,
                    target_weight=0.0,
                )
            )
            sell_count += 1

        if sell_count > 0 or decision.skipped_too_young > 0:
            skip_parts = []
            if decision.skipped_protected > 0:
                skip_parts.append(f"{decision.skipped_protected} 只盈利延续保护")
            if decision.skipped_target > 0:
                skip_parts.append(f"{decision.skipped_target} 只在新目标中")
            if decision.skipped_too_young > 0:
                skip_parts.append(f"{decision.skipped_too_young} 只未满持有期")
            skip_info = f"（{', '.join(skip_parts)}）" if skip_parts else ""
            logger.info(f"调仓日生成卖出指令: {sell_count} 只股票{skip_info}")

        return instructions
