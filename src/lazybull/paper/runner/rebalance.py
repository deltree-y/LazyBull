# -*- coding: utf-8 -*-
"""PaperRebalanceMixin：src/lazybull/paper/runner.py 拆分出的 _check_rebalance_day, _check_single_rebalance_day, _check_staggered_rebalance_day。"""

from ...trading.stagger import build_tranche_schedule_from_anchor
from loguru import logger
from typing import Tuple

class PaperRebalanceMixin:
    def _check_rebalance_day(
        self,
        trade_date: str,
        rebalance_freq: int,
        stagger_tranches: int = 1,
    ) -> Tuple[bool, int]:
        """检查是否为调仓日，并返回批次索引

        Args:
            trade_date: 交易日期 YYYYMMDD
            rebalance_freq: 调仓频率（交易日数）
            stagger_tranches: 分批调仓批次数（1=不分批）

        Returns:
            (是否为调仓日, tranche_idx)。stagger_tranches=1 时 tranche_idx 恒为 0。

        Raises:
            RuntimeError: 如果不是调仓日
        """
        # 加载调仓状态
        rebalance_state = self.paper_storage.load_rebalance_state()

        # 首次运行，允许执行（tranche_idx=0 全量建仓）
        if rebalance_state is None:
            logger.info("首次运行T0，允许执行")
            return True, 0

        last_rebalance_date = rebalance_state.get('last_rebalance_date')
        if not last_rebalance_date:
            logger.info("无上次调仓记录，允许执行")
            return True, 0

        # 分批调仓模式：基于锚定日推算排期表
        if stagger_tranches > 1:
            return self._check_staggered_rebalance_day(
                trade_date, rebalance_freq, stagger_tranches, rebalance_state
            )

        # 不分批模式：保持原有逻辑
        return self._check_single_rebalance_day(trade_date, rebalance_freq, last_rebalance_date)

    def _check_single_rebalance_day(
        self, trade_date: str, rebalance_freq: int, last_rebalance_date: str
    ) -> Tuple[bool, int]:
        """不分批模式的调仓日检查（原有逻辑）。"""
        try:
            trade_cal = self.loader.load_clean_trade_cal()
            if trade_cal is None:
                logger.error("无法加载交易日历，跳过调仓日检查")
                return True, 0

            trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()

            try:
                last_idx = trade_dates.index(last_rebalance_date)
                current_idx = trade_dates.index(trade_date)
            except ValueError as e:
                logger.error(f"日期不在交易日历中: {e}")
                return True, 0

            days_since_last = current_idx - last_idx

            if days_since_last >= rebalance_freq:
                logger.info(
                    f"距离上次调仓 {last_rebalance_date} 已过 [{days_since_last}] 个交易日，"
                    f"满足调仓频率 {rebalance_freq}，允许执行"
                )
                return True, 0
            else:
                raise RuntimeError(
                    f"当前不是调仓日！距离上次调仓 {last_rebalance_date} "
                    f"仅过 [{days_since_last}] 个交易日，"
                    f"需要至少 {rebalance_freq} 个交易日。"
                )
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"检查调仓日失败: {e}，跳过检查")
            return True, 0

    def _check_staggered_rebalance_day(
        self,
        trade_date: str,
        rebalance_freq: int,
        stagger_tranches: int,
        rebalance_state: dict,
    ) -> Tuple[bool, int]:
        """分批调仓模式的调仓日检查。

        基于 tranche_anchor_date（批次0锚定日）推算各批次排期，
        当前日在排期表中则允许执行，并返回对应的 tranche_idx。
        """
        try:
            trade_dates = self._get_open_trade_dates()
            if not trade_dates:
                logger.error("无法加载交易日历，跳过分批调仓日检查")
                return True, 0

            # 使用 tranche_anchor_date 作为锚定日，兼容旧版 last_rebalance_date
            anchor_date = rebalance_state.get('tranche_anchor_date') or rebalance_state.get(
                'last_rebalance_date'
            )
            if not anchor_date:
                logger.info("无分批锚定日记录，允许执行（tranche_idx=0）")
                return True, 0

            # 基于锚定日推算排期表
            schedule = build_tranche_schedule_from_anchor(
                anchor_date, trade_dates, rebalance_freq, stagger_tranches
            )

            if trade_date in schedule:
                tranche_idx = schedule[trade_date]
                logger.info(
                    f"分批调仓日命中: {trade_date} → 批次 {tranche_idx + 1}/{stagger_tranches}"
                    f"（锚定日 {anchor_date}）"
                )
                return True, tranche_idx

            # 不在排期表中，计算最近的调仓日
            future_dates = sorted(d for d in schedule if d >= trade_date)
            next_rebalance = future_dates[0] if future_dates else "无"
            raise RuntimeError(
                f"当前不是调仓日！分批模式（{stagger_tranches} 批，锚定日 {anchor_date}），"
                f"下一调仓日: {next_rebalance}"
            )
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"分批调仓日检查失败: {e}，跳过检查")
            return True, 0
