"""对称回补（terminal_loss 暴露门控 P2-4：减仓的对称操作）

**背景（R-004 §8 已登记的未验证项 2）**：P2-3 的主动减仓只做单向——减仓释放的现金要等
下一次调仓（约 20 交易日）才回补，于是"降暴露"实际覆盖了信号结束后一段时间：
实测 **68% 的收益来自"信号期外滞留"**，属**行为后果而非信号能力**；信号结束若上涨则持续付代价。
本模块实现**对称回补**：当暴露上限重新放开、且此前减仓释放的额度尚未回补时，
**立即**（T0 判定 → T+1 开盘）按比例把持仓补回，不新增第二条执行路径。

设计契约（与减仓严格对称）：

- **按比例对全部持仓加仓**，不做个股选择（个股截面选择已被 R-002 否定）；
- **只在两次调仓之间生效**：**买入计划生成日**（`date in self.pending_signals`，即 T0 排队新计划的当天）
  **不干预**，并把“释放额度”归零——新计划会用同一笔现金重新分配仓位，
  回补不得与调仓计划抢现金（**不得**用“调仓在途/补齐未完成”当条件：
  `pending_signals` 会在执行日前一直存在、`unfilled_slots` 会持续补齐窗口，
  用作条件会让回补在真实链路里永不生效，实测已证）；
- **上界三选一**：`回补额 = min(可用现金, 未回补的减仓释放额, λ_t×组合总值 − 当前持仓市值)`
  ⇒ 回补**永远不超过被减掉的那部分**（不把 λ 当目标仓位去做满仓），也永远不超上限；
- 共用减仓容差（`exposure_trim_tolerance`）：差额未超容差不动作，避免"减了又补"抖动；
- 不可买入（停牌/涨停/无行情）时**跳过该股并由次日判定重试**，不进延迟订单队列
  （与减仓同一理由：延迟队列按不同语义重试）；
- 引擎默认关闭（`exposure_replenish_enabled=False`）：不产生任何副作用，与既有回测**逐位一致**。

已知近似（登记）：按持仓市值顺序分配回补额（顺序 = 持仓字典顺序，非打分），
现金不足时**靠后的股票可能少补或不补**；加仓不改变原持仓的买入日（持有期与退出计划不变），
`buy_cost_cash` 按加权平均累加。若排队后该股已被整仓卖出（不再持有），该笔额度**不退回**
（对应现金已由后续调仓计划重新分配，退回会造成与其他持仓抢现金）。
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from loguru import logger

from ..common.date_utils import to_trade_date_str
from ..common.trade_status import is_tradeable
from ..trading.sizing import compute_lot_shares

#: 买入类型与触发类型（落盘到成交记录，供归因使用）
REPLENISH_BUY_TYPE = "risk_replenish"
REPLENISH_TRIGGER_TYPE = "exposure_gate"


class BacktestExposureReplenishMixin:
    """每日对称回补 mixin（暴露门控 P2-4）

    依赖 `exposure_trim.py` 的 `exposure_release_budget`（减仓累计释放并**未回补**的金额）：
    由减仓模块在每次部分卖出成交后累加，本模块在回补成交后扣减。
    开关关闭（`exposure_replenish_enabled=False`）或暴露表为 None 时不产生任何副作用。
    """

    def _init_exposure_replenish_state(self) -> None:
        """初始化回补状态（由引擎 `__init__` 调用）"""
        # {股票: {"trigger_date": 判定日, "signal_date": 判定日, "amount": 目标回补金额}}
        self.pending_exposure_replenishes: Dict[str, Dict[str, object]] = {}
        self.exposure_replenish_enabled: bool = False
        if not hasattr(self, "exposure_release_budget"):
            # 减仓模块未接线时的兜底（单一定义：释放额度 = 0 表示无可回补）
            self.exposure_release_budget: float = 0.0
        self.exposure_replenish_stats: Dict[str, object] = {
            "trigger_days": 0,  # 触发回补判定的交易日数
            "replenish_orders": 0,  # 生成的回补买单数（T0）
            "replenish_orders_filled": 0,  # 实际成交的买单数（T+1）
            "replenish_orders_skipped": 0,  # 不可交易而跳过的买单数（次日重判）
            "replenish_bought_amount": 0.0,  # 回补买入金额合计（含手续费）
            "skipped_days_below_tolerance": 0,  # 差额未达容差的交易日数
            "skipped_days_no_budget": 0,  # 无未回补释放额的交易日数
            "budget_reset_on_rebalance": 0,  # 新调仓计划生成而归零释放额的次数
            "budget_discount_release_amount": 0.0,  # 建仓折扣入释放额累计（A3 v2）
        }

    # ------------------------------------------------------------------ T0：判定
    def _queue_exposure_replenish(
        self,
        date: pd.Timestamp,
        trading_dates: list,
        date_to_idx: Dict[pd.Timestamp, int],
    ) -> None:
        """每日回补判定：暴露上限放开且存在未回补减仓额时，按比例把持仓排队到 T+1 买入

        Args:
            date: 判定日（T0）
            trading_dates: 交易日列表（保持接口一致，本方法不使用）
            date_to_idx: 日期到索引映射（保持接口一致，本方法不使用）
        """
        if not self.exposure_replenish_enabled or not self._has_exposure_source():
            return
        budget = float(getattr(self, "exposure_release_budget", 0.0) or 0.0)
        if budget <= 0:
            self.exposure_replenish_stats["skipped_days_no_budget"] += 1
            return
        # 新调仓计划生成日：计划会用同一笔现金重新分配仓位（按信号日 λ 缩放），
        # 释放额度归零，回补不得与调仓计划抢现金；两次调仓之间才允许回补。
        if date in self.pending_signals:
            self.exposure_release_budget = 0.0
            self.exposure_replenish_stats["budget_reset_on_rebalance"] += 1
            if self.verbose:
                logger.info(
                    f"暴露门控回补：{date.date()} 新调仓计划生成，"
                    f"释放额 {budget:.2f} 归零（交由新计划重新分配）"
                )
            return
        if not self.positions:
            return

        portfolio_value = self._calculate_portfolio_value(date)
        if portfolio_value <= 0:
            return
        position_value = portfolio_value - self.current_capital
        room = self._get_exposure_multiplier(date) * portfolio_value - position_value
        topup = min(self.current_capital, budget, max(room, 0.0))
        tolerance = portfolio_value * self.exposure_trim_tolerance
        if topup <= tolerance:
            self.exposure_replenish_stats["skipped_days_below_tolerance"] += 1
            return

        # 目标：对**全部持仓**按市值比例加仓（不做个股选择）；跳过正在排队卖出的股票
        candidates = []
        for stock in list(self.positions.keys()):
            if self.positions[stock]["shares"] <= 0:
                continue
            if (
                stock in self.pending_condition_sells
                or stock in self.pending_stop_loss_sells
                or stock in self.pending_exposure_trims
            ):
                continue
            value = self._position_market_value(date, stock)
            if value is None or value <= 0:
                continue
            candidates.append((stock, value))
        total_value = sum(value for _, value in candidates)
        if total_value <= 0:
            logger.warning(f"暴露门控回补：{date.date()} 判定可补 {topup:.2f}，但无可回补标的")
            return

        queued = 0
        for stock, value in candidates:
            amount = topup * (value / total_value)
            if amount <= 0:
                continue
            price = self._get_trade_price_open(date, stock) or self._get_trade_price(date, stock)
            if price is None or compute_lot_shares(amount, price) <= 0:
                continue  # 不足一手：比例语义优先，跳过（见模块 docstring 已知近似）
            self.pending_exposure_replenishes[stock] = {
                "trigger_date": date,
                "signal_date": date,
                "amount": amount,
            }
            queued += 1
        if queued == 0:
            logger.warning(
                f"暴露门控回补：{date.date()} 判定已放开 {topup:.2f}，但无可回补标的（均不足一手）"
            )
            return
        self.exposure_replenish_stats["trigger_days"] += 1
        self.exposure_replenish_stats["replenish_orders"] += queued
        # 释放额按本次计划回补额扣减（现金不足/不可交易由 T+1 执行侧消化）
        self.exposure_release_budget = max(0.0, budget - topup)
        if self.verbose:
            logger.info(
                f"暴露门控回补：{date.date()} 暴露上限放开，回补 {topup:.2f}"
                f"（敞口 {position_value:.2f} → 目标上界 "
                f"{self._get_exposure_multiplier(date) * portfolio_value:.2f}），"
                f"{queued} 只按比例 T+1 买入"
            )

    # ------------------------------------------------------------------ T+1：执行
    def _execute_pending_exposure_replenishes(
        self,
        date: pd.Timestamp,
        trading_dates: list,
        date_to_idx: Dict[pd.Timestamp, int],
    ) -> None:
        """执行待执行的回补买单（T+1 开盘）

        Args:
            date: 执行日（T+1）
            trading_dates: 交易日列表（保持接口一致，本方法不使用）
            date_to_idx: 日期到索引映射（保持接口一致，本方法不使用）
        """
        if not self.exposure_replenish_enabled or not self.pending_exposure_replenishes:
            return

        pending = self.pending_exposure_replenishes
        self.pending_exposure_replenishes = {}
        for stock, info in pending.items():
            if stock not in self.positions:
                continue
            tradeable, reason = self._can_replenish_now(date, stock)
            if not tradeable:
                self.exposure_replenish_stats["replenish_orders_skipped"] += 1
                # 未成交的额度退回释放额，由次日判定重试（不丢额度）
                self.exposure_release_budget = float(
                    getattr(self, "exposure_release_budget", 0.0)
                ) + float(info["amount"])
                if self.verbose:
                    logger.info(
                        f"暴露门控回补：{date.date()} {stock} 不可交易（{reason}），"
                        "跳过并等待次日重判"
                    )
                continue
            invested = self._add_to_position(
                date,
                stock,
                float(info["amount"]),
                buy_type=REPLENISH_BUY_TYPE,
                buy_reason=f"暴露门控回补（{info['trigger_date'].date()} 判定）",
            )
            # 未花掉的额度（部分成交/未成交）**退回释放额**，由次日判定重试（额度不丢）
            unspent = max(0.0, float(info["amount"]) - invested)
            if unspent > 0:
                self.exposure_release_budget = (
                    float(getattr(self, "exposure_release_budget", 0.0)) + unspent
                )
            if invested > 0:
                self.exposure_replenish_stats["replenish_orders_filled"] += 1
                self.exposure_replenish_stats["replenish_bought_amount"] += invested
            else:
                self.exposure_replenish_stats["replenish_orders_skipped"] += 1

    def _can_replenish_now(self, date: pd.Timestamp, stock: str) -> tuple:
        """买入可交易性检查（停牌/涨停/无行情 ⇒ 不可买入）；与交易状态检查同一实现。"""
        if self.price_data_cache is None:
            return True, ""
        trade_date_str = to_trade_date_str(date)
        quote = self.price_data_cache[self.price_data_cache["trade_date"] == trade_date_str]
        if quote.empty:
            return False, "无行情数据"
        return is_tradeable(stock, trade_date_str, quote, action="buy")

    # ------------------------------------------------------------------ 报告
    def get_exposure_replenish_report(self) -> Dict[str, object]:
        """返回回补统计（含开关状态、待执行队列长度与未回补释放额）"""
        report: Dict[str, object] = dict(self.exposure_replenish_stats or {})
        report["enabled"] = bool(self.exposure_replenish_enabled)
        report["pending_orders"] = len(self.pending_exposure_replenishes)
        report["release_budget"] = float(getattr(self, "exposure_release_budget", 0.0) or 0.0)
        return report
