"""每日风控减仓（terminal_loss 暴露门控 P2-3：主动减仓）

设计契约（见 CLAUDE.md「暴露覆盖契约」）：

- **判定每日进行**：每个交易日收盘后按当日暴露系数判断组合是否超配；
- **判定出风险立刻减仓**：T 日生成卖单、T+1 开盘执行，复用既有 T0/T1 指令链路，
  不新增第二条执行路径（引擎侧只允许"减仓"与"缩放买入预算"两件事）；
- **按比例对全部持仓减仓**，不做个股选择：个股截面退出已被风险登记 R-002 否定，
  本模块只做组合级暴露控制；
- 触发带容差（`EXPOSURE_TRIM_TOLERANCE`，组合总值的 3%），避免反复微调产生换手成本；
- 买入预算缩放（`exposure_override.py`）负责"暂停加仓"，两者配合才是完整降暴露语义：
  减仓把**存量**暴露在 T+1 开盘降到目标附近，缩放保证**增量**不再买回。

不可交易（停牌/跌停/无行情）时**不使用延迟订单队列整仓卖出**（那会放大成整仓卖出、
超出比例减仓语义），而是跳过该股并由次日的每日判定自然重试。
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from loguru import logger

# 触发容差：仅当超配金额超过组合总值的该比例才减仓（单一默认值，无多层回退）
EXPOSURE_TRIM_TOLERANCE = 0.03

# 卖出类型与触发类型（落盘到成交记录，供归因使用）
TRIM_SELL_TYPE = "risk_trim"
TRIM_TRIGGER_TYPE = "exposure_gate"


class BacktestExposureTrimMixin:
    """每日风控减仓 mixin

    引擎默认关闭：`exposure_table` 为 None 时本模块所有逻辑不产生任何副作用
    （`_queue_exposure_trim` 直接返回），保证开关关闭时逐位一致。
    """

    def _init_exposure_trim_state(self) -> None:
        """初始化减仓状态（由引擎 `__init__` 调用）"""
        # {股票: {"trigger_date": 判定日, "fraction": 卖出比例, "signal_date": 判定日}}
        self.pending_exposure_trims: Dict[str, Dict[str, object]] = {}
        self.exposure_trim_tolerance: float = EXPOSURE_TRIM_TOLERANCE
        self.exposure_trim_stats: Dict[str, object] = {
            "trigger_days": 0,  # 触发减仓判定的交易日数
            "trim_orders": 0,  # 生成的减仓卖单数（T0）
            "trim_orders_sold": 0,  # 实际成交的减仓卖单数（T+1）
            "trim_orders_skipped": 0,  # 不可交易而跳过的卖单数（次日重判）
            "trim_sold_amount": 0.0,  # 减仓成交金额合计
            "skipped_days_below_tolerance": 0,  # 超配但未达容差的交易日数
        }
        # 减仓释放的现金额度（**未回补部分**）：供对称回补（exposure_replenish.py）使用。
        # 每次部分卖出成交后累加，由回补模块在回补成交后扣减（单一数值定义，无多层回退）。
        self.exposure_release_budget: float = 0.0

    # ------------------------------------------------------------------ T0：判定
    def _queue_exposure_trim(
        self,
        date: pd.Timestamp,
        trading_dates: list,
        date_to_idx: Dict[pd.Timestamp, int],
    ) -> None:
        """每日风控判定：组合是否超配，超配则按比例把全部持仓排队到 T+1 卖出

        Args:
            date: 判定日（T0）
            trading_dates: 交易日列表（保持接口一致，本方法不使用）
            date_to_idx: 日期到索引映射（保持接口一致，本方法不使用）
        """
        if not self._has_exposure_source() or not self.positions:
            return

        multiplier = self._get_exposure_multiplier(date)
        if multiplier >= 1.0:
            return

        portfolio_value = self._calculate_portfolio_value(date)
        if portfolio_value <= 0:
            return

        position_value = portfolio_value - self.current_capital
        if position_value <= 0:
            return

        target_value = portfolio_value * multiplier
        excess = position_value - target_value
        tolerance = portfolio_value * self.exposure_trim_tolerance
        if excess <= tolerance:
            self.exposure_trim_stats["skipped_days_below_tolerance"] += 1
            return

        fraction = min(1.0, excess / position_value)
        queued = 0
        for stock in list(self.positions.keys()):
            if self.positions[stock]["shares"] <= 0:
                continue
            if (
                stock in self.pending_condition_sells
                or stock in self.pending_stop_loss_sells
                or stock in self.pending_exposure_trims
            ):
                continue  # 已在其他卖出队列（整仓卖出优先）
            if self._resolve_trim_shares(self.positions[stock]["shares"], fraction) <= 0:
                continue
            self.pending_exposure_trims[stock] = {
                "trigger_date": date,
                "signal_date": date,
                "fraction": fraction,
            }
            queued += 1

        self.exposure_trim_stats["trigger_days"] += 1
        if queued == 0:
            logger.warning(
                f"暴露门控减仓：{date.date()} 判定超配 {excess:.2f}（目标 {target_value:.2f}，"
                f"当前持仓市值 {position_value:.2f}），但无可减仓标的"
            )
            return
        self.exposure_trim_stats["trim_orders"] += queued
        if self.verbose:
            logger.info(
                f"暴露门控减仓：{date.date()} 判定超配 {excess:.2f}"
                f"（暴露系数 {multiplier:.2f}，目标持仓 {target_value:.2f}，"
                f"实际持仓 {position_value:.2f}），{queued} 只按 {fraction:.1%} 比例 T+1 减仓"
            )

    # ------------------------------------------------------------------ T+1：执行
    def _execute_pending_exposure_trims(
        self,
        date: pd.Timestamp,
        trading_dates: list,
        date_to_idx: Dict[pd.Timestamp, int],
    ) -> None:
        """执行待执行的减仓卖单（T+n 开盘）

        Args:
            date: 执行日（T+1）
            trading_dates: 交易日列表（保持接口一致，本方法不使用）
            date_to_idx: 日期到索引映射（保持接口一致，本方法不使用）
        """
        # 注意：政策源可能是在线 provider（此时 exposure_table 为 None），
        # 必须统一用 _has_exposure_source 判定，否则减仓单会被静默丢弃
        if not self._has_exposure_source() or not self.pending_exposure_trims:
            return

        pending = self.pending_exposure_trims
        self.pending_exposure_trims = {}

        for stock, info in pending.items():
            if stock not in self.positions:
                continue
            fraction = float(info["fraction"])
            if self._resolve_trim_shares(self.positions[stock]["shares"], fraction) <= 0:
                continue
            shares_before = self.positions[stock]["shares"]
            # 不使用延迟订单队列：不可交易时跳过，由次日每日判定重试（避免被放大成整仓卖出）
            self._sell_stock(
                date,
                stock,
                sell_type=TRIM_SELL_TYPE,
                sell_reason=f"暴露门控减仓（{info['trigger_date'].date()} 判定）",
                trigger_type=TRIM_TRIGGER_TYPE,
                fraction=fraction,
                allow_pending=False,
            )
            shares_after = self.positions[stock]["shares"] if stock in self.positions else 0
            if shares_after < shares_before:
                sold_shares = shares_before - shares_after
                price = self._get_trade_price_open(date, stock)
                if price is None:
                    price = self._get_trade_price(date, stock)
                self.exposure_trim_stats["trim_orders_sold"] += 1
                if price is not None:
                    self.exposure_trim_stats["trim_sold_amount"] += sold_shares * price
                    # 释放额度累加（对称回补的上界；见 exposure_replenish.py）
                    self.exposure_release_budget += sold_shares * price
            else:
                self.exposure_trim_stats["trim_orders_skipped"] += 1
                if self.verbose:
                    logger.info(f"暴露门控减仓：{date.date()} {stock} 不可交易，跳过并等待次日重判")

    # ------------------------------------------------------------------ 报告
    def get_exposure_trim_report(self) -> Dict[str, object]:
        """返回减仓统计（含容忍度、待执行队列长度等）"""
        report: Dict[str, object] = dict(self.exposure_trim_stats or {})
        report["tolerance"] = self.exposure_trim_tolerance
        report["pending_orders"] = len(self.pending_exposure_trims)
        report["release_budget"] = float(getattr(self, "exposure_release_budget", 0.0) or 0.0)
        return report
