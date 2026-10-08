"""回测信号执行 mixin。

v2 迁移复制件（P2a-T5，行为冻结）：与旧模块 ``backtest/signal_execution.py``
双源并存至 v2 切换；除模块头外，主体差异仅限 import 改指与 D8 提取式拆分
（见文末登记）。

import 改指：``common.date_utils`` → ``v2.common.date_utils``；
``portfolio`` 的 ``cap_and_normalize_weights`` / ``resolve_tranche_weight_cap``
→ ``v2.core.decision.weight_processor``；函数体内延迟 import 的
``portfolio.apply_industry_constraint`` → ``v2.core.decision.industry_constraint``
（T3 已迁复制件，保持函数内延迟形态不变）。

D8 提取式拆分（T5 落位时执行）：旧路径基线在册硬超限 ``_generate_signal``
（mccabe 17，函数 201 行 > 硬上限 150），复制到新路径 = 新增未豁免硬超限 ⇒
按「提取模块级私有 helper + 委托」拆出 5 个模块级私有函数
（排序候选准备 / T+1 买入日解析 / 行业约束与候选过滤 / 批次目标数量解析 /
权重归一化与上限 / 信号计划入队），公开签名、数值操作顺序、日志文本逐字保留。
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.v2.common.date_utils import to_trade_date_str
from src.lazybull.v2.core.decision.weight_processor import (
    cap_and_normalize_weights,
    resolve_tranche_weight_cap,
)


def _prepare_ranked_candidates(
    self: "BacktestSignalExecutionMixin",
    date: pd.Timestamp,
    price_data: pd.DataFrame,
) -> Optional[list]:
    """构建股票池并生成排序候选（D8 提取件；含净值快照与两处守卫日志，逐字保留）。

    Returns:
        排序候选列表；无可用数据或无候选时返回 None（调用方直接 return）
    """
    # 记录调仓日组合净值，用于止盈基准和"本调仓收益"计算
    self._last_rebalance_nav = self._calculate_portfolio_value(date)

    # 获取当日行情数据用于基础过滤（ST、停牌等基础过滤）
    trade_date_str = to_trade_date_str(date)
    date_quote = price_data[price_data["trade_date"] == trade_date_str]
    # 获取股票池（不过滤涨跌停，因为 T 日涨跌停不代表 T+1 日也涨跌停）
    # 但保留 ST、基本可交易性等过滤
    stock_universe = self.universe.get_stocks(date, quote_data=date_quote)

    # 调用扩展点获取额外数据（如 ML 特征）
    extra_data = self._build_signal_data(date)
    if extra_data is None:
        # None 表示该日期无可用数据，跳过信号生成
        if self.verbose:
            logger.warning(f"信号日 {date.date()} 无可用数据（_build_signal_data 返回 None），跳过")
        return None

    # 合并默认数据和额外数据
    signal_data = {}
    signal_data.update(extra_data)

    # 生成排序后的候选列表（返回所有候选，不仅仅是 top N）
    ranked_candidates = self.signal.generate_ranked(date, stock_universe, signal_data)

    if not ranked_candidates:
        if self.verbose:
            logger.warning(f"信号日 {date.date()} 无候选")
        return None

    return ranked_candidates


def _resolve_buy_date_quote(
    self: "BacktestSignalExecutionMixin",
    date: pd.Timestamp,
    trading_dates: List[pd.Timestamp],
    price_data: pd.DataFrame,
    date_to_idx: Dict,
) -> Optional[Tuple[pd.Timestamp, pd.DataFrame]]:
    """解析 T+1 买入日及其行情（D8 提取件，守卫日志文本逐字保留）。

    Returns:
        (buy_date, buy_date_quote)；没有 T+1 日时返回 None（调用方直接 return）
    """
    # 获取 T+1 日（买入日）的行情数据
    current_idx = date_to_idx.get(date)
    if current_idx is None or current_idx + 1 >= len(trading_dates):
        # 没有 T+1 日，无法买入
        if self.verbose:
            logger.warning(f"信号日 {date.date()} 之后没有交易日，无法执行")
        return None

    buy_date = trading_dates[current_idx + 1]
    buy_date_str = to_trade_date_str(buy_date)
    buy_date_quote = price_data[price_data["trade_date"] == buy_date_str]
    return buy_date, buy_date_quote


def _filter_signal_candidates(
    self: "BacktestSignalExecutionMixin",
    ranked_candidates: list,
    date: pd.Timestamp,
    existing_positions: set,
) -> Tuple[list, list]:
    """行业约束 + 扩展点过滤 + 排除已持仓（D8 提取件，顺序与日志文本逐字保留）。

    Returns:
        (ranked_candidates, ranked_candidates_for_selection)
    """
    # 应用行业约束（如果启用）
    if self.max_per_industry is not None:
        # 延迟导入
        from src.lazybull.v2.core.decision.industry_constraint import apply_industry_constraint

        ranked_candidates = apply_industry_constraint(
            ranked_candidates,
            self.industry_mapping,
            max_per_industry=self.max_per_industry,
            target_n=len(ranked_candidates),  # 保留所有候选，只改变顺序
            verbose=self.verbose,
            initial_industry_counts=self._get_industry_counts(existing_positions),
        )

    # 扩展点：子类可覆盖此方法对候选列表做额外过滤
    ranked_candidates = self._post_filter_candidates(ranked_candidates, date)

    # ── 持仓处理：排除已持仓股票 ──
    ranked_candidates_for_selection = ranked_candidates

    if existing_positions:
        # 排除已持仓的股票
        ranked_candidates_for_selection = [
            (stock, score) for stock, score in ranked_candidates if stock not in existing_positions
        ]
        if self.verbose:
            excluded = len(ranked_candidates) - len(ranked_candidates_for_selection)
            if excluded > 0:
                logger.info(
                    f"  排除已持仓股票: {excluded} 只 "
                    f"(持仓 {len(existing_positions)} 只, "
                    f"候选从 {len(ranked_candidates)} 缩减到 "
                    f"{len(ranked_candidates_for_selection)})"
                )
    else:
        ranked_candidates_for_selection = ranked_candidates

    return ranked_candidates, ranked_candidates_for_selection


def _resolve_signal_target_n(
    self: "BacktestSignalExecutionMixin",
    ranked_candidates: list,
    tranche_idx: int,
) -> Tuple[int, int]:
    """解析本批次目标买入数量（D8 提取件）。

    Returns:
        (target_n, overall_target_n)；target_n <= 0 时调用方直接 return
    """
    # 分批调仓拆分的是总 TopN，而不是每批各选 TopN。
    configured_target_n = getattr(self.signal, "top_n", None)
    overall_target_n = (
        configured_target_n
        if isinstance(configured_target_n, int) and configured_target_n > 0
        else len(ranked_candidates)
    )
    target_n = self._get_tranche_target_count(tranche_idx, overall_target_n)
    return target_n, overall_target_n


def _normalize_and_cap_signals(
    self: "BacktestSignalExecutionMixin",
    signals: Dict[str, float],
    date: pd.Timestamp,
    tranche_idx: int,
) -> Dict[str, float]:
    """归一化权重并按批次资金比例套单股权重上限（D8 提取件，顺序不变）。"""
    # 归一化权重（子类可覆写 _normalize_signals 以实现 ATR 加权等策略）
    signals = self._normalize_signals(signals, date)

    if self.max_weight_per_stock is not None and signals:
        tranche_weight_cap = resolve_tranche_weight_cap(
            self.max_weight_per_stock,
            self._get_tranche_capital_fraction(tranche_idx),
        )
        signals = cap_and_normalize_weights(
            signals,
            max_weight_per_stock=tranche_weight_cap,
            verbose=self.verbose,
        )

    return signals


def _queue_signal_plan(
    self: "BacktestSignalExecutionMixin",
    date: pd.Timestamp,
    signals: Dict[str, float],
    ranked_candidates: list,
    priority_candidates: list,
    target_n: int,
    overall_target_n: int,
    tranche_idx: int,
    decision_trace: Dict,
) -> Dict:
    """写入 pending_signals、保存最近候选并收口 decision_trace（D8 提取件，顺序不变）。"""
    # 应用权重限制（如果启用）
    # 同时保存 T0 候选优先级与槽位计划，供 T+1 顺位执行和后续补齐复用。
    slot_weights = [{"stock": stock, "weight": float(weight)} for stock, weight in signals.items()]
    self.pending_signals[date] = {
        "signals": signals,
        "ranked_candidates": ranked_candidates if self.enable_position_completion else [],
        "priority_candidates": list(priority_candidates),
        "slot_weights": slot_weights,
        "target_n": target_n,
        "desired_position_count": overall_target_n,
        "tranche_idx": tranche_idx,
        "decision_trace": decision_trace,
    }

    # 保存最近一次调仓候选列表，供整体止盈补位使用
    self._last_ranked_candidates = list(ranked_candidates)
    self._last_signal_date = date
    decision_trace["queued"] = True
    decision_trace["final_target_exposure"] = float(sum(signals.values()))
    decision_trace = self._finalize_decision_trace_for_signal_day(
        decision_trace=decision_trace,
        signal_date=date,
    )
    self.pending_signals[date]["decision_trace"] = decision_trace
    return decision_trace


class BacktestSignalExecutionMixin:
    """提供信号生成与候选规划相关实现。"""

    def _get_industry_counts(self, stocks: set) -> Dict[str, int]:
        """统计已有持仓在行业约束中的占用数量。"""
        industry_mapping = self.industry_mapping or {}
        industry_counts: Dict[str, int] = {}
        for stock in stocks:
            industry = industry_mapping.get(stock, "未知行业")
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        return industry_counts

    def _build_signal_data(self, date: pd.Timestamp) -> Optional[Dict]:
        """构建传递给信号生成器的额外数据（扩展点）

        子类可以重写此方法以注入特定数据（如 ML 特征）。

        Args:
            date: 信号生成日期

        Returns:
            数据字典，将与默认数据合并后传递给 signal.generate_ranked()
            返回 None 表示该日期无可用数据，将跳过信号生成
        """
        return {}

    def _post_filter_candidates(self, ranked_candidates: list, date: pd.Timestamp) -> list:
        """对排序候选列表做额外过滤（扩展点）

        Args:
            ranked_candidates: [(stock_code, score), ...] 已按分数降序排列
            date: 当前日期

        Returns:
            过滤后的候选列表
        """
        return ranked_candidates

    def _get_position_weight_for_planning(
        self,
        date: pd.Timestamp,
        stock: str,
        portfolio_value: Optional[float] = None,
    ) -> float:
        """获取指定持仓在规划日的组合权重。"""
        if stock not in self.positions:
            return 0.0

        total_value = float(portfolio_value or 0.0)
        if total_value <= 0:
            total_value = float(self._calculate_portfolio_value(date))
        if total_value <= 0:
            return 0.0

        info = self.positions[stock]
        shares = float(info.get("shares", 0) or 0)
        if shares <= 0:
            return 0.0

        trade_price = self._get_trade_price(date, stock)
        if trade_price is None:
            trade_price = info.get("last_known_price")
            if trade_price is None:
                trade_price = info.get("buy_trade_price", 0.0)
        else:
            info["last_known_price"] = trade_price

        if not trade_price:
            return 0.0

        return float(shares * trade_price / total_value)

    def _queue_condition_sell_refill_signal(
        self,
        date: pd.Timestamp,
        slot_weights: List[Dict[str, float]],
        price_data: pd.DataFrame,
        date_to_idx: Dict,
    ) -> None:
        """为持有期/盈利延续卖出生成 T0 买入计划，供下一交易日执行。"""
        if (
            not self.enable_position_completion
            or not slot_weights
            or date in self.pending_signals
            or price_data is None
            or price_data.empty
            or not hasattr(self.signal, "generate_ranked")
        ):
            return

        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx + 1 >= len(date_to_idx):
            return

        trade_date_str = to_trade_date_str(date)
        date_quote = price_data[price_data["trade_date"] == trade_date_str]
        stock_universe = self.universe.get_stocks(date, quote_data=date_quote)

        extra_data = self._build_signal_data(date)
        if extra_data is None:
            return

        signal_data = {}
        signal_data.update(extra_data)
        ranked_candidates = self.signal.generate_ranked(date, stock_universe, signal_data)
        if not ranked_candidates:
            return

        existing_positions = set(self.positions.keys()) if self.positions else set()
        if self.max_per_industry is not None:
            # 延迟导入
            from src.lazybull.v2.core.decision.industry_constraint import (
                apply_industry_constraint,
            )

            ranked_candidates = apply_industry_constraint(
                ranked_candidates,
                self.industry_mapping,
                max_per_industry=self.max_per_industry,
                target_n=len(ranked_candidates),
                verbose=self.verbose,
                initial_industry_counts=self._get_industry_counts(existing_positions),
            )

        ranked_candidates = self._post_filter_candidates(ranked_candidates, date)
        priority_candidates = [
            (stock, score) for stock, score in ranked_candidates if stock not in existing_positions
        ]
        if not priority_candidates:
            return

        normalized_slot_weights = []
        fallback_weight = 1.0 / max(len(slot_weights), 1)
        for slot in slot_weights:
            weight = float(slot.get("weight", 0.0) or 0.0)
            normalized_slot_weights.append(
                {
                    "stock": str(slot.get("stock", "")),
                    "weight": weight if weight > 0 else fallback_weight,
                }
            )

        planned_candidates = priority_candidates[: len(normalized_slot_weights)]
        if not planned_candidates:
            return

        signals = {}
        planned_slot_weights = []
        for slot_weight_info, (candidate_stock, _score) in zip(
            normalized_slot_weights, planned_candidates
        ):
            weight = float(slot_weight_info["weight"])
            signals[candidate_stock] = weight
            planned_slot_weights.append({"stock": candidate_stock, "weight": weight})

        desired_position_count = int(self._get_target_position_count() or len(self.positions))
        self.pending_signals[date] = {
            "signals": signals,
            "ranked_candidates": ranked_candidates,
            "priority_candidates": list(priority_candidates),
            "slot_weights": planned_slot_weights,
            "target_n": len(planned_slot_weights),
            "desired_position_count": desired_position_count,
            "tranche_idx": 0,
        }

        if self.verbose:
            logger.info(
                f"  持有期卖出补位计划: {date.date()} 生成 {len(planned_slot_weights)} 个待买槽位，"
                f"下一交易日按候选顺序执行"
            )

    def _get_holding_features_row(self, date: pd.Timestamp, stock: str) -> Optional[pd.Series]:
        """持仓强势度评分数据源 hook

        基类无特征数据,返回 None。BacktestEngineML 子类会从 features_by_date
        读取对应股票的截面特征行并返回。
        """
        return None

    def _generate_signal(
        self,
        date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
        date_to_idx: Dict,
        tranche_idx: int = 0,
    ) -> None:
        """生成信号（在 T 日生成，T+1 日执行买入）

        新逻辑：生成排序候选列表，在 T+1 日过滤不可交易股票并回填，确保 top N 全部可交易。

        Args:
            date: 信号生成日期
            trading_dates: 交易日列表
            price_data: 价格数据，包含行情信息
            date_to_idx: 日期到索引的映射
            tranche_idx: 分批调仓的批次索引（0-based）
        """
        ranked_candidates = _prepare_ranked_candidates(self, date, price_data)
        if ranked_candidates is None:
            return

        buy_info = _resolve_buy_date_quote(self, date, trading_dates, price_data, date_to_idx)
        if buy_info is None:
            return
        buy_date, buy_date_quote = buy_info

        existing_positions = set(self.positions.keys()) if self.positions else set()

        ranked_candidates, ranked_candidates_for_selection = _filter_signal_candidates(
            self, ranked_candidates, date, existing_positions
        )

        # 从排序候选中选择 top N 股票
        # 始终仅基于 T0 排名生成次日买入计划；
        # T+1 的可交易性统一在执行阶段处理，避免在计划阶段引入前视过滤。
        signals = {}
        candidates_checked = 0

        target_n, overall_target_n = _resolve_signal_target_n(self, ranked_candidates, tranche_idx)
        if target_n <= 0:
            return

        decision_trace = self._build_signal_decision_trace(
            date=date,
            target_n=target_n,
            candidate_count=len(ranked_candidates_for_selection),
            tranche_idx=tranche_idx,
        )

        priority_candidates = list(ranked_candidates_for_selection)

        for stock, score in priority_candidates[:target_n]:
            signals[stock] = score
            candidates_checked += 1

        if not signals:
            if self.verbose:
                logger.warning(
                    f"信号日 {date.date()} 所有候选在 T+1 日 {buy_date.date()} 均不可交易，"
                    f"检查了 {candidates_checked} 个候选"
                )
            return

        signals = _normalize_and_cap_signals(self, signals, date, tranche_idx)

        decision_trace = _queue_signal_plan(
            self,
            date,
            signals,
            ranked_candidates,
            priority_candidates,
            target_n,
            overall_target_n,
            tranche_idx,
            decision_trace,
        )

        tranche_tag = (
            f"[批次 {tranche_idx + 1}/{self.stagger_tranches}] "
            if self.stagger_tranches > 1
            else ""
        )
        self._log_rebalance_decision_summary(
            decision_trace=decision_trace,
            execution_date=buy_date,
            tranche_tag=tranche_tag,
        )

        # 分批调仓时始终打印信号生成汇总，便于确认各批次调度情况
        if self.verbose or self.stagger_tranches > 1:
            logger.info(
                f"  {tranche_tag}信号生成: {date.date()}, 选择 top {len(signals)}/{target_n} 股票"
                f"（未检查 T+1 可交易性，将在买入时处理）, "
                f"候选总数 {len(priority_candidates)} 个"
            )
