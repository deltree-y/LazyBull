# -*- coding: utf-8 -*-
"""PaperExecutionMixin：src/lazybull/paper/runner.py 拆分出的 run_t0, _build_pending_buys_from_failed_targets, _analyze_pending_buy_shares_backtest_style, _execute_pending_buys。"""

from ...common.constants import SHARE_LOT_SIZE
from ...common.trade_status import is_tradeable
from ...common.trading_config import TradingConfig
from ...features import ensure_features_for_date
from ...signals.ml_signal import MLSignal
from ...trading.buy_plan import REASON_ALREADY_BOUGHT
from ...trading.buy_plan import REASON_EXECUTION_FAILED
from ...trading.buy_plan import fill_slots_from_candidates
from ...trading.sizing import compute_lot_shares
from ...trading.stagger import get_tranche_target_count as _shared_tranche_target_count
from ..models import Order
from ..models import PendingBuy
from ..models import TargetWeight
from ..models import TradeInstruction
from ..models import normalize_trade_reason
from loguru import logger
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import pandas as pd

class PaperExecutionMixin:
    def run_t0(
        self,
        trade_date: str,
        buy_price_type: str = 'close',
        sell_price_type: str = 'close',
        universe_type: str = 'mainboard',
        top_n: int = 5,
        model_version: Optional[int] = None,
        rebalance_freq: int = 5,
        max_per_industry: Optional[int] = None,
        max_weight_per_stock: Optional[float] = None,
        exclude_st: bool = True,
        min_list_days: int = 365,
        trading_config: Optional[TradingConfig] = None,
        force_rebalance: bool = False,
        protected_stocks: Optional[set] = None,
    ) -> None:
        """T0工作流：拉取数据 + 生成T1待执行目标

        Args:
            trade_date: 交易日期 YYYYMMDD（T0日期）
            buy_price_type: T1买入价格类型 open/close
            sell_price_type: T1卖出价格类型 open/close
            universe_type: 股票池类型 mainboard
            top_n: 持仓股票数
            model_version: ML模型版本（可选）
            rebalance_freq: 调仓频率（交易日数）
            max_per_industry: 单行业最大持仓数量（可选）
            max_weight_per_stock: 单股最大权重（可选）
            exclude_st: 是否排除ST股票
            min_list_days: 最少上市天数
            protected_stocks: 盈利延续保护的股票集合（跳过卖出）
        """
        effective_config = trading_config or TradingConfig(
            buy_price=buy_price_type,
            sell_price=sell_price_type,
            universe=universe_type,
            top_n=top_n,
            model_version=model_version,
            rebalance_freq=rebalance_freq,
            max_per_industry=max_per_industry,
            max_weight_per_stock=max_weight_per_stock,
            exclude_st=exclude_st,
            min_list_days=min_list_days,

        )
        self.position_sizing = effective_config.position_sizing
        self.kelly_vol_window = effective_config.kelly_vol_window
        self.kelly_max_leverage = effective_config.kelly_max_leverage

        # 1. 校正交易日期
        corrected_date = self._correct_trade_date(trade_date)
        
        # 2. 检查幂等性
        if self.paper_storage.check_run_exists("t0", corrected_date):
            raise RuntimeError(
                f"T0 工作流已在 {corrected_date} 执行过，"
                f"不允许重复执行（幂等性保障）"
            )
        
        # 3. 检查调仓日（分批模式返回 tranche_idx）
        tranche_idx = 0
        if force_rebalance:
            logger.warning(f"强制执行 T0，跳过调仓日校验: {corrected_date}")
            self._resolved_rebalance_plan_date = corrected_date
        else:
            _, tranche_idx = self._check_rebalance_day(
                corrected_date,
                effective_config.rebalance_freq,
                stagger_tranches=effective_config.stagger_tranches,
            )

        # 分批调仓：空仓/拖尾提前调仓时触发 tranche 0 全量建仓（与回测对齐）
        stagger_tranches = effective_config.stagger_tranches
        if stagger_tranches > 1 and force_rebalance:
            tranche_idx = 0

        tranche_tag = (
            f"[批次 {tranche_idx + 1}/{stagger_tranches}] "
            if stagger_tranches > 1
            else ""
        )

        logger.info("=" * 80)
        logger.info(f"开始T0工作流 - {corrected_date}")
        logger.info("=" * 80)
        stagger_info = f", 分批调仓={stagger_tranches}批" if stagger_tranches > 1 else ""
        logger.info(f"调仓频率: {effective_config.rebalance_freq} 个交易日{stagger_info}")

        # 分批调仓：拆分 top_n 为本批槽位数
        overall_top_n = effective_config.top_n
        tranche_target_n = _shared_tranche_target_count(
            tranche_idx, overall_top_n, stagger_tranches
        )
        if stagger_tranches > 1:
            logger.info(
                f"{tranche_tag}本批槽位: {tranche_target_n}/{overall_top_n}"
            )
        if tranche_target_n <= 0:
            logger.warning(f"{tranche_tag}本批槽位为 0，跳过信号生成")
            return

        t1_date = self._get_next_trade_date(corrected_date)
        if not t1_date:
            logger.error(f"无法获取 {corrected_date} 的下一个交易日")
            return
        existing_instructions = self.paper_storage.load_instructions(t1_date) or []
        reserved_buy_stocks = {
            inst.ts_code for inst in existing_instructions if inst.action == 'buy'
        }
        
        # 4. 生成信号（ensure_features_for_date 内部自动下载缺失的 raw/clean 数据）
        # 分批调仓时 signal.top_n 保持为总 top_n（用于候选排序范围），
        # 但实际目标生成数量由 tranche_target_n 控制。
        logger.info("步骤2: 生成信号")
        self.signal = self.signal or MLSignal(
            top_n=overall_top_n,
            model_version=effective_config.model_version,
            verbose=False,
        )
        if hasattr(self.signal, "top_n"):
            self.signal.top_n = overall_top_n
        if effective_config.model_version_b is not None and hasattr(self.signal, "update_versions"):
            self.signal.update_versions(
                effective_config.model_version,
                effective_config.model_version_b,
            )
        elif effective_config.model_version is not None and hasattr(self.signal, "update_model_version"):
            self.signal.update_model_version(effective_config.model_version)
        targets = self._generate_signals(
            corrected_date,
            universe_type=effective_config.universe,
            top_n=tranche_target_n,
            model_version=effective_config.model_version,
            buy_price_type=effective_config.buy_price,
            max_per_industry=effective_config.max_per_industry,
            max_weight_per_stock=effective_config.max_weight_per_stock,
            exclude_st=effective_config.exclude_st,
            min_list_days=effective_config.min_list_days,
            protected_stocks=protected_stocks,
            excluded_stocks=reserved_buy_stocks,
            trading_config=effective_config,
            tranche_idx=tranche_idx,
        )
        
        if not targets:
            logger.warning("未生成任何目标权重")
            return
        
        # 6. 生成交易指令
        logger.info("步骤3: 生成交易指令")
        # 获取T0日的收盘价（用于计算指令股数）
        daily_data = self.loader.load_clean_daily(start_date=corrected_date, end_date=corrected_date)
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {corrected_date} 的价格数据")
            return
        
        current_prices = {}
        for _, row in daily_data.iterrows():
            price = row.get('close')
            if pd.isna(price) or float(price) <= 0:
                price = row.get('pre_close')
            if pd.isna(price):
                continue
            price_val = float(price)
            if price_val > 0:
                current_prices[row['ts_code']] = price_val
        
        # 生成指令（使用传入的 sell_price_type 参数，分批模式传入 tranche 参数）
        buy_instructions = self._generate_instructions(
            targets=targets,
            buy_price_type=buy_price_type,
            sell_price_type=sell_price_type,
            current_prices=current_prices,
            source_date=corrected_date,
            protected_stocks=protected_stocks,
            desired_position_count=overall_top_n,
            tranche_idx=tranche_idx,
            overall_top_n=overall_top_n,
            stagger_tranches=stagger_tranches,
        )

        # 调仓日同步生成卖出指令：将不在新目标且非保护持仓排队到 T+1 卖出，
        # 使卖出与买入对齐到同一执行日，消除卖出滞后一天的偏差。
        # 提前调仓（holding tail）场景下仅卖出已到期的持仓，跳过未满期的年轻持仓。
        target_codes = {t.ts_code for t in targets}
        trade_cal = self.loader.load_clean_trade_cal()
        trade_dates_list = (
            trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()
            if trade_cal is not None
            else None
        )
        sell_instructions = self._build_rebalance_sell_instructions(
            trade_date=corrected_date,
            sell_price_type=sell_price_type,
            protected_stocks=protected_stocks or set(),
            target_codes=target_codes,
            rebalance_freq=effective_config.rebalance_freq,
            trade_dates_list=trade_dates_list,
        )

        # 合并指令：卖出在前，买入在后（先卖后买）
        instructions = sell_instructions + buy_instructions

        if not instructions:
            logger.warning("未生成任何交易指令")
            return
        
        # 7. 持久化指令
        logger.info("步骤4: 保存交易指令")
        # T0生成的是T1执行的目标，所以需要获取T1日期
        # 保存交易指令（指令驱动模式）。
        # 同一交易日内，日度卖出规划可能已先写入次日指令，这里需要做合并而非覆盖。
        # 合并优先级：已存在的条件卖出/止损指令 > 调仓卖出 > 调仓买入。
        merged_instructions: List[TradeInstruction] = []
        seen_keys = set()
        for inst in [*existing_instructions, *instructions]:
            key = (inst.action, inst.ts_code)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_instructions.append(inst)
        self.paper_storage.save_instructions(t1_date, merged_instructions)
        if hasattr(self.signal, "_last_ranked_candidates") and self.signal._last_ranked_candidates:
            logger.info("保存本轮 ranked_candidates 供 T1 买入顺延使用")
            self.paper_storage.save_ranked_candidates(self.signal._last_ranked_candidates, corrected_date)
        
        # 8. 更新调仓状态（分批模式：批次0时更新锚定日）
        prev_state = self.paper_storage.load_rebalance_state() or {}
        scheduled_rebalance_date = getattr(
            self,
            '_resolved_rebalance_plan_date',
            corrected_date,
        )
        config_changed = (
            prev_state.get('rebalance_freq', effective_config.rebalance_freq)
            != effective_config.rebalance_freq
            or prev_state.get('stagger_tranches', 1) != stagger_tranches
        )
        rebalance_state = {
            'last_rebalance_date': corrected_date,
            'last_scheduled_rebalance_date': scheduled_rebalance_date,
            'rebalance_freq': effective_config.rebalance_freq,
            'stagger_tranches': stagger_tranches,
        }
        if stagger_tranches > 1 and (force_rebalance or config_changed or tranche_idx == 0):
            rebalance_state['tranche_anchor_date'] = scheduled_rebalance_date
        elif stagger_tranches > 1:
            if prev_state.get('tranche_anchor_date'):
                rebalance_state['tranche_anchor_date'] = prev_state['tranche_anchor_date']
        self.paper_storage.save_rebalance_state(rebalance_state)

        state = self._ensure_strategy_state(effective_config)
        state['last_rebalance_nav'] = self.account.get_total_value(current_prices)
        self._save_strategy_state()
        
        # 9. 保存执行记录
        buy_count = len(buy_instructions)
        sell_count = len(sell_instructions)
        run_record = {
            'trade_date': corrected_date,
            't1_date': t1_date,
            'buy_price_type': buy_price_type,
            'universe_type': universe_type,
            'top_n': top_n,
            'model_version': model_version,
            'rebalance_freq': effective_config.rebalance_freq,
            'targets_count': len(targets),
            'instructions_count': len(merged_instructions),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'tranche_idx': tranche_idx,
            'scheduled_rebalance_date': scheduled_rebalance_date,
            'stagger_tranches': stagger_tranches,
            'rebalance_state': rebalance_state,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.paper_storage.save_run_record("t0", corrected_date, run_record)
        
        logger.info("=" * 80)
        logger.info(
            f"T0工作流完成 - 已生成 {len(targets)} 个目标权重和 "
            f"{len(merged_instructions)} 条交易指令（买入 {buy_count}，卖出 {sell_count}）"
        )
        logger.info(f"下一交易日: {t1_date}")
        logger.info("=" * 80)

    def _build_pending_buys_from_failed_targets(
        self,
        failed_buy_targets: List[TargetWeight],
        trade_date: str,
        attempts: int = 0,
    ) -> List[PendingBuy]:
        """将失败买入目标转换为补位槽位。"""
        pending_buys: List[PendingBuy] = []
        fallback_weight = 1.0 / max(len(failed_buy_targets), 1)

        for target in failed_buy_targets:
            slot_weight = float(getattr(target, "target_weight", 0.0) or 0.0)
            if slot_weight <= 0:
                slot_weight = fallback_weight

            pending_buys.append(
                PendingBuy(
                    ts_code=str(getattr(target, "ts_code", "")),
                    target_weight=slot_weight,
                    reason=normalize_trade_reason(
                        str(getattr(target, 'reason', '买入失败')),
                        ensure_replenishment_prefix=True,
                    ),
                    create_date=trade_date,
                    attempts=attempts,
                    last_attempt_date="",
                    original_signal_date=str(
                        getattr(target, "original_signal_date", trade_date) or trade_date
                    ),
                )
            )

        return pending_buys

    def _analyze_pending_buy_shares_backtest_style(
        self,
        ts_code: str,
        price: float,
        target_weight: float,
        current_total_value: float,
    ) -> Tuple[int, str]:
        """按回测口径估算补位买入股数，并返回拒绝原因。"""
        if price <= 0 or target_weight <= 0 or current_total_value <= 0:
            if price <= 0:
                return 0, "无有效价格"
            if target_weight <= 0:
                return 0, "槽位权重<=0"
            return 0, "组合总资产<=0"

        target_value = current_total_value * target_weight
        buy_shares = compute_lot_shares(target_value, price, SHARE_LOT_SIZE)
        if buy_shares < SHARE_LOT_SIZE:
            return 0, "目标金额不足一手"

        cash = self.account.get_cash()
        if cash <= 0:
            return 0, "现金不足"

        amount = buy_shares * price
        buy_cost = self.broker.cost_model.calculate_buy_cost(amount)
        if amount + buy_cost <= cash:
            return buy_shares, "可买入"

        # 对齐回测：现金不足时按剩余现金缩量买入，而非直接放弃
        if cash <= buy_cost:
            return 0, "现金不足(连手续费都不够)"

        buy_shares = compute_lot_shares(cash - buy_cost, price, SHARE_LOT_SIZE)
        if buy_shares < SHARE_LOT_SIZE:
            return 0, "现金不足(缩量后不足一手)"

        while buy_shares >= SHARE_LOT_SIZE:
            amount = buy_shares * price
            buy_cost = self.broker.cost_model.calculate_buy_cost(amount)
            if amount + buy_cost <= cash:
                return buy_shares, "可买入(缩量)"
            buy_shares -= SHARE_LOT_SIZE

        return 0, "现金不足(含费用)"

    def _execute_pending_buys(
        self,
        pending_buys: List,
        buy_prices: Dict[str, float],
        trade_date: str,
        buy_price_type: str = 'close',
        universe_type: str = 'mainboard',
        exclude_st: bool = True,
        min_list_days: int = 365,
    ) -> List:
        """执行补位买入计划（仅买入，不触发卖出）
        
        Args:
            pending_buys: 补位买入计划列表
            buy_prices: 买入价格字典
            trade_date: 交易日期
            buy_price_type: 买入价格类型
            
        Returns:
            成交记录列表
        """
        from ..models import Fill, Order
        
        MAX_REPLENISHMENT_ATTEMPTS = 5
        
        logger.info("=" * 80)
        logger.info(f"执行补位买入计划 - {trade_date}")
        logger.info(f"待处理补位: {len(pending_buys)} 个")
        logger.info("=" * 80)

        # 当日行情用于可交易性检查（与回测 is_tradeable 对齐）
        date_quote = self.loader.load_clean_daily_by_date(trade_date)
        if date_quote is None:
            date_quote = pd.DataFrame()

        # 对齐回测：优先读取 T0 持久化候选池，再基于上一交易日重算候选池，
        # 并限制为「槽位数*2」。
        slot_count = len(pending_buys)
        candidate_buffer = slot_count * 2
        prev_trade_date = self._get_prev_trade_date(trade_date)

        candidate_codes: List[str] = []
        existing_positions = set(self.account.get_positions().keys())
        expected_signal_dates = {
            str(getattr(pb, 'original_signal_date', '') or '')
            for pb in pending_buys
            if str(getattr(pb, 'original_signal_date', '') or '')
        }
        if prev_trade_date:
            expected_signal_dates.add(prev_trade_date)

        rc_loaded = self.paper_storage.load_ranked_candidates()
        if isinstance(rc_loaded, tuple) and len(rc_loaded) == 2:
            ranked_candidates, signal_date = rc_loaded
            signal_date = str(signal_date)
            if signal_date in expected_signal_dates:
                for ts_code, _ in ranked_candidates:
                    if ts_code in existing_positions:
                        continue
                    candidate_codes.append(ts_code)
                    if len(candidate_codes) >= candidate_buffer:
                        break
                if candidate_codes:
                    logger.info(
                        f"补位候选池读取 T0 持久化排序: signal_date={signal_date}, "
                        f"有限候选 {len(candidate_codes)} 只（槽位 {slot_count}）"
                    )

        if (
            not candidate_codes
            and prev_trade_date
            and self.signal is not None
            and hasattr(self.signal, "generate_ranked")
        ):
            signal_data = self.storage.load_cs_train_day(prev_trade_date, subdir="cs_infer")
            if signal_data is None or signal_data.empty:
                ok, missing, _ = ensure_features_for_date(
                    self.storage,
                    self.loader,
                    self.feature_builder,
                    self.cleaner,
                    self.client,
                    prev_trade_date,
                    force=False,
                )
                self.missing_factors = missing
                if ok:
                    signal_data = self.storage.load_cs_train_day(prev_trade_date, subdir="cs_infer")

            if signal_data is not None and not signal_data.empty:
                try:
                    stocks: List[str] = []
                    stock_basic = self.loader.load_clean_stock_basic()
                    prev_daily_data = self.loader.load_clean_daily_by_date(prev_trade_date)
                    if (
                        stock_basic is not None
                        and not stock_basic.empty
                        and prev_daily_data is not None
                        and not prev_daily_data.empty
                    ):
                        universe = self._create_universe(
                            stock_basic,
                            universe_type,
                            exclude_st=exclude_st,
                            min_list_days=min_list_days,
                        )
                        stocks = universe.get_stocks(pd.Timestamp(prev_trade_date), prev_daily_data)

                    # 回退路径：股票池构建失败时，退回到特征文件内全量股票
                    if not stocks:
                        stocks = signal_data["ts_code"].dropna().astype(str).unique().tolist()

                    stocks = [s for s in stocks if s not in existing_positions]

                    ranked_candidates = self.signal.generate_ranked(
                        pd.Timestamp(prev_trade_date),
                        stocks,
                        {"features": signal_data},
                    )

                    for ts_code, _ in ranked_candidates:
                        if ts_code in existing_positions:
                            continue
                        candidate_codes.append(ts_code)
                        if len(candidate_codes) >= candidate_buffer:
                            break

                    if candidate_codes:
                        logger.info(
                            f"补位候选池已重算: 基于 {prev_trade_date}, "
                            f"有限候选 {len(candidate_codes)} 只（槽位 {slot_count}）"
                        )
                except Exception as exc:
                    logger.warning(f"补位候选池重算失败，回退队列代码模式: {exc}")

        # 回退路径：候选池重算失败时，沿用旧的队列代码顺序
        if not candidate_codes:
            candidate_codes = [pb.ts_code for pb in pending_buys if pb.ts_code]
            logger.warning(
                f"补位候选池为空，回退为队列代码模式（{len(candidate_codes)} 只）"
            )

        fills = []
        updated_pending_buys = []

        # 对齐回测：槽位目标金额基于当日组合总资产（非“补位均分现金”）
        current_total_value = self.account.get_total_value(buy_prices)
        if current_total_value <= 0:
            current_total_value = float(getattr(self.account, "initial_capital", 0.0) or 0.0)
        # 与 broker 路径保持一致：补位买入也要满足最小买入后市值阈值
        min_buy_value_threshold = self.broker._get_min_buy_value_threshold(buy_prices)
        
        # 前置检查：超次数放弃、同日已尝试跳过，其余槽位进入共享匹配骨架
        eligible_slots = []
        for pending_buy in pending_buys:
            if pending_buy.attempts >= MAX_REPLENISHMENT_ATTEMPTS:
                logger.warning(
                    f"补位 {pending_buy.ts_code} 已达最大尝试次数 ({MAX_REPLENISHMENT_ATTEMPTS})，放弃"
                )
                continue

            if pending_buy.last_attempt_date == trade_date:
                logger.info(f"补位 {pending_buy.ts_code} 今日已尝试，跳过（避免重复）")
                updated_pending_buys.append(pending_buy)
                continue

            eligible_slots.append(pending_buy)

        # 槽位匹配委托 trading.buy_plan 共享骨架，评估/执行/失败原因统计通过回调注入
        untradeable_stock_set = set()
        slot_reason_counters: Dict[int, Dict[str, int]] = {}

        def _record_slot_reason(slot, reason_text: str) -> None:
            counter = slot_reason_counters.setdefault(id(slot), {})
            counter[reason_text] = counter.get(reason_text, 0) + 1

        def _evaluate_candidate(ts_code: str, pending_buy) -> Tuple[bool, str]:
            if ts_code in untradeable_stock_set:
                return False, "当日不可交易(已缓存)"
            if ts_code in self.account.get_positions():
                return False, "已持仓"
            price = buy_prices.get(ts_code)
            if price is None or price <= 0:
                return False, "无价格数据"

            tradeable, reason = is_tradeable(ts_code, trade_date, date_quote, action="buy")
            if not tradeable:
                untradeable_stock_set.add(ts_code)
                return False, f"不可交易({reason})"

            buy_shares, share_reason = self._analyze_pending_buy_shares_backtest_style(
                ts_code=ts_code,
                price=price,
                target_weight=pending_buy.target_weight,
                current_total_value=current_total_value,
            )
            if buy_shares <= 0:
                return False, f"资金/股数约束({share_reason})"

            actual_buy_value = buy_shares * price
            if min_buy_value_threshold > 0 and actual_buy_value < min_buy_value_threshold:
                return False, "买入后市值过小"

            return True, ""

        def _execute_buy(ts_code: str, pending_buy) -> bool:
            price = float(buy_prices.get(ts_code) or 0.0)
            buy_shares, _ = self._analyze_pending_buy_shares_backtest_style(
                ts_code=ts_code,
                price=price,
                target_weight=pending_buy.target_weight,
                current_total_value=current_total_value,
            )
            if buy_shares <= 0:
                return False

            order = Order(
                ts_code=ts_code,
                action='buy',
                shares=buy_shares,
                price=price,
                target_weight=pending_buy.target_weight,
                current_weight=0.0,
                reason=pending_buy.reason,
            )

            fill = self.broker._execute_single_order(order, trade_date, buy_price_type)
            if not fill:
                return False

            fills.append(fill)
            slot_code = pending_buy.ts_code
            if prev_trade_date:
                logger.info(
                    f"补位成功: {trade_date} (基于 {prev_trade_date} 数据), "
                    f"槽位 {slot_code} (权重 {pending_buy.target_weight:.4f}) "
                    f"买入股票 {ts_code} 成功"
                )
            else:
                logger.info(
                    f"补位成功: 槽位 {slot_code} 买入股票 {ts_code}, "
                    f"买入 {fill.shares} 股, 成交价 {fill.price:.2f}"
                )
            return True

        def _on_reject(pending_buy, ts_code: str, reason: str) -> None:
            if reason == REASON_ALREADY_BOUGHT:
                reason = "当日已被其他槽位买入"
            elif reason == REASON_EXECUTION_FAILED:
                reason = "下单失败(执行层拒单)"
            _record_slot_reason(pending_buy, reason)

        match_result = fill_slots_from_candidates(
            slots=eligible_slots,
            candidates=candidate_codes,
            evaluate_candidate=_evaluate_candidate,
            execute_buy=_execute_buy,
            on_reject=_on_reject,
        )

        for pending_buy in match_result.unfilled:
            pending_buy.attempts += 1
            pending_buy.last_attempt_date = trade_date
            updated_pending_buys.append(pending_buy)
            slot_reason_counter = slot_reason_counters.get(id(pending_buy), {})
            reason_summary = "、".join(
                [
                    f"{reason}:{count}"
                    for reason, count in sorted(
                        slot_reason_counter.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:4]
                ]
            )
            if not reason_summary:
                reason_summary = "候选全部不匹配"
            logger.info(
                f"补位延迟: {trade_date}, 槽位 {pending_buy.ts_code} "
                f"(权重 {pending_buy.target_weight:.4f}) "
                f"候选池 {len(candidate_codes)} 只未匹配，"
                f"原因[{reason_summary}]，下次重试"
            )
        
        # 保存更新后的补位队列
        self.paper_storage.save_pending_buys(updated_pending_buys)
        
        logger.info(f"补位买入执行完成: 成功 {len(fills)} 个，失败 {len(updated_pending_buys)} 个")
        logger.info("=" * 80)
        
        return fills
