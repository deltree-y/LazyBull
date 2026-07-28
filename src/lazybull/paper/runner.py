"""纸面交易运行器"""

import gc
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..common.config import get_cost_settings
from ..common.print_table import format_row
from ..common.trade_status import is_tradeable
from ..common.trading_config import TradingConfig
from ..data import (
    DataCleaner,
    DataLoader,
    Storage,
    TushareClient,
    ensure_basic_data,
)
from ..features import FeatureBuilder, ensure_features_for_date
from ..signals.base import Signal
from ..signals.ml_signal import MLSignal
from ..trading.buy_plan import (
    REASON_ALREADY_BOUGHT,
    REASON_EXECUTION_FAILED,
    fill_slots_from_candidates,
)
from ..trading.sell_rules import (
    min_holding_days_for_rebalance_sell,
    select_rebalance_sell_candidates,
)
from ..trading.sizing import compute_kelly_weights, estimate_variance_from_prices
from ..universe.base import BasicUniverse
from ..portfolio.industry_constraint import load_industry_mapping, apply_industry_constraint
from .account import PaperAccount
from .broker import PaperBroker
from .models import NAVRecord, PendingBuy, TargetWeight, TradeInstruction, normalize_trade_reason
from .storage import PaperStorage

# 常量定义
SHARE_LOT_SIZE = 100         # A股买卖单位（手）
SEPARATOR_LENGTH = 100       # 分隔线长度


class PaperTradingRunner:
    """纸面交易运行器
    
    负责T0和T1的完整工作流
    """
    
    def __init__(
        self,
        signal: Optional[Signal] = None,
        initial_capital: float = 500000.0,
        data_root: Optional[str] = None,
        paper_root: Optional[str] = None,
        position_sizing: str = "equal",
        horizon: int = 5,
        verbose: bool = True,
    ):
        """初始化运行器

        Args:
            signal: 信号生成器（可选）
            initial_capital: 初始资金
            data_root: 数据根目录，未传时使用项目配置 data.root
            paper_root: 纸面交易数据目录，未传时默认使用 data.root/paper
            position_sizing: 仓位管理模式，equal|score（纸面交易不支持kelly）
            horizon: 特征构建的预测周期（天数），用于生成 y_ret_N 特征，默认 5
            verbose: 是否输出详细日志
        """
        # 初始化存储
        self.storage = Storage(data_root, verbose=verbose)
        self.paper_storage = PaperStorage(paper_root, verbose=verbose)
        
        # 初始化账户和经纪
        self.account = PaperAccount(initial_capital, self.paper_storage, verbose=verbose)
        self.broker = PaperBroker(self.account, storage=self.paper_storage, verbose=verbose, data_storage=self.storage)
        
        # 初始化信号生成器
        self.signal = signal
        self.position_sizing = position_sizing
        
        # 初始化数据加载器
        self.loader = DataLoader(self.storage, verbose=verbose)
        
        # 初始化TuShare客户端
        self.client = TushareClient(verbose=verbose)
        
        # 初始化数据清洗器和特征构建器（用于 ensure 功能）
        self.cleaner = DataCleaner(verbose=verbose)
        # 实盘模式使用 require_label=False，因为 T0 没有未来数据无法生成标签
        self.feature_builder = FeatureBuilder(horizon=horizon, require_label=False)

        self.horizon = horizon  # 保存 horizon 供其他地方使用
        self.verbose = verbose
        self.missing_factors: list = []  # 缺失的因子数据名称列表
        self._strategy_state: dict = self.paper_storage.load_strategy_state()
        self._trade_dates_cache: Optional[List[str]] = None
        self._kelly_cache_date: Optional[str] = None
        self._kelly_cache_df: Optional[pd.DataFrame] = None
        # 确保基础数据存在（如交易日历、股票基本信息等）
        #ensure_basic_data(self.storage, self.loader, self.cleaner, self.client)

    def _get_cost_setting(self, key: str, default: float) -> float:
        """读取成本配置，缺失时回退到默认值。"""
        try:
            return float(get_cost_settings().get(key, default))
        except (TypeError, ValueError):
            logger.warning(f"成本配置 {key} 非法，回退为默认值 {default}")
            return default

    def _ensure_strategy_state(
        self, trading_config: Optional[TradingConfig] = None
    ) -> dict:
        """初始化并返回策略运行状态。"""
        state = self._strategy_state or {}
        state.setdefault("last_rebalance_nav", None)
        self._strategy_state = state
        return state

    def _save_strategy_state(self) -> None:
        """持久化策略运行状态。"""
        self.paper_storage.save_strategy_state(self._strategy_state)

    def _get_open_trade_dates(self) -> List[str]:
        """返回开市交易日列表（带简单缓存）。"""
        if self._trade_dates_cache is None:
            self._trade_dates_cache = self._load_open_trade_dates_from_storage()
        return self._trade_dates_cache

    def _load_open_trade_dates_from_storage(self) -> List[str]:
        """从 clean/raw 交易日历中提取覆盖范围更完整的开市日列表。"""
        clean_dates = self._extract_open_trade_dates(self.loader.load_clean_trade_cal())
        raw_dates = self._extract_open_trade_dates(self.loader.load_trade_cal())

        if raw_dates and (not clean_dates or raw_dates[-1] > clean_dates[-1]):
            return raw_dates
        return clean_dates or raw_dates

    @staticmethod
    def _extract_open_trade_dates(trade_cal: Optional[pd.DataFrame]) -> List[str]:
        """从交易日历 DataFrame 中提取开市日列表。"""
        if trade_cal is None or trade_cal.empty:
            return []
        if "cal_date" not in trade_cal.columns or "is_open" not in trade_cal.columns:
            return []

        cal_dates = trade_cal["cal_date"]
        if pd.api.types.is_datetime64_any_dtype(cal_dates):
            cal_dates = cal_dates.dt.strftime("%Y%m%d")
        else:
            cal_dates = cal_dates.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)

        is_open = trade_cal["is_open"].astype(str)
        return cal_dates[is_open == "1"].dropna().tolist()

    def _ensure_trade_calendar_coverage(self, target_date: Optional[str]) -> List[str]:
        """确保交易日历至少覆盖到目标日期。"""
        normalized_target = str(target_date).strip() if target_date is not None else ""
        if normalized_target and len(normalized_target) == 10 and normalized_target[4] == "-":
            normalized_target = normalized_target.replace("-", "")
        if normalized_target and not normalized_target.isdigit():
            normalized_target = ""

        trade_dates = self._get_open_trade_dates()
        if trade_dates and (not normalized_target or trade_dates[-1] >= normalized_target):
            return trade_dates

        if normalized_target and ensure_basic_data(self.client, self.storage, normalized_target, force=False):
            self._trade_dates_cache = self._load_open_trade_dates_from_storage()
            return self._trade_dates_cache

        return trade_dates

    def _load_kelly_window_data(self, trade_date: str) -> Optional[pd.DataFrame]:
        """加载 Kelly 波动率估计所需的近窗价格数据。"""
        if self._kelly_cache_date == trade_date and self._kelly_cache_df is not None:
            return self._kelly_cache_df

        trade_dates = self._get_open_trade_dates()
        if trade_date not in trade_dates:
            return None

        current_idx = trade_dates.index(trade_date)
        start_idx = max(0, current_idx - max(self.horizon, 20, 2 * 60))
        start_date = trade_dates[start_idx]
        daily_df = self.loader.load_clean_daily(start_date=start_date, end_date=trade_date)
        self._kelly_cache_date = trade_date
        self._kelly_cache_df = daily_df
        return daily_df

    def _estimate_stock_variance(self, stock: str, trade_date: str) -> Optional[float]:
        """估计股票近期收益率方差，供 Kelly 仓位计算使用。"""
        daily_df = self._load_kelly_window_data(trade_date)
        if daily_df is None or daily_df.empty:
            return None

        stock_df = daily_df[daily_df["ts_code"] == stock].sort_values("trade_date")
        if len(stock_df) < 20:
            return None

        stock_df = stock_df.tail(max(20, min(self.horizon * 3, 120), 60))
        price_col = "close_adj" if "close_adj" in stock_df.columns else "close"
        # 方差口径统一由 trading.sizing 计算
        return estimate_variance_from_prices(stock_df[price_col].astype(float).to_numpy())

    def _kelly_weights(
        self,
        signals: Dict[str, float],
        trade_date: str,
        half: bool = False,
    ) -> Dict[str, float]:
        """计算 Kelly / 半 Kelly 仓位权重（委托 trading.sizing 共享实现）。"""
        result, _ = compute_kelly_weights(
            signals,
            variance_fn=lambda stock: self._estimate_stock_variance(stock, trade_date),
            half=half,
            max_leverage=float(getattr(self, "kelly_max_leverage", 1.0)),
        )
        return result

    def _correct_trade_date(self, input_date: str) -> str:
        """校正交易日期：非交易日自动滚动到下一交易日
        
        Args:
            input_date: 输入日期 YYYYMMDD
            
        Returns:
            校正后的交易日期 YYYYMMDD
        """
        try:
            normalized_input = str(input_date).strip()
            if normalized_input.lower() == "next":
                normalized_input = self._resolve_next_requested_trade_date()
            if len(normalized_input) == 10 and normalized_input[4] == "-" and normalized_input[7] == "-":
                normalized_input = normalized_input.replace("-", "")

            trade_dates = self._ensure_trade_calendar_coverage(normalized_input)
            if not trade_dates:
                logger.error("无法加载交易日历")
                return normalized_input
            
            # 检查输入日期是否为交易日
            if normalized_input in trade_dates:
                return normalized_input
            
            # 找到输入日期后的第一个交易日
            for date in trade_dates:
                if date > normalized_input:
                    logger.warning(
                        f"输入日期 {normalized_input} 不是交易日，"
                        f"已自动校正到下一交易日: {date}"
                    )
                    return date
            
            # 如果没有找到后续交易日，返回原日期（可能是未来日期）
            logger.warning(f"未找到 {normalized_input} 之后的交易日，使用原日期")
            return normalized_input
            
        except Exception as e:
            logger.error(f"校正交易日期失败: {e}")
            return str(input_date).strip()

    def _resolve_next_requested_trade_date(self) -> str:
        """将 next 解析为最近执行日之后的下一个交易日。"""
        today = datetime.now().strftime("%Y%m%d")
        trade_dates = self._ensure_trade_calendar_coverage(today)
        if not trade_dates:
            logger.warning("交易日历为空，next 回退为原始输入")
            return "next"

        last_trade_date = self.paper_storage.load_last_trade_date() or ""
        if not last_trade_date:
            account_state = self.paper_storage.load_account_state()
            if account_state and account_state.last_update:
                last_trade_date = str(account_state.last_update)

        if last_trade_date:
            future_dates = [date for date in trade_dates if date > last_trade_date]
            if future_dates:
                resolved_date = future_dates[0]
                logger.info(f"trade_date=next 解析为上次执行日 {last_trade_date} 之后的 {resolved_date}")
                return resolved_date

        today = pd.Timestamp.today().strftime("%Y%m%d")
        future_dates = [date for date in trade_dates if date >= today]
        if future_dates:
            resolved_date = future_dates[0]
            logger.info(f"trade_date=next 解析为从今日起的最近交易日 {resolved_date}")
            return resolved_date

        logger.warning("未找到可用交易日，next 回退为原始输入")
        return "next"
    
    def _check_rebalance_day(
        self, 
        trade_date: str, 
        rebalance_freq: int
    ) -> bool:
        """检查是否为调仓日
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            rebalance_freq: 调仓频率（交易日数）
            
        Returns:
            True 如果是调仓日
            
        Raises:
            RuntimeError: 如果不是调仓日
        """
        # 加载调仓状态
        rebalance_state = self.paper_storage.load_rebalance_state()
        
        # 首次运行，允许执行
        if rebalance_state is None:
            logger.info("首次运行T0，允许执行")
            return True
        
        last_rebalance_date = rebalance_state.get('last_rebalance_date')
        if not last_rebalance_date:
            logger.info("无上次调仓记录，允许执行")
            return True
        
        # 计算距离上次调仓的交易日数
        try:
            trade_cal = self.loader.load_clean_trade_cal()
            if trade_cal is None:
                logger.error("无法加载交易日历，跳过调仓日检查")
                return True
            
            # 筛选开市日
            trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
            
            # 找到两个日期的索引
            try:
                last_idx = trade_dates.index(last_rebalance_date)
                current_idx = trade_dates.index(trade_date)
            except ValueError as e:
                logger.error(f"日期不在交易日历中: {e}")
                return True
            
            # 计算间隔
            days_since_last = current_idx - last_idx
            
            if days_since_last >= rebalance_freq:
                logger.info(
                    f"距离上次调仓 {last_rebalance_date} 已过 [{days_since_last}] 个交易日，"
                    f"满足调仓频率 {rebalance_freq}，允许执行"
                )
                return True
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
            return True
    
    def _generate_instructions(
        self,
        targets: List[TargetWeight],
        buy_price_type: str,
        sell_price_type: str,
        current_prices: Dict[str, float],
        source_date: str,
        protected_stocks: Optional[set] = None,
        desired_position_count: Optional[int] = None,
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

        Returns:
            交易指令列表
        """
        instructions = []
        protected_stocks = protected_stocks or set()
        del sell_price_type, protected_stocks

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
            target_shares = int(target_value / price / SHARE_LOT_SIZE) * SHARE_LOT_SIZE

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
        
        # 3. 检查调仓日
        if force_rebalance:
            logger.warning(f"强制执行 T0，跳过调仓日校验: {corrected_date}")
        else:
            self._check_rebalance_day(corrected_date, effective_config.rebalance_freq)
        
        logger.info("=" * 80)
        logger.info(f"开始T0工作流 - {corrected_date}")
        logger.info("=" * 80)
        logger.info(f"调仓频率: {effective_config.rebalance_freq} 个交易日")
        
        # 4. 生成信号（ensure_features_for_date 内部自动下载缺失的 raw/clean 数据）
        logger.info("步骤2: 生成信号")
        self.signal = self.signal or MLSignal(
            top_n=effective_config.top_n,
            model_version=effective_config.model_version,
            verbose=False,
        )
        if hasattr(self.signal, "top_n"):
            self.signal.top_n = effective_config.top_n
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
            top_n=effective_config.top_n,
            model_version=effective_config.model_version,
            buy_price_type=effective_config.buy_price,
            max_per_industry=effective_config.max_per_industry,
            max_weight_per_stock=effective_config.max_weight_per_stock,
            exclude_st=effective_config.exclude_st,
            min_list_days=effective_config.min_list_days,
            protected_stocks=protected_stocks,
            trading_config=effective_config,
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
        
        # 生成指令（使用传入的 sell_price_type 参数）
        buy_instructions = self._generate_instructions(
            targets=targets,
            buy_price_type=buy_price_type,
            sell_price_type=sell_price_type,
            current_prices=current_prices,
            source_date=corrected_date,
            protected_stocks=protected_stocks,
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
        t1_date = self._get_next_trade_date(corrected_date)
        if not t1_date:
            logger.error(f"无法获取 {corrected_date} 的下一个交易日")
            return
        
        # 保存交易指令（指令驱动模式）。
        # 同一交易日内，日度卖出规划可能已先写入次日指令，这里需要做合并而非覆盖。
        # 合并优先级：已存在的条件卖出/止损指令 > 调仓卖出 > 调仓买入。
        existing_instructions = self.paper_storage.load_instructions(t1_date) or []
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
        
        # 8. 更新调仓状态
        rebalance_state = {
            'last_rebalance_date': corrected_date,
            'rebalance_freq': effective_config.rebalance_freq
        }
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
    
    def run_t1(
        self,
        trade_date: str,
        buy_price_type: str = 'close',
        sell_price_type: str = 'close'
    ) -> None:
        """T1工作流：读取待执行目标 + 执行订单 + 更新状态
        
        Args:
            trade_date: 交易日期 YYYYMMDD（T1日期）
            buy_price_type: 买入价格类型 open/close
            sell_price_type: 卖出价格类型 open/close（固定为close）
        """
        # 1. 校正交易日期
        corrected_date = self._correct_trade_date(trade_date)
        
        # 2. 检查幂等性
        if self.paper_storage.check_run_exists("t1", corrected_date):
            raise RuntimeError(
                f"T1 工作流已在 {corrected_date} 执行过，"
                f"不允许重复执行（幂等性保障）"
            )
        
        logger.info("=" * 80)
        logger.info(f"开始T1工作流 - {corrected_date}")
        logger.info("=" * 80)
        
        # 3. 读取交易指令
        logger.info("步骤1: 读取交易指令")
        instructions = self.paper_storage.load_instructions(corrected_date)
        
        # 4. 读取补位买入计划（增量买入）
        pending_buys = self.paper_storage.load_pending_buys()
        
        # 检查是否有任何待执行任务
        if not instructions and not pending_buys:
            logger.warning(f"未找到 {corrected_date} 的交易指令或补位买入计划，跳过执行")
            return
        
        if instructions:
            logger.info(f"读取到 {len(instructions)} 条交易指令")
        if pending_buys:
            logger.info(f"读取到 {len(pending_buys)} 个补位买入计划")
        
        # 6. 加载价格数据
        logger.info("步骤2: 加载价格数据")
        buy_prices, sell_prices = self._load_prices(corrected_date, buy_price_type, sell_price_type)
        
        if not buy_prices and not sell_prices:
            logger.error("无法加载价格数据")
            return
        
        fills_count = 0
        orders_count = 0
        
        # 7. 执行交易指令
        if instructions:
            logger.info("步骤3: 执行交易指令")
            fills = self.broker.execute_instructions(
                instructions,
                buy_prices,
                sell_prices,
                corrected_date
            )
            fills_count += len(fills) if fills else 0
            orders_count += len(instructions)
        
        same_day_pending_buys: List[PendingBuy] = []
        failed_buy_targets = self.broker.get_failed_buy_targets()
        if failed_buy_targets:
            logger.info("步骤3a: 处理当日买入失败的同日顺延补位")
            same_day_pending_buys = self._build_pending_buys_from_failed_targets(
                failed_buy_targets,
                corrected_date,
                attempts=0,
            )
            self.broker.clear_failed_buy_targets()
            logger.info(f"当日新增 {len(same_day_pending_buys)} 个失败买入槽位，立即按 T0 候选顺延")

        if same_day_pending_buys:
            pending_buys = same_day_pending_buys + list(pending_buys or [])

        # 8. 执行补位买入（如果有pending_buys）
        if pending_buys:
            logger.info("步骤3b: 处理补位买入计划")
            replenishment_fills = self._execute_pending_buys(
                pending_buys,
                buy_prices,
                corrected_date,
                buy_price_type
            )
            fills_count += len(replenishment_fills) if replenishment_fills else 0
            orders_count += len(replenishment_fills) if replenishment_fills else 0
        
        # 8. 更新账户状态
        logger.info("步骤5: 更新账户状态")
        self.account.update_last_date(corrected_date)
        self.account.save_state()
        
        # 9. 记录净值
        logger.info("步骤6: 记录净值")
        # 使用收盘价计算净值
        all_prices = {**sell_prices, **buy_prices}  # 合并价格字典
        self._record_nav(corrected_date, all_prices)
        
        # 11. 保存执行记录
        run_record = {
            'trade_date': corrected_date,
            'buy_price_type': buy_price_type,
            'sell_price_type': sell_price_type,
            'instructions_count': len(instructions) if instructions else 0,
            'pending_buys_count': len(pending_buys) if pending_buys else 0,
            'orders_count': orders_count,
            'fills_count': fills_count,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.paper_storage.save_run_record("t1", corrected_date, run_record)
        
        logger.info("=" * 80)
        logger.info(f"T1工作流完成 - {corrected_date}")
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
    
    def _estimate_pending_buy_shares(
        self,
        ts_code: str,
        price: float,
        target_weight: float,
        total_pending_count: int,
        pendding_capital_retention_ratio: float
    ) -> int:
        """估算补位买入股数（与_execute_pending_buys的实际执行口径一致）
        
        本方法封装了补位买入股数的计算逻辑，确保提示信息与实际执行一致。
        
        计算逻辑：
        1. total_cash = account.cash * (1 - pendding_capital_retention_ratio)
        2. available_cash = total_cash / total_pending_count  # 每个补位目标平均分配
        3. target_value = total_cash * target_weight
        4. 若 target_value + estimated_cost > available_cash，则 target_value = available_cash - estimated_cost
        5. buy_shares = floor(target_value / price / 100) * 100  # 按100股取整
        
        Args:
            ts_code: 股票代码
            price: 买入价格
            target_weight: 目标权重
            total_pending_count: 补位队列中的总数量
            pendding_capital_retention_ratio: 补位资金保留比例
            
        Returns:
            估算的买入股数（已按100股取整）。若不足一手，返回0
        """
        if price <= 0 or total_pending_count <= 0:
            return 0
        
        # 1. 计算总可用现金（扣除保留比例）
        total_cash = self.account.get_cash() * (1 - pendding_capital_retention_ratio)
        
        # 2. 平均分配到每个补位目标
        available_cash = total_cash / total_pending_count
        
        # 3. 根据目标权重计算买入金额
        target_value = total_cash * target_weight
        
        # 4. 预估成本
        estimated_cost = self.broker.cost_model.calculate_buy_cost(target_value)
        
        # 5. 检查是否超出可用现金
        if target_value + estimated_cost > available_cash:
            target_value = available_cash - estimated_cost
            if target_value <= 0:
                return 0
        
        # 6. 计算股数（按100股取整）
        buy_shares = int(target_value / price / 100) * 100
        
        return buy_shares

    def _estimate_pending_buy_shares_backtest_style(
        self,
        ts_code: str,
        price: float,
        target_weight: float,
        current_total_value: float,
    ) -> int:
        """按回测口径估算补位买入股数。

        回测补位的买入目标金额为 current_total_value * slot_weight，
        再按 A 股一手约束取整，若现金不足则按剩余现金缩量。
        """
        buy_shares, _ = self._analyze_pending_buy_shares_backtest_style(
            ts_code=ts_code,
            price=price,
            target_weight=target_weight,
            current_total_value=current_total_value,
        )
        return buy_shares

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
        buy_shares = int(target_value / price / SHARE_LOT_SIZE) * SHARE_LOT_SIZE
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

        buy_shares = int((cash - buy_cost) / price / SHARE_LOT_SIZE) * SHARE_LOT_SIZE
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
        from .models import Fill, Order
        
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
    
    def _generate_signals(
        self,
        trade_date: str,
        universe_type: str = 'mainboard',
        top_n: int = 5,
        model_version: Optional[int] = None,
        buy_price_type: str = 'close',
        max_per_industry: Optional[int] = None,
        max_weight_per_stock: Optional[float] = None,
        exclude_st: bool = True,
        min_list_days: int = 365,
        protected_stocks: Optional[set] = None,
        trading_config: Optional[TradingConfig] = None,
    ) -> List[TargetWeight]:
        """生成信号

        Args:
            trade_date: 交易日期 YYYYMMDD
            universe_type: 股票池类型
            top_n: 持仓股票数
            model_version: ML模型版本（可选）
            buy_price_type: T1买入价格类型 open/close（用于一手可买约束）
            max_per_industry: 单行业最大持仓数量（可选）
            max_weight_per_stock: 单股最大权重（可选）
            exclude_st: 是否排除ST股票
            min_list_days: 最少上市天数
            protected_stocks: 盈利延续保护股票集合（仅用于展示，默认空）

        Returns:
            目标权重列表
        """
        effective_config = trading_config or TradingConfig(
            buy_price=buy_price_type,
            universe=universe_type,
            top_n=top_n,
            model_version=model_version,
            max_per_industry=max_per_industry,
            max_weight_per_stock=max_weight_per_stock,
            exclude_st=exclude_st,
            min_list_days=min_list_days,

            position_sizing=self.position_sizing,
        )

        # 确保 features 数据存在
        logger.info(f"检查并确保 features 数据存在: {trade_date}")
        success, missing, feature_error = ensure_features_for_date(
            self.storage,
            self.loader,
            self.feature_builder,
            self.cleaner,
            self.client,
            trade_date,
            force=False
        )
        self.missing_factors = missing
        # 释放 FeatureBuilder 缓存，回收内存（纸面交易后续不再需要）
        self.feature_builder.clear_caches()
        if not success:
            logger.error(f"无法获取 features 数据: {trade_date}")
            self._last_feature_error = feature_error
            return []
        self._last_feature_error = ""

        # 加载股票池
        stock_basic = self.loader.load_clean_stock_basic()
        if stock_basic is None:
            logger.error("无法加载stock_basic数据")
            return []

        # 创建股票池
        universe = self._create_universe(
            stock_basic, universe_type,
            exclude_st=exclude_st, min_list_days=min_list_days,
        )

        # 加载价格数据
        daily_data = self.loader.load_clean_daily_by_date(trade_date)
        signal_data = self.storage.load_cs_train_day(trade_date, subdir="cs_infer")
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {trade_date} 的日线数据")
            return []

        # 获取股票列表
        date_ts = pd.Timestamp(trade_date)
        stocks = universe.get_stocks(date_ts, daily_data)
        
        if not stocks:
            logger.warning("股票池为空")
            return []
        
        logger.info(f"股票池大小: {len(stocks)}")
        
        # 使用信号生成器
        if self.signal is None:
            # 使用默认的ML信号
            if model_version is not None:
                self.signal = MLSignal(
                    top_n=effective_config.top_n,
                    model_version=effective_config.model_version,
                    verbose=False,
                )
            else:
                logger.warning("未指定信号生成器，使用等权")
                from ..signals.base import EqualWeightSignal
                self.signal = EqualWeightSignal(top_n=effective_config.top_n)
        elif hasattr(self.signal, "top_n"):
            self.signal.top_n = effective_config.top_n
            if (
                effective_config.model_version_b is not None
                and hasattr(self.signal, "update_versions")
            ):
                self.signal.update_versions(
                    effective_config.model_version,
                    effective_config.model_version_b,
                )
            elif effective_config.model_version is not None and hasattr(
                self.signal, "update_model_version"
            ):
                self.signal.update_model_version(effective_config.model_version)
        
        # 加载行业映射（如果启用行业约束）
        industry_mapping = {}
        if max_per_industry and max_per_industry > 0:
            shenwan_industry = self.loader.load_shenwan_industry()
            industry_mapping = load_industry_mapping(shenwan_industry, verbose=True)

        # 回收内存后再进入模型加载/预测（对内存受限设备如树莓派尤为重要）
        gc.collect()

        # 生成信号（原始分数字典，权重归一化由 _normalize_signals 统一处理）
        try:
            if hasattr(self.signal, "generate_ranked"):
                # MLSignal：使用 generate_ranked 获取排序候选，并应用一手可买约束
                raw_scores, signal_meta = self._generate_ranked_with_lot_constraint(
                    date_ts,
                    stocks,
                    signal_data,
                    daily_data,
                    effective_config.top_n,
                    buy_price_type,
                    max_per_industry=effective_config.max_per_industry,
                    industry_mapping=industry_mapping,
                    trading_config=effective_config,
                    existing_positions=set(self.account.get_positions().keys()),
                    return_meta=True,
                )
            else:
                raw_scores = self.signal.generate(
                    date_ts,
                    stocks,
                    {'features': signal_data}
                )
                signal_meta = {}

            signal_dict = self._normalize_signals(raw_scores, trade_date)

            # 与 backtest 保持一致：先做单股限权，避免后续归一化抹掉留仓位效果。
            if effective_config.max_weight_per_stock is not None and signal_dict:
                from ..portfolio import cap_and_normalize_weights

                signal_dict = cap_and_normalize_weights(
                    signal_dict,
                    max_weight_per_stock=effective_config.max_weight_per_stock,
                    verbose=True,
                )
        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return []

        if not signal_dict:
            self._save_strategy_state()
            logger.warning("门控后无有效目标权重")
            return []

        # 转换为目标权重，并增强信息
        targets = self._enhance_target_info(
            signal_dict,
            stock_basic,
            daily_data,
            trade_date
        )
        
        logger.info(f"生成 {len(targets)} 个目标权重")
        
        # 打印 T0 详细信息（与最终指令生成口径保持一致）
        self._print_t0_targets(
            targets,
            stock_basic,
            daily_data,
            protected_stocks=protected_stocks,
        )

        self._save_strategy_state()
        
        return targets
    
    def _generate_ranked_with_lot_constraint(
        self,
        date: pd.Timestamp,
        stocks: List[str],
        signal_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        top_n: int,
        buy_price_type: str,
        max_per_industry: Optional[int] = None,
        industry_mapping: Optional[Dict[str, str]] = None,
        trading_config: Optional[TradingConfig] = None,
        existing_positions: Optional[set] = None,
        return_meta: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, object]]]:
        """生成原始分数字典（含行业约束 + 一手可买约束顺延补足）

        返回原始 ml_score（非归一化权重），权重归一化由 _normalize_signals 统一处理。
        一手约束：按等权分配金额估算，不足1手的候选被跳过并顺延。

        Args:
            date: 当前日期
            stocks: 股票池
            signal_data: 特征数据
            daily_data: 日线数据（包含价格）
            top_n: 目标股票数
            buy_price_type: T1买入价格类型 open/close
            max_per_industry: 单行业最大持仓数量（可选）
            industry_mapping: 行业映射字典（可选）

        Returns:
            原始分数字典，若 return_meta=True 则额外返回信号元信息
        """
        existing_positions = existing_positions or set()

        # 使用 generate_ranked 获取完整排序候选列表
        ranked_candidates = self.signal.generate_ranked(
            date,
            stocks,
            {'features': signal_data}
        )

        if not ranked_candidates:
            logger.warning(f"{date.date()} 未获取到排序候选")
            return {}

        original_count = len(ranked_candidates)

        # 应用行业约束（在一手约束之前）
        if max_per_industry and max_per_industry > 0 and industry_mapping:
            ranked_candidates = apply_industry_constraint(
                ranked_candidates=ranked_candidates,
                industry_mapping=industry_mapping,
                max_per_industry=max_per_industry,
                target_n=top_n * 3,  # 多选一些候选，后续一手约束还会筛掉
                verbose=True,
            )
            logger.info(f"行业约束后候选数: {len(ranked_candidates)} (原始 {original_count})")

        # T0 选股应完全排除已持仓，
        # 避免为已持仓股票生成“补差买单”。
        if existing_positions:
            before = len(ranked_candidates)
            ranked_candidates = [
                (ts_code, score)
                for ts_code, score in ranked_candidates
                if ts_code not in existing_positions
            ]
            excluded = before - len(ranked_candidates)
            if excluded > 0:
                logger.info(
                    f"排除已持仓候选 {excluded} 只，"
                    f"候选数 {before} -> {len(ranked_candidates)}"
                )

        logger.info(f"等权+一手约束: 排序候选数 {len(ranked_candidates)}")

        target_n = top_n

        # 构建价格映射（使用 buy_price_type 指定的价格列）
        price_col = buy_price_type  # 'open' 或 'close'
        if price_col not in daily_data.columns:
            logger.warning(f"价格列 '{price_col}' 不存在，降级到 'close'")
            price_col = 'close'
        
        price_map = {}
        for _, row in daily_data.iterrows():
            ts_code = row['ts_code']
            price = row.get(price_col)
            if not pd.isna(price) and price > 0:
                price_map[ts_code] = price
        
        # 计算每只股票的等权分配金额
        total_capital = self.account.get_total_value(price_map)
        if total_capital <= 0:
            total_capital = self.account.initial_capital
        equal_weight_value = total_capital / max(target_n, 1)
        
        # 从排序候选中筛选可买至少1手的股票，保留原始分数
        selected = []  # List[Tuple[str, float]]
        skipped_stocks = []

        for ts_code, score in ranked_candidates:
            if len(selected) >= target_n:
                break

                selected.append((ts_code, score))
                continue

            price = price_map.get(ts_code)
            if price is None or price <= 0:
                skipped_stocks.append((ts_code, "无价格数据"))
                continue

            affordable_shares = int(equal_weight_value / price / SHARE_LOT_SIZE) * SHARE_LOT_SIZE
            if affordable_shares < SHARE_LOT_SIZE:
                skipped_stocks.append((ts_code, f"不足1手(价格={price:.2f}, 可买={affordable_shares}股)"))
                continue

            selected.append((ts_code, score))

        final_count = len(selected)
        skipped_count = len(skipped_stocks)

        logger.info(
            f"一手约束筛选: 最终目标数 {final_count}, "
            f"跳过 {skipped_count} 只 (原始候选 {original_count})"
        )

        if skipped_count > 0:
            examples = skipped_stocks[:5]
            for ts_code, reason in examples:
                logger.info(f"  跳过示例: {ts_code} - {reason}")
            if skipped_count > 5:
                logger.info(f"  ... 及其他 {skipped_count - 5} 只")

        if final_count < target_n:
            logger.warning(
                f"一手约束筛选: 候选不足，目标 {target_n} 只，实际仅 {final_count} 只可选"
            )

        if final_count == 0:
            result = {}
            meta = {
                'target_n': target_n,
                'ranked_candidates': ranked_candidates,
            }
            return (result, meta) if return_meta else result

        result = {ts_code: score for ts_code, score in selected}
        meta = {
            'target_n': target_n,
            'ranked_candidates': ranked_candidates,
        }
        return (result, meta) if return_meta else result

    def _normalize_signals(self, signals: Dict[str, float], trade_date: str) -> Dict[str, float]:
        """根据 position_sizing 将原始分数转换为归一化权重字典

        支持 equal、score、kelly、half_kelly 四种模式。
        """
        if not signals:
            return {}

        sizing = self.position_sizing

        if sizing == "equal":
            weight = 1.0 / len(signals)
            return {stock: weight for stock in signals}

        if sizing in ("kelly", "half_kelly"):
            return self._kelly_weights(signals, trade_date, half=(sizing == "half_kelly"))

        if sizing != "score":
            logger.warning(f"未知 position_sizing='{sizing}'，回退到 score 加权")
        positive = {s: v for s, v in signals.items() if v > 0}
        if not positive:
            weight = 1.0 / len(signals)
            return {stock: weight for stock in signals}
        total = sum(positive.values())
        if total < 1e-12:
            weight = 1.0 / len(positive)
            return {stock: weight for stock in positive}
        return {stock: score / total for stock, score in positive.items()}

    def _create_universe(
        self, stock_basic: pd.DataFrame, universe_type: str,
        exclude_st: bool = True, min_list_days: int = 365,
    ) -> BasicUniverse:
        """创建股票池

        Args:
            stock_basic: 股票基本信息
            universe_type: 股票池类型
            exclude_st: 是否排除ST股票
            min_list_days: 最少上市天数

        Returns:
            股票池实例
        """
        if universe_type == 'mainboard':
            mainboard_stocks = stock_basic[stock_basic['market'] == '主板'].copy()
            logger.info(f"主板股票数: {len(mainboard_stocks)} / {len(stock_basic)}")

            return BasicUniverse(
                stock_basic=mainboard_stocks,
                exclude_st=exclude_st,
                min_list_days=min_list_days,
                verbose=self.verbose,
            )
        else:
            # 默认全市场
            return BasicUniverse(
                stock_basic=stock_basic,
                exclude_st=exclude_st,
                min_list_days=min_list_days,
                verbose=self.verbose,
            )
    
    def _load_prices(
        self,
        trade_date: str,
        buy_price_type: str,
        sell_price_type: str
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """加载价格数据（分开盘/收盘）
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            buy_price_type: 买入价格类型 open/close
            sell_price_type: 卖出价格类型 open/close
            
        Returns:
            (buy_prices, sell_prices) 价格字典元组
            buy_prices: {ts_code: price} 买入价格字典
            sell_prices: {ts_code: price} 卖出价格字典
        """
        daily_data = self.loader.load_clean_daily_by_date(trade_date)
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {trade_date} 的日线数据")
            return {}, {}
        
        buy_prices = {}
        sell_prices = {}
        
        # 处理买入价格
        buy_col = buy_price_type  # 'open' 或 'close'
        if buy_col not in daily_data.columns:
            logger.warning(f"买入价格列 {buy_col} 不存在，降级到 close")
            buy_col = 'close'
        
        # 处理卖出价格
        sell_col = sell_price_type  # 'open' 或 'close'
        if sell_col not in daily_data.columns:
            logger.warning(f"卖出价格列 {sell_col} 不存在，降级到 close")
            sell_col = 'close'
        
        # 填充价格字典
        for _, row in daily_data.iterrows():
            ts_code = row['ts_code']
            
            # 买入价格（如果缺失，尝试降级）
            buy_price = row.get(buy_col)
            if pd.isna(buy_price) or buy_price <= 0:
                # open缺失，降级到close
                if buy_col == 'open' and 'close' in row:
                    buy_price = row['close']
                    if not pd.isna(buy_price) and buy_price > 0:
                        logger.debug(f"{ts_code} open价格缺失，使用close={buy_price}")
            
            if not pd.isna(buy_price) and buy_price > 0:
                buy_prices[ts_code] = buy_price
            
            # 卖出价格（如果缺失，尝试降级）
            sell_price = row.get(sell_col)
            if pd.isna(sell_price) or sell_price <= 0:
                # open缺失，降级到close
                if sell_col == 'open' and 'close' in row:
                    sell_price = row['close']
                    if not pd.isna(sell_price) and sell_price > 0:
                        logger.debug(f"{ts_code} open价格缺失，使用close={sell_price}")
            
            if not pd.isna(sell_price) and sell_price > 0:
                sell_prices[ts_code] = sell_price
        
        logger.info(f"加载价格数据: 买入({buy_price_type})={len(buy_prices)}只, "
                   f"卖出({sell_price_type})={len(sell_prices)}只")
        
        return buy_prices, sell_prices
    
    # ─────────────── 盈利延续持有 / 亏损提前换出 ───────────────

    def _calc_holding_days(
        self,
        buy_date: str,
        current_date: str,
        trade_dates_list: list,
    ) -> int:
        """计算两个日期之间的交易日数"""
        try:
            buy_idx = trade_dates_list.index(buy_date)
            cur_idx = trade_dates_list.index(current_date)
            return cur_idx - buy_idx
        except ValueError:
            return 0

    def _build_pnl_price_map_for_date(
        self,
        trade_date: str,
        price_type: str = "close",
    ) -> Dict[str, float]:
        """构建某交易日的绩效价格映射（优先后复权）。"""
        daily_data = self.loader.load_clean_daily_by_date(trade_date)
        if daily_data is None or daily_data.empty:
            return {}

        if str(price_type) == "open":
            candidates = ["open_adj", "open", "close_adj", "close"]
        else:
            candidates = ["close_adj", "close", "open_adj", "open"]

        result: Dict[str, float] = {}
        for _, row in daily_data.iterrows():
            ts_code = row["ts_code"]
            value = 0.0
            for col in candidates:
                col_val = row.get(col)
                if col_val is not None and not pd.isna(col_val) and float(col_val) > 0:
                    value = float(col_val)
                    break
            if value > 0:
                result[ts_code] = value
        return result

    def _resolve_buy_pnl_price_for_position(
        self,
        pos,
        buy_price_type: str,
        cache: Dict[str, Dict[str, float]],
    ) -> float:
        """解析持仓买入绩效价（优先持仓快照中的 buy_pnl_price）。"""
        buy_pnl_price = float(getattr(pos, "buy_pnl_price", 0.0) or 0.0)
        if buy_pnl_price > 0:
            return buy_pnl_price

        buy_date = getattr(pos, "buy_date", "")
        if not buy_date:
            return 0.0

        if buy_date not in cache:
            cache[buy_date] = self._build_pnl_price_map_for_date(buy_date, buy_price_type)

        return float(cache[buy_date].get(pos.ts_code, 0.0) or 0.0)

    @staticmethod
    def _regime_combined(features_df: pd.DataFrame, config: dict) -> float:
        """组合模式：vol_target + trend 双重保护"""
        trend_exp = PaperTradingRunner._regime_trend(features_df, config)

        # 趋势保护：上行趋势时跳过 vol_target
        if config.get("market_regime_trend_guard", True) and trend_exp >= 1.0:
            return 1.0

        vol_exp = PaperTradingRunner._regime_vol_target(features_df, config)
        combine = config.get("market_regime_combine_method", "min")
        if combine == "multiply":
            combined = vol_exp * trend_exp
        else:
            combined = min(vol_exp, trend_exp)
        min_exp = config.get("market_regime_min_exposure", 0.2)
        return float(np.clip(combined, min_exp, 1.0))

    def _record_nav(self, trade_date: str, prices: Dict[str, float]) -> None:
        """记录净值
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            prices: {ts_code: price} 价格字典
        """
        cash = self.account.get_cash()
        position_value = self.account.get_position_value(prices)
        total_value = cash + position_value
        nav = total_value / self.account.initial_capital
        
        nav_record = NAVRecord(
            trade_date=trade_date,
            cash=cash,
            position_value=position_value,
            total_value=total_value,
            nav=nav
        )
        
        self.paper_storage.append_nav(nav_record)
        logger.info(f"净值记录: 现金={cash:,.2f}, 持仓={position_value:,.2f}, "
                   f"总值={total_value:,.2f}, NAV={nav:.4f}")
    
    def _get_next_trade_date(self, trade_date: str) -> Optional[str]:
        """获取下一个交易日
        
        Args:
            trade_date: 当前交易日 YYYYMMDD
            
        Returns:
            下一个交易日 YYYYMMDD，不存在返回None
        """
        try:
            trade_dates = self._ensure_trade_calendar_coverage(trade_date)
            if not trade_dates:
                logger.error("无法加载交易日历")
                return None

            # 找到当前日期之后的第一个交易日；若当天本身是交易日，则返回其后一个交易日
            for i, date in enumerate(trade_dates):
                if date > trade_date:
                    return date
                if date == trade_date and i + 1 < len(trade_dates):
                    return trade_dates[i + 1]
            
            logger.debug(f"未找到 {trade_date} 的下一个交易日")
            return None
        except Exception as e:
            logger.error(f"获取下一个交易日失败: {e}")
            return None

    def _get_prev_trade_date(self, trade_date: str) -> Optional[str]:
        """获取上一个交易日。

        Args:
            trade_date: 当前交易日 YYYYMMDD

        Returns:
            上一个交易日 YYYYMMDD，不存在返回None
        """
        try:
            trade_dates = self._ensure_trade_calendar_coverage(trade_date)
            if not trade_dates:
                logger.error("无法加载交易日历")
                return None

            previous_date = None
            for date in trade_dates:
                if date >= trade_date:
                    break
                previous_date = date

            if previous_date is not None:
                return previous_date

            logger.warning(f"未找到 {trade_date} 的上一个交易日")
            return None
        except Exception as e:
            logger.error(f"获取上一个交易日失败: {e}")
            return None

    def _enhance_target_info(
        self,
        signal_dict: Dict[str, float],
        stock_basic: pd.DataFrame,
        daily_data: pd.DataFrame,
        trade_date: str
    ) -> List[TargetWeight]:
        """增强目标权重信息
        
        为每个目标添加股票名称等额外信息
        
        Args:
            signal_dict: {ts_code: weight} 信号字典
            stock_basic: 股票基本信息
            daily_data: 日线数据
            trade_date: 交易日期
            
        Returns:
            增强后的目标权重列表
        """
        # 构建股票名称映射
        name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
        
        # 构建价格映射
        price_map = {}
        if daily_data is not None:
            for _, row in daily_data.iterrows():
                price_map[row['ts_code']] = row.get('close', 0.0)
        
        # 转换为目标权重
        targets = []
        for ts_code, weight in signal_dict.items():
            name = name_map.get(ts_code, '-')
            price = price_map.get(ts_code, 0.0)
            
            # 构建原因字符串（包含权重信息）
            reason = f"信号生成 (权重={weight:.4f})"
            
            target = TargetWeight(
                ts_code=ts_code,
                target_weight=weight,
                reason=reason
            )
            targets.append(target)
        
        return targets
    
    def _print_t0_targets(
        self,
        targets: List[TargetWeight],
        stock_basic: pd.DataFrame,
        daily_data: pd.DataFrame,
        protected_stocks: Optional[set] = None,
    ) -> None:
        """打印 T0 目标详细信息（包含买入/减仓/清仓）
        
        输出包含：代码、名称、方向、参考价格、建议股数、原因
        
        Args:
            targets: 目标权重列表
            stock_basic: 股票基本信息
            daily_data: 日线数据
        """
        protected_stocks = protected_stocks or set()

        current_positions = self.account.get_positions()
        if not targets and not current_positions:
            logger.info("无 T0 目标")
            return
        
        logger.info("")
        logger.info("=" * SEPARATOR_LENGTH)
        logger.info("T0 建仓目标详情")
        logger.info("=" * SEPARATOR_LENGTH)
        
        # 构建股票名称和价格映射
        name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
        price_map = {}
        if daily_data is not None:
            for _, row in daily_data.iterrows():
                price_map[row['ts_code']] = row.get('close', 0.0)
        
        # 获取当前持仓
        
        # 与最终指令生成保持一致：按当前总资产并考虑现金保留比例
        capital_retention_ratio = self._get_cost_setting("capital_retention_ratio", 0.0)
        total_capital = self.account.get_total_value(price_map) * (1 - capital_retention_ratio)
        
        # 准备表格列宽和对齐
        widths = [12, 10, 6, 10, 10, 30]
        aligns = ['left', 'left', 'left', 'right', 'right', 'left']
        
        # 表头
        header = ["股票代码", "股票名称", "方向", "参考价格", "建议股数", "原因"]
        logger.info(format_row(header, widths, aligns))
        logger.info("-" * SEPARATOR_LENGTH)
        
        # 目标权重字典
        target_weights = {t.ts_code: (t.target_weight, t.reason) for t in targets}
        
        # 0. 处理所有目标股票（买入/加仓/减仓/清仓）
        all_stocks = set(target_weights.keys()) | set(current_positions.keys())
        
        # 1. 初始化存储列表和计数器
        rows_to_print = []
        stats = {"保留": 0, "清仓": 0, "减仓": 0, "加仓": 0, "买入": 0}

        for ts_code in all_stocks:
            target_weight, reason = target_weights.get(ts_code, (0.0, "退出持仓"))
            pos = current_positions.get(ts_code)
            current_shares = pos.shares if pos else 0
            
            name = name_map.get(ts_code, '-')
            price = price_map.get(ts_code, 0.0)
            
            if price <= 0:
                continue
            
            target_value = total_capital * target_weight
            target_shares = int(target_value / price / SHARE_LOT_SIZE) * SHARE_LOT_SIZE
            
            # 判断方向
            if target_shares > current_shares:
                direction = "买入" if current_shares == 0 else "加仓"
                suggested_shares = (target_shares - current_shares) // SHARE_LOT_SIZE * SHARE_LOT_SIZE
            elif target_shares < current_shares:
                raw_direction = "清仓" if target_shares == 0 else "减仓"
                direction = "保留"
                suggested_shares = 0
                if ts_code in protected_stocks:
                    reason_text = f"盈利延续保护（原目标: {raw_direction}）"
                else:
                    reason_text = f"持有期/条件驱动卖出（原目标: {raw_direction}）"
            else:
                continue

            if suggested_shares <= 0 and direction != "保留":
                continue
            
            # 统计数量
            if direction in stats:
                stats[direction] += 1
            
            if direction != "保留":
                reason_text = reason if reason else "信号生成"
            rows_to_print.append({
                'data': [ts_code, name, direction, f"{price:.2f}", str(suggested_shares), reason_text],
                'direction': direction
            })

        # 2. 按照指定顺序排序：保留 > 加仓 > 买入
        priority = {"保留": 0, "加仓": 1, "买入": 2}
        rows_to_print.sort(key=lambda x: priority.get(x['direction'], 99))

        # 3. 打印表格行
        for item in rows_to_print:
            logger.info(format_row(item['data'], widths, aligns))
        
        # 4. 打印统计摘要
        logger.info("-" * SEPARATOR_LENGTH)
        stats_str = (
            f"【操作统计】 保留: {stats['保留']} | "
            f"加仓: {stats['加仓']} | 买入: {stats['买入']}"
        )
        logger.info(stats_str)
        
        logger.info("=" * SEPARATOR_LENGTH)
        logger.info("")
    
    def run_retry(
        self,
        trade_date: str,
        sell_price_type: str = 'close'
    ) -> None:
        """重试延迟卖出订单
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            sell_price_type: 卖出价格类型 open/close
        """
        # 1. 校正交易日期
        corrected_date = self._correct_trade_date(trade_date)
        
        # 注意：retry 命令不加锁，允许同日多次执行
        
        logger.info("=" * 80)
        logger.info(f"重试延迟卖出 - {corrected_date}")
        logger.info("=" * 80)
        
        # 2. 重试延迟卖出
        fills = self.broker.retry_pending_sells(corrected_date, sell_price_type)
        
        # 3. 如果有成交，更新账户状态和净值
        if fills:
            logger.info("步骤1: 更新账户状态")
            self.account.update_last_date(corrected_date)
            self.account.save_state()
            
            logger.info("步骤2: 记录净值")
            # 加载价格
            buy_prices, sell_prices = self._load_prices(corrected_date, 'close', sell_price_type)
            all_prices = {**sell_prices, **buy_prices}
            self._record_nav(corrected_date, all_prices)
        
        logger.info("=" * 80)
        logger.info(f"重试完成 - {corrected_date}，成交 {len(fills)} 笔")
        logger.info("=" * 80)
    
    def generate_replacement_targets(
        self,
        trade_date: str,
        failed_count: int,
        universe_type: str = 'mainboard',
        model_version: Optional[int] = None,
        buy_price_type: str = 'close',
        original_signal_date: str = "",
        max_per_industry: Optional[int] = None,
        exclude_st: bool = True,
        min_list_days: int = 365,
        trading_config: Optional[TradingConfig] = None,
    ) -> List[TargetWeight]:
        """生成补位目标（当买入失败时使用）

        使用现有的信号生成链路，从候选中选择 top_k（k=失败数量）的补位股票，
        应用行业约束和一手可买约束，生成新的目标权重列表。

        Args:
            trade_date: 当前交易日期 YYYYMMDD（用于生成信号）
            failed_count: 失败买入的数量
            universe_type: 股票池类型
            model_version: ML模型版本
            buy_price_type: 买入价格类型（用于一手约束检查）
            original_signal_date: 原始信号日期（T0日期）
            max_per_industry: 单行业最大持仓数量（可选）
            exclude_st: 是否排除ST股票
            min_list_days: 最少上市天数

        Returns:
            补位目标权重列表
        """
        if failed_count <= 0:
            logger.info("无需生成补位目标")
            return []
        
        logger.info("=" * 80)
        logger.info(f"生成补位目标 - {trade_date}")
        logger.info(f"补位数量: {failed_count}")
        logger.info("=" * 80)
        
        # 1. 确保features数据存在
        logger.info(f"检查并确保 features 数据存在: {trade_date}")
        success, missing, _ = ensure_features_for_date(
            self.storage,
            self.loader,
            self.feature_builder,
            self.cleaner,
            self.client,
            trade_date,
            force=False
        )
        self.missing_factors = missing
        if not success:
            logger.error(f"无法获取 features 数据: {trade_date}")
            return []

        # 2. 加载股票池
        stock_basic = self.loader.load_clean_stock_basic()
        if stock_basic is None:
            logger.error("无法加载stock_basic数据")
            return []
        
        # 创建股票池
        universe = self._create_universe(
            stock_basic, universe_type,
            exclude_st=exclude_st, min_list_days=min_list_days,
        )

        # 3. 加载数据
        daily_data = self.loader.load_clean_daily_by_date(trade_date)
        signal_data = self.storage.load_cs_train_day(trade_date, subdir="cs_infer")
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {trade_date} 的日线数据")
            return []
        
        # 4. 获取股票列表（排除已持仓的）
        date_ts = pd.Timestamp(trade_date)
        stocks = universe.get_stocks(date_ts, daily_data)
        
        # 排除已持仓的股票（补位只考虑新股票）
        current_positions = set(self.account.get_positions().keys())
        stocks = [s for s in stocks if s not in current_positions]
        
        if not stocks:
            logger.warning("股票池为空（排除持仓后）")
            return []
        
        logger.info(f"股票池大小（排除持仓）: {len(stocks)}")
        
        if trading_config is not None:
            effective_config = replace(
                trading_config,
                buy_price=buy_price_type,
                universe=universe_type,
                top_n=failed_count,
                model_version=(
                    model_version
                    if model_version is not None
                    else trading_config.model_version
                ),
                max_per_industry=(
                    max_per_industry
                    if max_per_industry is not None
                    else trading_config.max_per_industry
                ),
                exclude_st=exclude_st,
                min_list_days=min_list_days,
                position_sizing=self.position_sizing,
            )
        else:
            effective_config = TradingConfig(
                buy_price=buy_price_type,
                universe=universe_type,
                top_n=failed_count,
                model_version=model_version,
                max_per_industry=max_per_industry,
                exclude_st=exclude_st,
                min_list_days=min_list_days,
                position_sizing=self.position_sizing,
            )

        # 5. 使用信号生成器获取排序候选
        if self.signal is None:
            if model_version is not None:
                self.signal = MLSignal(
                    top_n=effective_config.top_n,
                    model_version=effective_config.model_version,
                    verbose=False,
                )
            else:
                logger.warning("未指定信号生成器，使用等权")
                from ..signals.base import EqualWeightSignal
                self.signal = EqualWeightSignal(top_n=effective_config.top_n)
        elif hasattr(self.signal, "top_n"):
            self.signal.top_n = effective_config.top_n
            if (
                effective_config.model_version_b is not None
                and hasattr(self.signal, "update_versions")
            ):
                self.signal.update_versions(
                    effective_config.model_version,
                    effective_config.model_version_b,
                )
            elif effective_config.model_version is not None and hasattr(
                self.signal, "update_model_version"
            ):
                self.signal.update_model_version(effective_config.model_version)
        
        # 加载行业映射（如果启用行业约束）
        industry_mapping = {}
        if max_per_industry and max_per_industry > 0:
            shenwan_industry = self.loader.load_shenwan_industry()
            industry_mapping = load_industry_mapping(shenwan_industry, verbose=True)

        # 6. 生成排序候选（使用与T0相同的逻辑）
        try:
            if hasattr(self.signal, "generate_ranked"):
                raw_scores, signal_meta = self._generate_ranked_with_lot_constraint(
                    date_ts,
                    stocks,
                    signal_data,
                    daily_data,
                    effective_config.top_n,
                    buy_price_type,
                    max_per_industry=effective_config.max_per_industry,
                    industry_mapping=industry_mapping,
                    trading_config=effective_config,
                    existing_positions=current_positions,
                    return_meta=True,
                )
            else:
                raw_scores = self.signal.generate(
                    date_ts,
                    stocks,
                    {'features': signal_data}
                )
                signal_meta = {}

            signal_dict = self._normalize_signals(raw_scores, trade_date)
        except Exception as e:
            logger.error(f"补位信号生成失败: {e}")
            return []

        if not signal_dict:
            logger.warning("补位门控后无有效目标")
            return []
        
        # 7. 转换为目标权重
        targets = self._enhance_target_info(
            signal_dict,
            stock_basic,
            daily_data,
            trade_date
        )
        
        if len(targets) > failed_count:
            logger.warning(
                f"补位目标数量 {len(targets)} 超过缺口数 {failed_count}，"
                f"已截断为前 {failed_count} 个"
            )
            targets = targets[:failed_count]

        # 8. 修改reason以标识补位来源
        for target in targets:
            target.reason = f"补位-{target.reason}"
        
        logger.info(f"生成 {len(targets)} 个补位目标")
        
        # 9. 打印补位目标
        self._print_replacement_targets(targets, stock_basic, daily_data)
        
        logger.info("=" * 80)
        logger.info(f"补位目标生成完成 - {len(targets)} 个")
        logger.info("=" * 80)
        
        return targets
    
    def _print_replacement_targets(
        self,
        targets: List[TargetWeight],
        stock_basic: pd.DataFrame,
        daily_data: pd.DataFrame
    ) -> None:
        """打印补位目标（格式与T0输出一致）
        
        使用与实际执行一致的股数估算逻辑，包含现金保留比例、成本预估等。
        
        Args:
            targets: 目标权重列表
            stock_basic: 股票基本信息
            daily_data: 日线数据
        """
        if not targets:
            logger.info("无补位目标")
            return
        
        # 构建映射
        name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
        price_map = {}
        if daily_data is not None:
            for _, row in daily_data.iterrows():
                price_map[row['ts_code']] = row.get('close', 0.0)
        
        # 加载配置以获取资金保留比例
        pendding_capital_retention_ratio = self._get_cost_setting(
            "pendding_capital_retention_ratio", 0.3
        )
        
        # 打印表头
        logger.info("=" * 120)
        logger.info("补位买入目标详情（需要在下一交易日继续买入）")
        logger.info("=" * 120)
        logger.info(f"注意：以下股数为估算值，基于当前价格与现金（保留比例 {pendding_capital_retention_ratio:.1%}）")
        logger.info(f"实际执行时会受到执行日价格变化、补位队列长度变化等因素影响，但计算规则一致")
        logger.info("=" * 120)
        
        header = ["股票代码", "股票名称", "方向", "参考价格", "估算股数", "原因"]
        widths = [15, 12, 8, 12, 12, 60]
        aligns = ['left', 'left', 'left', 'right', 'right', 'left']
        logger.info(format_row(header, widths, aligns))
        logger.info("-" * 120)
        
        # 打印每行
        for target in targets:
            name = name_map.get(target.ts_code, '-')
            price = price_map.get(target.ts_code, 0.0)
            
            # 使用统一的估算方法计算建议股数
            if price > 0:
                suggested_shares = self._estimate_pending_buy_shares(
                    ts_code=target.ts_code,
                    price=price,
                    target_weight=target.target_weight,
                    total_pending_count=len(targets),
                    pendding_capital_retention_ratio=pendding_capital_retention_ratio
                )
            else:
                suggested_shares = 0
            
            # 如果不足一手，显示提示
            shares_display = str(suggested_shares) if suggested_shares > 0 else "0 (不足一手)"
            
            row = [
                target.ts_code,
                name,
                "买入",
                f"{price:.2f}" if price > 0 else "-",
                shares_display,
                target.reason
            ]
            logger.info(format_row(row, widths, aligns))
        
        logger.info("=" * 120)
