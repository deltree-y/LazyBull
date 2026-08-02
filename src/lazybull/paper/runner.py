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
from ..trading.sizing import (
    compute_kelly_weights,
    compute_lot_shares,
    estimate_variance_from_prices,
)
from ..trading.stagger import (
    build_tranche_schedule_from_anchor,
    get_tranche_capital_fraction as _shared_tranche_capital_fraction,
    get_tranche_target_count as _shared_tranche_target_count,
)
from ..universe.base import BasicUniverse
from ..portfolio.industry_constraint import load_industry_mapping, apply_industry_constraint
from .account import PaperAccount
from .broker import PaperBroker
from .models import NAVRecord, PendingBuy, TargetWeight, TradeInstruction, normalize_trade_reason
from .storage import PaperStorage

# 常量定义
SHARE_LOT_SIZE = 100         # A股买卖单位（手）
SEPARATOR_LENGTH = 100       # 分隔线长度

from ..common.constants import SEPARATOR_LENGTH, SHARE_LOT_SIZE

from .runner_calendar import PaperCalendarMixin
from .runner_rebalance import PaperRebalanceMixin
from .runner_instructions import PaperInstructionMixin
from .runner_execution import PaperExecutionMixin
from .runner_pricing import PaperPricingMixin
from .runner_signals import PaperSignalMixin
from .runner_replacement import PaperReplacementMixin

class PaperTradingRunner(
    PaperCalendarMixin,
    PaperRebalanceMixin,
    PaperInstructionMixin,
    PaperExecutionMixin,
    PaperPricingMixin,
    PaperSignalMixin,
    PaperReplacementMixin,
):
    """    纸面交易运行器
    
    负责T0和T1的完整工作流"""

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
