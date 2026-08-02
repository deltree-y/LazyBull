"""纸面交易经纪模块"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ...common.print_table import format_row

from ...common.cost import CostModel
from ...common.trade_status import evaluate_trade_status
from ...trading.sizing import compute_lot_shares, compute_min_buy_value_threshold
from ..account import PaperAccount
from ..models import Fill, Order, PendingBuy, TargetWeight, normalize_trade_reason
from ..storage import PaperStorage

from .tradability import PaperTradabilityMixin
from .execution import PaperExecutionMixin
from .retry import PaperRetryMixin
from .positions import PaperPositionsMixin

class PaperBroker(
    PaperTradabilityMixin,
    PaperExecutionMixin,
    PaperRetryMixin,
    PaperPositionsMixin,
):
    """纸面交易经纪

    负责生成订单、计算成本、打印明细、记录成交
    """

    def __init__(
        self,
        account: PaperAccount,
        cost_model: Optional[CostModel] = None,
        storage: Optional[PaperStorage] = None,
        verbose: bool = True,
        data_storage = None,  # 新增：数据存储实例（用于读取 raw/suspend 数据）
    ):
        """初始化经纪
        
        Args:
            account: 账户实例
            cost_model: 成本模型
            storage: 存储实例
            verbose: 是否输出详细日志
            data_storage: 数据存储实例（用于读取 raw/suspend 数据），如不提供则在需要时创建
        """
        self.account = account
        self.cost_model = cost_model or CostModel()
        self.storage = storage or PaperStorage()
        self.data_storage = data_storage  # 保存数据存储实例
        self.order_table_widths = [12, 6, 10, 10, 8, 8, 10, 12, 10, 10, 10, 10, 15]
        self.order_table_aligns = ['left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left']
        # 持仓表列：股票代码(名称)、股数、当前价格、买入均价、买入日期、持有交易日、持有剩余、当前市值、浮盈、收益率(%)、状态
        self.positions_table_widths = [20, 8, 10, 10, 12, 10, 10, 12, 12, 12, 8]
        self.positions_table_aligns = ['left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left']
        self.verbose = verbose
        # 加载延迟卖出队列
        self.pending_sells = self.storage.load_pending_sells()
        # 加载延迟买入队列（补位计划）
        self.pending_buys = self.storage.load_pending_buys()
        # 记录最近一次买入失败的目标（用于补位）
        self._failed_buy_targets = []
        # 停牌日历实例（延迟创建）
        self._suspend_calendar = None
        # 交易日历缓存（开市日列表）
        self._open_trade_dates: Optional[List[str]] = None
