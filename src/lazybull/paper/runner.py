"""纸面交易运行器"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ..data import DataLoader, Storage, TushareClient
from ..signals.base import Signal
from ..signals.ml_signal import MLSignal
from ..universe.base import BasicUniverse
from .account import PaperAccount
from .broker import PaperBroker
from .models import NAVRecord, TargetWeight
from .storage import PaperStorage


class PaperTradingRunner:
    """纸面交易运行器
    
    负责T0和T1的完整工作流
    """
    
    def __init__(
        self,
        signal: Optional[Signal] = None,
        initial_capital: float = 500000.0,
        data_root: str = "./data",
        paper_root: str = "./data/paper"
    ):
        """初始化运行器
        
        Args:
            signal: 信号生成器（可选）
            initial_capital: 初始资金
            data_root: 数据根目录
            paper_root: 纸面交易数据目录
        """
        # 初始化存储
        self.storage = Storage(data_root)
        self.paper_storage = PaperStorage(paper_root)
        
        # 初始化账户和经纪
        self.account = PaperAccount(initial_capital, self.paper_storage)
        self.broker = PaperBroker(self.account, storage=self.paper_storage)
        
        # 初始化信号生成器
        self.signal = signal
        
        # 初始化数据加载器
        self.loader = DataLoader(self.storage)
        
        # 初始化TuShare客户端
        self.client = TushareClient()
    
    def run_t0(
        self,
        trade_date: str,
        buy_price_type: str = 'close',
        universe_type: str = 'mainboard',
        top_n: int = 5,
        model_version: Optional[int] = None
    ) -> None:
        """T0工作流：拉取数据 + 生成T1待执行目标
        
        Args:
            trade_date: 交易日期 YYYYMMDD（T0日期）
            buy_price_type: T1买入价格类型 open/close
            universe_type: 股票池类型 mainboard
            top_n: 持仓股票数
            model_version: ML模型版本（可选）
        """
        logger.info("=" * 80)
        logger.info(f"开始T0工作流 - {trade_date}")
        logger.info("=" * 80)
        
        # 1. 拉取数据
        logger.info("步骤1: 拉取数据")
        self._download_data(trade_date)
        
        # 2. 生成信号
        logger.info("步骤2: 生成信号")
        targets = self._generate_signals(
            trade_date,
            universe_type=universe_type,
            top_n=top_n,
            model_version=model_version
        )
        
        if not targets:
            logger.warning("未生成任何目标权重")
            return
        
        # 3. 持久化待执行目标
        logger.info("步骤3: 保存待执行目标")
        # T0生成的是T1执行的目标，所以需要获取T1日期
        t1_date = self._get_next_trade_date(trade_date)
        if not t1_date:
            logger.error(f"无法获取 {trade_date} 的下一个交易日")
            return
        
        self.paper_storage.save_pending_weights(t1_date, targets)
        
        logger.info("=" * 80)
        logger.info(f"T0工作流完成 - 已生成 {len(targets)} 个目标权重，待T1执行")
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
        logger.info("=" * 80)
        logger.info(f"开始T1工作流 - {trade_date}")
        logger.info("=" * 80)
        
        # 1. 读取待执行目标
        logger.info("步骤1: 读取待执行目标")
        targets = self.paper_storage.load_pending_weights(trade_date)
        
        if not targets:
            logger.warning(f"未找到 {trade_date} 的待执行目标")
            return
        
        logger.info(f"读取到 {len(targets)} 个目标权重")
        
        # 2. 加载价格数据
        logger.info("步骤2: 加载价格数据")
        prices = self._load_prices(trade_date, buy_price_type, sell_price_type)
        
        if not prices:
            logger.error("无法加载价格数据")
            return
        
        # 3. 生成订单
        logger.info("步骤3: 生成订单")
        orders = self.broker.generate_orders(targets, prices, trade_date)
        
        if not orders:
            logger.warning("未生成任何订单")
        else:
            # 4. 执行订单并打印明细
            logger.info("步骤4: 执行订单")
            fills = self.broker.execute_orders(
                orders,
                trade_date,
                buy_price_type,
                sell_price_type
            )
        
        # 5. 更新账户状态
        logger.info("步骤5: 更新账户状态")
        self.account.update_last_date(trade_date)
        self.account.save_state()
        
        # 6. 记录净值
        logger.info("步骤6: 记录净值")
        self._record_nav(trade_date, prices)
        
        logger.info("=" * 80)
        logger.info(f"T1工作流完成 - {trade_date}")
        logger.info("=" * 80)
    
    def _download_data(self, trade_date: str) -> None:
        """下载数据
        
        Args:
            trade_date: 交易日期 YYYYMMDD
        """
        try:
            # 下载日线数据
            if not self.storage.is_data_exists("clean", "daily", trade_date):
                logger.info(f"下载日线数据: {trade_date}")
                daily_data = self.client.get_daily(trade_date=trade_date)
                adj_factor = self.client.get_adj_factor(trade_date=trade_date)
                
                # 简单清洗并保存
                if not daily_data.empty:
                    from ..data.cleaner import DataCleaner
                    cleaner = DataCleaner()
                    daily_clean = cleaner.clean_daily(daily_data, adj_factor)
                    self.storage.save_clean_by_date(daily_clean, "daily", trade_date)
                    logger.info(f"已保存clean数据: {len(daily_clean)} 条")
            else:
                logger.info(f"数据已存在，跳过下载: {trade_date}")
        except Exception as e:
            logger.error(f"下载数据失败: {e}")
            raise
    
    def _generate_signals(
        self,
        trade_date: str,
        universe_type: str = 'mainboard',
        top_n: int = 5,
        model_version: Optional[int] = None
    ) -> List[TargetWeight]:
        """生成信号
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            universe_type: 股票池类型
            top_n: 持仓股票数
            model_version: ML模型版本（可选）
            
        Returns:
            目标权重列表
        """
        # 加载股票池
        stock_basic = self.loader.load_clean_stock_basic()
        if stock_basic is None:
            logger.error("无法加载stock_basic数据")
            return []
        
        # 创建股票池（仅主板）
        universe = self._create_universe(stock_basic, universe_type)
        
        # 加载价格数据
        daily_data = self.loader.load_clean_daily_by_date(trade_date)
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
                self.signal = MLSignal(top_n=top_n, model_version=model_version)
            else:
                logger.warning("未指定信号生成器，使用等权")
                from ..signals.base import EqualWeightSignal
                self.signal = EqualWeightSignal(top_n=top_n)
        
        # 生成信号
        try:
            signal_dict = self.signal.generate(
                date_ts,
                stocks,
                {'daily': daily_data}
            )
        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return []
        
        # 转换为目标权重
        targets = []
        for ts_code, weight in signal_dict.items():
            targets.append(TargetWeight(
                ts_code=ts_code,
                target_weight=weight,
                reason="信号生成"
            ))
        
        logger.info(f"生成 {len(targets)} 个目标权重")
        return targets
    
    def _create_universe(self, stock_basic: pd.DataFrame, universe_type: str) -> BasicUniverse:
        """创建股票池
        
        Args:
            stock_basic: 股票基本信息
            universe_type: 股票池类型
            
        Returns:
            股票池实例
        """
        if universe_type == 'mainboard':
            # 仅沪深主板
            # 过滤逻辑：保留 ts_code 以 SH/SZ 开头，排除科创板、创业板、北交所
            # market 字段通常为 "主板"、"创业板"、"科创板" 等
            # 保守过滤：仅保留 market == "主板"
            mainboard_stocks = stock_basic[stock_basic['market'] == '主板'].copy()
            logger.info(f"主板股票数: {len(mainboard_stocks)} / {len(stock_basic)}")
            
            return BasicUniverse(
                stock_basic=mainboard_stocks,
                exclude_st=True,
                min_list_days=60
            )
        else:
            # 默认全市场
            return BasicUniverse(
                stock_basic=stock_basic,
                exclude_st=True,
                min_list_days=60
            )
    
    def _load_prices(
        self,
        trade_date: str,
        buy_price_type: str,
        sell_price_type: str
    ) -> Dict[str, float]:
        """加载价格数据
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            buy_price_type: 买入价格类型
            sell_price_type: 卖出价格类型
            
        Returns:
            {ts_code: price} 价格字典
        """
        daily_data = self.loader.load_clean_daily_by_date(trade_date)
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {trade_date} 的日线数据")
            return {}
        
        # 根据价格类型选择价格列
        # 注意：这里简化处理，买卖使用相同价格
        # 实际应用中可以分别处理
        price_col = 'open' if buy_price_type == 'open' else 'close'
        
        prices = {}
        for _, row in daily_data.iterrows():
            prices[row['ts_code']] = row[price_col]
        
        logger.info(f"加载价格数据: {len(prices)} 只股票")
        return prices
    
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
            trade_cal = self.loader.load_clean_trade_cal()
            if trade_cal is None:
                logger.error("无法加载交易日历")
                return None
            
            # 筛选开市日
            trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
            
            # 找到当前日期的下一个交易日
            for i, date in enumerate(trade_dates):
                if date == trade_date and i + 1 < len(trade_dates):
                    return trade_dates[i + 1]
            
            logger.warning(f"未找到 {trade_date} 的下一个交易日")
            return None
        except Exception as e:
            logger.error(f"获取下一个交易日失败: {e}")
            return None
