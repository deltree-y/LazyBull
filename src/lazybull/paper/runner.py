"""纸面交易运行器"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ..common.print_table import format_row
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
from ..universe.base import BasicUniverse
from .account import PaperAccount
from .broker import PaperBroker
from .models import NAVRecord, TargetWeight, TradeInstruction
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
        data_root: str = "./data",
        paper_root: str = "./data/paper",
        weight_method: str = "equal",
        horizon: int = 5,
        verbose: bool = True,
    ):
        """初始化运行器
        
        Args:
            signal: 信号生成器（可选）
            initial_capital: 初始资金
            data_root: 数据根目录
            paper_root: 纸面交易数据目录
            weight_method: 权重分配方法，"equal"表示等权，"score"表示按分数加权
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
        self.weight_method = weight_method
        
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
        # 确保基础数据存在（如交易日历、股票基本信息等）
        #ensure_basic_data(self.storage, self.loader, self.cleaner, self.client)
    
    def _correct_trade_date(self, input_date: str) -> str:
        """校正交易日期：非交易日自动滚动到下一交易日
        
        Args:
            input_date: 输入日期 YYYYMMDD
            
        Returns:
            校正后的交易日期 YYYYMMDD
        """
        try:
            trade_cal = self.loader.load_clean_trade_cal()
            if trade_cal is None:
                logger.error("无法加载交易日历")
                return input_date
            
            # 筛选开市日
            trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
            
            # 检查输入日期是否为交易日
            if input_date in trade_dates:
                return input_date
            
            # 找到输入日期后的第一个交易日
            for date in trade_dates:
                if date > input_date:
                    logger.warning(
                        f"输入日期 {input_date} 不是交易日，"
                        f"已自动校正到下一交易日: {date}"
                    )
                    return date
            
            # 如果没有找到后续交易日，返回原日期（可能是未来日期）
            logger.warning(f"未找到 {input_date} 之后的交易日，使用原日期")
            return input_date
            
        except Exception as e:
            logger.error(f"校正交易日期失败: {e}")
            return input_date
    
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
        source_date: str
    ) -> List[TradeInstruction]:
        """从目标权重生成明确的交易指令
        
        Args:
            targets: 目标权重列表
            buy_price_type: 买入价格类型 open/close
            sell_price_type: 卖出价格类型 open/close
            current_prices: 当前价格字典
            source_date: 源日期（T0日期）
            
        Returns:
            交易指令列表
        """
        instructions = []
        
        # 目标权重字典
        target_weights = {t.ts_code: (t.target_weight, t.reason) for t in targets}
        
        # 当前持仓
        current_positions = self.account.get_positions()
        
        # 使用账户总资金计算
        import yaml
        with open("configs/base.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        #total_capital = self.account.initial_capital #???应使用当前总资产,可以乘一个系数
        total_capital = self.account.get_total_value(current_prices) * (1 - cfg['costs']['capital_retention_ratio'])  # 乘以系数以留出现金空间，避免过度买入
        
        # 合并所有股票（目标+持仓）
        all_stocks = set(target_weights.keys()) | set(current_positions.keys())
        
        for ts_code in all_stocks:
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
                        target_weight=target_weight
                    ))
            elif target_shares < current_shares:
                # 卖出或减仓
                # 如果是清仓（目标权重为0），必须卖出全部
                if target_weight == 0:
                    shares = current_shares
                else:
                    shares = (current_shares - target_shares) // SHARE_LOT_SIZE * SHARE_LOT_SIZE
                
                if shares > 0:
                    instructions.append(TradeInstruction(
                        ts_code=ts_code,
                        action='sell',
                        shares=shares,
                        price_type=sell_price_type,
                        reason=reason if target_weight > 0 else "退出持仓",
                        source_date=source_date,
                        target_weight=target_weight
                    ))
        
        logger.info(f"生成 {len(instructions)} 条交易指令")
        return instructions
    
    def run_t0(
        self,
        trade_date: str,
        buy_price_type: str = 'close',
        sell_price_type: str = 'close',
        universe_type: str = 'mainboard',
        top_n: int = 5,
        model_version: Optional[int] = None,
        rebalance_freq: int = 5
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
        """
        # 1. 校正交易日期
        corrected_date = self._correct_trade_date(trade_date)
        
        # 2. 检查幂等性
        if self.paper_storage.check_run_exists("t0", corrected_date):
            raise RuntimeError(
                f"T0 工作流已在 {corrected_date} 执行过，"
                f"不允许重复执行（幂等性保障）"
            )
        
        # 3. 检查调仓日
        self._check_rebalance_day(corrected_date, rebalance_freq)
        
        logger.info("=" * 80)
        logger.info(f"开始T0工作流 - {corrected_date}")
        logger.info("=" * 80)
        logger.info(f"调仓频率: {rebalance_freq} 个交易日")
        
        # 4. 拉取数据
        logger.info("步骤1: 拉取数据")
        self._download_data(corrected_date)
        
        # 5. 生成信号
        logger.info("步骤2: 生成信号")
        self.signal = self.signal or MLSignal(
            top_n=top_n,
            model_version=model_version,
            weight_method=self.weight_method,
            verbose=False,
        )
        targets = self._generate_signals(
            corrected_date,
            universe_type=universe_type,
            top_n=top_n,
            model_version=model_version,
            buy_price_type=buy_price_type
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
            current_prices[row['ts_code']] = row.get('close', 0.0)
        
        # 生成指令（使用传入的 sell_price_type 参数）
        instructions = self._generate_instructions(
            targets=targets,
            buy_price_type=buy_price_type,
            sell_price_type=sell_price_type,
            current_prices=current_prices,
            source_date=corrected_date
        )
        
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
        
        # 保存交易指令（指令驱动模式）
        self.paper_storage.save_instructions(t1_date, instructions)
        
        # 8. 更新调仓状态
        rebalance_state = {
            'last_rebalance_date': corrected_date,
            'rebalance_freq': rebalance_freq
        }
        self.paper_storage.save_rebalance_state(rebalance_state)
        
        # 9. 保存执行记录
        run_record = {
            'trade_date': corrected_date,
            't1_date': t1_date,
            'buy_price_type': buy_price_type,
            'universe_type': universe_type,
            'top_n': top_n,
            'model_version': model_version,
            'rebalance_freq': rebalance_freq,
            'targets_count': len(targets),
            'instructions_count': len(instructions),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.paper_storage.save_run_record("t0", corrected_date, run_record)
        
        logger.info("=" * 80)
        logger.info(f"T0工作流完成 - 已生成 {len(targets)} 个目标权重和 {len(instructions)} 条交易指令")
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
    
    def _execute_pending_buys(
        self,
        pending_buys: List,
        buy_prices: Dict[str, float],
        trade_date: str,
        buy_price_type: str = 'close'
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
        from .models import Fill, Order, PendingBuy, TargetWeight
        
        MAX_REPLENISHMENT_ATTEMPTS = 5
        
        logger.info("=" * 80)
        logger.info(f"执行补位买入计划 - {trade_date}")
        logger.info(f"待处理补位: {len(pending_buys)} 个")
        logger.info("=" * 80)
        
        # 加载可交易性信息
        tradability = self.broker._load_tradability_info(trade_date)
        
        # 计算总资产
        all_prices = buy_prices
        total_value = self.account.get_total_value(all_prices)
        #算算手头还有多少现金
        import yaml
        with open("configs/base.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        total_cash = self.account.get_cash() * (1 - cfg['costs']['pendding_capital_retention_ratio'])  # 留出一定现金空间，避免过度买入导致后续无法补位
        available_cash = total_cash / len(pending_buys)  # 简单平均分配现金到每个补位目标
        
        fills = []
        updated_pending_buys = []
        failed_buy_targets = []
        
        for pending_buy in pending_buys:
            # 检查是否超过尝试次数
            if pending_buy.attempts >= MAX_REPLENISHMENT_ATTEMPTS:
                logger.warning(
                    f"补位 {pending_buy.ts_code} 已达最大尝试次数 ({MAX_REPLENISHMENT_ATTEMPTS})，放弃"
                )
                continue
            
            # 避免同日重复尝试
            if pending_buy.last_attempt_date == trade_date:
                logger.info(f"补位 {pending_buy.ts_code} 今日已尝试，跳过（避免重复）")
                updated_pending_buys.append(pending_buy)
                continue
            
            ts_code = pending_buy.ts_code
            
            # 检查是否已持仓
            if ts_code in self.account.get_positions():
                logger.info(f"补位 {ts_code} 已在持仓中，跳过")
                continue
            
            # 检查价格数据
            if ts_code not in buy_prices:
                logger.warning(f"补位 {ts_code} 无买入价格数据，记录失败并继续尝试")
                pending_buy.attempts += 1
                pending_buy.last_attempt_date = trade_date
                failed_buy_targets.append(TargetWeight(
                    ts_code=ts_code,
                    target_weight=pending_buy.target_weight,
                    reason=f"{pending_buy.reason}（无价格数据）"
                ))
                updated_pending_buys.append(pending_buy)
                continue
            
            # 检查可交易性
            can_buy, buy_reason = self.broker._check_can_buy(ts_code, tradability)
            if not can_buy:
                logger.warning(f"补位 {ts_code} 不可买入: {buy_reason}，记录失败并继续尝试")
                pending_buy.attempts += 1
                pending_buy.last_attempt_date = trade_date
                failed_buy_targets.append(TargetWeight(
                    ts_code=ts_code,
                    target_weight=pending_buy.target_weight,
                    reason=f"{pending_buy.reason}（{buy_reason}）"
                ))
                updated_pending_buys.append(pending_buy)
                continue
            
            # 使用统一的估算方法计算买入股数
            buy_shares = self._estimate_pending_buy_shares(
                ts_code=ts_code,
                price=buy_prices[ts_code],
                target_weight=pending_buy.target_weight,
                total_pending_count=len(pending_buys),
                pendding_capital_retention_ratio=cfg['costs']['pendding_capital_retention_ratio']
            )
            
            if buy_shares <= 0:
                logger.warning(f"补位 {ts_code} 不足一手，记录失败并继续尝试")
                pending_buy.attempts += 1
                pending_buy.last_attempt_date = trade_date
                failed_buy_targets.append(TargetWeight(
                    ts_code=ts_code,
                    target_weight=pending_buy.target_weight,
                    reason=f"{pending_buy.reason}（不足一手）"
                ))
                updated_pending_buys.append(pending_buy)
                continue
            
            # 创建订单
            order = Order(
                ts_code=ts_code,
                action='buy',
                shares=buy_shares,
                price=buy_prices[ts_code],
                target_weight=pending_buy.target_weight,
                current_weight=0.0,
                reason=pending_buy.reason
            )
            
            # 执行订单
            #logger.info(f"补位买入: {ts_code}, {buy_shares} 股, 目标权重 {pending_buy.target_weight:.2%}, 预估成本 {target_value:.2f}")
            fill = self.broker._execute_single_order(order, trade_date, buy_price_type)
            
            if fill:
                fills.append(fill)
                logger.info(f"补位成功: {ts_code}, 买入 {fill.shares} 股, 成交价 {fill.price:.2f}, 成交金额 {fill.shares * fill.price:.2f}")
                # 补位成功，不再保留在队列中
            else:
                logger.warning(f"补位执行失败: {ts_code}, ")
                pending_buy.attempts += 1
                pending_buy.last_attempt_date = trade_date
                updated_pending_buys.append(pending_buy)
        
        # 如果有新的失败买入，生成新的补位计划
        if failed_buy_targets:
            logger.info(f"补位执行失败 {len(failed_buy_targets)} 个，将生成新的补位计划")
            # 将失败的目标记录到broker，供后续处理
            self.broker._failed_buy_targets = failed_buy_targets
        
        # 保存更新后的补位队列
        self.paper_storage.save_pending_buys(updated_pending_buys)
        
        logger.info(f"补位买入执行完成: 成功 {len(fills)} 个，失败 {len(updated_pending_buys)} 个")
        logger.info("=" * 80)
        
        return fills
    
    def _download_data(self, trade_date: str) -> None:
        """下载并构建数据（复用仓库既有能力）
        
        Args:
            trade_date: 交易日期 YYYYMMDD
        """
        try:
            # 检查clean数据是否已存在
            if self.storage.is_data_exists("clean", "daily", trade_date):
                logger.info(f"数据已存在，跳过下载: {trade_date}")
                return
            
            # 1. 下载raw数据（复用TushareClient）
            logger.info(f"下载raw数据: {trade_date}")
            
            # 下载日线行情
            if not self.storage.is_data_exists("raw", "daily", trade_date):
                daily_data = self.client.get_daily(trade_date=trade_date)
                if not daily_data.empty:
                    self.storage.save_raw_by_date(daily_data, "daily", trade_date)
                    logger.info(f"  日线: 已保存 {len(daily_data)} 条记录")
            
            # 下载复权因子
            if not self.storage.is_data_exists("raw", "adj_factor", trade_date):
                adj_factor = self.client.get_adj_factor(trade_date=trade_date)
                if not adj_factor.empty:
                    self.storage.save_raw_by_date(adj_factor, "adj_factor", trade_date)
                    logger.info(f"  复权因子: 已保存 {len(adj_factor)} 条记录")
            
            # 下载停复牌信息
            if not self.storage.is_data_exists("raw", "suspend", trade_date):
                suspend = self.client.get_suspend_d(trade_date=trade_date)
                if not suspend.empty:
                    self.storage.save_raw_by_date(suspend, "suspend", trade_date)
                    logger.info(f"  停复牌: 已保存 {len(suspend)} 条记录")
            
            # 下载涨跌停信息
            if not self.storage.is_data_exists("raw", "stk_limit", trade_date):
                limit_up_down = self.client.get_stk_limit(trade_date=trade_date)
                if not limit_up_down.empty:
                    self.storage.save_raw_by_date(limit_up_down, "stk_limit", trade_date)
                    logger.info(f"  涨跌停: 已保存 {len(limit_up_down)} 条记录")
            
            # 2. 构建clean数据（复用DataCleaner）
            logger.info(f"构建clean数据: {trade_date}")
            from ..data.cleaner import DataCleaner
            cleaner = DataCleaner()
            
            # 加载raw数据
            daily_raw = self.storage.load_raw_by_date("daily", trade_date)
            adj_factor_raw = self.storage.load_raw_by_date("adj_factor", trade_date)
            
            if daily_raw is None or daily_raw.empty:
                logger.warning(f"未找到raw层daily数据，跳过clean构建")
                return
            
            # 处理缺失的复权因子
            if adj_factor_raw is None or adj_factor_raw.empty:
                logger.warning(f"未找到复权因子，使用默认值1.0")
                adj_factor_raw = daily_raw[['ts_code', 'trade_date']].copy()
                adj_factor_raw['adj_factor'] = 1.0
            
            # 清洗日线数据
            daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)
            
            # 添加可交易标记
            stock_basic = self.loader.load_clean_stock_basic()
            if stock_basic is None:
                # 如果没有stock_basic，尝试从raw加载并清洗
                stock_basic_raw = self.storage.load_raw("stock_basic")
                if stock_basic_raw is not None:
                    stock_basic = cleaner.clean_stock_basic(stock_basic_raw)
                    self.storage.save_clean(stock_basic, "stock_basic", is_force=True)
            
            if stock_basic is not None:
                suspend_raw = self.storage.load_raw_by_date("suspend", trade_date)
                limit_raw = self.storage.load_raw_by_date("stk_limit", trade_date)
                
                suspend_clean = None
                limit_clean = None
                
                if suspend_raw is not None and len(suspend_raw) > 0:
                    suspend_clean = cleaner.clean_suspend_info(suspend_raw)
                
                if limit_raw is not None and len(limit_raw) > 0:
                    limit_clean = cleaner.clean_limit_info(limit_raw)
                
                daily_clean = cleaner.add_tradable_universe_flag(
                    daily_clean,
                    stock_basic,
                    suspend_info_df=suspend_clean,
                    limit_info_df=limit_clean,
                    min_list_days=365
                )
            
            # 保存clean数据
            self.storage.save_clean_by_date(daily_clean, "daily", trade_date)
            logger.info(f"已保存clean数据: {len(daily_clean)} 条")
            
        except Exception as e:
            logger.error(f"下载数据失败: {e}")
            raise
    
    def _generate_signals(
        self,
        trade_date: str,
        universe_type: str = 'mainboard',
        top_n: int = 5,
        model_version: Optional[int] = None,
        buy_price_type: str = 'close'
    ) -> List[TargetWeight]:
        """生成信号
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            universe_type: 股票池类型
            top_n: 持仓股票数
            model_version: ML模型版本（可选）
            buy_price_type: T1买入价格类型 open/close（用于一手可买约束）
            
        Returns:
            目标权重列表
        """
        # 确保 features 数据存在
        logger.info(f"检查并确保 features 数据存在: {trade_date}")
        if not ensure_features_for_date(
            self.storage,
            self.loader,
            self.feature_builder,
            self.cleaner,
            self.client,
            trade_date,
            force=False
        ):
            logger.error(f"无法获取 features 数据: {trade_date}")
            return []
        
        # 加载股票池
        stock_basic = self.loader.load_clean_stock_basic()
        if stock_basic is None:
            logger.error("无法加载stock_basic数据")
            return []
        
        # 创建股票池（仅主板）
        universe = self._create_universe(stock_basic, universe_type)
        
        # 加载价格数据
        daily_data = self.loader.load_clean_daily_by_date(trade_date)
        signal_data = self.storage.load_cs_train_day(trade_date).copy()
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
                    top_n=top_n, 
                    model_version=model_version,
                    weight_method=self.weight_method,
                    verbose=False,
                )
            else:
                logger.warning("未指定信号生成器，使用等权")
                from ..signals.base import EqualWeightSignal
                self.signal = EqualWeightSignal(top_n=top_n)
        
        # 生成信号
        try:
            # 如果是等权策略且信号生成器是MLSignal，则使用generate_ranked获取排序候选并应用一手可买约束
            if self.weight_method == "equal" and isinstance(self.signal, MLSignal):
                signal_dict = self._generate_equal_weight_with_lot_constraint(
                    date_ts,
                    stocks,
                    signal_data,
                    daily_data,
                    top_n,
                    buy_price_type
                )
            else:
                # 其他情况（score加权或非MLSignal），使用原有逻辑
                signal_dict = self.signal.generate(
                    date_ts,
                    stocks,
                    {'features': signal_data}
                )
        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return []
        
        # 转换为目标权重，并增强信息
        targets = self._enhance_target_info(
            signal_dict,
            stock_basic,
            daily_data,
            trade_date
        )
        
        logger.info(f"生成 {len(targets)} 个目标权重")
        
        # 打印 T0 详细信息
        self._print_t0_targets(targets, stock_basic, daily_data)
        
        return targets
    
    def _generate_equal_weight_with_lot_constraint(
        self,
        date: pd.Timestamp,
        stocks: List[str],
        signal_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        top_n: int,
        buy_price_type: str
    ) -> Dict[str, float]:
        """等权策略下生成信号（含一手可买约束和顺延补足）
        
        对等权策略启用"一手可买约束"：如果按资金分配给某股票的金额不足以买入100股（1手），
        则跳过该股票并从候选中顺延选择下一只，直到凑足top_n个可买股票或候选耗尽。
        
        Args:
            date: 当前日期
            stocks: 股票池
            signal_data: 特征数据
            daily_data: 日线数据（包含价格）
            top_n: 目标股票数
            buy_price_type: T1买入价格类型 open/close
            
        Returns:
            信号字典 {股票代码: 权重}
        """
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
        logger.info(f"等权+一手约束: 原始排序候选数 {original_count}")
        
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
        total_capital = self.account.initial_capital
        equal_weight_value = total_capital / top_n
        
        # 从排序候选中筛选可买至少1手的股票
        selected_stocks = []
        skipped_stocks = []
        
        for ts_code, score in ranked_candidates:
            # 检查是否已凑足 top_n
            if len(selected_stocks) >= top_n:
                break
            
            # 获取价格
            price = price_map.get(ts_code)
            if price is None or price <= 0:
                # 无价格数据，跳过
                skipped_stocks.append((ts_code, "无价格数据"))
                continue
            
            # 计算可买股数（向下取整到100的倍数）
            affordable_shares = int(equal_weight_value / price / SHARE_LOT_SIZE) * SHARE_LOT_SIZE
            
            if affordable_shares < SHARE_LOT_SIZE:
                # 不足1手，跳过并记录
                skipped_stocks.append((ts_code, f"不足1手(价格={price:.2f}, 可买={affordable_shares}股)"))
                continue
            
            # 可买至少1手，加入选中列表
            selected_stocks.append(ts_code)
        
        # 日志输出
        final_count = len(selected_stocks)
        skipped_count = len(skipped_stocks)
        
        logger.info(
            f"等权+一手约束: 最终目标数 {final_count}, "
            f"跳过 {skipped_count} 只 (原始候选 {original_count})"
        )
        
        if skipped_count > 0:
            # 输出若干示例（最多5个）
            examples = skipped_stocks[:5]
            for ts_code, reason in examples:
                logger.info(f"  跳过示例: {ts_code} - {reason}")
            if skipped_count > 5:
                logger.info(f"  ... 及其他 {skipped_count - 5} 只")
        
        if final_count < top_n:
            logger.warning(
                f"等权+一手约束: 候选不足，目标 {top_n} 只，实际仅 {final_count} 只可选"
            )
        
        # 构建等权信号字典
        if final_count == 0:
            return {}
        
        weight = 1.0 / final_count
        signal_dict = {ts_code: weight for ts_code in selected_stocks}
        
        return signal_dict
    
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
                min_list_days=365,
                verbose=self.verbose,
            )
        else:
            # 默认全市场
            return BasicUniverse(
                stock_basic=stock_basic,
                exclude_st=True,
                min_list_days=365,
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
        daily_data: pd.DataFrame
    ) -> None:
        """打印 T0 目标详细信息（包含买入/减仓/清仓）
        
        输出包含：代码、名称、方向、参考价格、建议股数、原因
        
        Args:
            targets: 目标权重列表
            stock_basic: 股票基本信息
            daily_data: 日线数据
        """
        if not targets:
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
        current_positions = self.account.get_positions()
        
        # 使用账户总资金计算
        total_capital = self.account.initial_capital
        
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
        stats = {"清仓": 0, "减仓": 0, "加仓": 0, "买入": 0}

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
                direction = "清仓" if target_shares == 0 else "减仓"
                suggested_shares = (current_shares - target_shares) // SHARE_LOT_SIZE * SHARE_LOT_SIZE
            else:
                continue

            if suggested_shares <= 0:
                continue
            
            # 统计数量
            if direction in stats:
                stats[direction] += 1
            
            reason_text = reason if reason else "信号生成"
            rows_to_print.append({
                'data': [ts_code, name, direction, f"{price:.2f}", str(suggested_shares), reason_text],
                'direction': direction
            })

        # 2. 按照指定顺序排序：清仓 > 减仓 > 加仓 > 买入
        priority = {"清仓": 0, "减仓": 1, "加仓": 2, "买入": 3}
        rows_to_print.sort(key=lambda x: priority.get(x['direction'], 99))

        # 3. 打印表格行
        for item in rows_to_print:
            logger.info(format_row(item['data'], widths, aligns))
        
        # 4. 打印统计摘要
        logger.info("-" * SEPARATOR_LENGTH)
        stats_str = f"【操作统计】 清仓: {stats['清仓']} | 减仓: {stats['减仓']} | 加仓: {stats['加仓']} | 买入: {stats['买入']}"
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
        original_signal_date: str = ""
    ) -> List[TargetWeight]:
        """生成补位目标（当买入失败时使用）
        
        使用现有的信号生成链路，从候选中选择 top_k（k=失败数量）的补位股票，
        应用一手可买约束，生成新的目标权重列表。
        
        Args:
            trade_date: 当前交易日期 YYYYMMDD（用于生成信号）
            failed_count: 失败买入的数量
            universe_type: 股票池类型
            model_version: ML模型版本
            buy_price_type: 买入价格类型（用于一手约束检查）
            original_signal_date: 原始信号日期（T0日期）
            
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
        if not ensure_features_for_date(
            self.storage,
            self.loader,
            self.feature_builder,
            self.cleaner,
            self.client,
            trade_date,
            force=False
        ):
            logger.error(f"无法获取 features 数据: {trade_date}")
            return []
        
        # 2. 加载股票池
        stock_basic = self.loader.load_clean_stock_basic()
        if stock_basic is None:
            logger.error("无法加载stock_basic数据")
            return []
        
        # 创建股票池
        universe = self._create_universe(stock_basic, universe_type)
        
        # 3. 加载数据
        daily_data = self.loader.load_clean_daily_by_date(trade_date)
        signal_data = self.storage.load_cs_train_day(trade_date).copy()
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
        
        # 5. 使用信号生成器获取排序候选
        if self.signal is None:
            if model_version is not None:
                self.signal = MLSignal(
                    top_n=failed_count,
                    model_version=model_version,
                    weight_method=self.weight_method,
                    verbose=False,
                )
            else:
                logger.warning("未指定信号生成器，使用等权")
                from ..signals.base import EqualWeightSignal
                self.signal = EqualWeightSignal(top_n=failed_count)
        
        # 6. 生成排序候选（使用与T0相同的逻辑）
        try:
            if isinstance(self.signal, MLSignal):
                # 使用等权+一手约束逻辑
                signal_dict = self._generate_equal_weight_with_lot_constraint(
                    date_ts,
                    stocks,
                    signal_data,
                    daily_data,
                    failed_count,
                    buy_price_type
                )
            else:
                signal_dict = self.signal.generate(
                    date_ts,
                    stocks,
                    {'features': signal_data}
                )
        except Exception as e:
            logger.error(f"补位信号生成失败: {e}")
            return []
        
        # 7. 转换为目标权重
        targets = self._enhance_target_info(
            signal_dict,
            stock_basic,
            daily_data,
            trade_date
        )
        
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
        import yaml
        with open("configs/base.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        pendding_capital_retention_ratio = cfg['costs']['pendding_capital_retention_ratio']
        
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
