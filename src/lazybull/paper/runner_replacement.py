# -*- coding: utf-8 -*-
"""PaperReplacementMixin：src/lazybull/paper/runner.py 拆分出的 _enhance_target_info, _print_t0_targets, generate_replacement_targets, _print_replacement_targets。"""

from ..common.constants import SEPARATOR_LENGTH, SHARE_LOT_SIZE
from ..common.print_table import format_row
from ..common.trading_config import TradingConfig
from ..features import ensure_features_for_date
from ..portfolio.industry_constraint import load_industry_mapping
from ..signals.base import EqualWeightSignal
from ..signals.ml_signal import MLSignal
from ..trading.sizing import compute_lot_shares
from .models import TargetWeight
from dataclasses import replace
from loguru import logger
from typing import Dict
from typing import List
from typing import Optional
import pandas as pd

class PaperReplacementMixin:
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
            target_shares = compute_lot_shares(target_value, price, SHARE_LOT_SIZE)
            
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
        
        # 组合总资产（与 _execute_pending_buys 实际执行口径一致）
        current_total_value = self.account.get_total_value(price_map)
        if current_total_value <= 0:
            current_total_value = float(getattr(self.account, "initial_capital", 0.0) or 0.0)

        # 打印表头
        logger.info("=" * 120)
        logger.info("补位买入目标详情（需要在下一交易日继续买入）")
        logger.info("=" * 120)
        logger.info("注意：以下股数为估算值，基于当前价格与组合总资产（与实际执行口径一致）")
        logger.info("实际执行时会受到执行日价格变化、补位队列长度变化等因素影响，但计算规则一致")
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
            
            # 使用与实际执行一致的组合价值口径估算建议股数
            if price > 0:
                suggested_shares, _ = self._analyze_pending_buy_shares_backtest_style(
                    ts_code=target.ts_code,
                    price=price,
                    target_weight=target.target_weight,
                    current_total_value=current_total_value,
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
