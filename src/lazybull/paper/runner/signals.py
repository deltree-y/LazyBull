# -*- coding: utf-8 -*-
"""PaperSignalMixin：src/lazybull/paper/runner.py 拆分出的 _generate_signals, _generate_ranked_with_lot_constraint, _normalize_signals, _create_universe。"""

from ...common.constants import SHARE_LOT_SIZE
from ...common.trading_config import TradingConfig
from ...features import ensure_features_for_date
from ...portfolio import cap_and_normalize_weights, resolve_tranche_weight_cap
from ...portfolio.industry_constraint import apply_industry_constraint
from ...portfolio.industry_constraint import load_industry_mapping
from ...signals.base import EqualWeightSignal
from ...signals.ml_signal import MLSignal
from ...trading.sizing import compute_lot_shares
from ...trading.stagger import get_tranche_capital_fraction
from ...universe.base import BasicUniverse
from ..models import TargetWeight
from dataclasses import replace
from loguru import logger
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import gc
import pandas as pd

class PaperSignalMixin:
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
        excluded_stocks: Optional[set] = None,
        trading_config: Optional[TradingConfig] = None,
        tranche_idx: int = 0,
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
        portfolio_top_n = trading_config.top_n if trading_config is not None else top_n
        if trading_config is not None:
            # 纸面交易分批模式下，top_n 必须以调用方传入值为准（本批槽位数），
            # 不能被总配置中的 top_n 覆盖，否则会退化为首批一次买满总仓位。
            effective_config = replace(
                trading_config,
                buy_price=buy_price_type,
                universe=universe_type,
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
                max_weight_per_stock=(
                    max_weight_per_stock
                    if max_weight_per_stock is not None
                    else trading_config.max_weight_per_stock
                ),
                exclude_st=exclude_st,
                min_list_days=min_list_days,
                position_sizing=self.position_sizing,
            )
        else:
            effective_config = TradingConfig(
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
                from ...signals.base import EqualWeightSignal
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
        
        selection_exclusions = set(self.account.get_positions().keys())
        selection_exclusions.update(excluded_stocks or set())

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
                    top_n,
                    buy_price_type,
                    max_per_industry=effective_config.max_per_industry,
                    industry_mapping=industry_mapping,
                    trading_config=effective_config,
                    existing_positions=selection_exclusions,
                    portfolio_top_n=portfolio_top_n,
                    return_meta=True,
                )
            else:
                raw_scores = self.signal.generate(
                    date_ts,
                    stocks,
                    {'features': signal_data}
                )
                raw_scores = {
                    ts_code: score
                    for ts_code, score in raw_scores.items()
                    if ts_code not in selection_exclusions
                }
                signal_meta = {}

            signal_dict = self._normalize_signals(raw_scores, trade_date)

            # 全组合上限按本批预算比例换算为批内归一化权重上限。
            if effective_config.max_weight_per_stock is not None and signal_dict:
                tranche_capital_fraction = get_tranche_capital_fraction(
                    tranche_idx,
                    portfolio_top_n,
                    effective_config.stagger_tranches,
                )
                tranche_weight_cap = resolve_tranche_weight_cap(
                    effective_config.max_weight_per_stock,
                    tranche_capital_fraction,
                )
                signal_dict = cap_and_normalize_weights(
                    signal_dict,
                    max_weight_per_stock=tranche_weight_cap,
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
        portfolio_top_n: Optional[int] = None,
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

        # 已持仓和同一执行日已预留买单必须先排除，避免占用候选截断名额。
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

        # 行业约束按最终组合计数，而不是每个批次从零开始。
        if max_per_industry and max_per_industry > 0 and industry_mapping:
            initial_industry_counts: Dict[str, int] = {}
            for ts_code in existing_positions:
                industry = industry_mapping.get(ts_code, "未知行业")
                initial_industry_counts[industry] = initial_industry_counts.get(industry, 0) + 1
            ranked_candidates = apply_industry_constraint(
                ranked_candidates=ranked_candidates,
                industry_mapping=industry_mapping,
                max_per_industry=max_per_industry,
                target_n=top_n * 3,
                verbose=True,
                initial_industry_counts=initial_industry_counts,
            )
            logger.info(f"行业约束后候选数: {len(ranked_candidates)} (原始 {original_count})")

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
        capital_retention_ratio = self._get_cost_setting("capital_retention_ratio", 0.0)
        total_capital *= 1 - capital_retention_ratio
        if portfolio_top_n is None and trading_config is not None:
            portfolio_top_n = trading_config.top_n
        equal_weight_value = total_capital / max(portfolio_top_n or target_n, 1)
        
        # 从排序候选中筛选可买至少1手的股票，保留原始分数
        selected = []  # List[Tuple[str, float]]
        skipped_stocks = []

        for ts_code, score in ranked_candidates:
            if len(selected) >= target_n:
                break

            price = price_map.get(ts_code)
            if price is None or price <= 0:
                skipped_stocks.append((ts_code, "无价格数据"))
                continue

            affordable_shares = compute_lot_shares(
                equal_weight_value, price, SHARE_LOT_SIZE
            )
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
