"""特征与标签构建模块

实现按日截面特征构建，包括：
- 后复权收盘价计算
- 未来5日收益标签 (horizon=5)
- 基础数值特征
- 股票池过滤（ST、上市<60天、停牌）
- 涨跌停标记
- 技术指标因子（RSI、KDJ、MACD、布林带等）
- K线形态因子（振幅、上下影线等）
- 行业相关因子（alpha、偏离等）
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_date_columns, to_trade_date_str
from ..factors import (
    calculate_rsi,
    calculate_kdj,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_amplitude,
    calculate_shadows,
    calculate_volatility,
    add_industry_features,
    calculate_industry_alpha_windows,
    calculate_acceleration,
    calculate_volume_burst,
    compute_market_state_features,
    precompute_market_state_features,
    precompute_technical_factors,
)
from ..factors.normalization import cross_sectional_zscore

# 预计算 warmup 天数：固定 120 个交易日，确保 rolling/EWM 指标在历史充足时与 --start-date 无关
_WARMUP_TRADING_DAYS = 120


class FeatureBuilder:
    """特征构建器
    
    负责生成单日全市场截面训练数据，包含特征和标签
    """
    
    def __init__(
        self,
        min_list_days: int = 365,
        horizon: int = 5,
        horizons: List[int] = None,
        lookback_windows: List[int] = None,
        require_label: bool = True,
        verbose: bool = False,
    ):
        """初始化特征构建器

        Args:
            min_list_days: 最小上市自然日天数，默认365天（约12个月）
            horizon: 预测时间窗口（交易日），默认5天（自 0.5.0 版本起已废弃，请使用 horizons 参数）
            horizons: 预测时间窗口列表（交易日），默认[5, 10, 20]，生成多个标签 y_ret_5, y_ret_10, y_ret_20
            lookback_windows: 回看窗口列表，用于计算历史特征，默认[5, 10, 20]
            require_label: 是否要求标签非空，默认True（训练/回测模式）；设为False用于实盘/推理模式
            verbose: 是否输出详细日志
        """
        self.min_list_days = min_list_days
        # 如果 horizons 未指定，则使用旧参数 horizon 或默认值
        self.horizons = horizons or [5, 10, 20]
        # 保留旧参数向后兼容
        self.horizon = horizon if horizon in self.horizons else self.horizons[0]
        self.lookback_windows = lookback_windows or [5, 10, 20]
        self.require_label = require_label
        self.verbose = verbose
        # 实例级缓存：批量预计算的市场状态特征（首次调用时触发，后续 O(1) 取值）
        self._market_state_cache: Optional[pd.DataFrame] = None
        # 实例级缓存：批量预计算的技术指标与波动率因子（首次调用时触发，后续 O(1) 查表）
        self._tech_factor_cache: Optional[pd.DataFrame] = None
        # 优化3：技术指标按 trade_date 的字典索引，O(1) 取代 DataFrame 全量过滤
        self._tech_factor_cache_dict: Optional[Dict[str, pd.DataFrame]] = None
        # 优化2：交易日列表缓存 + O(1) 索引字典（_get_trading_dates 首次调用时填充）
        self._trading_dates_cache: Optional[List[str]] = None
        self._trading_date_index: Optional[Dict[str, int]] = None
        # 优化1：预计算的全量 daily_adj（含 pre_close_adj），循环外调用 precompute_daily_adj 填充
        self._daily_adj_precomputed: Optional[pd.DataFrame] = None
        # 优化4：daily_adj 按 trade_date 的字典索引，O(1) 取代 isin 全量扫描
        self._daily_adj_dict: Optional[Dict[str, pd.DataFrame]] = None
        
        if self.verbose:
            logger.info(
                f"特征构建器初始化: min_list_days={min_list_days}, "
                f"horizons={self.horizons}, lookback_windows={self.lookback_windows}, "
                f"require_label={require_label}"
            )
    
    def clear_caches(self) -> None:
        """释放所有内部缓存，降低内存占用

        在特征构建完成并保存后调用，适用于内存受限环境（如树莓派）。
        缓存会在下次 build_features_for_day 调用时按需重建。
        """
        cache_names = [
            '_market_state_cache', '_tech_factor_cache',
            '_tech_factor_cache_dict', '_trading_dates_cache',
            '_trading_date_index', '_daily_adj_precomputed',
            '_daily_adj_dict',
        ]
        cleared = []
        for name in cache_names:
            if getattr(self, name, None) is not None:
                setattr(self, name, None)
                cleared.append(name)
        if cleared:
            logger.debug(f"FeatureBuilder 缓存已释放: {', '.join(cleared)}")

    def precompute_daily_adj(self, daily_data: pd.DataFrame, adj_factor: pd.DataFrame) -> None:
        """预计算全量 daily_adj 并建立按交易日索引的字典（循环外调用一次）

        将原本在每次 build_features_for_day 内执行的全量 copy / sort / groupby.shift
        提前一次性完成，避免对 ~千万行 DataFrame 做 2000 次重复操作。

        Args:
            daily_data: 全量日线数据（clean 层，已含 close_adj 等复权价）
            adj_factor: 复权因子（clean 层场景下传入空 DataFrame 即可）
        """
        logger.info("预计算 daily_adj（含 pre_close_adj）并建立日期索引字典...")
        daily_adj = self._calculate_adj_close(daily_data, adj_factor)
        daily_adj = daily_adj.sort_values(['ts_code', 'trade_date'])
        daily_adj['pre_close_adj'] = daily_adj.groupby('ts_code')['close_adj'].shift(1)
        self._daily_adj_precomputed = daily_adj
        # 按 trade_date 分组建立字典，后续 O(1) 切片（各 sub_df 已 reset_index 成独立副本）
        self._daily_adj_dict = {
            d: sub_df.reset_index(drop=True)
            for d, sub_df in daily_adj.groupby('trade_date', sort=False)
        }
        logger.info(
            f"daily_adj 预计算完成：{len(daily_adj)} 条记录，"
            f"{len(self._daily_adj_dict)} 个交易日"
        )

    def build_features_for_day(
        self,
        trade_date: str,
        trade_cal: pd.DataFrame,
        daily_data: pd.DataFrame,
        adj_factor: pd.DataFrame,
        stock_basic: pd.DataFrame,
        daily_basic_data: Optional[pd.DataFrame] = None,
        moneyflow_data: Optional[pd.DataFrame] = None,
        suspend_info: Optional[pd.DataFrame] = None,
        limit_info: Optional[pd.DataFrame] = None,
        shenwan_industry: Optional[pd.DataFrame] = None,
        apply_industry_neutralization: bool = False,
        fundamental_data: Optional[pd.DataFrame] = None,
        margin_data: Optional[pd.DataFrame] = None,
        holder_data: Optional[pd.DataFrame] = None,
        earnings_data: Optional[pd.DataFrame] = None,
        cyq_perf_data: Optional[pd.DataFrame] = None,
        express_data: Optional[pd.DataFrame] = None,
        fund_portfolio_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """构建单个交易日的截面特征和标签

        Args:
            trade_date: 目标交易日，格式YYYYMMDD
            trade_cal: 交易日历DataFrame，需包含 cal_date, is_open
            daily_data: 日线行情DataFrame，需包含 ts_code, trade_date, close, pre_close,
                       pct_chg, vol, amount 等字段
            adj_factor: 复权因子DataFrame，需包含 ts_code, trade_date, adj_factor
            stock_basic: 股票基本信息DataFrame，需包含 ts_code, name, list_date
            daily_basic_data: 每日指标DataFrame（可选），包含 pb, pe_ttm, ps_ttm, dv_ttm, total_mv, circ_mv 等
            moneyflow_data: 资金流向DataFrame（可选），包含净流入、大单特大单净流入等
            suspend_info: 停复牌信息DataFrame（可选）
            limit_info: 涨跌停价格DataFrame（可选）
            shenwan_industry: 申万行业分类DataFrame（可选）
            apply_industry_neutralization: 是否应用行业中性化，默认False
            fundamental_data: 基本面因子（可选），当日已前向填充的数据
            margin_data: 融资融券因子（可选），当日数据
            holder_data: 股东人数因子（可选），当日数据
            earnings_data: 业绩预告/快报因子（可选），当日数据
            cyq_perf_data: 筹码胜率因子（可选），当日数据
            express_data: 业绩快报因子（可选），当日数据
            fund_portfolio_data: 基金持仓因子（可选），当日数据

        Returns:
            特征DataFrame，包含 trade_date, ts_code, 特征列, 标签列, 标记列
        """
        logger.info(f"开始构建 {trade_date} 的特征")
        
        # 1. 获取交易日序列（有缓存则 O(1) 返回）
        trading_dates = self._get_trading_dates(trade_cal)

        # 优化2：O(1) dict 查找替代 list.index()
        if self._trading_date_index is not None:
            current_idx = self._trading_date_index.get(trade_date, -1)
            if current_idx == -1:
                logger.warning(f"{trade_date} 不是交易日，跳过")
                return pd.DataFrame()
        else:
            if trade_date not in trading_dates:
                logger.warning(f"{trade_date} 不是交易日，跳过")
                return pd.DataFrame()
            current_idx = trading_dates.index(trade_date)

        # 2. 优化1：使用循环外预计算的 daily_adj，避免每日重复 copy/sort/groupby.shift
        if self._daily_adj_precomputed is not None:
            daily_adj = self._daily_adj_precomputed
        else:
            daily_adj = self._calculate_adj_close(daily_data, adj_factor)
            daily_adj = daily_adj.sort_values(['ts_code', 'trade_date'])
            daily_adj['pre_close_adj'] = daily_adj.groupby('ts_code')['close_adj'].shift(1)

        # 3. 获取当日数据
        current_data = daily_adj[daily_adj['trade_date'] == trade_date].copy()
        
        if len(current_data) == 0:
            logger.warning(f"{trade_date} 没有行情数据")
            return pd.DataFrame()

        # 4. 计算标签：未来N日收益（多个 horizon）
        labels = self._calculate_forward_returns(
            current_data,
            daily_adj,
            trade_date,
            trading_dates,
            current_idx
        )
        
        # 5. 计算特征：基于历史数据
        features = self._calculate_features(
            current_data,
            daily_adj,
            trade_date,
            trading_dates,
            current_idx,
            daily_basic_data,
            moneyflow_data,
            fundamental_data,
            margin_data,
            holder_data,
            earnings_data,
            cyq_perf_data,
            express_data,
            fund_portfolio_data,
        )
        logger.debug(f"{trade_date} 基础特征计算完成: {len(features.columns.tolist())} 列")

        # 6.5 合并申万行业分类
        if shenwan_industry is not None:
            features = self._merge_shenwan_industry(features, shenwan_industry)
            logger.debug(f"{trade_date} 合并申万行业分类完成: {len(features.columns.tolist())} 列")

        # 5.5 添加新增因子（技术指标、K线形态、波动率、行业等）
        features = self._add_advanced_factors(
            features,
            current_data,
            daily_adj,
            trade_date,
            trading_dates,
            current_idx,
            stock_basic
        )
        logger.debug(f"{trade_date} 高级特征计算完成: {len(features.columns.tolist())} 列")
        
        # 6. 合并特征和标签
        result = features.merge(labels, on=['trade_date', 'ts_code'], how='inner')
        logger.debug(f"{trade_date} 合并特征和标签完成: {len(result.columns.tolist())} 列")
        
        # 7. 添加过滤标记
        result = self._add_filter_flags(
            result,
            stock_basic,
            suspend_info,
            trade_date
        )
        logger.debug(f"{trade_date} 添加过滤标记完成: {len(result.columns.tolist())} 列")
        
        # 8. 添加涨跌停标记
        result = self._add_limit_flags(
            result,
            daily_data,
            limit_info,
            trade_date
        )
        logger.debug(f"{trade_date} 添加涨跌停标记完成: {len(result.columns.tolist())} 列")
        
        # 9. 应用过滤规则
        result = self._apply_filters(result)
        logger.debug(f"{trade_date} 应用过滤规则完成: {len(result)} 个样本")
        
        # 10. 应用行业中性化（如果启用）
        if apply_industry_neutralization and shenwan_industry is not None:
            result = self._apply_industry_neutralization(result)
            logger.debug(f"{trade_date} 行业中性化完成: {len(result.columns.tolist())} 列")
        
        # 11. 添加新增个股特征
        result = self._add_new_individual_features(result)
        logger.debug(f"{trade_date} 新增个股特征完成: {len(result.columns.tolist())} 列")

        # 12. 添加市场状态特征
        result = self._add_market_state_features(
            result, daily_adj, trade_date, trading_dates, current_idx, daily_basic_data
        )
        logger.debug(f"{trade_date} 市场状态特征完成: {len(result.columns.tolist())} 列")

        logger.info(f"{trade_date} 特征构建完成: {len(result)} 个样本")
        
        return result
    
    def _get_trading_dates(self, trade_cal: pd.DataFrame) -> List[str]:
        """从交易日历提取交易日列表（结果缓存，只计算一次）

        首次调用时计算并缓存结果及 O(1) 索引字典；后续调用直接返回缓存。

        Args:
            trade_cal: 交易日历DataFrame

        Returns:
            交易日列表（格式YYYYMMDD，排序）
        """
        if self._trading_dates_cache is not None:
            return self._trading_dates_cache

        if 'cal_date' in trade_cal.columns:
            # 如果是datetime格式，转换为字符串
            if pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
                trade_cal = trade_cal.copy()
                trade_cal['cal_date'] = trade_cal['cal_date'].dt.strftime('%Y%m%d')

            trading_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
        else:
            logger.error("交易日历缺少 cal_date 字段")
            return []

        self._trading_dates_cache = sorted(trading_dates)
        # 优化2：同时建立 O(1) 索引字典，替代后续所有 list.index() 调用
        self._trading_date_index = {d: i for i, d in enumerate(self._trading_dates_cache)}
        return self._trading_dates_cache

    def _get_lookback_dates(self, trade_date: str, n: int, trading_dates: List[str]) -> List[str]:
        """从全量交易日序列中，以 trade_date 为锚点向前回溯恰好 n 个交易日

        以 trade_date 在全量交易日序列中的位置为锚点，向前回溯 n 个交易日，
        确保窗口日期只由全量 trade_cal 决定，与构建脚本的 start/end 范围无关。

        Args:
            trade_date: 目标交易日（YYYYMMDD）
            n: 回溯交易日数
            trading_dates: 全量交易日序列（已排序，不重复）

        Returns:
            前 n 个交易日列表（不含 trade_date 本身）；历史不足时返回空列表
        """
        # 优化2：优先使用 O(1) 字典查找，回退到 O(n) list.index()
        if self._trading_date_index is not None:
            idx = self._trading_date_index.get(trade_date, -1)
            if idx == -1:
                return []
        else:
            if trade_date not in trading_dates:
                return []
            idx = trading_dates.index(trade_date)
        if idx < n:
            # 历史不足 n 个交易日
            return []
        return trading_dates[idx - n: idx]

    def _slice_by_trading_days(
        self,
        daily_df: pd.DataFrame,
        trading_dates: List[str],
        anchor_trade_date: str,
        warmup_days: int = _WARMUP_TRADING_DAYS,
    ) -> pd.DataFrame:
        """按交易日历回溯 warmup_days 个交易日切片数据

        以 anchor_trade_date 在全量 trading_dates 中的位置为锚点，向前回溯
        warmup_days 个交易日，返回该起点（含）之后属于全量交易日历的所有数据。

        通过确保两次构建的输入起点相同（无论 --start-date 如何），消除
        rolling/EWM 指标因历史截断点不同而产生的差异。

        Args:
            daily_df: 包含 trade_date 列的 DataFrame
            trading_dates: 已排序的全量交易日列表（YYYYMMDD 格式）
            anchor_trade_date: 锚点日期（首次处理的 trade_date），切片从该日
                               往前 warmup_days 个交易日处开始
            warmup_days: warmup 天数，默认 _WARMUP_TRADING_DAYS

        Returns:
            切片后的 DataFrame，仅保留 warmup 起始日（含）之后的交易日数据。
            若 anchor_trade_date 不在 trading_dates 中，则原样返回 daily_df。
        """
        if daily_df is None or len(daily_df) == 0:
            return daily_df
        if anchor_trade_date not in trading_dates:
            return daily_df

        anchor_idx = trading_dates.index(anchor_trade_date)
        warmup_start_idx = max(0, anchor_idx - warmup_days)
        # 过滤到 warmup 起始日（含）之后的所有交易日
        window_dates = set(trading_dates[warmup_start_idx:])
        return daily_df[daily_df['trade_date'].isin(window_dates)]

    def _calculate_adj_close(
        self,
        daily_data: pd.DataFrame,
        adj_factor: pd.DataFrame
    ) -> pd.DataFrame:
        """计算后复权收盘价和OHLC
        
        Args:
            daily_data: 日线行情DataFrame
            adj_factor: 复权因子DataFrame
            
        Returns:
            添加了 close_adj, open_adj, high_adj, low_adj 列的DataFrame
        """
        # 准备数据副本
        daily_adj = daily_data.copy()
        
        # 检查是否已经包含复权价格（clean 层数据）
        if 'close_adj' in daily_adj.columns:
            logger.info("数据已包含复权价格列，跳过复权计算")
            return daily_adj
        
        # 确保日期格式一致
        if pd.api.types.is_datetime64_any_dtype(daily_adj['trade_date']):
            daily_adj['trade_date'] = daily_adj['trade_date'].dt.strftime('%Y%m%d')
        
        if pd.api.types.is_datetime64_any_dtype(adj_factor['trade_date']):
            adj_factor = adj_factor.copy()
            adj_factor['trade_date'] = adj_factor['trade_date'].dt.strftime('%Y%m%d')
        
        # 合并复权因子
        daily_adj = daily_adj.merge(
            adj_factor[['ts_code', 'trade_date', 'adj_factor']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        # 计算后复权收盘价: close_adj = close * adj_factor
        daily_adj['close_adj'] = daily_adj['close'] * daily_adj['adj_factor']
        
        # 计算其他复权价格（如果存在）
        if 'open' in daily_adj.columns:
            daily_adj['open_adj'] = daily_adj['open'] * daily_adj['adj_factor']
        
        if 'high' in daily_adj.columns:
            daily_adj['high_adj'] = daily_adj['high'] * daily_adj['adj_factor']
        
        if 'low' in daily_adj.columns:
            daily_adj['low_adj'] = daily_adj['low'] * daily_adj['adj_factor']
        
        # 处理缺失的复权因子（如果有）
        missing_adj = daily_adj['adj_factor'].isna().sum()
        if missing_adj > 0:
            # 统计缺失复权因子的股票，便于区分"新上市（正常）"与"数据缺失（异常）"
            missing_codes = daily_adj.loc[daily_adj['adj_factor'].isna(), 'ts_code'].unique()
            logger.warning(
                f"有 {missing_adj} 条记录缺少复权因子（涉及 {len(missing_codes)} 只股票），"
                f"将使用原始价格（新上市股票属正常，除权日缺失则为数据问题）。"
                f"股票列表: {list(missing_codes[:10])}{'...' if len(missing_codes) > 10 else ''}"
            )
            daily_adj['close_adj'].fillna(daily_adj['close'], inplace=True)
            if 'open_adj' in daily_adj.columns:
                daily_adj['open_adj'].fillna(daily_adj['open'], inplace=True)
            if 'high_adj' in daily_adj.columns:
                daily_adj['high_adj'].fillna(daily_adj['high'], inplace=True)
            if 'low_adj' in daily_adj.columns:
                daily_adj['low_adj'].fillna(daily_adj['low'], inplace=True)
        
        return daily_adj
    
    def _calculate_forward_returns(
        self,
        current_data: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int
    ) -> pd.DataFrame:
        """计算未来N日收益标签（支持多个 horizon）
        
        Args:
            current_data: 当日数据
            daily_adj: 全部日线数据（含后复权价）
            trade_date: 当前交易日
            trading_dates: 交易日序列
            current_idx: 当前交易日在序列中的索引
            
        Returns:
            包含多个标签的DataFrame（y_ret_5, y_ret_10, y_ret_20等）
        """
        # 初始化结果DataFrame
        result = current_data[['trade_date', 'ts_code', 'close_adj']].copy()
        
        # 为每个 horizon 计算标签
        for horizon in self.horizons:
            label_col = f'y_ret_{horizon}'
            
            # 检查是否有足够的未来交易日
            if current_idx + horizon >= len(trading_dates):
                logger.warning(f"{trade_date} 后续交易日不足 {horizon} 天，{label_col} 标签为空")
                result[label_col] = np.nan
                continue
            
            # 获取未来第N个交易日
            future_date = trading_dates[current_idx + horizon]
            
            # 获取未来收盘价（优化4：优先 O(1) 字典取值，否则全量过滤）
            if self._daily_adj_dict is not None:
                _future_sub = self._daily_adj_dict.get(future_date)
                if _future_sub is not None:
                    future_data = _future_sub[['ts_code', 'close_adj']].copy()
                else:
                    future_data = pd.DataFrame(columns=['ts_code', 'close_adj'])
            else:
                future_data = daily_adj[daily_adj['trade_date'] == future_date][
                    ['ts_code', 'close_adj']
                ].copy()
            future_data.rename(columns={'close_adj': f'close_adj_future_{horizon}'}, inplace=True)
            
            # 合并当前和未来数据
            result = result.merge(
                future_data,
                on='ts_code',
                how='left'
            )
            
            # 计算收益率: (close_adj_future / close_adj) - 1
            # 添加除零保护：过滤掉收盘价为0或极小的样本
            valid_mask = result['close_adj'] > 1e-6
            future_col = f'close_adj_future_{horizon}'
            result.loc[valid_mask, label_col] = (
                result.loc[valid_mask, future_col] / result.loc[valid_mask, 'close_adj']
            ) - 1
            result.loc[~valid_mask, label_col] = np.nan
            
            # 删除中间列
            result.drop(columns=[future_col], inplace=True)
            
            # 记录缺失标签的样本数
            missing_labels = result[label_col].isna().sum()
            if missing_labels > 0:
                logger.warning(
                    f"{trade_date} 有 {missing_labels} 个样本缺失 {label_col}（未来{horizon}日收盘价缺失）"
                )
        
        # 删除 close_adj 列
        result.drop(columns=['close_adj'], inplace=True)
        
        return result
    
    def _calculate_features(
        self,
        current_data: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int,
        daily_basic_data: Optional[pd.DataFrame] = None,
        moneyflow_data: Optional[pd.DataFrame] = None,
        fundamental_data: Optional[pd.DataFrame] = None,
        margin_data: Optional[pd.DataFrame] = None,
        holder_data: Optional[pd.DataFrame] = None,
        earnings_data: Optional[pd.DataFrame] = None,
        cyq_perf_data: Optional[pd.DataFrame] = None,
        express_data: Optional[pd.DataFrame] = None,
        fund_portfolio_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """计算基础数值特征

        特征包括：
        - ret_1: 当日收益率
        - ret_N: 过去N日累计收益
        - vol_ratio_N: 过去N日平均成交量比
        - amount_ma_N: 过去N日平均成交额（保留）
        - ma_deviation_N: 收盘价与N日均线的偏离度
        - 价值红利因子：pb, pe_ttm, ep_ttm, bp, ps_ttm, dv_ttm, total_mv, circ_mv, log_total_mv等
        - 资金流因子：净流入、大单特大单净流入等
        - 基本面因子：roe_waa, or_yoy, netprofit_yoy 等（可选）
        - 另类因子：融资融券、股东人数、业绩预告/快报、人气榜（可选）

        注意：已删除 amount_ratio_N 和 vol_ma_N 特征
        
        Args:
            current_data: 当日数据
            daily_adj: 全部日线数据
            trade_date: 当前交易日
            trading_dates: 交易日序列
            current_idx: 当前交易日索引
            daily_basic_data: 每日指标数据（可选）
            moneyflow_data: 资金流向数据（可选）
            
        Returns:
            包含特征的DataFrame
        """
        # 初始化特征DataFrame，包含vol和amount用于后续过滤
        # 同时检查并保留 clean 层的标记列（如果存在）
        base_columns = ['trade_date', 'ts_code', 'vol', 'amount']
        clean_marker_columns = ['is_st', 'is_suspended', 'is_limit_up', 'is_limit_down', 'list_days', 'tradable']
        
        # 保留存在的 clean 层标记列
        columns_to_keep = base_columns.copy()
        for col in clean_marker_columns:
            if col in current_data.columns:
                columns_to_keep.append(col)
        
        features = current_data[columns_to_keep].copy()
        
        # 当日收益率（已在数据中）
        features = features.merge(
            current_data[['ts_code', 'pct_chg']],
            on='ts_code',
            how='left',
            suffixes=('', '_dup')
        )
        features.rename(columns={'pct_chg': 'ret_1'}, inplace=True)
        features['ret_1'] = features['ret_1'] / 100.0  # 转换为小数

        # 开盘强度（隔夜情绪代理）: open / pre_close - 1
        if 'open' in current_data.columns and 'pre_close' in current_data.columns:
            _open = current_data[['ts_code', 'open', 'pre_close']].copy()
            _open['opening_strength'] = np.where(
                _open['pre_close'] > 1e-6,
                _open['open'] / _open['pre_close'] - 1,
                np.nan,
            )
            features = features.merge(
                _open[['ts_code', 'opening_strength']], on='ts_code', how='left'
            )

        # 日内波动结构（多空力量对比）: (high - open) / (open - low)
        if all(c in current_data.columns for c in ['high', 'open', 'low']):
            _hloc = current_data[['ts_code', 'high', 'open', 'low']].copy()
            _up = _hloc['high'] - _hloc['open']
            _down = _hloc['open'] - _hloc['low']
            _hloc['intraday_vol_structure'] = np.where(
                _down > 1e-6, _up / _down, np.nan,
            )
            features = features.merge(
                _hloc[['ts_code', 'intraday_vol_structure']], on='ts_code', how='left'
            )

        # 计算回看特征
        for window in self.lookback_windows:
            # 以全量交易日历为锚点，向前回溯恰好 window 个交易日
            # 确保窗口日期只由 trade_cal 决定，与构建脚本的 start_date 无关
            hist_dates = self._get_lookback_dates(trade_date, window, trading_dates)
            if not hist_dates:
                # 历史不足 window 个交易日，填充空值
                features[f'ret_{window}'] = np.nan
                features[f'vol_ratio_{window}'] = np.nan
                features[f'ma_deviation_{window}'] = np.nan
                continue
            
            # 获取历史数据（优化4：优先按 trade_date 字典拼接，避免 isin 全量扫描）
            if self._daily_adj_dict is not None:
                _frames = [self._daily_adj_dict[d] for d in hist_dates if d in self._daily_adj_dict]
                hist_data = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
            else:
                hist_data = daily_adj[
                    daily_adj['trade_date'].isin(hist_dates)
                ]
            
            # 按股票分组计算特征
            hist_features = self._calculate_window_features(
                hist_data,
                current_data,
                window
            )
            
            # 合并特征
            features = features.merge(hist_features, on='ts_code', how='left')
        
        # 添加价值红利因子（从 daily_basic）
        if daily_basic_data is not None and len(daily_basic_data) > 0:
            features = self._add_value_dividend_features(features, daily_basic_data, trade_date)
        
        # 添加资金流因子（从 moneyflow）
        if moneyflow_data is not None and len(moneyflow_data) > 0:
            features = self._add_moneyflow_features(features, moneyflow_data, trade_date, trading_dates, current_idx)

        # 添加基本面因子（从 fina_indicator 季度数据前向填充）
        if fundamental_data is not None and len(fundamental_data) > 0:
            features = self._add_fundamental_features(features, fundamental_data, trade_date)

        # ── 另类数据因子 ──────────────────────────────────────────
        # 融资融券
        if margin_data is not None and len(margin_data) > 0:
            merge_cols = [c for c in margin_data.columns if c != "ts_code"]
            features = features.merge(
                margin_data[["ts_code"] + merge_cols], on="ts_code", how="left"
            )

        # 股东人数
        if holder_data is not None and len(holder_data) > 0:
            merge_cols = [c for c in holder_data.columns if c != "ts_code"]
            features = features.merge(
                holder_data[["ts_code"] + merge_cols], on="ts_code", how="left"
            )

        # 业绩预告/快报
        if earnings_data is not None and len(earnings_data) > 0:
            merge_cols = [c for c in earnings_data.columns if c != "ts_code"]
            features = features.merge(
                earnings_data[["ts_code"] + merge_cols], on="ts_code", how="left"
            )

        # 筹码胜率
        if cyq_perf_data is not None and len(cyq_perf_data) > 0:
            # weight_avg_bias = (close - weight_avg) / weight_avg
            if "weight_avg" in cyq_perf_data.columns and "close_adj" in features.columns:
                cyq = cyq_perf_data.copy()
                # 先合并获取 close_adj，再计算偏离度
                cyq_with_close = cyq.merge(
                    features[["ts_code", "close_adj"]], on="ts_code", how="left"
                )
                cyq_with_close["weight_avg_bias"] = np.where(
                    cyq_with_close["weight_avg"] > 1e-6,
                    (cyq_with_close["close_adj"] - cyq_with_close["weight_avg"])
                    / cyq_with_close["weight_avg"],
                    np.nan,
                )
                from ..factors.cyq_perf import CYQ_PERF_COLS
                merge_cols = [c for c in CYQ_PERF_COLS if c in cyq_with_close.columns]
                features = features.merge(
                    cyq_with_close[["ts_code"] + merge_cols], on="ts_code", how="left"
                )
            else:
                merge_cols = [c for c in cyq_perf_data.columns
                              if c != "ts_code" and c != "weight_avg"]
                features = features.merge(
                    cyq_perf_data[["ts_code"] + merge_cols], on="ts_code", how="left"
                )

        # 业绩快报
        if express_data is not None and len(express_data) > 0:
            merge_cols = [c for c in express_data.columns if c != "ts_code"]
            features = features.merge(
                express_data[["ts_code"] + merge_cols], on="ts_code", how="left"
            )

        # 基金持仓
        if fund_portfolio_data is not None and len(fund_portfolio_data) > 0:
            merge_cols = [c for c in fund_portfolio_data.columns if c != "ts_code"]
            features = features.merge(
                fund_portfolio_data[["ts_code"] + merge_cols], on="ts_code", how="left"
            )

        return features
    
    def _calculate_window_features(
        self,
        hist_data: pd.DataFrame,
        current_data: pd.DataFrame,
        window: int
    ) -> pd.DataFrame:
        """计算单个窗口的特征（优化版本，使用向量化计算）
        
        Args:
            hist_data: 历史窗口数据
            current_data: 当日数据
            window: 窗口大小
            
        Returns:
            窗口特征DataFrame
        """
        if len(hist_data) == 0:
            return pd.DataFrame(columns=['ts_code'])

        # 必须先按时间排序，保证 agg('first'/'last') 对应最早/最晚交易日
        hist_data = hist_data.sort_values(['ts_code', 'trade_date'])

        # 按股票分组，使用向量化操作计算特征
        # as_index=False 保留 ts_code 作为普通列
        grouped = hist_data.groupby('ts_code', as_index=False)
        
        # 计算累计收益率：(最后收盘价 / 第一个收盘价) - 1
        # 使用 agg 同时计算多个统计量
        window_features = grouped.agg({
            'close_adj': ['first', 'last', 'mean'],
            'vol': 'mean',
            'amount': 'mean'
        })
        
        # 展平多级列名
        # grouped.agg() 返回的 columns 是 MultiIndex
        # 第一列是 ('ts_code', '')，后面是 ('close_adj', 'first') 等
        new_columns = []
        for col in window_features.columns:
            if col[0] == 'ts_code':
                new_columns.append('ts_code')
            else:
                # 连接列名和聚合函数名
                new_columns.append('_'.join(col).strip('_'))
        window_features.columns = new_columns
        
        # 重命名列
        window_features = window_features.rename(columns={
            'close_adj_first': 'first_close',
            'close_adj_last': 'last_close',
            'close_adj_mean': 'ma_close',
            'vol_mean': 'mean_vol',
            'amount_mean': 'mean_amount'
        })
        
        # 计算累计收益率
        window_features[f'ret_{window}'] = (
            window_features['last_close'] / window_features['first_close']
        ) - 1
        
        # 合并当日数据计算比率
        current_vol_amount = current_data[['ts_code', 'vol', 'amount', 'close_adj']].copy()
        window_features = window_features.merge(current_vol_amount, on='ts_code', how='left')
        
        # 使用向量化操作计算比率（带除零保护）
        window_features[f'vol_ratio_{window}'] = np.where(
            window_features['mean_vol'] > 1e-6,
            window_features['vol'] / window_features['mean_vol'],
            np.nan
        )
        
        # 删除：不再生成 amount_ratio_* 特征
        # window_features[f'amount_ratio_{window}'] = ...
        
        window_features[f'ma_deviation_{window}'] = np.where(
            window_features['ma_close'] > 1e-6,
            (window_features['close_adj'] - window_features['ma_close']) / window_features['ma_close'],
            np.nan
        )
        
        # 保留需要的列（保留 amount_ma，删除 vol_ma 和 amount_ratio）
        keep_cols = ['ts_code', f'ret_{window}', f'vol_ratio_{window}', 
                     f'ma_deviation_{window}', 'mean_amount']
        window_features = window_features[keep_cols]
        
        # 重命名 mean_amount 为 amount_ma（保留 amount_ma，不保留 vol_ma）
        window_features = window_features.rename(columns={
            'mean_amount': f'amount_ma{window}'
        })
        
        return window_features
    
    def _add_advanced_factors(
        self,
        features: pd.DataFrame,
        current_data: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int,
        stock_basic: pd.DataFrame
    ) -> pd.DataFrame:
        """添加高级因子（技术指标、K线形态、波动率、行业等）
        
        Args:
            features: 基础特征DataFrame
            current_data: 当日数据
            daily_adj: 全部日线数据（含复权价）
            trade_date: 当前交易日
            trading_dates: 交易日序列
            current_idx: 当前交易日索引
            stock_basic: 股票基础信息
            
        Returns:
            添加了高级因子的DataFrame
        """
        result = features.copy()
        
        # 1. K线形态因子：振幅、上下影线
        logger.debug("计算K线形态因子...")
        if all(col in current_data.columns for col in ['high_adj', 'low_adj', 'pre_close', 'adj_factor']):
            amplitude_df = calculate_amplitude(current_data)
            result = result.merge(amplitude_df, on=['ts_code', 'trade_date'], how='left')
        
        logger.debug("计算K线形态因子（上下影线）...")
        if all(col in current_data.columns for col in ['open_adj', 'high_adj', 'low_adj', 'close_adj']):
            shadows_df = calculate_shadows(current_data)
            result = result.merge(shadows_df, on=['ts_code', 'trade_date'], how='left')
        
        # 2. 波动率因子（基于 ret_1 的 rolling std）——改为从预计算缓存取值
        logger.debug("获取波动率因子（批量预计算缓存）...")
        if 'ret_1' in result.columns and current_idx >= max(self.lookback_windows):
            tech_today = self._get_tech_factor_today(daily_adj, trade_date, trading_dates)
            vol_cols = [f'volatility_{w}' for w in self.lookback_windows
                        if f'volatility_{w}' in tech_today.columns]
            if vol_cols and len(tech_today) > 0:
                result = result.merge(
                    tech_today[['ts_code', 'trade_date'] + vol_cols],
                    on=['ts_code', 'trade_date'],
                    how='left',
                )
        
        # 3. 行业相关因子：industry_id, alpha_industry
        #logger.debug("计算行业相关因子...")
        #result = add_industry_features(result, stock_basic, ret_col='ret_1')
        
        # 计算多窗口行业 alpha（如果存在 ret_N，且行业列 sw_industry 存在）
        if all(f'ret_{w}' in result.columns for w in self.lookback_windows):
            # 优先使用 sw_industry（申万二级行业），若不存在则跳过行业 alpha 计算
            industry_col = 'sw_industry' if 'sw_industry' in result.columns else None
            if industry_col is not None:
                logger.debug("计算多个窗口的行业 alpha（基于申万二级行业 sw_industry）...")
                industry_alpha_df = calculate_industry_alpha_windows(
                    result, ret_windows=self.lookback_windows, industry_col=industry_col
                )
                result = result.merge(industry_alpha_df, on=['ts_code', 'trade_date'], how='left')

                # 行业动量特征（下沉到个股）
                from ..factors.industry import calculate_industry_momentum_features
                ind_mom_df = calculate_industry_momentum_features(
                    result, industry_col=industry_col, ret_col='ret_20',
                )
                result = result.merge(ind_mom_df, on=['ts_code', 'trade_date'], how='left')
            else:
                logger.debug("未找到行业列，跳过行业 alpha 计算")
        
        # 4. 动量加速度
        if 'ret_5' in result.columns and 'ret_10' in result.columns:
            logger.debug("计算动量加速度因子...")
            acceleration_df = calculate_acceleration(result)
            result = result.merge(acceleration_df, on=['ts_code', 'trade_date'], how='left')
        
        # 5. 量能突变（基于 vol_ratio 的截面 zscore）
        vol_ratio_cols = [f'vol_ratio_{w}' for w in self.lookback_windows]
        if all(col in result.columns for col in vol_ratio_cols):
            logger.debug("计算量能突变因子...")
            vol_burst_df = calculate_volume_burst(result, vol_ratio_windows=self.lookback_windows)
            result = result.merge(vol_burst_df, on=['ts_code', 'trade_date'], how='left')
        
        # 6. 技术指标：RSI, KDJ, MACD, 布林带——改为从预计算缓存取值
        # 需要足够的历史数据（至少30天用于 MACD(12,26,9) 和布林带(20)）
        if current_idx >= 30:
            tech_today = self._get_tech_factor_today(daily_adj, trade_date, trading_dates)
            tech_indicator_cols = [
                c for c in ['rsi_14', 'kdj_k', 'kdj_d', 'kdj_j',
                             'macd_dif', 'macd_dea', 'macd_hist',
                             'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_pct',
                             'atr_14', 'atr_pct_14']
                if c in tech_today.columns
            ]
            if tech_indicator_cols and len(tech_today) > 0:
                logger.debug("从预计算缓存获取技术指标（RSI/KDJ/MACD/布林带/ATR）...")
                result = result.merge(
                    tech_today[['ts_code', 'trade_date'] + tech_indicator_cols],
                    on=['ts_code', 'trade_date'],
                    how='left',
                )
        
        return result
    
    def _get_tech_factor_today(
        self,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """获取当日技术指标与波动率因子（内存缓存，首次触发批量预计算）

        首次调用时过滤 daily_adj 到全量交易日序列，再批量预计算所有指标并缓存到
        ``self._tech_factor_cache``；后续调用仅按 trade_date 过滤，
        实现 O(1) 查表，避免逐日重复计算。

        通过将 daily_adj 过滤到 trading_dates（全量 trade_cal 提取的交易日集合），
        确保滚动/EWM 指标的计算范围只依赖全量交易日历，与构建脚本的 start_date 无关。

        Args:
            daily_adj: 全量后复权日线数据
            trade_date: 目标交易日（YYYYMMDD）
            trading_dates: 全量交易日序列（已排序），用于过滤 daily_adj；为 None 时不过滤

        Returns:
            当日截面技术指标 DataFrame（ts_code + trade_date + 各指标列）
        """
        if self._tech_factor_cache is None:
            logger.info("首次构建：批量预计算技术指标与波动率因子（缓存中）...")
            # 统一从 trade_date 往前 _WARMUP_TRADING_DAYS 个交易日切片输入，
            # 消除 --start-date 不同导致的 rolling/EWM 指标历史起点差异
            if trading_dates is not None:
                daily_adj_for_cache = self._slice_by_trading_days(
                    daily_adj, trading_dates, trade_date
                )
                logger.debug(
                    f"技术指标预计算：daily_adj 按 warmup 窗口切片后剩余 "
                    f"{len(daily_adj_for_cache)} 条记录（原 {len(daily_adj)} 条）"
                )
            else:
                daily_adj_for_cache = daily_adj
            self._tech_factor_cache = precompute_technical_factors(
                daily_adj=daily_adj_for_cache,
                vol_windows=self.lookback_windows,
            )
            # 优化3：批量模式（_daily_adj_dict 已预计算）时建立 trade_date→sub_df 字典，O(1) 取值
            # 纸面交易（单日模式）跳过 dict 构建以节省内存（~15 MB）
            if (self._daily_adj_dict is not None
                    and self._tech_factor_cache is not None
                    and len(self._tech_factor_cache) > 0):
                self._tech_factor_cache_dict = {
                    d: sub_df.reset_index(drop=True)
                    for d, sub_df in self._tech_factor_cache.groupby('trade_date', sort=False)
                }

        if self._tech_factor_cache is None or len(self._tech_factor_cache) == 0:
            return pd.DataFrame(columns=['ts_code', 'trade_date'])

        # 优化3：O(1) 字典查表，替代全量 DataFrame 过滤
        if self._tech_factor_cache_dict is not None:
            return self._tech_factor_cache_dict.get(
                trade_date, pd.DataFrame(columns=['ts_code', 'trade_date'])
            )
        return self._tech_factor_cache[self._tech_factor_cache['trade_date'] == trade_date]

    def _add_value_dividend_features(
        self,
        features: pd.DataFrame,
        daily_basic_data: pd.DataFrame,
        trade_date: str
    ) -> pd.DataFrame:
        """添加价值红利因子（从 daily_basic）
        
        Args:
            features: 特征DataFrame
            daily_basic_data: daily_basic 数据
            trade_date: 当前交易日
            
        Returns:
            添加价值红利因子后的DataFrame
        """
        from ..common.feature_utils import log1p_transform
        
        # 筛选当日数据
        daily_basic_today = daily_basic_data[
            daily_basic_data['trade_date'] == trade_date
        ].copy()
        
        if len(daily_basic_today) == 0:
            logger.warning(f"{trade_date} 没有 daily_basic 数据，价值红利特征将为空")
            return features
        
        # 选择需要的列
        value_cols = ['ts_code', 'pb', 'pe_ttm', 'ps_ttm', 'dv_ttm', 
                      'total_mv', 'circ_mv', 'turnover_rate', 'volume_ratio']
        
        # 只保留存在的列
        existing_cols = ['ts_code'] + [c for c in value_cols[1:] if c in daily_basic_today.columns]
        daily_basic_today = daily_basic_today[existing_cols].copy()
        
        # 合并到 features
        features = features.merge(daily_basic_today, on='ts_code', how='left')

        # dv_ttm=NaN 表示未分红，语义上等同于 0（股息率为零）
        # 在此处填充，避免 NaN 传播到 zscore_dv_ttm，消除训练时 >30% NaN 警告
        if 'dv_ttm' in features.columns:
            features['dv_ttm'] = features['dv_ttm'].fillna(0)

        # 派生特征
        if 'pe_ttm' in features.columns:
            # ep_ttm = 1 / pe_ttm（市盈率倒数，即盈利收益率）
            # 处理 pe_ttm <= 0 或 NaN 的情况
            features['ep_ttm'] = np.where(
                (features['pe_ttm'].notna()) & (features['pe_ttm'] > 0),
                1.0 / features['pe_ttm'],
                np.nan
            )
            # 添加亏损标记（pe_ttm 为负或 NaN）
            features['is_loss'] = (
                (features['pe_ttm'].isna()) | (features['pe_ttm'] <= 0)
            ).astype(int)
        
        if 'pb' in features.columns:
            # bp = 1 / pb（市净率倒数，即账面市值比）
            # 处理 pb <= 0 或 NaN 的情况
            features['bp'] = np.where(
                (features['pb'].notna()) & (features['pb'] > 0),
                1.0 / features['pb'],
                np.nan
            )
        
        if 'total_mv' in features.columns:
            # log_total_mv = log1p(total_mv)（总市值对数变换）
            features['log_total_mv'] = log1p_transform(features['total_mv'])
        
        if 'circ_mv' in features.columns:
            # log_circ_mv = log1p(circ_mv)（流通市值对数变换）
            features['log_circ_mv'] = log1p_transform(features['circ_mv'])
        
        return features
    
    def _add_moneyflow_features(
        self,
        features: pd.DataFrame,
        moneyflow_data: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int
    ) -> pd.DataFrame:
        """添加资金流因子（从 moneyflow）
        
        计算净流入、大单特大单净流入的 rolling 特征
        
        Args:
            features: 特征DataFrame
            moneyflow_data: moneyflow 数据
            trade_date: 当前交易日
            trading_dates: 交易日序列
            current_idx: 当前交易日索引
            
        Returns:
            添加资金流因子后的DataFrame
        """
        from ..common.feature_utils import winsorize_series
        
        # 筛选当日数据
        moneyflow_today = moneyflow_data[
            moneyflow_data['trade_date'] == trade_date
        ].copy()
        
        if len(moneyflow_today) == 0:
            logger.warning(f"{trade_date} 没有 moneyflow 数据，资金流特征将为空")
            return features
        
        # 当日净流入特征（直接合并）
        merge_cols = ['ts_code', 'net_mf_amount']
        # 只保留存在的列
        merge_cols = [c for c in merge_cols if c in moneyflow_today.columns]
        if len(merge_cols) > 1:
            features = features.merge(
                moneyflow_today[merge_cols],
                on='ts_code',
                how='left'
            )
        
        # 计算大单、特大单净流入（当日）
        if 'buy_lg_amount' in moneyflow_today.columns and 'sell_lg_amount' in moneyflow_today.columns:
            moneyflow_today['lg_net_amount'] = (
                moneyflow_today['buy_lg_amount'] - moneyflow_today['sell_lg_amount']
            )
            features = features.merge(
                moneyflow_today[['ts_code', 'lg_net_amount']],
                on='ts_code',
                how='left'
            )
        
        if 'buy_elg_amount' in moneyflow_today.columns and 'sell_elg_amount' in moneyflow_today.columns:
            moneyflow_today['elg_net_amount'] = (
                moneyflow_today['buy_elg_amount'] - moneyflow_today['sell_elg_amount']
            )
            features = features.merge(
                moneyflow_today[['ts_code', 'elg_net_amount']],
                on='ts_code',
                how='left'
            )

            # 订单失衡（特大单）: (buy_elg - sell_elg) / (buy_elg + sell_elg)
            _total = moneyflow_today['buy_elg_amount'] + moneyflow_today['sell_elg_amount']
            moneyflow_today['order_imbalance'] = np.where(
                _total > 1e-6,
                (moneyflow_today['buy_elg_amount'] - moneyflow_today['sell_elg_amount']) / _total,
                np.nan,
            )
            features = features.merge(
                moneyflow_today[['ts_code', 'order_imbalance']],
                on='ts_code',
                how='left'
            )

        # 计算 rolling 特征（窗口 5, 20）
        for window in [5, 20]:
            # 以全量交易日历为锚点，向前回溯恰好 window 个交易日
            hist_dates = self._get_lookback_dates(trade_date, window, trading_dates)
            if not hist_dates:
                # 历史不足 window 个交易日，填充空值
                features[f'net_mf_amount_sum_{window}'] = np.nan
                features[f'net_mf_amount_mean_{window}'] = np.nan
                if 'lg_net_amount' in features.columns:
                    features[f'lg_net_amount_sum_{window}'] = np.nan
                if 'elg_net_amount' in features.columns:
                    features[f'elg_net_amount_sum_{window}'] = np.nan
                continue
            
            # 获取历史数据（只取全量交易日序列中确定的 window 个日期）
            hist_moneyflow = moneyflow_data[
                moneyflow_data['trade_date'].isin(hist_dates)
            ].copy()
            
            if len(hist_moneyflow) == 0:
                continue
            
            # 计算大单、特大单净流入（历史）
            if 'buy_lg_amount' in hist_moneyflow.columns and 'sell_lg_amount' in hist_moneyflow.columns:
                hist_moneyflow['lg_net_amount'] = (
                    hist_moneyflow['buy_lg_amount'] - hist_moneyflow['sell_lg_amount']
                )
            
            if 'buy_elg_amount' in hist_moneyflow.columns and 'sell_elg_amount' in hist_moneyflow.columns:
                hist_moneyflow['elg_net_amount'] = (
                    hist_moneyflow['buy_elg_amount'] - hist_moneyflow['sell_elg_amount']
                )
                # 历史订单失衡
                _total = hist_moneyflow['buy_elg_amount'] + hist_moneyflow['sell_elg_amount']
                hist_moneyflow['order_imbalance'] = np.where(
                    _total > 1e-6,
                    (hist_moneyflow['buy_elg_amount'] - hist_moneyflow['sell_elg_amount']) / _total,
                    np.nan,
                )

            # 按股票分组计算 rolling 特征
            # 只对存在的列进行聚合
            agg_dict = {}
            if 'net_mf_amount' in hist_moneyflow.columns:
                agg_dict['net_mf_amount'] = ['sum', 'mean']
            if 'lg_net_amount' in hist_moneyflow.columns:
                agg_dict['lg_net_amount'] = ['sum']
            if 'elg_net_amount' in hist_moneyflow.columns:
                agg_dict['elg_net_amount'] = ['sum']
            if 'order_imbalance' in hist_moneyflow.columns:
                agg_dict['order_imbalance'] = ['mean']
            
            if not agg_dict:
                # 没有可聚合的列，跳过
                continue
            
            rolling_features = hist_moneyflow.groupby('ts_code').agg(agg_dict).reset_index()
            
            # 展平列名
            new_columns = ['ts_code']
            for col in rolling_features.columns[1:]:
                if isinstance(col, tuple):
                    new_columns.append(f'{col[0]}_{col[1]}_{window}')
                else:
                    new_columns.append(col)
            rolling_features.columns = new_columns
            
            # 合并到 features
            features = features.merge(rolling_features, on='ts_code', how='left')
        
        # 对重尾列进行 winsorize 处理
        winsorize_cols = [c for c in features.columns if 'net_amount' in c or 'mf_amount' in c]
        for col in winsorize_cols:
            if col in features.columns:
                features[col] = winsorize_series(features[col], limits=(0.01, 0.01))
        
        return features

    def _add_fundamental_features(
        self,
        features: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        trade_date: str,
    ) -> pd.DataFrame:
        """添加基本面因子（从 fina_indicator 季度数据前向填充）

        Args:
            features: 特征 DataFrame
            fundamental_data: 当日基本面数据（已前向填充），含 ts_code, roe_waa 等
            trade_date: 当前交易日

        Returns:
            添加基本面因子后的 DataFrame
        """
        from ..factors.fundamental import FUNDA_COLS

        if fundamental_data is None or len(fundamental_data) == 0:
            logger.debug(f"{trade_date} 没有基本面数据，跳过")
            return features

        merge_cols = ['ts_code'] + [c for c in FUNDA_COLS if c in fundamental_data.columns]
        features = features.merge(
            fundamental_data[merge_cols],
            on='ts_code',
            how='left'
        )

        matched = features[FUNDA_COLS[0]].notna().sum() if FUNDA_COLS[0] in features.columns else 0
        if self.verbose:
            logger.debug(f"{trade_date} 基本面特征: 匹配 {matched}/{len(features)} 只股票")

        return features

    def _add_filter_flags(
        self,
        df: pd.DataFrame,
        stock_basic: pd.DataFrame,
        suspend_info: Optional[pd.DataFrame],
        trade_date: str
    ) -> pd.DataFrame:
        """添加过滤标记
        
        当 clean 层数据包含标记时直接复用，否则重新计算。
        注意：涨跌停标记由 _add_limit_flags 方法处理。
        
        Args:
            df: 特征DataFrame
            stock_basic: 股票基本信息
            suspend_info: 停复牌信息
            trade_date: 交易日期
            
        Returns:
            添加了过滤标记的DataFrame
        """
        result = df.copy()
        
        # 检查是否已有 clean 层的标记（tradable, is_st 等）
        # 注意：is_limit_up/is_limit_down 由 _add_limit_flags 单独检查和处理
        has_clean_flags = all(col in result.columns for col in ['is_st', 'is_suspended', 'tradable', 'list_days'])
        
        if has_clean_flags:
            logger.info("数据已包含 clean 层过滤标记，直接复用")
            return result
        
        # 如果没有clean标记，则需要自己计算
        logger.info("clean 层标记不存在，开始计算过滤标记")
        
        # 1. ST标记：通过股票名称判断
        stock_names = stock_basic[['ts_code', 'name']].copy()
        result = result.merge(stock_names, on='ts_code', how='left')
        
        # 判断ST：名称包含ST、*ST、S*ST等（使用更精确的匹配）
        # 匹配模式：开头可选的*或S，然后是ST，或者包含"退"字
        result['is_st'] = result['name'].fillna('').str.contains(
            r'^\*?S?\*?ST|退', 
            case=False, 
            regex=True
        ).astype(int)
        
        # 2. 上市天数
        stock_list_date = stock_basic[['ts_code', 'list_date']].copy()
        
        # 确保日期格式一致
        if pd.api.types.is_datetime64_any_dtype(stock_list_date['list_date']):
            stock_list_date['list_date'] = stock_list_date['list_date'].dt.strftime('%Y%m%d')
        
        result = result.merge(
            stock_list_date,
            on='ts_code',
            how='left',
            suffixes=('', '_basic')
        )
        
        # 计算上市天数
        # 注意：这里使用自然日天数作为粗略估计
        # 实际应该使用交易日历计算实际交易日数量，但为简化计算使用自然日
        # 对于min_list_days=365的设置，自然日365天约对应250个交易日（约12个月）
        try:
            trade_date_dt = pd.to_datetime(trade_date, format='%Y%m%d')
            result['list_date_dt'] = pd.to_datetime(result['list_date'], format='%Y%m%d', errors='coerce')
            result['list_days'] = (trade_date_dt - result['list_date_dt']).dt.days
            result.drop(columns=['list_date_dt'], inplace=True)
        except Exception as e:
            logger.warning(f"计算上市天数失败: {e}，使用默认值")
            result['list_days'] = 999  # 默认视为满足条件
        
        # 3. 停牌标记（使用统一列名 is_suspended）
        # 简化处理：如果当日成交量为0或极小，视为停牌
        if 'vol' in result.columns:
            result['is_suspended'] = (result['vol'] <= 0).astype(int)
        else:
            result['is_suspended'] = 0
        
        # 如果有停复牌信息，可以进一步完善
        if suspend_info is not None and len(suspend_info) > 0:
            # 新版API：suspend_info包含trade_date和suspend_type字段
            # 兼容旧版：如果有suspend_date字段，使用旧逻辑
            if 'suspend_date' in suspend_info.columns and 'resume_date' in suspend_info.columns:
                # 旧版逻辑：获取当日停牌的股票
                # 统一日期类型为字符串以避免类型不匹配
                suspend_info_normalized = normalize_date_columns(
                    suspend_info, ['suspend_date', 'resume_date'], to_str=True
                )
                trade_date_str = to_trade_date_str(trade_date)
                
                suspend_today = suspend_info_normalized[
                    (suspend_info_normalized['suspend_date'] <= trade_date_str) &
                    ((suspend_info_normalized['resume_date'] >= trade_date_str) | 
                     (suspend_info_normalized['resume_date'].isna()))
                ]['ts_code'].unique()
                
                result.loc[result['ts_code'].isin(suspend_today), 'is_suspended'] = 1
            elif 'trade_date' in suspend_info.columns and 'suspend_type' in suspend_info.columns:
                # 新版逻辑：筛选当日类型为'S'(停牌)的股票
                # 统一日期格式
                suspend_info_normalized = normalize_date_columns(
                    suspend_info, ['trade_date'], to_str=True
                )
                trade_date_str = to_trade_date_str(trade_date)
                
                suspend_today = suspend_info_normalized[
                    (suspend_info_normalized['trade_date'] == trade_date_str) &
                    (suspend_info_normalized['suspend_type'] == 'S')
                ]['ts_code'].unique()
                
                result.loc[result['ts_code'].isin(suspend_today), 'is_suspended'] = 1
        
        return result
    
    def _add_limit_flags(
        self,
        df: pd.DataFrame,
        daily_data: pd.DataFrame,
        limit_info: Optional[pd.DataFrame],
        trade_date: str
    ) -> pd.DataFrame:
        """添加涨跌停标记
        
        当 clean 层数据包含涨跌停标记时直接复用，否则重新计算。
        
        Args:
            df: 特征DataFrame
            daily_data: 日线行情
            limit_info: 涨跌停价格信息
            trade_date: 交易日期
            
        Returns:
            添加了涨跌停标记的DataFrame
        """
        result = df.copy()
        
        # 检查是否已有 clean 层的涨跌停标记
        has_clean_limit_flags = all(col in result.columns for col in ['is_limit_up', 'is_limit_down'])
        
        if has_clean_limit_flags:
            logger.info("数据已包含 clean 层涨跌停标记，直接复用")
            return result
        
        # 如果没有clean标记，则需要自己计算
        logger.info("clean 层涨跌停标记不存在，开始计算")
        
        # 获取当日行情
        current_daily = daily_data[daily_data['trade_date'] == trade_date][
            ['ts_code', 'close', 'pct_chg']
        ].copy()
        
        result = result.merge(current_daily, on='ts_code', how='left', suffixes=('', '_daily'))
        
        # 使用涨跌幅判断涨跌停
        # 注意：不同板块涨跌幅限制不同
        #   主板/中小板：±10%（非ST），±5%（ST）
        #   科创板（688xxx.SH）：±20%
        #   创业板注册制（300xxx/301xxx.SZ，2020-08-24起）：±20%
        result['is_limit_up'] = 0
        result['is_limit_down'] = 0

        non_st_mask = (result['is_st'] == 0)
        st_mask = (result['is_st'] == 1)

        # 科创板（688开头）/ 创业板注册制（300/301开头）：±20% 阈值
        kcb_mask = result['ts_code'].str.startswith('688')
        gem_mask = result['ts_code'].str.startswith('300') | result['ts_code'].str.startswith('301')
        reg_board_mask = (kcb_mask | gem_mask) & non_st_mask

        # 主板/其他非ST：±10% 阈值
        main_board_mask = ~(kcb_mask | gem_mask) & non_st_mask

        result.loc[reg_board_mask & (result['pct_chg'] >= 19.9), 'is_limit_up'] = 1
        result.loc[reg_board_mask & (result['pct_chg'] <= -19.9), 'is_limit_down'] = 1
        result.loc[main_board_mask & (result['pct_chg'] >= 9.9), 'is_limit_up'] = 1
        result.loc[main_board_mask & (result['pct_chg'] <= -9.9), 'is_limit_down'] = 1

        # ST股票：±5% 阈值
        result.loc[st_mask & (result['pct_chg'] >= 4.9), 'is_limit_up'] = 1
        result.loc[st_mask & (result['pct_chg'] <= -4.9), 'is_limit_down'] = 1
        
        # 如果有涨跌停价格信息，可以更精确地判断
        if limit_info is not None and len(limit_info) > 0:
            limit_today = limit_info[limit_info['trade_date'] == trade_date][
                ['ts_code', 'up_limit', 'down_limit']
            ].copy()
            
            if len(limit_today) > 0:
                result = result.merge(
                    limit_today,
                    on='ts_code',
                    how='left',
                    suffixes=('', '_limit')
                )
                
                # 使用价格对比（更精确）
                result.loc[
                    (result['close'] >= result['up_limit'] * 0.999),
                    'is_limit_up'
                ] = 1
                result.loc[
                    (result['close'] <= result['down_limit'] * 1.001),
                    'is_limit_down'
                ] = 1
                
                result.drop(columns=['up_limit', 'down_limit'], inplace=True, errors='ignore')
        
        # 清理不需要的列
        result.drop(columns=['close', 'pct_chg'], inplace=True, errors='ignore')
        
        return result
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用过滤规则（训练/推理共用）

        过滤条件：
        - 剔除 ST (is_st=1)
        - 剔除上市 < min_list_days 自然日（默认365天≈12个月）
        - 剔除停牌 (is_suspended=1)
        - 剔除标签缺失 (所有 y_ret_* 为空) - 仅当 require_label=True 时
        - 涨跌停不剔除，仅标记

        注：成交额/市值/金融股过滤在 MLSignal._apply_selection_filters 中执行，
        仅作用于实盘/回测选股阶段，不影响训练数据。

        Args:
            df: 特征DataFrame

        Returns:
            过滤后的DataFrame
        """
        original_count = len(df)

        st_count = (df['is_st'] == 1).sum()
        list_days_count = (df['list_days'] < self.min_list_days).sum()
        suspend_count = (df['is_suspended'] == 1).sum()

        # 统计各标签缺失情况
        label_missing_info = {}
        for horizon in self.horizons:
            label_col = f'y_ret_{horizon}'
            if label_col in df.columns:
                label_missing_info[label_col] = df[label_col].isna().sum()

        logger.info(
            f"过滤前样本数: {original_count}, "
            f"ST: {st_count}, 上市<{self.min_list_days}天: {list_days_count}, "
            f"停牌: {suspend_count}, 标签缺失: {label_missing_info}"
        )

        filter_mask = (
            (df['is_st'] == 0) &
            (df['list_days'] >= self.min_list_days) &
            (df['is_suspended'] == 0)
        )

        if self.require_label:
            label_mask = pd.Series([False] * len(df), index=df.index)
            for horizon in self.horizons:
                label_col = f'y_ret_{horizon}'
                if label_col in df.columns:
                    label_mask = label_mask | df[label_col].notna()
            filter_mask = filter_mask & label_mask
            logger.info("require_label=True, 将过滤所有标签均缺失的样本")
        else:
            logger.info("require_label=False, 保留标签缺失样本（实盘/推理模式）")

        result = df[filter_mask].copy()

        logger.info(f"过滤后样本数: {len(result)} （剔除 {original_count - len(result)} 只）")
        
        return result
    
    def _merge_shenwan_industry(
        self,
        features: pd.DataFrame,
        shenwan_industry: pd.DataFrame
    ) -> pd.DataFrame:
        """合并申万行业分类信息（支持 L3 三级与旧式 L2 两种格式）

        新式 L3 格式（优先）：shenwan_industry 含 sw_l3_code、sw_l3 列，同时含
          sw_l2_code/sw_l2、sw_l1_code/sw_l1 列，产出：
          - sw_industry / sw_industry_code / sw_industry_id（映射到 L3）
          - sw_l2 / sw_l2_code / sw_l2_id
          - sw_l1 / sw_l1_code / sw_l1_id

        旧式 L2 格式（兼容）：shenwan_industry 含 sw_code、sw_name 列，产出：
          - sw_industry / sw_industry_code / sw_industry_id

        Args:
            features: 特征DataFrame
            shenwan_industry: 申万行业分类DataFrame

        Returns:
            合并了行业信息的DataFrame
        """
        from ..factors.industry import generate_industry_encoding

        if shenwan_industry is None or len(shenwan_industry) == 0:
            logger.warning("申万行业分类数据为空，跳过合并")
            return features

        sw_cols = shenwan_industry.columns.tolist()

        # ---- 新式 L3 格式检测 ----
        is_l3_format = ('sw_l3_code' in sw_cols or 'sw_l3' in sw_cols)

        if is_l3_format:
            # 确定实际存在的列
            l3_cols = [c for c in ['sw_l3_code', 'sw_l3', 'sw_l2_code', 'sw_l2',
                                    'sw_l1_code', 'sw_l1', 'in_date']
                       if c in sw_cols]
            merge_cols = ['ts_code'] + l3_cols
            existing_merge_cols = [c for c in merge_cols if c in sw_cols]

            if len(existing_merge_cols) < 2:
                logger.warning(f"申万 L3 数据缺少必要字段，现有列：{sw_cols}")
                return features

            result = features.merge(
                shenwan_industry[existing_merge_cols],
                on='ts_code',
                how='left'
            )

            # L3 → sw_industry*
            if 'sw_l3_code' in result.columns:
                result = result.rename(columns={'sw_l3_code': 'sw_industry_code'})
            if 'sw_l3' in result.columns:
                result = result.rename(columns={'sw_l3': 'sw_industry'})

            # 生成 sw_industry_id
            if 'sw_industry' in result.columns:
                id_dict = generate_industry_encoding(result['sw_industry'])
                result['sw_industry_id'] = result['sw_industry'].map(id_dict)

            # 生成 sw_l2_id
            if 'sw_l2' in result.columns:
                id_dict_l2 = generate_industry_encoding(result['sw_l2'])
                result['sw_l2_id'] = result['sw_l2'].map(id_dict_l2)

            # 生成 sw_l1_id
            if 'sw_l1' in result.columns:
                id_dict_l1 = generate_industry_encoding(result['sw_l1'])
                result['sw_l1_id'] = result['sw_l1'].map(id_dict_l1)

            if self.verbose:
                industry_counts = result.get('sw_industry', pd.Series()).value_counts()
                logger.info(f"申万三级行业分布（前5）：\n{industry_counts.head()}")

        else:
            # ---- 旧式 L2/L1 格式 ----
            industry_cols = ['ts_code', 'sw_code', 'sw_name']
            existing_cols = [col for col in industry_cols if col in sw_cols]

            if len(existing_cols) < 2:
                logger.warning(f"申万行业数据缺少必要字段，现有列：{sw_cols}")
                return features

            result = features.merge(
                shenwan_industry[existing_cols],
                on='ts_code',
                how='left'
            )

            rename_map = {}
            if 'sw_name' in result.columns:
                rename_map['sw_name'] = 'sw_industry'
            if 'sw_code' in result.columns:
                rename_map['sw_code'] = 'sw_industry_code'
            if rename_map:
                result = result.rename(columns=rename_map)

            if 'sw_industry' in result.columns:
                id_dict = generate_industry_encoding(result['sw_industry'])
                result['sw_industry_id'] = result['sw_industry'].map(id_dict)

        return result
    
    def _apply_industry_neutralization(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """应用行业中性化（包含去均值和Z-Score两类）

        当数据包含 L3 层级信息（sw_industry_code、sw_l2_code、sw_l1_code）时，
        使用分层回退中性化（L3→L2→L1→全市场）；否则退化为单层 sw_industry 中性化。

        对指定的列进行行业中性化：
        1. 去均值（demean）：收益率/标签列，neu_ 前缀
        2. Z-Score：指标/特征列，zscore_ 前缀

        Args:
            features: 特征DataFrame，需包含 sw_industry 列（及可选的 sw_l2_code、sw_l1_code）

        Returns:
            添加了行业中性化列的DataFrame
        """
        from ..factors.normalization import industry_demean, industry_neutralization
        from ..factors.hierarchical_industry_neutralization import (
            hierarchical_demean,
            hierarchical_zscore,
        )

        # 检查必要的列是否存在
        if 'sw_industry' not in features.columns:
            logger.error(
                "缺少申万行业列 sw_industry，无法进行行业中性化。\n"
                "请确保已加载申万行业分类数据并通过参数传递给 build_features_for_day"
            )
            return features

        if 'tradable' not in features.columns:
            logger.warning("缺少 tradable 列，将使用全部样本进行统计")

        result = features.copy()

        # 判断是否有 L3 分层信息
        has_hierarchy = all(
            col in result.columns
            for col in ['sw_industry_code', 'sw_l2_code', 'sw_l1_code']
        )

        # ========================================
        # 1. 去均值（demean）中性化：收益率/标签列
        # ========================================
        demean_columns = []
        for horizon in self.horizons:
            label_col = f'y_ret_{horizon}'
            if label_col in result.columns:
                demean_columns.append(label_col)
        # ret_1 单独加入（lookback_windows 不含 1，但 ret_1 是重要的反转信号）
        if 'ret_1' in result.columns:
            demean_columns.append('ret_1')
        for window in self.lookback_windows:
            ret_col = f'ret_{window}'
            if ret_col in result.columns:
                demean_columns.append(ret_col)

        if len(demean_columns) > 0:
            if has_hierarchy:
                logger.info(
                    f"开始分层回退行业去均值（L3→L2→L1→全市场）：{len(demean_columns)} 个列"
                )
                try:
                    result = hierarchical_demean(
                        result,
                        columns=demean_columns,
                        l3_col='sw_industry_code',
                        l2_col='sw_l2_code',
                        l1_col='sw_l1_code',
                        tradable_col='tradable',
                        min_group_size=5,
                        prefix='neu_',
                    )
                    actual_new = [f'neu_{c}' for c in demean_columns if f'neu_{c}' in result.columns]
                    logger.info(f"分层去均值完成，新增 {len(actual_new)} 列")
                except Exception as e:
                    logger.error(f"分层行业去均值失败：{e}")
            else:
                logger.info(
                    f"开始行业去均值（按 sw_industry 分组）：{len(demean_columns)} 个列"
                )
                try:
                    result = industry_demean(
                        result,
                        columns=demean_columns,
                        industry_col='sw_industry',
                        tradable_col='tradable',
                        min_group_size=5,
                        prefix='neu_',
                        inplace=False
                    )
                    actual_new = [f'neu_{c}' for c in demean_columns if f'neu_{c}' in result.columns]
                    logger.info(f"去均值完成，新增 {len(actual_new)} 列")
                except Exception as e:
                    logger.error(f"行业去均值失败：{e}")
        else:
            logger.info("没有找到需要去均值的收益率/标签列")

        # ========================================
        # 2. Z-Score 中性化：指标/特征列
        # ========================================
        zscore_columns = [
            'pe_ttm', 'pb', 'bp', 'dv_ttm', 'log_total_mv',
            'amount_ma20', 'turnover_rate', 'volatility_5', 'volatility_10',
            'volatility_20', 'net_mf_amount', 'ma_deviation_20',
            'elg_net_amount_sum_20', 'acceleration', 'macd_hist', 'bb_width',
            # 基本面因子（季度数据前向填充，启用时才存在）
            'roe_waa', 'or_yoy', 'netprofit_yoy', 'debt_to_assets', 'q_gr_yoy',
            # 增强因子：开盘强度、日内波动结构、订单失衡
            'opening_strength', 'intraday_vol_structure', 'order_imbalance',
        ]
        existing_zscore_columns = [col for col in zscore_columns if col in result.columns]
        for window in self.lookback_windows:
            vol_col = f'volatility_{window}'
            if vol_col in result.columns and vol_col not in existing_zscore_columns:
                existing_zscore_columns.append(vol_col)

        if len(existing_zscore_columns) > 0:
            if has_hierarchy:
                logger.info(
                    f"开始分层回退行业内 Z-Score（L3→L2→L1→全市场）：{len(existing_zscore_columns)} 个特征"
                )
                try:
                    result = hierarchical_zscore(
                        result,
                        columns=existing_zscore_columns,
                        l3_col='sw_industry_code',
                        l2_col='sw_l2_code',
                        l1_col='sw_l1_code',
                        tradable_col='tradable',
                        min_group_size=5,
                        prefix='zscore_',
                    )
                    actual_new = [f'zscore_{c}' for c in existing_zscore_columns if f'zscore_{c}' in result.columns]
                    logger.info(f"分层 Z-Score 完成，新增 {len(actual_new)} 列")
                except Exception as e:
                    logger.error(f"分层行业内 Z-Score 失败：{e}")
            else:
                logger.info(
                    f"开始行业内 Z-Score（按 sw_industry 分组）：{len(existing_zscore_columns)} 个特征"
                )
                try:
                    result = industry_neutralization(
                        result,
                        columns=existing_zscore_columns,
                        industry_col='sw_industry',
                        tradable_col='tradable',
                        min_group_size=5,
                        prefix='zscore_',
                        inplace=False
                    )
                    actual_new = [f'zscore_{c}' for c in existing_zscore_columns if f'zscore_{c}' in result.columns]
                    logger.info(f"Z-Score 完成，新增 {len(actual_new)} 列")
                except Exception as e:
                    logger.error(f"行业内 Z-Score 失败：{e}")
        else:
            logger.info("没有找到需要 Z-Score 的特征列")

        return result


    def _add_new_individual_features(
        self,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """添加新增个股特征：is_new_stock、size、zscore_size、spec_score

        Args:
            result: 当日截面 DataFrame

        Returns:
            添加了新特征列的 DataFrame
        """
        # is_new_stock: 上市不足 365 天则为 1，否则为 0
        if 'list_days' in result.columns:
            result['is_new_stock'] = (result['list_days'] < 365).astype(int)
        else:
            logger.warning("缺少 list_days 列，is_new_stock 将全部设为 0（无法判断新股）")
            result['is_new_stock'] = 0

        # size: 流通市值
        if 'circ_mv' in result.columns:
            result['size'] = result['circ_mv']

        # zscore_size: 行业内对 log1p(size) 做 Z-Score
        if 'size' in result.columns and 'sw_industry' in result.columns:
            result['_log1p_size'] = np.log1p(result['size'])
            result = cross_sectional_zscore(
                result,
                columns=['_log1p_size'],
                group_col='sw_industry',
                tradable_col='tradable',
                min_group_size=5,
                suffix='_z',
            )
            if '_log1p_size_z' in result.columns:
                result.rename(columns={'_log1p_size_z': 'zscore_size'}, inplace=True)
            if '_log1p_size' in result.columns:
                result.drop(columns=['_log1p_size'], inplace=True)

        # spec_score: zscore_volatility_20 * (-zscore_size)
        if 'zscore_volatility_20' in result.columns and 'zscore_size' in result.columns:
            result['spec_score'] = (
                result['zscore_volatility_20'] * (-result['zscore_size'])
            )
        else:
            result['spec_score'] = np.nan

        return result

    def _add_market_state_features(
        self,
        result: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: list,
        current_idx: int,
        daily_basic_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """添加市场状态特征（每日一个标量，广播到所有股票）

        首次调用时对全部 trading_dates 批量预计算并缓存；
        后续调用直接按 trade_date 从缓存 O(1) 取值，避免逐日重复计算。

        Args:
            result: 当日截面 DataFrame
            daily_adj: 全量后复权日线数据
            trade_date: 目标交易日（YYYYMMDD）
            trading_dates: 已排序的交易日列表
            current_idx: trade_date 在 trading_dates 中的索引（保留，兼容旧调用）
            daily_basic_data: 全量每日指标数据（可选）

        Returns:
            添加了市场状态列的 DataFrame
        """
        try:
            # 首次进入时批量预计算并缓存
            if self._market_state_cache is None:
                logger.info("首次构建：批量预计算所有交易日市场状态特征（缓存中）...")
                # 统一从 trade_date 往前 _WARMUP_TRADING_DAYS 个交易日切片输入，
                # 消除 --start-date 不同导致的 rolling 指标历史起点差异
                sliced_daily_adj = self._slice_by_trading_days(
                    daily_adj, trading_dates, trade_date
                )
                sliced_daily_basic = (
                    self._slice_by_trading_days(daily_basic_data, trading_dates, trade_date)
                    if daily_basic_data is not None else None
                )
                self._market_state_cache = precompute_market_state_features(
                    daily_data=sliced_daily_adj,
                    trading_dates=trading_dates,
                    daily_basic_data=sliced_daily_basic,
                    tech_factor_df=self._tech_factor_cache,
                )

            # 按 trade_date O(1) 取值
            if trade_date in self._market_state_cache.index:
                row = self._market_state_cache.loc[trade_date]
                mkt_features = row.to_dict()
            else:
                # 安全回退：该日不在缓存中，逐日计算
                logger.warning(f"{trade_date} 不在市场状态缓存中，回退到逐日计算")
                mkt_features = compute_market_state_features(
                    daily_data=daily_adj,
                    trade_date=trade_date,
                    trading_dates=trading_dates,
                    current_idx=current_idx,
                    daily_basic_data=daily_basic_data,
                )
        except Exception as e:
            logger.error(f"计算市场状态特征失败：{e}")
            mkt_features = {
                'mkt_vol_cnt': np.nan,
                'mkt_vol_20': np.nan,
                'mkt_turnover_ratio': np.nan,
                'mkt_ret_avg_20': np.nan,
                'mkt_turnover_std': np.nan,
                'mkt_adv_dec_ratio': np.nan,
                'mkt_ma250_ratio': np.nan,
            }

        for feat_name, feat_val in mkt_features.items():
            result[feat_name] = feat_val

        return result
