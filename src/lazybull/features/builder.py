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
)


class FeatureBuilder:
    """特征构建器
    
    负责生成单日全市场截面训练数据，包含特征和标签
    """
    
    def __init__(
        self,
        min_list_days: int = 60,
        horizon: int = 5,
        horizons: List[int] = None,
        lookback_windows: List[int] = None,
        require_label: bool = True,
        verbose: bool = False,
    ):
        """初始化特征构建器
        
        Args:
            min_list_days: 最小上市天数，默认60天
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
        
        if self.verbose:
            logger.info(
                f"特征构建器初始化: min_list_days={min_list_days}, "
                f"horizons={self.horizons}, lookback_windows={self.lookback_windows}, "
                f"require_label={require_label}"
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
        apply_industry_neutralization: bool = False
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
                         新版API格式：ts_code, trade_date, suspend_type, suspend_timing
                         旧版格式（兼容）：ts_code, suspend_date, resume_date
            limit_info: 涨跌停价格DataFrame（可选）
            shenwan_industry: 申万行业分类DataFrame（可选），包含 ts_code, sw_code, sw_name
            apply_industry_neutralization: 是否应用行业中性化，默认False
            
        Returns:
            特征DataFrame，包含 trade_date, ts_code, 特征列, 标签列, 标记列
        """
        logger.info(f"开始构建 {trade_date} 的特征")
        
        # 1. 获取交易日序列
        trading_dates = self._get_trading_dates(trade_cal)
        
        if trade_date not in trading_dates:
            logger.warning(f"{trade_date} 不是交易日，跳过")
            return pd.DataFrame()
        
        current_idx = trading_dates.index(trade_date)
        
        # 2. 计算后复权收盘价
        daily_adj = self._calculate_adj_close(daily_data, adj_factor)
        
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
            moneyflow_data
        )
        logger.debug(f"{trade_date} 基础特征计算完成: {len(features.columns.tolist())} 列")
        
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
        
        # 6.5 合并申万行业分类
        if shenwan_industry is not None:
            result = self._merge_shenwan_industry(result, shenwan_industry)
            logger.debug(f"{trade_date} 合并申万行业分类完成: {len(result.columns.tolist())} 列")
        
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
        
        logger.info(f"{trade_date} 特征构建完成: {len(result)} 个样本")
        
        return result
    
    def _get_trading_dates(self, trade_cal: pd.DataFrame) -> List[str]:
        """从交易日历提取交易日列表
        
        Args:
            trade_cal: 交易日历DataFrame
            
        Returns:
            交易日列表（格式YYYYMMDD，排序）
        """
        if 'cal_date' in trade_cal.columns:
            # 如果是datetime格式，转换为字符串
            if pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
                trade_cal = trade_cal.copy()
                trade_cal['cal_date'] = trade_cal['cal_date'].dt.strftime('%Y%m%d')
            
            trading_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
        else:
            logger.error("交易日历缺少 cal_date 字段")
            return []
        
        return sorted(trading_dates)
    
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
            logger.warning(f"有 {missing_adj} 条记录缺少复权因子，将使用原始价格")
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
            
            # 获取未来收盘价
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
        moneyflow_data: Optional[pd.DataFrame] = None
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
        
        # 计算回看特征
        for window in self.lookback_windows:
            # 获取历史窗口数据
            if current_idx < window:
                # 历史数据不足，填充空值
                features[f'ret_{window}'] = np.nan
                features[f'vol_ratio_{window}'] = np.nan
                # 删除：不再生成 amount_ratio_{window}
                features[f'ma_deviation_{window}'] = np.nan
                continue
            
            # 历史日期范围
            hist_start_date = trading_dates[current_idx - window]
            hist_end_date = trading_dates[current_idx - 1]  # 不包含当日
            
            hist_dates = [
                d for d in trading_dates
                if hist_start_date <= d <= hist_end_date
            ]
            
            # 获取历史数据
            hist_data = daily_adj[
                (daily_adj['trade_date'].isin(hist_dates))
            ].copy()
            
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
            window_features['mean_vol'] > 0,
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
        
        # 2. 波动率因子（基于 ret_1 的 rolling std）
        logger.debug("计算波动率因子...")
        if 'ret_1' in result.columns and current_idx >= max(self.lookback_windows):
            # 获取历史窗口数据用于计算波动率
            lookback = max(self.lookback_windows) + 1  # 额外留1天用于计算 ret_1
            hist_start_date = trading_dates[max(0, current_idx - lookback)]
            hist_dates = [d for d in trading_dates if hist_start_date <= d <= trade_date]
            
            vol_hist_data = daily_adj[daily_adj['trade_date'].isin(hist_dates)].copy()
            
            # 确保包含 ret_1 列（从 pct_chg 计算）
            if 'ret_1' not in vol_hist_data.columns and 'pct_chg' in vol_hist_data.columns:
                vol_hist_data['ret_1'] = vol_hist_data['pct_chg'] / 100.0
            
            if 'ret_1' in vol_hist_data.columns:
                volatility_df = calculate_volatility(vol_hist_data, ret_col='ret_1', windows=self.lookback_windows)
                # 只保留当日的波动率
                volatility_today = volatility_df[volatility_df['trade_date'] == trade_date]
                if len(volatility_today) > 0:
                    result = result.merge(volatility_today, on=['ts_code', 'trade_date'], how='left')
        
        # 3. 行业相关因子：industry_id, alpha_industry
        logger.debug("计算行业相关因子...")
        result = add_industry_features(result, stock_basic, ret_col='ret_1')
        
        # 计算多窗口行业 alpha（如果存在 ret_N）
        if all(f'ret_{w}' in result.columns for w in self.lookback_windows):
            logger.debug("计算多个窗口的行业 alpha...")
            industry_alpha_df = calculate_industry_alpha_windows(
                result, ret_windows=self.lookback_windows, industry_col='industry'
            )
            result = result.merge(industry_alpha_df, on=['ts_code', 'trade_date'], how='left')
        
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
        
        # 6. 技术指标：RSI, KDJ, MACD, 布林带
        # 需要足够的历史数据（至少30天用于 MACD(12,26,9) 和布林带(20)）
        if current_idx >= 30:
            lookback = 50  # 给技术指标留足够历史
            hist_start_date = trading_dates[max(0, current_idx - lookback)]
            hist_dates = [d for d in trading_dates if hist_start_date <= d <= trade_date]
            
            tech_hist_data = daily_adj[daily_adj['trade_date'].isin(hist_dates)].copy()
            
            # RSI(14)
            logger.debug("计算RSI指标...")
            if 'close_adj' in tech_hist_data.columns:
                rsi_df = calculate_rsi(tech_hist_data, window=14)
                rsi_today = rsi_df[rsi_df['trade_date'] == trade_date]
                if len(rsi_today) > 0:
                    result = result.merge(rsi_today, on=['ts_code', 'trade_date'], how='left')
            
            # KDJ(9,3,3)
            logger.debug("计算KDJ指标...")
            if all(col in tech_hist_data.columns for col in ['high_adj', 'low_adj', 'close_adj']):
                kdj_df = calculate_kdj(tech_hist_data, n=9, m1=3, m2=3)
                kdj_today = kdj_df[kdj_df['trade_date'] == trade_date]
                if len(kdj_today) > 0:
                    result = result.merge(kdj_today, on=['ts_code', 'trade_date'], how='left')
            
            # MACD(12,26,9)
            logger.debug("计算MACD指标...")
            if 'close_adj' in tech_hist_data.columns:
                macd_df = calculate_macd(tech_hist_data, fast=12, slow=26, signal=9)
                macd_today = macd_df[macd_df['trade_date'] == trade_date]
                if len(macd_today) > 0:
                    result = result.merge(macd_today, on=['ts_code', 'trade_date'], how='left')
            
            # 布林带(20,2)
            logger.debug("计算布林带指标...")
            if 'close_adj' in tech_hist_data.columns:
                bb_df = calculate_bollinger_bands(tech_hist_data, window=20, num_std=2.0)
                bb_today = bb_df[bb_df['trade_date'] == trade_date]
                if len(bb_today) > 0:
                    result = result.merge(bb_today, on=['ts_code', 'trade_date'], how='left')
        
        return result
    
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
        
        # 计算 rolling 特征（窗口 5, 20）
        for window in [5, 20]:
            if current_idx < window:
                # 历史数据不足，填充空值
                features[f'net_mf_amount_sum_{window}'] = np.nan
                features[f'net_mf_amount_mean_{window}'] = np.nan
                if 'lg_net_amount' in features.columns:
                    features[f'lg_net_amount_sum_{window}'] = np.nan
                if 'elg_net_amount' in features.columns:
                    features[f'elg_net_amount_sum_{window}'] = np.nan
                continue
            
            # 历史日期范围
            hist_start_date = trading_dates[current_idx - window]
            hist_end_date = trading_dates[current_idx - 1]  # 不包含当日
            
            hist_dates = [
                d for d in trading_dates
                if hist_start_date <= d <= hist_end_date
            ]
            
            # 获取历史数据
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
            
            # 按股票分组计算 rolling 特征
            # 只对存在的列进行聚合
            agg_dict = {}
            if 'net_mf_amount' in hist_moneyflow.columns:
                agg_dict['net_mf_amount'] = ['sum', 'mean']
            if 'lg_net_amount' in hist_moneyflow.columns:
                agg_dict['lg_net_amount'] = ['sum']
            if 'elg_net_amount' in hist_moneyflow.columns:
                agg_dict['elg_net_amount'] = ['sum']
            
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
        # 对于min_list_days=60的设置，自然日60天大约对应40-45个交易日
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
        
        # 简化方法：使用涨跌幅判断（A股涨跌停通常为±10%，ST为±5%）
        # 这里使用9.9%和-9.9%作为阈值（考虑精度问题）
        result['is_limit_up'] = 0
        result['is_limit_down'] = 0
        
        # 非ST股票：涨跌幅 >= 9.9%
        non_st_mask = (result['is_st'] == 0)
        result.loc[non_st_mask & (result['pct_chg'] >= 9.9), 'is_limit_up'] = 1
        result.loc[non_st_mask & (result['pct_chg'] <= -9.9), 'is_limit_down'] = 1
        
        # ST股票：涨跌幅 >= 4.9%
        st_mask = (result['is_st'] == 1)
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
        """应用过滤规则
        
        过滤条件：
        - 剔除 ST (is_st=1)
        - 剔除上市 < 60天 (list_days < 60)
        - 剔除停牌 (is_suspended=1)
        - 剔除标签缺失 (所有 y_ret_* 为空) - 仅当 require_label=True 时
        - 涨跌停不剔除，仅标记
        
        Args:
            df: 特征DataFrame
            
        Returns:
            过滤后的DataFrame
        """
        original_count = len(df)
        
        # 记录过滤统计
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
        
        # 应用过滤
        # 基础过滤条件
        filter_mask = (
            (df['is_st'] == 0) &
            (df['list_days'] >= self.min_list_days) &
            (df['is_suspended'] == 0)
        )
        
        # 仅当 require_label=True 时过滤标签缺失
        # 要求至少有一个标签非空（而非所有标签都非空）
        if self.require_label:
            # 构建标签非空的条件：至少一个标签列非空
            label_mask = pd.Series([False] * len(df), index=df.index)
            for horizon in self.horizons:
                label_col = f'y_ret_{horizon}'
                if label_col in df.columns:
                    label_mask = label_mask | df[label_col].notna()
            
            filter_mask = filter_mask & label_mask
            logger.info(f"require_label=True, 将过滤所有标签均缺失的样本")
        else:
            logger.info(f"require_label=False, 保留标签缺失样本（实盘/推理模式）")
        
        result = df[filter_mask].copy()
        
        logger.info(f"过滤后样本数: {len(result)}")
        
        return result
    
    def _merge_shenwan_industry(
        self,
        features: pd.DataFrame,
        shenwan_industry: pd.DataFrame
    ) -> pd.DataFrame:
        """合并申万行业分类信息
        
        Args:
            features: 特征DataFrame
            shenwan_industry: 申万行业分类DataFrame，包含 ts_code, sw_code, sw_name
            
        Returns:
            合并了行业信息的DataFrame
        """
        if shenwan_industry is None or len(shenwan_industry) == 0:
            logger.warning("申万行业分类数据为空，跳过合并")
            return features
        
        # 选择需要的列
        industry_cols = ['ts_code', 'sw_code', 'sw_name']
        existing_cols = [col for col in industry_cols if col in shenwan_industry.columns]
        
        if len(existing_cols) < 2:  # 至少需要 ts_code + 一个行业字段
            logger.warning(f"申万行业数据缺少必要字段，现有列：{shenwan_industry.columns.tolist()}")
            return features
        
        # 合并行业信息
        result = features.merge(
            shenwan_industry[existing_cols],
            on='ts_code',
            how='left'
        )
        
        # 生成行业ID（整数编码）
        if 'sw_name' in result.columns:
            from ..factors.industry import generate_industry_encoding
            industry_id_dict = generate_industry_encoding(
                result['sw_name'],
                #result[['ts_code', 'trade_date', 'sw_name']],
                #industry_col='sw_name',
            )
            #logger.warning(f"industry_id_dict: {industry_id_dict}")
            result['industry_id'] = result['sw_name'].map(industry_id_dict)
            
            # 统计行业分布
            if self.verbose:
                industry_counts = result['sw_name'].value_counts()
                logger.info(f"行业分布（前5）：\n{industry_counts.head()}")
        
        return result
    
    def _apply_industry_neutralization(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """应用行业中性化（包含去均值和Z-Score两类）
        
        对指定的列进行行业中性化：
        1. 去均值（demean）：收益率/标签列，neu_前缀
        2. Z-Score：指标/特征列，_zscore后缀
        
        Args:
            features: 特征DataFrame，需包含 sw_name, tradable 列
            
        Returns:
            添加了行业中性化列的DataFrame
        """
        from ..factors.normalization import industry_demean, industry_neutralization
        
        # 检查必要的列是否存在
        if 'sw_name' not in features.columns:
            logger.error(
                "缺少申万行业列 sw_name，无法进行行业中性化。\n"
                "请确保已加载申万行业分类数据并通过参数传递给 build_features_for_day"
            )
            return features
        
        if 'tradable' not in features.columns:
            logger.warning("缺少 tradable 列，将使用全部样本进行统计")
        
        result = features.copy()
        
        # ========================================
        # 1. 去均值（demean）中性化：收益率/标签列
        # ========================================
        # 适用列：y_ret_5, y_ret_10, y_ret_20, ret_5, ret_10, ret_20
        # 命名规则：neu_ 前缀
        demean_columns = []
        
        # 标签列
        for horizon in self.horizons:
            label_col = f'y_ret_{horizon}'
            if label_col in result.columns:
                demean_columns.append(label_col)
        
        # 历史收益列
        for window in self.lookback_windows:
            ret_col = f'ret_{window}'
            if ret_col in result.columns:
                demean_columns.append(ret_col)
        
        if len(demean_columns) > 0:
            logger.info(f"开始行业去均值：{len(demean_columns)} 个收益率/标签列")
            logger.debug(f"去均值列表：{demean_columns}")
            
            try:
                result = industry_demean(
                    result,
                    columns=demean_columns,
                    industry_col='sw_name',
                    tradable_col='tradable',
                    min_group_size=5,
                    prefix='neu_',
                    inplace=False
                )
                
                # 统计新增的列
                new_cols = [f'neu_{col}' for col in demean_columns]
                actual_new_cols = [col for col in new_cols if col in result.columns]
                logger.info(f"去均值完成，新增 {len(actual_new_cols)} 列")
            except Exception as e:
                logger.error(f"行业去均值失败：{e}")
        else:
            logger.info("没有找到需要去均值的收益率/标签列")
        
        # ========================================
        # 2. Z-Score 中性化：指标/特征列
        # ========================================
        # 白名单（注意：从白名单中移除了 ret_20，因为用户明确只要去均值版）
        zscore_columns = [
            'pe_ttm',           # 市盈率
            'pb',               # 市净率
            'bp',               # 市净率倒数
            'dv_ttm',          # 股息率
            'log_total_mv',    # 对数总市值
            'amount_ma20',     # 20日均成交额
            'turnover_rate',   # 换手率
            'net_mf_amount',   # 净资金流入
            'ma_deviation_20', # 20日均线偏离度
        ]
        
        # 检查哪些列存在
        existing_zscore_columns = [col for col in zscore_columns if col in result.columns]
        
        # 添加波动率列（volatility_5, volatility_10, volatility_20）
        for window in self.lookback_windows:
            vol_col = f'volatility_{window}'
            if vol_col in result.columns:
                existing_zscore_columns.append(vol_col)
        
        if len(existing_zscore_columns) > 0:
            logger.info(f"开始行业内 Z-Score：{len(existing_zscore_columns)} 个特征")
            logger.debug(f"Z-Score 列表：{existing_zscore_columns}")
            
            try:
                result = industry_neutralization(
                    result,
                    columns=existing_zscore_columns,
                    industry_col='sw_name',
                    tradable_col='tradable',
                    min_group_size=5,
                    prefix='',  # 不使用前缀，而是使用后缀
                    inplace=False
                )
                
                # 将 neu_ 前缀改为 _zscore 后缀
                for col in existing_zscore_columns:
                    old_col = f'neu_{col}'
                    new_col = f'{col}_zscore'
                    if old_col in result.columns:
                        result[new_col] = result[old_col]
                        result.drop(columns=[old_col], inplace=True)
                
                    # 统计新增的列
                new_cols = [f'{col}_zscore' for col in existing_zscore_columns]
                actual_new_cols = [col for col in new_cols if col in result.columns]
                logger.info(f"Z-Score 完成，新增 {len(actual_new_cols)} 列")
            except Exception as e:
                logger.error(f"行业内 Z-Score 失败：{e}")
        else:
            logger.info("没有找到需要 Z-Score 的特征列")
        
        return result
