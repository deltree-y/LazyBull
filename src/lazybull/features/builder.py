"""特征与标签构建模块

实现按日截面特征构建，包括：
- 后复权收盘价计算
- 未来5日收益标签 (horizon=5)
- 基础数值特征
- 股票池过滤（ST、上市<60天、停牌）
- 涨跌停标记
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class FeatureBuilder:
    """特征构建器
    
    负责生成单日全市场截面训练数据，包含特征和标签
    """
    
    def __init__(
        self,
        min_list_days: int = 60,
        horizon: int = 5,
        lookback_windows: List[int] = None,
        volume_filter_pct: float = 20.0,
        volume_filter_enabled: bool = True
    ):
        """初始化特征构建器
        
        Args:
            min_list_days: 最小上市天数，默认60天
            horizon: 预测时间窗口（交易日），默认5天
            lookback_windows: 回看窗口列表，用于计算历史特征，默认[5, 10, 20]
            volume_filter_pct: 过滤成交量后N%的股票，默认20%
            volume_filter_enabled: 是否启用成交量过滤，默认True
        """
        self.min_list_days = min_list_days
        self.horizon = horizon
        self.lookback_windows = lookback_windows or [5, 10, 20]
        self.volume_filter_pct = volume_filter_pct
        self.volume_filter_enabled = volume_filter_enabled
        
        logger.info(
            f"特征构建器初始化: min_list_days={min_list_days}, "
            f"horizon={horizon}, lookback_windows={self.lookback_windows}, "
            f"volume_filter_pct={volume_filter_pct}%, volume_filter_enabled={volume_filter_enabled}"
        )
    
    def build_features_for_day(
        self,
        trade_date: str,
        trade_cal: pd.DataFrame,
        daily_data: pd.DataFrame,
        adj_factor: pd.DataFrame,
        stock_basic: pd.DataFrame,
        suspend_info: Optional[pd.DataFrame] = None,
        limit_info: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """构建单个交易日的截面特征和标签
        
        Args:
            trade_date: 目标交易日，格式YYYYMMDD
            trade_cal: 交易日历DataFrame，需包含 cal_date, is_open
            daily_data: 日线行情DataFrame，需包含 ts_code, trade_date, close, pre_close, 
                       pct_chg, vol, amount 等字段
            adj_factor: 复权因子DataFrame，需包含 ts_code, trade_date, adj_factor
            stock_basic: 股票基本信息DataFrame，需包含 ts_code, name, list_date
            suspend_info: 停复牌信息DataFrame（可选）
                         新版API格式：ts_code, trade_date, suspend_type, suspend_timing
                         旧版格式（兼容）：ts_code, suspend_date, resume_date
            limit_info: 涨跌停价格DataFrame（可选）
            
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
        
        # 4. 计算标签：未来5日收益
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
            current_idx
        )
        
        # 6. 合并特征和标签
        result = features.merge(labels, on=['trade_date', 'ts_code'], how='inner')
        
        # 7. 添加过滤标记
        result = self._add_filter_flags(
            result,
            stock_basic,
            suspend_info,
            trade_date
        )
        
        # 8. 添加涨跌停标记
        result = self._add_limit_flags(
            result,
            daily_data,
            limit_info,
            trade_date
        )
        
        # 9. 应用过滤规则
        result = self._apply_filters(result)
        
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
        """计算后复权收盘价
        
        Args:
            daily_data: 日线行情DataFrame
            adj_factor: 复权因子DataFrame
            
        Returns:
            添加了 close_adj 列的DataFrame
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
        
        # 处理缺失的复权因子（如果有）
        missing_adj = daily_adj['adj_factor'].isna().sum()
        if missing_adj > 0:
            logger.warning(f"有 {missing_adj} 条记录缺少复权因子，将使用原始收盘价")
            daily_adj['close_adj'].fillna(daily_adj['close'], inplace=True)
        
        return daily_adj
    
    def _calculate_forward_returns(
        self,
        current_data: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int
    ) -> pd.DataFrame:
        """计算未来N日收益标签
        
        Args:
            current_data: 当日数据
            daily_adj: 全部日线数据（含后复权价）
            trade_date: 当前交易日
            trading_dates: 交易日序列
            current_idx: 当前交易日在序列中的索引
            
        Returns:
            包含标签的DataFrame
        """
        # 检查是否有足够的未来交易日
        if current_idx + self.horizon >= len(trading_dates):
            logger.warning(f"{trade_date} 后续交易日不足 {self.horizon} 天，无法计算标签")
            # 返回空标签
            result = current_data[['trade_date', 'ts_code', 'close_adj']].copy()
            result['y_ret_5'] = np.nan
            return result
        
        # 获取未来第N个交易日
        future_date = trading_dates[current_idx + self.horizon]
        
        # 获取未来收盘价
        future_data = daily_adj[daily_adj['trade_date'] == future_date][
            ['ts_code', 'close_adj']
        ].copy()
        future_data.rename(columns={'close_adj': 'close_adj_future'}, inplace=True)
        
        # 合并当前和未来数据
        result = current_data[['trade_date', 'ts_code', 'close_adj']].merge(
            future_data,
            on='ts_code',
            how='left'
        )
        
        # 计算收益率: (close_adj_future / close_adj) - 1
        # 添加除零保护：过滤掉收盘价为0或极小的样本
        valid_mask = result['close_adj'] > 1e-6
        result.loc[valid_mask, 'y_ret_5'] = (
            result.loc[valid_mask, 'close_adj_future'] / result.loc[valid_mask, 'close_adj']
        ) - 1
        result.loc[~valid_mask, 'y_ret_5'] = np.nan
        
        # 删除中间列
        result.drop(columns=['close_adj', 'close_adj_future'], inplace=True)
        
        # 记录缺失标签的样本数
        missing_labels = result['y_ret_5'].isna().sum()
        if missing_labels > 0:
            logger.warning(
                f"{trade_date} 有 {missing_labels} 个样本缺失未来收盘价，标签为空"
            )
        
        return result
    
    def _calculate_features(
        self,
        current_data: pd.DataFrame,
        daily_adj: pd.DataFrame,
        trade_date: str,
        trading_dates: List[str],
        current_idx: int
    ) -> pd.DataFrame:
        """计算基础数值特征
        
        特征包括：
        - ret_1: 当日收益率
        - ret_N: 过去N日累计收益
        - vol_ratio_N: 过去N日平均成交量比
        - amount_ratio_N: 过去N日平均成交额比
        - ma_deviation_N: 收盘价与N日均线的偏离度
        
        Args:
            current_data: 当日数据
            daily_adj: 全部日线数据
            trade_date: 当前交易日
            trading_dates: 交易日序列
            current_idx: 当前交易日索引
            
        Returns:
            包含特征的DataFrame
        """
        # 初始化特征DataFrame，包含vol和amount用于后续过滤
        features = current_data[['trade_date', 'ts_code', 'vol', 'amount']].copy()
        
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
                features[f'amount_ratio_{window}'] = np.nan
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
        
        window_features[f'amount_ratio_{window}'] = np.where(
            window_features['mean_amount'] > 0,
            window_features['amount'] / window_features['mean_amount'],
            np.nan
        )
        
        window_features[f'ma_deviation_{window}'] = np.where(
            window_features['ma_close'] > 1e-6,
            (window_features['close_adj'] - window_features['ma_close']) / window_features['ma_close'],
            np.nan
        )
        
        # 保留需要的列
        keep_cols = ['ts_code', f'ret_{window}', f'vol_ratio_{window}', 
                     f'amount_ratio_{window}', f'ma_deviation_{window}']
        window_features = window_features[keep_cols]
        
        return window_features
    
    def _add_filter_flags(
        self,
        df: pd.DataFrame,
        stock_basic: pd.DataFrame,
        suspend_info: Optional[pd.DataFrame],
        trade_date: str
    ) -> pd.DataFrame:
        """添加过滤标记
        
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
        has_clean_flags = all(col in result.columns for col in ['is_st', 'is_suspended', 'tradable'])
        
        if has_clean_flags:
            logger.info("数据已包含 clean 层过滤标记，跳过标记添加")
            # 重命名以匹配特征构建器的命名（如果需要）
            if 'is_suspended' in result.columns and 'suspend' not in result.columns:
                result['suspend'] = result['is_suspended']
            return result
        
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
        
        # 3. 停牌标记
        # 简化处理：如果当日成交量为0或极小，视为停牌
        if 'vol' in result.columns:
            result['suspend'] = (result['vol'] <= 0).astype(int)
        else:
            result['suspend'] = 0
        
        # 如果有停复牌信息，可以进一步完善
        if suspend_info is not None and len(suspend_info) > 0:
            # 新版API：suspend_info包含trade_date和suspend_type字段
            # 兼容旧版：如果有suspend_date字段，使用旧逻辑
            if 'suspend_date' in suspend_info.columns and 'resume_date' in suspend_info.columns:
                # 旧版逻辑：获取当日停牌的股票
                suspend_today = suspend_info[
                    (suspend_info['suspend_date'] <= trade_date) &
                    ((suspend_info['resume_date'] >= trade_date) | (suspend_info['resume_date'].isna()))
                ]['ts_code'].unique()
                
                result.loc[result['ts_code'].isin(suspend_today), 'suspend'] = 1
            elif 'trade_date' in suspend_info.columns and 'suspend_type' in suspend_info.columns:
                # 新版逻辑：筛选当日类型为'S'(停牌)的股票
                suspend_today = suspend_info[
                    (suspend_info['trade_date'] == trade_date) &
                    (suspend_info['suspend_type'] == 'S')
                ]['ts_code'].unique()
                
                result.loc[result['ts_code'].isin(suspend_today), 'suspend'] = 1
        
        return result
    
    def _add_limit_flags(
        self,
        df: pd.DataFrame,
        daily_data: pd.DataFrame,
        limit_info: Optional[pd.DataFrame],
        trade_date: str
    ) -> pd.DataFrame:
        """添加涨跌停标记
        
        Args:
            df: 特征DataFrame
            daily_data: 日线行情
            limit_info: 涨跌停价格信息
            trade_date: 交易日期
            
        Returns:
            添加了涨跌停标记的DataFrame
        """
        result = df.copy()
        
        # 获取当日行情
        current_daily = daily_data[daily_data['trade_date'] == trade_date][
            ['ts_code', 'close', 'pct_chg']
        ].copy()
        
        result = result.merge(current_daily, on='ts_code', how='left', suffixes=('', '_daily'))
        
        # 简化方法：使用涨跌幅判断（A股涨跌停通常为±10%，ST为±5%）
        # 这里使用9.9%和-9.9%作为阈值（考虑精度问题）
        result['limit_up'] = 0
        result['limit_down'] = 0
        
        # 非ST股票：涨跌幅 >= 9.9%
        non_st_mask = (result['filter_is_st'] == 0)
        result.loc[non_st_mask & (result['pct_chg'] >= 9.9), 'limit_up'] = 1
        result.loc[non_st_mask & (result['pct_chg'] <= -9.9), 'limit_down'] = 1
        
        # ST股票：涨跌幅 >= 4.9%
        st_mask = (result['filter_is_st'] == 1)
        result.loc[st_mask & (result['pct_chg'] >= 4.9), 'limit_up'] = 1
        result.loc[st_mask & (result['pct_chg'] <= -4.9), 'limit_down'] = 1
        
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
                    'limit_up'
                ] = 1
                result.loc[
                    (result['close'] <= result['down_limit'] * 1.001),
                    'limit_down'
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
        - 剔除停牌 (suspend=1)
        - 剔除成交量后N%的股票（可配置）
        - 剔除标签缺失 (y_ret_5 为空)
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
        suspend_count = (df['suspend'] == 1).sum()
        missing_label_count = df['y_ret_5'].isna().sum()
        
        # 应用基础过滤
        result = df[
            (df['is_st'] == 0) &
            (df['list_days'] >= self.min_list_days) &
            (df['suspend'] == 0) &
            (df['y_ret_5'].notna())
        ].copy()
        
        # 成交量过滤
        volume_filtered_count = 0
        if self.volume_filter_enabled and 'vol' in result.columns and len(result) > 0:
            # 处理成交量缺失或为0的情况
            valid_vol_mask = (result['vol'].notna()) & (result['vol'] > 0)
            result_with_vol = result[valid_vol_mask].copy()
            result_no_vol = result[~valid_vol_mask].copy()
            
            if len(result_with_vol) > 0:
                # 计算成交量分位数阈值
                volume_threshold_pct = self.volume_filter_pct / 100.0
                volume_threshold = result_with_vol['vol'].quantile(volume_threshold_pct)
                
                # 过滤掉成交量在后N%的股票
                before_vol_filter = len(result_with_vol)
                result_with_vol = result_with_vol[result_with_vol['vol'] > volume_threshold].copy()
                volume_filtered_count = before_vol_filter - len(result_with_vol)
                
                # 合并有成交量和无成交量的数据（无成交量的已在停牌过滤中处理，这里可忽略）
                result = result_with_vol
        
        logger.info(
            f"过滤前样本数: {original_count}, "
            f"ST: {st_count}, 上市<{self.min_list_days}天: {list_days_count}, "
            f"停牌: {suspend_count}, 标签缺失: {missing_label_count}, "
            f"成交量过滤: {volume_filtered_count}"
        )
        logger.info(f"过滤后样本数: {len(result)}")
        
        return result
