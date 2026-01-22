"""数据清洗模块

实现 raw -> clean 的数据转换，包括：
- 去重（按主键 ts_code+trade_date）
- 类型统一（trade_date 为 YYYYMMDD 字符串，数值列转 float/int）
- 缺失值处理（adj_factor 回退，必要列报错或填充）
- 复权后行情计算（close_adj/open_adj/high_adj/low_adj）
- ST/停牌过滤标记（tradable_universe 列）
- 数据排序与校验
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class DataCleaner:
    """数据清洗器
    
    负责将 raw 层数据转换为标准化的 clean 层数据
    """
    
    def __init__(self, verbose: bool = False):
        """初始化数据清洗器"""
        self.verbose = verbose
        if self.verbose:
            logger.info("数据清洗器初始化完成")
    
    def clean_trade_cal(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """清洗交易日历数据
        
        Args:
            raw_df: 原始交易日历DataFrame
            
        Returns:
            清洗后的交易日历DataFrame
        """
        logger.info(f"开始清洗交易日历数据，原始记录数: {len(raw_df)}")
        
        df = raw_df.copy()
        
        # 1. 类型统一：trade_date 转为 YYYYMMDD 字符串
        df = self._standardize_date_columns(df, ['cal_date', 'pretrade_date'])
        
        # 2. 去重：按主键 (exchange, cal_date) 去重，保留最新记录
        df = self._deduplicate(df, ['exchange', 'cal_date'])
        
        # 3. 类型转换：is_open 转为 int
        if 'is_open' in df.columns:
            df['is_open'] = df['is_open'].astype(int)
        
        # 4. 排序：按 cal_date 排序
        df = df.sort_values('cal_date').reset_index(drop=True)
        
        logger.info(f"交易日历清洗完成，清洗后记录数: {len(df)}")
        
        return df
    
    def clean_stock_basic(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """清洗股票基本信息
        
        Args:
            raw_df: 原始股票基本信息DataFrame
            
        Returns:
            清洗后的股票基本信息DataFrame
        """
        logger.info(f"开始清洗股票基本信息，原始记录数: {len(raw_df)}")
        
        df = raw_df.copy()
        
        # 1. 类型统一
        df = self._standardize_date_columns(df, ['list_date'])
        
        # 2. ts_code 统一为字符串
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].astype(str)
        
        # 3. 去重：按主键 ts_code 去重
        df = self._deduplicate(df, ['ts_code'])
        
        # 4. 排序：按 ts_code 排序
        df = df.sort_values('ts_code').reset_index(drop=True)
        
        logger.info(f"股票基本信息清洗完成，清洗后记录数: {len(df)}")
        
        return df
    
    def clean_daily(
        self,
        raw_daily: pd.DataFrame,
        raw_adj_factor: pd.DataFrame
    ) -> pd.DataFrame:
        """清洗日线行情并计算复权价格
        
        Args:
            raw_daily: 原始日线行情DataFrame
            raw_adj_factor: 原始复权因子DataFrame
            
        Returns:
            清洗后的日线行情DataFrame（包含复权价格列）
        """
        logger.info(f"开始清洗日线行情数据，原始记录数: {len(raw_daily)}")
        
        df = raw_daily.copy()
        
        # 1. 类型统一：trade_date 转为 YYYYMMDD 字符串
        df = self._standardize_date_columns(df, ['trade_date'])
        
        # 2. ts_code 统一为字符串
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].astype(str)
        
        # 3. 数值列转换为 float
        numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 
                       'pct_chg', 'vol', 'amount']
        df = self._convert_numeric_columns(df, numeric_cols)
        
        # 4. 去重：按主键 (ts_code, trade_date) 去重
        df = self._deduplicate(df, ['ts_code', 'trade_date'])
        
        # 5. 合并复权因子并计算复权价格
        df = self._calculate_adjusted_prices(df, raw_adj_factor)
        
        # 6. 过滤异常数据：去除成交量/成交额为负的记录
        if 'vol' in df.columns:
            invalid_vol = (df['vol'] < 0).sum()
            if invalid_vol > 0:
                logger.warning(f"发现 {invalid_vol} 条成交量为负的记录，将被过滤")
                df = df[df['vol'] >= 0]
        
        if 'amount' in df.columns:
            invalid_amount = (df['amount'] < 0).sum()
            if invalid_amount > 0:
                logger.warning(f"发现 {invalid_amount} 条成交额为负的记录，将被过滤")
                df = df[df['amount'] >= 0]
        
        # 7. 排序：按 ts_code, trade_date 排序
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        # 8. 验证唯一性
        self._validate_uniqueness(df, ['ts_code', 'trade_date'])
        
        logger.info(f"日线行情清洗完成，清洗后记录数: {len(df)}")
        
        return df
    
    def clean_daily_basic(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """清洗每日指标数据
        
        Args:
            raw_df: 原始每日指标DataFrame
            
        Returns:
            清洗后的每日指标DataFrame
        """
        logger.info(f"开始清洗每日指标数据，原始记录数: {len(raw_df)}")
        
        df = raw_df.copy()
        
        # 1. 类型统一
        df = self._standardize_date_columns(df, ['trade_date'])
        
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].astype(str)
        
        # 2. 数值列转换
        numeric_cols = ['close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
                       'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
                       'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']
        df = self._convert_numeric_columns(df, numeric_cols)
        
        # 3. 去重
        df = self._deduplicate(df, ['ts_code', 'trade_date'])
        
        # 4. 排序
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        # 5. 验证唯一性
        self._validate_uniqueness(df, ['ts_code', 'trade_date'])
        
        logger.info(f"每日指标清洗完成，清洗后记录数: {len(df)}")
        
        return df
    
    def clean_suspend_info(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """清洗停复牌信息
        
        Args:
            raw_df: 原始停复牌DataFrame
            
        Returns:
            清洗后的停复牌DataFrame
        """
        logger.info(f"开始清洗停复牌信息，原始记录数: {len(raw_df)}")
        
        df = raw_df.copy()
        
        # 1. 类型统一
        date_cols = ['trade_date']
        # 兼容旧版字段
        if 'suspend_date' in df.columns:
            date_cols.append('suspend_date')
        if 'resume_date' in df.columns:
            date_cols.append('resume_date')
        
        df = self._standardize_date_columns(df, date_cols)
        
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].astype(str)
        
        # 2. 去重（根据字段决定主键）
        if 'trade_date' in df.columns:
            # 新版：按 (ts_code, trade_date) 去重
            df = self._deduplicate(df, ['ts_code', 'trade_date'])
            df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        elif 'suspend_date' in df.columns:
            # 旧版：按 (ts_code, suspend_date) 去重
            df = self._deduplicate(df, ['ts_code', 'suspend_date'])
            df = df.sort_values(['ts_code', 'suspend_date']).reset_index(drop=True)
        
        logger.info(f"停复牌信息清洗完成，清洗后记录数: {len(df)}")
        
        return df
    
    def clean_limit_info(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """清洗涨跌停信息
        
        Args:
            raw_df: 原始涨跌停DataFrame
            
        Returns:
            清洗后的涨跌停DataFrame
        """
        logger.info(f"开始清洗涨跌停信息，原始记录数: {len(raw_df)}")
        
        df = raw_df.copy()
        
        # 1. 类型统一
        df = self._standardize_date_columns(df, ['trade_date'])
        
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].astype(str)
        
        # 2. 数值列转换
        numeric_cols = ['pre_close', 'up_limit', 'down_limit']
        df = self._convert_numeric_columns(df, numeric_cols)
        
        # 3. 去重
        df = self._deduplicate(df, ['ts_code', 'trade_date'])
        
        # 4. 排序
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        logger.info(f"涨跌停信息清洗完成，清洗后记录数: {len(df)}")
        
        return df
    
    def add_tradable_universe_flag(
        self,
        daily_df: pd.DataFrame,
        stock_basic_df: pd.DataFrame,
        suspend_info_df: Optional[pd.DataFrame] = None,
        limit_info_df: Optional[pd.DataFrame] = None,
        min_list_days: int = 60
    ) -> pd.DataFrame:
        """为 daily 数据添加可交易标记
        
        标记逻辑：
        - is_st: 是否为 ST 股票（1=是, 0=否）
        - is_suspended: 是否停牌（1=是, 0=否）
        - is_limit_up: 是否涨停（1=是, 0=否）
        - is_limit_down: 是否跌停（1=是, 0=否）
        - list_days: 上市天数
        - tradable: 是否可交易（1=可交易, 0=不可交易）
          不可交易条件：ST或停牌或上市不足N天
        
        Args:
            daily_df: 清洗后的日线行情DataFrame
            stock_basic_df: 清洗后的股票基本信息DataFrame
            suspend_info_df: 清洗后的停复牌信息（可选）
            limit_info_df: 清洗后的涨跌停信息（可选）
            min_list_days: 最小上市天数，默认60天
            
        Returns:
            添加了标记列的DataFrame
        """
        logger.info(f"为日线数据添加可交易标记，记录数: {len(daily_df)}")
        
        df = daily_df.copy()
        
        # 1. 合并股票基本信息，获取名称和上市日期
        stock_info = stock_basic_df[['ts_code', 'name', 'list_date']].copy()
        df = df.merge(stock_info, on='ts_code', how='left')
        
        # 2. ST 标记
        df['is_st'] = df['name'].fillna('').str.contains(
            r'^\*?S?\*?ST|退', 
            case=False, 
            regex=True
        ).astype(int)
        
        # 3. 上市天数（使用自然日近似）
        df['list_days'] = 999  # 默认值
        valid_mask = df['list_date'].notna() & (df['list_date'] != '')
        if valid_mask.sum() > 0:
            try:
                df.loc[valid_mask, 'list_date_dt'] = pd.to_datetime(
                    df.loc[valid_mask, 'list_date'], 
                    format='%Y%m%d',
                    errors='coerce'
                )
                df.loc[valid_mask, 'trade_date_dt'] = pd.to_datetime(
                    df.loc[valid_mask, 'trade_date'],
                    format='%Y%m%d',
                    errors='coerce'
                )
                valid_dates = df['list_date_dt'].notna() & df['trade_date_dt'].notna()
                df.loc[valid_dates, 'list_days'] = (
                    df.loc[valid_dates, 'trade_date_dt'] - df.loc[valid_dates, 'list_date_dt']
                ).dt.days
                df.drop(columns=['list_date_dt', 'trade_date_dt'], inplace=True, errors='ignore')
            except Exception as e:
                logger.warning(f"计算上市天数失败: {e}")
        
        # 4. 停牌标记：先用成交量判断
        df['is_suspended'] = ((df['vol'] <= 0) | (df['vol'].isna())).astype(int)
        
        # 如果有停复牌信息，进一步完善
        if suspend_info_df is not None and len(suspend_info_df) > 0:
            # 新版 API
            if 'trade_date' in suspend_info_df.columns and 'suspend_type' in suspend_info_df.columns:
                suspend_dates = suspend_info_df[
                    suspend_info_df['suspend_type'] == 'S'
                ][['ts_code', 'trade_date']].copy()
                suspend_dates['_suspended'] = 1
                df = df.merge(suspend_dates, on=['ts_code', 'trade_date'], how='left')
                df['is_suspended'] = df['_suspended'].fillna(df['is_suspended']).astype(int)
                df.drop(columns=['_suspended'], inplace=True, errors='ignore')
        
        # 5. 涨跌停标记：使用涨跌幅判断
        df['is_limit_up'] = 0
        df['is_limit_down'] = 0
        
        if 'pct_chg' in df.columns:
            # 非 ST：±10%
            non_st = df['is_st'] == 0
            df.loc[non_st & (df['pct_chg'] >= 9.9), 'is_limit_up'] = 1
            df.loc[non_st & (df['pct_chg'] <= -9.9), 'is_limit_down'] = 1
            
            # ST：±5%
            st = df['is_st'] == 1
            df.loc[st & (df['pct_chg'] >= 4.9), 'is_limit_up'] = 1
            df.loc[st & (df['pct_chg'] <= -4.9), 'is_limit_down'] = 1
        
        # 如果有涨跌停信息，使用价格对比（更精确）
        if limit_info_df is not None and len(limit_info_df) > 0:
            limit_prices = limit_info_df[['ts_code', 'trade_date', 'up_limit', 'down_limit']].copy()
            df = df.merge(limit_prices, on=['ts_code', 'trade_date'], how='left')
            
            # 价格比对
            df.loc[df['close'] >= df['up_limit'] * 0.999, 'is_limit_up'] = 1
            df.loc[df['close'] <= df['down_limit'] * 1.001, 'is_limit_down'] = 1
            
            df.drop(columns=['up_limit', 'down_limit'], inplace=True, errors='ignore')
        
        # 6. 可交易标记：非 ST、非停牌、上市满足天数
        df['tradable'] = (
            (df['is_st'] == 0) &
            (df['is_suspended'] == 0) &
            (df['list_days'] >= min_list_days)
        ).astype(int)
        
        # 清理临时列
        df.drop(columns=['name', 'list_date'], inplace=True, errors='ignore')
        
        tradable_count = df['tradable'].sum()
        tradable_pct = 100.0 * tradable_count / len(df) if len(df) > 0 else 0
        
        logger.info(
            f"可交易标记添加完成: 可交易 {tradable_count} ({tradable_pct:.1f}%), "
            f"ST {df['is_st'].sum()}, 停牌 {df['is_suspended'].sum()}, "
            f"上市不足{min_list_days}天 {(df['list_days'] < min_list_days).sum()}"
        )
        
        return df
    
    def _standardize_date_columns(
        self,
        df: pd.DataFrame,
        date_cols: list
    ) -> pd.DataFrame:
        """统一日期列格式为 YYYYMMDD 字符串
        
        Args:
            df: DataFrame
            date_cols: 日期列名列表
            
        Returns:
            处理后的DataFrame
        """
        df = df.copy()
        
        for col in date_cols:
            if col not in df.columns:
                continue
            
            # 如果是 datetime 类型，转换为 YYYYMMDD 字符串
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y%m%d')
            # 如果是字符串类型，确保格式正确
            elif pd.api.types.is_string_dtype(df[col]):
                # 移除可能的分隔符
                df[col] = df[col].str.replace('-', '').str.replace('/', '')
                # 验证格式（应该是8位数字）
                invalid = ~df[col].str.match(r'^\d{8}$', na=False)
                if invalid.sum() > 0:
                    logger.warning(f"列 {col} 中有 {invalid.sum()} 个无效日期格式")
                    df.loc[invalid, col] = None
        
        return df
    
    def _convert_numeric_columns(
        self,
        df: pd.DataFrame,
        numeric_cols: list
    ) -> pd.DataFrame:
        """转换数值列为 float 类型
        
        Args:
            df: DataFrame
            numeric_cols: 数值列名列表
            
        Returns:
            处理后的DataFrame
        """
        df = df.copy()
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"列 {col} 转换为数值类型失败: {e}")
        
        return df
    
    def _deduplicate(
        self,
        df: pd.DataFrame,
        key_cols: list
    ) -> pd.DataFrame:
        """去重，保留最新记录
        
        Args:
            df: DataFrame
            key_cols: 主键列名列表
            
        Returns:
            去重后的DataFrame
        """
        original_count = len(df)
        
        # 检查重复
        duplicates = df.duplicated(subset=key_cols, keep=False)
        dup_count = duplicates.sum()
        
        if dup_count > 0:
            logger.warning(f"发现 {dup_count} 条重复记录（按 {key_cols} 判断），保留最新记录")
            
            # 保留最后一条（假定最后的是最新的）
            df = df.drop_duplicates(subset=key_cols, keep='last')
            
            logger.info(f"去重完成: {original_count} -> {len(df)} ({original_count - len(df)} 条被移除)")
        
        return df
    
    def _calculate_adjusted_prices(
        self,
        daily_df: pd.DataFrame,
        adj_factor_df: pd.DataFrame
    ) -> pd.DataFrame:
        """计算复权价格
        
        Args:
            daily_df: 日线行情DataFrame
            adj_factor_df: 复权因子DataFrame
            
        Returns:
            添加了复权价格列的DataFrame
        """
        logger.info("开始计算复权价格")
        
        df = daily_df.copy()
        
        # 标准化复权因子的日期格式
        adj_factor = adj_factor_df.copy()
        adj_factor = self._standardize_date_columns(adj_factor, ['trade_date'])
        
        if 'ts_code' in adj_factor.columns:
            adj_factor['ts_code'] = adj_factor['ts_code'].astype(str)
        
        # 去重复权因子
        adj_factor = self._deduplicate(adj_factor, ['ts_code', 'trade_date'])
        
        # 合并复权因子
        df = df.merge(
            adj_factor[['ts_code', 'trade_date', 'adj_factor']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        # 处理缺失的复权因子
        missing_adj = df['adj_factor'].isna().sum()
        if missing_adj > 0:
            logger.warning(f"有 {missing_adj} 条记录缺少复权因子，将使用 1.0 作为默认值")
            df['adj_factor'] = df['adj_factor'].fillna(1.0)
        
        # 计算复权价格: price_adj = price * adj_factor
        if 'close' in df.columns:
            df['close_adj'] = df['close'] * df['adj_factor']
        
        if 'open' in df.columns:
            df['open_adj'] = df['open'] * df['adj_factor']
        
        if 'high' in df.columns:
            df['high_adj'] = df['high'] * df['adj_factor']
        
        if 'low' in df.columns:
            df['low_adj'] = df['low'] * df['adj_factor']
        
        # 检查复权价格是否生成
        adj_cols = [c for c in ['close_adj', 'open_adj', 'high_adj', 'low_adj'] if c in df.columns]
        logger.info(f"复权价格计算完成，生成列: {adj_cols}")
        
        return df
    
    def _validate_uniqueness(
        self,
        df: pd.DataFrame,
        key_cols: list
    ) -> None:
        """验证主键唯一性
        
        Args:
            df: DataFrame
            key_cols: 主键列名列表
            
        Raises:
            ValueError: 如果存在重复主键
        """
        duplicates = df.duplicated(subset=key_cols, keep=False)
        dup_count = duplicates.sum()
        
        if dup_count > 0:
            dup_samples = df[duplicates].head(5)
            logger.error(f"数据验证失败: 发现 {dup_count} 条重复主键")
            logger.error(f"重复样本:\n{dup_samples[key_cols]}")
            raise ValueError(f"主键 {key_cols} 存在重复，请检查数据清洗逻辑")
        
        logger.debug(f"主键唯一性验证通过: {key_cols}")
