"""特征确保模块

提供确保 features 数据存在的封装函数
"""

from typing import Optional

import pandas as pd
from loguru import logger

from ..data import DataCleaner, DataLoader, Storage, TushareClient
from ..data.ensure import ensure_basic_data, ensure_clean_data_for_date
from .builder import FeatureBuilder

# 常量定义
FEATURE_DATA_HISTORY_MONTHS = 1  # 特征数据历史月数
FEATURE_DATA_FUTURE_MONTHS = 1   # 特征数据未来月数
HISTORICAL_DATA_MONTHS = 1       # 历史数据回看月数
MAX_HISTORICAL_DAYS = 30         # 最多检查的历史交易日数


def ensure_features_for_date(
    storage: Storage,
    loader: DataLoader,
    builder: FeatureBuilder,
    cleaner: DataCleaner,
    client: TushareClient,
    trade_date: str,
    force: bool = False
) -> bool:
    """确保指定日期的 features 数据存在，不存在则构建
    
    若发现 clean 数据缺失，会自动调用 clean 模块的 ensure 函数
    若发现 raw 数据缺失，会进一步触发 raw 模块的下载
    
    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        builder: FeatureBuilder 实例
        cleaner: DataCleaner 实例
        client: TushareClient 实例（用于在依赖缺失时下载）
        trade_date: 交易日期，格式 YYYYMMDD
        force: 是否强制重新构建
        
    Returns:
        是否成功
    """
    # 检查是否已存在
    if not force and storage.is_feature_exists(trade_date):
        logger.debug(f"features 数据已存在: {trade_date}")
        return True
    
    logger.info(f"构建 features 数据: {trade_date}")
    
    try:
        # 1. 确保基础数据存在
        if not ensure_basic_data(client, storage, trade_date, force=False):
            logger.error("无法获取基础数据（trade_cal/stock_basic）")
            return False
        
        # 2. 确保当日 clean 数据存在
        if not ensure_clean_data_for_date(
            storage, loader, cleaner, client, trade_date, force
        ):
            logger.error(f"无法获取 clean 数据: {trade_date}")
            return False
        
        # 3. 确保历史 clean 数据存在（features 需要历史数据计算特征）
        if not _ensure_historical_clean_data(
            storage, loader, cleaner, client, trade_date, force
        ):
            logger.warning(f"历史 clean 数据不完整，特征可能受影响: {trade_date}")
            # 不返回 False，继续尝试构建特征
        
        # 4. 加载基础数据
        trade_cal = loader.load_clean_trade_cal()
        stock_basic = loader.load_clean_stock_basic()
        
        if trade_cal is None or stock_basic is None:
            logger.error("缺少 clean 基础数据")
            return False
        
        # 转换日期格式
        if 'cal_date' in trade_cal.columns:
            if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
                trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
        
        # 5. 加载 clean 日线数据（扩展范围以包含历史数据）
        start_dt = pd.to_datetime(trade_date, format='%Y%m%d') - pd.DateOffset(
            months=FEATURE_DATA_HISTORY_MONTHS
        )
        end_dt = pd.to_datetime(trade_date, format='%Y%m%d') + pd.DateOffset(
            months=FEATURE_DATA_FUTURE_MONTHS
        )
        
        daily_clean = loader.load_clean_daily(
            start_dt.strftime('%Y%m%d'),
            end_dt.strftime('%Y%m%d')
        )
        
        if daily_clean is None or daily_clean.empty:
            logger.error(f"缺少 clean 日线数据: {trade_date}")
            return False
        
        logger.info(f"clean 日线数据: {len(daily_clean)} 条记录")
        
        # 6. 构建特征（无需传递 adj_factor，clean 数据已包含复权价格）
        features_df = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=trade_cal,
            daily_data=daily_clean,
            adj_factor=pd.DataFrame(),  # 空 DataFrame，clean 数据已包含复权价格
            stock_basic=stock_basic,
            suspend_info=None,
            limit_info=None
        )
        
        # 7. 保存结果
        if len(features_df) > 0:
            storage.save_cs_train_day(features_df, trade_date)#, has_label=builder.require_label)
            logger.info(f"已保存 features 数据: {len(features_df)} 条")
            return True
        else:
            logger.warning(f"没有有效样本: {trade_date}")
            return False
        
    except Exception as e:
        logger.error(f"构建 features 数据失败 {trade_date}: {e}")
        return False


def _ensure_historical_clean_data(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    client: TushareClient,
    trade_date: str,
    force: bool
) -> bool:
    """确保历史 clean 数据存在
    
    Features 构建需要历史数据来计算动量、均值等特征
    这里确保过去一个月的交易日数据存在
    
    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        cleaner: DataCleaner 实例
        client: TushareClient 实例
        trade_date: 当前交易日期，格式 YYYYMMDD
        force: 是否强制重新构建
        
    Returns:
        是否成功（至少部分历史数据可用）
    """
    # 获取交易日历
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        logger.warning("无法加载交易日历，跳过历史数据检查")
        return False
    
    # 确保日期格式统一
    if 'cal_date' in trade_cal.columns:
        if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
            trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
    
    # 获取过去一个月的交易日
    start_dt = pd.to_datetime(trade_date, format='%Y%m%d') - pd.DateOffset(
        months=HISTORICAL_DATA_MONTHS
    )
    
    trading_dates = trade_cal[
        (trade_cal['cal_date'] >= start_dt) &
        (trade_cal['cal_date'] < pd.to_datetime(trade_date, format='%Y%m%d')) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    if not trading_dates:
        logger.warning("未找到历史交易日")
        return False
    
    # 转换为 YYYYMMDD 格式
    trading_dates_str = [
        d.strftime('%Y%m%d') if isinstance(d, pd.Timestamp) else d
        for d in trading_dates
    ]
    
    logger.info(f"检查 {len(trading_dates_str)} 个历史交易日的 clean 数据")
    
    # 检查并补齐缺失的历史数据（最多补齐最近的指定个交易日）
    missing_count = 0
    success_count = 0
    
    for hist_date in trading_dates_str[-MAX_HISTORICAL_DAYS:]:  # 最多检查最近指定个交易日
        if not storage.is_data_exists("clean", "daily", hist_date):
            missing_count += 1
            # 尝试补齐
            if ensure_clean_data_for_date(
                storage, loader, cleaner, client, hist_date, force
            ):
                success_count += 1
    
    if missing_count > 0:
        logger.info(f"补齐了 {success_count}/{missing_count} 个历史交易日的 clean 数据")
    
    # 只要有部分数据可用就返回 True
    return True
