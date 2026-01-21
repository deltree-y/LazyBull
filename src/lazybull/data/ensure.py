"""数据确保模块

提供确保 raw/clean 数据存在的封装函数，按模块边界分层处理依赖
"""

from typing import List, Optional

import pandas as pd
from loguru import logger

from .cleaner import DataCleaner
from .loader import DataLoader
from .storage import Storage
from .tushare_client import TushareClient

# 常量定义
TRADE_CAL_HISTORY_MONTHS = 6  # 交易日历历史数据月数
TRADE_CAL_FUTURE_MONTHS = 6   # 交易日历未来数据月数
MIN_LIST_DAYS = 60             # 最小上市天数（约2个月交易日，用于稳定性分析）


def ensure_raw_data_for_date(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
    force: bool = False
) -> bool:
    """确保指定日期的 raw 数据存在，不存在则下载
    
    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trade_date: 交易日期，格式 YYYYMMDD
        force: 是否强制重新下载
        
    Returns:
        是否成功（True 表示数据已存在或下载成功）
    """
    # 检查是否已存在
    if not force and storage.is_data_exists("raw", "daily", trade_date):
        logger.debug(f"raw 数据已存在: {trade_date}")
        return True
    
    logger.info(f"下载 raw 数据: {trade_date}")
    
    try:
        # 下载日线行情
        daily_data = client.get_daily(trade_date=trade_date)
        if not daily_data.empty:
            storage.save_raw_by_date(daily_data, "daily", trade_date)
            logger.info(f"  日线: 已保存 {len(daily_data)} 条记录")
        
        # 下载复权因子
        adj_factor = client.get_adj_factor(trade_date=trade_date)
        if not adj_factor.empty:
            storage.save_raw_by_date(adj_factor, "adj_factor", trade_date)
            logger.info(f"  复权因子: 已保存 {len(adj_factor)} 条记录")
        
        # 下载停复牌信息
        suspend = client.get_suspend_d(trade_date=trade_date)
        if not suspend.empty:
            storage.save_raw_by_date(suspend, "suspend", trade_date)
            logger.info(f"  停复牌: 已保存 {len(suspend)} 条记录")
        
        # 下载涨跌停信息
        limit_up_down = client.get_stk_limit(trade_date=trade_date)
        if not limit_up_down.empty:
            storage.save_raw_by_date(limit_up_down, "stk_limit", trade_date)
            logger.info(f"  涨跌停: 已保存 {len(limit_up_down)} 条记录")
        
        return True
        
    except Exception as e:
        logger.error(f"下载 raw 数据失败 {trade_date}: {e}")
        return False


def ensure_basic_data(
    client: TushareClient,
    storage: Storage,
    end_date: str,
    force: bool = False
) -> bool:
    """确保基础数据（trade_cal 和 stock_basic）存在
    
    Args:
        client: TushareClient 实例
        storage: Storage 实例
        end_date: 结束日期，用于判断数据是否够新，格式 YYYYMMDD
        force: 是否强制重新下载
        
    Returns:
        是否成功
    """
    logger.info("检查基础数据...")
    
    # 检查 trade_cal
    need_download_trade_cal = force or not storage.check_basic_data_freshness("trade_cal", end_date)
    if need_download_trade_cal:
        logger.info("下载交易日历...")
        try:
            # 扩展日期范围以包含足够的历史和未来数据
            start_dt = pd.to_datetime(end_date, format='%Y%m%d') - pd.DateOffset(
                months=TRADE_CAL_HISTORY_MONTHS
            )
            end_dt = pd.to_datetime(end_date, format='%Y%m%d') + pd.DateOffset(
                months=TRADE_CAL_FUTURE_MONTHS
            )
            
            trade_cal = client.get_trade_cal(
                start_date=start_dt.strftime('19901219'),
                end_date=f"{end_dt.year}1231",    #直接指向目标年度最后一天
                exchange="SSE"
            )
            storage.save_raw(trade_cal, "trade_cal", is_force=True)
            logger.info(f"交易日历已下载: {len(trade_cal)} 条记录")
        except Exception as e:
            logger.error(f"下载交易日历失败: {e}")
            return False
    else:
        logger.info("交易日历已是最新")
    
    # 检查 stock_basic
    need_download_stock_basic = force or not storage.check_basic_data_freshness("stock_basic", end_date)
    if need_download_stock_basic:
        logger.info("下载股票基本信息...")
        try:
            stock_basic = client.get_stock_basic(list_status="L")
            storage.save_raw(stock_basic, "stock_basic", is_force=True)
            logger.info(f"股票基本信息已下载: {len(stock_basic)} 条记录")
        except Exception as e:
            logger.error(f"下载股票基本信息失败: {e}")
            return False
    else:
        logger.info("股票基本信息已存在")
    
    return True


def ensure_clean_data_for_date(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    client: TushareClient,
    trade_date: str,
    force: bool = False
) -> bool:
    """确保指定日期的 clean 数据存在，不存在则构建
    
    若发现 raw 数据缺失，会自动调用 ensure_raw_data_for_date 下载
    
    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        cleaner: DataCleaner 实例
        client: TushareClient 实例（用于在 raw 缺失时下载）
        trade_date: 交易日期，格式 YYYYMMDD
        force: 是否强制重新构建
        
    Returns:
        是否成功
    """
    # 检查是否已存在
    if not force and storage.is_data_exists("clean", "daily", trade_date):
        logger.debug(f"clean 数据已存在: {trade_date}")
        return True
    
    logger.info(f"构建 clean 数据: {trade_date}")
    
    # 确保 raw 数据存在
    if not ensure_raw_data_for_date(client, storage, trade_date, force):
        logger.error(f"无法获取 raw 数据: {trade_date}")
        return False
    
    try:
        # 确保基础 clean 数据存在
        _ensure_basic_clean_data(storage, cleaner)
        
        # 加载 raw 数据
        daily_raw = storage.load_raw_by_date("daily", trade_date)
        if daily_raw is None or daily_raw.empty:
            logger.error(f"未找到 raw 层 daily 数据: {trade_date}")
            return False
        
        adj_factor_raw = storage.load_raw_by_date("adj_factor", trade_date)
        if adj_factor_raw is None or adj_factor_raw.empty:
            logger.warning(f"未找到复权因子，使用默认值 1.0: {trade_date}")
            adj_factor_raw = daily_raw[['ts_code', 'trade_date']].copy()
            adj_factor_raw['adj_factor'] = 1.0
        
        # 清洗日线数据
        daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)
        
        # 添加可交易标记
        stock_basic = loader.load_clean_stock_basic()
        if stock_basic is not None:
            suspend_raw = storage.load_raw_by_date("suspend", trade_date)
            limit_raw = storage.load_raw_by_date("stk_limit", trade_date)
            
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
                min_list_days=MIN_LIST_DAYS
            )
        
        # 保存 clean 数据
        storage.save_clean_by_date(daily_clean, "daily", trade_date)
        logger.info(f"已保存 clean 数据: {len(daily_clean)} 条")
        
        return True
        
    except Exception as e:
        logger.error(f"构建 clean 数据失败 {trade_date}: {e}")
        return False


def _ensure_basic_clean_data(storage: Storage, cleaner: DataCleaner) -> None:
    """确保基础 clean 数据（trade_cal 和 stock_basic）存在
    
    内部辅助函数，不对外暴露
    """
    # 处理 trade_cal
    trade_cal_raw = storage.load_raw("trade_cal")
    if trade_cal_raw is not None:
        trade_cal_clean = cleaner.clean_trade_cal(trade_cal_raw)
        storage.save_clean(trade_cal_clean, "trade_cal", is_force=True)
    
    # 处理 stock_basic
    stock_basic_raw = storage.load_raw("stock_basic")
    if stock_basic_raw is not None:
        stock_basic_clean = cleaner.clean_stock_basic(stock_basic_raw)
        storage.save_clean(stock_basic_clean, "stock_basic", is_force=True)
