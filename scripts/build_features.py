#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接构建特征脚本（自动补齐依赖）

功能：
- 以feature为目标，直接构建特征
- 若发现raw或clean缺失，则自动在build feature流程中完成相应获取/计算
- 具有"补齐依赖"的能力
- 支持force参数强制重新构建
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataCleaner, DataLoader, Storage, TushareClient
from src.lazybull.features import FeatureBuilder


def ensure_basic_data(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False
) -> pd.DataFrame:
    """确保基础数据存在（trade_cal和stock_basic）
    
    如果不存在或不够新，则自动下载
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新下载
        
    Returns:
        trade_cal DataFrame
    """
    logger.info("检查基础数据...")
    
    # 检查trade_cal
    need_download_trade_cal = force or not storage.check_basic_data_freshness("trade_cal", end_date)
    if need_download_trade_cal:
        logger.info("下载交易日历...")
        # 扩展日期范围
        start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=6)
        end_dt = pd.to_datetime(end_date, format='%Y%m%d') + pd.DateOffset(months=6)
        
        trade_cal = client.get_trade_cal(
            start_date=start_dt.strftime('%Y%m%d'),
            end_date=end_dt.strftime('%Y%m%d'),
            exchange="SSE"
        )
        storage.save_raw(trade_cal, "trade_cal", is_force=True)
        logger.info(f"交易日历已下载: {len(trade_cal)} 条记录")
    else:
        logger.info("交易日历已是最新")
        trade_cal = storage.load_raw("trade_cal")
    
    # 检查stock_basic
    need_download_stock_basic = force or not storage.check_basic_data_freshness("stock_basic", end_date)
    if need_download_stock_basic:
        logger.info("下载股票基本信息...")
        stock_basic = client.get_stock_basic(list_status="L")
        storage.save_raw(stock_basic, "stock_basic", is_force=True)
        logger.info(f"股票基本信息已下载: {len(stock_basic)} 条记录")
    else:
        logger.info("股票基本信息已存在")
    
    return trade_cal


def ensure_raw_data(
    client: TushareClient,
    storage: Storage,
    trade_cal: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False
) -> None:
    """确保raw数据存在
    
    检查每个交易日的raw数据，缺失则自动下载
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        trade_cal: 交易日历DataFrame
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新下载
    """
    logger.info("检查raw数据...")
    
    # 扩展日期范围（需要历史数据用于特征计算）
    start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=1)
    end_dt = pd.to_datetime(end_date, format='%Y%m%d') + pd.DateOffset(months=1)
    
    # 获取交易日列表
    trading_dates = trade_cal[
        (trade_cal['cal_date'] >= start_dt.strftime('%Y%m%d')) &
        (trade_cal['cal_date'] <= end_dt.strftime('%Y%m%d')) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    missing_dates = []
    for trade_date in trading_dates:
        if force or not storage.is_data_exists("raw", "daily", trade_date):
            missing_dates.append(trade_date)
    
    if not missing_dates:
        logger.info("raw数据已完整")
        return
    
    logger.info(f"需要下载 {len(missing_dates)} 个交易日的raw数据")
    
    for i, trade_date in enumerate(missing_dates, 1):
        logger.info(f"[{i}/{len(missing_dates)}] 下载 {trade_date}...")
        
        try:
            # 下载日线行情
            daily_data = client.get_daily(trade_date=trade_date)
            if len(daily_data) > 0:
                storage.save_raw_by_date(daily_data, "daily", trade_date)
            
            # 下载每日指标
            daily_basic = client.get_daily_basic(trade_date=trade_date)
            if len(daily_basic) > 0:
                storage.save_raw_by_date(daily_basic, "daily_basic", trade_date)
            
            # 下载复权因子
            adj_factor = client.get_adj_factor(trade_date=trade_date)
            if len(adj_factor) > 0:
                storage.save_raw_by_date(adj_factor, "adj_factor", trade_date)
            
            # 下载停复牌信息
            suspend = client.get_suspend_d(trade_date=trade_date)
            if len(suspend) > 0:
                storage.save_raw_by_date(suspend, "suspend", trade_date)
            
            # 下载涨跌停信息
            limit_up_down = client.get_stk_limit(trade_date=trade_date)
            if len(limit_up_down) > 0:
                storage.save_raw_by_date(limit_up_down, "stk_limit", trade_date)
                
        except Exception as e:
            logger.error(f"下载 {trade_date} 失败: {str(e)}")
            continue


def ensure_clean_data(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    start_date: str,
    end_date: str,
    force: bool = False
) -> None:
    """确保clean数据存在
    
    检查每个交易日的clean数据，缺失则从raw数据构建
    
    Args:
        storage: Storage实例
        loader: DataLoader实例
        cleaner: DataCleaner实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新构建
    """
    logger.info("检查clean数据...")
    
    # 处理trade_cal和stock_basic
    trade_cal_raw = storage.load_raw("trade_cal")
    stock_basic_raw = storage.load_raw("stock_basic")
    
    if trade_cal_raw is None or stock_basic_raw is None:
        raise ValueError("缺少基础数据，无法构建clean数据")
    
    # 清洗并保存基础数据
    trade_cal_clean = cleaner.clean_trade_cal(trade_cal_raw)
    storage.save_clean(trade_cal_clean, "trade_cal", is_force=True)
    
    stock_basic_clean = cleaner.clean_stock_basic(stock_basic_raw)
    storage.save_clean(stock_basic_clean, "stock_basic", is_force=True)
    
    # 扩展日期范围
    start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=1)
    end_dt = pd.to_datetime(end_date, format='%Y%m%d') + pd.DateOffset(months=1)
    
    # 获取交易日列表
    trading_dates = trade_cal_clean[
        (trade_cal_clean['cal_date'] >= start_dt.strftime('%Y%m%d')) &
        (trade_cal_clean['cal_date'] <= end_dt.strftime('%Y%m%d')) &
        (trade_cal_clean['is_open'] == 1)
    ]['cal_date'].tolist()
    
    missing_dates = []
    for trade_date in trading_dates:
        if force or not storage.is_data_exists("clean", "daily", trade_date):
            missing_dates.append(trade_date)
    
    if not missing_dates:
        logger.info("clean数据已完整")
        return
    
    logger.info(f"需要构建 {len(missing_dates)} 个交易日的clean数据")
    
    for i, trade_date in enumerate(missing_dates, 1):
        logger.info(f"[{i}/{len(missing_dates)}] 构建 {trade_date}...")
        
        try:
            # 加载raw数据
            daily_raw = storage.load_raw_by_date("daily", trade_date)
            if daily_raw is None or len(daily_raw) == 0:
                logger.warning(f"  未找到raw数据，跳过")
                continue
            
            adj_factor_raw = storage.load_raw_by_date("adj_factor", trade_date)
            if adj_factor_raw is None or len(adj_factor_raw) == 0:
                adj_factor_raw = daily_raw[['ts_code', 'trade_date']].copy()
                adj_factor_raw['adj_factor'] = 1.0
            
            # 清洗数据
            daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)
            
            # 添加可交易标记
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
                stock_basic_clean,
                suspend_info_df=suspend_clean,
                limit_info_df=limit_clean,
                min_list_days=60
            )
            
            # 保存clean数据
            storage.save_clean_by_date(daily_clean, "daily", trade_date)
            
            # 处理daily_basic
            daily_basic_raw = storage.load_raw_by_date("daily_basic", trade_date)
            if daily_basic_raw is not None and len(daily_basic_raw) > 0:
                daily_basic_clean = cleaner.clean_daily_basic(daily_basic_raw)
                storage.save_clean_by_date(daily_basic_clean, "daily_basic", trade_date)
            
        except Exception as e:
            logger.error(f"构建 {trade_date} 失败: {str(e)}")
            continue


def build_features(
    storage: Storage,
    loader: DataLoader,
    builder: FeatureBuilder,
    start_date: str,
    end_date: str,
    force: bool = False
) -> None:
    """构建特征数据
    
    Args:
        storage: Storage实例
        loader: DataLoader实例
        builder: FeatureBuilder实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新构建
    """
    logger.info("=" * 60)
    logger.info("开始构建特征数据")
    logger.info("=" * 60)
    
    # 加载基础数据
    trade_cal = loader.load_clean_trade_cal()
    stock_basic = loader.load_clean_stock_basic()
    
    if trade_cal is None or stock_basic is None:
        raise ValueError("缺少clean基础数据")
    
    # 转换日期格式
    if 'cal_date' in trade_cal.columns:
        if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
            trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
    
    # 获取交易日列表
    trading_dates = loader.get_trading_dates(
        start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:8],
        end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:8]
    )
    
    trading_dates_str = [
        d.strftime('%Y%m%d') if isinstance(d, pd.Timestamp) else d
        for d in trading_dates
    ]
    
    logger.info(f"共 {len(trading_dates_str)} 个交易日需要构建特征")
    
    # 加载clean日线数据
    start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=1)
    end_dt = pd.to_datetime(end_date, format='%Y%m%d') + pd.DateOffset(months=1)
    
    daily_clean = loader.load_clean_daily(
        start_dt.strftime('%Y%m%d'),
        end_dt.strftime('%Y%m%d')
    )
    
    if daily_clean is None:
        raise ValueError("缺少clean日线数据")
    
    logger.info(f"clean日线数据: {len(daily_clean)} 条记录")
    
    # clean数据已包含复权价格
    adj_factor = pd.DataFrame(columns=['ts_code', 'trade_date', 'adj_factor'])
    
    # 构建特征
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, trade_date in enumerate(trading_dates_str, 1):
        logger.info(f"[{i}/{len(trading_dates_str)}] ({i/len(trading_dates_str):.1%}) 构建 {trade_date} 特征...")
        
        try:
            # 检查是否已存在
            if not force and storage.is_feature_exists(trade_date):
                logger.info(f"  特征已存在，跳过")
                skip_count += 1
                continue
            
            # 构建特征
            features_df = builder.build_features_for_day(
                trade_date=trade_date,
                trade_cal=trade_cal,
                daily_data=daily_clean,
                adj_factor=adj_factor,
                stock_basic=stock_basic,
                suspend_info=None,
                limit_info=None
            )
            
            # 保存结果
            if len(features_df) > 0:
                storage.save_cs_train_day(features_df, trade_date)
                success_count += 1
                logger.info(f"  已保存 {len(features_df)} 条特征")
            else:
                logger.warning(f"  没有有效样本")
                skip_count += 1
                
        except Exception as e:
            logger.error(f"  构建失败: {str(e)}")
            error_count += 1
            continue
    
    logger.info("=" * 60)
    logger.info("特征构建完成")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count} 个交易日")
    logger.info(f"跳过: {skip_count} 个交易日")
    logger.info(f"失败: {error_count} 个交易日")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="直接构建特征（自动补齐raw和clean依赖）"
    )
    parser.add_argument(
        "--start-date",
        default="20200101",
        help="开始日期，格式YYYYMMDD（默认：20200101）"
    )
    parser.add_argument(
        "--end-date",
        default="20251231",
        help="结束日期，格式YYYYMMDD（默认：20251231）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新构建所有数据"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过自动下载，如果raw数据缺失则报错"
    )
    parser.add_argument(
        "--min-list-days",
        type=int,
        default=60,
        help="最小上市天数（默认：60）"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="预测时间窗口（交易日）（默认：5）"
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logger(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("直接构建特征（自动补齐依赖）")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} - {args.end_date}")
    logger.info(f"强制重新构建: {'是' if args.force else '否'}")
    logger.info(f"跳过自动下载: {'是' if args.skip_download else '否'}")
    logger.info("=" * 60)
    
    try:
        # 初始化组件
        storage = Storage()
        loader = DataLoader(storage)
        cleaner = DataCleaner()
        builder = FeatureBuilder(
            min_list_days=args.min_list_days,
            horizon=args.horizon
        )
        
        # 如果不跳过下载，则确保数据完整
        if not args.skip_download:
            from src.lazybull.common.config import get_config
            get_config()
            client = TushareClient()
            
            # 1. 确保基础数据存在
            trade_cal = ensure_basic_data(
                client, storage,
                args.start_date, args.end_date,
                force=args.force
            )
            
            # 2. 确保raw数据存在
            ensure_raw_data(
                client, storage, trade_cal,
                args.start_date, args.end_date,
                force=args.force
            )
        
        # 3. 确保clean数据存在
        ensure_clean_data(
            storage, loader, cleaner,
            args.start_date, args.end_date,
            force=args.force
        )
        
        # 4. 构建特征
        build_features(
            storage, loader, builder,
            args.start_date, args.end_date,
            force=args.force
        )
        
        logger.info("=" * 60)
        logger.info("全部完成！")
        logger.info(f"特征数据位置: {storage.features_path}/cs_train")
        logger.info("=" * 60)
        
    except ValueError as e:
        logger.error("=" * 60)
        logger.error("构建失败")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("提示：如果跳过了自动下载，请先手动下载数据:")
        logger.error("  python scripts/download_raw.py")
        logger.error("=" * 60)
        sys.exit(1)
        
    except Exception as e:
        logger.exception(f"构建过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
