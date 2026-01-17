#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建clean和features数据脚本

功能：
- 假设raw数据已存在，若缺失则报错
- 只负责计算clean和feature并保存（partitioned存储）
- 不进行raw数据下载
- 支持force参数强制重新构建已存在的数据
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
from src.lazybull.data import DataCleaner, DataLoader, Storage
from src.lazybull.features import FeatureBuilder


def build_clean_data(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    start_date: str,
    end_date: str,
    force: bool = False
) -> None:
    """构建clean层数据
    
    Args:
        storage: Storage实例
        loader: DataLoader实例
        cleaner: DataCleaner实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新构建
    """
    logger.info("=" * 60)
    logger.info("开始构建clean层数据")
    logger.info("=" * 60)
    
    # 1. 检查并处理trade_cal
    logger.info("处理交易日历...")
    trade_cal_raw = storage.load_raw("trade_cal")
    if trade_cal_raw is None:
        raise ValueError("缺少raw层trade_cal数据，请先运行: python scripts/download_raw.py --only-basic")
    
    trade_cal_clean = cleaner.clean_trade_cal(trade_cal_raw)
    storage.save_clean(trade_cal_clean, "trade_cal", is_force=True)
    logger.info(f"交易日历清洗完成: {len(trade_cal_clean)} 条记录")
    
    # 2. 检查并处理stock_basic
    logger.info("处理股票基本信息...")
    stock_basic_raw = storage.load_raw("stock_basic")
    if stock_basic_raw is None:
        raise ValueError("缺少raw层stock_basic数据，请先运行: python scripts/download_raw.py --only-basic")
    
    stock_basic_clean = cleaner.clean_stock_basic(stock_basic_raw)
    storage.save_clean(stock_basic_clean, "stock_basic", is_force=True)
    logger.info(f"股票基本信息清洗完成: {len(stock_basic_clean)} 条记录")
    
    # 3. 按日期分区处理日线数据
    logger.info("使用分区模式处理日线数据...")
    
    # 获取交易日列表
    trading_dates = trade_cal_clean[
        (trade_cal_clean['cal_date'] >= start_date) &
        (trade_cal_clean['cal_date'] <= end_date) &
        (trade_cal_clean['is_open'] == 1)
    ]['cal_date'].tolist()
    
    logger.info(f"共 {len(trading_dates)} 个交易日需要处理")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, trade_date in enumerate(trading_dates, 1):
        logger.info(f"[{i}/{len(trading_dates)}] ({i/len(trading_dates):.1%}) 处理 {trade_date}...")
        
        try:
            # 检查clean数据是否已存在
            if not force and storage.is_data_exists("clean", "daily", trade_date):
                logger.info(f"  clean daily已存在，跳过")
                skip_count += 1
                continue
            
            # 加载raw数据
            daily_raw = storage.load_raw_by_date("daily", trade_date)
            if daily_raw is None or len(daily_raw) == 0:
                logger.warning(f"  未找到raw层daily数据，跳过")
                error_count += 1
                continue
            
            adj_factor_raw = storage.load_raw_by_date("adj_factor", trade_date)
            if adj_factor_raw is None or len(adj_factor_raw) == 0:
                logger.warning(f"  未找到复权因子，使用默认值1.0")
                adj_factor_raw = daily_raw[['ts_code', 'trade_date']].copy()
                adj_factor_raw['adj_factor'] = 1.0
            
            # 清洗日线数据
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
            success_count += 1
            logger.info(f"  已保存 {len(daily_clean)} 条clean记录")
            
            # 处理daily_basic
            daily_basic_raw = storage.load_raw_by_date("daily_basic", trade_date)
            if daily_basic_raw is not None and len(daily_basic_raw) > 0:
                if force or not storage.is_data_exists("clean", "daily_basic", trade_date):
                    daily_basic_clean = cleaner.clean_daily_basic(daily_basic_raw)
                    storage.save_clean_by_date(daily_basic_clean, "daily_basic", trade_date)
            
        except Exception as e:
            logger.error(f"处理 {trade_date} 失败: {str(e)}")
            error_count += 1
            continue
    
    logger.info("=" * 60)
    logger.info("clean层数据构建完成")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count} 个交易日")
    logger.info(f"跳过: {skip_count} 个交易日（已存在）")
    logger.info(f"失败: {error_count} 个交易日")


def build_features_data(
    storage: Storage,
    loader: DataLoader,
    builder: FeatureBuilder,
    start_date: str,
    end_date: str,
    force: bool = False
) -> None:
    """构建features层数据
    
    Args:
        storage: Storage实例
        loader: DataLoader实例
        builder: FeatureBuilder实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新构建
    """
    logger.info("=" * 60)
    logger.info("开始构建features层数据")
    logger.info("=" * 60)
    
    # 加载基础数据（从clean层）
    logger.info("加载基础数据...")
    trade_cal = loader.load_clean_trade_cal()
    stock_basic = loader.load_clean_stock_basic()
    
    if trade_cal is None:
        raise ValueError("缺少clean层trade_cal数据")
    if stock_basic is None:
        raise ValueError("缺少clean层stock_basic数据")
    
    # 转换日期格式
    if 'cal_date' in trade_cal.columns:
        if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
            trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
    
    # 获取交易日列表
    trading_dates = loader.get_trading_dates(
        start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:8],
        end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:8]
    )
    
    if len(trading_dates) == 0:
        raise ValueError(f"指定日期范围内没有交易日: {start_date} - {end_date}")
    
    # 转换为YYYYMMDD格式
    trading_dates_str = [
        d.strftime('%Y%m%d') if isinstance(d, pd.Timestamp) else d
        for d in trading_dates
    ]
    
    logger.info(f"共 {len(trading_dates_str)} 个交易日需要构建特征")
    
    # 加载clean层日线数据（扩展范围以包含历史数据）
    start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=1)
    end_dt = pd.to_datetime(end_date, format='%Y%m%d') + pd.DateOffset(months=1)
    
    daily_clean = loader.load_clean_daily(
        start_dt.strftime('%Y%m%d'),
        end_dt.strftime('%Y%m%d')
    )
    
    if daily_clean is None:
        raise ValueError("缺少clean层daily数据")
    
    logger.info(f"clean日线数据: {len(daily_clean)} 条记录")
    
    # clean数据已包含复权价格，使用空DataFrame
    adj_factor = pd.DataFrame(columns=['ts_code', 'trade_date', 'adj_factor'])
    
    # 构建特征
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, trade_date in enumerate(trading_dates_str, 1):
        logger.info(f"[{i}/{len(trading_dates_str)}] ({i/len(trading_dates_str):.1%}) 构建 {trade_date} 特征...")
        
        try:
            # 检查特征是否已存在
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
                logger.info(f"  已保存 {len(features_df)} 条特征记录")
            else:
                logger.warning(f"  没有有效样本，跳过保存")
                skip_count += 1
                
        except Exception as e:
            logger.error(f"  构建失败: {str(e)}")
            error_count += 1
            continue
    
    logger.info("=" * 60)
    logger.info("features层数据构建完成")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count} 个交易日")
    logger.info(f"跳过: {skip_count} 个交易日（已存在或无效样本）")
    logger.info(f"失败: {error_count} 个交易日")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="构建clean和features数据（假设raw已存在）"
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
        "--only-clean",
        action="store_true",
        help="仅构建clean层，不构建features"
    )
    parser.add_argument(
        "--only-features",
        action="store_true",
        help="仅构建features层，不构建clean"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新构建，即使文件已存在"
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
    logger.info("开始构建clean和features数据")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} - {args.end_date}")
    logger.info(f"仅构建clean: {'是' if args.only_clean else '否'}")
    logger.info(f"仅构建features: {'是' if args.only_features else '否'}")
    logger.info(f"强制重新构建: {'是' if args.force else '否'}")
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
        
        # 构建clean数据
        if not args.only_features:
            build_clean_data(
                storage, loader, cleaner,
                args.start_date, args.end_date,
                force=args.force
            )
        
        # 构建features数据
        if not args.only_clean:
            build_features_data(
                storage, loader, builder,
                args.start_date, args.end_date,
                force=args.force
            )
        
        logger.info("=" * 60)
        logger.info("数据构建完成！")
        logger.info(f"clean数据位置: {storage.clean_path}")
        logger.info(f"features数据位置: {storage.features_path}")
        logger.info("=" * 60)
        
    except ValueError as e:
        logger.error("=" * 60)
        logger.error("数据构建失败")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("请先下载raw数据:")
        logger.error("  python scripts/download_raw.py")
        logger.error("=" * 60)
        sys.exit(1)
        
    except Exception as e:
        logger.exception(f"构建过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
