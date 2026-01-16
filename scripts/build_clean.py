#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清洗数据构建脚本

从 raw 层数据生成 clean 层数据，包括：
- 去重、类型统一、缺失值处理
- 复权价格计算
- ST/停牌过滤标记
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataCleaner, DataLoader, Storage


def build_clean_for_date_range(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    start_date: str,
    end_date: str,
    use_partitioning: bool = True
):
    """为日期范围构建 clean 数据
    
    Args:
        storage: 存储实例
        loader: 数据加载器
        cleaner: 数据清洗器
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        use_partitioning: 是否使用分区存储
    """
    logger.info("=" * 60)
    logger.info("开始构建 clean 数据")
    logger.info("=" * 60)
    
    # 1. 加载并清洗交易日历
    logger.info("处理交易日历...")
    trade_cal_raw = storage.load_raw("trade_cal")
    if trade_cal_raw is not None:
        trade_cal_clean = cleaner.clean_trade_cal(trade_cal_raw)
        storage.save_clean(trade_cal_clean, "trade_cal", is_force=True)
        logger.info(f"交易日历清洗完成: {len(trade_cal_clean)} 条记录")
    else:
        logger.warning("未找到原始交易日历数据")
        trade_cal_clean = None
    
    # 2. 加载并清洗股票基本信息
    logger.info("处理股票基本信息...")
    stock_basic_raw = storage.load_raw("stock_basic")
    if stock_basic_raw is not None:
        stock_basic_clean = cleaner.clean_stock_basic(stock_basic_raw)
        storage.save_clean(stock_basic_clean, "stock_basic", is_force=True)
        logger.info(f"股票基本信息清洗完成: {len(stock_basic_clean)} 条记录")
    else:
        logger.warning("未找到原始股票基本信息")
        stock_basic_clean = None
    
    if not use_partitioning:
        # 3. 非分区模式：处理整体数据
        logger.info("使用非分区模式处理日线数据...")
        
        daily_raw = storage.load_raw("daily")
        adj_factor_raw = storage.load_raw("adj_factor")
        
        if daily_raw is None or adj_factor_raw is None:
            logger.error("缺少日线行情或复权因子数据")
            return
        
        # 清洗日线数据（包含复权价格计算）
        logger.info("清洗日线行情并计算复权价格...")
        daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)
        
        # 添加可交易标记
        if stock_basic_clean is not None:
            logger.info("添加可交易标记...")
            daily_clean = cleaner.add_tradable_universe_flag(
                daily_clean,
                stock_basic_clean,
                suspend_info_df=None,
                limit_info_df=None,
                min_list_days=60
            )
        
        # 保存
        storage.save_clean(daily_clean, "daily", is_force=True)
        logger.info(f"日线行情清洗完成: {len(daily_clean)} 条记录")
        
        # 处理 daily_basic
        daily_basic_raw = storage.load_raw("daily_basic")
        if daily_basic_raw is not None:
            logger.info("清洗每日指标...")
            daily_basic_clean = cleaner.clean_daily_basic(daily_basic_raw)
            storage.save_clean(daily_basic_clean, "daily_basic", is_force=True)
            logger.info(f"每日指标清洗完成: {len(daily_basic_clean)} 条记录")
        
    else:
        # 4. 分区模式：按交易日逐日处理
        logger.info("使用分区模式处理日线数据...")
        
        if trade_cal_clean is None:
            logger.error("缺少交易日历，无法进行分区处理")
            return
        
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
                # 检查 clean 数据是否已存在
                if storage.is_data_exists("clean", "daily", trade_date):
                    logger.info(f"  clean daily 已存在，跳过")
                    skip_count += 1
                    continue
                
                # 加载该日的 raw 数据
                daily_raw = storage.load_raw_by_date("daily", trade_date)
                adj_factor_raw = storage.load_raw_by_date("adj_factor", trade_date)
                
                if daily_raw is None or len(daily_raw) == 0:
                    logger.warning(f"  未找到日线数据，跳过")
                    skip_count += 1
                    continue
                
                if adj_factor_raw is None or len(adj_factor_raw) == 0:
                    logger.warning(f"  未找到复权因子，使用默认值 1.0")
                    # 创建默认复权因子
                    adj_factor_raw = daily_raw[['ts_code', 'trade_date']].copy()
                    adj_factor_raw['adj_factor'] = 1.0
                
                # 清洗日线数据
                daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)
                
                # 添加可交易标记
                if stock_basic_clean is not None:
                    suspend_raw = storage.load_raw_by_date("suspend", trade_date)
                    limit_raw = storage.load_raw_by_date("stk_limit", trade_date)
                    
                    # 清洗停复牌和涨跌停信息（如果存在）
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
                
                # 保存 clean 数据
                storage.save_clean_by_date(daily_clean, "daily", trade_date)
                success_count += 1
                
                # 处理 daily_basic（如果存在）
                daily_basic_raw = storage.load_raw_by_date("daily_basic", trade_date)
                if daily_basic_raw is not None and len(daily_basic_raw) > 0:
                    if not storage.is_data_exists("clean", "daily_basic", trade_date):
                        daily_basic_clean = cleaner.clean_daily_basic(daily_basic_raw)
                        storage.save_clean_by_date(daily_basic_clean, "daily_basic", trade_date)
                
            except Exception as e:
                logger.error(f"处理 {trade_date} 失败: {str(e)}")
                error_count += 1
                continue
        
        logger.info("=" * 60)
        logger.info("分区数据处理完成")
        logger.info("=" * 60)
        logger.info(f"成功: {success_count} 个交易日")
        logger.info(f"跳过: {skip_count} 个交易日（已存在）")
        logger.info(f"失败: {error_count} 个交易日")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建 clean 层数据")
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
        "--use-monolithic",
        action="store_true",
        help="使用非分区模式（不推荐，默认关闭）"
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logger(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("开始构建 clean 数据")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} - {args.end_date}")
    logger.info(f"分区模式: {'否' if args.use_monolithic else '是'}")
    
    try:
        # 初始化组件
        storage = Storage(enable_partitioning=not args.use_monolithic)
        loader = DataLoader(storage)
        cleaner = DataCleaner()
        
        # 构建 clean 数据
        build_clean_for_date_range(
            storage,
            loader,
            cleaner,
            args.start_date,
            args.end_date,
            use_partitioning=not args.use_monolithic
        )
        
        logger.info("=" * 60)
        logger.info("clean 数据构建完成！")
        logger.info(f"数据保存位置: {storage.clean_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception(f"构建过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
